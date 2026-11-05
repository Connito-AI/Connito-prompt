"""The vLLM engine's CPU-testable parts: the fp8-row export, the residual math against the HF hook, the shard layout,
and the greedy settings."""
from __future__ import annotations

import functools
import hashlib
import json
import re
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file
from torch import nn
from transformers import GenerationConfig
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeMLP

from connito.shared.modeling.families import FAMILIES
from connito.shared.modeling.quantization import FP8_DTYPE, _to_fp8
from connito.shared.modeling.residual import _residual_hook
from connito.test.conftest import TINY_TEXT_CONFIG

HERE = Path(__file__).resolve().parents[2] / "scripts" / "prompt_validator_vllm"
sys.path.insert(0, str(HERE))
import export_fp8row  # noqa: E402
from connito_vllm import residual_math, sampling  # noqa: E402

H, INTER, K, A, TAU = 16, 8, 3, 0.5, 0.05


def _hf_block(dtype: torch.dtype) -> SimpleNamespace:
    """The HF block's residual parts as install_residual_experts builds them: K Qwen MLPs and an fp32 router."""
    torch.manual_seed(0)
    text = Qwen3_5MoeTextConfig(hidden_size=H, moe_intermediate_size=INTER, hidden_act="silu")
    experts = nn.ModuleList(Qwen3_5MoeMLP(text, intermediate_size=INTER) for _ in range(K)).to(dtype)
    router = nn.Linear(H, K, bias=False, dtype=torch.float32)
    return SimpleNamespace(residual_experts=experts, residual_router=router)


def _shard(block: SimpleNamespace, layers=(0, 2)) -> dict[str, torch.Tensor]:
    """The block's tensors under the names a miner's shard uses, on each MoE layer."""
    shard = {}
    for layer in layers:
        shard[f"model.layers.{layer}.mlp.residual_router.weight"] = block.residual_router.weight.detach()
        for j, expert in enumerate(block.residual_experts):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                key = f"model.layers.{layer}.mlp.residual_experts.{j}.{proj}.weight"
                shard[key] = getattr(expert, proj).weight.detach()
    return shard


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_the_vllm_residual_adds_what_the_hf_hook_adds(dtype):
    block = _hf_block(dtype)
    x, out = torch.randn(5, H, dtype=dtype), torch.randn(5, H, dtype=dtype)
    hook = functools.partial(_residual_hook, threshold=A, temperature=TAU)
    with torch.no_grad():
        expected = hook(block, (x,), out) - out
        parts = residual_math.stack_shard(_shard(block, layers=(0,)), [0], K)[0]
        got = residual_math.residual_delta(x, parts["router"], parts["gate"], parts["up"], parts["down"], A, TAU)
    torch.testing.assert_close(out + got, out + expected, rtol=0, atol=0)


def test_stack_shard_stacks_each_layers_experts_in_order():
    block = _hf_block(torch.float32)
    stacked = residual_math.stack_shard(_shard(block), [0, 2], K)
    assert sorted(stacked) == [0, 2]
    parts = stacked[2]
    assert parts["router"].shape == (K, H)
    assert parts["gate"].shape == parts["up"].shape == (K, INTER, H) and parts["down"].shape == (K, H, INTER)
    assert torch.equal(parts["down"][1], block.residual_experts[1].down_proj.weight)


@pytest.mark.parametrize("change, error", [
    (lambda s: s.pop("model.layers.2.mlp.residual_experts.1.up_proj.weight"), "up_proj for experts"),
    (lambda s: s.pop("model.layers.0.mlp.residual_router.weight"), "no residual router"),
    (lambda s: s.update({"model.layers.0.mlp.gate.weight": torch.zeros(1)}), "not a residual key"),
    (lambda s: s.update({"model.layers.1.mlp.residual_router.weight": torch.zeros(K, H)}), "has no MoE block"),
])
def test_stack_shard_refuses_a_shard_that_does_not_fill_every_moe_layer(change, error):
    shard = _shard(_hf_block(torch.float32))
    change(shard)
    with pytest.raises(ValueError, match=error):
        residual_math.stack_shard(shard, [0, 2], K)


def test_stack_shard_refuses_another_k():
    with pytest.raises(ValueError, match="experts"):
        residual_math.stack_shard(_shard(_hf_block(torch.float32)), [0, 2], K + 1)


# --- the fp8-row export -----------------------------------------------------


def _read_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    out = {}
    for shard in path.glob("*.safetensors"):
        out.update(load_file(shard))
    return out


@pytest.fixture()
def exported(tmp_path, monkeypatch, tiny_qwen_checkpoint) -> tuple[Path, dict, dict[str, torch.Tensor]]:
    monkeypatch.setitem(FAMILIES, "qwen3_5_moe", replace(FAMILIES["qwen3_5_moe"], model_path=str(tiny_qwen_checkpoint)))
    out = tmp_path / "fp8row"
    manifest = export_fp8row.export(out)
    return out, manifest, load_file(tiny_qwen_checkpoint / "model.safetensors")


def test_the_export_writes_the_loaders_fp8_bytes_and_bf16_rounded_scales_per_expert(exported):
    out, _, hub = exported
    got = _read_checkpoint(out)
    inter, experts = TINY_TEXT_CONFIG["moe_intermediate_size"], TINY_TEXT_CONFIG["num_experts"]
    for layer in range(TINY_TEXT_CONFIG["num_hidden_layers"]):
        prefix = f"model.language_model.layers.{layer}.mlp.experts"
        gate_up, gate_up_scale = _to_fp8(hub[f"{prefix}.gate_up_proj"])
        down, down_scale = _to_fp8(hub[f"{prefix}.down_proj"])
        for e in range(experts):
            for proj, values, scale in (("gate_proj", gate_up[e, :inter], gate_up_scale[e, :inter]),
                                        ("up_proj", gate_up[e, inter:], gate_up_scale[e, inter:]),
                                        ("down_proj", down[e], down_scale[e])):
                weight, weight_scale = got[f"{prefix}.{e}.{proj}.weight"], got[f"{prefix}.{e}.{proj}.weight_scale"]
                assert weight.dtype == FP8_DTYPE and torch.equal(weight.view(torch.uint8), values.view(torch.uint8))
                assert weight_scale.dtype == torch.float32 and weight_scale.shape == (values.shape[0], 1)
                assert torch.equal(weight_scale, scale.to(torch.bfloat16).float())
        assert f"{prefix}.gate_up_proj" not in got and f"{prefix}.down_proj" not in got


def test_the_export_copies_every_other_text_tensor_unchanged_and_drops_vision_and_mtp(exported):
    out, manifest, hub = exported
    got = _read_checkpoint(out)
    kept = {k for k in hub if not k.startswith(("model.visual.", "mtp.")) and ".mlp.experts." not in k}
    assert kept and kept == {k for k in got if ".mlp.experts." not in k}
    for key in kept:
        assert got[key].dtype == hub[key].dtype and torch.equal(got[key], hub[key])
    assert set(manifest["tensors"]) == set(got)
    key = next(iter(kept))
    assert manifest["tensors"][key] == hashlib.sha256(hub[key].reshape(-1).view(torch.uint8).numpy()).hexdigest()
    assert manifest["revision"] == FAMILIES["qwen3_5_moe"].revision
    index = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
    assert set(index) == set(got)
    files = manifest["files"]
    assert {"config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json"} <= set(files)
    assert not any(name.endswith(".safetensors") or name == "manifest.json" for name in files)
    assert files["config.json"] == hashlib.sha256((out / "config.json").read_bytes()).hexdigest()


def test_the_export_quantizes_only_the_routed_experts_in_its_config(exported):
    out, _, _ = exported
    config = json.loads((out / "config.json").read_text())["quantization_config"]
    [group] = config["config_groups"].values()
    target = re.compile(group["targets"][0].removeprefix("re:"))
    assert group["weights"]["strategy"] == "channel" and group["weights"]["type"] == "float"
    # The names vLLM's compressed-tensors MoE method asks about, and the layers that stay bf16.
    assert all(target.match(f"model.layers.3.mlp.experts.0.{p}") for p in ("gate_proj", "up_proj", "down_proj"))
    for name in ("model.layers.3.mlp.shared_expert.gate_up_proj", "model.layers.3.mlp.shared_expert.down_proj",
                 "model.layers.3.mlp.gate", "model.layers.3.self_attn.qkv_proj", "lm_head"):
        assert not target.match(name)


# --- greedy settings ----------------------------------------------------------


def _generation_config(**fields) -> GenerationConfig:
    # The hub's generation_config.json at 995ad96: sampling defaults that greedy decoding never uses.
    hub = dict(bos_token_id=248044, do_sample=True, eos_token_id=[248046, 248044], pad_token_id=248044,
               temperature=1.0, top_k=20, top_p=0.95)
    return GenerationConfig(**{**hub, **fields})


def test_greedy_stops_at_the_eos_set_and_ignores_the_hubs_sampling_defaults():
    eos = {248046, 248044, 7}
    assert sampling.greedy(_generation_config(), eos, 1024) == {
        "temperature": 0.0, "max_tokens": 1024, "stop_token_ids": [7, 248044, 248046]}


def test_greedy_copies_a_hub_repetition_penalty_and_refuses_what_vllm_cannot_apply():
    assert sampling.greedy(_generation_config(repetition_penalty=1.05), {2}, 8)["repetition_penalty"] == 1.05
    with pytest.raises(ValueError, match="no_repeat_ngram_size"):
        sampling.greedy(_generation_config(no_repeat_ngram_size=3), {2}, 8)


@pytest.mark.parametrize("field", [{"num_beams": 2}, {"penalty_alpha": 0.6}, {"max_length": 50},
                                   {"encoder_repetition_penalty": 1.1}])
def test_greedy_refuses_any_hub_field_outside_the_allowlist(field):
    with pytest.raises(ValueError, match=next(iter(field))):
        sampling.greedy(_generation_config(**field), {2}, 8)


def test_the_engine_runs_the_residual_model_marlin_experts_and_no_prefix_cache():
    args = sampling.engine_args("/ckpt", {"k": 1, "threshold": 0.8, "temperature": 0.05}, enforce_eager=True)
    assert args["hf_overrides"] == {"architectures": ["ConnitoResidualQwen3_5MoeForCausalLM"],
                                    "connito_residual": {"k": 1, "threshold": 0.8, "temperature": 0.05}}
    assert (args["moe_backend"], args["generation_config"], args["enable_prefix_caching"]) == ("marlin", "vllm", False)
    assert args["worker_extension_cls"] == "connito_vllm.model.ResidualLoader" and args["enforce_eager"] is True
    with pytest.raises(ValueError, match="moe_backend"):
        sampling.engine_args("/ckpt", {"k": 1, "threshold": 0.8, "temperature": 0.05}, moe_backend="triton")


def test_the_model_refuses_an_engine_that_would_quantize_the_activations_or_lacks_the_residual_spec():
    spec = {"k": 1, "threshold": 0.8, "temperature": 0.05}
    assert sampling.residual_spec(SimpleNamespace(connito_residual=spec), "marlin") == spec
    for backend in ("auto", "triton", "cutlass"):
        with pytest.raises(ValueError, match="marlin"):
            sampling.residual_spec(SimpleNamespace(connito_residual=spec), backend)
    for hf_config in (SimpleNamespace(), SimpleNamespace(connito_residual={"k": 1, "threshold": 0.8})):
        with pytest.raises(ValueError, match=r"hf_overrides\['connito_residual'\].*temperature"):
            sampling.residual_spec(hf_config, "marlin")


def test_the_worker_env_drops_the_secrets_and_turns_off_the_flashinfer_sampler():
    env = sampling.worker_env({"HF_TOKEN": "x", "JEV_DIRECT_API_KEY": "y", "PATH": "/bin",
                               "CUDA_DEVICE_ORDER": "FASTEST_FIRST"})
    assert env == {"PATH": "/bin", **sampling.WORKER_ENV}
    assert env["VLLM_USE_FLASHINFER_SAMPLER"] == "0" and env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"


# --- swapping a shard in place ------------------------------------------------


def _params(layers=(0, 2)) -> dict[int, dict[str, torch.Tensor]]:
    return {layer: {"router": torch.zeros(K, H), "gate": torch.zeros(K, INTER, H), "up": torch.zeros(K, INTER, H),
                    "down": torch.zeros(K, H, INTER)} for layer in layers}


def test_load_into_copies_in_place_and_none_zeroes_every_tensor():
    params, block = _params(), _hf_block(torch.float32)
    pointers = {(layer, name): t.data_ptr() for layer, parts in params.items() for name, t in parts.items()}
    assert residual_math.load_into(params, _shard(block), K) == 8
    assert {(layer, name): t.data_ptr() for layer, parts in params.items() for name, t in parts.items()} == pointers
    assert torch.equal(params[2]["down"][1], block.residual_experts[1].down_proj.weight)
    assert residual_math.load_into(params, None, K) == 8
    assert all(not t.any() for parts in params.values() for t in parts.values())


def test_load_into_changes_nothing_when_a_shape_is_wrong():
    params = _params()
    residual_math.load_into(params, _shard(_hf_block(torch.float32)), K)
    before = {(layer, name): t.clone() for layer, parts in params.items() for name, t in parts.items()}
    # Layer 0 differs and comes first, so a loader that copies before checking would change it.
    bad = {k: v + 1 if ".layers.0." in k else v for k, v in _shard(_hf_block(torch.float32)).items()}
    for key, tensor, error in (("model.layers.2.mlp.residual_router.weight", torch.ones(K, H + 1), "residual_router"),
                               ("model.layers.2.mlp.residual_experts.2.down_proj.weight", torch.ones(H, INTER + 1),
                                "shapes differ")):
        with pytest.raises(ValueError, match=error):
            residual_math.load_into(params, {**bad, key: tensor}, K)
    assert all(torch.equal(params[layer][name], before[(layer, name)]) for layer, name in before)


def test_the_worker_keeps_the_cache_dirs_it_inherits_and_drops_only_secrets():
    # Those the run's start check reads (config.cache_dirs), as answers.run_worker passes them.
    caches = {name: f"/scratch/{name}" for name in (
        "VLLM_CACHE_ROOT", "VLLM_CONFIG_ROOT", "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "TMPDIR", "TRITON_CACHE_DIR",
        "TORCHINDUCTOR_CACHE_DIR", "CUDA_CACHE_PATH")}
    env = sampling.worker_env({**caches, "HF_TOKEN": "x", "BT_PW_WALLET": "y"})
    assert {name: env.get(name) for name in caches} == caches
    assert not {"HF_TOKEN", "BT_PW_WALLET"} & set(env)
