"""Tiny checkpoints of each model family, laid out like the real ones, for CPU tests."""

from __future__ import annotations

import inspect
import json
import signal
from dataclasses import replace
from pathlib import Path

import bittensor
import pytest
import torch
from bittensor.sp_core import Keypair
from safetensors.torch import save_file
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers.models.deepseek_v2 import DeepseekV2Config, DeepseekV2ForCausalLM
from transformers.models.qwen3_5_moe import Qwen3_5MoeConfig, Qwen3_5MoeForConditionalGeneration
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as qwen3_5_moe_modeling

from connito.prompt_validator.config import JUDGE_RULES_DATA
from connito.prompt_validator.stage_io import stage_path, write_rows
from connito.shared.modeling.families import FAMILIES


@pytest.fixture(autouse=True)
def sigterm_restored():
    """The prompt validator's main() installs its SIGTERM handler in the test process too: put the old one back."""
    before = signal.getsignal(signal.SIGTERM)
    yield
    signal.signal(signal.SIGTERM, before)


# A test that turns the miner-prompt judge on: the public release has no judge_rules.json, so it skips there.
needs_judge_rules = pytest.mark.skipif(not JUDGE_RULES_DATA, reason="the judge's rules, judge_rules.json, are not "
                                                                    "part of this checkout")


def pytest_configure(config):
    config.addinivalue_line("markers", "no_jev_price: the test sets no Jev price (conftest.jev_price)")


@pytest.fixture(autouse=True)
def jev_price(request, monkeypatch):
    """The price a deployment sets for Jev (jev.price_per_m has no default, and the client refuses to score without
    one): 1.0 USD per million input tokens, unless the test is marked no_jev_price."""
    if request.node.get_closest_marker("no_jev_price"):
        return
    from connito.prompt_validator import jev

    init = jev.JevClient.__init__

    def priced(self, cfg, *args, **kwargs):
        init(self, cfg if cfg.price_per_m is not None else cfg.model_copy(update={"price_per_m": 1.0}), *args,
             **kwargs)

    monkeypatch.setattr(jev.JevClient, "__init__", priced)


@pytest.fixture(autouse=True)
def allowed_gpus(monkeypatch):
    """The GPUs a deployment lets the prompt validator use (its compose file sets them): the GPU check refuses all
    of them when PV_ALLOWED_GPUS is unset. The GPU check's own tests pass their environment explicitly."""
    monkeypatch.setenv("PV_ALLOWED_GPUS", "2,3,4")


@pytest.fixture(autouse=True)
def torch_linear_attention(monkeypatch):
    """The tiny Qwen MoE's linear attention on its PyTorch path. With flash-linear-attention installed, transformers
    sends it to Triton kernels, which take CUDA tensors only: on a host with a GPU, these CPU tests would fail."""
    for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"):
        monkeypatch.setattr(qwen3_5_moe_modeling, name, inspect.unwrap(getattr(qwen3_5_moe_modeling, name)))


@pytest.fixture
def pv_chain(tmp_path):
    """A prompt validator config's chain: a plain hotkey under tmp_path, and ss58s that skip the chain lookup."""
    wallets = str(tmp_path / "wallets")
    keypair = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
    bittensor.Wallet(name="cold", hotkey="hot", path=wallets).hotkey_file.set_keypair(keypair, encrypt=False)
    return {"coldkey_name": "cold", "hotkey_name": "hot", "wallet_path": wallets,
            "coldkey_ss58": "cold", "hotkey_ss58": "hot"}


TINY_TEXT_CONFIG = dict(
    vocab_size=128,
    hidden_size=64,
    num_hidden_layers=4,
    layer_types=["linear_attention"] * 3 + ["full_attention"],
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    linear_key_head_dim=16,
    linear_value_head_dim=16,
    moe_intermediate_size=32,
    shared_expert_intermediate_size=32,
    num_experts=8,
    num_experts_per_tok=2,
    max_position_embeddings=128,
    # On, so a loader that leaves it on is caught.
    output_router_logits=True,
)
TINY_VISION_CONFIG = dict(
    depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64, num_position_embeddings=16,
)
# DeepSeek-V2-Lite's config.json in miniature: dense layer 0, MLA without a q LoRA, YaRN in the legacy form.
TINY_DEEPSEEK_V2_CONFIG = dict(
    architectures=["DeepseekV2ForCausalLM"],
    model_type="deepseek_v2",
    auto_map={"AutoModelForCausalLM": "modeling_deepseek.DeepseekV2ForCausalLM"},
    vocab_size=128,
    hidden_size=64,
    intermediate_size=96,
    moe_intermediate_size=32,
    num_hidden_layers=3,
    first_k_dense_replace=1,
    num_attention_heads=4,
    num_key_value_heads=4,
    kv_lora_rank=16,
    q_lora_rank=None,
    qk_nope_head_dim=16,
    qk_rope_head_dim=8,
    v_head_dim=16,
    n_routed_experts=8,
    n_shared_experts=2,
    num_experts_per_tok=2,
    topk_method="greedy",
    norm_topk_prob=False,
    max_position_embeddings=1280,
    rope_scaling=dict(
        type="yarn", factor=40, original_max_position_embeddings=32,
        beta_fast=32, beta_slow=1, mscale=0.707, mscale_all_dim=0.707,
    ),
    torch_dtype="bfloat16",
)


def _write_tokenizer(path: Path, tokenizer_config: dict) -> None:
    """A byte-level BPE `tokenizer.json` with no BOS template in its post-processor, as both real ones are."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(["hello world"], trainers.BpeTrainer(
        special_tokens=["<bos>", "<eos>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False,
    ))
    tokenizer.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))


@pytest.fixture()
def tiny_qwen_vlm() -> Qwen3_5MoeForConditionalGeneration:
    torch.manual_seed(0)
    config = Qwen3_5MoeConfig(text_config=TINY_TEXT_CONFIG, vision_config=TINY_VISION_CONFIG)
    return Qwen3_5MoeForConditionalGeneration(config).to(torch.bfloat16).eval()


@pytest.fixture()
def tiny_qwen_checkpoint(tmp_path: Path, tiny_qwen_vlm) -> Path:
    """The VLM saved with fused expert tensors and an `mtp.*` tensor, as the published checkpoint is."""
    path = tmp_path / "qwen3_5_moe"
    tiny_qwen_vlm.save_pretrained(path)
    # `save_pretrained` splits fused experts into per-expert keys; the real checkpoint keeps them fused.
    state = {name: tensor.contiguous() for name, tensor in tiny_qwen_vlm.state_dict().items()}
    state["mtp.fc.weight"] = torch.zeros(4, 4, dtype=torch.bfloat16)
    save_file(state, str(path / "model.safetensors"), metadata={"format": "pt"})
    _write_tokenizer(path, {"tokenizer_class": "Qwen2Tokenizer", "bos_token": None, "eos_token": "<eos>"})
    return path


@pytest.fixture()
def tiny_deepseek_v2_checkpoint(tmp_path: Path) -> Path:
    """Per-expert `mlp.experts.{i}.*_proj` tensors and the original config.json, as the published checkpoint is."""
    torch.manual_seed(0)
    model = DeepseekV2ForCausalLM(DeepseekV2Config(**TINY_DEEPSEEK_V2_CONFIG)).to(torch.bfloat16)
    path = tmp_path / "deepseek_v2"
    # transformers 5 splits its fused experts back into per-expert keys on save.
    model.save_pretrained(path)
    (path / "config.json").write_text(json.dumps(TINY_DEEPSEEK_V2_CONFIG))
    # The published tokenizer_config asks for BOS, but its tokenizer.json has no template that adds one.
    _write_tokenizer(path, {
        "tokenizer_class": "LlamaTokenizerFast", "add_bos_token": True, "bos_token": "<bos>", "eos_token": "<eos>",
    })
    return path


@pytest.fixture(params=sorted(FAMILIES))
def family_name(request, monkeypatch) -> str:
    """A family whose record points at its tiny checkpoint."""
    fixture = {"qwen3_5_moe": "tiny_qwen_checkpoint", "deepseek_v2": "tiny_deepseek_v2_checkpoint"}[request.param]
    path = request.getfixturevalue(fixture)
    monkeypatch.setitem(FAMILIES, request.param, replace(FAMILIES[request.param], model_path=str(path)))
    return request.param


@pytest.fixture
def write_selection():
    """write(pv, cycle_id, texts, selected): a cycle's miner prompts and the judge's selection, as those stages
    leave them (texts: prompt_id -> text)."""
    def write(pv, cycle_id: int, texts: dict[str, str], selected: list[str]) -> None:
        snapshot_dir = stage_path(pv.runs_dir, cycle_id, "snapshot").parent
        write_rows(snapshot_dir / "miner_prompts.jsonl", [
            {"prompt_id": p, "hotkey": "hk-m", "customer_task_id": "t-legal", "text": text, "content_hash": "0" * 64,
             "block": 100} for p, text in texts.items()])
        marker = stage_path(pv.runs_dir, cycle_id, "judge")
        write_rows(marker.parent / "selected.jsonl", [{"prompt_id": p, "hotkey": "hk-m"} for p in selected])
        write_rows(marker, [{"selected": len(selected)}])

    return write
