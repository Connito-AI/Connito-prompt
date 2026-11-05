"""Write the validator's Qwen3.6 base as a vLLM checkpoint: its own fp8 routed experts, everything else bf16.

Runs in the repo venv. The experts come from Connito's loader (get_base_model, fp8 scope "experts"), so the fp8 bytes
and per-row scales are _to_fp8's. Every other tensor is copied unchanged from the hub shards; vision and MTP are
dropped. Writes a sha256 per tensor and per other file to manifest.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

from connito.shared.config import ModelCfg
from connito.shared.modeling.families import FAMILIES
from connito.shared.modeling.mycelia import get_base_model
from connito.shared.modeling.quantization import FP8FusedExperts

FAMILY = "qwen3_5_moe"
_EXPERTS = re.compile(r"model\.language_model\.(layers\.\d+\.mlp\.experts)\.(gate_up_proj|down_proj)")
_DROP = ("model.visual.", "mtp.")
_FP8_WEIGHTS = {"num_bits": 8, "type": "float", "strategy": "channel", "symmetric": True, "dynamic": False}
# vLLM 0.32 has no compressed-tensors fp8 W8A16 MoE scheme (weights-only channel schemes must be int and packed), so
# the activations are declared fp8 per token. moe_backend="marlin" (connito_vllm.sampling) then runs the experts
# W8A16: Marlin never quantizes activations.
QUANTIZATION_CONFIG = {
    "quant_method": "compressed-tensors",
    "format": "float-quantized",
    "quantization_status": "compressed",
    "config_groups": {"routed_experts": {
        "targets": [r"re:.*mlp\.experts\.\d+\.(gate|up|down)_proj$"],
        "weights": _FP8_WEIGHTS,
        "input_activations": {"num_bits": 8, "type": "float", "strategy": "token", "symmetric": True, "dynamic": True},
        "output_activations": None,
        "format": "float-quantized",
    }},
    "ignore": ["lm_head"],
    "kv_cache_scheme": None,
}


def expert_tensors(prefix: str, proj: str, values: torch.Tensor, scale: torch.Tensor) -> dict[str, torch.Tensor]:
    """One fused projection [E, rows, cols] as per-expert keys; gate_up_proj's rows [0, I) are gate, [I, 2I) up."""
    if proj == "gate_up_proj":
        inter = values.shape[1] // 2
        parts = {"gate_proj": slice(0, inter), "up_proj": slice(inter, None)}
    else:
        parts = {proj: slice(None)}
    out = {}
    for e in range(values.shape[0]):
        for name, rows in parts.items():
            out[f"{prefix}.{e}.{name}.weight"] = values[e, rows].contiguous()
            # The value HF multiplies by: its dequant is bf16(fp8) * bf16(scale).
            out[f"{prefix}.{e}.{name}.weight_scale"] = scale[e, rows].to(torch.bfloat16).float().contiguous()
    return out


def _sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.reshape(-1).view(torch.uint8).numpy()).hexdigest()


def file_hashes(out: Path) -> dict[str, str]:
    """sha256 of every file but the weights and the manifest: config, index, tokenizer, chat template, generation."""
    return {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(out.iterdir())
            if f.is_file() and not f.name.endswith(".safetensors") and f.name != "manifest.json"}


def _hub_dir() -> Path:
    fam = FAMILIES[FAMILY]
    local = Path(fam.model_path)
    if local.is_dir():
        return local
    return Path(snapshot_download(fam.model_path, revision=fam.revision, local_files_only=True))


def export(out: Path) -> dict:
    """Write the checkpoint into out and return its manifest."""
    fam = FAMILIES[FAMILY]
    model = get_base_model(SimpleNamespace(role="validator", model=ModelCfg(family=FAMILY)))
    modules = dict(model.named_modules())
    hub = _hub_dir()
    index_path = hub / "model.safetensors.index.json"
    shards = sorted(set(json.loads(index_path.read_text())["weight_map"].values())) if index_path.exists() else [
        "model.safetensors"]
    out.mkdir(parents=True, exist_ok=True)

    weight_map, hashes, total = {}, {}, 0
    for shard in shards:
        tensors = {}
        with safe_open(hub / shard, framework="pt") as f:
            for key in f.keys():
                if key.startswith(_DROP):
                    continue
                match = _EXPERTS.fullmatch(key)
                if match is None:
                    tensors[key] = f.get_tensor(key)
                    continue
                experts = modules[f"model.{match[1]}"]
                if not isinstance(experts, FP8FusedExperts):
                    raise TypeError(f"{key}: the loader did not store it in fp8 ({type(experts).__name__})")
                proj = match[2]
                tensors.update(expert_tensors(f"model.language_model.{match[1]}", proj,
                                              getattr(experts, proj), getattr(experts, f"{proj}_scale")))
        if not tensors:
            continue
        save_file(tensors, str(out / shard), metadata={"format": "pt"})
        for name, tensor in tensors.items():
            weight_map[name] = shard
            hashes[name] = _sha256(tensor)
            total += tensor.numel() * tensor.element_size()
        print(f"{shard}: {len(tensors)} tensors", flush=True)

    for src in hub.iterdir():
        if src.is_file() and not src.name.endswith(".safetensors") and src.name != index_path.name:
            shutil.copyfile(src, out / src.name)
    config = json.loads((hub / "config.json").read_text())
    config["quantization_config"] = QUANTIZATION_CONFIG
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    index = {"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))}
    (out / index_path.name).write_text(json.dumps(index, indent=2) + "\n")
    manifest = {"repo": fam.model_path, "revision": fam.revision, "quantization_config": QUANTIZATION_CONFIG,
                "files": file_hashes(out), "tensors": dict(sorted(hashes.items()))}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="the checkpoint folder to write")
    manifest = export(parser.parse_args().out)
    print(f"{len(manifest['tensors'])} tensors, revision {manifest['revision']}", flush=True)


if __name__ == "__main__":
    main()
