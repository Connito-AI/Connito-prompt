"""Connito's residual experts as plain tensors: a shard's layout, the gate math and the in-place swap.

Torch only, so the vLLM model and the CPU tests share it. Mirrors connito/shared/modeling/residual.py without
importing connito, whose pins (torch 2.10) the vLLM venv (torch 2.13) cannot install.
"""
from __future__ import annotations

import re

import torch
import torch.nn.functional as F

# The keys a shard holds, as install_residual_experts names them on the HF model.
_KEY = re.compile(
    r"model\.layers\.(\d+)\.mlp\."
    r"(?:residual_router\.weight|residual_experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight)"
)
_PROJ = {"gate_proj": "gate", "up_proj": "up", "down_proj": "down"}


def stack_shard(
    state_dict: dict[str, torch.Tensor], moe_layers: list[int], k: int,
) -> dict[int, dict[str, torch.Tensor]]:
    """{layer: {"router": [K, H], "gate"/"up": [K, I, H], "down": [K, H, I]}}; refuses anything but K on every layer."""
    # A missing tensor would leave the previous miner's in place.
    found: dict[int, dict[str, dict[int, torch.Tensor] | torch.Tensor]] = {}
    for name, tensor in state_dict.items():
        match = _KEY.fullmatch(name)
        if match is None:
            raise ValueError(f"not a residual key: {name}")
        layer, expert, proj = int(match[1]), match[2], match[3]
        if layer not in moe_layers:
            raise ValueError(f"layer {layer} has no MoE block: {name}")
        parts = found.setdefault(layer, {"gate": {}, "up": {}, "down": {}})
        if expert is None:
            parts["router"] = tensor
        else:
            parts[_PROJ[proj]][int(expert)] = tensor
    out = {}
    for layer in moe_layers:
        parts = found.get(layer)
        if parts is None or "router" not in parts:
            raise ValueError(f"layer {layer}: no residual router in the shard")
        stacked = {"router": parts["router"]}
        for kind in ("gate", "up", "down"):
            if sorted(parts[kind]) != list(range(k)):
                raise ValueError(f"layer {layer}: {kind}_proj for experts {sorted(parts[kind])}, expected 0..{k - 1}")
            if len({parts[kind][j].shape for j in range(k)}) != 1:
                raise ValueError(f"layer {layer}: {kind}_proj shapes differ between experts")
            stacked[kind] = torch.stack([parts[kind][j] for j in range(k)])
        if stacked["router"].shape[0] != k:
            raise ValueError(f"layer {layer}: router has {stacked['router'].shape[0]} rows, expected K={k}")
        out[layer] = stacked
    return out


# h = out + Σ_j q_j(x)·E_j(x), q = sigmoid((sigmoid(W_r·x) − A) / τ) in fp32, E_j = down_j(silu(gate_j·x) * up_j·x)
def residual_delta(
    x: torch.Tensor, router: torch.Tensor, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor,
    threshold: float, temperature: float,
) -> torch.Tensor:
    """Σ_j q_j(x)·E_j(x) for tokens x [T, H]: what the residual hook adds to the MoE block's output."""
    # fp32 even under bf16 autocast: near A, bf16 resolves r only in steps of ~0.004 (see residual.py).
    with torch.autocast(x.device.type, enabled=False):
        r = torch.sigmoid(F.linear(x.float(), router.float()))
        q = torch.sigmoid((r - threshold) / temperature)
    # A loop over K, summed in the hook's order: the task's K is 1, so a bmm would save nothing.
    h = 0
    for j in range(router.shape[0]):
        expert = F.linear(F.silu(F.linear(x, gate[j])) * F.linear(x, up[j]), down[j])
        h = h + q[:, j, None].to(x.dtype) * expert
    return h


@torch.no_grad()
def load_into(params: dict[int, dict[str, torch.Tensor]], shard: dict[str, torch.Tensor] | None, k: int) -> int:
    """Copy a shard into {layer: {part: tensor}} in place (None zeroes them); every shape is checked first."""
    # In place, so captured CUDA graphs read the new values; checked first, so a refused shard leaves the model whole.
    if shard is None:
        for parts in params.values():
            for tensor in parts.values():
                tensor.zero_()
        return sum(map(len, params.values()))
    stacked = stack_shard(shard, list(params), k)
    for layer, parts in params.items():
        for name, tensor in parts.items():
            if tensor.shape != stacked[layer][name].shape:
                raise ValueError(f"layer {layer} residual_{name}: shard {tuple(stacked[layer][name].shape)}, "
                                 f"model {tuple(tensor.shape)}")
    for layer, parts in params.items():
        for name, tensor in parts.items():
            tensor.copy_(stacked[layer][name].to(device=tensor.device, dtype=tensor.dtype))
    return sum(map(len, params.values()))
