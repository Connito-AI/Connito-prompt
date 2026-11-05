"""vLLM 0.32's Qwen3_5MoeForCausalLM with Connito's residual experts added to every MoE block's whole output.

Qwen3NextSparseMoeBlock.forward (qwen3_next.py:247) returns routed plus gated shared experts, the HF hook's block.
The residual tensors start at zero and load_residual overwrites them in place, so captured CUDA graphs see each miner.
"""
from __future__ import annotations

import torch
from torch import nn
from vllm.model_executor.models.qwen3_5 import Qwen3_5MoeForCausalLM
from vllm.model_executor.models.qwen3_next import Qwen3NextSparseMoeBlock

from connito_vllm.residual_math import load_into, residual_delta
from connito_vllm.sampling import residual_spec

_PARTS = ("router", "gate", "up", "down")


class ResidualSparseMoeBlock(Qwen3NextSparseMoeBlock):
    """The block with the residual experts added to its whole output. Its tensors are attached by _attach."""

    def forward(self, hidden_states: torch.Tensor, already_sequence_parallel: bool = False) -> torch.Tensor:
        # First: a fused MoE kernel may write its output over its input.
        x = hidden_states.view(-1, hidden_states.shape[-1])
        delta = residual_delta(x, self.residual_router, self.residual_gate, self.residual_up, self.residual_down,
                               self.residual_threshold, self.residual_temperature)
        out = super().forward(hidden_states, already_sequence_parallel)
        return out + delta.view_as(out)


def _attach(block: Qwen3NextSparseMoeBlock, hidden: int, inter: int, spec: dict, dtype: torch.dtype) -> None:
    block.__class__ = ResidualSparseMoeBlock
    device = block.gate.weight.device
    k = spec["k"]

    def zeros(*shape, dtype=dtype):
        return nn.Parameter(torch.zeros(*shape, dtype=dtype, device=device), requires_grad=False)

    block.residual_router = zeros(k, hidden, dtype=torch.float32)  # fp32, as the HF router
    block.residual_gate = zeros(k, inter, hidden)
    block.residual_up = zeros(k, inter, hidden)
    block.residual_down = zeros(k, hidden, inter)
    block.residual_threshold = float(spec["threshold"])
    block.residual_temperature = float(spec["temperature"])


class ConnitoResidualQwen3_5MoeForCausalLM(Qwen3_5MoeForCausalLM):
    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        parallel = vllm_config.parallel_config
        if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
            raise ValueError("the residual experts are written for one GPU: tensor and pipeline parallel size 1")
        text = vllm_config.model_config.hf_text_config
        if text.hidden_act != "silu":
            raise ValueError(f"the residual experts are SwiGLU, but the model's act is {text.hidden_act}")
        # {"k": K, "threshold": A, "temperature": τ}: the snapshot meta's "residual", through hf_overrides.
        self.residual_spec = residual_spec(vllm_config.model_config.hf_config, vllm_config.kernel_config.moe_backend)
        self.residual_layers = []
        for index, layer in enumerate(self.model.layers):
            if isinstance(layer.mlp, Qwen3NextSparseMoeBlock):
                _attach(layer.mlp, text.hidden_size, text.moe_intermediate_size, self.residual_spec,
                        vllm_config.model_config.dtype)
                self.residual_layers.append(index)

    def load_weights(self, weights):
        loaded = super().load_weights(weights)
        # Not in the checkpoint: they start at zero, and each miner's come from load_residual.
        return loaded | {name for name, _ in self.named_parameters() if ".residual_" in name}

    def load_residual(self, shard: dict[str, torch.Tensor] | None) -> int:
        """Copy a miner's shard into the residual tensors in place, or zero them for None; the tensors written."""
        params = {layer: {part: getattr(self.model.layers[layer].mlp, f"residual_{part}") for part in _PARTS}
                  for layer in self.residual_layers}
        return load_into(params, shard, self.residual_spec["k"])


class ResidualLoader:
    """vLLM worker extension: load a miner's residual shard, or zero it, in place in the running model."""

    def load_residual(self, path: str | None) -> int:
        from safetensors.torch import load_file

        model = self.model_runner.model
        target = model if hasattr(model, "load_residual") else next(
            m for m in model.modules() if hasattr(type(m), "load_residual"))
        return target.load_residual(None if path is None else load_file(path, device="cpu"))

    def peak_gib(self) -> float:
        """The GPU memory peak since the last call, KV cache included."""
        import torch

        peak = torch.cuda.max_memory_allocated() / 2**30
        torch.cuda.reset_peak_memory_stats()
        return peak
