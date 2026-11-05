"""The vLLM general plugin (entry point connito_residual): registers the residual model in every vLLM process."""
from __future__ import annotations

ARCH = "ConnitoResidualQwen3_5MoeForCausalLM"


def register() -> None:
    from vllm import ModelRegistry

    # By path, so loading the plugin does not import the model (and CUDA) in processes that never build it.
    if ARCH not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(ARCH, f"connito_vllm.model:{ARCH}")
