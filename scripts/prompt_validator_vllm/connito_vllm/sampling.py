"""The engine and decoding settings under which vLLM answers as the HF answers stage does: greedy, same eos set."""
from __future__ import annotations

from connito_vllm.plugin import ARCH

# The only hub generation_config fields allowed off their defaults: ids and sampling settings HF's greedy never uses,
# and repetition_penalty, copied. Any other field could change a greedy answer in a way vLLM does not mirror.
_ALLOWED = {"bos_token_id", "eos_token_id", "pad_token_id", "do_sample", "temperature", "top_k", "top_p",
            "transformers_version", "repetition_penalty"}
_COPIED = {"repetition_penalty": "repetition_penalty"}
RESIDUAL_KEYS = ("k", "threshold", "temperature")
# Set by a vLLM process before it imports vllm. No FlashInfer sampler: its kernels compile at start-up with nvcc,
# which a host without the CUDA toolkit lacks, and greedy decoding never uses it.
WORKER_ENV = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "VLLM_NO_USAGE_STATS": "1", "HF_HUB_OFFLINE": "1",
              "VLLM_USE_FLASHINFER_SAMPLER": "0"}
SECRETS = ("JEV_DIRECT_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "WANDB_API_KEY")


def worker_env(environ: dict[str, str]) -> dict[str, str]:
    """environ without the secrets (bittensor's BT_PW_* passwords too), with WORKER_ENV."""
    return {k: v for k, v in environ.items() if k not in SECRETS and not k.startswith("BT_PW_")} | WORKER_ENV


def greedy(generation_config, eos: set[int], max_new_tokens: int) -> dict:
    """SamplingParams kwargs: greedy, stopping at any of eos, with the hub fields that change a greedy answer."""
    refused = sorted(set(generation_config.to_diff_dict()) - _ALLOWED)
    if refused:
        raise ValueError(f"the generation config sets {refused}, which vLLM cannot apply as HF does")
    copied = {name: getattr(generation_config, field) for field, name in _COPIED.items()
              if getattr(generation_config, field, None) not in (None, 1.0)}
    return {"temperature": 0.0, "max_tokens": max_new_tokens, "stop_token_ids": sorted(eos), **copied}


def residual_spec(hf_config, moe_backend: str) -> dict:
    """The residual gate from hf_overrides, once the engine is known to run the experts W8A16, as the validator."""
    # An explicit backend never falls back; "auto" or "triton" on an fp8 GPU would quantize the activations too.
    if moe_backend != "marlin":
        raise ValueError(f"moe_backend is {moe_backend!r}: only 'marlin' runs the fp8 experts W8A16")
    spec = getattr(hf_config, "connito_residual", None)
    if not isinstance(spec, dict) or any(key not in spec for key in RESIDUAL_KEYS):
        raise ValueError(f"hf_overrides['connito_residual'] must hold {RESIDUAL_KEYS}, got {spec!r}")
    return spec


def engine_args(checkpoint: str, residual: dict, **overrides) -> dict:
    """LLM kwargs for an export_fp8row checkpoint with the snapshot's residual gate {k, threshold, temperature}."""
    if "moe_backend" in overrides:
        raise ValueError("moe_backend is fixed to 'marlin': the validator's experts are W8A16")
    return {
        "model": checkpoint, "tokenizer": checkpoint, "dtype": "bfloat16", "seed": 0,
        "hf_overrides": {"architectures": [ARCH], "connito_residual": residual},
        # The hub's sampling defaults never reach a request: greedy() sets what HF's greedy uses.
        "generation_config": "vllm",
        # W8A16 on any GPU: the experts' fp8 weights dequantized in the kernel, activations left in bf16.
        "moe_backend": "marlin",
        "enable_prefix_caching": False,  # every model gets the same prompts: a cached prefix holds the last's KV
        "worker_extension_cls": "connito_vllm.model.ResidualLoader",
        **overrides,
    }
