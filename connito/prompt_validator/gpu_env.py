"""The GPU a prompt-validator process may use. Standard library only, so the entry point checks it before the stage
imports: fla starts CUDA as it is imported, so the GPU must be chosen by the process environment."""
from __future__ import annotations

import os
import sys
from collections.abc import Mapping

ALLOWED_ENV = "PV_ALLOWED_GPUS"  # the GPUs this host lets the prompt validator use, by nvidia-smi's index: "2,3"


def allowed_gpus(env: Mapping[str, str]) -> set[str] | None:
    """The indexes ALLOWED_ENV names, or None if it is unset, empty or not plain indexes: then no GPU is allowed."""
    gpus = {gpu.strip() for gpu in env.get(ALLOWED_ENV, "").split(",") if gpu.strip()}
    return gpus if gpus and all(gpu.isdigit() and str(int(gpu)) == gpu for gpu in gpus) else None


def require_gpu_env(env: Mapping[str, str] = os.environ, in_container: bool | None = None) -> None:
    """Exit with status 2 unless the process sees one GPU of PV_ALLOWED_GPUS, by nvidia-smi's index.

    Unset, nothing is allowed: a host that shares its GPUs must name the ones this process may take. In a container
    Docker passes in only the host GPU PV_HOST_GPU names, and inside it is device 0: there, CUDA_VISIBLE_DEVICES=0
    with PV_HOST_GPU one of PV_ALLOWED_GPUS. A deployment sets all three."""
    visible = env.get("CUDA_VISIBLE_DEVICES")
    allowed = allowed_gpus(env)
    if allowed is None:
        sys.stderr.write(
            f"prompt validator: set {ALLOWED_ENV} to the GPUs this process may use, by nvidia-smi's index (such as "
            f"{ALLOWED_ENV}=2,3), in the process environment (got {env.get(ALLOWED_ENV)!r}).\n"
        )
        raise SystemExit(2)
    if env.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID" and visible in allowed:
        return
    if in_container is None:
        in_container = os.path.exists("/.dockerenv")
    if in_container and visible == "0" and env.get("PV_HOST_GPU") in allowed:
        return
    sys.stderr.write(
        f"prompt validator: set CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES to one of the GPUs {ALLOWED_ENV} "
        f"allows ({', '.join(sorted(allowed, key=int))}), as nvidia-smi numbers them (got {visible!r}). Set them in "
        "the process environment: CUDA starts on import, so a .env is read too late.\n"
    )
    raise SystemExit(2)
