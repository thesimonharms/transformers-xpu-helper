"""Environment and runtime thread tuning for hybrid Intel CPUs."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping

from .config import XPUTrainingConfig
from .hardware import HardwareProfile, detect_device

logger = logging.getLogger(__name__)


_DEFAULT_ENV = {
    # Reduce oneDNN verbose spam unless the user opts in.
    "ONEDNN_VERBOSE": "0",
    # Prefer performance libraries that play well with Level Zero / SYCL.
    "IPEX_TILE_AS_DEVICE": "0",
}


def apply_runtime_env(
    config: XPUTrainingConfig | None = None,
    profile: HardwareProfile | None = None,
    *,
    extra: Mapping[str, str] | None = None,
    overwrite: bool = False,
) -> dict[str, str]:
    """Set process environment variables that improve XPU training throughput.

    Safe to call multiple times. Existing variables are left alone unless
    ``overwrite=True``.
    """
    if config is None:
        config = XPUTrainingConfig()
    if profile is None:
        profile = detect_device(prefer="xpu", profile_hint="255H").profile

    desired: dict[str, str] = {
        **_DEFAULT_ENV,
        "OMP_NUM_THREADS": str(config.omp_num_threads),
        "MKL_NUM_THREADS": str(config.mkl_num_threads),
        "OPENBLAS_NUM_THREADS": str(config.omp_num_threads),
        "NUMEXPR_NUM_THREADS": str(config.omp_num_threads),
        # Keep tokenizers from oversubscribing the hybrid core complex.
        "TOKENIZERS_PARALLELISM": "false",
    }
    if profile.shared_memory:
        # Hint to allocators that the process should leave headroom for the iGPU.
        desired.setdefault("PYTORCH_XPU_ALLOC_CONF", "expandable_segments:True")

    if extra:
        desired.update({k: str(v) for k, v in extra.items()})

    applied: dict[str, str] = {}
    for key, value in desired.items():
        if key in os.environ and not overwrite:
            applied[key] = os.environ[key]
            continue
        os.environ[key] = value
        applied[key] = value

    try:
        import torch

        torch.set_num_threads(config.omp_num_threads)
        if hasattr(torch, "set_num_interop_threads"):
            # Interop threads > 1 can oversubscribe small hybrid CPUs.
            interop = max(1, min(2, config.omp_num_threads // 4 or 1))
            try:
                torch.set_num_interop_threads(interop)
            except RuntimeError:
                # May already have been set after parallel work started.
                pass
    except ImportError:
        pass

    logger.debug("Applied XPU runtime env: %s", applied)
    return applied


def describe_env() -> dict[str, str]:
    keys = [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TOKENIZERS_PARALLELISM",
        "PYTORCH_XPU_ALLOC_CONF",
        "ZE_AFFINITY_MASK",
        "ONEAPI_DEVICE_SELECTOR",
    ]
    return {k: os.environ[k] for k in keys if k in os.environ}
