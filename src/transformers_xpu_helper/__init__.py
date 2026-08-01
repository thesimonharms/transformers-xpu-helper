"""transformers-xpu-helper — speed up Hugging Face Transformers training on Intel XPUs.

Tuned by default for the Intel Core Ultra 7 255H (Arrow Lake-H) with Arc Graphics 140T.
Uses native PyTorch XPU APIs (PyTorch 2.5+); Intel Extension for PyTorch is EOL.
"""

from __future__ import annotations

from .amp import AmpBundle, autocast_context, resolve_amp
from .config import XPUTrainingConfig, config_from_profile, ultra_255h_config
from .dataloader import dataloader_kwargs, wrap_dataloader
from .env import apply_runtime_env, describe_env
from .hardware import (
    CORE_ULTRA_7_255H,
    DeviceInfo,
    DeviceKind,
    HardwareProfile,
    detect_device,
    resolve_profile,
)
from .memory import (
    MemoryBudget,
    apply_memory_fraction,
    empty_cache,
    estimate_budget,
    format_bytes,
    suggest_batch_size,
    suggest_vision_batch_size,
)
from .optimize import (
    PreparedModel,
    enable_gradient_checkpointing,
    finalize_step,
    maybe_compile,
    prepare_for_training,
    training_step,
)
from .profiling import Profiler, StepStats

__version__ = "0.2.0"

__all__ = [
    "AmpBundle",
    "CORE_ULTRA_7_255H",
    "DeviceInfo",
    "DeviceKind",
    "HardwareProfile",
    "MemoryBudget",
    "PreparedModel",
    "Profiler",
    "StepStats",
    "XPUTrainingConfig",
    "apply_memory_fraction",
    "apply_runtime_env",
    "autocast_context",
    "config_from_profile",
    "dataloader_kwargs",
    "describe_env",
    "detect_device",
    "empty_cache",
    "enable_gradient_checkpointing",
    "estimate_budget",
    "finalize_step",
    "format_bytes",
    "maybe_compile",
    "prepare_for_training",
    "resolve_amp",
    "resolve_profile",
    "suggest_batch_size",
    "suggest_vision_batch_size",
    "training_step",
    "ultra_255h_config",
    "wrap_dataloader",
    "__version__",
]


def __getattr__(name: str):
    # Lazy optional exports that need the transformers extra.
    if name in {
        "build_seq2seq_training_arguments",
        "build_training_arguments",
        "is_vision_encoder_decoder",
        "optimize_model_for_trainer",
        "recommend_for_model",
    }:
        from . import trainer as _trainer

        return getattr(_trainer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
