"""Hugging Face Transformers / Accelerate integration."""

from __future__ import annotations

import logging
from typing import Any, Literal

from .config import XPUTrainingConfig, ultra_255h_config
from .env import apply_runtime_env
from .hardware import DeviceInfo, detect_device
from .memory import (
    apply_memory_fraction,
    suggest_batch_size,
    suggest_vision_batch_size,
)
from .optimize import enable_gradient_checkpointing, maybe_compile, prepare_for_training

logger = logging.getLogger(__name__)

TaskKind = Literal["nlp", "vision", "auto"]


def _amp_flags(
    config: XPUTrainingConfig, device_info: DeviceInfo
) -> tuple[bool, bool]:
    use_bf16 = config.precision in {"bf16", "auto"} and config.amp_dtype_name.value == "bf16"
    use_fp16 = config.precision == "fp16" or (
        config.precision == "auto" and config.amp_dtype_name.value == "fp16"
    )
    # Force consistency with Arc guidance: auto → bf16, no fp16 scaler path.
    if config.precision == "auto":
        use_bf16 = device_info.profile.preferred_amp.value == "bf16"
        use_fp16 = device_info.profile.preferred_amp.value == "fp16"
    return use_bf16, use_fp16 and not use_bf16


def _base_training_arg_kwargs(
    output_dir: str,
    config: XPUTrainingConfig,
    device_info: DeviceInfo,
) -> dict[str, Any]:
    use_bf16, use_fp16 = _amp_flags(config, device_info)
    return {
        "output_dir": output_dir,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "bf16": use_bf16,
        "fp16": use_fp16,
        "gradient_checkpointing": config.gradient_checkpointing,
        "dataloader_num_workers": config.dataloader_num_workers,
        "dataloader_pin_memory": config.dataloader_pin_memory,
        "dataloader_persistent_workers": config.dataloader_persistent_workers
        and config.dataloader_num_workers > 0,
        "torch_compile": config.torch_compile,
        "torch_compile_backend": "inductor",
        "torch_compile_mode": config.compile_mode,
        "seed": config.seed,
        "remove_unused_columns": False,
        "report_to": [],
        "use_cpu": not device_info.is_xpu and device_info.kind.value == "cpu",
    }


def _instantiate_training_arguments(cls: Any, args_kwargs: dict[str, Any]) -> Any:
    """Construct TrainingArguments / Seq2SeqTrainingArguments with version fallbacks."""
    try:
        return cls(**args_kwargs)
    except TypeError:
        # Retry without torch_compile_* keys on older releases.
        for key in ("torch_compile", "torch_compile_backend", "torch_compile_mode", "use_cpu"):
            args_kwargs.pop(key, None)
        return cls(**args_kwargs)


def build_training_arguments(
    output_dir: str,
    config: XPUTrainingConfig | None = None,
    device_info: DeviceInfo | None = None,
    **overrides: Any,
) -> Any:
    """Create ``transformers.TrainingArguments`` pre-tuned for Intel XPU / 255H.

    Requires the optional ``transformers`` extra.
    """
    try:
        from transformers import TrainingArguments
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "transformers is required for build_training_arguments. "
            "Install with: pip install 'transformers-xpu-helper[transformers]'"
        ) from exc

    config = config or ultra_255h_config()
    info = device_info or detect_device(prefer="xpu", profile_hint="255H")
    apply_runtime_env(config, info.profile)
    apply_memory_fraction(config, info)

    args_kwargs = _base_training_arg_kwargs(output_dir, config, info)
    args_kwargs.update(overrides)
    return _instantiate_training_arguments(TrainingArguments, args_kwargs)


def build_seq2seq_training_arguments(
    output_dir: str,
    config: XPUTrainingConfig | None = None,
    device_info: DeviceInfo | None = None,
    **overrides: Any,
) -> Any:
    """Create ``transformers.Seq2SeqTrainingArguments`` with the same XPU defaults.

    Use this for TrOCR / VisionEncoderDecoder / other seq2seq Trainer loops.
    Requires the optional ``transformers`` extra.
    """
    try:
        from transformers import Seq2SeqTrainingArguments
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "transformers is required for build_seq2seq_training_arguments. "
            "Install with: pip install 'transformers-xpu-helper[transformers]'"
        ) from exc

    config = config or ultra_255h_config()
    info = device_info or detect_device(prefer="xpu", profile_hint="255H")
    apply_runtime_env(config, info.profile)
    apply_memory_fraction(config, info)

    args_kwargs = _base_training_arg_kwargs(output_dir, config, info)
    # Sensible OCR / seq2seq defaults; callers override freely.
    args_kwargs.setdefault("predict_with_generate", False)
    args_kwargs.update(overrides)
    return _instantiate_training_arguments(Seq2SeqTrainingArguments, args_kwargs)


def optimize_model_for_trainer(
    model: Any,
    *,
    config: XPUTrainingConfig | None = None,
    device_info: DeviceInfo | None = None,
) -> Any:
    """Prepare a HF model ahead of ``Trainer`` (checkpointing + optional compile).

    Prefer letting ``TrainingArguments(torch_compile=True)`` handle compile when
    using Trainer; this helper is for custom loops or older transformers.
    """
    config = config or ultra_255h_config()
    info = device_info or detect_device(prefer="xpu", profile_hint="255H")
    apply_runtime_env(config, info.profile)

    if config.gradient_checkpointing:
        enable_gradient_checkpointing(model)

    # Trainer moves the model itself; only compile here if explicitly useful.
    if config.torch_compile and not info.is_xpu:
        # Skip compile on CPU fallback to keep unit tests / CI light.
        return model

    if config.extra.get("compile_before_trainer"):
        model, _ = maybe_compile(
            model,
            enabled=True,
            mode=config.compile_mode,
            fullgraph=config.compile_fullgraph,
        )
    return model


def is_vision_encoder_decoder(model: Any) -> bool:
    """Detect VisionEncoderDecoder / TrOCR-style models for batch heuristics."""
    if type(model).__name__ == "VisionEncoderDecoderModel":
        return True
    cfg = getattr(model, "config", None)
    if cfg is None:
        return False
    model_type = getattr(cfg, "model_type", "") or ""
    if model_type in {"vision-encoder-decoder", "trocr"}:
        return True
    if getattr(cfg, "is_encoder_decoder", False) and getattr(cfg, "vision_config", None) is not None:
        return True
    return False


def recommend_for_model(
    model: Any,
    *,
    seq_length: int = 512,
    image_hw: tuple[int, int] = (384, 384),
    task: TaskKind = "auto",
    config: XPUTrainingConfig | None = None,
) -> XPUTrainingConfig:
    """Adjust batch size / grad accum based on parameter count and Arc memory.

    For vision / OCR encoder-decoders (TrOCR), uses image-area scaling instead of
    NLP sequence length. Always re-smoke the recommended micro-batch on device.
    """
    config = config or ultra_255h_config()
    params = sum(p.numel() for p in model.parameters())

    resolved: Literal["nlp", "vision"]
    if task == "auto":
        resolved = "vision" if is_vision_encoder_decoder(model) else "nlp"
    else:
        resolved = task

    if resolved == "vision":
        batch = suggest_vision_batch_size(
            params, image_hw=image_hw, config=config
        )
    else:
        batch = suggest_batch_size(params, seq_length=seq_length, config=config)

    # Keep effective batch roughly stable (~8-16) via accumulation.
    micro = config.per_device_train_batch_size
    target_effective = max(8, micro * config.gradient_accumulation_steps)
    accum = max(1, target_effective // batch)
    return config.with_updates(
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        gradient_accumulation_steps=accum,
    )


# Re-export prepare_for_training for a single import surface.
__all__ = [
    "build_seq2seq_training_arguments",
    "build_training_arguments",
    "is_vision_encoder_decoder",
    "optimize_model_for_trainer",
    "prepare_for_training",
    "recommend_for_model",
]
