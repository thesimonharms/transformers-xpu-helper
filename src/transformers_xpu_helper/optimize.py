"""Model / optimizer preparation for Intel XPU training."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .amp import AmpBundle, resolve_amp
from .config import XPUTrainingConfig, ultra_255h_config
from .env import apply_runtime_env
from .hardware import DeviceInfo, detect_device
from .memory import apply_memory_fraction, empty_cache

logger = logging.getLogger(__name__)


@dataclass
class PreparedModel:
    """Result of :func:`prepare_for_training`."""

    model: Any
    optimizer: Any | None
    device_info: DeviceInfo
    config: XPUTrainingConfig
    amp: AmpBundle
    compiled: bool


def move_to_device(model: Any, device: str | DeviceInfo) -> Any:
    device_str = device.device if isinstance(device, DeviceInfo) else device
    return model.to(device_str)


def enable_gradient_checkpointing(model: Any) -> bool:
    """Enable gradient checkpointing when the model supports it."""
    fn = getattr(model, "gradient_checkpointing_enable", None)
    if callable(fn):
        try:
            fn(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            fn()
        enable_input = getattr(model, "enable_input_require_grads", None)
        if callable(enable_input):
            enable_input()
        return True

    logger.debug("Model does not expose gradient_checkpointing_enable()")
    return False


def maybe_compile(
    model: Any,
    *,
    enabled: bool = True,
    mode: str = "default",
    fullgraph: bool = False,
    backend: str = "inductor",
) -> tuple[Any, bool]:
    """Wrap ``model`` with ``torch.compile`` when available."""
    if not enabled:
        return model, False
    try:
        import torch
    except ImportError:
        return model, False

    compile_fn = getattr(torch, "compile", None)
    if not callable(compile_fn):
        logger.warning("torch.compile is unavailable in this PyTorch build")
        return model, False

    try:
        compiled = compile_fn(model, backend=backend, mode=mode, fullgraph=fullgraph)
        logger.info("Enabled torch.compile(backend=%s, mode=%s)", backend, mode)
        return compiled, True
    except Exception as exc:  # pragma: no cover - backend / platform specific
        logger.warning("torch.compile failed (%s); continuing in eager mode", exc)
        return model, False


def prepare_for_training(
    model: Any,
    optimizer: Any | None = None,
    *,
    config: XPUTrainingConfig | None = None,
    device_info: DeviceInfo | None = None,
    apply_env: bool = True,
    compile_model: bool | None = None,
) -> PreparedModel:
    """Move a Transformers (or torch) model onto XPU with 255H-oriented defaults.

    Steps:
      1. Apply hybrid-CPU thread / allocator env knobs
      2. Cap shared-memory XPU fraction
      3. Move model to XPU (or CPU when XPU is unavailable)
      4. Enable gradient checkpointing when configured
      5. Resolve BF16/FP16 autocast settings
      6. Optionally ``torch.compile`` the model
    """
    config = config or ultra_255h_config()
    info = device_info or detect_device(prefer="xpu", profile_hint="255H")

    if apply_env:
        apply_runtime_env(config, info.profile)
    apply_memory_fraction(config, info)

    if info.is_xpu:
        target = info.device
    else:
        target = "cpu"
        logger.info("XPU not available; placing model on CPU (profile=%s)", info.profile.name)

    try:
        model = move_to_device(model, target)
    except Exception as exc:
        logger.warning("Could not move model to %s (%s); using CPU", target, exc)
        model = move_to_device(model, "cpu")
        target = "cpu"

    if config.gradient_checkpointing:
        enable_gradient_checkpointing(model)

    amp = resolve_amp(config, info)

    should_compile = config.torch_compile if compile_model is None else compile_model
    # Only compile when we actually landed on XPU (or caller forced it).
    if should_compile and target == "cpu" and compile_model is not True:
        should_compile = False

    model, compiled = maybe_compile(
        model,
        enabled=should_compile,
        mode=config.compile_mode,
        fullgraph=config.compile_fullgraph,
    )

    if hasattr(model, "train"):
        model.train()

    return PreparedModel(
        model=model,
        optimizer=optimizer,
        device_info=info,
        config=config,
        amp=amp,
        compiled=compiled,
    )


def training_step(
    prepared: PreparedModel,
    batch: dict[str, Any] | tuple[Any, ...] | list[Any],
    *,
    criterion: Any | None = None,
) -> Any:
    """Run a single optimized training step (forward + backward).

    ``batch`` may be a Hugging Face-style dict of tensors or a ``(inputs, labels)``
    tuple. The optimizer is *not* stepped here so callers can accumulate gradients.
    """
    import torch

    model = prepared.model
    amp = prepared.amp
    device = prepared.device_info.device if prepared.device_info.is_xpu else "cpu"

    def _to_device(obj: Any) -> Any:
        if torch.is_tensor(obj):
            return obj.to(device, non_blocking=True)
        return obj

    if isinstance(batch, dict):
        batch = {k: _to_device(v) for k, v in batch.items()}
        with amp.autocast():
            outputs = model(**batch)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
    else:
        inputs, labels = batch[0], batch[1]
        inputs = _to_device(inputs)
        labels = _to_device(labels)
        with amp.autocast():
            outputs = model(inputs)
            if criterion is None:
                raise ValueError("criterion is required when batch is not a HF dict with labels")
            loss = criterion(outputs, labels)

    amp.backward(loss)
    return loss


def finalize_step(prepared: PreparedModel, *, step: int | None = None) -> None:
    """Optimizer step + optional cache clearing."""
    if prepared.optimizer is None:
        raise ValueError("PreparedModel.optimizer is required for finalize_step")
    prepared.amp.step(prepared.optimizer)
    prepared.amp.zero_grad(prepared.optimizer)
    if step is not None and prepared.config.empty_cache_steps > 0:
        if step % prepared.config.empty_cache_steps == 0:
            empty_cache(prepared.device_info)
