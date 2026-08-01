"""Automatic mixed precision helpers for Intel XPU training."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any

from .config import XPUTrainingConfig
from .hardware import AmpPolicy, DeviceInfo, detect_device

logger = logging.getLogger(__name__)


@dataclass
class AmpBundle:
    """Resolved AMP settings + optional GradScaler."""

    enabled: bool
    dtype: Any  # torch.dtype
    device_type: str
    use_grad_scaler: bool
    scaler: Any | None = None

    def autocast(self):
        if not self.enabled:
            return nullcontext()
        import torch

        return torch.autocast(device_type=self.device_type, dtype=self.dtype, enabled=True)

    def backward(self, loss: Any) -> None:
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def step(self, optimizer: Any) -> None:
        if self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()

    def zero_grad(self, optimizer: Any, set_to_none: bool = True) -> None:
        optimizer.zero_grad(set_to_none=set_to_none)


def _torch_dtype(policy: AmpPolicy):
    import torch

    return {
        AmpPolicy.BF16: torch.bfloat16,
        AmpPolicy.FP16: torch.float16,
        AmpPolicy.FP32: torch.float32,
    }[policy]


def resolve_amp(
    config: XPUTrainingConfig | None = None,
    device_info: DeviceInfo | None = None,
) -> AmpBundle:
    """Create AMP settings that are safe on Arc client GPUs.

    Core Ultra 7 255H / Arc 140T guidance:
    - Prefer BF16 autocast (XMX-friendly, no GradScaler required).
    - Avoid FP16 + GradScaler: client Arc historically lacks FP64 needed by the scaler.
    """
    import torch

    config = config or XPUTrainingConfig()
    info = device_info or detect_device(prefer="xpu", profile_hint="255H")

    policy = config.amp_dtype_name
    if config.precision == "auto":
        policy = info.profile.preferred_amp

    device_type = info.kind.value if info.kind.value in {"xpu", "cuda", "cpu"} else "cpu"
    enabled = policy is not AmpPolicy.FP32
    dtype = _torch_dtype(policy)

    use_scaler = bool(config.use_grad_scaler and policy is AmpPolicy.FP16)
    if use_scaler and not info.profile.supports_grad_scaler:
        logger.warning(
            "GradScaler requested but %s does not reliably support FP64; disabling scaler. "
            "Prefer BF16 autocast instead.",
            info.profile.name,
        )
        use_scaler = False

    scaler = None
    if use_scaler:
        # torch.amp.GradScaler is the unified API (PyTorch 2.x).
        try:
            scaler = torch.amp.GradScaler(device=device_type, enabled=True)
        except TypeError:
            scaler = torch.cuda.amp.GradScaler(enabled=True)  # type: ignore[attr-defined]

    return AmpBundle(
        enabled=enabled,
        dtype=dtype,
        device_type=device_type,
        use_grad_scaler=use_scaler,
        scaler=scaler,
    )


@contextmanager
def autocast_context(
    config: XPUTrainingConfig | None = None,
    device_info: DeviceInfo | None = None,
    amp: AmpBundle | None = None,
) -> Iterator[AmpBundle]:
    """Context manager yielding an :class:`AmpBundle` under autocast."""
    bundle = amp or resolve_amp(config, device_info)
    with bundle.autocast():
        yield bundle
