"""Shared-memory budgeting helpers for Arc iGPUs (Arc 140T)."""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass

from .config import XPUTrainingConfig
from .hardware import DeviceInfo, detect_device

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MemoryBudget:
    total_ram_bytes: int
    host_reserve_bytes: int
    trainable_bytes: int
    memory_fraction: float
    shared_memory: bool

    @property
    def trainable_gib(self) -> float:
        return self.trainable_bytes / (1024**3)


def system_ram_bytes() -> int:
    """Best-effort total system RAM in bytes."""
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages * page_size)
    except (AttributeError, OSError, ValueError):
        pass
    try:
        import psutil  # type: ignore

        return int(psutil.virtual_memory().total)
    except Exception:
        # Sensible laptop default when introspection fails (32 GiB).
        return 32 * 1024**3


def estimate_budget(
    config: XPUTrainingConfig | None = None,
    device_info: DeviceInfo | None = None,
) -> MemoryBudget:
    """Compute how much memory training should claim on a shared-memory iGPU."""
    config = config or XPUTrainingConfig()
    info = device_info or detect_device(prefer="xpu", profile_hint="255H")

    total = system_ram_bytes()
    if info.total_memory_bytes:
        # Some drivers report a carved-out XPU heap; prefer the smaller bound.
        if info.profile.shared_memory:
            total = min(total, info.total_memory_bytes)
        else:
            total = info.total_memory_bytes

    reserve = int(config.host_reserve_gib * 1024**3)
    usable = max(0, total - reserve)
    trainable = int(usable * config.memory_fraction)

    return MemoryBudget(
        total_ram_bytes=total,
        host_reserve_bytes=reserve,
        trainable_bytes=trainable,
        memory_fraction=config.memory_fraction,
        shared_memory=info.profile.shared_memory,
    )


def suggest_batch_size(
    approx_param_count: int,
    *,
    seq_length: int = 512,
    config: XPUTrainingConfig | None = None,
    device_info: DeviceInfo | None = None,
    bytes_per_param: float | None = None,
    activation_factor: float = 8.0,
    min_batch: int = 1,
    max_batch: int = 32,
) -> int:
    """Heuristic micro-batch size for AdamW-style fine-tuning on Arc 140T.

    Assumptions (BF16 weights + FP32 master weights + Adam moments roughly):
    ~12-16 bytes/param for optimizer state + weights, plus activation overhead
    scaled by sequence length. This is intentionally conservative for shared DRAM.
    """
    config = config or XPUTrainingConfig()
    budget = estimate_budget(config, device_info)

    if bytes_per_param is None:
        # bf16 params (2) + fp32 master (4) + two adam moments (8) ≈ 14
        bytes_per_param = 14.0

    static = approx_param_count * bytes_per_param
    # Rough activation working set per sample.
    per_sample = approx_param_count * activation_factor * (seq_length / 512.0)

    remaining = max(0.0, budget.trainable_bytes - static)
    if per_sample <= 0:
        return min_batch

    raw = int(remaining // per_sample)
    # Prefer powers-of-two-ish sizes that accumulate cleanly.
    candidates = [b for b in (1, 2, 4, 8, 16, 32) if min_batch <= b <= max_batch]
    chosen = min_batch
    for b in candidates:
        if b <= max(1, raw):
            chosen = b
    return chosen


def apply_memory_fraction(
    config: XPUTrainingConfig | None = None,
    device_info: DeviceInfo | None = None,
) -> float | None:
    """Ask the XPU allocator to respect a memory fraction when the API exists."""
    config = config or XPUTrainingConfig()
    info = device_info or detect_device(prefer="xpu", profile_hint="255H")
    if not info.is_xpu:
        return None

    try:
        import torch

        fraction = config.memory_fraction
        # set_per_process_memory_fraction mirrors the CUDA API when present.
        setter = getattr(torch.xpu, "set_per_process_memory_fraction", None)
        if callable(setter):
            setter(fraction, info.index)
            logger.info("Set XPU memory fraction to %.2f", fraction)
            return fraction
    except Exception as exc:  # pragma: no cover - device-specific
        logger.debug("Could not set XPU memory fraction: %s", exc)
    return None


def empty_cache(device_info: DeviceInfo | None = None) -> None:
    info = device_info or detect_device(prefer="xpu", profile_hint="255H")
    try:
        import torch

        if info.is_xpu and hasattr(torch, "xpu"):
            torch.xpu.empty_cache()
            synchronize = getattr(torch.xpu, "synchronize", None)
            if callable(synchronize):
                synchronize()
        elif info.kind.value == "cuda":
            torch.cuda.empty_cache()
    except Exception as exc:  # pragma: no cover
        logger.debug("empty_cache failed: %s", exc)


def format_bytes(num: int | float) -> str:
    if num <= 0:
        return "0 B"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    exp = min(int(math.log(num, 1024)), len(units) - 1)
    value = num / (1024**exp)
    return f"{value:.2f} {units[exp]}"
