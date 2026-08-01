"""Lightweight throughput / memory profiling for XPU training loops."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from .hardware import DeviceInfo, detect_device
from .memory import format_bytes

logger = logging.getLogger(__name__)


@dataclass
class StepStats:
    steps: int = 0
    total_seconds: float = 0.0
    samples: int = 0
    peak_memory_bytes: int = 0

    @property
    def samples_per_second(self) -> float:
        if self.total_seconds <= 0:
            return 0.0
        return self.samples / self.total_seconds

    @property
    def steps_per_second(self) -> float:
        if self.total_seconds <= 0:
            return 0.0
        return self.steps / self.total_seconds


def _default_device_info() -> DeviceInfo:
    return detect_device(prefer="xpu", profile_hint="255H")


@dataclass
class Profiler:
    device_info: DeviceInfo = field(default_factory=_default_device_info)
    stats: StepStats = field(default_factory=StepStats)

    def _current_memory(self) -> int:
        try:
            import torch

            if self.device_info.is_xpu and hasattr(torch, "xpu"):
                return int(torch.xpu.max_memory_allocated(self.device_info.index))
            if self.device_info.kind.value == "cuda":
                return int(torch.cuda.max_memory_allocated(self.device_info.index))
        except Exception:
            return 0
        return 0

    def _synchronize(self) -> None:
        try:
            import torch

            if self.device_info.is_xpu and hasattr(torch.xpu, "synchronize"):
                torch.xpu.synchronize()
            elif self.device_info.kind.value == "cuda":
                torch.cuda.synchronize()
        except Exception:
            pass

    @contextmanager
    def step(self, batch_size: int = 1) -> Iterator[None]:
        self._synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._synchronize()
            elapsed = time.perf_counter() - start
            self.stats.steps += 1
            self.stats.total_seconds += elapsed
            self.stats.samples += batch_size
            peak = self._current_memory()
            if peak > self.stats.peak_memory_bytes:
                self.stats.peak_memory_bytes = peak

    def summary(self) -> dict[str, Any]:
        return {
            "steps": self.stats.steps,
            "samples": self.stats.samples,
            "seconds": round(self.stats.total_seconds, 4),
            "samples_per_second": round(self.stats.samples_per_second, 3),
            "steps_per_second": round(self.stats.steps_per_second, 3),
            "peak_memory": format_bytes(self.stats.peak_memory_bytes),
            "device": self.device_info.device,
            "device_name": self.device_info.name,
            "profile": self.device_info.profile.name,
        }

    def log_summary(self) -> None:
        logger.info("XPU profile: %s", self.summary())
