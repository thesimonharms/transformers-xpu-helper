"""Hardware detection and profiles for Intel XPU training targets."""

from __future__ import annotations

import os
import platform
import re
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Any


class DeviceKind(str, Enum):
    XPU = "xpu"
    CUDA = "cuda"
    CPU = "cpu"
    MPS = "mps"


class AmpPolicy(str, Enum):
    """Preferred mixed-precision policy for a given device class."""

    BF16 = "bf16"
    FP16 = "fp16"
    FP32 = "fp32"


@dataclass(frozen=True)
class HardwareProfile:
    """Static tuning knobs for a known Intel SKU / GPU class."""

    name: str
    codename: str
    cpu_p_cores: int
    cpu_e_cores: int
    cpu_lp_e_cores: int
    gpu_name: str
    xe_cores: int
    shared_memory: bool
    preferred_amp: AmpPolicy
    # Arc client GPUs historically lack robust FP64; GradScaler needs it for FP16.
    supports_grad_scaler: bool
    # Fraction of *system* RAM the iGPU should be allowed to claim for tensors.
    default_memory_fraction: float
    # Host OS / Python / dataloader reserve (GiB) when budgeting shared memory.
    host_reserve_gib: float
    recommended_dataloader_workers: int
    recommended_omp_num_threads: int
    torch_compile_default: bool
    gradient_checkpointing_default: bool
    notes: tuple[str, ...] = ()

    @property
    def logical_cpus(self) -> int:
        return self.cpu_p_cores + self.cpu_e_cores + self.cpu_lp_e_cores


# Arrow Lake-H: Core Ultra 7 255H + Arc Graphics 140T (8 Xe cores, shared DRAM).
CORE_ULTRA_7_255H = HardwareProfile(
    name="Intel Core Ultra 7 255H",
    codename="Arrow Lake-H",
    cpu_p_cores=6,
    cpu_e_cores=8,
    cpu_lp_e_cores=2,
    gpu_name="Intel Arc Graphics 140T",
    xe_cores=8,
    shared_memory=True,
    preferred_amp=AmpPolicy.BF16,
    supports_grad_scaler=False,
    default_memory_fraction=0.55,
    host_reserve_gib=6.0,
    # Prefer a modest worker count so P-cores stay free for the host training loop.
    recommended_dataloader_workers=4,
    # Cap CPU math threads below total cores to avoid fighting the XPU runtime / Level Zero.
    recommended_omp_num_threads=8,
    torch_compile_default=True,
    gradient_checkpointing_default=True,
    notes=(
        "Arc 140T uses system memory; keep batch sizes modest and enable checkpointing.",
        "Prefer BF16 autocast without GradScaler on client Arc GPUs.",
        "torch.compile(inductor) is supported on native PyTorch XPU (2.5+).",
        "IPEX is EOL; use stock PyTorch with the xpu index wheels.",
    ),
)

# Nearby SKUs share the same Arc 140T and hybrid layout; reuse the 255H knobs.
CORE_ULTRA_7_265H = HardwareProfile(
    **{
        **CORE_ULTRA_7_255H.__dict__,
        "name": "Intel Core Ultra 7 265H",
    }
)

CORE_ULTRA_9_285H = HardwareProfile(
    **{
        **CORE_ULTRA_7_255H.__dict__,
        "name": "Intel Core Ultra 9 285H",
    }
)

GENERIC_ARC_CLIENT = HardwareProfile(
    name="Intel Arc Client GPU",
    codename="client-arc",
    cpu_p_cores=4,
    cpu_e_cores=4,
    cpu_lp_e_cores=0,
    gpu_name="Intel Arc Graphics",
    xe_cores=0,
    shared_memory=True,
    preferred_amp=AmpPolicy.BF16,
    supports_grad_scaler=False,
    default_memory_fraction=0.50,
    host_reserve_gib=6.0,
    recommended_dataloader_workers=2,
    recommended_omp_num_threads=4,
    torch_compile_default=True,
    gradient_checkpointing_default=True,
    notes=("Generic client Arc profile: shared memory, BF16 preferred, no GradScaler.",),
)

GENERIC_XPU = HardwareProfile(
    name="Intel XPU",
    codename="generic-xpu",
    cpu_p_cores=8,
    cpu_e_cores=0,
    cpu_lp_e_cores=0,
    gpu_name="Intel XPU",
    xe_cores=0,
    shared_memory=False,
    preferred_amp=AmpPolicy.BF16,
    supports_grad_scaler=True,
    default_memory_fraction=0.90,
    host_reserve_gib=4.0,
    recommended_dataloader_workers=4,
    recommended_omp_num_threads=8,
    torch_compile_default=True,
    gradient_checkpointing_default=False,
    notes=("Generic discrete / data-center XPU defaults.",),
)

CPU_ONLY = HardwareProfile(
    name="CPU",
    codename="cpu",
    cpu_p_cores=os.cpu_count() or 4,
    cpu_e_cores=0,
    cpu_lp_e_cores=0,
    gpu_name="none",
    xe_cores=0,
    shared_memory=False,
    preferred_amp=AmpPolicy.BF16,
    supports_grad_scaler=False,
    default_memory_fraction=0.0,
    host_reserve_gib=2.0,
    recommended_dataloader_workers=max(1, (os.cpu_count() or 4) // 2),
    recommended_omp_num_threads=os.cpu_count() or 4,
    torch_compile_default=False,
    gradient_checkpointing_default=False,
    notes=("CPU fallback profile.",),
)

_SKU_PATTERNS: list[tuple[re.Pattern[str], HardwareProfile]] = [
    (re.compile(r"ultra\s*7\s*255h", re.I), CORE_ULTRA_7_255H),
    (re.compile(r"255h", re.I), CORE_ULTRA_7_255H),
    (re.compile(r"ultra\s*7\s*265h", re.I), CORE_ULTRA_7_265H),
    (re.compile(r"265h", re.I), CORE_ULTRA_7_265H),
    (re.compile(r"ultra\s*9\s*285h", re.I), CORE_ULTRA_9_285H),
    (re.compile(r"285h", re.I), CORE_ULTRA_9_285H),
    (re.compile(r"arc\s*graphics\s*140t", re.I), CORE_ULTRA_7_255H),
    (re.compile(r"140t", re.I), CORE_ULTRA_7_255H),
]


@dataclass
class DeviceInfo:
    """Runtime snapshot of the selected accelerator."""

    kind: DeviceKind
    device: str
    index: int
    name: str
    profile: HardwareProfile
    total_memory_bytes: int | None = None
    xpu_available: bool = False
    torch_version: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_xpu(self) -> bool:
        return self.kind is DeviceKind.XPU


def _safe_torch():
    try:
        import torch

        return torch
    except ImportError:  # pragma: no cover - exercised in bare CI without torch
        return None


def _read_cpu_model() -> str:
    candidates: list[str] = []
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    candidates.append(line.split(":", 1)[1].strip())
                    break
    except OSError:
        pass
    if not candidates:
        candidates.append(platform.processor() or "")
    # Windows / macOS fallbacks are rare in this project's target env.
    return candidates[0] if candidates else ""


def _match_profile(text: str) -> HardwareProfile | None:
    for pattern, profile in _SKU_PATTERNS:
        if pattern.search(text):
            return profile
    return None


def resolve_profile(
    hint: str | None = None,
    *,
    device_name: str | None = None,
    prefer_255h: bool = True,
) -> HardwareProfile:
    """Pick a hardware profile from hints, CPU model, or XPU device name."""
    blobs = [hint or "", device_name or "", _read_cpu_model()]
    for blob in blobs:
        matched = _match_profile(blob)
        if matched is not None:
            return matched

    torch = _safe_torch()
    if torch is not None and hasattr(torch, "xpu") and torch.xpu.is_available():
        # Client Arc iGPUs almost always share host DRAM.
        name = ""
        try:
            name = torch.xpu.get_device_name(0)
        except Exception:
            name = ""
        matched = _match_profile(name)
        if matched is not None:
            return matched
        if prefer_255h:
            # This library is purpose-built around the 255H; default to its knobs
            # when an unnamed client XPU is present.
            return CORE_ULTRA_7_255H
        return GENERIC_ARC_CLIENT

    if prefer_255h and hint and "255" in hint:
        return CORE_ULTRA_7_255H
    return CPU_ONLY


def detect_device(
    *,
    prefer: str = "xpu",
    profile_hint: str | None = None,
    device_index: int = 0,
) -> DeviceInfo:
    """Detect the best available device and attach a matching hardware profile."""
    torch = _safe_torch()
    torch_version = getattr(torch, "__version__", None) if torch is not None else None

    prefer = prefer.lower()
    order = [prefer] + [d for d in ("xpu", "cuda", "mps", "cpu") if d != prefer]

    for kind_name in order:
        if torch is None:
            break
        if kind_name == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
            try:
                name = torch.xpu.get_device_name(device_index)
            except Exception:
                name = "Intel XPU"
            total = None
            try:
                props = torch.xpu.get_device_properties(device_index)
                total = int(getattr(props, "total_memory", 0) or 0) or None
            except Exception:
                total = None
            profile = resolve_profile(profile_hint, device_name=name)
            return DeviceInfo(
                kind=DeviceKind.XPU,
                device=f"xpu:{device_index}",
                index=device_index,
                name=name,
                profile=profile,
                total_memory_bytes=total,
                xpu_available=True,
                torch_version=torch_version,
            )
        if kind_name == "cuda" and torch.cuda.is_available():
            name = torch.cuda.get_device_name(device_index)
            total = int(torch.cuda.get_device_properties(device_index).total_memory)
            return DeviceInfo(
                kind=DeviceKind.CUDA,
                device=f"cuda:{device_index}",
                index=device_index,
                name=name,
                profile=GENERIC_XPU,
                total_memory_bytes=total,
                xpu_available=False,
                torch_version=torch_version,
            )
        mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if kind_name == "mps" and mps_ok:
            return DeviceInfo(
                kind=DeviceKind.MPS,
                device="mps",
                index=0,
                name="Apple MPS",
                profile=CPU_ONLY,
                total_memory_bytes=None,
                xpu_available=False,
                torch_version=torch_version,
            )

    profile = resolve_profile(profile_hint or "255H") if profile_hint else CPU_ONLY
    # Even without a live XPU, allow callers to request the 255H preset for planning.
    if profile_hint:
        profile = resolve_profile(profile_hint)
    return DeviceInfo(
        kind=DeviceKind.CPU,
        device="cpu",
        index=0,
        name=_read_cpu_model() or "CPU",
        profile=profile if profile_hint else CPU_ONLY,
        total_memory_bytes=None,
        xpu_available=bool(
            torch is not None and hasattr(torch, "xpu") and torch.xpu.is_available()
        ),
        torch_version=torch_version,
    )


@lru_cache(maxsize=1)
def default_device() -> DeviceInfo:
    """Cached detection, preferring XPU and the Core Ultra 7 255H profile."""
    return detect_device(prefer="xpu", profile_hint="255H")
