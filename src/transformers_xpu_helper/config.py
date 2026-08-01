"""Training configuration presets for Intel XPU / Core Ultra 7 255H."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal

from .hardware import (
    CORE_ULTRA_7_255H,
    AmpPolicy,
    DeviceInfo,
    HardwareProfile,
    detect_device,
)

Precision = Literal["bf16", "fp16", "fp32", "auto"]


@dataclass
class XPUTrainingConfig:
    """Opinionated training knobs sized for Intel client XPUs.

    Defaults target the Core Ultra 7 255H + Arc 140T: shared system memory,
    BF16 autocast without GradScaler, torch.compile, and gradient checkpointing.
    """

    device: str = "xpu"
    profile_name: str = CORE_ULTRA_7_255H.name
    precision: Precision = "auto"
    use_grad_scaler: bool = False
    torch_compile: bool = True
    compile_mode: str = "default"
    compile_fullgraph: bool = False
    gradient_checkpointing: bool = True
    gradient_accumulation_steps: int = 4
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = False
    dataloader_persistent_workers: bool = True
    memory_fraction: float = 0.55
    host_reserve_gib: float = 6.0
    omp_num_threads: int = 8
    mkl_num_threads: int = 8
    empty_cache_steps: int = 50
    sync_every_step: bool = False
    tf32: bool = False  # not applicable on XPU; kept for TrainingArguments parity
    seed: int = 42
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def amp_dtype_name(self) -> AmpPolicy:
        if self.precision == "auto":
            return AmpPolicy.BF16
        return AmpPolicy(self.precision)

    def with_updates(self, **kwargs: Any) -> XPUTrainingConfig:
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def config_from_profile(
    profile: HardwareProfile | None = None,
    *,
    device_info: DeviceInfo | None = None,
    precision: Precision = "auto",
    batch_size: int | None = None,
    grad_accum: int | None = None,
) -> XPUTrainingConfig:
    """Build an :class:`XPUTrainingConfig` from a hardware profile."""
    info = device_info or detect_device(prefer="xpu", profile_hint="255H")
    profile = profile or info.profile

    amp = profile.preferred_amp if precision == "auto" else AmpPolicy(precision)
    use_scaler = bool(profile.supports_grad_scaler and amp is AmpPolicy.FP16)

    cfg = XPUTrainingConfig(
        device=info.device if info.kind.value != "cpu" else "xpu",
        profile_name=profile.name,
        precision=amp.value if precision == "auto" else precision,
        use_grad_scaler=use_scaler,
        torch_compile=profile.torch_compile_default,
        gradient_checkpointing=profile.gradient_checkpointing_default,
        gradient_accumulation_steps=grad_accum if grad_accum is not None else 4,
        per_device_train_batch_size=batch_size if batch_size is not None else 2,
        per_device_eval_batch_size=batch_size if batch_size is not None else 2,
        dataloader_num_workers=profile.recommended_dataloader_workers,
        dataloader_pin_memory=False if profile.shared_memory else True,
        dataloader_persistent_workers=profile.recommended_dataloader_workers > 0,
        memory_fraction=profile.default_memory_fraction,
        host_reserve_gib=profile.host_reserve_gib,
        omp_num_threads=profile.recommended_omp_num_threads,
        mkl_num_threads=profile.recommended_omp_num_threads,
    )
    return cfg


def ultra_255h_config(**overrides: Any) -> XPUTrainingConfig:
    """Preset explicitly tuned for the Intel Core Ultra 7 255H."""
    info = detect_device(prefer="xpu", profile_hint="255H")
    # Force the 255H profile even when detection falls back to CPU (dev machines).
    from .hardware import CORE_ULTRA_7_255H

    cfg = config_from_profile(CORE_ULTRA_7_255H, device_info=info)
    if info.kind.value == "xpu":
        cfg = cfg.with_updates(device=info.device)
    else:
        cfg = cfg.with_updates(device="xpu")
    if overrides:
        cfg = cfg.with_updates(**overrides)
    return cfg
