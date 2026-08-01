"""DataLoader tuning for hybrid P/E-core Intel CPUs."""

from __future__ import annotations

from typing import Any

from .config import XPUTrainingConfig
from .hardware import DeviceInfo, detect_device


def dataloader_kwargs(
    config: XPUTrainingConfig | None = None,
    device_info: DeviceInfo | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Return keyword arguments suitable for ``torch.utils.data.DataLoader``.

    On the Core Ultra 7 255H we keep ``num_workers`` modest (E-cores handle
    prefetch) and disable ``pin_memory`` because Arc 140T shares host DRAM —
    pinning often adds overhead without a discrete HBM pool.
    """
    config = config or XPUTrainingConfig()
    info = device_info or detect_device(prefer="xpu", profile_hint="255H")

    workers = config.dataloader_num_workers
    kwargs: dict[str, Any] = {
        "num_workers": workers,
        "pin_memory": config.dataloader_pin_memory and not info.profile.shared_memory,
        "persistent_workers": bool(config.dataloader_persistent_workers and workers > 0),
        "prefetch_factor": 2 if workers > 0 else None,
    }
    # Drop None values so DataLoader does not reject prefetch_factor without workers.
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    kwargs.update(overrides)
    if kwargs.get("num_workers", 0) == 0:
        kwargs.pop("persistent_workers", None)
        kwargs.pop("prefetch_factor", None)
    return kwargs


def wrap_dataloader(dataset: Any, batch_size: int | None = None, **kwargs: Any) -> Any:
    """Construct a DataLoader with XPU-oriented defaults."""
    import torch

    from .config import ultra_255h_config

    config = kwargs.pop("config", None) or ultra_255h_config()
    device_info = kwargs.pop("device_info", None)
    bs = batch_size if batch_size is not None else config.per_device_train_batch_size
    dl_kwargs = dataloader_kwargs(config, device_info)
    dl_kwargs.update(kwargs)
    return torch.utils.data.DataLoader(dataset, batch_size=bs, **dl_kwargs)
