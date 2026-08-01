from __future__ import annotations

import torch
import torch.nn as nn

from transformers_xpu_helper.amp import resolve_amp
from transformers_xpu_helper.config import ultra_255h_config
from transformers_xpu_helper.dataloader import dataloader_kwargs
from transformers_xpu_helper.hardware import CORE_ULTRA_7_255H, DeviceInfo, DeviceKind
from transformers_xpu_helper.memory import (
    estimate_budget,
    format_bytes,
    suggest_batch_size,
    suggest_vision_batch_size,
)
from transformers_xpu_helper.optimize import finalize_step, prepare_for_training, training_step
from transformers_xpu_helper.profiling import Profiler


def _fake_xpu_info() -> DeviceInfo:
    return DeviceInfo(
        kind=DeviceKind.CPU,  # tests run without XPU
        device="cpu",
        index=0,
        name="test-cpu",
        profile=CORE_ULTRA_7_255H,
        xpu_available=False,
        torch_version=torch.__version__,
    )


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 4)

    def forward(self, x):
        return self.fc(x)


def test_resolve_amp_bf16_no_scaler():
    cfg = ultra_255h_config(precision="bf16", use_grad_scaler=True)
    info = _fake_xpu_info()
    amp = resolve_amp(cfg, info)
    assert amp.enabled is True
    assert amp.dtype == torch.bfloat16
    assert amp.use_grad_scaler is False
    assert amp.scaler is None


def test_memory_budget_and_batch_heuristic():
    cfg = ultra_255h_config()
    info = _fake_xpu_info()
    budget = estimate_budget(cfg, info)
    assert budget.trainable_bytes > 0
    assert "GiB" in format_bytes(budget.trainable_bytes) or "MiB" in format_bytes(
        budget.trainable_bytes
    )

    # ~110M params (bert-base-ish) should yield a small micro-batch on shared RAM.
    batch = suggest_batch_size(110_000_000, seq_length=512, config=cfg, device_info=info)
    assert 1 <= batch <= 8


def test_vision_batch_gc_allows_larger_than_no_gc():
    info = _fake_xpu_info()
    # Large-ish vision model (~300M) — GC should recommend >= no-GC.
    params = 300_000_000
    with_gc = suggest_vision_batch_size(
        params,
        image_hw=(384, 384),
        config=ultra_255h_config(gradient_checkpointing=True),
        device_info=info,
    )
    no_gc = suggest_vision_batch_size(
        params,
        image_hw=(384, 384),
        config=ultra_255h_config(gradient_checkpointing=False),
        device_info=info,
    )
    assert with_gc >= no_gc
    assert 1 <= no_gc <= with_gc <= 32


def test_vision_batch_scales_with_image_area():
    info = _fake_xpu_info()
    cfg = ultra_255h_config(gradient_checkpointing=True)
    params = 200_000_000
    small = suggest_vision_batch_size(
        params, image_hw=(224, 224), config=cfg, device_info=info
    )
    large = suggest_vision_batch_size(
        params, image_hw=(512, 512), config=cfg, device_info=info
    )
    assert small >= large


def test_nlp_batch_gc_allows_larger_than_no_gc():
    info = _fake_xpu_info()
    params = 110_000_000
    with_gc = suggest_batch_size(
        params,
        seq_length=512,
        config=ultra_255h_config(gradient_checkpointing=True),
        device_info=info,
    )
    no_gc = suggest_batch_size(
        params,
        seq_length=512,
        config=ultra_255h_config(gradient_checkpointing=False),
        device_info=info,
    )
    assert with_gc >= no_gc


def test_dataloader_kwargs_shared_memory():
    cfg = ultra_255h_config()
    info = _fake_xpu_info()
    kwargs = dataloader_kwargs(cfg, info)
    assert kwargs["num_workers"] == 4
    assert kwargs.get("pin_memory", False) is False


def test_prepare_and_train_step_cpu():
    cfg = ultra_255h_config(
        torch_compile=False,
        gradient_checkpointing=False,
        precision="fp32",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
    )
    model = Tiny()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    prepared = prepare_for_training(model, opt, config=cfg, device_info=_fake_xpu_info())
    assert prepared.compiled is False

    x = torch.randn(4, 16)
    y = torch.randint(0, 4, (4,))
    criterion = nn.CrossEntropyLoss()
    loss = training_step(prepared, (x, y), criterion=criterion)
    assert torch.isfinite(loss).item()
    finalize_step(prepared, step=1)


def test_profiler_smoke():
    profiler = Profiler(device_info=_fake_xpu_info())
    with profiler.step(batch_size=4):
        _ = sum(range(1000))
    summary = profiler.summary()
    assert summary["steps"] == 1
    assert summary["samples"] == 4
    assert summary["profile"] == CORE_ULTRA_7_255H.name
