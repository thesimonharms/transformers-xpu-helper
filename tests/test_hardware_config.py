from __future__ import annotations

import os

from transformers_xpu_helper.config import config_from_profile, ultra_255h_config
from transformers_xpu_helper.hardware import (
    CORE_ULTRA_7_255H,
    AmpPolicy,
    detect_device,
    resolve_profile,
)


def test_resolve_profile_255h():
    profile = resolve_profile("Intel Core Ultra 7 255H")
    assert profile is CORE_ULTRA_7_255H
    assert profile.gpu_name == "Intel Arc Graphics 140T"
    assert profile.xe_cores == 8
    assert profile.shared_memory is True
    assert profile.preferred_amp is AmpPolicy.BF16
    assert profile.supports_grad_scaler is False


def test_resolve_profile_arc_140t():
    assert resolve_profile(device_name="Intel Arc Graphics 140T") is CORE_ULTRA_7_255H


def test_ultra_255h_config_defaults():
    cfg = ultra_255h_config()
    assert cfg.precision in {"bf16", "auto"} or cfg.amp_dtype_name is AmpPolicy.BF16
    assert cfg.use_grad_scaler is False
    assert cfg.torch_compile is True
    assert cfg.gradient_checkpointing is True
    assert cfg.dataloader_pin_memory is False
    assert cfg.dataloader_num_workers == 4
    assert cfg.memory_fraction == 0.55
    assert cfg.profile_name == CORE_ULTRA_7_255H.name


def test_config_from_profile():
    cfg = config_from_profile(CORE_ULTRA_7_255H, batch_size=4, grad_accum=8)
    assert cfg.per_device_train_batch_size == 4
    assert cfg.gradient_accumulation_steps == 8


def test_detect_device_returns_profile():
    info = detect_device(prefer="xpu", profile_hint="255H")
    assert info.profile.name == CORE_ULTRA_7_255H.name
    assert info.device in {info.device}  # smoke
    assert info.kind.value in {"xpu", "cpu", "cuda", "mps"}


def test_apply_runtime_env(monkeypatch):
    from transformers_xpu_helper.env import apply_runtime_env

    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "TOKENIZERS_PARALLELISM"):
        monkeypatch.delenv(key, raising=False)

    cfg = ultra_255h_config()
    applied = apply_runtime_env(cfg, CORE_ULTRA_7_255H, overwrite=True)
    assert applied["OMP_NUM_THREADS"] == "8"
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"
