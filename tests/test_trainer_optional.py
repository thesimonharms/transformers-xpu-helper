from __future__ import annotations

import pytest

pytest.importorskip("transformers")

from transformers_xpu_helper import ultra_255h_config
from transformers_xpu_helper.trainer import build_training_arguments


def test_build_training_arguments():
    cfg = ultra_255h_config(torch_compile=False, per_device_train_batch_size=2)
    args = build_training_arguments("./tmp-xpu-helper-out", config=cfg, num_train_epochs=1)
    assert args.per_device_train_batch_size == 2
    assert args.bf16 is True or args.fp16 is False
    assert args.gradient_checkpointing is True
    assert args.dataloader_pin_memory is False
