from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch.nn as nn

pytest.importorskip("transformers")

from transformers_xpu_helper import ultra_255h_config
from transformers_xpu_helper.trainer import (
    build_seq2seq_training_arguments,
    build_training_arguments,
    is_vision_encoder_decoder,
    recommend_for_model,
)


def test_build_training_arguments():
    cfg = ultra_255h_config(torch_compile=False, per_device_train_batch_size=2)
    args = build_training_arguments("./tmp-xpu-helper-out", config=cfg, num_train_epochs=1)
    assert args.per_device_train_batch_size == 2
    assert args.bf16 is True or args.fp16 is False
    assert args.gradient_checkpointing is True
    assert args.dataloader_pin_memory is False


def test_build_seq2seq_training_arguments():
    cfg = ultra_255h_config(torch_compile=False, per_device_train_batch_size=4)
    args = build_seq2seq_training_arguments(
        "./tmp-xpu-helper-seq2seq",
        config=cfg,
        num_train_epochs=1,
        predict_with_generate=False,
        generation_max_length=64,
    )
    assert args.per_device_train_batch_size == 4
    assert args.gradient_checkpointing is True
    assert args.dataloader_pin_memory is False
    assert args.predict_with_generate is False
    assert getattr(args, "generation_max_length", 64) == 64


class _FakeVed(nn.Module):
    """Minimal stand-in for VisionEncoderDecoderModel detection."""

    def __init__(self, n: int = 1_000_000):
        super().__init__()
        self.fc = nn.Linear(8, 8)
        # Inflate param count for recommend_for_model without a real TrOCR.
        self._pad = nn.Parameter(self.fc.weight.new_zeros(n // 8, 8))
        self.config = SimpleNamespace(
            is_encoder_decoder=True,
            model_type="vision-encoder-decoder",
            vision_config=SimpleNamespace(),
        )


def test_is_vision_encoder_decoder_and_recommend():
    model = _FakeVed(2_000_000)
    assert is_vision_encoder_decoder(model) is True
    cfg = recommend_for_model(
        model,
        task="auto",
        image_hw=(384, 384),
        config=ultra_255h_config(torch_compile=False, gradient_checkpointing=True),
    )
    assert cfg.per_device_train_batch_size >= 1
    assert cfg.gradient_accumulation_steps >= 1


def test_recommend_nlp_task_force():
    model = _FakeVed(2_000_000)
    cfg = recommend_for_model(
        model,
        task="nlp",
        seq_length=128,
        config=ultra_255h_config(torch_compile=False),
    )
    assert cfg.per_device_train_batch_size >= 1
