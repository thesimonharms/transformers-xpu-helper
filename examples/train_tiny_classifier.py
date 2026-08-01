"""Minimal synthetic training loop using transformers-xpu-helper.

Runs on CPU when XPU is unavailable (useful for CI). On a Core Ultra 7 255H with
Arc 140T drivers + PyTorch XPU wheels, the same code targets the iGPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from transformers_xpu_helper import (
    Profiler,
    finalize_step,
    prepare_for_training,
    training_step,
    ultra_255h_config,
    wrap_dataloader,
)


class TinyClassifier(nn.Module):
    def __init__(self, dim: int = 64, n_classes: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def main() -> None:
    cfg = ultra_255h_config(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        torch_compile=False,  # keep the demo light on CPU / first-run CI
        gradient_checkpointing=False,
    )

    x = torch.randn(256, 64)
    y = torch.randint(0, 4, (256,))
    dataset = TensorDataset(x, y)
    loader: DataLoader = wrap_dataloader(dataset, config=cfg, shuffle=True)

    model = TinyClassifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    prepared = prepare_for_training(model, optimizer, config=cfg)
    profiler = Profiler(device_info=prepared.device_info)

    prepared.optimizer.zero_grad(set_to_none=True)
    step = 0
    for _epoch in range(2):
        for batch in loader:
            step += 1
            with profiler.step(batch_size=batch[0].size(0)):
                loss = training_step(prepared, batch, criterion=criterion)
                if step % cfg.gradient_accumulation_steps == 0:
                    finalize_step(prepared, step=step)
            if step % 10 == 0:
                print(f"step={step} loss={float(loss.detach().cpu()):.4f}")

    print("profile:", profiler.summary())
    print(
        f"device={prepared.device_info.device} "
        f"profile={prepared.device_info.profile.name} "
        f"amp={prepared.amp.dtype} compiled={prepared.compiled}"
    )


if __name__ == "__main__":
    main()
