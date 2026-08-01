# transformers-xpu-helper

Speed up and optimize [Hugging Face Transformers](https://huggingface.co/docs/transformers) training on **Intel XPUs**, with defaults tuned for the **Intel Core Ultra 7 255H** (Arrow Lake-H + **Arc Graphics 140T**).

Uses **native PyTorch XPU** (2.5+). [Intel Extension for PyTorch (IPEX) is EOL](https://pytorch-extension.intel.com/) — prefer stock PyTorch with the `xpu` wheel index.

## Why this exists

Client Arc GPUs share system memory, lack a comfortable GradScaler/FP64 path, and sit next to a hybrid P/E-core CPU. Stock Transformers/`Trainer` knobs assume discrete CUDA cards. This library applies 255H-oriented defaults:

| Concern | 255H / Arc 140T default |
| --- | --- |
| Precision | BF16 autocast (no GradScaler) |
| Graph mode | `torch.compile` (inductor) |
| Memory | Shared-DRAM budget + optional allocator fraction |
| Activations | Gradient checkpointing on |
| DataLoader | Modest workers, `pin_memory=False` |
| Host threads | Capped OMP/MKL threads to avoid oversubscription |

## Install

```bash
# PyTorch with Intel GPU (XPU) support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu

# This helper (+ optional HF integration)
pip install transformers-xpu-helper
pip install 'transformers-xpu-helper[transformers]'  # Trainer helpers
```

Intel GPU drivers are still required — see [PyTorch Getting Started on Intel GPU](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.md).

## Quick start (custom loop)

```python
import torch
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from transformers_xpu_helper import (
    finalize_step,
    prepare_for_training,
    training_step,
    ultra_255h_config,
)

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
optimizer = AdamW(model.parameters(), lr=2e-5)
cfg = ultra_255h_config(per_device_train_batch_size=2, gradient_accumulation_steps=4)

prepared = prepare_for_training(model, optimizer, config=cfg)

for step, batch in enumerate(train_loader, start=1):
    loss = training_step(prepared, batch)
    if step % cfg.gradient_accumulation_steps == 0:
        finalize_step(prepared, step=step)
```

## Quick start (Hugging Face Trainer)

```python
from transformers import Trainer, AutoModelForSequenceClassification
from transformers_xpu_helper import ultra_255h_config
from transformers_xpu_helper.trainer import build_training_arguments, recommend_for_model

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
cfg = recommend_for_model(model, seq_length=256)
args = build_training_arguments("./out", config=cfg, num_train_epochs=3, learning_rate=2e-5)

trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds)
trainer.train()
```

## Inspect the machine

```bash
xpu-helper-info
xpu-helper-info --json
```

## Core Ultra 7 255H profile

The built-in profile encodes:

- **CPU**: 6P + 8E + 2 LP-E (16 cores)
- **GPU**: Arc 140T, 8 Xe cores, shared DRAM, XMX
- **AMP**: BF16 preferred; GradScaler disabled
- **Workers**: 4 DataLoader workers, 8 OMP threads
- **Memory fraction**: ~55% of RAM after a 6 GiB host reserve

Nearby Arrow Lake-H SKUs (265H / 285H) reuse the same knobs.

## API surface

| Module | Role |
| --- | --- |
| `hardware` | Device detection + SKU profiles |
| `config` | `ultra_255h_config()` / `XPUTrainingConfig` |
| `optimize` | `prepare_for_training`, `training_step`, `finalize_step` |
| `amp` | Safe BF16/FP16 autocast + scaler policy |
| `memory` | Shared-memory budgets + batch-size heuristic |
| `dataloader` | XPU-friendly `DataLoader` kwargs |
| `trainer` | `TrainingArguments` factory for Transformers |
| `profiling` | Step throughput / peak memory helper |
| `env` | OMP / tokenizer / allocator environment |

## Development

```bash
pip install -e '.[dev]'
pytest
```

## License

MIT
