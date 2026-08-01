"""Sketch: TrOCR / VisionEncoderDecoder fine-tune via Seq2SeqTrainer + XPU helper.

Does not download a full OCR dataset — builds a tiny in-memory pair so the
wiring is clear. On a real cook:

  1. cfg = recommend_for_model(model, task="vision", image_hw=(384, 384))
  2. Smoke 1–2 steps at cfg.per_device_train_batch_size (and ± neighbors).
  3. Override TROCR_BATCH_SIZE / env if the smoke disagrees with the heuristic.
  4. args = build_seq2seq_training_arguments(out, config=cfg, ...)

Requires: pip install 'transformers-xpu-helper[transformers]' pillow
"""

from __future__ import annotations


def main() -> None:
    import torch
    from PIL import Image
    from torch.utils.data import Dataset
    from transformers import (
        Seq2SeqTrainer,
        TrOCRProcessor,
        VisionEncoderDecoderModel,
    )

    from transformers_xpu_helper import ultra_255h_config
    from transformers_xpu_helper.trainer import (
        build_seq2seq_training_arguments,
        recommend_for_model,
    )

    # Small printed checkpoint keeps the sketch downloadable; swap for
    # microsoft/trocr-large-printed (or your Hub tip) for real training.
    model_id = "microsoft/trocr-small-printed"
    processor = TrOCRProcessor.from_pretrained(model_id)
    model = VisionEncoderDecoderModel.from_pretrained(model_id)
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    class TinyOcr(Dataset):
        def __len__(self) -> int:
            return 8

        def __getitem__(self, idx: int):
            img = Image.new("RGB", (384, 384), color=(240, 240, 240))
            text = f"sample {idx}"
            pixel_values = processor(images=img, return_tensors="pt").pixel_values[0]
            labels = processor.tokenizer(
                text,
                padding="max_length",
                max_length=16,
                truncation=True,
                return_tensors="pt",
            ).input_ids[0]
            labels[labels == processor.tokenizer.pad_token_id] = -100
            return {"pixel_values": pixel_values, "labels": labels}

    cfg = recommend_for_model(
        model,
        task="vision",
        image_hw=(384, 384),
        config=ultra_255h_config(torch_compile=False),
    )
    print(
        f"recommended micro-batch={cfg.per_device_train_batch_size} "
        f"grad_accum={cfg.gradient_accumulation_steps} "
        f"gc={cfg.gradient_checkpointing}"
    )

    args = build_seq2seq_training_arguments(
        "./xpu-helper-trocr-sketch-out",
        config=cfg,
        num_train_epochs=1,
        max_steps=2,
        learning_rate=3e-5,
        logging_steps=1,
        save_strategy="no",
        eval_strategy="no",
        predict_with_generate=False,
        # TrOCR cooks often disable compile until smoke-proven on the SKU.
        torch_compile=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=TinyOcr(),
        processing_class=processor,
    )
    trainer.train()
    print("done", args.device, "cuda/xpu/cpu ok; torch", torch.__version__)


if __name__ == "__main__":
    main()
