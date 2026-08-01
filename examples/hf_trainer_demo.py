"""Example: wire Hugging Face Trainer to 255H-oriented TrainingArguments.

Requires: pip install 'transformers-xpu-helper[transformers]' datasets
"""

from __future__ import annotations


def main() -> None:
    from datasets import Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer

    from transformers_xpu_helper import ultra_255h_config
    from transformers_xpu_helper.trainer import build_training_arguments, recommend_for_model

    model_name = "prajjwal1/bert-tiny"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    texts = ["good film", "bad film", "lovely", "terrible"] * 16
    labels = [1, 0, 1, 0] * 16
    enc = tokenizer(texts, truncation=True, padding="max_length", max_length=32)
    enc["labels"] = labels
    ds = Dataset.from_dict(enc)

    cfg = recommend_for_model(model, seq_length=32, config=ultra_255h_config(torch_compile=False))
    args = build_training_arguments(
        "./xpu-helper-demo-out",
        config=cfg,
        num_train_epochs=1,
        learning_rate=5e-5,
        logging_steps=5,
        save_strategy="no",
        eval_strategy="no",
    )

    trainer = Trainer(model=model, args=args, train_dataset=ds)
    trainer.train()
    print("done", args.device)


if __name__ == "__main__":
    main()
