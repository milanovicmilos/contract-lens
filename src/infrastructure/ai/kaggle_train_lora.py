"""
Kaggle Training Script for Multi-label Risk Classifier.

Fine-tunes DeBERTa-v3-large with LoRA on the CUAD multi-label dataset.
Designed for Kaggle T4 x2 (16GB VRAM each) or P100 GPUs.

INPUTS:
- /kaggle/input/contractlens-cuad-multilabel/cuad_multilabel.jsonl
  Format: {"text": "...", "labels": [0,1,0,...]} (length 41)

OUTPUTS:
- /kaggle/working/deberta-cuad-classifier/  (model + tokenizer + LoRA adapters)
- /kaggle/working/eval_report.json          (per-category F1)
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from datasets import Dataset, load_dataset
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = os.environ.get("MODEL_NAME", "microsoft/deberta-v3-large")
DATASET_PATH = os.environ.get(
    "DATASET_PATH",
    "/kaggle/input/contractlens-cuad-multilabel/cuad_multilabel.jsonl",
)
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/kaggle/working/deberta-cuad-classifier")
NUM_LABELS = 41  # CUAD has 41 clause categories
MAX_LENGTH = 512
PROB_THRESHOLD = 0.5


# CUAD category list, must match prepare_multilabel_dataset.py output
CUAD_CATEGORIES = [
    "Document Name",
    "Parties",
    "Agreement Date",
    "Effective Date",
    "Expiration Date",
    "Renewal Term",
    "Notice Period To Terminate Renewal",
    "Governing Law",
    "Most Favored Nation",
    "Competitive Restriction Exception",
    "Non-Compete",
    "Exclusivity",
    "No-Solicit Of Customers",
    "No-Solicit Of Employees",
    "Non-Disparagement",
    "Termination For Convenience",
    "Rofr/Rofo/Rofn",
    "Change Of Control",
    "Anti-Assignment",
    "Revenue/Profit Sharing",
    "Price Restrictions",
    "Minimum Commitment",
    "Volume Restriction",
    "Ip Ownership Assignment",
    "Joint Ip Ownership",
    "License Grant",
    "Non-Transferable License",
    "Affiliate License-Licensor",
    "Affiliate License-Licensee",
    "Unlimited/All-You-Can-Eat-License",
    "Irrevocable Or Perpetual License",
    "Source Code Escrow",
    "Post-Termination Services",
    "Audit Rights",
    "Uncapped Liability",
    "Cap On Liability",
    "Liquidated Damages",
    "Warranty Duration",
    "Insurance",
    "Covenant Not To Sue",
    "Third Party Beneficiary",
]


def compute_metrics(eval_pred) -> Dict[str, float]:
    """Multi-label classification metrics: micro/macro F1 + precision/recall."""
    logits, labels = eval_pred
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs > PROB_THRESHOLD).astype(int)
    labels = labels.astype(int)

    return {
        "micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "micro_precision": precision_score(labels, preds, average="micro", zero_division=0),
        "micro_recall": recall_score(labels, preds, average="micro", zero_division=0),
    }


def tokenize_batch(batch: Dict[str, Any], tokenizer) -> Dict[str, Any]:
    """Tokenize text and cast labels to float (BCEWithLogitsLoss requirement)."""
    enc = tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH, padding="max_length")
    enc["labels"] = [[float(v) for v in row] for row in batch["labels"]]
    return enc


def load_and_prepare_dataset(tokenizer) -> Dataset:
    """Load JSONL, validate label shape, tokenize, split train/val."""
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. "
            "Run scripts/prepare_multilabel_dataset.py first and upload to Kaggle."
        )

    logger.info(f"Loading dataset from {DATASET_PATH}")
    raw = load_dataset("json", data_files=DATASET_PATH, split="train")

    sample_labels = raw[0]["labels"]
    if len(sample_labels) != NUM_LABELS:
        raise ValueError(
            f"Expected labels of length {NUM_LABELS}, got {len(sample_labels)}. "
            "Regenerate multilabel dataset with correct category count."
        )

    logger.info(f"Loaded {len(raw)} examples. Tokenizing...")
    tokenized = raw.map(
        lambda batch: tokenize_batch(batch, tokenizer),
        batched=True,
        remove_columns=[c for c in raw.column_names if c not in ("labels",)],
    )

    split = tokenized.train_test_split(test_size=0.1, seed=42)
    logger.info(f"Train: {len(split['train'])}, Val: {len(split['test'])}")
    return split


def apply_lora(model):
    """Wrap model with LoRA adapters for VRAM-efficient training."""
    try:
        from peft import LoraConfig, TaskType, get_peft_model

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["query_proj", "value_proj", "key_proj"],
        )
        model = get_peft_model(model, peft_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(
            f"LoRA applied. Trainable params: {trainable:,}/{total:,} ({100*trainable/total:.2f}%)"
        )
        return model
    except ImportError:
        logger.warning("peft not installed; running full fine-tuning (high VRAM usage).")
        return model


def per_category_report(trainer, dataset) -> Dict[str, Any]:
    """Generate per-category F1 report for thesis evaluation."""
    preds = trainer.predict(dataset)
    probs = 1.0 / (1.0 + np.exp(-preds.predictions))
    y_pred = (probs > PROB_THRESHOLD).astype(int)
    y_true = preds.label_ids.astype(int)

    report = classification_report(
        y_true, y_pred, target_names=CUAD_CATEGORIES, zero_division=0, output_dict=True
    )
    return report


def main():
    logger.info(f"Starting fine-tuning of {MODEL_NAME} on CUAD multi-label task")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
    )

    model = apply_lora(model)

    dataset = load_and_prepare_dataset(tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=3e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        num_train_epochs=4,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="micro_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=50,
        report_to="none",
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    logger.info("Starting training loop...")
    trainer.train()

    logger.info(f"Saving final model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    logger.info("Generating per-category evaluation report...")
    report = per_category_report(trainer, dataset["test"])
    report_path = Path(OUTPUT_DIR).parent / "eval_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Eval report saved to {report_path}")

    logger.info(
        f"Final metrics: micro_f1={report['micro avg']['f1-score']:.4f}, "
        f"macro_f1={report['macro avg']['f1-score']:.4f}"
    )


if __name__ == "__main__":
    main()
