"""
Local Multi-label Classifier Training Script.

Intended for fast iteration on a workstation GPU (or CPU for tiny models).
For full DeBERTa-v3-large training on the entire dataset, use the matching
Kaggle notebook (kaggle/notebook_classifier.py) on T4 x2.

Input format: JSONL with {"text": str, "labels": List[int]} where labels
is a length-41 binary vector. Produced by scripts/prepare_multilabel_dataset.py.

USAGE:
    python -m src.infrastructure.ai.train_classifier \
        --model microsoft/deberta-v3-small \
        --data data/processed/cuad_multilabel.jsonl \
        --output models/deberta-small-cuad \
        --epochs 2 \
        --batch 8
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
from datasets import load_dataset
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

NUM_LABELS = 41
PROB_THRESHOLD = 0.5

CUAD_CATEGORIES = [
    "Document Name", "Parties", "Agreement Date", "Effective Date", "Expiration Date",
    "Renewal Term", "Notice Period To Terminate Renewal", "Governing Law", "Most Favored Nation",
    "Competitive Restriction Exception", "Non-Compete", "Exclusivity", "No-Solicit Of Customers",
    "No-Solicit Of Employees", "Non-Disparagement", "Termination For Convenience",
    "Rofr/Rofo/Rofn", "Change Of Control", "Anti-Assignment", "Revenue/Profit Sharing",
    "Price Restrictions", "Minimum Commitment", "Volume Restriction", "Ip Ownership Assignment",
    "Joint Ip Ownership", "License Grant", "Non-Transferable License", "Affiliate License-Licensor",
    "Affiliate License-Licensee", "Unlimited/All-You-Can-Eat-License",
    "Irrevocable Or Perpetual License", "Source Code Escrow", "Post-Termination Services",
    "Audit Rights", "Uncapped Liability", "Cap On Liability", "Liquidated Damages",
    "Warranty Duration", "Insurance", "Covenant Not To Sue", "Third Party Beneficiary",
]


def compute_metrics(eval_pred) -> Dict[str, float]:
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


def tokenize_batch(batch: Dict[str, Any], tokenizer, max_length: int) -> Dict[str, Any]:
    enc = tokenizer(
        batch["text"], truncation=True, max_length=max_length, padding="max_length"
    )
    enc["labels"] = [[float(v) for v in row] for row in batch["labels"]]
    return enc


def train_classifier(
    dataset_path: Path,
    output_dir: Path,
    model_name: str = "microsoft/deberta-v3-small",
    epochs: int = 2,
    batch_size: int = 8,
    max_length: int = 512,
    learning_rate: float = 3e-5,
) -> Dict[str, Any]:
    """Run the full fine-tuning loop and write the model + an eval report."""
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
            "Run scripts/prepare_multilabel_dataset.py first."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Loading tokenizer & model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    config = AutoConfig.from_pretrained(
        model_name, num_labels=NUM_LABELS, problem_type="multi_label_classification"
    )
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

    logger.info(f"Loading dataset from {dataset_path}")
    raw = load_dataset("json", data_files=str(dataset_path), split="train")
    sample = raw[0]["labels"]
    if len(sample) != NUM_LABELS:
        raise ValueError(
            f"Expected labels of length {NUM_LABELS}; got {len(sample)}. "
            "Regenerate the multilabel dataset with the correct category count."
        )

    logger.info(f"Loaded {len(raw)} examples. Tokenizing...")
    tokenized = raw.map(
        lambda b: tokenize_batch(b, tokenizer, max_length),
        batched=True,
        remove_columns=[c for c in raw.column_names if c != "labels"],
    )
    split = tokenized.train_test_split(test_size=0.1, seed=42)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        num_train_epochs=epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="micro_f1",
        greater_is_better=True,
        fp16=False,  # default off for portability; set True on GPU
        logging_steps=50,
        report_to="none",
        warmup_ratio=0.1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    logger.info("Starting training...")
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    logger.info("Building per-category report...")
    preds = trainer.predict(split["test"])
    probs = 1.0 / (1.0 + np.exp(-preds.predictions))
    y_pred = (probs > PROB_THRESHOLD).astype(int)
    y_true = preds.label_ids.astype(int)
    report = classification_report(
        y_true, y_pred, target_names=CUAD_CATEGORIES, zero_division=0, output_dict=True
    )
    report_path = output_dir / "eval_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(
        f"DONE. micro_f1={report['micro avg']['f1-score']:.4f}, "
        f"macro_f1={report['macro avg']['f1-score']:.4f}; report: {report_path}"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a multi-label CUAD classifier locally")
    parser.add_argument("--model", default="microsoft/deberta-v3-small")
    parser.add_argument("--data", default="data/processed/cuad_multilabel.jsonl")
    parser.add_argument("--output", default="models/cuad-multilabel-classifier")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-5)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[3]
    train_classifier(
        dataset_path=root / args.data,
        output_dir=root / args.output,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        max_length=args.max_length,
        learning_rate=args.lr,
    )


if __name__ == "__main__":
    main()
