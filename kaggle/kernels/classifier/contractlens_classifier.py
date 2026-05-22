"""
ContractLens - Multi-label Classifier Training on Kaggle.

STANDALONE Kaggle script. No src/ imports. Copy directly into a Kaggle notebook
or push as a kernel:

    kaggle kernels push -p kaggle/

ENVIRONMENT:
- Accelerator: GPU T4 x2 (or P100)
- Internet: ON (for HuggingFace model download)
- Input dataset: contractlens-cuad-multilabel
  - file: cuad_multilabel.jsonl  ({"text", "labels": [0..1] x 41})

OUTPUT (in /kaggle/working/):
- deberta-cuad-classifier/  (LoRA-adapted model + tokenizer)
- eval_report.json          (sklearn classification_report)
"""

import json
import logging
import os
import sys

# ---------------------------------------------------------------------------
# 1. Install dependencies + CUDA-compatible torch
#    Kaggle may assign P100 (sm_60) or T4 (sm_75). Default torch on Kaggle
#    dropped sm_60 support; pin to torch 2.4 cu121 which still supports both.
# ---------------------------------------------------------------------------
def _ensure_deps():
    import subprocess

    def _pip(*args):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])

    # Reinstall torch with broad GPU compatibility (P100 sm_60 + T4 sm_75).
    # Pin transformers to a version that doesn't enforce torch>=2.6 for .bin loading
    # (CVE check that was added in transformers 5.x but we still need torch 2.4 for sm_60).
    _pip(
        "--upgrade",
        "--force-reinstall",
        "torch==2.4.0",
        "torchvision==0.19.0",
        "--index-url",
        "https://download.pytorch.org/whl/cu121",
    )
    _pip(
        "transformers==4.46.0",
        "tokenizers>=0.20,<0.21",
        "peft==0.13.0",
        "accelerate>=0.34,<1.0",
        "scikit-learn>=1.3.0",
        "datasets>=3.0,<3.5",
    )


_ensure_deps()

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


class MultiLabelTrainer(Trainer):
    """Class-weighted BCE for multi-label.

    Forces float labels and applies BCEWithLogitsLoss with pos_weight to
    counteract the extreme class imbalance in CUAD (e.g. Parties has 4071
    positives, Price Restrictions only 53). Without weighting, the model
    learns to never predict rare categories because the negative class
    overwhelms the loss signal.

    pos_weight per class = max(1, (N - n_pos) / max(n_pos, 1))
    """

    def __init__(self, *args, pos_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        logits = outputs.logits
        if self.pos_weight is not None:
            pw = self.pos_weight.to(logits.device)
            loss_fct = nn.BCEWithLogitsLoss(pos_weight=pw)
        else:
            loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_pos_weight(dataset, num_labels: int):
    """Inverse-frequency pos_weight per label from training set."""
    n_pos = np.zeros(num_labels, dtype=np.float64)
    total = 0
    for ex in dataset:
        n_pos += np.array(ex["labels"], dtype=np.float64)
        total += 1
    n_neg = total - n_pos
    # Avoid div-by-zero; cap weight to 100 to prevent loss explosion on ultra-rare classes
    pw = np.where(n_pos > 0, n_neg / np.maximum(n_pos, 1.0), 1.0)
    pw = np.clip(pw, 1.0, 100.0)
    return torch.tensor(pw, dtype=torch.float32)


def tune_thresholds(probs, y_true, num_labels: int):
    """Per-category threshold that maximizes F1 on the eval set."""
    best_thresholds = np.full(num_labels, PROB_THRESHOLD)
    for i in range(num_labels):
        if y_true[:, i].sum() == 0:
            continue
        best_f1 = 0.0
        for t in np.arange(0.1, 0.95, 0.05):
            preds = (probs[:, i] > t).astype(int)
            f1 = f1_score(y_true[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresholds[i] = t
    return best_thresholds

# ---------------------------------------------------------------------------
# 2. Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "microsoft/deberta-v3-base"  # safetensors available on main; safer with torch 2.4
# Kaggle mounts private CLI-attached datasets under /kaggle/input/datasets/<owner>/<slug>/.
# The discovery fallback in main() handles other mount conventions transparently.
DATASET_PATH = "/kaggle/input/datasets/milomilanovi/contractlens-cuad-multilabel/cuad_multilabel.jsonl"
OUTPUT_DIR = "/kaggle/working/deberta-cuad-classifier"
EVAL_REPORT_PATH = "/kaggle/working/eval_report.json"
NUM_LABELS = 41
MAX_LENGTH = 512
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
assert len(CUAD_CATEGORIES) == NUM_LABELS


# ---------------------------------------------------------------------------
# 3. Metrics + Tokenization
# ---------------------------------------------------------------------------
def compute_metrics(eval_pred):
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


def tokenize_fn(batch, tokenizer):
    enc = tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH, padding="max_length")
    enc["labels"] = [[float(v) for v in row] for row in batch["labels"]]
    return enc


# ---------------------------------------------------------------------------
# 4. Main training pipeline
# ---------------------------------------------------------------------------
def main():
    logger.info(f"GPU available: {torch.cuda.is_available()}, count: {torch.cuda.device_count()}")
    logger.info(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    logger.info(f"Loading dataset from {DATASET_PATH}")
    if not os.path.exists(DATASET_PATH):
        logger.error(f"Expected {DATASET_PATH} not present. Scanning /kaggle/input/ ...")
        for root, _dirs, files in os.walk("/kaggle/input"):
            for fname in files:
                logger.error(f"  found: {os.path.join(root, fname)}")
        # Try to auto-discover the dataset file
        discovered = None
        for root, _dirs, files in os.walk("/kaggle/input"):
            for fname in files:
                if fname == "cuad_multilabel.jsonl":
                    discovered = os.path.join(root, fname)
                    break
            if discovered:
                break
        if discovered is None:
            raise FileNotFoundError(
                f"Dataset not found at {DATASET_PATH} and not discovered under /kaggle/input. "
                "Attach 'milomilanovi/contractlens-cuad-multilabel' to the kernel."
            )
        logger.warning(f"Falling back to discovered path: {discovered}")
        dataset_path_to_use = discovered
    else:
        dataset_path_to_use = DATASET_PATH

    raw = load_dataset("json", data_files=dataset_path_to_use, split="train")
    sample = raw[0]["labels"]
    assert len(sample) == NUM_LABELS, f"Label vector length mismatch: {len(sample)} != {NUM_LABELS}"
    logger.info(f"Loaded {len(raw)} examples")

    tokenized = raw.map(
        lambda b: tokenize_fn(b, tokenizer),
        batched=True,
        remove_columns=[c for c in raw.column_names if c not in ("labels",)],
    )
    split = tokenized.train_test_split(test_size=0.1, seed=42)
    logger.info(f"Train: {len(split['train'])}, Val: {len(split['test'])}")

    logger.info("Loading base model + applying LoRA")
    # transformers 4.46 (pinned above) does not enforce the torch>=2.6 CVE check,
    # so the upstream pytorch_model.bin loads safely on torch 2.4. We avoid
    # use_safetensors=True because HF's auto-conversion service has been
    # intermittently failing with KeyError on 'event_id'.
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
    )

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=32,
        lora_alpha=64,
        lora_dropout=0.1,
        target_modules=["query_proj", "value_proj", "key_proj"],
    )
    model = get_peft_model(model, peft_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {trainable:,}/{total:,} ({100*trainable/total:.2f}%)")

    logger.info("Computing per-class pos_weight from training set distribution...")
    pos_weight = compute_pos_weight(split["train"], NUM_LABELS)
    logger.info(
        f"pos_weight stats: min={pos_weight.min().item():.2f}, "
        f"max={pos_weight.max().item():.2f}, mean={pos_weight.mean().item():.2f}"
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        num_train_epochs=8,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="micro_f1",
        greater_is_better=True,
        fp16=True,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=50,
        report_to="none",
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
    )

    trainer = MultiLabelTrainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        pos_weight=pos_weight,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info(f"Saving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    logger.info("Computing per-category classification report...")
    preds = trainer.predict(split["test"])
    probs = 1.0 / (1.0 + np.exp(-preds.predictions))
    y_true = preds.label_ids.astype(int)

    # Two reports: default 0.5 threshold, and per-class tuned thresholds (better
    # for rare categories where 0.5 is too conservative).
    y_pred_default = (probs > PROB_THRESHOLD).astype(int)
    report_default = classification_report(
        y_true, y_pred_default, target_names=CUAD_CATEGORIES, zero_division=0, output_dict=True
    )

    thresholds = tune_thresholds(probs, y_true, NUM_LABELS)
    y_pred_tuned = (probs > thresholds[None, :]).astype(int)
    report_tuned = classification_report(
        y_true, y_pred_tuned, target_names=CUAD_CATEGORIES, zero_division=0, output_dict=True
    )

    combined = {
        "default_threshold_0_5": report_default,
        "tuned_thresholds": report_tuned,
        "thresholds_per_category": {
            CUAD_CATEGORIES[i]: float(thresholds[i]) for i in range(NUM_LABELS)
        },
    }
    with open(EVAL_REPORT_PATH, "w") as f:
        json.dump(combined, f, indent=2)
    # Persist thresholds next to model for downstream HFClassifier loading.
    with open(f"{OUTPUT_DIR}/thresholds.json", "w") as f:
        json.dump(
            {CUAD_CATEGORIES[i]: float(thresholds[i]) for i in range(NUM_LABELS)},
            f,
            indent=2,
        )

    logger.info(f"Eval report written to {EVAL_REPORT_PATH}")
    logger.info(
        f"DEFAULT  threshold=0.5: micro_f1={report_default['micro avg']['f1-score']:.4f}, "
        f"macro_f1={report_default['macro avg']['f1-score']:.4f}"
    )
    logger.info(
        f"TUNED  per-class thresholds: micro_f1={report_tuned['micro avg']['f1-score']:.4f}, "
        f"macro_f1={report_tuned['macro avg']['f1-score']:.4f}"
    )


if __name__ == "__main__":
    main()
