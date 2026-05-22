"""
ContractLens - Extractive QA Training on Kaggle.

STANDALONE Kaggle script for DeBERTa-v3-large fine-tuning on CUAD SQuAD-format.
No src/ imports. Copy directly into a Kaggle notebook or push as a kernel.

ENVIRONMENT:
- Accelerator: GPU T4 x2 (or P100)
- Internet: ON
- Input dataset: contractlens-cuad-squad
  - file: cuad_squad.jsonl (SQuAD-format: id, context, question, answers, is_impossible)

OUTPUT (in /kaggle/working/):
- deberta-cuad-extractor/  (QA model + tokenizer)
- extractor_eval.json      (exact match + F1 token-level)
"""

import collections
import json
import logging
import os
import string
import subprocess
import sys
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Install CUDA-compatible torch (P100 sm_60 + T4 sm_75) before importing torch.
# ---------------------------------------------------------------------------
def _ensure_deps():
    def _pip(*args):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])

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
        "accelerate>=0.34,<1.0",
        "datasets>=3.0,<3.5",
    )


_ensure_deps()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "microsoft/deberta-v3-base"  # safetensors available on main; safer with torch 2.4
# Kaggle mounts private CLI-attached datasets under /kaggle/input/datasets/<owner>/<slug>/.
DATASET_PATH = "/kaggle/input/datasets/milomilanovi/contractlens-cuad-squad/cuad_squad.jsonl"
OUTPUT_DIR = "/kaggle/working/deberta-cuad-extractor"
EVAL_REPORT_PATH = "/kaggle/working/extractor_eval.json"

MAX_LENGTH = 512
DOC_STRIDE = 128


# ---------------------------------------------------------------------------
# 1. Preprocessing (standard SQuAD/CUAD tokenization with sliding window)
# ---------------------------------------------------------------------------
def prepare_train_features(examples, tokenizer, pad_on_right=True):
    """Tokenize (question, context) pairs with stride, mapping char answer to token positions."""
    tokenized = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=MAX_LENGTH,
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    tokenized["start_positions"] = []
    tokenized["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized.sequence_ids(i)

        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]

        if len(answers["answer_start"]) == 0:
            tokenized["start_positions"].append(cls_index)
            tokenized["end_positions"].append(cls_index)
        else:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            if not (
                offsets[token_start_index][0] <= start_char
                and offsets[token_end_index][1] >= end_char
            ):
                tokenized["start_positions"].append(cls_index)
                tokenized["end_positions"].append(cls_index)
            else:
                while (
                    token_start_index < len(offsets)
                    and offsets[token_start_index][0] <= start_char
                ):
                    token_start_index += 1
                tokenized["start_positions"].append(token_start_index - 1)

                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized["end_positions"].append(token_end_index + 1)

    return tokenized


def prepare_validation_features(examples, tokenizer, pad_on_right=True):
    """Validation features keep offset mapping for post-processing."""
    tokenized = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=MAX_LENGTH,
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    tokenized["example_id"] = []
    for i in range(len(tokenized["input_ids"])):
        sequence_ids = tokenized.sequence_ids(i)
        context_index = 1 if pad_on_right else 0
        sample_index = sample_mapping[i]
        tokenized["example_id"].append(examples["id"][sample_index])

        tokenized["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized["offset_mapping"][i])
        ]
    return tokenized


# ---------------------------------------------------------------------------
# 2. Post-processing: convert logits to text answers (best span search)
# ---------------------------------------------------------------------------
def postprocess_qa_predictions(
    examples, features, raw_predictions, n_best_size=20, max_answer_length=200
):
    all_start_logits, all_end_logits = raw_predictions
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    predictions = collections.OrderedDict()
    for example_index, example in enumerate(examples):
        feature_indices = features_per_example[example_index]
        min_null_score = None
        valid_answers = []
        context = example["context"]

        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]
            cls_index = features[feature_index]["input_ids"].index(0)  # rough
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append({
                        "score": start_logits[start_index] + end_logits[end_index],
                        "text": context[start_char:end_char],
                    })

        if valid_answers:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            best_answer = {"text": "", "score": 0.0}
        predictions[example["id"]] = best_answer["text"]
    return predictions


# ---------------------------------------------------------------------------
# 3. Evaluation: exact match + token-level F1 (CUAD standard)
# ---------------------------------------------------------------------------
def normalize_text(s: str) -> str:
    def remove_articles(text):
        return " ".join(w for w in text.split() if w not in ("a", "an", "the"))
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def compute_f1(pred: str, truth: str) -> float:
    pred_tokens = normalize_text(pred).split()
    truth_tokens = normalize_text(truth).split()
    if not pred_tokens or not truth_tokens:
        return float(pred_tokens == truth_tokens)
    common = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate_predictions(predictions: Dict[str, str], examples) -> Dict[str, float]:
    em_scores, f1_scores = [], []
    for ex in examples:
        pred = predictions.get(ex["id"], "")
        gold_texts = ex["answers"]["text"]
        if not gold_texts:
            em = 1.0 if pred == "" else 0.0
            f1 = 1.0 if pred == "" else 0.0
        else:
            em = max(float(normalize_text(pred) == normalize_text(t)) for t in gold_texts)
            f1 = max(compute_f1(pred, t) for t in gold_texts)
        em_scores.append(em)
        f1_scores.append(f1)
    return {
        "exact_match": float(np.mean(em_scores)),
        "f1": float(np.mean(f1_scores)),
        "num_examples": len(examples),
    }


# ---------------------------------------------------------------------------
# 4. Main training pipeline
# ---------------------------------------------------------------------------
def main():
    logger.info(f"GPU available: {torch.cuda.is_available()}")
    dataset_path_to_use = DATASET_PATH
    if not os.path.exists(DATASET_PATH):
        logger.error(f"Expected {DATASET_PATH} not present. Scanning /kaggle/input/ ...")
        discovered = None
        for root, _dirs, files in os.walk("/kaggle/input"):
            for fname in files:
                logger.error(f"  found: {os.path.join(root, fname)}")
                if fname == "cuad_squad.jsonl":
                    discovered = os.path.join(root, fname)
        if discovered is None:
            raise FileNotFoundError(
                f"Dataset not found at {DATASET_PATH} and not discovered under /kaggle/input."
            )
        logger.warning(f"Falling back to discovered path: {discovered}")
        dataset_path_to_use = discovered

    raw = load_dataset("json", data_files=dataset_path_to_use, split="train")
    logger.info(f"Loaded {len(raw)} QA examples")

    split = raw.train_test_split(test_size=0.1, seed=42)
    train_raw, val_raw = split["train"], split["test"]
    logger.info(f"Train: {len(train_raw)}, Val: {len(val_raw)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    pad_on_right = tokenizer.padding_side == "right"

    train_tokenized = train_raw.map(
        lambda ex: prepare_train_features(ex, tokenizer, pad_on_right),
        batched=True,
        remove_columns=train_raw.column_names,
    )

    val_tokenized = val_raw.map(
        lambda ex: prepare_validation_features(ex, tokenizer, pad_on_right),
        batched=True,
        remove_columns=val_raw.column_names,
    )

    # use_safetensors=True avoids torch.load CVE path that fails under torch 2.4.
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME, use_safetensors=True)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        learning_rate=3e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        warmup_ratio=0.1,
        fp16=True,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized.remove_columns(["example_id", "offset_mapping"]),
        tokenizer=tokenizer,
        data_collator=DefaultDataCollator(),
    )

    logger.info("Starting training...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info(f"Model saved to {OUTPUT_DIR}")

    logger.info("Running raw predictions on validation set...")
    raw_predictions = trainer.predict(
        val_tokenized.remove_columns(["example_id", "offset_mapping"])
    )
    predictions = postprocess_qa_predictions(
        val_raw,
        val_tokenized,
        (raw_predictions.predictions[0], raw_predictions.predictions[1]),
    )

    metrics = evaluate_predictions(predictions, val_raw)
    logger.info(f"FINAL: exact_match={metrics['exact_match']:.4f}, f1={metrics['f1']:.4f}")

    with open(EVAL_REPORT_PATH, "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
