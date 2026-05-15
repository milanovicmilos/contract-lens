"""
Hugging Face Trainer script for fine-tuning Extractive QA (DeBERTa) on CUAD.

This script parses the `cuad_squad.jsonl` export and fine-tunes a model
using standardized overlapping chunking (stride) from HF tokenizers.
"""

import argparse
import logging
from pathlib import Path

from datasets import load_dataset
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Standard QA preprocessing optimized for SQuAD format
def prepare_train_features(examples, tokenizer, pad_on_right=True, max_length=512, doc_stride=128):
    """
    Tokenize our examples with truncation and padding, but keep the overflows using a stride.
    This creates multiple features from a single long context.
    """
    # Tokenizer output includes offset mappings
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized_examples.sequence_ids(i)

        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]

        # If no answers are given, set the cls_index as answer.
        # Format expects answers to be a dict: {'text': [...], 'answer_start': [...]}
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # SQuAD/CUAD gives start character position
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Find context sequence
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (
                offsets[token_start_index][0] <= start_char
                and offsets[token_end_index][1] >= end_char
            ):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Find token positions
                while (
                    token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char
                ):
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)

                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples


def main():
    parser = argparse.ArgumentParser(description="Train local Extractive QA Prototype")
    parser.add_argument(
        "--model_name", type=str, default="microsoft/deberta-v3-base", help="Base model"
    )
    parser.add_argument(
        "--data_file", type=str, default="data/processed/cuad_squad.jsonl", help="JSONL Dataset"
    )
    parser.add_argument(
        "--output_dir", type=str, default="models/deberta-cuad-prototype", help="Output directory"
    )
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent.parent.parent.parent
    data_path = root_dir / args.data_file
    out_path = root_dir / args.output_dir

    if not data_path.exists():
        logger.error(
            f"Dataset not found at {data_path}. Run scripts/prepare_squad_dataset.py first."
        )
        return

    logger.info("Loading dataset via HuggingFace datasets...")
    # HF datasets load jsonl inherently
    raw_datasets = load_dataset("json", data_files={"train": str(data_path)})

    # Split into train/validation natively
    split = raw_datasets["train"].train_test_split(test_size=0.1, seed=42)

    logger.info(f"Loading Tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    pad_on_right = tokenizer.padding_side == "right"

    logger.info("Applying Map to chunk and tokenize data...")
    tokenized_datasets = split.map(
        lambda examples: prepare_train_features(
            examples,
            tokenizer=tokenizer,
            pad_on_right=pad_on_right,
            max_length=512,  # Typical context max length
            doc_stride=128,  # Overlapping chunks stride
        ),
        batched=True,
        remove_columns=split["train"].column_names,
    )

    logger.info("Initializing AutoModelForQuestionAnswering...")
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_name)

    training_args = TrainingArguments(
        output_dir=str(out_path),
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=50,
        fp16=True,  # Mixed precision
    )

    data_collator = DefaultDataCollator()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info(f"Saving final model to {out_path}")
    trainer.save_model(str(out_path))


if __name__ == "__main__":
    main()
