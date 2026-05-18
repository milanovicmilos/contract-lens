"""
Kaggle Training Script Template.

Due to memory constraints of training large models like DeBERTa-v3-large
on a standard PC, this script is designed to be executed directly on Kaggle
using their T4/P100 GPUs or TPUs.

INSTRUCTIONS:
1. Ensure your dataset is uploaded to Kaggle (e.g., using kaggle API: `kaggle datasets create`).
2. Update the paths below to point to the Kaggle /kaggle/input/... directories.
3. Run this script in a Kaggle Notebook or submit as a Kaggle script.
"""

import logging
import os

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)

# Constants
MODEL_NAME = "microsoft/deberta-v3-large"
# Kaggle input paths will typically look like this:
DATASET_PATH = os.environ.get("KAGGLE_DATASET_PATH", "./data/processed/cuad_squad.jsonl")
OUTPUT_DIR = os.environ.get("KAGGLE_OUTPUT_DIR", "./kaggle/working/models/deberta-v3-large")


def main():
    logging.basicConfig(level=logging.INFO)
    logger.info(f"Starting Kaggle training job for {MODEL_NAME}")

    # 1. Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Num labels depends on our dataset. Here we assume 4 for a prototype.
    # We use problem_type="multi_label_classification"
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=4, problem_type="multi_label_classification"
    )

    # Handle PEFT/LoRA setup here if memory is an issue even on 16GB GPUs
    try:
        from peft import LoraConfig, TaskType, get_peft_model

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS, inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1
        )
        model = get_peft_model(model, peft_config)
        logger.info("LoRA configuration applied.")
    except ImportError:
        logger.warning("PEFT not installed. Training full model (Beware OOM).")

    # 2. Dataset Loading (Mock logic for Kaggle env)
    logger.info(f"Looking for dataset at: {DATASET_PATH}")
    if os.path.exists(DATASET_PATH):
        dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

        # Preprocessing function
        def preprocess(examples):
            return tokenizer(
                examples["text"], truncation=True, max_length=512, padding="max_length"
            )

        tokenized_dataset = dataset.map(preprocess, batched=True)
    else:
        logger.warning(
            "Dataset not found. Creating dummy dataset for validation of the script pipeline."
        )
        # Dummy data for pipeline validation
        dummy_data = {
            "text": [
                "Contract liability is capped.",
                "This clause terminates the agreement in 30 days.",
            ],
            "labels": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        }
        tokenized_dataset = Dataset.from_dict(dummy_data).map(
            lambda x: tokenizer(x["text"], truncation=True, max_length=128, padding="max_length"),
            batched=True,
        )

    # 3. Training Arguments (Optimized for T4/P100 16GB VRAM)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=2e-5,
        per_device_train_batch_size=4,  # Keep small for DeBERTa large
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        fp16=torch.cuda.is_available(),  # Enable mixed precision
        logging_dir=f"{OUTPUT_DIR}/logs",
        report_to="none",  # Or "wandb"
    )

    # Split for dummy eval
    split = tokenized_dataset.train_test_split(test_size=0.1)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
    )

    # 4. Train
    logger.info("Starting training loop...")
    trainer.train()

    # 5. Save model
    logger.info(f"Saving final model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
