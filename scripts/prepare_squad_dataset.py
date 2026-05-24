"""
Prepares the CUAD dataset into a flattened SQuAD-like JSONL format suitable
for Hugging Face datasets and extractive QA training.
"""

import json
import logging
import sys
from pathlib import Path

# Ensure src is in PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.cuad_loader import CUADDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    root_dir = Path(__file__).resolve().parent.parent
    data_dir = root_dir / "data" / "processed"
    data_dir.mkdir(parents=True, exist_ok=True)

    out_file = data_dir / "cuad_squad.jsonl"

    logger.info("Initializing CUAD Loader...")
    cuad_dir = root_dir / "CUAD_v1"

    dataset = CUADDataset(dataset_dir=cuad_dir)
    logger.info("Loading texts and matches...")
    examples, categories = dataset.load()

    logger.info(f"Loaded {len(examples)} contracts. Extracting QAs...")

    # Flatten the dataset to a JSONL where each line is a context+QA pair
    # However, standard SQuAD has {id, context, question, answers: {text: [], answer_start: []}}
    # We will slice documents into chunks using HF tokenizer during training,
    # but here we provide document-level JSONL.

    written_count = 0
    with open(out_file, "w", encoding="utf-8") as f:
        for ex in examples:
            context = ex.context

            for qa in ex.qas:
                # SQuAD v2 style record
                record = {
                    "id": qa["id"],
                    "title": ex.title,
                    "context": context,
                    "question": qa["question"],
                    "answers": {
                        "text": [ans["text"] for ans in qa["answers"]],
                        "answer_start": [ans["answer_start"] for ans in qa["answers"]],
                    },
                    "is_impossible": qa.get("is_impossible", len(qa["answers"]) == 0),
                    "contract_id": ex.contract_id,
                }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                written_count += 1

    logger.info(f"Successfully saved {written_count} QA records to {out_file}")


if __name__ == "__main__":
    main()
