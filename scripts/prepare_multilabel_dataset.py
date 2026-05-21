"""
Convert CUAD SQuAD-format JSONL to multi-label classification format.

Input:  data/processed/cuad_squad.jsonl (SQuAD QA format, 1 row per question)
        Each row: {"id", "contract_id", "context", "question", "answers", "is_impossible"}

Output: data/processed/cuad_multilabel.jsonl (1 row per text window)
        Each row: {"text": str, "labels": List[int]} (length 41, binary)

STRATEGY:
- Aggregate per contract: which categories appear (is_impossible=False with answers).
- Generate two types of examples:
  1) POSITIVE windows: text spans containing actual answer texts -> labeled with categories
  2) NEGATIVE windows: random spans from contracts that don't overlap any answer
- This avoids the dilution problem of multi-label on full 50k+ char contracts.

USAGE:
    python scripts/prepare_multilabel_dataset.py \
        --input data/processed/cuad_squad.jsonl \
        --output data/processed/cuad_multilabel.jsonl \
        --window-size 512 \
        --max-negatives-per-contract 20
"""

import argparse
import hashlib
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


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
CATEGORY_TO_IDX = {name: i for i, name in enumerate(CUAD_CATEGORIES)}


def extract_category_from_id(qa_id: str) -> str:
    """SQuAD id format: '<contract_title>__<Category Name>'."""
    if "__" in qa_id:
        return qa_id.rsplit("__", 1)[1].strip()
    return ""


def load_squad_jsonl(path: Path) -> List[dict]:
    """Load SQuAD JSONL file row by row."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    logger.info(f"Loaded {len(rows)} SQuAD rows from {path}")
    return rows


def group_by_contract(rows: List[dict]) -> Dict[str, dict]:
    """Group SQuAD rows by contract_id, accumulating answer spans per category."""
    contracts: Dict[str, dict] = defaultdict(lambda: {"context": "", "spans": []})

    for row in rows:
        contract_id = row.get("contract_id", row.get("title", ""))
        category = extract_category_from_id(row.get("id", ""))

        if not category or category not in CATEGORY_TO_IDX:
            continue

        if not contracts[contract_id]["context"]:
            contracts[contract_id]["context"] = row["context"]

        if not row.get("is_impossible", True):
            answers = row.get("answers", {})
            texts = answers.get("text", [])
            starts = answers.get("answer_start", [])
            for text, start in zip(texts, starts):
                if text and start is not None and start >= 0:
                    contracts[contract_id]["spans"].append({
                        "category": category,
                        "start": int(start),
                        "end": int(start) + len(text),
                        "text": text,
                    })

    logger.info(f"Grouped into {len(contracts)} contracts")
    return contracts


def make_window_around_span(
    context: str, span: dict, window_size: int
) -> Tuple[int, int, str]:
    """Center a window of `window_size` chars around a span; clamp to context bounds."""
    span_center = (span["start"] + span["end"]) // 2
    half = window_size // 2
    win_start = max(0, span_center - half)
    win_end = min(len(context), win_start + window_size)
    win_start = max(0, win_end - window_size)
    return win_start, win_end, context[win_start:win_end]


def spans_overlapping_window(spans: List[dict], win_start: int, win_end: int) -> List[str]:
    """Return distinct categories whose spans overlap with [win_start, win_end)."""
    cats = set()
    for s in spans:
        if s["start"] < win_end and s["end"] > win_start:
            cats.add(s["category"])
    return sorted(cats)


def make_label_vector(categories: Set[str]) -> List[int]:
    vec = [0] * len(CUAD_CATEGORIES)
    for c in categories:
        if c in CATEGORY_TO_IDX:
            vec[CATEGORY_TO_IDX[c]] = 1
    return vec


def generate_positive_examples(
    contract: dict, window_size: int
) -> List[dict]:
    """For each answer span, generate a labeled window centered on it."""
    context = contract["context"]
    spans = contract["spans"]
    examples = []

    for span in spans:
        win_start, win_end, text = make_window_around_span(context, span, window_size)
        overlapping_cats = spans_overlapping_window(spans, win_start, win_end)
        examples.append({
            "text": text,
            "labels": make_label_vector(set(overlapping_cats)),
        })

    return examples


def generate_negative_examples(
    contract: dict, window_size: int, max_negatives: int, rng: random.Random
) -> List[dict]:
    """Sample random windows that do NOT overlap any answer span."""
    context = contract["context"]
    spans = contract["spans"]
    examples = []

    if len(context) < window_size:
        return examples

    occupied = [(s["start"], s["end"]) for s in spans]

    attempts = 0
    max_attempts = max_negatives * 10
    while len(examples) < max_negatives and attempts < max_attempts:
        attempts += 1
        win_start = rng.randint(0, len(context) - window_size)
        win_end = win_start + window_size

        overlaps = any(s_start < win_end and s_end > win_start for s_start, s_end in occupied)
        if overlaps:
            continue

        examples.append({
            "text": context[win_start:win_end],
            "labels": [0] * len(CUAD_CATEGORIES),
        })

    return examples


def deterministic_seed(contract_id: str) -> int:
    """Reproducible RNG seed per contract."""
    h = hashlib.md5(contract_id.encode()).hexdigest()
    return int(h[:8], 16)


def build_dataset(
    contracts: Dict[str, dict],
    window_size: int,
    max_negatives_per_contract: int,
) -> List[dict]:
    """Produce final multi-label dataset, balancing positive and negative examples."""
    examples = []
    pos_count = 0
    neg_count = 0

    for contract_id, contract in contracts.items():
        if not contract["context"]:
            continue

        rng = random.Random(deterministic_seed(contract_id))

        positives = generate_positive_examples(contract, window_size)
        examples.extend(positives)
        pos_count += len(positives)

        negatives = generate_negative_examples(
            contract, window_size, max_negatives_per_contract, rng
        )
        examples.extend(negatives)
        neg_count += len(negatives)

    logger.info(f"Generated {pos_count} positive + {neg_count} negative = {len(examples)} examples")
    return examples


def write_jsonl(path: Path, examples: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(examples)} examples to {path}")


def label_distribution_report(examples: List[dict]) -> Dict[str, int]:
    """Count examples per category for sanity check."""
    counts = {cat: 0 for cat in CUAD_CATEGORIES}
    for ex in examples:
        for i, v in enumerate(ex["labels"]):
            if v == 1:
                counts[CUAD_CATEGORIES[i]] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description="Convert CUAD SQuAD to multi-label format")
    parser.add_argument(
        "--input", type=str, default="data/processed/cuad_squad.jsonl",
        help="Path to SQuAD-formatted CUAD JSONL",
    )
    parser.add_argument(
        "--output", type=str, default="data/processed/cuad_multilabel.jsonl",
        help="Output multi-label JSONL path",
    )
    parser.add_argument(
        "--window-size", type=int, default=2000,
        help="Character window size (DeBERTa tokenizer chars ~ 4 per token, so 2000 ~ 500 tokens)",
    )
    parser.add_argument(
        "--max-negatives-per-contract", type=int, default=10,
        help="Max negative (no-clause) windows sampled per contract",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    input_path = project_root / args.input
    output_path = project_root / args.output

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1

    rows = load_squad_jsonl(input_path)
    contracts = group_by_contract(rows)
    examples = build_dataset(
        contracts,
        window_size=args.window_size,
        max_negatives_per_contract=args.max_negatives_per_contract,
    )

    # Shuffle deterministically for stable train/val splits downstream
    rng = random.Random(42)
    rng.shuffle(examples)

    write_jsonl(output_path, examples)

    dist = label_distribution_report(examples)
    logger.info("=== Label distribution (top 10) ===")
    for cat, n in sorted(dist.items(), key=lambda kv: -kv[1])[:10]:
        logger.info(f"  {cat:40s}: {n}")
    logger.info("=== Label distribution (bottom 5) ===")
    for cat, n in sorted(dist.items(), key=lambda kv: kv[1])[:5]:
        logger.info(f"  {cat:40s}: {n}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
