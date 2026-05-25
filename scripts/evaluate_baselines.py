"""
Baseline comparison script for ContractLens.

Evaluates two classifiers on the held-out CUAD eval split (same seed=42 /
test_size=0.1 split used during Kaggle training) and writes a structured
comparison report to docs/baseline_eval_report.json.

Classifiers compared:
  1. KeywordClassifier — regex heuristics, no ML (the lower bound)
  2. HFClassifier (v9.2) — DeBERTa-v3-base + LoRA, per-category tuned thresholds

Usage:
    python scripts/evaluate_baselines.py [--model-dir models/deberta-cuad-classifier]
                                         [--data data/processed/cuad_multilabel.jsonl]
                                         [--output docs/baseline_eval_report.json]
                                         [--skip-neural]   # keyword only, fast

The HFClassifier step takes ~25 s on CPU to load plus ~1 s per batch.
Run with --skip-neural to produce the keyword-only report quickly.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure repo root is on path when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.infrastructure.ai.hf_classifier import CUAD_CATEGORIES
from src.infrastructure.ai.keyword_classifier import KeywordClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_eval_split(
    jsonl_path: str,
    test_size: float = 0.1,
    seed: int = 42,
) -> Tuple[List[str], np.ndarray]:
    """Return (texts, labels) for the held-out eval split.

    Replicates the exact HuggingFace train_test_split used in the Kaggle
    training kernel so the comparison is on the identical 1 818-row set.
    """
    rows: List[Dict[str, Any]] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    n = len(rows)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    n_test = int(n * test_size)
    eval_indices = indices[:n_test]  # first n_test after permutation = test

    texts = [rows[i]["text"] for i in eval_indices]
    labels = np.array([rows[i]["labels"] for i in eval_indices], dtype=int)

    logger.info("Eval split: %d rows, %d categories", len(texts), labels.shape[1])
    return texts, labels


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def _preds_from_keyword(texts: List[str], classifier: KeywordClassifier) -> np.ndarray:
    cat_order = CUAD_CATEGORIES
    preds = np.zeros((len(texts), len(cat_order)), dtype=int)
    for i, text in enumerate(texts):
        scores = classifier.classify(text)
        for j, cat in enumerate(cat_order):
            preds[i, j] = 1 if scores.get(cat, 0.0) >= 0.5 else 0
    return preds


def _preds_from_neural(
    texts: List[str],
    model_dir: str,
    batch_size: int = 16,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Returns (binary_preds, thresholds_used)."""
    from src.infrastructure.ai.hf_classifier import HFClassifier

    logger.info("Loading HFClassifier from %s …", model_dir)
    clf = HFClassifier(model_name_or_path=model_dir, device=-1)

    # Load per-category thresholds directly from thresholds.json (same file
    # that HFClassifier.per_category_thresholds exposes once PR-H is merged).
    thresh_path = Path(model_dir) / "thresholds.json"
    if thresh_path.exists():
        with open(thresh_path, encoding="utf-8") as f:
            thresholds = json.load(f)
        logger.info("Loaded %d per-category thresholds from %s", len(thresholds), thresh_path)
    else:
        thresholds = {}
        logger.warning("thresholds.json not found — using global 0.55 threshold for all categories")

    default_thresh = 0.55

    preds = np.zeros((len(texts), len(CUAD_CATEGORIES)), dtype=int)
    for i, text in enumerate(texts):
        scores = clf.classify(text)
        for j, cat in enumerate(CUAD_CATEGORIES):
            thresh = thresholds.get(cat, default_thresh)
            preds[i, j] = 1 if scores.get(cat, 0.0) >= thresh else 0
        if (i + 1) % 200 == 0:
            logger.info("  Neural inference: %d / %d", i + 1, len(texts))

    return preds, thresholds


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    categories: List[str],
) -> Dict[str, Any]:
    """Compute per-category and aggregate F1 / precision / recall."""
    eps = 1e-9
    per_cat = {}
    for j, cat in enumerate(categories):
        tp = int(((y_true[:, j] == 1) & (y_pred[:, j] == 1)).sum())
        fp = int(((y_true[:, j] == 0) & (y_pred[:, j] == 1)).sum())
        fn = int(((y_true[:, j] == 1) & (y_pred[:, j] == 0)).sum())
        support = int(y_true[:, j].sum())
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        per_cat[cat] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    # Micro aggregate
    tp_all = int(((y_true == 1) & (y_pred == 1)).sum())
    fp_all = int(((y_true == 0) & (y_pred == 1)).sum())
    fn_all = int(((y_true == 1) & (y_pred == 0)).sum())
    micro_p = tp_all / (tp_all + fp_all + eps)
    micro_r = tp_all / (tp_all + fn_all + eps)
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r + eps)

    # Macro aggregate (only categories with support > 0)
    f1_vals = [v["f1"] for v in per_cat.values() if v["support"] > 0]
    macro_f1 = float(np.mean(f1_vals)) if f1_vals else 0.0

    return {
        "micro_f1": round(micro_f1, 4),
        "micro_precision": round(micro_p, 4),
        "micro_recall": round(micro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "per_category": per_cat,
    }


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------


def _print_comparison_table(keyword_metrics: Dict, neural_metrics: Optional[Dict]) -> None:
    print()
    print("=" * 72)
    print(f"{'Category':<38} {'Keyword F1':>10} {'v9.2 F1':>10} {'Support':>8}")
    print("-" * 72)
    per_k = keyword_metrics["per_category"]
    per_n = neural_metrics["per_category"] if neural_metrics else {}
    for cat in CUAD_CATEGORIES:
        k = per_k.get(cat, {})
        n = per_n.get(cat, {})
        k_f1 = f"{k.get('f1', 0):.3f}" if k.get("support", 0) > 0 else "   —  "
        n_f1 = f"{n.get('f1', 0):.3f}" if n else "  N/A "
        support = k.get("support", 0)
        print(f"  {cat:<36} {k_f1:>10} {n_f1:>10} {support:>8}")
    print("=" * 72)
    print(
        f"  {'MICRO F1':<36} {keyword_metrics['micro_f1']:>10.3f} "
        f"{neural_metrics['micro_f1'] if neural_metrics else 'N/A':>10} {'':>8}"
    )
    print(
        f"  {'MACRO F1':<36} {keyword_metrics['macro_f1']:>10.3f} "
        f"{neural_metrics['macro_f1'] if neural_metrics else 'N/A':>10} {'':>8}"
    )
    print("=" * 72)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="ContractLens baseline comparison")
    parser.add_argument(
        "--model-dir",
        default="models/deberta-cuad-classifier",
        help="Path to the v9.2 model directory (default: models/deberta-cuad-classifier)",
    )
    parser.add_argument(
        "--data",
        default="data/processed/cuad_multilabel.jsonl",
        help="Path to the CUAD multi-label JSONL (default: data/processed/cuad_multilabel.jsonl)",
    )
    parser.add_argument(
        "--output",
        default="docs/baseline_eval_report.json",
        help="Output JSON path (default: docs/baseline_eval_report.json)",
    )
    parser.add_argument(
        "--skip-neural",
        action="store_true",
        help="Only run the keyword baseline (fast, no model loading)",
    )
    args = parser.parse_args()

    # -------- load data
    logger.info("Loading eval split from %s …", args.data)
    texts, y_true = load_eval_split(args.data)

    # -------- keyword baseline
    logger.info("Running keyword classifier …")
    t0 = time.time()
    kw_clf = KeywordClassifier()
    y_pred_kw = _preds_from_keyword(texts, kw_clf)
    kw_time = time.time() - t0
    kw_metrics = _compute_metrics(y_true, y_pred_kw, CUAD_CATEGORIES)
    logger.info(
        "Keyword baseline — micro F1: %.3f  macro F1: %.3f  (%.1f s)",
        kw_metrics["micro_f1"],
        kw_metrics["macro_f1"],
        kw_time,
    )

    # -------- neural classifier
    neural_metrics: Optional[Dict] = None
    neural_thresholds: Dict[str, float] = {}
    if not args.skip_neural:
        if not Path(args.model_dir).exists():
            logger.warning(
                "Neural model directory %s not found. "
                "Run 'inv pull-models' to download. Using --skip-neural mode.",
                args.model_dir,
            )
        else:
            logger.info("Running v9.2 DeBERTa classifier …")
            t0 = time.time()
            y_pred_neural, neural_thresholds = _preds_from_neural(texts, args.model_dir)
            neural_time = time.time() - t0
            neural_metrics = _compute_metrics(y_true, y_pred_neural, CUAD_CATEGORIES)
            logger.info(
                "v9.2 DeBERTa — micro F1: %.3f  macro F1: %.3f  (%.1f s)",
                neural_metrics["micro_f1"],
                neural_metrics["macro_f1"],
                neural_time,
            )

    # -------- print table
    _print_comparison_table(kw_metrics, neural_metrics)

    # -------- save report
    report = {
        "eval_split": {"n_rows": len(texts), "test_size": 0.1, "seed": 42},
        "keyword_baseline": kw_metrics,
        "v9_2_deberta": neural_metrics,
        "thresholds_used": neural_thresholds,
        "notes": (
            "Keyword baseline: regex patterns, no ML. "
            "v9.2 DeBERTa: per-category F1-optimal thresholds from thresholds.json. "
            "Eval split replicates the Kaggle training split (seed=42, test_size=0.1)."
        ),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Report saved to %s", out)


if __name__ == "__main__":
    main()
