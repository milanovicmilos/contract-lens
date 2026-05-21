"""
Download trained model artifacts produced by the Kaggle kernels.

Pulls /kaggle/working output zips for the classifier and extractor kernels,
extracts them into models/, and writes a manifest with metric snapshots so
the API server can pick the right model directory via env vars.

USAGE:
    python scripts/pull_kaggle_models.py --kernels classifier extractor

This relies on the kaggle CLI being configured (~/.kaggle/kaggle.json).
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

KERNEL_DEFINITIONS: Dict[str, Dict[str, str]] = {
    "classifier": {
        "kernel_id": "milomilanovi/contractlens-classifier-training",
        "expected_artifact": "deberta-cuad-classifier",
        "metric_file": "eval_report.json",
        "model_dir_name": "deberta-cuad-classifier",
    },
    "extractor": {
        "kernel_id": "milomilanovi/contractlens-extractor-training",
        "expected_artifact": "deberta-cuad-extractor",
        "metric_file": "extractor_eval.json",
        "model_dir_name": "deberta-cuad-extractor",
    },
}


def kaggle_status(kernel_id: str) -> str:
    """Return the latest status string for a kernel run."""
    result = subprocess.run(
        ["kaggle", "kernels", "status", kernel_id],
        capture_output=True,
        text=True,
        check=False,
    )
    return (result.stdout or result.stderr).strip()


def kaggle_pull_output(kernel_id: str, dest: Path) -> bool:
    """Download all output files into dest. Returns True on success."""
    dest.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["kaggle", "kernels", "output", kernel_id, "-p", str(dest)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        logger.error(f"Failed to pull output for {kernel_id}: {result.stderr}")
        return False
    logger.info(f"Pulled outputs for {kernel_id} -> {dest}")
    return True


def write_manifest(model_root: Path, kernels: List[str]) -> Path:
    """Write models/MANIFEST.json summarising downloaded artifacts and metrics."""
    manifest: Dict[str, Dict] = {}
    for key in kernels:
        spec = KERNEL_DEFINITIONS[key]
        artifact_dir = model_root / spec["model_dir_name"]
        metric_path = model_root / spec["metric_file"]

        entry: Dict = {
            "kernel_id": spec["kernel_id"],
            "model_dir": (
                str(artifact_dir.relative_to(model_root.parent)) if artifact_dir.exists() else None
            ),
            "metrics_file": (
                str(metric_path.relative_to(model_root.parent)) if metric_path.exists() else None
            ),
        }
        if metric_path.exists():
            try:
                with open(metric_path, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
                if "micro avg" in metrics:
                    entry["headline"] = {
                        "micro_f1": metrics["micro avg"]["f1-score"],
                        "macro_f1": metrics["macro avg"]["f1-score"],
                    }
                elif "f1" in metrics:
                    entry["headline"] = {
                        "exact_match": metrics.get("exact_match"),
                        "f1": metrics.get("f1"),
                    }
            except Exception as exc:
                logger.warning(f"Could not parse metrics for {key}: {exc}")
        manifest[key] = entry

    manifest_path = model_root / "MANIFEST.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Wrote manifest to {manifest_path}")
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description="Pull Kaggle-trained model artifacts")
    parser.add_argument(
        "--kernels",
        nargs="+",
        default=["classifier", "extractor"],
        choices=list(KERNEL_DEFINITIONS),
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Destination directory (under project root)",
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Just print kernel status, do not download.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    model_root = project_root / args.models_dir

    if args.status_only:
        for key in args.kernels:
            spec = KERNEL_DEFINITIONS[key]
            print(f"{spec['kernel_id']}: {kaggle_status(spec['kernel_id'])}")
        return 0

    for key in args.kernels:
        spec = KERNEL_DEFINITIONS[key]
        status = kaggle_status(spec["kernel_id"])
        logger.info(f"{key} status: {status}")
        if "COMPLETE" not in status.upper() and "FINISHED" not in status.upper():
            logger.warning(
                f"{key} not complete yet (status={status}); skipping download. "
                "Re-run when kernel status is 'COMPLETE'."
            )
            continue
        if not kaggle_pull_output(spec["kernel_id"], model_root):
            logger.error(f"Aborting due to download failure for {key}")
            return 1

    write_manifest(model_root, args.kernels)
    return 0


if __name__ == "__main__":
    sys.exit(main())
