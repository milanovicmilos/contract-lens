"""
Hugging Face-based Multi-label Classifier implementation.

Supports three model loading modes:
1. Plain HF Hub model id (e.g., 'microsoft/deberta-v3-large') or local fine-tuned
   directory produced by `train_classifier.py`.
2. LoRA-adapted models produced by `kaggle_train_lora.py` / `notebook_classifier.py`:
   directories containing `adapter_config.json` are loaded via peft.PeftModel
   on top of the base model recorded in the adapter config.
3. Multi-label inference: the pipeline returns sigmoid probabilities for all
   labels when the model's config has `problem_type=multi_label_classification`.

Label remap: when the fine-tuned model has generic `LABEL_0..LABEL_N` ids,
we map them to the 41 CUAD category names so downstream policy lookups work.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

from src.application.interfaces.iclassifier import IClassifier

logger = logging.getLogger(__name__)


CUAD_CATEGORIES: List[str] = [
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


def _is_peft_adapter(model_path: str) -> bool:
    """Return True iff model_path points to a directory with adapter_config.json."""
    if not model_path:
        return False
    p = Path(model_path)
    return p.is_dir() and (p / "adapter_config.json").exists()


def _load_peft_model(adapter_path: str):
    """Reconstruct a base model + LoRA adapter and merge so the HF pipeline can use it."""
    import json as _json

    try:
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError(
            "Found a LoRA adapter directory but 'peft' is not installed. "
            "Install it via `pip install peft`."
        ) from exc

    adapter_cfg_path = Path(adapter_path) / "adapter_config.json"
    with open(adapter_cfg_path, "r", encoding="utf-8") as f:
        adapter_cfg = _json.load(f)

    base_model_name = adapter_cfg.get("base_model_name_or_path")
    if not base_model_name:
        raise RuntimeError(
            f"adapter_config.json at {adapter_cfg_path} missing base_model_name_or_path."
        )

    logger.info(f"Loading PEFT adapter '{adapter_path}' on top of base '{base_model_name}'")
    config = AutoConfig.from_pretrained(adapter_path)
    base = AutoModelForSequenceClassification.from_pretrained(base_model_name, config=config)
    model = PeftModel.from_pretrained(base, adapter_path)
    try:
        model = model.merge_and_unload()
    except Exception as exc:
        logger.warning(f"merge_and_unload failed ({exc}); using adapter wrapper directly.")
    return model


class HFClassifier(IClassifier):
    """Local multi-label classifier wrapper used by the orchestrator."""

    def __init__(
        self,
        model_name_or_path: str = "prajjwal1/bert-tiny",
        device: Optional[int] = None,
        label_mapping: Optional[List[str]] = None,
    ):
        """
        Args:
            model_name_or_path: HF Hub id or local path. If the local path
                contains adapter_config.json, the model is loaded as a
                PEFT adapter on top of its recorded base.
            device: -1 for CPU, 0+ for a specific GPU. If None, autodetect.
            label_mapping: Optional list of category names to override generic
                LABEL_N labels. Defaults to the 41 CUAD categories when the
                model has exactly 41 outputs.
        """
        if device is None:
            device = 0 if torch.cuda.is_available() else -1

        logger.info(f"Initializing HFClassifier with '{model_name_or_path}' on device={device}")
        try:
            if _is_peft_adapter(model_name_or_path):
                model = _load_peft_model(model_name_or_path)
                tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
                self.nlp_pipeline = pipeline(
                    "text-classification",
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    top_k=None,
                )
            else:
                self.nlp_pipeline = pipeline(
                    "text-classification",
                    model=model_name_or_path,
                    device=device,
                    top_k=None,
                )
        except Exception as exc:
            logger.error(f"Failed to load classifier '{model_name_or_path}': {exc}")
            raise RuntimeError(f"Pipeline initialization failed: {exc}") from exc

        self.label_mapping = self._resolve_label_mapping(label_mapping)

    def _resolve_label_mapping(self, override: Optional[List[str]]) -> Optional[Dict[str, str]]:
        """Build a 'LABEL_N -> human name' dict when applicable."""
        if override is not None:
            return {f"LABEL_{i}": name for i, name in enumerate(override)}

        try:
            model_config = self.nlp_pipeline.model.config
            num_labels = getattr(model_config, "num_labels", 0)
        except AttributeError:
            return None

        id2label = getattr(self.nlp_pipeline.model.config, "id2label", {}) or {}
        generic_labels = (
            all(str(v).startswith("LABEL_") for v in id2label.values()) if id2label else True
        )

        if num_labels == len(CUAD_CATEGORIES) and generic_labels:
            return {f"LABEL_{i}": name for i, name in enumerate(CUAD_CATEGORIES)}
        return None

    def classify(self, text: str) -> Dict[str, float]:
        if not text or not text.strip():
            return {}

        truncated = text[: int(os.getenv("CLASSIFIER_MAX_CHARS", "4000"))]
        try:
            results = self.nlp_pipeline(truncated)
            predictions = results[0] if results and isinstance(results[0], list) else results
            out: Dict[str, float] = {}
            for pred in predictions:
                label = pred["label"]
                if self.label_mapping and label in self.label_mapping:
                    label = self.label_mapping[label]
                out[label] = float(pred["score"])
            return out
        except Exception as exc:
            logger.error(f"Classification failed: {exc}")
            raise RuntimeError(f"Classification failed: {exc}") from exc
