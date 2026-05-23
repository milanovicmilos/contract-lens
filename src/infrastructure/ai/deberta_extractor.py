"""
DeBERTa-based Extractor implementation for Hugging Face models.

Uses AutoModelForQuestionAnswering directly (no `pipeline("question-answering")`)
because transformers 5.x removed the QA task from the public pipeline registry.
We replicate the same span-selection logic (top-k start/end index pairing,
impossible-answer handling) so the IExtractor contract is unchanged.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from src.application.interfaces.iextractor import ExtractionResult, IExtractor

logger = logging.getLogger(__name__)


class DebertaExtractor(IExtractor):
    """Local extractive QA model satisfying the IExtractor contract.

    Args:
        model_name_or_path: HF Hub id or local path to a fine-tuned QA model.
        device: -1 for CPU, 0+ for a specific GPU. If None, autodetect.
        max_length: Max encoded sequence length (truncates context).
        max_answer_length: Max number of tokens in a predicted span.
        n_best_size: How many start/end candidates to consider per pass.
    """

    def __init__(
        self,
        model_name_or_path: str = "deepset/deberta-v3-base-squad2",
        device: Optional[int] = None,
        max_length: int = 512,
        max_answer_length: int = 256,
        n_best_size: int = 20,
        impossible_threshold: float = 0.0,
    ) -> None:
        if device is None:
            device = 0 if torch.cuda.is_available() else -1

        logger.info(f"Loading local extraction model: {model_name_or_path} on device={device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name_or_path)
        self.device = torch.device("cuda", device) if device >= 0 else torch.device("cpu")
        self.model.to(self.device).eval()
        self.max_length = max_length
        self.max_answer_length = max_answer_length
        self.n_best_size = n_best_size
        self.impossible_threshold = impossible_threshold
        self._model_id = model_name_or_path
        logger.info("Extraction model loaded successfully.")

    def extract(self, context: str, question: str, top_k: int = 1) -> List[ExtractionResult]:
        if not context.strip():
            return []

        try:
            enc = self.tokenizer(
                question,
                context,
                max_length=self.max_length,
                truncation="only_second",
                padding="max_length",
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            offset_mapping = enc.pop("offset_mapping")[0].tolist()
            sequence_ids = enc.sequence_ids(0)

            with torch.no_grad():
                out = self.model(
                    input_ids=enc["input_ids"].to(self.device),
                    attention_mask=enc["attention_mask"].to(self.device),
                )

            start_logits = out.start_logits[0].cpu().numpy()
            end_logits = out.end_logits[0].cpu().numpy()

            # Mask out positions outside the context (question + special tokens get -inf)
            context_mask = np.array([sid == 1 for sid in sequence_ids], dtype=bool)
            start_logits = np.where(context_mask, start_logits, -1e9)
            end_logits = np.where(context_mask, end_logits, -1e9)

            cls_index = 0
            cls_logit = (out.start_logits[0, cls_index] + out.end_logits[0, cls_index]).item()

            start_idx_candidates = np.argsort(start_logits)[-self.n_best_size:][::-1]
            end_idx_candidates = np.argsort(end_logits)[-self.n_best_size:][::-1]

            candidates: List[ExtractionResult] = []
            for s in start_idx_candidates:
                for e in end_idx_candidates:
                    if e < s or (e - s + 1) > self.max_answer_length:
                        continue
                    if offset_mapping[s] is None or offset_mapping[e] is None:
                        continue
                    if not (context_mask[s] and context_mask[e]):
                        continue
                    start_char, _ = offset_mapping[s]
                    _, end_char = offset_mapping[e]
                    if start_char is None or end_char is None or end_char <= start_char:
                        continue
                    text = context[int(start_char):int(end_char)].strip()
                    if not text:
                        continue
                    score = float(start_logits[s] + end_logits[e])
                    # Optional impossible-answer suppression: only enabled when
                    # impossible_threshold > 0. The classic SQuAD2 rule says reject
                    # the span if CLS beats it; for CUAD-trained extractors the CLS
                    # logit is often dominant by default, so we keep all candidates
                    # unless the caller explicitly opts in to the suppression margin.
                    if self.impossible_threshold > 0 and score <= (cls_logit + self.impossible_threshold):
                        continue
                    candidates.append(
                        ExtractionResult(
                            text=text,
                            answer_start=int(start_char),
                            answer_end=int(end_char),
                            score=score,
                            question=question,
                            metadata={"model": self._model_id},
                        )
                    )

            candidates.sort(key=lambda r: -r.score)
            return candidates[:top_k]
        except Exception as exc:
            logger.error(f"Error during extraction: {exc}")
            raise

    def batch_extract(
        self, contexts: List[str], question: str, top_k: int = 1
    ) -> List[List[ExtractionResult]]:
        return [self.extract(ctx, question, top_k) for ctx in contexts]
