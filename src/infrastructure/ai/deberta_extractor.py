"""
DeBERTa-based Extractor implementation for Hugging Face models.

This infrastructure component implements IExtractor using the local
Hugging Face `transformers` pipeline, satisfying the Clean Architecture boundaries.
"""

import logging
from typing import List, Dict, Any

from transformers import pipeline

from src.application.interfaces.iextractor import IExtractor, ExtractionResult

logger = logging.getLogger(__name__)


class DebertaExtractor(IExtractor):
    """
    Local Deep Learning extractor mostly utilizing DeBERTa architecture
    for Extractive QA (SQuAD format).
    """

    def __init__(self, model_name_or_path: str = "deepset/deberta-v3-base-squad2", device: int = -1):
        """
        Initializes the Extractive QA pipeline.
        
        Args:
            model_name_or_path: Hugging Face model hub name or local path.
            device: GPU device ID (-1 for CPU, 0 for first GPU).
        """
        logger.info(f"Loading local extraction model: {model_name_or_path}")
        # Initialize standard HF QA pipeline
        # Use handle_impossible_answer=True because some clauses don't exist in the contract
        self.qa_pipeline = pipeline(
            task="question-answering",
            model=model_name_or_path,
            tokenizer=model_name_or_path,
            device=device,
            handle_impossible_answer=True
        )
        logger.info("Extraction model loaded successfully.")

    def extract(self, context: str, question: str, top_k: int = 1) -> List[ExtractionResult]:
        """Extract optimal spans from the context answering the question."""
        if not context.strip():
            return []

        try:
            # The pipeline returns a dict (top_k=1) or list of dicts (top_k>1)
            predictions = self.qa_pipeline(
                question=question,
                context=context,
                top_k=top_k,
                max_answer_len=256,
                handle_impossible_answer=True
            )
            
            if isinstance(predictions, dict):
                predictions = [predictions]
                
            results = []
            for pred in predictions:
                # SQuAD pipelines return "" for impossible answers but with high scores.
                # We filter out empty answers or impossible answers strictly if needed.
                if pred["answer"] == "" or pred["score"] < 0.01:
                    continue
                    
                result = ExtractionResult(
                    text=pred["answer"],
                    answer_start=pred["start"],
                    answer_end=pred["end"],
                    score=pred["score"],
                    question=question,
                    metadata={"model": self.qa_pipeline.model.name_or_path}
                )
                results.append(result)
                
            return results
        except Exception as e:
            logger.error(f"Error during extraction: {e}")
            raise

    def batch_extract(self, contexts: List[str], question: str, top_k: int = 1) -> List[List[ExtractionResult]]:
        """Batch extraction wrapper. (Naïve loop for PoC, can be optimized for DataLoader)"""
        return [self.extract(ctx, question, top_k) for ctx in contexts]
