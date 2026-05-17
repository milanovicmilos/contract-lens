"""
Hugging Face-based Multi-label Classifier implementation.

This infrastructure component implements IClassifier using the local
Hugging Face `transformers` sequence classification architectures,
respecting the Clean Architecture boundaries.
"""

import logging
from typing import Dict

from transformers import pipeline

from src.application.interfaces.iclassifier import IClassifier

logger = logging.getLogger(__name__)


class HFClassifier(IClassifier):
    """
    Local Deep Learning classifier for multi-label text classification.
    Used to categorize clauses into specific types (e.g., Termination, Governing Law).
    """

    def __init__(
        self,
        model_name_or_path: str = "typeform/distilbert-base-uncased-mnli",  # Placeholder default
        device: int = -1,
    ):
        """
        Initializes the Hugging Face text classification pipeline.

        Args:
            model_name_or_path: Path to the local fine-tuned model or HF Hub model name.
            device: -1 for CPU, 0+ for GPU.
        """
        logger.info(f"Initializing HFClassifier with model {model_name_or_path}")
        try:
            # Note: For multi-label, the model needs to be trained with problem_type="multi_label_classification".
            # `return_all_scores=True` (or `top_k=None` in newer transformers) returns probabilities for all labels.
            self.nlp_pipeline = pipeline(
                "text-classification",
                model=model_name_or_path,
                device=device,
                top_k=None,  # Returns all scores
            )
        except Exception as e:
            logger.error(f"Failed to load the pipeline for model {model_name_or_path}")
            raise RuntimeError(f"Pipeline initialization failed: {e}") from e

    def classify(self, text: str) -> Dict[str, float]:
        """
        Classifies a given text and returns risk labels with confidence scores.

        Args:
            text: The clause or paragraph text to classify.

        Returns:
            Dictionary mapping label names to probability scores (0.0 to 1.0).
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for classification.")
            return {}

        try:
            # Pipeline returns a list of lists: [[{'label': 'LABEL_0', 'score': 0.9}, ...]]
            results = self.nlp_pipeline(text)

            # Extract the first prediction since we only pass a single string at a time
            predictions = results[0]

            # Format the output into a dictionary mapping labels to scores
            return {pred["label"]: float(pred["score"]) for pred in predictions}

        except Exception as e:
            logger.error(f"Failed to classify text: {str(e)}")
            raise RuntimeError(f"Classification failed: {e}") from e
