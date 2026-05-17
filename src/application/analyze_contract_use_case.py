"""
Application Use Case: Analyze Contract.
Coordinates the NLP pipelines (Classifier, Extractor) with Domain Rules (RiskPolicy).
"""

import logging
from typing import List

from src.application.interfaces.iclassifier import IClassifier
from src.application.interfaces.iextractor import IExtractor
from src.application.interfaces.irisk_analyzer import RiskScore
from src.domain.risk_policy import RiskPolicy

logger = logging.getLogger(__name__)


class AnalyzeContractUseCase:
    """
    Coordinates the text processing, AI classification, extraction,
    and domain-level risk assessment.
    """

    def __init__(
        self,
        classifier: IClassifier,
        extractor: IExtractor,
        risk_policy: RiskPolicy,
        classification_threshold: float = 0.5,
    ):
        """
        Injects dependencies necessary for full contract processing.

        Args:
            classifier: Multi-label classifier (AI component).
            extractor: Question-answering/span extractor (AI component).
            risk_policy: Business rules for scoring risks.
            classification_threshold: Minimum confidence to consider a label present.
        """
        self.classifier = classifier
        self.extractor = extractor
        self.risk_policy = risk_policy
        self.classification_threshold = classification_threshold

    def execute(self, text_blocks: List[str]) -> List[RiskScore]:
        """
        Runs the full analysis pipeline on a list of text blocks.

        Args:
            text_blocks: List of paragraphs or sliding-window strings from the contract.

        Returns:
            A list of detected RiskScores across the entire text.
        """
        logger.info(f"Starting analysis on {len(text_blocks)} text blocks.")
        all_risks: List[RiskScore] = []

        for block_id, block in enumerate(text_blocks):
            if not block.strip():
                continue

            # Step 1: Classification - What kind of clauses are in this block?
            try:
                classifications = self.classifier.classify(block)
            except Exception as e:
                logger.error(f"Classification failed on block {block_id}: {e}")
                continue

            # Look for labels passing the threshold
            for label, score in classifications.items():
                if score >= self.classification_threshold:
                    logger.debug(f"Detected {label} (confidence: {score:.2f}) in block {block_id}.")

                    # Step 2: Extraction (Optional refining of span, if needed. For now treating block as span)
                    # You could use self.extractor.extract(...) for pinpoint accuracy
                    # For risk logic, we leverage the policy engine.

                    # Step 3: Domain Risk Assessment
                    risk_level, risk_score, justify = self.risk_policy.assess_risk(
                        category=label, text=block
                    )

                    all_risks.append(
                        RiskScore(
                            category=label,
                            risk_level=risk_level,
                            score=risk_score,
                            justification=justify,
                            extracted_span=block,
                            metadata={"block_id": block_id, "classification_confidence": score},
                        )
                    )

        logger.info(f"Analysis completed. Found {len(all_risks)} risk items.")
        return all_risks
