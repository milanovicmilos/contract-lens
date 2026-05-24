"""
Unit tests for the AI Extractor module utilizing Mocks.
This ensures we don't load heavy models during standard CI test runs.
"""

from unittest.mock import MagicMock, patch

import torch

from src.application.interfaces.iextractor import ExtractionResult
from src.infrastructure.ai.deberta_extractor import DebertaExtractor


def _patch_extractor_dependencies():
    """Common patch set: stub tokenizer + model so no HF download happens."""
    return (
        patch("src.infrastructure.ai.deberta_extractor.AutoTokenizer"),
        patch("src.infrastructure.ai.deberta_extractor.AutoModelForQuestionAnswering"),
        patch(
            "src.infrastructure.ai.deberta_extractor.torch.cuda.is_available", return_value=False
        ),
    )


class TestDebertaExtractor:
    def test_extractor_initialization(self):
        with (
            patch("src.infrastructure.ai.deberta_extractor.AutoTokenizer") as mock_tok,
            patch(
                "src.infrastructure.ai.deberta_extractor.AutoModelForQuestionAnswering"
            ) as mock_model,
        ):
            mock_tok.from_pretrained.return_value = MagicMock()
            mock_model.from_pretrained.return_value = MagicMock()
            extractor = DebertaExtractor(model_name_or_path="dummy-model", device=-1)

            mock_tok.from_pretrained.assert_called_once_with("dummy-model", use_fast=True)
            mock_model.from_pretrained.assert_called_once_with("dummy-model")
            assert extractor.model is not None
            assert extractor.tokenizer is not None
            assert extractor._model_id == "dummy-model"

    def test_extract_returns_correct_results(self):
        """The extractor should pair the highest start/end logits inside the context window
        and surface a non-empty ExtractionResult with the corresponding character offsets."""
        with (
            patch("src.infrastructure.ai.deberta_extractor.AutoTokenizer") as mock_tok,
            patch(
                "src.infrastructure.ai.deberta_extractor.AutoModelForQuestionAnswering"
            ) as mock_model,
        ):
            # Mock tokenizer encoding: token positions 1-5 correspond to context chars.
            mock_encoding = MagicMock()
            mock_encoding.__getitem__.side_effect = lambda k: {
                "input_ids": torch.tensor([[101, 200, 201, 202, 203, 204, 102]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 1]]),
            }[k]
            mock_encoding.pop.return_value = torch.tensor(
                [[[0, 0], [0, 4], [4, 8], [9, 13], [14, 18], [19, 27], [0, 0]]]
            )
            mock_encoding.sequence_ids.return_value = [None, 0, 0, 1, 1, 1, None]
            tok_inst = MagicMock(return_value=mock_encoding)
            tok_inst.return_value = mock_encoding
            mock_tok.from_pretrained.return_value = tok_inst

            mock_out = MagicMock()
            mock_out.start_logits = torch.tensor([[-9.0, -9.0, -9.0, 5.0, 0.0, 0.0, -9.0]])
            mock_out.end_logits = torch.tensor([[-9.0, -9.0, -9.0, 0.0, 0.0, 5.0, -9.0]])
            model_inst = MagicMock(return_value=mock_out)
            model_inst.to.return_value = model_inst
            mock_model.from_pretrained.return_value = model_inst

            extractor = DebertaExtractor(model_name_or_path="dummy-model", device=-1)
            results = extractor.extract(
                context="John Doe contractor party",
                question="Who is the contractor?",
                top_k=1,
            )

            assert len(results) == 1
            res = results[0]
            assert isinstance(res, ExtractionResult)
            assert res.answer_start == 9  # offset_mapping[3][0]
            assert res.answer_end == 27  # offset_mapping[5][1]
            assert res.text == "contractor party"
            assert res.question == "Who is the contractor?"
            assert res.score > 0

    def test_extract_handles_empty_context(self):
        with (
            patch("src.infrastructure.ai.deberta_extractor.AutoTokenizer") as mock_tok,
            patch(
                "src.infrastructure.ai.deberta_extractor.AutoModelForQuestionAnswering"
            ) as mock_model,
        ):
            mock_tok.from_pretrained.return_value = MagicMock()
            model_inst = MagicMock()
            model_inst.to.return_value = model_inst
            mock_model.from_pretrained.return_value = model_inst

            extractor = DebertaExtractor(model_name_or_path="dummy-model", device=-1)
            results = extractor.extract(context="   ", question="What?")

            assert results == []

    def test_extract_filters_impossible_answers(self):
        """When impossible_threshold>0 is set and CLS logit dominates every
        candidate span, impossible-answer suppression must return an empty list.

        The default (impossible_threshold=0) intentionally keeps all valid spans
        because CUAD-fine-tuned QA models often have a dominant CLS logit even
        when the answer exists; production callers can opt in to suppression."""
        with (
            patch("src.infrastructure.ai.deberta_extractor.AutoTokenizer") as mock_tok,
            patch(
                "src.infrastructure.ai.deberta_extractor.AutoModelForQuestionAnswering"
            ) as mock_model,
        ):
            mock_encoding = MagicMock()
            mock_encoding.__getitem__.side_effect = lambda k: {
                "input_ids": torch.tensor([[101, 200, 201, 202, 102]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
            }[k]
            mock_encoding.pop.return_value = torch.tensor(
                [[[0, 0], [0, 4], [4, 8], [9, 13], [0, 0]]]
            )
            mock_encoding.sequence_ids.return_value = [None, 0, 1, 1, None]
            tok_inst = MagicMock(return_value=mock_encoding)
            mock_tok.from_pretrained.return_value = tok_inst

            # CLS gets a huge logit; context candidates are weak.
            mock_out = MagicMock()
            mock_out.start_logits = torch.tensor([[10.0, -9.0, -9.0, -9.0, -9.0]])
            mock_out.end_logits = torch.tensor([[10.0, -9.0, -9.0, -9.0, -9.0]])
            model_inst = MagicMock(return_value=mock_out)
            model_inst.to.return_value = model_inst
            mock_model.from_pretrained.return_value = model_inst

            extractor = DebertaExtractor(
                model_name_or_path="dummy-model",
                device=-1,
                impossible_threshold=1.0,
            )
            results = extractor.extract(context="No answer here.", question="What?")

            assert results == []
