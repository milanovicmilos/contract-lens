"""
Unit tests for the AI Extractor module utilizing Mocks.
This ensures we don't load heavy models during standard CI test runs.
"""

import pytest
from unittest.mock import patch, MagicMock

from src.infrastructure.ai.deberta_extractor import DebertaExtractor
from src.application.interfaces.iextractor import ExtractionResult

class TestDebertaExtractor:
    
    @patch("src.infrastructure.ai.deberta_extractor.pipeline")
    def test_extractor_initialization(self, mock_pipeline):
        """Test that the pipeline is initialized correctly."""
        mock_pipeline.return_value = MagicMock()
        
        extractor = DebertaExtractor(model_name_or_path="dummy-model", device=-1)
        
        mock_pipeline.assert_called_once_with(
            task="question-answering",
            model="dummy-model",
            tokenizer="dummy-model",
            device=-1,
            handle_impossible_answer=True
        )
        assert extractor.qa_pipeline is not None

    @patch("src.infrastructure.ai.deberta_extractor.pipeline")
    def test_extract_returns_correct_results(self, mock_pipeline):
        """Test extraction correctly maps pipeline output to Domain/Application structs."""
        # Setup mock pipeline behavior
        mock_pipe_instance = MagicMock()
        mock_pipe_instance.model.name_or_path = "mock-model"
        
        # SQuAD pipeline returns a dict if top_k=1, or list of dicts if top_k>1.
        # Let's mock a scenario where pipeline returns a single dict (top_k=1).
        mock_pipe_instance.return_value = {
            "score": 0.95,
            "start": 10,
            "end": 20,
            "answer": "John Doe"
        }
        mock_pipeline.return_value = mock_pipe_instance
        
        extractor = DebertaExtractor("dummy-model")
        
        results = extractor.extract(
            context="My name is John Doe, the contractor.",
            question="Who is the contractor?",
            top_k=1
        )
        
        assert len(results) == 1
        res = results[0]
        assert isinstance(res, ExtractionResult)
        assert res.text == "John Doe"
        assert res.score == 0.95
        assert res.answer_start == 10
        assert res.answer_end == 20
        assert res.question == "Who is the contractor?"
        assert res.metadata["model"] == "mock-model"

    @patch("src.infrastructure.ai.deberta_extractor.pipeline")
    def test_extract_handles_empty_context(self, mock_pipeline):
        """Test that an empty context gracefully returns an empty list."""
        mock_pipe_instance = MagicMock()
        mock_pipeline.return_value = mock_pipe_instance
        
        extractor = DebertaExtractor("dummy-model")
        results = extractor.extract(context="   ", question="What?")
        
        assert results == []
        mock_pipe_instance.assert_not_called()

    @patch("src.infrastructure.ai.deberta_extractor.pipeline")
    def test_extract_filters_impossible_answers(self, mock_pipeline):
        """Test that empty string answers (impossible answers) are filtered out."""
        mock_pipe_instance = MagicMock()
        mock_pipe_instance.model.name_or_path = "mock-model"
        # SQuAD pipelines can return an empty string for impossible answers
        mock_pipe_instance.return_value = {"answer": "", "score": 0.99, "start": 0, "end": 0}
        mock_pipeline.return_value = mock_pipe_instance
        
        extractor = DebertaExtractor("dummy-model")
        results = extractor.extract(context="No answer here.", question="What?")
        
        assert len(results) == 0
