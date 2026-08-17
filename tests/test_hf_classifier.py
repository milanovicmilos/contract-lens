from unittest.mock import MagicMock, patch

import pytest

from src.infrastructure.ai.hf_classifier import HFClassifier


# Fixture to mock the transformers pipeline to avoid downloading models during CI
@pytest.fixture
def mock_pipeline():
    with patch("src.infrastructure.ai.hf_classifier.pipeline") as mock:
        yield mock


def test_hf_classifier_initialization_success(mock_pipeline):
    """Test that the classifier initializes without errors when pipeline succeeds."""
    mock_pipeline.return_value = MagicMock()

    classifier = HFClassifier(model_name_or_path="fake/model", device=-1)

    # Assert pipeline was built exactly once
    mock_pipeline.assert_called_once_with(
        "text-classification", model="fake/model", device=-1, top_k=None
    )
    assert classifier.nlp_pipeline is not None


def test_hf_classifier_initialization_failure(mock_pipeline):
    """Test that the classifier explicitly raises an error if pipeline fails to load."""
    mock_pipeline.side_effect = Exception("Model not found on HuggingFace Hub")

    with pytest.raises(RuntimeError) as exc_info:
        HFClassifier(model_name_or_path="invalid/model")

    assert "Pipeline initialization failed" in str(exc_info.value)


def test_hf_classifier_classify_valid_text(mock_pipeline):
    """Test standard classification flow mapping pipeline output to correct Dict structure."""
    # Setup mock pipeline to return standard HF text-classification top_k=None output
    mock_nlp_instance = MagicMock()
    mock_nlp_instance.return_value = [
        [{"label": "Termination", "score": 0.95}, {"label": "Governing Law", "score": 0.05}]
    ]
    mock_pipeline.return_value = mock_nlp_instance

    classifier = HFClassifier()
    result = classifier.classify("The contract may be terminated at will.")

    # Expected transformations
    assert isinstance(result, dict)
    assert "Termination" in result
    assert "Governing Law" in result
    assert result["Termination"] == 0.95
    assert result["Governing Law"] == 0.05


def test_hf_classifier_classify_empty_text(mock_pipeline):
    """Test handling of empty string."""
    mock_pipeline.return_value = MagicMock()
    classifier = HFClassifier()

    result = classifier.classify(" ")
    assert result == {}


def test_hf_classifier_classify_runtime_failure(mock_pipeline):
    """Test explicit error propagation if inference fails."""
    mock_nlp_instance = MagicMock()
    mock_nlp_instance.side_effect = Exception("CUDA OOM error")
    mock_pipeline.return_value = mock_nlp_instance

    classifier = HFClassifier()

    with pytest.raises(RuntimeError) as exc_info:
        classifier.classify("Some failing text")

    assert "Classification failed" in str(exc_info.value)
