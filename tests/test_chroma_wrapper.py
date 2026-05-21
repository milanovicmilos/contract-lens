"""
Basic tests for ChromaDB Wrapper to ensure the implementation is correct
and follows the boundaries defined in IVectorDatabase.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.infrastructure.database.chroma_wrapper import ChromaWrapper


@pytest.fixture
def mock_chroma_client():
    with (
        patch("src.infrastructure.database.chroma_wrapper.chromadb") as mock_chroma,
        patch("src.infrastructure.database.chroma_wrapper.CHROMA_AVAILABLE", True),
    ):
        mock_client_instance = MagicMock()
        mock_collection_instance = MagicMock()

        mock_client_instance.get_or_create_collection.return_value = mock_collection_instance
        mock_chroma.PersistentClient.return_value = mock_client_instance

        yield mock_chroma, mock_client_instance, mock_collection_instance


def test_chroma_init_success(mock_chroma_client):
    """Test standard successful initialization of the Chroma DB."""
    _, mock_client, mock_collection = mock_chroma_client

    db = ChromaWrapper(persist_directory="./fake_dir", collection_name="test_col")

    mock_client.get_or_create_collection.assert_called_once_with(name="test_col")
    assert db._collection is not None


def test_chroma_init_failure():
    """Test behavior when chromadb library is missing or fails."""
    with patch("src.infrastructure.database.chroma_wrapper.CHROMA_AVAILABLE", False):
        db = ChromaWrapper()
        assert db._collection is None


def test_add_texts(mock_chroma_client):
    """Test adding documents to the database."""
    _, _, mock_collection = mock_chroma_client

    db = ChromaWrapper()

    texts = ["Regulation 1", "Regulation 2"]
    metadatas = [{"source": "GDPR"}, {"source": "CCPA"}]

    db.add_texts(texts=texts, metadatas=metadatas)

    # Assert collection add was called
    mock_collection.add.assert_called_once()
    kwargs = mock_collection.add.call_args.kwargs
    assert "documents" in kwargs
    assert kwargs["documents"] == texts
    assert "metadatas" in kwargs
    assert kwargs["metadatas"] == metadatas
    assert "ids" in kwargs


def test_search(mock_chroma_client):
    """Test searching the vector database."""
    _, _, mock_collection = mock_chroma_client

    mock_collection.query.return_value = {
        "documents": [["Result 1", "Result 2"]],
        "metadatas": [[{"source": "law1"}, {"source": "law2"}]],
        "distances": [[0.1, 0.5]],
    }

    db = ChromaWrapper()
    results = db.search("Find laws", top_k=2)

    assert len(results) == 2
    assert results[0]["text"] == "Result 1"
    assert results[0]["metadata"] == {"source": "law1"}
    assert results[0]["score"] == 0.1
