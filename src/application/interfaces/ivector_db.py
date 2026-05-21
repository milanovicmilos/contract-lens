from abc import ABC, abstractmethod
from typing import Any, Dict, List


class IVectorDatabase(ABC):
    """
    Interface for Vector Database operations.
    Used for storing and retrieving legal constraints, regulatory documents,
    or reference clauses in the RAG pipeline.
    """

    @abstractmethod
    def add_texts(
        self, texts: List[str], metadatas: List[Dict[str, Any]] = None, ids: List[str] = None
    ) -> None:
        """
        Adds texts to the vector database.

        Args:
            texts: List of text strings to embed and add.
            metadatas: Optional list of metadata dictionaries.
            ids: Optional list of unique IDs for the texts.
        """
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Searches the vector database for the top_k most similar texts.

        Args:
            query: The search query.
            top_k: Number of results to return.

        Returns:
            A list of results containing at least 'text', 'score', and 'metadata'.
        """
        pass
