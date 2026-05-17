"""
ChromaDB implementation of the IVectorDatabase interface.
Used for local Retrieval-Augmented Generation (RAG).
"""

import logging
from typing import Any, Dict, List

try:
    import chromadb

    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    chromadb = None  # type: ignore

from src.application.interfaces.ivector_db import IVectorDatabase

logger = logging.getLogger(__name__)


class ChromaWrapper(IVectorDatabase):
    """
    Local Vector Database wrapper using Chroma.
    Maintains the privacy-first architecture by keeping embeddings local.
    """

    def __init__(
        self, persist_directory: str = "./data/chroma_db", collection_name: str = "regulations"
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self._collection = None
        self._init_db()

    def _init_db(self):
        """Initializes the actual Chroma client."""
        if not CHROMA_AVAILABLE or chromadb is None:
            logger.error("chromadb is not installed. Run `pip install chromadb`.")
            self._collection = None
            return

        try:
            logger.info(f"Initializing ChromaDB at {self.persist_directory}")
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name
            )
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            self._collection = None

    def add_texts(
        self, texts: List[str], metadatas: List[Dict[str, Any]] = None, ids: List[str] = None
    ) -> None:
        if not self._collection:
            logger.warning("ChromaDB is not initialized. Cannot add texts.")
            return

        if not ids:
            import uuid

            ids = [str(uuid.uuid4()) for _ in texts]

        try:
            self._collection.add(
                documents=texts,
                metadatas=metadatas or [{} for _ in texts],
                ids=ids,
            )
            logger.info(f"Added {len(texts)} documents to ChromaDB.")
        except Exception as e:
            logger.error(f"Error adding texts to ChromaDB: {e}")
            raise

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Searches documents by similarity."""
        if not self._collection:
            logger.warning("ChromaDB is not initialized. Cannot search texts.")
            return []

        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=top_k,
            )

            formatted_results = []
            if results["documents"] and len(results["documents"]) > 0:
                docs = results["documents"][0]
                metas = (
                    results["metadatas"][0]
                    if results["metadatas"]
                    else [{} for _ in docs]
                )
                distances = (
                    results["distances"][0] if results["distances"] else [0.0 for _ in docs]
                )

                for doc, meta, dist in zip(docs, metas, distances, strict=False):
                    formatted_results.append(
                        {
                            "text": doc,
                            "metadata": meta,
                            "score": dist,
                        }
                    )

            return formatted_results
        except Exception as e:
            logger.error(f"Error querying ChromaDB: {e}")
            return []
