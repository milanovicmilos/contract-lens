"""
ChromaDB implementation of the IVectorDatabase interface.

Used for local Retrieval-Augmented Generation (RAG) by the Legal
Consultant agent. Keeping embeddings local preserves the privacy-first
architecture — contract clauses and regulation snippets are embedded
without any vendor API call.

Embedding model selection
-------------------------
Chroma's default is sentence-transformers/all-MiniLM-L6-v2 (90 MB).
For legal text we recommend the larger BGE family — set
``EMBEDDING_MODEL=BAAI/bge-base-en-v1.5`` (440 MB) to improve recall on
clause-to-regulation matching. Smaller default keeps cold-start fast in
CI and on dev laptops; production overrides via env.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Dict, List, Optional

try:
    import chromadb
    from chromadb.utils import embedding_functions

    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    chromadb = None  # type: ignore
    embedding_functions = None  # type: ignore

from src.application.interfaces.ivector_db import IVectorDatabase

logger = logging.getLogger(__name__)


def _embedding_fn():
    """Build the embedding function from env, or return None to use Chroma's default."""
    if embedding_functions is None:
        return None
    model_name = (os.getenv("EMBEDDING_MODEL") or "").strip()
    if not model_name:
        return None  # Chroma's built-in default (all-MiniLM-L6-v2)
    try:
        return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
    except Exception as exc:
        logger.error(
            "Failed to build SentenceTransformer embedding function for %r: %s. "
            "Falling back to Chroma default. Install sentence-transformers or "
            "pick a different EMBEDDING_MODEL.",
            model_name,
            exc,
        )
        return None


class ChromaWrapper(IVectorDatabase):
    """Local Vector Database wrapper using Chroma."""

    def __init__(
        self,
        persist_directory: str = "./data/chroma_db",
        collection_name: str = "regulations",
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self._collection = None
        self._init_db()

    def _init_db(self) -> None:
        if not CHROMA_AVAILABLE or chromadb is None:
            logger.error("chromadb is not installed. Run `pip install chromadb`.")
            self._collection = None
            return

        try:
            logger.info("Initializing ChromaDB at %s", self.persist_directory)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            emb_fn = _embedding_fn()
            kwargs: Dict[str, Any] = {"name": self.collection_name}
            if emb_fn is not None:
                kwargs["embedding_function"] = emb_fn
            self._collection = self.client.get_or_create_collection(**kwargs)
        except Exception as exc:
            logger.error("Failed to initialize ChromaDB: %s", exc)
            self._collection = None

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        """Idempotent upsert keyed by id — re-running the seed script overwrites
        rows in place rather than duplicating them.
        """
        if not self._collection:
            logger.warning("ChromaDB is not initialized. Cannot add texts.")
            return

        if not ids:
            ids = [str(uuid.uuid4()) for _ in texts]

        try:
            self._collection.upsert(
                documents=texts,
                metadatas=metadatas or [{} for _ in texts],
                ids=ids,
            )
            logger.info("Upserted %d documents into ChromaDB.", len(texts))
        except Exception as exc:
            logger.error("Error upserting texts to ChromaDB: %s", exc)
            raise

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search documents by similarity (lower score = more similar)."""
        if not self._collection:
            logger.warning("ChromaDB is not initialized. Cannot search texts.")
            return []

        try:
            results = self._collection.query(query_texts=[query], n_results=top_k)
            formatted: List[Dict[str, Any]] = []
            if results["documents"] and len(results["documents"]) > 0:
                docs = results["documents"][0]
                metas = results["metadatas"][0] if results["metadatas"] else [{} for _ in docs]
                distances = results["distances"][0] if results["distances"] else [0.0 for _ in docs]
                for doc, meta, dist in zip(docs, metas, distances, strict=False):
                    formatted.append({"text": doc, "metadata": meta, "score": dist})
            return formatted
        except Exception as exc:
            logger.error("Error querying ChromaDB: %s", exc)
            return []

    def count(self) -> int:
        """Return the number of documents in the collection (0 if uninitialised)."""
        if not self._collection:
            return 0
        try:
            return int(self._collection.count())
        except Exception as exc:
            logger.error("Error counting ChromaDB collection: %s", exc)
            return 0
