"""Abstract base class for vector stores."""

from abc import ABC, abstractmethod
from typing import Any

import polars as pl


class VectorStore(ABC):
    """Abstract interface for vector storage and similarity search."""

    @abstractmethod
    async def upsert(
        self,
        ids: list[int],
        embeddings: list[list[float]],
        metadata: list[dict[str, Any]] | None = None,
    ) -> int:
        """Insert or update vectors with metadata.

        Args:
            ids: List of unique identifiers
            embeddings: List of embedding vectors
            metadata: Optional list of metadata dicts

        Returns:
            Number of records upserted
        """
        pass

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        """Search for similar vectors.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filters: Optional metadata filters

        Returns:
            Polars DataFrame with results and distances
        """
        pass

    @abstractmethod
    async def hybrid_search(
        self,
        query_embedding: list[float],
        query_text: str,
        top_k: int = 10,
    ) -> pl.DataFrame:
        """Hybrid search combining vector similarity and text matching.

        Args:
            query_embedding: Query vector
            query_text: Text query for filtering
            top_k: Number of results to return

        Returns:
            Polars DataFrame with results and combined scores
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the vector store is healthy."""
        pass
