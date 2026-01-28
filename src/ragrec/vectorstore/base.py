"""Abstract base class for vector stores."""

from abc import ABC, abstractmethod
from typing import Any


class VectorStore(ABC):
    """Abstract interface for vector storage and similarity search."""

    @abstractmethod
    async def insert(self, vectors: list[list[float]], metadata: list[dict[str, Any]]) -> None:
        """Insert vectors with metadata."""
        pass

    @abstractmethod
    async def search(
        self, query_vector: list[float], limit: int = 10
    ) -> list[dict[str, Any]]:
        """Search for similar vectors."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the vector store is healthy."""
        pass
