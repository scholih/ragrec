"""Abstract base class for embedding models."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray


class Embedder(ABC):
    """Abstract interface for embedding models."""

    @abstractmethod
    def encode_image(self, image_bytes: bytes) -> NDArray[np.float32]:
        """Encode an image to a vector embedding.

        Args:
            image_bytes: Raw image bytes (JPEG, PNG, etc.)

        Returns:
            Embedding vector as numpy array
        """
        pass

    @abstractmethod
    def encode_text(self, text: str) -> NDArray[np.float32]:
        """Encode text to a vector embedding.

        Args:
            text: Input text string

        Returns:
            Embedding vector as numpy array
        """
        pass

    @abstractmethod
    def batch_encode_images(
        self, image_bytes_list: list[bytes]
    ) -> NDArray[np.float32]:
        """Encode multiple images to vector embeddings (batched for efficiency).

        Args:
            image_bytes_list: List of raw image bytes

        Returns:
            2D numpy array where each row is an embedding
        """
        pass

    @abstractmethod
    def batch_encode_texts(self, texts: list[str]) -> NDArray[np.float32]:
        """Encode multiple texts to vector embeddings (batched for efficiency).

        Args:
            texts: List of text strings

        Returns:
            2D numpy array where each row is an embedding
        """
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the dimensionality of the embeddings."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier/version."""
        pass
