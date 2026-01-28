"""Customer behavior sequence encoder.

Generates behavioral embeddings from customer purchase history.
"""

from datetime import datetime, timedelta
from typing import Any

import numpy as np


class BehaviorEncoder:
    """Encode customer purchase sequences into behavior embeddings.

    Uses weighted average of product embeddings with recency weighting.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        output_dim: int = 256,
        recency_halflife_days: float = 30.0,
    ) -> None:
        """Initialize behavior encoder.

        Args:
            embedding_dim: Input product embedding dimension (default: 768 for SigLIP)
            output_dim: Output behavior embedding dimension (default: 256)
            recency_halflife_days: Days for recency weight to decay to 0.5
        """
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.recency_halflife_days = recency_halflife_days

        # PCA-like projection matrix for dimensionality reduction
        # In practice, we'd learn this from data, but for MVP we use random projection
        # with normalization (similar to Johnson-Lindenstrauss lemma)
        rng = np.random.default_rng(seed=42)
        self.projection = rng.standard_normal((embedding_dim, output_dim)).astype(np.float32)
        # Normalize columns to unit norm
        self.projection /= np.linalg.norm(self.projection, axis=0, keepdims=True)

    def encode_sequence(
        self,
        product_embeddings: list[np.ndarray],
        timestamps: list[datetime],
        reference_time: datetime | None = None,
    ) -> np.ndarray:
        """Encode a purchase sequence into a behavior embedding.

        Args:
            product_embeddings: List of product embeddings (each 768-dim)
            timestamps: List of purchase timestamps (aligned with embeddings)
            reference_time: Reference time for recency calculation (default: now)

        Returns:
            Behavior embedding (output_dim-dimensional)
        """
        if not product_embeddings:
            # Return zero embedding for customers with no purchases
            return np.zeros(self.output_dim, dtype=np.float32)

        if len(product_embeddings) != len(timestamps):
            raise ValueError("product_embeddings and timestamps must have same length")

        # Calculate recency weights
        if reference_time is None:
            reference_time = datetime.now()

        weights = self._calculate_recency_weights(timestamps, reference_time)

        # Weighted average of product embeddings
        embeddings_array = np.array(product_embeddings, dtype=np.float32)
        weights_array = np.array(weights, dtype=np.float32).reshape(-1, 1)

        weighted_avg = np.sum(embeddings_array * weights_array, axis=0) / np.sum(weights_array)

        # Project to lower dimension
        behavior_embedding = weighted_avg @ self.projection

        # L2 normalize
        norm = np.linalg.norm(behavior_embedding)
        if norm > 0:
            behavior_embedding = behavior_embedding / norm

        return behavior_embedding

    def _calculate_recency_weights(
        self,
        timestamps: list[datetime],
        reference_time: datetime,
    ) -> list[float]:
        """Calculate recency weights using exponential decay.

        More recent purchases are weighted higher.

        Args:
            timestamps: Purchase timestamps
            reference_time: Reference time for recency calculation

        Returns:
            List of recency weights (same length as timestamps)
        """
        # Exponential decay: weight = 0.5^(days_ago / halflife)
        weights = []
        for timestamp in timestamps:
            days_ago = (reference_time - timestamp).total_seconds() / 86400  # Convert to days

            # Exponential decay
            weight = 0.5 ** (days_ago / self.recency_halflife_days)
            weights.append(weight)

        return weights

    def encode_batch(
        self,
        sequences: list[tuple[list[np.ndarray], list[datetime]]],
        reference_time: datetime | None = None,
    ) -> np.ndarray:
        """Encode a batch of purchase sequences.

        Args:
            sequences: List of (product_embeddings, timestamps) tuples
            reference_time: Reference time for recency calculation (default: now)

        Returns:
            Array of behavior embeddings (batch_size x output_dim)
        """
        if reference_time is None:
            reference_time = datetime.now()

        batch_embeddings = []
        for product_embeddings, timestamps in sequences:
            embedding = self.encode_sequence(
                product_embeddings=product_embeddings,
                timestamps=timestamps,
                reference_time=reference_time,
            )
            batch_embeddings.append(embedding)

        return np.array(batch_embeddings, dtype=np.float32)


class CustomerBehaviorEncoder:
    """Customer behavior encoder for async interface."""

    def __init__(
        self,
        embedding_dim: int = 768,
        output_dim: int = 256,
        recency_halflife_days: float = 30.0,
    ) -> None:
        """Initialize customer behavior encoder.

        Args:
            embedding_dim: Input product embedding dimension
            output_dim: Output behavior embedding dimension
            recency_halflife_days: Days for recency weight to decay to 0.5
        """
        self.encoder = BehaviorEncoder(
            embedding_dim=embedding_dim,
            output_dim=output_dim,
            recency_halflife_days=recency_halflife_days,
        )

    async def embed(self, inputs: list[Any]) -> list[np.ndarray]:
        """Encode customer purchase sequences.

        Args:
            inputs: List of (product_embeddings, timestamps) tuples

        Returns:
            List of behavior embeddings
        """
        # Use current time as reference for all in batch
        reference_time = datetime.now()

        embeddings = []
        for product_embeddings, timestamps in inputs:
            embedding = self.encoder.encode_sequence(
                product_embeddings=product_embeddings,
                timestamps=timestamps,
                reference_time=reference_time,
            )
            embeddings.append(embedding)

        return embeddings

    @property
    def embedding_dim(self) -> int:
        """Get output embedding dimension."""
        return self.encoder.output_dim
