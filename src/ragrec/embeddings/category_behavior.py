"""Category-based customer behavior encoder.

Generates behavioral embeddings from purchase category distributions.
"""

from datetime import datetime
import numpy as np


class CategoryBehaviorEncoder:
    """Encode customer purchase patterns using category histograms.

    Creates embeddings from purchase counts across product categories
    with optional recency weighting. More interpretable and preserves
    behavioral diversity better than visual embedding averaging.
    """

    def __init__(
        self,
        categories: dict[str, list[str]],
        recency_halflife_days: float = 30.0,
    ) -> None:
        """Initialize category behavior encoder.

        Args:
            categories: Dict mapping category types to category IDs
                       e.g. {'section': [...], 'garment_group': [...], 'product_type': [...]}
            recency_halflife_days: Days for recency weight to decay to 0.5
        """
        self.categories = categories
        self.recency_halflife_days = recency_halflife_days

        # Build category index mapping
        self.category_to_idx = {}
        idx = 0
        for cat_type, cat_list in categories.items():
            for cat_id in sorted(cat_list):
                self.category_to_idx[cat_id] = idx
                idx += 1

        self.output_dim = len(self.category_to_idx)

    def encode_purchases(
        self,
        category_ids: list[str],
        timestamps: list[datetime],
        reference_time: datetime | None = None,
    ) -> np.ndarray:
        """Encode purchase history into category histogram embedding.

        Args:
            category_ids: List of category IDs for each purchase
            timestamps: List of purchase timestamps (aligned with category_ids)
            reference_time: Reference time for recency calculation (default: now)

        Returns:
            Category histogram embedding (output_dim-dimensional)
        """
        if not category_ids:
            return np.zeros(self.output_dim, dtype=np.float32)

        if len(category_ids) != len(timestamps):
            raise ValueError("category_ids and timestamps must have same length")

        # Calculate recency weights
        if reference_time is None:
            reference_time = datetime.now()

        weights = self._calculate_recency_weights(timestamps, reference_time)

        # Build weighted histogram
        histogram = np.zeros(self.output_dim, dtype=np.float32)

        for cat_id, weight in zip(category_ids, weights):
            if cat_id in self.category_to_idx:
                idx = self.category_to_idx[cat_id]
                histogram[idx] += weight

        # L1 normalization (sum to 1) to make it a probability distribution
        total = histogram.sum()
        if total > 0:
            histogram = histogram / total

        return histogram

    def _calculate_recency_weights(
        self,
        timestamps: list[datetime],
        reference_time: datetime,
    ) -> list[float]:
        """Calculate recency weights using exponential decay.

        Args:
            timestamps: Purchase timestamps
            reference_time: Reference time for recency calculation

        Returns:
            List of recency weights (same length as timestamps)
        """
        weights = []
        for timestamp in timestamps:
            days_ago = (reference_time - timestamp).total_seconds() / 86400
            weight = 0.5 ** (days_ago / self.recency_halflife_days)
            weights.append(weight)

        return weights
