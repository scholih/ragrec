"""Embedding strategy configuration by product category.

Determines whether to use visual or text embeddings based on category.
"""

from enum import Enum
from typing import Literal


class EmbeddingStrategy(str, Enum):
    """Embedding strategy for products."""

    VISUAL = "visual"  # SigLIP image embeddings
    TEXT = "text"  # SigLIP text embeddings
    HYBRID = "hybrid"  # Both visual and text (future)


# Category-to-strategy mapping
CATEGORY_EMBEDDING_STRATEGY: dict[str, EmbeddingStrategy] = {
    # Fashion categories - visual similarity matters
    "Clothing_Shoes_and_Jewelry": EmbeddingStrategy.VISUAL,
    "Fashion": EmbeddingStrategy.VISUAL,

    # Non-fashion categories - text/attributes matter more
    "Electronics": EmbeddingStrategy.TEXT,
    "All_Beauty": EmbeddingStrategy.TEXT,
    "Books": EmbeddingStrategy.TEXT,
    "Sports_and_Outdoors": EmbeddingStrategy.TEXT,
    "Home_and_Kitchen": EmbeddingStrategy.TEXT,
    "Toys_and_Games": EmbeddingStrategy.TEXT,
    "Tools_and_Home_Improvement": EmbeddingStrategy.TEXT,

    # Default fallback
    "default": EmbeddingStrategy.TEXT,
}


def get_embedding_strategy(category: str) -> EmbeddingStrategy:
    """Get embedding strategy for a category.

    Args:
        category: Product category name

    Returns:
        Embedding strategy to use
    """
    # Check exact match
    if category in CATEGORY_EMBEDDING_STRATEGY:
        return CATEGORY_EMBEDDING_STRATEGY[category]

    # Check partial match (case-insensitive)
    category_lower = category.lower()
    for cat_key, strategy in CATEGORY_EMBEDDING_STRATEGY.items():
        if cat_key.lower() in category_lower or category_lower in cat_key.lower():
            return strategy

    # Default to text
    return CATEGORY_EMBEDDING_STRATEGY["default"]


def needs_images(category: str) -> bool:
    """Check if category needs product images.

    Args:
        category: Product category name

    Returns:
        True if category uses visual embeddings
    """
    strategy = get_embedding_strategy(category)
    return strategy in (EmbeddingStrategy.VISUAL, EmbeddingStrategy.HYBRID)


def uses_text_embeddings(category: str) -> bool:
    """Check if category uses text embeddings.

    Args:
        category: Product category name

    Returns:
        True if category uses text embeddings
    """
    strategy = get_embedding_strategy(category)
    return strategy in (EmbeddingStrategy.TEXT, EmbeddingStrategy.HYBRID)
