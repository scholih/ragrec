"""Text-based product encoder using SigLIP text encoder.

For non-fashion products where visual similarity is less important.
"""

from typing import Any

import torch
import numpy as np
from transformers import AutoProcessor, AutoModel


class TextProductEncoder:
    """Encode product text (titles, descriptions) using SigLIP text encoder.

    Uses the same SigLIP model as visual encoder but operates on text.
    This enables:
    - Similarity by product attributes (brand, specs, category)
    - Compatibility with visual embeddings (same dimension)
    - Unified embedding space
    """

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        device: str | None = None,
    ) -> None:
        """Initialize text encoder.

        Args:
            model_name: HuggingFace model identifier
            device: Device to run model on (cuda/mps/cpu)
        """
        self.model_name = model_name

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device

        # Load model and processor
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

    def encode_text(self, text: str | list[str]) -> np.ndarray:
        """Encode text into embeddings.

        Args:
            text: Single text string or list of texts

        Returns:
            Embedding array (768-dim for SigLIP base)
        """
        # Ensure text is a list
        if isinstance(text, str):
            text = [text]
            return_single = True
        else:
            return_single = False

        # Process text
        inputs = self.processor(
            text=text,
            padding=True,
            truncation=True,
            max_length=77,  # SigLIP text max length
            return_tensors="pt",
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate embeddings
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)

        # Convert to numpy
        embeddings = outputs.cpu().numpy()

        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        if return_single:
            return embeddings[0]
        else:
            return embeddings

    def encode_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode batch of texts with batching.

        Args:
            texts: List of text strings
            batch_size: Number of texts per batch

        Returns:
            Embedding array (N x 768)
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_embeddings = self.encode_text(batch_texts)
            all_embeddings.append(batch_embeddings)

        return np.vstack(all_embeddings)

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        return 768  # SigLIP base dimension


class ProductTextFormatter:
    """Format product metadata into text for encoding.

    Creates rich text descriptions from product attributes.
    """

    @staticmethod
    def format_product(
        title: str,
        category: str | None = None,
        brand: str | None = None,
        features: list[str] | None = None,
        description: str | None = None,
    ) -> str:
        """Format product metadata into text.

        Args:
            title: Product title
            category: Product category
            brand: Brand name
            features: List of product features
            description: Product description

        Returns:
            Formatted text string
        """
        parts = []

        # Title is always first
        parts.append(title)

        # Add brand and category
        if brand:
            parts.append(f"Brand: {brand}")

        if category:
            parts.append(f"Category: {category}")

        # Add features
        if features:
            features_text = ". ".join(features[:5])  # Top 5 features
            parts.append(f"Features: {features_text}")

        # Add description snippet
        if description:
            # Take first 200 chars of description
            desc_snippet = description[:200].strip()
            if len(description) > 200:
                desc_snippet += "..."
            parts.append(desc_snippet)

        return ". ".join(parts)

    @staticmethod
    def format_amazon_product(product_data: dict[str, Any]) -> str:
        """Format Amazon product data.

        Args:
            product_data: Amazon product metadata dict

        Returns:
            Formatted text string
        """
        return ProductTextFormatter.format_product(
            title=product_data.get("title", ""),
            category=product_data.get("main_category", ""),
            brand=product_data.get("brand", ""),
            features=product_data.get("features", []),
            description=product_data.get("description", ""),
        )
