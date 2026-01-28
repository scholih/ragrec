"""Visual and sequence embedding models."""

from ragrec.embeddings.base import Embedder
from ragrec.embeddings.generator import generate_product_embeddings
from ragrec.embeddings.siglip import SigLIPEmbedder

__all__ = ["Embedder", "SigLIPEmbedder", "generate_product_embeddings"]
