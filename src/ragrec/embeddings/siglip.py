"""SigLIP embedding model implementation."""

import io
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from transformers import AutoModel, AutoImageProcessor, AutoTokenizer

from ragrec.embeddings.base import Embedder


class SigLIPEmbedder(Embedder):
    """SigLIP-based image and text embedder."""

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        device: str | None = None,
    ) -> None:
        """Initialize SigLIP embedder.

        Args:
            model_name: HuggingFace model identifier
            device: Device to use ('mps', 'cuda', 'cpu', or None for auto-detect)
        """
        self._model_name = model_name

        # Auto-detect device if not specified
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"  # Apple Metal
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self.device = torch.device(device)

        # Load model and processors
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        # Tokenizer loading has issues in transformers 5.0, skip for now
        # We'll load it lazily when needed for text encoding
        self._tokenizer = None

        # Move model to device
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

        # Get embedding dimension from model config
        # SigLIP base models have 768-dimensional embeddings
        self._embedding_dim = 768

    def encode_image(self, image_bytes: bytes) -> NDArray[np.float32]:
        """Encode a single image to embedding vector."""
        # Load image from bytes
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Process image
        inputs = self.image_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate embedding
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)

        # Extract tensor from output object (SigLIP returns BaseModelOutputWithPooling)
        # The actual embeddings are in the pooler_output field
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            embedding_tensor = outputs.pooler_output
        elif hasattr(outputs, 'last_hidden_state'):
            # Fallback: use mean pooling of last hidden state
            embedding_tensor = outputs.last_hidden_state.mean(dim=1)
        else:
            # Assume outputs is already a tensor
            embedding_tensor = outputs

        # Convert to numpy and normalize
        embedding = embedding_tensor.cpu().numpy()[0]
        embedding = embedding / np.linalg.norm(embedding)

        return embedding.astype(np.float32)

    def encode_text(self, text: str) -> NDArray[np.float32]:
        """Encode text to embedding vector."""
        # TODO: Tokenizer loading has issues in transformers 5.0
        # This will be fixed in a future version
        raise NotImplementedError(
            "Text encoding temporarily disabled due to transformers library issue"
        )

    def batch_encode_images(
        self, image_bytes_list: list[bytes], batch_size: int = 32
    ) -> NDArray[np.float32]:
        """Encode multiple images in batches."""
        all_embeddings = []

        for i in range(0, len(image_bytes_list), batch_size):
            batch = image_bytes_list[i : i + batch_size]

            # Load images
            images = [
                Image.open(io.BytesIO(img_bytes)).convert("RGB") for img_bytes in batch
            ]

            # Process batch
            inputs = self.image_processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate embeddings
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)

            # Extract tensor from output object
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                embedding_tensor = outputs.pooler_output
            elif hasattr(outputs, 'last_hidden_state'):
                # Fallback: use mean pooling of last hidden state
                embedding_tensor = outputs.last_hidden_state.mean(dim=1)
            else:
                # Assume outputs is already a tensor
                embedding_tensor = outputs

            # Convert to numpy and normalize
            embeddings = embedding_tensor.cpu().numpy()
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms

            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings).astype(np.float32)

    def batch_encode_texts(
        self, texts: list[str], batch_size: int = 32
    ) -> NDArray[np.float32]:
        """Encode multiple texts in batches."""
        # TODO: Tokenizer loading has issues in transformers 5.0
        # This will be fixed in a future version
        raise NotImplementedError(
            "Text encoding temporarily disabled due to transformers library issue"
        )

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimensionality."""
        return self._embedding_dim

    @property
    def model_name(self) -> str:
        """Return model identifier."""
        return self._model_name
