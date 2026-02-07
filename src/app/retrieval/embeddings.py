from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingsClient:
    """CPU-friendly embeddings client using sentence-transformers.

    Defaults to BAAI/bge-base-en-v1.5.
    """

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", device: str = "cpu") -> None:
        self.model = SentenceTransformer(model_name, device=device)

    def embed(self, texts: list[str], *, normalize: bool = True) -> np.ndarray:
        vectors = self.model.encode(
            texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=normalize
        )
        return vectors.astype(np.float32)
