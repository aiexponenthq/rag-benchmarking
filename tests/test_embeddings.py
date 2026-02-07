from __future__ import annotations

import numpy as np

from app.retrieval.embeddings import EmbeddingsClient


def test_embeddings_shape_and_dtype() -> None:
    client = EmbeddingsClient()
    vecs = client.embed(["hello world", "goodbye world"])
    assert isinstance(vecs, np.ndarray)
    assert vecs.shape[0] == 2
    assert vecs.dtype == np.float32
    assert not np.isnan(vecs).any()
