from __future__ import annotations

from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.config.settings import get_settings


def get_qdrant_client() -> QdrantClient:
    settings = get_settings()
    if not (settings.qdrant_url and settings.qdrant_api_key):
        raise RuntimeError("Qdrant Cloud is not configured. Set QDRANT_URL and QDRANT_API_KEY.")
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=30)


def ensure_collection(
    client: QdrantClient, collection: str, vector_size: int, desired_vector_name: str | None = None
) -> tuple[str, str | None]:
    """Ensure collection exists and return (collection_name, vector_name_if_named).

    If the collection exists, attempt to detect if it uses a named-vector schema and return the 
    name. If it does not exist, create either a single-vector collection or a named-vector 
    collection if desired_vector_name is provided.
    """
    existing = [c.name for c in client.get_collections().collections]
    if collection in existing:
        # Detect vector schema
        name = _detect_named_vector_from_dump(client, collection)
        if name is None and desired_vector_name:
            # Prefer a sibling collection with desired named vector schema
            new_collection = f"{collection}__{desired_vector_name}"
            if new_collection in existing:
                return (new_collection, desired_vector_name)
            try:
                client.create_collection(
                    collection_name=new_collection,
                    vectors_config={
                        desired_vector_name: qmodels.VectorParams(
                            size=vector_size, distance=qmodels.Distance.COSINE
                        )
                    },
                )
            except Exception:
                # If it already exists or any race, just proceed to use it
                pass
            return (new_collection, desired_vector_name)
        return (collection, name)
    # Create collection
    if desired_vector_name:
        client.create_collection(
            collection_name=collection,
            vectors_config={
                desired_vector_name: qmodels.VectorParams(
                    size=vector_size, distance=qmodels.Distance.COSINE
                )
            },
        )
        return (collection, desired_vector_name)
    else:
        client.create_collection(
            collection_name=collection,
            vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE),
        )
        return (collection, None)


def _resolve_vector_schema(client: QdrantClient, collection: str) -> tuple[bool, str | None]:
    """Return (is_named_vectors, vector_name_if_named).

    Avoid direct isinstance checks on typing-based classes to prevent TypeErrors.
    """
    info = client.get_collection(collection)
    vectors_cfg = info.config.params.vectors  # type: ignore[attr-defined]

    # Case 1: single vector config typically has attribute 'size'
    if hasattr(vectors_cfg, "size"):
        return (False, None)

    # Case 2: named vectors often presented as mapping/dict-like under 'params'
    params = getattr(vectors_cfg, "params", None)
    if isinstance(params, dict) and len(params) > 0:
        name = next(iter(params.keys()))
        return (True, name)

    # Case 3: vectors_cfg itself might be a dict mapping names to params
    if isinstance(vectors_cfg, dict) and len(vectors_cfg) > 0:
        name = next(iter(vectors_cfg.keys()))
        return (True, name)

    return (False, None)


def _detect_named_vector_from_dump(client: QdrantClient, collection: str) -> str | None:
    """Fallback: inspect raw model dump to extract a named vector key if present."""
    info = client.get_collection(collection)
    data = info.model_dump(exclude_none=True)  # type: ignore[attr-defined]
    vectors = (
        data.get("config", {}).get("params", {}).get("vectors") if isinstance(data, dict) else None
    )
    if isinstance(vectors, dict):
        if "size" in vectors:
            return None
        if "params" in vectors and isinstance(vectors["params"], dict) and vectors["params"]:
            return next(iter(vectors["params"].keys()))
        if vectors:
            return next(iter(vectors.keys()))
    return None


def upsert_points(
    client: QdrantClient,
    collection: str,
    embeddings: np.ndarray,
    payloads: list[dict[str, Any]],
    vector_name: str | None = None,
) -> None:
    assert embeddings.shape[0] == len(payloads)
    points = []
    from uuid import uuid4

    for _idx, (vec, payload) in enumerate(zip(embeddings, payloads, strict=True)):
        if vector_name:
            points.append(
                qmodels.PointStruct(
                    id=str(uuid4()),
                    vector={vector_name: vec.tolist()},  # dict for named vector
                    payload=payload,
                ),
            )
        else:
            points.append(
                qmodels.PointStruct(id=str(uuid4()), vector=vec.tolist(), payload=payload),
            )
    client.upsert(collection_name=collection, points=points, wait=True)


def search(
    client: QdrantClient,
    collection: str,
    query_vector: np.ndarray,
    top_k: int = 5,
    filters: qmodels.Filter | None = None,
    vector_name: str | None = None,
) -> list[qmodels.ScoredPoint]:
    qv: Any = (vector_name, query_vector.tolist()) if vector_name else query_vector.tolist()
    return client.search(
        collection_name=collection,
        query_vector=qv,
        limit=top_k,
        query_filter=filters,
        with_payload=True,
    )
