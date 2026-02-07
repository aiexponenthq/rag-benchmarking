from __future__ import annotations

from typing import Any

from app.config.settings import get_settings
from app.retrieval.embeddings import EmbeddingsClient
from app.retrieval.qdrant_store import (
    get_qdrant_client,
)
from app.retrieval.qdrant_store import (
    search as qdrant_search,
)


def _resolve_collection_and_vector_name() -> tuple[str, str | None]:
    """Heuristic to choose the right collection and vector name for querying.

    - Prefer a sibling collection with suffix `__content` (created by ingestion) if present.
    - Otherwise use the base collection.
    - Vector name is `content` for the sibling; otherwise None and Qdrant default is used.
    """
    settings = get_settings()
    base = settings.qdrant_collection
    # Prefer sibling if it exists
    preferred = f"{base}__content"
    client = get_qdrant_client()
    collections = [c.name for c in client.get_collections().collections]
    if preferred in collections:
        return preferred, "content"
    return base, None


def retrieve_top_chunks(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Embed the query and fetch top-k chunks from Qdrant Cloud.

    Returns a list of payload dicts with at least keys: text, source_id, chunk_index, score.
    """
    if not query or not query.strip():
        return []
    embedder = EmbeddingsClient()
    qvec = embedder.embed([query])[0]
    client = get_qdrant_client()
    collection, vector_name = _resolve_collection_and_vector_name()
    try:
        results = qdrant_search(client, collection, qvec, top_k=top_k, vector_name=vector_name)
    except Exception as e:
        from app.exceptions import VectorDBError

        raise VectorDBError(f"Qdrant search failed: {str(e)}") from e
    payloads: list[dict[str, Any]] = []
    for r in results:
        payload = dict(r.payload or {})
        payload["score"] = r.score
        payloads.append(payload)
    return payloads
