from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


def source_attribution_accuracy(
    cited_ids: list[str],
    retrieved_ids: list[str],
) -> dict:
    """
    Deterministic metric — no LLM required.

    Checks whether source IDs cited in the answer were actually retrieved.
    Hallucinated sources are IDs the agent cited but never retrieved.

    Returns:
        score: float — fraction of cited IDs that were actually retrieved (attribution precision)
        hallucinated_sources: list[str] — cited IDs not in retrieved set
        valid_sources: list[str] — cited IDs that were retrieved
        coverage: float — fraction of retrieved IDs that were cited
    """
    if not cited_ids:
        return {
            "score": 1.0,
            "hallucinated_sources": [],
            "valid_sources": [],
            "coverage": 1.0,
        }

    retrieved_set = set(retrieved_ids)
    hallucinated = [cid for cid in cited_ids if cid not in retrieved_set]
    valid = [cid for cid in cited_ids if cid in retrieved_set]

    attribution_precision = len(valid) / len(cited_ids)
    coverage = len(set(cited_ids) & retrieved_set) / len(retrieved_set) if retrieved_set else 0.0

    return {
        "score": attribution_precision,
        "hallucinated_sources": hallucinated,
        "valid_sources": valid,
        "coverage": coverage,
    }
