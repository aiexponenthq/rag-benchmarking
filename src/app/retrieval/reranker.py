from __future__ import annotations

from typing import Any

try:
    from FlagEmbedding import FlagReranker
except Exception:  # pragma: no cover - optional dependency during import
    FlagReranker = None  # type: ignore[assignment]


class CrossEncoderReranker:
    """Cross-encoder reranker using BAAI/bge-reranker-v2-m3 by default."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = "cpu") -> None:
        if FlagReranker is None:
            raise RuntimeError("FlagEmbedding is not installed")
        # use_fp16=True is fine on CPU via bfloat16 emulation; can set False if issues
        self.reranker = FlagReranker(model_name, use_fp16=True, device=device)

    def rerank(self, query: str, chunks: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        pairs: list[tuple[str, str]] = [(query, c.get("text", "")) for c in chunks]
        scores = self.reranker.compute_score(pairs, normalize=True)
        scored = [dict(c, rerank_score=float(s)) for c, s in zip(chunks, scores, strict=True)]
        scored.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        return scored[:top_k]
