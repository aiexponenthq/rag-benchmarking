from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.engine.rag_engine import RAGEngine, RetrievedChunk

router = APIRouter(prefix="/v1", tags=["query"])


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=20)
    rerank: bool = Field(default=False)


class QueryResponse(BaseModel):
    answer: str
    citations: list[RetrievedChunk]
    timings_ms: dict[str, float] | None = None
    tokens: dict[str, int] | None = None
    groundedness: float | None = None


def get_rag_engine() -> RAGEngine:
    return RAGEngine()


@router.post("/query", response_model=QueryResponse)
def post_query(req: QueryRequest, engine: RAGEngine = Depends(get_rag_engine)) -> QueryResponse:
    try:
        result = engine.query(req.query, req.top_k, req.rerank)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    # Token usage is provider-specific; placeholder zeros for now
    tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    return QueryResponse(
        answer=result.answer,
        citations=result.citations,
        timings_ms=result.timings,
        tokens=tokens,
        groundedness=result.groundedness,
    )
