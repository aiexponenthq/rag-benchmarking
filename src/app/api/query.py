from __future__ import annotations

import asyncio
import functools
from typing import Any

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
async def post_query(
    req: QueryRequest, engine: RAGEngine = Depends(get_rag_engine)
) -> QueryResponse:
    """Async query endpoint.

    ``RAGEngine.query()`` calls sentence-transformers (CPU-bound / sync) and
    blocking HTTP calls to OpenAI / Gemini.  We run it in a thread-pool
    executor so the FastAPI event loop is never blocked.

    Pattern for sync code in async FastAPI:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, functools.partial(sync_fn, *args))
    """
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            None,
            functools.partial(engine.query, req.query, req.top_k, req.rerank),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    # Token usage: LLMClient.generate() stores self._last_token_usage after
    # each call.  Guard against mock engines (test environments) where the
    # attribute may be a Mock object rather than a dict.
    _raw = getattr(getattr(engine, "llm", None), "_last_token_usage", None)
    last_usage: dict[str, int] = (
        _raw
        if isinstance(_raw, dict)
        else {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    )

    return QueryResponse(
        answer=result.answer,
        citations=result.citations,
        timings_ms=result.timings,
        tokens=last_usage,
        groundedness=result.groundedness,
    )
