from __future__ import annotations

import asyncio
import functools
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

import app.eval.ragas_runner as rr
from app.api.security import get_api_key
from app.eval.reporting import write_report_files
from app.eval.result_store import ResultStore
from harness.schemas import AgentTrace

router = APIRouter(prefix="/v1", tags=["evaluate"])

_result_store = ResultStore()


class EvalSample(BaseModel):
    question: str
    contexts: list[str] = Field(default_factory=list)
    answer: str
    ground_truths: list[str] = Field(default_factory=list)


class EvalRequest(BaseModel):
    samples: list[EvalSample]
    metrics: list[str] | None = None
    out_json: str | None = None
    out_md: str | None = None


@router.post("/evaluate")
async def post_evaluate(req: EvalRequest) -> dict[str, Any]:
    """Async evaluate endpoint.

    RAGAS evaluation is I/O-heavy (LLM API calls) but uses its own internal
    asyncio via ``allow_nest_asyncio=True``.  We offload the entire call to a
    thread-pool executor to keep the FastAPI event loop free and to avoid
    nested-event-loop conflicts on Python 3.11.
    """
    loop = asyncio.get_event_loop()
    try:
        samples = [s.model_dump() for s in req.samples]
        result: dict[str, Any] = await loop.run_in_executor(
            None,
            functools.partial(rr.run_evaluation, samples, metrics=req.metrics),
        )
        paths = write_report_files(
            result,
            out_json=Path(req.out_json) if req.out_json else None,
            out_md=Path(req.out_md) if req.out_md else None,
        )
        return {"result": result, "written": paths}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


class AgentEvalRequest(BaseModel):
    trace: AgentTrace
    metrics: list[str] = Field(
        default=["source_attribution_accuracy", "agent_faithfulness", "tool_call_accuracy"]
    )


@router.post("/evaluate/agent")
async def post_evaluate_agent(
    request: AgentEvalRequest,
    _: str | None = Depends(get_api_key),
) -> dict[str, Any]:
    """Evaluate an agentic RAG trace using agentic-specific metrics."""
    import re

    from app.eval.agentic_metrics import source_attribution_accuracy
    from app.eval.agentic_llm_metrics import (
        compute_agent_faithfulness,
        compute_retrieval_necessity,
        compute_tool_call_accuracy,
    )

    scores: dict[str, float] = {}
    details: dict[str, Any] = {}

    for metric in request.metrics:
        if metric == "source_attribution_accuracy":
            cited = re.findall(r'\[source:\s*([^\]]+)\]', request.trace.final_answer)
            retrieved = [c.source_id for c in request.trace.retrieved_chunks]
            r = source_attribution_accuracy(cited, retrieved)
            scores[metric] = r["score"]
            details[metric] = r
        elif metric == "agent_faithfulness":
            loop = asyncio.get_event_loop()
            r = await loop.run_in_executor(None, compute_agent_faithfulness, request.trace)
            scores[metric] = r["score"]
            details[metric] = r
        elif metric == "tool_call_accuracy":
            loop = asyncio.get_event_loop()
            r = await loop.run_in_executor(None, compute_tool_call_accuracy, request.trace)
            scores[metric] = r["score"]
            details[metric] = r
        elif metric == "retrieval_necessity":
            contexts = [c.content for c in request.trace.retrieved_chunks]
            loop = asyncio.get_event_loop()
            r = await loop.run_in_executor(
                None,
                functools.partial(
                    compute_retrieval_necessity,
                    request.trace.question,
                    request.trace.final_answer,
                    contexts,
                ),
            )
            scores[metric] = r["score"]
            details[metric] = r

    return {
        "scores": scores,
        "details": details,
        "trace_id": request.trace.trace_id,
    }


@router.get("/runs")
async def list_runs(
    limit: int = 50,
    _: str | None = Depends(get_api_key),
) -> list[dict[str, Any]]:
    """List recent evaluation runs from the result store."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, functools.partial(_result_store.list_runs, limit))
