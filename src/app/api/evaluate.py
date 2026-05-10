from __future__ import annotations

import asyncio
import functools
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

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
async def post_evaluate(req: EvalRequest, _: str | None = Depends(get_api_key)) -> dict[str, Any]:
    """Evaluate RAG samples against the requested metrics.

    Routes through EvaluationRunner which handles:
    - Deterministic metrics (source_attribution_accuracy) — no LLM needed
    - RAGAS metrics (faithfulness, answer_relevancy, context_precision, context_recall) — needs Gemini/OpenAI
    - Retrieval metrics (precision_at_k, recall_at_k, mrr, ndcg_at_k) — no LLM needed

    RAGAS calls are offloaded to a thread-pool executor to avoid blocking
    the FastAPI event loop.
    """
    from harness.runner import EvaluationRunner
    from harness.schemas import EvalSample as HarnessEvalSample
    from harness.schemas import RunConfig

    loop = asyncio.get_running_loop()
    try:
        # Convert API samples to harness EvalSample objects
        harness_samples = [
            HarnessEvalSample(
                question=s.question,
                contexts=s.contexts,
                answer=s.answer,
                ground_truths=s.ground_truths,
            )
            for s in req.samples
        ]

        config = RunConfig(metrics=req.metrics or ["faithfulness", "answer_relevancy"])
        runner = EvaluationRunner(config)

        result = await loop.run_in_executor(None, runner.evaluate, harness_samples)

        # Convert BenchmarkReport to dict for response
        output: dict[str, Any] = {
            "metrics": result.metrics,
            "skipped_metrics": result.skipped_metrics,
            "skip_reasons": result.skip_reasons,
            "run_id": result.run_id,
            "n_samples": result.n_samples,
        }

        paths = write_report_files(
            {
                "metrics": result.metrics,
                "per_sample": {},
                "skipped_metrics": result.skipped_metrics,
            },
            out_json=Path(req.out_json) if req.out_json else None,
            out_md=Path(req.out_md) if req.out_md else None,
        )
        output["written"] = paths
        return output

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


class AgentEvalRequest(BaseModel):
    trace: AgentTrace
    metrics: list[str] = Field(default=["source_attribution_accuracy", "agent_faithfulness", "tool_call_accuracy"])


@router.post("/evaluate/agent")
async def post_evaluate_agent(
    request: AgentEvalRequest,
    _: str | None = Depends(get_api_key),
) -> dict[str, Any]:
    """Evaluate an agentic RAG trace using agentic-specific metrics."""
    import re

    from app.eval.agentic_llm_metrics import (
        compute_agent_faithfulness,
        compute_retrieval_necessity,
        compute_tool_call_accuracy,
    )
    from app.eval.agentic_metrics import source_attribution_accuracy

    scores: dict[str, float] = {}
    details: dict[str, Any] = {}

    for metric in request.metrics:
        if metric == "source_attribution_accuracy":
            cited = re.findall(r"\[source:\s*([^\]]+)\]", request.trace.final_answer)
            retrieved = [c.source_id for c in request.trace.retrieved_chunks]
            r = source_attribution_accuracy(cited, retrieved)
            scores[metric] = r["score"]
            details[metric] = r
        elif metric == "agent_faithfulness":
            loop = asyncio.get_running_loop()
            r = await loop.run_in_executor(None, compute_agent_faithfulness, request.trace)
            scores[metric] = r["score"]
            details[metric] = r
        elif metric == "tool_call_accuracy":
            loop = asyncio.get_running_loop()
            r = await loop.run_in_executor(None, compute_tool_call_accuracy, request.trace)
            scores[metric] = r["score"]
            details[metric] = r
        elif metric == "retrieval_necessity":
            contexts = [c.content for c in request.trace.retrieved_chunks]
            loop = asyncio.get_running_loop()
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

    # v1.0.1 audit RB-NEW3: renamed `scores` → `metrics` to match the
    # /v1/evaluate sibling endpoint shape (which has always emitted
    # `metrics`). Sibling-endpoint contract drift was the only remaining
    # API contract issue in the v1.0.0 audit. CHANGELOG documents this
    # as the one breaking change in v1.0.1; consumers reading
    # `r["scores"]` should switch to `r["metrics"]`.
    return {
        "metrics": scores,
        "details": details,
        "trace_id": request.trace.trace_id,
    }


@router.get("/runs")
async def list_runs(
    limit: int = 50,
    _: str | None = Depends(get_api_key),
) -> list[dict[str, Any]]:
    """List recent evaluation runs from the result store."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, functools.partial(_result_store.list_runs, limit))


@router.post("/runs/compare")
async def compare_runs(
    run_ids: list[str],
    _: str | None = Depends(get_api_key),
) -> dict[str, Any]:
    """Compare metrics across multiple named runs."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, functools.partial(_result_store.compare_runs, run_ids))
