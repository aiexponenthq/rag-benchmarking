from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import app.eval.ragas_runner as rr
from app.eval.reporting import write_report_files

router = APIRouter(prefix="/v1", tags=["evaluate"])


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
def post_evaluate(req: EvalRequest) -> dict[str, Any]:
    try:
        samples = [s.model_dump() for s in req.samples]
        result = rr.run_evaluation(samples, metrics=req.metrics)
        paths = write_report_files(
            result,
            out_json=Path(req.out_json) if req.out_json else None,
            out_md=Path(req.out_md) if req.out_md else None,
        )
        return {"result": result, "written": paths}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
