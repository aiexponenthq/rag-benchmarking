from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class MetricGroup(StrEnum):
    CLASSIC = "classic"
    RETRIEVAL = "retrieval"
    AGENTIC_V1 = "agentic_v1"
    AGENTIC_V2 = "agentic_v2"
    FULL = "full"


class ToolCallType(StrEnum):
    RETRIEVE = "retrieve"
    WEB_SEARCH = "web_search"
    CODE_EXEC = "code_exec"
    CALCULATOR = "calculator"
    OTHER = "other"


class RetrievedChunk(BaseModel):
    source_id: str
    content: str
    score: float | None = None


class EvalSample(BaseModel):
    """Universal input for classic RAG evaluation. Works with any RAG system.

    The ``ground_truths`` field is a list (Ragas convention) so a sample can
    carry multiple acceptable references. The legacy singular ``ground_truth``
    keyword is accepted at construction time for backwards compatibility (it
    becomes the first element of ``ground_truths``).
    """

    sample_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str = Field(..., min_length=1)
    contexts: list[str] = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    ground_truths: list[str] = Field(default_factory=list)
    retrieved_doc_ids: list[str] = Field(default_factory=list)
    relevant_doc_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"populate_by_name": True, "extra": "allow"}

    def __init__(self, **data: Any) -> None:
        # Backwards-compat: accept legacy singular `ground_truth=` kwarg and
        # promote to the plural list. Pre-existing user code keeps working.
        if "ground_truth" in data and "ground_truths" not in data:
            gt = data.pop("ground_truth")
            data["ground_truths"] = [gt] if gt else []
        super().__init__(**data)

    @property
    def ground_truth(self) -> str | None:
        """Legacy singular accessor. Returns the first ground_truths entry."""
        return self.ground_truths[0] if self.ground_truths else None

    @field_validator("question", "answer")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be empty or whitespace")
        return v


class ToolCall(BaseModel):
    tool_name: str
    tool_input: dict[str, Any]
    tool_output: str
    step_index: int
    latency_ms: float | None = None


class ReasoningStep(BaseModel):
    step_index: int
    thought: str
    action: str
    observation: str
    cited_sources: list[str] = Field(default_factory=list)


class AgentTrace(BaseModel):
    """Input for agentic RAG evaluation. Captures the full reasoning trace.

    ``ground_truths`` is a list (Ragas convention). Legacy singular
    ``ground_truth`` is accepted at construction for backwards compatibility.
    """

    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str = Field(..., min_length=1)
    final_answer: str = Field(..., min_length=1)
    ground_truths: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    reasoning_steps: list[ReasoningStep] = Field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    conversation_history: list[dict[str, str]] = Field(default_factory=list)
    total_tokens: int | None = None
    total_latency_ms: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"populate_by_name": True, "extra": "allow"}

    def __init__(self, **data: Any) -> None:
        # Backwards-compat shim, parity with EvalSample. Without this the
        # legacy singular kwarg is silently dropped under Pydantic v2 — the
        # exact contract failure the audit RB-H5 was meant to close.
        if "ground_truth" in data and "ground_truths" not in data:
            gt = data.pop("ground_truth")
            data["ground_truths"] = [gt] if gt else []
        super().__init__(**data)

    @property
    def ground_truth(self) -> str | None:
        """Legacy singular accessor. Returns the first ground_truths entry."""
        return self.ground_truths[0] if self.ground_truths else None


class EvalResult(BaseModel):
    sample_id: str
    metrics: dict[str, float]
    details: dict[str, Any] = Field(default_factory=dict)
    errors: dict[str, str] = Field(default_factory=dict)


class BenchmarkReport(BaseModel):
    run_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    n_samples: int
    metrics: dict[str, float]
    per_sample: list[EvalResult] = Field(default_factory=list)
    skipped_metrics: list[str] = Field(default_factory=list)
    skip_reasons: dict[str, str] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)


class RunConfig(BaseModel):
    metrics: list[str] = Field(default_factory=list)
    metric_group: MetricGroup | None = None
    judge_model: str = "gemini-2.5-flash"
    judge_temperature: float = 0.0
    k: int = 5
    faithfulness_threshold: float = 0.7
    run_id: str | None = None


METRIC_GROUPS: dict[str, list[str]] = {
    # `classic` = the four canonical Ragas metrics. context_precision and
    # context_recall require ground_truths; they short-circuit to skipped
    # when unavailable (see ragas_runner.py).
    "classic": ["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
    "retrieval": ["precision_at_k", "recall_at_k", "mrr", "ndcg_at_k"],
    "agentic_v1": [
        "source_attribution_accuracy",
        "retrieval_necessity",
        "agent_faithfulness",
        "tool_call_accuracy",
    ],
    "agentic_v2": [
        "multihop_faithfulness",
        "agent_trajectory_efficiency",
        "reasoning_hallucination",
        "context_coherence_across_turns",
    ],
    # `full` = every metric group concatenated. Keep in sync when groups change
    # — audit RB-M9 caught a prior version that omitted context_precision /
    # context_recall / agentic_v2.
    "full": [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "precision_at_k",
        "recall_at_k",
        "mrr",
        "ndcg_at_k",
        "source_attribution_accuracy",
        "retrieval_necessity",
        "agent_faithfulness",
        "tool_call_accuracy",
        "multihop_faithfulness",
        "agent_trajectory_efficiency",
        "reasoning_hallucination",
        "context_coherence_across_turns",
    ],
}
