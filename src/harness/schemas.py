from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class MetricGroup(str, Enum):
    CLASSIC = "classic"
    RETRIEVAL = "retrieval"
    AGENTIC_V1 = "agentic_v1"
    AGENTIC_V2 = "agentic_v2"
    FULL = "full"


class ToolCallType(str, Enum):
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
    """Universal input for classic RAG evaluation. Works with any RAG system."""
    sample_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str = Field(..., min_length=1)
    contexts: list[str] = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    ground_truth: str | None = None
    retrieved_doc_ids: list[str] = Field(default_factory=list)
    relevant_doc_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

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
    """Input for agentic RAG evaluation. Captures the full reasoning trace."""
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str = Field(..., min_length=1)
    final_answer: str = Field(..., min_length=1)
    ground_truth: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    reasoning_steps: list[ReasoningStep] = Field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    conversation_history: list[dict[str, str]] = Field(default_factory=list)
    total_tokens: int | None = None
    total_latency_ms: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


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
    judge_model: str = "gemini-1.5-flash"
    judge_temperature: float = 0.0
    k: int = 5
    faithfulness_threshold: float = 0.7
    run_id: str | None = None


METRIC_GROUPS: dict[str, list[str]] = {
    "classic": ["faithfulness", "answer_relevancy"],
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
    "full": [
        "faithfulness", "answer_relevancy",
        "precision_at_k", "recall_at_k", "mrr", "ndcg_at_k",
        "source_attribution_accuracy", "retrieval_necessity",
        "agent_faithfulness", "tool_call_accuracy",
    ],
}
