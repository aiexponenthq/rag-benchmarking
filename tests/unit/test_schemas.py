import uuid

import pytest
from harness.schemas import (
    AgentTrace,
    BenchmarkReport,
    EvalSample,
    RunConfig,
    ToolCall,
)
from pydantic import ValidationError


def test_eval_sample_minimal():
    s = EvalSample(
        question="What is RAG?",
        contexts=["RAG stands for Retrieval-Augmented Generation."],
        answer="RAG is a technique combining retrieval with generation.",
    )
    assert s.sample_id is not None
    uuid.UUID(s.sample_id)  # raises ValueError if not a valid UUID
    assert s.ground_truth is None


def test_eval_sample_with_ground_truth():
    s = EvalSample(
        question="What is RAG?",
        contexts=["RAG stands for Retrieval-Augmented Generation."],
        answer="RAG is a technique combining retrieval with generation.",
        ground_truth="Retrieval-Augmented Generation combines retrieval with LLM generation.",
        relevant_doc_ids=["doc-1", "doc-2"],
        retrieved_doc_ids=["doc-1", "doc-3"],
    )
    assert s.relevant_doc_ids == ["doc-1", "doc-2"]


def test_eval_sample_requires_question():
    with pytest.raises(ValidationError):
        EvalSample(question="", contexts=["ctx"], answer="ans")


def test_eval_sample_legacy_singular_kwarg_promoted():
    """Backward-compat: EvalSample(ground_truth="x") promotes to plural list."""
    s = EvalSample(
        question="q",
        contexts=["c"],
        answer="a",
        ground_truth="legacy-single",
    )
    assert s.ground_truths == ["legacy-single"]
    assert s.ground_truth == "legacy-single"  # legacy property accessor


def test_agent_trace_legacy_singular_kwarg_promoted():
    """Regression test for RB-NEW1: AgentTrace(ground_truth="x") must promote to
    plural list (the EvalSample shim was applied, the AgentTrace shim was
    initially missed and silently dropped the kwarg)."""
    t = AgentTrace(
        question="q",
        final_answer="a",
        ground_truth="legacy-single",
    )
    assert t.ground_truths == ["legacy-single"]
    assert t.ground_truth == "legacy-single"  # legacy property accessor


def test_agent_trace_plural_kwarg():
    """AgentTrace(ground_truths=[...]) is the canonical shape."""
    t = AgentTrace(
        question="q",
        final_answer="a",
        ground_truths=["one", "two"],
    )
    assert t.ground_truths == ["one", "two"]
    assert t.ground_truth == "one"  # accessor returns first element


def test_agent_trace_minimal():
    t = AgentTrace(
        question="What is RAG?",
        final_answer="RAG combines retrieval with generation.",
        tool_calls=[
            ToolCall(
                tool_name="retrieve",
                tool_input={"query": "What is RAG?"},
                tool_output="RAG stands for Retrieval-Augmented Generation.",
                step_index=0,
            )
        ],
    )
    assert len(t.tool_calls) == 1
    assert t.conversation_history == []


def test_benchmark_report_score_range():
    report = BenchmarkReport(
        run_id="run-001",
        n_samples=10,
        metrics={"faithfulness": 0.85, "answer_relevancy": 0.90},
    )
    assert 0.0 <= report.metrics["faithfulness"] <= 1.0


def test_run_config_defaults():
    config = RunConfig(metrics=["faithfulness"])
    assert config.judge_model == "gemini-2.5-flash"
    assert config.metric_group is None


def test_eval_sample_unique_ids():
    s1 = EvalSample(question="Q1?", contexts=["ctx"], answer="A1.")
    s2 = EvalSample(question="Q2?", contexts=["ctx"], answer="A2.")
    assert s1.sample_id != s2.sample_id
