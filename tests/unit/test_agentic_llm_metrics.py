import json
from unittest.mock import MagicMock, patch

import pytest

from app.eval.agentic_llm_metrics import (
    compute_agent_faithfulness,
    compute_retrieval_necessity,
    compute_tool_call_accuracy,
)
from harness.schemas import AgentTrace, ReasoningStep, RetrievedChunk, ToolCall


def make_trace():
    return AgentTrace(
        question="What is the 2025 EU AI Act deadline?",
        final_answer="The GPAI obligations apply from August 2025.",
        tool_calls=[
            ToolCall(
                tool_name="retrieve",
                tool_input={"query": "EU AI Act 2025 deadline"},
                tool_output="Article 53 obligations apply from August 2025.",
                step_index=0,
            )
        ],
        reasoning_steps=[
            ReasoningStep(
                step_index=0,
                thought="I need to check the EU AI Act deadline for GPAI.",
                action="retrieve",
                observation="August 2025.",
                cited_sources=["doc-1"],
            )
        ],
        retrieved_chunks=[
            RetrievedChunk(
                source_id="doc-1",
                content="Article 53 obligations apply from August 2025.",
            )
        ],
    )


MOCK_FAITHFUL_RESPONSE = json.dumps(
    {
        "step_analysis": [{"step_index": 0, "claims": ["August 2025"], "supported": [True], "faithfulness_score": 1.0}],
        "trace_faithfulness_score": 1.0,
        "worst_step": 0,
        "critical_hallucinations": [],
    }
)

MOCK_TOOL_RESPONSE = json.dumps(
    {
        "tool_evaluations": [
            {
                "step_index": 0,
                "tool_name": "retrieve",
                "necessary": True,
                "correct_tool": True,
                "input_quality": 1.0,
                "score": 1.0,
                "reason": "Correct retrieval",
            }
        ],
        "overall_score": 1.0,
    }
)

MOCK_NECESSITY_RESPONSE = json.dumps(
    {
        "necessity": "NECESSARY",
        "parametric_answer_possible": False,
        "retrieval_contribution": "essential",
        "score": 1.0,
        "reasoning": "Requires specific regulatory date.",
    }
)


def _mock_llm(response: str):
    m = MagicMock()
    m.generate.return_value = response
    return m


def test_agent_faithfulness_full():
    trace = make_trace()
    with patch("app.eval.agentic_llm_metrics.LLMClient", return_value=_mock_llm(MOCK_FAITHFUL_RESPONSE)):
        result = compute_agent_faithfulness(trace)
    assert result["score"] == pytest.approx(1.0)
    assert result["worst_step"] == 0
    assert result["critical_hallucinations"] == []


def test_agent_faithfulness_no_steps_returns_neutral():
    trace = AgentTrace(
        question="Q?",
        final_answer="A.",
    )
    with patch("app.eval.agentic_llm_metrics.LLMClient", return_value=_mock_llm(MOCK_FAITHFUL_RESPONSE)):
        result = compute_agent_faithfulness(trace)
    # No steps → no reasoning to evaluate → neutral 1.0
    assert result["score"] == pytest.approx(1.0)


def test_tool_call_accuracy_full():
    trace = make_trace()
    with patch("app.eval.agentic_llm_metrics.LLMClient", return_value=_mock_llm(MOCK_TOOL_RESPONSE)):
        result = compute_tool_call_accuracy(trace)
    assert result["score"] == pytest.approx(1.0)
    assert len(result["tool_evaluations"]) == 1


def test_tool_call_accuracy_no_calls_returns_perfect():
    trace = AgentTrace(question="Q?", final_answer="A.")
    with patch("app.eval.agentic_llm_metrics.LLMClient", return_value=_mock_llm(MOCK_TOOL_RESPONSE)):
        result = compute_tool_call_accuracy(trace)
    assert result["score"] == pytest.approx(1.0)
    assert result["tool_evaluations"] == []


def test_retrieval_necessity_essential():
    with patch("app.eval.agentic_llm_metrics.LLMClient", return_value=_mock_llm(MOCK_NECESSITY_RESPONSE)):
        result = compute_retrieval_necessity(
            question="What is the 2025 deadline?",
            answer="August 2025.",
            contexts=["Article 53 applies from August 2025."],
        )
    assert result["score"] == pytest.approx(1.0)
    assert result["necessity"] == "NECESSARY"


def test_faithfulness_malformed_response_returns_zero():
    trace = make_trace()
    with patch("app.eval.agentic_llm_metrics.LLMClient", return_value=_mock_llm("not json")):
        result = compute_agent_faithfulness(trace)
    assert result["score"] == pytest.approx(0.0)
