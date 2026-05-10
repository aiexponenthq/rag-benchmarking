from __future__ import annotations

import json
import logging
from typing import Any

from app.llm.client import LLMClient
from harness.schemas import AgentTrace

logger = logging.getLogger(__name__)


def _call_judge(system: str, user: str) -> dict[str, Any]:
    """Call LLM judge and parse JSON. Returns empty dict on failure."""
    llm = LLMClient()
    raw = llm.generate(system, user).strip()
    try:
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        return json.loads(raw)
    except Exception as exc:
        logger.warning("Judge parse failure: %s | raw: %.200s", exc, raw)
        return {}


_AGENT_FAITHFULNESS_SYSTEM = """You are evaluating factual consistency across an AI agent's full reasoning trace.

For each reasoning step, verify every factual claim is supported by the tool output at that step
or a previously retrieved source.
A claim is NOT supported if it appears in the reasoning but not in any tool output, or contradicts the tool output.

Return JSON only:
{
  "step_analysis": [
    {"step_index": int, "claims": ["..."], "supported": [true|false], "faithfulness_score": float}
  ],
  "trace_faithfulness_score": float,
  "worst_step": int,
  "critical_hallucinations": ["..."]
}"""

_TOOL_ACCURACY_SYSTEM = """You are evaluating whether an AI agent made appropriate tool calls to answer a question.
For each tool call assess: was it necessary, was the correct tool chosen, was the input well-formed?
Score each: 0 (wrong), 0.5 (partial), 1 (correct).

Return JSON only:
{
  "tool_evaluations": [
    {"step_index": int, "tool_name": str, "necessary": bool, "correct_tool": bool,
     "input_quality": float, "score": float, "reason": str}
  ],
  "overall_score": float
}"""

_NECESSITY_SYSTEM = """Evaluate whether retrieval was necessary to answer this question.
Categories: NECESSARY / HELPFUL / UNNECESSARY

Return JSON only:
{
  "necessity": "NECESSARY"|"HELPFUL"|"UNNECESSARY",
  "parametric_answer_possible": bool,
  "retrieval_contribution": "none"|"marginal"|"significant"|"essential",
  "score": float,
  "reasoning": str
}"""


def compute_agent_faithfulness(trace: AgentTrace) -> dict:
    """
    Evaluates faithfulness across the entire agent reasoning trace.
    Returns score 1.0 when no reasoning steps present (neutral).
    """
    if not trace.reasoning_steps:
        return {
            "score": 1.0,
            "worst_step": -1,
            "critical_hallucinations": [],
            "step_analysis": [],
        }

    steps_text = "\n\n".join(
        f"Step {s.step_index}: {s.thought}\nObservation: {s.observation}" for s in trace.reasoning_steps
    )
    sources_text = (
        "\n\n".join(f"[{c.source_id}]: {c.content}" for c in trace.retrieved_chunks) or "No explicit chunks provided."
    )

    user = f"FULL AGENT TRACE:\n{steps_text}\n\nSOURCES:\n{sources_text}"
    data = _call_judge(_AGENT_FAITHFULNESS_SYSTEM, user)

    score = float(data.get("trace_faithfulness_score", 0.0))
    return {
        "score": max(0.0, min(1.0, score)),
        "worst_step": data.get("worst_step", -1),
        "critical_hallucinations": data.get("critical_hallucinations", []),
        "step_analysis": data.get("step_analysis", []),
    }


def compute_tool_call_accuracy(trace: AgentTrace) -> dict:
    """
    Evaluates whether the agent made appropriate tool calls.
    Returns score 1.0 with empty evaluations when no tool calls present.
    """
    if not trace.tool_calls:
        return {"score": 1.0, "tool_evaluations": []}

    calls_text = "\n".join(
        f"Step {tc.step_index}: {tc.tool_name}({tc.tool_input}) → {tc.tool_output[:200]}" for tc in trace.tool_calls
    )
    user = (
        f"QUESTION: {trace.question}\n\n"
        f"TOOL CALLS:\n{calls_text}\n\n"
        "AVAILABLE TOOLS: retrieve, web_search, code_exec, calculator"
    )
    data = _call_judge(_TOOL_ACCURACY_SYSTEM, user)
    score = float(data.get("overall_score", 0.0))
    return {
        "score": max(0.0, min(1.0, score)),
        "tool_evaluations": data.get("tool_evaluations", []),
    }


def compute_retrieval_necessity(
    question: str,
    answer: str,
    contexts: list[str],
) -> dict:
    """
    Evaluates whether retrieval was actually necessary for this query.
    High score = retrieval was essential; low score = retrieval was unnecessary.
    """
    context_text = "\n\n".join(contexts)
    user = f"QUESTION: {question}\n\nRETRIEVED CONTEXT:\n{context_text}\n\nFINAL ANSWER: {answer}"
    data = _call_judge(_NECESSITY_SYSTEM, user)
    score = float(data.get("score", 0.0))
    return {
        "score": max(0.0, min(1.0, score)),
        "necessity": data.get("necessity", "UNKNOWN"),
        "retrieval_contribution": data.get("retrieval_contribution", "UNKNOWN"),
        "reasoning": data.get("reasoning", ""),
    }
