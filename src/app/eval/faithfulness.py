from __future__ import annotations

import json
import logging

from app.llm.client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM = """You are a strict factual consistency evaluator for RAG systems.

Step 1 — Decompose the ANSWER into atomic factual claims.
Each claim must be a single verifiable statement.
Output a JSON list of strings, key "claims".

Step 2 — For each claim, determine if it is FULLY SUPPORTED by the CONTEXT.
A claim is supported if, and only if, there is specific text in the CONTEXT that
directly entails the claim. Indirect inference is NOT support.
Output a JSON list of booleans, key "supported", same length as "claims".

Return only valid JSON in this exact format:
{
  "claims": ["claim 1", "claim 2"],
  "supported": [true, false]
}"""

_USER = """CONTEXT:
{context}

ANSWER:
{answer}"""


def compute_faithfulness(answer: str, contexts: list[str]) -> dict:
    """
    Compute claim-level faithfulness score.

    Returns:
        {
            "score": float,          # 0.0 (unfaithful) to 1.0 (faithful)
            "claims": list[str],     # atomic claims decomposed from answer
            "supported": list[bool], # whether each claim is supported
        }
    score is 1.0 when answer has no factual claims (neutral/empty answer).
    score is 0.0 on LLM judge failure.
    """
    llm = LLMClient()
    context_text = "\n\n".join(contexts)
    user = _USER.format(context=context_text, answer=answer)

    try:
        raw = llm.generate(_SYSTEM, user).strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        data = json.loads(raw)
        claims: list[str] = data.get("claims", [])
        supported: list[bool] = data.get("supported", [])

        if not claims:
            return {"score": 1.0, "claims": [], "supported": []}

        # Guard against length mismatch from the LLM
        min_len = min(len(claims), len(supported))
        claims = claims[:min_len]
        supported = supported[:min_len]

        score = sum(1 for s in supported if s) / len(claims)
        return {"score": score, "claims": claims, "supported": supported}

    except Exception as exc:
        logger.warning("Faithfulness evaluation failed: %s", exc)
        return {"score": 0.0, "claims": [], "supported": []}
