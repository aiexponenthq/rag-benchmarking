from __future__ import annotations

from app.llm.client import LLMClient


def compute_groundedness(answer: str, contexts: list[str]) -> float:
    """Compute a simple groundedness score in [0,1] using an LLM-as-judge prompt.

    This is a lightweight rubric: the judge must return only a float between 0 and 1.
    """
    rubric = (
        "You are a strict evaluator. Given the CONTEXT and an ANSWER, return a single float "
        "between 0 and 1 indicating how well the answer is directly supported by the context "
        "(1 = fully supported, 0 = unsupported). Respond with only the number."
    )
    ctx = "\n\n".join(contexts)
    user = f"CONTEXT:\n{ctx}\n\nANSWER:\n{answer}\n\nScore:"
    llm = LLMClient()
    raw = llm.generate(rubric, user).strip()
    try:
        val = float(raw.split()[0])
        if val < 0:
            return 0.0
        if val > 1:
            return 1.0
        return val
    except Exception:
        return 0.0
