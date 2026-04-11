from __future__ import annotations

from app.eval.faithfulness import compute_faithfulness


def compute_groundedness(answer: str, contexts: list[str]) -> float:
    """
    Compute a groundedness score in [0, 1].
    Delegates to claim-decomposition faithfulness evaluator.
    Preserved for backward compatibility — RAGEngine calls this method.
    """
    result = compute_faithfulness(answer, contexts)
    return result["score"]
