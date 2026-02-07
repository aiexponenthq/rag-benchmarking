from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def run_evaluation(
    samples: Sequence[dict[str, Any]],
    metrics: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run a RAG evaluation on provided samples using RAGAS.

    Parameters
    ----------
    samples: Sequence[Dict[str, Any]]
        Items with keys: question (str), contexts (List[str]), answer (str), 
        ground_truths (List[str])
    metrics: Sequence[str] | None
        Metric names to compute. Defaults to faithfulness, answer_relevancy, 
        context_precision, context_recall.

    Returns
    -------
    Dict[str, Any]
        Aggregate metrics with means and per-sample scores if available.
    """

    # Import ragas lazily so that importing this module doesn't require it
    import os

    from datasets import Dataset
    from langchain_google_genai import ChatGoogleGenerativeAI
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        faithfulness,
    )

    selected = metrics or [
        "faithfulness",
        "answer_relevancy",
        # For simplicity in the POC, compute precision/recall only when reference is present
        # "context_precision",
        # "context_recall",
    ]

    name_to_metric = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        # "context_precision": context_precision,
        # "context_recall": context_recall,
    }

    metric_objs = [name_to_metric[m] for m in selected if m in name_to_metric]
    ds = Dataset.from_list(list(samples))
    # Configure Gemini as judge via LangChain LLM interface
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is required to run RAGAS with Gemini judge")

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", google_api_key=gemini_api_key, temperature=0.0
    )
    result = evaluate(ds, metrics=metric_objs, llm=llm)
    # Ragas returns a Dataset with columns for metric scores and an aggregate .to_pandas().mean()
    df = result.to_pandas()
    aggregates: dict[str, float] = {}
    for m in selected:
        if m in df.columns:
            aggregates[m] = float(df[m].mean())
    return {
        "metrics": aggregates,
    }
