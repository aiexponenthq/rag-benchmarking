from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def run_evaluation(
    samples: Sequence[dict[str, Any]],
    metrics: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run a RAG evaluation on provided samples using RAGAS 0.4.x.

    Parameters
    ----------
    samples: Sequence[Dict[str, Any]]
        Items with keys: question (str), contexts (List[str]), answer (str),
        ground_truths (List[str]).  ground_truths is optional — when absent
        or empty the retrieval metrics (context_precision, context_recall) are
        automatically skipped and only LLM-judge metrics are computed.
    metrics: Sequence[str] | None
        Metric names to compute. Defaults to faithfulness, answer_relevancy,
        and — when ground truth is present — context_precision, context_recall.

    Returns
    -------
    Dict[str, Any]
        Aggregate metrics with means and per-sample scores if available.

    Notes
    -----
    RAGAS 0.4.x column mapping (SingleTurnSample):
        question          -> user_input
        answer            -> response
        contexts          -> retrieved_contexts
        ground_truths[0]  -> reference   (first element only; ragas expects a
                                          single string reference, not a list)

    Retrieval metrics require ``reference`` — they use LLMContextPrecisionWithReference
    and LLMContextRecall from ``ragas.metrics._context_precision / _context_recall``.
    These are the stable 0.4.x classes (the ``ragas.metrics.collections`` variants
    require a new InstructorBaseRagasLLM interface that is not yet LangChain-compatible).
    """

    import os

    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    from ragas.metrics._answer_relevance import AnswerRelevancy
    from ragas.metrics._context_precision import LLMContextPrecisionWithReference
    from ragas.metrics._context_recall import LLMContextRecall
    from ragas.metrics._faithfulness import Faithfulness

    # ------------------------------------------------------------------ #
    # Determine which metrics to run                                       #
    # ------------------------------------------------------------------ #
    all_requested = list(metrics or ["faithfulness", "answer_relevancy", "context_precision", "context_recall"])

    # Retrieval metrics need ground truth — inspect first sample to decide
    has_ground_truth = any(bool(s.get("ground_truths") or s.get("reference")) for s in samples)

    RETRIEVAL_METRICS = {"context_precision", "context_recall"}
    selected: list[str] = []
    for m in all_requested:
        if m in RETRIEVAL_METRICS and not has_ground_truth:
            # Graceful skip: no reference contexts available
            continue
        selected.append(m)

    # Build metric objects — LLM is injected AFTER ragas_llm is created below.
    # Store as dict so we can set the llm on each object once it's ready.
    name_to_metric: dict[str, Any] = {
        "faithfulness": Faithfulness(),
        "answer_relevancy": AnswerRelevancy(),
        "context_precision": LLMContextPrecisionWithReference(),
        "context_recall": LLMContextRecall(),
    }

    metric_objs = [name_to_metric[m] for m in selected if m in name_to_metric]

    # ------------------------------------------------------------------ #
    # Build EvaluationDataset from caller-supplied dicts                   #
    # ------------------------------------------------------------------ #
    # Translate legacy field names to SingleTurnSample field names.
    # SingleTurnSample accepted fields (ragas 0.4.x):
    #   user_input, retrieved_contexts, response, reference
    ragas_samples: list[SingleTurnSample] = []
    for s in samples:
        ground_truth_list: list[str] = s.get("ground_truths") or []
        reference: str | None = s.get("reference") or (ground_truth_list[0] if ground_truth_list else None)
        ragas_samples.append(
            SingleTurnSample(
                user_input=s.get("user_input") or s.get("question", ""),
                retrieved_contexts=s.get("retrieved_contexts") or s.get("contexts") or [],
                response=s.get("response") or s.get("answer", ""),
                reference=reference,
            )
        )

    dataset = EvaluationDataset(samples=ragas_samples)

    # ------------------------------------------------------------------ #
    # Configure judge LLM (LangChain wrapper for RAGAS 0.4.x)             #
    # ------------------------------------------------------------------ #
    from langchain_google_genai import ChatGoogleGenerativeAI
    from ragas.llms import LangchainLLMWrapper

    from app.config.settings import get_settings as _get_settings

    _settings = _get_settings()

    gemini_api_key = _settings.gemini_api_key or os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is required. Set it in your .env file.")

    _gemini_model = _settings.gemini_model or "gemini-2.5-flash"

    # langchain-google-genai 4.x reads the key from GOOGLE_API_KEY env var only
    # (the google_api_key constructor parameter was removed in 4.x)
    os.environ["GOOGLE_API_KEY"] = gemini_api_key

    langchain_llm = ChatGoogleGenerativeAI(
        model=_gemini_model,
        temperature=0.0,
    )
    ragas_llm = LangchainLLMWrapper(langchain_llm)

    # Inject the LLM into every metric object explicitly.
    # Without this, metrics default to OpenAI even when a different LLM
    # is passed to evaluate().
    for metric in metric_objs:
        metric.llm = ragas_llm

    # AnswerRelevancy also needs embeddings — inject HuggingFace (sentence-transformers)
    # so it never falls back to OpenAI embeddings.
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    _hf_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
    for metric in metric_objs:
        if hasattr(metric, "embeddings"):
            metric.embeddings = _hf_embeddings

    # ------------------------------------------------------------------ #
    # Run evaluation                                                       #
    # ------------------------------------------------------------------ #
    result = evaluate(
        dataset=dataset,
        metrics=metric_objs,
        llm=ragas_llm,
        show_progress=False,
        raise_exceptions=True,
    )

    import logging as _logging
    import math as _math

    _logger = _logging.getLogger(__name__)

    try:
        df = result.to_pandas()
    except Exception as exc:
        _logger.warning("RAGAS result.to_pandas() failed: %s — trying scores dict", exc)
        # Fallback: try to read scores directly from the result object
        scores = getattr(result, "scores", {}) or {}
        aggregates = {m: float(scores.get(m, float("nan"))) for m in selected}
        per_sample = {}
        skipped = [m for m in all_requested if m not in selected]
        return {
            "metrics": {k: (None if _math.isnan(v) else v) for k, v in aggregates.items()},
            "per_sample": per_sample,
            "skipped_metrics": skipped,
            "skip_reason": None,
        }

    aggregates: dict[str, float] = {}
    per_sample: dict[str, list[float]] = {}
    for m in selected:
        if m in df.columns:
            vals = df[m].dropna().tolist()
            mean_val = float(df[m].mean()) if vals else float("nan")
            # Replace NaN with None so JSON serialises cleanly (NaN → null → None)
            aggregates[m] = mean_val if not _math.isnan(mean_val) else None  # type: ignore[assignment]
            per_sample[m] = [float(v) for v in vals]

    skipped = [m for m in all_requested if m not in selected]

    return {
        "metrics": aggregates,
        "per_sample": per_sample,
        "skipped_metrics": skipped,
        "skip_reason": (
            "ground_truths not provided — context_precision and context_recall require reference" if skipped else None
        ),
    }
