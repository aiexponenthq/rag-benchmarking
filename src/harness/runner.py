from __future__ import annotations

import logging
import uuid
from datetime import datetime

from harness.schemas import (
    BenchmarkReport,
    EvalResult,
    EvalSample,
    METRIC_GROUPS,
    RunConfig,
)

logger = logging.getLogger(__name__)

try:
    from app.eval.ragas_runner import run_evaluation
except ImportError:  # pragma: no cover
    run_evaluation = None  # type: ignore[assignment]

# Metrics handled by RAGAS
_RAGAS_METRICS = {"faithfulness", "answer_relevancy", "context_precision", "context_recall"}
# Metrics requiring ground_truth
_GROUND_TRUTH_REQUIRED = {"context_precision", "context_recall"}
# Metrics requiring relevant_doc_ids
_RELEVANT_IDS_REQUIRED = {"precision_at_k", "recall_at_k", "mrr", "ndcg_at_k"}


def _resolve_metrics(config: RunConfig) -> list[str]:
    """Expand metric_group into concrete list; merge with explicit metrics."""
    metrics = list(config.metrics)
    if config.metric_group:
        for m in METRIC_GROUPS.get(config.metric_group.value, []):
            if m not in metrics:
                metrics.append(m)
    return metrics


class EvaluationRunner:
    """
    Orchestrates metric computation over EvalSamples.
    Does not care how the samples were produced — works with any RAG system.
    """

    def __init__(self, config: RunConfig) -> None:
        self.config = config
        self._metrics = _resolve_metrics(config)

    def evaluate(self, samples: list[EvalSample]) -> BenchmarkReport:
        run_id = self.config.run_id or str(uuid.uuid4())
        skipped: list[str] = []
        skip_reasons: dict[str, str] = {}

        # Pre-flight: determine what can actually be computed
        has_ground_truths = all(s.ground_truth for s in samples)
        has_relevant_ids = all(s.relevant_doc_ids for s in samples)

        active: list[str] = []
        for m in self._metrics:
            if m in _GROUND_TRUTH_REQUIRED and not has_ground_truths:
                skipped.append(m)
                skip_reasons[m] = "requires ground_truth on all samples"
            elif m in _RELEVANT_IDS_REQUIRED and not has_relevant_ids:
                skipped.append(m)
                skip_reasons[m] = "requires relevant_doc_ids on all samples"
            else:
                active.append(m)

        # Split metrics by handler
        ragas_metrics = [m for m in active if m in _RAGAS_METRICS]
        other_metrics = [m for m in active if m not in _RAGAS_METRICS]

        aggregate: dict[str, float] = {}
        per_sample_scores: dict[str, list[float]] = {}

        # --- RAGAS metrics ---
        if ragas_metrics:
            ragas_samples = [
                {
                    "question": s.question,
                    "contexts": s.contexts,
                    "answer": s.answer,
                    "ground_truths": [s.ground_truth] if s.ground_truth else [],
                }
                for s in samples
            ]
            result = run_evaluation(ragas_samples, metrics=ragas_metrics)
            aggregate.update(result.get("metrics", {}))
            per_sample_scores.update(result.get("per_sample", {}))
            for m in result.get("skipped_metrics", []):
                if m not in skipped:
                    skipped.append(m)
                    skip_reasons[m] = result.get("skip_reason") or "skipped by RAGAS runner"

        # --- Other metrics (stubs until Phase 2/3 tasks implement them) ---
        for m in other_metrics:
            if m == "source_attribution_accuracy":
                # Stub: will be replaced in Task 3.1
                aggregate[m] = 1.0
                per_sample_scores[m] = [1.0] * len(samples)
            elif m in _RELEVANT_IDS_REQUIRED:
                # Already filtered above; shouldn't reach here
                pass
            else:
                logger.warning("Metric '%s' not yet implemented, skipping", m)
                skipped.append(m)
                skip_reasons[m] = "not yet implemented"

        # Build per-sample EvalResult list
        per_sample_results = [
            EvalResult(
                sample_id=s.sample_id,
                metrics={
                    metric: per_sample_scores[metric][i]
                    for metric in aggregate
                    if metric in per_sample_scores and i < len(per_sample_scores[metric])
                },
            )
            for i, s in enumerate(samples)
        ]

        return BenchmarkReport(
            run_id=run_id,
            created_at=datetime.utcnow(),
            n_samples=len(samples),
            metrics=aggregate,
            per_sample=per_sample_results,
            skipped_metrics=skipped,
            skip_reasons=skip_reasons,
            config=self.config.model_dump(mode="json"),
        )
