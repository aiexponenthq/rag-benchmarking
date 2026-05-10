# CI/CD Integration Guide

## Integrating RAG Evaluation into GitHub Actions

Running evaluation in CI gives you a continuous signal on whether changes to your RAG pipeline (prompts, chunking strategy, embedding model, reranker) are improvements or regressions.

---

## Basic Setup: Evaluate on Every PR

```yaml
# .github/workflows/rag-eval.yml
name: RAG Evaluation

on:
  pull_request:
    paths:
      - 'src/**'
      - 'data/**'
      - 'scripts/**'

jobs:
  evaluate:
    runs-on: ubuntu-latest
    services:
      rag-benchmarking:
        image: your-org/rag-benchmarking:latest
        ports:
          - 5001:5000
        env:
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
          API_KEY: ci-test-key
          ENFORCE_API_KEY: "true"
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Wait for server to be ready
        run: |
          for i in {1..30}; do
            curl -sf http://localhost:5001/health && break
            sleep 2
          done

      - name: Run evaluation
        run: |
          python scripts/evaluate.py data/golden/qa.jsonl \
            --metrics faithfulness answer_relevancy \
            --out-json reports/ci-results.json \
            --out-md reports/ci-results.md

      - name: Check thresholds
        run: |
          python - <<'EOF'
          import json, sys

          with open("reports/ci-results.json") as f:
              results = json.load(f)

          THRESHOLDS = {
              "faithfulness": 0.75,
              "answer_relevancy": 0.80,
          }

          failures = []
          for metric, threshold in THRESHOLDS.items():
              score = results.get("scores", {}).get(metric, 0)
              if score < threshold:
                  failures.append(f"{metric}: {score:.3f} < {threshold}")

          if failures:
              print("Evaluation thresholds not met:")
              for f in failures:
                  print(f"  {f}")
              sys.exit(1)
          else:
              print("All evaluation thresholds met")
          EOF

      - name: Upload results
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: eval-results-${{ github.run_id }}
          path: reports/
```

---

## Threshold Strategy

**Do not assert exact scores in CI.** LLM-as-judge metrics are non-deterministic even at temperature 0; expect ±0.02 variance run-to-run. Use thresholds to catch regressions, not to pin to a specific value.

| Approach | When to use |
|---|---|
| Minimum threshold (`score >= X`) | Catch significant regressions |
| Rolling average (last N runs) | Detect gradual drift over time |
| Relative delta (`<= 0.05 drop vs baseline`) | Compare two configurations head-to-head |

**Recommended starting thresholds:**

| Metric | Threshold | Notes |
|---|---|---|
| `faithfulness` | ≥ 0.75 | Below this, the system is regularly hallucinating |
| `answer_relevancy` | ≥ 0.80 | Below this, answers are frequently off-topic |
| `source_attribution_accuracy` | ≥ 0.95 | Deterministic — can use a strict assertion |

Start conservative. Tighten thresholds as your pipeline matures.

---

## Comparing RAG Configurations

Use the Python SDK (`src/app/sdk/client.py`) to run two configurations against the same sample set and compare results directly:

```python
from app.sdk.client import RagEval

client = RagEval(api_url="http://localhost:5001", api_key="ci-test-key")

# Load the golden dataset
import json
with open("data/golden/qa.jsonl") as f:
    samples = [json.loads(line) for line in f]

# Evaluate configuration A (e.g., your current pipeline output)
report_a = client.evaluate(samples, metrics=["faithfulness", "answer_relevancy"])

# Produce answers with configuration B (e.g., a new chunking strategy)
samples_b = [run_my_rag(s["question"]) for s in samples]
report_b = client.evaluate(samples_b, metrics=["faithfulness", "answer_relevancy"])

# Compare run IDs
comparison = client.compare_runs([report_a["run_id"], report_b["run_id"]])
for metric, scores in comparison["metrics"].items():
    score_a, score_b = scores
    delta = score_b - score_a
    direction = "+" if delta >= 0 else ""
    print(f"{metric}: {score_a:.3f} -> {score_b:.3f} ({direction}{delta:.3f})")
```

---

## Using Retrieval Metrics in CI

Retrieval metrics (`precision_at_k`, `recall_at_k`, `mrr`, `ndcg_at_k`) are fully deterministic — they require no LLM judge call and use only `relevant_doc_ids` from the dataset. They are fast and cheap to run on every commit.

```bash
# Retrieval-only evaluation (no GEMINI_API_KEY required)
python scripts/evaluate.py data/golden/qa.jsonl \
  --metrics precision_at_k recall_at_k mrr ndcg_at_k \
  --out-json reports/retrieval-results.json
```

Because these are deterministic, you can use strict equality assertions in CI:

```python
with open("reports/retrieval-results.json") as f:
    results = json.load(f)

assert results["metrics"]["precision_at_k"] >= 0.65, \
    f"Retrieval precision dropped: {results['scores']['precision_at_k']:.3f}"
```

---

## Using `source_attribution_accuracy` as a CI Gate

`source_attribution_accuracy` (from the `agentic_v1` metric group) is deterministic: it checks whether the document IDs cited in an agent's response are present in the retrieved set. No LLM call is made. This makes it safe to use as a hard gate in CI:

```python
report = client.evaluate(
    samples,
    metrics=["source_attribution_accuracy"]
)
score = report["metrics"]["source_attribution_accuracy"]
assert score >= 0.95, \
    f"Agent is hallucinating source citations: attribution accuracy = {score:.3f}"
```

---

## Running the Full Metric Suite Locally Before Pushing

The `full` metric group covers all available metrics. Run this locally before opening a PR against a production pipeline:

```bash
# Requires GEMINI_API_KEY set in environment
export GEMINI_API_KEY=your_key_here

python scripts/evaluate.py data/golden/qa.jsonl \
  --out-json reports/full-results.json \
  --out-md reports/full-results.md

cat reports/full-results.md
```

This produces both a machine-readable JSON report and a human-readable Markdown summary.

---

## Environment Variables for CI

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | Yes (for LLM metrics) | Judge model API key |
| `API_KEY` | Yes (if `ENFORCE_API_KEY=true`) | Key for the evaluation server |
| `ENFORCE_API_KEY` | No | Set `"true"` to require `X-API-Key` on all requests |
| `HOST_PORT` | No | Override host port (default: `5001`) — container internal port stays `5000` |

Store `GEMINI_API_KEY` as a GitHub Actions secret — never hardcode it in workflow files.
