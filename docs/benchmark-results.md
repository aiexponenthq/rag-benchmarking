# Benchmark Results — Golden Dataset v1.0

## Dataset

- 50 samples across 10 domains (RAG Fundamentals, Vector Databases, EU AI Act, Evaluation Metrics, LLMs, Python/FastAPI, MLOps, AI Security, Data Engineering, Responsible AI)
- Each sample has: `question`, `contexts`, `answer`, `ground_truths`, `relevant_doc_ids`, `sample_id`
- See `data/golden/qa.jsonl`

> **Note:** Scores below are representative baselines computed using Gemini 1.5 Flash as judge at `temperature=0.0`. Your scores will vary based on your LLM provider, judge model, and RAG system configuration. These figures are provided as illustrative reference points, not guarantees.

---

## Classic Metrics — Baseline Scores

Metric group: `classic` — LLM-as-judge metrics requiring `question`, `contexts`, and `answer`.

| Metric | Score | Interpretation |
|---|---|---|
| `faithfulness` | 0.89 | High — most answers fully grounded in context |
| `answer_relevancy` | 0.91 | High — answers address the questions directly |

---

## Retrieval Metrics — Baseline Scores

Metric group: `retrieval` — deterministic metrics using `relevant_doc_ids`. K=5.

| Metric | Score | Interpretation |
|---|---|---|
| `precision_at_k` | 0.72 | Moderate — 3–4 of top-5 retrieved docs are relevant |
| `recall_at_k` | 0.81 | Good — most relevant docs found in top-5 |
| `mrr` | 0.78 | Good — first relevant doc usually in top-2 positions |
| `ndcg_at_k` | 0.76 | Good — relevant docs concentrated near top |

---

## Score Interpretation

| Score Range | Label | Meaning |
|---|---|---|
| ≥ 0.90 | Excellent | Production-ready for high-stakes use |
| 0.75–0.89 | Good | Suitable for most enterprise deployments |
| 0.60–0.74 | Moderate | Acceptable for internal tools; improve before customer-facing |
| < 0.60 | Needs work | Significant hallucination or retrieval gaps |

---

## What These Scores Mean for Your System

### Faithfulness below 0.89

If your `faithfulness` score is significantly below 0.89 on this dataset:

1. **Check your chunking strategy** — larger chunks often improve faithfulness by giving the LLM more supporting text
2. **Review your prompt template** — explicitly instruct the LLM to only use the provided context; penalise external knowledge
3. **Increase top-K retrieval** — more context gives the LLM more to work with before it reaches for unsupported claims

### Context precision below 0.75

If your `precision_at_k` is below 0.75:

1. **Add a reranker** — a cross-encoder reranker (e.g., BGE-Reranker, Cohere Rerank) after initial retrieval can sharply improve precision
2. **Review your embedding model** — domain-specific or fine-tuned models typically outperform general-purpose models on specialised corpora
3. **Reduce chunk overlap** — excessive overlap can cause near-duplicate chunks to fill your top-K slots, artificially deflating precision

---

## Running the Benchmark

```bash
# Start the server
docker compose up -d

# Run classic LLM-as-judge metrics against the golden dataset
python scripts/evaluate.py data/golden/qa.jsonl \
  --metrics faithfulness answer_relevancy \
  --out-json reports/golden-results.json \
  --out-md reports/golden-results.md

# Or use metric groups for convenience
# (requires GEMINI_API_KEY in environment)
python scripts/evaluate.py data/golden/qa.jsonl \
  --out-json reports/golden-results.json

# View JSON results
cat reports/golden-results.json | python -m json.tool
```

The default output path is `reports/ragas_report.json` with a matching `reports/ragas_report.md` summary.

---

## Reproducibility Notes

- Judge model: Gemini 1.5 Flash (`gemini-1.5-flash`)
- Judge temperature: `0.0`
- K for retrieval metrics: `5`
- Dataset: `data/golden/qa.jsonl` (50 samples, SHA recorded in `reports/`)
- LLM-as-judge scores are **not fully deterministic** even at temperature 0. Expect ±0.02 variance across runs.
- Retrieval metrics (`precision_at_k`, `recall_at_k`, `mrr`, `ndcg_at_k`) are fully deterministic given fixed inputs.
