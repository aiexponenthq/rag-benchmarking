# Golden Dataset Methodology

## Overview

The golden dataset (`data/golden/qa.jsonl`) contains 50 evaluation samples across 10 domains relevant to AI governance, RAG systems, and enterprise AI deployment.

Each record is a self-contained JSONL line. The dataset is used as the primary reference corpus for the benchmark scores published in `docs/benchmark-results.md`.

---

## Domain Coverage

10 domains × 5 samples each (50 total):

| # | Domain | Prefix | Topics covered |
|---|---|---|---|
| 1 | RAG Fundamentals | `rag-` | Chunking, retrieval pipelines, reranking, top-K |
| 2 | Vector Databases | `vdb-` | HNSW indexing, cosine similarity, Qdrant |
| 3 | EU AI Act | `euai-` | Articles 4, 9, 15, 53; high-risk system classification |
| 4 | RAG Evaluation Metrics | `eval-` | Faithfulness, NDCG, the RAG triad |
| 5 | LLMs and Transformers | `llm-` | Attention mechanisms, temperature, hallucination |
| 6 | Python / FastAPI | `py-` | Pydantic, async/await, dependency injection |
| 7 | MLOps | `mlops-` | Model drift, A/B testing, containerisation |
| 8 | AI Security | `sec-` | Prompt injection, PII handling, data poisoning |
| 9 | Data Engineering | `de-` | ETL, Kafka, dbt, change-data-capture |
| 10 | Responsible AI | `rai-` | Bias, explainability, NIST AI RMF |

---

## Sample Structure

Each sample in the JSONL file contains the following fields:

```json
{
  "question":        "What does RAG stand for and what problem does it solve?",
  "contexts":        ["RAG stands for Retrieval-Augmented Generation ...", "Traditional LLMs are limited ..."],
  "answer":          "RAG stands for Retrieval-Augmented Generation. It solves the hallucination problem by grounding LLM responses in retrieved documents.",
  "ground_truths":   ["RAG stands for Retrieval-Augmented Generation and reduces hallucination by grounding responses in retrieved documents."],
  "relevant_doc_ids": ["rag-1", "rag-2"],
  "sample_id":       "03fd8be3-1051-4213-acda-0fc242447018"
}
```

| Field | Type | Used by |
|---|---|---|
| `question` | `str` | All metrics |
| `contexts` | `list[str]` | `faithfulness`, `answer_relevancy` |
| `answer` | `str` | All LLM-as-judge metrics |
| `ground_truths` | `list[str]` | Context precision/recall scoring |
| `relevant_doc_ids` | `list[str]` | `precision_at_k`, `recall_at_k`, `mrr`, `ndcg_at_k` |
| `sample_id` | `str` (UUID) | Traceability and result linking |

> **Note:** The field is `ground_truths` (plural, a list), not `ground_truth`. This matches the RAGAS library convention.

---

## Design Decisions

### Why 50 samples?

50 samples provides enough statistical power for relative comparisons — detecting changes of approximately ±0.05 in metric scores — while remaining fast to evaluate (roughly 2–5 minutes per metric group with Gemini 2.5 Flash as judge). For absolute metric stability, 200+ samples are recommended; this dataset is designed for fast iteration rather than production certification.

### Why these domains?

Domains were selected to reflect the primary users of this tool: teams building AI governance and compliance tooling. The EU AI Act domain is particularly relevant given this tool's positioning as **partial Article 15(1) accuracy input** for high-risk AI systems. Robustness in the regulatory sense and cybersecurity are out of scope for this harness — those legs of Article 15 require dedicated tooling (see the README scope note).

The selection also covers the full engineering stack (Python/FastAPI, MLOps, Data Engineering) so that platform teams — not just ML researchers — can use the dataset as a meaningful evaluation signal.

### How answers were written

Answers were authored to be:

- **Faithful to the provided context** — no external facts or knowledge injected beyond what appears in `contexts`
- **Concise but complete** — a single coherent response rather than a bullet dump
- **In plain English** — paraphrased, not copied verbatim from the context passage

This makes the dataset suitable for faithfulness evaluation: a well-functioning RAG system should reproduce answers that score near 1.0 on faithfulness for these samples, since the source context is sufficient.

### How `relevant_doc_ids` were assigned

Document IDs use the format `{domain_prefix}-{n}` (e.g., `rag-1`, `euai-3`). Each sample lists the IDs of the documents that genuinely contain enough information to answer the question. Samples where the answer spans multiple documents list multiple IDs; samples with a single sufficient source list one.

---

## Limitations

- **English-only** — all questions, contexts, and answers are in English; multilingual RAG systems will require separate domain-specific datasets.
- **5 samples per domain** — this is insufficient for reliable domain-specific sub-analyses. Treat per-domain breakdowns as directional indicators only.
- **Single annotator** — ground truth answers were not validated by multiple independent annotators; there may be valid alternative phrasings that would score lower with a strict judge.
- **Judge-relative scores** — the baseline scores in `docs/benchmark-results.md` are calibrated for Gemini 2.5 Flash at temperature 0.0. Other judge models (GPT-4o, Claude 3.5 Sonnet, etc.) will produce different absolute scores; relative comparisons within the same judge are reliable, cross-judge comparisons are not.
- **Static context** — contexts are short, curated passages. Real-world RAG systems dealing with long, noisy documents will typically see lower faithfulness and precision scores.

---

## Extending the Dataset

To add new samples, append to `data/golden/qa.jsonl` following the same schema. The generation script is at `scripts/generate_golden.py`:

```bash
# Regenerate the dataset (overwrites existing file)
python scripts/generate_golden.py --output data/golden/qa.jsonl --n 50

# To add new domains or samples:
# Edit the SAMPLES list in scripts/generate_golden.py, then re-run
```

Each new sample must include all six fields. `sample_id` should be a fresh UUID:

```python
import uuid
print(uuid.uuid4())
```

To contribute domain-specific samples, open a GitHub issue with the `golden-dataset` label and include the proposed samples as a JSONL snippet.
