# How RAG Benchmarking Compares to Other Evaluation Tools

## Summary

| Feature | RAG Benchmarking | RAGAS | TruLens | DeepEval | Promptfoo |
|---|---|---|---|---|---|
| **Framework-agnostic** | Yes — any RAG system via adapter | Partial | Partial | Yes | Yes |
| **LangChain adapter** | Yes | Built-in | Built-in | Yes | Yes |
| **LlamaIndex adapter** | Yes | Built-in | Built-in | Yes | Partial |
| **Classic RAG metrics** | Yes (faithfulness, answer_relevancy, context_precision/recall) | Yes | Yes | Yes | Partial |
| **Retrieval metrics** | Yes (Precision@K, Recall@K, MRR, NDCG) | No | Partial | Partial | No |
| **Agentic-era metrics** | Yes (4 metrics: agent_faithfulness, tool_call_accuracy, retrieval_necessity, source_attribution) | Partial (Tool call Accuracy per docs.ragas.io) | No | Partial | No |
| **LLM-as-judge** | Yes (Gemini, OpenAI) | Yes | Yes | Yes | Yes |
| **Deterministic metrics** | Yes (source_attribution_accuracy) | No | No | Yes | Yes |
| **REST API** | Yes (FastAPI, API key auth) | No | No | No | No (CLI / yaml) |
| **Python SDK** | Yes | Direct library | Direct library | Direct library | Yes |
| **Run comparison** | Yes (SQLite, /v1/runs/compare) | No | Yes (dashboard) | No | Yes (web UI) |
| **Self-hosted** | Yes (Docker Compose) | Yes | Yes | Yes | Yes |
| **EU AI Act framing** | Partial Art. 15(1) accuracy input | No | No | No | No |
| **Open source** | Yes (Apache 2.0) | Yes (Apache 2.0) | Yes (MIT) | Yes (Apache 2.0) | Yes (MIT) |

> **Note (audit 2026-05-10):** RAGAS's stable docs at docs.ragas.io list `Tool call Accuracy`
> under the *Agents or Tool use cases* section, so the agentic-metrics row is honestly
> "Partial" rather than "No". An earlier version of this table had RAGAS at "No" — fixed.
> Promptfoo column added; it is primarily a prompt-evaluation framework with some RAG
> support, so RAG-specific metrics show up as Partial / No rather than direct competitors.

---

## When to Use Each Tool

### Use RAG Benchmarking when:
- You need **agentic evaluation** — multi-step agents, tool use, reasoning traces
- You want a **server-based evaluation API** that your team can hit from any language
- You need **EU AI Act Article 15** evaluation evidence
- You're comparing RAG systems across runs over time with run history
- You have an existing RAG system in any framework and need a thin adapter

### Use RAGAS when:
- You want direct Python library integration without a server
- You're using LangChain or LlamaIndex natively and want RAGAS's built-in chains
- You need reference-free evaluation at the lowest setup friction
- You only need the RAG Triad (context relevance, faithfulness, answer relevance)

### Use TruLens when:
- You want a visual dashboard for monitoring evaluation results over time
- You're already using TruLens for observability and want evaluation in the same UI
- You need continuous monitoring rather than batch benchmark runs

### Use DeepEval when:
- You need a large library of predefined metrics out of the box
- You want LLM-based testing integrated into a pytest-style workflow
- You need G-Eval or similar custom LLM-as-judge metric definitions

---

## Detailed Comparison

### RAGAS

RAGAS is the most widely-used open-source RAG evaluation library. It provides the RAG Triad and integrates natively with LangChain and LlamaIndex.

**Where RAGAS wins:**
- Zero-friction setup for LangChain/LlamaIndex users
- Large community and documentation
- Reference-free evaluation (no ground truth required for most metrics)
- `EvaluationDataset` API is clean

**Where RAG Benchmarking adds:**
- Retrieval metrics (Precision@K, Recall@K, MRR, NDCG) — not in RAGAS
- Agentic metrics for tool-using agents — not in RAGAS
- REST API for language-agnostic integration
- Run comparison and history
- EU AI Act positioning

**Notes:** RAG Benchmarking uses RAGAS internally for the classic metrics (faithfulness, answer_relevancy, context_precision, context_recall). It is a superset, not a replacement.

---

### TruLens

TruLens is an observability and evaluation platform from Truera. It provides real-time monitoring with a local dashboard.

**Where TruLens wins:**
- Visual dashboard for continuous monitoring
- Instrumentation at the RAG chain level (traces)
- Good for production monitoring, not just offline benchmarking

**Where RAG Benchmarking adds:**
- Lighter dependency footprint
- REST API for CI/CD integration
- Agentic metrics

---

### DeepEval

DeepEval is a testing framework with pytest integration and a large metric library.

**Where DeepEval wins:**
- pytest-native — familiar to engineers who write tests
- Large set of predefined metrics
- G-Eval for custom metric definition
- Confident AI dashboard integration

**Where RAG Benchmarking adds:**
- Framework-agnostic server API
- Agentic-era metrics
- Simpler setup (no cloud account required)
- EU AI Act compliance framing

---

## Metric Coverage Comparison

| Metric | RAG Benchmarking | RAGAS | TruLens | DeepEval |
|---|---|---|---|---|
| Faithfulness | ✓ (claim-level) | ✓ | ✓ | ✓ |
| Answer Relevancy | ✓ | ✓ | ✓ | ✓ |
| Context Precision | ✓ | ✓ | ✓ | ✓ |
| Context Recall | ✓ | ✓ | ✓ | ✓ |
| Precision@K | ✓ | ✗ | ✗ | Partial |
| Recall@K | ✓ | ✗ | ✗ | Partial |
| MRR | ✓ | ✗ | ✗ | ✗ |
| NDCG | ✓ | ✗ | ✗ | ✗ |
| Agent Faithfulness | ✓ | ✗ | ✗ | Partial |
| Tool Call Accuracy | ✓ | ✓ (per docs.ragas.io) | ✗ | ✓ |
| Source Attribution | ✓ (deterministic) | ✗ | ✗ | ✗ |
| Retrieval Necessity | ✓ | ✗ | ✗ | ✗ |
| Custom metrics | Via plugin | Via custom metrics | Via feedback functions | Via G-Eval |
