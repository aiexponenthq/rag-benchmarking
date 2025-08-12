# Agentic RAG Benchmarking POC

This tutorial-style repository teaches you how to build, evaluate, and productionize an Agentic RAG system using open-source tools on a Mac (CPU) with cloud backends. It complements and operationalizes the practices in the article: [The Complete Enterprise Guide to RAG Evaluation and Benchmarking](https://aiexponent.com/the-complete-enterprise-guide-to-rag-evaluation-and-benchmarking/).

## Goals
- Agentic RAG pipeline (analyze → retrieve → rerank → synthesize → self-check → cite)
- Cloud-first defaults (hosted LLMs, Qdrant Cloud), local fallbacks (Ollama, local Qdrant)
- Rigorous evaluation: RAG Triad (Context Relevance, Faithfulness/Groundedness, Answer Relevance) and retrieval metrics (Precision@k, Recall@k, MRR, NDCG)
- FastAPI service with `/query`, `/ingest`, `/evaluate`, `/health`, `/metrics` on internal port 5000
- Docker-first packaging; local dev via conda on macOS
- TDD and CI gating

## Scope & Objectives

By the end of this tutorial you will be able to:
- Understand core RAG quality dimensions (Context Relevance, Faithfulness/Groundedness, Answer Relevance)
- Stand up a production-like RAG service on your Mac, using cloud LLMs and Qdrant Cloud for storage
- Ingest, retrieve, rerank, synthesize, and self-check groundedness
- Evaluate RAG quality using RAGAS with a Gemini judge and export reports
- Extend the system with agentic behaviors (retry on low groundedness, stricter guardrails)

## Technical Architecture (high level)
- API service: FastAPI exposing `/v1/query`, `/v1/evaluate`, `/health`
- Retrieval: sentence-transformers (BGE-base) + Qdrant Cloud
- Reranking (optional): BGE reranker cross-encoder
- Generation: Gemini (recommended) or OpenAI; pluggable
- Self-check: LLM-as-judge groundedness scoring with optional retry
- Evaluation: RAGAS with Gemini judge; JSON/Markdown report
- Packaging: Docker compose (internal port 5000), conda for local dev

## Local development
- Create a conda environment on macOS at your preferred path (e.g., `~/conda_envs/rag_agentic`).
- Install dependencies (listed in `pyproject.toml`).
- Secrets and endpoints go in `.env` (use `.env.example` as a template).
- Run tests with `pytest`. Lint with `ruff`, format with `black`, type-check with `mypy`.

## Configuration (.env)
Minimal keys for POC:

- LLM (generation):
  - `LLM_PROVIDER=gemini` (recommended)
  - `GEMINI_API_KEY=...` (required for generation and RAGAS judge)
  - Optional OpenAI: `LLM_PROVIDER=openai`, `OPENAI_API_KEY=...`
- Vector DB (Qdrant Cloud):
  - `QDRANT_URL=https://<cluster>.gcp.cloud.qdrant.io:6333`
  - `QDRANT_API_KEY=...`
  - `QDRANT_COLLECTION=agentic_rag_poc`
- Self-check controls:
  - `SELF_CHECK_MIN_GROUNDEDNESS=0.7`
  - `SELF_CHECK_RETRY=true`

See `.env.example` for the full list.

## Run

Conda (dev):

```bash
conda activate rag_agentic
uvicorn app.main:app --host 0.0.0.0 --port 5000
```

Docker (cloud defaults):

```bash
HOST_PORT=5001 docker compose up -d
```

Health:

```bash
curl http://localhost:5001/health
```

## Ingest data (Retrieval v0)

```bash
conda activate rag_agentic
python -m app.retrieval.ingest_cli data/sample/guide.md
```

This embeds with BGE-base (CPU) and upserts to Qdrant Cloud. If the target collection uses a different vector schema, the CLI creates a sibling collection `agentic_rag_poc__content` and uses a named vector `content` for portability.

## Query API

`POST /v1/query`

Body:

```json
{
  "query": "What is RAG?",
  "top_k": 3,
  "rerank": true
}
```

Response (fields abbreviated):

```json
{
  "answer": "...",
  "citations": [{"text": "...", "source_id": "...", "chunk_index": 0, "score": 0.91}],
  "timings_ms": {"retrieve": 42.1, "rerank": 25.3, "generate": 210.5, "self_check": 180.2},
  "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
  "groundedness": 0.82
}
```

Notes:
- `rerank=true` enables cross-encoder reranking (`BAAI/bge-reranker-v2-m3`). If unavailable, endpoint falls back gracefully.
- If `groundedness < SELF_CHECK_MIN_GROUNDEDNESS` and `SELF_CHECK_RETRY=true`, the service retries with expanded context and adopts the improved result.

## Evaluation (RAGAS)

Scripted:

```bash
conda activate rag_agentic
source .env
python scripts/evaluate.py data/golden/qa.jsonl --metrics faithfulness answer_relevancy --out reports/ragas_report.json
```

API:

```bash
curl -X POST http://localhost:5001/v1/evaluate \
 -H 'Content-Type: application/json' \
 -d '{
  "samples":[{"question":"What does RAG combine?","contexts":["RAG combines retrieval and generation."],"answer":"RAG combines retrieval and generation.","ground_truths":["RAG combines retrieval and generation."]}],
  "metrics":["faithfulness","answer_relevancy"],
  "out_json":"reports/ragas_report.json",
  "out_md":"reports/ragas_report.md"
}'
```

The Gemini judge is configured by default via `GEMINI_API_KEY`.

## Project Structure

```
src/
  app/
    api/               # FastAPI routers (query, evaluate)
    config/            # Settings via pydantic-settings
    eval/              # RAGAS runner and reporting helper
    llm/               # LLM client (Gemini/OpenAI)
    logging/           # JSON logger configuration
    quality/           # self_check (LLM-as-judge groundedness)
    retrieval/         # chunking, embeddings, qdrant client, reranker, service
    main.py            # FastAPI app wiring
data/
  golden/              # small golden set for evaluation
  sample/              # tiny example doc for ingestion
scripts/
  evaluate.py          # CLI to run RAGAS and save reports
```

## Key Concepts (with references)

- RAG Triad (Context Relevance, Faithfulness/Groundedness, Answer Relevance): see the article’s definitions and enterprise targets in [AiExponent guide](https://aiexponent.com/the-complete-enterprise-guide-to-rag-evaluation-and-benchmarking/)
- RAGAS (reference-free evaluation): [RAGAS Docs](https://github.com/explodinggradients/ragas)
- Qdrant (vector DB): [Qdrant Docs](https://qdrant.tech/documentation/)
- Sentence-Transformers (embeddings): [ST Docs](https://www.sbert.net/)
- BGE Reranker (cross-encoder): [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)
- Gemini (Generation/Judge): [Gemini API](https://ai.google.dev/)
- FastAPI: [FastAPI Docs](https://fastapi.tiangolo.com/)

## Extensions & Ideas
- Planner + multi-hop retrieval (LangGraph)
- Domain allow-list & prompt-injection filters for tool usage
- Token/cost accounting per provider; Prometheus `/metrics`
- Add retrieval metrics (MRR, nDCG) and dashboards (TruLens)
- Compare variants: rerank on/off, K-sweep, different models; publish tables

## Tutorial Steps
1) Setup env and run the API (Conda or Docker)
2) Ingest sample docs into Qdrant Cloud
3) Query with and without reranking, inspect citations
4) Enable groundedness and observe retry behavior when low
5) Run evaluation on `data/golden/qa.jsonl`, produce JSON/MD report

## How to Run (summary)
See “Run”, “Ingest data”, “Query API”, and “Evaluation” sections above for exact commands.

## Licensing
Licensed under the Apache License, Version 2.0. See `LICENSE`.

## Acknowledgments
- Evaluation targets and methodology draw from the article: [The Complete Enterprise Guide to RAG Evaluation and Benchmarking](https://aiexponent.com/the-complete-enterprise-guide-to-rag-evaluation-and-benchmarking/).


