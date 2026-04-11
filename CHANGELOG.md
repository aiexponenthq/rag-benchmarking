# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0-rc1] — 2026-04-08

### Breaking Changes
- Evaluation is now decoupled from the built-in RAG pipeline via `harness.RAGEvaluable` protocol. Existing `/v1/evaluate` endpoint is backward-compatible.

### Added
- `src/harness/` — framework-agnostic evaluation harness (Protocol, EvaluationRunner, ResultStore, schemas)
- `EvalSample`, `AgentTrace`, `BenchmarkReport`, `RunConfig` Pydantic v2 schemas
- `RAGEvaluable` Protocol — plug any RAG system in with one `run()` method
- Retrieval metrics: Precision@K, Recall@K, MRR, NDCG@K (`src/app/eval/retrieval_metrics.py`)
- Claim-decomposition faithfulness replaces single-rubric self_check (`src/app/eval/faithfulness.py`)
- Agentic metrics (deterministic): `source_attribution_accuracy` (`src/app/eval/agentic_metrics.py`)
- Agentic metrics (LLM-as-judge): `agent_faithfulness`, `tool_call_accuracy`, `retrieval_necessity` (`src/app/eval/agentic_llm_metrics.py`)
- `POST /v1/evaluate/agent` — agentic trace evaluation endpoint
- `GET /v1/runs` — list historical benchmark runs
- Harness-level `ResultStore` — SQLite persistence for `BenchmarkReport` with `compare_runs`
- Python SDK: `RagEval` client with LangChain and LlamaIndex adapters (`src/app/sdk/`)
- 50-sample golden dataset across 10 domains (`data/golden/qa.jsonl`)

### Fixed
- Token accounting now returns real usage from Gemini + OpenAI APIs (was hardcoded zeros)
- `context_precision` and `context_recall` enabled in RAGAS runner with ground_truth guard
- All API endpoints are async via `run_in_executor`
- Test suite expanded from 11 to 80+ tests
- Fixed `datetime.utcnow()` deprecation — replaced with `datetime.now(UTC)`

## [0.2.0-rc1] - 2026-02-07

### Added
- **Authentication**: Implemented API Key middleware. Clients must now provide `X-API-Key` header.
- **Observability**: Added correlation IDs (`X-Trace-Id`) to requests and logs.
- **Structured Logging**: Enhanced `RAGEngine` logs with event-specific metadata (retrieval counts, loop metrics).
- **Deployment Guide**: Comprehensive [deployment documentation](DEPLOYMENT.md).

### Changed
- **Refactor**: Extracted RAG logic from `api/query.py` into `src/app/engine/rag_engine.py`.
- **Error Handling**: Replaced generic 500 errors with specific status codes (503 for Service Unavailable).
- **Settings**: Externalized prompt templates to environment variables/settings.

### Security
- Added `API_KEY` validation for `/v1/query` and `/v1/evaluate` endpoints.

## [0.1.0] - 2026-01-30

### Added
- Initial POC release.
- Retrieval-Augmented Generation pipeline using Qdrant and Gemini/OpenAI.
- RAGAS evaluation suite.
- Docker Compose setup.
