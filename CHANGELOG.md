# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.1] - 2026-05-10

Cleanup release. Closes the three non-blocking items the v1.0.0 sign-off
round flagged (`docs/audit/T-N4-readiness-2026-05-10.md` §11.3 G6-A, G6-B
and §11.1 RB-NEW3) before they accrue into a larger release. Same-day as
v1.0.0 because the items are mechanical.

### ⚠️ Breaking change (one)

- **`/v1/evaluate/agent` response payload key renamed `scores` → `metrics`.**
  This brings the agent endpoint into shape-parity with `/v1/evaluate`
  (which has always emitted `metrics`). The RB-NEW3 sibling-endpoint
  contract drift caught in the v1.0.0 Authenticity Shield re-review was
  deferred to v1.0.1 with the explicit promise of a loud CHANGELOG note;
  this is that note. **Migration:** if your code reads `r["scores"]`
  from the agent endpoint, change it to `r["metrics"]`. The shape of
  the inner dict (per-metric score floats) is unchanged.

  Affected callers known at release time:
  - `tests/integration/test_sdk.py` — fixed in this release
  - `tests/e2e/test_api_harness.py` — fixed in this release
  - `docs/ci-cd-guide.md` examples — fixed in this release
  - `src/app/sdk/client.py` module docstring example — fixed in this release

### Changed

- **G6-A (SDK docstring):** `RagEval.evaluate()` docstring at
  `src/app/sdk/client.py:60` now documents the canonical
  `ground_truths: list[str]` plural-list shape and explicitly notes
  the legacy singular `ground_truth=` kwarg is still accepted via the
  backward-compat shim. The runtime behaviour was already correct in
  v1.0.0; this fixes the documented contract.
- **G6-B (built-in RAG pipeline default):** `LLMClient.gemini_model`
  fallback at `src/app/llm/client.py:52` flipped from the deprecated
  `gemini-1.5-flash` to `gemini-2.5-flash`, aligning with the eval
  harness judge default. Affects only the optional built-in RAG
  pipeline; the eval harness was already on 2.5-flash. Override via
  the `GEMINI_MODEL` env var as before.

### Verification

- 80 unit tests + 3 schema regression tests pass (the v1.0.0 shim tests
  for `EvalSample` and `AgentTrace` legacy-kwarg promotion still hold).
- Test-mock + e2e tests updated to assert against the new `metrics`
  key on the agent endpoint.

## [1.0.0] - 2026-05-10

First Production/Stable release. Closes the 2026-05-10 Top-N=4 launch readiness audit (Path A: narrowed scope + authenticity-pass cleanup). Full action list at `docs/audit/T-N4-readiness-2026-05-10.md` in the aiexponent monorepo. G6 Responsible AI sign-off pending re-review against this tag.

### Repositioned (regulatory framing)

- **Article 15 framing narrowed.** README + docs no longer assert "audit-ready evidence for Article 15 compliance" — Art. 15 is a tripartite obligation (accuracy + robustness + cybersecurity) and this harness covers only Art. 15(1) accuracy + faithfulness. The cybersecurity scope disclaimer (added in v0.2.x) is now matched by an equally explicit robustness scope disclaimer.
- **Comparison table updated.** RAGAS gets a "Partial" for agentic metrics (per `docs.ragas.io` Tool call Accuracy) — was incorrectly "No". Promptfoo column added.

### Fixed (production-affecting)

- **Schema reconciliation (RB-H5).** `EvalSample.ground_truths` and `AgentTrace.ground_truths` are now plural-list (Ragas convention), matching the runner, the API request body, the golden dataset, and the methodology docs. The legacy singular `ground_truth=` kwarg still works at construction via a backward-compat shim, and the `.ground_truth` property is preserved as a read-only first-element accessor. **Pasting the documented sample shape no longer raises a Pydantic validation error.**
- **`METRIC_GROUPS["full"]`** now actually includes every metric group (was missing `context_precision`, `context_recall`, and the entire `agentic_v2` set per audit RB-M9).
- **`MetricGroup.classic`** now includes `context_precision` + `context_recall` to match the canonical Ragas grouping.

### Changed

- `pyproject.toml` version `1.0.0-rc1` → `1.0.0`; classifier `Beta` → `Production/Stable`.
- `RunConfig.judge_model` default `gemini-1.5-flash` → `gemini-2.5-flash`. Headline 0.958 / 0.810 figures re-validated against 2.5-flash. Cross-judge absolute comparisons remain unreliable; pin the same judge across runs you compare.
- `LICENSE` replaced with the verbatim Apache-2.0 SPDX template; copyright moved to a `NOTICE` file per Apache 2.0 §4(d).
- `publish-pypi.yml` hardened to RiskForge parity: CycloneDX SBOM at build, Sigstore build-provenance attestation, GitHub Release auto-created with wheel + sdist + sbom artefacts.
- `docs/comparison.md`: Ragas tool_call_accuracy ✗ → ✓ with citation; Promptfoo column added; EU AI Act framing row downgraded to "Partial Art. 15(1) accuracy input".
- `docs/benchmark-results.md`: retrieval-metric score table removed (audit RB-H3 — scores were retriever-config-dependent, not reproducible); replaced with a `Repro:` command + honest "indicative" note.
- `docs/dataset-methodology.md`: judge model citations 1.5 → 2.5; "cybersecurity" dropped from the positioning sentence.
- `DEPLOYMENT.md`: rewritten for v1.0; "POC" framing dropped; legacy `/v1/query` reference removed; `HOST_PORT` default documented as 5001.
- `docker-compose.yml`: default host port `5000` → `5001` (container internal port stays 5000).
- `docs/ci-cd-guide.md`: port references aligned to 5001.
- `SECURITY.md`: SQLite path corrected; supported-versions table updated for 1.0.x.
- `tool.ruff` line-length 100 → 120 with audit-comment; full `ruff format` applied (23 files, no logic changes).
- `.github/workflows/ci.yml`: ruff check + ruff format + mypy added (mypy non-blocking for v1.0; flips to required at v1.1).
- `README.md`: `report["scores"]` → `report["metrics"]` (matches actual return shape); Article 15 section reframed honestly.

### Removed

- Pre-existing committed `eval_results.db` artefact untracked + gitignored.

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
