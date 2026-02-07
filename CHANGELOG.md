# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
