# Deployment Guide

This guide covers how to deploy `rag-benchmarking` (v1.0.0+) as a self-hosted evaluation service. The CLI/SDK can be used without the server; this guide is specifically for the FastAPI service that exposes `/v1/evaluate`, `/v1/evaluate/agent`, `/v1/runs`, and `/v1/runs/compare`.

## Prerequisites

- **Docker & Docker Compose**: required for containerized deployment.
- **Qdrant Cloud Account** *(optional)*: only needed if you wire your own retriever in front of the harness. The eval API itself is retriever-agnostic.
- **LLM Provider API Key**: Gemini (recommended) or OpenAI, used by RAGAS-backed metrics (`faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`) and the LLM-judge agentic metrics. Deterministic metrics (retrieval + `source_attribution_accuracy`) need no LLM key.

## Configuration

The application is configured via environment variables. Create a `.env` file in the root directory (use `.env.example` as a template).

### Critical Settings

| Variable | Description | Required | Default |
| :--- | :--- | :--- | :--- |
| `API_KEY` | Master API Key for `/v1/*` endpoints. | **Yes** for production | `None` (open-mode, dev only) |
| `ENFORCE_API_KEY` | When `true`, requests without `X-API-Key` are rejected. | Recommended `true` | `false` |
| `GEMINI_API_KEY` | Google Gemini API Key. | Yes (if using Gemini judge) | - |
| `OPENAI_API_KEY` | OpenAI API Key. | Yes (if using OpenAI judge) | - |
| `HOST_PORT` | Host-side port for the docker-compose service. Container internal port stays `5000`. | No | `5001` |

### Tuning Settings

| Variable | Description | Default |
| :--- | :--- | :--- |
| `LOG_LEVEL` | Logging verbosity (DEBUG, INFO, WARNING, ERROR). | `INFO` |

## Deployment Options

### 1. Docker Compose (recommended)

```bash
docker compose up -d --build
```

The API will be available at `http://localhost:5001`. (Container listens on internal `:5000`; the compose file maps it to host `:5001` by default — override with `HOST_PORT=…` if needed.)

```bash
curl http://localhost:5001/health
docker compose logs -f rag-api
```

### 2. Kubernetes

Use the project `Dockerfile`. Mount API keys via a `Secret`, expose via a `Service`/`Ingress`. The container listens on `5000` internally.

```yaml
env:
  - name: API_KEY
    valueFrom:
      secretKeyRef:
        name: rag-benchmarking-secrets
        key: api-key
  - name: ENFORCE_API_KEY
    value: "true"
  - name: GEMINI_API_KEY
    valueFrom:
      secretKeyRef:
        name: rag-benchmarking-secrets
        key: gemini-api-key
```

## Security

> [!IMPORTANT]
> This application uses a simple `X-API-Key` header for authentication. There is no built-in user model.

- All clients **must** send `X-API-Key` with every request to `/v1/evaluate`, `/v1/evaluate/agent`, `/v1/runs`, and `/v1/runs/compare` when `ENFORCE_API_KEY=true`.
- **Key rotation**: update the `API_KEY` env var and restart the service. Old keys are invalidated immediately.
- The harness makes no outbound calls except to your configured LLM judge provider — see `SECURITY.md` for the full data-flow.

## Observability

- **Tracing**: every response carries an `X-Trace-Id` header. Include it in bug reports.
- **Logs**: JSON to stdout. Wire your log collector (Fluentd, Datadog Agent, Vector) to parse the JSON lines.
- **Health**: `GET /health` returns `200 OK` once the service is ready to take traffic.

## Versioning + upgrade

- Releases follow [Semantic Versioning](https://semver.org/). Breaking changes bump the major version; metric-definition changes are documented in `CHANGELOG.md` so historical run comparisons remain interpretable.
- The `eval_results.db` SQLite file is created on first run at the project root by default and is gitignored. Back it up before upgrading if you depend on cross-version run history.
