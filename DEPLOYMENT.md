# Deployment Guide

This guide covers how to deploy the Agentic RAG Benchmarking POC in a production-like environment.

## Prerequisites

- **Docker & Docker Compose**: Required for containerized deployment.
- **Qdrant Cloud Account**: You need a Qdrant Cloud cluster (or local instance) and an API Key.
- **LLM Provider API Key**: Gemini (recommended) or OpenAI API Key.

## Configuration

The application is configured via environment variables. Create a `.env` file in the root directory (use `.env.example` as a template).

### Critical Settings

| Variable | Description | Required | Default |
| :--- | :--- | :--- | :--- |
| `API_KEY` | **Security**: Master API Key for accessing endpoints. | **Yes** | `None` (Open Mode - Dev Only) |
| `QDRANT_URL` | URL of your Qdrant instance. | **Yes** | - |
| `QDRANT_API_KEY` | API Key for Qdrant. | **Yes** | - |
| `GEMINI_API_KEY` | Google Gemini API Key. | Yes (if using Gemini) | - |
| `OPENAI_API_KEY` | OpenAI API Key. | Yes (if using OpenAI) | - |

### Tuning Settings

| Variable | Description | Default |
| :--- | :--- | :--- |
| `SELF_CHECK_MIN_GROUNDEDNESS` | Threshold (0.0-1.0) for retrying generation. | `0.7` |
| `SELF_CHECK_RETRY` | Enable/Disable retry logic. | `True` |
| `LOG_LEVEL` | Logging verbosity (DEBUG, INFO, WARNING, ERROR). | `INFO` |

## Deployment Options

### 1. Docker Compose (Recommended)

This is the standard way to run the service.

1.  **Build and Run**:
    ```bash
    docker compose up -d --build
    ```

2.  **Verify Status**:
    The API will be available at `http://localhost:5001`.
    ```bash
    curl http://localhost:5001/health
    ```

3.  **Access Logs**:
    ```bash
    docker compose logs -f app
    ```

### 2. Kubernetes (K8s)

For Kubernetes, use the generated `Dockerfile`.

1.  **Build Image**:
    ```bash
    docker build -t agentic-rag-benchmarking:latest .
    ```

2.  **Deploy**:
    Create a Deployment and Service manifest. Ensure you map the `.env` variables to a Kubernetes `Secret`.

    ```yaml
    env:
      - name: API_KEY
        valueFrom:
          secretKeyRef:
            name: rag-secrets
            key: api-key
      ...
    ```

## Security

> [!IMPORTANT]
> This application uses a simple API Key header for authentication.

- All clients **MUST** send the `X-API-Key` header with every request to `/v1/query` and `/v1/evaluate`.
- **Rotations**: To rotate the key, update the `API_KEY` environment variable and restart the service.

## Observability

- **Tracing**: The application adds an `X-Trace-Id` header to every response. Include this ID in bug reports.
- **Logs**: Logs are output in JSON format to `stdout`. Configure your log collector (e.g., Fluentd, Datadog Agent) to parse these JSON lines.
