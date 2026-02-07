from __future__ import annotations

import platform
import time
import uuid
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from app import __version__
from app.api.evaluate import router as eval_router
from app.api.query import router as query_router
from app.api.security import get_api_key
from app.config.settings import get_settings
from app.exceptions import LLMError, RAGException, VectorDBError
from app.logging.json_logger import configure_json_logging, trace_id_var

app = FastAPI(title="Agentic RAG Benchmarking POC", version=__version__)
app.include_router(query_router, dependencies=[Depends(get_api_key)])
app.include_router(eval_router, dependencies=[Depends(get_api_key)])


@app.middleware("http")
async def trace_middleware(request: Request, call_next):
    trace_id = request.headers.get("X-Trace-Id") or str(uuid.uuid4())
    trace_id_var.set(trace_id)
    response = await call_next(request)
    response.headers["X-Trace-Id"] = trace_id
    return response


@app.exception_handler(VectorDBError)
async def vector_db_error_handler(request: Request, exc: VectorDBError):
    return JSONResponse(
        status_code=503,
        content={"detail": "Vector Database Service Unavailable", "error": str(exc)},
    )


@app.exception_handler(LLMError)
async def llm_error_handler(request: Request, exc: LLMError):
    return JSONResponse(
        status_code=503,
        content={"detail": "LLM Service Unavailable", "error": str(exc)},
    )


@app.exception_handler(RAGException)
async def rag_exception_handler(request: Request, exc: RAGException):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal RAG Error", "error": str(exc)},
    )


@app.on_event("startup")
def _startup() -> None:
    settings = get_settings()
    configure_json_logging(settings.log_level)


@app.get("/health")
def health() -> dict[str, Any]:
    settings = get_settings()
    # GPU status (Mac): unavailable by default for this POC
    gpu_status = {"available": False, "details": {"device": None}}
    model_status = {
        "providers": {
            "openai": bool(settings.openai_api_key),
            "gemini": bool(settings.gemini_api_key),
        },
        "loaded": False,
    }
    vectordb_status = {
        "provider": "qdrant",
        "configured": bool(settings.qdrant_url and settings.qdrant_api_key),
        "collection": settings.qdrant_collection,
    }
    return {
        "status": "healthy",
        "timestamp": int(time.time()),
        "version": __version__,
        "system": {"python": platform.python_version(), "platform": platform.platform()},
        "model": model_status,
        "gpu": gpu_status,
        "vectordb": vectordb_status,
        "last_successful_prediction_at": None,
    }
