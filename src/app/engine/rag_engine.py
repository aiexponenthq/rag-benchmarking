from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

import app.llm.client as llm_client
import app.retrieval.service as retrieval_service
from app.config.settings import get_settings
from app.utils.timing import timer

logger = logging.getLogger(__name__)


class RetrievedChunk(BaseModel):
    text: str
    source_id: str
    chunk_index: int
    score: float


class RAGResult(BaseModel):
    answer: str
    citations: list[RetrievedChunk]
    timings: dict[str, float]
    groundedness: float | None = None


class RAGEngine:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.llm = llm_client.LLMClient()

    def query(self, query: str, top_k: int, rerank: bool) -> RAGResult:
        """
        Execute the full RAG pipeline including retrieval, reranking, generation, 
        and optional retry.
        """
        timings: dict[str, float] = {}

        # 1. Retrieve
        logger.info("Starting RAG query", extra={"query": query, "top_k": top_k, "rerank": rerank})
        with timer() as t_retr:
            chunks = retrieval_service.retrieve_top_chunks(query, top_k=max(top_k, 10))
        timings["retrieve"] = t_retr["elapsed_ms"]

        if not chunks:
            logger.warning("No chunks retrieved for query", extra={"query": query})
            return RAGResult(answer="", citations=[], timings=timings)

        logger.info("Retrieved chunks", extra={"count": len(chunks)})

        # 2. Rerank
        if rerank:
            try:
                from app.retrieval.reranker import CrossEncoderReranker

                with timer() as t_rr:
                    reranker = CrossEncoderReranker()
                    chunks = reranker.rerank(query, chunks, top_k=len(chunks))
                timings["rerank"] = t_rr["elapsed_ms"]
            except Exception:
                pass

        current_chunks = chunks[:top_k]

        # 3. Generate
        with timer() as t_gen:
            answer = self._call_llm(query, current_chunks)
        timings["generate"] = t_gen["elapsed_ms"]

        # 4. Self-Check
        groundedness = None
        try:
            from app.quality.self_check import compute_groundedness

            with timer() as t_sc:
                groundedness = compute_groundedness(
                    answer, [c.get("text", "") for c in current_chunks]
                )
            timings["self_check"] = t_sc["elapsed_ms"]
        except Exception:
            pass

        # 5. Retry if needed
        if (
            groundedness is not None
            and groundedness < self.settings.self_check_min_groundedness
            and self.settings.self_check_retry
        ):

            logger.info(
                "Groundedness below threshold, attempting retry",
                extra={
                    "groundedness": groundedness,
                    "threshold": self.settings.self_check_min_groundedness,
                },
            )
            try:
                # Retry logic
                retry_result = self._retry_workflow(query, top_k, rerank, groundedness)
                if retry_result:
                    logger.info(
                        "Retry successful, adopting new answer",
                        extra={"new_groundedness": retry_result["groundedness"]},
                    )
                    answer = retry_result["answer"]
                    groundedness = retry_result["groundedness"]
                    current_chunks = retry_result["chunks"]
                    timings.update(retry_result["timings"])
                else:
                    logger.info("Retry did not improve groundedness")
            except Exception as e:
                logger.error("Error during retry workflow", extra={"error": str(e)})
                pass

        citations = [
            RetrievedChunk(
                text=c.get("text", ""),
                source_id=c.get("source_id", ""),
                chunk_index=int(c.get("chunk_index", 0)),
                score=float(c.get("score", 0.0)),
            )
            for c in current_chunks
        ]

        return RAGResult(
            answer=answer, citations=citations, timings=timings, groundedness=groundedness
        )

    def _call_llm(self, query: str, chunks: list[dict[str, Any]]) -> str:
        context_blocks = "\n\n".join(
            [f"[source: {c.get('source_id','')}]\n{c.get('text','')}" for c in chunks]
        )
        user_prompt = self.settings.user_prompt_template.format(
            context_blocks=context_blocks, query=query
        )
        return self.llm.generate(self.settings.system_prompt, user_prompt)

    def _retry_workflow(
        self, query: str, top_k: int, rerank: bool, current_score: float
    ) -> dict[str, Any] | None:
        timings = {}

        # Expand retrieval
        with timer() as t_retr:
            more_chunks = retrieval_service.retrieve_top_chunks(query, top_k=20)
        timings["retrieve_retry"] = t_retr["elapsed_ms"]

        if rerank:
            try:
                from app.retrieval.reranker import CrossEncoderReranker

                with timer() as t_rr:
                    reranker = CrossEncoderReranker()
                    more_chunks = reranker.rerank(query, more_chunks, top_k=len(more_chunks))
                timings["rerank_retry"] = t_rr["elapsed_ms"]
            except Exception:
                pass

        more_chunks = more_chunks[:top_k]

        # Generate
        with timer() as t_gen:
            answer = self._call_llm(query, more_chunks)
        timings["generate_retry"] = t_gen["elapsed_ms"]

        # Check
        groundedness = None
        try:
            from app.quality.self_check import compute_groundedness

            with timer() as t_sc:
                groundedness = compute_groundedness(
                    answer, [c.get("text", "") for c in more_chunks]
                )
            timings["self_check_retry"] = t_sc["elapsed_ms"]
        except Exception:
            return None

        if groundedness is not None and groundedness >= current_score:
            return {
                "answer": answer,
                "groundedness": groundedness,
                "chunks": more_chunks,
                "timings": timings,
            }
        return None
