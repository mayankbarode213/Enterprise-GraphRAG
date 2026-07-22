"""
VectorAgent — wraps VectorRetriever and records a ReAct reasoning step.
"""
from __future__ import annotations

import logging
import time

from app.vector.retriever import VectorRetriever
from app.schemas.models import ReasoningStep, VectorResult

logger = logging.getLogger(__name__)


class VectorAgent:
    """Agent that performs semantic similarity retrieval."""

    def __init__(self) -> None:
        self._retriever = VectorRetriever()

    async def run(self, query: str) -> tuple[VectorResult, ReasoningStep]:
        """
        Execute vector retrieval and return result + reasoning step.
        """
        t0 = time.perf_counter()
        logger.info("VectorAgent.run | query=%.80s", query)

        result = await self._retriever.retrieve(query)

        latency_ms = (time.perf_counter() - t0) * 1000

        top_score = result.chunks[0].relevance_score if result.chunks else 0.0
        sources = list(set(c.source for c in result.chunks))
        sources_clean = [s.replace(".txt", "").replace("_", " ").title() for s in sources[:3]]
        sources_formatted = ", ".join(sources_clean) if sources_clean else "unstructured plant records"

        thought_text = (
            f"Executing FAISS vector search to locate document passages for '{query}'. "
            f"Retrieved semantic context from {sources_formatted} (top relevance score: {top_score:.3f})."
        )

        step = ReasoningStep(
            thought=thought_text,
            action="vector_retrieve",
            observation=(
                f"VectorRAG retrieved {len(result.chunks)} chunks "
                f"(top relevance: {top_score:.3f}) "
                f"in {latency_ms:.1f}ms."
            ),
        )

        logger.info(
            "VectorAgent complete | chunks=%d | latency=%.1fms",
            len(result.chunks),
            latency_ms,
        )
        return result, step
