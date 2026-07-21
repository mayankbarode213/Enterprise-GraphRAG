"""
VectorRetriever — similarity search over FAISSVectorStore + LLM synthesis.

Intentionally limited to semantic similarity — this is what fails
on the canonical multi-hop breaking-point query.
"""
from __future__ import annotations

import logging
import time

from langchain_openai import ChatOpenAI

from app.vector.embedding import EmbeddingService
from app.vector.store import FAISSVectorStore
from app.schemas.models import Citation, VectorResult
from settings import settings

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 1536  # text-embedding-3-small


class VectorRetriever:
    """
    Async retriever: embed query → FAISS similarity search → LLM synthesis.
    """

    def __init__(self) -> None:
        self._embedder = EmbeddingService()
        self._llm = ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            temperature=0,
        )
        self._store = FAISSVectorStore(
            persist_dir=settings.vector_dir_path,
            dimension=EMBEDDING_DIM,
        )

    async def retrieve(self, query: str, k: int | None = None) -> VectorResult:
        """
        Embed query → top-k similarity search → LLM answer synthesis.

        Args:
            query: User's natural language question.
            k: Number of chunks to retrieve (defaults to settings.vector_top_k).

        Returns:
            Validated VectorResult.
        """
        top_k = k or settings.vector_top_k
        t0 = time.perf_counter()

        # Step 1: Embed the query
        query_embedding = await self._embedder.embed_one(query)

        # Step 2: FAISS similarity search
        if self._store.count() == 0:
            logger.warning("VectorRetriever: FAISS store is empty. Run ingest_vectors.py first.")
            return VectorResult(
                query=query,
                chunks=[],
                answer="Vector store is empty — please run scripts/ingest_vectors.py first to load documents.",
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            )

        search_results = self._store.search(query_embedding, k=top_k)
        latency_embed = (time.perf_counter() - t0) * 1000

        # Step 3: Build citations
        citations: list[Citation] = []
        for doc, score in search_results:
            # FAISS IndexFlatIP on L2-normalised vectors returns cosine similarity in [-1, 1]
            # Clip to [0, 1] for display
            similarity = max(0.0, min(1.0, score))
            citations.append(
                Citation(
                    source=doc.metadata.get("source", "unknown"),
                    content=doc.content,
                    relevance_score=round(similarity, 4),
                )
            )

        logger.info(
            "VectorRetriever found %d chunks | embed_latency=%.1fms",
            len(citations),
            latency_embed,
        )

        # Step 4: LLM synthesis from retrieved chunks
        context = "\n\n---\n\n".join(
            f"[Source: {c.source}]\n{c.content}" for c in citations
        )
        prompt = (
            f"Answer the following question using ONLY the provided context. "
            f"If the context does not contain enough information to fully answer the question, "
            f"say so explicitly — do not guess or hallucinate.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

        llm_response = await self._llm.ainvoke(prompt)
        answer_text = str(llm_response.content).strip()
        total_latency = (time.perf_counter() - t0) * 1000

        token_count = 0
        if hasattr(llm_response, "usage_metadata") and llm_response.usage_metadata:
            token_count = llm_response.usage_metadata.get("total_tokens", 0)
        elif hasattr(llm_response, "response_metadata") and llm_response.response_metadata:
            usage = llm_response.response_metadata.get("token_usage", {})
            token_count = usage.get("total_tokens", 0)

        result = VectorResult(
            query=query,
            chunks=citations,
            answer=answer_text,
            latency_ms=round(total_latency, 2),
            tokens_used=token_count,
        )
        logger.info(
            "VectorRetriever complete | latency=%.1fms | tokens=%d",
            total_latency,
            token_count,
        )
        return result
