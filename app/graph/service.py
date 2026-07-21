"""
GraphService — high-level orchestration of graph retrieval.
Agents call this; it delegates to GraphRetriever and enriches the result.
"""
from __future__ import annotations

import logging

from app.graph.client import Neo4jClient
from app.graph.retriever import GraphRetriever
from app.schemas.models import GraphResult

logger = logging.getLogger(__name__)


class GraphService:
    """Thin service layer wrapping GraphRetriever with logging and error handling."""

    def __init__(self, client: Neo4jClient | None = None) -> None:
        self._retriever = GraphRetriever(client=client)

    async def query(self, user_query: str, use_text2cypher: bool = False) -> GraphResult:
        """
        Execute a graph retrieval for the given user query.

        Args:
            user_query: Natural language question from the user.
            use_text2cypher: If True, use LLM-generated Cypher (bonus path).

        Returns:
            Validated GraphResult.
        """
        logger.info("GraphService.query called | use_t2c=%s | query=%.100s", use_text2cypher, user_query)
        try:
            if use_text2cypher:
                result = await self._retriever.retrieve_text2cypher(user_query)
            else:
                result = await self._retriever.retrieve(user_query)

            logger.info(
                "GraphService.query success | entities=%d | latency=%.1fms",
                len(result.entities),
                result.latency_ms,
            )
            return result

        except Exception as exc:
            logger.error("GraphService.query failed: %s", exc, exc_info=True)
            return GraphResult(
                query=user_query,
                answer=f"Graph retrieval failed: {exc}",
                latency_ms=0.0,
            )
