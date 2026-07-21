"""
GraphAgent — wraps GraphService and records a ReAct reasoning step.
"""
from __future__ import annotations

import logging
import time

from app.graph.service import GraphService
from app.schemas.models import GraphResult, GraphResultType, ReasoningStep

logger = logging.getLogger(__name__)


class GraphAgent:
    """Agent that performs knowledge graph retrieval."""

    def __init__(self) -> None:
        self._service = GraphService()

    async def run(self, query: str, use_text2cypher: bool = False) -> tuple[GraphResult, ReasoningStep]:
        """
        Execute graph retrieval and return result + reasoning step.
        """
        t0 = time.perf_counter()
        logger.info("GraphAgent.run | query=%.80s | use_t2c=%s", query, use_text2cypher)

        result = await self._service.query(query, use_text2cypher=use_text2cypher)

        latency_ms = (time.perf_counter() - t0) * 1000


        is_summary = getattr(result, "result_type", None) == GraphResultType.SUMMARY
        action_name = "graph_aggregate" if is_summary else "graph_retrieve"
        
        intent_str = getattr(result, "intent", None)
        root_ent = getattr(result, "root_entity", None)
        depth = getattr(result, "depth_hops", None)
        
        if intent_str:
            thought_text = (
                f"Detected intent '{intent_str}' targeting root entity '{root_ent or 'unknown'}'. "
                f"Executing Cypher path traversal (max depth: {depth or 'N/A'} hops) to find connected nodes."
            )
        elif is_summary:
            thought_text = (
                f"Query requires analytical aggregation over graph relationships. "
                f"Executing parameterized Cypher aggregation query."
            )
        else:
            thought_text = (
                f"Query requires multi-hop relationship traversal. "
                f"Executing parameterized Cypher query to retrieve entities connected to the search terms."
            )

        step = ReasoningStep(
            thought=thought_text,
            action=action_name,
            observation=(
                f"GraphRAG returned {len(result.entities)} entities across {len(result.paths)} paths "
                f"in {latency_ms:.1f}ms. Answer synthesized from graph context."
            ),
        )

        logger.info(
            "GraphAgent complete | entities=%d | latency=%.1fms",
            len(result.entities),
            latency_ms,
        )
        return result, step
