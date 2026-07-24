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
    """Agent that performs knowledge graph retrieval with dynamic ReAct reasoning."""

    def __init__(self) -> None:
        self._service = GraphService()

    async def run(self, query: str, use_text2cypher: bool = False) -> tuple[GraphResult, ReasoningStep]:
        """
        Execute graph retrieval and return result + ReAct reasoning step.
        """
        t0 = time.perf_counter()
        logger.info("GraphAgent.run | query=%.80s | use_t2c=%s", query, use_text2cypher)

        # 1. Execute Graph Retrieval
        result = await self._service.query(query, use_text2cypher=use_text2cypher)
        latency_ms = (time.perf_counter() - t0) * 1000

        # Print detailed GraphPath objects to terminal for inspection
        print("\n" + "=" * 70)
        print(f"📊 [GRAPH RETRIEVAL RESULTS] Found {len(result.paths)} GraphPath(s) | {len(result.entities)} Entity(ies)")
        print("=" * 70)
        for idx, path in enumerate(result.paths, 1):
            nodes_fmt = " ──▶ ".join(f"{n.name} [{n.type}]" for n in path.nodes)
            rels_fmt = ", ".join(path.relationships)
            print(f"  Path #{idx} (length={path.path_length}):")
            print(f"    Nodes        : {nodes_fmt}")
            print(f"    Relationships: [{rels_fmt}]")
        print("=" * 70 + "\n")

        is_summary = getattr(result, "result_type", None) == GraphResultType.SUMMARY
        action_name = "graph_aggregate" if is_summary else "graph_retrieve"

        # 2. Extract Natural Query-Specific ReAct Thought
        entity_names = [e.name for e in result.entities[:4] if hasattr(e, "name") and e.name]
        entity_str = f" starting from '{entity_names[0]}'" if entity_names else ""
        entity_types = list({e.type for e in result.entities if hasattr(e, "type") and e.type})
        type_str = ", ".join(entity_types[:5]) if entity_types else "Nodes"
        intent_str = getattr(result, "intent", None)
        root_ent = getattr(result, "root_entity", None)
        depth = getattr(result, "depth_hops", None)

        if use_text2cypher:
            t2c_llm_reasoning = getattr(result, "t2c_thought", "")
            cypher_snippet = getattr(result, "cypher_used", "")
            
            if t2c_llm_reasoning:
                thought_text = f"{t2c_llm_reasoning} \n\n[Cypher: `{cypher_snippet}`]\n\n"
            else:
                thought_text = (
                    f"Generated dynamic Text-to-Cypher query: `{cypher_snippet or 'MATCH ...'}`. "
                    f"Executing LLM-crafted Cypher statement against Neo4j database."
                )
        elif intent_str:
            thought_text = (
                f"Executing multi-hop graph traversal{entity_str} across {depth or '6'} relationship hops "
                f"to trace causal lineage between supplier batches, maintenance events, and downstream defects."
            )
        elif is_summary:
            thought_text = (
                f"Aggregating quality statistics and defect frequency across suppliers and components."
            )
        else:
            thought_text = (
                f"Traversing graph neighborhood around query entities to discover multi-hop causal chains."
            )

        observation_text = (
            f"Retrieved {len(result.entities)} entities ({type_str}) across {len(result.paths)} causal paths "
            f"with {getattr(result, 'relationships_traversed', 0)} edge traversals in {latency_ms:.1f}ms."
        )

        step = ReasoningStep(
            thought=thought_text,
            action=action_name,
            observation=observation_text,
        )

        logger.info(
            "GraphAgent complete | entities=%d | latency=%.1fms",
            len(result.entities),
            latency_ms,
        )
        return result, step


