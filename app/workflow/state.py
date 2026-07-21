"""
LangGraph state definition — the shared state object passed between all nodes.
Uses TypedDict for compatibility with LangGraph's state channels.
"""
from __future__ import annotations

from typing import TypedDict

from app.schemas.models import (
    FinalResponse,
    GraphResult,
    ReasoningStep,
    RoutingDecision,
    VectorResult,
)


class AgentStateDict(TypedDict, total=False):
    """
    LangGraph state — partial (total=False) so nodes only update relevant fields.

    Flow:
      START
        → router_node      (adds: routing_decision, reasoning_steps[0])
        → graph_node       (adds: graph_result, reasoning_steps[1])    ← conditional
          OR
        → vector_node      (adds: vector_result, reasoning_steps[1])   ← conditional
        → synthesizer_node (adds: reasoning_steps[2], partial answer data)
        → guardrail_node   (adds: final_response)
      END
    """

    query: str
    forced_mode: str
    routing_decision: RoutingDecision
    graph_result: GraphResult
    vector_result: VectorResult
    reasoning_steps: list[ReasoningStep]
    final_response: FinalResponse
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    total_latency_ms: float
    use_text2cypher: bool
    # Transient field: set by synthesizer_node, consumed by guardrail_node
    _synthesized_answer: str
