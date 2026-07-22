"""
LangGraph StateGraph — the multi-agent orchestration pipeline.

Architecture:
                    Supervisor (START)
                          │
                    Router Node
                    (LLM decides)
                   /             \\
           Graph Node        Vector Node
           (Neo4j)           (FAISS)
                   \\             /
                Synthesizer Node
                (asyncio.gather)
                          │
                  Guardrail Node
                  (Pydantic v2)
                          │
                         END
"""
from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from app.agents.graph_agent import GraphAgent
from app.agents.guardrail import GuardrailAgent
from app.agents.router import RouterAgent
from app.agents.synthesizer import SynthesizerAgent
from app.agents.vector_agent import VectorAgent
from app.schemas.models import FinalResponse, RoutingDecision, ToolChoice, ReasoningStep
from app.workflow.state import AgentStateDict

logger = logging.getLogger(__name__)

# ── Instantiate agents (shared across invocations) ─────────────────────────────
_router = RouterAgent()
_graph_agent = GraphAgent()
_vector_agent = VectorAgent()
_synthesizer = SynthesizerAgent()
_guardrail = GuardrailAgent()


# ── Node functions ─────────────────────────────────────────────────────────────

async def router_node(state: AgentStateDict) -> AgentStateDict:
    """Route the query to graph or vector based on LLM reasoning or user forced mode."""
    query = state["query"]
    steps = list(state.get("reasoning_steps", []))
    forced_mode = state.get("forced_mode")

    # Run the router LLM to check if the query is out-of-domain and should be rejected
    decision, step = await _router.route(query)

    if decision.tool == ToolChoice.REJECT:
        # Out-of-domain query is globally rejected regardless of forced modes
        logger.info("[router_node] Global OOD reject triggered for query: %.80s", query)
    else:
        # If query is in-domain and the user forced a specific mode, override the routing
        if forced_mode in ["graph", "vector"]:
            tool = ToolChoice.GRAPH if forced_mode == "graph" else ToolChoice.VECTOR
            decision = RoutingDecision(
                tool=tool,
                confidence=1.0,
                reason=f"Forced selection of {forced_mode} by user."
            )
            step = ReasoningStep(
                thought=f"User forced execution using {forced_mode}.",
                action="route_query",
                observation=f"Routed directly to {forced_mode}."
            )

    steps.append(step)

    logger.info("[router_node] -> %s (confidence=%.2f)", decision.tool.value, decision.confidence)
    return {
        **state,
        "routing_decision": decision,
        "reasoning_steps": steps,
    }


async def graph_node(state: AgentStateDict) -> AgentStateDict:
    """Execute graph retrieval via Neo4j."""
    query = state["query"]
    use_t2c = state.get("use_text2cypher", False)
    steps = list(state.get("reasoning_steps", []))
    tokens = state.get("total_tokens", 0)
    latency = state.get("total_latency_ms", 0.0)

    result, step = await _graph_agent.run(query, use_text2cypher=use_t2c)
    steps.append(step)

    logger.info("[graph_node] entities=%d", len(result.entities))
    return {
        **state,
        "graph_result": result,
        "reasoning_steps": steps,
        "total_tokens": tokens + result.tokens_used,
        "total_latency_ms": latency + result.latency_ms,
    }


async def vector_node(state: AgentStateDict) -> AgentStateDict:
    """Execute vector similarity search via FAISS."""
    query = state["query"]
    steps = list(state.get("reasoning_steps", []))
    tokens = state.get("total_tokens", 0)
    latency = state.get("total_latency_ms", 0.0)

    result, step = await _vector_agent.run(query)
    steps.append(step)

    logger.info("[vector_node] chunks=%d", len(result.chunks))
    return {
        **state,
        "vector_result": result,
        "reasoning_steps": steps,
        "total_tokens": tokens + result.tokens_used,
        "total_latency_ms": latency + result.latency_ms,
    }


async def synthesizer_node(state: AgentStateDict) -> AgentStateDict:
    """Synthesize the final answer from whichever retrieval result is available."""
    query = state["query"]
    decision: RoutingDecision = state["routing_decision"]
    steps = list(state.get("reasoning_steps", []))
    tokens = state.get("total_tokens", 0)
    latency = state.get("total_latency_ms", 0.0)

    answer, tok, prompt_tok, completion_tok, lat, step = await _synthesizer.synthesize(
        query=query,
        routing_decision=decision,
        graph_result=state.get("graph_result"),
        vector_result=state.get("vector_result"),
    )
    steps.append(step)

    logger.info("[synthesizer_node] answer_len=%d tokens=%d (prompt=%d, completion=%d)", len(answer), tok, prompt_tok, completion_tok)
    return {
        **state,
        "_synthesized_answer": answer,
        "reasoning_steps": steps,
        "total_tokens": tokens + tok,
        "prompt_tokens": state.get("prompt_tokens", 0) + prompt_tok,
        "completion_tokens": state.get("completion_tokens", 0) + completion_tok,
        "total_latency_ms": latency + lat,
    }


async def guardrail_node(state: AgentStateDict) -> AgentStateDict:
    """Validate the answer via two-layer guardrail (content + Pydantic schema).

    validate() always returns a structured FinalResponse — never raises —
    so LangGraph cannot intercept an unhandled node exception.
    rejected=True means a guardrail layer caught a bad answer and substituted
    a safe fallback with confidence=0.0.
    """
    query = state["query"]
    decision: RoutingDecision = state["routing_decision"]
    steps = list(state.get("reasoning_steps", []))
    answer = state.get("_synthesized_answer", "")  # type: ignore[arg-type]

    final_response, step, rejected = await _guardrail.validate(
        query=query,
        answer=answer,
        routing_decision=decision,
        reasoning_steps=steps,
        tokens_used=state.get("total_tokens", 0),
        prompt_tokens=state.get("prompt_tokens", 0),
        completion_tokens=state.get("completion_tokens", 0),
        latency_ms=state.get("total_latency_ms", 0.0),
        graph_result=state.get("graph_result"),
        vector_result=state.get("vector_result"),
    )
    steps.append(step)

    if rejected:
        logger.warning("[guardrail_node] REJECTED — returning fallback response | reason=%s", step.observation)
    else:
        logger.info("[guardrail_node] PASSED tool=%s", final_response.tool_used.value)

    return {
        **state,
        "final_response": final_response,
        "reasoning_steps": steps,
    }




async def reject_node(state: AgentStateDict) -> AgentStateDict:
    """Handle out-of-domain queries by refusing them immediately."""
    query = state["query"]
    decision = state["routing_decision"]
    steps = list(state.get("reasoning_steps", []))
    
    refusal_answer = (
        "I am designed to assist only with manufacturing quality control, "
        "equipment maintenance, and supplier risk analysis for the Chakan assembly plant. "
        "Your query is outside this domain."
    )
    
    # Create final response directly
    final_response = FinalResponse(
        query=query,
        tool_used=ToolChoice.REJECT,
        answer=refusal_answer,
        reasoning=steps,
        confidence=decision.confidence,
        tokens_used=0,
        prompt_tokens=0,
        completion_tokens=0,
        latency_ms=0.0,
    )
    
    return {
        **state,
        "final_response": final_response,
    }


# ── Conditional edge ───────────────────────────────────────────────────────────

def route_after_router(state: AgentStateDict) -> str:
    """LangGraph conditional edge: branch to graph_node, vector_node, or reject_node."""
    decision: RoutingDecision = state["routing_decision"]
    if decision.tool == ToolChoice.GRAPH:
        return "graph_node"
    elif decision.tool == ToolChoice.REJECT:
        return "reject_node"
    return "vector_node"


# ── Build the graph ────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Construct and compile the LangGraph StateGraph."""
    builder = StateGraph(AgentStateDict)

    # Register nodes
    builder.add_node("router_node", router_node)
    builder.add_node("graph_node", graph_node)
    builder.add_node("vector_node", vector_node)
    builder.add_node("reject_node", reject_node)
    builder.add_node("synthesizer_node", synthesizer_node)
    builder.add_node("guardrail_node", guardrail_node)

    # Edges
    builder.add_edge(START, "router_node")

    # Conditional: router → graph OR vector OR reject
    builder.add_conditional_edges(
        "router_node",
        route_after_router,
        {
            "graph_node": "graph_node",
            "vector_node": "vector_node",
            "reject_node": "reject_node",
        },
    )

    # Both retrieval paths merge into synthesizer
    builder.add_edge("graph_node", "synthesizer_node")
    builder.add_edge("vector_node", "synthesizer_node")

    # reject_node routes directly to END
    builder.add_edge("reject_node", END)

    # Synthesizer → guardrail → END
    builder.add_edge("synthesizer_node", "guardrail_node")
    builder.add_edge("guardrail_node", END)

    return builder.compile()


# Module-level compiled graph
agent_graph = build_graph()


async def run_pipeline(
    query: str,
    forced_mode: str | None = None,
    use_text2cypher: bool = False,
) -> FinalResponse:
    """
    Entry point for the full multi-agent pipeline.

    Args:
        query: Natural language question from the user.
        forced_mode: Optional forced routing mode ('graph' or 'vector').
        use_text2cypher: If True, uses dynamic Text-to-Cypher for graph retrieval.

    Returns:
        Validated FinalResponse.
    """
    initial_state: AgentStateDict = {
        "query": query,
        "reasoning_steps": [],
        "total_tokens": 0,
        "total_latency_ms": 0.0,
        "use_text2cypher": use_text2cypher,
    }
    if forced_mode:
        initial_state["forced_mode"] = forced_mode

    logger.info("Pipeline START | query=%.100s | forced_mode=%s", query, forced_mode)
    final_state = await agent_graph.ainvoke(initial_state)
    response: FinalResponse = final_state["final_response"]
    logger.info(
        "Pipeline END | tool=%s | tokens=%d | latency=%.1fms",
        response.tool_used.value,
        response.tokens_used,
        response.latency_ms,
    )
    return response
