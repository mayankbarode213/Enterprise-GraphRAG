"""
Pydantic v2 schemas — the single source of truth for all data structures
flowing through the multi-agent pipeline.

Every LLM output is validated against one of these models before being
passed downstream, providing schema-level guardrails.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Enums ─────────────────────────────────────────────────────────────────────

class ToolChoice(str, Enum):
    GRAPH = "graph"
    VECTOR = "vector"
    REJECT = "reject"


class DefectSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ── Routing ───────────────────────────────────────────────────────────────────

class RoutingDecision(BaseModel):
    """LLM-produced routing decision validated before any downstream call."""

    tool: ToolChoice = Field(
        ...,
        description="Which retrieval tool to invoke: 'graph' or 'vector'",
    )
    reason: str = Field(
        ...,
        min_length=10,
        description="Human-readable explanation of why this tool was chosen",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Router confidence score between 0.0 and 1.0",
    )
    requires_multi_hop: bool = Field(
        default=False,
        description="True if the query requires multi-hop relationship traversal",
    )

    @field_validator("reason")
    @classmethod
    def reason_must_be_meaningful(cls, v: str) -> str:
        if v.strip().lower() in {"n/a", "none", ""}:
            raise ValueError("reason must be a meaningful explanation, not empty or 'n/a'")
        return v.strip()


# ── Entities ──────────────────────────────────────────────────────────────────

class Entity(BaseModel):
    """A node retrieved from the knowledge graph."""

    id: str = Field(..., description="Unique entity identifier (e.g. 'M001')")
    type: str = Field(..., description="Node label (e.g. 'Machine', 'Supplier')")
    name: str = Field(..., description="Human-readable entity name")
    properties: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional properties from the graph node",
    )


class GraphPath(BaseModel):
    """A single traversal path through the knowledge graph."""

    nodes: list[Entity] = Field(..., min_length=1)
    relationships: list[str] = Field(
        default_factory=list,
        description="Relationship types along the path (e.g. ['SUPPLIED_BATCH', 'MANUFACTURED'])",
    )
    path_length: int = Field(default=0)

    @model_validator(mode="after")
    def compute_path_length(self) -> "GraphPath":
        self.path_length = len(self.nodes)
        return self


# ── Citations ──────────────────────────────────────────────────────────────────

class Citation(BaseModel):
    """A single text chunk retrieved from the vector store."""

    source: str = Field(..., description="Document filename or identifier")
    content: str = Field(..., description="The retrieved chunk text")
    relevance_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Cosine similarity score from the vector store",
    )


class GraphOperation(str, Enum):
    TRAVERSAL = "traversal"                  # Multi-hop path traversal
    AGGREGATION = "aggregation"              # Analytical aggregation over relationships
    NEIGHBORHOOD = "neighborhood"            # Local entity neighborhood extraction
    SHORTEST_PATH = "shortest_path"          # Algorithmic path search
    DEPENDENCY_ANALYSIS = "dependency_analysis"  # Upstream/downstream dependency graph
    SUBGRAPH = "subgraph"                    # Subgraph extraction


class GraphResultType(str, Enum):
    LINEAGE = "lineage"          # Multi-hop path traversal / root-cause lineage tree
    SUMMARY = "summary"          # Aggregation / grouped summary report
    NEIGHBORHOOD = "neighborhood"# Entity neighborhood / list


# ── Retrieval Results ──────────────────────────────────────────────────────────

class GraphResult(BaseModel):
    """Structured result from the GraphRAG retriever."""

    query: str
    result_type: GraphResultType = Field(
        default=GraphResultType.LINEAGE,
        description="Whether this result represents a path lineage, a summary report, or neighborhood",
    )
    operation: GraphOperation = Field(
        default=GraphOperation.TRAVERSAL,
        description="The fundamental graph operation performed (traversal, aggregation, neighborhood, etc.)",
    )
    entities: list[Entity] = Field(default_factory=list)
    paths: list[GraphPath] = Field(default_factory=list)
    cypher_used: str = Field(default="", description="The Cypher query that was executed")
    t2c_thought: str = Field(default="", description="The LLM internal reasoning explaining why it generated this Cypher query")
    answer: str = Field(default="", description="LLM-synthesized answer from graph context")
    intent: str | None = Field(default=None, description="The detected query intent")
    latency_ms: float = Field(default=0.0, ge=0.0)
    tokens_used: int = Field(default=0, ge=0)
    root_entity: str | None = Field(default=None, description="The anchor/root entity name and type for the traversal")
    depth_hops: int | None = Field(default=None, description="The maximum traversal depth in hops")



class VectorResult(BaseModel):
    """Structured result from the Vector RAG retriever."""

    query: str
    chunks: list[Citation] = Field(default_factory=list)
    answer: str = Field(default="", description="LLM-synthesized answer from retrieved chunks")
    latency_ms: float = Field(default=0.0, ge=0.0)
    tokens_used: int = Field(default=0, ge=0)


# ── Reasoning ─────────────────────────────────────────────────────────────────

class ReasoningStep(BaseModel):
    """One step in a ReAct (Reason + Act) reasoning trace."""

    thought: str = Field(..., description="The agent's internal reasoning")
    action: str = Field(..., description="The action taken (e.g. 'call graph_retrieve')")
    observation: str = Field(..., description="The result of the action")


# ── Final Output ──────────────────────────────────────────────────────────────

class FinalResponse(BaseModel):
    """
    The fully validated final output of the multi-agent pipeline.

    This is the schema that the Guardrail agent enforces before returning
    any answer to the user. A ValidationError here demonstrates the Pydantic
    guardrail in action.
    """

    query: str = Field(..., description="The original user query")
    tool_used: ToolChoice = Field(..., description="Which tool ultimately answered the query")
    answer: str = Field(
        ...,
        min_length=10,
        description="The final synthesized answer",
    )
    reasoning: list[ReasoningStep] = Field(
        default_factory=list,
        description="The full ReAct reasoning trace",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Agent confidence in the final answer",
    )
    tokens_used: int = Field(default=0, ge=0)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    graph_result: GraphResult | None = Field(
        default=None,
        description="Raw graph retrieval result, if graph tool was used",
    )
    vector_result: VectorResult | None = Field(
        default=None,
        description="Raw vector retrieval result, if vector tool was used",
    )

    @field_validator("answer")
    @classmethod
    def answer_must_not_be_placeholder(cls, v: str) -> str:
        placeholders = {"i don't know", "n/a", "none", "no answer", "unknown"}
        if v.strip().lower() in placeholders:
            raise ValueError(
                "FinalResponse.answer must not be a placeholder. "
                "The agent must provide a substantive answer."
            )
        return v.strip()


# ── Agent State ────────────────────────────────────────────────────────────────

class AgentState(BaseModel):
    """
    LangGraph state object passed between nodes in the state graph.
    All fields are optional because state is built up incrementally.
    """

    query: str = Field(default="")
    routing_decision: RoutingDecision | None = Field(default=None)
    graph_result: GraphResult | None = Field(default=None)
    vector_result: VectorResult | None = Field(default=None)
    reasoning_steps: list[ReasoningStep] = Field(default_factory=list)
    final_response: FinalResponse | None = Field(default=None)
    error: str | None = Field(default=None)
    total_tokens: int = Field(default=0)
    total_latency_ms: float = Field(default=0.0)

    def add_reasoning_step(self, thought: str, action: str, observation: str) -> None:
        self.reasoning_steps.append(
            ReasoningStep(thought=thought, action=action, observation=observation)
        )
