"""
tests/test_schemas.py

Unit tests for Pydantic v2 schema validation.
These tests demonstrate the guardrail behavior — most importantly, that invalid
LLM output (empty answers, out-of-range confidence, etc.) raises ValidationError.
"""
import pytest
from pydantic import ValidationError

from app.schemas.models import (
    Citation,
    Entity,
    FinalResponse,
    GraphPath,
    GraphResult,
    ReasoningStep,
    RoutingDecision,
    ToolChoice,
    VectorResult,
)


class TestRoutingDecision:
    def test_valid_graph_routing(self):
        decision = RoutingDecision(
            tool=ToolChoice.GRAPH,
            reason="Query requires multi-hop graph traversal across supplier and defect nodes.",
            confidence=0.95,
            requires_multi_hop=True,
        )
        assert decision.tool == ToolChoice.GRAPH
        assert decision.requires_multi_hop is True

    def test_valid_vector_routing(self):
        decision = RoutingDecision(
            tool=ToolChoice.VECTOR,
            reason="Simple factual lookup about a single supplier entity.",
            confidence=0.80,
        )
        assert decision.tool == ToolChoice.VECTOR

    def test_confidence_out_of_range(self):
        with pytest.raises(ValidationError) as exc_info:
            RoutingDecision(
                tool=ToolChoice.GRAPH,
                reason="Multi-hop traversal needed for relationship chain.",
                confidence=1.5,  # ← invalid: > 1.0
            )
        assert "confidence" in str(exc_info.value)

    def test_reason_too_short(self):
        with pytest.raises(ValidationError):
            RoutingDecision(
                tool=ToolChoice.GRAPH,
                reason="ok",  # ← invalid: min_length=10
                confidence=0.9,
            )

    def test_reason_cannot_be_na(self):
        with pytest.raises(ValidationError):
            RoutingDecision(
                tool=ToolChoice.VECTOR,
                reason="n/a",  # ← rejected by custom validator
                confidence=0.5,
            )


class TestFinalResponse:
    """Demonstrates the Pydantic guardrail catching bad LLM output."""

    def _valid_response(self) -> FinalResponse:
        return FinalResponse(
            query="Which supplier caused the defect?",
            tool_used=ToolChoice.GRAPH,
            answer="FastFit Industries supplied Batch B003 which caused Defects D002 and D004.",
            reasoning=[
                ReasoningStep(
                    thought="Multi-hop query detected",
                    action="route_to_graph",
                    observation="Graph traversal returned 3 paths",
                )
            ],
            confidence=0.92,
            tokens_used=150,
            latency_ms=320.5,
        )

    def test_valid_response(self):
        r = self._valid_response()
        assert r.tool_used == ToolChoice.GRAPH
        assert r.confidence == 0.92

    def test_empty_answer_rejected(self):
        """GUARDRAIL DEMO: Empty LLM output must be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            FinalResponse(
                query="test query",
                tool_used=ToolChoice.VECTOR,
                answer="",  # ← invalid: min_length=10
                reasoning=[],
                confidence=0.5,
                tokens_used=0,
                latency_ms=0.0,
            )
        assert "answer" in str(exc_info.value)

    def test_placeholder_answer_rejected(self):
        """GUARDRAIL DEMO: 'I don't know' placeholder must be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            FinalResponse(
                query="test query",
                tool_used=ToolChoice.GRAPH,
                answer="i don't know",  # ← rejected by custom validator
                reasoning=[],
                confidence=0.1,
                tokens_used=5,
                latency_ms=10.0,
            )
        assert "FinalResponse.answer" in str(exc_info.value)

    def test_confidence_must_be_0_to_1(self):
        with pytest.raises(ValidationError):
            FinalResponse(
                query="test",
                tool_used=ToolChoice.GRAPH,
                answer="The defect was caused by batch B003 from FastFit Industries.",
                reasoning=[],
                confidence=-0.1,  # ← invalid: ge=0.0
                tokens_used=0,
                latency_ms=0.0,
            )

    def test_negative_tokens_rejected(self):
        with pytest.raises(ValidationError):
            FinalResponse(
                query="test",
                tool_used=ToolChoice.VECTOR,
                answer="The answer from the document chunks is comprehensive.",
                reasoning=[],
                confidence=0.8,
                tokens_used=-50,  # ← invalid: ge=0
                latency_ms=0.0,
            )


class TestGraphPath:
    def test_path_length_computed(self):
        nodes = [
            Entity(id="S001", type="Supplier", name="FastFit"),
            Entity(id="B003", type="Batch", name="Batch_HS"),
            Entity(id="D002", type="Defect", name="Seal_Leak"),
        ]
        path = GraphPath(nodes=nodes, relationships=["SUPPLIED_BATCH", "RESULTED_IN"])
        assert path.path_length == 3

    def test_empty_nodes_rejected(self):
        with pytest.raises(ValidationError):
            GraphPath(nodes=[], relationships=[])


class TestCitation:
    def test_relevance_score_range(self):
        with pytest.raises(ValidationError):
            Citation(source="doc.txt", content="some content", relevance_score=1.5)
