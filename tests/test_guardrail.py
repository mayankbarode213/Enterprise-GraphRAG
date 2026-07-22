"""
tests/test_guardrail.py

Unit tests for GuardrailAgent — demonstrates two-layer guardrails:
  Layer 1: Content guardrail (placeholder / hallucination / failure phrases)
           validate() returns rejected=True with fallback FinalResponse (never raises)
  Layer 2: Pydantic schema guardrail (field types, value ranges, length)
           validate() wraps ValidationError and returns rejected=True with fallback
"""
import pytest
from pydantic import ValidationError

from app.agents.guardrail import ContentGuardrailError, GuardrailAgent
from app.schemas.models import ReasoningStep, RoutingDecision, ToolChoice


@pytest.fixture
def guardrail():
    return GuardrailAgent()


@pytest.fixture
def valid_routing():
    return RoutingDecision(
        tool=ToolChoice.GRAPH,
        reason="Multi-hop traversal required to trace supplier batch to defect chain.",
        confidence=0.92,
        requires_multi_hop=True,
    )


@pytest.fixture
def reasoning_steps():
    return [
        ReasoningStep(
            thought="Multi-hop query detected",
            action="route_to_graph",
            observation="Graph returned 3 paths",
        )
    ]


class TestGuardrailAgent:

    async def test_valid_answer_passes(self, guardrail, valid_routing, reasoning_steps):
        """A substantive answer should pass both guardrail layers (rejected=False)."""
        response, step, rejected = await guardrail.validate(
            query="Which supplier caused the defect?",
            answer="FastFit Industries supplied Batch B003 which manufactured the Hydraulic Seal H50 installed on CNC_Lathe_Alpha, causing defects D002 and D004.",
            routing_decision=valid_routing,
            reasoning_steps=reasoning_steps,
            tokens_used=200,
            prompt_tokens=150,
            completion_tokens=50,
            latency_ms=450.0,
        )
        assert rejected is False
        assert response.tool_used == ToolChoice.GRAPH
        assert response.confidence == 0.92
        assert "guardrail_validate" in step.action

    # ── Layer 1: Content Guardrail (soft rejection — no raise) ───────────────

    async def test_empty_answer_fails_layer1(self, guardrail, valid_routing, reasoning_steps):
        """Empty answer must be rejected by Layer 1 (returned as rejected=True, confidence=0)."""
        response, step, rejected = await guardrail.validate(
            query="Which supplier caused the defect?",
            answer="",
            routing_decision=valid_routing,
            reasoning_steps=reasoning_steps,
            tokens_used=0,
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=100.0,
        )
        assert rejected is True
        assert response.confidence == 0.0
        assert "guardrail_reject" in step.action
        assert "too short" in step.observation

    async def test_failure_phrase_blocked_with_zero_entities(self, guardrail, valid_routing, reasoning_steps):
        """'Not identified due to lack of traversed relationships' + 0 entities → rejected=True."""
        from app.schemas.models import GraphOperation, GraphResult, GraphResultType
        empty_graph = GraphResult(
            query="test",
            result_type=GraphResultType.LINEAGE,
            operation=GraphOperation.TRAVERSAL,
            entities=[],
            intent="text2cypher",
            latency_ms=100.0,
            depth_hops=0,
        )
        response, step, rejected = await guardrail.validate(
            query="Which supplier caused the defect?",
            answer="Originating Supplier: Not identified due to lack of traversed relationships. Batches from the Same Supplier: Not identified due to lack of traversed relationships.",
            routing_decision=valid_routing,
            reasoning_steps=reasoning_steps,
            tokens_used=200,
            prompt_tokens=150,
            completion_tokens=50,
            latency_ms=450.0,
            graph_result=empty_graph,
        )
        assert rejected is True
        assert response.confidence == 0.0
        assert "failure phrase" in step.observation

    async def test_hallucinated_batch_names_blocked(self, guardrail, valid_routing, reasoning_steps):
        """Generic 'Batch A', 'Batch B', 'Batch C' hallucination → rejected=True."""
        response, step, rejected = await guardrail.validate(
            query="Which batches caused the defect?",
            answer="The affected batches are Batch A, Batch B, and Batch C from the supplier.",
            routing_decision=valid_routing,
            reasoning_steps=reasoning_steps,
            tokens_used=200,
            prompt_tokens=150,
            completion_tokens=50,
            latency_ms=450.0,
        )
        assert rejected is True
        assert response.confidence == 0.0
        assert "Hallucination pattern" in step.observation

    async def test_vector_fallback_message(self, guardrail, reasoning_steps):
        """Guardrail rejection in vector mode must produce a VectorRAG-specific fallback message."""
        vector_routing = RoutingDecision(
            tool=ToolChoice.VECTOR,
            reason="Semantic lookup requested by user.",
            confidence=1.0,
        )
        response, step, rejected = await guardrail.validate(
            query="Which supplier batches indirectly caused all defects?",
            answer="The affected batches are Batch A, Batch B, and Batch C.",
            routing_decision=vector_routing,
            reasoning_steps=reasoning_steps,
            tokens_used=100,
            prompt_tokens=80,
            completion_tokens=20,
            latency_ms=200.0,
        )
        assert rejected is True
        assert "VectorRAG could not retrieve sufficient textual context" in response.answer
        assert "Text-to-Cypher" not in response.answer

    # ── Layer 2: Pydantic Schema (demo raises — only for testing) ─────────────

    def test_demonstrate_schema_failure_raises(self, guardrail):
        """The Layer 2 demo method must raise ValidationError as expected."""
        with pytest.raises(ValidationError):
            guardrail.demonstrate_validation_failure()

    def test_demonstrate_content_failure_raises(self, guardrail):
        """The Layer 1 demo method must raise ContentGuardrailError as expected."""
        with pytest.raises(ContentGuardrailError):
            guardrail.demonstrate_content_failure()
