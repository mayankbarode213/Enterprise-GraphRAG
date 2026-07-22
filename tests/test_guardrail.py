"""
tests/test_guardrail.py

Unit tests for GuardrailAgent — demonstrates two-layer guardrails:
  Layer 1: Content guardrail (placeholder / hallucination / failure phrases)
  Layer 2: Pydantic schema guardrail (field types, value ranges, length)
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
        """A substantive answer should pass both guardrail layers."""
        response, step = await guardrail.validate(
            query="Which supplier caused the defect?",
            answer="FastFit Industries supplied Batch B003 which manufactured the Hydraulic Seal H50 installed on CNC_Lathe_Alpha, causing defects D002 and D004.",
            routing_decision=valid_routing,
            reasoning_steps=reasoning_steps,
            tokens_used=200,
            prompt_tokens=150,
            completion_tokens=50,
            latency_ms=450.0,
        )
        assert response.tool_used == ToolChoice.GRAPH
        assert response.confidence == 0.92
        assert "guardrail_validate" in step.action

    # ── Layer 2: Pydantic Schema Guardrail ────────────────────────────────────

    async def test_empty_answer_fails_layer1(self, guardrail, valid_routing, reasoning_steps):
        """Empty answer must be rejected by Layer 1 content guardrail (too short)."""
        with pytest.raises(ContentGuardrailError) as exc_info:
            await guardrail.validate(
                query="Which supplier caused the defect?",
                answer="",
                routing_decision=valid_routing,
                reasoning_steps=reasoning_steps,
                tokens_used=0,
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=100.0,
            )
        assert "too short" in exc_info.value.reason


    def test_demonstrate_schema_failure_raises(self, guardrail):
        """The Layer 2 demo method must raise ValidationError as expected."""
        with pytest.raises(ValidationError):
            guardrail.demonstrate_validation_failure()

    # ── Layer 1: Content Guardrail ────────────────────────────────────────────

    async def test_failure_phrase_blocked_with_zero_entities(self, guardrail, valid_routing, reasoning_steps):
        """'Not identified due to lack of traversed relationships' + 0 entities must be blocked."""
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
        with pytest.raises(ContentGuardrailError) as exc_info:
            await guardrail.validate(
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
        assert "failure phrase" in exc_info.value.reason

    async def test_hallucinated_batch_names_blocked(self, guardrail, valid_routing, reasoning_steps):
        """Generic 'Batch A', 'Batch B', 'Batch C' hallucination must be blocked."""
        with pytest.raises(ContentGuardrailError) as exc_info:
            await guardrail.validate(
                query="Which batches caused the defect?",
                answer="The affected batches are Batch A, Batch B, and Batch C from the supplier.",
                routing_decision=valid_routing,
                reasoning_steps=reasoning_steps,
                tokens_used=200,
                prompt_tokens=150,
                completion_tokens=50,
                latency_ms=450.0,
            )
        assert "Hallucination pattern" in exc_info.value.reason

    def test_demonstrate_content_failure_raises(self, guardrail):
        """The Layer 1 demo method must raise ContentGuardrailError as expected."""
        with pytest.raises(ContentGuardrailError):
            guardrail.demonstrate_content_failure()
