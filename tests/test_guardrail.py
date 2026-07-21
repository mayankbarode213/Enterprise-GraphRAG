"""
tests/test_guardrail.py

Unit tests for GuardrailAgent — demonstrates Pydantic guardrails catching bad output.
"""
import pytest
from pydantic import ValidationError

from app.agents.guardrail import GuardrailAgent
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
        """A substantive answer should pass validation."""
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

    async def test_empty_answer_fails(self, guardrail, valid_routing, reasoning_steps):
        """Empty answer must be rejected by the guardrail."""
        with pytest.raises(ValidationError) as exc_info:
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
        assert "answer" in str(exc_info.value)

    async def test_placeholder_answer_fails(self, guardrail, valid_routing, reasoning_steps):
        """'I don't know' placeholder must be rejected."""
        with pytest.raises(ValidationError):
            await guardrail.validate(
                query="Which supplier caused the defect?",
                answer="i don't know",
                routing_decision=valid_routing,
                reasoning_steps=reasoning_steps,
                tokens_used=10,
                prompt_tokens=5,
                completion_tokens=5,
                latency_ms=200.0,
            )

    def test_demonstrate_validation_failure_raises(self, guardrail):
        """The demo method must raise ValidationError as expected."""
        with pytest.raises(ValidationError):
            guardrail.demonstrate_validation_failure()
