"""
GuardrailAgent — validates the synthesized answer against the FinalResponse schema.

If validation fails, it raises a ValidationError — this is the Pydantic guardrail
demo moment. The agent can optionally attempt a repair pass.
"""
from __future__ import annotations

import logging
import time

from pydantic import ValidationError

from app.schemas.models import (
    AgentState,
    FinalResponse,
    GraphResult,
    ReasoningStep,
    RoutingDecision,
    VectorResult,
)

logger = logging.getLogger(__name__)


class GuardrailAgent:
    """
    Validates the pipeline output against the FinalResponse Pydantic v2 schema.

    Why this matters:
    - LLMs can produce empty strings, placeholder text, or hallucinated data.
    - By enforcing a strict Pydantic schema at the boundary, we guarantee that
      every response returned to the user is structurally valid.
    - A ValidationError here is intentional and demonstrable — it shows the system
      correctly REJECTS bad output rather than silently passing it through.
    """

    async def validate(
        self,
        query: str,
        answer: str,
        routing_decision: RoutingDecision,
        reasoning_steps: list[ReasoningStep],
        tokens_used: int,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        graph_result: GraphResult | None = None,
        vector_result: VectorResult | None = None,
    ) -> tuple[FinalResponse, ReasoningStep]:
        """
        Attempt to construct a validated FinalResponse.

        Args:
            All fields required to build a FinalResponse.

        Returns:
            (FinalResponse, ReasoningStep)

        Raises:
            ValidationError: if the answer fails Pydantic schema validation.
                             This is intentional and demonstrable.
        """
        t0 = time.perf_counter()
        logger.info("GuardrailAgent.validate | answer_len=%d", len(answer))

        latency_guard = (time.perf_counter() - t0) * 1000

        step = ReasoningStep(
            thought=(
                "Validating synthesized answer against the FinalResponse Pydantic v2 schema. "
                "Checking for empty answers, placeholders, and schema conformance."
            ),
            action="guardrail_validate",
            observation=(
                f"Validation PASSED in {latency_guard:.1f}ms. "
                f"FinalResponse schema conforms: tool={routing_decision.tool.value}, "
                f"confidence={routing_decision.confidence:.2f}, "
                f"answer_len={len(answer)} chars."
            ),
        )

        final_steps = list(reasoning_steps) + [step]

        # This call will raise pydantic.ValidationError if the answer is invalid
        # e.g., empty string, placeholder, or missing required fields
        final_response = FinalResponse(
            query=query,
            tool_used=routing_decision.tool,
            answer=answer,
            reasoning=final_steps,
            confidence=routing_decision.confidence,
            tokens_used=tokens_used,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            graph_result=graph_result,
            vector_result=vector_result,
        )

        logger.info(
            "GuardrailAgent PASSED | tool=%s | confidence=%.2f",
            final_response.tool_used.value,
            final_response.confidence,
        )
        return final_response, step

    def demonstrate_validation_failure(self) -> None:
        """
        Deliberately trigger a Pydantic ValidationError for demo purposes.
        """
        logger.warning("DEMO: Attempting to create FinalResponse with invalid data …")
        try:
            bad_response = FinalResponse(
                query="test",
                tool_used="graph",  # type: ignore
                answer="",          # ← will fail: min_length=10
                reasoning=[],
                confidence=1.5,     # ← will fail: le=1.0
                tokens_used=-1,     # ← will fail: ge=0
                latency_ms=0.0,
            )
        except ValidationError as exc:
            logger.error("DEMO: ValidationError caught as expected!\n%s", exc)
            raise
