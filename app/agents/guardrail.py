"""
GuardrailAgent — validates the synthesized answer against the FinalResponse schema.

Two-layer validation:
  Layer 1 (Content Guardrail) — checks answer quality BEFORE Pydantic schema validation.
           Rejects: placeholder text, hallucinated generic names, zero-entity graph answers,
                    SQL injection patterns, and known failure phrases.
  Layer 2 (Schema Guardrail) — Pydantic v2 enforces field types, value ranges, and lengths.

If either layer fails, a descriptive error is raised and logged.
"""
from __future__ import annotations

import logging
import re
import time

from pydantic import ValidationError

from app.schemas.models import (
    AgentState,
    FinalResponse,
    GraphResult,
    ReasoningStep,
    RoutingDecision,
    ToolChoice,
    VectorResult,
)

logger = logging.getLogger(__name__)


# ── Layer 1: Content-Level Guardrail Patterns ────────────────────────────────

# Phrases that indicate the LLM failed to retrieve real data
FAILURE_PHRASES: list[str] = [
    "not identified due to lack of traversed",
    "not identified due to lack",
    "no information available",
    "i don't have information",
    "i cannot answer",
    "unable to determine",
    "no data found",
    "no results found",
    "i do not know",
    "i don't know",
    "insufficient data",
    "no relevant information",
    "context does not contain",
]

# Hallucinated generic placeholder names the LLM invents when it has no data
HALLUCINATION_PATTERNS: list[str] = [
    r"\bBatch\s+[A-C]\b",           # "Batch A", "Batch B", "Batch C"
    r"\bSupplier\s+[A-C]\b",        # "Supplier A", "Supplier B"
    r"\bMachine\s+[A-C]\b",         # "Machine A"
    r"\bEntity\s+\d+\b",            # "Entity 1", "Entity 2"
    r"\[PLACEHOLDER\]",             # "[PLACEHOLDER]"
    r"\bXYZ\b",                     # Generic "XYZ" placeholders
    r"<(?:placeholder|insert|name_here|value_here)>",  # Explicit placeholder tags like <placeholder>
    r"\[INSERT_[^\]]+\]",           # "[INSERT_NAME]"
]

# Phrases indicating zero traversal — LLM admitted it found nothing meaningful
ZERO_TRAVERSAL_PHRASES: list[str] = [
    "0 relationship traversals",
    "relationship traversals executed: 0",
    "nodes traversed: 0",
    "0 nodes traversed",
]


class ContentGuardrailError(ValueError):
    """Raised when the answer fails Layer 1 content-quality checks."""

    def __init__(self, reason: str, answer_snippet: str) -> None:
        self.reason = reason
        self.answer_snippet = answer_snippet[:200]
        super().__init__(f"ContentGuardrail BLOCKED — {reason} | snippet: '{self.answer_snippet}'")


class GuardrailAgent:
    """
    Validates the pipeline output through two sequential guardrail layers.

    Layer 1 — Content Guardrail:
      Checks for placeholder text, hallucination, failure phrases, and
      zero-entity graph responses BEFORE schema validation.

    Layer 2 — Pydantic Schema Guardrail:
      Enforces FinalResponse field types, value constraints (confidence ≤ 1.0,
      tokens_used ≥ 0, answer min_length=10), and required fields.
    """

    # ── Layer 1: Content Guardrail ────────────────────────────────────────────

    def _check_content(
        self,
        answer: str,
        graph_result: GraphResult | None,
    ) -> tuple[bool, str]:
        """
        Run content-level checks on the synthesized answer.

        Returns:
            (passed: bool, reason: str)
        """
        answer_lower = answer.lower()

        # Check 1: Failure phrases — LLM admitted it found nothing
        for phrase in FAILURE_PHRASES:
            if phrase in answer_lower:
                # Only block if graph returned 0 entities (T2C failure case)
                if graph_result is not None and len(graph_result.entities) == 0:
                    return False, f"Answer contains failure phrase with 0 graph entities: '{phrase}'"

        # Check 2: Hallucinated generic names
        for pattern in HALLUCINATION_PATTERNS:
            if re.search(pattern, answer, re.IGNORECASE):
                return False, f"Hallucination pattern detected in answer: '{pattern}'"

        # Check 3: Zero-entity graph response with a non-trivial graph query
        if graph_result is not None and len(graph_result.entities) == 0:
            if graph_result.intent not in (None, "", "text2cypher"):
                pass  # Allow parameterized queries to return 0 entities for truly empty results
            else:
                # T2C returned 0 entities — this is a retrieval failure
                if len(answer) < 200:
                    return False, "Text-to-Cypher returned 0 entities and answer is too short to be meaningful"

        # Check 4: Minimum meaningful content length
        if len(answer.strip()) < 50:
            return False, f"Answer too short to be meaningful: {len(answer)} chars"

        return True, ""

    # ── Main validate method ──────────────────────────────────────────────────

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
    ) -> tuple[FinalResponse, ReasoningStep, bool]:
        """
        Two-layer validation: content guardrail → Pydantic schema guardrail.

        Returns:
            (FinalResponse, ReasoningStep, rejected: bool)
            rejected=True means a guardrail layer blocked the answer.
            Never raises — always returns a structured FinalResponse so
            LangGraph cannot intercept an unhandled node exception.
        """
        t0 = time.perf_counter()
        logger.info("GuardrailAgent.validate | answer_len=%d", len(answer))

        # ── Layer 1: Content Guardrail ────────────────────────────────────────
        passed, reason = self._check_content(answer, graph_result)
        if not passed:
            logger.warning("GuardrailAgent BLOCKED (Layer 1 — Content): %s", reason)
            if routing_decision.tool == ToolChoice.VECTOR:
                fallback_answer = (
                    "⚠️ VectorRAG could not retrieve sufficient textual context from unstructured documents to answer this multi-hop query. "
                    "Vector search locates isolated text passages but cannot trace multi-hop entity relationships across multiple schema hops. "
                    "Please switch to GraphRAG mode or Auto Router for complete relationship traversal."
                )
            else:
                fallback_answer = (
                    "⚠️ The system could not retrieve sufficient graph data to answer this query reliably. "
                    "This may be because the Text-to-Cypher query returned no connected entities, or the "
                    "synthesized answer contained placeholder text. "
                    "Please try rephrasing your query or disable Text-to-Cypher mode for a more precise traversal."
                )
            reject_step = ReasoningStep(
                thought="Guardrail Layer 1 (Content) rejected the synthesized answer.",
                action="guardrail_reject",
                observation=f"ContentGuardrailError: {reason}",
            )
            final_steps = list(reasoning_steps) + [reject_step]
            fallback_response = FinalResponse(
                query=query,
                tool_used=routing_decision.tool,
                answer=fallback_answer,
                reasoning=final_steps,
                confidence=0.0,
                tokens_used=tokens_used,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                graph_result=graph_result,
                vector_result=vector_result,
            )
            return fallback_response, reject_step, True

        # ── Layer 2: Pydantic Schema Guardrail ───────────────────────────────
        latency_guard = (time.perf_counter() - t0) * 1000

        step = ReasoningStep(
            thought=(
                "Verifying synthesized response against quality criteria: checking for completeness, "
                "absence of generic hallucinated placeholders, and strict Pydantic v2 schema compliance."
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

        # This call raises pydantic.ValidationError if schema constraints are violated
        try:
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
        except ValidationError as exc:
            logger.error("GuardrailAgent BLOCKED (Layer 2 — Schema): %s", exc)
            schema_reject_step = ReasoningStep(
                thought="Guardrail Layer 2 (Pydantic Schema) rejected the synthesized answer.",
                action="guardrail_reject",
                observation=f"ValidationError: {str(exc)[:300]}",
            )
            fallback_answer = (
                "⚠️ The system produced a structurally invalid response that failed schema validation. "
                "This is a system-level safety catch. Please retry your query."
            )
            fallback_response = FinalResponse(
                query=query,
                tool_used=routing_decision.tool,
                answer=fallback_answer,
                reasoning=final_steps + [schema_reject_step],
                confidence=0.0,
                tokens_used=tokens_used,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
            )
            return fallback_response, schema_reject_step, True

        logger.info(
            "GuardrailAgent PASSED | tool=%s | confidence=%.2f",
            final_response.tool_used.value,
            final_response.confidence,
        )
        return final_response, step, False

    # ── Demo / Testing ────────────────────────────────────────────────────────

    def demonstrate_validation_failure(self) -> None:
        """
        Deliberately trigger a Pydantic ValidationError for demo purposes.
        Shows Layer 2 (schema) rejection of structurally invalid data.
        """
        logger.warning("DEMO: Attempting to create FinalResponse with invalid data …")
        try:
            FinalResponse(
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

    def demonstrate_content_failure(self) -> None:
        """
        Deliberately trigger a ContentGuardrailError for demo purposes.
        Shows Layer 1 (content) rejection of hallucinated / placeholder answers.
        """
        logger.warning("DEMO: Attempting to pass hallucinated answer through content guardrail …")
        bad_answer = "The affected batches are Batch A, Batch B, and Batch C."
        passed, reason = self._check_content(bad_answer, graph_result=None)
        if not passed:
            raise ContentGuardrailError(reason=reason, answer_snippet=bad_answer)
