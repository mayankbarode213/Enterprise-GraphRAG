"""
RouterAgent — LLM-powered routing with no keyword rules.

The LLM reads the query, reasons about whether it requires multi-hop
relationship traversal (→ graph) or semantic similarity (→ vector),
and returns a validated RoutingDecision.
"""
from __future__ import annotations

import json
import logging
import time

from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from app.prompts.router_prompt import ROUTER_SYSTEM_PROMPT, ROUTER_USER_TEMPLATE
from app.schemas.models import ReasoningStep, RoutingDecision, ToolChoice
from settings import settings

logger = logging.getLogger(__name__)


class RouterAgent:
    """
    LLM-based query router implementing a single ReAct step:
    Thought (LLM reasoning) → Action (tool selection) → Observation (validation result).
    """

    def __init__(self) -> None:
        self._llm = ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            temperature=0,        # deterministic routing
            model_kwargs={"response_format": {"type": "json_object"}},
        )

    async def route(self, query: str) -> tuple[RoutingDecision, ReasoningStep]:
        """
        Invoke the LLM router and return:
        - A validated RoutingDecision
        - A ReasoningStep capturing the thought/action/observation trace

        Raises:
            ValidationError: if the LLM returns output that doesn't match the schema.
        """
        t0 = time.perf_counter()
        user_message = ROUTER_USER_TEMPLATE.format(query=query)

        messages = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        logger.info("RouterAgent.route | query=%.80s", query)
        response = await self._llm.ainvoke(messages)
        raw_content = str(response.content).strip()

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.debug("Router LLM response (%.1fms): %s", latency_ms, raw_content[:300])

        # Parse JSON
        try:
            payload = json.loads(raw_content)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"RouterAgent: LLM returned invalid JSON: {raw_content[:200]}"
            ) from exc

        # Validate with Pydantic — this is the guardrail in action
        decision = RoutingDecision(**payload)

        reasoning_step = ReasoningStep(
            thought=decision.reason,
            action=f"route_to_{decision.tool.value}",
            observation=(
                f"RoutingDecision validated successfully. "
                f"tool={decision.tool.value} confidence={decision.confidence:.2f}"
            ),
        )

        logger.info(
            "RouterAgent decision: tool=%s confidence=%.2f latency=%.1fms",
            decision.tool.value,
            decision.confidence,
            latency_ms,
        )
        return decision, reasoning_step
