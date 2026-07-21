"""
tests/test_router.py

Unit tests for the RouterAgent.
Mocks the LLM call to test routing logic without API calls.
"""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.router import RouterAgent
from app.schemas.models import ToolChoice


class TestRouterAgent:

    @pytest.fixture
    def router(self):
        return RouterAgent()

    def _mock_llm_response(self, tool: str, reason: str, confidence: float, multi_hop: bool = False):
        """Create a mock LLM response."""
        payload = {
            "tool": tool,
            "reason": reason,
            "confidence": confidence,
            "requires_multi_hop": multi_hop,
        }
        mock_response = MagicMock()
        mock_response.content = json.dumps(payload)
        return mock_response

    async def test_multi_hop_query_routes_to_graph(self, router):
        """Complex relational queries should route to graph."""
        mock_resp = self._mock_llm_response(
            tool="graph",
            reason="Query requires traversal across Supplier, Batch, Component, Machine, and Defect nodes.",
            confidence=0.95,
            multi_hop=True,
        )
        with patch("langchain_openai.ChatOpenAI.ainvoke", new=AsyncMock(return_value=mock_resp)):
            decision, step = await router.route(
                "Which supplier batches caused defects after machine maintenance by Apex?"
            )

        assert decision.tool == ToolChoice.GRAPH
        assert decision.requires_multi_hop is True
        assert decision.confidence == 0.95
        assert step.action == "route_to_graph"

    async def test_simple_factual_query_routes_to_vector(self, router):
        """Simple factual lookups should route to vector."""
        mock_resp = self._mock_llm_response(
            tool="vector",
            reason="Simple factual question about supplier rating can be answered from a single document.",
            confidence=0.88,
            multi_hop=False,
        )
        with patch("langchain_openai.ChatOpenAI.ainvoke", new=AsyncMock(return_value=mock_resp)):
            decision, step = await router.route("What is PrecisionParts Co.'s quality rating?")

        assert decision.tool == ToolChoice.VECTOR
        assert decision.confidence == 0.88
        assert step.action == "route_to_vector"

    async def test_invalid_json_raises_value_error(self, router):
        """Malformed LLM JSON should raise ValueError."""
        mock_resp = MagicMock()
        mock_resp.content = "This is not JSON at all"

        with patch("langchain_openai.ChatOpenAI.ainvoke", new=AsyncMock(return_value=mock_resp)):
            with pytest.raises(ValueError, match="invalid JSON"):
                await router.route("some query")

    async def test_invalid_confidence_raises_validation_error(self, router):
        """Out-of-range confidence should raise Pydantic ValidationError."""
        from pydantic import ValidationError

        mock_resp = MagicMock()
        mock_resp.content = json.dumps({
            "tool": "graph",
            "reason": "Multi-hop reasoning required for this complex query.",
            "confidence": 2.0,  # ← invalid
            "requires_multi_hop": True,
        })

        with patch("langchain_openai.ChatOpenAI.ainvoke", new=AsyncMock(return_value=mock_resp)):
            with pytest.raises(ValidationError):
                await router.route("some query")
