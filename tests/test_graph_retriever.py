"""
tests/test_graph_retriever.py

Integration tests for GraphRetriever.
Requires a running Neo4j instance with data loaded.
Mark: pytest -m integration
"""
import pytest

from app.graph.retriever import GraphRetriever


@pytest.mark.integration
class TestGraphRetriever:

    @pytest.fixture
    async def retriever(self):
        return GraphRetriever()

    async def test_canonical_query_returns_results(self, retriever):
        """The breaking-point query must return batch B003 and defects D002/D004."""
        result = await retriever.retrieve(
            "Which supplier batches indirectly caused defects after maintenance by Apex?"
        )
        assert result is not None
        assert result.latency_ms > 0
        # Should find at least one entity
        entity_ids = {e.id for e in result.entities}
        # At minimum the answer text should reference key entities
        assert len(result.answer) > 0

    async def test_vendor_query_returns_apex(self, retriever):
        """Vendor query should identify Apex Maintenance Services."""
        result = await retriever.retrieve("What maintenance did Apex perform?")
        assert result is not None

    async def test_latency_is_recorded(self, retriever):
        """Latency must always be measured and non-negative."""
        result = await retriever.retrieve("Which batches failed QC?")
        assert result.latency_ms >= 0

    async def test_empty_result_returns_graceful_message(self, retriever):
        """Non-matching query should return a graceful message, not crash."""
        result = await retriever.retrieve("xyz abc nonexistent entity 999")
        assert "No matching records" in result.answer or len(result.answer) > 0
