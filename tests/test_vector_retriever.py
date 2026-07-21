"""
tests/test_vector_retriever.py

Unit tests for VectorRetriever — mocks FAISS store and embedding calls.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.schemas.models import VectorResult
from app.vector.store import FAISSVectorStore, VectorStoreDoc


class TestVectorRetriever:
    """Unit tests with mocked external dependencies."""

    @pytest.fixture
    def retriever(self):
        from app.vector.retriever import VectorRetriever
        r = VectorRetriever.__new__(VectorRetriever)
        r._embedder = MagicMock()
        r._embedder.embed_one = AsyncMock(return_value=[0.1] * 1536)
        r._llm = MagicMock()
        r._store = MagicMock(spec=FAISSVectorStore)
        return r

    async def test_returns_vector_result(self, retriever):
        """VectorRetriever must return a valid VectorResult."""
        doc = VectorStoreDoc(
            doc_id="doc_002::chunk_0",
            content="FastFit Industries supplied Batch B003 which had a QC failure.",
            metadata={"source": "doc_002_supplier_fastfit.txt", "chunk_index": 0},
        )
        retriever._store.count.return_value = 1
        retriever._store.search.return_value = [(doc, 0.92)]

        mock_llm_resp = MagicMock()
        mock_llm_resp.content = "Based on the context, FastFit Industries supplied a defective batch B003."
        mock_llm_resp.usage_metadata = {"total_tokens": 150}
        retriever._llm.ainvoke = AsyncMock(return_value=mock_llm_resp)

        result = await retriever.retrieve("What did FastFit supply?")

        assert isinstance(result, VectorResult)
        assert len(result.chunks) == 1
        assert result.chunks[0].source == "doc_002_supplier_fastfit.txt"
        assert result.latency_ms >= 0

    async def test_empty_store_returns_graceful_result(self, retriever):
        """Empty vector store should return a graceful message, not crash."""
        retriever._store.count.return_value = 0

        result = await retriever.retrieve("test query")

        assert isinstance(result, VectorResult)
        assert result.chunks == []
        assert "empty" in result.answer.lower() or "ingest" in result.answer.lower()

    async def test_relevance_score_clipped_to_0_1(self, retriever):
        """Cosine scores must always be in [0, 1] after clipping."""
        doc = VectorStoreDoc(doc_id="x", content="text", metadata={"source": "f.txt"})
        retriever._store.count.return_value = 1
        retriever._store.search.return_value = [(doc, 0.95)]

        mock_resp = MagicMock()
        mock_resp.content = "The document contains relevant manufacturing defect information."
        mock_resp.usage_metadata = {"total_tokens": 30}
        retriever._llm.ainvoke = AsyncMock(return_value=mock_resp)

        result = await retriever.retrieve("test")
        assert 0.0 <= result.chunks[0].relevance_score <= 1.0


class TestFAISSVectorStore:
    """Unit tests for the FAISS store itself."""

    @pytest.fixture
    def tmp_store(self, tmp_path):
        return FAISSVectorStore(persist_dir=tmp_path, dimension=4)

    def test_empty_on_creation(self, tmp_store):
        assert tmp_store.count() == 0

    def test_upsert_and_count(self, tmp_store):
        import numpy as np
        emb = list(np.random.rand(4).astype(float))
        tmp_store.upsert(["doc1"], [emb], ["hello world"], [{"source": "f.txt"}])
        assert tmp_store.count() == 1

    def test_search_returns_results(self, tmp_store):
        import numpy as np
        emb = list(np.ones(4, dtype=float))
        tmp_store.upsert(["doc1"], [emb], ["test content"], [{}])
        results = tmp_store.search(emb, k=1)
        assert len(results) == 1
        doc, score = results[0]
        assert doc.doc_id == "doc1"
        assert 0.0 <= score <= 1.01  # slight float tolerance

    def test_empty_search_returns_empty(self, tmp_store):
        import numpy as np
        results = tmp_store.search(list(np.ones(4, dtype=float)), k=5)
        assert results == []


@pytest.mark.integration
class TestVectorRetrieverIntegration:
    """Integration tests — require populated FAISS store."""

    async def test_breaking_point_query_does_not_crash(self):
        from app.vector.retriever import VectorRetriever
        retriever = VectorRetriever()
        result = await retriever.retrieve(
            "Which supplier batches indirectly caused defects after Apex maintenance?"
        )
        assert isinstance(result, VectorResult)
        assert len(result.answer) > 0
