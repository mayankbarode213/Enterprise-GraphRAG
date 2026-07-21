"""
VectorIngestor — chunks documents, embeds them, and upserts into FAISSVectorStore.
Idempotent: same chunk_id overwrites previous entry.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from app.vector.chunker import Chunk, chunk_documents
from app.vector.embedding import EmbeddingService
from app.vector.store import FAISSVectorStore
from settings import settings

logger = logging.getLogger(__name__)

DOCUMENTS_DIR = Path(__file__).parents[2] / "data" / "documents"
print("DOCUMENTS_DIR:", DOCUMENTS_DIR)
EMBEDDING_DIM = 1536  # text-embedding-3-small dimension


def _get_store() -> FAISSVectorStore:
    """Return (or load) the persistent FAISS vector store."""
    return FAISSVectorStore(
        persist_dir=settings.vector_dir_path,  # reusing the same path setting
        dimension=EMBEDDING_DIM,
    )


async def ingest(documents_dir: Path | None = None) -> int:
    """
    Full ingestion pipeline:
    1. Chunk all .txt documents
    2. Embed chunks via OpenAI
    3. Upsert into FAISSVectorStore

    Returns: number of chunks upserted.
    """
    doc_dir = documents_dir or DOCUMENTS_DIR
    store = _get_store()
    embedder = EmbeddingService()

    chunks: list[Chunk] = chunk_documents(doc_dir)
    if not chunks:
        logger.warning("No chunks produced — nothing to ingest")
        return 0

    logger.info("Embedding %d chunks …", len(chunks))
    texts = [c.content for c in chunks]
    embeddings = await embedder.embed_many(texts)

    store.upsert(
        doc_ids=[c.chunk_id for c in chunks],
        embeddings=embeddings,
        contents=texts,
        metadatas=[{"source": c.source, "chunk_index": c.chunk_index} for c in chunks],
    )

    logger.info("Upserted %d chunks into FAISS vector store", len(chunks))
    return len(chunks)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(ingest())
