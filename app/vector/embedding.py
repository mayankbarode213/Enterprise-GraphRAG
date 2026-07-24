"""
EmbeddingService — LangChain wrapper around OpenAI embeddings.
Provides async batch embedding, query embedding, and direct access to OpenAIEmbeddings.
"""
from __future__ import annotations

import logging
from typing import Sequence

from langchain_openai import OpenAIEmbeddings

from settings import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Async OpenAI embedding client backed by LangChain's OpenAIEmbeddings."""

    def __init__(self) -> None:
        self._embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
        )

    @property
    def embeddings(self) -> OpenAIEmbeddings:
        """Expose underlying LangChain OpenAIEmbeddings instance."""
        return self._embeddings

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts using LangChain OpenAIEmbeddings."""
        if not texts:
            return []
        return await self._embeddings.aembed_documents(texts)

    async def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed an arbitrary number of texts."""
        return await self.embed_batch(list(texts))

    async def embed_one(self, text: str) -> list[float]:
        """Embed a single query string."""
        return await self._embeddings.aembed_query(text)
