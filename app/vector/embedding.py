"""
EmbeddingService — async wrapper around OpenAI embeddings.
Batches requests and handles retries transparently.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Sequence

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from settings import settings

logger = logging.getLogger(__name__)

_MAX_BATCH = 100  # OpenAI limit per request

class EmbeddingService:
    """Async OpenAI embedding client with batching and retry."""

    def __init__(self) -> None:

        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.embedding_model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Respects OpenAI's per-request limit."""
        response = await self._client.embeddings.create(
            model=self._model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    async def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed an arbitrary number of texts, batching transparently."""
        texts_list = list(texts)
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts_list), _MAX_BATCH):
            batch = texts_list[i : i + _MAX_BATCH]
            embeddings = await self.embed_batch(batch)
            all_embeddings.extend(embeddings)
            logger.debug(
                "Embedded batch %d–%d of %d",
                i + 1,
                min(i + _MAX_BATCH, len(texts_list)),
                len(texts_list),
            )

        return all_embeddings

    async def embed_one(self, text: str) -> list[float]:
        """Embed a single string."""
        results = await self.embed_batch([text])
        return results[0]
