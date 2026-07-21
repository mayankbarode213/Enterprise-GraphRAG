"""
SemanticCache — prompt-level caching to reduce redundant LLM calls.

Implementation: in-memory dict (hash → response).
Production path: Replace the dict with Redis using semantic similarity lookup
(embed prompt → cosine search over cached embeddings → return if score > threshold).

This stub demonstrates the architecture and is presentable in interviews.
"""
from __future__ import annotations

import hashlib
import logging
import re

logger = logging.getLogger(__name__)


def _normalize(text: str) -> str:
    """Normalize a prompt for cache key generation."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def _hash(text: str) -> str:
    """SHA-256 hash of normalized text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class SemanticCache:
    """
    Stub semantic cache.

    Production upgrade path:
    1. Embed the incoming prompt using EmbeddingService.
    2. Query Redis vector index (e.g., RediSearch + HNSW) for nearest neighbor.
    3. If cosine distance < threshold (e.g., 0.05), return cached response.
    4. Otherwise, call LLM and store (embedding, response) in Redis with TTL.

    Benefits:
    - Eliminates duplicate LLM calls for rephrased but semantically identical queries.
    - Reduces latency by ~90% for cached queries.
    - Cuts token costs proportionally.
    """

    def __init__(self) -> None:
        self._store: dict[str, str] = {}
        self.hits = 0
        self.misses = 0

    def get(self, prompt: str) -> str | None:
        """Return cached response or None on cache miss."""
        key = _hash(_normalize(prompt))
        result = self._store.get(key)
        if result is not None:
            self.hits += 1
            logger.debug("SemanticCache HIT | key=%s", key[:16])
        else:
            self.misses += 1
            logger.debug("SemanticCache MISS | key=%s", key[:16])
        return result

    def set(self, prompt: str, response: str) -> None:
        """Store a prompt→response pair."""
        key = _hash(_normalize(prompt))
        self._store[key] = response
        logger.debug("SemanticCache SET | key=%s | response_len=%d", key[:16], len(response))

    def stats(self) -> dict[str, int | float]:
        """Return cache statistics."""
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total": total,
            "hit_rate": round(self.hits / total, 3) if total > 0 else 0.0,
            "cached_entries": len(self._store),
        }

    def clear(self) -> None:
        """Clear all cached entries."""
        self._store.clear()
        self.hits = 0
        self.misses = 0


# Module-level singleton
cache = SemanticCache()
