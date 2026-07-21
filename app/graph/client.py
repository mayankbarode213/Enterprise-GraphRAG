"""
Async Neo4j driver wrapper with connection pooling and context manager support.
All graph operations go through this client.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

# pyrefly: ignore [missing-import]
from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
from tenacity import retry, stop_after_attempt, wait_exponential

from settings import settings

logger = logging.getLogger(__name__)


class Neo4jClient:
    """
    Thread-safe async Neo4j client.

    Usage:
        client = Neo4jClient()
        await client.connect()
        async with client.session() as session:
            result = await session.run("RETURN 1")
        await client.close()

    Or use as async context manager:
        async with Neo4jClient() as client:
            ...
    """

    def __init__(self) -> None:
        self._driver: AsyncDriver | None = None

    async def connect(self) -> None:
        """Open the driver connection pool."""
        self._driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
            max_connection_pool_size=10,
            connection_timeout=30.0,
        )
        await self._verify_connectivity()
        logger.info("Neo4j driver connected to %s", settings.neo4j_uri)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def _verify_connectivity(self) -> None:
        """Ping Neo4j; retries up to 5 times with exponential back-off."""
        if self._driver is None:
            raise RuntimeError("Driver not initialised")
        await self._driver.verify_connectivity()
        logger.debug("Neo4j connectivity verified")

    async def close(self) -> None:
        """Drain the connection pool gracefully."""
        if self._driver:
            await self._driver.close()
            self._driver = None
            logger.info("Neo4j driver closed")

    @asynccontextmanager
    async def session(self, database: str = "neo4j") -> AsyncGenerator[AsyncSession, None]:
        """Yield a session; always closes it afterwards."""
        if self._driver is None:
            raise RuntimeError("Call connect() before requesting a session")
        async with self._driver.session(database=database) as s:
            yield s

    async def run_query(
        self,
        cypher: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Execute a read Cypher query and return results as plain dicts.

        Args:
            cypher: Parameterized Cypher string.
            parameters: Query parameters (prevents injection).

        Returns:
            List of record dicts.
        """
        params = parameters or {}
        async with self.session() as s:
            result = await s.run(cypher, params)
            records = await result.data()
            logger.debug(
                "Cypher executed | rows=%d | query=%.120s",
                len(records),
                cypher,
            )
            return records

    async def run_write(
        self,
        cypher: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a write Cypher query inside an explicit write transaction."""
        params = parameters or {}

        async def _tx(tx: Any) -> list[dict[str, Any]]:
            result = await tx.run(cypher, params)
            return await result.data()

        async with self.session() as s:
            return await s.execute_write(_tx)

    # ── Async context manager ──────────────────────────────────────────────────
    async def __aenter__(self) -> "Neo4jClient":
        await self.connect()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()


# Module-level singleton — agents import and await .connect() once
_client: Neo4jClient | None = None
_lock = asyncio.Lock()


async def get_client() -> Neo4jClient:
    """Return the shared Neo4j client, connecting on first call."""
    global _client
    async with _lock:
        if _client is None:
            _client = Neo4jClient()
            await _client.connect()
            try:
                from app.graph.loader import ensure_graph_loaded
                await ensure_graph_loaded(_client)
            except Exception as exc:
                logger.warning("Failed to run ensure_graph_loaded on client init: %s", exc)
    return _client

