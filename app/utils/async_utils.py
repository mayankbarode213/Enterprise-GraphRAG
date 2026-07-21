"""
Async utilities — helpers for concurrent execution patterns.

The primary pattern used throughout this project:
    asyncio.gather() for concurrent retrieval + logging + token tracking.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

T = TypeVar("T")


async def timed(coro: Awaitable[T]) -> tuple[T, float]:
    """
    Await a coroutine and return (result, elapsed_ms).

    Usage:
        result, ms = await timed(some_async_call())
    """
    t0 = time.perf_counter()
    result = await coro
    return result, (time.perf_counter() - t0) * 1000


async def gather_with_timeout(
    *coros: Awaitable[Any],
    timeout: float = 30.0,
) -> list[Any]:
    """
    Run coroutines concurrently with an overall timeout.

    Args:
        *coros: Awaitable coroutines to gather.
        timeout: Seconds before raising asyncio.TimeoutError.

    Returns:
        List of results in the same order as inputs.
    """
    return await asyncio.wait_for(
        asyncio.gather(*coros),
        timeout=timeout,
    )


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    max_attempts: int = 3,
    delay_seconds: float = 1.0,
    backoff: float = 2.0,
) -> T:
    """
    Retry an async function with exponential backoff.

    Args:
        fn: Zero-argument async callable to retry.
        max_attempts: Maximum number of attempts.
        delay_seconds: Initial delay between retries.
        backoff: Multiply delay by this factor each attempt.

    Returns:
        Result of the first successful call.

    Raises:
        The last exception if all attempts fail.
    """
    last_exc: Exception | None = None
    delay = delay_seconds

    for attempt in range(1, max_attempts + 1):
        try:
            return await fn()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < max_attempts:
                await asyncio.sleep(delay)
                delay *= backoff

    raise last_exc  # type: ignore[misc]
