"""
Lightweight in-memory metrics collector.

Tracks: latency histogram, token counter, route distribution.
In production, replace with Prometheus counters + Grafana dashboards.
"""
from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import DefaultDict


@dataclass
class MetricsCollector:
    """
    Simple in-process metrics store.

    Production equivalent: Prometheus Counter/Histogram + OpenTelemetry spans.
    Could also use LangSmith callbacks for per-run tracing.
    """

    latencies_ms: list[float] = field(default_factory=list)
    token_counts: list[int] = field(default_factory=list)
    route_counts: DefaultDict[str, int] = field(default_factory=lambda: defaultdict(int))
    error_counts: DefaultDict[str, int] = field(default_factory=lambda: defaultdict(int))
    total_calls: int = 0

    def record(
        self,
        tool: str,
        latency_ms: float,
        tokens: int,
        success: bool = True,
    ) -> None:
        """Record a single pipeline invocation."""
        self.total_calls += 1
        self.latencies_ms.append(latency_ms)
        self.token_counts.append(tokens)
        self.route_counts[tool] += 1
        if not success:
            self.error_counts[tool] += 1

    def summary(self) -> dict:
        """Return a summary dict suitable for logging or display."""
        lats = self.latencies_ms
        toks = self.token_counts

        return {
            "total_calls": self.total_calls,
            "route_distribution": dict(self.route_counts),
            "error_counts": dict(self.error_counts),
            "latency_ms": {
                "mean": round(statistics.mean(lats), 2) if lats else 0,
                "median": round(statistics.median(lats), 2) if lats else 0,
                "min": round(min(lats), 2) if lats else 0,
                "max": round(max(lats), 2) if lats else 0,
                "p95": round(sorted(lats)[int(len(lats) * 0.95)], 2) if len(lats) >= 2 else 0,
            },
            "tokens": {
                "total": sum(toks),
                "mean_per_call": round(statistics.mean(toks), 1) if toks else 0,
                "estimated_cost_usd": round(sum(toks) / 1_000_000 * 0.15, 6),  # gpt-4o-mini rate
            },
            "production_note": (
                "In production: replace with Prometheus metrics, "
                "Grafana dashboards, LangSmith traces, and OpenTelemetry spans."
            ),
        }


# Module-level singleton
metrics = MetricsCollector()
