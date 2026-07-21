"""
scripts/compare.py — Head-to-head demo script.

Runs the canonical breaking-point query (and supporting queries) through
BOTH Vector RAG and GraphRAG and displays a side-by-side comparison.

Usage:
    python scripts/compare.py
    python scripts/compare.py --query "Which batches failed QC?"
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from app.graph.service import GraphService
from app.utils.logging import setup_logging
from app.vector.retriever import VectorRetriever
from evaluation.ground_truth import get_breaking_point_query, load_ground_truth
from evaluation.metrics import compute_entity_recall, is_correct
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

console = Console(width=160)

CANONICAL_QUERY = (
    "Which supplier batch indirectly caused all defects reported after the emergency maintenance performed by Apex Industrial Services on Robotic Welding Cell RW-101?"
)


async def run_comparison(query: str, expected_ids: list[str] | None = None) -> None:
    """Run both systems on the query and display results."""

    console.print(Rule("[bold yellow]Query[/bold yellow]"))
    console.print(f"\n[bold]{query}[/bold]\n")

    graph_service = GraphService()
    vector_retriever = VectorRetriever()

    # Run both concurrently
    console.print("[dim]Running Graph RAG and Vector RAG concurrently...[/dim]\n")
    t0 = time.perf_counter()
    graph_result, vector_result = await asyncio.gather(
        graph_service.query(query),
        vector_retriever.retrieve(query),
    )
    total_ms = (time.perf_counter() - t0) * 1000

    # ── Display results side by side ───────────────────────────────────────────
    graph_correct = is_correct(expected_ids or [], graph_result.answer) if expected_ids else None
    vector_correct = is_correct(expected_ids or [], vector_result.answer) if expected_ids else None

    graph_panel = Panel(
        f"[bold]Answer:[/bold]\n{graph_result.answer[:800]}\n\n"
        f"[dim]Entities: {len(graph_result.entities)} | Paths: {len(graph_result.paths)} | "
        f"Latency: {graph_result.latency_ms:.1f}ms[/dim]"
        + (f"\n[bold green]CORRECT[/bold green]" if graph_correct is True
           else f"\n[bold red]INCORRECT[/bold red]" if graph_correct is False else ""),
        title="[bold cyan]GRAPH RAG (Neo4j)[/bold cyan]",
        border_style="cyan",
        expand=True,
    )

    top_score = f"{vector_result.chunks[0].relevance_score:.3f}" if vector_result.chunks else "0.000"

    vector_panel = Panel(
        f"[bold]Answer:[/bold]\n{vector_result.answer[:800]}\n\n"
        f"[dim]Chunks: {len(vector_result.chunks)} | "
        f"Top score: {top_score} | "
        f"Latency: {vector_result.latency_ms:.1f}ms[/dim]"
        + (f"\n[bold green]CORRECT[/bold green]" if vector_correct is True
           else f"\n[bold red]INCORRECT[/bold red]" if vector_correct is False else ""),
        title="[bold magenta]VECTOR RAG (FAISS)[/bold magenta]",
        border_style="magenta",
        expand=True,
    )

    console.print(Columns([graph_panel, vector_panel]))

    # ── Metrics summary ────────────────────────────────────────────────────────
    if expected_ids:
        graph_recall = compute_entity_recall(expected_ids, graph_result.answer)
        vector_recall = compute_entity_recall(expected_ids, vector_result.answer)

        console.print(Rule("[bold yellow]Metrics[/bold yellow]"))
        console.print(f"\n[bold]Expected entities:[/bold] {', '.join(expected_ids)}")
        console.print(
            f"Graph recall:  [cyan]{graph_recall:.1%}[/cyan]  | "
            f"Vector recall: [magenta]{vector_recall:.1%}[/magenta]"
        )
        console.print(f"Total wall time (both systems, concurrent): {total_ms:.0f}ms\n")

        if graph_recall > vector_recall:
            console.print(Panel(
                "[bold green]GRAPH RAG WINS[/bold green]\n\n"
                "This is the breaking point: the query requires multi-hop relationship traversal "
                "(Supplier -> Batch -> Component -> Machine -> Maintenance -> Vendor -> Incident -> Defect).\n\n"
                "Vector RAG cannot traverse this 5-hop chain because the answer is distributed "
                "across 6+ document chunks with no single chunk containing the full causal path.\n\n"
                "Graph RAG traverses the relationship graph directly and returns the exact answer.",
                border_style="green",
            ))


async def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description="GraphRAG vs Vector RAG head-to-head comparison")
    parser.add_argument("--query", type=str, default=None, help="Custom query (default: canonical breaking-point)")
    args = parser.parse_args()

    console.print(Panel.fit(
        "[bold]GraphRAG vs Vector RAG - Head-to-Head Demo[/bold]\n"
        "[dim]Manufacturing Defect Analysis Domain[/dim]",
        border_style="yellow",
    ))
    console.print()

    if args.query:
        await run_comparison(args.query)
    else:
        # Canonical breaking-point query with ground truth
        gt = load_ground_truth()
        bp = get_breaking_point_query(gt)
        expected_ids = []
        for batch in bp["expected_answer"].get("supplier_batches", []):
            expected_ids.extend(batch.get("defects_caused", []))
            if b := batch.get("batch_id"):
                expected_ids.append(b)

        await run_comparison(bp["question"], expected_ids=expected_ids)


if __name__ == "__main__":
    asyncio.run(main())
