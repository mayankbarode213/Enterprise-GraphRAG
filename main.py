"""
main.py — Interactive multi-agent pipeline entry point.

Usage:
    python main.py
    python main.py --query "Which supplier batches caused defects after maintenance?"
    python main.py --demo    # Run the canonical breaking-point query
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from app.utils.logging import setup_logging
from app.telemetry.metrics import metrics
from app.workflow.graph import run_pipeline
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

console = Console()

DEMO_QUERY = (
    "Which supplier batches indirectly caused all defects reported after "
    "CNC_Lathe_Alpha's maintenance performed by Apex Maintenance Services?"
)


def _print_response(response: "FinalResponse") -> None:  # type: ignore[name-defined]
    from app.schemas.models import FinalResponse  # local import for display
    console.print(Rule("[bold cyan]Final Response[/bold cyan]"))
    console.print(Panel(
        f"[bold]Tool Used:[/bold] {response.tool_used.value.upper()}\n\n"
        f"[bold]Answer:[/bold]\n{response.answer}\n\n"
        f"[bold]Confidence:[/bold] {response.confidence:.2f}\n"
        f"[bold]Tokens Used:[/bold] {response.tokens_used}\n"
        f"[bold]Latency:[/bold] {response.latency_ms:.1f}ms",
        border_style="green" if response.tool_used.value == "graph" else "magenta",
    ))

    if response.reasoning:
        console.print(Rule("[dim]ReAct Reasoning Trace[/dim]"))
        for i, step in enumerate(response.reasoning, 1):
            console.print(f"[bold cyan]Step {i}[/bold cyan]")
            console.print(f"  [yellow]Thought:[/yellow]  {step.thought}")
            console.print(f"  [blue]Action:[/blue]   {step.action}")
            console.print(f"  [green]Observation:[/green] {step.observation}\n")


async def run_interactive() -> None:
    """Interactive REPL loop."""
    console.print(Panel.fit(
        "[bold]GraphRAG Multi-Agent System[/bold]\n"
        "[dim]Manufacturing Defect Analysis | Type 'quit' to exit[/dim]",
        border_style="yellow",
    ))

    while True:
        try:
            query = console.input("\n[bold yellow]Query:[/bold yellow] ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if query.lower() in {"quit", "exit", "q"}:
            break
        if not query:
            continue

        try:
            response = await run_pipeline(query)
            metrics.record(
                tool=response.tool_used.value,
                latency_ms=response.latency_ms,
                tokens=response.tokens_used,
            )
            _print_response(response)
        except Exception as exc:
            console.print(f"[bold red]Error:[/bold red] {exc}")

    # Print session summary
    if metrics.total_calls > 0:
        console.print(Rule("[dim]Session Summary[/dim]"))
        summary = metrics.summary()
        console.print(f"  Total queries: {summary['total_calls']}")
        console.print(f"  Route distribution: {summary['route_distribution']}")
        console.print(f"  Mean latency: {summary['latency_ms']['mean']}ms")
        console.print(f"  Total tokens: {summary['tokens']['total']}")


async def run_single(query: str) -> None:
    """Run a single query and exit."""
    response = await run_pipeline(query)
    metrics.record(
        tool=response.tool_used.value,
        latency_ms=response.latency_ms,
        tokens=response.tokens_used,
    )
    _print_response(response)


def main() -> None:
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    if sys.stderr and hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass
    setup_logging()

    parser = argparse.ArgumentParser(
        description="GraphRAG Multi-Agent System — Manufacturing Defect Analysis"
    )
    parser.add_argument("--query", type=str, default=None, help="Run a single query and exit")
    parser.add_argument("--demo", action="store_true", help="Run the canonical breaking-point demo query")
    args = parser.parse_args()

    if args.demo:
        asyncio.run(run_single(DEMO_QUERY))
    elif args.query:
        asyncio.run(run_single(args.query))
    else:
        asyncio.run(run_interactive())


if __name__ == "__main__":
    main()
