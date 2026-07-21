"""
scripts/load_graph.py — One-shot Neo4j data loader.

Usage:
    python scripts/load_graph.py
"""
import asyncio
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parents[1]))

from app.graph.loader import load_all
from app.utils.logging import setup_logging
from rich.console import Console
from rich.panel import Panel

console = Console()


async def main() -> None:
    setup_logging()
    console.print(Panel.fit(
        "[bold cyan]GraphRAG — Neo4j Data Loader[/bold cyan]\n"
        "Loading manufacturing dataset into Neo4j ...",
        border_style="cyan",
    ))

    try:
        result = await load_all()
        console.print(f"\n[bold green][SUCCESS] Load complete![/bold green]")
        console.print(f"   Nodes loaded       : [yellow]{result['nodes']}[/yellow]")
        console.print(f"   Relationships loaded: [yellow]{result['relationships']}[/yellow]")
        console.print("\n[dim]Neo4j Browser: http://localhost:7474[/dim]")
    except Exception as exc:
        console.print(f"\n[bold red][ERROR] Load failed:[/bold red] {exc}")
        console.print("[dim]Is Neo4j running? Try: docker-compose up -d neo4j[/dim]")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
