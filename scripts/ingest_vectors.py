"""
scripts/ingest_vectors.py — One-shot FAISS ingestion script.

Usage:
    python scripts/ingest_vectors.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from app.utils.logging import setup_logging
from app.vector.ingest import ingest
from rich.console import Console
from rich.panel import Panel

console = Console()


async def main() -> None:
    setup_logging()
    console.print(Panel.fit(
        "[bold magenta]GraphRAG — Vector Store Ingestor[/bold magenta]\n"
        "Chunking documents, embedding, and loading into FAISS ...",
        border_style="magenta",
    ))

    try:
        chunk_count = await ingest()
        console.print(f"\n[bold green][SUCCESS] Ingestion complete![/bold green]")
        console.print(f"   Chunks ingested: [yellow]{chunk_count}[/yellow]")
        console.print(f"   Collection: [yellow]manufacturing_docs[/yellow]")
    except Exception as exc:
        console.print(f"\n[bold red][ERROR] Ingestion failed:[/bold red] {exc}")
        console.print("[dim]Check your OPENAI_API_KEY in .env[/dim]")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
