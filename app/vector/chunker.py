"""
TextChunker — splits documents into overlapping chunks for vector ingestion using LangChain.

Strategy: character-based splitting with token awareness via tiktoken & LangChain Text Splitter.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from settings import settings

logger = logging.getLogger(__name__)

_ENCODING = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    """A single document chunk ready for embedding."""

    chunk_id: str
    source: str          # original document filename
    content: str
    token_count: int
    chunk_index: int


def _count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def chunk_text(
    text: str,
    source: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    """
    Split text into overlapping chunks using LangChain's RecursiveCharacterTextSplitter.

    Args:
        text: Raw document text.
        source: Document identifier (filename).
        chunk_size: Max tokens per chunk (defaults to settings.chunk_size).
        overlap: Overlap tokens between chunks (defaults to settings.chunk_overlap).

    Returns:
        List of Chunk objects.
    """
    size = chunk_size or settings.chunk_size
    ovlp = overlap or settings.chunk_overlap

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=size,
        chunk_overlap=ovlp,
    )

    split_texts = splitter.split_text(text)
    chunks: list[Chunk] = []

    for idx, content in enumerate(split_texts):
        cleaned_content = content.strip()
        if not cleaned_content:
            continue
        token_count = _count_tokens(cleaned_content)
        chunks.append(
            Chunk(
                chunk_id=f"{source}::chunk_{idx}",
                source=source,
                content=cleaned_content,
                token_count=token_count,
                chunk_index=idx,
            )
        )

    logger.debug("Chunked '%s' → %d chunks", source, len(chunks))
    return chunks


def chunk_documents(documents_dir: Path) -> list[Chunk]:
    """
    Chunk all .txt files in a directory.

    Returns:
        Flat list of all chunks from all documents.
    """
    all_chunks: list[Chunk] = []
    txt_files = sorted(documents_dir.glob("*.txt"))

    if not txt_files:
        logger.warning("No .txt files found in %s", documents_dir)
        return all_chunks

    for path in txt_files:
        text = path.read_text(encoding="utf-8")
        chunks = chunk_text(text, source=path.name)
        all_chunks.extend(chunks)
        logger.info("'%s' → %d chunk(s)", path.name, len(chunks))

    logger.info("Total chunks from %d documents: %d", len(txt_files), len(all_chunks))
    return all_chunks
