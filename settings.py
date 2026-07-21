"""
Global application settings — loaded from environment variables / .env file.
Uses pydantic-settings v2 for type-safe configuration.
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── OpenAI ────────────────────────────────────────────────────────────────
    openai_api_key: str = "mock-key"
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # ── Neo4j ─────────────────────────────────────────────────────────────────
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    # ── Vector Store (FAISS) ──────────────────────────────────────────────────
    vector_persist_dir: str = "./vector_store"   # kept for backward compat (used as FAISS persist dir)
    vector_collection_name: str = "manufacturing_docs"

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_file: str = "./logs/agent.jsonl"

    # ── Evaluation ────────────────────────────────────────────────────────────
    ground_truth_path: str = "./ground_truth/answers.json"

    # ── Retrieval ─────────────────────────────────────────────────────────────
    vector_top_k: int = 3
    chunk_size: int = 256
    chunk_overlap: int = 50

    # ── Derived helpers ───────────────────────────────────────────────────────
    @property
    def log_file_path(self) -> Path:
        p = Path(self.log_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def vector_dir_path(self) -> Path:
        """Path to the FAISS vector store directory (kept as vector_dir_path for compatibility)."""
        p = Path(self.vector_persist_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


# Singleton — import this everywhere
settings = Settings()
