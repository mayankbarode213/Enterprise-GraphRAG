"""
FAISSVectorStore — FAISS-backed vector store using LangChain.

Combines LangChain's FAISS vector store wrapper with cosine similarity.
Supports idempotent upsert, similarity search, disk persistence, and clear.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document

from app.vector.embedding import EmbeddingService

logger = logging.getLogger(__name__)


@dataclass
class VectorStoreDoc:
    """A stored document with its metadata."""
    doc_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class FAISSVectorStore:
    """
    Persistent FAISS-backed vector store built on top of LangChain's FAISS class.
    """

    def __init__(self, persist_dir: Path, dimension: int = 1536) -> None:
        self._dir = Path(persist_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._dim = dimension
        self._embedding_service = EmbeddingService()
        self._vectorstore: FAISS | None = None
        self._load()

    # ── Persistence ────────────────────────────────────────────────────────────

    def _index_file(self) -> Path:
        return self._dir / "index.faiss"

    def _load(self) -> None:
        """Load existing LangChain FAISS index from disk or create fresh."""
        if self._index_file().exists():
            try:
                self._vectorstore = FAISS.load_local(
                    folder_path=str(self._dir),
                    embeddings=self._embedding_service.embeddings,
                    allow_dangerous_deserialization=True,
                )
                logger.info(
                    "FAISSVectorStore loaded via LangChain | docs=%d",
                    self.count(),
                )
                return
            except Exception as exc:
                logger.warning("Could not load FAISS index from disk: %s. Creating fresh.", exc)

        self._create_empty_store()

    def _create_empty_store(self) -> None:
        """Initialize an empty FAISS store using IndexFlatIP (cosine similarity on unit vectors)."""
        index = faiss.IndexFlatIP(self._dim)
        docstore = InMemoryDocstore({})
        index_to_docstore_id: dict[int, str] = {}
        self._vectorstore = FAISS(
            embedding_function=self._embedding_service.embeddings,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        )
        logger.info("FAISSVectorStore created fresh | dim=%d", self._dim)

    def _save(self) -> None:
        """Persist the vector store using LangChain's save_local."""
        if self._vectorstore is not None:
            self._vectorstore.save_local(folder_path=str(self._dir))
            logger.debug("FAISSVectorStore saved | docs=%d", self.count())

    # ── CRUD Interface ─────────────────────────────────────────────────────────

    def upsert(
        self,
        doc_ids: list[str],
        embeddings: list[list[float]],
        contents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """
        Upsert documents and pre-computed embeddings into the store.
        Normalizes vectors for cosine similarity search via IndexFlatIP.
        """
        if not doc_ids:
            return

        assert self._vectorstore is not None
        metas = metadatas or [{} for _ in doc_ids]

        # Check if any doc_ids already exist in store
        existing_doc_ids = set(self._vectorstore.docstore._dict.keys())
        has_overlap = any(did in existing_doc_ids for did in doc_ids)

        if has_overlap:
            # Rebuild store preserving existing non-overlapping documents + adding updates
            new_updates = {
                did: (emb, content, meta)
                for did, emb, content, meta in zip(doc_ids, embeddings, contents, metas)
            }
            all_docs: dict[str, tuple[list[float], str, dict[str, Any]]] = {}

            # Save kept docs (Note: in LangChain docstore, content and metadata are stored)
            for did, doc in self._vectorstore.docstore._dict.items():
                if did not in new_updates:
                    # Keep existing doc text & metadata
                    # We generate embedding on demand if needed or re-embed
                    all_docs[did] = ([], doc.page_content, doc.metadata)

            # Add updates
            all_docs.update(new_updates)

            self._create_empty_store()
            assert self._vectorstore is not None

            # Re-insert all documents
            texts_to_add: list[str] = []
            ids_to_add: list[str] = []
            metas_to_add: list[dict[str, Any]] = []
            embs_to_add: list[list[float]] = []

            for did, (emb, content, meta) in all_docs.items():
                texts_to_add.append(content)
                ids_to_add.append(did)
                metas_to_add.append(meta)
                if not emb:
                    # Re-embed if missing
                    emb = self._embedding_service.embeddings.embed_query(content)
                embs_to_add.append(emb)

            vectors = np.array(embs_to_add, dtype=np.float32)
            faiss.normalize_L2(vectors)

            text_embeddings = list(zip(texts_to_add, vectors.tolist()))
            self._vectorstore.add_embeddings(
                text_embeddings=text_embeddings,
                metadatas=metas_to_add,
                ids=ids_to_add,
            )
        else:
            vectors = np.array(embeddings, dtype=np.float32)
            faiss.normalize_L2(vectors)

            text_embeddings = list(zip(contents, vectors.tolist()))
            self._vectorstore.add_embeddings(
                text_embeddings=text_embeddings,
                metadatas=metas,
                ids=doc_ids,
            )

        self._save()
        logger.info("FAISSVectorStore upserted %d documents | total=%d", len(doc_ids), self.count())

    def search(
        self,
        query_embedding: list[float],
        k: int = 5,
    ) -> list[tuple[VectorStoreDoc, float]]:
        """
        Search for the k most similar documents by query vector.
        """
        assert self._vectorstore is not None
        if self.count() == 0:
            return []

        k = min(k, self.count())
        query_vec = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_vec)

        # FAISS similarity search by vector with score
        doc_score_pairs = self._vectorstore.similarity_search_with_score_by_vector(
            embedding=query_vec[0].tolist(),
            k=k,
        )

        results: list[tuple[VectorStoreDoc, float]] = []
        for doc, score in doc_score_pairs:
            # Map LangChain Document back to VectorStoreDoc
            # Doc ID is retrieved from metadata or docstore key lookup
            doc_id = doc.metadata.get("doc_id", doc.metadata.get("source", "unknown"))
            # Find matching doc_id in docstore if available
            for stored_id, stored_doc in self._vectorstore.docstore._dict.items():
                if stored_doc.page_content == doc.page_content and stored_doc.metadata == doc.metadata:
                    doc_id = stored_id
                    break

            store_doc = VectorStoreDoc(
                doc_id=doc_id,
                content=doc.page_content,
                metadata=doc.metadata,
            )
            results.append((store_doc, float(score)))

        return results

    def count(self) -> int:
        """Return total document count in store."""
        if self._vectorstore is None or self._vectorstore.docstore is None:
            return 0
        return len(self._vectorstore.docstore._dict)

    def clear(self) -> None:
        """Clear all stored vectors and documents."""
        self._create_empty_store()
        self._save()
        logger.info("FAISSVectorStore cleared")
