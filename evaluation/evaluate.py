"""
Evaluation harness — runs both retrievers against ground truth queries
and produces a comparison report (JSON + Markdown).

Run:
    python evaluation/evaluate.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from app.graph.service import GraphService
from app.utils.logging import setup_logging
from app.vector.retriever import VectorRetriever
from evaluation.ground_truth import get_breaking_point_query, get_supporting_queries, load_ground_truth
from evaluation.metrics import (
    compute_entity_recall,
    compute_f1,
    compute_precision,
    extract_entity_ids,
    is_correct,
)

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"


def _expected_ids(query_data: dict[str, Any]) -> list[str]:
    """Extract expected entity IDs from ground truth query data."""
    expected = query_data.get("expected_answer", {})
    ids: list[str] = []

    # Breaking point query has supplier_batches list
    for batch in expected.get("supplier_batches", []):
        ids.extend(batch.get("defects_caused", []))
        if batch_id := batch.get("batch_id"):
            ids.append(batch_id)
        if sid := batch.get("supplier_id"):
            ids.append(sid)

    # Supporting queries
    for key in ["machines", "defects", "batches", "failed_batches"]:
        for val in expected.get(key, []):
            if isinstance(val, str):
                ids.append(val)
            elif isinstance(val, dict) and "batch_id" in val:
                ids.append(val["batch_id"])

    # Clean and extract actual IDs from the list of strings
    clean_ids = set()
    for item in ids:
        found = extract_entity_ids(item)
        if found:
            clean_ids.update(found)
        else:
            clean_ids.add(item)
    return list(clean_ids)


async def evaluate_query(
    question: str,
    expected_ids: list[str],
    graph_service: GraphService,
    vector_retriever: VectorRetriever,
) -> dict[str, Any]:
    """Run one query through both systems and collect metrics."""

    # ── Graph retrieval ────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    graph_result = await graph_service.query(question)
    graph_latency = (time.perf_counter() - t0) * 1000

    graph_ids = extract_entity_ids(graph_result.answer)
    graph_recall = compute_entity_recall(expected_ids, graph_result.answer)
    graph_precision = compute_precision(expected_ids, graph_result.answer)
    graph_f1 = compute_f1(graph_precision, graph_recall)

    # ── Vector retrieval ───────────────────────────────────────────────────────
    t1 = time.perf_counter()
    vector_result = await vector_retriever.retrieve(question)
    vector_latency = (time.perf_counter() - t1) * 1000

    vector_recall = compute_entity_recall(expected_ids, vector_result.answer)
    vector_precision = compute_precision(expected_ids, vector_result.answer)
    vector_f1 = compute_f1(vector_precision, vector_recall)

    return {
        "question": question,
        "expected_entity_ids": expected_ids,
        "graph": {
            "answer": graph_result.answer[:500],
            "entity_ids_found": sorted(graph_ids),
            "recall": round(graph_recall, 3),
            "precision": round(graph_precision, 3),
            "f1": round(graph_f1, 3),
            "correct": is_correct(expected_ids, graph_result.answer),
            "latency_ms": round(graph_latency, 1),
            "tokens": graph_result.tokens_used,
        },
        "vector": {
            "answer": vector_result.answer[:500],
            "entity_ids_found": sorted(extract_entity_ids(vector_result.answer)),
            "recall": round(vector_recall, 3),
            "precision": round(vector_precision, 3),
            "f1": round(vector_f1, 3),
            "correct": is_correct(expected_ids, vector_result.answer),
            "latency_ms": round(vector_latency, 1),
            "tokens": vector_result.tokens_used,
        },
        "winner": "graph" if graph_f1 >= vector_f1 else "vector",
    }


def _build_markdown_report(results: list[dict[str, Any]]) -> str:
    """Build a human-readable Markdown comparison report."""
    lines = [
        "# GraphRAG vs Vector RAG — Evaluation Report",
        "",
        "## Summary",
        "",
        "| Query | Graph F1 | Vector F1 | Graph Correct | Vector Correct | Winner |",
        "|-------|----------|-----------|---------------|----------------|--------|",
    ]

    total_graph_correct = 0
    total_vector_correct = 0

    for r in results:
        q_short = r["question"][:60] + ("…" if len(r["question"]) > 60 else "")
        g = r["graph"]
        v = r["vector"]
        winner = "🏆 Graph" if r["winner"] == "graph" else "Vector"
        lines.append(
            f"| {q_short} | {g['f1']:.3f} | {v['f1']:.3f} | "
            f"{'✅' if g['correct'] else '❌'} | "
            f"{'✅' if v['correct'] else '❌'} | {winner} |"
        )
        if g["correct"]:
            total_graph_correct += 1
        if v["correct"]:
            total_vector_correct += 1

    n = len(results)
    lines += [
        "",
        "## Overall Results",
        "",
        f"- **GraphRAG**: {total_graph_correct}/{n} correct ({total_graph_correct/n*100:.0f}%)",
        f"- **VectorRAG**: {total_vector_correct}/{n} correct ({total_vector_correct/n*100:.0f}%)",
        "",
        "## Detailed Results",
        "",
    ]

    for i, r in enumerate(results, 1):
        g = r["graph"]
        v = r["vector"]
        lines += [
            f"### Query {i}: {r['question'][:80]}",
            "",
            f"**Expected entities**: `{', '.join(r['expected_entity_ids'])}`",
            "",
            "#### Graph RAG",
            f"- **Correct**: {'✅ YES' if g['correct'] else '❌ NO'}",
            f"- **Recall**: {g['recall']:.3f} | **Precision**: {g['precision']:.3f} | **F1**: {g['f1']:.3f}",
            f"- **Entities found**: `{', '.join(g['entity_ids_found']) or 'none'}`",
            f"- **Latency**: {g['latency_ms']:.1f}ms",
            f"- **Answer snippet**: _{g['answer'][:200]}_",
            "",
            "#### Vector RAG",
            f"- **Correct**: {'✅ YES' if v['correct'] else '❌ NO'}",
            f"- **Recall**: {v['recall']:.3f} | **Precision**: {v['precision']:.3f} | **F1**: {v['f1']:.3f}",
            f"- **Entities found**: `{', '.join(v['entity_ids_found']) or 'none'}`",
            f"- **Latency**: {v['latency_ms']:.1f}ms",
            f"- **Answer snippet**: _{v['answer'][:200]}_",
            "",
            "---",
            "",
        ]

    return "\n".join(lines)


async def run_evaluation() -> None:
    """Main evaluation runner."""
    setup_logging()
    logger.info("Starting evaluation harness …")

    gt = load_ground_truth()
    bp = get_breaking_point_query(gt)
    supporting = get_supporting_queries(gt)

    queries = [(bp["question"], _expected_ids(bp))]
    for sq in supporting:
        queries.append((sq["question"], _expected_ids(sq)))

    graph_service = GraphService()
    vector_retriever = VectorRetriever()

    results: list[dict[str, Any]] = []
    for question, expected_ids in queries:
        logger.info("Evaluating: %s", question[:80])
        try:
            row = await evaluate_query(question, expected_ids, graph_service, vector_retriever)
            results.append(row)
        except Exception as exc:
            logger.error("Evaluation failed for '%s': %s", question[:60], exc)

    # Save JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved JSON results to %s", json_path)

    # Save Markdown
    md_path = RESULTS_DIR / "comparison.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_build_markdown_report(results))
    logger.info("Saved Markdown report to %s", md_path)

    print(f"\n[SUCCESS] Evaluation complete. Results saved to {RESULTS_DIR}/")
    print(f"   JSON : {json_path}")
    print(f"   Report: {md_path}")


if __name__ == "__main__":
    asyncio.run(run_evaluation())
