"""
FastAPI Server for Enterprise GraphRAG.

Provides REST API endpoints for:
- Chat query processing (/api/query)
- Serving frontend static files
"""
from __future__ import annotations

import os
import sys
import json
import asyncio
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rich.console import Console
from rich.rule import Rule

from app.workflow.graph import run_pipeline
from app.workflow.evaluator import RAGEvaluator
from app.utils.logging import setup_logging
from app.schemas.models import FinalResponse

console = Console()

# Enable UTF-8 on Windows
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

setup_logging()

app = FastAPI(title="Enterprise GraphRAG")

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")

app.mount(
    "/static",
    StaticFiles(directory=FRONTEND_DIR),
    name="static",
)


@app.get("/")
async def index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


class QueryRequest(BaseModel):
    query: str
    mode: str = "auto"
    use_text2cypher: bool = False


def _print_reasoning_trace(response: Any, prefix: str = "") -> None:
    """Print structured ReAct reasoning steps to the server stdout console."""
    reasoning = getattr(response, "reasoning", None)
    if reasoning is None and isinstance(response, dict):
        reasoning = response.get("reasoning", [])
        
    if not reasoning:
        return
        
    title = f"{prefix} ReAct Reasoning Trace" if prefix else "ReAct Reasoning Trace"
    console.print(Rule(f"[dim]{title}[/dim]"))
    
    for i, step in enumerate(reasoning, 1):
        thought = getattr(step, "thought", None)
        if thought is None and isinstance(step, dict):
            thought = step.get("thought", "")
        action = getattr(step, "action", None)
        if action is None and isinstance(step, dict):
            action = step.get("action", "")
        observation = getattr(step, "observation", None)
        if observation is None and isinstance(step, dict):
            observation = step.get("observation", "")
            
        console.print(f"[bold cyan]Step {i}[/bold cyan]")
        console.print(f"  [yellow]Thought:[/yellow]  {thought}")
        console.print(f"  [blue]Action:[/blue]   {action}")
        console.print(f"  [green]Observation:[/green] {observation}\n")


@app.post("/api/query")
async def query(req: QueryRequest):
    print("\n" + "=" * 60)
    print(f"📥 [SERVER RECEIVED /api/query] mode={req.mode} | use_text2cypher={req.use_text2cypher}")
    print("=" * 60 + "\n")

    if req.mode == "compare":
        graph_task = asyncio.create_task(
            run_pipeline(req.query, forced_mode="graph", use_text2cypher=req.use_text2cypher)
        )
        vector_task = asyncio.create_task(
            run_pipeline(req.query, forced_mode="vector", use_text2cypher=req.use_text2cypher)
        )

        graph_res, vector_res = await asyncio.gather(
            graph_task,
            vector_task,
            return_exceptions=True,
        )

        graph_res_data = None
        if isinstance(graph_res, Exception):
            graph_res_data = {"error": str(graph_res)}
        else:
            _print_reasoning_trace(graph_res, prefix="[Graph RAG]")
            graph_res_data = graph_res.model_dump()

        vector_res_data = None
        if isinstance(vector_res, Exception):
            vector_res_data = {"error": str(vector_res)}
        else:
            _print_reasoning_trace(vector_res, prefix="[Vector RAG]")
            vector_res_data = vector_res.model_dump()

        evaluation = None

        if not isinstance(graph_res, Exception) and not isinstance(vector_res, Exception):
            is_graph_reject = getattr(graph_res, "tool_used", None) == "reject"
            is_vector_reject = getattr(vector_res, "tool_used", None) == "reject"

            if is_graph_reject or is_vector_reject:
                evaluation = {
                    "winner": "N/A (Rejected)",
                    "graph_eval": {
                        "correct_tool": False,
                        "completeness_score": 0,
                        "multihop_support": False,
                        "explainability_score": 0,
                    },
                    "vector_eval": {
                        "correct_tool": False,
                        "completeness_score": 0,
                        "multihop_support": False,
                        "explainability_score": 0,
                    },
                    "graph_reasons": [
                        "Query was rejected as out-of-domain by Input Guardrails.",
                        "No Neo4j database traversal was executed."
                    ],
                    "vector_reasons": [
                        "Query was rejected as out-of-domain by Input Guardrails.",
                        "No FAISS semantic search was executed."
                    ]
                }
            else:
                evaluator = RAGEvaluator()
                evaluation = await evaluator.evaluate(
                    req.query,
                    graph_res.answer,
                    vector_res.answer,
                )

        return {
            "compare": True,
            "graph": graph_res_data,
            "vector": vector_res_data,
            "evaluation": evaluation,
        }

    forced = req.mode if req.mode in ("graph", "vector") else None

    result = await run_pipeline(
        req.query,
        forced_mode=forced,
        use_text2cypher=req.use_text2cypher,
    )

    _print_reasoning_trace(result)

    return result.model_dump()