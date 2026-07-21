# GraphRAG vs Vector RAG — Multi-Agent PoC

> **Manufacturing Defect Analysis** | LangGraph · Neo4j · FAISS · OpenAI · Pydantic v2 · asyncio

---

## 🎯 What This Demonstrates

A production-quality proof of concept showing **exactly where Vector RAG breaks** and **why GraphRAG succeeds** — driven by a multi-agent system with a LangGraph ReAct loop.

### The Breaking-Point Query

> *"Which supplier batches indirectly caused all defects reported after Robotic Welding Cell RW-101's maintenance performed by Apex Industrial Services Pvt. Ltd.?"*

This query requires a **5-hop relationship traversal**:

```
Supplier → Batch → Component → Machine → Maintenance → Vendor → Incident → Defect
```

**Vector RAG fails** because the answer is fragmented across 6+ documents. No single chunk contains the full causal chain. Semantic similarity ≠ relationship traversal.

**GraphRAG succeeds** because it traverses typed edges directly in Neo4j with a parameterized Cypher query.

---

## 🏗️ Architecture

```
                        ┌─────────────────┐
                        │  User Query     │
                        └────────┬────────┘
                                 │
                        ┌────────▼────────┐
                        │  Router Agent   │  ← LLM decides (ReAct)
                        │  (GPT-4o-mini)  │    → RoutingDecision (Pydantic)
                        └────────┬────────┘
                    ┌────────────┤
                    ▼            ▼
          ┌──────────────┐  ┌──────────────┐
          │ Graph Agent  │  │ Vector Agent │
          │  (Neo4j)     │  │    (FAISS)   │
          └──────┬───────┘  └──────┬───────┘
                 └──────────┬──────┘
                            ▼
                  ┌─────────────────┐
                  │  Synthesizer    │  ← asyncio.gather()
                  │  (GPT-4o-mini)  │    concurrent LLM + logging + metrics
                  └────────┬────────┘
                           ▼
                  ┌─────────────────┐
                  │  Guardrail      │  ← Pydantic v2 FinalResponse schema
                  │  (Pydantic v2)  │    raises ValidationError on bad output
                  └────────┬────────┘
                           ▼
                  ┌─────────────────┐
                  │  Final Response │
                  └─────────────────┘
```

---

## 📁 Project Structure

```
Enterprise-GraphRAG/
├── app/
│   ├── agents/         # Router, GraphAgent, VectorAgent, Synthesizer, Guardrail
│   ├── graph/          # Neo4j client, schema, loader, queries, retriever, service
│   ├── vector/         # Chunker, embedding, store (FAISS), ingest, retriever
│   ├── workflow/       # LangGraph StateGraph + state definition
│   ├── schemas/        # All Pydantic v2 models
│   ├── prompts/        # System prompts for router and synthesizer
│   ├── services/       # SemanticCache stub
│   ├── telemetry/      # In-memory metrics collector
│   └── utils/          # Structured JSON logging, async utilities
├── data/
│   ├── entities.csv    # 12 graph nodes (Machine, Supplier, Batch, Defect …)
│   ├── relationships.csv  # 23 typed edges
│   └── documents/      # 12 natural-language documents for Vector RAG
├── ground_truth/
│   └── answers.json    # Ground truth for evaluation harness
├── evaluation/
│   ├── evaluate.py     # Runs both systems, produces comparison.json + .md
│   ├── metrics.py      # Precision, recall, F1, entity extraction
│   └── ground_truth.py # Ground truth loader
├── scripts/
│   ├── load_graph.py   # One-shot Neo4j data loader
│   ├── ingest_vectors.py  # One-shot FAISS ingestion
│   └── compare.py      # Side-by-side head-to-head demo
├── tests/              # pytest unit + integration tests
├── docs/               # Design docs, breaking point analysis, decision log
├── docker/Dockerfile
├── docker-compose.yml
├── main.py             # Interactive CLI entry point
├── server.py           # FastAPI REST API server
├── settings.py         # pydantic-settings config
└── requirements.txt
```

---

## ⚡ Quick Start

### 1. Prerequisites
- Python 3.11+
- Docker + Docker Compose
- OpenAI API key

### 2. Clone & Setup
```bash
git clone <your-repo-url>
cd Enterprise-GraphRAG

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
```

### 3. Configure Environment
```bash
copy .env.example .env
# Edit .env and set your OPENAI_API_KEY
```

### 4. Start Neo4j
```bash
docker-compose up -d neo4j
# Wait ~30 seconds for Neo4j to be ready
# Browser UI: http://localhost:7474 (neo4j/password)
```

### 5. Load Data
```bash
# Load graph into Neo4j
python scripts/load_graph.py

# Embed documents into FAISS
python scripts/ingest_vectors.py
```

### 6. Run the Demo
```bash
# Head-to-head comparison (canonical breaking-point query)
python scripts/compare.py

# Run evaluation harness (produces comparison.md)
python evaluation/evaluate.py

# Interactive agent
python main.py

# Single demo query
python main.py --demo
```

### 7. Run Tests
```bash
# Unit tests (no Neo4j/FAISS needed)
pytest tests/ -v -m "not integration"

# All tests (requires running Neo4j + loaded FAISS)
pytest tests/ -v
```

---

### 7. Launch the Interactive API

Start the FastAPI server:
```bash
python -m uvicorn server:app --port 8000
```
or choose any available port:

```bash
python -m uvicorn server:app --port <PORT>
```

Example:

```bash
python -m uvicorn server:app --port 8080
```

Once the server starts:

#### Interactive UI: http://localhost:8000

#### Swagger API Docs: http://localhost:8000/docs

#### ReDoc Documentation: http://localhost:8000/redoc

If using a different port, replace 8000 with your chosen port.

### 8. Run the Head-to-Head Demo
```bash
python scripts/compare.py
```

### 9. Run Evaluation
```bash
python evaluation/evaluate.py
```

Outputs:
```
evaluation/results/
├── comparison.json
└── comparison.md
```

### 10. Run Interactive CLI
```bash
python main.py
```

Single canonical demo:
```bash
python main.py --demo
```
### 11. Run Tests
```bash
# Unit tests (no Neo4j/FAISS needed)
pytest tests/ -v -m "not integration"
```

Full suite:
```bash
pytest tests/ -v
```

## 💻 Available Interfaces

| Interface | Command | Purpose |
|-----------|---------|---------|
| **Interactive CLI** | `python main.py` | Chat with the multi-agent system |
| **Demo Query** | `python main.py --demo` | Runs the canonical GraphRAG vs VectorRAG comparison |
| **FastAPI Server** | `python -m uvicorn server:app --port 8000` | Launches the REST API with interactive Swagger UI |
| **Comparison Script** | `python scripts/compare.py` | Runs a side-by-side GraphRAG vs VectorRAG comparison |
| **Evaluation Harness** | `python evaluation/evaluate.py` | Executes the evaluation suite and generates benchmark metrics (`comparison.json` and `comparison.md`) |


## 🔑 Key Technical Highlights

### Pydantic v2 Guardrails
Every LLM output is validated against a strict schema before being passed downstream:
```python
# This WILL raise ValidationError 
FinalResponse(
    answer="",       # min_length=10 → FAIL
    confidence=1.5,  # le=1.0 → FAIL
    tokens_used=-1,  # ge=0 → FAIL
)
```

### asyncio.gather() in SynthesizerAgent
```python
llm_response, metadata, _ = await asyncio.gather(
    _call_llm(),         # ~1-2s LLM call
    _collect_metadata(), # ~0ms concurrent
    _log_trace(),        # ~0ms concurrent
)
```

### LangGraph Conditional Routing
```
START → router_node → [graph_node | vector_node] → synthesizer → guardrail → END
```
The router's `RoutingDecision.tool` field determines the conditional edge.

### Parameterized Cypher (never string interpolation)
```cypher
MATCH (sup:Supplier)-[:SUPPLIED_BATCH]->(b:Batch)-[:MANUFACTURED]->...
WHERE m.name = $machine_name AND v.name = $vendor_name
```

---


## 🏭 Production Readiness Notes

| Concern | Current (PoC) | Production |
|---------|--------------|------------|
| Observability | In-memory metrics | LangSmith + Prometheus + Grafana |
| Tracing | JSON log file | OpenTelemetry spans |
| Caching | In-memory dict | Redis semantic cache (embed → cosine search) |
| Graph | Local Neo4j | Neo4j AuraDB / Enterprise |
| Vector store | FAISS | Pre-built Python 3.13 wheels, no C++ compiler needed |
| LLM | GPT-4o-mini | GPT-4o / fine-tuned model |
| Cost tracking | Token count × rate | LangSmith cost tracking |
| Latency | ~5-8s | <500ms with caching |

---

## 📖 Documentation

- [Design Document](docs/design.md) — Problem statement, hypothesis, tech choices
- [Breaking Point Analysis](docs/breaking_point.md) — Why Vector RAG fails (detailed)
- [Decision Log](docs/decision_log.md) — Rationale for every architectural choice
