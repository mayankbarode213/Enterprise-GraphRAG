# Decision Log

## DL-001: LangGraph over Haystack

**Decision**: Use LangGraph for multi-agent orchestration.

**Rationale**:
- LangGraph's `StateGraph` provides an explicit, inspectable state machine — every node's inputs and outputs are typed
- Conditional edges (`add_conditional_edges`) make the routing logic visible and testable
- Native `async` support with `ainvoke` throughout

**Why not Haystack**: Haystack is pipeline-based and less flexible for dynamic routing. Its state model is implicit. LangGraph's directed graph model maps 1:1 to the architecture diagram.

---

## DL-002: Neo4j over NetworkX

**Decision**: Use Neo4j as the graph database.

**Rationale**:
- Neo4j is the industry standard for production knowledge graphs
- Cypher is expressive and parameterized (prevents injection)
- APOC plugin adds shortest-path, graph algorithms, and procedures used in production
- Persistent storage with Docker — data survives restarts
- Neo4j Browser (port 7474) enables live visual graph exploration

**Why not NetworkX**: In-memory only, no persistence, no query language, not production-relevant.

## DL-003: FAISS over ChromaDB

**Decision**: Use FAISS (`faiss-cpu`) as the vector store.

**Rationale**:
- Pre-built Windows wheels for Python 3.13 are available (`faiss-cpu`), requiring zero C++ compiler tooling.
- Thin, self-contained, and highly portable persistent index on disk (`index.faiss`).
- High-performance cosine similarity calculations natively supported via `IndexFlatIP` on unit-normalized vectors.
- Matches requirements of the assignment spec, avoiding the heavier service client dependencies of ChromaDB.

**Why not ChromaDB**: ChromaDB adds a larger package footprint and requires more complex client-server configuration or multi-process lock management, whereas FAISS keeps the vector store lightweight and easy to bundle in a single workspace.

---

## DL-004: Async (asyncio) Architecture

**Decision**: Use `asyncio` throughout, with `asyncio.gather()` for concurrent operations.

**Rationale**:
- Assignment explicitly requires asyncio
- Graph database calls, LLM calls, and embedding calls are all I/O-bound → ideal for async
- `asyncio.gather()` in SynthesizerAgent concurrently runs: LLM synthesis + metadata collection + logging
- Measurable latency savings: without gather(), these would be sequential; with gather(), they overlap

**Demonstration point**: In the SynthesizerAgent, we fire 3 coroutines concurrently. The wall-clock latency is dominated by the slowest (LLM call, ~1-2s), while logging and metadata collection add ~0ms because they run in parallel.

---

## DL-005: Pydantic v2

**Decision**: Use Pydantic v2 for all data models and schema validation.

**Rationale**:
- Assignment explicitly requires Pydantic v2
- `model_validator` and `field_validator` allow custom guardrail logic
- `ValidationError` is raised immediately when invalid data is passed — this is the "guardrail demo" moment
- JSON serialization/deserialization is built-in
- `pydantic-settings` integrates config loading from .env files


---

## DL-006: ReAct (Reason + Act) Pattern

**Decision**: Implement ReAct pattern in the router and agent nodes.

**Rationale**:
- Assignment requires ReAct loop
- Each agent node produces a `ReasoningStep` with thought/action/observation
- The full trace is captured in `FinalResponse.reasoning` — visible and auditable
