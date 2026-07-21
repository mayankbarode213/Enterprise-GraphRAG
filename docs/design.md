# Design Document — GraphRAG vs Vector RAG PoC

## Problem Statement

Modern RAG (Retrieval-Augmented Generation) systems typically rely on vector similarity search over chunked documents. While this works well for semantic lookup within a single document or topic, it fundamentally fails when answering questions that require **multi-hop relationship traversal** — i.e., connecting facts distributed across multiple entities through a chain of typed relationships.

This PoC demonstrates this failure mode in a manufacturing defect analysis domain.

## Hypothesis

> **GraphRAG outperforms Vector RAG for queries requiring multi-hop relationship traversal.**

Specifically: any query that asks "what caused what, through which entities" cannot be reliably answered by vector similarity search because:
1. The answer is fragmented across multiple documents/chunks
2. No single chunk contains the complete causal chain
3. Semantic similarity ≠ logical/relational connectivity

## Success Criteria

| Metric | Target |
|--------|--------|
| GraphRAG recall on canonical query | ≥ 90% |
| VectorRAG recall on canonical query | ≤ 40% |
| GraphRAG F1 > VectorRAG F1 | By ≥ 50 points |
| Pipeline latency (graph path) | < 3 seconds |
| All Pydantic schemas validated | 100% |
| Unit tests pass | 100% |

## Technology Choices

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Agent orchestration | LangGraph | Native async, explicit state machine, conditional edges |
| Graph database | Neo4j | Industry standard, Cypher is expressive, APOC plugins |
| Vector store | FAISS | Persistent local index on disk, cosine similarity, no server process needed |
| LLM | OpenAI GPT-4o-mini | Cost-effective, JSON mode support, fast |
| Embeddings | text-embedding-3-small | High quality, low cost |
| Schema validation | Pydantic v2 | Strict, fast, self-documenting |
| Config | pydantic-settings | Type-safe env var loading |
| Concurrency | asyncio | Native Python async, gather() for concurrency |

## Alternatives Considered

| Decision | Alternative | Why Rejected |
|----------|-------------|--------------|
| LangGraph | Haystack | LangGraph's StateGraph gives explicit control over conditional routing |
| Neo4j | NetworkX | Neo4j is production-grade, persistent, has APOC; NetworkX is in-memory only |
| FAISS | ChromaDB | ChromaDB adds a larger dependency footprint and locks, whereas FAISS keeps the store lightweight and easy to bundle |
| GPT-4o-mini | Claude/Gemini | OpenAI JSON mode is stable; cost-effective for demos |

## Trade-offs

**GraphRAG**
- ✅ Precise multi-hop traversal
- ✅ Explainable (Cypher shows exact path)
- ❌ Requires schema design upfront
- ❌ Cold start: data must be modelled and loaded

**Vector RAG**
- ✅ Zero schema design
- ✅ Works out-of-the-box on any documents
- ❌ Fails on relational/causal queries
- ❌ Hallucination risk when context is fragmented

## Assumptions

1. The domain dataset is small (≤50 nodes) — sufficient for PoC demonstration
2. OpenAI API access is available
3. Docker is available for Neo4j
4. The canonical breaking-point query is fixed and representative

## Future Improvements

1. **Hybrid retrieval**: Use vector search to identify the most relevant graph subgraph, then Cypher to traverse it
2. **LLM → Cypher**: Replace parameterized Cypher with LLM-generated queries for arbitrary questions
3. **Knowledge graph auto-construction**: Use NLP/LLM to extract entities and relationships from unstructured text automatically
4. **Multi-modal**: Add image-based defect detection linked to the graph
5. **Production LangSmith**: Replace in-memory metrics with LangSmith tracing
