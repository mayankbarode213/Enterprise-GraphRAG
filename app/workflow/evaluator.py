"""
Evaluator node - compares GraphRAG and VectorRAG answers to assess accuracy, completeness, and reasoning.
"""
from __future__ import annotations

import json
import logging
from typing import Any
from langchain_openai import ChatOpenAI
from settings import settings

logger = logging.getLogger(__name__)

EVALUATOR_SYSTEM_PROMPT = """You are an expert manufacturing quality inspector and AI systems evaluator.
You will compare two answers generated for a quality engineering question: one from a GraphRAG system (Neo4j structured database) and one from a VectorRAG system (semantic text search).

CRITICAL EVALUATION PRINCIPLES:
1. **Query Fulfillment & Accuracy First**: The PRIMARY factor for winning is whether the system ACCURATELY and DIRECTLY answered the user's specific core question (e.g., if the user asks for a DATE, document finding, or specific attribute, the system that actually provides the exact date/fact wins over a system that claims 'no records found' or provides unrequested structural data).
2. **Correct Tool Choice**:
   - **VectorRAG** is the correct tool for: unstructured text findings, single-entity attribute lookups from free-text documents, or queries where the answer lies in narrative document chunks rather than graph relationships.
   - **GraphRAG** is the correct tool for: multi-hop relational dependency tracing, supplier-to-defect lineage, component-to-machine hierarchy, operational path traversals, AND **complete machine operational history queries** (e.g., "show maintenance history of Machine X including vendors, incidents, and defects"). Machine history queries inherently require traversing Machine→Maintenance→Vendor AND Machine→Maintenance←Incident→Defect — this is ALWAYS a multi-hop graph query.
   - **DO NOT** assume VectorRAG wins simply because the query mentions dates or incident logs. If the query asks for a COMPLETE HISTORY of a named machine including multiple entity types (vendors, incidents, defects), that is a multi-hop graph traversal and GraphRAG is the correct tool.
3. **Scoring Answer Completeness (1-10)**:
   - A system that fails to answer the requested metric/date MUST receive a LOW completeness score (1-4).
   - A system that directly answers the requested metric/date with precise evidence MUST receive a HIGH completeness score (8-10).
   - Do NOT award a high completeness score to a GraphRAG response just because it output a large structural report if it missed the user's explicit question.
   - **GraphRAG BONUS**: If GraphRAG correctly traverses multi-hop relationships (e.g., Machine→Maintenance→Vendor, Incident→Defect) AND provides the requested entity data (maintenance events, vendors, defects), award it a HIGH completeness score (8-10) even if the format is a structured report rather than a plain table.
4. **Selecting the Winner**:
   - If VectorRAG answered the core question correctly (e.g. found the exact date/document detail) while GraphRAG missed the answer or stated no records were found, **VectorRAG MUST be declared the Winner**.
   - If GraphRAG traced multi-hop paths accurately to answer a lineage or operational history query while VectorRAG gave incomplete or unverifiable context, **GraphRAG MUST be declared the Winner**.
   - For machine operational history queries (maintenance history + vendors + incidents + defects), GraphRAG is the structurally correct tool. If GraphRAG provides a complete multi-hop answer covering all requested entity types (maintenance events, vendors, incidents, defects), **GraphRAG MUST win** even if VectorRAG also provides a prose answer.
   - If both answered equally well, select **Tie**.
5. **Detailed & Evidentiary Reasoning (CRITICAL)**:
   - Your reasons in `"graph_reasons"` and `"vector_reasons"` MUST be specific, factual, and comparative. Avoid vague descriptions like "provided a complete trace" or "lacked relationships".
   - **Specify exact data points**: Explicitly name the missing batches (e.g. `BAT_HS_2026_002`), suppliers, components, or machine IDs that one system retrieved but the other missed.
   - **Expose structural blind spots**: If VectorRAG failed, explain *why* (e.g. "Failed because the causal connection between Defect X and Batch Y is split across separate documents and not present in any single text segment, making semantic retrieval impossible").
   - **Quantify the complexity**: Mention the exact number of hops or unique relationship types successfully traversed by GraphRAG (e.g. "successfully traversed 8 hops starting from the root defect back to the originating supplier, fanning out to active batch BAT_HS_2026_002").

Your response MUST be a JSON object matching this structure:
{
  "winner": "GraphRAG" | "VectorRAG" | "Tie",
  "graph_reasons": [
    "Reason 1 (Must be highly specific, naming nodes/hops)",
    "Reason 2"
  ],
  "vector_reasons": [
    "Reason 1 (Must explain exact missing facts or structural document boundaries)",
    "Reason 2"
  ],
  "graph_eval": {
    "correct_tool": boolean,
    "completeness_score": number (1-10),
    "multihop_support": boolean,
    "explainability_score": number (1-10)
  },
  "vector_eval": {
    "correct_tool": boolean,
    "completeness_score": number (1-10),
    "multihop_support": boolean,
    "explainability_score": number (1-10)
  }
}
Do not add any markdown block wrapping like ```json or prefix text. Output only raw JSON.
"""

class RAGEvaluator:
    def __init__(self) -> None:
        self._llm = ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            temperature=0.1,
        )

    async def evaluate(self, query: str, graph_answer: str, vector_answer: str) -> dict[str, Any]:
        """
        Evaluate and compare GraphRAG vs VectorRAG answers.
        """
        user_message = f"""Query: {query}

                        --- GraphRAG Answer ---
                        {graph_answer}

                        --- VectorRAG Answer ---
                        {vector_answer}
                        """
        try:
            response = await self._llm.ainvoke(
                [
                    {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ]
            )
            # Remove any possible code block formatting
            text = response.content.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].startswith("```"):
                    lines = lines[:-1]
                text = "\n".join(lines).strip()
            
            return json.loads(text)
        except Exception as e:
            logger.error("Failed to run evaluation LLM: %s", e)
            return {
                "winner": "Tie",
                "graph_reasons": ["Failed to generate reasons due to an evaluation error."],
                "vector_reasons": ["Failed to generate reasons due to an evaluation error."],
                "graph_eval": {
                    "correct_tool": True,
                    "completeness_score": 5,
                    "multihop_support": False,
                    "explainability_score": 5
                },
                "vector_eval": {
                    "correct_tool": True,
                    "completeness_score": 5,
                    "multihop_support": False,
                    "explainability_score": 5
                }
            }
