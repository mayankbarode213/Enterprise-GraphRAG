"""
cypher_guardrail.py — Security & Syntax Guardrail for Text-to-Cypher

Enforces strict read-only permissions and sanitizes LLM-generated Cypher queries
before execution on Neo4j.
"""
import re
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# List of forbidden Cypher mutation/destructive keywords
FORBIDDEN_KEYWORDS = [
    r"\bCREATE\b",
    r"\bMERGE\b",
    r"\bDELETE\b",
    r"\bDETACH\b",
    r"\bSET\b",
    r"\bREMOVE\b",
    r"\bDROP\b",
    r"\bALTER\b",
    r"\bGRANT\b",
    r"\bREVOKE\b",
    r"\bCALL\s+dbms\b",
]


class CypherGuardrail:
    """Validates and sanitizes LLM-generated Cypher queries."""

    @classmethod
    def validate_and_sanitize(cls, cypher: str) -> Tuple[bool, str, str]:
        """
        Validate Cypher query safety and syntax.

        Returns:
            (is_valid, sanitized_cypher, error_message)
        """
        clean_cypher = cypher.strip()

        # 1. Check for Forbidden Write / Destructive Mutations
        for pattern in FORBIDDEN_KEYWORDS:
            if re.search(pattern, clean_cypher, re.IGNORECASE):
                msg = f"Forbidden mutation operation detected in Cypher query: {pattern}"
                logger.warning("CypherGuardrail BLOCKED: %s", msg)
                return False, "", msg

        # 2. Enforce READ-ONLY Start (Must contain MATCH or RETURN)
        if not re.search(r"\b(MATCH|OPTIONAL MATCH|WITH|RETURN)\b", clean_cypher, re.IGNORECASE):
            msg = "Query does not contain valid Cypher read clauses (MATCH / RETURN)."
            logger.warning("CypherGuardrail BLOCKED: %s", msg)
            return False, "", msg

        # 3. Sanitize invalid subquery MATCH inside WHERE clauses:
        if re.search(r"WHERE.*?\(\s*MATCH", clean_cypher, re.IGNORECASE | re.DOTALL):
            logger.info("CypherGuardrail: Fixing invalid inline subquery in WHERE clause.")
            clean_cypher = re.sub(
                r"\s*AND\s+[\w\.]+\s*>\s*\(\s*MATCH.*?\)",
                "",
                clean_cypher,
                flags=re.IGNORECASE | re.DOTALL
            )

        # 4. Fix duplicate/multi-WHERE syntax errors: e.g. "WHERE ... WHERE ..." -> "WHERE ... AND ..."
        while len(re.findall(r"\bWHERE\b", clean_cypher, re.IGNORECASE)) > 1:
            logger.info("CypherGuardrail: Replacing duplicate WHERE clause with AND.")
            clean_cypher = re.sub(
                r"(\bWHERE\b.*?)\s+\bWHERE\b",
                r"\1 AND",
                clean_cypher,
                flags=re.IGNORECASE | re.DOTALL,
                count=1
            )

        logger.info("CypherGuardrail PASSED cleanly.")
        return True, clean_cypher, ""
