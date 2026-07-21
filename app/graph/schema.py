"""
Neo4j schema: constraints, indexes, and label / relationship constants.
Run schema.apply_schema(client) once at startup before loading data.
"""
from __future__ import annotations

import logging

from app.graph.client import Neo4jClient

logger = logging.getLogger(__name__)

# ── Label constants ────────────────────────────────────────────────────────────
LABELS = {
    "Machine": "Machine",
    "Component": "Component",
    "Supplier": "Supplier",
    "Batch": "Batch",
    "Maintenance": "Maintenance",
    "Vendor": "Vendor",
    "Operator": "Operator",
    "Incident": "Incident",
    "Defect": "Defect",
    "ProductionLine": "ProductionLine",
}

# ── Relationship type constants ────────────────────────────────────────────────
REL_TYPES = {
    "SUPPLIED_BATCH": "SUPPLIED_BATCH",
    "MANUFACTURED": "MANUFACTURED",
    "INSTALLED_ON": "INSTALLED_ON",
    "UNDERWENT": "UNDERWENT",
    "PERFORMED_BY": "PERFORMED_BY",
    "RESULTED_IN": "RESULTED_IN",
    "REPORTED_AS": "REPORTED_AS",
    "LOCATED_IN": "LOCATED_IN",
    "OPERATES": "OPERATES",
}

# ── DDL statements ─────────────────────────────────────────────────────────────
_CONSTRAINTS: list[str] = [
    "CREATE CONSTRAINT machine_id IF NOT EXISTS FOR (n:Machine) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT component_id IF NOT EXISTS FOR (n:Component) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT supplier_id IF NOT EXISTS FOR (n:Supplier) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT batch_id IF NOT EXISTS FOR (n:Batch) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT maintenance_id IF NOT EXISTS FOR (n:Maintenance) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT vendor_id IF NOT EXISTS FOR (n:Vendor) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT operator_id IF NOT EXISTS FOR (n:Operator) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT incident_id IF NOT EXISTS FOR (n:Incident) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT defect_id IF NOT EXISTS FOR (n:Defect) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT production_line_id IF NOT EXISTS FOR (n:ProductionLine) REQUIRE n.id IS UNIQUE",
]

_INDEXES: list[str] = [
    "CREATE INDEX machine_name IF NOT EXISTS FOR (n:Machine) ON (n.name)",
    "CREATE INDEX supplier_name IF NOT EXISTS FOR (n:Supplier) ON (n.name)",
    "CREATE INDEX vendor_name IF NOT EXISTS FOR (n:Vendor) ON (n.name)",
    "CREATE INDEX batch_name IF NOT EXISTS FOR (n:Batch) ON (n.name)",
    "CREATE INDEX defect_type IF NOT EXISTS FOR (n:Defect) ON (n.type)",
]


async def apply_schema(client: Neo4jClient) -> None:
    """Apply all constraints and indexes idempotently."""
    logger.info("Applying Neo4j schema constraints and indexes …")
    for stmt in _CONSTRAINTS + _INDEXES:
        try:
            await client.run_write(stmt)
            logger.debug("Applied: %s", stmt[:80])
        except Exception as exc:  # noqa: BLE001
            # Constraint already exists — safe to ignore
            logger.debug("Schema statement skipped (%s): %s", type(exc).__name__, stmt[:80])
    logger.info("Schema applied successfully")
