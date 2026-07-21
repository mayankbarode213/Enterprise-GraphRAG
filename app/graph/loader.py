"""
Bulk-loads the manufacturing dataset from CSV files into Neo4j.
Idempotent — uses MERGE so re-running is safe.
"""
from __future__ import annotations

import asyncio
import csv
import logging
from pathlib import Path

from app.graph.client import Neo4jClient, get_client
from app.graph.schema import apply_schema

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parents[2] / "data"
print("DATA_DIR:", DATA_DIR)
ENTITIES_CSV = DATA_DIR / "entities.csv"
print("ENTITIES_CSV:", ENTITIES_CSV)
RELATIONSHIPS_CSV = DATA_DIR / "relationships.csv"
print("RELATIONSHIPS_CSV:", RELATIONSHIPS_CSV)

# Cypher templates — MERGE ensures idempotency
_MERGE_NODE = """
MERGE (n:{label} {{id: $id}})
SET n += $props
"""

_MERGE_REL = """
MATCH (a {{id: $source_id}})
MATCH (b {{id: $target_id}})
MERGE (a)-[r:{rel_type}]->(b)
SET r.properties = $properties
"""


def _parse_properties(raw: str) -> dict[str, str]:
    """Parse 'key=value,key2=value2' string into a dict."""
    result: dict[str, str] = {}
    if not raw or raw.strip() == "":
        return result
    for pair in raw.split(","):
        if "=" in pair:
            k, _, v = pair.partition("=")
            result[k.strip()] = v.strip()
    return result


async def load_entities(client: Neo4jClient) -> int:
    """Load all nodes from entities.csv. Returns count of loaded nodes."""
    count = 0
    with open(ENTITIES_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["type"].strip()
            cypher = _MERGE_NODE.format(label=label)
            
            # Extract all non-empty properties from the CSV row
            props = {
                "id": row["id"].strip(),
                "name": row["name"].strip(),
                "label": label,
            }
            for k, v in row.items():
                if v and v.strip() != "" and k not in ["id", "type"]:
                    props[k.strip()] = v.strip()
            
            await client.run_write(
                cypher,
                {
                    "id": row["id"].strip(),
                    "props": props,
                },
            )
            count += 1
    logger.info("Loaded %d entity nodes into Neo4j", count)
    return count


async def load_relationships(client: Neo4jClient) -> int:
    """Load all edges from relationships.csv. Returns count of loaded relationships."""
    count = 0
    with open(RELATIONSHIPS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel_type = row["relationship"].strip()
            cypher = _MERGE_REL.format(rel_type=rel_type)
            await client.run_write(
                cypher,
                {
                    "source_id": row["source_id"].strip(),
                    "target_id": row["target_id"].strip(),
                    "properties": str(_parse_properties(row.get("properties", ""))),
                },
            )
            count += 1
    logger.info("Loaded %d relationships into Neo4j", count)
    return count


async def load_all(client: Neo4jClient | None = None) -> dict[str, int]:
    """Full pipeline: schema → entities → relationships."""
    if client is None:
        client = await get_client()

    await apply_schema(client)
    nodes = await load_entities(client)
    rels = await load_relationships(client)
    logger.info("Graph load complete: %d nodes, %d relationships", nodes, rels)
    return {"nodes": nodes, "relationships": rels}


async def ensure_graph_loaded(client: Neo4jClient) -> None:
    """Ensure that the graph is seeded with entities if empty or missing canonical records."""
    try:
        res = await client.run_query("MATCH (s:Supplier {id: 'SUP_SHAKTI'}) RETURN count(s) AS cnt")
        if not res or res[0].get("cnt", 0) == 0:
            logger.info("Canonical graph data missing in Neo4j. Auto-seeding graph...")
            await load_all(client)
    except Exception as exc:
        logger.warning("Auto-seed check failed: %s", exc)



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(load_all())

