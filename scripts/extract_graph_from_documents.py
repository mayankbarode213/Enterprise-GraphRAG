"""
extract_graph_from_documents.py — Graph Extraction Script

Reads raw unstructured text documents from `data/documents/`, uses an LLM
to extract entities and relationships matching the Pydantic schema, and exports
`entities.csv` and `relationships.csv`.

Demonstrates that GraphRAG and VectorRAG originate from the exact same raw data!
"""
import asyncio
import csv
import logging
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DOCS_DIR = DATA_DIR / "documents"
ENTITIES_CSV = DATA_DIR / "entities.csv"
RELATIONSHIPS_CSV = DATA_DIR / "relationships.csv"


# ── Pydantic Schemas for Structured Output ────────────────────────────────────

class ExtractedEntity(BaseModel):
    id: str = Field(..., description="Unique entity ID, e.g., SUP_SHAKTI, MACH_RW101, BAT_HS_2026_001")
    type: str = Field(..., description="Entity type: Supplier, Batch, Component, Machine, ProductionLine, Vendor, Operator, Maintenance, Incident, Defect")
    name: str = Field(..., description="Human-readable name")
    plant: Optional[str] = Field(None, description="Plant name, e.g., Chakan Plant")
    production_line: Optional[str] = Field(None, description="Associated production line ID")
    country: Optional[str] = Field(None, description="Country of origin")
    rating: Optional[str] = Field(None, description="Rating string, e.g., 3.7 / 5")
    part_number: Optional[str] = Field(None, description="Part number, e.g., HS-501")
    material: Optional[str] = Field(None, description="Material, e.g., Nitrile Rubber")
    qc_status: Optional[str] = Field(None, description="QC status: PASSED, FAILED")
    production_date: Optional[str] = Field(None, description="Production or delivery date YYYY-MM-DD")
    quantity: Optional[str] = Field(None, description="Quantity string")
    specialty: Optional[str] = Field(None, description="Vendor specialty")
    commissioned: Optional[str] = Field(None, description="Commission date YYYY-MM-DD")
    shift: Optional[str] = Field(None, description="Operator shift: Morning, Evening, Night")
    experience_years: Optional[str] = Field(None, description="Years of experience")
    severity: Optional[str] = Field(None, description="Severity: Critical, High, Medium, Low")
    maintenance_type: Optional[str] = Field(None, description="Maintenance type: Scheduled, Emergency")
    date: Optional[str] = Field(None, description="Event date YYYY-MM-DD")
    notes: Optional[str] = Field(None, description="Additional notes or descriptions")


class ExtractedRelationship(BaseModel):
    source_id: str = Field(..., description="Source entity ID")
    source_type: str = Field(..., description="Source entity type")
    relationship: str = Field(..., description="UPPERCASE relationship type: SUPPLIED_BATCH, MANUFACTURED_AS, INSTALLED_ON, LOCATED_IN, OPERATES, UNDERWENT, PERFORMED_BY, TRIGGERED_MAINTENANCE, REPORTED_AS, AFFECTED_COMPONENT, CAUSED_BY_BATCH")
    target_id: str = Field(..., description="Target entity ID")
    target_type: str = Field(..., description="Target entity type")


class DocumentExtractionResult(BaseModel):
    entities: List[ExtractedEntity] = Field(default_factory=list)
    relationships: List[ExtractedRelationship] = Field(default_factory=list)


# ── LLM Extraction Logic ──────────────────────────────────────────────────────

async def extract_from_document(llm: ChatOpenAI, file_path: Path) -> DocumentExtractionResult:
    """Extract entities and relationships from a single document text file."""
    text = file_path.read_text(encoding="utf-8")
    
    prompt = f"""You are a Knowledge Graph Extraction Agent for a manufacturing plant.
Extract all Entities and Relationships described in the following raw document text.

CRITICAL INSTRUCTIONS:
1. Extract ALL Entities with their exact IDs (e.g. SUP_SHAKTI, MACH_RW101, BAT_HS_2026_001, COMP_HYD_SEAL, INC_RW101_001, DEF_HYD_LEAK, MNT_RW101_002, VEN_APEX).
2. Extract ALL explicit relationships connecting these entities.
3. Use UPPERCASE for relationship types (e.g. SUPPLIED_BATCH, MANUFACTURED_AS, INSTALLED_ON, LOCATED_IN, OPERATES, UNDERWENT, PERFORMED_BY, TRIGGERED_MAINTENANCE, REPORTED_AS, AFFECTED_COMPONENT, CAUSED_BY_BATCH).

Document File: {file_path.name}
Document Content:
{text}
"""

    structured_llm = llm.with_structured_output(DocumentExtractionResult)
    try:
        result: DocumentExtractionResult = await structured_llm.ainvoke(prompt)
        logger.info(f"Extracted {len(result.entities)} entities, {len(result.relationships)} relationships from {file_path.name}")
        return result
    except Exception as exc:
        logger.error(f"Failed extraction for {file_path.name}: {exc}")
        return DocumentExtractionResult()


async def main():
    logger.info("Starting LLM-based Graph Extraction from data/documents/...")
    
    if not DOCS_DIR.exists():
        logger.error(f"Documents directory not found at {DOCS_DIR}")
        return

    doc_files = list(DOCS_DIR.glob("*.txt"))
    if not doc_files:
        logger.error("No .txt files found in data/documents/")
        return

    logger.info(f"Found {len(doc_files)} text documents for extraction.")

    llm = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        temperature=0.0,
    )

    tasks = [extract_from_document(llm, f) for f in doc_files]
    results: List[DocumentExtractionResult] = await asyncio.gather(*tasks)

    # ── Aggregate and Deduplicate ──────────────────────────────────────────────
    all_entities: Dict[str, Dict[str, Any]] = {}
    all_relationships: set = set()
    rel_dicts: List[Dict[str, str]] = []

    entity_fields = [
        "id", "type", "name", "plant", "production_line", "country", "rating",
        "part_number", "material", "qc_status", "production_date", "quantity",
        "specialty", "commissioned", "shift", "experience_years", "severity",
        "maintenance_type", "date", "notes"
    ]

    for res in results:
        for entity in res.entities:
            e_dict = entity.model_dump()
            ent_id = e_dict["id"]
            if ent_id not in all_entities:
                all_entities[ent_id] = e_dict
            else:
                for k, v in e_dict.items():
                    if v is not None and all_entities[ent_id].get(k) is None:
                        all_entities[ent_id][k] = v

        for rel in res.relationships:
            r_tuple = (rel.source_id, rel.source_type, rel.relationship, rel.target_id, rel.target_type)
            if r_tuple not in all_relationships:
                all_relationships.add(r_tuple)
                rel_dicts.append({
                    "source_id": rel.source_id,
                    "source_type": rel.source_type,
                    "relationship": rel.relationship,
                    "target_id": rel.target_id,
                    "target_type": rel.target_type,
                })

    logger.info(f"Extraction Summary: Total Unique Entities = {len(all_entities)}, Total Unique Relationships = {len(rel_dicts)}")

    # ── Export Entities CSV ────────────────────────────────────────────────────
    with open(ENTITIES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=entity_fields)
        writer.writeheader()
        for e in sorted(all_entities.values(), key=lambda x: x["id"]):
            clean_row = {k: ("" if v is None else str(v)) for k, v in e.items()}
            writer.writerow(clean_row)

    logger.info(f"Exported entities to {ENTITIES_CSV}")

    # ── Export Relationships CSV ───────────────────────────────────────────────
    rel_fields = ["source_id", "source_type", "relationship", "target_id", "target_type"]
    with open(RELATIONSHIPS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rel_fields)
        writer.writeheader()
        for r in sorted(rel_dicts, key=lambda x: (x["source_id"], x["relationship"], x["target_id"])):
            writer.writerow(r)

    logger.info(f"Exported relationships to {RELATIONSHIPS_CSV}")
    logger.info("LLM Graph Extraction completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
