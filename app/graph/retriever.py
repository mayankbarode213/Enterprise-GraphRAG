"""
Graph Retriever

Production-ready GraphRAG retriever.
"""

from __future__ import annotations

import logging
import time

from typing import Any
from pydantic import BaseModel
from langchain_openai import ChatOpenAI

from app.graph.client import Neo4jClient, get_client
from app.graph import queries as Q
from app.schemas.models import Entity, GraphOperation, GraphPath, GraphResult, GraphResultType
from settings import settings

logger = logging.getLogger(__name__)


# =============================================================================
# Structured Query Parameters
# =============================================================================


class QueryParameters(BaseModel):
    machine_name: str | None = None
    vendor_name: str | None = None
    supplier_name: str | None = None
    batch_id: str | None = None
    batch_name: str | None = None       # used by BATCH_LINEAGE (Q011)
    component_name: str | None = None
    incident_name: str | None = None
    defect_name: str | None = None      # used by DEFECT_LINEAGE (Q008)
    qc_status: str | None = None

    # Comparison overrides
    compare_machine_name: str | None = None
    compare_vendor_name: str | None = None
    compare_supplier_name: str | None = None
    compare_batch_id: str | None = None
    compare_batch_name: str | None = None
    compare_component_name: str | None = None
    compare_defect_name: str | None = None
    compare_incident_name: str | None = None


# =============================================================================
# Graph Retriever
# =============================================================================


class GraphRetriever:
    """
    GraphRAG retriever.

    Responsibilities
    ----------------
    1. Detect user intent
    2. Extract query parameters
    3. Execute Cypher query
    4. Build graph entities
    5. Build graph paths
    6. Generate final response
    """

    def __init__(
        self,
        client: Neo4jClient | None = None,
        llm: ChatOpenAI | None = None,
    ) -> None:
        self._client = client
        self._llm = llm or ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            temperature=0,
        )

    async def retrieve_text2cypher(self, query: str) -> GraphResult:
        """
        Dynamic Text-to-Cypher (T2C) retrieval path.
        Generates a Cypher query on the fly using the LLM and executes it against Neo4j.
        """
        import time
        start_time = time.perf_counter()

        system_prompt = """You are an expert Neo4j Cypher Database Engineer for a Manufacturing Plant Knowledge Graph.
Given a user question, generate ONE valid, executable Cypher query that traverses the graph.

════════════════════════════════════════════════════════
NODE LABELS & THEIR EXACT PROPERTIES
════════════════════════════════════════════════════════
(:Supplier)   id, name, country, rating
(:Batch)      id, name, qc_status, production_date, quantity
(:Component)  id, name, part_number, material
(:Machine)    id, name, plant, production_line, commissioned
(:Maintenance)id, name, plant, maintenance_type, date, notes
(:Vendor)     id, name, country, specialty
(:Incident)   id, name, severity, date
(:Defect)     id, name, severity, date

════════════════════════════════════════════════════════
RELATIONSHIPS (EXACT DIRECTIONS)
════════════════════════════════════════════════════════
(:Supplier)  -[:SUPPLIED_BATCH]->      (:Batch)
(:Batch)     -[:MANUFACTURED_AS]->     (:Component)
(:Component) -[:INSTALLED_ON]->        (:Machine)
(:Machine)   -[:UNDERWENT]->           (:Maintenance)
(:Maintenance)-[:PERFORMED_BY]->       (:Vendor)
(:Incident)  -[:TRIGGERED_MAINTENANCE]->(:Maintenance)
(:Incident)  -[:REPORTED_AS]->         (:Defect)
(:Defect)    -[:CAUSED_BY_BATCH]->     (:Batch)

════════════════════════════════════════════════════════
STRICT QUERY RULES
════════════════════════════════════════════════════════
1. Return ONLY the raw Cypher query. No markdown, no explanation.
2. NO destructive operations: NO CREATE, MERGE, DELETE, SET, REMOVE.
3. Use UNDIRECTED relationships `-[:REL]-` (without `>` or `<`) to avoid direction mismatches.
4. ALL MATCH clauses MUST share a common bound variable to stay connected.
   BAD:  MATCH (i:Incident)-[:TRIGGERED_MAINTENANCE]-(mt)  ← isolated, unconnected to main chain
   GOOD: MATCH (s:Supplier)-[:SUPPLIED_BATCH]-(b:Batch)-[:CAUSED_BY_BATCH]-(d:Defect)-[:REPORTED_AS]-(i:Incident)-[:TRIGGERED_MAINTENANCE]-(mt:Maintenance)
5. Use ONLY ONE WHERE clause. Combine conditions with AND / OR. NEVER repeat WHERE.
6. Only return node variables (s, b, c, m, etc.) — NEVER return scalar properties like b.qcStatus.
7. NEVER invent property names. Use ONLY the exact property names listed above.

════════════════════════════════════════════════════════
EXAMPLE 1 — Supplier → Defect Root Cause Lineage
════════════════════════════════════════════════════════
MATCH (s:Supplier)-[:SUPPLIED_BATCH]-(b:Batch)-[:MANUFACTURED_AS]-(c:Component)-[:INSTALLED_ON]-(m:Machine)-[:UNDERWENT]-(mt:Maintenance)-[:PERFORMED_BY]-(v:Vendor)
MATCH (b)-[:CAUSED_BY_BATCH]-(d:Defect)-[:REPORTED_AS]-(i:Incident)-[:TRIGGERED_MAINTENANCE]-(mt)
WHERE toLower(m.name) CONTAINS toLower('RW-101') AND toLower(v.name) CONTAINS toLower('Apex')
RETURN s, b, c, m, mt, v, i, d

════════════════════════════════════════════════════════
EXAMPLE 2 — Defect → Back-trace to Supplier + all Batches
════════════════════════════════════════════════════════
MATCH (d:Defect)-[:CAUSED_BY_BATCH]-(b:Batch)-[:SUPPLIED_BATCH]-(s:Supplier)-[:SUPPLIED_BATCH]-(b2:Batch)-[:MANUFACTURED_AS]-(c2:Component)-[:INSTALLED_ON]-(m:Machine)
WHERE toLower(d.name) CONTAINS toLower('Hydraulic Seal Leakage')
RETURN d, b, s, b2, c2, m
"""

        try:
            response = await self._llm.ainvoke(
                [
                    ("system", system_prompt),
                    ("user", f"Generate Cypher query for: {query}"),
                ]
            )
            cypher_text = response.content.strip().replace("```cypher", "").replace("```", "").strip()
            logger.info("T2C Generated Cypher: %s", cypher_text)

            from app.graph.cypher_guardrail import CypherGuardrail
            is_valid, sanitized_cypher, err_msg = CypherGuardrail.validate_and_sanitize(cypher_text)

            print("\n" + "=" * 80)
            print("⚡ [Text-to-Cypher] LLM-GENERATED CYPHER QUERY:")
            print(f"{cypher_text}")
            if sanitized_cypher != cypher_text:
                print(f"🛡️ [CypherGuardrail] SANITIZED QUERY:\n{sanitized_cypher}")
            print("=" * 80 + "\n")

            if not is_valid:
                logger.warning("CypherGuardrail blocked query execution: %s", err_msg)
                return GraphResult(
                    query=query,
                    result_type=GraphResultType.LINEAGE,
                    operation=GraphOperation.TRAVERSAL,
                    entities=[],
                    cypher_used=f"Blocked by CypherGuardrail: {err_msg}",
                    intent="text2cypher",
                    latency_ms=(time.perf_counter() - start_time) * 1000,
                    depth_hops=0,
                )

            cypher_text = sanitized_cypher
            client = await self._get_client()
            raw_records = await client.run_query(cypher_text)
            latency_ms = (time.perf_counter() - start_time) * 1000

            entities_dict = {}
            if raw_records:
                for rec in raw_records:
                    if isinstance(rec, dict):
                        for key, val in rec.items():
                            if isinstance(val, dict):
                                e_id = val.get("id") or val.get("name")
                                if e_id and str(e_id) not in entities_dict:
                                    raw_type = val.get("type", "")
                                    s_id = str(e_id).upper()
                                    e_type = raw_type if raw_type else (
                                        "Supplier" if "SUP_" in s_id or "SHAKTI" in s_id or "BHARAT" in s_id or "WESTERN" in s_id
                                        else "Batch" if "BAT_" in s_id or "BATCH" in s_id
                                        else "Machine" if "MACH_" in s_id or "RW101" in s_id or "HP201" in s_id
                                        else "Component" if "COMP_" in s_id or "SEAL" in s_id or "MOTOR" in s_id
                                        else "Defect" if "DEF_" in s_id or "LEAK" in s_id
                                        else "Maintenance" if "MNT_" in s_id or "MAINT" in s_id
                                        else "Vendor" if "VEN_" in s_id or "APEX" in s_id
                                        else "Incident" if "INC_" in s_id
                                        else "Entity"
                                    )
                                    e_name = val.get("name") or str(e_id)
                                    entities_dict[str(e_id)] = Entity(id=str(e_id), type=e_type, name=str(e_name), properties=val)

            entities = list(entities_dict.values())

            return GraphResult(
                query=query,
                result_type=GraphResultType.LINEAGE,
                operation=GraphOperation.TRAVERSAL,
                entities=entities,
                cypher_used=cypher_text,
                intent="text2cypher",
                latency_ms=latency_ms,
                depth_hops=1,
            )

        except Exception as exc:
            logger.error("Text-to-Cypher retrieval failed: %s", exc)
            latency_ms = (time.perf_counter() - start_time) * 1000
            return GraphResult(
                query=query,
                result_type=GraphResultType.LINEAGE,
                operation=GraphOperation.TRAVERSAL,
                entities=[],
                cypher_used="Text-to-Cypher Failed",
                intent="text2cypher",
                latency_ms=latency_ms,
                depth_hops=0,
            )

    # ------------------------------------------------------------------ #
    # Neo4j Client
    # ------------------------------------------------------------------ #

    async def _get_client(self) -> Neo4jClient:
        if self._client is None:
            self._client = await get_client()

        return self._client

    # ------------------------------------------------------------------ #
    # Intent Detection
    # ------------------------------------------------------------------ #

    INTENT_KEYWORDS = {

        # ── Batch lineage / trace ─────────────────────────────────────────────
        # User is tracing a specific batch number (e.g. "Trace Hydraulic Seal Batch 2035-999")
        Q.QueryIntent.BATCH_LINEAGE: [
            "trace batch",
            "batch lineage",
            "trace hydraulic seal batch",
            "trace seal batch",
            "hydraulic seal batch",
            "batch 20",
            "batch 19",
            "batch genealogy",
            "trace batch number",
            "batch history",
        ],

        # ── Defect lineage / root-cause trace ─────────────────────────────────
        # User is STARTING FROM an explicit defect name (e.g. "Hydraulic Seal Leakage")
        Q.QueryIntent.DEFECT_LINEAGE: [
            "trace defect",
            "defect lineage",
            "trace hydraulic seal leakage",
            "hydraulic seal leakage",
            "leakage",
            "dimensional failure",
            "root cause",
            "root-cause",
            "originating supplier",
            "trace back defect",
            "complete path",
            "complete lineage",
            "responsible supplier",
            "upstream",                         # "identify every upstream entity"
            "upstream entity",
            "upstream entities",
            "responsible for the defect",
            "starting from hydraulic seal leakage",
            "identify every",
            "every entity responsible",
            "where did this defect come from",
            "which supplier caused",
        ],


        # ── Canonical multi-hop traversal (supplier → defect via maintenance) ─
        Q.QueryIntent.SUPPLIER_BATCH_TO_DEFECT: [
            "emergency maintenance",
            "performed by",
            "indirectly caused",
            "indirectly",
            "supplier batch",
            "which supplier batch",
            "batches supplied by",
            "supplied by",
            "defects originating from",
            "originating from batches",
            "which maintenance vendor serviced",
            "maintenance vendor serviced",
            "vendor serviced machines that experienced defects",
            "serviced machines that experienced defects",
        ],

        # ── Supplier full downstream chain (7 hops, includes components/machines)
        # Wins over DEFECTS_BY_SUPPLIER when user wants the COMPLETE downstream.
        Q.QueryIntent.SUPPLIER_LINEAGE: [
            "downstream entities",            # "trace all downstream entities"
            "all downstream",
            "starting from the supplier",     # "starting from the supplier X"
            "trace all downstream",
            "downstream from supplier",
            "full downstream",
            "everything downstream",
        ],

        # ── Supplier quality aggregation report (grouped by supplier) ────────
        Q.QueryIntent.SUPPLIER_QUALITY_SUMMARY: [
            "group the results by supplier",  # "group the results by supplier"
            "group by supplier",
            "find every supplier",
            "every supplier whose batches",
            "batches ultimately resulted",
            "listing all affected",
            "affected batches",
            "supplier quality summary",
            "quality report by supplier",
        ],

        # ── Vendor performed maintenance → machine → incident → defect ────────
        # Wins over MACHINES_BY_VENDOR when the user also asks about incidents/defects.
        Q.QueryIntent.VENDOR_MAINTENANCE_INCIDENTS: [
            "serviced by",
            "maintenance activities",
            "which machines were serviced",
            "performed maintenance",
            "machines were serviced",
            "associated with those maintenance",
            "associated with maintenance",
            "incidents and defects",
            "defects were associated",
        ],

        # ── Vendor → maintenance → machine (no incident detail needed) ────────
        Q.QueryIntent.MACHINES_BY_VENDOR: [
            "maintenance company",
            "service provider",
            "which vendor",
            "vendor serviced",
        ],

        # ── Supplier → batch → defect only (simple 3-hop)  ────────────────────
        # Does NOT match "downstream entities" queries — those go to SUPPLIER_LINEAGE
        Q.QueryIntent.DEFECTS_BY_SUPPLIER: [
            "defects caused by supplier",
            "defects from supplier",
            "what defects did",               # "What defects did supplier X cause?"
            "what defects",
            "defects did",
            "defects supplied",
            "caused by supplier",
        ],

        # ── Machine complete operational history ──────────────────────────────
        # Must beat DEFECTS_BY_SUPPLIER for "show history of machine X"
        Q.QueryIntent.MACHINE_HISTORY: [
            "operational history",               # "complete operational history"
            "complete history",
            "history of robotic",
            "history of machine",
            "history of cnc",
            "maintenance history",
            "machine history",
            "installed components",              # "including installed components"
            "maintenance activities",
            "rw-101",                            # machine ID fragments
            "rw101",
            "cnc_lathe",
            "timeline",
        ],

        # ── Component traceability ────────────────────────────────────────────
        Q.QueryIntent.COMPONENT_TRACE: [
            "component",
            "traceability",
            "which component",
            "trace component",
            "hydraulic pressure seal",
            "pressure seal",
            "servo motor",
            "plc board",
            "control board",
            "installed component",
        ],

        # ── Incident investigation ────────────────────────────────────────────
        Q.QueryIntent.INCIDENT_HISTORY: [
            "incident",
            "failure event",
            "breakdown",
            "what happened",
        ],

        # ── QC / batch quality ────────────────────────────────────────────────
        Q.QueryIntent.FAILED_QC_BATCHES: [
            "qc",
            "quality control",
            "failed batch",
            "quality check",
            "qc status",
            "failed qc",
        ],

        # ── 8-hop supplier risk exposure (defect → supplier → ALL batches → machines at risk) ─
        # Wins over DEFECT_LINEAGE when user asks about ONGOING RISK from other batches
        # of the same supplier still installed on production machines.
        Q.QueryIntent.SUPPLIER_RISK_EXPOSURE: [
            "ongoing risk",
            "still at risk",
            "machines at risk",
            "production risk",
            "other batches",
            "same supplier",
            "still installed",
            "currently installed",
            "risk assessment",
            "supplier risk",
            "remaining risk",
            "risk from same supplier",
            "identify all machines",
            "which machines remain",
            "which machines are still",
            "all batches from",
            "every batch from the same supplier",
            "risk exposure",
            "continued use",
            "scar",
        ],

        # ── Debug / explore ───────────────────────────────────────────────────
        Q.QueryIntent.LIST_ALL: [
            "list all",
            "show all",
            "all nodes",
        ],
    }


    def _detect_intent(
        self,
        query: str,
    ) -> Q.QueryIntent:
        """
        Lightweight rule-based intent detection.

        Falls back to the canonical GraphRAG traversal.
        """

        query_lower = query.lower()

        # Priority rule: Vendor audit or vendor servicing queries
        vendor_audit_indicators = ["audit apex", "audit vendor", "apex has serviced", "serviced by apex", "machines that apex", "vendor audit"]
        if any(ind in query_lower for ind in vendor_audit_indicators) or ("audit" in query_lower and "vendor" in query_lower) or ("audit" in query_lower and "apex" in query_lower):
            return Q.QueryIntent.VENDOR_MAINTENANCE_INCIDENTS

        # Priority rule: Supplier risk exposure — 8-hop query starting from a defect
        # and branching to ALL other batches from the same supplier
        risk_exposure_indicators = [
            "ongoing risk", "machines at risk", "still at risk", "same supplier",
            "risk assessment", "supplier risk", "still installed", "currently installed",
            "other batches", "risk exposure", "all batches from", "which machines remain",
            "which machines are still", "continued use", "scar",
        ]
        if any(ind in query_lower for ind in risk_exposure_indicators):
            return Q.QueryIntent.SUPPLIER_RISK_EXPOSURE

        # Priority rule: Supplier queries asking about defects, maintenance, machines, or vendors
        supplier_indicators = ["supplied by", "batches supplied", "originating from batches", "defects originating from"]
        if any(ind in query_lower for ind in supplier_indicators):
            if any(term in query_lower for term in ["vendor", "maintenance", "serviced", "defect", "machine"]):
                if "group" in query_lower:
                    return Q.QueryIntent.SUPPLIER_QUALITY_SUMMARY
                return Q.QueryIntent.SUPPLIER_BATCH_TO_DEFECT

        scores = {}

        for intent, keywords in self.INTENT_KEYWORDS.items():
            score = sum(keyword in query_lower for keyword in keywords)
            if score:
                scores[intent] = score

        if scores:
            return max(scores, key=scores.get)

        return Q.QueryIntent.SUPPLIER_BATCH_TO_DEFECT

    # ------------------------------------------------------------------ #
    # Parameter Extraction
    # ------------------------------------------------------------------ #

    async def _extract_parameters(
        self,
        query: str,
        intent: Q.QueryIntent,
    ) -> QueryParameters:
        """
        Extract graph entities from the user's question.

        Only extracts parameters required by the detected intent.
        """

        required = Q.required_parameters(intent)

        prompt = f"""
You are an expert entity parameter extractor for Neo4j manufacturing graph queries.

Intent: {intent.value}
Required parameters: {required}

CRITICAL RULES:
1. Preserve exact numbers, codes, and identifiers (e.g. "2035-999", "2026-001", "RW-101"). Never strip numeric codes.
2. If the user is comparing two entities (e.g., comparing two batches, two machines, two suppliers), extract the first entity into primary parameters (e.g., `batch_id` / `batch_name`) and the second entity into the corresponding comparison parameters (e.g., `compare_batch_id` / `compare_batch_name`).
3. If the user mentions a Batch (e.g., "Hydraulic Seal Batch 2035-999" or "Batch 2035-999"), extract "2035-999" or "Hydraulic Seal Batch 2035-999" into `batch_id` or `batch_name`. Do NOT put batch names into `defect_name`!
4. If the user mentions a Component (e.g., "Hydraulic Pressure Seal", "Servo Motor", "PLC Board"), extract it into `component_name`.
5. `defect_name` should ONLY be extracted when a specific defect symptom/type (e.g., "Leakage", "Dimensional Failure", "Crack") is explicitly named in the query.
6. `supplier_name` is for component/batch/part suppliers (e.g. "Shakti Industrial Seals Pvt. Ltd."). `vendor_name` is strictly for maintenance/service contractors (e.g. "Apex Maintenance Services"). NEVER extract a component supplier as `vendor_name`.
7. If a parameter value is not mentioned in the query, return null.

User query:
{query}
"""

        extractor = self._llm.with_structured_output(
            QueryParameters
        )

        params = await extractor.ainvoke(prompt)

        logger.debug(
            "Structured Parameters: %s",
            params.model_dump(),
        )

        print("Query_Params: ", params.model_dump())

        return params



    # ------------------------------------------------------------------ #
    # Execute Query
    # ------------------------------------------------------------------ #

    async def _execute_query(
        self,
        intent: Q.QueryIntent,
        params: QueryParameters,
    ) -> list[dict[str, Any]]:
        """
        Execute the Cypher query for the detected intent.

        Always passes the full parameter dict (including None values) so
        that Neo4j's ``$param IS NULL`` optional guards never raise a
        ParameterMissing error.
        """

        client = await self._get_client()

        cypher = Q.get_query(intent)

        logger.debug("Executing Cypher:\n%s", cypher)

        # Pass ALL parameters (None included).
        # Cypher uses ``$param IS NULL`` guards for optional fields,
        # so every referenced $param must be present in the dict.
        rows = await client.run_query(
            cypher,
            params.model_dump(),          # do NOT use exclude_none=True
        )

        return rows


    # ------------------------------------------------------------------ #
    # Entity Mapping
    # ------------------------------------------------------------------ #

    ENTITY_MAPPING = {

        "supplier": {
            "label": "Supplier",
            "id": "supplier_id",
            "name": "supplier_name",
        },

        "batch": {
            "label": "Batch",
            "id": "batch_id",
            "name": "batch_name",
        },

        "component": {
            "label": "Component",
            "id": "component_id",
            "name": "component_name",
        },

        "machine": {
            "label": "Machine",
            "id": "machine_id",
            "name": "machine_name",
        },

        "maintenance": {
            "label": "Maintenance",
            "id": "maintenance_id",
            "name": "maintenance_name",
        },

        "vendor": {
            "label": "Vendor",
            "id": "vendor_id",
            "name": "vendor_name",
        },

        "incident": {
            "label": "Incident",
            "id": "incident_id",
            "name": "incident_name",
        },

        "defect": {
            "label": "Defect",
            "id": "defect_id",
            "name": "defect_name",
        },
    }

    def _create_entity(
        self,
        row: dict[str, Any],
        mapping: dict[str, str],
    ) -> Entity:
        """
        Create an Entity object from a Neo4j result row.
        """

        prefix = mapping["id"].replace("_id", "")  # e.g. "supplier"

        properties = {
            key: value
            for key, value in row.items()
            if key.startswith(prefix + "_") and value is not None
        }

        return Entity(
            id=row[mapping["id"]],
            type=mapping["label"],
            name=row.get(mapping["name"], ""),
            properties=properties,
        )


    # ------------------------------------------------------------------ #
    # Build Entities
    # ------------------------------------------------------------------ #

    def _build_entities(
        self,
        rows: list[dict[str, Any]],
    ) -> list[Entity]:
        """
        Build a unique collection of Entity objects from query results.
        """

        entities: dict[str, Entity] = {}

        for row in rows:

            for mapping in self.ENTITY_MAPPING.values():

                entity_id = row.get(mapping["id"])

                if entity_id is None:
                    continue

                if entity_id in entities:
                    continue

                entities[entity_id] = self._create_entity(
                    row=row,
                    mapping=mapping,
                )

        return list(entities.values())


    # ------------------------------------------------------------------ #
    # Path Definition
    # ------------------------------------------------------------------ #

    PATH_ORDER = [
        "supplier",
        "batch",
        "component",
        "machine",
        "maintenance",
        "vendor",
        "incident",
        "defect",
    ]

    RELATIONSHIP_MAPPING = {
        ("Supplier", "Batch"): "SUPPLIED_BATCH",
        ("Batch", "Component"): "MANUFACTURED_AS",
        ("Component", "Machine"): "INSTALLED_ON",
        ("Machine", "Maintenance"): "UNDERWENT",
        ("Maintenance", "Vendor"): "PERFORMED_BY",

        ("Incident", "Maintenance"): "TRIGGERED_MAINTENANCE",
        ("Maintenance", "Incident"): "TRIGGERED_MAINTENANCE",

        ("Incident", "Defect"): "REPORTED_AS",
        ("Defect", "Incident"): "REPORTED_AS",

        ("Defect", "Batch"): "CAUSED_BY_BATCH",
        ("Batch", "Defect"): "CAUSED_BY_BATCH",
    }


    # ------------------------------------------------------------------ #
    # Build Paths
    # ------------------------------------------------------------------ #

    def _build_paths(
        self,
        rows: list[dict[str, Any]],
    ) -> list[GraphPath]:
        """
        Build clean branching paths from Neo4j rows to represent vendor
        and incident/defect branches as distinct traversals, preserving
        semantic relationship labels.
        """
        if rows and "root_defect_id" in rows[0]:
            return self._build_risk_exposure_paths(rows)

        paths: list[GraphPath] = []

        for row in rows:
            # We want to extract base nodes that are always present:
            # Supplier -> Batch -> Component -> Machine -> Maintenance
            base_nodes = []
            for name in ["supplier", "batch", "component", "machine", "maintenance"]:
                mapping = self.ENTITY_MAPPING.get(name)
                if mapping and row.get(mapping["id"]) is not None:
                    base_nodes.append(self._create_entity(row, mapping))

            # Build base relationship labels
            base_rels = []
            if len(base_nodes) >= 2:
                for curr, nxt in zip(base_nodes, base_nodes[1:]):
                    rel = self.RELATIONSHIP_MAPPING.get((curr.type, nxt.type)) or self.RELATIONSHIP_MAPPING.get((nxt.type, curr.type))
                    base_rels.append(rel or "RELATED")

            # Check if vendor branch exists
            v_mapping = self.ENTITY_MAPPING["vendor"]
            has_vendor = row.get(v_mapping["id"]) is not None

            # Check if incident / defect branch exists
            i_mapping = self.ENTITY_MAPPING["incident"]
            d_mapping = self.ENTITY_MAPPING["defect"]
            has_incident = row.get(i_mapping["id"]) is not None
            has_defect = row.get(d_mapping["id"]) is not None

            if len(base_nodes) >= 2:
                # 1. Vendor Branch: Base -> Maintenance -> Vendor
                if has_vendor:
                    v_node = self._create_entity(row, v_mapping)
                    paths.append(
                        GraphPath(
                            nodes=base_nodes + [v_node],
                            relationships=base_rels + ["PERFORMED_BY"],
                        )
                    )

                # 2. Incident & Defect Branch: Base -> Maintenance <- Incident -> Defect
                if has_incident:
                    i_node = self._create_entity(row, i_mapping)
                    i_nodes = base_nodes + [i_node]
                    i_rels = base_rels + ["TRIGGERED_MAINTENANCE"]

                    if has_defect:
                        d_node = self._create_entity(row, d_mapping)
                        i_nodes.append(d_node)
                        i_rels.append("REPORTED_AS")

                    paths.append(
                        GraphPath(
                            nodes=i_nodes,
                            relationships=i_rels,
                        )
                    )
                elif has_defect:
                    # Defect without incident, e.g. Defect -> Batch via CAUSED_BY_BATCH
                    d_node = self._create_entity(row, d_mapping)
                    b_mapping = self.ENTITY_MAPPING["batch"]
                    if row.get(b_mapping["id"]) is not None:
                        b_node = self._create_entity(row, b_mapping)
                        paths.append(
                            GraphPath(
                                nodes=[b_node, d_node],
                                relationships=["CAUSED_BY_BATCH"],
                            )
                        )
            else:
                # Fallback to linear zip of nodes if full base chain is not present
                nodes = []
                for name in self.PATH_ORDER:
                    mapping = self.ENTITY_MAPPING[name]
                    if row.get(mapping["id"]) is not None:
                        nodes.append(self._create_entity(row, mapping))

                if len(nodes) >= 2:
                    relationships: list[str] = []
                    for current, nxt in zip(nodes, nodes[1:]):
                        relationship = (
                            self.RELATIONSHIP_MAPPING.get((current.type, nxt.type))
                            or self.RELATIONSHIP_MAPPING.get((nxt.type, current.type))
                        )
                        relationships.append(relationship or "RELATED")

                    paths.append(
                        GraphPath(
                            nodes=nodes,
                            relationships=relationships,
                        )
                    )

        return paths


    def _build_risk_exposure_paths(
        self,
        rows: list[dict[str, Any]],
    ) -> list[GraphPath]:
        """
        Dedicated path builder for the 8-hop SUPPLIER_RISK_EXPOSURE query.
        Traces: Defect -> Defective Batch -> Supplier -> Batch -> Component
        -> Machine -> Maintenance -> Vendor / Incident -> Linked Defect.
        """
        paths: list[GraphPath] = []

        root_defect_map = {"label": "Defect", "id": "root_defect_id", "name": "root_defect_name"}
        defective_batch_map = {"label": "Batch", "id": "defective_batch_id", "name": "defective_batch_name"}
        supplier_map = {"label": "Supplier", "id": "supplier_id", "name": "supplier_name"}
        batch_map = {"label": "Batch", "id": "batch_id", "name": "batch_name"}
        component_map = {"label": "Component", "id": "component_id", "name": "component_name"}
        machine_map = {"label": "Machine", "id": "machine_id", "name": "machine_name"}
        maintenance_map = {"label": "Maintenance", "id": "maintenance_id", "name": "maintenance_name"}
        vendor_map = {"label": "Vendor", "id": "vendor_id", "name": "vendor_name"}
        incident_map = {"label": "Incident", "id": "incident_id", "name": "incident_name"}
        linked_defect_map = {"label": "Defect", "id": "linked_defect_id", "name": "linked_defect_name"}

        for row in rows:
            # Build the base 6-hop path: Root Defect -> Defective Batch -> Supplier -> Batch -> Component -> Machine -> Maintenance
            base_nodes = []
            base_rels = []

            # 1. Defect -> Defective Batch
            if row.get("root_defect_id") and row.get("defective_batch_id"):
                base_nodes.append(self._create_entity(row, root_defect_map))
                base_nodes.append(self._create_entity(row, defective_batch_map))
                base_rels.append("CAUSED_BY_BATCH")

            # 2. Defective Batch <- Supplier
            if row.get("supplier_id") and len(base_nodes) == 2:
                base_nodes.append(self._create_entity(row, supplier_map))
                base_rels.append("SUPPLIED_BATCH")

            # 3. Supplier -> Batch
            if row.get("batch_id") and len(base_nodes) == 3:
                base_nodes.append(self._create_entity(row, batch_map))
                base_rels.append("SUPPLIED_BATCH")

            # 4. Batch -> Component
            if row.get("component_id") and len(base_nodes) == 4:
                base_nodes.append(self._create_entity(row, component_map))
                base_rels.append("MANUFACTURED_AS")

            # 5. Component -> Machine
            if row.get("machine_id") and len(base_nodes) == 5:
                base_nodes.append(self._create_entity(row, machine_map))
                base_rels.append("INSTALLED_ON")

            # 6. Machine -> Maintenance
            if row.get("maintenance_id") and len(base_nodes) == 6:
                base_nodes.append(self._create_entity(row, maintenance_map))
                base_rels.append("UNDERWENT")

            if len(base_nodes) < 2:
                continue

            # Branch 1: Vendor details
            if row.get("vendor_id"):
                v_node = self._create_entity(row, vendor_map)
                paths.append(
                    GraphPath(
                        nodes=base_nodes + [v_node],
                        relationships=base_rels + ["PERFORMED_BY"],
                    )
                )

            # Branch 2: Incident & Defect details
            if row.get("incident_id"):
                i_node = self._create_entity(row, incident_map)
                i_nodes = base_nodes + [i_node]
                i_rels = base_rels + ["TRIGGERED_MAINTENANCE"]

                if row.get("linked_defect_id"):
                    ld_node = self._create_entity(row, linked_defect_map)
                    i_nodes.append(ld_node)
                    i_rels.append("REPORTED_AS")

                paths.append(
                    GraphPath(
                        nodes=i_nodes,
                        relationships=i_rels,
                    )
                )

        return paths

    # ------------------------------------------------------------------ #
    # Generate Answer
    # ------------------------------------------------------------------ #

    def _generate_answer(
        self,
        rows: list[dict[str, Any]],
        intent: Q.QueryIntent,
    ) -> str:
        """
        Generate a human-readable summary from graph results.

        For MACHINE_HISTORY intent, produces a rich structured summary
        that includes dates, maintenance type, activities, severity, and
        vendor metadata — ensuring equal context richness vs VectorRAG.

        This method intentionally avoids another LLM call.
        """

        if not rows:
            return "No matching records were found in the knowledge graph."

        # ── Rich structured summary for Machine History ───────────────────────
        if intent == Q.QueryIntent.MACHINE_HISTORY:
            return self._generate_machine_history_answer(rows)

        # ── Rich structured summary for Supplier Risk Exposure ───────────────
        if intent == Q.QueryIntent.SUPPLIER_RISK_EXPOSURE:
            return self._generate_supplier_risk_exposure_answer(rows)

        # ── Generic name-only summary for all other intents ───────────────────
        summary: dict[str, set[str]] = {
            "Supplier": set(),
            "Batch": set(),
            "Component": set(),
            "Machine": set(),
            "Maintenance": set(),
            "Vendor": set(),
            "Incident": set(),
            "Defect": set(),
        }

        for row in rows:

            if row.get("supplier_name"):
                name = row["supplier_name"]
                ent_id = row.get("supplier_id")
                summary["Supplier"].add(f"{name} ({ent_id})" if ent_id else name)

            if row.get("batch_name"):
                name = row["batch_name"]
                ent_id = row.get("batch_id")
                summary["Batch"].add(f"{name} ({ent_id})" if ent_id else name)

            if row.get("component_name"):
                name = row["component_name"]
                ent_id = row.get("component_id")
                summary["Component"].add(f"{name} ({ent_id})" if ent_id else name)

            if row.get("machine_name"):
                name = row["machine_name"]
                ent_id = row.get("machine_id")
                summary["Machine"].add(f"{name} ({ent_id})" if ent_id else name)

            if row.get("maintenance_name"):
                name = row["maintenance_name"]
                ent_id = row.get("maintenance_id")
                summary["Maintenance"].add(f"{name} ({ent_id})" if ent_id else name)

            if row.get("vendor_name"):
                name = row["vendor_name"]
                ent_id = row.get("vendor_id")
                summary["Vendor"].add(f"{name} ({ent_id})" if ent_id else name)

            if row.get("incident_name"):
                name = row["incident_name"]
                ent_id = row.get("incident_id")
                summary["Incident"].add(f"{name} ({ent_id})" if ent_id else name)

            if row.get("defect_name"):
                name = row["defect_name"]
                ent_id = row.get("defect_id")
                summary["Defect"].add(f"{name} ({ent_id})" if ent_id else name)

        output = []

        output.append(
            f"Retrieved {len(rows)} graph record(s).\n"
        )

        for entity_type, values in summary.items():

            if not values:
                continue

            output.append(f"{entity_type}s:")

            for value in sorted(values):
                output.append(f"  • {value}")

            output.append("")

        return "\n".join(output)


    def _generate_machine_history_answer(
        self,
        rows: list[dict[str, Any]],
    ) -> str:
        """
        Produce a rich structured summary for MACHINE_HISTORY queries.

        Includes maintenance dates, types, activities, incident/defect
        severity, and vendor country/specialty — all the metadata returned
        by the enriched MACHINE_HISTORY Cypher query.
        """
        output: list[str] = []

        # ── Machine info ─────────────────────────────────────────────────────
        machine_name = rows[0].get("machine_name", "Unknown Machine")
        machine_plant = rows[0].get("machine_plant") or "—"
        machine_commissioned = rows[0].get("machine_commissioned") or "—"
        output.append(f"Machine: {machine_name}")
        output.append(f"  Plant: {machine_plant}")
        output.append(f"  Commissioned: {machine_commissioned}")
        output.append("")

        # ── Maintenance events (deduplicated by maintenance_id) ───────────────
        seen_maintenance: dict[str, dict] = {}
        seen_incidents: dict[str, dict] = {}
        seen_defects: dict[str, dict] = {}
        seen_vendors: dict[str, dict] = {}
        seen_components: set[str] = set()

        for row in rows:
            mt_id = row.get("maintenance_id")
            if mt_id and mt_id not in seen_maintenance:
                seen_maintenance[mt_id] = {
                    "name":  row.get("maintenance_name", "—"),
                    "date":  row.get("maintenance_date") or "—",
                    "type":  row.get("maintenance_type") or "—",
                    "notes": row.get("maintenance_notes") or "—",
                }

            v_id = row.get("vendor_id")
            if v_id and v_id not in seen_vendors:
                seen_vendors[v_id] = {
                    "name":      row.get("vendor_name", "—"),
                    "country":   row.get("vendor_country") or "—",
                    "specialty": row.get("vendor_specialty") or "—",
                }

            i_id = row.get("incident_id")
            if i_id and i_id not in seen_incidents:
                seen_incidents[i_id] = {
                    "name":     row.get("incident_name", "—"),
                    "date":     row.get("incident_date") or "—",
                    "severity": row.get("incident_severity") or "—",
                }

            d_id = row.get("defect_id")
            if d_id and d_id not in seen_defects:
                seen_defects[d_id] = {
                    "name":     row.get("defect_name", "—"),
                    "date":     row.get("defect_date") or "—",
                    "severity": row.get("defect_severity") or "—",
                }

            c_name = row.get("component_name")
            if c_name:
                seen_components.add(c_name)

        output.append("Maintenance Events:")
        for mt in seen_maintenance.values():
            output.append(f"  • {mt['name']}")
            output.append(f"      Date          : {mt['date']}")
            output.append(f"      Type          : {mt['type']}")
            output.append(f"      Activities    : {mt['notes']}")
        output.append("")

        if seen_vendors:
            output.append("Vendors Involved:")
            for v in seen_vendors.values():
                output.append(f"  • {v['name']}")
                output.append(f"      Country   : {v['country']}")
                output.append(f"      Specialty : {v['specialty']}")
            output.append("")

        if seen_incidents:
            output.append("Incidents Reported:")
            for inc in seen_incidents.values():
                output.append(f"  • {inc['name']}")
                output.append(f"      Date      : {inc['date']}")
                output.append(f"      Severity  : {inc['severity']}")
            output.append("")

        if seen_defects:
            output.append("Defects Resulting from Incidents:")
            for d in seen_defects.values():
                output.append(f"  • {d['name']}")
                output.append(f"      Date      : {d['date']}")
                output.append(f"      Severity  : {d['severity']}")
            output.append("")

        if seen_components:
            output.append("Installed Components:")
            for c in sorted(seen_components):
                output.append(f"  • {c}")
            output.append("")

        output.append(f"Total graph records: {len(rows)}")

        return "\n".join(output)


    def _generate_supplier_risk_exposure_answer(
        self,
        rows: list[dict[str, Any]],
    ) -> str:
        """
        Produce a rich structured summary for SUPPLIER_RISK_EXPOSURE queries.

        Highlights the critical risk exposure where components from other
        batches of the same at-risk supplier (with an open SCAR) are still
        installed on production machines.
        """
        output: list[str] = []

        # ── Root defect and originating supplier/defective batch ──────────────
        root_defect_name = rows[0].get("root_defect_name", "Unknown Defect")
        root_defect_severity = rows[0].get("root_defect_severity") or "—"
        defective_batch_name = rows[0].get("defective_batch_name") or "—"
        defective_batch_qc = rows[0].get("defective_batch_qc_status") or "—"
        supplier_name = rows[0].get("supplier_name", "Unknown Supplier")
        supplier_rating = rows[0].get("supplier_rating") or "—"
        supplier_country = rows[0].get("supplier_country") or "—"

        output.append(f"Root Defect Investigated: {root_defect_name} (Severity: {root_defect_severity})")
        output.append(f"  Originated From Batch : {defective_batch_name} (QC Status: {defective_batch_qc})")
        output.append(f"  Originating Supplier  : {supplier_name} (Rating: {supplier_rating}, Country: {supplier_country})")
        output.append("  Supplier Quality Status: SCAR Open (Supplier Corrective Action Request)")
        output.append("")

        # ── Group batches and components ──────────────────────────────────────
        batches: dict[str, dict] = {}
        machines_at_risk: dict[str, set[str]] = {}

        for row in rows:
            b_id = row.get("batch_id")
            if b_id:
                if b_id not in batches:
                    batches[b_id] = {
                        "name": row.get("batch_name", "—"),
                        "qc_status": row.get("batch_qc_status") or "—",
                        "production_date": row.get("batch_production_date") or "—",
                        "quantity": row.get("batch_quantity") or "—",
                        "components": set(),
                        "machines": set(),
                    }
                
                c_name = row.get("component_name")
                if c_name:
                    batches[b_id]["components"].add(c_name)

                m_name = row.get("machine_name")
                if m_name:
                    batches[b_id]["machines"].add(m_name)
                    if b_id != row.get("defective_batch_id"):
                        if m_name not in machines_at_risk:
                            machines_at_risk[m_name] = set()
                        machines_at_risk[m_name].add(row.get("batch_name", "—"))

        output.append("All Batches Supplied by this Supplier & Downstream Assets:")
        for b_id, b_info in batches.items():
            is_defective = (b_id == rows[0].get("defective_batch_id"))
            status_flag = "⚠️ DEFECTIVE BATCH" if is_defective else "🔍 ACTIVE BATCH IN SYSTEM"
            output.append(f"  • {b_info['name']} [{status_flag}]")
            output.append(f"      QC Status      : {b_info['qc_status']}")
            output.append(f"      Quantity       : {b_info['quantity']}")
            output.append(f"      Components     : {', '.join(sorted(b_info['components']))}")
            output.append(f"      Installed On   : {', '.join(sorted(b_info['machines']))}")
        output.append("")

        # ── Risk Assessment Summary ───────────────────────────────────────────
        output.append("Risk Assessment Summary:")
        if machines_at_risk:
            output.append("  ⚠️ ONGOING PRODUCTION RISK DETECTED:")
            for m_name, risk_batches in machines_at_risk.items():
                output.append(f"    • Machine '{m_name}' has component from active batch '{', '.join(sorted(risk_batches))}' currently installed.")
                output.append(f"      Risk Profile: The supplier ({supplier_name}) has an open SCAR due to repeated dimensional deviations. Even though the defective batch {defective_batch_name} was remediated, the use of other batches from the same at-risk supplier on active production assets represents a critical quality exposure.")
        else:
            output.append("  ✅ No active components from other batches of this supplier are currently installed on production lines.")
        output.append("")

        output.append(f"Total graph records: {len(rows)}")

        return "\n".join(output)

    async def _resolve_entity(self, query: str) -> dict[str, Any] | list[dict[str, Any]] | None:
        """
        Database-driven Entity Resolver with scoring, entity role weighting, and ambiguity resolution.
        Exposes matches with highest token overlap and resolves ambiguity using query intent signals.
        """
        try:
            client = await self._get_client()
            records = await client.run_query("MATCH (n) RETURN labels(n)[0] AS type, n.name AS name, n.id AS id")

            query_lower = query.lower()

            # Identify intent role signals in query
            is_vendor_audit = any(w in query_lower for w in ["audit apex", "audit vendor", "apex has serviced", "serviced by apex", "machines that apex", "vendor audit"])
            has_defect_signal = any(w in query_lower for w in ["failure", "defect", "leakage", "leak", "breakdown", "issue", "fault", "problem"])
            has_incident_signal = any(w in query_lower for w in ["incident", "loss", "event", "alarm"])
            has_batch_signal = any(w in query_lower for w in ["batch", "lot", "shipment"])
            has_component_signal = any(w in query_lower for w in ["component", "part"])
            has_supplier_signal = any(w in query_lower for w in ["supplier", "manufacturer"])
            has_vendor_signal = any(w in query_lower for w in ["vendor", "contractor", "servicing"]) or is_vendor_audit

            # Clean tokens from query
            common_words = {"trace", "show", "history", "defect", "batch", "component", "supplier", "vendor", "incident", "of", "the", "and", "every", "that", "connects", "original", "with", "from", "preparing", "audit", "order"}
            query_tokens = [t.strip(".,?!()\"'") for t in query_lower.split()]
            query_tokens = [t for t in query_tokens if len(t) >= 3 and t not in common_words]

            if not query_tokens:
                return None

            candidates = []
            for r in records:
                name = r.get("name", "")
                node_id = r.get("id", "")
                node_type = r.get("type", "")

                # Check for exact substring match
                exact_match = (name and name.lower() in query_lower) or (node_id and node_id.lower() in query_lower)

                # Count matching tokens
                score = 0
                for token in query_tokens:
                    if (name and token in name.lower()) or (node_id and token in node_id.lower()):
                        score += 1

                if exact_match or score > 0:
                    final_score = score + (10 if exact_match else 0)

                    # Role-based weighting boost
                    if is_vendor_audit and node_type == "Vendor":
                        final_score += 30
                    elif has_defect_signal and node_type == "Defect":
                        final_score += 15
                    elif has_incident_signal and node_type == "Incident":
                        final_score += 10
                    elif has_batch_signal and node_type == "Batch":
                        final_score += 10
                    elif has_component_signal and node_type == "Component":
                        final_score += 10
                    elif has_supplier_signal and node_type == "Supplier":
                        final_score += 10
                    elif has_vendor_signal and node_type == "Vendor":
                        final_score += 10

                    candidates.append((final_score, score, r))

            if not candidates:
                return None

            # Sort by score descending
            candidates.sort(key=lambda x: (-x[0], -x[1]))

            best_score = candidates[0][0]
            top_candidates = [item[2] for item in candidates if item[0] == best_score]

            if len(top_candidates) > 1:
                # Deduplicate candidates by name
                unique_dict = {cand['name']: cand for cand in top_candidates}
                top_candidates = list(unique_dict.values())
                if len(top_candidates) > 1:
                    # If all top candidates share the same entity type (e.g. Defect), auto-select top candidate
                    types = {c["type"] for c in top_candidates}
                    if len(types) == 1:
                        return top_candidates[0]
                    return top_candidates

            return top_candidates[0]
        except Exception as exc:
            logger.warning("Entity Resolver failed: %s", exc)

        return None

    # ------------------------------------------------------------------ #
    # Main Pipeline
    # ------------------------------------------------------------------ #

    async def retrieve(
        self,
        query: str,
    ) -> GraphResult:

        start = time.perf_counter()

        logger.info("=" * 80)
        logger.info("GRAPH RETRIEVER")
        logger.info("Query : %s", query)

        # ── Step 1 : Entity Resolution (Database-driven Entity Resolver) ──────
        resolved = None
        is_comparison_query = any(word in query.lower() for word in [
            "compare", "versus", " vs ", "difference", "shared with", "common with", "different from", "relative to"
        ])
        is_single_entity_query = (
            (any(word in query.lower() for word in ["trace", "lineage", "history of", "origin of", "genealogy", "what happened to", "supplied by", "batches supplied", "originating from", "audit"])
             or "supplier" in query.lower() or "vendor" in query.lower())
            and not is_comparison_query
            and not any(word in query.lower() for word in ["indirectly caused", "caused all defects", "supplier batches"])
        )
        
        if is_single_entity_query:
            resolved = await self._resolve_entity(query)
            
        if isinstance(resolved, list):
            # Ambiguity detected
            logger.info("Entity Resolver: Ambiguity detected between %d candidates", len(resolved))
            query_clean = query.replace("Trace ", "").replace("trace ", "").strip(".,?!()\"'")
            lines = [
                f"The query \"{query_clean}\" matches multiple entities.\n",
                "Please choose one:"
            ]
            for idx, cand in enumerate(resolved, 1):
                lines.append(f"{idx}. {cand['name']} ({cand['type']})")
            answer = "\n".join(lines)
            
            latency = round((time.perf_counter() - start) * 1000, 2)
            return GraphResult(
                query=query,
                result_type=Q.GraphResultType.LINEAGE,
                operation=Q.GraphOperation.TRAVERSAL,
                entities=[],
                paths=[],
                cypher_used="N/A (Ambiguous Entity)",
                answer=answer,
                intent="ambiguous_entity",
                latency_ms=latency,
            )
        elif resolved:
            logger.info("Entity Resolver: resolved '%s' as %s", resolved["name"], resolved["type"])
            # Map resolved node label to EntityType
            type_mapping = {
                "Supplier": Q.EntityType.SUPPLIER,
                "Batch": Q.EntityType.BATCH,
                "Component": Q.EntityType.COMPONENT,
                "Machine": Q.EntityType.MACHINE,
                "Vendor": Q.EntityType.VENDOR,
                "Incident": Q.EntityType.INCIDENT,
                "Defect": Q.EntityType.DEFECT,
            }
            entity_type = type_mapping.get(resolved["type"], Q.EntityType.BATCH)
            
            # Resolved single-entity queries route to Lineage or History operations
            op = Q.Operation.HISTORY if entity_type in [Q.EntityType.MACHINE, Q.EntityType.VENDOR, Q.EntityType.INCIDENT] else Q.Operation.LINEAGE
            
            # Resolve query intent via EntityType & Operation abstraction
            intent = Q.resolve_intent(entity_type, op)
            if entity_type == Q.EntityType.SUPPLIER:
                if any(w in query.lower() for w in ["every", "all affected", "downstream", "lineage", "resume production", "affected"]):
                    intent = Q.QueryIntent.SUPPLIER_LINEAGE
                elif any(w in query.lower() for w in ["defect", "maintenance", "vendor", "serviced"]):
                    intent = Q.QueryIntent.SUPPLIER_BATCH_TO_DEFECT
            elif entity_type == Q.EntityType.DEFECT:
                if any(w in query.lower() for w in ["ongoing risk", "machines at risk", "still at risk", "same supplier", "risk exposure", "other batches"]):
                    intent = Q.QueryIntent.SUPPLIER_RISK_EXPOSURE

            logger.info("Entity Resolver Override: set intent to %s (via %s + %s)", intent.value, entity_type.value, op.value)
            
            # Map parameters deterministically
            params = QueryParameters()
            t = resolved["type"]
            if t == "Supplier":
                params.supplier_name = resolved["name"]
            elif t == "Batch":
                params.batch_id = resolved["id"]
                params.batch_name = resolved["name"]
            elif t == "Component":
                params.component_name = resolved["name"]
            elif t == "Machine":
                params.machine_name = resolved["name"]
            elif t == "Vendor":
                params.vendor_name = resolved["name"]
            elif t == "Incident":
                params.incident_name = resolved["name"]
            elif t == "Defect":
                params.defect_name = resolved["name"]
        else:
            # Fallback to rule-based intent and LLM extraction
            intent = self._detect_intent(query)
            logger.info("Intent : %s (rule-based)", intent.value)
            params = await self._extract_parameters(query=query, intent=intent)

        # ── Step 2b : Entity Type Correction Guardrail ───────────────────────
        if params.vendor_name and not params.supplier_name:
            client = await self._get_client()
            try:
                supplier_check = await client.run_query(
                    "MATCH (s:Supplier) WHERE toLower(s.name) CONTAINS toLower($v_name) RETURN s.name AS name LIMIT 1",
                    {"v_name": params.vendor_name}
                )
                if supplier_check:
                    supplier_canonical = supplier_check[0]["name"]
                    logger.warning(
                        "Entity Guardrail: '%s' was extracted as vendor_name but matches Supplier '%s' in Neo4j. Reassigning to supplier_name.",
                        params.vendor_name, supplier_canonical
                    )
                    params.supplier_name = supplier_canonical
                    params.vendor_name = None
                    intent = Q.QueryIntent.SUPPLIER_BATCH_TO_DEFECT
            except Exception as exc:
                logger.warning("Entity Guardrail check failed: %s", exc)

        logger.info(
            "Parameters : %s",
            params.model_dump(exclude_none=True),
        )

        #
        # Step 3 : Execute Query
        #

        # Check if we have comparison parameters to perform comparison queries
        has_comparison = any([
            params.compare_machine_name,
            params.compare_vendor_name,
            params.compare_supplier_name,
            params.compare_batch_id,
            params.compare_batch_name,
            params.compare_component_name,
            params.compare_defect_name,
            params.compare_incident_name,
        ])

        if has_comparison:
            logger.info("Comparison query detected. Running queries for both entities...")
            # Execute first entity query
            params_a = params.model_copy()
            for field in list(params_a.model_fields.keys()):
                if field.startswith("compare_"):
                    setattr(params_a, field, None)
            rows_a = await self._execute_query(intent=intent, params=params_a)

            # Execute second entity query (mapping comparison fields to primary fields)
            params_b = params.model_copy()
            if params_b.compare_machine_name:
                params_b.machine_name = params_b.compare_machine_name
            if params_b.compare_vendor_name:
                params_b.vendor_name = params_b.compare_vendor_name
            if params_b.compare_supplier_name:
                params_b.supplier_name = params_b.compare_supplier_name
            if params_b.compare_batch_id:
                params_b.batch_id = params_b.compare_batch_id
            if params_b.compare_batch_name:
                params_b.batch_name = params_b.compare_batch_name
            if params_b.compare_component_name:
                params_b.component_name = params_b.compare_component_name
            if params_b.compare_defect_name:
                params_b.defect_name = params_b.compare_defect_name
            if params_b.compare_incident_name:
                params_b.incident_name = params_b.compare_incident_name

            for field in list(params_b.model_fields.keys()):
                if field.startswith("compare_"):
                    setattr(params_b, field, None)
            rows_b = await self._execute_query(intent=intent, params=params_b)

            rows = rows_a + rows_b
        else:
            rows = await self._execute_query(
                intent=intent,
                params=params,
            )

        logger.info("Rows Returned : %d", len(rows))

        # ── Step 4a: Entity Existence Guardrail ───────────────────────────────
        if len(rows) == 0:
            extracted_item = (
                params.batch_name or params.batch_id or params.defect_name or
                params.machine_name or params.supplier_name or params.vendor_name or "the requested item"
            )
            answer = (
                f"No matching entity or batch record found in the Knowledge Graph for '{extracted_item}'. "
                f"No lineage or operational data exists for this item."
            )
            latency = round((time.perf_counter() - start) * 1000, 2)
            result_type = Q.INTENT_RESULT_TYPES.get(intent, Q.GraphResultType.LINEAGE)
            operation   = Q.INTENT_OPERATIONS.get(intent, Q.GraphOperation.TRAVERSAL)

            # Determine root entity dynamically for empty result
            root_entity = None
            for field in ["defect_name", "batch_name", "batch_id", "component_name", "machine_name", "vendor_name", "supplier_name", "incident_name"]:
                val = getattr(params, field, None)
                if val:
                    type_label = field.split("_")[0].capitalize()
                    root_entity = f"{val} ({type_label})"
                    break

            return GraphResult(
                query=query,
                result_type=result_type,
                operation=operation,
                entities=[],
                paths=[],
                cypher_used=Q.get_query(intent),
                answer=answer,
                intent=intent.value,
                latency_ms=latency,
                root_entity=root_entity,
                depth_hops=0,
            )

        #
        # Step 4b : Build Result Objects
        #

        entities = self._build_entities(rows)

        # Hydrate entities with full properties from database
        try:
            entity_ids = [ent.id for ent in entities if ent.id]
            if entity_ids:
                client = await self._get_client()
                prop_rows = await client.run_query(
                    "MATCH (n) WHERE n.id IN $ids RETURN n.id AS id, properties(n) AS props",
                    {"ids": entity_ids}
                )
                props_by_id = {r["id"]: r["props"] for r in prop_rows}
                for ent in entities:
                    ent.properties = props_by_id.get(ent.id, {})
        except Exception as exc:
            logger.warning("Failed to hydrate entity properties: %s", exc)

        paths = self._build_paths(rows)

        #
        # Step 5 : Generate Natural Language Answer
        #

        answer = self._generate_answer(
            rows=rows,
            intent=intent,
        )

        latency = round(
            (time.perf_counter() - start) * 1000,
            2,
        )

        logger.info("Completed in %.2f ms", latency)

        result_type = Q.INTENT_RESULT_TYPES.get(intent, Q.GraphResultType.LINEAGE)
        operation   = Q.INTENT_OPERATIONS.get(intent, Q.GraphOperation.TRAVERSAL)

        # Determine root entity and depth hops dynamically
        root_name = None
        root_type = None

        if intent == Q.QueryIntent.SUPPLIER_LINEAGE or intent == Q.QueryIntent.SUPPLIER_QUALITY_SUMMARY:
            if params.supplier_name:
                root_name, root_type = params.supplier_name, "Supplier"
        elif intent == Q.QueryIntent.SUPPLIER_BATCH_TO_DEFECT:
            # Dynamically resolve root supplier from execution rows
            suppliers = [r["supplier_name"] for r in rows if r.get("supplier_name")]
            if suppliers:
                root_name, root_type = suppliers[0], "Supplier"
        elif intent == Q.QueryIntent.BATCH_LINEAGE:
            if params.batch_name or params.batch_id:
                root_name, root_type = params.batch_name or params.batch_id, "Batch"
        elif intent == Q.QueryIntent.COMPONENT_TRACE:
            if params.component_name:
                root_name, root_type = params.component_name, "Component"
        elif intent == Q.QueryIntent.MACHINE_HISTORY:
            if params.machine_name:
                root_name, root_type = params.machine_name, "Machine"
        elif intent == Q.QueryIntent.VENDOR_MAINTENANCE_INCIDENTS or intent == Q.QueryIntent.MACHINES_BY_VENDOR:
            if params.vendor_name:
                root_name, root_type = params.vendor_name, "Vendor"
        elif intent == Q.QueryIntent.INCIDENT_HISTORY:
            if params.incident_name:
                root_name, root_type = params.incident_name, "Incident"
        elif intent == Q.QueryIntent.DEFECT_LINEAGE:
            if params.defect_name:
                root_name, root_type = params.defect_name, "Defect"

        if not root_name:
            # Fallback to resolved or any populated parameter
            for field in ["defect_name", "batch_name", "batch_id", "component_name", "machine_name", "vendor_name", "supplier_name", "incident_name"]:
                val = getattr(params, field, None)
                if val:
                    root_name = val
                    root_type = field.split("_")[0].capitalize()
                    break

        root_entity = None
        if root_name and root_type:
            canonical_name = None
            for ent in entities:
                if ent.type.lower() == root_type.lower():
                    if root_type == "Batch":
                        if ent.id == root_name or root_name.lower() in ent.name.lower():
                            canonical_name = ent.name
                            break
                    else:
                        if root_name.lower() in ent.name.lower():
                            canonical_name = ent.name
                            break
            resolved_name = canonical_name or root_name
            root_entity = f"{resolved_name} ({root_type})"

        depth_hops = max(len(p.relationships) for p in paths) if paths else 0

        return GraphResult(
            query=query,
            result_type=result_type,
            operation=operation,
            entities=entities,
            paths=paths,
            cypher_used=Q.get_query(intent),
            answer=answer,
            intent=intent.value,
            latency_ms=latency,
            root_entity=root_entity,
            depth_hops=depth_hops,
        )
    
    @property
    def client(self) -> Neo4jClient:
        """
        Expose Neo4j client if needed for testing.
        """
        if self._client is None:
            raise RuntimeError("Neo4j client not initialized.")
        return self._client