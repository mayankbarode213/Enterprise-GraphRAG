"""
app/graph/queries.py

Central registry of parameterized Cypher queries.

Design goals
------------
• No hardcoded entity names
• Parameterized Cypher only
• Production-ready
• Easy to extend
• Compatible with current manufacturing graph

String filters use ``toLower(x) CONTAINS toLower($param)`` so that
LLM-extracted names (which may be abbreviated or slightly different from
the exact stored values) still match.

Graph schema

Supplier
    ──SUPPLIED_BATCH──▶ Batch
Batch
    ──MANUFACTURED_AS──▶ Component
Component
    ──INSTALLED_ON──▶ Machine
Machine
    ──UNDERWENT──▶ Maintenance
Maintenance
    ──PERFORMED_BY──▶ Vendor
Incident
    ──TRIGGERED_MAINTENANCE──▶ Maintenance
Incident
    ──REPORTED_AS──▶ Defect
Defect
    ──CAUSED_BY_BATCH──▶ Batch
"""

from __future__ import annotations

from enum import Enum


from app.schemas.models import GraphOperation, GraphResultType


# ============================================================================
# Entity Types and Operations (Scalable abstraction layer)
# ============================================================================

class EntityType(str, Enum):
    SUPPLIER = "supplier"
    VENDOR = "vendor"
    MACHINE = "machine"
    BATCH = "batch"
    COMPONENT = "component"
    INCIDENT = "incident"
    DEFECT = "defect"


class Operation(str, Enum):
    LINEAGE = "lineage"
    HISTORY = "history"
    COMPARE = "compare"
    NEIGHBORS = "neighbors"
    AGGREGATION = "aggregation"
    SHORTEST_PATH = "shortest_path"
    DEPENDENCY = "dependency"


class IntentCategory(str, Enum):
    TRAVERSAL = "traversal"  # Path traversal / root-cause chain queries
    ANALYTICS = "analytics"  # Aggregation / summary / grouped queries


class QueryIntent(str, Enum):
    # ── Traversal Intents ───────────────────────────────────────────────────
    SUPPLIER_BATCH_TO_DEFECT     = "supplier_batch_to_defect"
    SUPPLIER_LINEAGE             = "supplier_lineage"              # full chain: supplier → batch → component → machine → maintenance → incident → defect
    BATCH_LINEAGE                = "batch_lineage"                 # batch → supplier, component, machine, maintenance, defect
    DEFECT_LINEAGE               = "defect_lineage"                # reverse traversal: defect → supplier
    COMPONENT_TRACE              = "component_trace"
    INCIDENT_HISTORY             = "incident_history"
    SUPPLIER_RISK_EXPOSURE       = "supplier_risk_exposure"        # 8-hop: defect → batch → supplier → ALL batches → components → machines → maintenance → vendor/incidents

    # ── Analytics / Summary Intents ─────────────────────────────────────────
    SUPPLIER_QUALITY_SUMMARY     = "supplier_quality_summary"      # aggregation: supplier → batches, machines, incidents, defects
    MACHINE_HISTORY              = "machine_history"               # complete operational history summary
    DEFECTS_BY_SUPPLIER          = "defects_by_supplier"           # simple: supplier → batch → defect (3 hops)
    MACHINES_BY_VENDOR           = "machines_by_vendor"
    VENDOR_MAINTENANCE_INCIDENTS = "vendor_maintenance_incidents"  # vendor → machine → incidents → defects
    FAILED_QC_BATCHES            = "failed_qc_batches"
    LIST_ALL                     = "list_all"


# Architecture: Intent → GraphOperation → Cypher Template
INTENT_OPERATIONS: dict[QueryIntent, GraphOperation] = {
    QueryIntent.SUPPLIER_BATCH_TO_DEFECT:     GraphOperation.TRAVERSAL,
    QueryIntent.SUPPLIER_LINEAGE:             GraphOperation.TRAVERSAL,
    QueryIntent.BATCH_LINEAGE:                GraphOperation.TRAVERSAL,
    QueryIntent.DEFECT_LINEAGE:               GraphOperation.TRAVERSAL,
    QueryIntent.COMPONENT_TRACE:              GraphOperation.TRAVERSAL,
    QueryIntent.INCIDENT_HISTORY:             GraphOperation.TRAVERSAL,
    QueryIntent.SUPPLIER_RISK_EXPOSURE:       GraphOperation.TRAVERSAL,

    QueryIntent.SUPPLIER_QUALITY_SUMMARY:     GraphOperation.AGGREGATION,
    QueryIntent.MACHINE_HISTORY:              GraphOperation.AGGREGATION,
    QueryIntent.DEFECTS_BY_SUPPLIER:          GraphOperation.AGGREGATION,
    QueryIntent.MACHINES_BY_VENDOR:           GraphOperation.AGGREGATION,
    QueryIntent.VENDOR_MAINTENANCE_INCIDENTS: GraphOperation.AGGREGATION,
    QueryIntent.FAILED_QC_BATCHES:            GraphOperation.AGGREGATION,
    QueryIntent.LIST_ALL:                     GraphOperation.NEIGHBORHOOD,
}


# Map each QueryIntent to its result type (LINEAGE vs SUMMARY)
INTENT_RESULT_TYPES: dict[QueryIntent, GraphResultType] = {
    QueryIntent.SUPPLIER_BATCH_TO_DEFECT:     GraphResultType.LINEAGE,
    QueryIntent.SUPPLIER_LINEAGE:             GraphResultType.LINEAGE,
    QueryIntent.BATCH_LINEAGE:                GraphResultType.LINEAGE,
    QueryIntent.DEFECT_LINEAGE:               GraphResultType.LINEAGE,
    QueryIntent.COMPONENT_TRACE:              GraphResultType.LINEAGE,
    QueryIntent.INCIDENT_HISTORY:             GraphResultType.LINEAGE,
    QueryIntent.SUPPLIER_RISK_EXPOSURE:       GraphResultType.LINEAGE,

    QueryIntent.SUPPLIER_QUALITY_SUMMARY:     GraphResultType.SUMMARY,
    QueryIntent.MACHINE_HISTORY:              GraphResultType.SUMMARY,
    QueryIntent.DEFECTS_BY_SUPPLIER:          GraphResultType.SUMMARY,
    QueryIntent.MACHINES_BY_VENDOR:           GraphResultType.SUMMARY,
    QueryIntent.VENDOR_MAINTENANCE_INCIDENTS: GraphResultType.SUMMARY,
    QueryIntent.FAILED_QC_BATCHES:            GraphResultType.SUMMARY,
    QueryIntent.LIST_ALL:                     GraphResultType.SUMMARY,
}


# Unified Mapping: (EntityType, Operation) → QueryIntent
ENTITY_OPERATION_MAP: dict[tuple[EntityType, Operation], QueryIntent] = {
    (EntityType.SUPPLIER, Operation.LINEAGE): QueryIntent.SUPPLIER_LINEAGE,
    (EntityType.BATCH, Operation.LINEAGE): QueryIntent.BATCH_LINEAGE,
    (EntityType.COMPONENT, Operation.LINEAGE): QueryIntent.COMPONENT_TRACE,
    (EntityType.DEFECT, Operation.LINEAGE): QueryIntent.DEFECT_LINEAGE,
    
    (EntityType.MACHINE, Operation.HISTORY): QueryIntent.MACHINE_HISTORY,
    (EntityType.VENDOR, Operation.HISTORY): QueryIntent.VENDOR_MAINTENANCE_INCIDENTS,
    (EntityType.INCIDENT, Operation.HISTORY): QueryIntent.INCIDENT_HISTORY,
    
    (EntityType.SUPPLIER, Operation.AGGREGATION): QueryIntent.SUPPLIER_QUALITY_SUMMARY,
    (EntityType.BATCH, Operation.COMPARE): QueryIntent.BATCH_LINEAGE,
}


def resolve_intent(entity_type: EntityType, op: Operation) -> QueryIntent:
    """
    Resolve dynamic entity type and operation pairs into a concrete Cypher QueryIntent.
    """
    intent = ENTITY_OPERATION_MAP.get((entity_type, op))
    if not intent:
        # Fallbacks for scalability
        if op == Operation.LINEAGE:
            if entity_type == EntityType.SUPPLIER:
                return QueryIntent.SUPPLIER_LINEAGE
            return QueryIntent.BATCH_LINEAGE
        elif op == Operation.COMPARE:
            return QueryIntent.BATCH_LINEAGE
        elif op == Operation.HISTORY:
            return QueryIntent.MACHINE_HISTORY
        return QueryIntent.SUPPLIER_BATCH_TO_DEFECT
    return intent



# ============================================================================
# Q001
#
# Supplier
#   ↓
# Batch
#   ↓
# Component
#   ↓
# Machine
#   ↓
# Maintenance
#   ↑
# Incident
#   ↓
# Defect
#   ↓
# Batch
#
# Canonical GraphRAG query
# ============================================================================

SUPPLIER_BATCH_TO_DEFECT = """
MATCH (s:Supplier)-[:SUPPLIED_BATCH]->(b:Batch)
MATCH (b)-[:MANUFACTURED_AS]->(c:Component)
MATCH (c)-[:INSTALLED_ON]->(m:Machine)
MATCH (m)-[:UNDERWENT]->(mt:Maintenance)
MATCH (mt)-[:PERFORMED_BY]->(v:Vendor)
MATCH (i:Incident)-[:TRIGGERED_MAINTENANCE]->(mt)
MATCH (i)-[:REPORTED_AS]->(d:Defect)
MATCH (d)-[:CAUSED_BY_BATCH]->(b)

WHERE
($machine_name IS NULL OR (m IS NOT NULL AND toLower(m.name) CONTAINS toLower($machine_name)))
AND
($vendor_name IS NULL OR (v IS NOT NULL AND toLower(v.name) CONTAINS toLower($vendor_name)))
AND
($supplier_name IS NULL OR toLower(s.name) CONTAINS toLower($supplier_name))
AND
($batch_id IS NULL OR b.id = $batch_id)

RETURN DISTINCT

s.id      AS supplier_id,
s.name    AS supplier_name,

b.id      AS batch_id,
b.name    AS batch_name,

c.id      AS component_id,
c.name    AS component_name,

m.id      AS machine_id,
m.name    AS machine_name,

mt.id     AS maintenance_id,
mt.name   AS maintenance_name,

v.id      AS vendor_id,
v.name    AS vendor_name,

i.id      AS incident_id,
i.name    AS incident_name,

d.id      AS defect_id,
d.name    AS defect_name

ORDER BY
supplier_name,
batch_name,
defect_name
"""


# ============================================================================
# Q002
#
# Maintenance history by vendor
# ============================================================================

MACHINES_BY_VENDOR = """
MATCH
(m:Machine)-[:UNDERWENT]->(mt:Maintenance)
          -[:PERFORMED_BY]->(v:Vendor)

WHERE
($vendor_name IS NULL OR toLower(v.name) CONTAINS toLower($vendor_name))

RETURN

m.id      AS machine_id,
m.name    AS machine_name,

mt.id     AS maintenance_id,
mt.name   AS maintenance_name,

v.id      AS vendor_id,
v.name    AS vendor_name

ORDER BY
machine_name,
maintenance_name
"""


# ============================================================================
# Q003
#
# Supplier -> Defects
# ============================================================================

DEFECTS_BY_SUPPLIER = """
MATCH
(s:Supplier)-[:SUPPLIED_BATCH]->(b:Batch),
(b)-[:MANUFACTURED_AS]->(:Component),
(i:Incident)-[:REPORTED_AS]->(d:Defect),
(d)-[:CAUSED_BY_BATCH]->(b)

WHERE
($supplier_name IS NULL OR toLower(s.name) CONTAINS toLower($supplier_name))

RETURN DISTINCT

s.id      AS supplier_id,
s.name    AS supplier_name,

b.id      AS batch_id,
b.name    AS batch_name,

d.id      AS defect_id,
d.name    AS defect_name

ORDER BY
batch_name,
defect_name
"""


# ============================================================================
# Q003b
#
# Supplier Lineage  (full downstream chain)
# Supplier → Batch → Component → Machine → Maintenance → Incident → Defect
#
# Answers: "Starting from supplier X, trace all downstream entities"
# Unlike DEFECTS_BY_SUPPLIER this traverses the FULL manufacturing graph
# so components, machines, and maintenance activities are all included.
# ============================================================================

SUPPLIER_LINEAGE = """
MATCH
(s:Supplier)-[:SUPPLIED_BATCH]->(b:Batch),
(b)-[:MANUFACTURED_AS]->(c:Component),
(c)-[:INSTALLED_ON]->(m:Machine),
(m)-[:UNDERWENT]->(mt:Maintenance)

OPTIONAL MATCH (mt)-[:PERFORMED_BY]->(v:Vendor)

OPTIONAL MATCH (i:Incident)-[:TRIGGERED_MAINTENANCE]->(mt)

OPTIONAL MATCH (i)-[:REPORTED_AS]->(d:Defect)

WHERE
($supplier_name IS NULL OR toLower(s.name) CONTAINS toLower($supplier_name))

RETURN DISTINCT

s.id      AS supplier_id,
s.name    AS supplier_name,

b.id      AS batch_id,
b.name    AS batch_name,

c.id      AS component_id,
c.name    AS component_name,

m.id      AS machine_id,
m.name    AS machine_name,

mt.id     AS maintenance_id,
mt.name   AS maintenance_name,

v.id      AS vendor_id,
v.name    AS vendor_name,

i.id      AS incident_id,
i.name    AS incident_name,

d.id      AS defect_id,
d.name    AS defect_name

ORDER BY
component_name,
maintenance_name,
defect_name
"""


# ============================================================================
# Q004
#
# Complete machine history
# ============================================================================

MACHINE_HISTORY = """
MATCH (m:Machine)-[:UNDERWENT]->(mt:Maintenance)

OPTIONAL MATCH (mt)-[:PERFORMED_BY]->(v:Vendor)

OPTIONAL MATCH (c:Component)-[:INSTALLED_ON]->(m)

OPTIONAL MATCH (i:Incident)-[:TRIGGERED_MAINTENANCE]->(mt)

OPTIONAL MATCH (i)-[:REPORTED_AS]->(d:Defect)

WHERE
($machine_name IS NULL OR toLower(m.name) CONTAINS toLower($machine_name))

RETURN

m.id                    AS machine_id,
m.name                  AS machine_name,
m.plant                 AS machine_plant,
m.commissioned          AS machine_commissioned,

mt.id                   AS maintenance_id,
mt.name                 AS maintenance_name,
mt.date                 AS maintenance_date,
mt.maintenance_type     AS maintenance_type,
mt.notes                AS maintenance_notes,

v.id                    AS vendor_id,
v.name                  AS vendor_name,
v.country               AS vendor_country,
v.specialty             AS vendor_specialty,

c.id                    AS component_id,
c.name                  AS component_name,

i.id                    AS incident_id,
i.name                  AS incident_name,
i.severity              AS incident_severity,
i.date                  AS incident_date,

d.id                    AS defect_id,
d.name                  AS defect_name,
d.severity              AS defect_severity,
d.date                  AS defect_date

ORDER BY
maintenance_date,
maintenance_name,
incident_name,
defect_name
"""

# ============================================================================
# Q005
#
# Failed QC batches
# ============================================================================

FAILED_QC_BATCHES = """
MATCH

(s:Supplier)-[:SUPPLIED_BATCH]->(b:Batch)

WHERE
(
    $qc_status IS NULL
    OR
    toLower(b.qc_status) CONTAINS toLower($qc_status)
)

RETURN

s.id      AS supplier_id,
s.name    AS supplier_name,

b.id      AS batch_id,
b.name    AS batch_name,

b.qc_status      AS qc_status,
b.production_date AS production_date

ORDER BY
batch_name
"""


# ============================================================================
# Q006
#
# Component Traceability
# Component -> Machine -> Maintenance -> Vendor
# ============================================================================

COMPONENT_TRACE = """
MATCH

(c:Component)
-[:INSTALLED_ON]->
(m:Machine)
-[:UNDERWENT]->
(mt:Maintenance)
-[:PERFORMED_BY]->
(v:Vendor)

OPTIONAL MATCH

(b:Batch)
-[:MANUFACTURED_AS]->
(c)

OPTIONAL MATCH

(s:Supplier)
-[:SUPPLIED_BATCH]->
(b)

WHERE

(
    $component_name IS NULL
    OR
    toLower(c.name) CONTAINS toLower($component_name)
)

RETURN

c.id AS component_id,
c.name AS component_name,

b.id AS batch_id,
b.name AS batch_name,

s.id AS supplier_id,
s.name AS supplier_name,

m.id AS machine_id,
m.name AS machine_name,

mt.id AS maintenance_id,
mt.name AS maintenance_name,

v.id AS vendor_id,
v.name AS vendor_name

ORDER BY
component_name
"""


# ============================================================================
# Q007
#
# Incident History
# ============================================================================

INCIDENT_HISTORY = """
MATCH

(i:Incident)
-[:TRIGGERED_MAINTENANCE]->
(mt:Maintenance)

OPTIONAL MATCH

(mt)-[:PERFORMED_BY]->(v:Vendor)

OPTIONAL MATCH

(i)-[:REPORTED_AS]->(d:Defect)

OPTIONAL MATCH

(d)-[:CAUSED_BY_BATCH]->(b:Batch)

OPTIONAL MATCH

(b)<-[:SUPPLIED_BATCH]-(s:Supplier)

WHERE

(
    $incident_name IS NULL
    OR
    toLower(i.name) CONTAINS toLower($incident_name)
)

RETURN

i.id AS incident_id,
i.name AS incident_name,

mt.id AS maintenance_id,
mt.name AS maintenance_name,

v.id AS vendor_id,
v.name AS vendor_name,

d.id AS defect_id,
d.name AS defect_name,

b.id AS batch_id,
b.name AS batch_name,

s.id AS supplier_id,
s.name AS supplier_name

ORDER BY
incident_name,
defect_name
"""


# ============================================================================
# Q008
#
# Defect Lineage  (anchored root-cause traversal: Defect → Supplier)
#
# Starts from the defect and follows CAUSED_BY_BATCH to get the one batch
# explicitly responsible, then recovers the full lineage chain outward.
#
# Chain:
#   Defect ──CAUSED_BY_BATCH──▶ Batch ◀──SUPPLIED_BATCH── Supplier
#   Batch  ──MANUFACTURED_AS──▶ Component ──INSTALLED_ON──▶ Machine
#   Machine ──UNDERWENT──▶ Maintenance ◀──TRIGGERED_MAINTENANCE── Incident
#   Incident ──REPORTED_AS──▶ Defect   (closes the loop for validation)
#
# Anchoring on CAUSED_BY_BATCH prevents false positives from other batches
# that happened to manufacture a component installed on the same machine.
# ============================================================================

DEFECT_LINEAGE = """
// Step 1: Anchor on the defect and follow the explicit causal edge.
MATCH (d:Defect)-[:CAUSED_BY_BATCH]->(b:Batch)

// Step 2: Recover the full downstream / upstream chain.
MATCH (s:Supplier)-[:SUPPLIED_BATCH]->(b)
MATCH (b)-[:MANUFACTURED_AS]->(c:Component)
MATCH (c)-[:INSTALLED_ON]->(m:Machine)
MATCH (m)-[:UNDERWENT]->(mt:Maintenance)
OPTIONAL MATCH (mt)-[:PERFORMED_BY]->(v:Vendor)
MATCH (i:Incident)-[:TRIGGERED_MAINTENANCE]->(mt)
MATCH (i)-[:REPORTED_AS]->(d)

WHERE
($defect_name IS NULL OR toLower(d.name) CONTAINS toLower($defect_name))

RETURN DISTINCT

s.id      AS supplier_id,
s.name    AS supplier_name,

b.id      AS batch_id,
b.name    AS batch_name,

c.id      AS component_id,
c.name    AS component_name,

m.id      AS machine_id,
m.name    AS machine_name,

mt.id     AS maintenance_id,
mt.name   AS maintenance_name,

v.id      AS vendor_id,
v.name    AS vendor_name,

i.id      AS incident_id,
i.name    AS incident_name,

d.id      AS defect_id,
d.name    AS defect_name

ORDER BY
defect_name,
supplier_name
"""


# ============================================================================
# Q009
#
# Vendor Maintenance Incidents
# vendor → maintenance → machine + (incidents) + (defects)
#
# Answers: "Which machines were serviced by vendor X and what incidents
# and defects were associated with those maintenance activities?"
# ============================================================================

VENDOR_MAINTENANCE_INCIDENTS = """
MATCH (v:Vendor)<-[:PERFORMED_BY]-(mt:Maintenance)<-[:UNDERWENT]-(m:Machine)

OPTIONAL MATCH (m)<-[:INSTALLED_ON]-(c:Component)<-[:MANUFACTURED_AS]-(b1:Batch)<-[:SUPPLIED_BATCH]-(s1:Supplier)
OPTIONAL MATCH (i:Incident)-[:TRIGGERED_MAINTENANCE]->(mt)
OPTIONAL MATCH (i)-[:REPORTED_AS]->(d:Defect)
OPTIONAL MATCH (d)-[:CAUSED_BY_BATCH]->(b2:Batch)<-[:SUPPLIED_BATCH]-(s2:Supplier)

WITH v, mt, m, c,
     coalesce(s1, s2) AS s,
     coalesce(b1, b2) AS b,
     i, d

WHERE ($vendor_name IS NULL OR toLower(v.name) CONTAINS toLower($vendor_name))

RETURN DISTINCT

v.id      AS vendor_id,
v.name    AS vendor_name,

m.id      AS machine_id,
m.name    AS machine_name,

c.id      AS component_id,
c.name    AS component_name,

b.id      AS batch_id,
b.name    AS batch_name,

s.id      AS supplier_id,
s.name    AS supplier_name,

mt.id     AS maintenance_id,
mt.name   AS maintenance_name,

i.id      AS incident_id,
i.name    AS incident_name,

d.id      AS defect_id,
d.name    AS defect_name

ORDER BY
vendor_name,
machine_name,
supplier_name
"""


# ============================================================================
# Q010
#
# Supplier Quality Summary (Aggregation Query)
# Supplier → Batch → Component → Machine → Maintenance ← Incident → Defect
#
# Grouped summary of suppliers whose batches caused defects, listing all
# affected batches, components, machines, maintenance events, incidents, defects.
# ============================================================================

SUPPLIER_QUALITY_SUMMARY = """
MATCH
(s:Supplier)-[:SUPPLIED_BATCH]->(b:Batch),
(b)-[:MANUFACTURED_AS]->(c:Component),
(c)-[:INSTALLED_ON]->(m:Machine),
(m)-[:UNDERWENT]->(mt:Maintenance)

OPTIONAL MATCH (mt)-[:PERFORMED_BY]->(v:Vendor)

MATCH (i:Incident)-[:TRIGGERED_MAINTENANCE]->(mt),
(i)-[:REPORTED_AS]->(d:Defect),
(d)-[:CAUSED_BY_BATCH]->(b)

WHERE
($supplier_name IS NULL OR toLower(s.name) CONTAINS toLower($supplier_name))

RETURN DISTINCT

s.id      AS supplier_id,
s.name    AS supplier_name,

b.id      AS batch_id,
b.name    AS batch_name,

c.id      AS component_id,
c.name    AS component_name,

m.id      AS machine_id,
m.name    AS machine_name,

mt.id     AS maintenance_id,
mt.name   AS maintenance_name,

v.id      AS vendor_id,
v.name    AS vendor_name,

i.id      AS incident_id,
i.name    AS incident_name,

d.id      AS defect_id,
d.name    AS defect_name

ORDER BY
supplier_name,
batch_name,
machine_name,
defect_name
"""


# ============================================================================
# Debug
# ============================================================================

LIST_ALL_NODES = """
MATCH (n)

RETURN

labels(n) AS labels,
n.id AS id,
n.name AS name

ORDER BY
labels,
id

LIMIT 100
"""


LIST_NODES_BY_LABEL = """
MATCH (n)

WHERE
$label IN labels(n)

RETURN

n.id AS id,
n.name AS name

ORDER BY
n.name
"""

# ============================================================================
# Q011
#
# Batch Lineage
# Batch → Supplier, Component, Machine, Maintenance, Vendor, Incident, Defect
#
# Answers: "Trace Hydraulic Seal Batch 2035-999" or "Batch 2026-001 lineage"
# ============================================================================

BATCH_LINEAGE = """
MATCH (b:Batch)

WHERE
($batch_id IS NOT NULL AND (b.id = $batch_id OR toLower(b.name) = toLower($batch_id)))
OR
($batch_name IS NOT NULL AND (toLower(b.name) = toLower($batch_name) OR b.id = $batch_name))

OPTIONAL MATCH (s:Supplier)-[:SUPPLIED_BATCH]->(b)

OPTIONAL MATCH (b)-[:MANUFACTURED_AS]->(c:Component)

OPTIONAL MATCH (c)-[:INSTALLED_ON]->(m:Machine)

OPTIONAL MATCH (m)-[:UNDERWENT]->(mt:Maintenance)

OPTIONAL MATCH (mt)-[:PERFORMED_BY]->(v:Vendor)

OPTIONAL MATCH (i:Incident)-[:TRIGGERED_MAINTENANCE]->(mt)

OPTIONAL MATCH (i)-[:REPORTED_AS]->(d:Defect)

RETURN DISTINCT

s.id      AS supplier_id,
s.name    AS supplier_name,

b.id      AS batch_id,
b.name    AS batch_name,

c.id      AS component_id,
c.name    AS component_name,

m.id      AS machine_id,
m.name    AS machine_name,

mt.id     AS maintenance_id,
mt.name   AS maintenance_name,

v.id      AS vendor_id,
v.name    AS vendor_name,

i.id      AS incident_id,
i.name    AS incident_name,

d.id      AS defect_id,
d.name    AS defect_name

ORDER BY
batch_name,
component_name
"""



# ============================================================================
# Q012 — BREAKING POINT (8 hops)
#
# Supplier Risk Exposure Assessment
#
# Defect (0)
#   →[CAUSED_BY_BATCH]→   Defective Batch (1)
#   ←[SUPPLIED_BATCH]←    Supplier (2)
#   →[SUPPLIED_BATCH]→    ALL Batches from same supplier (3)
#   →[MANUFACTURED_AS]→   Components (4)
#   →[INSTALLED_ON]→      Machines at Risk (5)
#   →[UNDERWENT]→         Maintenance Events (6)
#   →[PERFORMED_BY]→      Vendors (7)
#   ←[TRIGGERED_MAINTENANCE]← Incidents (7)
#   →[REPORTED_AS]→       Other Defects (8)
#
# Answers: "Starting from defect X, trace back to originating supplier and
#           identify all machines still at risk from other batches of the
#           same supplier"
# ============================================================================

SUPPLIER_RISK_EXPOSURE = """
// Step 1: Identify the defective batch and originating supplier from the named defect
MATCH (d_root:Defect)-[:CAUSED_BY_BATCH]->(b_defective:Batch)<-[:SUPPLIED_BATCH]-(s:Supplier)

WHERE
($defect_name IS NULL OR toLower(d_root.name) CONTAINS toLower($defect_name))

// Step 2: From that supplier, fan out to ALL batches they have ever supplied
WITH d_root, b_defective, s
MATCH (s)-[:SUPPLIED_BATCH]->(b_all:Batch)

// Step 3: Trace each batch forward through the manufacturing graph
MATCH (b_all)-[:MANUFACTURED_AS]->(c:Component)
MATCH (c)-[:INSTALLED_ON]->(m:Machine)

// Step 4: Find all maintenance events and their outcomes on those machines
OPTIONAL MATCH (m)-[:UNDERWENT]->(mt:Maintenance)
OPTIONAL MATCH (mt)-[:PERFORMED_BY]->(v:Vendor)
OPTIONAL MATCH (i:Incident)-[:TRIGGERED_MAINTENANCE]->(mt)
OPTIONAL MATCH (i)-[:REPORTED_AS]->(d_linked:Defect)

RETURN DISTINCT

d_root.id               AS root_defect_id,
d_root.name             AS root_defect_name,
d_root.severity         AS root_defect_severity,

b_defective.id          AS defective_batch_id,
b_defective.name        AS defective_batch_name,
b_defective.qc_status   AS defective_batch_qc_status,

s.id                    AS supplier_id,
s.name                  AS supplier_name,
s.country               AS supplier_country,
s.rating                AS supplier_rating,

b_all.id                AS batch_id,
b_all.name              AS batch_name,
b_all.qc_status         AS batch_qc_status,
b_all.production_date   AS batch_production_date,
b_all.quantity          AS batch_quantity,

c.id                    AS component_id,
c.name                  AS component_name,

m.id                    AS machine_id,
m.name                  AS machine_name,
m.plant                 AS machine_plant,

mt.id                   AS maintenance_id,
mt.name                 AS maintenance_name,
mt.date                 AS maintenance_date,
mt.maintenance_type     AS maintenance_type,

v.id                    AS vendor_id,
v.name                  AS vendor_name,

i.id                    AS incident_id,
i.name                  AS incident_name,
i.severity              AS incident_severity,

d_linked.id             AS linked_defect_id,
d_linked.name           AS linked_defect_name,
d_linked.severity       AS linked_defect_severity

ORDER BY
supplier_name,
batch_name,
machine_name,
maintenance_date
"""


QUERY_REGISTRY = {
    QueryIntent.SUPPLIER_BATCH_TO_DEFECT:     SUPPLIER_BATCH_TO_DEFECT,
    QueryIntent.SUPPLIER_LINEAGE:             SUPPLIER_LINEAGE,
    QueryIntent.SUPPLIER_QUALITY_SUMMARY:     SUPPLIER_QUALITY_SUMMARY,
    QueryIntent.BATCH_LINEAGE:                BATCH_LINEAGE,
    QueryIntent.DEFECT_LINEAGE:               DEFECT_LINEAGE,
    QueryIntent.MACHINES_BY_VENDOR:           MACHINES_BY_VENDOR,
    QueryIntent.DEFECTS_BY_SUPPLIER:          DEFECTS_BY_SUPPLIER,
    QueryIntent.VENDOR_MAINTENANCE_INCIDENTS: VENDOR_MAINTENANCE_INCIDENTS,
    QueryIntent.FAILED_QC_BATCHES:            FAILED_QC_BATCHES,
    QueryIntent.COMPONENT_TRACE:              COMPONENT_TRACE,
    QueryIntent.INCIDENT_HISTORY:             INCIDENT_HISTORY,
    QueryIntent.MACHINE_HISTORY:              MACHINE_HISTORY,
    QueryIntent.SUPPLIER_RISK_EXPOSURE:       SUPPLIER_RISK_EXPOSURE,
    QueryIntent.LIST_ALL:                     LIST_ALL_NODES,
}

REQUIRED_PARAMETERS = {
    QueryIntent.SUPPLIER_BATCH_TO_DEFECT: [
        "machine_name",
        "vendor_name",
        "supplier_name",
        "batch_id",
    ],
    QueryIntent.SUPPLIER_LINEAGE: [
        "supplier_name",
    ],
    QueryIntent.SUPPLIER_QUALITY_SUMMARY: [
        "supplier_name",
    ],
    QueryIntent.BATCH_LINEAGE: [
        "batch_id",
        "batch_name",
    ],
    QueryIntent.DEFECT_LINEAGE: [
        "defect_name",
    ],
    QueryIntent.MACHINES_BY_VENDOR: [
        "vendor_name",
    ],
    QueryIntent.DEFECTS_BY_SUPPLIER: [
        "supplier_name",
    ],
    QueryIntent.VENDOR_MAINTENANCE_INCIDENTS: [
        "vendor_name",
    ],
    QueryIntent.COMPONENT_TRACE: [
        "component_name",
    ],
    QueryIntent.MACHINE_HISTORY: [
        "machine_name",
    ],
    QueryIntent.INCIDENT_HISTORY: [
        "incident_name",
    ],
    QueryIntent.FAILED_QC_BATCHES: [
        "qc_status",
    ],
    QueryIntent.SUPPLIER_RISK_EXPOSURE: [
        "defect_name",
    ],
    QueryIntent.LIST_ALL: [],
}

def get_query(intent: QueryIntent) -> str:
    return QUERY_REGISTRY[intent]

def required_parameters(intent: QueryIntent) -> list[str]:
    return REQUIRED_PARAMETERS.get(intent, [])