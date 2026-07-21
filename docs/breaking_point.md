# Breaking Point Analysis

## The Canonical Query

> **"Which supplier batches indirectly caused all defects reported after Robotic Welding Cell RW-101's maintenance performed by Apex Industrial Services Pvt. Ltd.?"**

## Why Vector RAG Fails

### 1. Information Fragmentation
The answer to this query requires synthesizing information from **6 different documents**:

| Document | Contains |
|----------|----------|
| `shakti_seals_supplier.txt` | Shakti Industrial Seals Pvt. Ltd. supplies Batch BAT_HS_2026_001 |
| `robotic_welding_cell_rw101.txt` | Machine has Hydraulic Pressure Seal installed |
| `maintenance_mnt_rw101_002.txt` | Apex Industrial Services Pvt. Ltd. performed emergency maintenance MNT_RW101_002 |
| `incident_inc_rw101_001.txt` | Incident INC_RW101_001 (Hydraulic Pressure Loss) linked to seal failure |
| `defect_def_hyd_leak.txt` | Defect DEF_HYD_LEAK (Hydraulic Seal Leakage, critical) |
| `defect_def_dimension.txt` | Defect DEF_DIMENSION (Seal Dimensional Failure, high) |
| `vendor_apex.txt` | Apex is vendor VEN_APEX |

**No single chunk contains the full chain.** Vector similarity search returns the top-k chunks most semantically similar to the query, but:
- Chunk A says "Shakti Industrial Seals supplied BAT_HS_2026_001"
- Chunk B says "Seal from BAT_HS_2026_001 installed on Robotic Welding Cell RW-101"
- Chunk C says "Apex Industrial Services performed emergency maintenance"
- Chunk D says "DEF_HYD_LEAK and DEF_DIMENSION raised after seal failure"

These chunks are **individually retrieved but never connected**. The LLM sees independent fragments and cannot reliably synthesize the causal chain.

### 2. Semantic Similarity ≠ Relationship Traversal
The query asks for entities **indirectly** connected through typed relationships. Vector similarity matches on semantics (keywords, topics) — not on typed graph edges.

A vector search for "supplier batches causing defects after maintenance" will return documents mentioning these keywords, but has no concept of:
- `SUPPLIED_BATCH` direction (which supplier, which batch)
- `UNDERWENT` edge (which machine underwent which maintenance event)
- The 5-hop chain connecting all of them

### 3. The Full Relationship Chain

```
SUP_SHAKTI (Shakti Industrial Seals Pvt. Ltd.)
  └─[SUPPLIED_BATCH]→ BAT_HS_2026_001 (Hydraulic Seal Batch 2026-001)
       └─[MANUFACTURED]→ COMP_HYD_SEAL (Hydraulic Pressure Seal)
            └─[INSTALLED_ON]→ MACH_RW101 (Robotic Welding Cell RW-101)
                 └─[UNDERWENT]→ MNT_RW101_002 (Emergency Maintenance RW101)
                      └─[PERFORMED_BY]→ VEN_APEX (Apex Industrial Services Pvt. Ltd.)
                           (MNT_RW101_002)
                           └─[TRIGGERED_MAINTENANCE]→ INC_RW101_001 (Hydraulic Pressure Loss)
                                └─[REPORTED_AS]→ DEF_HYD_LEAK (Hydraulic Seal Leakage)
                                └─[REPORTED_AS]→ DEF_DIMENSION (Seal Dimensional Failure)
```

**5 hops. 9 entity types. All connected by typed relationships.**

This is the breaking point.

## Why GraphRAG Succeeds

The Cypher query traverses this exact path:
```cypher
MATCH (sup:Supplier)-[:SUPPLIED_BATCH]->(b:Batch)
      -[:MANUFACTURED]->(c:Component)
      -[:INSTALLED_ON]->(m:Machine)
      -[:UNDERWENT]->(mt:Maintenance)
      -[:PERFORMED_BY]->(v:Vendor)
WHERE m.name = 'Robotic Welding Cell RW-101'
  AND v.name = 'Apex Industrial Services Pvt. Ltd.'
WITH sup, b, c, m, mt, v
MATCH (mt)-[:TRIGGERED_MAINTENANCE]->(inc:Incident)-[:REPORTED_AS]->(d:Defect)
RETURN DISTINCT sup.name, b.id, d.id, d.name
```

**The graph traversal follows typed edges. No ambiguity. No fragmentation. Exact answer.**
