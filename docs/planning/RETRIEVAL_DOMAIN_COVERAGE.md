# Retrieval Domain Coverage (Current PoC)

Date: 2026-03-08  
Scope: GenAI TMT GraphRAG (current implementation)

## 1) Purpose
Define what the current retrieval system can answer reliably, what is partially supported, and what is out of scope.

This file is the contract for:
- query design
- evaluation dataset design
- expected behavior in demo

## 2) Coverage Tiers

### Tier A: Reliable (Deterministic / Cypher-first)
Use when query has clear anchor and strict condition.

- `id_lookup`
  - Example: `TMTID 662401 คือยาอะไร`
  - Expected: high precision, stable repeatability

- `list` by manufacturer name
  - Example: `รายการยาของผู้ผลิต GPO`
  - Expected: returns TP list with manufacturer match

- `count` by substance/manufacturer/basic hierarchy
  - Example: `จำนวนยาที่มีส่วนผสม selexipag`
  - Expected: deterministic numeric answer (no LLM hallucinated count)

### Tier B: Moderate (Hybrid Retrieval)
Use when query is semantic or under-specified.

- `lookup` (general info)
  - Example: `ข้อมูลยา paracetamol`
  - Expected: good recall, quality depends on ranking/context packing

- `verify`
  - Example: `MACROPHAR ผลิต paracetamol หรือไม่`
  - Expected: verdict + evidence; sensitive to slot extraction quality

- `compare`
  - Example: `เปรียบเทียบ paracetamol กับ ibuprofen`
  - Expected: acceptable for basic compare; can degrade on noisy NER

### Tier C: Out of Scope (Current PoC)
- NLEM policy questions (currently blocked by system policy)
- Clinical recommendation / treatment decision
- Side-effect counseling beyond evidence in current TMT graph
- Non-TMT external knowledge

## 3) Supported Query Axes

- Action axis: `lookup`, `list`, `count`, `verify`, `compare`, `id_lookup`
- Topic axis (effective): `manufacturer`, `substance`, `formula`, `hierarchy`, `general`
- Data axis: TMT node/relations (SUBS, VTM, GP, GPU, TP, TPU + manufacturer/trade_name/fsn/tmtid)

## 4) Current Gap (Important)

Query family:
- `manufacturer list by substance + strength`
- Example: `ชื่อโรงงานที่ผลิตยา Paracetamol 500 mg ทั้งหมด 5 ชื่อ`

Why it fails in some runs:
- route may not anchor to TP/manufacturer strongly enough
- extraction may keep generic query text without strong manufacturer intent
- retrieval may return GP-heavy results (no trade/manufacturer-rich context)

Status:
- Partially supported after recent TP-focused list improvements
- Still not guaranteed for all phrasings

## 5) Acceptance Rules for "In-Domain Answerable"

A query is considered "in-domain answerable" if:
- It has at least one anchor: `tmtid` or `manufacturer` or `substance/drug`
- Requested output can be derived from TMT graph fields/relations
- It does not require blocked/out-of-scope datasets (e.g., NLEM policy mode now)

If not satisfied:
- system should return controlled fallback
- avoid fabricated detail

## 6) Demo-Safe Query Set (Recommended)

Prefer these during demo:
- Exact ID lookup
- List by manufacturer
- Count by substance
- Trade-name listing with clear substance (+ optional strength)
- Basic compare with two explicit drug names

Avoid these during demo (until upgraded):
- open-ended clinical advice
- deep policy/NLEM reimbursement queries
- heavily ambiguous Thai colloquial queries without anchor entity

## 7) Next Retrieval Work (after domain lock)

1. Strengthen manufacturer-by-substance route (TP-first with strength constraint)  
2. Add query-shape classifier for `โรงงาน/ผู้ผลิต` phrasing robustness  
3. Add evaluation slice by query family (not only overall average)
