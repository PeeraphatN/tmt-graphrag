# Concept Paper: From Flat Composite Intent to Hierarchical Intent Classification (Action + Topics + Slots)

Date: 2026-02-20  
Project: GenAI GraphRAG (TMT domain)  
Scope: Intent routing design for retrieval and search planning

## 1) Executive Summary
The current intent design is workable for core in-domain questions, but it does not match real query patterns well enough.  
The largest issue is that the current labels are cross-product labels (`topics_action`, such as `manufacturer_find`) with only 3 actions (`find/check/count`).  
Real traffic includes patterns that are not represented explicitly (for example ID lookup, list-style requests, compare-like requests, and out-of-domain chat).

This paper proposes Hierarchical Intent Classification (HIC):
- `action_intent` (single-label): `lookup`, `verify`, `list`, `count`, `compare`, `unknown`
- `topics_intents` (multi-label): `manufacturer`, `substance`, `nlem`, `formula`, `hierarchy`, `id_lookup`
- `slots` (structured extracted values): `drug`, `tmtid`, `manufacturer`, `strength`, etc.

HIC is expected to improve routing quality and retrieval precision/recall for mixed or compositional queries, with a small acceptable latency increase.

## 2) Current Approach (As-Is)
Current intent dataset and classifier behavior:
- Dataset: `experiments/question_understanding/intent_classification/data/source_snapshots/legacy_intent_dataset.json`
- Labels: 18 (`6 topics x 3 actions`)
- Samples: 270 total
- Actions represented: `find`, `check`, `count` only
- Topics represented: `manufacturer`, `substance`, `nlem`, `hierarchy`, `formula`, `general`

Current AQT behavior:
- Vector centroid classifier predicts a fine label and target/topics-like class.
- Rule-based strategy maps to `retrieve`, `verify`, `count`.
- No explicit `list`, `compare`, or `unknown` action class.
- No explicit `id_lookup` topics class.

## 3) Evidence from Real Data
Real query source used for assessment:
- `logs/ragas_data.jsonl`
- 44 total queries, 25 unique

Observed real query patterns:
- Queries with `TMTID` keyword: 2/44 (4.5%)
- Trade-name request style (for example "want trade names"): 17/44 (38.6%)
- Compare-like phrase: 1/44 (2.3%)
- OOD/chat-like greeting: 1/44 (2.3%)

Runtime intent behavior on these 44 queries (current pipeline):
- Target distribution:
  - `GENERAL`: 20
  - `MANUFACTURER`: 14
  - `NLEM`: 8
  - `FORMULA`: 2
- Strategy distribution:
  - `RETRIEVE`: 33
  - `VERIFY`: 8
  - `COUNT`: 3

Confidence quality signal:
- Margin `< 0.03`: 22/44 queries (50.0%)
- Median margin: 0.0292

Interpretation:
- Many queries are routed to generic retrieve behavior.
- A large portion of traffic is near decision boundaries.
- Existing label space is too narrow for real request variety.

## 4) Experimental Design
Objective:
- Compare Flat Composite Intent (FCI) vs Hierarchical Intent Classification (HIC) under the same data and embedding conditions.
- Verify not only in-domain performance, but also compositional generalization.

Primary hypotheses:
- H1: In easy in-domain splits, FCI and HIC are similar.
- H2: In compositional holdout splits, HIC outperforms FCI on action/topics routing quality.

Systems under test:
- FCI (Flat Composite Intent, baseline):
  - Single centroid classifier over fine labels (`topics_action`, 18 classes).
  - Predicted fine label is back-parsed to action and topics.
- HIC (Hierarchical Intent Classification):
  - Split-head centroid classifiers:
    - action head
    - topics head
  - Final decision is composed from both heads.

Dataset and source:
- Source file: `experiments/question_understanding/intent_classification/data/source_snapshots/legacy_intent_dataset.json`
- Derived HIC reference file: `experiments/question_understanding/intent_classification/data/intent_dataset_v2_reference.json`
- Total samples used in benchmark: 270
- Note: This benchmark uses existing text only (no new text augmentation yet).

Controlled variables:
- Same embedding model for both systems (`bge-m3`).
- Same train/test partitions for both systems in each experiment.
- Same cosine-centroid nearest prediction rule.
- Same preprocessing path for labels and records.
- Fixed random seed: 42.

Experimental splits:
- Split A: Random Stratified Split
  - Method: split each fine label into train/test with constant ratio.
  - Purpose: estimate in-domain performance where train/test distributions are similar.
- Split B: Compositional Holdout Split
  - Method: hide selected fine labels from training, evaluate on those hidden labels only.
  - Holdout labels used:
    - `manufacturer_check`
    - `substance_count`
    - `nlem_find`
    - `hierarchy_check`
    - `formula_count`
    - `general_find`
  - Purpose: test generalization to unseen action+topics combinations.
  - Meaning of "hidden":
    - FCI never sees those exact fine labels.
    - HIC can still combine learned action and topics patterns from separate heads.

Evaluation metrics:
- `action_acc`, `action_f1`: ability to predict what user wants to do.
- `topics_acc`, `topics_f1`: ability to predict which information dimension is needed.
- `joint_acc`: strict score, requires both action and topics to be correct on the same sample.
- Inference latency:
  - measured on prediction step only (embedding already computed),
  - averaged over repeated loops.

Interpretation protocol:
- Random split results indicate "fit to seen pattern families".
- Compositional holdout results indicate "robustness to unseen composition".
- For production-oriented routing, compositional metrics have higher decision weight.

Limitations of current experiment design:
- No slot extraction evaluation yet (`tmtid`, manufacturer span, strength span, etc.).
- No end-to-end retrieval quality metrics in this benchmark (`P@k`, `R@k`, answer faithfulness).
- Single-seed reporting (future work should add repeated seeds and confidence intervals).

## 5) Experimental Benchmark Evidence
Experiment script:
- `experiments/question_understanding/intent_classification/intent_structure_fci_hic/benchmark_intent_v2_vs_legacy.py`
Result file:
- `experiments/question_understanding/intent_classification/results/benchmark_intent_v2_vs_legacy_20260220_134337.json`

### 5.1 Random stratified split (in-domain style)
Full metrics table:

| Model | action_acc | action_f1 | topics_acc | topics_f1 | joint_acc | latency_ms |
|---|---:|---:|---:|---:|---:|---:|
| FCI | 0.8667 | 0.8647 | 0.8333 | 0.8320 | 0.7333 | 0.2077 |
| HIC | 0.8667 | 0.8647 | 0.8333 | 0.8343 | 0.7000 | 0.2843 |
| Delta (HIC - FCI) | +0.0000 | +0.0000 | +0.0000 | +0.0023 | -0.0333 | +0.0765 |

Support and class scope:

| Item | Value |
|---|---:|
| test_size | 90 |
| FCI class count (fine labels) | 18 |
| HIC action class count | 3 |
| HIC topics class count | 6 |

Reading:
- FCI and HIC are similar on top-line action/topics in easy in-domain split.
- HIC has slightly lower strict `joint_acc` in this split.

### 5.2 Compositional holdout split (harder generalization test)
Full metrics table:

| Model | action_acc | action_f1 | topics_acc | topics_f1 | joint_acc | latency_ms |
|---|---:|---:|---:|---:|---:|---:|
| FCI | 0.3333 | 0.3465 | 0.6111 | 0.6077 | 0.0000 | 0.2007 |
| HIC | 0.5333 | 0.5360 | 0.6667 | 0.6727 | 0.2556 | 0.3028 |
| Delta (HIC - FCI) | +0.2000 | +0.1895 | +0.0556 | +0.0650 | +0.2556 | +0.1021 |

Support and holdout setting:

| Item | Value |
|---|---:|
| test_size | 90 |
| FCI class count seen in train | 12 |
| HIC action class count | 3 |
| HIC topics class count | 6 |

Holdout fine labels:
- `manufacturer_check`
- `substance_count`
- `nlem_find`
- `hierarchy_check`
- `formula_count`
- `general_find`

Interpretation:
- HIC is clearly better when query combinations are less seen or unseen.
- This is the expected production-like benefit for evolving query styles.

### 5.3 Real Query Case Studies (Production-Like Examples)
Real query sample:
- Source: `logs/ragas_data.jsonl`
- Evaluated unique queries: 25
- Comparison artifact: `experiments/question_understanding/intent_classification/results/real_query_fci_vs_hic_preview.json`
- Cases where FCI and HIC disagree: 6/25

Representative examples:

| Real query | Expected intent meaning | FCI output | HIC output | Observation |
|---|---|---|---|---|
| `Ã Â¸Å¾Ã Â¸Â²Ã Â¸Â£Ã Â¸Â²Ã Â¸Â­Ã Â¸Â¢Ã Â¸Â¹Ã Â¹Ë†Ã Â¹Æ’Ã Â¸â„¢Ã Â¸Å¡Ã Â¸Â±Ã Â¸ÂÃ Â¸Å Ã Â¸ÂµÃ Â¸Â¢Ã Â¸Â²Ã Â¸Â«Ã Â¸Â¥Ã Â¸Â±Ã Â¸ÂÃ Â¹Æ’Ã Â¸Å Ã Â¹Ë†Ã Â¸Â«Ã Â¸Â£Ã Â¸Â·Ã Â¸Â­Ã Â¹â€žÃ Â¸Â¡Ã Â¹Ë†` | `verify + nlem` | `lookup + nlem` | `verify + nlem` | HIC aligns better with yes/no verification intent. |
| `Paracetamol Ã Â¸Â­Ã Â¸Â¢Ã Â¸Â¹Ã Â¹Ë†Ã Â¹Æ’Ã Â¸â„¢Ã Â¸Å¡Ã Â¸Â±Ã Â¸ÂÃ Â¸Å Ã Â¸ÂµÃ Â¸Â¢Ã Â¸Â²Ã Â¸Â«Ã Â¸Â¥Ã Â¸Â±Ã Â¸ÂÃ Â¹Æ’Ã Â¸Å Ã Â¹Ë†Ã Â¸Â«Ã Â¸Â£Ã Â¸Â·Ã Â¸Â­Ã Â¹â€žÃ Â¸Â¡Ã Â¹Ë†` | `verify + nlem` | `lookup + nlem` | `verify + nlem` | Same pattern in English/Thai mixed form; HIC is more action-correct. |
| `MACROPHAR Ã Â¸Å“Ã Â¸Â¥Ã Â¸Â´Ã Â¸â€¢Ã Â¸Â¢Ã Â¸Â²Ã Â¸Å¾Ã Â¸Â²Ã Â¸Â£Ã Â¸Â²Ã Â¹â€žÃ Â¸Â«Ã Â¸Â¡` | `verify + manufacturer` | `lookup + manufacturer` | `verify + manufacturer` | HIC captures verification cue from "Ã Â¹â€žÃ Â¸Â«Ã Â¸Â¡" more reliably. |
| `paracetamal Ã Â¸ÂÃ Â¸ÂµÃ Â¹Ë† ml Ã Â¹â‚¬Ã Â¸â€ºÃ Â¹â€¡Ã Â¸â„¢Ã Â¸Â­Ã Â¸Â±Ã Â¸â„¢Ã Â¸â€¢Ã Â¸Â£Ã Â¸Â²Ã Â¸Â¢` | likely safety/unknown (not simple count) | `lookup + formula` | `count + formula` | Both are weak here; action space lacks safety-risk intent. |
| `Ã Â¸ÂªÃ Â¸Â§Ã Â¸Â±Ã Â¸ÂªÃ Â¸â€Ã Â¸Âµ Ã Â¸Å“Ã Â¸Â¡Ã Â¸Å Ã Â¸Â·Ã Â¹Ë†Ã Â¸Â­Ã Â¸Â­Ã Â¸Â°Ã Â¹â€žÃ Â¸Â£Ã Â¹Æ’Ã Â¸Â«Ã Â¹â€°Ã Â¸â€”Ã Â¸Â²Ã Â¸Â¢` | `unknown` (OOD/chit-chat) | `verify + general` | `lookup + general` | Both fail as expected because `unknown` class is not modeled yet. |
| `Ã Â¸â€šÃ Â¸Â­ TMTID Ã Â¸â€šÃ Â¸Â­Ã Â¸â€¡Ã Â¸Â¢Ã Â¸Â² cefazolin ...` | `lookup + id_lookup` | `lookup + hierarchy` | `lookup + hierarchy` | Both miss `id_lookup` topics because this topics is absent in current label space. |

What these cases show:
- HIC improves action semantics on verification-style real queries.
- Some production patterns still fail under both methods due to taxonomy gaps, not model weakness alone.
- Priority gaps to close in next data design:
  - `unknown` action
  - `id_lookup` topics
  - safety/toxicity style intents (or explicit fallback policy to non-RAG flow)

## 6) Problem Statement
Why change is necessary:
- The current taxonomy couples "what user wants to do" and "which information area is needed" into one label.
- Cross-product labels do not scale well and are hard to extend.
- Important production intents are not first-class citizens (`list`, `compare`, `unknown`, `id_lookup`).
- Routing is overly biased toward generic retrieve path under ambiguity.

## 7) Proposed Hierarchical Intent Classification (To-Be)
### 7.1 Intent schema
- `action_intent` (single-label):
  - `lookup`, `verify`, `list`, `count`, `compare`, `unknown`
- `topics_intents` (multi-label):
  - `manufacturer`, `substance`, `nlem`, `formula`, `hierarchy`, `id_lookup`
- `slots` (structured extraction):
  - `drug_name`, `tmtid`, `manufacturer`, `strength`, `dose_form`, `nlem_category`, etc.

### 7.2 Routing logic
- `action_intent` selects operator family:
  - `lookup` -> hybrid retrieve
  - `verify` -> existence/claim verification flow
  - `list` -> list retrieval with filters and limits
  - `count` -> aggregate count query
  - `compare` -> dual retrieval + diff
  - `unknown` -> safe fallback or clarification
- `topics_intents` choose search scope/boost/filter/traversal depth.
- `id_lookup` has highest override priority (exact lookup path first).
- `slots` become deterministic constraints for Cypher and fulltext.

## 8) Expected Benefits
- Better precision/recall due to clearer routing and constraints.
- Better compositional generalization (supported by holdout benchmark).
- Lower token waste for small local LLM by reducing irrelevant context.
- Better observability and debugging (scores per action/topics + slot source).

## 9) Trade-offs and Risks
- Slightly higher inference latency (two heads + slot resolver).
- Higher implementation complexity than a single flat classifier.
- Requires re-labeling and data pipeline support for action/topics/slot format.

Mitigation:
- Keep fast deterministic pre-checks (`id_lookup`, units, obvious patterns).
- Cache embeddings and route decisions.
- Add confidence thresholds and fallback policy.

## 10) Success Criteria for Adoption
Adopt HIC if the following are met:
- Action accuracy and topics F1 improve on compositional holdout.
- End-to-end retrieval improves on production-like eval set (Precision@k, Recall@k, answer faithfulness).
- Token usage per answer decreases for equivalent or better answer quality.
- No major latency regression beyond agreed budget.

## 11) Recommendation
Proceed with HIC design and phased rollout.  
Maintain FCI path as fallback during transition, but use HIC as the primary experiment track for production-oriented evaluation.

