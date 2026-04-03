# Codex Persistent Knowledge

This file is a persistent memory reference for ongoing work in this repository.
Update it when architecture, workflow, or operating rules change.

## Project Context

- Repository: `C:\Work\PlayGround\Intern\GenAI`
- Domain: GraphRAG for TMT/healthcare
- Current initiative: Intent V2 + adaptive hybrid retrieval + NER integration

## Branching Policy (Current)

- `master`: frozen baseline reference, do not use as active dev line.
- `integration/intent-v2-hybrid`: main integration branch for current initiative.
- `task/*`: short-lived implementation branches, merge quickly into integration.
- `experiment/*`: benchmark/evidence branches only, not main development path.

## Known Active Branches

- `integration/intent-v2-hybrid`
- `task/intent-v2-feature-flag-scaffold`
- `experiment/intent-classification-benchmarks`
- `experiment/ner-wangchanberta` (remote behind local by 3 commits; contains heavy artifacts)
- `experiment/ner-wangchanberta-lite` (train + final fine-tuned model, lighter publish path)

## Baseline and Milestone Tags

- `baseline-master-20260209`
- `exp-intent-benchmarks-20260223`
- `exp-ner-wangchanberta-20260210`
- `exp-search-hybrid-rerank-20260219`

## Intent V2 Design Anchors

- Intent schema: `action_intent` + `topics_intents` + `slots`
- Bundle contract file: `apps/api/src/schemas/intent_bundle.py`
- Shadow compare script: `experiments/question_understanding/intent_classification/integration_with_app/shadow_compare/test_intent_bundle_shadow.py`
- Latest shadow logs: `test_results/*intent_bundle_shadow_compare.jsonl`

## Integration Rules

1. Prefer feature flags over branch explosion.
2. Keep task branches small and fast to merge.
3. Preserve reproducibility with timestamped outputs in `test_results/` or `experiments/.../results/`.
4. Do not rewrite history on shared branches unless explicitly approved.

## Feature Flags (Target)

- `INTENT_V2_ENABLED`
- `INTENT_V2_USE_NER`
- `INTENT_V2_ADAPTIVE_PLANNER`

## Environment Notes

- Python env: `.venv` in repo root.
- Core deps validated in venv (neo4j, ollama, langchain, fastapi, numpy, sklearn).
- `.env` required and currently present.
- Docker file in this repo is `docker-compose.yml` (not `compose.yaml`).

## Operational Checks

Run quick environment + pipeline smoke:

```powershell
.\.venv\Scripts\python.exe -c "import sys; from pathlib import Path; sys.path.insert(0, str(Path('apps/api').resolve())); from src.config import validate_env; validate_env(); print('ENV_OK')"
.\.venv\Scripts\python.exe experiments\question_understanding\intent_classification\integration_with_app\shadow_compare\test_intent_bundle_shadow.py
```

## Current Risks / Watchlist

- Python 3.14 + legacy pydantic-v1 paths in some dependencies can emit warnings.
- Large model artifacts in some experiment branches may cause push timeout unless using Git LFS.
- Keep `AGENTS.md` ignored locally if it appears as untracked noise.

## Flow Audit TODO (2026-03-07)

Top-impact tasks from latest flow audit:

1. `[DONE][VALIDATED]` Fix fulltext query construction to avoid over-constrained Lucene queries.
   - Target: `apps/api/src/query_processor.py`
   - Goal: recover non-zero fulltext channel hits on list/lookup queries.
2. `[DONE][VALIDATED]` Add robust graph anchor fallback when fulltext anchor misses.
   - Target: `apps/api/src/services/search.py` (`_graph_anchor_search`)
   - Goal: make traversal usable for Thai/manufacturer-heavy queries.
3. `[DONE][VALIDATED]` Add deterministic-empty fallback in manufacturer list route.
   - Target: `apps/api/src/services/search.py` (`execute_listing_query`)
   - Goal: avoid zero-result hard fail; fallback to graph-hybrid route.

Next queue after top 3:

4. `[DONE][VALIDATED]` Remove/relax hard cap effect for large manufacturer result sets (`k=200`) during eval mode.
   - Change: `apps/api/src/services/search.py` now supports eval-aware cap relaxation via `RETRIEVAL_EVAL_MODE`.
   - Validation: top-20 large manufacturer recall with `k=200` moved from `0.528` (eval off) to `1.000` (eval on).
5. `[DONE][VALIDATED]` Align fulltext index schema in DB with search expectations (handle existing IF NOT EXISTS drift).
   - Change: `apps/api/src/services/database.py` now detects fulltext schema drift and auto-rebuilds index.
   - Validation: `tmt_fulltext_index` properties reconciled from `['name','fsn','embedding_text']` to expected search fields.
6. `[DONE][VALIDATED]` Rationalize reranker initialization (single load path) to reduce startup/VRAM overhead.
   - Change: singleton accessor `get_reranker()` in `apps/api/src/services/ranking_service.py`; both `pipeline` and `search` use shared instance.
   - Validation: runtime check confirms `GraphRAGPipeline().reranker is get_reranker()` is `True`.

## Model Candidate Notes (2026-03-07)

### Candidate: `ThaiLLM/ThaiLLM-8B-SFT-IQ`

- Scope: Thai medical QA model specialized for citation-grounded RAG.
- Positioning from model card: intended for medical information query with context-only answering + citations.
- Important limitation: model card explicitly states it does **not** include a retriever; must supply external context.
- Baseline architecture/details: Qwen3-family, decoder-only, 8B, BF16, context length 4096.
- Reported benchmark on card (medical query setting):
  - `Qwen3-8B-Bas`: BLEU 0.267, citation Jaccard 0.075
  - `ThaiLLM-8B-SFT`: BLEU 0.406, citation Jaccard 0.133
  - `ThaiLLM-8B-SFT-IQ`: BLEU 0.4363, citation Jaccard 0.5485
- Prompting style: model card suggests strict JSON with citations and context-only answer policy.

### Deployment Fit for This Repo

- Positive fit:
  - Aligns with current direction (RAG + citation-grounded answer generation).
  - Thai-focused domain behavior is relevant for TMT/healthcare use cases.
- Cautions:
  - 4096 context is smaller than some modern long-context models; retrieval packaging must stay compact.
  - Need strict prompt contract to preserve JSON+citation output format.
  - Inference cost/latency for 8B BF16 may be high on limited local hardware.
  - Verify licensing/governance policy before production use (license field is not clearly surfaced in quick card scrape).

### Practical Runtime Options

- Original HF weights (safetensors): ~16.4 GB repo size.
- Community GGUF options available (mradermacher static + imatrix quants), including practical local sizes such as:
  - `Q4_K_M` ~5.0 GB (common speed/quality balance)
  - `Q6_K` ~6.7 GB (higher quality)
  - `Q8_0` ~8.7 GB (best quality among common quant tiers)

### Source Links

- Base model card: `https://huggingface.co/ThaiLLM/ThaiLLM-8B-SFT-IQ`
- Base files: `https://huggingface.co/ThaiLLM/ThaiLLM-8B-SFT-IQ/tree/main`
- GGUF (static): `https://huggingface.co/mradermacher/ThaiLLM-8B-SFT-IQ-GGUF`
- GGUF (imatrix): `https://huggingface.co/mradermacher/ThaiLLM-8B-SFT-IQ-i1-GGUF`

## Intent/NER Priority Matrix (2026-02-23)

| ID | Group | Problem | Impact | Effort | Priority |
|---|---|---|---|---|---|
| I2 | Intent | Retrieval profile biased to `fulltext_heavy` in most test cases | Lower recall for abstract/general queries | M | P0 |
| N3 | NER | Noisy NER slots flow directly into retrieval plan | Immediate precision drop from wrong query/filter | M | P0 |
| N1 | NER | Over-extraction (whole sentence captured as entity) | Retrieval drift and token waste | M | P0 |
| I1 | Intent | Topic routing misses domain topic in some cases (e.g., substance -> general) | Wrong search scope, lower precision/recall | M | P1 |
| N2 | NER | Label confusion (e.g., `tmtid` interpreted as `brand`) | Slot semantic mismatch | M | P1 |
| I3 | Intent | High ambiguity rate on target prediction | Unstable routing decisions | S-M | P1 |
| I4 | Intent | `compare` intent does not split A/B entities yet | Limited compare retrieval quality | M | P2 |
| N4 | NER | Slot confidence uses coarse fixed value | Weak confidence-aware filtering/ranking | S | P2 |

### Execution Order

1. P0 first: `I2 -> N1/N3`
2. P1 next: `I1 -> N2 -> I3`
3. P2 last: `I4 -> N4`

## Retrieval PoC Gate (Semantic-First, Locked 2026-03-07)

- Criteria source: `experiments/retrieval_eval/eval/poc_acceptance_criteria_semantic_v1.json`
- Checker script: `experiments/retrieval_eval/check_poc_acceptance_semantic.py`
- Latest evaluation run input:
  - `experiments/retrieval_eval/results/phase3_uniform_static_runs_20260307_135203.jsonl`
- Latest gate report:
  - `experiments/retrieval_eval/results/phase3_uniform_static_runs_20260307_135203_poc_semantic_acceptance.json`

### Current Gate Status

- `uniform`: `FAIL`
- `static`: `FAIL`

### Largest Gaps (both policies)

1. `overall_semantic_ndcg@10` just below target (`uniform=0.5951`, `static=0.5929`, target `>= 0.60`)
2. `count_exact_match` low (`0.1885 < 0.45`)
3. `count_mape` high (`162.72 > 40.0`)

### Post-fix Snapshot (2026-03-07)

- Retrieval improvements after lookup/list routing fixes:
  - `lookup_hit@10`: `0.3313 -> 0.5875`
  - `list_hit@10`: `0.7000 -> 0.9333`
  - `overall_semantic_hit@10`: `~0.60 -> ~0.81`

## Lookup Fallback Ablation (2026-03-08)

Purpose:
- Validate that lookup deterministic fallback materially improves anchor-style semantic lookup
  (trade-name/manufacturer phrasing), not only generic phase-2 silver queries.

Code / outputs:
- Script: `experiments/retrieval_eval/run_lookup_fallback_ablation.py`
- Result JSONL: `experiments/retrieval_eval/results/lookup_fallback_ablation_20260308_224634.jsonl`
- Summary: `experiments/retrieval_eval/results/lookup_fallback_ablation_20260308_224634_summary.json`

Setup:
- Policy: `static`
- Query generation from `phase1_ground_truth` anchors
- Samples: `20` substance anchors + `20` manufacturer anchors
- Total queries: `120` (all resolved to `lookup`)
- Comparison:
  1. baseline lookup: `search_general(...)` (no fallback)
  2. lookup fallback path: `execute_lookup_query(...)`

Observed impact (fallback - baseline):
- `hit@1`: `+0.100000`
- `hit@3`: `+0.141667`
- `hit@5`: `+0.108333`
- `hit@10`: `+0.233333`
- `mrr`: `+0.140514`
- `ndcg@10`: `+0.135777`
- Avg latency: `+17.55 ms`

Route diagnostics:
- `fallback_attempt_count`: `120/120`
- `fallback_used_count`: `98/120`
- `resolved_operator_distribution`: `lookup=120`

Benchmark alignment change:
- `experiments/retrieval_eval/run_phase3_uniform_static.py` lookup path now calls
  `execute_lookup_query(...)` (production lookup router) instead of direct `search_general(...)`.
- This makes phase-3 lookup metrics include fallback behavior and route diagnostics.

Interpretation:
- Lookup fallback is effective on anchor-heavy lookup intent phrasing and should be evaluated
  with targeted anchor sets, not only generic phase-2 silver mix.

## I2 Rebalancing Plan (Draft 2026-02-23)

Goal: improve overall question coverage (especially abstract/mixed queries) while keeping small-model + low-token constraints.

### I2 Root-Cause Priorities

- P0 (`Policy bias`): retrieval rules in AQT push too many queries to `fulltext_heavy`.
- P1 (`Feature bias`): NER/regex signals can inflate specificity and over-trigger fulltext.
- P2 (`Data scope bias`): PoC set is skewed toward direct/specific questions.

### Phase Plan

1. Phase 1 - Policy rebalance (P0)
   - Target file: `apps/api/src/services/aqt.py`
   - Adjust `choose_retrieval_profile()` thresholds and default routing balance.
   - Add guardrail: maintain minimum vector contribution for non-`id_lookup`.
2. Phase 2 - Signal quality gate (P1)
   - Target file: `apps/api/src/services/aqt.py`
   - Filter low-quality NER spans before they affect `entity_ratio`.
   - Separate hard entities (`tmtid`, dose/strength, manufacturer) from soft semantic spans.
3. Phase 3 - Evaluation rebalance (P2)
   - Target files: `test_aqt_cli.py`, benchmark set under `experiments/intent_benchmarks/`
   - Split test buckets: `specific`, `mixed`, `abstract`.
   - Report metrics by bucket, not only global average.

### Success Criteria

- Reduce `fulltext_heavy` share on abstract/mixed buckets without harming specific bucket precision.
- Improve recall@k on abstract/mixed queries.
- Keep token/context budget within PoC limits (no major increase in average context size).
- Preserve exact-id path behavior (`id_lookup` precision remains unchanged).

### Safety and Rollback

- Use feature flags for staged rollout:
  - `INTENT_V2_ADAPTIVE_PLANNER`
  - `INTENT_V2_USE_NER`
- If regression occurs, disable only the latest phase via flags and keep prior stable behavior.

### Latest AQT Evaluation Snapshot (`aqt_out.json`)

Date: 2026-02-23 (post minor sanitization update)

Scope:
- 11 CLI test cases (specific + mixed + abstract)

Observed metrics:
- Action intent correctness (manual expected mapping): `11/11`
- Topic intent correctness (manual expected mapping): `9/11`
- Retrieval mode distribution: `fulltext_heavy=10`, `vector_heavy=1`
- NER availability in metadata: `11/11 = true`
- NER sanitization impact:
  - `entity_count_before=14`, `entity_count_after=13` (drop=1)
  - cases with any drop: `1/11`

Key improvements:
- `ner_sanitized` metadata now present, with dropped entity/slot audit trail.
- One noisy manufacturer entity (`"à¸œà¸¹à¹‰à¸œà¸¥à¸´à¸•"`) correctly dropped in Manufacturer listing case.
- Exact `id_lookup` path still preserved (tmtid filter + must_match + depth=1).

Remaining issues (linked to I2 / NER quality):
- I2 still active: routing heavily biased to `fulltext_heavy` (10/11).
- Abstract English case still routed `fulltext_heavy` due noisy drug/query span.
- Topic misses:
  - `Count with substance` -> `general` (expected substance-like scope)
  - `English count query` -> `hierarchy` (expected general/count scope)
- Residual noisy slots:
  - `brand = "TMTID 662401"`
  - long-span `drug/query` values from full sentence in some English cases

## Proposed Threshold Table (Draft, Pending Confirmation)

Status: `NOT CONFIRMED` (store for review only, do not implement yet)

### Global Rules

| Rule | Value |
|---|---|
| `id_lookup` override | keep current `vector=0.15`, `fulltext=0.85` |
| Non-`id_lookup` vector floor | `vector >= 0.35` |
| Non-`id_lookup` fulltext cap | `fulltext <= 0.65` |

### Threshold Matrix (Phase 1 Candidate)

| Strategy | Condition | Retrieval Mode | Vector | Fulltext |
|---|---|---|---:|---:|
| `retrieve` | `is_abstract = true` or `entity_ratio < 0.30` | `vector_heavy` | 0.70 | 0.30 |
| `retrieve` | `0.30 <= entity_ratio < 0.60` | `balanced` | 0.55 | 0.45 |
| `retrieve` | `entity_ratio >= 0.60` | `fulltext_heavy` | 0.40 | 0.60 |
| `verify` | `entity_ratio >= 0.60` and not abstract | `fulltext_heavy` | 0.40 | 0.60 |
| `verify` | `0.35 <= entity_ratio < 0.60` | `balanced` | 0.50 | 0.50 |
| `verify` | `entity_ratio < 0.35` or abstract | `vector_heavy` | 0.60 | 0.40 |
| `count` / `list` | `entity_ratio >= 0.60` | `fulltext_heavy` | 0.35 | 0.65 |
| `count` / `list` | otherwise | `balanced` | 0.50 | 0.50 |

## Research References (Adaptive Hybrid Retrieval)

### Search Method (Hybrid Fusion / Ranking)

1. Bruch, Gai, Ingber (2022), *An Analysis of Fusion Functions for Hybrid Retrieval*  
   Link: https://arxiv.org/abs/2210.11934
2. Cormack, Clarke, Buttcher (2009), *Reciprocal rank fusion outperforms condorcet and individual rank learning methods*  
   Link: https://dblp.org/rec/conf/sigir/CormackCB09
3. Louis, van Dijck, Spanakis (2024), *Know When to Fuse: Investigating Non-English Hybrid Retrieval in the Legal Domain*  
   Link: https://arxiv.org/abs/2409.01357

### Query Performance Prediction (QPP) / Adaptive Routing Signals

1. Cronen-Townsend, Zhou, Croft (2002), *Predicting query performance*  
   Link: https://ir.webis.de/anthology/2002.sigirconf_conference-2002.40/
2. Shtok, Kurland, Carmel (2009), *Predicting Query Performance by Query-Drift Estimation*  
   Link: https://ir.webis.de/anthology/2009.ictir_conference-2009.30/
3. Hauff, Hiemstra, de Jong (2008), *A survey of pre-retrieval query performance predictors*  
   Link: https://research.utwente.nl/en/publications/a-survey-of-pre-retrieval-query-performance-predictors/
4. Vlachou, Macdonald (2023), *On Coherence-based Predictors for Dense Query Performance Prediction*  
   Link: https://arxiv.org/abs/2310.11405
5. Faggioli et al. (2023), *Query Performance Prediction for Neural IR: Are We There Yet?*  
   Link: https://arxiv.org/abs/2302.09947
6. Mothe, Ullah (2023), *Selective Query Processing: a Risk-Sensitive Selection of System Configurations*  
   Link: https://arxiv.org/abs/2305.18311

### Notes for Current Design

- Research direction indicates hybrid retrieval should use adaptive weighting rather than fixed weights.
- Neural-IR QPP is harder and less stable than sparse settings; design should include guardrails:
  - clamp weight range
  - fallback to balanced mode when confidence/clarity is low

## Update Checklist (When things change)

Update this file for:

- branch policy changes
- new canonical tags
- new feature flags
- new benchmark entrypoints
- changed run commands

## AQT Maturity Assessment vs High-End GraphRAG (2026-02-26)

Source evidence:
- `aqt_out.json` (latest local run in `.venv`)
- runtime path: `test_aqt_cli.py`

### Score Snapshot

Current overall PoC score: `6.5/10`  
Target score: `8.0/10`

| Dimension | Current | Target |
|---|---:|---:|
| Intent routing quality | 7.0 | 8.0 |
| NER + slot quality | 6.0 | 7.5 |
| Adaptive hybrid retrieval policy | 6.0 | 8.0 |
| Graph reasoning + expansion precision | 5.5 | 7.5 |
| Grounding / explainability | 4.5 | 7.0 |
| Observability / rollout safety | 7.0 | 8.0 |

### Latest Summary (PoC reality check)

- NER runtime available in the latest `.venv` run: `11/11` cases.
- Fallback query slot (`query_slot_mode=fallback_rule`) occurred in `3/11` cases (expected for no-slot queries).
- Compare case now splits multi-drug correctly (`Paracetamol`, `Ibuprofen`).
- Main residual issue: slot field confusion (entity extracted but mapped to wrong slot), e.g. brand/manufacturer/id overlap.

### Gap to 8/10 (Top blockers)

1. Slot disambiguation is not strict enough (`brand` vs `manufacturer` vs `tmtid` collisions).
2. Adaptive retrieval policy is still threshold-driven and static; limited uncertainty control.
3. Evidence is not first-class in final answer output (weak citation/trace surface for users).

### 8/10 Roadmap (Fast PoC track)

Phase A (P0, 1-2 days): Slot trust policy
- Add deterministic conflict rules:
  - if `id_lookup` exists, suppress NER brand-like spans containing ID tokens.
  - if rule manufacturer exists, demote conflicting NER brand span with same text.
  - keep multi-entity compare slots, but enforce per-slot allowlist by question intent.
- KPI:
  - field-confusion error reduced by >=50% on current 11-case set + shadow cases.

Phase B (P0/P1, 2-3 days): Adaptive retrieval with guardrails
- Replace fixed threshold branches with calibrated piecewise weighting:
  - maintain vector floor and fulltext cap for non-id queries.
  - introduce ambiguity-aware fallback to balanced mode.
  - use slot confidence buckets to modulate profile (not just entity_ratio).
- KPI:
  - improve mixed/abstract recall without dropping exact/specific precision.
  - keep token/context budget stable (+/-10%).

Phase C (P1, 2-3 days): Evidence-first response contract
- Pass structured evidence triples (node/relation/source fields) through extraction to formatter.
- Require answer sections: `answer`, `supporting_evidence`, `confidence_note`.
- KPI:
  - every non-count answer contains explicit evidence references.
  - manual faithfulness review improves (no unsupported claims in sampled set).

### Exit Criteria for PoC = 8/10

- Routing quality: action/topic accuracy stable at >= current baseline with fewer slot misroutes.
- Retrieval quality: measurable gain on mixed/abstract bucket while preserving id-lookup precision.
- Answer grounding: evidence-cited outputs become default behavior.
- Ops: feature-flag rollout path remains reversible per phase.

## Phase 1 Execution Log (2026-02-26)

Scope completed:
- Added synthetic regression dataset:
  - `experiments/intent_benchmarks/aqt_synthetic_phase1_cases.json`
- Added synthetic test runner:
  - `experiments/question_understanding/intent_classification/integration_with_app/shadow_compare/test_aqt_phase1_synthetic.py`
- Implemented slot trust policy in `apps/api/src/services/aqt.py`:
  1. suppress `brand` when exact `tmtid` signal exists
  2. suppress `brand` that conflicts with deterministic manufacturer signal
  3. suppress `brand` noise in compare queries when drug signal exists
  4. preserve fallback `query` only when no usable NER slot remains

Validation snapshot:
- Synthetic regression: `8/8 PASS`
  - output: `test_results/20260226_154314_aqt_phase1_synthetic.jsonl`
- Shadow compare regression: `8/8 PASS`
  - output: `test_results/20260226_154356_intent_bundle_shadow_compare.jsonl`
- Extended CLI test now includes base + synthetic cases:
  - `test_aqt_cli.py` loads synthetic set automatically if file exists
  - latest total cases: `19`

Known residual noise after Phase 1:
- Some manufacturer-style aliases may still appear as `brand` in generic listing contexts (e.g. short alias tokens).
- This is tracked for Phase 2/next policy tuning (confidence-aware disambiguation + slot scoring).

