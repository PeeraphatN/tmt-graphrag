# Retrieval Evaluation Experiment Draft

## 1) Objective
- Measure search correctness and ranking quality against real graph ground truth.
- Compare three weighting policies:
  - `policy_a_equal`: equal weights (`vector=fulltext=graph`)
  - `policy_b_fixed`: current lookup table policy (existing behavior)
  - `policy_c_dynamic`: adaptive weight policy from real input features
- Evaluate by operator: `lookup`, `list`, `compare`, `verify`, `count`.
- Separate retrieval quality from reranker impact (`pre-rerank` vs `post-rerank`).

## 2) Phase 1: Build Ground Truth from Real Neo4j Data
- Extract real nodes with stratified sampling across levels: `SUBS`, `VTM`, `GP`, `GPU`, `TP`, `TPU`.
- Build two gold sets:
  - `single-target gold`: one clear expected node (for example exact `tmtid`).
  - `multi-target gold`: multiple expected nodes via deterministic relation traversal.
- Build deterministic `qrels` (query relevance labels) from ontology paths and filters.
- Build deterministic count gold from fixed scopes (for example `SUBS->VTM->GP->GPU/TP/TPU`).
- Output artifacts:
  - `eval/ground_truth_nodes.jsonl`
  - `eval/qrels.jsonl`
  - `eval/count_gold.jsonl`

## 3) Phase 2: Build Silver Standard Query Set
- Generate TH/EN questions from ground truth templates + rule-based paraphrases.
- Attach metadata per query:
  - `expected_operator`
  - `expected_slots`
  - `gold_node_ids` or `gold_count_scope`
  - `difficulty` (`easy`, `medium`, `hard`)
  - `language` (`th`, `en`)
- Coverage requirements:
  - exact id lookup
  - name lookup
  - manufacturer filter
  - substance traversal
  - compare two entities
  - deterministic count
- Output artifact:
  - `eval/silver_queries.jsonl`

## 4) Phase 3: Policy Comparison Protocol
- Run the same query set on all three policies (`policy_a_equal`, `policy_b_fixed`, `policy_c_dynamic`).
- Keep all budgets identical across policies:
  - same `k`
  - same traversal depth
  - same reranker model/top_k
  - same timeout limits
- Log both stages:
  - `pre_rerank`: retrieval output before reranker
  - `post_rerank`: final ranked output after reranker
- Per-query log fields:
  - `operator`, `route`, `weights`, `seed_results`, `expanded_nodes`, latency
  - rank position of first gold hit
  - full ranked list of returned node ids (for evaluation replay)

## 5) Metrics
- Retrieval operators (`lookup`, `list`, `compare`):
  - `Hit@k`, `Precision@k`, `Recall@k`, `MRR`, `nDCG`
  - `FirstRelevantRank` (lower is better)
  - `RerankGain` = (`post-rerank metric`) - (`pre-rerank metric`)
- Verify:
  - `Accuracy`, `Precision`, `Recall`, `F1`
- Count:
  - `Exact Match`, `MAE`, `MAPE`
  - optional: exact-match by count-scope type (substance/manufacturer/hierarchy)
- Efficiency:
  - latency (`p50`, `p95`)

## 6) Controls and Anti-Overfit
- Fixed random seed and reproducible sampling.
- Keep dev/test split for dynamic policy tuning:
  - `dev`: tune dynamic weight function
  - `test`: locked evaluation only
- Do not tune on final test set.
- Report slices by:
  - operator
  - level
  - difficulty
  - language (`TH`/`EN`)

## 7) Acceptance Criteria (Research Comparison Track)
- `policy_c_dynamic` outperforms `policy_b_fixed` on `Recall@10` and `MRR` in test split.
- `Precision@10` does not regress beyond agreed threshold.
- `count` exact match improves or remains equal to best baseline.
- latency regression stays within agreed budget.

## 8) PoC Acceptance Gate (Semantic-First, Locked 2026-03-07)
- Scope for semantic quality gate:
  - include only semantic operators: `lookup`, `list`, `compare` (stored as `analyze_compare`)
  - exclude exact-id flow (`id_lookup`) from semantic retrieval scoring
  - verify is tracked as `verify_non_id` (only records where `strategy != id_lookup`)
- Quality thresholds:
  - `overall_semantic_hit@10 >= 0.75`
  - `lookup_hit@10 >= 0.55`
  - `list_hit@10 >= 0.80`
  - `compare_hit@10 >= 0.85`
  - `overall_semantic_mrr >= 0.55`
  - `overall_semantic_ndcg@10 >= 0.60`
- Count thresholds:
  - `count_exact_match >= 0.45`
  - `count_mape <= 40.0`
- System thresholds:
  - `error_rate <= 0.01`
  - `latency_p95_ms <= 2500`
- Stability rule:
  - must pass all thresholds for at least 2 consecutive benchmark runs on a frozen eval set.

## 9) Deliverables
- `eval/summary.json`
- `eval/summary.md`
- full per-policy result tables
- per-operator breakdown tables
- error buckets:
  - AQT error
  - retrieval miss
  - rerank miss
  - graph traversal scope error
  - count scope mismatch

## 10) Immediate Next Step
- Start Phase 1:
  - define sample size per level
  - finalize deterministic Cypher templates for qrels/count gold
  - generate first draft of `ground_truth_nodes.jsonl` and `qrels.jsonl`
