# Experiment Repro Manifest

Last updated: 2026-02-23  
Repo: `C:\Work\PlayGround\Intern\GenAI`

## Baseline Policy

- `master` is the frozen baseline reference branch.
- Do not merge experimental work into `master` unless explicitly approved.

## Branch Mapping

1. `master` (`48c70cb`)
2. `experiment/intent-classification-benchmarks` (`e466e44`)
3. `experiment/ner-wangchanberta` (`52fcdd6`)
4. `feature/search-hybrid-rerank` (`46ab1f3`)

Lineage (current):

```text
master -> intent-classification-benchmarks -> ner-wangchanberta -> search-hybrid-rerank
```

## Repro Anchors (Tags)

- `baseline-master-20260209` -> `master` baseline (`48c70cb`)
- `exp-intent-benchmarks-20260223` -> intent benchmark + shadow bundle point (`e466e44`)
- `exp-ner-wangchanberta-20260210` -> NER experiment point (`52fcdd6`)
- `exp-search-hybrid-rerank-20260219` -> search experiment point (`46ab1f3`)

## Experiment Inventory

### 1) Intent Benchmarks

- Branch: `experiment/intent-classification-benchmarks`
- Core files:
  - `experiments/intent_benchmarks/benchmark_intent.py`
  - `experiments/intent_benchmarks/benchmark_llm.py`
  - `experiments/intent_benchmarks/benchmark_intent_v2_vs_legacy.py`
  - `experiments/intent_benchmarks/intent_dataset_v2_reference.json`
- Existing result artifacts:
  - `experiments/intent_benchmarks/results/benchmark_intent_20260202_161033.txt`
  - `experiments/intent_benchmarks/results/benchmark_intent_20260206_140509.txt`
  - `experiments/intent_benchmarks/results/benchmark_intent_v2_vs_legacy_20260220_134337.json`
  - `experiments/intent_benchmarks/results/real_query_fci_vs_hic_preview.json`

Re-run commands:

```powershell
git checkout exp-intent-benchmarks-20260223
python experiments/intent_benchmarks/benchmark_intent.py
python experiments/intent_benchmarks/benchmark_llm.py
python experiments/intent_benchmarks/benchmark_intent_v2_vs_legacy.py
```

### 2) IntentBundle Shadow Compare

- Branch: `experiment/intent-classification-benchmarks`
- Core files:
  - `src/schemas/intent_bundle.py`
  - `scripts/test_intent_bundle_shadow.py`
- Existing result artifacts:
  - `test_results/20260223_121035_intent_bundle_shadow_compare.jsonl`
  - `test_results/20260223_121124_intent_bundle_shadow_compare.jsonl`

Re-run command:

```powershell
git checkout exp-intent-benchmarks-20260223
$env:PYTHONIOENCODING='utf-8'
python scripts/test_intent_bundle_shadow.py
```

### 3) NER WangchanBERTa Experiment

- Branch: `experiment/ner-wangchanberta`
- Core files:
  - `experiments/name_entity_extraction_benckmarks/finetune_ner.py`
  - `experiments/name_entity_extraction_benckmarks/ner_inference_helper.py`
  - `experiments/name_entity_extraction_benckmarks/train.json`
  - `src/services/aqt.py` (NER integration point)
- Existing artifacts:
  - `experiments/name_entity_extraction_benckmarks/ner_model_output/*`

Re-run command (example):

```powershell
git checkout exp-ner-wangchanberta-20260210
python experiments/name_entity_extraction_benckmarks/finetune_ner.py
```

### 4) Search Hybrid + Rerank Experiment

- Branch: `feature/search-hybrid-rerank`
- Core files:
  - `src/services/search.py`
  - `src/pipeline.py`
  - `src/services/aqt.py`
- Verification script:
  - `scripts/test_intent_router.py`

Re-run command:

```powershell
git checkout exp-search-hybrid-rerank-20260219
python scripts/test_intent_router.py
```

## Environment Notes

- Use the project virtual environment before running experiments.
- Required services for retrieval experiments:
  - Neo4j
  - Ollama
- Suggested startup:

```powershell
docker-compose up -d
```

## Repro Rules

1. Every experiment run should produce a timestamped artifact in `experiments/.../results/` or `test_results/`.
2. Save deterministic settings in code/config:
   - random seed
   - embedding model
   - train/test split method
3. Do not overwrite previous results; append new timestamped files.
4. Keep `master` unchanged as the baseline reference.
