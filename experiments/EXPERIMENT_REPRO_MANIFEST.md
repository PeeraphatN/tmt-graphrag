# Experiment Repro Manifest

Last updated: 2026-03-27  
Repo: `C:\Work\PlayGround\Intern\GenAI`

## Layout Policy

Experiments now live under stable folder-based homes instead of being described primarily as branch-only work:

- `experiments/question_understanding/intent_classification`
- `experiments/question_understanding/ner_finetuning`
- `experiments/retrieval/retrieval_eval`

`master` remains the historical baseline branch, but this manifest now records the canonical file layout in the current repo tree.

## 1) Intent Classification

Canonical root:
- `experiments/question_understanding/intent_classification`

Key areas:
- `data/`
- `data/source_snapshots/`
- `embedding_model_selection/`
- `intent_structure_fci_hic/`
- `baselines/llm_prompting/`
- `integration_with_app/`
- `results/`

Standalone entry points:
- `python experiments/question_understanding/intent_classification/data/builders/build_intent_dataset_v2_reference.py`
- `python experiments/question_understanding/intent_classification/data/builders/generate_dataset.py`
- `python experiments/question_understanding/intent_classification/embedding_model_selection/benchmark_intent.py`
- `python experiments/question_understanding/intent_classification/baselines/llm_prompting/benchmark_llm.py`
- `python experiments/question_understanding/intent_classification/intent_structure_fci_hic/benchmark_intent_v2_vs_legacy.py`

App-integration entry points:
- `python experiments/question_understanding/intent_classification/integration_with_app/shadow_compare/test_intent_bundle_shadow.py`
- `python experiments/question_understanding/intent_classification/integration_with_app/shadow_compare/test_aqt_phase1_synthetic.py`
- `python experiments/question_understanding/intent_classification/integration_with_app/aqt_cli/test_aqt_cli.py`

Notes:
- standalone intent builders and benchmarks use `data/source_snapshots/legacy_intent_dataset.json` instead of reading `src/api/intent_dataset.json`
- app-coupled checks are isolated under `integration_with_app/`

## 2) NER Fine-Tuning

Canonical root:
- `experiments/question_understanding/ner_finetuning`

Kept in Git:
- `run/extract_entities.py`
- `run/generate_ner_data.py`
- `run/finetune_ner.py`
- `run/analyze_all_entities.py`
- `run/ner_inference_helper.py`
- `configs/ner_inference_config.json`
- `configs/train_config.reference.json`
- `data/dataset_manifest.json`
- reproducibility splits and small entity lists

Intentionally cleared from repo surface:
- trained model weights
- tokenizer exports
- large trainer output directories

Execution policy:
- standalone NER flow lives under `run/` and must not import `src/`
- optional app-coupling checks live under `integration_with_app/`
- `run/generate_ner_data.py` writes `artifacts/generated_records.jsonl` so integration checks can run later without changing the standalone generator

Representative entry points:
- `python experiments/question_understanding/ner_finetuning/run/extract_entities.py`
- `python experiments/question_understanding/ner_finetuning/run/generate_ner_data.py`
- `python experiments/question_understanding/ner_finetuning/run/finetune_ner.py`
- `python experiments/question_understanding/ner_finetuning/integration_with_app/run_aqt_sanity.py`

## 3) Retrieval Evaluation

Canonical root:
- `experiments/retrieval/retrieval_eval`

Key areas:
- `data/phase1_ground_truth.json`
- `data/phase2_silver_queries.json`
- `data/poc_acceptance_criteria_semantic_v1.json`
- `run/`
- `integration_with_app/`
- `docs/`
- `results/`

Representative entry points:
- `python experiments/retrieval/retrieval_eval/integration_with_app/build_phase1_ground_truth.py`
- `python experiments/retrieval/retrieval_eval/integration_with_app/build_phase2_silver_queries.py`
- `python experiments/retrieval/retrieval_eval/integration_with_app/run_phase3_uniform_static.py`
- `python experiments/retrieval/retrieval_eval/integration_with_app/run_lookup_fallback_ablation.py`
- `python experiments/retrieval/retrieval_eval/run/check_poc_acceptance_semantic.py --runs-jsonl <runs.jsonl>`

## Repro Rules

1. Put executable experiment logic in the experiment folder that owns it; avoid creating new repo-level ad hoc script folders for experiment work.
2. Keep deterministic datasets, configs, and small result summaries in Git when they help reproduction.
3. Keep large generated artifacts out of Git, especially NER model weights and export directories.
4. Write standalone outputs into the owning experiment's `results/` or `artifacts/` subtree.
5. If a script imports `src/`, move it under `integration_with_app/` instead of mixing it with standalone experiment code.
