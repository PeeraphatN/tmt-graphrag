# Intent Classification

Canonical home for question-understanding experiments that replace LLM-only parsing with intent classification.

Layout:

- `data/`: benchmark datasets, deterministic builders, and source snapshots needed to run without `src/`
- `embedding_model_selection/`: embedding-model benchmark scripts
- `intent_structure_fci_hic/`: standalone FCI/HIC structure experiments
- `baselines/llm_prompting/`: LLM-only baselines
- `integration_with_app/`: checks that intentionally import AQT or other app code
- `docs/`: concept papers, summaries, and historical notes
- `results/`: committed benchmark outputs and small reproducibility artifacts

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

- `data/source_snapshots/legacy_intent_dataset.json` is a frozen copy of the legacy intent seed dataset so builders and benchmarks do not need `src/api/intent_dataset.json`
- if a new script needs `from src...`, place it under `integration_with_app/` instead of mixing it into the standalone benchmark folders
