# Retrieval Research

Canonical home for retrieval evaluation and ablation work that supports, but is separate from, the product runtime.

Layout:

- `retrieval_eval/data/`: ground-truth and silver-query datasets
- `retrieval_eval/run/`: standalone analysis and visualization scripts
- `retrieval_eval/integration_with_app/`: runners that intentionally call the backend app
- `retrieval_eval/docs/`: drafts and research notes
- `retrieval_eval/results/`: committed experiment outputs and plots

Key entry points:

- `python experiments/retrieval/retrieval_eval/integration_with_app/build_phase1_ground_truth.py`
- `python experiments/retrieval/retrieval_eval/integration_with_app/build_phase2_silver_queries.py`
- `python experiments/retrieval/retrieval_eval/integration_with_app/run_phase3_uniform_static.py`
- `python experiments/retrieval/retrieval_eval/integration_with_app/run_lookup_fallback_ablation.py`
- `python experiments/retrieval/retrieval_eval/run/check_poc_acceptance_semantic.py --runs-jsonl <runs.jsonl>`

Integration rule:

- retrieval eval scripts that call backend services should target `apps/api` as the canonical runtime
- keep backend-coupled scripts under `integration_with_app/` instead of mixing them into standalone analysis folders
