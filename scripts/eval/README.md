# Eval Scripts

Helpers for generating and evaluating app-level RAGAS-style outputs.

- `generate_ragas_data.py`: generate evaluation logs from pipeline runs
- `evaluate_ragas.py`: score collected logs with RAGAS metrics
- `validate_log.py`: quick sanity check for the generated JSONL log

These scripts read from or write to the repo-level `logs/` directory.
