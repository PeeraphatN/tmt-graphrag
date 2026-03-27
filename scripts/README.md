# Scripts

This directory now holds only lightweight repo utilities and smoke checks.

## Top-Level Scripts

- `inspect_graph_schema.py`: inspect the current Neo4j schema and sample fields
- `test_intent_router.py`: smoke-check the current intent routing behavior
- `test_pipeline_chain.py`: basic pipeline initialization and run smoke test
- `test_logger.py`: shared JSONL logger used by smoke scripts

## Subdirectories

- `eval/`: RAGAS-style evaluation helpers and log validation
- `legacy_graph/`: one-off graph maintenance and NLEM enrichment helpers kept for reference

## Guidance

- Prefer experiment-specific runners inside `experiments/.../run` for research work.
- Prefer `apps/api` and `apps/web` for product runtime work.
- Treat `legacy_graph/` as maintenance history, not the main product path.
