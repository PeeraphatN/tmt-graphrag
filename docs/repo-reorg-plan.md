# Repo Reorg Plan

## Goal

Split the repository into two clear surfaces:

1. Product runtime for the GraphRAG PoC
2. Research and experiment workspaces for question understanding and retrieval studies

## Phase 1 Scope

This phase only creates the target structure and documents where files should move next.
No existing runtime files are moved in this phase.

## Target Structure

```text
/
  apps/
    api/
    web/
  experiments/
    _meta/
    question_understanding/
      intent_classification/
      ner_finetuning/
    retrieval/
      retrieval_eval/
  infra/
  docs/
```

## Migration Map

### Product Runtime

- `/src` -> `/apps/api/src`
- `/frontend` -> `/apps/web`
- `/requirements.txt` -> `/apps/api/requirements.txt`
- `/docker-compose.yml` -> `/infra/docker-compose.yml`
- `/.env.example` -> `/infra/.env.example`
- `/README.md` -> `/docs/README.md`

### Question Understanding Research

- `/experiments/intent_benchmarks` -> `/experiments/question_understanding/intent_classification`
- `/scripts/test_intent_bundle_shadow.py` -> `/experiments/question_understanding/intent_classification/intent_structure_fci_hic/shadow_compare`
- `/scripts/test_aqt_phase1_synthetic.py` -> `/experiments/question_understanding/intent_classification/intent_structure_fci_hic/shadow_compare`
- `/planning/INTENT_V2_BRANCH_STRATEGY.md` -> `/experiments/_meta/question_understanding_branch_strategy.md`

### NER Fine-Tuning Research

- `/experiments/name_entity_extraction_benckmarks` -> `/experiments/question_understanding/ner_finetuning`
- keep only pre-run essentials in Git:
  - training and inference scripts
  - configs
  - small datasets needed to reproduce runs
  - smoke tests
- move or exclude post-run artifacts:
  - model weights
  - tokenizer exports
  - trainer outputs
  - large generated dumps

### Retrieval Research

- `/experiments/retrieval_eval` -> `/experiments/retrieval/retrieval_eval`

## Migration Rules

- Do not delete old locations until imports, commands, and docs are updated.
- Treat `test_results/`, model outputs, cache files, and temporary files as generated artifacts.
- Recover missing NER pre-run assets before cleaning the old NER experiment tree.
- Update `.gitignore` before moving large generated outputs into the new structure.
