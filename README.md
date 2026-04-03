# TMT GraphRAG PoC

GraphRAG proof-of-concept for Thai medicinal terminology search and question answering.

## Canonical Structure

The repo now uses the new application-first layout as the source of truth:

- `apps/api`: FastAPI backend and GraphRAG runtime
- `apps/web`: Next.js frontend
- `infra`: Docker Compose and shared environment templates
- `experiments/question_understanding`: intent classification and NER research
- `experiments/retrieval`: retrieval evaluation and analysis

The old root-level runtime surfaces have been retired. Start new work only from the canonical homes above.

## What This Repo Contains

There are two main parts:

1. Product PoC
   The runnable GraphRAG application used for demos and integration work.
2. Research Work
   Experiments for question understanding and retrieval evaluation.

## Run The Product App

### Option 1: Run with Docker Compose

```powershell
Copy-Item infra\.env.example infra\.env
cd infra
docker compose up --build
```

Services:

- API: `http://localhost:8000`
- Web: `http://localhost:3000`
- Neo4j Browser: `http://localhost:7474`

### Option 2: Run Locally

Backend:

```powershell
cd apps/api
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item ..\..\infra\.env.example .env
python src/api/main.py
```

Frontend:

```powershell
cd apps/web
Copy-Item .env.local.example .env.local
npm install
npm run dev
```

## Experiments

### Question Understanding

Located in `experiments/question_understanding`.

Main areas:

- `intent_classification`: embedding model selection and FCI/HIC intent structure work
- `ner_finetuning`: NER data generation and fine-tuning pipeline

NER runbook:

- `experiments/question_understanding/ner_finetuning/README.md`

### Retrieval Evaluation

Located in `experiments/retrieval/retrieval_eval`.

This area contains retrieval experiments, evaluation scripts, draft documents, and generated result sets.

Experiment integration checks that need runtime services should target `apps/api` as the current backend source of truth.

## Recommended Entry Points

When handing this repo to another engineer, start from:

- `apps/api/README.md`
- `apps/web/README.md`
- `infra/README.md`
- `experiments/question_understanding/ner_finetuning/README.md`

## What Should Not Be Committed

Keep generated or machine-specific artifacts out of Git:

- model weights and checkpoints
- trainer outputs
- large experiment artifacts
- local caches
- logs
- virtual environments
- local `node_modules`

## Current Handoff Direction

The repo now uses a clean split between product runtime and research assets. New work should stay inside the canonical structure above.
