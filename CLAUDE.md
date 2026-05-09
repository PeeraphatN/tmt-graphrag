# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

GraphRAG proof-of-concept for **Thai Medicinal Terminology (TMT)** search and Q&A. Two halves:

- **Product runtime** under `apps/` and `infra/` — runnable demo (FastAPI + Next.js + Neo4j + Ollama).
- **Research** under `experiments/` — intent classification, NER fine-tuning, retrieval evaluation. Research scripts that need backend services target `apps/api` as the canonical runtime; runners that hit the app live under each experiment's `integration_with_app/` subfolder.

The legacy root-level `src/`, `frontend/`, `scripts/` layout described in [AGENTS.md](AGENTS.md) has been retired. Treat `AGENTS.md` as historical — start new work from `apps/api` and `apps/web`.

## Common Commands

### Backend (`apps/api`)

```powershell
cd apps/api
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item ..\..\infra\.env.example .env
python src/api/main.py            # serves http://localhost:8000 (uvicorn, hot reload)
```

Run unit tests (no Neo4j / Ollama / model weights required — heavy deps are mocked in [apps/api/tests/conftest.py](apps/api/tests/conftest.py)):

```powershell
cd apps/api
pytest tests/                     # all tests
pytest tests/test_core.py::TestDetectStrategy::test_count_thai   # single test
```

### Frontend (`apps/web`)

```powershell
cd apps/web
Copy-Item .env.local.example .env.local
npm install
npm run dev      # http://localhost:3000
npm run build
npm run lint     # eslint
```

### Full stack via Docker Compose

```powershell
Copy-Item infra\.env.example infra\.env
cd infra
docker compose up --build         # neo4j, ollama, api (8000), web (3000), neo4j browser (7474)
```

Compose builds `apps/api` and `apps/web` from the repo root context (`context: ..`); volume-mounts `apps/api/{artifacts,cache,logs}` into the API container.

## Backend Architecture

The backend is a layered pipeline composed as an LCEL chain in [apps/api/src/pipeline.py](apps/api/src/pipeline.py). Reading `pipeline.py` end-to-end is the fastest way to understand the system; the chain runs:

```
Transform (AQT) -> Search -> Extract -> Format
```

### Stages

1. **AQT — Advanced Query Transformation** ([src/services/aqt.py](apps/api/src/services/aqt.py))
   Rule-based + ML, **no LLM**. Combines:
   - Strategy detection (`retrieve` / `count` / `list` / `verify`) via regex over Thai+English patterns.
   - Intent classification ([src/services/intent_classifier.py](apps/api/src/services/intent_classifier.py)) — embedding-similarity to precomputed centroids over the labeled set in [src/api/intent_dataset.json](apps/api/src/api/intent_dataset.json).
   - Optional NER ([src/services/ner_service.py](apps/api/src/services/ner_service.py)) gated by `INTENT_V2_USE_NER` for slot extraction (DRUG, BRAND, MANUFACTURER, FORM, STRENGTH).
   - Deterministic filter extraction (TMT id, NLEM category, manufacturer alias).
   - Outputs a `GraphRAGQuery` ([src/schemas/query.py](apps/api/src/schemas/query.py)) plus an `IntentBundle` ([src/schemas/intent_bundle.py](apps/api/src/schemas/intent_bundle.py)) with the adaptive retrieval profile (mode, vector/fulltext weights, entity ratio).

2. **Search** ([src/services/search.py](apps/api/src/services/search.py))
   Hybrid Neo4j retrieval combining vector similarity, fulltext, and graph traversal (depth = `GRAPH_TRAVERSAL_DEPTH`). The function `advanced_graphrag_search` returns a `route` dict; routes are split into:
   - **Deterministic Cypher routes** (`id_lookup`, `analyze_count`, non-fallback `list`) — full payload is preserved, the `HYBRID_FINAL_TOP_K = 25` seed cap is **skipped**, and the reranker is **not** applied. See `_is_cypher_deterministic_route` in `pipeline.py`.
   - **Hybrid routes** — capped at top 25, and the cross-encoder reranker ([src/services/ranking_service.py](apps/api/src/services/ranking_service.py)) is applied **only** for the `lookup` operator.

3. **Extract** ([src/services/extraction.py](apps/api/src/services/extraction.py)) — projects seed nodes + expanded nodes + relationships into a structured context dict consumed by the formatter.

4. **Format** ([src/services/formatting.py](apps/api/src/services/formatting.py)) — LCEL formatter calling the LLM via `langchain_ollama`. **Exception:** `strategy == "count"` bypasses the LLM and uses the deterministic numeric renderer in `pipeline._render_count_answer` to avoid hallucinated numbers.

### Cross-cutting infrastructure

- **Result cache** ([src/cache/result_cache.py](apps/api/src/cache/result_cache.py)) — both exact and semantic (embedding-similarity) answer-cache lookups; question-embedding is computed once in `pipeline.run()` and threaded through `transform_query` to avoid recomputation.
- **NLEM gating** — questions about Thailand's National List of Essential Medicines are short-circuited with a "not supported" reply unless `NLEM_QA_ENABLED=true`. Pattern list is `NLEM_UNSUPPORTED_PATTERNS` in `pipeline.py`.
- **Logging** — central `setup_logging()` in [src/logging_config.py](apps/api/src/logging_config.py) is called from `api/main.py` at import time. Reads `LOG_LEVEL` (default `INFO`), formats `%(asctime)s [%(levelname)s] %(name)s: %(message)s`, writes to stdout, and is idempotent. New code must use `logging.getLogger(__name__)` — no `print()`.
- **Module path** — code uses absolute imports rooted at `src.*`. [apps/api/src/api/main.py](apps/api/src/api/main.py) prepends `apps/api` to `sys.path` at startup; tests do the same in [conftest.py](apps/api/tests/conftest.py). Run Python from `apps/api/`, not from the repo root.
- **API surface** — `GET /` (status), `GET /health` (pipeline readiness), `POST /chat` (`{message}` → `{response}`). The pipeline runs on a threadpool (`run_in_threadpool`) wrapped in `asyncio.wait_for(..., timeout=CHAT_TIMEOUT_SECONDS)` so a stuck LLM/retrieval returns HTTP 504 instead of hanging the request.

### Feature flags (env)

Defined in [src/config.py](apps/api/src/config.py); validated by `validate_env()` at startup:

- `INTENT_V2_ENABLED` (default true) — IntentV2 classifier + IntentBundle.
- `INTENT_V2_USE_NER` (default true in code; **default false in [infra/docker-compose.yml](infra/docker-compose.yml)** because NER weights are not committed). To enable NER place the model under `apps/api/artifacts/ner/final_model` and set the flag.
- `INTENT_V2_ADAPTIVE_PLANNER` (default true) — adaptive retrieval profile per intent.
- `NLEM_QA_ENABLED` (default false) — gate NLEM questions.
- `RETRIEVAL_EVAL_MODE` — used by retrieval experiments to bypass caches/keep raw payloads.
- `CHAT_TIMEOUT_SECONDS` (default `120`) — `/chat` request timeout; on expiry the endpoint returns HTTP 504.
- `RERANKER_DEVICE` (default `auto`; also `cpu` / `cuda`) — device for the cross-encoder reranker. `auto` picks CUDA if available, else CPU; `cuda` raises if CUDA isn't visible. Use `cpu` to keep VRAM free for the LLM on the 8 GB card. VRAM free/total is logged at startup by [ranking_service.py](apps/api/src/services/ranking_service.py).
- `LOG_LEVEL` (default `INFO`) — root logger level for the API process.

`NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `OLLAMA_URL`, `OLLAMA_EMBED_URL`, `LLM_MODEL`, `EMBED_MODEL`, `VECTOR_INDEX_NAME`, `FULLTEXT_INDEX_NAME`, `EMBEDDING_DIM` are all required.

## Frontend Architecture

Next.js 16 (App Router) + React 19 + Tailwind v4. Single-page chat client under [apps/web/app/](apps/web/app/) calling the API at `NEXT_PUBLIC_API_BASE_URL` (set to `http://api:8000` in docker-compose, otherwise `http://localhost:8000`).

## Testing Notes

- The `apps/api/tests/conftest.py` stubs `langchain_core`, `langchain_ollama`, `ollama`, `neo4j`, `torch`, `transformers`, `sentence_transformers`, and `pythainlp` via `MagicMock` **before** importing any `src.*` module. This is what makes pure-utility tests fast and dependency-free; if you add a test that needs real instances, override the stub explicitly inside the test rather than removing it from `conftest.py`.
- [src/services/manufacturer_lookup.py](apps/api/src/services/manufacturer_lookup.py) reads [apps/api/manufacturers.json](apps/api/manufacturers.json) at module import — patch both `load_manufacturers` and `find_manufacturer_with_alias` when importing AQT in tests, as `test_core.py` already does.

## Pre-commit

`detect-secrets` runs on commit ([.pre-commit-config.yaml](.pre-commit-config.yaml)) with `experiments/**.json[l]` and `infra/.env.example` excluded.

## Conventions

- Python 3.10+, 4-space indent, type hints, snake_case modules under `src/services/`.
- Frontend uses TypeScript and ESLint via `eslint-config-next`.
- Don't commit: model weights, trainer outputs, large experiment artifacts, local caches, logs, virtual envs, `node_modules`. The `experiments/retrieval/.../results/*.jsonl` files have been removed from git on the current branch — keep new runs out unless they're explicitly canonical reference results.

## Improvement Roadmap (topdown, dependency-ordered)

Status of the agreed-upon hardening pass before lab handoff. Each tier unblocks the next — don't skip ahead. Constraint: Ryzen 5 5600G + RTX 4060 8 GB, so VRAM-aware defaults matter.

### Tier 0 — Critical correctness ✅
- [x] `detect_strategy` priority bug — yes/no questions like "Is X in the NLEM list?" were classified as `list` because `\blist\b` matched the noun before verify patterns. Fixed in [src/services/aqt.py](apps/api/src/services/aqt.py) with `VERIFY_LEADING_PATTERN` (sentence-leading auxiliary verbs) evaluated between count and list. 27/27 tests pass.

### Tier 1 — Hygiene ✅
- [x] [apps/api/requirements.txt](apps/api/requirements.txt) pinned (exact pins for 11 packages; `>=` for fastapi/uvicorn/pythainlp).
- [x] [LICENSE](LICENSE) — MIT.
- [x] [CLAUDE.md](CLAUDE.md) — this file.
- [x] UI screenshot at [docs/planning/assets/ui_screenshot.png](docs/planning/assets/ui_screenshot.png).

### Tier 2 — Observability ✅
- [x] [src/logging_config.py](apps/api/src/logging_config.py) — central `setup_logging()`, idempotent, reads `LOG_LEVEL`.
- [x] 72 `print()` calls replaced with `logger.*` across [pipeline.py](apps/api/src/pipeline.py), [services/search.py](apps/api/src/services/search.py), [services/intent_classifier.py](apps/api/src/services/intent_classifier.py), [api/main.py](apps/api/src/api/main.py). Timing → `debug`, errors → `error(..., exc_info=True)`.

### Tier 3 — Reliability ✅
- [x] `/chat` wraps the threadpool call in `asyncio.wait_for` → HTTP 504 on `CHAT_TIMEOUT_SECONDS` (default 120).
- [x] `RERANKER_DEVICE=auto|cpu|cuda` resolved in [ranking_service.py](apps/api/src/services/ranking_service.py); VRAM free/total logged at reranker init so we can see headroom on the 8 GB card.
- [x] Both env vars documented in [infra/.env.example](infra/.env.example).

### Tier 4 — Quality gates ⏳ (next)
- [ ] **CI** — `.github/workflows/test.yml` running `pytest apps/api/tests/` on push/PR. ~30 min.
- [ ] **Pipeline pure-function tests** — unit tests for `_is_cypher_deterministic_route`, `_is_nlem_question`, `_render_count_answer` in [pipeline.py](apps/api/src/pipeline.py). ~1 h. Depends on CI being green.
- [ ] **pytest-cov gate** — add `pytest-cov`, `--cov=src --cov-fail-under=<N>` in CI. ~15 min. Depends on the test additions above so the threshold isn't trivially low.

### Tier 5 — Documentation ⏳ (last)
- [ ] **README (English, engineering-doc style for lab handoff — not CV showcase).** Sections: Architecture, Prerequisites, Quick Start (Docker), Local Dev, Configuration table (full env list from [src/config.py](apps/api/src/config.py)), Running Tests, Project Layout, Experiments, Known Limitations. Reference `docs/planning/assets/ui_screenshot.png`. ~2 h. Depends on Tier 4 being stable so config and test commands are accurate.

### Explicitly deferred (not blocking handoff)
- Splitting [services/search.py](apps/api/src/services/search.py) (~2.5k lines) and [services/aqt.py](apps/api/src/services/aqt.py) (~1.5k lines) into sub-modules.
- Operational `nvidia-smi dmon` VRAM check during a real pipeline run — runtime sanity step, no code change.
