# System Architecture — TMT GraphRAG

เอกสารนี้สรุปสถาปัตยกรรมของระบบ GraphRAG สำหรับ Thai Medicinal Terminology (TMT)
ครอบคลุมทั้ง runtime ของ product และส่วน research โดยอ้างอิงโค้ดจริงในสาขา
canonical layout ใต้ [apps/](../apps/) และ [infra/](../infra/)

> ส่วน legacy ใต้ root (`src/`, `frontend/`, `scripts/`) ถูกย้ายออกแล้ว — ดู
> [docs/repo-reorg-plan.md](repo-reorg-plan.md) สำหรับประวัติการ refactor

---

## 1. Goals & Scope

ระบบเป็น GraphRAG proof-of-concept ที่รับคำถามเกี่ยวกับยา (ภาษาไทย/อังกฤษ)
แล้วตอบจาก knowledge graph ของ TMT โดยมีเป้าหมายหลัก 3 ข้อ

1. **Hybrid retrieval ที่ไม่พึ่ง LLM ใน routing** — ใช้ rule + embedding similarity
   ทำ intent classification เพื่อให้ latency ต่ำและพฤติกรรมคงที่
2. **Deterministic Cypher path สำหรับคำถามเชิงโครงสร้าง** — ID lookup, count, list
   ไม่ต้องผ่าน vector/fulltext เพื่อกัน hallucination ของตัวเลข/ID
3. **แยก research จาก runtime** — โค้ดที่ใช้ใน experiments อยู่ใต้ [experiments/](../experiments/)
   และเรียก runtime ผ่าน `apps/api` เป็น canonical entry-point

---

## 2. High-Level Diagram

```
                ┌──────────────────┐
                │   apps/web       │  Next.js 16 + React 19 + Tailwind v4
                │   (port 3000)    │  เป็น single-page chat client
                └────────┬─────────┘
                         │ POST /chat  { message }
                         ▼
                ┌──────────────────┐
                │   apps/api       │  FastAPI (Uvicorn)
                │   (port 8000)    │  Endpoints: /, /health, /chat
                └────────┬─────────┘
                         │ run_in_threadpool(pipeline.run)
                         ▼
        ┌──────────────────────────────────────────┐
        │        GraphRAGPipeline (LCEL)           │
        │  Transform → Search → Extract → Format   │
        └───┬───────────┬───────────┬───────────┬──┘
            │           │           │           │
            ▼           ▼           ▼           ▼
     ┌──────────┐  ┌────────┐  ┌────────┐  ┌────────┐
     │  AQT     │  │ Neo4j  │  │ Struct │  │ Ollama │
     │ Intent + │  │ Vector │  │ context│  │  LLM   │
     │ Filters  │  │  + FT  │  │ projct │  │ (chat) │
     │ + NER    │  │  +Graph│  │        │  │        │
     └──────────┘  └────────┘  └────────┘  └────────┘
                         │
                         ▼
                  ┌──────────────┐
                  │  Result &    │
                  │  Query Cache │
                  └──────────────┘
```

External services (containerized): **Neo4j 5.15** (graph DB), **Ollama** (LLM + embedding),
ดู [infra/docker-compose.yml](../infra/docker-compose.yml)

---

## 3. Component Map

### 3.1 Runtime — [apps/](../apps/)

| Component | Path | Responsibility |
|-----------|------|----------------|
| API server | [apps/api/src/api/main.py](../apps/api/src/api/main.py) | FastAPI app, lifespan สร้าง pipeline + warmup, CORS, threadpool offload |
| Pipeline | [apps/api/src/pipeline.py](../apps/api/src/pipeline.py) | LCEL chain เชื่อม 4 stage, NLEM gate, count renderer, log JSONL |
| AQT | [apps/api/src/services/aqt.py](../apps/api/src/services/aqt.py) | Strategy detection + intent classification + filter extraction (rule + embedding) |
| Intent classifier | [apps/api/src/services/intent_classifier.py](../apps/api/src/services/intent_classifier.py) | Centroid similarity บน labeled dataset, ไม่ใช้ LLM |
| NER | [apps/api/src/services/ner_service.py](../apps/api/src/services/ner_service.py) | Slot extraction (DRUG/BRAND/MANUFACTURER/FORM/STRENGTH); gated ด้วย `INTENT_V2_USE_NER` |
| Search | [apps/api/src/services/search.py](../apps/api/src/services/search.py) | Hybrid retrieval (vector + fulltext + graph traversal) ผ่าน operator routing |
| Extract | [apps/api/src/services/extraction.py](../apps/api/src/services/extraction.py) | ฉาย seed/expanded nodes + relationships ให้กลายเป็น context dict |
| Format | [apps/api/src/services/formatting.py](../apps/api/src/services/formatting.py) | LCEL formatter เรียก LLM ผ่าน `langchain_ollama` |
| Reranker | [apps/api/src/services/ranking_service.py](../apps/api/src/services/ranking_service.py) | Cross-encoder rerank (ใช้เฉพาะ lookup route) |
| Result cache | [apps/api/src/cache/result_cache.py](../apps/api/src/cache/result_cache.py) | Exact + semantic answer cache, query cache |
| DB driver | [apps/api/src/services/database.py](../apps/api/src/services/database.py) | Neo4j driver init + index setup |
| Web client | [apps/web/app/page.tsx](../apps/web/app/page.tsx) | Chat UI ดูดข้อมูลจาก `NEXT_PUBLIC_API_BASE_URL` |

### 3.2 Schemas

- [GraphRAGQuery](../apps/api/src/schemas/query.py) — โครงสร้างคำถามที่ผ่าน AQT แล้ว
  ประกอบด้วย `target_type`, `strategy`, `query`, filters, **adaptive retrieval profile**
  (`retrieval_mode`, `vector_weight`, `fulltext_weight`, `entity_ratio`)
- [IntentBundle](../apps/api/src/schemas/intent_bundle.py) — payload ของ IntentV2
  (`action_intent`, `topics_intents`, `slots`, `control_features`, `retrieval_plan`)
  ที่แนบไปกับ `GraphRAGQuery.intent_bundle`

### 3.3 Infrastructure — [infra/](../infra/)

[docker-compose.yml](../infra/docker-compose.yml) ประกาศ 4 service:

| Service | Image | Ports | หน้าที่ |
|---------|-------|-------|---------|
| `neo4j` | `neo4j:5.15.0` + APOC | 7474, 7687 | Graph DB เก็บ TMT |
| `ollama` | `ollama/ollama:latest` | 11434 | Run LLM + embedding model |
| `api` | build จาก `apps/api/Dockerfile` | 8000 | FastAPI runtime |
| `web` | build จาก `apps/web/Dockerfile` | 3000 | Next.js client |

ทั้งหมดอยู่บน bridge network `genai-network` และ build context = repo root (`..`)
volume mounts: `apps/api/{artifacts,cache,logs}` → ใน container ใช้ persist
intent centroid cache, NER weights, และ ragas log

### 3.4 Research — [experiments/](../experiments/)

แยกออกจาก runtime; โค้ดที่ต้องเรียก backend services ใช้ subfolder
`integration_with_app/` ที่ import จาก `apps/api/src.*` แทนที่จะ duplicate ตรรกะ

---

## 4. Backend Pipeline — 4 Stages

LCEL chain ใน [pipeline.py](../apps/api/src/pipeline.py):

```python
RunnablePassthrough.assign(query_obj=_step_transform)   # AQT
| RunnablePassthrough.assign(results=_step_search)      # Hybrid retrieval
| RunnablePassthrough.assign(context=_step_extract)     # Project to dict
| RunnablePassthrough.assign(answer=formatter_step)     # LLM (or deterministic)
```

### 4.1 Transform — AQT (no LLM)

ลำดับใน [aqt.py](../apps/api/src/services/aqt.py):

1. เก็บ `question_raw` และ normalize เบาๆ สำหรับ classifier/strategy เท่านั้น
2. รัน NER → sanitize slots/entities (ถ้า `INTENT_V2_USE_NER=true`)
3. Merge deterministic filters: `tmtid` / `manufacturer` / `nlem_category`
4. ประกอบ search term จาก id + sanitized NER slots + raw fallback
5. คำนวณ adaptive retrieval profile (mode, weights, entity_ratio) → ห่อเป็น `IntentBundle`

**Strategy** เลือกหนึ่งใน `retrieve / count / list / verify` ผ่าน regex
ภาษาไทย+อังกฤษ (`COUNT_PATTERNS`, `LIST_PATTERNS`, `VERIFY_PATTERNS` ใน aqt.py)

### 4.2 Search — Hybrid Retrieval + Operator Routing

`advanced_graphrag_search` คืน `route` dict ที่บอก operator:

| Operator | Path | Top-K cap | Reranker |
|----------|------|-----------|----------|
| `id_lookup` | Cypher deterministic | **ไม่ cap** | ปิด |
| `analyze_count` | Cypher deterministic | ไม่ cap | ปิด |
| `list` (fallback_used=False) | Cypher deterministic | ไม่ cap | ปิด |
| `list` (fallback_used=True) | Hybrid | 25 | ปิด |
| `lookup` | Hybrid (vector+FT+graph) | 25 | **เปิด** (cross-encoder) |
| อื่นๆ | Hybrid | 25 | ปิด |

ตรรกะ routing อยู่ใน `_is_cypher_deterministic_route` ของ
[pipeline.py:76](../apps/api/src/pipeline.py#L76) — ทำให้ deterministic route
รักษา payload เต็มและไม่ผ่านการจัด rank ซ้ำ

ความลึกของ graph traversal ปรับผ่าน `GRAPH_TRAVERSAL_DEPTH` (default 2)

### 4.3 Extract

[extraction.py](../apps/api/src/services/extraction.py) ฉาย seed + expanded nodes +
relationships ให้เป็น dict ที่ formatter ใช้สร้าง prompt

### 4.4 Format

- เคส `strategy == "count"` → bypass LLM ใช้ deterministic renderer
  `_render_count_answer` ใน [pipeline.py:250](../apps/api/src/pipeline.py#L250)
  เพื่อกันการ hallucinate ตัวเลข
- เคสอื่น → LCEL chain จาก `get_formatter_chain()` เรียก LLM ผ่าน `langchain_ollama`

---

## 5. Data Layer

### 5.1 TMT Hierarchy

โครงข้อมูลใช้ลำดับ TMT มาตรฐาน — ดู `HIERARCHY_LEVELS` ใน
[search.py:46](../apps/api/src/services/search.py#L46):

```
SUBS → VTM → GP → GPU → TP → TPU
```

Relationship types ที่ traversal ใช้ — `GRAPH_TRAVERSAL_REL_TYPES`
([search.py:30](../apps/api/src/services/search.py#L30))
เช่น `HAS_ACTIVE_SUBSTANCE`, `HAS_VTM`, `HAS_TRADE_PRODUCT`, ฯลฯ
(double-direction ทุกคู่เพื่อให้ traverse ได้สองทาง)

### 5.2 Indexes

- Vector index — ตั้งชื่อตาม `VECTOR_INDEX_NAME` env (dim = `EMBEDDING_DIM`)
- Fulltext index — ตั้งชื่อตาม `FULLTEXT_INDEX_NAME`
- ตรวจ existence ผ่าน `check_index_exists`
  ([search.py](../apps/api/src/services/search.py)) มี cache ในตัวเพื่อลด round-trip

### 5.3 Manufacturer alias table

[manufacturers.json](../apps/api/manufacturers.json) ถูกโหลดที่ module import ของ
[manufacturer_lookup.py](../apps/api/src/services/manufacturer_lookup.py) — เป็น
dictionary mapping จากชื่อย่อ/ชื่อภาษาไทยไปยัง canonical manufacturer name

---

## 6. Cross-Cutting Concerns

### 6.1 Caching

ใน [result_cache.py](../apps/api/src/cache/result_cache.py):

- **Answer cache** — exact (hash ของคำถาม) + semantic (cosine similarity ของ embedding)
- **Query cache** — เก็บผลลัพธ์ AQT (`GraphRAGQuery`) ตาม embedding similarity
- คำถาม-embedding คำนวณครั้งเดียวใน `pipeline.run()` แล้วส่งต่อไป AQT เพื่อกัน recompute

### 6.2 NLEM Gate

`NLEM_QA_ENABLED=false` (default) → คำถามที่ตรงกับ `NLEM_UNSUPPORTED_PATTERNS`
ถูก short-circuit ตอบข้อความ "ยังไม่รองรับ" โดยไม่เข้า pipeline
([pipeline.py:57](../apps/api/src/pipeline.py#L57))

### 6.3 Concurrency

FastAPI `/chat` ใช้ `run_in_threadpool(pipeline.run, ...)` กัน blocking event loop
เพราะ pipeline เรียก Neo4j driver แบบ sync และ Ollama HTTP แบบ blocking

### 6.4 Warmup

`pipeline.warmup()` ถูกเรียกที่ FastAPI lifespan startup —
embed test text + classification LLM + reranker เพื่อลด cold-start latency
ของ request แรก

### 6.5 Logging

ทุก request ที่สำเร็จเขียน JSONL ลง [apps/api/logs/ragas_data.jsonl](../apps/api/logs/)
ผ่าน `_log_interaction` (ใช้ใน RAGAS evaluation) — ฟิลด์: `id`, `timestamp`,
`question`, `contexts`, `answer`, `ground_truth`

### 6.6 Timing Telemetry

`pipeline.run()` วัด `transform / search / extract / llm` แยกกันแล้วพิมพ์
summary ทุก request — เก็บใน `self._timing` dict

---

## 7. Configuration & Feature Flags

ดู [config.py](../apps/api/src/config.py) ทั้งหมด

**Required env** (ถ้าขาดจะ raise ที่ `validate_env()`):

```
NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
OLLAMA_URL, OLLAMA_EMBED_URL,
LLM_MODEL, CLASSIFICATION_MODEL, EMBED_MODEL,
VECTOR_INDEX_NAME, FULLTEXT_INDEX_NAME, EMBEDDING_DIM
```

**Feature flags:**

| Flag | Default | ผลที่เกิด |
|------|---------|-----------|
| `INTENT_V2_ENABLED` | true | เปิด IntentV2 + IntentBundle |
| `INTENT_V2_USE_NER` | true (compose: false) | เปิด NER service (ต้องมี weights) |
| `INTENT_V2_ADAPTIVE_PLANNER` | true | คำนวณ adaptive retrieval profile |
| `NLEM_QA_ENABLED` | false | กันไม่ให้ตอบ NLEM (ยังไม่มีข้อมูล) |
| `RETRIEVAL_EVAL_MODE` | false | research mode: bypass cache, keep raw payload |
| `GRAPH_TRAVERSAL_DEPTH` | 2 | ความลึกของ neighbor expansion |
| `LIST_MAX_K_CAP` | 200 | จำกัด list-route result count |
| `NER_CONFIDENCE_THRESHOLD` | 0.60 | NER slot ที่ต่ำกว่านี้ทิ้ง |

---

## 8. Request Lifecycle

```
client (apps/web)
    │ 1. POST /chat { message }
    ▼
FastAPI /chat
    │ 2. run_in_threadpool(pipeline.run, message)
    ▼
GraphRAGPipeline.run
    │ 3. NLEM gate check
    │ 4. embed_text(question)               ─→ Ollama embed
    │ 5. chain.invoke({question, q_embed})
    │
    │   5a. Transform — transform_query(...)
    │       - intent_classifier (centroid sim)
    │       - NER (optional)
    │       - rule-based filter extract
    │       → GraphRAGQuery + IntentBundle
    │
    │   5b. Search — advanced_graphrag_search(...)
    │       - route by operator
    │       - Cypher deterministic | Hybrid (vector+FT+graph)
    │       - selective rerank (lookup only)
    │       - hybrid seed cap = 25
    │
    │   5c. Extract — extract_structured_data(...)
    │       → context dict
    │
    │   5d. Format
    │       - count → _render_count_answer (no LLM)
    │       - else  → formatter_chain.invoke (Ollama LLM)
    │
    │ 6. set_cached_answer_semantic(question, q_embed, answer)
    │ 7. _log_interaction → ragas_data.jsonl
    ▼
ChatResponse { response }
```

---

## 9. Module Path Convention

ทุก import ใช้ absolute path รูท `src.*` —
[apps/api/src/api/main.py:9](../apps/api/src/api/main.py#L9) prepend
`apps/api` เข้า `sys.path` ตอน startup
[apps/api/tests/conftest.py](../apps/api/tests/conftest.py) ทำเหมือนกัน
**รัน Python จาก `apps/api/` เท่านั้น ไม่ใช่จาก repo root**

---

## 10. Testing Strategy

- Unit tests ใช้ MagicMock stub ของ `langchain_core`, `langchain_ollama`, `ollama`,
  `neo4j`, `torch`, `transformers`, `sentence_transformers`, `pythainlp` ใน
  [conftest.py](../apps/api/tests/conftest.py) — ทำให้ test รันโดยไม่ต้องมี
  service จริงและไม่ต้องโหลด weights
- ถ้า test ต้องการ instance จริง ให้ override stub ใน test แทนการลบจาก conftest
- Integration / retrieval evaluation อยู่ใน [experiments/retrieval/](../experiments/retrieval/)
  ใช้ `RETRIEVAL_EVAL_MODE=true` เพื่อ bypass cache

---

## 11. ส่วนที่ยังไม่ได้ implement / ข้อจำกัดปัจจุบัน

- NLEM Q&A ปิดอยู่ (ต้องมีการ ingest ข้อมูลก่อน)
- NER weights ไม่ commit; production deploy ต้อง mount ผ่าน volume `artifacts/ner/final_model`
- Reranker ใช้เฉพาะ `lookup` route เท่านั้น (ตามค่า ENABLE_COMPARE_EXPANDED_RERANK = False)
- Pipeline เป็น sync ภายใน — concurrent request ใช้ FastAPI threadpool ไม่ใช่ async

---

## Reference Files

- Pipeline entry — [apps/api/src/pipeline.py](../apps/api/src/pipeline.py)
- API surface — [apps/api/src/api/main.py](../apps/api/src/api/main.py)
- Config & flags — [apps/api/src/config.py](../apps/api/src/config.py)
- Compose topology — [infra/docker-compose.yml](../infra/docker-compose.yml)
- Project guide — [CLAUDE.md](../CLAUDE.md)
