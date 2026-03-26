import json
import re
import time
import uuid
import pathlib
import sys
from datetime import datetime
from operator import itemgetter

from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda

from src.config import validate_env, GRAPH_TRAVERSAL_DEPTH, NLEM_QA_ENABLED

# Service Imports (Refactored)
from src.services.database import init_driver, setup_indexes
from src.services.search import advanced_graphrag_search
from src.services.extraction import extract_structured_data
from src.services.aqt import transform_query
from src.services.formatting import get_formatter_chain
from src.services.verification import verify_semantic_match
from src.services.intent_classifier import get_intent_classifier
from src.cache.result_cache import (
    get_cached_answer_semantic, set_cached_answer_semantic,
    get_cached_query, set_cached_query,
    get_cache_stats
)
from src.models.embeddings import embed_text
from src.schemas.query import GraphRAGQuery
from src.services.ranking_service import get_reranker


def _configure_stdout_utf8() -> None:
    """
    Best-effort UTF-8 stdout/stderr setup for Windows shells.
    Prevents UnicodeEncodeError in debug logs.
    """
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    try:
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


_configure_stdout_utf8()

# Log path configuration
LOG_PATH = "./logs/ragas_data.jsonl"
pathlib.Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)

# Intent Classification dataset
INTENT_DATASET = "./api/intent_dataset.json"
HYBRID_FINAL_TOP_K = 25

NLEM_UNSUPPORTED_PATTERNS = [
    r"\bnlem\b",
    r"บัญชียาหลัก",
    r"เบิกได้",
    r"เบิก",
    r"reimburse",
    r"reimbursement",
    r"essential medicines?",
    r"national list",
]


def _is_nlem_question(question: str) -> bool:
    text = str(question or "").strip().lower()
    if not text:
        return False
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in NLEM_UNSUPPORTED_PATTERNS)


def _is_cypher_deterministic_route(route: dict) -> bool:
    """
    Cypher-based deterministic routes should keep full payload.
    """
    if not isinstance(route, dict):
        return False

    operator = str(route.get("operator") or "").lower()
    if operator in {"id_lookup", "analyze_count"}:
        return True

    if operator == "list":
        # Deterministic list routes have fallback_used=False.
        # Hybrid list fallback marks fallback_used=True.
        return route.get("fallback_used") is not True

    return False


class GraphRAGPipeline:
    def __init__(self):
        """Initialize the pipeline, validate env, and setup connections."""
        try:
            validate_env()
        except RuntimeError as e:
            print(f"Configuration Error: {e}")
            raise

        init_driver()
        print("=== Neo4j + Ollama Hybrid Retriever Demo (Modular - LangChain Phase 3) ===")
        
        try:
            setup_indexes()
        except Exception as e:
            print(f"Warning: Could not setup indexes: {e}")

        # Initialize (shared singleton) Reranker
        self.reranker = get_reranker()
        # self.reranker = None  # Configurable: Set to Reranker() if needed

        # Initialize Intent Classifier (compute centroids at startup)
        print("\n🧠 Initializing Intent Classifier...")
        self.intent_classifier = get_intent_classifier()
        self.intent_classifier.initialize()

        # build the LCEL chain
        self.chain = self._build_chain()
        self._timing = {}

    def _build_chain(self) -> Runnable:
        """
        Constructs the LCEL chain for the RAG pipeline.
        Flow: Input -> Transform -> Search (+Rerank) -> Extract -> Format -> Output
        """
        formatter_chain = get_formatter_chain()
        formatter_step = RunnableLambda(lambda x: self._step_format_answer(x, formatter_chain))

        return (
            RunnablePassthrough.assign(query_obj=self._step_transform)
            | RunnablePassthrough.assign(results=self._step_search)
            | RunnablePassthrough.assign(context=self._step_extract)
            | RunnablePassthrough.assign(answer=formatter_step)
        )

    # ==========================
    # Chain Steps (Runnables)
    # ==========================

    def _step_transform(self, inputs: dict) -> GraphRAGQuery:
        """Step 1: AQT - Intent Classification + Filter Extraction (no LLM)"""
        if not hasattr(self, "_timing"):
            self._timing = {}
        start_time = time.perf_counter()
        question = inputs["question"]
        q_embedding = inputs.get("q_embedding")  # From run() for cache optimization
        
        print(f"\n→ Process: AQT (Intent + Filters) ...")
        
        # transform_query now handles both intent classification and filter extraction
        query_obj = transform_query(question, q_embedding)
        
        # Print debug info
        raw_intent = getattr(query_obj, '_raw_intent', 'unknown')
        confidence = getattr(query_obj, '_intent_confidence', 0.0)
        target_conf = getattr(query_obj, '_target_confidence', 0.0)
        target_margin = getattr(query_obj, '_target_margin', 0.0)
        
        print(f"   Intent: {raw_intent} (fine_conf: {confidence:.4f}, target_conf: {target_conf:.4f}, margin: {target_margin:.4f})")
        print(f"   Target Type: {query_obj.target_type.upper()} | Strategy: {query_obj.strategy}")
        print(f"   Filters: NLEM={query_obj.nlem_filter}, Cat={query_obj.nlem_category}, Manu={query_obj.manufacturer_filter}")
        print(
            "   Retrieval Profile: "
            f"mode={query_obj.retrieval_mode}, "
            f"entity_ratio={query_obj.entity_ratio:.2f}, "
            f"weights(v={query_obj.vector_weight:.2f}, f={query_obj.fulltext_weight:.2f})"
        )
        print(f"   Search Term: '{query_obj.query}'")
        if getattr(query_obj, "id_lookup", None):
            print(f"   ID Lookup: {query_obj.id_lookup}")
        bundle = getattr(query_obj, "intent_bundle", None)
        if bundle:
            action = bundle.get("action_intent", "unknown")
            topics = bundle.get("topics_intents", [])
            print(f"   IntentV2: action={action}, topics={topics}")
        
        elapsed = time.perf_counter() - start_time
        print(f"   ⏱️ Transform Time: {elapsed:.3f}s")
        self._timing['transform'] = elapsed
        
        return query_obj

    def _step_search(self, inputs: dict) -> dict:
        """Step 2: Search (Vector + Fulltext + selective rerank)"""
        if not hasattr(self, "_timing"):
            self._timing = {}
        start_time = time.perf_counter()
        query_obj = inputs["query_obj"]
        question = inputs["question"]
        
        print(f"\\n→ Process: Advanced GraphRAG Search ...")
        results = advanced_graphrag_search(query_obj, k=50, depth=GRAPH_TRAVERSAL_DEPTH)
        # Selective reranking: apply only on lookup route.
        route = results.get("route") or {}
        route_operator = str(route.get("operator") or "").lower()
        if self.reranker and route_operator == "lookup" and results.get("seed_results"):
            print(f"\n-> Process: Re-ranking lookup seeds ({len(results['seed_results'])} candidates) ...")
            reranked = self.reranker.rerank(question, results["seed_results"], top_k=50)
            results["seed_results"] = reranked
            if reranked:
                print(f"   Top candidate: {reranked[0].get('score')} -> {reranked[0].get('rerank_score')}")

        # Final seed cap for hybrid routes only.
        # Cypher deterministic routes keep all results as requested.
        if (
            not _is_cypher_deterministic_route(route)
            and isinstance(results.get("seed_results"), list)
            and len(results["seed_results"]) > HYBRID_FINAL_TOP_K
        ):
            before = len(results["seed_results"])
            results["seed_results"] = results["seed_results"][:HYBRID_FINAL_TOP_K]
            print(f"   Hybrid seed cap applied: {before} -> {len(results['seed_results'])}")

        elapsed = time.perf_counter() - start_time
        print(f"   ⏱️ Search Time: {elapsed:.3f}s")
        self._timing['search'] = elapsed
        
        return results

    def _step_extract(self, inputs: dict) -> dict:
        """Step 3: Extract Structured Data"""
        if not hasattr(self, "_timing"):
            self._timing = {}
        start_time = time.perf_counter()
        results = inputs["results"]
        query_obj = inputs["query_obj"]
        
        print(f"\\n→ Process: Data Extraction ...")
        structured = extract_structured_data(results, query_obj.target_type)

        # Debug info
        num_seeds = len(results.get("seed_results", []))
        num_expanded = len(results.get("expanded_nodes", []))
        num_rels = len(results.get("relationships", []))
        num_entities = len(structured.get("entities", []))
        
        print(f"   Found: {num_seeds} primary nodes, {num_expanded} related nodes, {num_rels} relationships")
        print(f"   Context: {num_entities} entities will be sent to LLM")
        
        elapsed = time.perf_counter() - start_time
        print(f"   ⏱️ Extraction Time: {elapsed:.3f}s")
        self._timing['extract'] = elapsed
        
        return structured

    def _render_count_answer(self, question: str, results: dict, context: dict | None) -> str:
        """
        Deterministic renderer for count queries.
        Avoids LLM hallucination for numeric responses.
        """
        count_value = None
        if isinstance(context, dict):
            count_value = context.get("count")
        if count_value is None and isinstance(results, dict):
            count_value = results.get("result")

        try:
            count_int = int(count_value)
        except Exception:
            return "ไม่พบจำนวนที่เชื่อถือได้จากผลการค้นหา"

        route = results.get("route", {}) if isinstance(results, dict) else {}
        mode = str(route.get("count_mode", "")).strip()
        if mode:
            return f"ผลการนับสำหรับคำถามนี้: {count_int} รายการ"
        return f"ผลการนับ: {count_int} รายการ"

    def _step_format_answer(self, inputs: dict, formatter_chain: Runnable) -> str:
        """
        Step 4: Format answer.
        - count strategy uses deterministic rendering
        - others use LLM formatter
        """
        results = inputs.get("results") or {}
        strategy = str(results.get("strategy", "")).strip().lower()
        if strategy == "count":
            return self._render_count_answer(
                question=str(inputs.get("question", "")),
                results=results,
                context=inputs.get("context"),
            )
        return formatter_chain.invoke(inputs)

    # ==========================
    # Main Execution
    # ==========================

    def run(self, question: str, ground_truth: str = None) -> str:
        """
        Execute the full RAG pipeline for a given question.
        Returns the final answer.
        """
        question = question.strip()
        if not question:
            return ""
        if not NLEM_QA_ENABLED and _is_nlem_question(question):
            return (
                "ขณะนี้ระบบยังไม่รองรับคำถามเกี่ยวกับบัญชียาหลักแห่งชาติ (NLEM) "
                "เนื่องจากยังไม่ได้นำเข้าข้อมูลส่วนนี้ในฐานความรู้"
            )

        # 1. Generate Embedding (for Intent Classification)
        print("\n→ Generating question embedding...")
        q_embedding = embed_text(question)

        # 2. Invoke Chain
        print("\n🚀 Invoking Pipeline Chain...")
        self._timing = {}  # Reset timing dict
        pipeline_start = time.perf_counter()
        
        try:
            # The chain returns a dict with all steps' outputs: {question, query_obj, results, context, answer}
            output = self.chain.invoke({"question": question, "q_embedding": q_embedding})
            answer = output["answer"]
            results = output["results"]
            
            # Calculate LLM time (total - other steps)
            pipeline_total = time.perf_counter() - pipeline_start
            other_time = sum(self._timing.values())
            llm_time = pipeline_total - other_time
            self._timing['llm'] = llm_time
            
            # Print timing summary
            print(f"\n⏱️ Pipeline Timing Summary:")
            print(f"   Transform: {self._timing.get('transform', 0):.3f}s")
            print(f"   Search:    {self._timing.get('search', 0):.3f}s")
            print(f"   Extract:   {self._timing.get('extract', 0):.3f}s")
            print(f"   LLM:       {self._timing.get('llm', 0):.3f}s")
            print(f"   ────────────────────")
            print(f"   Total:     {pipeline_total:.3f}s")
            print("\nตอบ:\n", answer)

            # 3. Update Answer Cache
            set_cached_answer_semantic(question, q_embedding, answer)

            # 4. Log Interaction
            self._log_interaction(question, results, answer, ground_truth)

            return answer

        except Exception as e:
            print(f"❌ Pipeline Execution Failed: {e}")
            import traceback
            traceback.print_exc()
            return "ขออภัย เกิดข้อผิดพลาดในระบบประมวลผล"

    def warmup(self):
        """Warmup models (Embedding & LLM) to reduce first-inference latency."""
        print("🔥 Warming up models (this may take a few seconds)...")
        
        # 1. Warmup Embedding
        try:
            embed_text("warmup")
            print("   ✅ Embedding Model Ready")
        except Exception as e:
            print(f"   ⚠️ Embedding Warmup Failed: {e}")

        # 2. Warmup Classification LLM (Small model)
        try:
            # Import locally to avoid circular deps if any
            from src.models.llm import get_llm
            from src.config import CLASSIFICATION_MODEL
            
            llm = get_llm(model=CLASSIFICATION_MODEL)
            llm.invoke("Hi")
            print("   ✅ Classification Model Ready")
        except Exception as e:
            print(f"   ⚠️ LLM Warmup Failed: {e}")
        
        # 3. Warmup Reranker
        try:
            if self.reranker:
                 self.reranker.rerank("warmup", [{"node": {"fsn": "test"}}], top_k=1)
                 print("   ✅ Reranker Ready")
        except Exception as e:
             print(f"   ⚠️ Reranker Warmup Failed: {e}")

        print("🔥 System Ready!\n")

    def _log_interaction(self, question: str, results: dict, answer: str, ground_truth: str = None):
        """Append interaction log to JSONL file."""
        contexts = []
        all_nodes = results.get("seed_results", []) + results.get("expanded_nodes", [])
        
        for item in all_nodes:
            node = item["node"]
            props = dict(node)
            text_parts = []
            
            # Prioritize semantic fields
            if "fsn" in props: text_parts.append(str(props["fsn"]))
            if "trade_name" in props: text_parts.append(f"trade_name: {props['trade_name']}")
            if "generic_name" in props: text_parts.append(f"generic_name: {props['generic_name']}")
            if "manufacturer" in props: text_parts.append(f"manufacturer: {props['manufacturer']}")
            if "nlem" in props: text_parts.append(f"nlem: {props['nlem']}")
            
            if text_parts:
                contexts.append(" | ".join(text_parts))

        record = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "contexts": contexts,
            "answer": answer,
            "ground_truth": ground_truth
        }

        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def print_cache_stats(self):
        """Print current cache statistics."""
        stats = get_cache_stats()
        print("\n\n📊 Cache Statistics:")
        print(f"   Answer Cache: {stats['answer_cache']['hits_exact']} exact + {stats['answer_cache']['hits_semantic']} semantic / {stats['answer_cache']['misses']} misses")
        print(f"   Query Cache:  {stats['query_cache']['hits']} hits / {stats['query_cache']['misses']} misses")
