import json
import uuid
import pathlib
from datetime import datetime
from operator import itemgetter

from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda

from src.config import validate_env, GRAPH_TRAVERSAL_DEPTH

# Service Imports (Refactored)
from src.services.database import init_driver, setup_indexes
from src.services.search import advanced_graphrag_search
from src.services.extraction import extract_structured_data
from src.services.aqt import transform_query
from src.services.formatting import get_formatter_chain
from src.services.verification import verify_semantic_match
from src.cache.result_cache import (
    get_cached_answer_semantic, set_cached_answer_semantic,
    get_cached_query, set_cached_query,
    get_cache_stats
)
from src.models.embeddings import embed_text
from src.schemas.query import GraphRAGQuery
from src.services.ranking_service import Reranker

# Log path configuration
LOG_PATH = "./logs/ragas_data.jsonl"
pathlib.Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)

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

        # Initialize Reranker
        self.reranker = Reranker()
        # self.reranker = None  # Configurable: Set to Reranker() if needed

        # build the LCEL chain
        self.chain = self._build_chain()

    def _build_chain(self) -> Runnable:
        """
        Constructs the LCEL chain for the RAG pipeline.
        Flow: Input -> Transform -> Search (+Rerank) -> Extract -> Format -> Output
        """
        formatter_chain = get_formatter_chain()

        return (
            RunnablePassthrough.assign(query_obj=self._step_transform)
            | RunnablePassthrough.assign(results=self._step_search)
            | RunnablePassthrough.assign(context=self._step_extract)
            | RunnablePassthrough.assign(answer=formatter_chain) 
        )

    # ==========================
    # Chain Steps (Runnables)
    # ==========================

    def _step_transform(self, inputs: dict) -> GraphRAGQuery:
        """Step 1: Transform Query (with Caching)"""
        question = inputs["question"]
        print(f"\\n→ Process: Query Transformation (Self-Querying) ...")
        
        # cached_query = get_cached_query(question)
        # if cached_query:
        #     print("   ⚡ [CACHE HIT] Query Transform")
        #     query_obj = GraphRAGQuery(
        #         query=cached_query["query"],
        #         target_type=cached_query["target_type"],
        #         nlem_filter=cached_query.get("nlem_filter"),
        #         nlem_category=cached_query.get("nlem_category"),
        #         manufacturer_filter=cached_query.get("manufacturer_filter"),
        #     )
        # else:
        #     query_obj = transform_query(question)
        #     set_cached_query(question, query_obj)

        query_obj = transform_query(question)
        
        

        print(f"   Intent: {query_obj.target_type.upper()}")
        print(f"   Filters: NLEM={query_obj.nlem_filter}, Cat={query_obj.nlem_category}")
        print(f"   Search Term: '{query_obj.query}'")
        
        return query_obj

    def _step_search(self, inputs: dict) -> dict:
        """Step 2: Search (Vector + Fulltext + Rerank)"""
        query_obj = inputs["query_obj"]
        question = inputs["question"]
        
        print(f"\\n→ Process: Advanced GraphRAG Search ...")
        results = advanced_graphrag_search(query_obj, k=50, depth=GRAPH_TRAVERSAL_DEPTH)

        # Reranking logic
        if self.reranker and results.get("seed_results"):
            print(f"\\n→ Process: Re-ranking ({len(results['seed_results'])} candidates) ...")
            reranked = self.reranker.rerank(question, results["seed_results"], top_k=50)
            results["seed_results"] = reranked
            if reranked:
                print(f"   Top candidate: {reranked[0].get('score')} -> {reranked[0].get('rerank_score')}")

        return results

    def _step_extract(self, inputs: dict) -> dict:
        """Step 3: Extract Structured Data"""
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
        
        return structured

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

        # 1. Generate Embedding (for semantic cache)
        print("\n→ Generating question embedding...")
        q_embedding = embed_text(question)

        # 2. Check Answer Cache (Layer 3)
        # cached_answer, is_semantic = get_cached_answer_semantic(
        #     question, 
        #     q_embedding,
        #     verification_fn=verify_semantic_match
        # )
        # if cached_answer:
        #     if is_semantic:
        #         print("⚡ [CACHE HIT - SEMANTIC] คำถามคล้ายกันถูกดึงจาก Cache")
        #     else:
        #         print("⚡ [CACHE HIT - EXACT] คำตอบถูกดึงจาก Cache")
        #     print("\nตอบ:\n", cached_answer)
        #     return cached_answer

        # 3. Invoke Chain
        print("\n🚀 Invoking Pipeline Chain...")
        try:
            # The chain returns a dict with all steps' outputs: {question, query_obj, results, context, answer}
            output = self.chain.invoke({"question": question})
            answer = output["answer"]
            results = output["results"]
            
            print("\nตอบ:\n", answer)

            # 4. Update Answer Cache
            set_cached_answer_semantic(question, q_embedding, answer)

            # 5. Log Interaction
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
