import json
import uuid
import pathlib
from datetime import datetime

from src.config import validate_env, GRAPH_TRAVERSAL_DEPTH
from src.knowledge_graph import (
    init_driver, 
    setup_indexes, 
    extract_structured_data,
    advanced_graphrag_search
)
from src.chains.query_transform_chain import transform_query
from src.chains.formatter_chain import format_answer_llm
from src.chains.verify_cache_chain import verify_semantic_match
from src.cache.result_cache import (
    get_cached_answer_semantic, set_cached_answer_semantic,
    get_cached_query, set_cached_query,
    get_cache_stats
)
from src.models.embeddings import embed_text
from src.schemas.query import GraphRAGQuery

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
        
        print("🔥 System Ready!\n")

    def run(self, question: str) -> str:
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
        cached_answer, is_semantic = get_cached_answer_semantic(
            question, 
            q_embedding,
            verification_fn=verify_semantic_match
        )
        if cached_answer:
            if is_semantic:
                print("⚡ [CACHE HIT - SEMANTIC] คำถามคล้ายกันถูกดึงจาก Cache")
            else:
                print("⚡ [CACHE HIT - EXACT] คำตอบถูกดึงจาก Cache")
            print("\nตอบ:\n", cached_answer)
            return cached_answer

        # 3. Query Transformation (Layer 1 Cache)
        print(f"\n→ Process: Query Transformation (Self-Querying) ...")
        cached_query = get_cached_query(question)
        
        if cached_query:
            print("   ⚡ [CACHE HIT] Query Transform")
            query_obj = GraphRAGQuery(
                query=cached_query["query"],
                target_type=cached_query["target_type"],
                nlem_filter=cached_query.get("nlem_filter"),
                nlem_category=cached_query.get("nlem_category"),
                manufacturer_filter=cached_query.get("manufacturer_filter"),
            )
        else:
            query_obj = transform_query(question)
            set_cached_query(question, query_obj)

        print(f"   Intent: {query_obj.target_type.upper()}")
        print(f"   Filters: NLEM={query_obj.nlem_filter}, Cat={query_obj.nlem_category}")
        print(f"   Search Term: '{query_obj.query}'")

        # 4. Advanced Graph Search & Routing
        print(f"\n→ Process: Advanced GraphRAG Search ...")
        results = advanced_graphrag_search(query_obj, k=30, depth=GRAPH_TRAVERSAL_DEPTH)

        # 5. Extract Structured Data
        print(f"\n→ Process: Data Extraction ...")
        structured = extract_structured_data(results, query_obj.target_type)
        
        # Debug info
        num_seeds = len(results.get("seed_results", []))
        num_expanded = len(results.get("expanded_nodes", []))
        num_rels = len(results.get("relationships", []))
        num_entities = len(structured.get("entities", []))
        
        print(f"   Found: {num_seeds} primary nodes, {num_expanded} related nodes, {num_rels} relationships")
        print(f"   Context: {num_entities} entities will be sent to LLM")

        # 6. Generate Answer (LLM)
        print("\n→ ส่งให้ LLM ตอบ (Structured Mode - LangChain) ...")
        answer = format_answer_llm(question, structured)
        print("\nตอบ:\n", answer)

        # 7. Update Answer Cache
        set_cached_answer_semantic(question, q_embedding, answer)

        # 8. Log Interaction
        self._log_interaction(question, results, answer)

        return answer

    def _log_interaction(self, question: str, results: dict, answer: str):
        """Append interaction log to JSONL file."""
        contexts = []
        all_nodes = results.get("seed_results", []) + results.get("expanded_nodes", [])
        
        for item in all_nodes:
            node = item["node"]
            props = dict(node)
            text_parts = []
            
            if "fsn" in props: text_parts.append(str(props["fsn"]))
            if "embedding_text" in props: text_parts.append(str(props["embedding_text"]))
            if "trade_name" in props: text_parts.append(f"trade_name: {props['trade_name']}")
            
            if text_parts:
                contexts.append(" | ".join(text_parts))

        record = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "contexts": contexts,
            "answer": answer,
        }

        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def print_cache_stats(self):
        """Print current cache statistics."""
        stats = get_cache_stats()
        print("\n\n📊 Cache Statistics:")
        print(f"   Answer Cache: {stats['answer_cache']['hits_exact']} exact + {stats['answer_cache']['hits_semantic']} semantic / {stats['answer_cache']['misses']} misses")
        print(f"   Query Cache:  {stats['query_cache']['hits']} hits / {stats['query_cache']['misses']} misses")
