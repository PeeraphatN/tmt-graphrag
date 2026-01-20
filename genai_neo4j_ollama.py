import json
import uuid
from datetime import datetime
import pathlib
import atexit

from src.config import validate_env, LLM_MODEL, GRAPH_TRAVERSAL_DEPTH
from src.knowledge_graph import (
    init_driver, 
    close_driver, 
    setup_indexes, 
    graphrag_search, 
    extract_structured_data,
    advanced_graphrag_search
)
from src.chains.query_transform_chain import transform_query
from src.chains.formatter_chain import format_answer_llm
from src.cache.result_cache import (
    get_cached_answer, set_cached_answer,
    get_cached_query, set_cached_query,
    get_cache_stats
)
from src.schemas.query import GraphRAGQuery

# Register cleanup
atexit.register(close_driver)

# ==============================
# MAIN PROGRAM
# ==============================

LOG_PATH = "./logs/ragas_data.jsonl"
pathlib.Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)

def log_interaction(question: str, results: dict, answer: str):
    contexts = []
    # Combine seed and expanded nodes for logging
    all_nodes = results.get("seed_results", []) + results.get("expanded_nodes", [])
    
    for item in all_nodes:
        node = item["node"]
        props = dict(node)

        text_parts = []
        if "fsn" in props:
            text_parts.append(str(props["fsn"]))
        if "embedding_text" in props:
            text_parts.append(str(props["embedding_text"]))
        if "trade_name" in props:
            text_parts.append(f"trade_name: {props['trade_name']}")
        if "manufacturer" in props:
            text_parts.append(f"manufacturer: {props['manufacturer']}")

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

def main():
    print("Starting main...")
    try:
        validate_env()
    except RuntimeError as e:
        print(e)
        return

    init_driver()
    print("=== Neo4j + Ollama Hybrid Retriever Demo (Modular - LangChain Phase 3) ===")
    print(f"Running on {LLM_MODEL}")
    
    # Setup Indexes
    try:
        setup_indexes()
    except Exception as e:
        print(f"Warning: Could not setup indexes: {e}")
        
    print("พิมพ์คำถาม หรือ 'exit' เพื่อออก")

    try:
        while True:
            q = input("\nถาม: ").strip()
            if q.lower() in ("exit", "quit"):
                break
            
            # ===== CACHE CHECK: Layer 3 (Answer) =====
            cached_answer = get_cached_answer(q)
            if cached_answer:
                print("\n⚡ [CACHE HIT] คำตอบถูกดึงจาก Cache")
                print("\nตอบ:\n", cached_answer)
                continue

            print(f"\n→ Process: Query Transformation (Self-Querying) ...")
            
            # ===== CACHE CHECK: Layer 1 (Query Transform) =====
            cached_query = get_cached_query(q)
            if cached_query:
                print("   ⚡ [CACHE HIT] Query Transform")
                # Reconstruct GraphRAGQuery from cached dict
                query_obj = GraphRAGQuery(
                    query=cached_query["query"],
                    target_type=cached_query["target_type"],
                    nlem_filter=cached_query.get("nlem_filter"),
                    nlem_category=cached_query.get("nlem_category"),
                    manufacturer_filter=cached_query.get("manufacturer_filter"),
                )
            else:
                query_obj = transform_query(q)
                set_cached_query(q, query_obj)
            
            print(f"   Intent: {query_obj.target_type.upper()}")
            print(f"   Filters: NLEM={query_obj.nlem_filter}, Cat={query_obj.nlem_category}")
            print(f"   Search Term: '{query_obj.query}'")

            # 2. Advanced Search (Routing based on filters)
            print(f"\n→ Process: Advanced GraphRAG Search ...")
            results = advanced_graphrag_search(query_obj, k=30, depth=GRAPH_TRAVERSAL_DEPTH)
            
            # 3. Extract Structured Data
            print(f"\n→ Process: Data Extraction ...")
            structured = extract_structured_data(results, query_obj.target_type)
            
            # Debug context size (approx)
            json_ctx = json.dumps(structured, ensure_ascii=False)
            print(f"   Structured Data Size: {len(json_ctx)} chars")

            # Show search stats
            num_seeds = len(results.get("seed_results", []))
            num_expanded = len(results.get("expanded_nodes", []))
            num_rels = len(results.get("relationships", []))
            print(f"   Found: {num_seeds} primary nodes, {num_expanded} related nodes, {num_rels} relationships")

            print("\n→ ส่งให้ LLM ตอบ (Structured Mode - LangChain) ...")
            answer = format_answer_llm(q, structured)
            print("\nตอบ:\n", answer)
            
            # ===== CACHE SAVE: Layer 3 (Answer) =====
            set_cached_answer(q, answer)

            log_interaction(q, results, answer)
    except (KeyboardInterrupt, EOFError):
        # Show cache stats on exit
        stats = get_cache_stats()
        print("\n\n📊 Cache Statistics:")
        print(f"   Answer Cache: {stats['answer_cache']['hits']} hits / {stats['answer_cache']['misses']} misses")
        print(f"   Query Cache:  {stats['query_cache']['hits']} hits / {stats['query_cache']['misses']} misses")
        print("\nออกจากโปรแกรม")


if __name__ == "__main__":
    main()