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
    extract_structured_data
)
from src.query_processor import classify_question
from src.llm_service import ask_ollama_structured

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
    print("=== Neo4j + Ollama Hybrid Retriever Demo (Modular) ===")
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

            print(f"\n→ ค้นหาแบบ GraphRAG (Hybrid + Relationship Traversal, depth={GRAPH_TRAVERSAL_DEPTH}) ...")
            
            # 1. Classify
            q_type = classify_question(q)
            print(f"   Question Type: {q_type}")

            results = graphrag_search(q, k=10, depth=GRAPH_TRAVERSAL_DEPTH)
            
            # 2. Extract Structured Data
            structured = extract_structured_data(results, q_type)
            # Debug context size (approx)
            json_ctx = json.dumps(structured, ensure_ascii=False)
            print(f"   Structured Data Size: {len(json_ctx)} chars")

            # Show search stats
            num_seeds = len(results.get("seed_results", []))
            num_expanded = len(results.get("expanded_nodes", []))
            num_rels = len(results.get("relationships", []))
            print(f"   Found: {num_seeds} primary nodes, {num_expanded} related nodes, {num_rels} relationships")

            # Debug structured data
            # print("\nDebug Structured Data:")
            # print(json.dumps(structured, ensure_ascii=False, indent=2)) 

            print("\n→ ส่งให้ LLM ตอบ (Structured Mode) ...")
            answer = ask_ollama_structured(q, structured)
            print("\nตอบ:\n", answer)

            log_interaction(q, results, answer)
    except (KeyboardInterrupt, EOFError):
        print("\nออกจากโปรแกรม")


if __name__ == "__main__":
    main()