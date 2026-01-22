
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import GraphRAGPipeline
from src.schemas.query import GraphRAGQuery

def test_retrieval():
    print("Initializing Pipeline for Verification...")
    try:
        pipeline = GraphRAGPipeline()
        pipeline.warmup()
    except Exception as e:
        print(f"Failed to init pipeline: {e}")
        return

    test_cases = [
        # Case 1: Synonym (Expansion Check)
        {"question": "ยาพาราเซตามอลมีสรรพคุณอะไร", "expected_term": "Paracetamol"},
        # Case 2: Broad category (Re-ranking Check)
        {"question": "ยาแก้ปวด", "expected_behavior": "Should rank specific painkillers higher than generic categories"}
    ]

    print("\nStarting Retrieval Verification...")
    
    for case in test_cases:
        q = case["question"]
        print(f"\n--------------------------------------------------")
        print(f"Testing Question: '{q}'")
        
        # 1. Test Query Transformation
        from src.chains.query_transform_chain import transform_query
        query_obj = transform_query(q)
        print(f"   [Query Transform]")
        print(f"   Term: {query_obj.query}")
        print(f"   Expanded: {query_obj.expanded_queries}")
        
        # 2. Test Retrieval & Re-ranking (via pipeline logic)
        # We manually call internal methods to inspect
        from src.knowledge_graph import advanced_graphrag_search
        from src.config import GRAPH_TRAVERSAL_DEPTH
        
        results = advanced_graphrag_search(query_obj, k=30, depth=GRAPH_TRAVERSAL_DEPTH)
        seeds = results.get("seed_results", [])
        print(f"   [Retrieval] Found {len(seeds)} seed nodes.")
        
        if pipeline.reranker and seeds:
            print(f"   [Re-ranking]")
            reranked = pipeline.reranker.rerank(q, seeds, top_k=10)
            print(f"   Top 5 after re-ranking:")
            for i, item in enumerate(reranked[:5]):
                node = item["node"]
                name = node.get("fsn") or node.get("trade_name") or "Unknown"
                print(f"     {i+1}. {name} (Score: {item.get('rerank_score'):.4f})")
        else:
            print("   [Re-ranking] Skipped (No reranker or no seeds)")

if __name__ == "__main__":
    test_retrieval()
