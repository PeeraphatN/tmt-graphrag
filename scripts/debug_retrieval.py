import sys
import os

# Ensure project root is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# from src.pipeline import GraphRAGPipeline # Causing import issues
from src.knowledge_graph import advanced_graphrag_search
from src.chains.query_transform_chain import transform_query
from src.models.embeddings import embed_text

def debug_query(question: str):
    print(f"\n🔎 Debugging Query: '{question}'")
    
    # 1. Check Query Transform
    print("\n[1] Checking Query Transformation...")
    try:
        query_obj = transform_query(question)
        print(f"    Intent: {query_obj.target_type}")
        print(f"    Filters: NLEM={query_obj.nlem_filter}, Cat={query_obj.nlem_category}")
        print(f"    Search Query: '{query_obj.query}'")
    except Exception as e:
        print(f"    ❌ Error: {e}")
        return

    # 2. Check Vector SearchRaw Results
    print("\n[2] Checking Raw Vector Search (Top 5)...")
    try:
        # We manually call the search logic to inspect raw scores
        results = advanced_graphrag_search(query_obj, k=5, depth=0) # depth=0 to just check seed nodes
        seeds = results.get("seed_results", [])
        
        if not seeds:
            print("    ❌ No seed nodes found! (Vector Search Failed)")
        else:
            for i, seed in enumerate(seeds):
                node = seed["node"]
                score = seed["score"]
                is_nlem = node.get("is_nlem", "N/A")
                print(f"    #{i+1} Score: {score:.4f} | NLEM: {is_nlem} | Name: {node.get('fsn', 'N/A')}")
    except Exception as e:
        print(f"    ❌ Error in Search: {e}")

if __name__ == "__main__":
    from src.config import validate_env
    from src.knowledge_graph import init_driver
    validate_env()
    init_driver()
    
    # Test the problematic query
    debug_query("ยา Alendronate sodium อยู่ในบัญชียาหลักไหม?")
