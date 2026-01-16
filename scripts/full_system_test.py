"""
Full System Helper Test Script
To verify RAG accuracy against Neo4j Ground Truth.
"""
import sys
import os
import json
import re

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.knowledge_graph import init_driver, advanced_graphrag_search
from src.chains.query_transform_chain import transform_query
from src.chains.formatter_chain import format_answer_llm
from src.schemas.query import GraphRAGQuery

# Test Cases
TEST_CASES = [
    {
        "category": "NLEM",
        "question": "ยาในหมวด nlem มีทั้งหมดกี่ตัว",
        "expected_intent": "nlem",
        "expected_filter_key": "nlem_filter",
        "expected_filter_value": True
    },
    {
        "category": "NLEM",
        "question": "ยา Calcitonin อยู่ในบัญชีอะไร",
        "expected_intent": "nlem",
        "verify_keyword": "ง" # Category
    },
    {
        "category": "Ingredient",
        "question": "ยาที่มี Paracetamol 500 mg มีชื่อการค้าอะไรบ้าง",
        "expected_intent": "ingredient",
        "verify_keyword": "ACETAPHEN" # One of the results
    },
    {
        "category": "General",
        "question": "Simvastatin 20 mg คือยาอะไร",
        "expected_intent": "general",
        "verify_keyword": "ZOCOR"
    }
]

def run_test():
    print("=== Starting Full System Verification ===\n")
    driver = init_driver()
    results_summary = []

    for i, case in enumerate(TEST_CASES):
        print(f"Test Case {i+1}: {case['question']}")
        
        # 1. Test Query Transformation
        try:
            q_obj = transform_query(case["question"])
            print(f"   Intent: {q_obj.target_type}")
            
            intent_pass = True
            if "expected_intent" in case:
                if q_obj.target_type != case["expected_intent"]:
                    print(f"   [FAIL] Intent mismatch. Expected {case['expected_intent']}, got {q_obj.target_type}")
                    intent_pass = False
            
            if "expected_filter_key" in case:
                val = getattr(q_obj, case["expected_filter_key"])
                if val != case["expected_filter_value"]:
                     print(f"   [FAIL] Filter mismatch. Expected {case['expected_filter_value']}, got {val}")
                     intent_pass = False
            
            if intent_pass:
                print("   [PASS] Query Transformation")
            
            # 2. Test Retrieval (Coverage)
            search_res = advanced_graphrag_search(q_obj, k=50) # Get enough results
            found_nodes = search_res["seed_results"] + search_res["expanded_nodes"]
            
            # Quick check if result isn't empty
            if not found_nodes:
                print("   [WARN] No nodes found via search.")
            else:
                print(f"   [PASS] Retrieval found {len(search_res['seed_results'])} seeds, {len(search_res['expanded_nodes'])} expanded.")

            # 3. Test LLM Answer (Accuracy)
            # Use format_answer_llm directly to simulate RAG
            structured = {
                 "entities": [dict(n["node"]) for n in found_nodes[:20]] # Simulation extract
            }
            # For verification, we check if keywords exist in the RAW NODE DATA first (Ground Truth)
            # Because checking LLM text is fuzzy.
            
            verify_pass = True
            if "verify_keyword" in case:
                keyword = case["verify_keyword"].lower()
                found_in_graph = False
                for item in found_nodes:
                    node = item["node"]
                    props = str(dict(node)).lower()
                    if keyword in props:
                        found_in_graph = True
                        break
                
                if found_in_graph:
                     print(f"   [PASS] Verification keyword '{case['verify_keyword']}' found in Graph Results.")
                else:
                     print(f"   [FAIL] Verification keyword '{case['verify_keyword']}' NOT found in Graph Results.")
                     verify_pass = False
            
            results_summary.append({
                "question": case["question"],
                "intent_pass": intent_pass,
                "retrieval_count": len(found_nodes),
                "verify_pass": verify_pass
            })
            
        except Exception as e:
            print(f"   [ERROR] {e}")
            results_summary.append({"question": case["question"], "error": str(e)})
        
        print("-" * 30)

    print("\n=== Summary ===")
    for res in results_summary:
        status = "PASS" if res.get("intent_pass") and res.get("verify_pass") else "FAIL"
        print(f"{res['question'][:30]}... : {status} (Nodes: {res.get('retrieval_count',0)})")

if __name__ == "__main__":
    run_test()
