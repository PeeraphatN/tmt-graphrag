import asyncio
import sys
import os

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.pipeline import GraphRAGPipeline
from src.schemas.query import SearchStrategy

async def test_intent_router():
    pipeline = GraphRAGPipeline()
    
    test_cases = [
        {
            "question": "มียากี่ตัวในบัญชี ง", 
            "expected_strategy": SearchStrategy.COUNT
        },
        {
            "question": "ขอรายชื่อยาของ SUN PHARMACEUTICAL",
            "expected_strategy": SearchStrategy.LIST
        },
        {
            "question": "ยา Plavix อยู่ในบัญชีไหม",
            "expected_strategy": SearchStrategy.VERIFY
        },
         {
            "question": "ยา Paracetamol สรรพคุณคืออะไร",
            "expected_strategy": SearchStrategy.RETRIEVE
        }
    ]

    print("=== Testing Intent Router & Specialized Search ===\n")

    for case in test_cases:
        q = case["question"]
        print(f"\n🧪 Testing Input: '{q}'")
        
        # 1. Test Transform Step (AQT)
        print("   -> Running AQT Transform...")
        input_dict = {"question": q}
        query_obj = pipeline._step_transform(input_dict)
        
        # NOTE: _step_transform returns the GraphRAGQuery object directly
        strategy = query_obj.strategy
        print(f"   -> Detected Strategy: {strategy}")
        
        if strategy != case["expected_strategy"]:
            print(f"   ❌ MISMATCH: Expected {case['expected_strategy']}, got {strategy}")
        else:
            print(f"   ✅ Strategy Match!")

        # 2. Test Search Step
        print("   -> Running Search Step...")
        # Mocking the input structure for _step_search which expects {"query_obj": ..., "question": ...}
        search_input = {"query_obj": query_obj, "question": q}
        results = pipeline._step_search(search_input)
        
        # Check result structure based on strategy
        search_res = results
        
        if strategy == SearchStrategy.COUNT:
            count_val = search_res.get("result")
            print(f"   -> Count Result: {count_val}")
            if isinstance(count_val, int):
                 print(f"   ✅ Search Success (Got Integer)")
            else:
                 print(f"   ❌ Search Failed (Expected Integer)")

        elif strategy == SearchStrategy.LIST:
            seed_res = search_res.get("seed_results", [])
            print(f"   -> List Result: Found {len(seed_res)} items")
            if len(seed_res) > 0:
                 print(f"   ✅ Search Success (Got List)")
            else:
                 print(f"   ⚠️ Search Warning (Got Empty List)")

        elif strategy == SearchStrategy.VERIFY:
            seed_res = search_res.get("seed_results", [])
            print(f"   -> Verify Result: Found {len(seed_res)} matching nodes")
            # For verification, finding 0 nodes is also a valid result (meaning "No")
            print(f"   ✅ Search Success (Executed)")

        elif strategy == SearchStrategy.RETRIEVE:
            seed_res = search_res.get("seed_results", [])
            print(f"   -> Retrieve Result: Found {len(seed_res)} candidates")
            if len(seed_res) > 0:
                 print(f"   ✅ Search Success (Got Candidates)")

if __name__ == "__main__":
    asyncio.run(test_intent_router())
