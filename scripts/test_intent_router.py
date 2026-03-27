import sys
import os
import asyncio
import time
import json  # Added import

# Add graphrag root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import GraphRAGPipeline
from src.schemas.query import SearchStrategy
from test_logger import TestLogger

async def test_intent_router():
    print("=== Testing Intent Router (Property-Aware + Rule-Based Logic) ===")
    print("Goal: Verify that complexity rules and property filters are correctly extracted.\n")
    
    # Initialize Pipeline (only need part of it, but simpler to init all)
    pipeline = GraphRAGPipeline()
    
    # Initialize Logger
    logger = TestLogger(test_name="intent_router_verification")
    
    test_cases = [
        # 1. Count & NLEM Logic
        {
            "input": "มียากี่ตัวในบัญชี ง",
            "expected_strategy": SearchStrategy.COUNT,
            "expected_target": "nlem",
            "description": "Count NLEM Category Ngor"
        },
        # 2. Manufacturer List Logic
        {
            "input": "ขอรายชื่อยาขององค์การเภสัชกรรม",
            "expected_strategy": SearchStrategy.LIST,
            "expected_target": "manufacturer",
            "description": "List GPO drugs"
        },
        # 3. Verify Logic (Existence Check)
        {
            "input": "ยา Paracap เบิกได้ไหม",
            "expected_strategy": SearchStrategy.VERIFY,
            "expected_target": "nlem",
            "description": "Verify Reimbursement (NLEM)"
        },
        # 4. Verify Logic (Manufacturer Check)
        {
            "input": "Does Pfizer make Viagra?",
            "expected_strategy": SearchStrategy.VERIFY,
            "expected_target": "manufacturer",
            "description": "Verify Manufacturer (English)"
        },
        # 5. General Retrieval
        {
            "input": "สรรพคุณของ Paracetamol",
            "expected_strategy": SearchStrategy.RETRIEVE,
            "expected_target": "general",
            "description": "General Info Lookup"
        },
        # 6. Substance Logic
        {
            "input": "Tiffy มีส่วนผสมอะไรบ้าง",
            "expected_strategy": SearchStrategy.RETRIEVE, # Often falls to retrieve or list, checking extraction mainly
            "expected_target": "substance",
            "description": "Substance Lookup"
        }
    ]

    print(f"{'TEST CASE':<40} | {'STRATEGY':<10} | {'TARGET':<12} | {'QUERY':<15} | {'FILTERS'}")
    print("-" * 100)

    for case in test_cases:
        query_text = case["input"]
        
        start_time = time.perf_counter()
        
        try:
            # 1. Transform Step
            input_dict = {"question": query_text}
            query_obj = pipeline._step_transform(input_dict)
            
            # Extract actual values
            actual_strategy = query_obj.strategy.value
            actual_target = query_obj.target_type.value
            actual_query = query_obj.query
            
            # Format filters for display
            filters = {}
            if query_obj.nlem_filter: filters['nlem'] = True
            if query_obj.nlem_category: filters['cat'] = query_obj.nlem_category
            if query_obj.manufacturer_filter: filters['manu'] = query_obj.manufacturer_filter
            filters_str = json.dumps(filters, ensure_ascii=False)

            # Check correctness
            strategy_match = (actual_strategy == case["expected_strategy"].value)
            target_match = (actual_target == case["expected_target"])
            
            status = "PASS" if (strategy_match and target_match) else "FAIL"
            status_icon = "✅" if status == "PASS" else "❌"

            # Print concise row
            print(f"{status_icon} {query_text[:37]:<37} | {actual_strategy:<10} | {actual_target:<12} | {actual_query:<15} | {filters_str}")

            end_time = time.perf_counter()
            latency = (end_time - start_time) * 1000

            # Log Evidence
            logger.log(
                query=query_text,
                intent=f"{actual_strategy}/{actual_target}",
                status=status,
                latency_ms=latency,
                actual_result=f"Query='{actual_query}', Filters={filters_str}",
                expected_criteria=f"Strategy={case['expected_strategy'].value}, Target={case['expected_target']}",
            )

        except Exception as e:
            print(f"❌ ERROR: {case['input']} -> {e}")

    print("\n" + "="*50)
    print(f"📄 Evidence log saved to: {logger.filepath}")

if __name__ == "__main__":
    asyncio.run(test_intent_router())
