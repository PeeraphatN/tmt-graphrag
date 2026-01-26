import sys
import os
import asyncio
import time

# Add graphrag root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import GraphRAGPipeline
from src.schemas.query import SearchStrategy
from src.utils.test_logger import TestLogger

async def test_intent_router():
    print("=== Testing Intent Router & Specialized Search with Evidence Logging ===")
    
    # Initialize Pipeline
    pipeline = GraphRAGPipeline()
    
    # Initialize Logger
    logger = TestLogger(test_name="intent_router_verification")
    
    test_cases = [
        {
            "input": "มียากี่ตัวในบัญชี ง",
            "expected_strategy": SearchStrategy.COUNT,
            "expected_result_type": int,
            "description": "Count NLEM Category Ngor"
        },
        {
            "input": "ขอรายชื่อยาของ SUN PHARMACEUTICAL",
            "expected_strategy": SearchStrategy.LIST,
            "expected_result_type": list,
            "description": "List drugs by Manufacturer"
        },
        {
            "input": "ยา Alendronate sodium อยู่ในบัญชีไหม",
            "expected_strategy": SearchStrategy.VERIFY,
            "expected_result_type": list,
            "description": "Verify drug existence (Trade Name)"
        },
        {
            "input": "ยา Paracetamol สรรพคุณคืออะไร",
            "expected_strategy": SearchStrategy.RETRIEVE,
            "expected_result_type": list, 
            "description": "General Retrieval (Properties)"
        }
    ]

    for case in test_cases:
        query_text = case["input"]
        print(f"\n🧪 Testing Input: '{query_text}'")
        
        start_time = time.perf_counter()
        
        try:
            # 1. Transform Step (to get strategy and query object)
            # The pipeline chain usually does this automatically, but accessing _step methods means we manage state manually.
            input_dict = {"question": query_text}
            query_obj = pipeline._step_transform(input_dict)
            
            # 2. Search Step
            # _step_search expects the state from the previous step (which contains query_obj)
            # We mock the state dictionary that the chain would pass
            search_input = {
                "question": query_text,
                "query_obj": query_obj
            }
            results = pipeline._step_search(search_input)
            
            # 3. Extract strategy and result
            # Note: _step_search returns a dict, likely containing 'strategy' and data
            actual_strategy_str = results.get("strategy", "unknown")
            actual_result_summary = "None"
            status = "FAIL"
            note = ""

            # Check Strategy Match
            # Use lower() for safe comparison if needed, or direct if enum value matches string
            is_strategy_match = False
            if str(actual_strategy_str).lower() == str(case["expected_strategy"].value).lower():
                is_strategy_match = True
            
            # Check Result Validity based on strategy
            if case["expected_strategy"] == SearchStrategy.COUNT:
                raw_res = results.get("result")
                actual_result_summary = f"Count={raw_res}"
                if isinstance(raw_res, int):
                    status = "PASS"
                else:
                    note = f"Type Mismatch: Expected int, got {type(raw_res)}"
            
            elif case["expected_strategy"] in [SearchStrategy.LIST, SearchStrategy.VERIFY, SearchStrategy.RETRIEVE]:
                # List-based results in 'seed_results'
                seeds = results.get("seed_results", [])
                actual_result_summary = f"Items={len(seeds)}"
                if isinstance(seeds, list):
                    status = "PASS"
                    # For Verify, 0 items might be valid "Not Found", but the search itself worked.
                    # For Retrieve/List, usually expect > 0, but 0 is technically a valid list type return.
                else:
                    note = f"Type Mismatch: Expected list, got {type(seeds)}"

            # Strategy mismatch overrides pass
            if not is_strategy_match:
                status = "FAIL"
                note = f"Strategy Mismatch: Expected {case['expected_strategy'].value}, Got {actual_strategy_str}. " + note

            end_time = time.perf_counter()
            latency = (end_time - start_time) * 1000

            # Log Evidence
            logger.log(
                query=query_text,
                intent=str(actual_strategy_str),
                status=status,
                latency_ms=latency,
                actual_result=actual_result_summary,
                expected_criteria=f"Strategy={case['expected_strategy'].value}",
                metadata={"debug_note": note}
            )

        except Exception as e:
            end_time = time.perf_counter()
            latency = (end_time - start_time) * 1000
            print(f"❌ Exception: {e}")
            logger.log(
                query=query_text,
                intent="ERROR",
                status="FAIL",
                latency_ms=latency,
                actual_result=str(e),
                expected_criteria="No Exception",
                metadata={"exception_trace": str(e)}
            )

    logger.summary()
    print(f"\n📄 Evidence log saved to: {logger.filepath}")

if __name__ == "__main__":
    asyncio.run(test_intent_router())
