import os
import json
import time
from datetime import datetime
from typing import Any, Dict, Optional

class TestLogger:
    """
    Standardized logger for producing empirical evidence (JSONL logs) from test runs.
    """
    def __init__(self, test_name: str, log_dir: str = "test_results"):
        self.test_name = test_name
        self.log_dir = log_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"{self.timestamp}_{test_name}.jsonl"
        self.filepath = os.path.join(self.log_dir, self.filename)
        self.results = []
        
        # Ensure log directory exists
        os.makedirs(self.log_dir, exist_ok=True)
        print(f"📝 Test Logger Initialized: {self.filepath}")

    def log(self, 
            query: str, 
            intent: str, 
            status: str, 
            latency_ms: float, 
            actual_result: Any, 
            expected_criteria: str = None, 
            metadata: Dict = None):
        """
        Log a single test case result.
        
        Args:
            query: The input question.
            intent: Detected intent (e.g., RETRIEVE, COUNT).
            status: "PASS" or "FAIL".
            latency_ms: Time taken in milliseconds.
            actual_result: Summary of the result (e.g., count=22, or top result name).
            expected_criteria: Description of what was expected (e.g., "Count > 0").
            metadata: Any extra debug info (filters used, strategy detected, etc.).
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "test_run_id": self.timestamp,
            "query": query,
            "intent": intent,
            "status": status,
            "latency_ms": round(latency_ms, 2),
            "result": str(actual_result),
            "expected": expected_criteria,
            "metadata": metadata or {}
        }
        
        # Write to memory
        self.results.append(entry)
        
        # Write to file (Append mode)
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
        # Console output
        icon = "✅" if status == "PASS" else "❌"
        print(f"   {icon} [{status}] Latency: {entry['latency_ms']}ms | Evidence: {entry['filepath'] if 'filepath' in entry else 'See Log'}")

    def summary(self):
        """
        Print final summary stats.
        """
        total = len(self.results)
        passed = sum(1 for r in self.results if r['status'] == 'PASS')
        failed = total - passed
        print("\n" + "="*40)
        print(f"📊 TEST SUMMARY: {self.test_name}")
        print(f"   Total : {total}")
        print(f"   Pass  : {passed}")
        print(f"   Fail  : {failed}")
        print(f"   Log   : {self.filepath}")
        print("="*40 + "\n")
