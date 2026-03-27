import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_FILE = PROJECT_ROOT / "logs" / "ragas_data.jsonl"

try:
    with LOG_FILE.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()
        print(f"Total lines: {len(lines)}")
        if lines:
            print("Last line sample:")
            print(json.loads(lines[-1])["question"])
            print("Successfully parsed JSON.")
        else:
            print("File is empty.")
except Exception as exc:
    print(f"Error reading file: {exc}")
