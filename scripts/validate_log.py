
import json

LOG_FILE = "./logs/ragas_data.jsonl"

try:
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
        print(f"Total lines: {len(lines)}")
        if lines:
            print("Last line sample:")
            print(json.loads(lines[-1])["question"])
            print("Successfully parsed JSON.")
        else:
            print("File is empty.")
except Exception as e:
    print(f"Error reading file: {e}")
