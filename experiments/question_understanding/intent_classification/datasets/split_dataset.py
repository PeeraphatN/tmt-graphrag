
import json
import os
import random
from sklearn.model_selection import train_test_split

INPUT_FILE = "experiments/intent_benchmarks/intent_gold_samples.json"
TRAIN_FILE = "experiments/intent_benchmarks/benchmark_train.json"
TEST_FILE = "experiments/intent_benchmarks/benchmark_test.json"

# Fixed seed for reproducibility
random.seed(42)

def split_dataset():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    train_set = {}
    test_set = {}
    
    total_samples = 0
    test_count = 0
    
    # Stratified Split (Hold out 25% for Testing ~5 samples per class)
    for label, samples in data.items():
        # Shuffle to ensure randomness before split
        random.shuffle(samples)
        
        # Split
        tr, te = train_test_split(samples, test_size=0.25, random_state=42, shuffle=True)
        
        train_set[label] = tr
        test_set[label] = te
        
        total_samples += len(samples)
        test_count += len(te)
        
    # Save
    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        json.dump(train_set, f, ensure_ascii=False, indent=2)
        
    with open(TEST_FILE, "w", encoding="utf-8") as f:
        json.dump(test_set, f, ensure_ascii=False, indent=2)

    print(f"✅ Split Completed:")
    print(f"   Total Source: {total_samples}")
    print(f"   Train (Support): {total_samples - test_count} (Saved to {TRAIN_FILE})")
    print(f"   Test (Evaluation): {test_count} (Saved to {TEST_FILE})")

if __name__ == "__main__":
    split_dataset()
