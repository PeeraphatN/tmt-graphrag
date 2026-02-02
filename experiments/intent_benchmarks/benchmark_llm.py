import json
import time
import ollama
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score, matthews_corrcoef
import re
import random

# Configuration
TRAIN_PATH = "experiments/intent_benchmarks/benchmark_train.json"
TEST_PATH = "experiments/intent_benchmarks/benchmark_test.json"

# Models to Benchmark (Generative)
MODELS = [
    "llama3.2:1b",
    "deepseek-r1:1.5b",
    "qwen2.5:3b",
    "gemma3:4b",
    "phi4-mini:latest",
    "llama3",           # 8B
    "qwen2.5:7b-instruct",
    "qwen3:latest",
]

# Intent Definitions for Zero-shot
INTENTS = {
    "manufacturer": "Questions about the drug company, brand owner, or producer.",
    "ingredient": "Questions about active substances, chemical composition, or drug components.",
    "nlem": "Questions about National List of Essential Medicines, reimbursement, or rights (Gold Card, Civil Servant).",
    "hierarchy": "Questions about drug levels (Generic vs Trade), or relationships between drugs.",
    "formula": "Questions about dosage form (tablet, syrup, cream) or packaging.",
    "general": "General questions about TMT code, trade name search, or unspecified drug info."
}

def load_data(filepath):
    """Load dataset from JSON."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    samples = []
    for label, sentences in data.items():
        for sent in sentences:
            samples.append({"text": sent, "label": label})
    return samples

def get_few_shot_examples(train_data, n_shots=1):
    """Randomly select n examples per class from training data."""
    examples = {}
    random.shuffle(train_data)
    for item in train_data:
        label = item['label']
        if label not in examples:
            examples[label] = []
        if len(examples[label]) < n_shots:
            examples[label].append(item['text'])
    return examples

def construct_prompt(query, mode="zero-shot", few_shot_examples=None):
    """Constructs the prompt for the LLM."""
    
    intent_desc = "\n".join([f"- {k}: {v}" for k, v in INTENTS.items()])
    
    prompt = f"""You are a medical intent classifier for the Thai Medicines Terminology (TMT) system.
Classify the given query into exactly one of the following intents:

{intent_desc}

"""

    if mode == "few-shot" and few_shot_examples:
        prompt += "Examples:\n"
        for label, texts in few_shot_examples.items():
            for text in texts:
                prompt += f"Query: \"{text}\"\nIntent: {{\"intent\": \"{label}\"}}\n"
        prompt += "\n"

    prompt += f"""Query: "{query}"

Respond strictly with valid JSON only. Format: {{"intent": "class_name"}}
Do not explain.
JSON:"""
    return prompt

def clean_json_response(response_text):
    """Extracts and parses JSON from LLM response."""
    try:
        # Try direct parse
        return json.loads(response_text)
    except:
        # Try finding JSON pattern
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
        
        # Fallback: simple string matching if JSON fails
        cleaned = response_text.lower().strip()
        for label in INTENTS.keys():
            if label in cleaned:
                return {"intent": label}
        
        return {"intent": "unknown"}

def run_prediction(model, prompt):
    """Calls Ollama generate API."""
    start_time = time.perf_counter()
    try:
        response = ollama.generate(model=model, prompt=prompt, options={"temperature": 0.0}) # Deterministic
        latency = (time.perf_counter() - start_time) * 1000 # ms
        
        # Capture usage stats directly from response object (if available)
        # Ollama python client returns dict with 'eval_count', 'prompt_eval_count' etc.
        prompt_tokens = response.get('prompt_eval_count', 0)
        eval_tokens = response.get('eval_count', 0)
        
        return response['response'], latency, prompt_tokens, eval_tokens
    except Exception as e:
        print(f"Error calling {model}: {e}")
        return "{}", 0, 0, 0

def get_model_info(model_name):
    """Fetches model metadata via Ollama Show."""
    try:
        dataset = ollama.show(model_name)
        # Parse modelfile or details for context length
        # Usually found in modelfile under PARAMETER num_ctx
        context_window = "Unknown"
        
        if 'modelfile' in dataset:
            match = re.search(r'PARAMETER num_ctx (\d+)', dataset['modelfile'])
            if match:
                context_window = int(match.group(1))
            else:
                context_window = "Default (4096)"
        
        return {"context_window": context_window}
    except Exception as e:
        return {"context_window": "Unknown"}

def run_benchmark():
    print("=== LLM PROMPTING BENCHMARK (Zero-shot vs Few-shot) ===")
    
    # 1. Load Data
    train_data = load_data(TRAIN_PATH)
    test_data = load_data(TEST_PATH)
    print(f"Train: {len(train_data)} | Test: {len(test_data)}")
    
    # Prepare Few-shot examples
    few_shot_ex = get_few_shot_examples(train_data, n_shots=2)

    results_log = []

    for model in MODELS:
        print(f"\n{'='*50}")
        print(f" 🤖 Evaluation: {model}")
        print(f"{'='*50}")

        # Fetch Model Info
        model_info = get_model_info(model)
        ctx_window = model_info['context_window']
        print(f"   ℹ️ Context Window: {ctx_window}")

        for mode in ["zero-shot", "few-shot"]:
            print(f"\n   📍 Mode: {mode.upper()}")
            
            y_true = []
            y_pred = []
            latencies = []
            prompt_token_counts = []
            parse_errors = 0

            for i, sample in enumerate(test_data):
                if i % 10 == 0: print(f"      Processing {i}/{len(test_data)}...", end="\r")
                
                query = sample['text']
                label = sample['label']
                
                prompt = construct_prompt(query, mode, few_shot_ex)
                raw_res, lat, p_tok, e_tok = run_prediction(model, prompt)
                
                parsed = clean_json_response(raw_res)
                pred = parsed.get("intent", "unknown")
                
                # Check for Valid Label
                if pred not in INTENTS:
                    parse_errors += 1
                    # Try to rescue partial matches again heavily or assign 'unknown'
                    pred = "unknown"

                y_true.append(label)
                y_pred.append(pred)
                latencies.append(lat)
                prompt_token_counts.append(p_tok)
            
            print(f"      Processing {len(test_data)}/{len(test_data)}... Done.")

            # Calculate Metrics
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            avg_lat = np.mean(latencies)
            avg_prompt_tokens = np.mean(prompt_token_counts)
            
            print(f"      [RESULT] Acc: {acc:.4f} | F1: {f1:.4f} | Latency: {avg_lat:.2f}ms | Avg Prompt Tok: {avg_prompt_tokens:.1f} | ParseErr: {parse_errors}")
            
            results_log.append({
                "model": model,
                "mode": mode,
                "context_window": ctx_window,
                "accuracy": acc,
                "f1": f1,
                "latency": avg_lat,
                "avg_prompt_tokens": avg_prompt_tokens,
                "parse_error": parse_errors
            })

    # Summary Table
    print(f"\n{'='*100}")
    print(f" 📊 FINAL LLM BENCHMARK SUMMARY")
    print(f"{'='*100}")
    print(f"{'Model':<20} {'Mode':<10} {'CtxWin':<10} | {'Acc':<8} {'F1':<8} {'Lat(ms)':<8} {'PrToks':<6} {'ParseErr':<8}")
    print("-" * 100)
    
    for r in results_log:
        print(f"{r['model']:<20} {r['mode']:<10} {str(r['context_window']):<10} | {r['accuracy']:<8.4f} {r['f1']:<8.4f} {r['latency']:<8.2f} {r['avg_prompt_tokens']:<6.1f} {r['parse_error']:<8}")

if __name__ == "__main__":
    run_benchmark()
