import json
import time
import ollama
import numpy as np
import os
from sklearn.metrics import accuracy_score, classification_report, f1_score, matthews_corrcoef, confusion_matrix, precision_score, recall_score
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
    "manufacturer_find": "Find specific manufacturer, distributor, or copyright holder of a drug.",
    "manufacturer_check": "Check verification of manufacturer, country of origin, or production standards.",
    "manufacturer_count": "Count products, brands, or factories related to a manufacturer.",
    "ingredient_find": "Find active ingredients, components, formula details, or chemical names.",
    "ingredient_check": "Check if a drug contains specific substances (e.g. alcohol, steroids, peanut).",
    "ingredient_count": "Count number of ingredients, substances, or active components.",
    "nlem_find": "Find National List of Essential Medicines (NLEM) category, reimbursement type, or rights.",
    "nlem_check": "Check reimbursement eligibility (Gold Card, Civil Servant) or NLEM status.",
    "nlem_count": "Count drugs in specific NLEM categories or reimbursement lists.",
    "hierarchy_find": "Find generic name, drug relationships (parent/child), or TMT hierarchy levels.",
    "hierarchy_check": "Check if drug is Generic/Original, or verification of TMT level (GP/TP/VTM).",
    "hierarchy_count": "Count trade names, generic substitutions, or hierarchy nodes.",
    "formula_find": "Find dosage form, strength, concentration, or packaging details.",
    "formula_check": "Check specific characteristics (tablet/syrup, sugar-free, extended release).",
    "formula_count": "Count available strengths, dosage forms, or package sizes.",
    "general_find": "Find general info, TMT ID, update date, or official name.",
    "general_check": "Check TMT ID existence, data validity, or status in system.",
    "general_count": "Count total records, updates, or items in the database."
}

class Logger:
    def __init__(self):
        self.logs = []
    
    def log(self, message=""):
        print(message)
        self.logs.append(message)
        
    def get_content(self):
        return "\n".join(self.logs)

def save_results_to_file(content):
    """Save benchmark results to a timestamped file."""
    results_dir = "experiments/intent_benchmarks/results"
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_llm_{timestamp}.txt"
    filepath = os.path.join(results_dir, filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"\n✅ Results saved to: {filepath}")


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
        prompt_tokens = response.get('prompt_eval_count') or 0
        eval_tokens = response.get('eval_count') or 0
        
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
    logger = Logger()
    logger.log("=== LLM PROMPTING BENCHMARK (Zero-shot vs Few-shot) ===")
    
    # 1. Load Data
    train_data = load_data(TRAIN_PATH)
    test_data = load_data(TEST_PATH)
    logger.log(f"Train: {len(train_data)} | Test: {len(test_data)}")
    
    # Prepare Few-shot examples
    few_shot_ex = get_few_shot_examples(train_data, n_shots=2)

    results_log = []

    for model in MODELS:
        logger.log(f"\n{'='*50}")
        logger.log(f" 🤖 Evaluation: {model}")
        logger.log(f"{'='*50}")

        # Fetch Model Info
        model_info = get_model_info(model)
        ctx_window = model_info['context_window']
        logger.log(f"   ℹ️ Context Window: {ctx_window}")

        for mode in ["zero-shot", "few-shot"]:
            logger.log(f"\n   📍 Mode: {mode.upper()}")
            
            y_true = []
            y_pred = []
            latencies = []
            prompt_token_counts = []
            parse_errors = 0

            for i, sample in enumerate(test_data):
                if i % 10 == 0: 
                    # We still print progress to console directly for better UX (don't log this)
                    print(f"      Processing {i}/{len(test_data)}...", end="\r")
                
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
            
            logger.log(f"      Processing {len(test_data)}/{len(test_data)}... Done.")

            # Calculate Metrics
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
            cm = confusion_matrix(y_true, y_pred)
            
            avg_lat = np.mean(latencies)
            avg_prompt_tokens = np.mean(prompt_token_counts)
            
            logger.log(f"      [RESULT] Acc: {acc:.4f} | F1: {f1:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f}")
            logger.log(f"      Latency: {avg_lat:.2f}ms | Avg Prompt Tok: {avg_prompt_tokens:.1f} | ParseErr: {parse_errors}")
            logger.log(f"      Confusion Matrix:\n{cm}")
            
            results_log.append({
                "model": model,
                "mode": mode,
                "context_window": ctx_window,
                "accuracy": acc,
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "cm": cm,
                "latency": avg_lat,
                "avg_prompt_tokens": avg_prompt_tokens,
                "parse_error": parse_errors
            })

    # Summary Table
    logger.log(f"\n{'='*100}")
    logger.log(f" 📊 FINAL LLM BENCHMARK SUMMARY")
    logger.log(f"{'='*100}")
    logger.log(f"{'Model':<20} {'Mode':<10} {'CtxWin':<10} | {'Acc':<8} {'F1':<8} {'Prec':<8} {'Rec':<8} {'Lat(ms)':<8}")
    logger.log("-" * 100)
    
    for r in results_log:
        logger.log(f"{r['model']:<20} {r['mode']:<10} {str(r['context_window']):<10} | {r['accuracy']:<8.4f} {r['f1']:<8.4f} {r['precision']:<8.4f} {r['recall']:<8.4f} {r['latency']:<8.2f}")

    save_results_to_file(logger.get_content())

if __name__ == "__main__":
    run_benchmark()
