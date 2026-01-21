import json
import os
import sys
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from langchain_ollama import ChatOllama, OllamaEmbeddings

# Add parent directory to path to import config if needed
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.config import OLLAMA_URL, LLM_MODEL, EMBED_MODEL

LOG_FILE = "./logs/ragas_data.jsonl"
RESULT_FILE = "./eval_results.csv"

def load_data():
    """Load logs from JSONL and convert to Ragas Dataset format."""
    if not os.path.exists(LOG_FILE):
        print(f"❌ Log file not found at {LOG_FILE}")
        return None

    questions = []
    answers = []
    contexts = []

    print(f"📂 Loading logs from {LOG_FILE}...")
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                questions.append(record.get("question", ""))
                answers.append(record.get("answer", ""))
                # contexts must be list[str]
                ctx = record.get("contexts", [])
                contexts.append(ctx if isinstance(ctx, list) else [])
            except json.JSONDecodeError:
                continue
    
    if not questions:
        print("❌ No data found in logs.")
        return None

    data_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        # Ground truth is optional for Faithfulness/Relevance
        # "ground_truth": [[""]] * len(questions) 
    }
    
    return Dataset.from_dict(data_dict)

def main():
    dataset = load_data()
    if dataset is None:
        return

    print("🤖 Configuring Local Models for Evaluation...")
    print(f"   Judge LLM: {LLM_MODEL} (via {OLLAMA_URL})")
    print(f"   Embeddings: {EMBED_MODEL}")

    # Initialize Local Models
    # Note: Ragas uses these to generate the critique
    llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_URL, temperature=0)
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)

    print("🚀 Starting Evaluation (Faithfulness & Answer Relevance)...")
    print("   This may take a while depending on your GPU...")

    try:
        # Run Evaluation
        results = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=llm,
            embeddings=embeddings,
        )

        print("\n✅ Evaluation Complete!")
        print(results)

        # Save results
        df = results.to_pandas()
        df.to_csv(RESULT_FILE, index=False)
        print(f"\n💾 Results saved to {RESULT_FILE}")

        # Display average scores
        print("\n📊 Average Scores:")
        print(df[["faithfulness", "answer_relevance"]].mean())

    except Exception as e:
        print(f"\n❌ Evaluation Failed: {e}")
        print("Tip: Ensure 'ragas' and 'datasets' are installed: pip install ragas datasets pandas")

if __name__ == "__main__":
    main()
