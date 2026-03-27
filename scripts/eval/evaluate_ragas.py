import json
import sys
from pathlib import Path

from datasets import Dataset
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.config import EMBED_MODEL, LLM_MODEL, OLLAMA_URL

LOG_FILE = PROJECT_ROOT / "logs" / "ragas_data.jsonl"
RESULT_FILE = PROJECT_ROOT / "logs" / "ragas_eval.csv"


def load_data():
    """Load logs from JSONL and convert to a RAGAS Dataset."""
    if not LOG_FILE.exists():
        print(f"Log file not found at {LOG_FILE}")
        return None

    questions = []
    answers = []
    contexts = []

    print(f"Loading logs from {LOG_FILE}...")
    with LOG_FILE.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            questions.append(record.get("question", ""))
            answers.append(record.get("answer", ""))
            ctx = record.get("contexts", [])
            contexts.append(ctx if isinstance(ctx, list) else [])

    if not questions:
        print("No data found in logs.")
        return None

    return Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }
    )


def main():
    dataset = load_data()
    if dataset is None:
        return

    print("Configuring local models for evaluation...")
    print(f"  Judge LLM: {LLM_MODEL} (via {OLLAMA_URL})")
    print(f"  Embeddings: {EMBED_MODEL}")

    llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_URL, temperature=0)
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)

    print("Starting evaluation (faithfulness + answer relevance)...")
    try:
        results = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=llm,
            embeddings=embeddings,
        )
        print("\nEvaluation complete.")
        print(results)

        df = results.to_pandas()
        RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(RESULT_FILE, index=False)
        print(f"\nResults saved to {RESULT_FILE}")
        print("\nAverage scores:")
        print(df[["faithfulness", "answer_relevance"]].mean())
    except Exception as exc:
        print(f"\nEvaluation failed: {exc}")
        print("Tip: ensure ragas, datasets, and pandas are installed.")


if __name__ == "__main__":
    main()
