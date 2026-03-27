import sys
from pathlib import Path

from neo4j import GraphDatabase

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER
from src.pipeline import GraphRAGPipeline

LOG_FILE = PROJECT_ROOT / "logs" / "ragas_data.jsonl"


def get_control_questions_from_db():
    print("Connecting to Neo4j to fetch ground truths...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    questions = []

    with driver.session() as session:
        print("  Fetch NLEM samples...")
        result = session.run(
            """
            MATCH (n:TMT)
            WHERE n.nlem = true AND n.fsn IS NOT NULL AND n.nlem_category IS NOT NULL
            RETURN n.fsn AS name, n.nlem_category AS cat
            LIMIT 3
            """
        )
        for record in result:
            name = record["name"]
            cat = record["cat"]
            questions.append(
                {
                    "question": f"ยา {name} อยู่ในบัญชียาหลักหรือไม่",
                    "ground_truth": f"ใช่ ยา {name} อยู่ในบัญชียาหลักแห่งชาติ หมวดยา {cat}",
                }
            )

        print("  Fetch manufacturer samples...")
        result = session.run(
            """
            MATCH (n:TMT)
            WHERE n.manufacturer IS NOT NULL AND n.trade_name IS NOT NULL
            RETURN n.trade_name AS trade, n.manufacturer AS manu
            LIMIT 3
            """
        )
        for record in result:
            trade = record["trade"]
            manu = record["manu"]
            questions.append(
                {
                    "question": f"ใครผลิตยา {trade}",
                    "ground_truth": f"ยา {trade} ผลิตโดยบริษัท {manu}",
                }
            )

        print("  Fetch TMTID samples...")
        result = session.run(
            """
            MATCH (n:TMT)
            WHERE n.tmtid IS NOT NULL AND n.fsn IS NOT NULL
            RETURN n.fsn AS name, n.tmtid AS id
            LIMIT 2
            """
        )
        for record in result:
            name = record["name"]
            tmtid = record["id"]
            questions.append(
                {
                    "question": f"ขอ TMTID ของยา {name}",
                    "ground_truth": f"TMTID ของยา {name} คือ {tmtid}",
                }
            )

    driver.close()
    return questions


def main():
    print("=== Generating RAGAS Data (Dynamic Ground Truths) ===")

    try:
        control_questions = get_control_questions_from_db()
        print(f"Loaded {len(control_questions)} questions from DB.")
    except Exception as exc:
        print(f"Failed to fetch from DB: {exc}")
        return

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    if LOG_FILE.exists():
        print(f"Clearing old log file: {LOG_FILE}")
        LOG_FILE.unlink()

    print("Initializing pipeline...")
    pipeline = GraphRAGPipeline()

    print(f"\nRunning {len(control_questions)} control questions...")
    for index, item in enumerate(control_questions, start=1):
        question = item["question"]
        ground_truth = item["ground_truth"]
        print(f"\n[{index}/{len(control_questions)}] Asking: {question}")
        try:
            pipeline.run(question=question, ground_truth=ground_truth)
            print("  Complete.")
        except Exception as exc:
            print(f"  Failed: {exc}")

    print(f"\nGeneration complete. Data saved to {LOG_FILE}")


if __name__ == "__main__":
    main()
