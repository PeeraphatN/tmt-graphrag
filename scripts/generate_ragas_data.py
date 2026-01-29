
import sys
import os
import json
from neo4j import GraphDatabase

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import GraphRAGPipeline
from src.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

LOG_FILE = "./logs/ragas_data.jsonl"

def get_control_questions_from_db():
    print("🔌 Connecting to Neo4j to fetch Ground Truths...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    questions = []

    with driver.session() as session:
        # 1. NLEM Questions
        # Fetch 3 distinct NLEM items
        print("   Fetch NLEM samples...")
        result = session.run("""
            MATCH (n:TMT) 
            WHERE n.nlem = true AND n.fsn IS NOT NULL AND n.nlem_category IS NOT NULL
            RETURN n.fsn AS name, n.nlem_category AS cat
            LIMIT 3
        """)
        for r in result:
            name = r["name"]
            cat = r["cat"]
            questions.append({
                "question": f"ยา {name} อยู่ในบัญชียาหลักหรือไม่",
                "ground_truth": f"ใช่ ยา {name} อยู่ในบัญชียาหลักแห่งชาติ หมวดยา {cat}"
            })

        # 2. Manufacturer Questions
        # Fetch 3 distinct items with known manufacturer
        print("   Fetch Manufacturer samples...")
        result = session.run("""
            MATCH (n:TMT) 
            WHERE n.manufacturer IS NOT NULL AND n.trade_name IS NOT NULL
            RETURN n.trade_name AS trade, n.manufacturer AS manu
            LIMIT 3
        """)
        for r in result:
            trade = r["trade"]
            manu = r["manu"]
            questions.append({
                "question": f"ใครผลิตยา {trade}",
                "ground_truth": f"ยา {trade} ผลิตโดยบริษัท {manu}"
            })

        # 3. Ingredient Questions (Using CONTAINS or similar if available, else simple lookup)
        # Fallback: Just ask about properties or use embedding text logic to infer content
        # Since HAS_INGREDIENT is missing, we rely on the node properties (usually embedded in FSN or properties)
        # Let's try to ask about TMTID which is definite
        print("   Fetch TMTID samples...")
        result = session.run("""
            MATCH (n:TMT) 
            WHERE n.tmtid IS NOT NULL AND n.fsn IS NOT NULL
            RETURN n.fsn AS name, n.tmtid AS id
            LIMIT 2
        """)
        for r in result:
            name = r["name"]
            tid = r["id"]
            questions.append({
                "question": f"ขอ TMTID ของยา {name}",
                "ground_truth": f"TMTID ของยา {name} คือ {tid}"
            })

    driver.close()
    return questions

def main():
    print("=== Generating RAGAS Data (Dynamic Ground Truths) ===")
    
    # 1. Fetch Questions dynamically
    try:
        control_questions = get_control_questions_from_db()
        print(f"✅ Loaded {len(control_questions)} questions from DB.")
    except Exception as e:
        print(f"❌ Failed to fetch from DB: {e}")
        return

    # 2. Clear old logs
    if os.path.exists(LOG_FILE):
        print(f"🧹 Clearing old log file: {LOG_FILE}")
        os.remove(LOG_FILE)
    
    # 3. Initialize Pipeline
    print("🔧 Initializing Pipeline...")
    pipeline = GraphRAGPipeline()
    # pipeline.warmup()

    # 4. Process Questions
    print(f"\n🚀 Running {len(control_questions)} control questions...")
    
    for i, item in enumerate(control_questions):
        q = item["question"]
        gt = item["ground_truth"]
        
        print(f"\n[{i+1}/{len(control_questions)}] Asking: {q}")
        try:
            # Pipeline run automatically logs to ragas_data.jsonl
            pipeline.run(question=q, ground_truth=gt)
            print("   ✅ Complete.")
        except Exception as e:
            print(f"   ❌ Failed: {e}")

    print(f"\n✨ Generation Complete! Data saved to {LOG_FILE}")

if __name__ == "__main__":
    main()
