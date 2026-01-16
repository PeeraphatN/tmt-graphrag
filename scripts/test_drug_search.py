"""Test: Search by drug name, check if NLEM info comes back"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.config import validate_env
from src.knowledge_graph import init_driver, close_driver, graphrag_search

validate_env()
init_driver()

# Search for a specific NLEM drug by name
q = "alendronic acid 70 mg"
print(f"🔍 Searching: {q}")
print("-" * 50)

results = graphrag_search(q, k=5, depth=1)
seed_results = results.get("seed_results", [])

for item in seed_results:
    props = dict(item["node"])
    print(f"\n📦 {props.get('fsn', 'N/A')}")
    print(f"   TMTID: {props.get('tmtid')}")
    print(f"   NLEM: {props.get('nlem', 'Not set')}")
    print(f"   Category: {props.get('nlem_category', 'N/A')}")
    # Show embedding_text suffix
    emb = props.get('embedding_text', '')
    if emb and 'บัญชี' in emb:
        print(f"   ✅ Embedding has NLEM info!")

close_driver()
