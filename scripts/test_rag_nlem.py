"""Test RAG search for NLEM after embedding update"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.config import validate_env
from src.knowledge_graph import init_driver, close_driver, graphrag_search
import json

validate_env()
init_driver()

# Key test: Can we find NLEM drugs by searching for "บัญชียาหลัก"?
q = "ยาตัวไหนอยู่ในบัญชียาหลักแห่งชาติ"
print(f"🔍 Question: {q}")
print("-" * 50)

results = graphrag_search(q, k=10, depth=1)
seed_results = results.get("seed_results", [])

print(f"\nFound {len(seed_results)} results:\n")

# Check if we found NLEM drugs
nlem_found = 0
for item in seed_results[:10]:
    node = item["node"]
    props = dict(node)
    fsn = props.get("fsn", "N/A")[:60]
    nlem = props.get("nlem", False)
    emb_text = props.get("embedding_text", "")[:80]
    
    marker = "✅ NLEM" if nlem else "❌"
    print(f"  {marker} {fsn}")
    if "บัญชียาหลัก" in emb_text:
        nlem_found += 1

close_driver()

print(f"\n=== Results: {nlem_found}/{len(seed_results)} have NLEM in embedding_text ===")
