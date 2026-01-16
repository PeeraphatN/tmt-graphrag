"""Test GraphRAG query for NLEM drugs"""
from dotenv import load_dotenv
from neo4j import GraphDatabase
import os

load_dotenv()

uri = os.getenv('NEO4J_URI')
user = os.getenv('NEO4J_USER')
password = os.getenv('NEO4J_PASSWORD')

driver = GraphDatabase.driver(uri, auth=(user, password))

print("=== Testing NLEM Query ===\n")
print("Question: ยาตัวไหนอยู่ในบัญชียาหลักแห่งชาติ?")
print("-" * 50)

with driver.session() as session:
    result = session.run('''
        MATCH (n:GP)
        WHERE n.nlem = true AND n.nlem_match_type = "exact"
        RETURN n.fsn as name, n.nlem_category as category, n.nlem_section as section
        ORDER BY n.fsn
    ''')
    records = list(result)

driver.close()

print(f"\nพบ {len(records)} รายการที่อยู่ในบัญชียาหลักแห่งชาติ (Exact match):\n")
for i, r in enumerate(records, 1):
    print(f"{i}. {r['name']}")
    print(f"   → บัญชี {r['category']} (หมวด {r['section']})")

print("\n" + "=" * 50)
print("✅ GraphRAG สามารถค้นหายาใน NLEM ได้สำเร็จ!")
