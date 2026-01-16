"""Verify NLEM properties were added correctly"""
from dotenv import load_dotenv
from neo4j import GraphDatabase
import os

load_dotenv()

uri = os.getenv('NEO4J_URI')
user = os.getenv('NEO4J_USER')
password = os.getenv('NEO4J_PASSWORD')

driver = GraphDatabase.driver(uri, auth=(user, password))

with driver.session() as session:
    result = session.run('''
        MATCH (n:GP)
        WHERE n.nlem = true
        RETURN n.tmtid as tmtid, n.fsn as fsn, 
               n.nlem_category as category, 
               n.nlem_match_type as match_type
        ORDER BY n.nlem_match_type, n.tmtid
    ''')
    records = list(result)

driver.close()

# Write to file
with open('nlem_verify_results.txt', 'w', encoding='utf-8') as f:
    f.write(f"=== GP Nodes with NLEM property ===\n")
    f.write(f"Total: {len(records)} nodes\n\n")
    
    current_type = None
    for r in records:
        if r['match_type'] != current_type:
            current_type = r['match_type']
            f.write(f"\n--- {current_type.upper()} ---\n")
        f.write(f"{r['tmtid']}: {r['fsn'][:70]}... [บัญชี {r['category']}]\n")

print(f"Found {len(records)} GP nodes with NLEM property. See nlem_verify_results.txt")
