"""Search for NLEM drugs in TMT Neo4j database"""
from dotenv import load_dotenv
from neo4j import GraphDatabase
import os

load_dotenv()

uri = os.getenv('NEO4J_URI')
user = os.getenv('NEO4J_USER')
password = os.getenv('NEO4J_PASSWORD')

driver = GraphDatabase.driver(uri, auth=(user, password))

# Drugs from NLEM page (6.6 Drugs affecting bone metabolism)
drugs = ['Alendronate', 'alendronic', 'Calcitonin', 'Pamidronate', 'pamidronic', 'Zoledronic']

output_lines = []
output_lines.append(f"Connecting to {uri}...\n")

with driver.session() as session:
    for drug in drugs:
        output_lines.append(f"\n=== {drug} ===")
        result = session.run('''
            MATCH (n:GP)
            WHERE toLower(n.fsn) CONTAINS toLower($drug)
            RETURN n.tmtid as tmtid, n.fsn as fsn
            LIMIT 20
        ''', drug=drug)
        records = list(result)
        if records:
            for r in records:
                fsn = r['fsn'][:90] + '...' if len(r['fsn']) > 90 else r['fsn']
                output_lines.append(f"  {r['tmtid']}: {fsn}")
        else:
            output_lines.append('  (No GP found)')

driver.close()

output_lines.append(f"\n=== Done ===")

# Write to file
with open('nlem_search_results.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(output_lines))

print("Results written to nlem_search_results.txt")
