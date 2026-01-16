"""Check embedding_text was updated correctly"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from neo4j import GraphDatabase
load_dotenv()

driver = GraphDatabase.driver(
    os.getenv('NEO4J_URI'),
    auth=(os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD'))
)

with driver.session() as session:
    result = session.run('''
        MATCH (n:GP)
        WHERE n.nlem = true
        RETURN n.tmtid as tmtid, n.fsn as fsn, n.embedding_text as emb_text
        LIMIT 5
    ''')
    for r in result:
        print(f"TMTID: {r['tmtid']}")
        print(f"  FSN: {r['fsn'][:50]}")
        print(f"  EMB: {r['emb_text'][-60:] if r['emb_text'] else 'NULL'}")
        print()

driver.close()
