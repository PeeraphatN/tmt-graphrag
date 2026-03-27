"""Check whether embedding_text was updated correctly."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")),
)

with driver.session() as session:
    result = session.run(
        """
        MATCH (n:GP)
        WHERE n.nlem = true
        RETURN n.tmtid as tmtid, n.fsn as fsn, n.embedding_text as emb_text
        LIMIT 5
        """
    )
    for record in result:
        print(f"TMTID: {record['tmtid']}")
        print(f"  FSN: {record['fsn'][:50]}")
        print(f"  EMB: {record['emb_text'][-60:] if record['emb_text'] else 'NULL'}")
        print()

driver.close()
