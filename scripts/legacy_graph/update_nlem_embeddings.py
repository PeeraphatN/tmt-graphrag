"""
Update embedding_text and re-embed GP nodes that have NLEM properties.
This simulates what the NLEM enrichment pipeline should do after adding properties.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.embeddings import embed_text

load_dotenv()

uri = os.getenv("NEO4J_URI")
user = os.getenv("NEO4J_USER")
password = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(uri, auth=(user, password))

print("=== Updating embedding_text for NLEM nodes ===\n")

with driver.session() as session:
    result = session.run(
        """
        MATCH (n:GP)
        WHERE n.nlem = true
        RETURN n.tmtid as tmtid, n.fsn as fsn, n.embedding_text as old_text,
               n.nlem_category as category, n.nlem_section as section,
               n.nlem_name as nlem_name, n.nlem_match_type as match_type
        """
    )
    nodes = list(result)

    print(f"Found {len(nodes)} NLEM nodes to update\n")

    updated_count = 0
    for node in nodes:
        tmtid = node["tmtid"]
        fsn = node["fsn"]
        old_text = node["old_text"] or fsn
        category = node["category"]
        section = node["section"]
        nlem_name = node["nlem_name"]

        nlem_suffix = f" | บัญชียาหลักแห่งชาติ บัญชี {category} หมวด {section} {nlem_name}"
        new_text = old_text + nlem_suffix

        print(f"  Embedding {tmtid}: {fsn[:40]}...")
        new_embedding = embed_text(new_text)

        if new_embedding:
            session.run(
                """
                MATCH (n:GP {tmtid: $tmtid})
                SET n.embedding_text = $new_text,
                    n.embedding = $embedding
                """,
                tmtid=tmtid,
                new_text=new_text,
                embedding=new_embedding,
            )
            updated_count += 1
            print("    Updated")
        else:
            print("    Embedding failed")

driver.close()

print(f"\n=== Done! Updated {updated_count}/{len(nodes)} nodes ===")
print("Now RAG should find NLEM drugs when searching for 'บัญชียาหลักแห่งชาติ'")
