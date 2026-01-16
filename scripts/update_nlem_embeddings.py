"""
Update embedding_text and re-embed GP nodes that have NLEM properties.
This simulates what the NLEM enrichment pipeline should do after adding properties.
"""
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

uri = os.getenv('NEO4J_URI')
user = os.getenv('NEO4J_USER')
password = os.getenv('NEO4J_PASSWORD')

driver = GraphDatabase.driver(uri, auth=(user, password))

# Import embedding function
from src.models.embeddings import embed_text

print("=== Updating embedding_text for NLEM nodes ===\n")

with driver.session() as session:
    # 1. Find all GP nodes with NLEM property
    result = session.run('''
        MATCH (n:GP)
        WHERE n.nlem = true
        RETURN n.tmtid as tmtid, n.fsn as fsn, n.embedding_text as old_text,
               n.nlem_category as category, n.nlem_section as section,
               n.nlem_name as nlem_name, n.nlem_match_type as match_type
    ''')
    nodes = list(result)
    
    print(f"Found {len(nodes)} NLEM nodes to update\n")
    
    updated_count = 0
    for node in nodes:
        tmtid = node['tmtid']
        fsn = node['fsn']
        old_text = node['old_text'] or fsn
        category = node['category']
        section = node['section']
        nlem_name = node['nlem_name']
        
        # Create enriched embedding text
        nlem_suffix = f" | บัญชียาหลักแห่งชาติ บัญชี {category} หมวด {section} {nlem_name}"
        new_text = old_text + nlem_suffix
        
        # Generate new embedding
        print(f"  Embedding {tmtid}: {fsn[:40]}...")
        new_embedding = embed_text(new_text)
        
        if new_embedding:
            # Update node
            session.run('''
                MATCH (n:GP {tmtid: $tmtid})
                SET n.embedding_text = $new_text,
                    n.embedding = $embedding
            ''', tmtid=tmtid, new_text=new_text, embedding=new_embedding)
            updated_count += 1
            print(f"    ✅ Updated")
        else:
            print(f"    ⚠️ Embedding failed")

driver.close()

print(f"\n=== Done! Updated {updated_count}/{len(nodes)} nodes ===")
print("Now RAG should find NLEM drugs when searching for 'บัญชียาหลักแห่งชาติ'")
