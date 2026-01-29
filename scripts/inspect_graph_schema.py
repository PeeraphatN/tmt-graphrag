
import os
import sys
from neo4j import GraphDatabase

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

def inspect():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    queries = {
        "Node Labels": "CALL db.labels()",
        "Relationship Types": "CALL db.relationshipTypes()",
        "TMT Node Properties (Sample)": """
            MATCH (n) 
            WHERE n.fsn IS NOT NULL 
            RETURN keys(n) AS props LIMIT 1
        """,
        "NLEM Sample": """
            MATCH (n) 
            WHERE n.nlem = true 
            RETURN n.fsn, n.nlem, n.nlem_category LIMIT 1
        """,
        "Manufacturer Sample": """
            MATCH (n) 
            WHERE n.manufacturer IS NOT NULL 
            RETURN n.manufacturer LIMIT 5
        """,
        "Ingredient Sample": """
            MATCH (n)-[r:HAS_INGREDIENT]->(i)
            RETURN n.fsn, i.fsn LIMIT 1
        """
    }

    with driver.session() as session:
        for title, q in queries.items():
            print(f"\\n=== {title} ===")
            try:
                result = session.run(q)
                for record in result:
                    print(record)
            except Exception as e:
                print(f"Error: {e}")

    driver.close()

if __name__ == "__main__":
    inspect()
