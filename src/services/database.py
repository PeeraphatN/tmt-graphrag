from neo4j import GraphDatabase
from src.config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, 
    VECTOR_INDEX_NAME, FULLTEXT_INDEX_NAME, EMBEDDING_DIM
)

driver = None

def init_driver():
    global driver
    if driver is None:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        print("Connected to Neo4j.")
    return driver

def close_driver():
    global driver
    if driver is not None:
        driver.close()
        driver = None

def check_index_exists(session, index_name: str) -> bool:
    """
    Check if a Neo4j index exists.
    """
    try:
        res = session.run("SHOW INDEXES YIELD name")
        names = [r["name"] for r in res]
        return index_name in names
    except Exception as e:
        print(f"Error checking index: {e}")
        return False

def setup_indexes():
    """
    Create Vector and Fulltext indexes if they don't exist.
    """
    print("Setting up indexes...")
    drv = init_driver()
    with drv.session() as session:
        # 1. Vector Index
        session.run(f"""
        CREATE VECTOR INDEX {VECTOR_INDEX_NAME} IF NOT EXISTS
        FOR (n:TMT)
        ON (n.embedding_vec)
        OPTIONS {{indexConfig: {{
            `vector.dimensions`: {EMBEDDING_DIM},
            `vector.similarity_function`: 'cosine'
        }}}}
        """)
        
        # 2. Fulltext Index
        session.run(f"""
        CREATE FULLTEXT INDEX {FULLTEXT_INDEX_NAME} IF NOT EXISTS
        FOR (n:TMT)
        ON EACH [
            n.name, 
            n.fsn, 
            n.trade_name, 
            n.generic_name, 
            n.active_ingredient, 
            n.active_ingredients,
            n.strength,
            n.strengths,
            n.dosageform,
            n.manufacturer
        ]
        """)
        
        print("Waiting for indexes to come online...")
        try:
            session.run(f"CALL db.awaitIndex('{VECTOR_INDEX_NAME}', 30)")
            session.run(f"CALL db.awaitIndex('{FULLTEXT_INDEX_NAME}', 30)")
        except Exception as e:
            print(f"Warning: Index await failed: {e}")

        print("Indexes checked/created.")
        
        # Debug: Show indexes
        res = session.run("SHOW INDEXES YIELD name, state, type")
        print("Current Indexes:")
        for r in res:
            print(f"- {r['name']} ({r['type']}): {r['state']}")

def run_cypher(cypher: str):
    try:
        drv = init_driver()
        with drv.session() as session:
            result = list(session.run(cypher))
            return result
    except Exception as e:
        return f"Cypher Error: {e}"

def fetch_nodes_by_element_ids(element_ids: list[str]) -> list:
    """
    Fetch Neo4j nodes by elementId().
    """
    if not element_ids:
        return []

    # Dedupe while preserving order
    element_ids = list(dict.fromkeys([x for x in element_ids if x]))

    drv = init_driver()
    with drv.session() as session:
        q = """
        UNWIND $ids AS id
        MATCH (n) WHERE elementId(n) = id
        RETURN n
        """
        recs = session.run(q, ids=element_ids)
        return [r["n"] for r in recs]

