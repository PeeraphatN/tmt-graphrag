from neo4j import GraphDatabase
from src.config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, 
    VECTOR_INDEX_NAME, FULLTEXT_INDEX_NAME, EMBEDDING_DIM
)

driver = None
EXPECTED_FULLTEXT_PROPS = [
    "name",
    "fsn",
    "trade_name",
    "generic_name",
    "active_substance",
    "active_substances",
    "strength",
    "strengths",
    "dosageform",
    "manufacturer",
]

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


def _escape_identifier(name: str) -> str:
    return str(name or "").replace("`", "``")


def _get_index_details(session, index_name: str) -> dict | None:
    try:
        rec = session.run(
            """
            SHOW INDEXES
            YIELD name, type, entityType, labelsOrTypes, properties, state
            WHERE name = $name
            RETURN name, type, entityType, labelsOrTypes, properties, state
            """,
            name=index_name,
        ).single()
        return dict(rec) if rec else None
    except Exception as e:
        print(f"Warning: could not inspect index {index_name}: {e}")
        return None


def _fulltext_index_drifted(details: dict | None) -> bool:
    if not details:
        return False
    if str(details.get("type", "")).upper() != "FULLTEXT":
        return True
    if str(details.get("entityType", "")).upper() != "NODE":
        return True
    labels = set(details.get("labelsOrTypes") or [])
    if labels != {"TMT"}:
        return True
    current_props = set(details.get("properties") or [])
    expected_props = set(EXPECTED_FULLTEXT_PROPS)
    return current_props != expected_props

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
        
        # 2. Fulltext Index (with drift reconciliation)
        fulltext_details = _get_index_details(session, FULLTEXT_INDEX_NAME)
        if _fulltext_index_drifted(fulltext_details):
            current_props = fulltext_details.get("properties") if fulltext_details else None
            print(
                f"Fulltext index drift detected: {FULLTEXT_INDEX_NAME}\n"
                f"  current={current_props}\n"
                f"  expected={EXPECTED_FULLTEXT_PROPS}\n"
                "Rebuilding fulltext index..."
            )
            safe_index = _escape_identifier(FULLTEXT_INDEX_NAME)
            session.run(f"DROP INDEX `{safe_index}` IF EXISTS")

        session.run(f"""
        CREATE FULLTEXT INDEX {FULLTEXT_INDEX_NAME} IF NOT EXISTS
        FOR (n:TMT)
        ON EACH [
            n.name,
            n.fsn,
            n.trade_name,
            n.generic_name,
            n.active_substance,
            n.active_substances,
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
        res = session.run("SHOW INDEXES YIELD name, state, type, properties")
        print("Current Indexes:")
        for r in res:
            print(f"- {r['name']} ({r['type']}): {r['state']} props={r.get('properties')}")

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

