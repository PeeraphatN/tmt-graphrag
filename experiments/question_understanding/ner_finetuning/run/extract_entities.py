import json
import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
AUTH = (
    os.getenv("NEO4J_USER", "neo4j"),
    os.getenv("NEO4J_PASSWORD", "password"),
)

RUN_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = RUN_DIR.parent
DATA_DIR = EXPERIMENT_DIR / "data"


def run_query(query: str) -> list[str]:
    driver = None
    try:
        driver = GraphDatabase.driver(URI, auth=AUTH)
        driver.verify_connectivity()
        with driver.session() as session:
            result = session.run(query)
            return [
                str(record["name"]).strip()
                for record in result
                if record.get("name") and str(record["name"]).strip()
            ]
    except Exception as exc:
        print(f"Error executing query: {query}")
        print(f"Error details: {exc}")
        return []
    finally:
        if driver:
            driver.close()


def save_json(data: list[str], filename: str) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    filepath = DATA_DIR / filename
    cleaned = [str(item).strip() for item in data if item and str(item).strip()]
    unique_data = sorted(set(cleaned))
    filepath.write_text(json.dumps(list(unique_data), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(unique_data)} unique items to {filepath.name}")


def main() -> None:
    print(f"Connecting to Neo4j at {URI}...")

    print("\n1. Extracting DRUG names (SUBS + VTM)...")
    drugs_subs = run_query("MATCH (n:SUBS) RETURN DISTINCT n.name as name")
    drugs_vtm = run_query("MATCH (n:VTM) RETURN DISTINCT n.name as name")
    save_json(drugs_subs + drugs_vtm, "entities_drug.json")

    print("\n2. Extracting BRAND names (TP)...")
    brands = run_query("MATCH (n:TP) WHERE n.trade_name IS NOT NULL RETURN DISTINCT n.trade_name as name")
    if not brands:
        brands = run_query("MATCH (n:TP) RETURN DISTINCT n.name as name")
    save_json(brands, "entities_brand.json")

    print("\n3. Extracting MANUFACTURER names (TP)...")
    manufacturers = run_query("MATCH (n:TP) WHERE n.manufacturer IS NOT NULL RETURN DISTINCT n.manufacturer as name")
    save_json(manufacturers, "entities_manufacturer.json")

    print("\n4. Extracting DOSAGE FORM (GP)...")
    forms = run_query("MATCH (n:GP) WHERE n.dosageform IS NOT NULL RETURN DISTINCT n.dosageform as name")
    save_json(forms, "entities_form.json")

    print("\n5. Extracting STRENGTH (GP)...")
    strengths = run_query("MATCH (n:GP) WHERE n.strength IS NOT NULL RETURN DISTINCT n.strength as name")
    save_json(strengths, "entities_strength.json")

    print("\nExtraction complete.")


if __name__ == "__main__":
    main()