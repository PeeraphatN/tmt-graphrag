from dotenv import load_dotenv
load_dotenv()

import json
from src.services.database import init_driver

def load_manufacturer_from_neo4j() -> set[str]:
    """Load all unique manufacturer names from Neo4j."""
    query = """
    MATCH (n:TMT)
    WHERE n.manufacturer IS NOT NULL AND n.manufacturer <> ''
    RETURN DISTINCT n.manufacturer AS manu
    ORDER BY manu
    """
    
    driver = init_driver()
    manufacturers = set()
    
    with driver.session() as session:
        result = session.run(query)
        for record in result:
            manu = record["manu"]
            if manu and manu.strip():
                manufacturers.add(manu.strip())
    
    return manufacturers


def save_manufacturers_to_json(manufacturers: set[str], output_path: str = "manufacturers.json"):
    """Save manufacturers to JSON file."""
    data = {
        "count": len(manufacturers),
        "manufacturers": sorted(list(manufacturers))
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Saved {len(manufacturers)} manufacturers to {output_path}")


if __name__ == "__main__":
    print("🔍 Loading manufacturers from Neo4j...")
    manufacturers = load_manufacturer_from_neo4j()
    print(f"   Found {len(manufacturers)} unique manufacturers")
    
    # Save to JSON
    save_manufacturers_to_json(manufacturers)
    
    # Print sample
    print("\n📋 Sample (first 20):")
    for manu in sorted(manufacturers)[:20]:
        print(f"   - {manu}")