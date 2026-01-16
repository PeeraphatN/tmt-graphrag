
from src.knowledge_graph import init_driver, close_driver

def inspect():
    drv = init_driver()
    with drv.session() as session:
        print("--- Node Labels ---")
        labels = session.run("MATCH (n) RETURN labels(n) as label, count(*) as count")
        for r in labels:
            print(f"{r['label']}: {r['count']}")
            
        print("\n--- Relationship Types ---")
        rels = session.run("MATCH ()-[r]->() RETURN type(r) as type, count(*) as count")
        for r in rels:
            print(f"{r['type']}: {r['count']}")
            
        print("\n--- Level Stats (TMT nodes) ---")
        levels = session.run("MATCH (n:TMT) RETURN n.level as level, count(*) as count")
        for r in levels:
            print(f"{r['level']}: {r['count']}")

if __name__ == "__main__":
    try:
        inspect()
    finally:
        close_driver()
