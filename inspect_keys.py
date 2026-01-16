
from src.knowledge_graph import init_driver, close_driver

def inspect_keys():
    drv = init_driver()
    with drv.session() as session:
        for label in ['VTM', 'GP', 'GPU', 'TP', 'TPU', 'SUBS']:
            print(f"--- Keys for {label} ---")
            res = session.run(f"MATCH (n:{label}) RETURN keys(n) as keys LIMIT 1")
            for r in res:
                print(r['keys'])
            print()

if __name__ == "__main__":
    try:
        inspect_keys()
    finally:
        close_driver()
