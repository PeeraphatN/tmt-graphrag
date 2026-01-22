import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.knowledge_graph import init_driver

def check_node(tmtid):
    driver = init_driver()
    with driver.session() as session:
        result = session.run(
            "MATCH (n:TMT {tmtid: $tmtid}) RETURN n.fsn, n.nlem, n.nlem_category", 
            tmtid=tmtid
        )
        record = result.single()
        if record:
            print(f"Name: {record['n.fsn']}")
            print(f"NLEM: {record['n.nlem']}")
            print(f"Category: {record['n.nlem_category']}")
        else:
            print("Node not found")

if __name__ == "__main__":
    check_node("740860")
