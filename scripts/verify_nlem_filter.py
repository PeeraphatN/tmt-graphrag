from src.knowledge_graph import extract_structured_data
import json

class MockNode:
    def __init__(self, labels, props, element_id):
        self.labels = labels
        self._props = props
        self.element_id = element_id
    def __getitem__(self, key):
        return self._props.get(key)
    def __contains__(self, key):
        return key in self._props
    def __iter__(self):
        return iter(self._props)
    def keys(self):
        return self._props.keys()

# 22 Seed NLEM Nodes
seed_results = []
for i in range(22):
    node = MockNode(["GP"], {"tmtid": f"S{i}", "level": "GP", "fsn": f"Drug S{i}", "nlem": True}, f"node_s{i}")
    seed_results.append({"node": node})

# 5 Expanded non-NLEM nodes
expanded_nodes = []
for i in range(5):
    node = MockNode(["GP"], {"tmtid": f"E{i}", "level": "GP", "fsn": f"Drug E{i}", "nlem": False}, f"node_e{i}")
    expanded_nodes.append({"node": node})

results = {
    "seed_results": seed_results,
    "expanded_nodes": expanded_nodes,
    "relationships": []
}

print("Testing with target_type='nlem'...")
structured_nlem = extract_structured_data(results, 'nlem')
print(f"Entities Count (NLEM): {len(structured_nlem['entities'])}")

print("\nTesting with target_type='general'...")
structured_general = extract_structured_data(results, 'general')
print(f"Entities Count (General): {len(structured_general['entities'])}")
