import re
from src.services.database import init_driver, fetch_nodes_by_element_ids

def get_search_config(target_type: str) -> dict:
    """
    Returns search configuration based on target_type (intent).
    Centralizes logic for allowed_levels, required_fields, and max_entities.
    """
    nlem_fields = ['nlem', 'nlem_category', 'nlem_section']
    base_fields = ['trade_name', 'manufacturer', 'fsn', 'tmtid', 'level']
    
    config = {
        "allowed_levels": {'TP', 'TPU', 'GP', 'GPU'},
        "required_fields": base_fields + nlem_fields,
        "max_entities": 50
    }

    if target_type == 'manufacturer':
        config["required_fields"] = ['trade_name', 'manufacturer', 'fsn', 'tmtid', 'level', 'container_text'] + nlem_fields
        config["allowed_levels"] = {'TP', 'TPU'}
        config["max_entities"] = 40
    elif target_type == 'ingredient':
        config["required_fields"] = ['fsn', 'active_ingredient', 'active_ingredients', 'strength', 'tmtid', 'level', 'dosageform'] + nlem_fields
        config["allowed_levels"] = {'GP', 'GPU', 'TP', 'TPU', 'VTM', 'SUBS'}
        config["max_entities"] = 60
    elif target_type == 'nlem':
        config["required_fields"] = ['fsn', 'tmtid', 'level'] + nlem_fields
        config["allowed_levels"] = {'GP'}
        config["max_entities"] = 80
    elif target_type == 'hierarchy':
        config["required_fields"] = ['level', 'fsn', 'tmtid'] + nlem_fields
        config["allowed_levels"] = {'SUBS', 'VTM', 'GP', 'GPU', 'TP', 'TPU'}
        config["max_entities"] = 60
    elif target_type == 'formula':
         config["required_fields"] = ['trade_name', 'manufacturer', 'fsn', 'tmtid', 'level'] + nlem_fields
         config["allowed_levels"] = {'TP', 'TPU', 'GP', 'GPU'}
         config["max_entities"] = 50

    return config

# Helper functions for extraction
def _fallback_trade_and_mfr_from_fsn(fsn: str) -> tuple[str | None, str | None]:
    if not fsn or '(' not in fsn:
        return None, None
    brand = fsn.split('(')[0].strip()
    m = re.search(r'\(([^)]+)\)', fsn)
    mfr = m.group(1).strip() if m else None
    return (brand or None), (mfr or None)

def _fallback_ingredients_from_fsn(fsn: str) -> str | None:
    if not fsn:
        return None
    parts = re.findall(r'\(([^)]+)\)', fsn)
    if len(parts) >= 2:
        return parts[1].strip()
    return None

def extract_structured_data(results: dict, question_type: str) -> dict:
    """
    Extract deterministic structured data from GraphRAG results.
    """
    seed_results = results.get("seed_results", [])
    expanded_nodes = results.get("expanded_nodes", [])
    relationships = results.get("relationships", [])
    strategy = results.get("strategy", "retrieve")

    # -------------------------
    # Handling Special Strategies
    # -------------------------
    if strategy == 'count':
        # Direct return of count result
        return {
            "strategy": "count",
            "count": results.get("result", 0),
            "entities": [] # No entities for count op
        }

    # -------------------------
    # Property filter + level scope
    # -------------------------
    # Use centralized config
    config = get_search_config(question_type)
    required_fields = config["required_fields"]
    allowed_levels = config["allowed_levels"]
    max_entities = config["max_entities"]

    # -------------------------
    # Collect candidate nodes (seed + expanded)
    # -------------------------
    candidate_nodes = []
    for item in seed_results:
        candidate_nodes.append(item["node"])
    for item in expanded_nodes:
        candidate_nodes.append(item["node"])

    # -------------------------
    # Entity completion from relationship endpoints
    # -------------------------
    endpoint_ids = []
    for rel in relationships:
        if rel.get("start_id"):
            endpoint_ids.append(rel["start_id"])
        if rel.get("end_id"):
            endpoint_ids.append(rel["end_id"])

    endpoint_nodes = fetch_nodes_by_element_ids(endpoint_ids)
    candidate_nodes.extend(endpoint_nodes)

    # -------------------------
    # Build entities deterministically
    # -------------------------
    entities = []
    seen = set()

    for node in candidate_nodes:
        props = dict(node)
        level = props.get("level")
        tmtid = props.get("tmtid")

        if not level or level not in allowed_levels:
            continue
        if not tmtid:
            continue
            
        key = (level, str(tmtid))
        if key in seen:
            continue
        seen.add(key)

        entity = {"labels": list(node.labels)}

        for field in required_fields:
            if field in props and props[field] not in (None, ""):
                entity[field] = props[field]

        # FSN fallback (trade_name/manufacturer)
        fsn = entity.get("fsn") or props.get("fsn")
        if fsn:
            # trade_name fallback เฉพาะ TP/TPU
            if level in ("TP", "TPU") and not entity.get("trade_name"):
                b, mfr = _fallback_trade_and_mfr_from_fsn(fsn)
                if b:
                    entity["trade_name_fallback"] = b
                if not entity.get("manufacturer") and mfr:
                    entity["manufacturer_fallback"] = mfr

            # manufacturer fallback เฉพาะ TP/TPU
            if level in ("TP", "TPU") and not entity.get("manufacturer"):
                _, mfr = _fallback_trade_and_mfr_from_fsn(fsn)
                if mfr:
                    entity["manufacturer_fallback"] = mfr

            # formula/ingredient fallback
            if question_type in ("formula", "ingredient") and "ingredients_fallback" not in entity:
                ing = _fallback_ingredients_from_fsn(fsn)
                if ing:
                    entity["ingredients_fallback"] = ing

        entities.append(entity)

    # -------------------------
    # Sort & limit
    # -------------------------
    def sort_key(e: dict):
        lvl = e.get("level", "")
        name = e.get("trade_name") or e.get("trade_name_fallback") or e.get("fsn", "")
        return (0 if lvl == "TP" else 1, name)

    if question_type == "manufacturer":
        entities.sort(key=sort_key)
    else:
        entities.sort(key=lambda e: e.get("fsn", ""))

    entities = entities[:max_entities]

    # -------------------------
    # Evidence with elementIds
    # -------------------------
    evidence = []
    for rel in relationships[:20]:
        evidence.append({
            "start_id": rel.get("start_id"),
            "end_id": rel.get("end_id"),
            "rel": rel.get("type"),
            "start": f"{rel['start_labels'][0] if rel.get('start_labels') else 'Node'}:{rel.get('start_name','Unknown')}",
            "end": f"{rel['end_labels'][0] if rel.get('end_labels') else 'Node'}:{rel.get('end_name','Unknown')}",
        })

    return {
        "question_type": question_type,
        "entities": entities,
        "evidence": evidence,
        "total_entities": len(entities),
        "total_relationships": len(evidence)
    }
