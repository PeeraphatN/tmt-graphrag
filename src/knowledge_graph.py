from neo4j import GraphDatabase
import json
import re
from src.config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, 
    VECTOR_INDEX_NAME, FULLTEXT_INDEX_NAME, EMBEDDING_DIM,
    GRAPH_TRAVERSAL_DEPTH
)
from src.query_processor import sanitize_fulltext_query
from src.llm_service import get_embedding

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

def hybrid_search(question: str, k: int = 5) -> list[dict]:
    """
    Perform hybrid search using Vector Search and Fulltext Search,
    combined via Reciprocal Rank Fusion (RRF).
    """
    # 1. Get Embedding
    query_vec = get_embedding(question)
    if not query_vec:
        return []

    results = {} # node_element_id -> {node: ..., v_score: ..., t_score: ...}

    drv = init_driver()
    with drv.session() as session:
        # 2. Vector Search
        vector_query = f"""
        CALL db.index.vector.queryNodes($index_name, $k, $embedding)
        YIELD node, score
        RETURN node, score
        """
        try:
            v_recs = session.run(vector_query, index_name=VECTOR_INDEX_NAME, k=k, embedding=query_vec)
            for i, rec in enumerate(v_recs):
                node = rec["node"]
                nid = node.element_id if hasattr(node, 'element_id') else node.id
                results[nid] = {
                    "node": node,
                    "v_score": rec["score"],
                    "t_score": 0.0,
                    "v_rank": i,
                    "t_rank": 9999
                }
        except Exception as e:
            print(f"Vector search error: {e}")

        # 3. Fulltext Search (skip if pure Thai with no English/numbers)
        sanitized_query, use_fulltext = sanitize_fulltext_query(question)
        
        if use_fulltext and sanitized_query:
            fulltext_query = f"""
            CALL db.index.fulltext.queryNodes($index_name, $search_text, {{limit: $k}})
            YIELD node, score
            RETURN node, score
            """
            try:
                t_recs = session.run(fulltext_query, index_name=FULLTEXT_INDEX_NAME, search_text=sanitized_query, k=k)
                for i, rec in enumerate(t_recs):
                    node = rec["node"]
                    nid = node.element_id if hasattr(node, 'element_id') else node.id
                    
                    if nid in results:
                        results[nid]["t_score"] = rec["score"]
                        results[nid]["t_rank"] = i
                    else:
                        results[nid] = {
                            "node": node,
                            "v_score": 0.0,
                            "t_score": rec["score"],
                            "v_rank": 9999,
                            "t_rank": i
                        }
            except Exception as e:
                print(f"Fulltext search error: {e}")
        else:
            print("   Skipping fulltext search (pure Thai or no keywords)")


    # 4. RRF Calculation
    k_rrf = 60
    final_results = []
    for nid, data in results.items():
        rrf_score = (1 / (k_rrf + data["v_rank"] + 1)) + (1 / (k_rrf + data["t_rank"] + 1))
        data["rrf_score"] = rrf_score
        final_results.append(data)

    final_results.sort(key=lambda x: x["rrf_score"], reverse=True)
    
    return final_results[:k]

def expand_context(node_ids: list[str], depth: int = 2) -> dict:
    """
    Expand context by traversing relationships from seed nodes.
    Returns nodes, relationships, and paths found.
    """
    if not node_ids:
        return {"nodes": [], "relationships": [], "paths": []}
    
    drv = init_driver()
    expanded_nodes = {}
    relationships = []
    paths = []
    
    with drv.session() as session:
        # Query to get related nodes and relationships
        # Using variable-length path pattern to traverse up to 'depth' hops
        traversal_query = """
        MATCH path = (n)-[r*1..$depth]-(m)
        WHERE elementId(n) IN $node_ids
        RETURN path, 
               [rel IN relationships(path) | {
                   type: type(rel),
                   start_id: elementId(startNode(rel)),
                   end_id: elementId(endNode(rel)),
                   start_labels: labels(startNode(rel)),
                   end_labels: labels(endNode(rel)),
                   start_name: coalesce(startNode(rel).name, startNode(rel).fsn, startNode(rel).trade_name, 'Unknown'),
                   end_name: coalesce(endNode(rel).name, endNode(rel).fsn, endNode(rel).trade_name, 'Unknown')
               }] as rels,
               [node IN nodes(path) | node] as path_nodes
        LIMIT 50
        """
        
        try:
            # Neo4j doesn't support parameter in path length, so we build query dynamically
            actual_query = traversal_query.replace("$depth", str(depth))
            records = session.run(actual_query, node_ids=node_ids)
            
            for rec in records:
                # Collect unique nodes
                for node in rec["path_nodes"]:
                    nid = node.element_id if hasattr(node, 'element_id') else node.id
                    if nid not in expanded_nodes:
                        expanded_nodes[nid] = {
                            "node": node,
                            "labels": list(node.labels),
                            "is_seed": nid in node_ids
                        }
                
                # Collect relationships
                for rel_info in rec["rels"]:
                    rel_key = f"{rel_info['start_id']}-{rel_info['type']}-{rel_info['end_id']}"
                    if rel_key not in [r.get('key') for r in relationships]:
                        rel_info['key'] = rel_key
                        relationships.append(rel_info)
                
                # Store path info
                path = rec["path"]
                if path:
                    paths.append(path)
                    
        except Exception as e:
            print(f"Relationship traversal error: {e}")
    
    return {
        "nodes": list(expanded_nodes.values()),
        "relationships": relationships,
        "paths": paths
    }

def graphrag_search(question: str, k: int = 5, depth: int = None) -> dict:
    """
    GraphRAG search: Hybrid search + Relationship traversal.
    """
    if depth is None:
        depth = GRAPH_TRAVERSAL_DEPTH
    
    # Step 1: Get seed nodes via hybrid search
    seed_results = hybrid_search(question, k=k)
    
    if not seed_results:
        return {
            "seed_results": [],
            "expanded_nodes": [],
            "relationships": []
        }
    
    # Step 2: Extract node IDs for expansion
    seed_node_ids = []
    for item in seed_results:
        node = item["node"]
        nid = node.element_id if hasattr(node, 'element_id') else node.id
        seed_node_ids.append(nid)
    
    # Step 3: Expand context via relationship traversal
    expanded = expand_context(seed_node_ids, depth=depth)
    
    # Step 4: Filter out seed nodes from expanded (to avoid duplication)
    non_seed_nodes = [n for n in expanded["nodes"] if not n.get("is_seed", False)]
    
    return {
        "seed_results": seed_results,
        "expanded_nodes": non_seed_nodes,
        "relationships": expanded["relationships"]
    }

def execute_count_query(query_obj) -> dict:
    """
    Strategy: COUNT
    Executes a Cypher COUNT query based on filters.
    """
    # Build filter clause - Base label is TMT, usually looking for GP level for counts
    where_parts = ["n:TMT"]
    params = {}
    
    if query_obj.target_type == 'nlem' or query_obj.nlem_filter:
        where_parts.append("n.nlem = true")
        if query_obj.nlem_category:
            where_parts.append("n.nlem_category = $cat")
            params['cat'] = query_obj.nlem_category
            
    if query_obj.manufacturer_filter:
        where_parts.append("n.manufacturer CONTAINS $manu")
        params['manu'] = query_obj.manufacturer_filter
        
    where_clause = " WHERE " + " AND ".join(where_parts)
    
    cypher = f"""
    MATCH (n)
    {where_clause}
    RETURN count(n) as total
    """
    
    print(f"   [Strategy: COUNT] Executing: {where_clause}")
    print(f"   Params: {params}")
    
    drv = init_driver()
    count = 0
    with drv.session() as session:
        res = session.run(cypher, **params)
        record = res.single()
        if record:
            count = record["total"]
            
    return {
        "strategy": "count",
        "result": count,
        "seed_results": [],
        "expanded_nodes": [],
        "relationships": []
    }

def execute_listing_query(query_obj, k: int = 100) -> dict:
    """
    Strategy: LIST
    Executes a Cypher MATCH query to get a raw list of items.
    """
    where_parts = ["n:TMT"]
    params = {}
    
    if query_obj.target_type == 'nlem' or query_obj.nlem_filter:
        where_parts.append("n.nlem = true")
        if query_obj.nlem_category:
            where_parts.append("n.nlem_category = $cat")
            params['cat'] = query_obj.nlem_category
            
    if query_obj.manufacturer_filter:
        # Note: Manufacturers are often at TP (Trade Product) level
        # If filtering by manufacturer, allow TP level as well
        where_parts = ["n:TMT"]
        where_parts.append("n.manufacturer CONTAINS $manu")
        params['manu'] = query_obj.manufacturer_filter
        
    where_clause = " WHERE " + " AND ".join(where_parts)
    
    cypher = f"""
    MATCH (n)
    {where_clause}
    RETURN n
    LIMIT $k
    """
    
    print(f"   [Strategy: LIST] Executing: {where_clause} LIMIT {k}")
    print(f"   Params: {params}")
    
    drv = init_driver()
    nodes = []
    with drv.session() as session:
        res = session.run(cypher, k=k, **params)
        nodes = [rec["n"] for rec in res]
        
    # Return as 'seed_results' styled dict so extract step can process it
    seed_results = [{"node": n, "score": 1.0, "rrf_score": 1.0} for n in nodes]
    
    return {
        "strategy": "list",
        "seed_results": seed_results,
        "expanded_nodes": [],
        "relationships": []
    }

def execute_verification_query(query_obj) -> dict:
    """
    Strategy: VERIFY
    Checks for existence of a specific entity with filters.
    """
    # Ensure there is a query term
    if not query_obj.query:
        return search_general(query_obj, k=5)
        
    params = {'q': query_obj.query}
    # For verification, search typically on TMT nodes
    where_parts = ["n:TMT", "(toLower(n.fsn) CONTAINS toLower($q) OR toLower(n.generic_name) CONTAINS toLower($q))"]
    
    if query_obj.nlem_filter:
        where_parts.append("n.nlem = true")
    
    where_clause = " WHERE " + " AND ".join(where_parts)
    
    cypher = f"""
    MATCH (n)
    {where_clause}
    RETURN n
    LIMIT 10
    """
    
    print(f"   [Strategy: VERIFY] Checking: {query_obj.query}")
    print(f"   Params: {params}")
    
    drv = init_driver()
    nodes = []
    with drv.session() as session:
        res = session.run(cypher, **params)
        nodes = [rec["n"] for rec in res]
        
    seed_results = [{"node": n, "score": 1.0, "rrf_score": 1.0} for n in nodes]
    
    return {
        "strategy": "verify",
        "seed_results": seed_results,
        "expanded_nodes": [],
        "relationships": []
    }

def search_general(query_obj, k: int = 10, depth: int = 2) -> dict:
    """
    Strategy: RETRIEVE (Default)
    Standard Hybrid Search + Graph Traversal
    """
    print(f"   [Strategy: RETRIEVE] Hybrid search for: {query_obj.query}")
    
    clean_query = query_obj.query if query_obj.query else "ยา"
    
    # ----------------------------------------------------
    # Hybrid Search Logic (Same as before)
    # ----------------------------------------------------
    search_queries = [clean_query]
    if hasattr(query_obj, 'expanded_queries') and query_obj.expanded_queries:
        search_queries.extend(query_obj.expanded_queries)
    search_queries = list(dict.fromkeys([q for q in search_queries if q]))
    
    all_seed_results = []
    seen_seed_ids = set()
    
    for q_str in search_queries:
        results = hybrid_search(q_str, k=k)
        for item in results:
            node = item["node"]
            nid = node.element_id if hasattr(node, 'element_id') else node.id
            if nid not in seen_seed_ids:
                seen_seed_ids.add(nid)
                item["found_by_query"] = q_str
                all_seed_results.append(item)
    
    # Expand context
    seed_node_ids = [ (n["node"].element_id if hasattr(n["node"], 'element_id') else n["node"].id) for n in all_seed_results ]
    if not seed_node_ids:
        return {"seed_results": [], "expanded_nodes": [], "relationships": []}

    expanded = expand_context(seed_node_ids, depth=depth)
    non_seed_nodes = [n for n in expanded["nodes"] if not n.get("is_seed", False)]
    
    return {
        "strategy": "retrieve",
        "seed_results": all_seed_results,
        "expanded_nodes": non_seed_nodes,
        "relationships": expanded["relationships"]
    }

def advanced_graphrag_search(query_obj, k: int = 10, depth: int = 2) -> dict:
    """
    Intent Router: Dispatches the search to specific functions based on query_obj.strategy.
    """
    strategy = getattr(query_obj, 'strategy', 'retrieve')
    
    # 1. Router Logic
    if strategy == 'count':
        return execute_count_query(query_obj)
        
    elif strategy == 'list':
        # Increase K for listing
        return execute_listing_query(query_obj, k=50)
        
    elif strategy == 'verify':
        return execute_verification_query(query_obj)
        
    else:
        # Default: RETRIEVE
        return search_general(query_obj, k=k, depth=depth)

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
    # Common NLEM fields to include if present
    nlem_fields = ['nlem', 'nlem_category', 'nlem_section']

    if question_type == 'manufacturer':
        required_fields = ['trade_name', 'manufacturer', 'fsn', 'tmtid', 'level', 'container_text'] + nlem_fields
        allowed_levels = {'TP', 'TPU'}
        max_entities = 40  # Increased from 20
    elif question_type == 'ingredient':
        required_fields = ['fsn', 'active_ingredient', 'active_ingredients', 'strength', 'tmtid', 'level', 'dosageform'] + nlem_fields
        allowed_levels = {'GP', 'GPU', 'TP', 'TPU', 'VTM', 'SUBS'}
        max_entities = 60  # Increased from 30
    elif question_type == 'nlem':
        required_fields = ['fsn', 'tmtid', 'level'] + nlem_fields
        allowed_levels = {'GP'}
        max_entities = 80 # Increased from 50
    elif question_type == 'hierarchy':
        required_fields = ['level', 'fsn', 'tmtid'] + nlem_fields
        allowed_levels = {'SUBS', 'VTM', 'GP', 'GPU', 'TP', 'TPU'}
        max_entities = 60  # Increased from 40
    elif question_type == 'formula':
        # formula เทียบจาก FSN วงเล็บที่ 2 เป็นหลัก (fallback)
        required_fields = ['trade_name', 'manufacturer', 'fsn', 'tmtid', 'level'] + nlem_fields
        allowed_levels = {'TP', 'TPU', 'GP', 'GPU'}
        max_entities = 50  # Increased from 30
    else:
        required_fields = ['trade_name', 'manufacturer', 'fsn', 'tmtid', 'level'] + nlem_fields
        allowed_levels = {'TP', 'TPU', 'GP', 'GPU'}
        max_entities = 50  # Increased from 30

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
            
        # Strict filter for NLEM query type -> REMOVED to allow "Is X in NLEM?" (Negative answer)
        # if question_type == 'nlem' and not props.get('nlem'):
        #    continue

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

def run_cypher(cypher: str):
    try:
        drv = init_driver()
        with drv.session() as session:
            result = list(session.run(cypher))
            return result
    except Exception as e:
        return f"Cypher Error: {e}"
