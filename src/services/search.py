"""
Search Service.
Handles Advanced Search strategies (Intent Routing), Vector/Fulltext access, and Graph Traversal.
"""
from src.services.database import init_driver, check_index_exists
from src.services.extraction import get_search_config
from src.config import VECTOR_INDEX_NAME, FULLTEXT_INDEX_NAME, GRAPH_TRAVERSAL_DEPTH
from src.query_processor import sanitize_fulltext_query
from src.llm_service import get_embedding

def hybrid_search(question: str, k: int = 5, allowed_levels: list[str] = None) -> list[dict]:
    """
    Perform hybrid search (Vector + Fulltext) with RRF fusion.
    """
    embedding = get_embedding(question)
    if not embedding:
        return []
    
    drv = init_driver()
    results = {}

    # Build WHERE clause for pre-filtering
    where_clause = ""
    params = {"index_name": VECTOR_INDEX_NAME, "k": k, "embedding": embedding}
    
    if allowed_levels:
        where_clause = "WHERE node.level IN $allowed_levels"
        params["allowed_levels"] = list(allowed_levels)

    with drv.session() as session:
        # 1. Vector Search
        
        # Check if index exists first (optional, but good for safety)
        if not check_index_exists(session, VECTOR_INDEX_NAME):
            print(f"Index {VECTOR_INDEX_NAME} not found. Skipping vector search.")
        else:
            vector_query = f"""
            CALL db.index.vector.queryNodes($index_name, $k, $embedding)
            YIELD node, score
            {where_clause}
            RETURN node, score
            """
            
            try:
                v_recs = session.run(vector_query, **params)
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

        # 2. Fulltext Search (skip if pure Thai with no English/numbers)
        sanitized_query, use_fulltext = sanitize_fulltext_query(question)
        
        if use_fulltext and sanitized_query:
            fulltext_params = {
                "index_name": FULLTEXT_INDEX_NAME, 
                "search_text": sanitized_query, 
                "k": k
            }
            if allowed_levels:
                fulltext_params["allowed_levels"] = list(allowed_levels)

            fulltext_query = f"""
            CALL db.index.fulltext.queryNodes($index_name, $search_text, {{limit: $k}})
            YIELD node, score
            {where_clause}
            RETURN node, score
            """
            try:
                t_recs = session.run(fulltext_query, **fulltext_params)
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
            # print("   Skipping fulltext search (pure Thai or no keywords)")
            pass

    # 3. RRF Calculation
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

def search_general(query_obj, k: int = 10, depth: int = 2) -> dict:
    """
    Strategy: RETRIEVE (Default)
    Standard Hybrid Search + Graph Traversal
    """
    print(f"   [Strategy: RETRIEVE] Hybrid search for: {query_obj.query}")
    
    clean_query = query_obj.query if query_obj.query else "ยา"
    
    search_queries = [clean_query]
    if hasattr(query_obj, 'expanded_queries') and query_obj.expanded_queries:
        search_queries.extend(query_obj.expanded_queries)
    search_queries = list(dict.fromkeys([q for q in search_queries if q]))
    
    all_seed_results = []
    seen_seed_ids = set()
    
    for q_str in search_queries:
        # Get allowed levels from config to pre-filter search
        config = get_search_config(query_obj.target_type if hasattr(query_obj, 'target_type') else 'general')
        
        results = hybrid_search(q_str, k=k, allowed_levels=config["allowed_levels"])
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

def execute_count_query(query_obj) -> dict:
    """
    Strategy: COUNT
    Executes a Cypher COUNT query based on filters.
    """
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
    
    drv = init_driver()
    nodes = []
    with drv.session() as session:
        res = session.run(cypher, k=k, **params)
        nodes = [rec["n"] for rec in res]
        
    seed_results = [{"node": n, "score": 1.0, "rrf_score": 1.0} for n in nodes]
    
    return {
        "strategy": "list",
        "seed_results": seed_results,
        "expanded_nodes": [],
        "relationships": []
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
        return execute_listing_query(query_obj, k=50)
        
    elif strategy == 'verify':
        print(f"   [Strategy: VERIFY] Redirecting to General Search for broader context...")
        return search_general(query_obj, k=k, depth=depth)
        
    else:
        # Default: RETRIEVE
        return search_general(query_obj, k=k, depth=depth)
