from neo4j import GraphDatabase
import requests
import textwrap
import json

import os
import atexit
from dotenv import load_dotenv
import pathlib
import uuid
from datetime import datetime

# Load environment variables
load_dotenv()

# ==============================
# CONFIG
# ==============================

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

OLLAMA_URL = os.getenv("OLLAMA_URL")
OLLAMA_EMBED_URL = os.getenv("OLLAMA_EMBED_URL")
LLM_MODEL = os.getenv("LLM_MODEL")
EMBED_MODEL = os.getenv("EMBED_MODEL")

VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME")
FULLTEXT_INDEX_NAME = os.getenv("FULLTEXT_INDEX_NAME")
EMBEDDING_DIM_VALUE = os.getenv("EMBEDDING_DIM")
EMBEDDING_DIM = int(EMBEDDING_DIM_VALUE) if EMBEDDING_DIM_VALUE else None
GRAPH_TRAVERSAL_DEPTH = int(os.getenv("GRAPH_TRAVERSAL_DEPTH", "2"))

print("Connecting to Neo4j...")
driver = None


def validate_env():
    required_vars = {
        "NEO4J_URI": NEO4J_URI,
        "NEO4J_USER": NEO4J_USER,
        "NEO4J_PASSWORD": NEO4J_PASSWORD,
        "OLLAMA_URL": OLLAMA_URL,
        "OLLAMA_EMBED_URL": OLLAMA_EMBED_URL,
        "LLM_MODEL": LLM_MODEL,
        "EMBED_MODEL": EMBED_MODEL,
        "VECTOR_INDEX_NAME": VECTOR_INDEX_NAME,
        "FULLTEXT_INDEX_NAME": FULLTEXT_INDEX_NAME,
        "EMBEDDING_DIM": EMBEDDING_DIM,
    }

    missing = [name for name, value in required_vars.items() if value in (None, "")]
    if missing:
        missing_list = ", ".join(missing)
        raise RuntimeError(f"Missing required environment variables: {missing_list}")


def close_driver():
    global driver
    if driver is not None:
        driver.close()
        driver = None


atexit.register(close_driver)


def init_driver():
    global driver
    if driver is None:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        print("Connected to Neo4j.")
    return driver

# ==============================
# Embeddings & Indexes
# ==============================

def get_embedding(text: str) -> list[float]:
    """
    Generate embedding using Ollama
    """
    payload = {
        "model": EMBED_MODEL,
        "prompt": text,
    }
    try:
        resp = requests.post(OLLAMA_EMBED_URL, json=payload)
        resp.raise_for_status()
        return resp.json()["embedding"]
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return []

def setup_indexes():
    """
    Create Vector and Fulltext indexes if they don't exist.
    Based on schema: TMT label is common across most nodes.
    Properties: embedding_vec (Vector), name, fsn, etc. (Text)
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

# ==============================
# Query Preprocessing & Classification
# ==============================

# Thai stopwords (obvious ones)
THAI_STOPWORDS = {
    'อะไร', 'บ้าง', 'ใคร', 'ผลิต', 'เป็น', 'ยังไง', 'อย่างไร', 'ไหน', 'ทำไม',
    'มี', 'ได้', 'และ', 'หรือ', 'ที่', 'ของ', 'ใน', 'จาก', 'ให้', 'กับ',
    'แสดง', 'บอก', 'หา', 'ค้นหา', 'ดู', 'ต้องการ', 'ขอ', 'ช่วย'
}

# Lucene special characters that need escaping
LUCENE_SPECIAL_CHAR_SET = set(list(r'+-!(){}[]^"~*?:\/'))

def escape_lucene_char(char: str) -> str:
    return f'\\{char}' if char in LUCENE_SPECIAL_CHAR_SET else char

import re


def extract_keywords(question: str) -> list[str]:
    """
    Extract high-signal tokens from question.
    Returns tokens that are:
    - English words/drug names
    - Numbers and units (mg, g, mL, etc.)
    - Patterns like "1 g/20 mL"
    """
    keywords = []
    
    # Pattern for units with numbers: "1 g/20 mL", "500 mg"
    unit_pattern = r'\d+\.?\d*\s*(?:mg|g|mL|mcg|IU|%|ml|MG|G|ML)(?:/\d+\.?\d*\s*(?:mg|g|mL|mcg|IU|%|ml|MG|G|ML))?'
    unit_matches = re.findall(unit_pattern, question, re.IGNORECASE)
    keywords.extend(unit_matches)
    
    # Pattern for English words (drug names, etc.)
    english_pattern = r'[A-Za-z][A-Za-z0-9\-]*[A-Za-z0-9]|[A-Za-z]'
    english_matches = re.findall(english_pattern, question)
    # Filter out very short words (likely not drug names)
    keywords.extend([w for w in english_matches if len(w) >= 2])
    
    # Pattern for numbers that might be important (e.g., dosage)
    number_pattern = r'\d+\.?\d*'
    numbers = re.findall(number_pattern, question)
    # Only include if not already part of unit pattern
    for num in numbers:
        if not any(num in unit for unit in unit_matches):
            keywords.append(num)
    
    # Extract Thai words that are not stopwords
    thai_pattern = r'[\u0E00-\u0E7F]+'
    thai_words = re.findall(thai_pattern, question)
    meaningful_thai = [w for w in thai_words if w not in THAI_STOPWORDS and len(w) > 1]
    keywords.extend(meaningful_thai)
    
    # Deduplicate while preserving order
    seen = set()
    unique_keywords = []
    for kw in keywords:
        kw_lower = kw.lower() if kw.isascii() else kw
        if kw_lower not in seen:
            seen.add(kw_lower)
            unique_keywords.append(kw)
    
    return unique_keywords


def has_english_or_numbers(question: str) -> bool:
    """Check if question contains English letters or numbers."""
    return bool(re.search(r'[A-Za-z0-9]', question))


def sanitize_fulltext_query(question: str) -> tuple[str, bool]:
    """
    Sanitize question for Lucene fulltext query.
    
    Returns:
        tuple: (sanitized_query, should_use_fulltext)
        - sanitized_query: AND-joined escaped tokens
        - should_use_fulltext: False if pure Thai with no English/numbers
    """
    # Check if we should skip fulltext (pure Thai)
    if not has_english_or_numbers(question):
        return "", False
    
    # Extract keywords
    keywords = extract_keywords(question)
    
    if not keywords:
        return "", False
    
    # Escape special characters in each keyword
    escaped_keywords = []
    for kw in keywords:
        escaped = ''.join(escape_lucene_char(c) for c in kw)
        if escaped.strip():
            escaped_keywords.append(escaped)
    
    if not escaped_keywords:
        return "", False
    
    # Join with AND
    sanitized = ' AND '.join(escaped_keywords)
    return sanitized, True


def classify_question(question: str) -> str:
    """
    Classify question type using rule-based approach (no LLM).
    
    Returns one of:
    - 'manufacturer': บริษัท/ผู้ผลิต queries
    - 'ingredient': ส่วนผสม/active ingredient queries
    - 'formula': สูตร/formula comparison queries
    - 'hierarchy': ลำดับ/hierarchy queries
    - 'general': other queries
    """
    q_lower = question.lower()
    
    # Manufacturer patterns
    manufacturer_patterns = [
        r'บริษัท.*ผลิต', r'ผู้ผลิต', r'ใครผลิต', r'ผลิตโดย',
        r'manufacturer', r'บริษัท.*ยา', r'ยา.*บริษัท'
    ]
    for pattern in manufacturer_patterns:
        if re.search(pattern, question, re.IGNORECASE):
            return 'manufacturer'
    
    # Ingredient patterns
    ingredient_patterns = [
        r'ส่วนผสม', r'สาร.*ออกฤทธิ์', r'active.?ingredient',
        r'ประกอบด้วย', r'มี.*อะไร.*บ้าง', r'ingredient'
    ]
    for pattern in ingredient_patterns:
        if re.search(pattern, question, re.IGNORECASE):
            return 'ingredient'
    
    # Formula patterns
    formula_patterns = [
        r'สูตร.*เดียวกัน', r'เปรียบเทียบ.*สูตร', r'formula',
        r'สูตร.*ต่างกัน', r'same.*formula'
    ]
    for pattern in formula_patterns:
        if re.search(pattern, question, re.IGNORECASE):
            return 'formula'
    
    # Hierarchy patterns
    hierarchy_patterns = [
        r'hierarchy', r'ลำดับ', r'มาจาก.*สาร', r'vtm', r'gp', r'tp',
        r'subs', r'gpu', r'tpu', r'ระดับ'
    ]
    for pattern in hierarchy_patterns:
        if re.search(pattern, question, re.IGNORECASE):
            return 'hierarchy'
    
    return 'general'


# ==============================
# Hybrid Retrieval (Vector + Fulltext + RRF)
# ==============================

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
    
    Args:
        node_ids: List of node element IDs from hybrid_search
        depth: How many hops to traverse (default: 2)
    
    Returns:
        dict with 'nodes', 'relationships', and 'paths'
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
    Combines vector/fulltext search with graph structure exploration.
    
    Args:
        question: User question
        k: Number of seed nodes to retrieve
        depth: Traversal depth (uses GRAPH_TRAVERSAL_DEPTH if None)
    
    Returns:
        dict with 'seed_results', 'expanded_nodes', 'relationships'
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


# ==============================
# Run Cypher Against Neo4j
# ==============================

def run_cypher(cypher: str):
    try:
        drv = init_driver()
        with drv.session() as session:
            result = list(session.run(cypher))
            return result
    except Exception as e:
        return f"Cypher Error: {e}"


# ==============================
# Convert Neo4j Path → LLM Context
# ==============================

def format_path(path) -> str:
    nodes = path.nodes
    rels = path.relationships

    lines = []
    for i, rel in enumerate(rels):
        start = nodes[i]
        end = nodes[i + 1]

        start_label = list(start.labels)[0] if start.labels else "Node"
        end_label = list(end.labels)[0] if end.labels else "Node"

        lines.append(
            f"({start_label} {dict(start)}) -[:{rel.type}]-> ({end_label} {dict(end)})"
        )

    return "Path:\n" + "\n".join(lines)

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

def format_hybrid_results(results: list[dict]) -> str:
    if not results:
        return "ไม่พบข้อมูลที่เกี่ยวข้องในกราฟ"
        
    lines = []
    for item in results:
        node = item["node"]
        props = dict(node)
        
        # Remove embedding vectors from display to save tokens
        if "embedding_vec" in props:
            del props["embedding_vec"]
        if "embedding" in props:
            del props["embedding"]
        
        # Truncate long embedding_text to reduce context size
        if "embedding_text" in props and len(props["embedding_text"]) > 300:
            props["embedding_text"] = props["embedding_text"][:300] + "..."
            
        labels = list(node.labels)
        lines.append(f"Node Labels: {labels}\nProperties: {json.dumps(props, ensure_ascii=False)}")
        
    return "\n---\n".join(lines)


def format_graphrag_results(results: dict) -> str:
    """
    Format GraphRAG results including nodes and relationships.
    
    Args:
        results: dict from graphrag_search with seed_results, expanded_nodes, relationships
    
    Returns:
        Formatted string for LLM context
    """
    seed_results = results.get("seed_results", [])
    expanded_nodes = results.get("expanded_nodes", [])
    relationships = results.get("relationships", [])
    
    if not seed_results and not expanded_nodes:
        return "ไม่พบข้อมูลที่เกี่ยวข้องในกราฟ"
    
    sections = []
    
    # Section 1: Primary Results (Seed Nodes) - max 20
    if seed_results:
        seed_lines = ["=== PRIMARY RESULTS (ผลลัพธ์หลัก) ==="]
        seen_tmtids = set()
        for item in seed_results[:20]:
            node = item["node"]
            props = dict(node)
            
            # Dedupe by tmtid
            tmtid = props.get("tmtid", "")
            if tmtid and tmtid in seen_tmtids:
                continue
            if tmtid:
                seen_tmtids.add(tmtid)
            
            # Remove embedding vectors and long fields
            for key in ["embedding_vec", "embedding", "embedding_text"]:
                if key in props:
                    del props[key]
            
            # Truncate long string fields
            for key in ["strengths", "fsn"]:
                if key in props and isinstance(props[key], str) and len(props[key]) > 200:
                    props[key] = props[key][:200] + "..."
            
            labels = list(node.labels)
            seed_lines.append(f"[{', '.join(labels)}] {json.dumps(props, ensure_ascii=False)}")
        
        sections.append("\n".join(seed_lines))
    
    # Section 2: Relationships - max 20
    if relationships:
        rel_lines = ["=== RELATIONSHIPS (ความสัมพันธ์) ==="]
        for rel in relationships[:20]:
            start_label = rel['start_labels'][0] if rel['start_labels'] else 'Node'
            end_label = rel['end_labels'][0] if rel['end_labels'] else 'Node'
            rel_lines.append(
                f"({start_label}:{rel['start_name']}) -[:{rel['type']}]-> ({end_label}:{rel['end_name']})"
            )
        
        if len(relationships) > 20:
            rel_lines.append(f"... และอีก {len(relationships) - 20} relationships")
        
        sections.append("\n".join(rel_lines))
    
    # Section 3: Related Nodes (Expanded) - max 10
    if expanded_nodes:
        exp_lines = ["=== RELATED NODES (โหนดที่เกี่ยวข้อง) ==="]
        for item in expanded_nodes[:10]:
            node = item["node"]
            props = dict(node)
            
            # Remove heavy fields
            for key in ["embedding_vec", "embedding", "embedding_text"]:
                if key in props:
                    del props[key]
            
            labels = item.get("labels", [])
            exp_lines.append(f"[{', '.join(labels)}] {json.dumps(props, ensure_ascii=False)}")
        
        if len(expanded_nodes) > 10:
            exp_lines.append(f"... และอีก {len(expanded_nodes) - 10} related nodes")
        
        sections.append("\n".join(exp_lines))
    
    return "\n\n".join(sections)


def extract_structured_data(results: dict, question_type: str) -> dict:
    """
    Extract deterministic structured data from GraphRAG results.
    - Collect entities from: seed_results + expanded_nodes + relationship endpoints
    - Dedupe by (level, tmtid)
    - Prefer direct fields (trade_name/manufacturer); FSN parsing only as fallback
    - Keep evidence with elementIds for traceability
    """
    seed_results = results.get("seed_results", [])
    expanded_nodes = results.get("expanded_nodes", [])
    relationships = results.get("relationships", [])

    # -------------------------
    # Property filter + level scope
    # -------------------------
    if question_type == 'manufacturer':
        required_fields = ['trade_name', 'manufacturer', 'fsn', 'tmtid', 'level', 'container_text']
        allowed_levels = {'TP', 'TPU'}
        max_entities = 20
    elif question_type == 'ingredient':
        required_fields = ['fsn', 'active_ingredient', 'active_ingredients', 'strength', 'tmtid', 'level', 'dosageform']
        allowed_levels = {'GP', 'GPU', 'TP', 'TPU', 'VTM', 'SUBS'}
        max_entities = 30
    elif question_type == 'hierarchy':
        required_fields = ['level', 'fsn', 'tmtid']
        allowed_levels = {'SUBS', 'VTM', 'GP', 'GPU', 'TP', 'TPU'}
        max_entities = 40
    elif question_type == 'formula':
        # formula เทียบจาก FSN วงเล็บที่ 2 เป็นหลัก (fallback)
        required_fields = ['trade_name', 'manufacturer', 'fsn', 'tmtid', 'level']
        allowed_levels = {'TP', 'TPU', 'GP', 'GPU'}
        max_entities = 30
    else:
        required_fields = ['trade_name', 'manufacturer', 'fsn', 'tmtid', 'level']
        allowed_levels = {'TP', 'TPU', 'GP', 'GPU'}
        max_entities = 30

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
    # FSN fallback parsers
    # -------------------------
    def fallback_trade_and_mfr_from_fsn(fsn: str) -> tuple[str | None, str | None]:
        """
        FSN รูปแบบทั่วไป:
        BRAND (MANUFACTURER) (INGREDIENTS) ... (TP/TPU)
        - trade_name_fallback: ก่อน '(' แรก
        - manufacturer_fallback: วงเล็บแรก
        """
        if not fsn or '(' not in fsn:
            return None, None
        brand = fsn.split('(')[0].strip()
        m = re.search(r'\(([^)]+)\)', fsn)
        mfr = m.group(1).strip() if m else None
        return (brand or None), (mfr or None)

    def fallback_ingredients_from_fsn(fsn: str) -> str | None:
        """
        ดึงวงเล็บที่ 2 (จากซ้าย) เป็น ingredients (fallback)
        """
        if not fsn:
            return None
        parts = re.findall(r'\(([^)]+)\)', fsn)
        if len(parts) >= 2:
            return parts[1].strip()
        return None

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
                b, mfr = fallback_trade_and_mfr_from_fsn(fsn)
                if b:
                    entity["trade_name_fallback"] = b
                if not entity.get("manufacturer") and mfr:
                    entity["manufacturer_fallback"] = mfr

            # manufacturer fallback เฉพาะ TP/TPU
            if level in ("TP", "TPU") and not entity.get("manufacturer"):
                _, mfr = fallback_trade_and_mfr_from_fsn(fsn)
                if mfr:
                    entity["manufacturer_fallback"] = mfr

            # formula/ingredient fallback
            if question_type in ("formula", "ingredient") and "ingredients_fallback" not in entity:
                ing = fallback_ingredients_from_fsn(fsn)
                if ing:
                    entity["ingredients_fallback"] = ing

        entities.append(entity)

    # -------------------------
    # Sort & limit (manufacturer: TP ก่อน TPU)
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



def format_structured_context(structured_data: dict) -> str:
    """
    Format structured data as JSON for LLM context.
    """
    return json.dumps(structured_data, ensure_ascii=False, indent=2)


# ==============================
# LLM QA USING CONTEXT
# ==============================

def ask_ollama_structured(question: str, structured_data: dict) -> str:
    """
    Ask Ollama using structured JSON data (formatter role).
    Enforces:
    - Use ONLY JSON
    - List ALL entities provided (no omission)
    - Output table only (no narrative)
    """
    entities = structured_data.get("entities", [])
    if not entities:
        return "ไม่พบข้อมูลในกราฟ"

    system_prompt = """คุณเป็น "Formatter" สำหรับข้อมูลยา TMT (Thai Medicinal Terminology)

    ข้อกำหนด (ต้องทำตาม):
    - ตอบเป็นภาษาไทยเท่านั้น (ยกเว้นชื่อยา/บริษัท/หน่วย mg, g, mL)
    - ห้ามขึ้นต้น/แทรกภาษาอังกฤษ เช่น "Based on the provided JSON"
    - แสดงผลเป็นรายการ bullet "ความแรง" ที่พบจาก fsn/strength ใน JSON เท่านั้น
    - ห้ามตีความเพิ่มนอก JSON
    """

    json_context = json.dumps(structured_data, ensure_ascii=False, indent=2)

    user_message = f"""คำถาม: {question}

JSON:
```json
{json_context}
```"""

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
        "options": {
            "num_ctx": 8192,
            "temperature": 0
        }
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    except requests.exceptions.HTTPError as e:
        print(f"Ollama HTTP Error: {e}")
        print(f"Response: {e.response.text if e.response else 'No response'}")
        return "เกิดข้อผิดพลาดจาก LLM"
    except Exception as e:
        print(f"Ollama Error: {e}")
        return "เกิดข้อผิดพลาดในการเชื่อมต่อ LLM"

# ==============================
# MAIN PROGRAM
# ==============================

LOG_PATH = "./logs/ragas_data.jsonl"
pathlib.Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)

def log_interaction(question: str, results: dict, answer: str):
    contexts = []
    # Combine seed and expanded nodes for logging
    all_nodes = results.get("seed_results", []) + results.get("expanded_nodes", [])
    
    for item in all_nodes:
        node = item["node"]
        props = dict(node)

        text_parts = []
        if "fsn" in props:
            text_parts.append(str(props["fsn"]))
        if "embedding_text" in props:
            text_parts.append(str(props["embedding_text"]))
        if "trade_name" in props:
            text_parts.append(f"trade_name: {props['trade_name']}")
        if "manufacturer" in props:
            text_parts.append(f"manufacturer: {props['manufacturer']}")

        if text_parts:
            contexts.append(" | ".join(text_parts))

    record = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "contexts": contexts,
        "answer": answer,
    }

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    print("Starting main...")
    try:
        validate_env()
    except RuntimeError as e:
        print(e)
        return

    init_driver()
    print("=== Neo4j + Ollama Hybrid Retriever Demo ===")
    print(f"Running on {LLM_MODEL}")
    
    # Setup Indexes
    try:
        setup_indexes()
    except Exception as e:
        print(f"Warning: Could not setup indexes: {e}")
        
    print("พิมพ์คำถาม หรือ 'exit' เพื่อออก")

    try:
        while True:
            q = input("\nถาม: ").strip()
            if q.lower() in ("exit", "quit"):
                break

            print(f"\n→ ค้นหาแบบ GraphRAG (Hybrid + Relationship Traversal, depth={GRAPH_TRAVERSAL_DEPTH}) ...")
            
            # 1. Classify
            q_type = classify_question(q)
            print(f"   Question Type: {q_type}")

            results = graphrag_search(q, k=10, depth=GRAPH_TRAVERSAL_DEPTH)
            
            # 2. Extract Structured Data
            structured = extract_structured_data(results, q_type)
            # Debug context size (approx)
            json_ctx = json.dumps(structured, ensure_ascii=False)
            print(f"   Structured Data Size: {len(json_ctx)} chars")

            # Show search stats
            num_seeds = len(results.get("seed_results", []))
            num_expanded = len(results.get("expanded_nodes", []))
            num_rels = len(results.get("relationships", []))
            print(f"   Found: {num_seeds} primary nodes, {num_expanded} related nodes, {num_rels} relationships")

            # Debug structured data
            print("\nDebug Structured Data:")
            print(json.dumps(structured, ensure_ascii=False, indent=2)) 

            print("\n→ ส่งให้ LLM ตอบ (Structured Mode) ...")
            answer = ask_ollama_structured(q, structured)
            print("\nตอบ:\n", answer)

            log_interaction(q, results, answer)
    except (KeyboardInterrupt, EOFError):
        print("\nออกจากโปรแกรม")


if __name__ == "__main__":
    main()