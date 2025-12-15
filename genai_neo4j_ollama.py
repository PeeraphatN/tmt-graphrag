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

        # 3. Fulltext Search
        fulltext_query = f"""
        CALL db.index.fulltext.queryNodes($index_name, $search_text, {{limit: $k}})
        YIELD node, score
        RETURN node, score
        """
        try:
            t_recs = session.run(fulltext_query, index_name=FULLTEXT_INDEX_NAME, search_text=question, k=k)
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
    
    # Section 1: Primary Results (Seed Nodes)
    if seed_results:
        seed_lines = ["=== PRIMARY RESULTS (ผลลัพธ์หลัก) ==="]
        for item in seed_results:
            node = item["node"]
            props = dict(node)
            
            # Remove embedding vectors
            for key in ["embedding_vec", "embedding"]:
                if key in props:
                    del props[key]
            
            # Truncate long text
            if "embedding_text" in props and len(props["embedding_text"]) > 300:
                props["embedding_text"] = props["embedding_text"][:300] + "..."
            
            labels = list(node.labels)
            seed_lines.append(f"[{', '.join(labels)}] {json.dumps(props, ensure_ascii=False)}")
        
        sections.append("\n".join(seed_lines))
    
    # Section 2: Relationships
    if relationships:
        rel_lines = ["=== RELATIONSHIPS (ความสัมพันธ์) ==="]
        for rel in relationships[:20]:  # Limit to 20 relationships
            start_label = rel['start_labels'][0] if rel['start_labels'] else 'Node'
            end_label = rel['end_labels'][0] if rel['end_labels'] else 'Node'
            rel_lines.append(
                f"({start_label}:{rel['start_name']}) -[:{rel['type']}]-> ({end_label}:{rel['end_name']})"
            )
        
        if len(relationships) > 20:
            rel_lines.append(f"... และอีก {len(relationships) - 20} relationships")
        
        sections.append("\n".join(rel_lines))
    
    # Section 3: Related Nodes (Expanded)
    if expanded_nodes:
        exp_lines = ["=== RELATED NODES (โหนดที่เกี่ยวข้อง) ==="]
        for item in expanded_nodes[:10]:  # Limit to 10 expanded nodes
            node = item["node"]
            props = dict(node)
            
            # Remove embedding vectors
            for key in ["embedding_vec", "embedding"]:
                if key in props:
                    del props[key]
            
            # Truncate long text
            if "embedding_text" in props and len(props["embedding_text"]) > 200:
                props["embedding_text"] = props["embedding_text"][:200] + "..."
            
            labels = item.get("labels", [])
            exp_lines.append(f"[{', '.join(labels)}] {json.dumps(props, ensure_ascii=False)}")
        
        if len(expanded_nodes) > 10:
            exp_lines.append(f"... และอีก {len(expanded_nodes) - 10} related nodes")
        
        sections.append("\n".join(exp_lines))
    
    return "\n\n".join(sections)


# ==============================
# LLM QA USING CONTEXT
# ==============================

def ask_ollama(question: str, graph_context: str) -> str:

    if graph_context.strip() == "ไม่พบข้อมูลที่เกี่ยวข้องในกราฟ":
        return "ไม่พบข้อมูลในกราฟ"

    system_prompt = f"""คุณเป็นผู้เชี่ยวชาญด้านข้อมูลยาของประเทศไทย ที่ทำงานกับฐานข้อมูล TMT (Thai Medicinal Terminology)

🎯 หน้าที่หลัก:
ตอบคำถามโดยวิเคราะห์ข้อมูลจาก GRAPH CONTEXT ที่ให้มาเท่านั้น ห้ามใช้ความรู้ภายนอก

📊 โครงสร้างข้อมูล TMT (Hierarchy):
SUBS → VTM → GP → GPU → TP → TPU
(สารเคมี → สารออกฤทธิ์ → ยาสามัญ → หน่วยจ่ายสามัญ → ยาการค้า → หน่วยจ่ายการค้า)

ระดับข้อมูล:
• SUBS (Substance): สารเคมีพื้นฐาน - properties: name, fsn, tmtid
• VTM (Virtual Therapeutic Moiety): สารออกฤทธิ์ - properties: activeingredient(s), fsn
• GP (Generic Product): ยาสามัญ - properties: strength, dosageform, fsn
• GPU (Generic Product Use Unit): หน่วยจ่ายสามัญ - properties: genericname, dispunit, contunit
• TP (Trade Product): ยาการค้า ⭐ สำคัญที่สุด - properties: trade_name, manufacturer, fsn
• TPU (Trade Product Use Unit): หน่วยจ่ายการค้า - properties: fsn

🔍 FSN PARSING RULES (สำคัญมาก!):
FSN (Fully Specified Name) สำหรับ TP มีโครงสร้าง:
"<BRAND> (<MANUFACTURER>) (<INGREDIENTS>) <DOSAGE_FORM> (TP)"

วิธีแยกส่วนประกอบ:
- วงเล็บที่ 1: ชื่อผู้ผลิต (manufacturer)
- วงเล็บที่ 2: ส่วนผสม/สูตรยา (ingredients) ⭐ ใช้เปรียบเทียบสูตร
- วงเล็บสุดท้าย: ระดับข้อมูล (TP)

ตัวอย่าง:
"AMINOLEBAN (ไทยโอซูก้า) (amino acids 8 g/100 mL) solution for infusion (TP)"
→ Brand: AMINOLEBAN
→ Manufacturer: ไทยโอซูก้า
→ Ingredients: amino acids 8 g/100 mL
→ Dosage Form: solution for infusion

📋 กฎการทำงาน (STRICT RULES):

1️⃣ GROUNDING - ยึดติดกับข้อมูลที่มี
   ✅ ใช้เฉพาะข้อมูลจาก GRAPH CONTEXT
   ✅ อ้างอิง properties: fsn, manufacturer, trade_name, tmtid, level
   ✅ ตอนเป็นภาษาไทยเท่านั้น ยกเว้น properties
   ❌ ห้ามใช้ความรู้ทั่วไปเกี่ยวกับยา
   ❌ ห้ามสร้างชื่อบริษัท/ยาที่ไม่มีใน context
   ❌ ห้ามอธิบายสรรพคุณ/กลไกยา/ผลข้างเคียง

2️⃣ MANUFACTURER QUERIES - คำถามเกี่ยวกับบริษัท
   เมื่อถูกถามว่า "บริษัท X ผลิต/จำหน่ายยาอะไรบ้าง":
   
   ขั้นตอน:
   a) ค้นหา nodes ทั้งหมดที่ level="TP" และ manufacturer ตรงกับชื่อบริษัท
   b) แสดงรายการยาทั้งหมดที่พบ (ห้ามตัดทิ้ง)
   c) สำหรับแต่ละยา แสดง:
      - ชื่อการค้า (trade_name)
      - ส่วนผสม (แยกจาก FSN วงเล็บที่ 2)
      - รูปแบบยา (dosage form)
      - TMT ID
   
   รูปแบบคำตอบที่ดี:
   "บริษัท [ชื่อ] ผลิตยาทั้งหมด X รายการ:
   
   1. [ชื่อการค้า]
      - ส่วนผสม: [ingredients]
      - รูปแบบ: [dosage form]
      - TMT ID: [tmtid]
   
   2. [ชื่อการค้า]
      ..."

3️⃣ FORMULA COMPARISON - เปรียบเทียบสูตรยา
   เมื่อถูกถามว่า "ยา A และ B ใช้สูตรเดียวกันไหม":
   
   ขั้นตอน:
   a) แยกส่วนผสมจาก FSN ของแต่ละยา (วงเล็บชุดที่ 2)
   b) เปรียบเทียบข้อความส่วนผสมแบบตรงตัว (case-insensitive, trim whitespace)
   c) ตอบชัดเจน:
      - "ใช้สูตรเดียวกัน" + แสดงส่วนผสม
      - "ใช้สูตรต่างกัน" + แสดงส่วนผสมของทั้งสองตัว
   
   ⚠️ การเปรียบเทียบส่วนผสมจาก FSN ถือว่าเป็นการใช้ข้อมูลจากกราฟ ไม่ใช่การเดา

4️⃣ INGREDIENT EXTRACTION - การแยกส่วนผสม
   วิธีแยกส่วนผสมจาก FSN:
   
   Pattern: "<BRAND> (<MANUFACTURER>) (<INGREDIENTS>) <FORM> (TP)"
   
   ขั้นตอน:
   1. หาวงเล็บทั้งหมดใน FSN
   2. วงเล็บที่ 2 (นับจากซ้าย) = ส่วนผสม
   3. ดึงข้อความออกมา (ไม่รวมวงเล็บ)
   
   ตัวอย่าง:
   FSN: "ACETAR (ACETATE RINGER'S INJECTION) (ไทยโอซูก้า) (calcium chloride dihydrate 20 mg/100 mL + potassium chloride 30 mg/100 mL + sodium acetate trihydrate 380 mg/100 mL + sodium chloride 600 mg/100 mL) solution for infusion (TP)"
   → Ingredients: "calcium chloride dihydrate 20 mg/100 mL + potassium chloride 30 mg/100 mL + sodium acetate trihydrate 380 mg/100 mL + sodium chloride 600 mg/100 mL"

5️⃣ RESPONSE QUALITY - คุณภาพคำตอบ
   ✅ ตอบเป็นภาษาไทยที่เป็นธรรมชาติ
   ✅ ใช้ชื่อยา/บริษัทภาษาอังกฤษตามที่ปรากฏใน context (ไม่ต้องแปล)
   ✅ จัดรูปแบบให้อ่านง่าย (ใช้ bullet points, numbering)
   ✅ แสดงข้อมูลครบถ้วนตามที่มี
   ❌ ห้ามตัดข้อมูลทิ้ง (เช่น แสดงแค่บางรายการ)
   ❌ ห้ามสรุปแบบคร่าวๆ ถ้าถามรายละเอียด

6️⃣ EDGE CASES - กรณีพิเศษ
   - ถ้าไม่พบข้อมูล: ตอบ "ไม่พบข้อมูลในกราฟ"
   - ถ้าข้อมูลไม่ครบ: บอกว่าข้อมูลใดที่มี และข้อมูลใดที่ไม่มี
   - ถ้าคำถามคลุมเครือ: ตอบตามข้อมูลที่มี และบอกว่าตีความคำถามอย่างไร

🧠 วิธีคิดก่อนตอบ (REASONING PROCESS):
1. อ่านคำถาม → ระบุว่าถามเรื่องอะไร (บริษัท? สูตร? ส่วนผสม?)
2. สแกน GRAPH CONTEXT → หา nodes ที่เกี่ยวข้อง
3. แยก properties ที่ต้องใช้ → fsn, manufacturer, trade_name, level
4. ประมวลผลตามกฎ → แยกส่วนผสม, เปรียบเทียบ, จัดกลุ่ม
5. จัดรูปแบบคำตอบ → ชัดเจน อ่านง่าย ครบถ้วน

--- GRAPH CONTEXT ---
{graph_context}
--- END CONTEXT ---
"""

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        "stream": False,
        "options": {
            "num_ctx": 8192,
            "temperature": 0.1
        }
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    except requests.exceptions.HTTPError as e:
        print(f"Ollama HTTP Error: {e}")
        print(f"Response: {e.response.text if e.response else 'No response'}")
        return "เกิดข้อผิดพลาดจาก LLM (context อาจใหญ่เกินไป)"
    except Exception as e:
        print(f"Ollama Error: {e}")
        return "เกิดข้อผิดพลาดในการเชื่อมต่อ LLM"




# ==============================
# MAIN PROGRAM
# ==============================

LOG_PATH = "./logs/ragas_data.jsonl"
pathlib.Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)

def log_interaction(question: str, results: list[dict], answer: str):
    contexts = []
    for item in results:
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
            results = graphrag_search(q, k=10, depth=GRAPH_TRAVERSAL_DEPTH)
            ctx = format_graphrag_results(results)

            # Show search stats
            num_seeds = len(results.get("seed_results", []))
            num_expanded = len(results.get("expanded_nodes", []))
            num_rels = len(results.get("relationships", []))
            print(f"   Found: {num_seeds} primary nodes, {num_expanded} related nodes, {num_rels} relationships")

            # Check context size
            ctx_size = len(ctx)
            print(f"   Context size: {ctx_size} characters")
            if ctx_size > 15000:
                print("   Warning: Context is large, may cause issues with LLM")

            # Uncomment to debug context
            # print(ctx)

            print("\n→ ส่งให้ LLM ตอบ ...")
            answer = ask_ollama(q, ctx)
            print("\nตอบ:\n", answer)

            log_interaction(q, results, answer)
    except (KeyboardInterrupt, EOFError):
        print("\nออกจากโปรแกรม")


if __name__ == "__main__":
    main()