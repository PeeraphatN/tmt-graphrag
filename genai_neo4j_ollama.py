from neo4j import GraphDatabase
import requests
import textwrap
import json

# ==============================
# CONFIG
# ==============================

NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "REDACTED_NEO4J_PASSWORD"

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
LLM_MODEL = "qwen2.5:7b-instruct"
EMBED_MODEL = "bge-m3:latest"

VECTOR_INDEX_NAME = "tmt_embedding_index"
FULLTEXT_INDEX_NAME = "tmt_fulltext_index"
EMBEDDING_DIM = 1024

print("Connecting to Neo4j...")
driver = None

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

def main():
    print("Starting main...")
    init_driver()
    print("=== Neo4j + Ollama Hybrid Retriever Demo ===")
    
    # Setup Indexes
    try:
        setup_indexes()
    except Exception as e:
        print(f"Warning: Could not setup indexes: {e}")
        
    print("พิมพ์คำถาม หรือ 'exit' เพื่อออก")

    while True:
        q = input("\nถาม: ").strip()
        if q.lower() in ("exit", "quit"):
            break

        print("\n→ ค้นหาแบบ Hybrid (Vector + Fulltext) ...")
        results = hybrid_search(q, k=20)
        ctx = format_hybrid_results(results)
        
        # Check context size
        ctx_size = len(ctx)
        print(f"Context size: {ctx_size} characters")
        if ctx_size > 15000:
            print("Warning: Context is large, may cause issues with LLM")
        
        print(ctx)

        print("\n→ ส่งให้ LLM ตอบ ...")
        answer = ask_ollama(q, ctx)
        print("\nตอบ:\n", answer)


if __name__ == "__main__":
    main()
