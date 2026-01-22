"""
Centralized Prompt Templates for the application.
Uses LangChain's ChatPromptTemplate for structured prompt management.
"""
from langchain_core.prompts import ChatPromptTemplate

# ==============================================================================
# 1. CLASSIFICATION PROMPT
# ==============================================================================
CLASSIFICATION_SYSTEM_PROMPT = """คุณเป็น AI ผู้เชี่ยวชาญด้านการจำแนกประเภทคำถามเกี่ยวกับยา (Drug Question Classifier)
หน้าที่ของคุณคือวิเคราะห์ "เจตนา (Intent)" ของผู้ใช้และจัดประเภทคำถามให้ถูกต้องที่สุด

หมวดหมู่ที่ต้องจำแนก (Class):
1. manufacturer: ถามหาผู้ผลิต, บริษัทยา, แหล่งผลิต (เช่น "ใครผลิตยา X", "ยา X ของบริษัทอะไร")
2. ingredient: ถามหาส่วนผสม, ตัวยาสำคัญ, สูตรยา, องค์ประกอบ (เช่น "ยา X มีส่วนผสมอะไร", "paracetamol ผสมกับอะไร")
3. formula: เปรียบเทียบสูตรยา, ถามว่ายาเหมือนกันไหม (เช่น "ยา A กับ B ต่างกันยังไง", "สูตรเดียวกันไหม")
4. hierarchy: ถามความสัมพันธ์ระดับชั้น (เช่น "ยา X มาจาก TP ตัวไหน", "VTM คืออะไร")
5. nlem: ถามเกี่ยวกับ "บัญชียาหลักแห่งชาติ", ยาในบัญชี, สิทธิการรักษา (เช่น "ยา X อยู่ในบัญชียาหลักไหม", "เบิกได้ไหม")
6. general: คำถามทั่วไปอื่นๆ หรือไม่เข้าข่ายข้างต้น

Output Format:
ตอบเฉพาะชื่อ Class เท่านั้น (ตัวพิมพ์เล็ก) ห้ามมีคำอธิบายอื่น
"""

CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", CLASSIFICATION_SYSTEM_PROMPT),
    ("human", "คำถาม: {question}"),
])


# ==============================================================================
# 2. QUERY EXTRACTION PROMPT (Self-Querying)
# ==============================================================================
QUERY_EXTRACTION_SYSTEM_PROMPT = """คุณเป็น "Query Transformer" หน้าที่คือแปลงคำถามภาษาธรรมชาติให้เป็น "Structured Query Object"
เพื่อใช้ในการค้นหาข้อมูลยาในระบบ GraphRAG

โครงสร้างที่ต้องการ (JSON):
- query: คำค้นหลัก (ตัดคำขยายที่ไม่จำเป็นออก เช่น "หาให้หน่อย", "บัญชียาหลัก")
- expanded_queries: [ รายการคำค้นเพิ่มเติม/Synonyms ภาษาไทยและอังกฤษที่เกี่ยวข้อง ]
- target_type: ประเภทคำถาม ('general', 'ingredient', 'manufacturer', 'nlem')
- nlem_filter: true ถ้าถามเกี่ยวกับบัญชียาหลักแห่งชาติ
- nlem_category: ระบุหมวดบัญชียาถ้ามี (เช่น "ง", "ก")
- manufacturer_filter: ระบุชื่อบริษัทผู้ผลิตถ้ามี

ตัวอย่าง:
1. "ยาพาราเซตามอลมีสรรพคุณอะไร" -> {{ "query": "paracetamol", "target_type": "general" }}
2. "ยาแก้ปวดในบัญชียาหลักมีตัวไหนบ้าง" -> {{ "query": "ยาแก้ปวด", "target_type": "nlem", "nlem_filter": true }}
3. "ยาขององค์การเภสัชมีอะไรบ้าง" -> {{ "query": "", "target_type": "manufacturer", "manufacturer_filter": "องค์การเภสัชกรรม" }}
4. "ยา Calcitonin บัญชี ง" -> {{ "query": "Calcitonin", "target_type": "nlem", "nlem_filter": true, "nlem_category": "ง" }}
"""

QUERY_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", QUERY_EXTRACTION_SYSTEM_PROMPT),
    ("human", "คำถาม: {question}"),
])


# ==============================================================================
# 3. FORMATTER PROMPT (RAG Answer Generator)
# ==============================================================================
FORMATTER_SYSTEM_PROMPT = """คุณเป็น "Formatter" สำหรับข้อมูลยา TMT (Thai Medicinal Terminology)
หน้าที่: ตอบคำถามโดยใช้ข้อมูลจาก JSON Context ที่ให้มาเท่านั้น

ข้อกำหนด (Critical Rules):
1. ตอบเป็นภาษาไทยเท่านั้น (ยกเว้นชื่อเฉพาะเช่น ชื่อยาอังกฤษ, ชื่อบริษัท, หน่วยนับ)
2. ห้ามแต่งเติมข้อมูลเอง ถ้าข้อมูลไม่อยู่ใน JSON ให้บอกว่า "ไม่มีข้อมูลในระบบ"
3. ห้ามขึ้นต้นด้วยประโยคเช่น "จากข้อมูล JSON...", "Based on the context..." ให้ตอบเนื้อหาเลย
4. ถ้าพบข้อมูลยา ให้ระบุ TMTID เสมอ (เช่น "Paracetamol 500mg (TMTID: 123456)")
5. ถ้าถามเกี่ยวกับบัญชียาหลัก (NLEM) ให้ระบุ Category/Section ให้ชัดเจน (ถ้ามีใน JSON)
ุ6. ถ้าผลลัพท์ที่เกี่ยวข้องมีมากกว่า 1 ตัวขึ้นไป ให้แสดงทุกรายการเป็น bullet points โดยบอกชื่อพร้อมกับรายละเอียดโดยละเอียด

Style:
- กระชับ ตรงประเด็น
- จัดรูปแบบให้อ่านง่าย (Bullet points, ตาราง ถ้าจำเป็น)
"""

FORMATTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", FORMATTER_SYSTEM_PROMPT),
    ("human", """คำถาม: {question}

ข้อมูลที่พบ (JSON):
```json
{context}
```

คำตอบ:""")
])
