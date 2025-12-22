import requests
import json
from src.config import OLLAMA_EMBED_URL, OLLAMA_URL, LLM_MODEL, EMBED_MODEL

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

def format_structured_context(structured_data: dict) -> str:
    """
    Format structured data as JSON for LLM context.
    """
    return json.dumps(structured_data, ensure_ascii=False, indent=2)

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
    - หากมีข้อมูลที่ตรงกับคำถาม ให้แสดงข้อมูลที่พบจาก JSON เท่านั้น
    - หากไม่มีข้อมูลที่ตรงกับคำถาม ให้แสดงข้อมูลที่พบจาก JSON เท่านั้น และอธิบายว่าไม่มีข้อมูลที่ตรงกับคำถาม
    - หากเจอข้อมูลที่ตรงกับคำถามให้แนบ tmtid ของแต่ละข้อมูลที่พบจาก JSON มาด้วย
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
