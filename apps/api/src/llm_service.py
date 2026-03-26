"""
LLM Service - Now powered by LangChain wrappers.
This module provides backward-compatible functions for the rest of the codebase.
"""
import json
from src.models.embeddings import embed_text
from src.models.llm import get_llm, ask_llm


def get_embedding(text: str) -> list[float]:
    """
    Generate embedding using LangChain OllamaEmbeddings.
    (Backward-compatible wrapper)
    """
    try:
        return embed_text(text)
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
    Now uses LangChain ChatOllama under the hood.
    """
    entities = structured_data.get("entities", [])
    if not entities:
        return "ไม่พบข้อมูลในกราฟ"

    json_context = json.dumps(structured_data, ensure_ascii=False, indent=2)
    
    try:
        return ask_llm(question, json_context)
    except Exception as e:
        print(f"LLM Error: {e}")
        return "เกิดข้อผิดพลาดในการเชื่อมต่อ LLM"
