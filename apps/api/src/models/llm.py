"""
LLM wrapper using LangChain ChatOllama.
Replaces raw requests.post() calls to Ollama API.
"""
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from src.config import OLLAMA_URL, LLM_MODEL

# Singleton instance
_llm_instance = None


def get_llm(
    model: str = None,
    temperature: float = 0,
    num_ctx: int = 8192,
) -> ChatOllama:
    """
    Get a ChatOllama instance (singleton pattern).
    
    Args:
        model: Model name (default: from config)
        temperature: Generation temperature (default: 0 for deterministic)
        num_ctx: Context window size (default: 8192)
    
    Returns:
        ChatOllama instance
    """
    global _llm_instance
    
    if _llm_instance is None:
        # Extract base URL from OLLAMA_URL (remove /api/chat if present)
        base_url = OLLAMA_URL.replace("/api/chat", "")
        
        _llm_instance = ChatOllama(
            model=model or LLM_MODEL,
            base_url=base_url,
            temperature=temperature,
            num_ctx=num_ctx,
        )
    
    return _llm_instance


def ask_llm(question: str, context: str, system_prompt: str = None) -> str:
    """
    Convenience function to ask the LLM a question with context.
    
    Args:
        question: User question
        context: JSON context from retrieval
        system_prompt: Optional system prompt override
    
    Returns:
        LLM response as string
    """
    llm = get_llm()
    
    default_system = """คุณเป็น "Formatter" สำหรับข้อมูลยา TMT (Thai Medicinal Terminology)

ข้อกำหนด (ต้องทำตาม):
- ตอบเป็นภาษาไทยเท่านั้น (ยกเว้นชื่อยา/บริษัท/หน่วย mg, g, mL)
- ห้ามขึ้นต้น/แทรกภาษาอังกฤษ เช่น "Based on the provided JSON"
- หากมีข้อมูลที่ตรงกับคำถาม ให้แสดงข้อมูลที่พบจาก JSON เท่านั้น
- หากไม่มีข้อมูลที่ตรงกับคำถาม ให้อธิบายว่าไม่มีข้อมูลที่ตรงกับคำถาม
- หากเจอข้อมูลที่ตรงกับคำถามให้แนบ tmtid ของแต่ละข้อมูลที่พบจาก JSON มาด้วย
"""
    
    messages = [
        SystemMessage(content=system_prompt or default_system),
        HumanMessage(content=f"คำถาม: {question}\n\nJSON:\n```json\n{context}\n```")
    ]
    
    response = llm.invoke(messages)
    return response.content
