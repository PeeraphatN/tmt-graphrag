"""
Formatting Service.
Uses LLM to generate the final answer based on retrieved context.
"""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from src.models.llm import get_llm
from src.prompts.templates import FORMATTER_PROMPT
import json

def get_formatter_chain() -> Runnable:
    """
    Creates a chain that takes {"question": str, "context": str/dict} 
    and returns the final answer (str).
    """
    llm = get_llm(temperature=0)
    
    chain = (
        {
            "question": lambda x: x["question"],
            "context": lambda x: x["context"] if isinstance(x["context"], str) else json.dumps(x["context"], ensure_ascii=False, indent=2)
        }
        | FORMATTER_PROMPT 
        | llm 
        | StrOutputParser()
    )
    
    return chain

def format_answer_llm(question: str, structured_data: dict) -> str:
    """
    Helper function to run formatting immediately.
    """
    chain = get_formatter_chain()
    try:
        return chain.invoke({"question": question, "context": structured_data})
    except Exception as e:
        print(f"Formatter Error: {e}")
        return "เกิดข้อผิดพลาดในการสร้างคำตอบ"
