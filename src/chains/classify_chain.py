"""
Classification Chain
Uses LLM to determine the type of user question.
"""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from src.models.llm import get_llm
from src.prompts.templates import CLASSIFICATION_PROMPT

def get_classify_chain() -> Runnable:
    """
    Creates a chain that takes {"question": str} and returns a classification label (str).
    Categories: manufacturer, ingredient, formula, hierarchy, nlem, general
    """
    llm = get_llm(temperature=0) # Deterministic
    
    chain = (
        CLASSIFICATION_PROMPT 
        | llm 
        | StrOutputParser() 
        | (lambda x: x.strip().lower()) # Normalize output
    )
    
    return chain

def classify_question_llm(question: str) -> str:
    """
    Helper function to run classification immediately.
    """
    chain = get_classify_chain()
    try:
        return chain.invoke({"question": question})
    except Exception as e:
        print(f"Classification Error: {e}")
        return "general" # Fallback
