"""
AQT Service (Advance Query Transformation).
Extracts structured query and filters from natural language input.
"""
from langchain_core.runnables import Runnable
from src.models.llm import get_llm
from src.prompts.templates import QUERY_EXTRACTION_PROMPT
from src.schemas.query import GraphRAGQuery
from src.config import CLASSIFICATION_MODEL

def get_query_transform_chain() -> Runnable:
    """
    Creates a chain that takes {"question": str} and returns a GraphRAGQuery object.
    """
    # Use CLASSIFICATION_MODEL for query transformation
    llm = get_llm(model=CLASSIFICATION_MODEL, temperature=0)
    
    # Since Ollama structured output can be tricky, we'll use the prompt to force JSON 
    # and default Pydantic parsing.
    structured_llm = llm.with_structured_output(GraphRAGQuery)
    
    chain = QUERY_EXTRACTION_PROMPT | structured_llm
    
    return chain

def transform_query(question: str) -> GraphRAGQuery:
    """Run transformation immediately."""
    chain = get_query_transform_chain()
    try:
        return chain.invoke({"question": question})
    except Exception as e:
        print(f"Query Transform Error: {e}")
        # Fallback
        return GraphRAGQuery(query=question, target_type="general")
