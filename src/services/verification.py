from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.models.llm import get_llm
from src.config import CLASSIFICATION_MODEL

def verify_semantic_match(new_question: str, cached_question: str, cached_answer: str) -> bool:
    """
    Verify if the cached question/answer pair is valid for the new question.
    Returns True if valid (YES), False otherwise (NO).
    """
    
    # Use the lightweight classification model for speed
    llm = get_llm(model=CLASSIFICATION_MODEL, temperature=0.0)

    prompt = ChatPromptTemplate.from_template("""
    You are a semantic verification assistant.
    Your task is to determine if the "Cached Question" and "Cached Answer" can validly answer the "New Question".
    
    Rules:
    1. If the New Question asks for exactly the same information as the Cached Question (even if phrased differently), return YES.
    2. If the New Question asks about the SAME TOPIC but a DIFFERENT ASPECT (e.g., "indications" vs "side effects"), return NO.
    3. If the New Question asks about a DIFFERENT DRUG or TOPIC entirely, return NO.
    4. Ignore minor typos or grammatical differences.

    New Question: {new_q}
    Cached Question: {cached_q}
    Cached Answer (Context): {cached_a}

    Answer strictly with YES or NO.
    """)

    chain = prompt | llm | StrOutputParser()
    
    try:
        result = chain.invoke({
            "new_q": new_question,
            "cached_q": cached_question,
            # Truncate answer to save context window if too long
            "cached_a": cached_answer[:500] if cached_answer else ""
        })
        
        cleaned_result = result.strip().upper()
        # print(f"   [Verify] '{new_question}' vs '{cached_question}' -> {cleaned_result}")
        
        return "YES" in cleaned_result
    except Exception as e:
        print(f"   [Verify] Error: {e}")
        # Fail safe: if verification fails, assume False to be safe (or True if optimistic)
        # Choosing False to prevent bad cache hits
        return False
