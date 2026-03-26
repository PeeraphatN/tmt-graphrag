"""
Embeddings wrapper using LangChain OllamaEmbeddings.
Replaces raw requests.post() calls for embedding generation.
"""
from langchain_ollama import OllamaEmbeddings
from src.config import OLLAMA_EMBED_URL, EMBED_MODEL

# Singleton instance
_embeddings_instance = None


def get_embeddings(model: str = None) -> OllamaEmbeddings:
    """
    Get an OllamaEmbeddings instance (singleton pattern).
    
    Args:
        model: Model name (default: from config)
    
    Returns:
        OllamaEmbeddings instance
    """
    global _embeddings_instance
    
    if _embeddings_instance is None:
        # Extract base URL from OLLAMA_EMBED_URL
        # Example: "http://localhost:11434/api/embeddings" -> "http://localhost:11434"
        base_url = OLLAMA_EMBED_URL.replace("/api/embeddings", "")
        
        _embeddings_instance = OllamaEmbeddings(
            model=model or EMBED_MODEL,
            base_url=base_url,
        )
    
    return _embeddings_instance


def embed_text(text: str) -> list[float]:
    """
    Generate embedding for a single text.
    
    Args:
        text: Text to embed
    
    Returns:
        Embedding vector as list of floats
    """
    embeddings = get_embeddings()
    return embeddings.embed_query(text)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for multiple texts (batch).
    
    Args:
        texts: List of texts to embed
    
    Returns:
        List of embedding vectors
    """
    embeddings = get_embeddings()
    return embeddings.embed_documents(texts)
