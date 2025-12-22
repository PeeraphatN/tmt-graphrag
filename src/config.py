import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ==============================
# CONFIG
# ==============================

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

OLLAMA_URL = os.getenv("OLLAMA_URL")
OLLAMA_EMBED_URL = os.getenv("OLLAMA_EMBED_URL")
LLM_MODEL = os.getenv("LLM_MODEL")
EMBED_MODEL = os.getenv("EMBED_MODEL")

VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME")
FULLTEXT_INDEX_NAME = os.getenv("FULLTEXT_INDEX_NAME")
EMBEDDING_DIM_VALUE = os.getenv("EMBEDDING_DIM")
EMBEDDING_DIM = int(EMBEDDING_DIM_VALUE) if EMBEDDING_DIM_VALUE else None
GRAPH_TRAVERSAL_DEPTH = int(os.getenv("GRAPH_TRAVERSAL_DEPTH", "2"))

def validate_env():
    required_vars = {
        "NEO4J_URI": NEO4J_URI,
        "NEO4J_USER": NEO4J_USER,
        "NEO4J_PASSWORD": NEO4J_PASSWORD,
        "OLLAMA_URL": OLLAMA_URL,
        "OLLAMA_EMBED_URL": OLLAMA_EMBED_URL,
        "LLM_MODEL": LLM_MODEL,
        "EMBED_MODEL": EMBED_MODEL,
        "VECTOR_INDEX_NAME": VECTOR_INDEX_NAME,
        "FULLTEXT_INDEX_NAME": FULLTEXT_INDEX_NAME,
        "EMBEDDING_DIM": EMBEDDING_DIM,
    }

    missing = [name for name, value in required_vars.items() if value in (None, "")]
    if missing:
        missing_list = ", ".join(missing)
        raise RuntimeError(f"Missing required environment variables: {missing_list}")
