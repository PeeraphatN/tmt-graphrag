import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}

# ==============================
# CONFIG
# ==============================

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

OLLAMA_URL = os.getenv("OLLAMA_URL")
OLLAMA_EMBED_URL = os.getenv("OLLAMA_EMBED_URL")
LLM_MODEL = os.getenv("LLM_MODEL")
CLASSIFICATION_MODEL = os.getenv("CLASSIFICATION_MODEL", LLM_MODEL) # Default to LLM_MODEL if not set
EMBED_MODEL = os.getenv("EMBED_MODEL")

VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME")
FULLTEXT_INDEX_NAME = os.getenv("FULLTEXT_INDEX_NAME")
EMBEDDING_DIM_VALUE = os.getenv("EMBEDDING_DIM")
EMBEDDING_DIM = int(EMBEDDING_DIM_VALUE) if EMBEDDING_DIM_VALUE else None
GRAPH_TRAVERSAL_DEPTH = int(os.getenv("GRAPH_TRAVERSAL_DEPTH", "2"))

# IntentV2 + NER feature flags
INTENT_V2_ENABLED = _as_bool(os.getenv("INTENT_V2_ENABLED"), default=True)
INTENT_V2_USE_NER = _as_bool(os.getenv("INTENT_V2_USE_NER"), default=True)
INTENT_V2_ADAPTIVE_PLANNER = _as_bool(os.getenv("INTENT_V2_ADAPTIVE_PLANNER"), default=True)

# NER runtime config
NER_MODEL_DIR = os.getenv(
    "NER_MODEL_DIR",
    "experiments/name_entity_extraction_benckmarks/ner_model_output/final_model",
)
NER_CONFIDENCE_THRESHOLD = float(os.getenv("NER_CONFIDENCE_THRESHOLD", "0.60"))
NER_MAX_SEQ_LENGTH = int(os.getenv("NER_MAX_SEQ_LENGTH", "128"))

def validate_env():
    required_vars = {
        "NEO4J_URI": NEO4J_URI,
        "NEO4J_USER": NEO4J_USER,
        "NEO4J_PASSWORD": NEO4J_PASSWORD,
        "OLLAMA_URL": OLLAMA_URL,
        "OLLAMA_EMBED_URL": OLLAMA_EMBED_URL,
        "LLM_MODEL": LLM_MODEL,
        "CLASSIFICATION_MODEL": CLASSIFICATION_MODEL,
        "EMBED_MODEL": EMBED_MODEL,
        "VECTOR_INDEX_NAME": VECTOR_INDEX_NAME,
        "FULLTEXT_INDEX_NAME": FULLTEXT_INDEX_NAME,
        "EMBEDDING_DIM": EMBEDDING_DIM,
    }

    missing = [name for name, value in required_vars.items() if value in (None, "")]
    if missing:
        missing_list = ", ".join(missing)
        raise RuntimeError(f"Missing required environment variables: {missing_list}")
