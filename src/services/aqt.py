"""
AQT Service (Advance Query Transformation).

Hybrid approach combining:
1. Intent Classification (Centroid-based) - Already implemented
2. NER (Entity Extraction) - TODO: Will be developed later

For now, uses regex patterns for filter extraction as a placeholder for NER.
"""
import re
from src.schemas.query import GraphRAGQuery
from src.services.intent_classifier import get_intent_classifier
from src.services.manufacturer_lookup import find_manufacturer_with_alias, load_manufacturers


# ============================================================
# FILTER EXTRACTION (Placeholder for future NER)
# ============================================================

# Load manufacturers at module import (cached)
load_manufacturers()


# NLEM/Reimbursement patterns
NLEM_PATTERNS = [
    r"บัญชียาหลัก",
    r"เบิกได้",
    r"เบิกจ่ายตรง",
    r"สิทธิ์.*(?:บัตรทอง|ข้าราชการ|ประกันสังคม)",
    r"(?:NLEM|ED|NED)",
    r"นอกบัญชี",
    r"ยา\s*จ\.",
]

NLEM_CATEGORY_PATTERNS = {
    "ก": [r"บัญชี\s*(?:ยา|)?\s*ก(?!\S)", r"cat(?:egory)?\s*a\b", r"หมวด\s*ก"],
    "ข": [r"บัญชี\s*(?:ยา|)?\s*ข(?!\S)", r"cat(?:egory)?\s*b\b", r"หมวด\s*ข"],
    "ค": [r"บัญชี\s*(?:ยา|)?\s*ค(?!\S)", r"cat(?:egory)?\s*c\b", r"หมวด\s*ค"],
    "ง": [r"บัญชี\s*(?:ยา|)?\s*ง(?!\S)", r"cat(?:egory)?\s*d\b", r"หมวด\s*ง", r"ยา\s*จ\.?\s*2"],
    "จ": [r"บัญชี\s*(?:ยา|)?\s*จ(?!\S)", r"cat(?:egory)?\s*e\b", r"หมวด\s*จ", r"ยาเฉพาะ"],
}


def extract_manufacturer(question: str) -> str:
    """Extract manufacturer name using dynamic lookup from Neo4j data."""
    return find_manufacturer_with_alias(question)


def extract_nlem_filter(question: str) -> bool:
    """Check if question asks about NLEM/reimbursement (Placeholder for NER)."""
    q_lower = question.lower()
    
    for pattern in NLEM_PATTERNS:
        if re.search(pattern, q_lower, re.IGNORECASE):
            return True
    
    return None


def extract_nlem_category(question: str) -> str:
    """Extract specific NLEM category if mentioned (Placeholder for NER)."""
    q_lower = question.lower()
    
    for category, patterns in NLEM_CATEGORY_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, q_lower, re.IGNORECASE):
                return category
    
    return None


def extract_drug_name(question: str) -> str:
    """Extract the main drug/entity name from question (Placeholder for NER)."""
    # Pattern for English drug names (most common case)
    english_pattern = r"(?:ยา\s*)?([A-Za-z][A-Za-z0-9\-]+)(?:\s+\d+\s*(?:mg|ml|g)?)?"
    
    matches = re.findall(english_pattern, question)
    if matches:
        # Filter out common non-drug words
        stopwords = {'GPO', 'NLEM', 'ED', 'NED', 'TMT', 'GP', 'TP', 'VTM', 'TPU', 'GPU', 
                     'Pfizer', 'Bayer', 'Sanofi', 'GSK', 'Merck', 'Abbott', 'Berlin'}
        for match in matches:
            if match.upper() not in stopwords and len(match) > 2:
                return match
    
    # If no English name, clean the question for Thai drug names
    cleaned = question
    remove_patterns = [
        r"ของบริษัท.*", r"ผลิตโดย.*", r"เบิกได้ไหม", r"ใช่.*ไหม", 
        r"หรือเปล่า", r"หรือไม่", r"ครับ", r"ค่ะ", r"มั้ย", r"ไหม",
        r"อะไร", r"อย่างไร", r"ยังไง", r"เท่าไหร่", r"กี่",
        r"มี.*บ้าง", r"ขอ.*หน่อย", r"ช่วย.*หน่อย"
    ]
    for pattern in remove_patterns:
        cleaned = re.sub(pattern, "", cleaned).strip()
    
    return cleaned if cleaned else question


# ============================================================
# MAIN TRANSFORMATION FUNCTION
# ============================================================

def transform_query(question: str, q_embedding=None) -> GraphRAGQuery:
    """
    Transform natural language question into structured query object.
    
    This is the main AQT function that combines:
    1. Intent Classification (Centroid-based) - determines target_type and strategy
    2. Filter Extraction (Regex-based) - extracts manufacturer, nlem, etc.
    
    Args:
        question: User's natural language question
        q_embedding: Pre-computed embedding (optional, to avoid re-embedding)
    
    Returns:
        GraphRAGQuery with all fields populated
    """
    # ============================================================
    # Step 1: Intent Classification (Centroid-based)
    # ============================================================
    classifier = get_intent_classifier()
    if not classifier._initialized:
        classifier.initialize()
    
    intent_result = classifier.classify(question, q_embedding)
    
    # Parse intent result
    target_type = intent_result.get("base_intent", "general")
    action = intent_result.get("action", "find")
    confidence = intent_result.get("confidence", 0.0)
    
    # Map action to strategy
    if action == "count":
        strategy = "count"
    elif action == "check":
        strategy = "verify"
    else:
        strategy = "retrieve"
    
    # ============================================================
    # Step 2: Filter Extraction (Regex-based, placeholder for NER)
    # ============================================================
    manufacturer = extract_manufacturer(question)
    nlem_filter = extract_nlem_filter(question)
    nlem_category = extract_nlem_category(question)
    search_term = extract_drug_name(question)
    
    # ============================================================
    # Step 3: Build Query Object
    # ============================================================
    query_obj = GraphRAGQuery(
        query=search_term,
        target_type=target_type,
        strategy=strategy,
        nlem_filter=nlem_filter,
        nlem_category=nlem_category,
        manufacturer_filter=manufacturer
    )
    
    # Store metadata for debugging/logging
    query_obj._intent_confidence = confidence
    query_obj._raw_intent = intent_result.get("intent", "unknown")
    
    return query_obj


def get_aqt_info() -> dict:
    """Get information about the AQT system configuration."""
    return {
        "intent_classifier": "Centroid-based (bge-m3)",
        "ner": "Regex-based (placeholder)",
        "llm_dependency": False,
        "version": "2.0.0-hybrid"
    }
