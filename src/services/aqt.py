"""
AQT Service (Advanced Query Transformation).

This version separates intent into two explicit axes:
1) target_type: domain focus (manufacturer / ingredient / nlem / ...)
2) strategy: action shape (retrieve / count / list / verify)

It also computes an adaptive retrieval profile used by hybrid search:
- entity_ratio
- retrieval_mode
- vector_weight / fulltext_weight
"""
from __future__ import annotations

import re

from src.schemas.query import GraphRAGQuery, RetrievalMode
from src.services.intent_classifier import get_intent_classifier
from src.services.manufacturer_lookup import find_manufacturer_with_alias, load_manufacturers

# Load once at module import (cached in service)
load_manufacturers()


# ============================================================
# Regex and lexical rules
# ============================================================

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9\-\./%]+|[\u0E00-\u0E7F]+")
UNIT_PATTERN = re.compile(r"^\d+(?:\.\d+)?(?:mg|g|mcg|ml|iu|%)?$", re.IGNORECASE)

COUNT_PATTERNS = [
    r"\bhow many\b",
    r"\bcount\b",
    r"\bnumber of\b",
    r"กี่",
    r"จำนวน",
    r"นับ",
]
LIST_PATTERNS = [
    r"\blist\b",
    r"\bshow all\b",
    r"\ball\b",
    r"รายการ",
    r"รายชื่อ",
    r"แสดง",
]
VERIFY_PATTERNS = [
    r"\bis\b",
    r"\bdoes\b",
    r"\bcan\b",
    r"\bexists?\b",
    r"ไหม",
    r"หรือไม่",
    r"ใช่",
    r"จริง",
]

NLEM_PATTERNS = [
    r"\bnlem\b",
    r"\bed\b",
    r"\bned\b",
    r"บัญชียาหลัก",
    r"เบิก",
    r"สิทธิ",
    r"reimburse",
    r"reimbursement",
]

NLEM_CATEGORY_PATTERNS = {
    "ก": [r"บัญชี\s*ก", r"category\s*a", r"cat\s*a"],
    "ข": [r"บัญชี\s*ข", r"category\s*b", r"cat\s*b"],
    "ค": [r"บัญชี\s*ค", r"category\s*c", r"cat\s*c"],
    "ง": [r"บัญชี\s*ง", r"category\s*d", r"cat\s*d"],
    "จ": [r"บัญชี\s*จ", r"category\s*e", r"cat\s*e"],
}

EN_STOPWORDS = {
    "what",
    "which",
    "with",
    "that",
    "this",
    "drug",
    "medicine",
    "please",
    "help",
    "show",
    "list",
    "count",
    "does",
    "is",
    "are",
    "can",
    "do",
    "did",
    "make",
    "made",
    "manufacture",
    "manufacturer",
}

ABSTRACT_HINTS = {
    "overview",
    "summary",
    "difference",
    "compare",
    "information",
    "อธิบาย",
    "สรุป",
    "ภาพรวม",
    "ข้อมูล",
}

MEDICAL_HINTS = {
    "nlem",
    "ed",
    "ned",
    "tmt",
    "vtm",
    "gp",
    "gpu",
    "tp",
    "tpu",
}


# ============================================================
# Feature extraction helpers
# ============================================================

def _contains_pattern(question: str, patterns: list[str]) -> bool:
    q = question.lower()
    return any(re.search(p, q, re.IGNORECASE) for p in patterns)


def detect_strategy(question: str) -> str:
    """
    Detect action-oriented strategy.
    Priority: count > list > verify > retrieve
    """
    q = question.strip()

    if _contains_pattern(q, COUNT_PATTERNS):
        return "count"
    if _contains_pattern(q, LIST_PATTERNS):
        return "list"
    if _contains_pattern(q, VERIFY_PATTERNS):
        return "verify"
    return "retrieve"


def extract_manufacturer(question: str) -> str | None:
    return find_manufacturer_with_alias(question)


def extract_nlem_filter(question: str) -> bool | None:
    return True if _contains_pattern(question, NLEM_PATTERNS) else None


def extract_nlem_category(question: str) -> str | None:
    q = question.lower()
    for category, patterns in NLEM_CATEGORY_PATTERNS.items():
        if any(re.search(p, q, re.IGNORECASE) for p in patterns):
            return category
    return None


def extract_drug_name(question: str, manufacturer: str | None = None) -> str:
    """
    Extract a compact core search term.
    """
    english_pattern = r"(?:ยา\s*)?([A-Za-z][A-Za-z0-9\-]+)(?:\s+\d+\s*(?:mg|ml|g|mcg|iu)?)?"
    matches = re.findall(english_pattern, question, re.IGNORECASE)
    manufacturer_tokens = set()
    if manufacturer:
        manufacturer_tokens = {t.lower() for t in TOKEN_PATTERN.findall(manufacturer)}
    if matches:
        for token in matches:
            token_lower = token.lower()
            if token_lower in manufacturer_tokens:
                continue
            if len(token) >= 3 and token_lower not in EN_STOPWORDS:
                return token

    cleaned = question
    remove_patterns = [
        r"ของบริษัท.*",
        r"ผลิตโดย.*",
        r"เบิกได้ไหม",
        r"ใช่.*ไหม",
        r"หรือเปล่า",
        r"หรือไม่",
        r"ครับ",
        r"ค่ะ",
        r"มั้ย",
        r"ไหม",
        r"อะไร",
        r"อย่างไร",
        r"ยังไง",
        r"เท่าไหร่",
        r"กี่",
        r"มี.*บ้าง",
        r"ขอ.*หน่อย",
        r"ช่วย.*หน่อย",
    ]
    for pattern in remove_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()

    cleaned = " ".join(cleaned.split())
    return cleaned if cleaned else question.strip()


def compute_entity_profile(
    question: str,
    manufacturer: str | None = None,
    nlem_filter: bool | None = None,
    nlem_category: str | None = None,
) -> tuple[int, int, float, bool]:
    """
    Estimate entity specificity from lexical features.
    Returns: token_count, entity_token_count, entity_ratio, is_abstract
    """
    tokens = TOKEN_PATTERN.findall(question.lower())
    if not tokens:
        return 0, 0, 0.0, False

    entity_tokens = 0
    for token in tokens:
        if UNIT_PATTERN.match(token):
            entity_tokens += 1
            continue
        if any(ch.isdigit() for ch in token):
            entity_tokens += 1
            continue
        if token in MEDICAL_HINTS:
            entity_tokens += 1
            continue
        if re.search(r"[A-Za-z]", token):
            if len(token) >= 3 and token not in EN_STOPWORDS:
                entity_tokens += 1
            continue

    if manufacturer:
        entity_tokens += max(1, len(TOKEN_PATTERN.findall(manufacturer)))
    if nlem_filter:
        entity_tokens += 1
    if nlem_category:
        entity_tokens += 1

    token_count = len(tokens)
    entity_token_count = min(entity_tokens, token_count)
    entity_ratio = entity_token_count / token_count

    lower_q = question.lower()
    is_abstract = (token_count >= 10 and not re.search(r"[A-Za-z0-9]", lower_q)) or any(
        hint in lower_q for hint in ABSTRACT_HINTS
    )
    return token_count, entity_token_count, entity_ratio, is_abstract


def choose_retrieval_profile(
    strategy: str,
    entity_ratio: float,
    is_abstract: bool,
) -> tuple[RetrievalMode, float, float]:
    """
    Adaptive profile for hybrid retrieval weights.
    """
    # Action-specific defaults
    if strategy in {"count", "list"}:
        return RetrievalMode.FULLTEXT_HEAVY, 0.30, 0.70

    if strategy == "verify":
        if entity_ratio >= 0.40:
            return RetrievalMode.FULLTEXT_HEAVY, 0.35, 0.65
        return RetrievalMode.BALANCED, 0.50, 0.50

    # retrieve strategy
    if entity_ratio >= 0.50:
        return RetrievalMode.FULLTEXT_HEAVY, 0.35, 0.65
    if entity_ratio <= 0.20 and is_abstract:
        return RetrievalMode.VECTOR_HEAVY, 0.70, 0.30
    return RetrievalMode.BALANCED, 0.50, 0.50


# ============================================================
# Main transformation
# ============================================================

def transform_query(question: str, q_embedding=None) -> GraphRAGQuery:
    """
    Transform natural language into GraphRAGQuery.
    """
    classifier = get_intent_classifier()
    if not classifier._initialized:
        classifier.initialize()

    # Axis 1: semantic target classification (vector centroid)
    intent_result = classifier.classify(question, q_embedding)
    target_type = intent_result.get("target_type", intent_result.get("base_intent", "general"))

    # Axis 2: action strategy (rules)
    strategy = detect_strategy(question)

    # Property filters
    manufacturer = extract_manufacturer(question)
    nlem_filter = extract_nlem_filter(question)
    nlem_category = extract_nlem_category(question)
    if nlem_category and nlem_filter is None:
        nlem_filter = True

    search_term = extract_drug_name(question, manufacturer=manufacturer)

    # Target override when filter signal is explicit
    if nlem_filter:
        target_type = "nlem"
    elif manufacturer:
        target_type = "manufacturer"

    # Query fallback for list/count style questions
    if strategy in {"count", "list"} and not search_term:
        search_term = "ยา"

    token_count, entity_token_count, entity_ratio, is_abstract = compute_entity_profile(
        question=question,
        manufacturer=manufacturer,
        nlem_filter=nlem_filter,
        nlem_category=nlem_category,
    )
    retrieval_mode, vector_weight, fulltext_weight = choose_retrieval_profile(
        strategy=strategy,
        entity_ratio=entity_ratio,
        is_abstract=is_abstract,
    )

    query_obj = GraphRAGQuery(
        query=search_term,
        target_type=target_type,
        strategy=strategy,
        nlem_filter=nlem_filter,
        nlem_category=nlem_category,
        manufacturer_filter=manufacturer,
        token_count=token_count,
        entity_token_count=entity_token_count,
        entity_ratio=entity_ratio,
        retrieval_mode=retrieval_mode,
        vector_weight=vector_weight,
        fulltext_weight=fulltext_weight,
    )

    # Debug metadata
    query_obj._intent_confidence = intent_result.get("confidence", 0.0)
    query_obj._target_confidence = intent_result.get("target_confidence", 0.0)
    query_obj._target_margin = intent_result.get("target_margin", 0.0)
    query_obj._raw_intent = intent_result.get("intent", "unknown")
    query_obj._intent_top_targets = intent_result.get("top_targets", [])
    query_obj._is_abstract = is_abstract

    return query_obj


def get_aqt_info() -> dict:
    return {
        "intent_classifier": "Centroid-based target classification (bge-m3)",
        "strategy_detector": "Rule-based action classifier",
        "adaptive_profile": True,
        "llm_dependency": False,
        "version": "3.0.0-adaptive-intent",
    }
