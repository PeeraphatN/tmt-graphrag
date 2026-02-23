"""
AQT Service (Advanced Query Transformation).

IntentV2 flow:
1) Target classification from centroid classifier.
2) Rule strategy detection (retrieve/count/list/verify).
3) Slot extraction from rules + optional NER model.
4) Build IntentBundle payload and adaptive retrieval plan for search.
"""
from __future__ import annotations

import re

from src.config import INTENT_V2_ADAPTIVE_PLANNER, INTENT_V2_ENABLED, INTENT_V2_USE_NER
from src.schemas.intent_bundle import (
    ActionIntent,
    IntentBundle,
    IntentControlFeatures,
    RetrievalPlan,
    SlotSource,
    SlotValue,
    TopicsIntent,
)
from src.schemas.query import GraphRAGQuery, RetrievalMode
from src.services.intent_classifier import get_intent_classifier
from src.services.manufacturer_lookup import find_manufacturer_with_alias, load_manufacturers
from src.services.ner_service import get_ner_service

# Load once at module import (cached in service)
load_manufacturers()


# ============================================================
# Regex and lexical rules
# ============================================================

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9\-\./%]+|[\u0E00-\u0E7F]+")
UNIT_PATTERN = re.compile(r"^\d+(?:\.\d+)?(?:mg|g|mcg|ml|iu|%)?$", re.IGNORECASE)
DOSE_UNIT_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\s*(?:mg|g|mcg|ml|iu|%)\b", re.IGNORECASE)
TMTID_PATTERN = re.compile(r"\b(?:tmtid|tmt-id|tmt)\s*[:#]?\s*(\d{5,10})\b", re.IGNORECASE)
BARE_ID_PATTERN = re.compile(r"\b\d{6,10}\b")

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
COMPARE_PATTERNS = [
    r"\bcompare\b",
    r"\bdifference\b",
    r"\bvs\.?\b",
    r"เปรียบเทียบ",
    r"เทียบ",
    r"ต่างกัน",
]
UNKNOWN_PATTERNS = [
    r"^hi\b",
    r"^hello\b",
    r"^hey\b",
    r"^สวัสดี",
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
    "tmtid",
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
    q = question.lower().strip()
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


def detect_compare(question: str) -> bool:
    return _contains_pattern(question, COMPARE_PATTERNS)


def detect_unknown(question: str) -> bool:
    q = question.strip().lower()
    return _contains_pattern(q, UNKNOWN_PATTERNS) and len(q.split()) <= 8


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


def extract_tmtid(question: str) -> str | None:
    match = TMTID_PATTERN.search(question)
    if match:
        return str(match.group(1))
    bare_match = BARE_ID_PATTERN.search(question)
    return str(bare_match.group(0)) if bare_match else None


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


def _strategy_to_action_intent(strategy: str, question: str, is_ambiguous: bool) -> ActionIntent:
    if detect_compare(question):
        return ActionIntent.COMPARE
    if strategy == "count":
        return ActionIntent.COUNT
    if strategy == "list":
        return ActionIntent.LIST
    if strategy == "verify":
        return ActionIntent.VERIFY
    if detect_unknown(question) and is_ambiguous:
        return ActionIntent.UNKNOWN
    return ActionIntent.LOOKUP


def _target_to_topics(target: str) -> TopicsIntent:
    mapping = {
        "manufacturer": TopicsIntent.MANUFACTURER,
        "ingredient": TopicsIntent.INGREDIENT,
        "nlem": TopicsIntent.NLEM,
        "formula": TopicsIntent.FORMULA,
        "hierarchy": TopicsIntent.HIERARCHY,
        "general": TopicsIntent.GENERAL,
    }
    return mapping.get(str(target), TopicsIntent.GENERAL)


def _to_dict(model_obj):
    if hasattr(model_obj, "model_dump"):
        return model_obj.model_dump(mode="json")
    return model_obj.dict()


def _build_intent_bundle(
    question: str,
    query_obj: GraphRAGQuery,
    intent_result: dict,
    ner_payload: dict | None,
) -> IntentBundle:
    strategy_value = query_obj.strategy.value if hasattr(query_obj.strategy, "value") else str(query_obj.strategy)
    target_value = query_obj.target_type.value if hasattr(query_obj.target_type, "value") else str(query_obj.target_type)
    is_ambiguous = bool(intent_result.get("is_ambiguous", False))

    action_intent = _strategy_to_action_intent(strategy_value, question, is_ambiguous)
    topics_intents: list[TopicsIntent] = [_target_to_topics(target_value)]
    slot_values: list[SlotValue] = []

    if query_obj.id_lookup:
        if TopicsIntent.ID_LOOKUP not in topics_intents:
            topics_intents.append(TopicsIntent.ID_LOOKUP)
        slot_values.append(
            SlotValue(name="tmtid", value=str(query_obj.id_lookup), confidence=1.0, source=SlotSource.RULE)
        )

    if query_obj.manufacturer_filter:
        if TopicsIntent.MANUFACTURER not in topics_intents:
            topics_intents.append(TopicsIntent.MANUFACTURER)
        slot_values.append(
            SlotValue(
                name="manufacturer",
                value=str(query_obj.manufacturer_filter),
                confidence=0.95,
                source=SlotSource.RULE,
            )
        )

    if query_obj.nlem_filter is not None:
        if TopicsIntent.NLEM not in topics_intents:
            topics_intents.append(TopicsIntent.NLEM)
        slot_values.append(
            SlotValue(
                name="nlem",
                value=str(bool(query_obj.nlem_filter)).lower(),
                confidence=0.95,
                source=SlotSource.RULE,
            )
        )
    if query_obj.nlem_category:
        slot_values.append(
            SlotValue(
                name="nlem_category",
                value=str(query_obj.nlem_category),
                confidence=0.95,
                source=SlotSource.RULE,
            )
        )

    # Capture NER slots for downstream routing/context.
    ner_slots = (ner_payload or {}).get("slots", {})
    for key in ("brand", "drug", "strength", "form"):
        value = ner_slots.get(key)
        if value:
            slot_values.append(SlotValue(name=key, value=str(value), confidence=0.85, source=SlotSource.NER))

    if query_obj.query:
        slot_values.append(SlotValue(name="query", value=str(query_obj.query), confidence=0.70, source=SlotSource.RULE))

    # Add formula topic when form/strength appears.
    if ner_slots.get("strength") or ner_slots.get("form"):
        if TopicsIntent.FORMULA not in topics_intents:
            topics_intents.append(TopicsIntent.FORMULA)

    top_target_scores = {
        item.get("target", ""): float(item.get("score", 0.0)) for item in intent_result.get("top_targets", [])
    }
    control_features = IntentControlFeatures(
        token_count=query_obj.token_count,
        entity_token_count=query_obj.entity_token_count,
        entity_ratio=query_obj.entity_ratio,
        has_exact_id=1.0 if query_obj.id_lookup else 0.0,
        has_dose_unit=1.0 if DOSE_UNIT_PATTERN.search(question) else 0.0,
        ambiguity=max(0.0, min(1.0, 1.0 - float(intent_result.get("target_margin", 0.0)))),
    )

    filters = {}
    if query_obj.nlem_filter is not None:
        filters["nlem"] = query_obj.nlem_filter
    if query_obj.nlem_category:
        filters["nlem_category"] = query_obj.nlem_category
    if query_obj.manufacturer_filter:
        filters["manufacturer"] = query_obj.manufacturer_filter
    if query_obj.id_lookup:
        filters["tmtid"] = query_obj.id_lookup

    retrieval_mode = query_obj.retrieval_mode.value if hasattr(query_obj.retrieval_mode, "value") else str(query_obj.retrieval_mode)
    retrieval_plan = RetrievalPlan(
        retrieval_mode=retrieval_mode,
        vector_weight=query_obj.vector_weight,
        fulltext_weight=query_obj.fulltext_weight,
        top_k=query_obj.limit,
        depth=1 if query_obj.id_lookup else 2,
        filters=filters,
        must_match=list(query_obj.must_match),
    )

    metadata = {
        "legacy_raw_intent": intent_result.get("intent", "unknown"),
        "legacy_target_confidence": float(intent_result.get("target_confidence", 0.0)),
        "legacy_target_margin": float(intent_result.get("target_margin", 0.0)),
        "is_ambiguous": is_ambiguous,
        "ner_available": bool((ner_payload or {}).get("available", False)),
    }
    if ner_payload and ner_payload.get("error"):
        metadata["ner_error"] = ner_payload["error"]

    return IntentBundle(
        query=question,
        action_intent=action_intent,
        topics_intents=topics_intents,
        slots=slot_values,
        action_scores={action_intent.value: float(intent_result.get("confidence", 0.0))},
        topics_scores=top_target_scores,
        control_features=control_features,
        adaptive_retrieval_weights=retrieval_plan,
        metadata=metadata,
    )


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

    # Property filters (rules)
    manufacturer = extract_manufacturer(question)
    nlem_filter = extract_nlem_filter(question)
    nlem_category = extract_nlem_category(question)
    if nlem_category and nlem_filter is None:
        nlem_filter = True

    tmtid = extract_tmtid(question)
    search_term = extract_drug_name(question, manufacturer=manufacturer)

    # Optional NER refinement
    ner_payload = None
    if INTENT_V2_USE_NER:
        ner_payload = get_ner_service().extract(question)
        if not manufacturer and (ner_payload or {}).get("manufacturer_filter"):
            manufacturer = str(ner_payload["manufacturer_filter"])
        ner_query = str((ner_payload or {}).get("query", "")).strip()
        if ner_query and len(ner_query) >= 2:
            search_term = ner_query

    # Target override when filter signal is explicit
    if nlem_filter:
        target_type = "nlem"
    elif manufacturer:
        target_type = "manufacturer"

    # If form/strength exists, favor formula scope.
    ner_slots = (ner_payload or {}).get("slots", {})
    if target_type == "general" and (ner_slots.get("strength") or ner_slots.get("form")):
        target_type = "formula"

    # Exact ID lookup path always has priority for query term.
    if tmtid:
        search_term = tmtid

    # Query fallback for list/count style questions
    if strategy in {"count", "list"} and not search_term:
        search_term = "ยา"

    token_count, entity_token_count, entity_ratio, is_abstract = compute_entity_profile(
        question=question,
        manufacturer=manufacturer,
        nlem_filter=nlem_filter,
        nlem_category=nlem_category,
    )

    if INTENT_V2_ADAPTIVE_PLANNER:
        retrieval_mode, vector_weight, fulltext_weight = choose_retrieval_profile(
            strategy=strategy,
            entity_ratio=entity_ratio,
            is_abstract=is_abstract,
        )
    else:
        retrieval_mode, vector_weight, fulltext_weight = RetrievalMode.BALANCED, 0.50, 0.50

    # Override profile for exact ID lookup to maximize precision.
    if tmtid:
        retrieval_mode, vector_weight, fulltext_weight = RetrievalMode.FULLTEXT_HEAVY, 0.15, 0.85

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
        id_lookup=tmtid,
        must_match=[tmtid] if tmtid else [],
    )

    # Debug metadata
    query_obj._intent_confidence = intent_result.get("confidence", 0.0)
    query_obj._target_confidence = intent_result.get("target_confidence", 0.0)
    query_obj._target_margin = intent_result.get("target_margin", 0.0)
    query_obj._raw_intent = intent_result.get("intent", "unknown")
    query_obj._intent_top_targets = intent_result.get("top_targets", [])
    query_obj._is_abstract = is_abstract
    query_obj._ner_payload = ner_payload or {}

    # IntentV2 bundle payload for search/debug.
    if INTENT_V2_ENABLED:
        bundle = _build_intent_bundle(
            question=question,
            query_obj=query_obj,
            intent_result=intent_result,
            ner_payload=ner_payload,
        )
        query_obj.intent_bundle = _to_dict(bundle)
    else:
        query_obj.intent_bundle = None

    return query_obj


def get_aqt_info() -> dict:
    return {
        "intent_classifier": "Centroid-based target classification (bge-m3)",
        "strategy_detector": "Rule-based action classifier",
        "adaptive_profile": INTENT_V2_ADAPTIVE_PLANNER,
        "intent_v2_enabled": INTENT_V2_ENABLED,
        "ner_enabled": INTENT_V2_USE_NER,
        "llm_dependency": False,
        "version": "4.0.0-intent-v2-ner-hybrid",
    }
