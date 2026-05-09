"""
AQT Service (Advanced Query Transformation).

IntentV2 flow (raw-first):
1) Keep question_raw and lightly normalize for classifier/strategy only.
2) Run NER extraction and sanitize noisy slots/entities.
3) Merge deterministic filters (tmtid/manufacturer/nlem).
4) Build search term from id + sanitized NER slots + raw fallback.
5) Compute adaptive retrieval profile and build IntentBundle.
"""
from __future__ import annotations
import os
import re
from typing import Any

from src.config import (
    INTENT_V2_ADAPTIVE_PLANNER,
    INTENT_V2_ENABLED,
    INTENT_V2_USE_NER,
    NER_CONFIDENCE_THRESHOLD,
)
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
# Yes/no English questions ("Is X ...?") should be classified as verify even
# when the sentence contains "list" as a noun ("the NLEM list"). Matched
# before LIST_PATTERNS in detect_strategy to override the noun-shaped match.
VERIFY_LEADING_PATTERN = re.compile(
    r"^\s*(is|are|was|were|does|do|did|can|could|should|will|would|has|have|had)\b",
    re.IGNORECASE,
)
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

# Unicode-safe Thai keywords (avoid source encoding issues in regex literals)
TH_COUNT_KEYWORDS = (
    "\u0e08\u0e33\u0e19\u0e27\u0e19",  # จำนวน
    "\u0e01\u0e35\u0e48",              # กี่
    "\u0e19\u0e31\u0e1a",              # นับ
)
TH_LIST_KEYWORDS = (
    "\u0e23\u0e32\u0e22\u0e01\u0e32\u0e23",  # รายการ
    "\u0e23\u0e32\u0e22\u0e0a\u0e37\u0e48\u0e2d",  # รายชื่อ
    "\u0e41\u0e2a\u0e14\u0e07",          # แสดง
    "\u0e17\u0e31\u0e49\u0e07\u0e2b\u0e21\u0e14",  # ทั้งหมด
)
TH_VERIFY_KEYWORDS = (
    "\u0e44\u0e2b\u0e21",              # ไหม
    "\u0e2b\u0e23\u0e37\u0e2d\u0e44\u0e21\u0e48",  # หรือไม่
    "\u0e08\u0e23\u0e34\u0e07\u0e44\u0e2b\u0e21",  # จริงไหม
)

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

NER_NOISE_TERMS = EN_STOPWORDS | {
    "drug",
    "drugs",
    "medicine",
    "medicines",
    "data",
    "info",
    "query",
    "ยา",
    "ข้อมูล",
    "ขอ",
    "ระหว่าง",
    "เปรียบเทียบ",
    "เปรย",
    "บเทยบ",
    "รายการ",
    "ทั้งหมด",
}

NER_LABEL_TO_SLOT = {
    "BRAND": "brand",
    "DRUG": "drug",
    "MANUFACTURER": "manufacturer",
    "FORM": "form",
    "STRENGTH": "strength",
}
NER_ALLOWED_LABELS = set(NER_LABEL_TO_SLOT.keys())
NER_SANITIZE_MIN_CONFIDENCE = max(0.60, float(NER_CONFIDENCE_THRESHOLD))

# PoC: keep as many entities as possible, but cap per slot to control payload size.
# brand/manufacturer default=2; drug defaults to 3x brand.
NER_MAX_BRAND = int(os.getenv("NER_MAX_BRAND", "2"))
NER_MAX_MANUFACTURER = int(os.getenv("NER_MAX_MANUFACTURER", str(NER_MAX_BRAND)))
NER_DRUG_MULTIPLIER = int(os.getenv("NER_DRUG_MULTIPLIER", "3"))
NER_MAX_DRUG = int(os.getenv("NER_MAX_DRUG", str(max(1, NER_MAX_BRAND) * max(1, NER_DRUG_MULTIPLIER))))

CONNECTOR_TOKENS = {
    "กับ",
    "กบ",
    "และ",
    "หรือ",
    "compare",
    "vs",
    "v.s.",
    "versus",
    "and",
    "or",
    "with",
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

MANUFACTURER_CONTEXT_TERMS = {
    "manufacturer",
    "produce",
    "produced",
    "made",
    "company",
    "ผู้ผลิต",
    "ผลิตโดย",
    "บริษัท",
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
    Priority: count > verify-leading > list > verify > retrieve

    The verify-leading check sits between count and list so that yes/no
    questions ("Is paracetamol in the NLEM list?") aren't misclassified as
    list just because the noun "list" appears in the sentence.
    """
    q = question.strip()

    if _contains_pattern(q, COUNT_PATTERNS) or any(k in q for k in TH_COUNT_KEYWORDS):
        return "count"
    if VERIFY_LEADING_PATTERN.match(q):
        return "verify"
    if _contains_pattern(q, LIST_PATTERNS) or any(k in q for k in TH_LIST_KEYWORDS):
        return "list"
    if _contains_pattern(q, VERIFY_PATTERNS) or any(k in q for k in TH_VERIFY_KEYWORDS):
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


def _light_normalize_question(question: str) -> str:
    """
    Preserve context while normalizing whitespace only.
    """
    return " ".join(str(question).strip().split())


def _safe_confidence(item: dict[str, Any]) -> float:
    raw_value = item.get("confidence", item.get("score", 0.0))
    try:
        return float(raw_value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _normalize_slot_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def _canonical_slot_text(value: Any) -> str:
    text = _normalize_slot_text(value).lower()
    if not text:
        return ""
    return " ".join(TOKEN_PATTERN.findall(text))


def _is_tmtid_like_text(value: Any, tmtid: str | None) -> bool:
    text = _normalize_slot_text(value).lower()
    if not text:
        return False
    if tmtid and str(tmtid).lower() in text:
        return True
    if ("tmtid" in text or "tmt" in text) and re.search(r"\b\d{5,10}\b", text):
        return True
    return False


def _contains_signal_token(text: str) -> bool:
    tokens = TOKEN_PATTERN.findall(text.lower())
    for token in tokens:
        if token in NER_NOISE_TERMS:
            continue
        if len(token) >= 2 or any(ch.isdigit() for ch in token):
            return True
    return False


def _is_text_from_question(question_raw: str, value: str) -> bool:
    question_lower = question_raw.lower()
    value_lower = value.lower()
    if value_lower in question_lower:
        return True

    value_tokens = set(TOKEN_PATTERN.findall(value_lower))
    if not value_tokens:
        return False
    question_tokens = set(TOKEN_PATTERN.findall(question_lower))
    overlap = len(value_tokens & question_tokens) / max(1, len(value_tokens))
    return overlap >= 0.5


def _split_compare_entities(value: str) -> list[str]:
    """
    Split a potentially combined compare phrase into individual entities.
    Examples:
    - "Paracetamol กับ Ibuprofen" -> ["Paracetamol", "Ibuprofen"]
    - "Paracetamol กบ Ibuprofen" -> ["Paracetamol", "Ibuprofen"]
    """
    text = _normalize_slot_text(value)
    if not text:
        return []

    tokens = TOKEN_PATTERN.findall(text)
    if not tokens:
        return []

    parts: list[str] = []
    current: list[str] = []
    for token in tokens:
        token_lower = token.lower()
        if token_lower in CONNECTOR_TOKENS:
            if current:
                parts.append(_normalize_slot_text(" ".join(current)))
                current = []
            continue
        current.append(token)

    if current:
        parts.append(_normalize_slot_text(" ".join(current)))

    # Remove empty/noise-only parts and deduplicate while keeping order.
    cleaned: list[str] = []
    seen: set[str] = set()
    for part in parts:
        if not part:
            continue
        if part.lower() in NER_NOISE_TERMS:
            continue
        if _is_connector_only(part):
            continue
        key = part.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(part)
    return cleaned


def _sanitize_ner_slot_value(
    question_raw: str,
    slot_name: str,
    raw_value: Any,
    split_compare: bool = True,
) -> tuple[str | None, str]:
    value = _normalize_slot_text(raw_value)
    if not value:
        return None, "empty"

    value_lower = value.lower()
    if value_lower in NER_NOISE_TERMS:
        return None, "noise_term"
    if not _contains_signal_token(value):
        return None, "no_signal_token"

    # Post-process: Reject English question patterns
    english_patterns = ["how many", "what is", "overview of", "information about", "list of", "number of"]
    if any(pattern in value_lower for pattern in english_patterns):
        return None, "english_question"

    # Post-process: Reject obvious Thai intent fragments that are not entities.
    thai_intent_fragments = ["เปรียบเทียบ", "ระหว่าง", "ข้อมูล", "ขอ", "เปรย", "บเทยบ"]
    if slot_name in {"drug", "brand"} and any(fragment in value_lower for fragment in thai_intent_fragments):
        if not re.search(r"[A-Za-z0-9]", value):
            return None, "thai_intent_fragment"

    # Post-process: normalize compare entities for single-slot usage
    if split_compare and slot_name == "drug":
        parts = _split_compare_entities(value)
        if parts:
            value = parts[0]
            if not value:
                return None, "empty_after_split"

    if slot_name in {"query", "brand", "drug", "manufacturer"} and not _is_text_from_question(question_raw, value):
        return None, "not_in_question"

    if slot_name == "manufacturer":
        canonical = find_manufacturer_with_alias(value)
        if not canonical:
            return None, "unknown_manufacturer"
        value = _normalize_slot_text(canonical)

    if slot_name == "strength" and not any(ch.isdigit() for ch in value):
        return None, "invalid_strength"

    return value, "ok"


def _select_top_entities(entities: list[dict[str, Any]]) -> dict[str, str]:
    top_by_slot: dict[str, tuple[float, int, str]] = {}
    for entity in entities:
        label = str(entity.get("label", "")).upper().strip()
        slot_name = NER_LABEL_TO_SLOT.get(label)
        if not slot_name:
            continue
        confidence = _safe_confidence(entity)
        text = _normalize_slot_text(entity.get("text"))
        if not text:
            continue
        
        # Apply post-processing to individual entities for single slots.
        if slot_name == "drug":
            parts = _split_compare_entities(text)
            if parts:
                text = parts[0]
                if not text:
                    continue

        current = top_by_slot.get(slot_name)
        candidate = (confidence, len(text), text)
        if current is None or candidate > current:
            top_by_slot[slot_name] = candidate
    return {slot: payload[2] for slot, payload in top_by_slot.items()}


def _is_connector_only(value: str) -> bool:
    tokens = [t.lower() for t in TOKEN_PATTERN.findall(str(value))]
    if not tokens:
        return False
    return all(token in CONNECTOR_TOKENS for token in tokens)


def _collect_slot_entities(
    sanitized_entities: list[dict[str, Any]],
    slot_name: str,
    max_items: int,
) -> list[str]:
    if max_items <= 0:
        return []

    candidates: list[tuple[float, int, str]] = []
    for entity in sanitized_entities:
        label = str(entity.get("label", "")).upper().strip()
        if NER_LABEL_TO_SLOT.get(label) != slot_name:
            continue
        text = _normalize_slot_text(entity.get("text"))
        if not text:
            continue
        if _is_connector_only(text):
            continue
        confidence = _safe_confidence(entity)
        candidates.append((confidence, len(text), text))

    candidates.sort(reverse=True)
    results: list[str] = []
    seen: set[str] = set()
    for _, __, text in candidates:
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        results.append(text)
        if len(results) >= max_items:
            break
    return results


def _build_multi_entity_slots(
    sanitized_entities: list[dict[str, Any]],
    max_brand: int,
    max_manufacturer: int,
    max_drug: int,
) -> dict[str, list[str]]:
    """Build multi-entity slots with post-processing for better entity extraction"""
    
    # First pass: collect entities with post-processing
    processed_entities = []
    for entity in sanitized_entities:
        label = str(entity.get("label", "")).upper().strip()
        slot_name = NER_LABEL_TO_SLOT.get(label)
        if not slot_name:
            continue
            
        text = _normalize_slot_text(entity.get("text"))
        if not text:
            continue
            
        # Post-process comparison entities for drug slot
        if slot_name == "drug":
            parts = _split_compare_entities(text)
            if not parts:
                continue
            for i, part in enumerate(parts):
                if not part or _is_connector_only(part):
                    continue
                # Keep highest confidence for first part and slightly reduce follow-up parts.
                adjusted_confidence = entity["confidence"] * (0.9 if i > 0 else 1.0)
                processed_entity = dict(entity)
                processed_entity["text"] = part
                processed_entity["label"] = label
                processed_entity["confidence"] = adjusted_confidence
                processed_entities.append(processed_entity)
        else:
            processed_entities.append(entity)
    
    # Build slots with limits
    slots_multi = {
        "brand": _collect_slot_entities(processed_entities, "brand", max_brand),
        "manufacturer": _collect_slot_entities(processed_entities, "manufacturer", max_manufacturer),
        "drug": _collect_slot_entities(processed_entities, "drug", max_drug),
    }
    
    return slots_multi


def _sanitize_ner_payload(question_raw: str, ner_payload: dict | None) -> dict[str, Any]:
    """
    Filter NER output to reduce slot noise before downstream routing.
    """
    base_payload: dict[str, Any] = {
        "available": False,
        "query": "",
        "manufacturer_filter": None,
        "slots": {},
        "entities": [],
    }
    if not isinstance(ner_payload, dict):
        return base_payload

    if ner_payload.get("error"):
        base_payload["error"] = ner_payload["error"]
    if not bool(ner_payload.get("available", False)):
        return base_payload

    sanitized_entities: list[dict[str, Any]] = []
    seen_entities: set[tuple[str, str]] = set()
    dropped_entities: list[dict[str, Any]] = []

    for raw_entity in ner_payload.get("entities", []) or []:
        if not isinstance(raw_entity, dict):
            continue
        label = str(raw_entity.get("label", "")).upper().strip()
        if label not in NER_ALLOWED_LABELS:
            continue

        confidence = _safe_confidence(raw_entity)
        if confidence < NER_SANITIZE_MIN_CONFIDENCE:
            dropped_entities.append({"label": label, "text": raw_entity.get("text"), "reason": "low_confidence"})
            continue

        slot_name = NER_LABEL_TO_SLOT[label]
        sanitized_text, reason = _sanitize_ner_slot_value(
            question_raw,
            slot_name,
            raw_entity.get("text"),
            split_compare=False,
        )
        if not sanitized_text:
            dropped_entities.append({"label": label, "text": raw_entity.get("text"), "reason": reason})
            continue

        dedupe_key = (label, sanitized_text.lower())
        if dedupe_key in seen_entities:
            continue
        seen_entities.add(dedupe_key)

        entity = dict(raw_entity)
        entity["label"] = label
        entity["text"] = sanitized_text
        entity["confidence"] = confidence
        sanitized_entities.append(entity)

    slots_multi = _build_multi_entity_slots(
        sanitized_entities,
        NER_MAX_BRAND,
        NER_MAX_MANUFACTURER,
        NER_MAX_DRUG,
    )

    top_from_entities = _select_top_entities(sanitized_entities)
    raw_slots = ner_payload.get("slots", {}) or {}

    slot_candidates = {
        "brand": [raw_slots.get("brand"), top_from_entities.get("brand")],
        "drug": [raw_slots.get("drug"), top_from_entities.get("drug")],
        "strength": [raw_slots.get("strength"), top_from_entities.get("strength")],
        "form": [raw_slots.get("form"), top_from_entities.get("form")],
        "manufacturer": [
            raw_slots.get("manufacturer"),
            ner_payload.get("manufacturer_filter"),
            top_from_entities.get("manufacturer"),
        ],
    }

    sanitized_slots: dict[str, str] = {}
    dropped_slots: dict[str, str] = {}
    for slot_name, candidates in slot_candidates.items():
        best_value: str | None = None
        last_reason = "empty"
        for candidate in candidates:
            value, reason = _sanitize_ner_slot_value(question_raw, slot_name, candidate)
            if value:
                best_value = value
                break
            last_reason = reason
        if best_value:
            sanitized_slots[slot_name] = best_value
        elif any(_normalize_slot_text(candidate) for candidate in candidates):
            dropped_slots[slot_name] = last_reason


    sanitized_payload: dict[str, Any] = dict(ner_payload)
    sanitized_payload["available"] = True
    sanitized_payload["entities"] = sanitized_entities
    sanitized_payload["slots"] = sanitized_slots
    sanitized_payload["slots_multi"] = slots_multi
    sanitized_payload["manufacturer_filter"] = sanitized_slots.get("manufacturer")
    sanitized_payload["sanitized"] = {
        "entity_count_before": len(ner_payload.get("entities", []) or []),
        "entity_count_after": len(sanitized_entities),
        "slot_count_after": len(sanitized_slots),
        "slot_multi_count": {k: len(v) for k, v in slots_multi.items()},
        "dropped_entities": dropped_entities[:10],
        "dropped_slots": dropped_slots,
    }
    return sanitized_payload


def _apply_slot_trust_policy(
    question_raw: str,
    ner_payload: dict[str, Any] | None,
    tmtid: str | None,
    manufacturer: str | None,
    strategy: str,
) -> dict[str, Any] | None:
    """
    Phase 1 trust policy:
    1) Suppress brand-like slots/entities when exact TMTID signal exists.
    2) If deterministic manufacturer matches NER brand text, drop brand duplicate.
    3) For compare queries with drug signal, suppress brand noise.
    """
    if not isinstance(ner_payload, dict):
        return ner_payload
    if not bool(ner_payload.get("available", False)):
        return ner_payload

    payload: dict[str, Any] = dict(ner_payload)
    slots = dict(payload.get("slots", {}) or {})
    slots_multi_raw = payload.get("slots_multi", {}) or {}
    slots_multi = {
        "brand": list(slots_multi_raw.get("brand", []) or []),
        "manufacturer": list(slots_multi_raw.get("manufacturer", []) or []),
        "drug": list(slots_multi_raw.get("drug", []) or []),
    }
    entities = [dict(item) for item in (payload.get("entities", []) or []) if isinstance(item, dict)]

    dropped: dict[str, list[str]] = {
        "brand_tmtid_conflict": [],
        "brand_manufacturer_conflict": [],
        "brand_compare_noise": [],
    }

    def _track_drop(reason: str, value: Any) -> None:
        text = _normalize_slot_text(value)
        if text:
            dropped.setdefault(reason, []).append(text)

    def _drop_brand_entities(predicate, reason: str) -> None:
        nonlocal entities
        filtered: list[dict[str, Any]] = []
        for item in entities:
            label = str(item.get("label", "")).upper().strip()
            text = _normalize_slot_text(item.get("text"))
            if label == "BRAND" and predicate(text):
                _track_drop(reason, text)
                continue
            filtered.append(item)
        entities = filtered

    # Rule 1: exact id lookup should suppress brand values carrying id-like text.
    if tmtid:
        if _is_tmtid_like_text(slots.get("brand"), tmtid):
            _track_drop("brand_tmtid_conflict", slots.get("brand"))
            slots.pop("brand", None)

        filtered_brand_values: list[str] = []
        for value in slots_multi.get("brand", []):
            if _is_tmtid_like_text(value, tmtid):
                _track_drop("brand_tmtid_conflict", value)
                continue
            filtered_brand_values.append(value)
        slots_multi["brand"] = filtered_brand_values

        _drop_brand_entities(lambda text: _is_tmtid_like_text(text, tmtid), "brand_tmtid_conflict")

    # Rule 2: deterministic manufacturer should win over same-text NER brand.
    manufacturer_norm = _canonical_slot_text(manufacturer)
    if manufacturer_norm:
        def _is_same_as_manufacturer(value: Any) -> bool:
            return _canonical_slot_text(value) == manufacturer_norm

        def _is_manufacturer_context_brand(value: Any) -> bool:
            normalized = _canonical_slot_text(value)
            if not normalized:
                return False
            tokens = set(normalized.split())
            return any(token in MANUFACTURER_CONTEXT_TERMS for token in tokens)

        if _is_same_as_manufacturer(slots.get("brand")):
            _track_drop("brand_manufacturer_conflict", slots.get("brand"))
            slots.pop("brand", None)
        elif _is_manufacturer_context_brand(slots.get("brand")):
            _track_drop("brand_manufacturer_conflict", slots.get("brand"))
            slots.pop("brand", None)

        filtered_brand_values = []
        for value in slots_multi.get("brand", []):
            if _is_same_as_manufacturer(value) or _is_manufacturer_context_brand(value):
                _track_drop("brand_manufacturer_conflict", value)
                continue
            filtered_brand_values.append(value)
        slots_multi["brand"] = filtered_brand_values

        _drop_brand_entities(
            lambda value: _is_same_as_manufacturer(value) or _is_manufacturer_context_brand(value),
            "brand_manufacturer_conflict",
        )

    # Rule 3: compare queries with clear drug signal should ignore brand noise.
    is_compare_query = strategy == "compare" or detect_compare(question_raw)
    has_drug_signal = bool(_normalize_slot_text(slots.get("drug"))) or bool(slots_multi.get("drug"))
    if is_compare_query and has_drug_signal:
        if slots.get("brand"):
            _track_drop("brand_compare_noise", slots.get("brand"))
            slots.pop("brand", None)

        if slots_multi.get("brand"):
            for value in slots_multi["brand"]:
                _track_drop("brand_compare_noise", value)
            slots_multi["brand"] = []

        _drop_brand_entities(lambda text: bool(_normalize_slot_text(text)), "brand_compare_noise")

    payload["slots"] = slots
    payload["slots_multi"] = slots_multi
    payload["entities"] = entities
    payload["manufacturer_filter"] = slots.get("manufacturer")

    sanitized = dict(payload.get("sanitized", {}) or {})
    policy_dropped = {
        key: list(dict.fromkeys(values))[:10]
        for key, values in dropped.items()
        if values
    }
    if policy_dropped:
        sanitized["policy_dropped"] = policy_dropped
    sanitized["slot_count_after"] = len(slots)
    sanitized["slot_multi_count"] = {k: len(v) for k, v in slots_multi.items()}
    payload["sanitized"] = sanitized

    return payload


LOOKUP_PREFIX_PATTERNS = (
    re.compile(r"^\s*(?:drug\s+information\s+about|information\s+about|info\s+about)\s+(.+)$", re.IGNORECASE),
    re.compile(r"^\s*(?:\u0e02\u0e49\u0e2d\u0e21\u0e39\u0e25\u0e22\u0e32)\s+(.+)$", re.IGNORECASE),
)

LIST_SUBSTANCE_PATTERNS = (
    re.compile(r"(?:list\s+drugs?\s+containing|drugs?\s+containing)\s+(.+)$", re.IGNORECASE),
    re.compile(r"(?:\u0e2a\u0e48\u0e27\u0e19\u0e1c\u0e2a\u0e21)\s+(.+)$", re.IGNORECASE),
)

LIST_MANUFACTURER_PATTERNS = (
    re.compile(r"(?:list\s+drugs?\s+by\s+manufacturer|by\s+manufacturer|manufacturer)\s+(.+)$", re.IGNORECASE),
    re.compile(r"(?:\u0e1c\u0e39\u0e49\u0e1c\u0e25\u0e34\u0e15|\u0e1c\u0e25\u0e34\u0e15\u0e42\u0e14\u0e22)\s+(.+)$", re.IGNORECASE),
)

DRUG_ANCHOR_PATTERNS = (
    re.compile(
        r"(?:\u0e22\u0e32|drug)\s+([A-Za-z][A-Za-z0-9\-]+(?:\s+\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|iu|%))?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"([A-Za-z][A-Za-z0-9\-]+)\s+(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|iu|%)",
        re.IGNORECASE,
    ),
)

TH_QUERY_NOISE_TOKENS = {
    "\u0e1c\u0e21",  # ผม
    "\u0e09\u0e31\u0e19",  # ฉัน
    "\u0e2d\u0e22\u0e32\u0e01",  # อยาก
    "\u0e02\u0e2d",  # ขอ
    "\u0e0a\u0e37\u0e48\u0e2d",  # ชื่อ
    "\u0e17\u0e32\u0e07",  # ทาง
    "\u0e01\u0e32\u0e23",  # การ
    "\u0e04\u0e49\u0e32",  # ค้า
    "\u0e42\u0e23\u0e07\u0e07\u0e32\u0e19",  # โรงงาน
    "\u0e1c\u0e39\u0e49\u0e1c\u0e25\u0e34\u0e15",  # ผู้ผลิต
    "\u0e17\u0e31\u0e49\u0e07\u0e2b\u0e21\u0e14",  # ทั้งหมด
    "\u0e0a\u0e37\u0e48\u0e2d",  # ชื่อ
    "\u0e23\u0e32\u0e22\u0e01\u0e32\u0e23",  # รายการ
    "\u0e2d\u0e30\u0e44\u0e23",  # อะไร
}


def _clean_search_candidate(value: str) -> str:
    text = _normalize_slot_text(value).strip(" ,.;:()[]{}\"'")
    text = re.sub(r"\s+", " ", text)
    if text.endswith("?"):
        text = text[:-1].strip()
    return text


def _is_query_like_text(value: str) -> bool:
    normalized = _normalize_slot_text(value).lower()
    if not normalized:
        return True
    query_like_tokens = {
        "drug information",
        "information about",
        "list drugs",
        "by manufacturer",
        "\u0e02\u0e49\u0e2d\u0e21\u0e39\u0e25\u0e22\u0e32",
        "\u0e23\u0e32\u0e22\u0e01\u0e32\u0e23\u0e22\u0e32",
    }
    return any(token in normalized for token in query_like_tokens)


def _extract_by_patterns(question_raw: str, patterns: tuple[re.Pattern, ...]) -> str | None:
    text = _normalize_slot_text(question_raw)
    if not text:
        return None
    for pattern in patterns:
        match = pattern.search(text)
        if not match:
            continue
        candidate = _clean_search_candidate(match.group(1))
        if candidate:
            return candidate
    return None


def _fallback_search_anchor(question_raw: str, manufacturer: str | None = None) -> str:
    text = _normalize_slot_text(question_raw)
    if not text:
        return ""

    for patterns in (LOOKUP_PREFIX_PATTERNS, LIST_SUBSTANCE_PATTERNS, LIST_MANUFACTURER_PATTERNS):
        candidate = _extract_by_patterns(text, patterns)
        if candidate and not _is_query_like_text(candidate):
            return candidate

    for pattern in DRUG_ANCHOR_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        if len(match.groups()) == 3:
            candidate = f"{match.group(1)} {match.group(2)} {match.group(3)}"
        else:
            candidate = match.group(1)
        cleaned = _clean_search_candidate(candidate)
        if cleaned and not _is_query_like_text(cleaned):
            return cleaned

    tokens = TOKEN_PATTERN.findall(text)
    filtered: list[str] = []
    manufacturer_tokens = set(TOKEN_PATTERN.findall((manufacturer or "").lower()))
    for token in tokens:
        token_lower = token.lower()
        if token_lower in EN_STOPWORDS:
            continue
        if token_lower in TH_QUERY_NOISE_TOKENS:
            continue
        if manufacturer and token_lower in manufacturer_tokens:
            continue
        if len(token) < 3 and not any(ch.isdigit() for ch in token):
            continue
        filtered.append(token)

    # Prefer first obvious drug-like English token (e.g., Paracetamol),
    # then attach nearest strength token if present.
    for idx, token in enumerate(filtered):
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9\-]{2,}", token):
            candidate = token
            if idx + 1 < len(filtered) and re.fullmatch(r"\d+(?:\.\d+)?", filtered[idx + 1]):
                candidate = f"{candidate} {filtered[idx + 1]}"
                if idx + 2 < len(filtered) and re.fullmatch(r"(?:mg|mcg|g|ml|iu|%)", filtered[idx + 2], re.IGNORECASE):
                    candidate = f"{candidate} {filtered[idx + 2]}"
            return candidate

    if filtered:
        return " ".join(filtered[:8])
    return text


def build_search_term(
    question_raw: str,
    tmtid: str | None,
    manufacturer: str | None,
    ner_slots: dict[str, Any] | None,
    ner_slots_multi: dict[str, list[str]] | None,
) -> str:
    if tmtid:
        return str(tmtid)

    slots = ner_slots or {}
    slots_multi = ner_slots_multi or {}
    
    # Priority: use multi-entity slots for better search term construction
    candidate_order = []
    
    # Add brand entities (multi if available, fallback to single)
    brand_entities = slots_multi.get("brand", [])
    if brand_entities:
        candidate_order.extend(brand_entities)
    elif slots.get("brand"):
        candidate_order.append(slots["brand"])
    
    # Add drug entities (multi if available, fallback to single)
    drug_entities = slots_multi.get("drug", [])
    if drug_entities:
        candidate_order.extend(drug_entities)
    elif slots.get("drug"):
        candidate_order.append(slots["drug"])
    
    # Build search term from prioritized entities
    for candidate in candidate_order:
        cleaned = _clean_search_candidate(str(candidate))
        if not cleaned:
            continue
        if _is_query_like_text(cleaned):
            continue
        return cleaned

    # Fallback only when NER slots cannot provide a reliable anchor.
    fallback = _fallback_search_anchor(question_raw, manufacturer=manufacturer)
    if fallback:
        return fallback
    return _light_normalize_question(question_raw)


def compute_entity_profile(
    question: str,
    manufacturer: str | None = None,
    nlem_filter: bool | None = None,
    nlem_category: str | None = None,
    ner_payload: dict | None = None,
) -> tuple[int, int, float, bool]:
    """
    Estimate entity specificity from hybrid signals (lexical + regex + NER).
    Returns: token_count, entity_token_count, entity_ratio, is_abstract
    """
    tokens = TOKEN_PATTERN.findall(question.lower())
    if not tokens:
        return 0, 0, 0.0, False

    entity_tokens: set[str] = set()
    for token in tokens:
        if UNIT_PATTERN.match(token):
            entity_tokens.add(token)
            continue
        if any(ch.isdigit() for ch in token):
            entity_tokens.add(token)
            continue
        if token in MEDICAL_HINTS:
            entity_tokens.add(token)
            continue
        if re.search(r"[A-Za-z]", token):
            if len(token) >= 3 and token not in EN_STOPWORDS:
                entity_tokens.add(token)
            continue

    # Regex entity spans provide strong, deterministic cues.
    for m in DOSE_UNIT_PATTERN.finditer(question.lower()):
        for token in TOKEN_PATTERN.findall(m.group(0)):
            entity_tokens.add(token)

    tmtid_match = TMTID_PATTERN.search(question)
    if tmtid_match:
        entity_tokens.add(str(tmtid_match.group(1)).lower())

    for m in BARE_ID_PATTERN.finditer(question):
        entity_tokens.add(str(m.group(0)).lower())

    if manufacturer:
        for token in TOKEN_PATTERN.findall(manufacturer.lower()):
            entity_tokens.add(token)
    if nlem_filter:
        entity_tokens.add("nlem")
    if nlem_category:
        entity_tokens.add(str(nlem_category).lower())

    # NER entities/slots are merged when available to improve specificity signal.
    if ner_payload and ner_payload.get("available"):
        ner_entities = ner_payload.get("entities", [])
        for item in ner_entities:
            if not isinstance(item, dict):
                continue
            confidence = float(item.get("confidence", item.get("score", 0.0)) or 0.0)
            if confidence < NER_SANITIZE_MIN_CONFIDENCE:
                continue
            text = str(item.get("text", "")).strip().lower()
            if not text:
                continue
            for token in TOKEN_PATTERN.findall(text):
                if len(token) >= 2 or any(ch.isdigit() for ch in token):
                    entity_tokens.add(token)

        ner_slots = ner_payload.get("slots", {}) or {}
        for key in ("drug", "brand", "manufacturer", "strength", "form"):
            value = ner_slots.get(key)
            if not value:
                continue
            for token in TOKEN_PATTERN.findall(str(value).lower()):
                if len(token) >= 2 or any(ch.isdigit() for ch in token):
                    entity_tokens.add(token)

    token_count = len(tokens)
    entity_token_count = min(len(entity_tokens), token_count)
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
    Adaptive profile for hybrid retrieval weights (Phase 2 draft table).
    Notes:
    - id_lookup override is handled by caller.
    - non-id profiles keep vector >= 0.35 and fulltext <= 0.65 guardrails.
    """
    ratio = max(0.0, min(1.0, float(entity_ratio)))

    # Count/List
    if strategy in {"count", "list"}:
        if ratio >= 0.60:
            return RetrievalMode.FULLTEXT_HEAVY, 0.35, 0.65
        return RetrievalMode.BALANCED, 0.50, 0.50

    # Verify
    if strategy == "verify":
        if ratio >= 0.60 and not is_abstract:
            return RetrievalMode.FULLTEXT_HEAVY, 0.40, 0.60
        if 0.35 <= ratio < 0.60:
            return RetrievalMode.BALANCED, 0.50, 0.50
        return RetrievalMode.VECTOR_HEAVY, 0.60, 0.40

    # Retrieve (default)
    if is_abstract or ratio < 0.30:
        return RetrievalMode.VECTOR_HEAVY, 0.70, 0.30
    if ratio < 0.60:
        return RetrievalMode.BALANCED, 0.55, 0.45
    return RetrievalMode.FULLTEXT_HEAVY, 0.40, 0.60


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
        "substance": TopicsIntent.SUBSTANCE,
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
    seen_slot_values: set[tuple[str, str]] = set()

    def _append_slot(name: str, value: Any, confidence: float, source: SlotSource) -> bool:
        text = _normalize_slot_text(value)
        if not text:
            return False
        key = (name, text.lower())
        if key in seen_slot_values:
            return False
        seen_slot_values.add(key)
        slot_values.append(SlotValue(name=name, value=text, confidence=confidence, source=source))
        return True

    if query_obj.id_lookup:
        if TopicsIntent.ID_LOOKUP not in topics_intents:
            topics_intents.append(TopicsIntent.ID_LOOKUP)
        _append_slot("tmtid", query_obj.id_lookup, 1.0, SlotSource.RULE)

    if query_obj.manufacturer_filter:
        if TopicsIntent.MANUFACTURER not in topics_intents:
            topics_intents.append(TopicsIntent.MANUFACTURER)
        _append_slot("manufacturer", query_obj.manufacturer_filter, 0.95, SlotSource.RULE)

    if query_obj.nlem_filter is not None:
        if TopicsIntent.NLEM not in topics_intents:
            topics_intents.append(TopicsIntent.NLEM)
        _append_slot("nlem", str(bool(query_obj.nlem_filter)).lower(), 0.95, SlotSource.RULE)
    if query_obj.nlem_category:
        _append_slot("nlem_category", query_obj.nlem_category, 0.95, SlotSource.RULE)

    # Capture NER slots for downstream routing/context.
    ner_slots = (ner_payload or {}).get("slots", {})
    ner_slots_multi = (ner_payload or {}).get("slots_multi", {})
    ner_slot_added = False

    # Multi-entity aware slots for searchable classes.
    for key in ("brand", "drug", "manufacturer"):
        multi_values = []
        if isinstance(ner_slots_multi, dict):
            multi_values = list(ner_slots_multi.get(key, []) or [])
        if multi_values:
            for idx, value in enumerate(multi_values):
                if _append_slot(key, value, 0.85 if idx == 0 else 0.82, SlotSource.NER):
                    ner_slot_added = True
            continue
        if _append_slot(key, ner_slots.get(key), 0.85, SlotSource.NER):
            ner_slot_added = True

    # Keep single-value pharmacological attributes.
    for key in ("strength", "form"):
        if _append_slot(key, ner_slots.get(key), 0.85, SlotSource.NER):
            ner_slot_added = True

    # Only expose rule query as fallback when NER slots are unavailable.
    if not ner_slot_added:
        _append_slot("query", query_obj.query, 0.70, SlotSource.RULE)


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

    metadata: dict[str, Any] = {
        "legacy_raw_intent": str(intent_result.get("intent", "")),
        "legacy_intent_confidence": float(intent_result.get("confidence", 0.0)),
        "legacy_intent_top_targets": list(intent_result.get("top_targets", []) or []),
        "legacy_target_confidence": float(intent_result.get("target_confidence", 0.0)),
        "legacy_target_margin": float(intent_result.get("target_margin", 0.0)),
        "is_abstract": bool(getattr(query_obj, "_is_abstract", False)),
        "is_ambiguous": is_ambiguous,
        "ner_available": bool((ner_payload or {}).get("available", False)),
        "query_slot_mode": "fallback_rule" if not ner_slot_added else "ner_primary",
        "ner_slots_multi": dict((ner_payload or {}).get("slots_multi", {}) or {}),
    }
    if ner_payload and ner_payload.get("sanitized"):
        metadata["ner_sanitized"] = ner_payload["sanitized"]
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
    question_raw = str(question or "").strip()
    question_for_intent = _light_normalize_question(question_raw)

    classifier = get_intent_classifier()
    if not classifier._initialized:
        classifier.initialize()

    # Axis 1: semantic target classification (vector centroid)
    intent_result = classifier.classify(question_for_intent, q_embedding)
    target_type = intent_result.get("target_type", intent_result.get("base_intent", "general"))

    # Axis 2: action strategy (rules)
    strategy = detect_strategy(question_for_intent)

    # NER extraction + sanitization (no aggressive cleanup of user question).
    ner_payload: dict[str, Any] | None = None
    if INTENT_V2_USE_NER:
        ner_payload = _sanitize_ner_payload(question_raw, get_ner_service().extract(question_raw))

    # Property filters (deterministic)
    tmtid = extract_tmtid(question_raw)
    manufacturer = extract_manufacturer(question_raw)
    nlem_filter = extract_nlem_filter(question_raw)
    nlem_category = extract_nlem_category(question_raw)
    if nlem_category and nlem_filter is None:
        nlem_filter = True

    if not manufacturer and (ner_payload or {}).get("manufacturer_filter"):
        manufacturer = str(ner_payload["manufacturer_filter"])

    ner_payload = _apply_slot_trust_policy(
        question_raw=question_raw,
        ner_payload=ner_payload,
        tmtid=tmtid,
        manufacturer=manufacturer,
        strategy=strategy,
    )

    ner_slots = (ner_payload or {}).get("slots", {})
    ner_slots_multi = (ner_payload or {}).get("slots_multi", {})
    search_term = build_search_term(
        question_raw=question_raw,
        tmtid=tmtid,
        manufacturer=manufacturer,
        ner_slots=ner_slots,
        ner_slots_multi=ner_slots_multi,
    )

    # Target override when filter signal is explicit
    if nlem_filter:
        target_type = "nlem"
    elif manufacturer:
        target_type = "manufacturer"

    # If form/strength exists, favor formula scope.
    if target_type == "general" and (ner_slots.get("strength") or ner_slots.get("form")):
        target_type = "formula"

    # Query fallback for list/count style questions
    if strategy in {"count", "list"} and not search_term:
        search_term = "ยา"

    token_count, entity_token_count, entity_ratio, is_abstract = compute_entity_profile(
        question=question_raw,
        manufacturer=manufacturer,
        nlem_filter=nlem_filter,
        nlem_category=nlem_category,
        ner_payload=ner_payload,
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
    query_obj._question_raw = question_raw
    query_obj._question_for_intent = question_for_intent
    query_obj._ner_payload = ner_payload or {}

    # IntentV2 bundle payload for search/debug.
    if INTENT_V2_ENABLED:
        bundle = _build_intent_bundle(
            question=question_raw,
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
        "version": "4.1.0-intent-v2-raw-first-ner-sanitized",
    }
