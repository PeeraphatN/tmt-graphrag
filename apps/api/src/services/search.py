"""
Search Service.
Operator-driven search with 3-channel retrieval:
- vector similarity
- fulltext search
- graph-traversal candidate generation
"""

from __future__ import annotations

import math
import re
from typing import Any

from src.config import FULLTEXT_INDEX_NAME, LIST_MAX_K_CAP, RETRIEVAL_EVAL_MODE, VECTOR_INDEX_NAME
from src.llm_service import get_embedding
from src.query_processor import sanitize_fulltext_query
from src.services.database import check_index_exists, init_driver
from src.services.ranking_service import get_reranker

COMPARE_SPLIT_PATTERN = re.compile(
    r"\b(?:vs\.?|versus|and|or|with)\b|[\u0E00-\u0E7F]*\s*(?:à¸à¸±à¸š|à¸à¸š|à¹à¸¥à¸°|à¸«à¸£à¸·à¸­)\s*[\u0E00-\u0E7F]*",
    re.IGNORECASE,
)

GRAPH_TRAVERSAL_REL_TYPES = (
    "HAS_ACTIVE_SUBSTANCE",
    "IS_ACTIVE_SUBSTANCE_OF",
    "HAS_VTM",
    "IS_VTM_OF",
    "HAS_GENERIC_UNIT",
    "IS_GENERIC_UNIT_OF",
    "HAS_TRADE_PRODUCT",
    "IS_TRADE_PRODUCT_OF",
    "HAS_TRADE_UNIT",
    "IS_UNIT_OF_TRADE_PRODUCT",
    "HAS_TRADE_EQUIVALENT",
    "IS_TRADE_EQUIVALENT_OF",
)

PRODUCT_LEVELS = ("GP", "GPU", "TP", "TPU")
HIERARCHY_LEVELS = ("SUBS", "VTM", "GP", "GPU", "TP", "TPU")
ENABLE_COMPARE_EXPANDED_RERANK = False

_EMBEDDING_CACHE: dict[str, list[float]] = {}
_INDEX_EXISTS_CACHE: dict[str, bool] = {}

LIST_SUBSTANCE_PATTERNS = (
    re.compile(r"(?:list\s+drugs?\s+containing|drugs?\s+containing)\s+(.+)$", re.IGNORECASE),
    re.compile(r"(?:\u0e2a\u0e48\u0e27\u0e19\u0e1c\u0e2a\u0e21)\s+(.+)$", re.IGNORECASE),
)

LOOKUP_ANCHOR_SUBSTANCE_PATTERNS = LIST_SUBSTANCE_PATTERNS + (
    re.compile(r"(?:trade\s+name(?:s)?(?:\s+of)?|brand\s+name(?:s)?(?:\s+of)?)\s+(.+)$", re.IGNORECASE),
    re.compile(r"(?:who\s+manufactures|manufacturer\s+of)\s+(.+)$", re.IGNORECASE),
    re.compile(
        r"(?:\u0e0a\u0e37\u0e48\u0e2d\u0e17\u0e32\u0e07\u0e01\u0e32\u0e23\u0e04\u0e49\u0e32(?:\u0e02\u0e2d\u0e07\u0e22\u0e32)?|"
        r"\u0e22\u0e35\u0e48\u0e2b\u0e49\u0e2d\u0e22\u0e32)\s*(.+)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:\u0e42\u0e23\u0e07\u0e07\u0e32\u0e19\u0e17\u0e35\u0e48\u0e1c\u0e25\u0e34\u0e15|"
        r"\u0e1c\u0e39\u0e49\u0e1c\u0e25\u0e34\u0e15)\s*(.+)$",
        re.IGNORECASE,
    ),
)

LIST_MANUFACTURER_PATTERNS = (
    re.compile(r"(?:list\s+drugs?\s+by\s+manufacturer|by\s+manufacturer|manufacturer)\s+(.+)$", re.IGNORECASE),
    re.compile(r"(?:\u0e1c\u0e39\u0e49\u0e1c\u0e25\u0e34\u0e15|\u0e1c\u0e25\u0e34\u0e15\u0e42\u0e14\u0e22)\s+(.+)$", re.IGNORECASE),
)

MANUFACTURER_REQUEST_HINTS = (
    "manufacturer",
    "manufactures",
    "factory",
    "\u0e1c\u0e39\u0e49\u0e1c\u0e25\u0e34\u0e15",  # à¸œà¸¹à¹‰à¸œà¸¥à¸´à¸•
    "\u0e1a\u0e23\u0e34\u0e29\u0e31\u0e17",      # à¸šà¸£à¸´à¸©à¸±à¸—
    "\u0e42\u0e23\u0e07\u0e07\u0e32\u0e19",      # à¹‚à¸£à¸‡à¸‡à¸²à¸™
    "\u0e1c\u0e25\u0e34\u0e15\u0e42\u0e14\u0e22",  # à¸œà¸¥à¸´à¸•à¹‚à¸”à¸¢
)

TRADE_NAME_REQUEST_HINTS = (
    "trade name",
    "\u0e0a\u0e37\u0e48\u0e2d\u0e17\u0e32\u0e07\u0e01\u0e32\u0e23\u0e04\u0e49\u0e32",
)


COUNT_SUBSTANCE_SCOPE_LEVELS = ("GP",)
COUNT_MANUFACTURER_SCOPE_LEVELS = ("TP",)
COUNT_PRODUCT_LEVELS = ("GP", "GPU", "TP", "TPU")
COUNT_HIERARCHY_LEVELS = ("SUBS", "VTM", "GP", "GPU", "TP", "TPU")

STRENGTH_TOKEN_PATTERN = re.compile(
    r"(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>mg|mcg|μg|ug|g|ml|iu|%)\b",
    re.IGNORECASE,
)
STRENGTH_UNIT_CANONICAL = {
    "μg": "mcg",
    "ug": "mcg",
}


def _enum_value(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _canonicalize_strength_unit(unit: str) -> str:
    normalized = _normalize_text(unit).lower()
    return STRENGTH_UNIT_CANONICAL.get(normalized, normalized)


def _normalize_strength_expression(text: str) -> str:
    normalized_text = _normalize_text(text)
    if not normalized_text:
        return ""

    def _replace(match: re.Match[str]) -> str:
        number = match.group("num")
        unit = _canonicalize_strength_unit(match.group("unit"))
        return f"{number}{unit}"

    return STRENGTH_TOKEN_PATTERN.sub(_replace, normalized_text)


def _normalize_query_for_search(text: Any) -> str:
    return _normalize_strength_expression(_normalize_text(text))


def _strip_strength_tokens(text: Any) -> str:
    cleaned = STRENGTH_TOKEN_PATTERN.sub("", _normalize_text(text))
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.;:()[]{}\"'")
    return cleaned


def _build_search_variants(text: Any) -> list[str]:
    raw = _normalize_text(text)
    if not raw:
        return []
    base = _normalize_strength_expression(raw)

    variants: list[str] = []
    seen: set[str] = set()

    def _push(value: str) -> None:
        normalized = _normalize_text(value)
        if not normalized:
            return
        key = normalized.lower()
        if key in seen:
            return
        seen.add(key)
        variants.append(normalized)

    # Keep original text first to preserve prior ranking behavior.
    _push(raw)
    _push(base)

    # Add spaced-strength variant to improve exact/fulltext matching when DB keeps "500 mg" form.
    spaced = re.sub(
        r"(?P<num>\d+(?:\.\d+)?)(?P<unit>mg|mcg|g|ml|iu|%)\b",
        r"\g<num> \g<unit>",
        base,
        flags=re.IGNORECASE,
    )
    _push(spaced)
    return variants


def _get_embedding_cached(text: str) -> list[float]:
    key = _normalize_text(text)
    if not key:
        return []
    cached = _EMBEDDING_CACHE.get(key)
    if cached is not None:
        return cached
    value = get_embedding(key) or []
    _EMBEDDING_CACHE[key] = value
    return value


def _check_index_exists_cached(session, index_name: str) -> bool:
    if index_name in _INDEX_EXISTS_CACHE:
        return _INDEX_EXISTS_CACHE[index_name]
    exists = bool(check_index_exists(session, index_name))
    _INDEX_EXISTS_CACHE[index_name] = exists
    return exists


def reset_runtime_caches() -> None:
    _EMBEDDING_CACHE.clear()
    _INDEX_EXISTS_CACHE.clear()


def _build_filters_from_query_obj(query_obj) -> dict:
    filters = {}
    if getattr(query_obj, "nlem_filter", None) is not None:
        filters["nlem"] = query_obj.nlem_filter
    if getattr(query_obj, "nlem_category", None):
        filters["nlem_category"] = query_obj.nlem_category
    if getattr(query_obj, "manufacturer_filter", None):
        filters["manufacturer"] = query_obj.manufacturer_filter
    if getattr(query_obj, "id_lookup", None):
        filters["tmtid"] = query_obj.id_lookup
    return filters


def _extract_slot_values(intent_bundle: dict | None, slot_name: str) -> list[str]:
    if not isinstance(intent_bundle, dict):
        return []
    values: list[str] = []
    seen: set[str] = set()
    for slot in intent_bundle.get("slots", []) or []:
        if not isinstance(slot, dict):
            continue
        if _normalize_text(slot.get("name")) != slot_name:
            continue
        value = _normalize_text(slot.get("value"))
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        values.append(value)
    return values


def _resolve_action_intent(query_obj) -> str:
    bundle = getattr(query_obj, "intent_bundle", None)
    if isinstance(bundle, dict):
        action = _normalize_text(bundle.get("action_intent")).lower()
        if action:
            return action

    strategy = _enum_value(getattr(query_obj, "strategy", "retrieve")).lower()
    if strategy == "verify":
        return "verify"
    if strategy == "count":
        return "count"
    if strategy == "list":
        return "list"
    return "lookup"


def _resolve_operator(query_obj) -> str:
    if getattr(query_obj, "id_lookup", None):
        return "id_lookup"

    action = _resolve_action_intent(query_obj)
    if action == "verify":
        return "verify"
    if action == "count":
        return "analyze_count"
    if action == "compare":
        return "analyze_compare"
    if action == "list":
        return "list"
    return "lookup"


def _merge_seed_results(result_groups: list[list[dict]]) -> list[dict]:
    merged: list[dict] = []
    seen_ids: set[str] = set()
    for group in result_groups:
        for item in group:
            node = item.get("node")
            if node is None:
                continue
            nid = node.element_id if hasattr(node, "element_id") else node.id
            if nid in seen_ids:
                continue
            seen_ids.add(nid)
            merged.append(item)
    return merged


def _split_compare_terms(query_text: str) -> list[str]:
    text = _normalize_query_for_search(query_text)
    if not text:
        return []

    parts = [_normalize_query_for_search(part) for part in COMPARE_SPLIT_PATTERN.split(text)]
    parts = [part for part in parts if len(part) >= 2]

    deduped: list[str] = []
    seen: set[str] = set()
    for part in parts:
        key = part.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(part)
    return deduped


def _safe_top_seed(seed_results: list[dict]) -> dict[str, Any]:
    if not seed_results:
        return {}
    item = seed_results[0]
    node = item.get("node")
    if node is None:
        return {}
    props = dict(node)
    return {
        "tmtid": props.get("tmtid"),
        "level": props.get("level"),
        "name": props.get("trade_name") or props.get("fsn") or props.get("name"),
        "manufacturer": props.get("manufacturer"),
        "score": item.get("rrf_score", item.get("score")),
    }


def _build_filter_parts(
    alias: str,
    allowed_levels: list[str] | set[str] | tuple[str, ...] | None,
    filters: dict | None,
    param_prefix: str,
) -> tuple[list[str], dict[str, Any]]:
    parts: list[str] = []
    params: dict[str, Any] = {}

    if allowed_levels:
        key = f"{param_prefix}allowed_levels"
        parts.append(f"{alias}.level IN ${key}")
        params[key] = list(allowed_levels)

    if filters:
        if filters.get("nlem") is not None:
            key = f"{param_prefix}nlem_val"
            parts.append(f"{alias}.nlem = ${key}")
            params[key] = bool(filters["nlem"])

        if filters.get("nlem_category"):
            key = f"{param_prefix}nlem_cat"
            parts.append(f"{alias}.nlem_category = ${key}")
            params[key] = str(filters["nlem_category"])

        if filters.get("manufacturer"):
            key = f"{param_prefix}manu"
            parts.append(f"toLower(coalesce({alias}.manufacturer, '')) CONTAINS ${key}")
            params[key] = str(filters["manufacturer"]).lower()

        if filters.get("tmtid"):
            key = f"{param_prefix}tmtid"
            parts.append(f"{alias}.tmtid = ${key}")
            params[key] = str(filters["tmtid"])

    return parts, params


def _normalize_channel_weights(vector_weight: float, fulltext_weight: float, graph_weight: float) -> dict[str, float]:
    v = max(0.0, float(vector_weight))
    f = max(0.0, float(fulltext_weight))
    g = max(0.0, float(graph_weight))
    total = v + f + g
    if total <= 1e-9:
        return {"vector": 1.0 / 3.0, "fulltext": 1.0 / 3.0, "graph": 1.0 / 3.0}
    return {"vector": v / total, "fulltext": f / total, "graph": g / total}


def _exact_match_candidates(
    query_text: str,
    allowed_levels: list[str] | set[str] | tuple[str, ...] | None,
    filters: dict | None,
    channel_k: int = 50,
) -> list[dict[str, Any]]:
    """
    Deterministic lexical channel for lookup:
    prefer exact/startswith matches on key text fields before generic contains.
    """
    normalized = _normalize_text(query_text).lower()
    if len(normalized) < 3:
        return []

    parts, params = _build_filter_parts("n", allowed_levels, filters, "exact_")
    where_prefix = f"{' AND '.join(parts)} AND " if parts else ""
    params.update({"exact_q": normalized, "exact_k": int(max(1, channel_k))})

    cypher = f"""
    MATCH (n:TMT)
    WHERE {where_prefix} (
        toLower(coalesce(n.fsn, '')) CONTAINS $exact_q
        OR toLower(coalesce(n.trade_name, '')) CONTAINS $exact_q
        OR toLower(coalesce(n.name, '')) CONTAINS $exact_q
        OR toLower(coalesce(n.generic_name, '')) CONTAINS $exact_q
        OR toLower(coalesce(n.active_substance, '')) CONTAINS $exact_q
        OR toLower(coalesce(n.active_substances, '')) CONTAINS $exact_q
    )
    WITH n,
         CASE
             WHEN toLower(coalesce(n.fsn, '')) = $exact_q THEN 0
             WHEN toLower(coalesce(n.trade_name, '')) = $exact_q THEN 0
             WHEN toLower(coalesce(n.name, '')) = $exact_q THEN 0
             WHEN toLower(coalesce(n.generic_name, '')) = $exact_q THEN 0
             WHEN toLower(coalesce(n.fsn, '')) STARTS WITH $exact_q THEN 1
             WHEN toLower(coalesce(n.trade_name, '')) STARTS WITH $exact_q THEN 1
             WHEN toLower(coalesce(n.name, '')) STARTS WITH $exact_q THEN 1
             WHEN toLower(coalesce(n.generic_name, '')) STARTS WITH $exact_q THEN 1
             ELSE 2
         END AS pri,
         abs(size(toLower(coalesce(n.fsn, n.trade_name, n.name, n.generic_name, ''))) - size($exact_q)) AS delta
    RETURN n, pri, delta
    ORDER BY pri ASC, delta ASC
    LIMIT $exact_k
    """

    drv = init_driver()
    candidates: list[dict[str, Any]] = []
    with drv.session() as session:
        records = session.run(cypher, **params)
        for idx, rec in enumerate(records):
            candidates.append(
                {
                    "node": rec["n"],
                    "exact_rank": idx,
                    "exact_pri": rec.get("pri", 2),
                    "exact_delta": rec.get("delta", 9999),
                }
            )
    return candidates


def _default_graph_weight(strategy_label: str, target_type: str) -> float:
    strategy = _normalize_text(strategy_label).lower()
    target = _normalize_text(target_type).lower()

    if strategy in {"verify", "compare", "list"}:
        base = 0.35
    elif strategy in {"count", "id_lookup"}:
        base = 0.20
    else:
        base = 0.28

    if target in {"hierarchy", "substance", "manufacturer"}:
        base += 0.05

    return max(0.15, min(0.55, base))


def _resolve_channel_weights(
    strategy_label: str,
    target_type: str,
    vector_weight: float,
    fulltext_weight: float,
    graph_weight_override: float | None = None,
) -> dict[str, float]:
    v = max(0.0, float(vector_weight))
    f = max(0.0, float(fulltext_weight))
    vf_total = v + f
    if vf_total <= 1e-9:
        v = 0.5
        f = 0.5
        vf_total = 1.0

    g = (
        max(0.0, min(0.9, float(graph_weight_override)))
        if graph_weight_override is not None
        else _default_graph_weight(strategy_label, target_type)
    )
    remain = max(0.0, 1.0 - g)
    scaled_v = remain * (v / vf_total)
    scaled_f = remain * (f / vf_total)
    return _normalize_channel_weights(scaled_v, scaled_f, g)


def _collect_graph_seed_terms(question: str, seed_terms: list[str] | None, filters: dict | None) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()

    def _push(term: str) -> None:
        value = _normalize_text(term)
        if len(value) < 2:
            return
        key = value.lower()
        if key in seen:
            return
        seen.add(key)
        terms.append(value)

    _push(question)
    for term in seed_terms or []:
        _push(term)
    if filters and filters.get("manufacturer"):
        _push(str(filters["manufacturer"]))
    if filters and filters.get("tmtid"):
        _push(str(filters["tmtid"]))

    return terms[:4]


def _rrf_component(rank: int, k_rrf: int = 60) -> float:
    if rank >= 9999:
        return 0.0
    return 1.0 / (k_rrf + rank + 1)


def get_search_config(target_type: str) -> dict:
    """
    Returns search configuration based on target_type (intent).
    """
    nlem_fields = ["nlem", "nlem_category", "nlem_section"]
    base_fields = ["trade_name", "manufacturer", "fsn", "tmtid", "level"]

    config = {
        "allowed_levels": set(HIERARCHY_LEVELS),
        "required_fields": base_fields + nlem_fields,
        "max_entities": 60,
    }

    if target_type == "manufacturer":
        config["required_fields"] = ["trade_name", "manufacturer", "fsn", "tmtid", "level", "container_text"] + nlem_fields
        config["allowed_levels"] = {"TP", "TPU"}
        config["max_entities"] = 40
    elif target_type == "substance":
        config["required_fields"] = ["fsn", "active_substance", "active_substances", "strength", "tmtid", "level", "dosageform"] + nlem_fields
        config["allowed_levels"] = {"GP", "GPU", "TP", "TPU", "VTM", "SUBS"}
        config["max_entities"] = 60
    elif target_type == "nlem":
        config["required_fields"] = ["fsn", "tmtid", "level"] + nlem_fields
        config["allowed_levels"] = {"GP"}
        config["max_entities"] = 80
    elif target_type == "hierarchy":
        config["required_fields"] = ["level", "fsn", "tmtid"] + nlem_fields
        config["allowed_levels"] = {"SUBS", "VTM", "GP", "GPU", "TP", "TPU"}
        config["max_entities"] = 60
    elif target_type == "formula":
        config["required_fields"] = ["trade_name", "manufacturer", "fsn", "tmtid", "level"] + nlem_fields
        config["allowed_levels"] = {"TP", "TPU", "GP", "GPU"}
        config["max_entities"] = 50

    return config


def _vector_channel_search(
    session,
    question: str,
    channel_k: int,
    allowed_levels: list[str] | None,
    filters: dict | None,
) -> list[dict]:
    embedding = _get_embedding_cached(question)
    if not embedding:
        return []

    if not _check_index_exists_cached(session, VECTOR_INDEX_NAME):
        print(f"Index {VECTOR_INDEX_NAME} not found. Skipping vector search.")
        return []

    parts, params = _build_filter_parts("node", allowed_levels, filters, "vec_")
    where_clause = f"WHERE {' AND '.join(parts)}" if parts else ""
    query = f"""
    CALL db.index.vector.queryNodes($index_name, $k, $embedding)
    YIELD node, score
    {where_clause}
    RETURN node, score
    """
    query_params = {"index_name": VECTOR_INDEX_NAME, "k": channel_k, "embedding": embedding}
    query_params.update(params)
    try:
        recs = session.run(query, **query_params)
        return [{"node": rec["node"], "score": float(rec["score"] or 0.0)} for rec in recs]
    except Exception as exc:
        print(f"Vector search error: {exc}")
        return []


def _fulltext_channel_search(
    session,
    question: str,
    channel_k: int,
    allowed_levels: list[str] | None,
    filters: dict | None,
) -> list[dict]:
    sanitized_query, use_fulltext = sanitize_fulltext_query(question)
    if not use_fulltext or not sanitized_query:
        return []

    if not _check_index_exists_cached(session, FULLTEXT_INDEX_NAME):
        print(f"Index {FULLTEXT_INDEX_NAME} not found. Skipping fulltext search.")
        return []

    parts, params = _build_filter_parts("node", allowed_levels, filters, "ft_")
    where_clause = f"WHERE {' AND '.join(parts)}" if parts else ""
    query = f"""
    CALL db.index.fulltext.queryNodes($index_name, $search_text, {{limit: $k}})
    YIELD node, score
    {where_clause}
    RETURN node, score
    """
    query_params = {"index_name": FULLTEXT_INDEX_NAME, "search_text": sanitized_query, "k": channel_k}
    query_params.update(params)
    try:
        recs = session.run(query, **query_params)
        return [{"node": rec["node"], "score": float(rec["score"] or 0.0)} for rec in recs]
    except Exception as exc:
        print(f"Fulltext search error: {exc}")
        return []


def _graph_anchor_search(
    session,
    seed_terms: list[str],
    anchor_k: int,
    filters: dict | None,
) -> list[dict]:
    anchors: dict[str, dict[str, Any]] = {}
    if not seed_terms:
        return []

    fulltext_available = _check_index_exists_cached(session, FULLTEXT_INDEX_NAME)

    def _upsert_anchor(node, score: float, rank: int) -> None:
        nid = node.element_id if hasattr(node, "element_id") else node.id
        prev = anchors.get(nid)
        if prev is None or score > prev["score"]:
            anchors[nid] = {
                "node": node,
                "score": score,
                "rank": rank,
            }

    for term in seed_terms:
        normalized_term = _normalize_text(term).lower()
        if not normalized_term:
            continue
        found_in_term = 0
        sanitized, use_fulltext = sanitize_fulltext_query(term)
        if fulltext_available and use_fulltext and sanitized:
            try:
                recs = session.run(
                    """
                    CALL db.index.fulltext.queryNodes($index_name, $search_text, {limit: $k})
                    YIELD node, score
                    RETURN node, score
                    """,
                    index_name=FULLTEXT_INDEX_NAME,
                    search_text=sanitized,
                    k=anchor_k,
                )
                for rank, rec in enumerate(recs):
                    _upsert_anchor(rec["node"], float(rec["score"] or 0.0), rank)
                    found_in_term += 1
            except Exception as exc:
                print(f"Graph anchor fulltext error: {exc}")

        # Fallback anchor path:
        # when fulltext is unavailable/empty or query sanitizer becomes too strict.
        if found_in_term == 0:
            try:
                recs = session.run(
                    """
                    MATCH (n:TMT)
                    WHERE toLower(coalesce(n.name, '')) CONTAINS $term
                       OR toLower(coalesce(n.fsn, '')) CONTAINS $term
                       OR toLower(coalesce(n.trade_name, '')) CONTAINS $term
                       OR toLower(coalesce(n.generic_name, '')) CONTAINS $term
                       OR toLower(coalesce(n.active_substance, '')) CONTAINS $term
                       OR toLower(coalesce(n.active_substances, '')) CONTAINS $term
                       OR toLower(coalesce(n.manufacturer, '')) CONTAINS $term
                       OR toLower(coalesce(n.tmtid, '')) = $term
                    RETURN n AS node
                    ORDER BY CASE n.level
                        WHEN 'SUBS' THEN 1
                        WHEN 'VTM' THEN 2
                        WHEN 'GP' THEN 3
                        WHEN 'TP' THEN 4
                        WHEN 'GPU' THEN 5
                        WHEN 'TPU' THEN 6
                        ELSE 7
                    END,
                    coalesce(n.trade_name, n.fsn, n.name)
                    LIMIT $k
                    """,
                    term=normalized_term,
                    k=anchor_k,
                )
                for rank, rec in enumerate(recs):
                    _upsert_anchor(rec["node"], 200.0 - rank, rank)
            except Exception as exc:
                print(f"Graph anchor lexical fallback error: {exc}")

    manu = str(filters.get("manufacturer", "")).strip().lower() if filters else ""
    if manu:
        try:
            recs = session.run(
                """
                MATCH (n:TMT)
                WHERE n.level IN ['TP','TPU']
                  AND toLower(coalesce(n.manufacturer, '')) CONTAINS $manu
                RETURN n AS node
                LIMIT $k
                """,
                manu=manu,
                k=anchor_k,
            )
            for rank, rec in enumerate(recs):
                _upsert_anchor(rec["node"], 350.0 - rank, rank)
        except Exception as exc:
            print(f"Graph anchor manufacturer fallback error: {exc}")

    tmtid = str(filters.get("tmtid", "")).strip() if filters else ""
    if tmtid:
        try:
            recs = session.run(
                """
                MATCH (n:TMT {tmtid: $tmtid})
                RETURN n AS node
                LIMIT 3
                """,
                tmtid=tmtid,
            )
            for rec in recs:
                _upsert_anchor(rec["node"], 999.0, 0)
        except Exception as exc:
            print(f"Graph anchor id error: {exc}")

    anchor_list = list(anchors.values())
    anchor_list.sort(key=lambda item: (-item["score"], item["rank"]))
    return anchor_list[:anchor_k]


def _graph_traversal_channel_search(
    session,
    question: str,
    channel_k: int,
    allowed_levels: list[str] | None,
    filters: dict | None,
    seed_terms: list[str] | None,
    max_hops: int = 2,
) -> list[dict]:
    if max_hops < 1:
        return []

    terms = _collect_graph_seed_terms(question, seed_terms, filters)
    anchors = _graph_anchor_search(session, terms, anchor_k=max(8, channel_k), filters=filters)
    if not anchors:
        return []

    anchor_ids = [
        anchor["node"].element_id if hasattr(anchor["node"], "element_id") else anchor["node"].id
        for anchor in anchors
    ]

    parts, params = _build_filter_parts("cand", allowed_levels, filters, "g_")
    extra_filter = f" AND {' AND '.join(parts)}" if parts else ""

    traversal_cypher = f"""
    MATCH path = (anchor)-[*0..{max_hops}]-(cand:TMT)
    WHERE elementId(anchor) IN $anchor_ids
      AND all(rel IN relationships(path) WHERE type(rel) IN $rel_types)
      {extra_filter}
    WITH cand AS node, min(length(path)) AS min_hop, count(path) AS support
    RETURN node, min_hop, support
    ORDER BY min_hop ASC, support DESC
    LIMIT $k
    """
    query_params = {
        "anchor_ids": anchor_ids,
        "rel_types": list(GRAPH_TRAVERSAL_REL_TYPES),
        "k": channel_k,
    }
    query_params.update(params)

    rows: list[dict] = []
    try:
        recs = session.run(traversal_cypher, **query_params)
        for rec in recs:
            min_hop = int(rec["min_hop"] or 0)
            support = int(rec["support"] or 0)
            score = (1.0 / (1.0 + min_hop)) + min(0.45, math.log1p(support) / 6.0)
            rows.append(
                {
                    "node": rec["node"],
                    "score": score,
                    "hop": min_hop,
                    "support": support,
                }
            )
    except Exception as exc:
        print(f"Graph traversal search error: {exc}")
        return []

    return rows


def _fuse_three_channels(
    vector_rows: list[dict],
    fulltext_rows: list[dict],
    graph_rows: list[dict],
    weights: dict[str, float],
    k: int,
) -> list[dict]:
    fused: dict[str, dict[str, Any]] = {}

    def _upsert(rows: list[dict], prefix: str) -> None:
        for rank, row in enumerate(rows):
            node = row.get("node")
            if node is None:
                continue
            nid = node.element_id if hasattr(node, "element_id") else node.id
            bucket = fused.setdefault(
                nid,
                {
                    "node": node,
                    "v_score": 0.0,
                    "t_score": 0.0,
                    "g_score": 0.0,
                    "v_rank": 9999,
                    "t_rank": 9999,
                    "g_rank": 9999,
                    "graph_hop": None,
                    "graph_support": None,
                },
            )
            bucket[f"{prefix}_score"] = float(row.get("score", 0.0) or 0.0)
            bucket[f"{prefix}_rank"] = rank
            if prefix == "g":
                bucket["graph_hop"] = row.get("hop")
                bucket["graph_support"] = row.get("support")

    _upsert(vector_rows, "v")
    _upsert(fulltext_rows, "t")
    _upsert(graph_rows, "g")

    final_results: list[dict] = []
    for row in fused.values():
        rrf_score = (
            weights["vector"] * _rrf_component(int(row["v_rank"]))
            + weights["fulltext"] * _rrf_component(int(row["t_rank"]))
            + weights["graph"] * _rrf_component(int(row["g_rank"]))
        )
        row["rrf_score"] = float(rrf_score)
        final_results.append(row)

    final_results.sort(key=lambda item: item["rrf_score"], reverse=True)
    return final_results[:k]


def hybrid_search(
    question: str,
    k: int = 5,
    allowed_levels: list[str] = None,
    filters: dict = None,
    vector_weight: float = 0.5,
    fulltext_weight: float = 0.5,
    graph_weight: float = 0.0,
    seed_terms: list[str] | None = None,
    max_graph_hops: int = 2,
) -> list[dict]:
    """
    3-channel search (Vector + Fulltext + Graph Traversal) fused with weighted RRF.
    """
    channel_k = max(int(k) * 2, 12)
    weights = _normalize_channel_weights(vector_weight, fulltext_weight, graph_weight)
    drv = init_driver()

    with drv.session() as session:
        vector_rows = _vector_channel_search(session, question, channel_k, allowed_levels, filters)
        fulltext_rows = _fulltext_channel_search(session, question, channel_k, allowed_levels, filters)
        graph_rows = _graph_traversal_channel_search(
            session,
            question,
            channel_k,
            allowed_levels,
            filters,
            seed_terms=seed_terms,
            max_hops=max_graph_hops,
        )

    return _fuse_three_channels(vector_rows, fulltext_rows, graph_rows, weights, max(1, int(k)))


def expand_context(node_ids: list[str], depth: int = 2) -> dict:
    """
    Expand context by traversing relationships from seed nodes.
    """
    if not node_ids:
        return {"nodes": [], "relationships": [], "paths": []}

    drv = init_driver()
    expanded_nodes = {}
    relationships = []
    seen_rel_keys: set[str] = set()
    paths = []

    with drv.session() as session:
        traversal_query = f"""
        MATCH path = (n)-[r*1..{max(1, int(depth))}]-(m)
        WHERE elementId(n) IN $node_ids
        RETURN path,
               [rel IN relationships(path) | {{
                   type: type(rel),
                   start_id: elementId(startNode(rel)),
                   end_id: elementId(endNode(rel)),
                   start_labels: labels(startNode(rel)),
                   end_labels: labels(endNode(rel)),
                   start_name: coalesce(startNode(rel).name, startNode(rel).fsn, startNode(rel).trade_name, 'Unknown'),
                   end_name: coalesce(endNode(rel).name, endNode(rel).fsn, endNode(rel).trade_name, 'Unknown')
               }}] AS rels,
               [node IN nodes(path) | node] AS path_nodes
        LIMIT 80
        """
        try:
            records = session.run(traversal_query, node_ids=node_ids)
            for rec in records:
                for node in rec["path_nodes"]:
                    nid = node.element_id if hasattr(node, "element_id") else node.id
                    if nid not in expanded_nodes:
                        expanded_nodes[nid] = {
                            "node": node,
                            "labels": list(node.labels),
                            "is_seed": nid in node_ids,
                        }

                for rel_info in rec["rels"]:
                    rel_key = f"{rel_info['start_id']}-{rel_info['type']}-{rel_info['end_id']}"
                    if rel_key in seen_rel_keys:
                        continue
                    seen_rel_keys.add(rel_key)
                    rel_info["key"] = rel_key
                    relationships.append(rel_info)

                path = rec["path"]
                if path:
                    paths.append(path)
        except Exception as exc:
            print(f"Relationship traversal error: {exc}")

    return {
        "nodes": list(expanded_nodes.values()),
        "relationships": relationships,
        "paths": paths,
    }


def search_general(
    query_obj,
    k: int = 10,
    depth: int = 2,
    *,
    query_override: str | None = None,
    filters_override: dict | None = None,
    allowed_levels_override: set[str] | list[str] | None = None,
    vector_weight_override: float | None = None,
    fulltext_weight_override: float | None = None,
    graph_weight_override: float | None = None,
    strategy_label: str = "retrieve",
    rerank_expanded: bool = False,
) -> dict:
    """
    Generic lookup operator:
    3-channel retrieval -> graph expansion -> optional rerank.
    """
    strategy_name = _enum_value(getattr(query_obj, "strategy", "retrieve"))
    raw_query = query_override or (query_obj.query if getattr(query_obj, "query", None) else "drug")
    clean_query = _normalize_query_for_search(raw_query) or "drug"
    print(f"   [Strategy: {strategy_name.upper()}] 3-channel search for: {clean_query}")

    raw_search_queries = [raw_query]
    if query_override is None and getattr(query_obj, "expanded_queries", None):
        raw_search_queries.extend(query_obj.expanded_queries)

    search_queries: list[str] = []
    seen_queries: set[str] = set()
    for raw in raw_search_queries:
        normalized = _normalize_text(raw)
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen_queries:
            continue
        seen_queries.add(key)
        search_queries.append(normalized)

    target_type = _enum_value(getattr(query_obj, "target_type", "general"))
    config = get_search_config(target_type)
    if allowed_levels_override is not None:
        allowed_levels = allowed_levels_override
    elif strategy_label == "lookup":
        # Lookup should prioritize recall; strict target-level filtering causes misses when
        # target_type is misclassified by AQT.
        allowed_levels = set(HIERARCHY_LEVELS)
    else:
        allowed_levels = config["allowed_levels"]
    filters = dict(filters_override) if filters_override is not None else _build_filters_from_query_obj(query_obj)

    vector_weight = (
        float(vector_weight_override)
        if vector_weight_override is not None
        else float(getattr(query_obj, "vector_weight", 0.5))
    )
    fulltext_weight = (
        float(fulltext_weight_override)
        if fulltext_weight_override is not None
        else float(getattr(query_obj, "fulltext_weight", 0.5))
    )
    weights = _resolve_channel_weights(
        strategy_label=strategy_label,
        target_type=target_type,
        vector_weight=vector_weight,
        fulltext_weight=fulltext_weight,
        graph_weight_override=graph_weight_override,
    )

    bundle = getattr(query_obj, "intent_bundle", None)
    seed_terms: list[str] = []
    for slot_name in ("drug", "brand", "manufacturer", "query"):
        seed_terms.extend(_extract_slot_values(bundle, slot_name))
    if getattr(query_obj, "must_match", None):
        seed_terms.extend([_normalize_text(v) for v in query_obj.must_match if _normalize_text(v)])

    seed_by_id: dict[str, dict] = {}
    for query_text in search_queries:
        candidates = hybrid_search(
            query_text,
            k=max(8, int(k)),
            allowed_levels=list(allowed_levels) if allowed_levels is not None else None,
            filters=filters,
            vector_weight=weights["vector"],
            fulltext_weight=weights["fulltext"],
            graph_weight=weights["graph"],
            seed_terms=seed_terms + [query_text],
            max_graph_hops=min(3, max(1, int(depth))),
        )
        for item in candidates:
            node = item["node"]
            nid = node.element_id if hasattr(node, "element_id") else node.id
            enriched = dict(item)
            enriched["found_by_query"] = query_text
            prev = seed_by_id.get(nid)
            if prev is None or float(enriched.get("rrf_score", 0.0)) > float(prev.get("rrf_score", 0.0)):
                seed_by_id[nid] = enriched

        if strategy_label == "lookup":
            exact_candidates = _exact_match_candidates(
                query_text,
                allowed_levels=allowed_levels,
                filters=filters,
                channel_k=max(8, int(k)),
            )
            for idx, item in enumerate(exact_candidates):
                node = item["node"]
                nid = node.element_id if hasattr(node, "element_id") else node.id
                boost = 0.45 if idx == 0 else max(0.15, 0.35 - (idx * 0.02))
                prev = seed_by_id.get(nid)
                if prev is None:
                    seed_by_id[nid] = {
                        "node": node,
                        "score": 1.0,
                        "rrf_score": boost,
                        "v_rank": 9999,
                        "t_rank": 9999,
                        "g_rank": 9999,
                        "exact_rank": idx,
                        "found_by_query": query_text,
                    }
                else:
                    prev["rrf_score"] = float(prev.get("rrf_score", 0.0)) + boost
                    prev["exact_rank"] = min(int(prev.get("exact_rank", 9999)), idx)
                    prev["exact_boost"] = float(prev.get("exact_boost", 0.0)) + boost

    all_seed_results = list(seed_by_id.values())
    all_seed_results.sort(key=lambda row: float(row.get("rrf_score", 0.0)), reverse=True)
    all_seed_results = all_seed_results[: max(int(k) * 3, 24)]

    seed_node_ids = [
        item["node"].element_id if hasattr(item["node"], "element_id") else item["node"].id
        for item in all_seed_results
    ]
    if not seed_node_ids:
        return {
            "strategy": strategy_label,
            "seed_results": [],
            "expanded_nodes": [],
            "relationships": [],
            "route_plan": {
                "query": clean_query,
                "filters": filters,
                "allowed_levels": list(allowed_levels) if allowed_levels is not None else [],
                "vector_weight": weights["vector"],
                "fulltext_weight": weights["fulltext"],
                "graph_weight": weights["graph"],
                "depth": depth,
            },
        }

    expanded = expand_context(seed_node_ids, depth=depth)
    non_seed_nodes = [n for n in expanded["nodes"] if not n.get("is_seed", False)]
    if rerank_expanded and non_seed_nodes:
        print(f"   [Pruning] Reranking {len(non_seed_nodes)} expanded nodes...")
        try:
            non_seed_nodes = get_reranker().rerank(clean_query, non_seed_nodes, top_k=20)
        except Exception as exc:
            print(f"   [Warning] Reranking failed, using original list: {exc}")

    channel_hits = {
        "vector": sum(1 for row in all_seed_results if int(row.get("v_rank", 9999)) < 9999),
        "fulltext": sum(1 for row in all_seed_results if int(row.get("t_rank", 9999)) < 9999),
        "graph": sum(1 for row in all_seed_results if int(row.get("g_rank", 9999)) < 9999),
        "exact": sum(1 for row in all_seed_results if int(row.get("exact_rank", 9999)) < 9999),
    }

    return {
        "strategy": strategy_label,
        "seed_results": all_seed_results,
        "expanded_nodes": non_seed_nodes,
        "relationships": expanded["relationships"],
        "route_plan": {
            "query": clean_query,
            "filters": filters,
            "allowed_levels": list(allowed_levels) if allowed_levels is not None else [],
            "vector_weight": weights["vector"],
            "fulltext_weight": weights["fulltext"],
            "graph_weight": weights["graph"],
            "depth": depth,
            "channel_hits": channel_hits,
        },
    }


def _run_single_count_query(cypher: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = params or {}
    drv = init_driver()
    with drv.session() as session:
        record = session.run(cypher, **params).single()
        return dict(record) if record else {}


def execute_id_lookup_query(query_obj, depth: int = 1) -> dict:
    """
    Exact lookup path for TMTID-based queries.
    """
    tmtid = getattr(query_obj, "id_lookup", None)
    if not tmtid:
        return search_general(query_obj, k=10, depth=depth)

    print(f"   [Strategy: ID_LOOKUP] Exact lookup for TMTID={tmtid}")
    drv = init_driver()
    nodes = []
    with drv.session() as session:
        recs = session.run(
            """
            MATCH (n:TMT {tmtid: $tmtid})
            RETURN n
            LIMIT 10
            """,
            tmtid=tmtid,
        )
        nodes = [rec["n"] for rec in recs]

    if not nodes:
        print("   [ID_LOOKUP] Exact match not found, falling back to search.")
        fallback = search_general(
            query_obj,
            k=10,
            depth=depth,
            graph_weight_override=0.45,
            strategy_label="id_lookup_fallback",
        )
        fallback["route"] = {
            "operator": "id_lookup",
            "fallback_used": True,
            "reason": "exact_not_found",
        }
        return fallback

    seed_results = [{"node": n, "score": 1.0, "rrf_score": 1.0, "found_by_query": tmtid} for n in nodes]
    seed_node_ids = [n.element_id if hasattr(n, "element_id") else n.id for n in nodes]
    expanded = expand_context(seed_node_ids, depth=max(1, min(depth, 2)))
    non_seed_nodes = [n for n in expanded["nodes"] if not n.get("is_seed", False)]

    return {
        "strategy": "id_lookup",
        "seed_results": seed_results,
        "expanded_nodes": non_seed_nodes,
        "relationships": expanded["relationships"],
        "route": {
            "operator": "id_lookup",
            "fallback_used": False,
            "exact_match_count": len(seed_results),
        },
    }


def _count_from_substance_graph(substance: str) -> dict[str, Any]:
    cypher = """
    MATCH (subs:TMT {level:'SUBS'})
    WHERE toLower(coalesce(subs.name, '')) = $substance
    MATCH (subs)-[sv]->(v:TMT {level:'VTM'})
    WHERE type(sv) IN ['IS_ACTIVE_SUBSTANCE_OF','IS_ACTIVE_INGREDIENT_OF']
    MATCH (gp:TMT {level:'GP'})-[gv]->(v)
    WHERE type(gv) IN ['HAS_VTM','HAS_ACTIVE_INGREDIENT']
    OPTIONAL MATCH (gp)-[:HAS_GENERIC_UNIT]->(gpu:TMT {level:'GPU'})
    OPTIONAL MATCH (gp)-[:HAS_TRADE_PRODUCT]->(tp:TMT {level:'TP'})
    OPTIONAL MATCH (tp)-[:HAS_TRADE_UNIT]->(tpu:TMT {level:'TPU'})
    RETURN count(DISTINCT subs) AS subs_count,
           count(DISTINCT v) AS vtm_count,
           count(DISTINCT gp) AS gp_count,
           count(DISTINCT gpu) AS gpu_count,
           count(DISTINCT tp) AS tp_count,
           count(DISTINCT tpu) AS tpu_count,
           count(DISTINCT gp) AS total
    """
    return _run_single_count_query(cypher, {"substance": substance.lower()})


def _count_from_text_contains(term: str, levels: tuple[str, ...] | list[str] | None = None) -> dict[str, Any]:
    selected_levels = list(levels) if levels else list(COUNT_PRODUCT_LEVELS)
    cypher = """
    MATCH (n:TMT)
    WHERE n.level IN $levels
      AND (
        toLower(coalesce(n.name, '')) CONTAINS $term
        OR toLower(coalesce(n.fsn, '')) CONTAINS $term
        OR toLower(coalesce(n.trade_name, '')) CONTAINS $term
        OR toLower(coalesce(n.generic_name, '')) CONTAINS $term
        OR toLower(coalesce(n.active_substance, '')) CONTAINS $term
        OR toLower(coalesce(n.active_substances, '')) CONTAINS $term
      )
    RETURN count(DISTINCT n) AS total
    """
    return _run_single_count_query(cypher, {"term": term.lower(), "levels": selected_levels})


def _resolve_substance_anchor(term: str) -> str:
    normalized = _normalize_query_for_search(term).lower()
    if not normalized:
        return ""

    drv = init_driver()
    with drv.session() as session:
        exact = session.run(
            """
            MATCH (s:TMT {level:'SUBS'})
            WHERE toLower(coalesce(s.name, '')) = $term
            RETURN s.name AS name
            LIMIT 1
            """,
            term=normalized,
        ).single()
        if exact and exact.get("name"):
            return _normalize_text(exact["name"])

        fuzzy = session.run(
            """
            MATCH (s:TMT {level:'SUBS'})
            WHERE
                toLower(coalesce(s.name, '')) CONTAINS $term
                OR $term CONTAINS toLower(coalesce(s.name, ''))
            RETURN s.name AS name,
                   CASE
                       WHEN toLower(coalesce(s.name, '')) STARTS WITH $term THEN 0
                       WHEN $term STARTS WITH toLower(coalesce(s.name, '')) THEN 1
                       ELSE 2
                   END AS pri,
                   abs(size(toLower(coalesce(s.name, ''))) - size($term)) AS delta
            ORDER BY pri ASC, delta ASC
            LIMIT 1
            """,
            term=normalized,
        ).single()
        if fuzzy and fuzzy.get("name"):
            return _normalize_text(fuzzy["name"])

    return _normalize_query_for_search(term)


def _is_global_product_count_query(question_text: str) -> bool:
    text = _normalize_text(question_text).lower()
    if "tmt" not in text:
        return False
    product_hints = {
        "how many drugs",
        "produce",
        "products",
        "\u0e08\u0e33\u0e19\u0e27\u0e19\u0e22\u0e32",  # à¸ˆà¸³à¸™à¸§à¸™à¸¢à¸²
        "\u0e1c\u0e25\u0e34\u0e15",  # à¸œà¸¥à¸´à¸•
        "\u0e22\u0e32",  # à¸¢à¸²
    }
    return any(hint in text for hint in product_hints)


def _is_global_hierarchy_count_query(question_text: str) -> bool:
    text = _normalize_text(question_text).lower()
    if "tmt" not in text:
        return False
    hierarchy_hints = {
        "all nodes",
        "all entities",
        "\u0e17\u0e31\u0e49\u0e07\u0e2b\u0e21\u0e14",  # à¸—à¸±à¹‰à¸‡à¸«à¸¡à¸”
        "\u0e42\u0e2b\u0e19\u0e14",  # à¹‚à¸«à¸™à¸”
        "\u0e43\u0e19\u0e23\u0e30\u0e1a\u0e1a",  # à¹ƒà¸™à¸£à¸°à¸šà¸š
    }
    return any(hint in text for hint in hierarchy_hints)


def execute_count_query(query_obj) -> dict:
    """
    Strategy: COUNT (graph-aware).
    """
    target_type = _enum_value(getattr(query_obj, "target_type", "general")).lower()
    query_text = _normalize_query_for_search(getattr(query_obj, "query", "")).lower()
    question_raw = _get_question_raw(query_obj)
    question_text = question_raw.lower()
    bundle = getattr(query_obj, "intent_bundle", None)
    drug_slots = _extract_slot_values(bundle, "drug")
    brand_slots = _extract_slot_values(bundle, "brand")
    manufacturer = _normalize_query_for_search(getattr(query_obj, "manufacturer_filter", None))
    inferred_substance = _normalize_query_for_search(_infer_substance_from_question(question_raw) or "")

    has_explicit_anchor = bool(drug_slots or brand_slots or manufacturer)
    if not has_explicit_anchor and (target_type == "hierarchy" or "tmt" in question_text):
        if _is_global_hierarchy_count_query(question_raw) and not _is_global_product_count_query(question_raw):
            drv = init_driver()
            with drv.session() as session:
                rows = session.run(
                    """
                    MATCH (n:TMT)
                    WHERE n.level IN $levels
                    RETURN n.level AS level, count(*) AS c
                    ORDER BY c DESC
                    """,
                    levels=list(COUNT_HIERARCHY_LEVELS),
                ).data()
            breakdown = {row["level"]: int(row["c"]) for row in rows}
            total = int(sum(breakdown.values()))
            return {
                "strategy": "count",
                "result": total,
                "result_detail": {"level_breakdown": breakdown},
                "seed_results": [],
                "expanded_nodes": [],
                "relationships": [],
                "route": {
                    "operator": "analyze_count",
                    "count_mode": "hierarchy_breakdown",
                    "where_clause": "level IN [SUBS,VTM,GP,GPU,TP,TPU]",
                },
            }

        if _is_global_product_count_query(question_raw):
            result = _run_single_count_query(
                """
                MATCH (n:TMT)
                WHERE n.level IN $levels
                RETURN count(n) AS total
                """,
                {"levels": list(COUNT_PRODUCT_LEVELS)},
            )
            return {
                "strategy": "count",
                "result": int(result.get("total", 0) or 0),
                "result_detail": result,
                "seed_results": [],
                "expanded_nodes": [],
                "relationships": [],
                "route": {
                    "operator": "analyze_count",
                    "count_mode": "produced_product_levels",
                    "where_clause": "level IN [GP,GPU,TP,TPU]",
                },
            }

    substance_candidate = _normalize_query_for_search(drug_slots[0]) if drug_slots else inferred_substance
    if substance_candidate:
        substance = _resolve_substance_anchor(substance_candidate)
        result = _count_from_substance_graph(substance)
        total = int(result.get("total", 0) or 0)
        if total > 0:
            return {
                "strategy": "count",
                "result": total,
                "result_detail": result,
                "seed_results": [],
                "expanded_nodes": [],
                "relationships": [],
                "route": {
                    "operator": "analyze_count",
                    "count_mode": "substance_graph_gp",
                    "substance": substance,
                    "where_clause": "SUBS -> VTM -> GP",
                },
            }

        text_fallback = _count_from_text_contains(substance_candidate, levels=COUNT_SUBSTANCE_SCOPE_LEVELS)
        return {
            "strategy": "count",
            "result": int(text_fallback.get("total", 0) or 0),
            "result_detail": text_fallback,
            "seed_results": [],
            "expanded_nodes": [],
            "relationships": [],
            "route": {
                "operator": "analyze_count",
                "count_mode": "text_fallback_gp",
                "substance": substance_candidate,
                "fallback_used": True,
                "where_clause": "level IN [GP] AND contains(term)",
            },
        }

    if manufacturer:
        result = _run_single_count_query(
            """
            MATCH (n:TMT)
            WHERE n.level IN $levels
              AND toLower(coalesce(n.manufacturer, '')) CONTAINS $manu
            RETURN count(DISTINCT n) AS total
            """,
            {"manu": manufacturer.lower(), "levels": list(COUNT_MANUFACTURER_SCOPE_LEVELS)},
        )
        return {
            "strategy": "count",
            "result": int(result.get("total", 0) or 0),
            "result_detail": result,
            "seed_results": [],
            "expanded_nodes": [],
            "relationships": [],
            "route": {
                "operator": "analyze_count",
                "count_mode": "manufacturer_products",
                "where_clause": "level IN [TP] AND manufacturer contains",
            },
        }

    if target_type == "nlem" or getattr(query_obj, "nlem_filter", None):
        params: dict[str, Any] = {}
        extra = ""
        if getattr(query_obj, "nlem_category", None):
            extra = " AND n.nlem_category = $cat"
            params["cat"] = query_obj.nlem_category
        result = _run_single_count_query(
            f"""
            MATCH (n:TMT)
            WHERE n.level = 'GP' AND n.nlem = true{extra}
            RETURN count(n) AS total
            """,
            params,
        )
        return {
            "strategy": "count",
            "result": int(result.get("total", 0) or 0),
            "result_detail": result,
            "seed_results": [],
            "expanded_nodes": [],
            "relationships": [],
            "route": {
                "operator": "analyze_count",
                "count_mode": "nlem_gp",
                "where_clause": "level=GP AND nlem=true",
            },
        }

    if target_type == "hierarchy":
        if _is_global_product_count_query(question_raw) or _is_global_product_count_query(query_text):
            result = _run_single_count_query(
                """
                MATCH (n:TMT)
                WHERE n.level IN $levels
                RETURN count(n) AS total
                """,
                {"levels": list(COUNT_PRODUCT_LEVELS)},
            )
            return {
                "strategy": "count",
                "result": int(result.get("total", 0) or 0),
                "result_detail": result,
                "seed_results": [],
                "expanded_nodes": [],
                "relationships": [],
                "route": {
                    "operator": "analyze_count",
                    "count_mode": "produced_product_levels",
                    "where_clause": "level IN [GP,GPU,TP,TPU]",
                },
            }

        drv = init_driver()
        with drv.session() as session:
            rows = session.run(
                """
                MATCH (n:TMT)
                WHERE n.level IN $levels
                RETURN n.level AS level, count(*) AS c
                ORDER BY c DESC
                """,
                levels=list(COUNT_HIERARCHY_LEVELS),
            ).data()
        breakdown = {row["level"]: int(row["c"]) for row in rows}
        total = int(sum(breakdown.values()))
        return {
            "strategy": "count",
            "result": total,
            "result_detail": {"level_breakdown": breakdown},
            "seed_results": [],
            "expanded_nodes": [],
            "relationships": [],
            "route": {
                "operator": "analyze_count",
                "count_mode": "hierarchy_breakdown",
                "where_clause": "level IN [SUBS,VTM,GP,GPU,TP,TPU]",
            },
        }

    query_fallback_term = _normalize_query_for_search(getattr(query_obj, "query", ""))
    generic_term = (
        inferred_substance
        if inferred_substance
        else (brand_slots[0] if brand_slots else query_fallback_term)
    )
    if generic_term:
        resolved_term = _resolve_substance_anchor(generic_term)
        graph_result = _count_from_substance_graph(resolved_term)
        graph_total = int(graph_result.get("total", 0) or 0)
        if graph_total > 0:
            return {
                "strategy": "count",
                "result": graph_total,
                "result_detail": graph_result,
                "seed_results": [],
                "expanded_nodes": [],
                "relationships": [],
                "route": {
                    "operator": "analyze_count",
                    "count_mode": "substance_graph_gp_fallback",
                    "substance": resolved_term,
                    "where_clause": "SUBS -> VTM -> GP",
                },
            }

        result = _count_from_text_contains(generic_term, levels=COUNT_SUBSTANCE_SCOPE_LEVELS)
        return {
            "strategy": "count",
            "result": int(result.get("total", 0) or 0),
            "result_detail": result,
            "seed_results": [],
            "expanded_nodes": [],
            "relationships": [],
            "route": {
                "operator": "analyze_count",
                "count_mode": "text_contains_gp",
                "where_clause": "level IN [GP] AND contains(term)",
            },
        }

    result = _run_single_count_query(
        """
        MATCH (n:TMT)
        WHERE n.level IN $levels
        RETURN count(n) AS total
        """,
        {"levels": list(COUNT_PRODUCT_LEVELS)},
    )
    return {
        "strategy": "count",
        "result": int(result.get("total", 0) or 0),
        "result_detail": result,
        "seed_results": [],
        "expanded_nodes": [],
        "relationships": [],
        "route": {
            "operator": "analyze_count",
            "count_mode": "all_product_levels",
            "where_clause": "level IN [GP,GPU,TP,TPU]",
        },
    }

def _get_question_raw(query_obj) -> str:
    raw = _normalize_text(getattr(query_obj, "_question_raw", None))
    if raw:
        return raw
    return _normalize_text(getattr(query_obj, "query", ""))


def _clean_anchor_candidate(value: str) -> str:
    text = _normalize_text(value).strip(" ,.;:()[]{}\"'")
    text = re.sub(r"\s+", " ", text)
    if text.endswith("?"):
        text = text[:-1].strip()
    return text


def _extract_tail_by_patterns(question: str, patterns: tuple[re.Pattern, ...]) -> str | None:
    text = _normalize_text(question)
    if not text:
        return None
    for pattern in patterns:
        match = pattern.search(text)
        if not match:
            continue
        candidate = _clean_anchor_candidate(match.group(1))
        if candidate:
            return candidate
    return None


def _infer_substance_from_question(question: str) -> str | None:
    candidate = _extract_tail_by_patterns(question, LIST_SUBSTANCE_PATTERNS)
    if candidate:
        return candidate
    return None


def _infer_substance_for_lookup_anchor(question: str) -> str | None:
    candidate = _extract_tail_by_patterns(question, LOOKUP_ANCHOR_SUBSTANCE_PATTERNS)
    if candidate:
        return candidate
    return None


def _infer_manufacturer_from_question(question: str) -> str | None:
    candidate = _extract_tail_by_patterns(question, LIST_MANUFACTURER_PATTERNS)
    if candidate:
        return candidate
    return None


def _is_manufacturer_request(question: str) -> bool:
    text = _normalize_text(question).lower()
    if not text:
        return False
    return any(hint in text for hint in MANUFACTURER_REQUEST_HINTS)


def _is_trade_name_request(question: str) -> bool:
    text = _normalize_text(question).lower()
    if not text:
        return False
    return any(hint in text for hint in TRADE_NAME_REQUEST_HINTS)


def _lookup_has_anchor_hint(query_obj) -> bool:
    bundle = getattr(query_obj, "intent_bundle", None)
    question_raw = _get_question_raw(query_obj)
    drug_slots = _extract_slot_values(bundle, "drug")
    brand_slots = _extract_slot_values(bundle, "brand")
    manufacturer_filter = _normalize_query_for_search(getattr(query_obj, "manufacturer_filter", None))
    inferred_substance = _normalize_query_for_search(_infer_substance_for_lookup_anchor(question_raw) or "")
    inferred_manufacturer = _normalize_query_for_search(_infer_manufacturer_from_question(question_raw) or "")

    has_anchor = bool(drug_slots or brand_slots or manufacturer_filter or inferred_substance or inferred_manufacturer)
    if not has_anchor:
        return False

    return _is_trade_name_request(question_raw) or _is_manufacturer_request(question_raw)


def _lookup_confidence_state(search_result: dict, k: int) -> tuple[bool, str]:
    seeds = search_result.get("seed_results", []) or []
    if not seeds:
        return True, "empty_seed"

    if len(seeds) < max(3, min(6, int(k))):
        return True, "few_seed_candidates"

    top = seeds[0]
    top_score = float(top.get("rrf_score", 0.0) or 0.0)
    second_score = float(seeds[1].get("rrf_score", 0.0) or 0.0) if len(seeds) > 1 else 0.0
    gap = top_score - second_score
    support = sum(
        1
        for rank_key in ("v_rank", "t_rank", "g_rank", "exact_rank")
        if int(top.get(rank_key, 9999)) < 9999
    )

    if support <= 1 and gap < 0.003:
        return True, "weak_top_support"
    return False, "stable"


def _merge_lookup_fallback(primary: dict, fallback: dict, k: int) -> list[dict]:
    merged: dict[str, dict] = {}
    for item in primary.get("seed_results", []) or []:
        node = item.get("node")
        if node is None:
            continue
        node_id = node.element_id if hasattr(node, "element_id") else node.id
        merged[node_id] = dict(item)

    fallback_seeds = fallback.get("seed_results", []) or []
    for idx, item in enumerate(fallback_seeds):
        node = item.get("node")
        if node is None:
            continue
        node_id = node.element_id if hasattr(node, "element_id") else node.id
        # Prioritize deterministic anchors but keep ranking smooth.
        boost = max(0.08, 0.22 - (idx * 0.01))
        existing = merged.get(node_id)
        if existing is None:
            merged[node_id] = {
                "node": node,
                "score": 1.0,
                "rrf_score": boost,
                "v_rank": 9999,
                "t_rank": 9999,
                "g_rank": 9999,
                "fallback_rank": idx,
                "found_by_query": "lookup_fallback",
            }
            continue
        existing["rrf_score"] = float(existing.get("rrf_score", 0.0) or 0.0) + boost
        existing["fallback_rank"] = min(int(existing.get("fallback_rank", 9999)), idx)
        existing["fallback_boost"] = float(existing.get("fallback_boost", 0.0) or 0.0) + boost

    ranked = list(merged.values())
    ranked.sort(key=lambda row: float(row.get("rrf_score", 0.0) or 0.0), reverse=True)
    return ranked[: max(int(k) * 3, 24)]


def _extract_lookup_anchor_hints(query_obj) -> dict[str, Any]:
    bundle = getattr(query_obj, "intent_bundle", None)
    question_raw = _get_question_raw(query_obj)
    query_text = _normalize_query_for_search(getattr(query_obj, "query", ""))

    drug_slots = _extract_slot_values(bundle, "drug")
    brand_slots = _extract_slot_values(bundle, "brand")
    manufacturer_slots = _extract_slot_values(bundle, "manufacturer")
    strength_slots = _extract_slot_values(bundle, "strength")

    substance = ""
    for candidate in drug_slots + brand_slots:
        substance = _normalize_query_for_search(candidate)
        if substance:
            break
    if not substance:
        substance = _normalize_query_for_search(_infer_substance_for_lookup_anchor(question_raw) or "")
    if not substance:
        substance = query_text

    # Strip request phrases from fallback anchor text.
    substance = re.sub(
        r"^(?:trade\s+name(?:s)?(?:\s+of)?|brand\s+name(?:s)?(?:\s+of)?|"
        r"who\s+manufactures|manufacturer\s+of)\s+",
        "",
        substance,
        flags=re.IGNORECASE,
    )
    substance = _strip_strength_tokens(substance)
    substance = _normalize_query_for_search(substance)

    manufacturer = _normalize_query_for_search(getattr(query_obj, "manufacturer_filter", None))
    if not manufacturer:
        for candidate in manufacturer_slots:
            manufacturer = _normalize_query_for_search(candidate)
            if manufacturer:
                break
    if not manufacturer:
        manufacturer = _normalize_query_for_search(_infer_manufacturer_from_question(question_raw) or "")

    strength = _normalize_strength_expression(strength_slots[0]) if strength_slots else ""
    if not strength:
        strength_match = re.search(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|iu|%)\b", question_raw, re.IGNORECASE)
        if strength_match:
            strength = _normalize_strength_expression(strength_match.group(0))

    return {
        "substance": substance,
        "manufacturer": manufacturer,
        "strength": strength,
        "is_trade_name_request": _is_trade_name_request(question_raw),
        "is_manufacturer_request": _is_manufacturer_request(question_raw),
    }


def _execute_lookup_anchor_query(query_obj, k: int) -> dict[str, Any]:
    hints = _extract_lookup_anchor_hints(query_obj)
    substance = hints.get("substance", "")
    manufacturer = hints.get("manufacturer", "")
    strength = hints.get("strength", "")
    strength_compact = strength.replace(" ", "").lower() if strength else ""
    is_trade_name_request = bool(hints.get("is_trade_name_request"))
    is_manufacturer_request = bool(hints.get("is_manufacturer_request"))

    if not (substance or manufacturer):
        return {"seed_results": [], "route_mode": "none"}

    drv = init_driver()
    with drv.session() as session:
        if substance:
            tp_records = session.run(
                """
                MATCH (tp:TMT {level:'TP'})
                WHERE (
                    toLower(coalesce(tp.trade_name, '')) CONTAINS $substance
                    OR toLower(coalesce(tp.generic_name, '')) CONTAINS $substance
                    OR toLower(coalesce(tp.fsn, '')) CONTAINS $substance
                    OR toLower(coalesce(tp.active_substance, '')) CONTAINS $substance
                    OR toLower(coalesce(tp.active_substances, '')) CONTAINS $substance
                )
                  AND (
                    $strength = ''
                    OR toLower(coalesce(tp.fsn, '')) CONTAINS $strength
                    OR replace(toLower(coalesce(tp.fsn, '')), ' ', '') CONTAINS $strength_compact
                  )
                  AND (
                    $manufacturer = ''
                    OR toLower(coalesce(tp.manufacturer, '')) CONTAINS $manufacturer
                  )
                RETURN DISTINCT tp AS n
                ORDER BY coalesce(tp.trade_name, tp.fsn, tp.name)
                LIMIT $k
                """,
                substance=substance.lower(),
                strength=strength.lower(),
                strength_compact=strength_compact,
                manufacturer=manufacturer.lower(),
                k=max(10, int(k)),
            )
            tp_nodes = [rec["n"] for rec in tp_records]
            if tp_nodes:
                if is_manufacturer_request:
                    route_mode = "lookup_anchor_tp_manufacturer"
                elif is_trade_name_request:
                    route_mode = "lookup_anchor_tp_trade_name"
                else:
                    route_mode = "lookup_anchor_tp_substance"
                return {
                    "seed_results": [{"node": n, "score": 1.0, "rrf_score": 1.0} for n in tp_nodes],
                    "route_mode": route_mode,
                }

            gp_records = session.run(
                """
                MATCH (gp:TMT {level:'GP'})
                WHERE (
                    toLower(coalesce(gp.fsn, '')) CONTAINS $substance
                    OR toLower(coalesce(gp.name, '')) CONTAINS $substance
                    OR toLower(coalesce(gp.active_substance, '')) CONTAINS $substance
                    OR toLower(coalesce(gp.active_substances, '')) CONTAINS $substance
                )
                  AND (
                    $strength = ''
                    OR toLower(coalesce(gp.fsn, '')) CONTAINS $strength
                    OR replace(toLower(coalesce(gp.fsn, '')), ' ', '') CONTAINS $strength_compact
                  )
                RETURN DISTINCT gp AS n
                ORDER BY coalesce(gp.fsn, gp.name)
                LIMIT $k
                """,
                substance=substance.lower(),
                strength=strength.lower(),
                strength_compact=strength_compact,
                k=max(10, int(k)),
            )
            gp_nodes = [rec["n"] for rec in gp_records]
            if gp_nodes:
                return {
                    "seed_results": [{"node": n, "score": 1.0, "rrf_score": 1.0} for n in gp_nodes],
                    "route_mode": "lookup_anchor_gp_substance",
                }

        if manufacturer:
            manu_records = session.run(
                """
                MATCH (tp:TMT {level:'TP'})
                WHERE toLower(coalesce(tp.manufacturer, '')) CONTAINS $manufacturer
                RETURN DISTINCT tp AS n
                ORDER BY coalesce(tp.trade_name, tp.fsn, tp.name)
                LIMIT $k
                """,
                manufacturer=manufacturer.lower(),
                k=max(10, int(k)),
            )
            manu_nodes = [rec["n"] for rec in manu_records]
            if manu_nodes:
                return {
                    "seed_results": [{"node": n, "score": 1.0, "rrf_score": 1.0} for n in manu_nodes],
                    "route_mode": "lookup_anchor_tp_manufacturer_only",
                }

    return {"seed_results": [], "route_mode": "none"}


def execute_listing_query(query_obj, k: int = 100) -> dict:
    """
    Strategy: LIST with graph-aware deterministic routes.
    """
    bundle = getattr(query_obj, "intent_bundle", None)
    question_raw = _get_question_raw(query_obj)
    question_lower = question_raw.lower()
    target_type = _enum_value(getattr(query_obj, "target_type", "general")).lower()
    drug_slots = _extract_slot_values(bundle, "drug")
    strength_slots = _extract_slot_values(bundle, "strength")
    manufacturer = _normalize_query_for_search(getattr(query_obj, "manufacturer_filter", None))
    substance_hint = (
        _normalize_query_for_search(drug_slots[0])
        if drug_slots
        else _normalize_query_for_search(_infer_substance_from_question(question_raw))
    )
    query_hint = _normalize_query_for_search(getattr(query_obj, "query", ""))
    manufacturer_hint = manufacturer or _normalize_query_for_search(_infer_manufacturer_from_question(question_raw))
    strength_hint = _normalize_strength_expression(strength_slots[0]) if strength_slots else ""
    if not strength_hint:
        strength_match = re.search(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|iu|%)\b", question_raw, re.IGNORECASE)
        if strength_match:
            strength_hint = _normalize_strength_expression(strength_match.group(0))
    if substance_hint:
        embedded_strength = STRENGTH_TOKEN_PATTERN.search(substance_hint)
        if embedded_strength:
            if not strength_hint:
                strength_hint = _normalize_strength_expression(embedded_strength.group(0))
            stripped_substance = _strip_strength_tokens(substance_hint)
            if stripped_substance:
                substance_hint = stripped_substance
    strength_compact = strength_hint.replace(" ", "").lower() if strength_hint else ""

    wants_trade_name = (
        target_type == "formula"
        or "trade name" in question_lower
        or "\u0e0a\u0e37\u0e48\u0e2d\u0e17\u0e32\u0e07\u0e01\u0e32\u0e23\u0e04\u0e49\u0e32" in question_raw
    )
    wants_manufacturer_names = _is_manufacturer_request(question_raw) and not manufacturer_hint
    if (wants_trade_name or wants_manufacturer_names) and not substance_hint:
        if query_hint:
            substance_hint = query_hint
    requested_k = max(1, int(k))
    cap = max(1, int(LIST_MAX_K_CAP))

    # Eval mode guard: relax cap for large-result diagnostics (manufacturer/list stress tests).
    if RETRIEVAL_EVAL_MODE and requested_k >= cap:
        requested_k = max(requested_k, 1000)
        cap = max(cap, 5000)

    k = min(requested_k, cap)

    drv = init_driver()

    if substance_hint:
        substance = substance_hint.lower()
        if wants_manufacturer_names:
            cypher_manu_direct = """
            MATCH (tp:TMT {level:'TP'})
            WHERE (
                toLower(coalesce(tp.trade_name, '')) CONTAINS $substance
                OR toLower(coalesce(tp.generic_name, '')) CONTAINS $substance
                OR toLower(coalesce(tp.fsn, '')) CONTAINS $substance
                OR toLower(coalesce(tp.active_substance, '')) CONTAINS $substance
                OR toLower(coalesce(tp.active_substances, '')) CONTAINS $substance
            )
              AND (
                $strength = ''
                OR toLower(coalesce(tp.fsn, '')) CONTAINS $strength
                OR replace(toLower(coalesce(tp.fsn, '')), ' ', '') CONTAINS $strength_compact
              )
              AND coalesce(tp.manufacturer, '') <> ''
            WITH toLower(tp.manufacturer) AS mkey, collect(tp)[0] AS n
            RETURN n
            ORDER BY coalesce(n.manufacturer, '')
            LIMIT $k
            """
            cypher_manu_graph = """
            MATCH (subs:TMT {level:'SUBS'})
            WHERE toLower(coalesce(subs.name, '')) = $substance
            MATCH (subs)-[sv]->(v:TMT {level:'VTM'})
            WHERE type(sv) IN ['IS_ACTIVE_SUBSTANCE_OF','IS_ACTIVE_INGREDIENT_OF']
            MATCH (gp:TMT {level:'GP'})-[gv]->(v)
            WHERE type(gv) IN ['HAS_VTM','HAS_ACTIVE_INGREDIENT']
            MATCH (gp)-[:HAS_TRADE_PRODUCT]->(tp:TMT {level:'TP'})
            WHERE (
                $strength = ''
                OR toLower(coalesce(tp.fsn, '')) CONTAINS $strength
                OR replace(toLower(coalesce(tp.fsn, '')), ' ', '') CONTAINS $strength_compact
            )
              AND coalesce(tp.manufacturer, '') <> ''
            WITH toLower(tp.manufacturer) AS mkey, collect(tp)[0] AS n
            RETURN n
            ORDER BY coalesce(n.manufacturer, '')
            LIMIT $k
            """
            with drv.session() as session:
                nodes = [
                    rec["n"]
                    for rec in session.run(
                        cypher_manu_direct,
                        substance=substance,
                        strength=strength_hint.lower(),
                        strength_compact=strength_compact,
                        k=k,
                    )
                ]
                if not nodes:
                    nodes = [
                        rec["n"]
                        for rec in session.run(
                            cypher_manu_graph,
                            substance=substance,
                            strength=strength_hint.lower(),
                            strength_compact=strength_compact,
                            k=k,
                        )
                    ]
            if nodes:
                return {
                    "strategy": "list",
                    "seed_results": [{"node": n, "score": 1.0, "rrf_score": 1.0} for n in nodes],
                    "expanded_nodes": [],
                    "relationships": [],
                    "route": {
                        "operator": "list",
                        "list_mode": "substance_manufacturer_tp_unique",
                        "substance": substance,
                        "strength": strength_hint,
                        "limit": k,
                    },
                }

        if wants_trade_name:
            cypher_tp_direct = """
            MATCH (n:TMT {level:'TP'})
            WHERE (
                toLower(coalesce(n.trade_name, '')) CONTAINS $substance
                OR toLower(coalesce(n.generic_name, '')) CONTAINS $substance
                OR toLower(coalesce(n.fsn, '')) CONTAINS $substance
                OR toLower(coalesce(n.active_substance, '')) CONTAINS $substance
                OR toLower(coalesce(n.active_substances, '')) CONTAINS $substance
            )
              AND (
                $strength = ''
                OR toLower(coalesce(n.fsn, '')) CONTAINS $strength
                OR replace(toLower(coalesce(n.fsn, '')), ' ', '') CONTAINS $strength_compact
              )
            RETURN DISTINCT n
            ORDER BY coalesce(n.trade_name, n.fsn, n.name)
            LIMIT $k
            """
            cypher_tp_graph = """
            MATCH (subs:TMT {level:'SUBS'})
            WHERE toLower(coalesce(subs.name, '')) = $substance
            MATCH (subs)-[sv]->(v:TMT {level:'VTM'})
            WHERE type(sv) IN ['IS_ACTIVE_SUBSTANCE_OF','IS_ACTIVE_INGREDIENT_OF']
            MATCH (gp:TMT {level:'GP'})-[gv]->(v)
            WHERE type(gv) IN ['HAS_VTM','HAS_ACTIVE_INGREDIENT']
            MATCH (gp)-[:HAS_TRADE_PRODUCT]->(tp:TMT {level:'TP'})
            WHERE (
                $strength = ''
                OR toLower(coalesce(tp.fsn, '')) CONTAINS $strength
                OR replace(toLower(coalesce(tp.fsn, '')), ' ', '') CONTAINS $strength_compact
            )
            RETURN DISTINCT tp AS n
            ORDER BY coalesce(tp.trade_name, tp.fsn, tp.name)
            LIMIT $k
            """
            with drv.session() as session:
                nodes = [
                    rec["n"]
                    for rec in session.run(
                        cypher_tp_direct,
                        substance=substance,
                        strength=strength_hint.lower(),
                        strength_compact=strength_compact,
                        k=k,
                    )
                ]
                if not nodes:
                    nodes = [
                        rec["n"]
                        for rec in session.run(
                            cypher_tp_graph,
                            substance=substance,
                            strength=strength_hint.lower(),
                            strength_compact=strength_compact,
                            k=k,
                        )
                    ]

            if nodes:
                return {
                    "strategy": "list",
                    "seed_results": [{"node": n, "score": 1.0, "rrf_score": 1.0} for n in nodes],
                    "expanded_nodes": [],
                    "relationships": [],
                    "route": {
                        "operator": "list",
                        "list_mode": "substance_trade_name_tp",
                        "substance": substance,
                        "strength": strength_hint,
                        "limit": k,
                    },
                }

        cypher = """
        MATCH (subs:TMT {level:'SUBS'})
        WHERE toLower(coalesce(subs.name, '')) = $substance
        MATCH (subs)-[sv]->(v:TMT {level:'VTM'})
        WHERE type(sv) IN ['IS_ACTIVE_SUBSTANCE_OF','IS_ACTIVE_INGREDIENT_OF']
        MATCH (gp:TMT {level:'GP'})-[gv]->(v)
        WHERE type(gv) IN ['HAS_VTM','HAS_ACTIVE_INGREDIENT']
        RETURN DISTINCT gp AS n
        ORDER BY coalesce(gp.fsn, gp.name, gp.trade_name)
        LIMIT $k
        """
        with drv.session() as session:
            nodes = [rec["n"] for rec in session.run(cypher, substance=substance, k=k)]

            if not nodes:
                fallback_cypher = """
                MATCH (gp:TMT {level:'GP'})
                WHERE toLower(coalesce(gp.name, '')) CONTAINS $substance
                   OR toLower(coalesce(gp.fsn, '')) CONTAINS $substance
                   OR toLower(coalesce(gp.active_substance, '')) CONTAINS $substance
                   OR toLower(coalesce(gp.active_substances, '')) CONTAINS $substance
                RETURN DISTINCT gp AS n
                ORDER BY coalesce(gp.fsn, gp.name, gp.trade_name)
                LIMIT $k
                """
                nodes = [rec["n"] for rec in session.run(fallback_cypher, substance=substance, k=k)]

        if nodes:
            return {
                "strategy": "list",
                "seed_results": [{"node": n, "score": 1.0, "rrf_score": 1.0} for n in nodes],
                "expanded_nodes": [],
                "relationships": [],
                "route": {
                    "operator": "list",
                    "list_mode": "substance_gp_only",
                    "substance": substance,
                    "limit": k,
                },
            }

    if manufacturer_hint:
        manu_term = manufacturer_hint.lower()
        cypher_exact = """
        MATCH (n:TMT {level:'TP'})
        WHERE toLower(coalesce(n.manufacturer, '')) = $manu
        RETURN n
        ORDER BY coalesce(n.trade_name, n.fsn, n.name)
        LIMIT $k
        """
        cypher_contains = """
        MATCH (n:TMT {level:'TP'})
        WHERE toLower(coalesce(n.manufacturer, '')) CONTAINS $manu
        RETURN n
        ORDER BY coalesce(n.trade_name, n.fsn, n.name)
        LIMIT $k
        """
        with drv.session() as session:
            nodes = [rec["n"] for rec in session.run(cypher_exact, manu=manu_term, k=k)]
            if not nodes:
                nodes = [rec["n"] for rec in session.run(cypher_contains, manu=manu_term, k=k)]
        if nodes:
            return {
                "strategy": "list",
                "seed_results": [{"node": n, "score": 1.0, "rrf_score": 1.0} for n in nodes],
                "expanded_nodes": [],
                "relationships": [],
                "route": {
                    "operator": "list",
                    "list_mode": "manufacturer_tp_only",
                    "limit": k,
                },
            }

        fallback = search_general(
            query_obj,
            k=min(k, 80),
            depth=1,
            query_override=manufacturer_hint,
            filters_override={"manufacturer": manufacturer_hint},
            allowed_levels_override={"TP"},
            strategy_label="list",
            graph_weight_override=0.45,
            rerank_expanded=False,
        )
        fallback["route"] = {
            "operator": "list",
            "list_mode": "manufacturer_graph_hybrid_fallback",
            "limit": k,
            "fallback_used": True,
            "reason": "manufacturer_deterministic_empty",
        }
        return fallback

    lookup = search_general(
        query_obj,
        k=min(k, 60),
        depth=1,
        strategy_label="list",
        graph_weight_override=0.45,
        rerank_expanded=False,
    )
    lookup["route"] = {
        "operator": "list",
        "list_mode": "graph_hybrid_fallback",
        "limit": k,
        "fallback_used": True,
    }
    return lookup


def execute_verify_query(query_obj, k: int = 20, depth: int = 1) -> dict:
    """
    Verify operator with strict-first fallback chain.
    """
    base_filters = _build_filters_from_query_obj(query_obj)
    target_type = _enum_value(getattr(query_obj, "target_type", "general"))
    strict_levels = get_search_config(target_type)["allowed_levels"]
    attempts: list[dict[str, Any]] = []

    strict = search_general(
        query_obj,
        k=max(8, min(k, 20)),
        depth=depth,
        filters_override=base_filters,
        allowed_levels_override=strict_levels,
        strategy_label="verify",
    )
    strict_seed = len(strict.get("seed_results", []))
    attempts.append(
        {
            "name": "strict",
            "seed_count": strict_seed,
            "filters": strict.get("route_plan", {}).get("filters", {}),
            "allowed_levels": strict.get("route_plan", {}).get("allowed_levels", []),
            "weights": {
                "vector": strict.get("route_plan", {}).get("vector_weight"),
                "fulltext": strict.get("route_plan", {}).get("fulltext_weight"),
                "graph": strict.get("route_plan", {}).get("graph_weight"),
            },
        }
    )
    if strict_seed > 0:
        strict["route"] = {
            "operator": "verify",
            "fallback_used": False,
            "attempts": attempts,
        }
        return strict

    relaxed_filters = dict(base_filters)
    if "manufacturer" in relaxed_filters:
        relaxed_filters.pop("manufacturer", None)
    elif "nlem_category" in relaxed_filters:
        relaxed_filters.pop("nlem_category", None)
    elif "nlem" in relaxed_filters:
        relaxed_filters.pop("nlem", None)

    attempt2 = search_general(
        query_obj,
        k=max(8, min(k, 20)),
        depth=depth,
        filters_override=relaxed_filters,
        allowed_levels_override=strict_levels,
        vector_weight_override=0.33,
        fulltext_weight_override=0.33,
        graph_weight_override=0.34,
        strategy_label="verify",
    )
    attempt2_seed = len(attempt2.get("seed_results", []))
    attempts.append(
        {
            "name": "relax_one_filter",
            "seed_count": attempt2_seed,
            "filters": attempt2.get("route_plan", {}).get("filters", {}),
            "allowed_levels": attempt2.get("route_plan", {}).get("allowed_levels", []),
            "weights": {
                "vector": attempt2.get("route_plan", {}).get("vector_weight"),
                "fulltext": attempt2.get("route_plan", {}).get("fulltext_weight"),
                "graph": attempt2.get("route_plan", {}).get("graph_weight"),
            },
        }
    )
    if attempt2_seed > 0:
        attempt2["route"] = {
            "operator": "verify",
            "fallback_used": True,
            "fallback_tier": 1,
            "attempts": attempts,
        }
        return attempt2

    broader_levels = get_search_config("general")["allowed_levels"]
    attempt3 = search_general(
        query_obj,
        k=max(8, min(k, 20)),
        depth=depth,
        filters_override=relaxed_filters,
        allowed_levels_override=broader_levels,
        vector_weight_override=0.33,
        fulltext_weight_override=0.33,
        graph_weight_override=0.34,
        strategy_label="verify",
    )
    attempt3_seed = len(attempt3.get("seed_results", []))
    attempts.append(
        {
            "name": "broaden_scope",
            "seed_count": attempt3_seed,
            "filters": attempt3.get("route_plan", {}).get("filters", {}),
            "allowed_levels": attempt3.get("route_plan", {}).get("allowed_levels", []),
            "weights": {
                "vector": attempt3.get("route_plan", {}).get("vector_weight"),
                "fulltext": attempt3.get("route_plan", {}).get("fulltext_weight"),
                "graph": attempt3.get("route_plan", {}).get("graph_weight"),
            },
        }
    )

    attempt3["route"] = {
        "operator": "verify",
        "fallback_used": True,
        "fallback_tier": 2,
        "attempts": attempts,
    }
    return attempt3


def execute_compare_query(query_obj, k: int = 10, depth: int = 2) -> dict:
    """
    Compare operator: dual retrieval for A/B entities, then merge and expand.
    """
    bundle = getattr(query_obj, "intent_bundle", None)
    compare_terms = _extract_slot_values(bundle, "drug")
    if len(compare_terms) < 2:
        compare_terms.extend([t for t in _extract_slot_values(bundle, "brand") if t not in compare_terms])
    if len(compare_terms) < 2:
        compare_terms.extend([t for t in _split_compare_terms(getattr(query_obj, "query", "")) if t not in compare_terms])
    if not compare_terms:
        compare_terms = [getattr(query_obj, "query", "") or "drug"]
    compare_terms = compare_terms[:2]

    compare_filters = _build_filters_from_query_obj(query_obj)
    compare_filters.pop("manufacturer", None)
    compare_filters.pop("tmtid", None)
    compare_levels = get_search_config("substance")["allowed_levels"]

    result_groups: list[list[dict]] = []
    per_term: list[dict[str, Any]] = []

    for term in compare_terms:
        result = search_general(
            query_obj,
            k=max(8, min(k, 20)),
            depth=min(depth, 2),
            query_override=term,
            filters_override=compare_filters,
            allowed_levels_override=compare_levels,
            vector_weight_override=0.35,
            fulltext_weight_override=0.30,
            graph_weight_override=0.35,
            strategy_label="compare",
            rerank_expanded=False,
        )
        seeds = result.get("seed_results", [])
        result_groups.append(seeds)
        per_term.append(
            {
                "term": term,
                "seed_count": len(seeds),
                "top_seed": _safe_top_seed(seeds),
            }
        )

    merged_seed_results = _merge_seed_results(result_groups)
    seed_node_ids = [
        item["node"].element_id if hasattr(item["node"], "element_id") else item["node"].id
        for item in merged_seed_results
    ]
    if not seed_node_ids:
        return {
            "strategy": "compare",
            "seed_results": [],
            "expanded_nodes": [],
            "relationships": [],
            "compare_terms": compare_terms,
            "compare_detail": per_term,
            "route": {
                "operator": "analyze_compare",
                "fallback_used": False,
            },
        }

    expanded = expand_context(seed_node_ids, depth=min(depth, 2))
    non_seed_nodes = [n for n in expanded["nodes"] if not n.get("is_seed", False)]
    if ENABLE_COMPARE_EXPANDED_RERANK and non_seed_nodes:
        compare_query = " vs ".join(compare_terms)
        try:
            non_seed_nodes = get_reranker().rerank(compare_query, non_seed_nodes, top_k=20)
        except Exception as exc:
            print(f"   [Warning] Compare rerank failed, using original list: {exc}")

    return {
        "strategy": "compare",
        "seed_results": merged_seed_results,
        "expanded_nodes": non_seed_nodes,
        "relationships": expanded["relationships"],
        "compare_terms": compare_terms,
        "compare_detail": per_term,
        "route": {
            "operator": "analyze_compare",
            "fallback_used": False,
        },
    }


def execute_lookup_query(
    query_obj,
    k: int = 10,
    depth: int = 2,
    *,
    vector_weight_override: float | None = None,
    fulltext_weight_override: float | None = None,
    graph_weight_override: float | None = None,
) -> dict:
    """
    Lookup operator with guarded deterministic fallback.
    Keeps hybrid retrieval as primary path, then uses list-style deterministic anchors
    only when confidence is weak or question shape strongly suggests anchored lookup.
    """
    primary_graph_weight = 0.35 if graph_weight_override is None else graph_weight_override
    primary = search_general(
        query_obj,
        k=k,
        depth=depth,
        strategy_label="lookup",
        vector_weight_override=vector_weight_override,
        fulltext_weight_override=fulltext_weight_override,
        graph_weight_override=primary_graph_weight,
    )

    low_confidence, confidence_reason = _lookup_confidence_state(primary, k=k)
    anchor_hint = _lookup_has_anchor_hint(query_obj)
    should_attempt_fallback = low_confidence or anchor_hint

    route_info: dict[str, Any] = {
        "operator": "lookup",
        "fallback_used": False,
        "fallback_attempted": bool(should_attempt_fallback),
        "fallback_trigger": {
            "low_confidence": low_confidence,
            "anchor_hint": anchor_hint,
            "reason": confidence_reason,
        },
    }

    if should_attempt_fallback:
        fallback_k = min(max(int(k) * 2, 25), 50)
        anchor_fallback = _execute_lookup_anchor_query(query_obj, k=fallback_k)
        anchor_mode = str(anchor_fallback.get("route_mode") or "")
        anchor_seed_count = len(anchor_fallback.get("seed_results", []) or [])

        route_info["fallback_route_mode"] = anchor_mode
        route_info["fallback_seed_count"] = anchor_seed_count

        if anchor_seed_count > 0 and anchor_mode != "none":
            primary["seed_results"] = _merge_lookup_fallback(primary, anchor_fallback, k=k)
            route_info["fallback_used"] = True
        else:
            fallback = execute_listing_query(query_obj, k=fallback_k)
            fallback_route = fallback.get("route", {}) if isinstance(fallback, dict) else {}
            fallback_mode = str(fallback_route.get("list_mode") or "")
            fallback_seed_count = len(fallback.get("seed_results", []) or [])
            route_info["fallback_route_mode"] = fallback_mode
            route_info["fallback_seed_count"] = fallback_seed_count

            # Only trust deterministic list routes for lookup fallback.
            deterministic_modes = {
                "substance_manufacturer_tp_unique",
                "substance_trade_name_tp",
                "substance_gp_only",
                "manufacturer_tp_only",
            }
            if fallback_seed_count > 0 and fallback_mode in deterministic_modes:
                primary["seed_results"] = _merge_lookup_fallback(primary, fallback, k=k)
                route_info["fallback_used"] = True

    primary["route"] = route_info
    return primary


def advanced_graphrag_search(query_obj, k: int = 10, depth: int = 2) -> dict:
    """
    Operator-driven router after AQT.
    """
    operator = _resolve_operator(query_obj)

    if operator == "id_lookup":
        return execute_id_lookup_query(query_obj, depth=1)
    if operator == "analyze_count":
        return execute_count_query(query_obj)
    if operator == "analyze_compare":
        return execute_compare_query(query_obj, k=k, depth=depth)
    if operator == "verify":
        return execute_verify_query(query_obj, k=k, depth=1)
    if operator == "list":
        return execute_listing_query(query_obj, k=50)
    return execute_lookup_query(query_obj, k=k, depth=depth)

