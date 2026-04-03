"""
Shadow comparison test for legacy AQT output vs IntentBundle schema.

This script does NOT change runtime behavior.
It only logs side-by-side outputs for evaluation and migration planning.
"""
from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
API_APP_ROOT = PROJECT_ROOT / "apps" / "api"
INTEGRATION_DIR = Path(__file__).resolve().parents[1]
INTENT_EXPERIMENT_DIR = Path(__file__).resolve().parents[2]
LOG_DIR = INTEGRATION_DIR / "results" / "shadow_compare"

# Add the canonical backend app to import path.
sys.path.insert(0, str(API_APP_ROOT))


def _configure_stdout_utf8() -> None:
    """
    Best-effort UTF-8 stdout/stderr setup for Windows shells.
    Prevents UnicodeEncodeError while printing Thai/Unicode logs.
    """
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    try:
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


_configure_stdout_utf8()

from src.schemas.intent_bundle import (
    ActionIntent,
    TopicsIntent,
    IntentBundle,
    IntentControlFeatures,
    RetrievalPlan,
    SlotSource,
    SlotValue,
)
from src.services.aqt import transform_query


TMTID_PATTERN = re.compile(r"\b(?:tmtid|tmt-id|tmt)\s*[:#]?\s*(\d{5,10})\b", re.IGNORECASE)
BARE_ID_PATTERN = re.compile(r"\b\d{6,10}\b")
DOSE_UNIT_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\s*(?:mg|g|mcg|ml|iu|%)\b", re.IGNORECASE)


def _to_dict(model_obj):
    if hasattr(model_obj, "model_dump"):
        return model_obj.model_dump(mode="json")
    return model_obj.dict()


def _map_strategy_to_action(strategy: str) -> ActionIntent:
    mapping = {
        "retrieve": ActionIntent.LOOKUP,
        "verify": ActionIntent.VERIFY,
        "list": ActionIntent.LIST,
        "count": ActionIntent.COUNT,
        "compare": ActionIntent.COMPARE,
    }
    return mapping.get(str(strategy), ActionIntent.UNKNOWN)


def _map_target_to_topics(target: str) -> TopicsIntent:
    mapping = {
        "manufacturer": TopicsIntent.MANUFACTURER,
        "substance": TopicsIntent.SUBSTANCE,
        "nlem": TopicsIntent.NLEM,
        "formula": TopicsIntent.FORMULA,
        "hierarchy": TopicsIntent.HIERARCHY,
        "general": TopicsIntent.GENERAL,
    }
    return mapping.get(str(target), TopicsIntent.GENERAL)


def _extract_tmtid(question: str) -> str | None:
    match = TMTID_PATTERN.search(question)
    if match:
        return match.group(1)
    bare = BARE_ID_PATTERN.search(question)
    return bare.group(0) if bare else None


def build_shadow_bundle(question: str, legacy_query_obj) -> IntentBundle:
    strategy = getattr(legacy_query_obj, "strategy", "retrieve")
    target_type = getattr(legacy_query_obj, "target_type", "general")
    target_value = target_type.value if hasattr(target_type, "value") else str(target_type)
    strategy_value = strategy.value if hasattr(strategy, "value") else str(strategy)

    action_intent = _map_strategy_to_action(strategy_value)
    topics_intents: list[TopicsIntent] = [_map_target_to_topics(target_value)]
    slots: list[SlotValue] = []

    tmtid = _extract_tmtid(question)
    has_exact_id = 1.0 if tmtid else 0.0
    has_dose_unit = 1.0 if DOSE_UNIT_PATTERN.search(question) else 0.0

    if tmtid:
        if TopicsIntent.ID_LOOKUP not in topics_intents:
            topics_intents.append(TopicsIntent.ID_LOOKUP)
        slots.append(
            SlotValue(
                name="tmtid",
                value=tmtid,
                confidence=1.0,
                source=SlotSource.RULE,
            )
        )

    manufacturer = getattr(legacy_query_obj, "manufacturer_filter", None)
    if manufacturer:
        slots.append(
            SlotValue(
                name="manufacturer",
                value=str(manufacturer),
                confidence=0.95,
                source=SlotSource.RULE,
            )
        )

    nlem_filter = getattr(legacy_query_obj, "nlem_filter", None)
    if nlem_filter is not None:
        slots.append(
            SlotValue(
                name="nlem",
                value=str(bool(nlem_filter)).lower(),
                confidence=0.95,
                source=SlotSource.RULE,
            )
        )

    nlem_category = getattr(legacy_query_obj, "nlem_category", None)
    if nlem_category:
        slots.append(
            SlotValue(
                name="nlem_category",
                value=str(nlem_category),
                confidence=0.95,
                source=SlotSource.RULE,
            )
        )

    search_term = getattr(legacy_query_obj, "query", "") or ""
    if search_term and search_term.lower() != question.lower():
        slots.append(
            SlotValue(
                name="drug_name",
                value=str(search_term),
                confidence=0.7,
                source=SlotSource.RULE,
            )
        )

    filters = {}
    if nlem_filter is not None:
        filters["nlem"] = bool(nlem_filter)
    if nlem_category:
        filters["nlem_category"] = str(nlem_category)
    if manufacturer:
        filters["manufacturer"] = str(manufacturer)

    control_features = IntentControlFeatures(
        token_count=int(getattr(legacy_query_obj, "token_count", 0) or 0),
        entity_token_count=int(getattr(legacy_query_obj, "entity_token_count", 0) or 0),
        entity_ratio=float(getattr(legacy_query_obj, "entity_ratio", 0.0) or 0.0),
        has_exact_id=has_exact_id,
        has_dose_unit=has_dose_unit,
        ambiguity=max(
            0.0,
            min(
                1.0,
                1.0 - float(getattr(legacy_query_obj, "_target_margin", 0.0) or 0.0),
            ),
        ),
    )

    retrieval_mode = getattr(legacy_query_obj, "retrieval_mode", "balanced")
    retrieval_mode_value = retrieval_mode.value if hasattr(retrieval_mode, "value") else str(retrieval_mode)
    retrieval_plan = RetrievalPlan(
        retrieval_mode=retrieval_mode_value,
        vector_weight=float(getattr(legacy_query_obj, "vector_weight", 0.5) or 0.5),
        fulltext_weight=float(getattr(legacy_query_obj, "fulltext_weight", 0.5) or 0.5),
        top_k=int(getattr(legacy_query_obj, "limit", 10) or 10),
        depth=2,
        filters=filters,
        must_match=[tmtid] if tmtid else [],
    )

    metadata = {
        "legacy_raw_intent": getattr(legacy_query_obj, "_raw_intent", "unknown"),
        "legacy_target_confidence": float(getattr(legacy_query_obj, "_target_confidence", 0.0) or 0.0),
        "legacy_target_margin": float(getattr(legacy_query_obj, "_target_margin", 0.0) or 0.0),
        "legacy_top_targets": getattr(legacy_query_obj, "_intent_top_targets", []),
        "note": "shadow bundle generated from legacy AQT output",
    }

    return IntentBundle(
        query=question,
        action_intent=action_intent,
        topics_intents=topics_intents,
        slots=slots,
        action_scores={},
        topics_scores={},
        control_features=control_features,
        adaptive_retrieval_weights=retrieval_plan,
        metadata=metadata,
    )


def run_shadow_compare() -> Path:
    test_queries = [
        "what is tmtid 662401",
        "Paracetamol Ã Â¸Â­Ã Â¸Â¢Ã Â¸Â¹Ã Â¹Ë†Ã Â¹Æ’Ã Â¸â„¢Ã Â¸Å¡Ã Â¸Â±Ã Â¸ÂÃ Â¸Å Ã Â¸ÂµÃ Â¸Â¢Ã Â¸Â²Ã Â¸Â«Ã Â¸Â¥Ã Â¸Â±Ã Â¸ÂÃ Â¹Æ’Ã Â¸Å Ã Â¹Ë†Ã Â¸Â«Ã Â¸Â£Ã Â¸Â·Ã Â¸Â­Ã Â¹â€žÃ Â¸Â¡Ã Â¹Ë†",
        "Does Pfizer make Viagra?",
        "Ã Â¸â€šÃ Â¸Â­Ã Â¸Â£Ã Â¸Â²Ã Â¸Â¢Ã Â¸Å Ã Â¸Â·Ã Â¹Ë†Ã Â¸Â­Ã Â¸Â¢Ã Â¸Â²Ã Â¸â€šÃ Â¸Â­Ã Â¸â€¡Ã Â¸Â­Ã Â¸â€¡Ã Â¸â€žÃ Â¹Å’Ã Â¸ÂÃ Â¸Â²Ã Â¸Â£Ã Â¹â‚¬Ã Â¸Â Ã Â¸ÂªÃ Â¸Â±Ã Â¸Å Ã Â¸ÂÃ Â¸Â£Ã Â¸Â£Ã Â¸Â¡",
        "Ã Â¸Â¡Ã Â¸ÂµÃ Â¸Â¢Ã Â¸Â²Ã Â¸ÂÃ Â¸ÂµÃ Â¹Ë†Ã Â¸â€¢Ã Â¸Â±Ã Â¸Â§Ã Â¹Æ’Ã Â¸â„¢Ã Â¸Å¡Ã Â¸Â±Ã Â¸ÂÃ Â¸Å Ã Â¸Âµ Ã Â¸â€¡",
        "Tiffy Ã Â¸Â¡Ã Â¸ÂµÃ Â¸ÂªÃ Â¹Ë†Ã Â¸Â§Ã Â¸â„¢Ã Â¸Å“Ã Â¸ÂªÃ Â¸Â¡Ã Â¸Â­Ã Â¸Â°Ã Â¹â€žÃ Â¸Â£Ã Â¸Å¡Ã Â¹â€°Ã Â¸Â²Ã Â¸â€¡",
        "Ã Â¸â€šÃ Â¸Â­Ã Â¸â€šÃ Â¹â€°Ã Â¸Â­Ã Â¸Â¡Ã Â¸Â¹Ã Â¸Â¥Ã Â¸Â¢Ã Â¸Â² cefazolin",
        "Ã Â¸ÂªÃ Â¸Â§Ã Â¸Â±Ã Â¸ÂªÃ Â¸â€Ã Â¸Âµ Ã Â¸Å“Ã Â¸Â¡Ã Â¸Å Ã Â¸Â·Ã Â¹Ë†Ã Â¸Â­Ã Â¸Â­Ã Â¸Â°Ã Â¹â€žÃ Â¸Â£Ã Â¹Æ’Ã Â¸Â«Ã Â¹â€°Ã Â¸â€”Ã Â¸Â²Ã Â¸Â¢",
    ]

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = LOG_DIR / f"{stamp}_intent_bundle_shadow_compare.jsonl"

    ok_count = 0
    fail_count = 0

    with out_path.open("w", encoding="utf-8") as f:
        for idx, question in enumerate(test_queries, start=1):
            start = time.perf_counter()
            try:
                legacy_query_obj = transform_query(question)
                bundle = build_shadow_bundle(question, legacy_query_obj)

                legacy_strategy = (
                    legacy_query_obj.strategy.value
                    if hasattr(legacy_query_obj.strategy, "value")
                    else str(legacy_query_obj.strategy)
                )
                legacy_target = (
                    legacy_query_obj.target_type.value
                    if hasattr(legacy_query_obj.target_type, "value")
                    else str(legacy_query_obj.target_type)
                )
                mapped_legacy_action = _map_strategy_to_action(legacy_strategy).value

                bundle_dict = _to_dict(bundle)
                comparison = {
                    "action_equivalent": mapped_legacy_action == bundle.action_intent.value,
                    "topics_contains_legacy_target": legacy_target in [t.value for t in bundle.topics_intents],
                    "weights_consistent": (
                        abs(
                            float(getattr(legacy_query_obj, "vector_weight", 0.5))
                            - float(bundle.adaptive_retrieval_weights.vector_weight)
                        )
                        < 1e-9
                        and abs(
                            float(getattr(legacy_query_obj, "fulltext_weight", 0.5))
                            - float(bundle.adaptive_retrieval_weights.fulltext_weight)
                        )
                        < 1e-9
                    ),
                }
                duration_ms = (time.perf_counter() - start) * 1000.0

                row = {
                    "timestamp": datetime.now().isoformat(),
                    "query": question,
                    "latency_ms": round(duration_ms, 2),
                    "legacy": {
                        "strategy": legacy_strategy,
                        "target_type": legacy_target,
                        "query": legacy_query_obj.query,
                        "nlem_filter": legacy_query_obj.nlem_filter,
                        "nlem_category": legacy_query_obj.nlem_category,
                        "manufacturer_filter": legacy_query_obj.manufacturer_filter,
                        "entity_ratio": legacy_query_obj.entity_ratio,
                        "retrieval_mode": (
                            legacy_query_obj.retrieval_mode.value
                            if hasattr(legacy_query_obj.retrieval_mode, "value")
                            else str(legacy_query_obj.retrieval_mode)
                        ),
                        "vector_weight": legacy_query_obj.vector_weight,
                        "fulltext_weight": legacy_query_obj.fulltext_weight,
                        "raw_intent": getattr(legacy_query_obj, "_raw_intent", "unknown"),
                    },
                    "intent_bundle_shadow": bundle_dict,
                    "comparison": comparison,
                    "status": "PASS" if all(comparison.values()) else "REVIEW",
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                ok_count += 1
                print(
                    f"[{idx}/{len(test_queries)}] PASS | "
                    f"{legacy_strategy}/{legacy_target} -> {bundle.action_intent.value}/{[x.value for x in bundle.topics_intents]}"
                )
            except Exception as exc:
                duration_ms = (time.perf_counter() - start) * 1000.0
                err = {
                    "timestamp": datetime.now().isoformat(),
                    "query": question,
                    "latency_ms": round(duration_ms, 2),
                    "status": "ERROR",
                    "error": str(exc),
                }
                f.write(json.dumps(err, ensure_ascii=False) + "\n")
                fail_count += 1
                print(f"[{idx}/{len(test_queries)}] ERROR | {question} | {exc}")

    print("\n=== Shadow Compare Summary ===")
    print(f"Log file : {out_path}")
    print(f"Success  : {ok_count}")
    print(f"Errors   : {fail_count}")
    return out_path


if __name__ == "__main__":
    run_shadow_compare()


