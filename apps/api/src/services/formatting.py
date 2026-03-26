"""
Formatting Service.
Uses LLM to generate the final answer based on retrieved context.
"""
from __future__ import annotations

import json
import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from src.models.llm import get_llm
from src.prompts.templates import FORMATTER_PROMPT


def _get_action_intent(payload: dict) -> str:
    query_obj = payload.get("query_obj")
    bundle = getattr(query_obj, "intent_bundle", None) or {}
    action = str(bundle.get("action_intent") or "").strip().lower()
    if action:
        return action
    strategy = getattr(query_obj, "strategy", "retrieve")
    if hasattr(strategy, "value"):
        strategy = strategy.value
    strategy = str(strategy).strip().lower()
    if strategy == "retrieve":
        return "lookup"
    if strategy:
        return strategy
    return "lookup"


def _extract_requested_limit(question: str, action_intent: str) -> int:
    text = str(question or "")
    if action_intent != "list":
        return 5

    # Capture explicit numeric request, e.g. "5 ชื่อ", "top 5", "5 items"
    patterns = [
        r"\btop\s*(\d+)\b",
        r"\b(\d+)\s*(?:items?|results?|names?)\b",
        r"(\d+)\s*(?:ชื่อ|รายการ|ตัว)",
        r"\b(\d+)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue
        try:
            value = int(match.group(1))
            return max(1, min(20, value))
        except Exception:
            continue
    return 5


def _compact_entities(entities: list[dict], max_items: int) -> list[dict]:
    compact = []
    for entity in entities[:max_items]:
        if not isinstance(entity, dict):
            continue
        compact.append(
            {
                "tmtid": entity.get("tmtid"),
                "level": entity.get("level"),
                "trade_name": entity.get("trade_name") or entity.get("trade_name_fallback"),
                "fsn": entity.get("fsn"),
                "manufacturer": entity.get("manufacturer") or entity.get("manufacturer_fallback"),
            }
        )
    return compact


def _build_formatter_context(payload: dict) -> dict:
    context = payload.get("context", {})
    if not isinstance(context, dict):
        return {"raw_context": str(context or "")}

    action_intent = _get_action_intent(payload)
    question = str(payload.get("question") or "")
    requested_limit = _extract_requested_limit(question, action_intent)

    entities = context.get("entities", []) or []
    evidence = context.get("evidence", []) or []

    # Keep payload compact for small-context local LLMs.
    if action_intent == "compare":
        entity_cap = 12
    elif action_intent == "verify":
        entity_cap = 10
    elif action_intent == "list":
        entity_cap = min(20, max(8, requested_limit * 3))
    else:
        entity_cap = 12

    compact = {
        "action_intent": action_intent,
        "requested_limit": requested_limit,
        "question_type": str(context.get("question_type", "")),
        "total_entities": int(context.get("total_entities", len(entities) or 0)),
        "entities": _compact_entities(entities, entity_cap),
        "evidence": evidence[:8],
    }

    if "count" in context:
        compact["count"] = context.get("count")
    if "strategy" in context:
        compact["strategy"] = context.get("strategy")
    return compact

def get_formatter_chain() -> Runnable:
    """
    Creates a chain that takes {"question": str, "context": str/dict} 
    and returns the final answer (str).
    """
    llm = get_llm(temperature=0)
    
    chain = (
        {
            "question": lambda x: x["question"],
            "context": lambda x: (
                x["context"]
                if isinstance(x["context"], str)
                else json.dumps(
                    _build_formatter_context(x),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            ),
        }
        | FORMATTER_PROMPT 
        | llm 
        | StrOutputParser()
    )
    
    return chain
