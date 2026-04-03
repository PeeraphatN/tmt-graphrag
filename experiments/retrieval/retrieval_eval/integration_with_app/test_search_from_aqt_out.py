#!/usr/bin/env python3
"""
Replay search from questions in aqt_out.json and log retrieval behavior/results.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]
API_APP_ROOT = PROJECT_ROOT / "apps" / "api"
RETRIEVAL_EVAL_DIR = Path(__file__).resolve().parents[1]
INTENT_EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "question_understanding" / "intent_classification" / "integration_with_app"
DEFAULT_INPUT = INTENT_EXPERIMENT_DIR / "results" / "aqt_cli" / "aqt_out.json"
DEFAULT_OUT_DIR = RETRIEVAL_EVAL_DIR / "results" / "search_replay"

sys.path.insert(0, str(API_APP_ROOT))

from src.services.aqt import transform_query
from src.services.search import advanced_graphrag_search


def _configure_stdout_utf8() -> None:
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


def _normalize_cases(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        if not question:
            continue
        normalized.append(item)
    return normalized


def _extract_top_seed(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    if not seed_results:
        return {}
    first = seed_results[0]
    node = first.get("node")
    if node is None:
        return {}
    try:
        props = dict(node)
    except Exception:
        props = {}
    return {
        "tmtid": props.get("tmtid"),
        "level": props.get("level"),
        "name": props.get("trade_name") or props.get("fsn") or props.get("name"),
        "manufacturer": props.get("manufacturer"),
        "score": first.get("rrf_score", first.get("score")),
    }


def _as_mode(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)


def run(input_path: Path, k: int, depth: int) -> tuple[Path, Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    cases = _normalize_cases(payload)
    if not cases:
        raise ValueError(f"No valid cases found in: {input_path}")

    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_jsonl = out_dir / f"{stamp}_search_from_aqt_out.jsonl"
    out_summary = out_dir / f"{stamp}_search_from_aqt_out_summary.json"

    new_mode_counter: Counter[str] = Counter()
    operator_counter: Counter[str] = Counter()
    old_mode_counter: Counter[str] = Counter()
    changed_weight_count = 0
    success_count = 0
    fail_count = 0
    total_seed = 0
    total_expanded = 0
    total_rels = 0

    print("=" * 90)
    print("Replay Search From aqt_out.json")
    print("=" * 90)
    print(f"Input: {input_path} | cases={len(cases)} | k={k} | depth={depth}")

    with out_jsonl.open("w", encoding="utf-8") as f:
        for idx, case in enumerate(cases, start=1):
            question = str(case.get("question", "")).strip()
            old_plan = dict(case.get("retrieval_plan", {}) or {})
            old_mode = str(old_plan.get("retrieval_mode", ""))
            if old_mode:
                old_mode_counter[old_mode] += 1

            started = time.perf_counter()
            record: dict[str, Any] = {
                "timestamp": datetime.now().isoformat(),
                "index": idx,
                "question": question,
                "description": case.get("description", ""),
                "source": case.get("source", "aqt_out"),
                "old_retrieval_plan": old_plan,
            }
            try:
                query_obj = transform_query(question)
                results = advanced_graphrag_search(query_obj, k=k, depth=depth)
                elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)

                new_mode = _as_mode(getattr(query_obj, "retrieval_mode", ""))
                new_vector = float(getattr(query_obj, "vector_weight", 0.0) or 0.0)
                new_fulltext = float(getattr(query_obj, "fulltext_weight", 0.0) or 0.0)
                new_mode_counter[new_mode] += 1

                old_vector = float(old_plan.get("vector_weight", new_vector) or new_vector)
                old_fulltext = float(old_plan.get("fulltext_weight", new_fulltext) or new_fulltext)
                changed_weight = abs(old_vector - new_vector) > 1e-9 or abs(old_fulltext - new_fulltext) > 1e-9
                if changed_weight:
                    changed_weight_count += 1

                seed_results = list(results.get("seed_results", []) or [])
                expanded_nodes = list(results.get("expanded_nodes", []) or [])
                relationships = list(results.get("relationships", []) or [])
                seed_count = len(seed_results)
                expanded_count = len(expanded_nodes)
                rel_count = len(relationships)
                route_info = dict(results.get("route", {}) or {})
                operator = str(route_info.get("operator", ""))
                if operator:
                    operator_counter[operator] += 1

                success_count += 1
                total_seed += seed_count
                total_expanded += expanded_count
                total_rels += rel_count

                record.update(
                    {
                        "status": "ok",
                        "latency_ms": elapsed_ms,
                        "new_retrieval_plan": {
                            "retrieval_mode": new_mode,
                            "vector_weight": new_vector,
                            "fulltext_weight": new_fulltext,
                            "strategy": _as_mode(getattr(query_obj, "strategy", "")),
                            "target_type": _as_mode(getattr(query_obj, "target_type", "")),
                            "entity_ratio": float(getattr(query_obj, "entity_ratio", 0.0) or 0.0),
                            "is_abstract": bool(getattr(query_obj, "_is_abstract", False)),
                        },
                        "weight_changed_from_input": changed_weight,
                        "search_metrics": {
                            "seed_count": seed_count,
                            "expanded_count": expanded_count,
                            "relationship_count": rel_count,
                        },
                        "strategy_output": str(results.get("strategy", "")),
                        "route": route_info,
                        "count_result": results.get("result"),
                        "compare_terms": list(results.get("compare_terms", []) or []),
                        "compare_detail": list(results.get("compare_detail", []) or []),
                        "top_seed": _extract_top_seed(seed_results),
                    }
                )
                print(
                    f"[{idx:02d}] OK op={operator or '-'} mode={new_mode} "
                    f"w(v={new_vector:.2f},f={new_fulltext:.2f}) "
                    f"seed={seed_count} expanded={expanded_count} rel={rel_count}"
                )
            except Exception as exc:
                fail_count += 1
                elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
                record.update(
                    {
                        "status": "error",
                        "latency_ms": elapsed_ms,
                        "error": str(exc),
                    }
                )
                print(f"[{idx:02d}] ERROR {exc}")

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    avg_seed = round(total_seed / success_count, 3) if success_count else 0.0
    avg_expanded = round(total_expanded / success_count, 3) if success_count else 0.0
    avg_rels = round(total_rels / success_count, 3) if success_count else 0.0
    summary = {
        "timestamp": datetime.now().isoformat(),
        "input_path": str(input_path),
        "total_cases": len(cases),
        "success_count": success_count,
        "fail_count": fail_count,
        "weight_changed_count": changed_weight_count,
        "old_mode_distribution": dict(old_mode_counter),
        "new_mode_distribution": dict(new_mode_counter),
        "operator_distribution": dict(operator_counter),
        "avg_seed_count": avg_seed,
        "avg_expanded_count": avg_expanded,
        "avg_relationship_count": avg_rels,
        "output_jsonl": str(out_jsonl),
    }
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("-" * 90)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("-" * 90)
    return out_jsonl, out_summary


def main() -> None:
    _configure_stdout_utf8()
    parser = argparse.ArgumentParser(description="Replay Graph search using AQT CLI output")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Path to AQT CLI output JSON")
    parser.add_argument("--k", type=int, default=10, help="Top-k seeds for search")
    parser.add_argument("--depth", type=int, default=2, help="Graph traversal depth")
    args = parser.parse_args()

    run(Path(args.input), k=max(1, args.k), depth=max(0, args.depth))


if __name__ == "__main__":
    main()
