#!/usr/bin/env python3
"""
Run manual Cypher COUNT queries against the same question set in aqt_out.json.
Focused on COUNT intent to compare hand-written query logic vs router output.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]
API_APP_ROOT = PROJECT_ROOT / "apps" / "api"
RETRIEVAL_EVAL_DIR = Path(__file__).resolve().parents[1]
INTENT_EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "question_understanding" / "intent_classification" / "integration_with_app"
DEFAULT_AQT_INPUT = INTENT_EXPERIMENT_DIR / "results" / "aqt_cli" / "aqt_out.json"
DEFAULT_ROUTER_RESULTS_DIR = RETRIEVAL_EVAL_DIR / "results" / "search_replay"
DEFAULT_OUT_DIR = RETRIEVAL_EVAL_DIR / "results" / "manual_count"

sys.path.insert(0, str(API_APP_ROOT))

from src.services.database import init_driver


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


def _find_latest_router_jsonl(results_dir: Path) -> Path | None:
    if not results_dir.exists():
        return None
    candidates = sorted(results_dir.glob("*_search_from_aqt_out.jsonl"))
    if not candidates:
        return None
    return candidates[-1]


def _load_router_counts(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        question = str(row.get("question", "")).strip()
        if not question:
            continue
        route = dict(row.get("route", {}) or {})
        if route.get("operator") != "analyze_count":
            continue
        records[question] = {
            "count_result": row.get("count_result"),
            "route": route,
        }
    return records


def _slot_values(case: dict[str, Any], slot_name: str) -> list[str]:
    values: list[str] = []
    for slot in case.get("slots", []) or []:
        if not isinstance(slot, dict):
            continue
        if str(slot.get("name", "")).strip() != slot_name:
            continue
        value = " ".join(str(slot.get("value", "")).split()).strip()
        if value:
            values.append(value)
    return values


def _build_manual_count_query(case: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
    question = str(case.get("question", "")).strip()
    question_l = question.lower()

    drug_values = _slot_values(case, "drug")
    manufacturer_values = _slot_values(case, "manufacturer")

    if drug_values:
        substance = drug_values[0].strip().lower()
        cypher = """
        MATCH (subs:TMT {level:'SUBS'})
        WHERE toLower(coalesce(subs.name, '')) = $substance
        MATCH (subs)-[:IS_ACTIVE_SUBSTANCE_OF]->(v:TMT {level:'VTM'})<-[:HAS_VTM]-(gp:TMT {level:'GP'})
        OPTIONAL MATCH (gp)-[:HAS_GENERIC_UNIT]->(gpu:TMT {level:'GPU'})
        OPTIONAL MATCH (gp)-[:HAS_TRADE_PRODUCT]->(tp:TMT {level:'TP'})
        OPTIONAL MATCH (tp)-[:HAS_TRADE_UNIT]->(tpu:TMT {level:'TPU'})
        RETURN count(DISTINCT v) as vtm_count,
               count(DISTINCT gp) as gp_count,
               count(DISTINCT gpu) as gpu_count,
               count(DISTINCT tp) as tp_count,
               count(DISTINCT tpu) as tpu_count,
               count(DISTINCT gp) + count(DISTINCT gpu) + count(DISTINCT tp) + count(DISTINCT tpu) as total
        """
        return cypher, {"substance": substance}, "substance_graph_descendants"

    if manufacturer_values:
        manufacturer = manufacturer_values[0].strip().lower()
        cypher = """
        MATCH (n:TMT)
        WHERE n.level IN ['TP','TPU']
          AND toLower(coalesce(n.manufacturer, '')) CONTAINS $manufacturer
        RETURN count(DISTINCT n) as total
        """
        return cypher, {"manufacturer": manufacturer}, "manufacturer_products"

    if "how many" in question_l or "à¸ˆà¸³à¸™à¸§à¸™" in question_l or "à¸à¸µà¹ˆ" in question:
        cypher = """
        MATCH (n:TMT)
        WHERE n.level IN ['GP','GPU','TP','TPU']
        RETURN count(n) as total
        """
        return cypher, {}, "all_product_levels"

    cypher = """
    MATCH (n:TMT)
    RETURN count(n) as total
    """
    return cypher, {}, "all_tmt_nodes"


def _run_manual_count(cypher: str, params: dict[str, Any]) -> dict[str, Any]:
    drv = init_driver()
    with drv.session() as session:
        record = session.run(cypher, **params).single()
        if not record:
            return {"total": 0}
        return dict(record)


def run(aqt_path: Path, router_jsonl: Path | None) -> tuple[Path, Path]:
    if not aqt_path.exists():
        raise FileNotFoundError(f"Input file not found: {aqt_path}")

    payload = json.loads(aqt_path.read_text(encoding="utf-8"))
    cases = _normalize_cases(payload)
    count_cases = [c for c in cases if str(c.get("action_intent", "")).strip().lower() == "count"]
    if not count_cases:
        raise ValueError("No count cases found in input.")

    router_counts = _load_router_counts(router_jsonl)

    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_jsonl = out_dir / f"{stamp}_manual_count_eval.jsonl"
    out_summary = out_dir / f"{stamp}_manual_count_eval_summary.json"

    rows: list[dict[str, Any]] = []
    more_constrained = 0
    broader = 0
    same = 0
    no_router_baseline = 0

    print("=" * 90)
    print("Manual COUNT Query Evaluation")
    print("=" * 90)
    print(f"Input: {aqt_path} | count_cases={len(count_cases)}")
    if router_jsonl:
        print(f"Router baseline: {router_jsonl}")
    else:
        print("Router baseline: not provided")

    for idx, case in enumerate(count_cases, start=1):
        question = str(case.get("question", "")).strip()
        cypher, params, query_kind = _build_manual_count_query(case)
        manual_row = _run_manual_count(cypher, params)
        manual_total = int(manual_row.get("total", 0) or 0)

        router_row = router_counts.get(question, {})
        router_total = router_row.get("count_result")
        if router_total is None:
            no_router_baseline += 1
            delta = None
        else:
            delta = manual_total - int(router_total)
            if delta < 0:
                more_constrained += 1
            elif delta > 0:
                broader += 1
            else:
                same += 1

        row = {
            "index": idx,
            "question": question,
            "description": case.get("description", ""),
            "manual_query_kind": query_kind,
            "manual_cypher": " ".join(cypher.split()),
            "manual_params": params,
            "manual_result": manual_row,
            "router_count_result": router_total,
            "router_route": router_row.get("route"),
            "delta_manual_minus_router": delta,
        }
        rows.append(row)
        print(
            f"[{idx:02d}] {query_kind} | manual={manual_total} "
            f"| router={router_total if router_total is not None else '-'} "
            f"| delta={delta if delta is not None else '-'}"
        )

    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "aqt_input": str(aqt_path),
        "router_baseline": str(router_jsonl) if router_jsonl else None,
        "total_count_cases": len(count_cases),
        "more_constrained_cases": more_constrained,
        "broader_cases": broader,
        "same_cases": same,
        "missing_router_baseline_cases": no_router_baseline,
        "output_jsonl": str(out_jsonl),
    }
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("-" * 90)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("-" * 90)
    return out_jsonl, out_summary


def main() -> None:
    _configure_stdout_utf8()
    parser = argparse.ArgumentParser(description="Evaluate manual COUNT Cypher against router baseline.")
    parser.add_argument("--input", default=str(DEFAULT_AQT_INPUT), help="Path to AQT CLI output JSON")
    parser.add_argument(
        "--router-jsonl",
        default="",
        help="Path to *_search_from_aqt_out.jsonl. If empty, latest file in retrieval replay results is used.",
    )
    args = parser.parse_args()

    router_path: Path | None = None
    if args.router_jsonl:
        router_path = Path(args.router_jsonl)
    else:
        router_path = _find_latest_router_jsonl(DEFAULT_ROUTER_RESULTS_DIR)

    run(Path(args.input), router_path)


if __name__ == "__main__":
    main()
