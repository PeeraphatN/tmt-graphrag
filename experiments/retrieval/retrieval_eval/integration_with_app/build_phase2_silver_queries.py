#!/usr/bin/env python3
"""
Phase 2 silver-standard query builder.

Builds synthetic-but-grounded evaluation queries from phase1_ground_truth.json.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
API_APP_ROOT = REPO_ROOT / "apps" / "api"
sys.path.insert(0, str(API_APP_ROOT))

from src.services.database import init_driver


SUBSTANCE_DEFAULT_SCOPE = "GP"
MANUFACTURER_DEFAULT_SCOPE = "TP"


@dataclass(frozen=True)
class BuildConfig:
    seed: int
    node_lookup_count: int
    substance_set_count: int
    manufacturer_set_count: int
    compare_pair_count: int


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


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _choose_display_name(node: dict[str, Any]) -> str:
    candidates = [
        _normalize_text(node.get("trade_name")),
        _normalize_text(node.get("name")),
        _normalize_text(node.get("generic_name")),
        _normalize_text(node.get("fsn")),
    ]
    for candidate in candidates:
        if candidate:
            return candidate
    return _normalize_text(node.get("tmtid"))


def _clean_fsn_level_suffix(text: str) -> str:
    value = _normalize_text(text)
    if value.endswith(")"):
        idx = value.rfind("(")
        if idx > 0:
            tag = value[idx + 1 : -1].strip().upper()
            if tag in {"SUBS", "VTM", "GP", "GPU", "TP", "TPU"}:
                return _normalize_text(value[:idx])
    return value


def _fetch_node_brief_by_ids(tmtids: list[str]) -> dict[str, dict[str, Any]]:
    ids = sorted({str(tid).strip() for tid in tmtids if str(tid).strip()})
    if not ids:
        return {}
    drv = init_driver()
    with drv.session() as session:
        rows = session.run(
            """
            UNWIND $ids AS tid
            MATCH (n:TMT {tmtid: tid})
            RETURN
                n.tmtid AS tmtid,
                n.level AS level,
                n.name AS name,
                n.fsn AS fsn,
                n.trade_name AS trade_name,
                n.manufacturer AS manufacturer
            """,
            ids=ids,
        ).data()
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        tid = _normalize_text(row.get("tmtid"))
        out[tid] = {
            "tmtid": tid,
            "level": _normalize_text(row.get("level")),
            "name": row.get("name"),
            "fsn": row.get("fsn"),
            "trade_name": row.get("trade_name"),
            "manufacturer": row.get("manufacturer"),
        }
    return out


class QueryBuilder:
    def __init__(self) -> None:
        self._counter = 0
        self._queries: list[dict[str, Any]] = []
        self._seen: set[tuple[str, str]] = set()

    @property
    def queries(self) -> list[dict[str, Any]]:
        return self._queries

    def add(
        self,
        *,
        query: str,
        language: str,
        expected_operator: str,
        metric_family: str,
        expected_slots: list[dict[str, Any]],
        gold: dict[str, Any],
        source: dict[str, Any],
        difficulty: str,
        tags: list[str] | None = None,
    ) -> None:
        q = _normalize_text(query)
        if not q:
            return
        key = (q.lower(), expected_operator)
        if key in self._seen:
            return
        self._seen.add(key)
        self._counter += 1
        self._queries.append(
            {
                "query_id": f"silver_q_{self._counter:06d}",
                "query": q,
                "language": language,
                "difficulty": difficulty,
                "expected_operator": expected_operator,
                "metric_family": metric_family,
                "expected_slots": expected_slots,
                "gold": gold,
                "source": source,
                "tags": tags or [],
            }
        )


def _sample_rows(rng: random.Random, rows: list[dict[str, Any]], target_count: int) -> list[dict[str, Any]]:
    if target_count <= 0:
        return []
    if len(rows) <= target_count:
        return list(rows)
    sampled = rng.sample(rows, target_count)
    sampled.sort(key=lambda item: json.dumps(item, ensure_ascii=False))
    return sampled


def _sorted_unique_tmtids(values: list[Any]) -> list[str]:
    return sorted({str(v).strip() for v in values if str(v).strip()})


def build_silver_queries(phase1: dict[str, Any], config: BuildConfig) -> dict[str, Any]:
    rng = random.Random(config.seed)
    qb = QueryBuilder()

    node_pool = list(phase1.get("node_pool", []) or [])
    id_cases = list(phase1.get("id_lookup_ground_truth", []) or [])
    substance_sets = list(phase1.get("substance_ground_truth_sets", []) or [])
    manufacturer_sets = list(phase1.get("manufacturer_ground_truth_sets", []) or [])

    # -------------------------------------------------------------------------
    # 1) ID lookup queries (easy)
    # -------------------------------------------------------------------------
    for case in id_cases:
        tmtid = _normalize_text(case.get("tmtid"))
        if not tmtid:
            continue
        name = _normalize_text(case.get("name"))
        qb.add(
            query=f"TMTID {tmtid} à¸„à¸·à¸­à¸¢à¸²à¸­à¸°à¹„à¸£",
            language="th",
            expected_operator="id_lookup",
            metric_family="retrieval",
            expected_slots=[{"name": "tmtid", "value": tmtid}],
            gold={"relevant_tmtids": [tmtid], "count_value": None, "verify_label": None},
            source={"type": "id_lookup_ground_truth", "anchor": {"tmtid": tmtid, "name": name}},
            difficulty="easy",
            tags=["id", "exact"],
        )
        qb.add(
            query=f"what is tmtid {tmtid}",
            language="en",
            expected_operator="id_lookup",
            metric_family="retrieval",
            expected_slots=[{"name": "tmtid", "value": tmtid}],
            gold={"relevant_tmtids": [tmtid], "count_value": None, "verify_label": None},
            source={"type": "id_lookup_ground_truth", "anchor": {"tmtid": tmtid, "name": name}},
            difficulty="easy",
            tags=["id", "exact"],
        )

    # -------------------------------------------------------------------------
    # 2) Name lookup queries (easy/medium)
    # -------------------------------------------------------------------------
    named_nodes = []
    for node in node_pool:
        display = _clean_fsn_level_suffix(_choose_display_name(node))
        if not display or len(display) < 3:
            continue
        if display.isdigit():
            continue
        named_nodes.append({"node": node, "display": display})

    sampled_named = _sample_rows(rng, named_nodes, config.node_lookup_count)
    for i, item in enumerate(sampled_named):
        node = item["node"]
        display = item["display"]
        tmtid = _normalize_text(node.get("tmtid"))
        if not tmtid:
            continue
        if i % 2 == 0:
            query = f"à¸‚à¹‰à¸­à¸¡à¸¹à¸¥à¸¢à¸² {display}"
            language = "th"
        else:
            query = f"drug information about {display}"
            language = "en"
        qb.add(
            query=query,
            language=language,
            expected_operator="lookup",
            metric_family="retrieval",
            expected_slots=[{"name": "query", "value": display}],
            gold={"relevant_tmtids": [tmtid], "count_value": None, "verify_label": None},
            source={"type": "node_pool", "anchor": {"tmtid": tmtid, "display_name": display, "level": node.get("level")}},
            difficulty="medium" if node.get("level") in {"TP", "TPU"} else "easy",
            tags=["name_lookup"],
        )

    # -------------------------------------------------------------------------
    # 3) Substance relation queries (list + count)
    # -------------------------------------------------------------------------
    sampled_substances = _sample_rows(rng, substance_sets, config.substance_set_count)
    for row in sampled_substances:
        anchor = dict(row.get("anchor", {}) or {})
        substance = _normalize_text(anchor.get("substance_name"))
        subs_tmtid = _normalize_text(anchor.get("subs_tmtid"))
        gold_map = dict(row.get("gold_tmtids", {}) or {})
        counts = dict(row.get("counts", {}) or {})

        relevant = _sorted_unique_tmtids(list(gold_map.get("gp") or []))
        count_value = int(counts.get("gp", 0) or 0)

        if substance:
            qb.add(
                query=f"à¸£à¸²à¸¢à¸à¸²à¸£à¸¢à¸²à¸—à¸µà¹ˆà¸¡à¸µà¸ªà¹ˆà¸§à¸™à¸œà¸ªà¸¡ {substance}",
                language="th",
                expected_operator="list",
                metric_family="retrieval",
                expected_slots=[{"name": "drug", "value": substance}],
                gold={
                    "relevant_tmtids": relevant,
                    "count_value": None,
                    "verify_label": None,
                    "retrieval_scope": SUBSTANCE_DEFAULT_SCOPE,
                },
                source={
                    "type": "substance_ground_truth_sets",
                    "anchor": {
                        "subs_tmtid": subs_tmtid,
                        "substance_name": substance,
                        "scope": SUBSTANCE_DEFAULT_SCOPE,
                    },
                },
                difficulty="medium",
                tags=["substance", "relation_list"],
            )
            qb.add(
                query=f"list drugs containing {substance}",
                language="en",
                expected_operator="list",
                metric_family="retrieval",
                expected_slots=[{"name": "drug", "value": substance}],
                gold={
                    "relevant_tmtids": relevant,
                    "count_value": None,
                    "verify_label": None,
                    "retrieval_scope": SUBSTANCE_DEFAULT_SCOPE,
                },
                source={
                    "type": "substance_ground_truth_sets",
                    "anchor": {
                        "subs_tmtid": subs_tmtid,
                        "substance_name": substance,
                        "scope": SUBSTANCE_DEFAULT_SCOPE,
                    },
                },
                difficulty="medium",
                tags=["substance", "relation_list"],
            )
            qb.add(
                query=f"à¸ˆà¸³à¸™à¸§à¸™à¸¢à¸²à¸—à¸µà¹ˆà¸¡à¸µà¸ªà¹ˆà¸§à¸™à¸œà¸ªà¸¡ {substance}",
                language="th",
                expected_operator="analyze_count",
                metric_family="count",
                expected_slots=[{"name": "drug", "value": substance}],
                gold={
                    "relevant_tmtids": relevant,
                    "count_value": count_value,
                    "verify_label": None,
                    "count_scope": SUBSTANCE_DEFAULT_SCOPE,
                    "count_components": {
                        "vtm": int(counts.get("vtm", 0) or 0),
                        "gp": int(counts.get("gp", 0) or 0),
                        "gpu": int(counts.get("gpu", 0) or 0),
                        "tp": int(counts.get("tp", 0) or 0),
                        "tpu": int(counts.get("tpu", 0) or 0),
                    },
                },
                source={
                    "type": "substance_ground_truth_sets",
                    "anchor": {
                        "subs_tmtid": subs_tmtid,
                        "substance_name": substance,
                        "scope": SUBSTANCE_DEFAULT_SCOPE,
                    },
                },
                difficulty="medium",
                tags=["substance", "count"],
            )

    # -------------------------------------------------------------------------
    # 4) Manufacturer relation queries (list + count + verify true/false)
    # -------------------------------------------------------------------------
    sampled_manufacturers = _sample_rows(rng, manufacturer_sets, config.manufacturer_set_count)
    manufacturer_names = [_normalize_text(row.get("anchor", {}).get("manufacturer_norm")) for row in sampled_manufacturers]
    manufacturer_names = [m for m in manufacturer_names if m]

    verify_needed_ids: list[str] = []
    for row in sampled_manufacturers:
        gold_map = dict(row.get("gold_tmtids", {}) or {})
        tp_ids = _sorted_unique_tmtids(list(gold_map.get("tp") or []))
        tpu_ids = _sorted_unique_tmtids(list(gold_map.get("tpu") or []))
        first_id = tp_ids[0] if tp_ids else (tpu_ids[0] if tpu_ids else "")
        if first_id:
            verify_needed_ids.append(first_id)
    brief_by_id = _fetch_node_brief_by_ids(verify_needed_ids)

    for row in sampled_manufacturers:
        anchor = dict(row.get("anchor", {}) or {})
        manufacturer = _normalize_text(anchor.get("manufacturer_norm"))
        counts = dict(row.get("counts", {}) or {})
        gold_map = dict(row.get("gold_tmtids", {}) or {})
        tp_ids = _sorted_unique_tmtids(list(gold_map.get("tp") or []))
        tpu_ids = _sorted_unique_tmtids(list(gold_map.get("tpu") or []))
        relevant = tp_ids
        total_products = int(counts.get("tp", 0) or 0)

        if manufacturer:
            qb.add(
                query=f"à¸£à¸²à¸¢à¸à¸²à¸£à¸¢à¸²à¸‚à¸­à¸‡à¸œà¸¹à¹‰à¸œà¸¥à¸´à¸• {manufacturer}",
                language="th",
                expected_operator="list",
                metric_family="retrieval",
                expected_slots=[{"name": "manufacturer", "value": manufacturer}],
                gold={
                    "relevant_tmtids": relevant,
                    "count_value": None,
                    "verify_label": None,
                    "retrieval_scope": MANUFACTURER_DEFAULT_SCOPE,
                },
                source={
                    "type": "manufacturer_ground_truth_sets",
                    "anchor": {"manufacturer_norm": manufacturer, "scope": MANUFACTURER_DEFAULT_SCOPE},
                },
                difficulty="medium",
                tags=["manufacturer", "relation_list"],
            )
            qb.add(
                query=f"list drugs by manufacturer {manufacturer}",
                language="en",
                expected_operator="list",
                metric_family="retrieval",
                expected_slots=[{"name": "manufacturer", "value": manufacturer}],
                gold={
                    "relevant_tmtids": relevant,
                    "count_value": None,
                    "verify_label": None,
                    "retrieval_scope": MANUFACTURER_DEFAULT_SCOPE,
                },
                source={
                    "type": "manufacturer_ground_truth_sets",
                    "anchor": {"manufacturer_norm": manufacturer, "scope": MANUFACTURER_DEFAULT_SCOPE},
                },
                difficulty="medium",
                tags=["manufacturer", "relation_list"],
            )
            qb.add(
                query=f"à¸ˆà¸³à¸™à¸§à¸™à¸¢à¸²à¸‚à¸­à¸‡à¸œà¸¹à¹‰à¸œà¸¥à¸´à¸• {manufacturer}",
                language="th",
                expected_operator="analyze_count",
                metric_family="count",
                expected_slots=[{"name": "manufacturer", "value": manufacturer}],
                gold={
                    "relevant_tmtids": relevant,
                    "count_value": total_products,
                    "verify_label": None,
                    "count_scope": MANUFACTURER_DEFAULT_SCOPE,
                    "count_components": {
                        "tp": int(counts.get("tp", 0) or 0),
                        "tpu": int(counts.get("tpu", 0) or 0),
                        "total_products": int(counts.get("total_products", 0) or 0),
                    },
                },
                source={
                    "type": "manufacturer_ground_truth_sets",
                    "anchor": {"manufacturer_norm": manufacturer, "scope": MANUFACTURER_DEFAULT_SCOPE},
                },
                difficulty="medium",
                tags=["manufacturer", "count"],
            )

            # Verify true/false using tmtid + manufacturer.
            verify_id = relevant[0] if relevant else (tpu_ids[0] if tpu_ids else "")
            if verify_id:
                tmtid = verify_id
                node_brief = brief_by_id.get(tmtid, {})
                display_name = _clean_fsn_level_suffix(_choose_display_name(node_brief)) if node_brief else ""
                if display_name:
                    qb.add(
                        query=f"tmtid {tmtid} à¸œà¸¥à¸´à¸•à¹‚à¸”à¸¢ {manufacturer} à¹ƒà¸Šà¹ˆà¹„à¸«à¸¡",
                        language="th",
                        expected_operator="verify",
                        metric_family="verify",
                        expected_slots=[{"name": "tmtid", "value": tmtid}, {"name": "manufacturer", "value": manufacturer}],
                        gold={
                            "relevant_tmtids": [tmtid],
                            "count_value": None,
                            "verify_label": True,
                            "claim": {"tmtid": tmtid, "manufacturer": manufacturer, "drug_name": display_name},
                        },
                        source={"type": "manufacturer_ground_truth_sets", "anchor": {"manufacturer_norm": manufacturer, "tmtid": tmtid}},
                        difficulty="hard",
                        tags=["verify", "positive"],
                    )

                    other_manu = manufacturer
                    if len(manufacturer_names) > 1:
                        for candidate in manufacturer_names:
                            if candidate and candidate != manufacturer:
                                other_manu = candidate
                                break
                    if other_manu and other_manu != manufacturer:
                        qb.add(
                            query=f"tmtid {tmtid} à¸œà¸¥à¸´à¸•à¹‚à¸”à¸¢ {other_manu} à¹ƒà¸Šà¹ˆà¹„à¸«à¸¡",
                            language="th",
                            expected_operator="verify",
                            metric_family="verify",
                            expected_slots=[{"name": "tmtid", "value": tmtid}, {"name": "manufacturer", "value": other_manu}],
                            gold={
                                "relevant_tmtids": [tmtid],
                                "count_value": None,
                                "verify_label": False,
                                "claim": {"tmtid": tmtid, "manufacturer": other_manu, "drug_name": display_name},
                            },
                            source={"type": "manufacturer_ground_truth_sets", "anchor": {"manufacturer_norm": manufacturer, "tmtid": tmtid}},
                            difficulty="hard",
                            tags=["verify", "negative"],
                        )

    # -------------------------------------------------------------------------
    # 5) Compare queries
    # -------------------------------------------------------------------------
    compare_candidates = [row for row in sampled_substances if _normalize_text(row.get("anchor", {}).get("substance_name"))]
    rng.shuffle(compare_candidates)
    pair_count = min(config.compare_pair_count, len(compare_candidates) // 2)
    for i in range(pair_count):
        a = compare_candidates[2 * i]
        b = compare_candidates[2 * i + 1]
        ing_a = _normalize_text(a.get("anchor", {}).get("substance_name"))
        ing_b = _normalize_text(b.get("anchor", {}).get("substance_name"))
        if not ing_a or not ing_b or ing_a.lower() == ing_b.lower():
            continue

        gold_a = _sorted_unique_tmtids(list((a.get("gold_tmtids", {}) or {}).get("gp", [])))
        gold_b = _sorted_unique_tmtids(list((b.get("gold_tmtids", {}) or {}).get("gp", [])))
        union_gold = sorted(set(gold_a) | set(gold_b))

        qb.add(
            query=f"à¹€à¸›à¸£à¸µà¸¢à¸šà¹€à¸—à¸µà¸¢à¸šà¸‚à¹‰à¸­à¸¡à¸¹à¸¥à¸£à¸°à¸«à¸§à¹ˆà¸²à¸‡ {ing_a} à¸à¸±à¸š {ing_b}",
            language="th",
            expected_operator="analyze_compare",
            metric_family="retrieval",
            expected_slots=[{"name": "drug", "value": ing_a}, {"name": "drug", "value": ing_b}],
            gold={
                "relevant_tmtids": union_gold,
                "count_value": None,
                "verify_label": None,
                "compare_targets": {"a": gold_a, "b": gold_b},
                "retrieval_scope": SUBSTANCE_DEFAULT_SCOPE,
            },
            source={
                "type": "substance_ground_truth_sets_pair",
                "anchor": {"substance_a": ing_a, "substance_b": ing_b, "scope": SUBSTANCE_DEFAULT_SCOPE},
            },
            difficulty="hard",
            tags=["compare", "substance_pair"],
        )
        qb.add(
            query=f"compare {ing_a} vs {ing_b}",
            language="en",
            expected_operator="analyze_compare",
            metric_family="retrieval",
            expected_slots=[{"name": "drug", "value": ing_a}, {"name": "drug", "value": ing_b}],
            gold={
                "relevant_tmtids": union_gold,
                "count_value": None,
                "verify_label": None,
                "compare_targets": {"a": gold_a, "b": gold_b},
                "retrieval_scope": SUBSTANCE_DEFAULT_SCOPE,
            },
            source={
                "type": "substance_ground_truth_sets_pair",
                "anchor": {"substance_a": ing_a, "substance_b": ing_b, "scope": SUBSTANCE_DEFAULT_SCOPE},
            },
            difficulty="hard",
            tags=["compare", "substance_pair"],
        )

    # -------------------------------------------------------------------------
    # 6) Global deterministic count probes
    # -------------------------------------------------------------------------
    db_counts = dict(phase1.get("counts", {}).get("db_level_counts", {}) or {})
    product_total = int(db_counts.get("GP", 0)) + int(db_counts.get("GPU", 0)) + int(db_counts.get("TP", 0)) + int(db_counts.get("TPU", 0))
    hierarchy_total = sum(int(v) for v in db_counts.values())

    qb.add(
        query="how many drugs does TMT produce",
        language="en",
        expected_operator="analyze_count",
        metric_family="count",
        expected_slots=[{"name": "query", "value": "how many drugs does TMT produce"}],
        gold={
            "relevant_tmtids": [],
            "count_value": product_total,
            "verify_label": None,
            "count_scope": "all product levels GP/GPU/TP/TPU",
        },
        source={"type": "db_level_counts", "anchor": {"levels": ["GP", "GPU", "TP", "TPU"]}},
        difficulty="easy",
        tags=["count", "global_product_total"],
    )

    qb.add(
        query="à¸ˆà¸³à¸™à¸§à¸™à¹‚à¸«à¸™à¸”à¸¢à¸²à¸—à¸±à¹‰à¸‡à¸«à¸¡à¸”à¹ƒà¸™à¸£à¸°à¸šà¸š TMT",
        language="th",
        expected_operator="analyze_count",
        metric_family="count",
        expected_slots=[{"name": "query", "value": "à¸ˆà¸³à¸™à¸§à¸™à¹‚à¸«à¸™à¸”à¸¢à¸²à¸—à¸±à¹‰à¸‡à¸«à¸¡à¸”"}],
        gold={
            "relevant_tmtids": [],
            "count_value": hierarchy_total,
            "verify_label": None,
            "count_scope": "all levels SUBS/VTM/GP/GPU/TP/TPU",
        },
        source={"type": "db_level_counts", "anchor": {"levels": list(db_counts.keys())}},
        difficulty="easy",
        tags=["count", "global_all_levels"],
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    by_operator = Counter(q["expected_operator"] for q in qb.queries)
    by_language = Counter(q["language"] for q in qb.queries)
    by_metric = Counter(q["metric_family"] for q in qb.queries)
    by_difficulty = Counter(q["difficulty"] for q in qb.queries)

    return {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "phase": "phase2_silver_queries",
            "seed": config.seed,
            "source_ground_truth_phase1": "experiments/retrieval/retrieval_eval/data/phase1_ground_truth.json",
            "script": "experiments/retrieval/retrieval_eval/run/build_phase2_silver_queries.py",
        },
        "sampling": {
            "node_lookup_count": config.node_lookup_count,
            "substance_set_count": config.substance_set_count,
            "manufacturer_set_count": config.manufacturer_set_count,
            "compare_pair_count": config.compare_pair_count,
        },
        "summary": {
            "total_queries": len(qb.queries),
            "by_operator": dict(by_operator),
            "by_language": dict(by_language),
            "by_metric_family": dict(by_metric),
            "by_difficulty": dict(by_difficulty),
        },
        "queries": qb.queries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build phase-2 silver standard query set.")
    parser.add_argument(
        "--phase1",
        default="experiments/retrieval/retrieval_eval/data/phase1_ground_truth.json",
        help="Input phase1 ground truth JSON file.",
    )
    parser.add_argument(
        "--output",
        default="experiments/retrieval/retrieval_eval/data/phase2_silver_queries.json",
        help="Output silver query JSON file.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--node-lookup-count", type=int, default=160, help="Sample size for name-lookup queries.")
    parser.add_argument("--substance-set-count", type=int, default=60, help="Substance relation set sample size.")
    parser.add_argument("--manufacturer-set-count", type=int, default=60, help="Manufacturer relation set sample size.")
    parser.add_argument("--compare-pair-count", type=int, default=35, help="Compare pair sample size.")
    return parser.parse_args()


def main() -> None:
    _configure_stdout_utf8()
    args = parse_args()
    phase1_path = Path(args.phase1)
    if not phase1_path.exists():
        raise FileNotFoundError(f"Phase1 ground truth file not found: {phase1_path}")

    phase1 = json.loads(phase1_path.read_text(encoding="utf-8"))
    config = BuildConfig(
        seed=args.seed,
        node_lookup_count=max(1, args.node_lookup_count),
        substance_set_count=max(1, args.substance_set_count),
        manufacturer_set_count=max(1, args.manufacturer_set_count),
        compare_pair_count=max(1, args.compare_pair_count),
    )

    payload = build_silver_queries(phase1, config)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Silver queries generated.")
    print(f"Output: {output_path}")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
