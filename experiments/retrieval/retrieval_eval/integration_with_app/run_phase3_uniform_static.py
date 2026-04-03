#!/usr/bin/env python3
"""
Phase 3 benchmark runner (uniform vs static weighting, RRF-only).

Runs queries from phase2_silver_queries.json against live GraphRAG search,
then computes retrieval/count/verify metrics per policy.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
API_APP_ROOT = REPO_ROOT / "apps" / "api"
RETRIEVAL_EVAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API_APP_ROOT))

from src.services.aqt import transform_query
from src.services import search as search_service

DEFAULT_QUERIES = RETRIEVAL_EVAL_DIR / "data" / "phase2_silver_queries.json"
DEFAULT_OUT_DIR = RETRIEVAL_EVAL_DIR / "results"
SUPPORTED_POLICIES = {"uniform", "static"}


def _configure_utf8() -> None:
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


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = _normalize_text(value).lower()
    return text in {"true", "1", "yes", "y"}


def _node_to_dict(node: Any) -> dict[str, Any]:
    if node is None:
        return {}
    if isinstance(node, dict):
        return dict(node)
    try:
        return dict(node)
    except Exception:
        pass
    props: dict[str, Any] = {}
    for key in ("tmtid", "level", "name", "fsn", "trade_name", "manufacturer"):
        try:
            props[key] = node.get(key)  # type: ignore[attr-defined]
        except Exception:
            continue
    return props


def _extract_ranked_tmtids(search_result: dict[str, Any], limit: int = 200) -> list[str]:
    ranked: list[str] = []
    seen: set[str] = set()
    for item in search_result.get("seed_results", []) or []:
        node = item.get("node") if isinstance(item, dict) else None
        props = _node_to_dict(node)
        tmtid = _normalize_text(props.get("tmtid"))
        if not tmtid or tmtid in seen:
            continue
        seen.add(tmtid)
        ranked.append(tmtid)
        if len(ranked) >= limit:
            break
    return ranked


def _first_non_none(values: list[Any]) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _clone_query_obj(query_obj: Any) -> Any:
    if hasattr(query_obj, "model_copy"):
        return query_obj.model_copy(deep=True)
    if hasattr(query_obj, "copy"):
        try:
            return query_obj.copy(deep=True)
        except TypeError:
            return query_obj.copy()
    return copy.deepcopy(query_obj)


def _policy_weights(policy: str, query_obj: Any) -> tuple[float | None, float | None, float | None]:
    if policy == "uniform":
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    if policy == "static":
        return (
            _as_float(getattr(query_obj, "vector_weight", 0.5), 0.5),
            _as_float(getattr(query_obj, "fulltext_weight", 0.5), 0.5),
            None,
        )
    raise ValueError(f"Unsupported policy: {policy}")


def _run_lookup_with_policy(query_obj: Any, policy: str, k: int, depth: int) -> dict[str, Any]:
    v, f, g = _policy_weights(policy, query_obj)
    return search_service.execute_lookup_query(
        query_obj,
        k=k,
        depth=depth,
        vector_weight_override=v,
        fulltext_weight_override=f,
        graph_weight_override=g,
    )


def _run_verify_with_policy(query_obj: Any, policy: str, k: int, depth: int) -> dict[str, Any]:
    v, f, g = _policy_weights(policy, query_obj)

    base_filters = search_service._build_filters_from_query_obj(query_obj)
    target_type = search_service._enum_value(getattr(query_obj, "target_type", "general"))
    strict_levels = search_service.get_search_config(target_type)["allowed_levels"]

    strict = search_service.search_general(
        query_obj,
        k=max(8, min(k, 20)),
        depth=depth,
        filters_override=base_filters,
        allowed_levels_override=strict_levels,
        strategy_label="verify",
        vector_weight_override=v,
        fulltext_weight_override=f,
        graph_weight_override=g,
        rerank_expanded=False,
    )
    strict_seed = len(strict.get("seed_results", []))
    if strict_seed > 0:
        strict["route"] = {
            "operator": "verify",
            "policy": policy,
            "fallback_used": False,
            "attempts": [{"name": "strict", "seed_count": strict_seed}],
        }
        return strict

    relaxed_filters = dict(base_filters)
    if "manufacturer" in relaxed_filters:
        relaxed_filters.pop("manufacturer", None)
    elif "nlem_category" in relaxed_filters:
        relaxed_filters.pop("nlem_category", None)
    elif "nlem" in relaxed_filters:
        relaxed_filters.pop("nlem", None)

    attempt2 = search_service.search_general(
        query_obj,
        k=max(8, min(k, 20)),
        depth=depth,
        filters_override=relaxed_filters,
        allowed_levels_override=strict_levels,
        strategy_label="verify",
        vector_weight_override=v,
        fulltext_weight_override=f,
        graph_weight_override=g,
        rerank_expanded=False,
    )
    attempt2_seed = len(attempt2.get("seed_results", []))
    if attempt2_seed > 0:
        attempt2["route"] = {
            "operator": "verify",
            "policy": policy,
            "fallback_used": True,
            "fallback_tier": 1,
            "attempts": [
                {"name": "strict", "seed_count": strict_seed},
                {"name": "relax_one_filter", "seed_count": attempt2_seed},
            ],
        }
        return attempt2

    broader_levels = search_service.get_search_config("general")["allowed_levels"]
    attempt3 = search_service.search_general(
        query_obj,
        k=max(8, min(k, 20)),
        depth=depth,
        filters_override=relaxed_filters,
        allowed_levels_override=broader_levels,
        strategy_label="verify",
        vector_weight_override=v,
        fulltext_weight_override=f,
        graph_weight_override=g,
        rerank_expanded=False,
    )
    attempt3_seed = len(attempt3.get("seed_results", []))
    attempt3["route"] = {
        "operator": "verify",
        "policy": policy,
        "fallback_used": True,
        "fallback_tier": 2,
        "attempts": [
            {"name": "strict", "seed_count": strict_seed},
            {"name": "relax_one_filter", "seed_count": attempt2_seed},
            {"name": "broaden_scope", "seed_count": attempt3_seed},
        ],
    }
    return attempt3


def _run_compare_with_policy(query_obj: Any, policy: str, k: int, depth: int) -> dict[str, Any]:
    v, f, g = _policy_weights(policy, query_obj)

    bundle = getattr(query_obj, "intent_bundle", None)
    compare_terms = search_service._extract_slot_values(bundle, "drug")
    if len(compare_terms) < 2:
        compare_terms.extend([t for t in search_service._extract_slot_values(bundle, "brand") if t not in compare_terms])
    if len(compare_terms) < 2:
        compare_terms.extend([t for t in search_service._split_compare_terms(getattr(query_obj, "query", "")) if t not in compare_terms])
    if not compare_terms:
        compare_terms = [getattr(query_obj, "query", "") or "drug"]
    compare_terms = compare_terms[:2]

    compare_filters = search_service._build_filters_from_query_obj(query_obj)
    compare_filters.pop("manufacturer", None)
    compare_filters.pop("tmtid", None)
    compare_levels = search_service.get_search_config("substance")["allowed_levels"]

    result_groups: list[list[dict[str, Any]]] = []
    per_term: list[dict[str, Any]] = []

    for term in compare_terms:
        result = search_service.search_general(
            query_obj,
            k=max(8, min(k, 20)),
            depth=min(depth, 2),
            query_override=term,
            filters_override=compare_filters,
            allowed_levels_override=compare_levels,
            vector_weight_override=v,
            fulltext_weight_override=f,
            graph_weight_override=g,
            strategy_label="compare",
            rerank_expanded=False,
        )
        seeds = result.get("seed_results", [])
        result_groups.append(seeds)
        per_term.append({"term": term, "seed_count": len(seeds), "top_seed": search_service._safe_top_seed(seeds)})

    merged_seed_results = search_service._merge_seed_results(result_groups)
    seed_node_ids = [
        item["node"].element_id if hasattr(item["node"], "element_id") else item["node"].id
        for item in merged_seed_results
        if isinstance(item, dict) and item.get("node") is not None
    ]

    if not seed_node_ids:
        return {
            "strategy": "compare",
            "seed_results": [],
            "expanded_nodes": [],
            "relationships": [],
            "compare_terms": compare_terms,
            "compare_detail": per_term,
            "route": {"operator": "analyze_compare", "policy": policy, "fallback_used": False},
        }

    expanded = search_service.expand_context(seed_node_ids, depth=min(depth, 2))
    non_seed_nodes = [n for n in expanded["nodes"] if not n.get("is_seed", False)]
    return {
        "strategy": "compare",
        "seed_results": merged_seed_results,
        "expanded_nodes": non_seed_nodes,
        "relationships": expanded["relationships"],
        "compare_terms": compare_terms,
        "compare_detail": per_term,
        "route": {"operator": "analyze_compare", "policy": policy, "fallback_used": False},
    }


def _run_query_with_policy(query_obj: Any, policy: str, k: int, depth: int) -> dict[str, Any]:
    operator = search_service._resolve_operator(query_obj)

    if operator == "id_lookup":
        result = search_service.execute_id_lookup_query(query_obj, depth=1)
    elif operator == "analyze_count":
        result = search_service.execute_count_query(query_obj)
    elif operator == "analyze_compare":
        result = _run_compare_with_policy(query_obj, policy=policy, k=k, depth=depth)
    elif operator == "verify":
        result = _run_verify_with_policy(query_obj, policy=policy, k=k, depth=1)
    elif operator == "list":
        result = search_service.execute_listing_query(query_obj, k=50)
    else:
        result = _run_lookup_with_policy(query_obj, policy=policy, k=k, depth=depth)

    route = dict(result.get("route", {}) or {})
    route.setdefault("operator", operator)
    route.setdefault("policy", policy)
    result["route"] = route
    return result


def _precision_at_k(pred: list[str], gold: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    topk = pred[:k]
    if not topk:
        return 0.0
    hits = sum(1 for t in topk if t in gold)
    return hits / float(k)


def _recall_at_k(pred: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    topk = pred[:k]
    if not topk:
        return 0.0
    hits = sum(1 for t in topk if t in gold)
    return hits / float(len(gold))


def _hit_at_k(pred: list[str], gold: set[str], k: int) -> float:
    if not pred or not gold:
        return 0.0
    return 1.0 if any(t in gold for t in pred[:k]) else 0.0


def _mrr(pred: list[str], gold: set[str]) -> float:
    if not pred or not gold:
        return 0.0
    for i, t in enumerate(pred, start=1):
        if t in gold:
            return 1.0 / float(i)
    return 0.0


def _ndcg_at_k(pred: list[str], gold: set[str], k: int) -> float:
    if k <= 0 or not gold:
        return 0.0
    topk = pred[:k]
    if not topk:
        return 0.0

    dcg = 0.0
    for i, t in enumerate(topk, start=1):
        rel = 1.0 if t in gold else 0.0
        if rel > 0.0:
            dcg += rel / math.log2(i + 1.0)

    ideal_len = min(k, len(gold))
    if ideal_len <= 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 1.0) for i in range(1, ideal_len + 1))
    if idcg <= 1e-12:
        return 0.0
    return dcg / idcg


def _predict_verify_label(query_item: dict[str, Any], search_result: dict[str, Any]) -> bool:
    claim = ((query_item.get("gold") or {}).get("claim") or {})
    claim_tmtid = _normalize_text(claim.get("tmtid"))
    claim_manufacturer = _normalize_text(claim.get("manufacturer")).lower()

    for item in search_result.get("seed_results", []) or []:
        node = item.get("node") if isinstance(item, dict) else None
        props = _node_to_dict(node)
        tmtid = _normalize_text(props.get("tmtid"))
        manufacturer = _normalize_text(props.get("manufacturer")).lower()

        if claim_tmtid and tmtid != claim_tmtid:
            continue
        if claim_manufacturer and claim_manufacturer not in manufacturer:
            continue
        return True

    return False


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def _build_summary(records: list[dict[str, Any]], policies: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "total_records": len(records),
        "policy_summaries": {},
    }

    for policy in policies:
        policy_rows = [r for r in records if r.get("policy") == policy]
        ok_rows = [r for r in policy_rows if r.get("status") == "ok"]
        err_rows = [r for r in policy_rows if r.get("status") != "ok"]

        op_counter = Counter(str(r.get("expected_operator", "")) for r in ok_rows)
        latency_values = [_as_float(r.get("latency_ms"), 0.0) for r in ok_rows]

        retrieval_rows = [r for r in ok_rows if r.get("metric_family") == "retrieval"]
        count_rows = [r for r in ok_rows if r.get("metric_family") == "count"]
        verify_rows = [r for r in ok_rows if r.get("metric_family") == "verify"]

        retrieval_metrics: dict[str, Any] = {}
        for key in ("hit@1", "hit@3", "hit@5", "hit@10", "p@5", "r@5", "mrr", "ndcg@10"):
            values = [_as_float((r.get("metrics") or {}).get(key), 0.0) for r in retrieval_rows]
            retrieval_metrics[key] = _mean(values)

        count_em = [_as_float((r.get("metrics") or {}).get("exact_match"), 0.0) for r in count_rows]
        count_mae = [_as_float((r.get("metrics") or {}).get("abs_error"), 0.0) for r in count_rows]
        count_mape = [
            _as_float((r.get("metrics") or {}).get("ape"), float("nan"))
            for r in count_rows
            if (r.get("metrics") or {}).get("ape") is not None
        ]
        count_mape = [v for v in count_mape if not math.isnan(v)]

        tp = sum(1 for r in verify_rows if _as_bool((r.get("metrics") or {}).get("gold_label")) and _as_bool((r.get("metrics") or {}).get("pred_label")))
        fp = sum(1 for r in verify_rows if not _as_bool((r.get("metrics") or {}).get("gold_label")) and _as_bool((r.get("metrics") or {}).get("pred_label")))
        fn = sum(1 for r in verify_rows if _as_bool((r.get("metrics") or {}).get("gold_label")) and not _as_bool((r.get("metrics") or {}).get("pred_label")))
        tn = sum(1 for r in verify_rows if not _as_bool((r.get("metrics") or {}).get("gold_label")) and not _as_bool((r.get("metrics") or {}).get("pred_label")))

        verify_acc_values = [_as_float((r.get("metrics") or {}).get("accuracy"), 0.0) for r in verify_rows]
        verify_precision = (tp / (tp + fp)) if (tp + fp) else 0.0
        verify_recall = (tp / (tp + fn)) if (tp + fn) else 0.0
        verify_f1 = (2 * verify_precision * verify_recall / (verify_precision + verify_recall)) if (verify_precision + verify_recall) else 0.0

        by_operator: dict[str, Any] = {}
        for operator, rows in defaultdict(list, {
            op: [r for r in ok_rows if str(r.get("expected_operator", "")) == op] for op in sorted(op_counter.keys())
        }).items():
            fam = _first_non_none([r.get("metric_family") for r in rows])
            op_info: dict[str, Any] = {"metric_family": fam, "count": len(rows)}
            if fam == "retrieval":
                op_info["hit@10"] = _mean([_as_float((r.get("metrics") or {}).get("hit@10"), 0.0) for r in rows])
                op_info["mrr"] = _mean([_as_float((r.get("metrics") or {}).get("mrr"), 0.0) for r in rows])
            elif fam == "count":
                op_info["exact_match"] = _mean([_as_float((r.get("metrics") or {}).get("exact_match"), 0.0) for r in rows])
                op_info["mae"] = _mean([_as_float((r.get("metrics") or {}).get("abs_error"), 0.0) for r in rows])
            elif fam == "verify":
                op_info["accuracy"] = _mean([_as_float((r.get("metrics") or {}).get("accuracy"), 0.0) for r in rows])
            by_operator[operator] = op_info

        summary["policy_summaries"][policy] = {
            "records_total": len(policy_rows),
            "success_count": len(ok_rows),
            "error_count": len(err_rows),
            "operator_distribution": dict(op_counter),
            "latency_ms_avg": _mean(latency_values),
            "retrieval": {
                "count": len(retrieval_rows),
                **retrieval_metrics,
            },
            "count": {
                "count": len(count_rows),
                "exact_match": _mean(count_em),
                "mae": _mean(count_mae),
                "mape": _mean(count_mape),
            },
            "verify": {
                "count": len(verify_rows),
                "accuracy": _mean(verify_acc_values),
                "precision": round(verify_precision, 6),
                "recall": round(verify_recall, 6),
                "f1": round(verify_f1, 6),
                "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            },
            "by_operator": by_operator,
        }

    return summary


def run(
    queries_path: Path,
    out_dir: Path,
    policies: list[str],
    k: int,
    depth: int,
    max_queries: int | None = None,
) -> tuple[Path, Path]:
    if hasattr(search_service, "reset_runtime_caches"):
        search_service.reset_runtime_caches()

    payload = json.loads(queries_path.read_text(encoding="utf-8"))
    queries = list(payload.get("queries", [])) if isinstance(payload, dict) else []
    queries = [q for q in queries if isinstance(q, dict) and _normalize_text(q.get("query"))]

    if max_queries is not None and max_queries > 0:
        queries = queries[: max_queries]

    if not queries:
        raise ValueError(f"No valid queries in {queries_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_jsonl = out_dir / f"phase3_uniform_static_runs_{stamp}.jsonl"
    out_summary = out_dir / f"phase3_uniform_static_summary_{stamp}.json"

    records: list[dict[str, Any]] = []

    print("=" * 100)
    print("Phase 3 Benchmark: uniform vs static")
    print("=" * 100)
    print(f"Input: {queries_path}")
    print(f"Queries: {len(queries)} | Policies: {policies} | k={k} depth={depth}")

    for idx, item in enumerate(queries, start=1):
        question = _normalize_text(item.get("query"))
        if not question:
            continue

        try:
            base_query_obj = transform_query(question)
        except Exception as exc:
            for policy in policies:
                records.append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "query_index": idx,
                        "query_id": item.get("query_id"),
                        "query": question,
                        "policy": policy,
                        "expected_operator": item.get("expected_operator"),
                        "metric_family": item.get("metric_family"),
                        "language": item.get("language"),
                        "difficulty": item.get("difficulty"),
                        "status": "error",
                        "latency_ms": 0.0,
                        "error": f"transform_query failed: {exc}",
                    }
                )
                print(f"[{idx:04d}] {policy:7s} ERROR transform_query failed: {exc}")
            continue

        for policy in policies:
            started = time.perf_counter()
            base_record: dict[str, Any] = {
                "timestamp": datetime.now().isoformat(),
                "query_index": idx,
                "query_id": item.get("query_id"),
                "query": question,
                "policy": policy,
                "expected_operator": item.get("expected_operator"),
                "metric_family": item.get("metric_family"),
                "language": item.get("language"),
                "difficulty": item.get("difficulty"),
            }

            try:
                query_obj = _clone_query_obj(base_query_obj)
                if policy == "uniform":
                    query_obj.vector_weight = 1.0 / 3.0
                    query_obj.fulltext_weight = 1.0 / 3.0
                elif policy == "static":
                    # Keep AQT-provided weights as-is
                    pass

                search_result = _run_query_with_policy(query_obj, policy=policy, k=k, depth=depth)
                latency_ms = round((time.perf_counter() - started) * 1000.0, 3)

                ranked_tmtids = _extract_ranked_tmtids(search_result)
                gold = (item.get("gold") or {})
                gold_tmtids = [str(x) for x in (gold.get("relevant_tmtids") or []) if _normalize_text(x)]
                gold_set = set(gold_tmtids)

                metrics: dict[str, Any] = {}
                metric_family = str(item.get("metric_family", ""))
                if metric_family == "retrieval":
                    metrics = {
                        "hit@1": _hit_at_k(ranked_tmtids, gold_set, 1),
                        "hit@3": _hit_at_k(ranked_tmtids, gold_set, 3),
                        "hit@5": _hit_at_k(ranked_tmtids, gold_set, 5),
                        "hit@10": _hit_at_k(ranked_tmtids, gold_set, 10),
                        "p@5": _precision_at_k(ranked_tmtids, gold_set, 5),
                        "r@5": _recall_at_k(ranked_tmtids, gold_set, 5),
                        "mrr": _mrr(ranked_tmtids, gold_set),
                        "ndcg@10": _ndcg_at_k(ranked_tmtids, gold_set, 10),
                    }
                elif metric_family == "count":
                    gold_count = gold.get("count_value")
                    pred_count = search_result.get("result")
                    pred_int = int(pred_count) if pred_count is not None else 0
                    gold_int = int(gold_count) if gold_count is not None else 0
                    abs_error = abs(pred_int - gold_int)
                    ape = (abs_error / gold_int) if gold_int > 0 else None
                    metrics = {
                        "pred_count": pred_int,
                        "gold_count": gold_int,
                        "exact_match": 1.0 if pred_int == gold_int else 0.0,
                        "abs_error": float(abs_error),
                        "ape": float(ape) if ape is not None else None,
                    }
                elif metric_family == "verify":
                    gold_label = _as_bool(gold.get("verify_label"))
                    pred_label = _predict_verify_label(item, search_result)
                    metrics = {
                        "gold_label": gold_label,
                        "pred_label": pred_label,
                        "accuracy": 1.0 if pred_label == gold_label else 0.0,
                    }

                route = dict(search_result.get("route", {}) or {})
                route_plan = dict(search_result.get("route_plan", {}) or {})
                seed_count = len(search_result.get("seed_results", []) or [])
                expanded_count = len(search_result.get("expanded_nodes", []) or [])
                rel_count = len(search_result.get("relationships", []) or [])

                rec = {
                    **base_record,
                    "status": "ok",
                    "latency_ms": latency_ms,
                    "strategy": search_result.get("strategy"),
                    "metrics": metrics,
                    "ranked_tmtids": ranked_tmtids[:50],
                    "gold_relevant_tmtids": gold_tmtids[:300],
                    "search_counts": {
                        "seed_count": seed_count,
                        "expanded_count": expanded_count,
                        "relationship_count": rel_count,
                    },
                    "route": route,
                    "route_plan": {
                        "vector_weight": route_plan.get("vector_weight"),
                        "fulltext_weight": route_plan.get("fulltext_weight"),
                        "graph_weight": route_plan.get("graph_weight"),
                        "channel_hits": route_plan.get("channel_hits"),
                    },
                }
                records.append(rec)

                print(
                    f"[{idx:04d}] {policy:7s} ok op={item.get('expected_operator'):>14s} "
                    f"fam={item.get('metric_family'):>9s} seed={seed_count:3d} "
                    f"lat={latency_ms:8.2f}ms"
                )
            except Exception as exc:
                latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
                rec = {
                    **base_record,
                    "status": "error",
                    "latency_ms": latency_ms,
                    "error": str(exc),
                }
                records.append(rec)
                print(f"[{idx:04d}] {policy:7s} ERROR {exc}")

    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = _build_summary(records, policies)
    summary.update(
        {
            "input_queries_path": str(queries_path),
            "output_jsonl": str(out_jsonl),
            "config": {
                "k": k,
                "depth": depth,
                "policies": policies,
                "max_queries": max_queries,
                "rrf_only": True,
            },
        }
    )
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("-" * 100)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("-" * 100)

    return out_jsonl, out_summary


def main() -> None:
    _configure_utf8()

    parser = argparse.ArgumentParser(description="Run Phase 3 benchmark (uniform vs static)")
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES, help="Path to phase2 query set JSON")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory")
    parser.add_argument("--k", type=int, default=10, help="Top-k seeds for retrieval")
    parser.add_argument("--depth", type=int, default=2, help="Graph expansion depth")
    parser.add_argument("--max-queries", type=int, default=None, help="Optional cap on number of queries")
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["uniform", "static"],
        help="Policies to run (uniform, static)",
    )
    args = parser.parse_args()

    policies = [str(p).strip().lower() for p in args.policies if str(p).strip()]
    if not policies:
        raise ValueError("No policies specified")
    unsupported = [p for p in policies if p not in SUPPORTED_POLICIES]
    if unsupported:
        raise ValueError(f"Unsupported policies: {unsupported}. Supported: {sorted(SUPPORTED_POLICIES)}")

    run(
        queries_path=args.queries,
        out_dir=args.out_dir,
        policies=policies,
        k=max(1, int(args.k)),
        depth=max(0, int(args.depth)),
        max_queries=args.max_queries,
    )


if __name__ == "__main__":
    main()
