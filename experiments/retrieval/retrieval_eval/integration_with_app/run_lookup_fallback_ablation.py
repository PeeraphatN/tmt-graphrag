#!/usr/bin/env python3
"""
Ablation for lookup fallback effectiveness.

Compares:
1) baseline lookup (search_general only)
2) lookup with fallback (execute_lookup_query)

The query set is generated from phase1 ground-truth anchors:
- substance-based lookup anchors
- manufacturer-based lookup anchors
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
API_APP_ROOT = REPO_ROOT / "apps" / "api"
RETRIEVAL_EVAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API_APP_ROOT))

from src.services import search as search_service
from src.services.aqt import transform_query

DEFAULT_GROUND_TRUTH = RETRIEVAL_EVAL_DIR / "data" / "phase1_ground_truth.json"
DEFAULT_OUT_DIR = RETRIEVAL_EVAL_DIR / "results"


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


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
    if not pred or not gold or k <= 0:
        return 0.0
    dcg = 0.0
    for i, t in enumerate(pred[:k], start=1):
        if t in gold:
            dcg += 1.0 / math.log2(i + 1.0)
    ideal_hits = min(k, len(gold))
    if ideal_hits <= 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 1.0) for i in range(1, ideal_hits + 1))
    if idcg <= 0:
        return 0.0
    return dcg / idcg


def _clone_query_obj(query_obj: Any) -> Any:
    if hasattr(query_obj, "model_copy"):
        return query_obj.model_copy(deep=True)
    if hasattr(query_obj, "copy"):
        try:
            return query_obj.copy(deep=True)
        except TypeError:
            return query_obj.copy()
    return copy.deepcopy(query_obj)


def _policy_weights(policy: str, query_obj: Any) -> tuple[float, float, float]:
    policy_norm = _normalize_text(policy).lower()
    if policy_norm == "uniform":
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    # static by default
    v = _as_float(getattr(query_obj, "vector_weight", 0.5), 0.5)
    f = _as_float(getattr(query_obj, "fulltext_weight", 0.5), 0.5)
    vf = max(1e-9, v + f)
    base_g = 0.35
    remain = max(0.0, 1.0 - base_g)
    return (remain * (v / vf), remain * (f / vf), base_g)


def _metrics(pred: list[str], gold: set[str]) -> dict[str, float]:
    return {
        "hit@1": _hit_at_k(pred, gold, 1),
        "hit@3": _hit_at_k(pred, gold, 3),
        "hit@5": _hit_at_k(pred, gold, 5),
        "hit@10": _hit_at_k(pred, gold, 10),
        "p@5": _precision_at_k(pred, gold, 5),
        "r@5": _recall_at_k(pred, gold, 5),
        "mrr": _mrr(pred, gold),
        "ndcg@10": _ndcg_at_k(pred, gold, 10),
    }


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 6)


def _product_gold_ids(bucket: dict[str, Any]) -> list[str]:
    gold = (bucket.get("gold_tmtids") or {})
    tps = [str(x) for x in (gold.get("tp") or []) if _normalize_text(x)]
    tpus = [str(x) for x in (gold.get("tpu") or []) if _normalize_text(x)]
    deduped: list[str] = []
    seen: set[str] = set()
    for x in tps + tpus:
        if x in seen:
            continue
        seen.add(x)
        deduped.append(x)
    return deduped


def _build_queries(
    ground_truth: dict[str, Any],
    sample_substances: int,
    sample_manufacturers: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    subs_sets = list(ground_truth.get("substance_ground_truth_sets", []) or [])
    man_sets = list(ground_truth.get("manufacturer_ground_truth_sets", []) or [])

    if sample_substances > 0:
        sample_substances = min(sample_substances, len(subs_sets))
        subs_sets = rng.sample(subs_sets, sample_substances)
    if sample_manufacturers > 0:
        sample_manufacturers = min(sample_manufacturers, len(man_sets))
        man_sets = rng.sample(man_sets, sample_manufacturers)

    queries: list[dict[str, Any]] = []
    qid = 1

    for item in subs_sets:
        anchor = item.get("anchor") or {}
        substance = _normalize_text(anchor.get("substance_name"))
        if not substance:
            continue
        gold_ids = _product_gold_ids(item)
        if not gold_ids:
            continue

        templates = [
            ("th", "trade_name", f"à¸Šà¸·à¹ˆà¸­à¸—à¸²à¸‡à¸à¸²à¸£à¸„à¹‰à¸²à¸‚à¸­à¸‡à¸¢à¸² {substance} à¸¡à¸µà¸­à¸°à¹„à¸£à¸šà¹‰à¸²à¸‡"),
            ("en", "trade_name", f"what are the trade names of {substance}"),
            ("th", "manufacturer_by_substance", f"à¹ƒà¸„à¸£à¹€à¸›à¹‡à¸™à¸œà¸¹à¹‰à¸œà¸¥à¸´à¸•à¸¢à¸² {substance}"),
            ("en", "manufacturer_by_substance", f"who manufactures {substance}"),
        ]

        for lang, template, q in templates:
            queries.append(
                {
                    "query_id": f"lookup_anchor_q_{qid:06d}",
                    "query": q,
                    "language": lang,
                    "expected_operator": "lookup",
                    "metric_family": "retrieval",
                    "gold": {"relevant_tmtids": gold_ids},
                    "tags": ["lookup_anchor", template, "substance"],
                }
            )
            qid += 1

    for item in man_sets:
        anchor = item.get("anchor") or {}
        manufacturer = _normalize_text(anchor.get("manufacturer_norm"))
        if not manufacturer:
            continue
        gold_ids = _product_gold_ids(item)
        if not gold_ids:
            continue

        templates = [
            ("th", "manufacturer_lookup", f"à¸‚à¹‰à¸­à¸¡à¸¹à¸¥à¸¢à¸²à¸—à¸µà¹ˆà¸œà¸¥à¸´à¸•à¹‚à¸”à¸¢ {manufacturer}"),
            ("en", "manufacturer_lookup", f"drug information manufactured by {manufacturer}"),
        ]
        for lang, template, q in templates:
            queries.append(
                {
                    "query_id": f"lookup_anchor_q_{qid:06d}",
                    "query": q,
                    "language": lang,
                    "expected_operator": "lookup",
                    "metric_family": "retrieval",
                    "gold": {"relevant_tmtids": gold_ids},
                    "tags": ["lookup_anchor", template, "manufacturer"],
                }
            )
            qid += 1

    return queries


def run(
    ground_truth_path: Path,
    out_dir: Path,
    policy: str,
    k: int,
    depth: int,
    sample_substances: int,
    sample_manufacturers: int,
    seed: int,
) -> tuple[Path, Path]:
    if hasattr(search_service, "reset_runtime_caches"):
        search_service.reset_runtime_caches()

    ground_truth = json.loads(ground_truth_path.read_text(encoding="utf-8"))
    queries = _build_queries(
        ground_truth=ground_truth,
        sample_substances=sample_substances,
        sample_manufacturers=sample_manufacturers,
        seed=seed,
    )
    if not queries:
        raise ValueError("No lookup anchor queries were generated.")

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_jsonl = out_dir / f"lookup_fallback_ablation_{stamp}.jsonl"
    out_summary = out_dir / f"lookup_fallback_ablation_{stamp}_summary.json"

    records: list[dict[str, Any]] = []
    print("=" * 100)
    print("Lookup Fallback Ablation")
    print("=" * 100)
    print(f"Ground truth: {ground_truth_path}")
    print(f"Queries: {len(queries)} | policy={policy} | k={k} depth={depth}")

    for idx, item in enumerate(queries, start=1):
        question = _normalize_text(item.get("query"))
        gold_set = set(str(x) for x in (item.get("gold", {}) or {}).get("relevant_tmtids", []) if _normalize_text(x))
        if not question or not gold_set:
            continue

        row: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "query_index": idx,
            "query_id": item.get("query_id"),
            "query": question,
            "language": item.get("language"),
            "tags": item.get("tags", []),
            "gold_size": len(gold_set),
        }

        try:
            qobj = transform_query(question)
            v, f, g = _policy_weights(policy, qobj)
            op = search_service._resolve_operator(qobj)
            row["resolved_operator"] = op
            row["weights"] = {"vector": v, "fulltext": f, "graph": g}

            # Baseline: lookup without fallback.
            started = time.perf_counter()
            baseline_result = search_service.search_general(
                qobj,
                k=k,
                depth=depth,
                strategy_label="lookup",
                vector_weight_override=v,
                fulltext_weight_override=f,
                graph_weight_override=g,
                rerank_expanded=False,
            )
            baseline_ms = round((time.perf_counter() - started) * 1000.0, 3)
            baseline_ranked = _extract_ranked_tmtids(baseline_result)
            baseline_metrics = _metrics(baseline_ranked, gold_set)

            # New path: lookup with fallback.
            qobj2 = _clone_query_obj(qobj)
            started = time.perf_counter()
            fallback_result = search_service.execute_lookup_query(
                qobj2,
                k=k,
                depth=depth,
                vector_weight_override=v,
                fulltext_weight_override=f,
                graph_weight_override=g,
            )
            fallback_ms = round((time.perf_counter() - started) * 1000.0, 3)
            fallback_ranked = _extract_ranked_tmtids(fallback_result)
            fallback_metrics = _metrics(fallback_ranked, gold_set)

            route = fallback_result.get("route", {}) if isinstance(fallback_result, dict) else {}
            row.update(
                {
                    "status": "ok",
                    "baseline": {
                        "latency_ms": baseline_ms,
                        "seed_count": len(baseline_result.get("seed_results", []) or []),
                        "metrics": baseline_metrics,
                    },
                    "fallback": {
                        "latency_ms": fallback_ms,
                        "seed_count": len(fallback_result.get("seed_results", []) or []),
                        "metrics": fallback_metrics,
                        "route": {
                            "fallback_used": bool(route.get("fallback_used")),
                            "fallback_attempted": bool(route.get("fallback_attempted")),
                            "fallback_route_mode": route.get("fallback_route_mode"),
                            "fallback_seed_count": route.get("fallback_seed_count"),
                            "fallback_trigger": route.get("fallback_trigger"),
                        },
                    },
                }
            )
            records.append(row)
            print(
                f"[{idx:04d}] ok op={op:>8s} "
                f"base(h10={baseline_metrics['hit@10']:.0f},mrr={baseline_metrics['mrr']:.3f}) "
                f"fb(h10={fallback_metrics['hit@10']:.0f},mrr={fallback_metrics['mrr']:.3f}) "
                f"used={bool(route.get('fallback_used'))}"
            )
        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)
            records.append(row)
            print(f"[{idx:04d}] ERROR {exc}")

    with out_jsonl.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    ok_rows = [r for r in records if r.get("status") == "ok"]
    error_rows = [r for r in records if r.get("status") != "ok"]

    def _collect(path: str) -> list[float]:
        parts = path.split(".")
        vals: list[float] = []
        for r in ok_rows:
            cur: Any = r
            for p in parts:
                cur = cur.get(p, None) if isinstance(cur, dict) else None
            if cur is None:
                continue
            vals.append(_as_float(cur, 0.0))
        return vals

    fallback_used_count = sum(
        1 for r in ok_rows if bool(((r.get("fallback") or {}).get("route") or {}).get("fallback_used"))
    )
    fallback_attempt_count = sum(
        1 for r in ok_rows if bool(((r.get("fallback") or {}).get("route") or {}).get("fallback_attempted"))
    )
    resolved_operator_dist = dict(Counter(str(r.get("resolved_operator", "")) for r in ok_rows))

    summary = {
        "generated_at": datetime.now().isoformat(),
        "input_ground_truth": str(ground_truth_path),
        "output_jsonl": str(out_jsonl),
        "config": {
            "policy": policy,
            "k": k,
            "depth": depth,
            "sample_substances": sample_substances,
            "sample_manufacturers": sample_manufacturers,
            "seed": seed,
        },
        "counts": {
            "total": len(records),
            "ok": len(ok_rows),
            "error": len(error_rows),
            "fallback_attempt_count": fallback_attempt_count,
            "fallback_used_count": fallback_used_count,
            "resolved_operator_distribution": resolved_operator_dist,
        },
        "baseline": {
            "hit@1": _mean(_collect("baseline.metrics.hit@1")),
            "hit@3": _mean(_collect("baseline.metrics.hit@3")),
            "hit@5": _mean(_collect("baseline.metrics.hit@5")),
            "hit@10": _mean(_collect("baseline.metrics.hit@10")),
            "p@5": _mean(_collect("baseline.metrics.p@5")),
            "r@5": _mean(_collect("baseline.metrics.r@5")),
            "mrr": _mean(_collect("baseline.metrics.mrr")),
            "ndcg@10": _mean(_collect("baseline.metrics.ndcg@10")),
            "latency_ms_avg": _mean(_collect("baseline.latency_ms")),
        },
        "fallback": {
            "hit@1": _mean(_collect("fallback.metrics.hit@1")),
            "hit@3": _mean(_collect("fallback.metrics.hit@3")),
            "hit@5": _mean(_collect("fallback.metrics.hit@5")),
            "hit@10": _mean(_collect("fallback.metrics.hit@10")),
            "p@5": _mean(_collect("fallback.metrics.p@5")),
            "r@5": _mean(_collect("fallback.metrics.r@5")),
            "mrr": _mean(_collect("fallback.metrics.mrr")),
            "ndcg@10": _mean(_collect("fallback.metrics.ndcg@10")),
            "latency_ms_avg": _mean(_collect("fallback.latency_ms")),
        },
    }
    summary["delta_fallback_minus_baseline"] = {
        kname: round(
            _as_float(summary["fallback"].get(kname), 0.0) - _as_float(summary["baseline"].get(kname), 0.0),
            6,
        )
        for kname in ("hit@1", "hit@3", "hit@5", "hit@10", "p@5", "r@5", "mrr", "ndcg@10", "latency_ms_avg")
    }
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("-" * 100)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("-" * 100)
    return out_jsonl, out_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lookup fallback ablation")
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH, help="Path to phase1 ground truth")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory")
    parser.add_argument("--policy", type=str, default="static", choices=["static", "uniform"], help="Weight policy")
    parser.add_argument("--k", type=int, default=10, help="Top-k retrieval seeds")
    parser.add_argument("--depth", type=int, default=2, help="Graph expansion depth")
    parser.add_argument("--sample-substances", type=int, default=20, help="Sample size from substance anchors")
    parser.add_argument("--sample-manufacturers", type=int, default=20, help="Sample size from manufacturer anchors")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    args = parser.parse_args()

    run(
        ground_truth_path=args.ground_truth,
        out_dir=args.out_dir,
        policy=args.policy,
        k=max(1, int(args.k)),
        depth=max(1, int(args.depth)),
        sample_substances=max(1, int(args.sample_substances)),
        sample_manufacturers=max(1, int(args.sample_manufacturers)),
        seed=int(args.seed),
    )


if __name__ == "__main__":
    main()
