import argparse
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean


SEMANTIC_OPS = {"lookup", "list", "analyze_compare", "compare"}
COMPARE_OPS = {"analyze_compare", "compare"}


def _safe_mean(values):
    return mean(values) if values else None


def _percentile(values, q):
    if not values:
        return None
    if q <= 0:
        return min(values)
    if q >= 100:
        return max(values)
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (q / 100.0)
    floor_i = math.floor(k)
    ceil_i = math.ceil(k)
    if floor_i == ceil_i:
        return sorted_vals[int(k)]
    d0 = sorted_vals[floor_i] * (ceil_i - k)
    d1 = sorted_vals[ceil_i] * (k - floor_i)
    return d0 + d1


def _load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _evaluate_threshold(metric, threshold):
    value = metric.get("value")
    if value is None:
        return {"pass": False, "reason": "missing metric"}

    op = threshold["op"]
    target = threshold["value"]
    if op == ">=":
        passed = value >= target
    elif op == "<=":
        passed = value <= target
    else:
        raise ValueError(f"Unsupported operator: {op}")
    return {"pass": passed, "target": target, "op": op}


def _compute_policy_metrics(rows):
    metrics = {}

    status_ok = sum(1 for r in rows if r.get("status") == "ok")
    error_rate = 1.0 - (status_ok / len(rows)) if rows else None
    latencies = [float(r.get("latency_ms", 0.0)) for r in rows if r.get("latency_ms") is not None]
    latency_p95 = _percentile(latencies, 95)

    retrieval_rows = [
        r
        for r in rows
        if r.get("metric_family") == "retrieval"
        and str(r.get("expected_operator", "")).lower() in SEMANTIC_OPS
    ]
    lookup_rows = [r for r in retrieval_rows if str(r.get("expected_operator", "")).lower() == "lookup"]
    list_rows = [r for r in retrieval_rows if str(r.get("expected_operator", "")).lower() == "list"]
    compare_rows = [
        r for r in retrieval_rows if str(r.get("expected_operator", "")).lower() in COMPARE_OPS
    ]

    count_rows = [r for r in rows if r.get("metric_family") == "count"]
    verify_non_id_rows = [
        r
        for r in rows
        if r.get("metric_family") == "verify" and str(r.get("strategy", "")).lower() != "id_lookup"
    ]

    def metric_avg(records, key):
        vals = []
        for r in records:
            m = r.get("metrics") or {}
            if key in m and m[key] is not None:
                vals.append(float(m[key]))
        return _safe_mean(vals)

    metrics["overall_semantic_hit@10"] = {
        "value": metric_avg(retrieval_rows, "hit@10"),
        "count": len(retrieval_rows),
    }
    metrics["lookup_hit@10"] = {
        "value": metric_avg(lookup_rows, "hit@10"),
        "count": len(lookup_rows),
    }
    metrics["list_hit@10"] = {
        "value": metric_avg(list_rows, "hit@10"),
        "count": len(list_rows),
    }
    metrics["compare_hit@10"] = {
        "value": metric_avg(compare_rows, "hit@10"),
        "count": len(compare_rows),
    }
    metrics["overall_semantic_mrr"] = {
        "value": metric_avg(retrieval_rows, "mrr"),
        "count": len(retrieval_rows),
    }
    metrics["overall_semantic_ndcg@10"] = {
        "value": metric_avg(retrieval_rows, "ndcg@10"),
        "count": len(retrieval_rows),
    }
    metrics["count_exact_match"] = {
        "value": metric_avg(count_rows, "exact_match"),
        "count": len(count_rows),
    }
    metrics["count_mape"] = {
        "value": metric_avg(count_rows, "ape"),
        "count": len(count_rows),
    }
    metrics["error_rate"] = {"value": error_rate, "count": len(rows)}
    metrics["latency_p95_ms"] = {"value": latency_p95, "count": len(latencies)}
    metrics["verify_non_id_accuracy"] = {
        "value": metric_avg(verify_non_id_rows, "accuracy"),
        "count": len(verify_non_id_rows),
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Check PoC semantic acceptance criteria from phase3 JSONL.")
    parser.add_argument(
        "--runs-jsonl",
        required=True,
        help="Path to phase3 runs jsonl",
    )
    parser.add_argument(
        "--criteria-json",
        default="experiments/retrieval/retrieval_eval/data/poc_acceptance_criteria_semantic_v1.json",
        help="Path to semantic acceptance criteria json",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional output path (default: sidecar next to runs)",
    )
    args = parser.parse_args()

    runs_path = Path(args.runs_jsonl)
    criteria_path = Path(args.criteria_json)
    rows = _load_jsonl(runs_path)
    criteria = json.loads(criteria_path.read_text(encoding="utf-8"))

    by_policy = defaultdict(list)
    for r in rows:
        by_policy[str(r.get("policy", "unknown"))].append(r)

    report = {
        "generated_at": datetime.now().isoformat(),
        "runs_jsonl": str(runs_path),
        "criteria_json": str(criteria_path),
        "policies": {},
    }

    thresholds = criteria.get("thresholds", {})
    for policy, policy_rows in sorted(by_policy.items()):
        pm = _compute_policy_metrics(policy_rows)
        checks = {}
        all_pass = True
        for metric_name, threshold in thresholds.items():
            metric = pm.get(metric_name, {"value": None})
            verdict = _evaluate_threshold(metric, threshold)
            checks[metric_name] = {
                "value": metric.get("value"),
                "count": metric.get("count"),
                **verdict,
            }
            all_pass = all_pass and bool(verdict["pass"])
        report["policies"][policy] = {
            "overall_pass": all_pass,
            "metrics": pm,
            "checks": checks,
        }

    if args.output_json:
        out_path = Path(args.output_json)
    else:
        out_path = runs_path.with_name(runs_path.stem + "_poc_semantic_acceptance.json")

    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 100)
    print("PoC Semantic Acceptance Report")
    print("=" * 100)
    for policy, data in report["policies"].items():
        print(f"[{policy}] overall_pass={data['overall_pass']}")
        for metric_name, check in data["checks"].items():
            value = check.get("value")
            op = check.get("op")
            target = check.get("target")
            status = "PASS" if check.get("pass") else "FAIL"
            print(f"  - {metric_name}: value={value} {op} {target} => {status}")
    print("-" * 100)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
