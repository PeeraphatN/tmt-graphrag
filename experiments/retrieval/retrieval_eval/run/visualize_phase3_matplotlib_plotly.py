#!/usr/bin/env python3
"""
Generate richer visualizations for Phase 3 benchmark using matplotlib + plotly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


RETRIEVAL_METRICS = ["hit@1", "hit@3", "hit@5", "hit@10", "mrr", "ndcg@10", "p@5", "r@5"]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            obj = json.loads(text)
        except Exception:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _summary_policy(summary: dict[str, Any], policy: str) -> dict[str, Any]:
    return (summary.get("policy_summaries", {}) or {}).get(policy, {}) or {}


def _plot_retrieval_metrics(summary: dict[str, Any], out_path: Path) -> None:
    u = _summary_policy(summary, "uniform").get("retrieval", {}) or {}
    s = _summary_policy(summary, "static").get("retrieval", {}) or {}

    u_vals = [_as_float(u.get(m), 0.0) for m in RETRIEVAL_METRICS]
    s_vals = [_as_float(s.get(m), 0.0) for m in RETRIEVAL_METRICS]

    x = np.arange(len(RETRIEVAL_METRICS))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, u_vals, width, label="uniform", color="#4f46e5")
    ax.bar(x + width / 2, s_vals, width, label="static", color="#059669")

    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(RETRIEVAL_METRICS, rotation=20)
    ax.set_ylabel("score")
    ax.set_title("Phase 3 Retrieval Metrics (RRF-only)")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend()

    for i, (uv, sv) in enumerate(zip(u_vals, s_vals)):
        ax.text(i - width / 2, uv + 0.01, f"{uv:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + width / 2, sv + 0.01, f"{sv:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_latency_distribution(run_rows: list[dict[str, Any]], out_path: Path) -> None:
    uniform = [_as_float(r.get("latency_ms"), 0.0) for r in run_rows if r.get("policy") == "uniform" and r.get("status") == "ok"]
    static = [_as_float(r.get("latency_ms"), 0.0) for r in run_rows if r.get("policy") == "static" and r.get("status") == "ok"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bins = 60
    if uniform:
        ax.hist(uniform, bins=bins, alpha=0.45, label="uniform", color="#4f46e5", density=True)
    if static:
        ax.hist(static, bins=bins, alpha=0.45, label="static", color="#059669", density=True)

    ax.set_title("Latency Distribution by Policy (RRF-only)")
    ax.set_xlabel("latency (ms)")
    ax.set_ylabel("density")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_operator_breakdown(summary: dict[str, Any], out_path: Path) -> None:
    u_ops = (_summary_policy(summary, "uniform").get("by_operator", {}) or {})
    s_ops = (_summary_policy(summary, "static").get("by_operator", {}) or {})

    retrieval_ops = [op for op in sorted(set(u_ops.keys()) | set(s_ops.keys())) if (u_ops.get(op, {}) or s_ops.get(op, {})).get("metric_family") == "retrieval"]
    if not retrieval_ops:
        retrieval_ops = ["id_lookup", "lookup", "list", "analyze_compare"]

    u_hit = [_as_float((u_ops.get(op, {}) or {}).get("hit@10"), 0.0) for op in retrieval_ops]
    s_hit = [_as_float((s_ops.get(op, {}) or {}).get("hit@10"), 0.0) for op in retrieval_ops]
    u_mrr = [_as_float((u_ops.get(op, {}) or {}).get("mrr"), 0.0) for op in retrieval_ops]
    s_mrr = [_as_float((s_ops.get(op, {}) or {}).get("mrr"), 0.0) for op in retrieval_ops]

    x = np.arange(len(retrieval_ops))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

    axes[0].bar(x - width / 2, u_hit, width, label="uniform", color="#4f46e5")
    axes[0].bar(x + width / 2, s_hit, width, label="static", color="#059669")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(retrieval_ops, rotation=20)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("hit@10 by operator")
    axes[0].grid(axis="y", linestyle="--", alpha=0.25)

    axes[1].bar(x - width / 2, u_mrr, width, label="uniform", color="#4f46e5")
    axes[1].bar(x + width / 2, s_mrr, width, label="static", color="#059669")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(retrieval_ops, rotation=20)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("MRR by operator")
    axes[1].grid(axis="y", linestyle="--", alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle("Operator Breakdown (Retrieval Operators)")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _build_plotly_dashboard(summary: dict[str, Any], run_rows: list[dict[str, Any]], out_path: Path) -> None:
    u = _summary_policy(summary, "uniform")
    s = _summary_policy(summary, "static")

    ur = (u.get("retrieval", {}) or {})
    sr = (s.get("retrieval", {}) or {})

    uniform_lat = [_as_float(r.get("latency_ms"), 0.0) for r in run_rows if r.get("policy") == "uniform" and r.get("status") == "ok"]
    static_lat = [_as_float(r.get("latency_ms"), 0.0) for r in run_rows if r.get("policy") == "static" and r.get("status") == "ok"]

    u_ops = (u.get("by_operator", {}) or {})
    s_ops = (s.get("by_operator", {}) or {})
    retrieval_ops = [op for op in sorted(set(u_ops.keys()) | set(s_ops.keys())) if (u_ops.get(op, {}) or s_ops.get(op, {})).get("metric_family") == "retrieval"]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Retrieval Metrics",
            "Latency Box Plot",
            "Operator hit@10",
            "Operator MRR",
        ),
    )

    fig.add_trace(
        go.Bar(name="uniform", x=RETRIEVAL_METRICS, y=[_as_float(ur.get(m), 0.0) for m in RETRIEVAL_METRICS], marker_color="#4f46e5"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(name="static", x=RETRIEVAL_METRICS, y=[_as_float(sr.get(m), 0.0) for m in RETRIEVAL_METRICS], marker_color="#059669"),
        row=1,
        col=1,
    )

    fig.add_trace(go.Box(name="uniform", y=uniform_lat, marker_color="#4f46e5"), row=1, col=2)
    fig.add_trace(go.Box(name="static", y=static_lat, marker_color="#059669"), row=1, col=2)

    fig.add_trace(
        go.Bar(name="uniform hit@10", x=retrieval_ops, y=[_as_float((u_ops.get(op, {}) or {}).get("hit@10"), 0.0) for op in retrieval_ops], marker_color="#4f46e5"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(name="static hit@10", x=retrieval_ops, y=[_as_float((s_ops.get(op, {}) or {}).get("hit@10"), 0.0) for op in retrieval_ops], marker_color="#059669"),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Bar(name="uniform mrr", x=retrieval_ops, y=[_as_float((u_ops.get(op, {}) or {}).get("mrr"), 0.0) for op in retrieval_ops], marker_color="#4f46e5"),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Bar(name="static mrr", x=retrieval_ops, y=[_as_float((s_ops.get(op, {}) or {}).get("mrr"), 0.0) for op in retrieval_ops], marker_color="#059669"),
        row=2,
        col=2,
    )

    fig.update_layout(
        title="Phase 3 Dashboard (RRF-only, uniform vs static)",
        barmode="group",
        height=900,
        width=1400,
        legend_tracegroupgap=8,
    )
    fig.update_yaxes(range=[0, 1.05], row=1, col=1)
    fig.update_yaxes(title_text="latency (ms)", row=1, col=2)
    fig.update_yaxes(range=[0, 1.05], row=2, col=1)
    fig.update_yaxes(range=[0, 1.05], row=2, col=2)

    fig.write_html(str(out_path), include_plotlyjs=True, full_html=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Phase 3 results with matplotlib + plotly")
    parser.add_argument("--summary", required=True, type=Path, help="phase3 summary json")
    parser.add_argument("--runs", required=True, type=Path, help="phase3 runs jsonl")
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/retrieval/retrieval_eval/results/plots"), help="output directory")
    args = parser.parse_args()

    summary = _load_json(args.summary)
    run_rows = _load_jsonl(args.runs)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    stem = args.summary.stem.replace("phase3_uniform_static_summary_", "")
    png1 = args.out_dir / f"phase3_{stem}_retrieval_metrics.png"
    png2 = args.out_dir / f"phase3_{stem}_latency_distribution.png"
    png3 = args.out_dir / f"phase3_{stem}_operator_breakdown.png"
    html1 = args.out_dir / f"phase3_{stem}_dashboard_plotly.html"

    _plot_retrieval_metrics(summary, png1)
    _plot_latency_distribution(run_rows, png2)
    _plot_operator_breakdown(summary, png3)
    _build_plotly_dashboard(summary, run_rows, html1)

    print("Generated files:")
    print(f"- {png1}")
    print(f"- {png2}")
    print(f"- {png3}")
    print(f"- {html1}")


if __name__ == "__main__":
    main()
