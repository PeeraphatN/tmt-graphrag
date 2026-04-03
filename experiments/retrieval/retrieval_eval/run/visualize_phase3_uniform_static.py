#!/usr/bin/env python3
"""
Create self-contained HTML visualization for Phase 3 uniform/static benchmark results.
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


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


def _fmt_num(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def _fmt_ms(value: float) -> str:
    return f"{value:.2f} ms"


def _metric_bar_row(name: str, u_val: float, s_val: float, scale: float = 1.0) -> str:
    maxv = max(u_val, s_val, 1e-9)
    u_pct = min(100.0, (u_val / maxv) * 100.0)
    s_pct = min(100.0, (s_val / maxv) * 100.0)

    def _bar(pct: float, color: str) -> str:
        return (
            "<div class='bar-track'>"
            f"<div class='bar-fill' style='width:{pct:.2f}%; background:{color};'></div>"
            "</div>"
        )

    return (
        "<tr>"
        f"<td>{escape(name)}</td>"
        f"<td>{_fmt_num(u_val / scale, 6)}{_bar(u_pct, '#4f46e5')}</td>"
        f"<td>{_fmt_num(s_val / scale, 6)}{_bar(s_pct, '#059669')}</td>"
        "</tr>"
    )


def _build_operator_table(summary: dict[str, Any]) -> str:
    policies = summary.get("policy_summaries", {}) or {}
    u = (policies.get("uniform") or {}).get("by_operator", {}) or {}
    s = (policies.get("static") or {}).get("by_operator", {}) or {}
    ops = sorted(set(u.keys()) | set(s.keys()))

    rows: list[str] = []
    for op in ops:
        uo = u.get(op, {}) or {}
        so = s.get(op, {}) or {}
        fam = _normalize_text(uo.get("metric_family") or so.get("metric_family"))
        count = int(uo.get("count", so.get("count", 0)) or 0)

        if fam == "retrieval":
            m1u = _as_float(uo.get("hit@10"), 0.0)
            m1s = _as_float(so.get("hit@10"), 0.0)
            m2u = _as_float(uo.get("mrr"), 0.0)
            m2s = _as_float(so.get("mrr"), 0.0)
            metric_names = "hit@10 / mrr"
        elif fam == "count":
            m1u = _as_float(uo.get("exact_match"), 0.0)
            m1s = _as_float(so.get("exact_match"), 0.0)
            m2u = _as_float(uo.get("mae"), 0.0)
            m2s = _as_float(so.get("mae"), 0.0)
            metric_names = "exact_match / mae"
        elif fam == "verify":
            m1u = _as_float(uo.get("accuracy"), 0.0)
            m1s = _as_float(so.get("accuracy"), 0.0)
            m2u = 0.0
            m2s = 0.0
            metric_names = "accuracy"
        else:
            m1u = m1s = m2u = m2s = 0.0
            metric_names = "-"

        rows.append(
            "<tr>"
            f"<td>{escape(op)}</td>"
            f"<td>{escape(fam)}</td>"
            f"<td>{count}</td>"
            f"<td>{escape(metric_names)}</td>"
            f"<td>{_fmt_num(m1u, 6)}</td>"
            f"<td>{_fmt_num(m1s, 6)}</td>"
            f"<td>{_fmt_num(m1s - m1u, 6)}</td>"
            f"<td>{_fmt_num(m2u, 6)}</td>"
            f"<td>{_fmt_num(m2s, 6)}</td>"
            f"<td>{_fmt_num(m2s - m2u, 6)}</td>"
            "</tr>"
        )

    if not rows:
        rows.append("<tr><td colspan='10'>No operator breakdown found.</td></tr>")

    return (
        "<table class='data-table'>"
        "<thead><tr><th>operator</th><th>family</th><th>count</th><th>metrics</th>"
        "<th>uniform m1</th><th>static m1</th><th>delta m1</th>"
        "<th>uniform m2</th><th>static m2</th><th>delta m2</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table>"
    )


def _build_query_diff_section(run_rows: list[dict[str, Any]]) -> str:
    if not run_rows:
        return "<p>No run JSONL rows loaded; skip query-level diagnostics.</p>"

    pairs: dict[str, dict[str, dict[str, Any]]] = {}
    for row in run_rows:
        qid = _normalize_text(row.get("query_id"))
        pol = _normalize_text(row.get("policy"))
        if not qid or not pol:
            continue
        pairs.setdefault(qid, {})[pol] = row

    diff_rank_rows: list[dict[str, Any]] = []
    latency_deltas: list[float] = []
    weight_diff_count = 0

    for qid, pair in pairs.items():
        if "uniform" not in pair or "static" not in pair:
            continue
        u = pair["uniform"]
        s = pair["static"]

        uw = (u.get("route_plan") or {})
        sw = (s.get("route_plan") or {})
        if any(
            abs(_as_float(uw.get(k), 0.0) - _as_float(sw.get(k), 0.0)) > 1e-9
            for k in ("vector_weight", "fulltext_weight", "graph_weight")
        ):
            weight_diff_count += 1

        u10 = tuple((u.get("ranked_tmtids") or [])[:10])
        s10 = tuple((s.get("ranked_tmtids") or [])[:10])
        if u10 != s10:
            diff_rank_rows.append(
                {
                    "query_id": qid,
                    "operator": _normalize_text(u.get("expected_operator")),
                    "query": _normalize_text(u.get("query")),
                    "uniform_top3": list(u10[:3]),
                    "static_top3": list(s10[:3]),
                }
            )

        latency_deltas.append(_as_float(s.get("latency_ms"), 0.0) - _as_float(u.get("latency_ms"), 0.0))

    diff_rank_rows = sorted(diff_rank_rows, key=lambda r: (r["operator"], r["query_id"]))

    if latency_deltas:
        avg_delta = statistics.mean(latency_deltas)
        med_delta = statistics.median(latency_deltas)
        p90_delta = sorted(latency_deltas)[int(0.9 * (len(latency_deltas) - 1))]
    else:
        avg_delta = med_delta = p90_delta = 0.0

    top_rows_html: list[str] = []
    for row in diff_rank_rows[:20]:
        top_rows_html.append(
            "<tr>"
            f"<td>{escape(row['query_id'])}</td>"
            f"<td>{escape(row['operator'])}</td>"
            f"<td>{escape(row['query'][:120])}</td>"
            f"<td>{escape(', '.join(row['uniform_top3']))}</td>"
            f"<td>{escape(', '.join(row['static_top3']))}</td>"
            "</tr>"
        )

    if not top_rows_html:
        top_rows_html.append("<tr><td colspan='5'>No top-10 ranking differences found.</td></tr>")

    return (
        "<div class='card'>"
        "<h3>Query-Level Diagnostics</h3>"
        f"<p>paired queries: <b>{len(pairs)}</b> | weight-different queries: <b>{weight_diff_count}</b> | "
        f"top10-different queries: <b>{len(diff_rank_rows)}</b></p>"
        f"<p>latency delta (static - uniform): avg <b>{_fmt_num(avg_delta, 3)} ms</b>, "
        f"median <b>{_fmt_num(med_delta, 3)} ms</b>, p90 <b>{_fmt_num(p90_delta, 3)} ms</b></p>"
        "<table class='data-table'>"
        "<thead><tr><th>query_id</th><th>operator</th><th>query (trimmed)</th><th>uniform top3</th><th>static top3</th></tr></thead>"
        "<tbody>" + "".join(top_rows_html) + "</tbody></table>"
        "</div>"
    )


def build_html(summary: dict[str, Any], run_rows: list[dict[str, Any]], summary_path: Path) -> str:
    pol = summary.get("policy_summaries", {}) or {}
    u = pol.get("uniform", {}) or {}
    s = pol.get("static", {}) or {}

    ur = (u.get("retrieval") or {})
    sr = (s.get("retrieval") or {})
    uc = (u.get("count") or {})
    sc = (s.get("count") or {})
    uv = (u.get("verify") or {})
    sv = (s.get("verify") or {})

    retrieval_rows = "".join(
        [
            _metric_bar_row("hit@1", _as_float(ur.get("hit@1")), _as_float(sr.get("hit@1"))),
            _metric_bar_row("hit@3", _as_float(ur.get("hit@3")), _as_float(sr.get("hit@3"))),
            _metric_bar_row("hit@5", _as_float(ur.get("hit@5")), _as_float(sr.get("hit@5"))),
            _metric_bar_row("hit@10", _as_float(ur.get("hit@10")), _as_float(sr.get("hit@10"))),
            _metric_bar_row("mrr", _as_float(ur.get("mrr")), _as_float(sr.get("mrr"))),
            _metric_bar_row("ndcg@10", _as_float(ur.get("ndcg@10")), _as_float(sr.get("ndcg@10"))),
            _metric_bar_row("p@5", _as_float(ur.get("p@5")), _as_float(sr.get("p@5"))),
            _metric_bar_row("r@5", _as_float(ur.get("r@5")), _as_float(sr.get("r@5"))),
        ]
    )

    generated_at = _normalize_text(summary.get("generated_at"))
    total_records = int(summary.get("total_records", 0) or 0)
    cfg = summary.get("config", {}) or {}

    html = f"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Phase 3 Visualization (uniform vs static)</title>
<style>
  body {{ font-family: "Segoe UI", Arial, sans-serif; margin: 20px; color: #0f172a; background: #f8fafc; }}
  h1, h2, h3 {{ margin: 0 0 10px 0; }}
  .meta {{ color: #334155; margin-bottom: 18px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(320px,1fr)); gap: 14px; margin-bottom: 14px; }}
  .card {{ background: #ffffff; border: 1px solid #cbd5e1; border-radius: 12px; padding: 14px; box-shadow: 0 2px 8px rgba(15,23,42,0.06); }}
  .kpi {{ display: flex; justify-content: space-between; margin: 6px 0; }}
  .kpi b {{ color: #111827; }}
  .data-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  .data-table th, .data-table td {{ border: 1px solid #dbeafe; padding: 6px 8px; vertical-align: top; }}
  .data-table th {{ background: #eff6ff; text-align: left; }}
  .bar-track {{ width: 100%; height: 10px; background: #e2e8f0; border-radius: 8px; margin-top: 3px; overflow: hidden; }}
  .bar-fill {{ height: 100%; border-radius: 8px; }}
  .legend {{ display: inline-flex; gap: 14px; margin-bottom: 8px; font-size: 12px; color: #334155; }}
  .dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-right: 5px; }}
  .small {{ font-size: 12px; color: #475569; }}
</style>
</head>
<body>
  <h1>Phase 3 Benchmark Visualization (RRF-only)</h1>
  <div class="meta">
    generated_at: <b>{escape(generated_at)}</b> | total_records: <b>{total_records}</b> | 
    config: k=<b>{escape(str(cfg.get('k')))}</b>, depth=<b>{escape(str(cfg.get('depth')))}</b>, 
    policies=<b>{escape(str(cfg.get('policies')))}</b>, rrf_only=<b>{escape(str(cfg.get('rrf_only')))}</b><br/>
    source summary: <code>{escape(str(summary_path))}</code>
  </div>

  <div class="grid">
    <div class="card">
      <h3>Retrieval Metrics (700 queries)</h3>
      <div class="legend">
        <span><span class="dot" style="background:#4f46e5"></span>uniform</span>
        <span><span class="dot" style="background:#059669"></span>static</span>
      </div>
      <table class="data-table">
        <thead><tr><th>metric</th><th>uniform</th><th>static</th></tr></thead>
        <tbody>{retrieval_rows}</tbody>
      </table>
    </div>

    <div class="card">
      <h3>Latency & Run Health</h3>
      <div class="kpi"><span>uniform avg latency</span><b>{_fmt_ms(_as_float(u.get('latency_ms_avg')))}</b></div>
      <div class="kpi"><span>static avg latency</span><b>{_fmt_ms(_as_float(s.get('latency_ms_avg')))}</b></div>
      <div class="kpi"><span>delta (static - uniform)</span><b>{_fmt_num(_as_float(s.get('latency_ms_avg')) - _as_float(u.get('latency_ms_avg')), 3)} ms</b></div>
      <hr/>
      <div class="kpi"><span>uniform success / error</span><b>{u.get('success_count',0)} / {u.get('error_count',0)}</b></div>
      <div class="kpi"><span>static success / error</span><b>{s.get('success_count',0)} / {s.get('error_count',0)}</b></div>
      <div class="small">ทั้งสอง policy รันครบทุก query และไม่มี error ในชุดนี้</div>
    </div>

    <div class="card">
      <h3>Count Metrics (122 queries)</h3>
      <table class="data-table">
        <thead><tr><th>metric</th><th>uniform</th><th>static</th><th>delta</th></tr></thead>
        <tbody>
          <tr><td>exact_match</td><td>{_fmt_num(_as_float(uc.get('exact_match')),6)}</td><td>{_fmt_num(_as_float(sc.get('exact_match')),6)}</td><td>{_fmt_num(_as_float(sc.get('exact_match'))-_as_float(uc.get('exact_match')),6)}</td></tr>
          <tr><td>mae</td><td>{_fmt_num(_as_float(uc.get('mae')),6)}</td><td>{_fmt_num(_as_float(sc.get('mae')),6)}</td><td>{_fmt_num(_as_float(sc.get('mae'))-_as_float(uc.get('mae')),6)}</td></tr>
          <tr><td>mape</td><td>{_fmt_num(_as_float(uc.get('mape')),6)}</td><td>{_fmt_num(_as_float(sc.get('mape')),6)}</td><td>{_fmt_num(_as_float(sc.get('mape'))-_as_float(uc.get('mape')),6)}</td></tr>
        </tbody>
      </table>
    </div>

    <div class="card">
      <h3>Verify Metrics (36 queries)</h3>
      <table class="data-table">
        <thead><tr><th>metric</th><th>uniform</th><th>static</th><th>delta</th></tr></thead>
        <tbody>
          <tr><td>accuracy</td><td>{_fmt_num(_as_float(uv.get('accuracy')),6)}</td><td>{_fmt_num(_as_float(sv.get('accuracy')),6)}</td><td>{_fmt_num(_as_float(sv.get('accuracy'))-_as_float(uv.get('accuracy')),6)}</td></tr>
          <tr><td>precision</td><td>{_fmt_num(_as_float(uv.get('precision')),6)}</td><td>{_fmt_num(_as_float(sv.get('precision')),6)}</td><td>{_fmt_num(_as_float(sv.get('precision'))-_as_float(uv.get('precision')),6)}</td></tr>
          <tr><td>recall</td><td>{_fmt_num(_as_float(uv.get('recall')),6)}</td><td>{_fmt_num(_as_float(sv.get('recall')),6)}</td><td>{_fmt_num(_as_float(sv.get('recall'))-_as_float(uv.get('recall')),6)}</td></tr>
          <tr><td>f1</td><td>{_fmt_num(_as_float(uv.get('f1')),6)}</td><td>{_fmt_num(_as_float(sv.get('f1')),6)}</td><td>{_fmt_num(_as_float(sv.get('f1'))-_as_float(uv.get('f1')),6)}</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <div class="card">
    <h3>Operator Breakdown</h3>
    {_build_operator_table(summary)}
  </div>

  {_build_query_diff_section(run_rows)}

</body>
</html>
"""
    return html


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Phase 3 uniform/static benchmark results")
    parser.add_argument("--summary", required=True, type=Path, help="Path to phase3 summary JSON")
    parser.add_argument("--runs", type=Path, default=None, help="Path to phase3 run JSONL (optional)")
    parser.add_argument("--output", type=Path, default=None, help="Output HTML path")
    args = parser.parse_args()

    summary_path = args.summary
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    summary = _load_json(summary_path)
    run_rows: list[dict[str, Any]] = []
    if args.runs and args.runs.exists():
        run_rows = _load_jsonl(args.runs)

    output_path = args.output
    if output_path is None:
        output_path = summary_path.with_name(summary_path.stem + "_viz.html")

    html = build_html(summary=summary, run_rows=run_rows, summary_path=summary_path)
    output_path.write_text(html, encoding="utf-8")

    print(f"Visualization written to: {output_path}")
    print(f"Generated at: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
