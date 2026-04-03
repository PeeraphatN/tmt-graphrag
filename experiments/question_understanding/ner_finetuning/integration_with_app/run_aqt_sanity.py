from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
API_APP_ROOT = PROJECT_ROOT / "apps" / "api"
INTEGRATION_DIR = Path(__file__).resolve().parent
NER_EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = NER_EXPERIMENT_DIR / "artifacts" / "generated_records.jsonl"
RESULTS_DIR = INTEGRATION_DIR / "results"

if str(API_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(API_APP_ROOT))

from src.services.aqt import transform_query


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Generated records not found: {path}. Run generate_ner_data.py first to create artifacts/generated_records.jsonl"
        )

    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def run_aqt_sanity(records: list[dict], sample_size: int, seed: int) -> dict:
    rng = random.Random(seed)
    sample = records if len(records) <= sample_size else rng.sample(records, sample_size)

    evaluated = 0
    non_empty_query = 0
    errors = 0

    for record in sample:
        try:
            query_obj = transform_query(record["text"])
            evaluated += 1
            if str(getattr(query_obj, "query", "")).strip():
                non_empty_query += 1
        except Exception:
            errors += 1

    query_non_empty_rate = (non_empty_query / evaluated) if evaluated else 0.0

    nlem_candidates = [record for record in records if record.get("group") == "nlem.verify"]
    nlem_sample_size = min(sample_size, len(nlem_candidates))
    nlem_sample = nlem_candidates if len(nlem_candidates) <= nlem_sample_size else rng.sample(nlem_candidates, nlem_sample_size)

    nlem_evaluated = 0
    nlem_true = 0
    nlem_errors = 0
    for record in nlem_sample:
        try:
            query_obj = transform_query(record["text"])
            nlem_evaluated += 1
            if getattr(query_obj, "nlem_filter", None) is True:
                nlem_true += 1
        except Exception:
            nlem_errors += 1

    nlem_true_rate = (nlem_true / nlem_evaluated) if nlem_evaluated else 0.0

    return {
        "input_records": len(records),
        "query_non_empty": {
            "passed": query_non_empty_rate >= 0.90,
            "rate": query_non_empty_rate,
            "threshold": 0.90,
            "evaluated": evaluated,
            "errors": errors,
        },
        "nlem_filter_true": {
            "passed": nlem_true_rate >= 0.85,
            "rate": nlem_true_rate,
            "threshold": 0.85,
            "evaluated": nlem_evaluated,
            "errors": nlem_errors,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AQT sanity checks against generated NER-style questions.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Path to generated_records.jsonl")
    parser.add_argument("--sample-size", type=int, default=500, help="Maximum number of generated records to sample")
    parser.add_argument("--seed", type=int, default=20260209, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(args.input)
    report = run_aqt_sanity(records=records, sample_size=args.sample_size, seed=args.seed)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"aqt_sanity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Input: {args.input}")
    print(f"Records: {report['input_records']}")
    print(f"Query non-empty rate: {report['query_non_empty']['rate']:.3f}")
    print(f"NLEM true rate: {report['nlem_filter_true']['rate']:.3f}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
