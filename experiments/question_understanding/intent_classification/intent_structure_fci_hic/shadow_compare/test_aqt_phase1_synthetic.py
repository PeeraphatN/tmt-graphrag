#!/usr/bin/env python3
"""
Synthetic regression checks for AQT Phase 1 slot trust policy.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
INTENT_EXPERIMENT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = INTENT_EXPERIMENT_DIR / "data" / "aqt_synthetic_phase1_cases.json"
OUTPUT_DIR = INTENT_EXPERIMENT_DIR / "results" / "synthetic_regression"

sys.path.append(str(PROJECT_ROOT))

from src.services.aqt import transform_query


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


def _slot_map(bundle: dict) -> dict[str, list[str]]:
    mapped: dict[str, list[str]] = {}
    for slot in bundle.get("slots", []) or []:
        name = str(slot.get("name", "")).strip()
        value = str(slot.get("value", "")).strip()
        if not name or not value:
            continue
        mapped.setdefault(name, []).append(value)
    return mapped


def _check_case(case: dict, bundle: dict) -> tuple[bool, list[str], dict[str, list[str]]]:
    checks = dict(case.get("checks", {}) or {})
    slots_by_name = _slot_map(bundle)
    failures: list[str] = []

    for slot_name in checks.get("must_have", []) or []:
        if not slots_by_name.get(slot_name):
            failures.append(f"missing_required_slot:{slot_name}")

    for slot_name in checks.get("must_not_have", []) or []:
        if slots_by_name.get(slot_name):
            failures.append(f"unexpected_slot:{slot_name}")

    min_drug_count = checks.get("min_drug_count")
    if min_drug_count is not None:
        actual_drug_count = len(slots_by_name.get("drug", []))
        if actual_drug_count < int(min_drug_count):
            failures.append(f"drug_count_below_min:{actual_drug_count}<{min_drug_count}")

    return (len(failures) == 0, failures, slots_by_name)


def run() -> Path:
    _configure_stdout_utf8()

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Synthetic dataset not found: {DATASET_PATH}")

    cases = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError("Synthetic dataset must be a JSON list.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_aqt_phase1_synthetic.jsonl"

    pass_count = 0
    fail_count = 0

    print("=" * 80)
    print("AQT Phase 1 Synthetic Regression")
    print("=" * 80)
    print(f"Dataset: {DATASET_PATH} ({len(cases)} cases)")

    with out_path.open("w", encoding="utf-8") as f:
        for idx, case in enumerate(cases, start=1):
            question = str(case.get("question", "")).strip()
            description = str(case.get("description", f"case-{idx}"))
            case_id = str(case.get("id", f"case_{idx:03d}"))
            if not question:
                continue

            query_obj = transform_query(question)
            bundle = query_obj.intent_bundle or {}

            passed, failures, slots_by_name = _check_case(case, bundle)
            status = "PASS" if passed else "FAIL"
            if passed:
                pass_count += 1
            else:
                fail_count += 1

            print(f"\n[{idx:02d}] {case_id} :: {description}")
            print(f"Q: {question}")
            print(f"Status: {status}")
            print(f"Slots: {slots_by_name}")
            if failures:
                print(f"Failures: {failures}")

            record = {
                "timestamp": datetime.now().isoformat(),
                "case_id": case_id,
                "description": description,
                "question": question,
                "status": status,
                "failures": failures,
                "slots": bundle.get("slots", []),
                "slots_multi": (bundle.get("metadata", {}) or {}).get("ner_slots_multi", {}),
                "metadata": bundle.get("metadata", {}),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\n" + "-" * 80)
    print(f"Summary: pass={pass_count} fail={fail_count} total={pass_count + fail_count}")
    print(f"Output: {out_path}")
    print("-" * 80)
    return out_path


if __name__ == "__main__":
    run()
