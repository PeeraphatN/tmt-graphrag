#!/usr/bin/env python3
"""
Phase 1 ground truth builder for retrieval evaluation.

Outputs a single JSON file with:
- stratified node pool across TMT levels
- deterministic id-lookup ground truth cases
- substance relation-based multi-target ground truth sets
- manufacturer relation-based multi-target ground truth sets
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
API_APP_ROOT = REPO_ROOT / "apps" / "api"
sys.path.insert(0, str(API_APP_ROOT))

from src.services.database import init_driver


LEVELS = ("SUBS", "VTM", "GP", "GPU", "TP", "TPU")


@dataclass(frozen=True)
class BuildConfig:
    seed: int
    sample_per_level: int
    id_lookup_count: int
    substance_set_count: int
    manufacturer_set_count: int
    min_gp_per_substance: int
    min_manufacturer_product_count: int
    max_gold_ids_per_set: int


def _configure_stdout_utf8() -> None:
    import sys

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


def _sample_rows(rng: random.Random, rows: list[dict[str, Any]], target_count: int) -> list[dict[str, Any]]:
    if len(rows) <= target_count:
        return rows
    sampled = rng.sample(rows, target_count)
    sampled.sort(key=lambda item: str(item.get("tmtid", "")))
    return sampled


def _fetch_level_counts() -> dict[str, int]:
    drv = init_driver()
    with drv.session() as session:
        rows = session.run(
            """
            MATCH (n:TMT)
            RETURN n.level AS level, count(*) AS c
            ORDER BY level
            """
        ).data()
    return {str(row["level"]): int(row["c"]) for row in rows if row.get("level")}


def _fetch_nodes_for_level(level: str) -> list[dict[str, Any]]:
    drv = init_driver()
    with drv.session() as session:
        rows = session.run(
            """
            MATCH (n:TMT {level: $level})
            RETURN
                n.tmtid AS tmtid,
                n.level AS level,
                n.name AS name,
                n.fsn AS fsn,
                n.trade_name AS trade_name,
                n.generic_name AS generic_name,
                n.manufacturer AS manufacturer,
                n.active_substance AS active_substance,
                n.active_substances AS active_substances,
                n.strength AS strength,
                n.dosageform AS dosageform,
                n.nlem AS nlem,
                n.nlem_category AS nlem_category
            ORDER BY n.tmtid
            """,
            level=level,
        ).data()

    normalized: list[dict[str, Any]] = []
    for row in rows:
        normalized.append(
            {
                "tmtid": _normalize_text(row.get("tmtid")),
                "level": _normalize_text(row.get("level")),
                "name": row.get("name"),
                "fsn": row.get("fsn"),
                "trade_name": row.get("trade_name"),
                "generic_name": row.get("generic_name"),
                "manufacturer": row.get("manufacturer"),
                "active_substance": row.get("active_substance"),
                "active_substances": row.get("active_substances"),
                "strength": row.get("strength"),
                "dosageform": row.get("dosageform"),
                "nlem": row.get("nlem"),
                "nlem_category": row.get("nlem_category"),
            }
        )
    return normalized


def _fetch_substance_sets(config: BuildConfig, rng: random.Random) -> list[dict[str, Any]]:
    drv = init_driver()
    with drv.session() as session:
        rows = session.run(
            """
            MATCH (subs:TMT {level:'SUBS'})-[:IS_ACTIVE_SUBSTANCE_OF]->(v:TMT {level:'VTM'})<-[:HAS_VTM]-(gp:TMT {level:'GP'})
            OPTIONAL MATCH (gp)-[:HAS_GENERIC_UNIT]->(gpu:TMT {level:'GPU'})
            OPTIONAL MATCH (gp)-[:HAS_TRADE_PRODUCT]->(tp:TMT {level:'TP'})
            OPTIONAL MATCH (tp)-[:HAS_TRADE_UNIT]->(tpu:TMT {level:'TPU'})
            WITH subs,
                 count(DISTINCT v) AS vtm_count,
                 count(DISTINCT gp) AS gp_count,
                 count(DISTINCT gpu) AS gpu_count,
                 count(DISTINCT tp) AS tp_count,
                 count(DISTINCT tpu) AS tpu_count,
                 collect(DISTINCT v.tmtid)[..$max_ids] AS vtm_ids,
                 collect(DISTINCT gp.tmtid)[..$max_ids] AS gp_ids,
                 collect(DISTINCT gpu.tmtid)[..$max_ids] AS gpu_ids,
                 collect(DISTINCT tp.tmtid)[..$max_ids] AS tp_ids,
                 collect(DISTINCT tpu.tmtid)[..$max_ids] AS tpu_ids
            WHERE gp_count >= $min_gp
            RETURN
                subs.tmtid AS subs_tmtid,
                subs.name AS substance_name,
                vtm_count,
                gp_count,
                gpu_count,
                tp_count,
                tpu_count,
                vtm_ids,
                gp_ids,
                gpu_ids,
                tp_ids,
                tpu_ids
            ORDER BY gp_count DESC, tp_count DESC, subs_tmtid
            """,
            max_ids=config.max_gold_ids_per_set,
            min_gp=config.min_gp_per_substance,
        ).data()

    if len(rows) > config.substance_set_count:
        rows = rng.sample(rows, config.substance_set_count)

    rows.sort(key=lambda item: str(item.get("subs_tmtid", "")))
    output: list[dict[str, Any]] = []
    for row in rows:
        output.append(
            {
                "set_type": "substance_descendants",
                "anchor": {
                    "subs_tmtid": _normalize_text(row.get("subs_tmtid")),
                    "substance_name": _normalize_text(row.get("substance_name")),
                },
                "counts": {
                    "vtm": int(row.get("vtm_count", 0) or 0),
                    "gp": int(row.get("gp_count", 0) or 0),
                    "gpu": int(row.get("gpu_count", 0) or 0),
                    "tp": int(row.get("tp_count", 0) or 0),
                    "tpu": int(row.get("tpu_count", 0) or 0),
                },
                "gold_tmtids": {
                    "vtm": list(row.get("vtm_ids") or []),
                    "gp": list(row.get("gp_ids") or []),
                    "gpu": list(row.get("gpu_ids") or []),
                    "tp": list(row.get("tp_ids") or []),
                    "tpu": list(row.get("tpu_ids") or []),
                },
            }
        )
    return output


def _fetch_manufacturer_sets(config: BuildConfig, rng: random.Random) -> list[dict[str, Any]]:
    drv = init_driver()
    with drv.session() as session:
        rows = session.run(
            """
            MATCH (tp:TMT {level:'TP'})
            WHERE tp.manufacturer IS NOT NULL
              AND trim(tp.manufacturer) <> ''
            OPTIONAL MATCH (tp)-[:HAS_TRADE_UNIT]->(tpu:TMT {level:'TPU'})
            WITH toLower(trim(tp.manufacturer)) AS manufacturer_norm,
                 count(DISTINCT tp) AS tp_count,
                 count(DISTINCT tpu) AS tpu_count,
                 collect(DISTINCT tp.tmtid)[..$max_ids] AS tp_ids,
                 collect(DISTINCT tpu.tmtid)[..$max_ids] AS tpu_ids
            WITH manufacturer_norm,
                 tp_count,
                 tpu_count,
                 tp_ids,
                 tpu_ids,
                 tp_count + tpu_count AS total_products
            WHERE total_products >= $min_products
            RETURN
                manufacturer_norm,
                tp_count,
                tpu_count,
                total_products,
                tp_ids,
                tpu_ids
            ORDER BY total_products DESC, manufacturer_norm
            """,
            min_products=config.min_manufacturer_product_count,
            max_ids=config.max_gold_ids_per_set,
        ).data()

    if len(rows) > config.manufacturer_set_count:
        rows = rng.sample(rows, config.manufacturer_set_count)

    rows.sort(key=lambda item: str(item.get("manufacturer_norm", "")))
    output: list[dict[str, Any]] = []
    for row in rows:
        output.append(
            {
                "set_type": "manufacturer_products",
                "anchor": {
                    "manufacturer_norm": _normalize_text(row.get("manufacturer_norm")),
                },
                "counts": {
                    "tp": int(row.get("tp_count", 0) or 0),
                    "tpu": int(row.get("tpu_count", 0) or 0),
                    "total_products": int(row.get("total_products", 0) or 0),
                },
                "gold_tmtids": {
                    "tp": list(row.get("tp_ids") or []),
                    "tpu": list(row.get("tpu_ids") or []),
                },
            }
        )
    return output


def build_ground_truth(config: BuildConfig) -> dict[str, Any]:
    rng = random.Random(config.seed)
    level_counts = _fetch_level_counts()

    selected_nodes: list[dict[str, Any]] = []
    selected_per_level: dict[str, int] = {}
    for level in LEVELS:
        level_nodes = _fetch_nodes_for_level(level)
        sampled_nodes = _sample_rows(rng, level_nodes, config.sample_per_level)
        selected_nodes.extend(sampled_nodes)
        selected_per_level[level] = len(sampled_nodes)

    id_lookup_nodes = _sample_rows(rng, selected_nodes, config.id_lookup_count)
    id_lookup_cases = [
        {
            "tmtid": row["tmtid"],
            "level": row["level"],
            "name": row.get("trade_name") or row.get("fsn") or row.get("name"),
        }
        for row in id_lookup_nodes
    ]

    substance_sets = _fetch_substance_sets(config, rng)
    manufacturer_sets = _fetch_manufacturer_sets(config, rng)

    return {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "phase": "phase1_ground_truth",
            "seed": config.seed,
            "levels": list(LEVELS),
            "script": "experiments/retrieval/retrieval_eval/run/build_phase1_ground_truth.py",
        },
        "sampling": {
            "sample_per_level": config.sample_per_level,
            "id_lookup_count": config.id_lookup_count,
            "substance_set_count": config.substance_set_count,
            "manufacturer_set_count": config.manufacturer_set_count,
            "min_gp_per_substance": config.min_gp_per_substance,
            "min_manufacturer_product_count": config.min_manufacturer_product_count,
            "max_gold_ids_per_set": config.max_gold_ids_per_set,
        },
        "counts": {
            "db_level_counts": level_counts,
            "selected_node_counts": selected_per_level,
            "selected_node_total": len(selected_nodes),
            "id_lookup_case_count": len(id_lookup_cases),
            "substance_set_count": len(substance_sets),
            "manufacturer_set_count": len(manufacturer_sets),
        },
        "node_pool": selected_nodes,
        "id_lookup_ground_truth": id_lookup_cases,
        "substance_ground_truth_sets": substance_sets,
        "manufacturer_ground_truth_sets": manufacturer_sets,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build phase-1 ground truth JSON from Neo4j.")
    parser.add_argument(
        "--output",
        default="experiments/retrieval/retrieval_eval/data/phase1_ground_truth.json",
        help="Output JSON path.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--sample-per-level", type=int, default=200, help="Node sample per level.")
    parser.add_argument("--id-lookup-count", type=int, default=120, help="ID lookup case count.")
    parser.add_argument("--substance-set-count", type=int, default=80, help="Substance relation set count.")
    parser.add_argument("--manufacturer-set-count", type=int, default=80, help="Manufacturer relation set count.")
    parser.add_argument("--min-gp-per-substance", type=int, default=5, help="Minimum GP count for substance sets.")
    parser.add_argument("--min-manufacturer-product-count", type=int, default=20, help="Minimum product count for manufacturer sets.")
    parser.add_argument("--max-gold-ids-per-set", type=int, default=300, help="Maximum ID list size per set.")
    return parser.parse_args()


def main() -> None:
    _configure_stdout_utf8()
    args = parse_args()
    config = BuildConfig(
        seed=args.seed,
        sample_per_level=max(1, args.sample_per_level),
        id_lookup_count=max(1, args.id_lookup_count),
        substance_set_count=max(1, args.substance_set_count),
        manufacturer_set_count=max(1, args.manufacturer_set_count),
        min_gp_per_substance=max(1, args.min_gp_per_substance),
        min_manufacturer_product_count=max(1, args.min_manufacturer_product_count),
        max_gold_ids_per_set=max(20, args.max_gold_ids_per_set),
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = build_ground_truth(config)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Ground truth generated.")
    print(f"Output: {output_path}")
    print(json.dumps(payload["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
