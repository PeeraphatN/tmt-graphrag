import argparse
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from pythainlp.tokenize import word_tokenize as thai_word_tokenize
except Exception:
    thai_word_tokenize = None


RUN_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = RUN_DIR.parent
DATA_DIR = EXPERIMENT_DIR / "data"
ARTIFACTS_DIR = EXPERIMENT_DIR / "artifacts"
FILES = {
    "DRUG": DATA_DIR / "entities_drug.json",
    "BRAND": DATA_DIR / "entities_brand.json",
    "MANUFACTURER": DATA_DIR / "entities_manufacturer.json",
    "FORM": DATA_DIR / "entities_form.json",
    "STRENGTH": DATA_DIR / "entities_strength.json",
}

LABEL_LIST = [
    "O",
    "B-DRUG",
    "I-DRUG",
    "B-BRAND",
    "I-BRAND",
    "B-MANUFACTURER",
    "I-MANUFACTURER",
    "B-FORM",
    "I-FORM",
    "B-STRENGTH",
    "I-STRENGTH",
]

ENTITY_RULES = {
    "DRUG": {"min_len": 2, "max_len": 80},
    "BRAND": {"min_len": 2, "max_len": 80},
    "MANUFACTURER": {"min_len": 2, "max_len": 60},
    "FORM": {"min_len": 2, "max_len": 40},
    "STRENGTH": {"min_len": 2, "max_len": 40, "max_plus": 1},
}

BLACKLIST_KEYWORDS = [
    "ราคา",
    "ผลข้างเคียง",
    "วิธีใช้",
    "ข้อห้าม",
    "ปฏิกิริยาระหว่างยา",
]

TEMPLATE_GROUP_WEIGHTS = {
    "general.retrieve": 0.20,
    "manufacturer.mix": 0.20,
    "ingredient.retrieve": 0.15,
    "formula.mix": 0.15,
    "nlem.verify": 0.15,
    "hierarchy.retrieve": 0.05,
    "short.practical": 0.10,
}

TEMPLATES = [
    {"id": "G01", "group": "general.retrieve", "target_type": "general", "strategy": "retrieve", "profile": "practical.thai_first", "weight": 1.0, "text": "ข้อมูลยา {DRUG}"},
    {"id": "G02", "group": "general.retrieve", "target_type": "general", "strategy": "retrieve", "profile": "practical.thai_first", "weight": 1.0, "text": "ขอข้อมูลของ {BRAND}"},
    {"id": "G03", "group": "general.retrieve", "target_type": "general", "strategy": "retrieve", "profile": "practical.thai_first", "weight": 1.0, "text": "{DRUG} คือยาอะไร"},
    {"id": "G04", "group": "general.retrieve", "target_type": "general", "strategy": "retrieve", "profile": "practical.thai_first", "weight": 1.0, "text": "{BRAND} คือยาอะไร"},
    {"id": "G05", "group": "general.retrieve", "target_type": "general", "strategy": "retrieve", "profile": "practical.thai_first", "weight": 1.0, "text": "รายละเอียดของ {DRUG} คืออะไร"},

    {"id": "M01", "group": "manufacturer.mix", "target_type": "manufacturer", "strategy": "retrieve", "profile": "practical.thai_first", "weight": 1.0, "text": "{BRAND} ผลิตโดยบริษัทอะไร"},
    {"id": "M02", "group": "manufacturer.mix", "target_type": "manufacturer", "strategy": "retrieve", "profile": "practical.thai_first", "weight": 1.0, "text": "ผู้ผลิตของ {BRAND} คือใคร"},
    {"id": "M03", "group": "manufacturer.mix", "target_type": "manufacturer", "strategy": "verify", "profile": "practical.thai_first", "weight": 1.0, "text": "ยา {DRUG} ของ {MANUFACTURER} มีไหม"},
    {"id": "M04", "group": "manufacturer.mix", "target_type": "manufacturer", "strategy": "verify", "profile": "practical.thai_first", "weight": 1.0, "text": "{BRAND} เป็นของ {MANUFACTURER} หรือไม่"},
    {"id": "M05", "group": "manufacturer.mix", "target_type": "manufacturer", "strategy": "verify", "profile": "practical.thai_first", "weight": 1.0, "text": "{DRUG} ผลิตโดย {MANUFACTURER} ใช่ไหม"},

    {"id": "I01", "group": "ingredient.retrieve", "target_type": "ingredient", "strategy": "retrieve", "profile": "practical.thai_first", "weight": 1.0, "text": "{BRAND} มีตัวยาสำคัญอะไร"},
    {"id": "I02", "group": "ingredient.retrieve", "target_type": "ingredient", "strategy": "retrieve", "profile": "practical.thai_first", "weight": 1.0, "text": "ส่วนประกอบของ {DRUG} คืออะไร"},
    {"id": "I03", "group": "ingredient.retrieve", "target_type": "ingredient", "strategy": "retrieve", "profile": "practical.thai_first", "weight": 1.0, "text": "{DRUG} มี active ingredient อะไร"},
    {"id": "I04", "group": "ingredient.retrieve", "target_type": "ingredient", "strategy": "retrieve", "profile": "practical.thai_first", "weight": 1.0, "text": "{BRAND} {STRENGTH} มีสารสำคัญอะไร"},

    {"id": "F01", "group": "formula.mix", "target_type": "formula", "strategy": "retrieve", "profile": "practical.thai_first", "weight": 1.0, "text": "{BRAND} มีรูปแบบยาอะไร"},
    {"id": "F02", "group": "formula.mix", "target_type": "formula", "strategy": "verify", "profile": "practical.thai_first", "weight": 1.0, "text": "{DRUG} มีรูปแบบ {FORM} ไหม"},
    {"id": "F03", "group": "formula.mix", "target_type": "formula", "strategy": "verify", "profile": "practical.thai_first", "weight": 1.0, "text": "{BRAND} เป็น {FORM} หรือไม่"},
    {"id": "F04", "group": "formula.mix", "target_type": "formula", "strategy": "verify", "profile": "practical.thai_first", "weight": 1.0, "text": "{DRUG} {STRENGTH} มีในระบบไหม"},

    {"id": "N01", "group": "nlem.verify", "target_type": "nlem", "strategy": "verify", "profile": "practical.thai_first", "weight": 1.0, "text": "{BRAND} เบิกได้ไหม"},
    {"id": "N02", "group": "nlem.verify", "target_type": "nlem", "strategy": "verify", "profile": "practical.thai_first", "weight": 1.0, "text": "ยา {DRUG} อยู่ในบัญชียาหลักหรือไม่"},
    {"id": "N03", "group": "nlem.verify", "target_type": "nlem", "strategy": "verify", "profile": "practical.thai_first", "weight": 1.0, "text": "{DRUG} {STRENGTH} เบิกได้หรือไม่"},
    {"id": "N04", "group": "nlem.verify", "target_type": "nlem", "strategy": "verify", "profile": "practical.thai_first", "weight": 1.0, "text": "{BRAND} อยู่ใน NLEM ไหม"},
    {"id": "N05", "group": "nlem.verify", "target_type": "nlem", "strategy": "verify", "profile": "practical.thai_first", "weight": 1.0, "text": "{DRUG} อยู่ในบัญชียาหลักหมวด ก หรือไม่"},

    {"id": "H01", "group": "hierarchy.retrieve", "target_type": "hierarchy", "strategy": "retrieve", "profile": "practical.thai_first", "weight": 1.0, "text": "{BRAND} เป็นยาสามัญตัวไหน"},
    {"id": "H02", "group": "hierarchy.retrieve", "target_type": "hierarchy", "strategy": "retrieve", "profile": "practical.thai_first", "weight": 1.0, "text": "{DRUG} มีชื่อการค้าอะไร"},
    {"id": "H03", "group": "hierarchy.retrieve", "target_type": "hierarchy", "strategy": "retrieve", "profile": "practical.thai_first", "weight": 1.0, "text": "{BRAND} อยู่ในกลุ่มยาอะไร"},

    {"id": "S01", "group": "short.practical", "target_type": "general", "strategy": "retrieve", "profile": "practical.thai_first", "weight": 1.0, "text": "{BRAND}"},
    {"id": "S02", "group": "short.practical", "target_type": "general", "strategy": "retrieve", "profile": "practical.thai_first", "weight": 1.0, "text": "{DRUG}"},
    {"id": "S03", "group": "short.practical", "target_type": "nlem", "strategy": "verify", "profile": "practical.thai_first", "weight": 1.0, "text": "{BRAND} เบิกได้ไหม"},
    {"id": "S04", "group": "short.practical", "target_type": "manufacturer", "strategy": "retrieve", "profile": "practical.thai_first", "weight": 1.0, "text": "{DRUG} ของ {MANUFACTURER}"},
]

SLOT_PATTERN = re.compile(r"\{(\w+)\}")
TOKENIZER_ENGINE = "pythainlp_newmm" if thai_word_tokenize else "regex_fallback"

GROUP_REQUIRED_KEYWORDS = {
    "nlem.verify": ["nlem", "เบิก", "บัญชียาหลัก"],
    "manufacturer.mix": ["ผู้ผลิต", "บริษัท", "ผลิตโดย"],
}


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def load_entities() -> dict[str, list[str]]:
    entities: dict[str, list[str]] = {}
    for label, path in FILES.items():
        with open(path, "r", encoding="utf-8") as f:
            entities[label] = json.load(f)
    return entities


def filter_entities(raw_entities: dict[str, list[str]]) -> dict[str, list[str]]:
    pools: dict[str, list[str]] = {}
    for label, values in raw_entities.items():
        rule = ENTITY_RULES[label]
        seen = set()
        filtered: list[str] = []

        for value in values:
            normalized = normalize_text(value)
            if not normalized:
                continue

            if len(normalized) < rule["min_len"] or len(normalized) > rule["max_len"]:
                continue

            if "max_plus" in rule and normalized.count("+") > rule["max_plus"]:
                continue

            if normalized in seen:
                continue

            seen.add(normalized)
            filtered.append(normalized)

        pools[label] = filtered

    return pools


def validate_template_catalog() -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    required_fields = {"id", "group", "target_type", "strategy", "profile", "weight", "text"}
    allowed_target_types = {"general", "manufacturer", "ingredient", "formula", "nlem", "hierarchy"}
    allowed_strategies = {"retrieve", "verify"}

    for spec in TEMPLATES:
        missing = required_fields - set(spec.keys())
        if missing:
            raise ValueError(f"Template {spec.get('id')} missing fields: {sorted(missing)}")

        if spec["group"] not in TEMPLATE_GROUP_WEIGHTS:
            raise ValueError(f"Unknown group in template {spec['id']}: {spec['group']}")
        if spec["target_type"] not in allowed_target_types:
            raise ValueError(f"Unknown target_type in template {spec['id']}: {spec['target_type']}")
        if spec["strategy"] not in allowed_strategies:
            raise ValueError(f"Unknown strategy in template {spec['id']}: {spec['strategy']}")
        if not isinstance(spec["weight"], (int, float)) or spec["weight"] <= 0:
            raise ValueError(f"Invalid weight in template {spec['id']}: {spec['weight']}")

        grouped[spec["group"]].append(spec)

    missing_groups = [g for g in TEMPLATE_GROUP_WEIGHTS if g not in grouped]
    if missing_groups:
        raise ValueError(f"No templates configured for groups: {missing_groups}")

    return grouped


def choose_template(rng: random.Random, grouped_templates: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    groups = list(TEMPLATE_GROUP_WEIGHTS.keys())
    group_weights = [TEMPLATE_GROUP_WEIGHTS[g] for g in groups]
    group = rng.choices(groups, weights=group_weights, k=1)[0]

    candidates = grouped_templates[group]
    template_weights = [float(item["weight"]) for item in candidates]
    return rng.choices(candidates, weights=template_weights, k=1)[0]


def render_template(
    template_spec: dict[str, Any],
    entity_pools: dict[str, list[str]],
    rng: random.Random,
) -> tuple[str | None, list[tuple[int, int, str]], str | None]:
    template_text = template_spec["text"]
    slots = SLOT_PATTERN.findall(template_text)
    if not slots:
        return None, [], "no_slots"

    fill_values: dict[str, str] = {}
    for slot in slots:
        candidates = entity_pools.get(slot, [])
        if not candidates:
            return None, [], "slot_pool_empty"
        if slot not in fill_values:
            fill_values[slot] = rng.choice(candidates)

    parts = re.split(r"(\{\w+\})", template_text)
    full_text = ""
    spans: list[tuple[int, int, str]] = []
    cursor = 0

    for part in parts:
        if part.startswith("{") and part.endswith("}"):
            slot_name = part[1:-1]
            value = fill_values.get(slot_name)
            if not value:
                return None, [], "missing_slot_value"
            start = cursor
            end = start + len(value)
            spans.append((start, end, slot_name))
            full_text += value
            cursor = end
        else:
            full_text += part
            cursor += len(part)

    if not spans:
        return None, [], "no_entity_span"

    return full_text, spans, None


def is_graph_answerable(text: str, template_spec: dict[str, Any], has_entity: bool) -> tuple[bool, str | None]:
    lowered = text.lower()

    for keyword in BLACKLIST_KEYWORDS:
        if keyword in lowered:
            return False, "blacklist_keyword"

    if not has_entity:
        return False, "no_entity_slot"

    required_keywords = GROUP_REQUIRED_KEYWORDS.get(template_spec["group"], [])
    if required_keywords and not any(keyword in lowered for keyword in required_keywords):
        if template_spec["group"] == "manufacturer.mix":
            # Allow verify phrasing such as "{DRUG} ของ {MANUFACTURER} มีไหม".
            if template_spec.get("strategy") == "verify" and "ของ" in lowered:
                return True, None
            return False, "missing_manufacturer_keyword"
        if template_spec["group"] == "nlem.verify":
            return False, "missing_nlem_keyword"
        return False, "missing_group_keyword"

    return True, None


def tokenize_and_align(text: str, spans: list[tuple[int, int, str]]) -> tuple[list[str] | None, list[str] | None, str | None]:
    try:
        if thai_word_tokenize:
            tokens = thai_word_tokenize(text, engine="newmm")
        else:
            tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    except Exception:
        return None, None, "tokenization_error"

    if not tokens:
        return None, None, "tokenization_empty"

    token_spans: list[tuple[int, int]] = []
    cursor = 0
    for token in tokens:
        index = text.find(token, cursor)
        if index < 0:
            index = text.find(token)
            if index < 0:
                return None, None, "token_not_found"
        token_spans.append((index, index + len(token)))
        cursor = index + len(token)

    ner_tags = ["O"] * len(tokens)
    for start, end, label in spans:
        overlap_indices = []
        for i, (token_start, token_end) in enumerate(token_spans):
            if max(start, token_start) < min(end, token_end):
                overlap_indices.append(i)

        if not overlap_indices:
            return None, None, "span_token_miss"

        for index_position, token_index in enumerate(overlap_indices):
            if ner_tags[token_index] != "O":
                return None, None, "tag_collision"
            prefix = "B" if index_position == 0 else "I"
            ner_tags[token_index] = f"{prefix}-{label}"

    return tokens, ner_tags, None


def validate_iob_sequence(tags: list[str]) -> bool:
    previous_label = None
    for tag in tags:
        if tag == "O":
            previous_label = None
            continue

        if "-" not in tag:
            return False

        prefix, label = tag.split("-", 1)
        if prefix == "B":
            previous_label = label
            continue

        if prefix == "I":
            if previous_label != label:
                return False
            continue

        return False

    return True


def generate_record(
    template_spec: dict[str, Any],
    entity_pools: dict[str, list[str]],
    rng: random.Random,
) -> tuple[dict[str, Any] | None, str | None]:
    text, spans, render_error = render_template(template_spec, entity_pools, rng)
    if render_error:
        return None, render_error

    answerable, gate_error = is_graph_answerable(text, template_spec, has_entity=bool(spans))
    if not answerable:
        return None, gate_error

    tokens, ner_tags, align_error = tokenize_and_align(text, spans)
    if align_error:
        return None, align_error

    item = {
        "id": str(rng.getrandbits(64)),
        "tokens": tokens,
        "ner_tags": ner_tags,
    }
    return {
        "item": item,
        "text": text,
        "template_id": template_spec["id"],
        "group": template_spec["group"],
        "target_type": template_spec["target_type"],
        "strategy": template_spec["strategy"],
        "profile": template_spec["profile"],
    }, None


def generate_dataset(
    num_samples: int,
    entity_pools: dict[str, list[str]],
    grouped_templates: dict[str, list[dict[str, Any]]],
    seed: int,
    max_attempts: int,
) -> tuple[list[dict[str, Any]], Counter, int]:
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    discard_reasons: Counter = Counter()

    attempts = 0
    while len(records) < num_samples and attempts < max_attempts:
        attempts += 1
        template_spec = choose_template(rng, grouped_templates)
        record, error = generate_record(template_spec, entity_pools, rng)
        if record:
            records.append(record)
        else:
            discard_reasons[error or "unknown_error"] += 1

    return records, discard_reasons, attempts


def split_records(records: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    train_end = int(total * 0.8)
    val_end = int(total * 0.9)
    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


def save_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=None)


def save_manifest(train: list[dict[str, Any]], val: list[dict[str, Any]], test: list[dict[str, Any]]) -> None:
    manifest = {
        "label_list": LABEL_LIST,
        "num_train": len(train),
        "num_val": len(val),
        "num_test": len(test),
    }
    with open(DATA_DIR / "dataset_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def compute_distribution(records: list[dict[str, Any]], key: str) -> dict[str, dict[str, float]]:
    total = len(records)
    counts = Counter(record[key] for record in records)
    all_keys = sorted(counts.keys())

    return {
        value: {
            "count": counts[value],
            "ratio": (counts[value] / total) if total else 0.0,
        }
        for value in all_keys
    }


def compute_label_distribution(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    counter = Counter()
    total = 0
    for record in records:
        for tag in record["item"]["ner_tags"]:
            counter[tag] += 1
            total += 1

    labels = list(LABEL_LIST)
    labels.extend(sorted(tag for tag in counter if tag not in LABEL_LIST))

    result: dict[str, dict[str, float]] = {}
    for label in labels:
        count = counter.get(label, 0)
        result[label] = {
            "count": count,
            "ratio": (count / total) if total else 0.0,
        }
    return result


def run_template_smoke_test(
    entity_pools: dict[str, list[str]],
    target_per_template: int,
    max_attempts_per_template: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    details = {}
    all_passed = True

    for template_spec in TEMPLATES:
        success = 0
        attempts = 0
        reason_counter = Counter()

        while success < target_per_template and attempts < max_attempts_per_template:
            attempts += 1
            _, error = generate_record(template_spec, entity_pools, rng)
            if error:
                reason_counter[error] += 1
                continue
            success += 1

        passed = success >= target_per_template
        all_passed = all_passed and passed
        details[template_spec["id"]] = {
            "passed": passed,
            "generated": success,
            "required": target_per_template,
            "attempts": attempts,
            "discard_reasons": dict(reason_counter),
        }

    return {
        "passed": all_passed,
        "required_per_template": target_per_template,
        "details": details,
    }


def run_label_integrity_test(records: list[dict[str, Any]]) -> dict[str, Any]:
    length_mismatch = 0
    invalid_iob = 0
    checked = 0

    for record in records:
        tokens = record["item"]["tokens"]
        tags = record["item"]["ner_tags"]
        checked += 1

        if len(tokens) != len(tags):
            length_mismatch += 1
            continue

        if not validate_iob_sequence(tags):
            invalid_iob += 1

    return {
        "passed": (length_mismatch == 0 and invalid_iob == 0),
        "checked": checked,
        "length_mismatch": length_mismatch,
        "invalid_iob": invalid_iob,
    }


def run_graph_answerable_gate_test(records: list[dict[str, Any]]) -> dict[str, Any]:
    failures = 0
    failure_reasons = Counter()

    for record in records:
        has_entity = any(tag != "O" for tag in record["item"]["ner_tags"])
        ok, reason = is_graph_answerable(
            text=record["text"],
            template_spec={
                "group": record["group"],
                "strategy": record["strategy"],
            },
            has_entity=has_entity,
        )
        if not ok:
            failures += 1
            failure_reasons[reason or "unknown_error"] += 1

    return {
        "passed": failures == 0,
        "checked": len(records),
        "failed": failures,
        "failure_reasons": dict(failure_reasons),
    }


def run_distribution_test(records: list[dict[str, Any]], tolerance: float) -> dict[str, Any]:
    group_counts = Counter(record["group"] for record in records)
    total = len(records)

    details = {}
    passed = True
    for group, expected_ratio in TEMPLATE_GROUP_WEIGHTS.items():
        actual_ratio = (group_counts[group] / total) if total else 0.0
        delta = abs(actual_ratio - expected_ratio)
        within_tolerance = delta <= tolerance
        if not within_tolerance:
            passed = False

        details[group] = {
            "expected_ratio": expected_ratio,
            "actual_ratio": actual_ratio,
            "delta": delta,
            "within_tolerance": within_tolerance,
        }

    return {
        "passed": passed,
        "tolerance": tolerance,
        "details": details,
    }

def save_generated_records_preview(records: list[dict[str, Any]]) -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = ARTIFACTS_DIR / "generated_records.jsonl"
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            payload = {
                "id": record["item"]["id"],
                "text": record["text"],
                "group": record["group"],
                "target_type": record["target_type"],
                "strategy": record["strategy"],
                "template_id": record["template_id"],
                "profile": record["profile"],
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return output_path
def build_generation_report(
    records: list[dict[str, Any]],
    discard_reasons: Counter,
    attempts: int,
    requested_samples: int,
    checks: dict[str, Any],
) -> dict[str, Any]:
    num_generated = len(records)
    num_discarded = int(sum(discard_reasons.values()))
    all_passed = all(check.get("passed", True) for check in checks.values() if isinstance(check, dict) and "passed" in check)

    return {
        "generated_at": int(time.time()),
        "tokenizer_engine": TOKENIZER_ENGINE,
        "requested_samples": requested_samples,
        "num_generated": num_generated,
        "num_discarded": num_discarded,
        "num_attempts": attempts,
        "discard_reasons": dict(discard_reasons),
        "distribution": {
            "group": compute_distribution(records, "group"),
            "target_type": compute_distribution(records, "target_type"),
            "strategy": compute_distribution(records, "strategy"),
            "template_id": compute_distribution(records, "template_id"),
            "label": compute_label_distribution(records),
        },
        "quality_checks": checks,
        "all_quality_checks_passed": all_passed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate NER dataset from graph-answerable Thai templates.")
    parser.add_argument("--num-samples", type=int, default=10000, help="Target number of generated samples.")
    parser.add_argument("--seed", type=int, default=20260209, help="Random seed.")
    parser.add_argument("--max-attempts", type=int, default=300000, help="Max generation attempts before stopping.")
    parser.add_argument("--smoke-per-template", type=int, default=20, help="Required successful samples per template in smoke test.")
    parser.add_argument("--smoke-max-attempts", type=int, default=400, help="Max attempts per template in smoke test.")
    parser.add_argument("--distribution-tolerance", type=float, default=0.03, help="Allowed ratio deviation for distribution test.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grouped_templates = validate_template_catalog()

    print("Generating NER dataset...")
    if not thai_word_tokenize:
        print("Warning: pythainlp is unavailable. Using regex fallback tokenizer.")
    raw_entities = load_entities()
    entity_pools = filter_entities(raw_entities)

    for label, pool in entity_pools.items():
        print(f"  {label}: {len(pool)} candidates after filtering")

    records, discard_reasons, attempts = generate_dataset(
        num_samples=args.num_samples,
        entity_pools=entity_pools,
        grouped_templates=grouped_templates,
        seed=args.seed,
        max_attempts=args.max_attempts,
    )

    if len(records) < args.num_samples:
        print(
            f"Warning: generated {len(records)} samples (< {args.num_samples}) "
            f"within max attempts={args.max_attempts}."
        )

    train_records, val_records, test_records = split_records(records, seed=args.seed + 1)
    train_data = [record["item"] for record in train_records]
    val_data = [record["item"] for record in val_records]
    test_data = [record["item"] for record in test_records]

    save_json(DATA_DIR / "train.json", train_data)
    save_json(DATA_DIR / "validation.json", val_data)
    save_json(DATA_DIR / "test.json", test_data)
    save_manifest(train_data, val_data, test_data)

    checks = {
        "template_smoke_test": run_template_smoke_test(
            entity_pools=entity_pools,
            target_per_template=args.smoke_per_template,
            max_attempts_per_template=args.smoke_max_attempts,
            seed=args.seed + 2,
        ),
        "label_integrity_test": run_label_integrity_test(records),
        "graph_answerable_gate_test": run_graph_answerable_gate_test(records),
        "distribution_test": run_distribution_test(records, tolerance=args.distribution_tolerance),
        "aqt_sanity_test": {
            "enabled": False,
            "skipped": True,
            "reason": "moved_to_integration_with_app/run_aqt_sanity.py",
        },
    }
    report = build_generation_report(
        records=records,
        discard_reasons=discard_reasons,
        attempts=attempts,
        requested_samples=args.num_samples,
        checks=checks,
    )
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS_DIR / "generation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Done. Generated: {len(records)}")
    print(f"Train: {len(train_data)} | Validation: {len(val_data)} | Test: {len(test_data)}")
    print(f"Discarded: {sum(discard_reasons.values())} | Attempts: {attempts}")
    preview_path = save_generated_records_preview(records)

    print(f"Report: {ARTIFACTS_DIR / 'generation_report.json'}")
    print(f"Preview: {preview_path}")


if __name__ == "__main__":
    main()



