"""
Experimental benchmark: Legacy fine-intent centroid vs V2 split-head centroid.

Legacy:
- Single centroid classifier over 18 fine labels
- Parse predicted fine label -> action + topics

V2:
- Action centroid head (lookup/verify/count)
- Topics centroid head (manufacturer/ingredient/nlem/hierarchy/formula/general)

Goal:
- Verify if splitting action/topics heads improves intent quality.
"""
from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import ollama
from sklearn.metrics import accuracy_score, f1_score

DATASET_PATH = Path("src/api/intent_dataset.json")
RESULTS_DIR = Path("experiments/intent_benchmarks/results")
EMBED_MODEL = "bge-m3"
SEED = 42
TEST_RATIO = 0.33

ACTION_MAP = {
    "find": "lookup",
    "check": "verify",
    "count": "count",
}

KNOWN_TOPICS = {
    "manufacturer",
    "ingredient",
    "nlem",
    "hierarchy",
    "formula",
    "general",
}

# Hold out one fine label per topics to test compositional generalization.
HOLDOUT_FINE_LABELS = [
    "manufacturer_check",
    "ingredient_count",
    "nlem_find",
    "hierarchy_check",
    "formula_count",
    "general_find",
]


def parse_fine_label(fine_label: str) -> tuple[str, str]:
    if "_" not in fine_label:
        return fine_label, "unknown"
    topics, suffix = fine_label.rsplit("_", 1)
    topics = topics if topics in KNOWN_TOPICS else "general"
    action = ACTION_MAP.get(suffix, "unknown")
    return topics, action


def load_records() -> list[dict]:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    records = []
    for fine_label, texts in dataset.items():
        topics, action = parse_fine_label(fine_label)
        for text in texts:
            records.append(
                {
                    "text": text,
                    "fine_label": fine_label,
                    "topics_label": topics,
                    "action_label": action,
                }
            )
    return records


def stratified_split(records: list[dict], test_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    grouped = defaultdict(list)
    for rec in records:
        grouped[rec["fine_label"]].append(rec)

    rng = random.Random(seed)
    train, test = [], []
    for fine_label, rows in grouped.items():
        shuffled = list(rows)
        rng.shuffle(shuffled)
        n_test = max(1, int(round(len(shuffled) * test_ratio)))
        test.extend(shuffled[:n_test])
        train.extend(shuffled[n_test:])
    return train, test


def compositional_holdout_split(records: list[dict], holdout_labels: list[str]) -> tuple[list[dict], list[dict]]:
    holdout = set(holdout_labels)
    train = [r for r in records if r["fine_label"] not in holdout]
    test = [r for r in records if r["fine_label"] in holdout]
    return train, test


def embed_texts(texts: list[str], model: str) -> np.ndarray:
    vectors = []
    for idx, text in enumerate(texts, start=1):
        response = ollama.embeddings(model=model, prompt=text)
        vectors.append(np.array(response["embedding"], dtype=np.float32))
        if idx % 25 == 0 or idx == len(texts):
            print(f"  Embedded {idx}/{len(texts)}")
    return np.vstack(vectors)


def build_centroids(vectors: np.ndarray, labels: list[str]) -> dict[str, np.ndarray]:
    grouped = defaultdict(list)
    for vec, label in zip(vectors, labels):
        grouped[label].append(vec)
    return {label: np.mean(np.vstack(items), axis=0) for label, items in grouped.items()}


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


def predict_labels(vectors: np.ndarray, centroids: dict[str, np.ndarray]) -> list[str]:
    labels = list(centroids.keys())
    c_matrix = np.vstack([centroids[label] for label in labels])
    v_norm = normalize_rows(vectors)
    c_norm = normalize_rows(c_matrix)
    sims = v_norm @ c_norm.T
    pred_idx = np.argmax(sims, axis=1)
    return [labels[i] for i in pred_idx]


def benchmark_inference_latency(
    vectors: np.ndarray,
    legacy_centroids: dict[str, np.ndarray],
    action_centroids: dict[str, np.ndarray],
    topics_centroids: dict[str, np.ndarray],
    loops: int = 100,
) -> tuple[float, float]:
    start = time.perf_counter()
    for _ in range(loops):
        _ = predict_labels(vectors, legacy_centroids)
    legacy_ms = (time.perf_counter() - start) * 1000 / loops

    start = time.perf_counter()
    for _ in range(loops):
        _ = predict_labels(vectors, action_centroids)
        _ = predict_labels(vectors, topics_centroids)
    v2_ms = (time.perf_counter() - start) * 1000 / loops
    return legacy_ms, v2_ms


def evaluate(
    test_records: list[dict],
    test_vectors: np.ndarray,
    legacy_centroids: dict[str, np.ndarray],
    action_centroids: dict[str, np.ndarray],
    topics_centroids: dict[str, np.ndarray],
) -> dict:
    y_action_true = [r["action_label"] for r in test_records]
    y_topics_true = [r["topics_label"] for r in test_records]

    # Legacy
    legacy_fine_pred = predict_labels(test_vectors, legacy_centroids)
    legacy_action_pred = []
    legacy_topics_pred = []
    for fine in legacy_fine_pred:
        topics, action = parse_fine_label(fine)
        legacy_action_pred.append(action)
        legacy_topics_pred.append(topics)

    # V2 split heads
    v2_action_pred = predict_labels(test_vectors, action_centroids)
    v2_topics_pred = predict_labels(test_vectors, topics_centroids)

    legacy_joint = [
        int(a == ta and t == tf)
        for a, t, ta, tf in zip(legacy_action_pred, legacy_topics_pred, y_action_true, y_topics_true)
    ]
    v2_joint = [
        int(a == ta and t == tf)
        for a, t, ta, tf in zip(v2_action_pred, v2_topics_pred, y_action_true, y_topics_true)
    ]

    return {
        "legacy": {
            "action_acc": accuracy_score(y_action_true, legacy_action_pred),
            "action_f1": f1_score(y_action_true, legacy_action_pred, average="macro", zero_division=0),
            "topics_acc": accuracy_score(y_topics_true, legacy_topics_pred),
            "topics_f1": f1_score(y_topics_true, legacy_topics_pred, average="macro", zero_division=0),
            "joint_acc": float(np.mean(legacy_joint)),
        },
        "v2": {
            "action_acc": accuracy_score(y_action_true, v2_action_pred),
            "action_f1": f1_score(y_action_true, v2_action_pred, average="macro", zero_division=0),
            "topics_acc": accuracy_score(y_topics_true, v2_topics_pred),
            "topics_f1": f1_score(y_topics_true, v2_topics_pred, average="macro", zero_division=0),
            "joint_acc": float(np.mean(v2_joint)),
        },
        "support": {
            "test_size": len(test_records),
            "legacy_num_classes": len(legacy_centroids),
            "v2_action_classes": len(action_centroids),
            "v2_topics_classes": len(topics_centroids),
        },
    }


def print_summary(title: str, results: dict, legacy_ms: float, v2_ms: float) -> None:
    legacy = results["legacy"]
    v2 = results["v2"]
    support = results["support"]

    print(f"\n=== {title} ===")
    print(f"Test size: {support['test_size']}")
    print(
        "Class counts: "
        f"legacy={support['legacy_num_classes']} "
        f"vs v2(action={support['v2_action_classes']}, topics={support['v2_topics_classes']})"
    )
    print("\nMetrics")
    print(f"  Legacy action_acc: {legacy['action_acc']:.4f} | action_f1: {legacy['action_f1']:.4f}")
    print(f"  V2     action_acc: {v2['action_acc']:.4f} | action_f1: {v2['action_f1']:.4f}")
    print(f"  Legacy topics_acc : {legacy['topics_acc']:.4f} | topics_f1 : {legacy['topics_f1']:.4f}")
    print(f"  V2     topics_acc : {v2['topics_acc']:.4f} | topics_f1 : {v2['topics_f1']:.4f}")
    print(f"  Legacy joint_acc : {legacy['joint_acc']:.4f}")
    print(f"  V2     joint_acc : {v2['joint_acc']:.4f}")

    print("\nInference latency (predict test set, avg over loops)")
    print(f"  Legacy: {legacy_ms:.3f} ms")
    print(f"  V2    : {v2_ms:.3f} ms")

    print("Delta (V2 - Legacy)")
    print(f"  action_acc: {v2['action_acc'] - legacy['action_acc']:+.4f}")
    print(f"  topics_acc : {v2['topics_acc'] - legacy['topics_acc']:+.4f}")
    print(f"  joint_acc : {v2['joint_acc'] - legacy['joint_acc']:+.4f}")


def save_results(payload: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"benchmark_intent_v2_vs_legacy_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def run_experiment(title: str, train_records: list[dict], test_records: list[dict]) -> dict:
    print(f"\nRunning experiment: {title}")
    print(f"Train size: {len(train_records)} | Test size: {len(test_records)}")

    all_texts = [r["text"] for r in train_records + test_records]
    print(f"Embedding with model: {EMBED_MODEL}")
    all_vectors = embed_texts(all_texts, model=EMBED_MODEL)
    train_vectors = all_vectors[: len(train_records)]
    test_vectors = all_vectors[len(train_records) :]

    legacy_centroids = build_centroids(train_vectors, [r["fine_label"] for r in train_records])
    action_centroids = build_centroids(train_vectors, [r["action_label"] for r in train_records])
    topics_centroids = build_centroids(train_vectors, [r["topics_label"] for r in train_records])

    results = evaluate(
        test_records=test_records,
        test_vectors=test_vectors,
        legacy_centroids=legacy_centroids,
        action_centroids=action_centroids,
        topics_centroids=topics_centroids,
    )
    legacy_ms, v2_ms = benchmark_inference_latency(
        vectors=test_vectors,
        legacy_centroids=legacy_centroids,
        action_centroids=action_centroids,
        topics_centroids=topics_centroids,
    )
    print_summary(title, results, legacy_ms, v2_ms)
    return {
        "results": results,
        "latency_ms": {"legacy": legacy_ms, "v2": v2_ms},
    }


def main() -> None:
    print("Loading records...")
    records = load_records()
    random_train, random_test = stratified_split(records, test_ratio=TEST_RATIO, seed=SEED)
    comp_train, comp_test = compositional_holdout_split(records, holdout_labels=HOLDOUT_FINE_LABELS)

    random_outcome = run_experiment("Random Stratified Split", random_train, random_test)
    compositional_outcome = run_experiment("Compositional Holdout Split", comp_train, comp_test)

    payload = {
        "meta": {
            "dataset": str(DATASET_PATH),
            "embed_model": EMBED_MODEL,
            "seed": SEED,
            "test_ratio": TEST_RATIO,
            "holdout_fine_labels": HOLDOUT_FINE_LABELS,
        },
        "random_split": random_outcome,
        "compositional_holdout": compositional_outcome,
    }
    out_path = save_results(payload)
    print(f"\nSaved result: {out_path}")


if __name__ == "__main__":
    main()
