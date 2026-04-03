import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

try:
    from pythainlp.tokenize import word_tokenize as thai_word_tokenize
except Exception:
    thai_word_tokenize = None


DEFAULT_CONFIG = {
    "entity_threshold": 0.60,
    "query_priority": ["BRAND", "DRUG"],
    "mapping_rules": {"manufacturer_filter": "MANUFACTURER"},
    "auxiliary_entities": ["FORM", "STRENGTH"],
    "max_seq_length": 128,
}


def load_tokenizer_with_fallback(model_dir: Path) -> AutoTokenizer:
    try:
        return AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    except Exception as fast_error:
        print(f"Fast tokenizer failed: {fast_error}")
        print("Falling back to slow tokenizer (use_fast=False)...")
        try:
            return AutoTokenizer.from_pretrained(str(model_dir), use_fast=False)
        except Exception as slow_error:
            raise RuntimeError(
                "Failed to load tokenizer with both fast and slow backends. "
                "Please ensure 'sentencepiece' is installed."
            ) from slow_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NER inference and map output for GraphRAG query extraction.")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to trained model directory.")
    parser.add_argument("--text", type=str, default="", help="Input text to extract entities from.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override confidence threshold from config.",
    )
    parser.add_argument("--max-seq-length", type=int, default=0, help="Override max sequence length from config.")
    parser.add_argument("--json", action="store_true", help="Print only JSON output.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_inference_config(model_dir: Path) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    config_path = model_dir / "ner_inference_config.json"
    if config_path.exists():
        from_file = load_json(config_path)
        if isinstance(from_file, dict):
            config.update(from_file)
    return config


def tokenize_text(text: str) -> list[str]:
    if thai_word_tokenize is not None:
        tokens = thai_word_tokenize(text, engine="newmm", keep_whitespace=False)
        return [token for token in tokens if token]
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def compute_token_offsets(text: str, tokens: list[str]) -> list[tuple[int, int]]:
    offsets = []
    cursor = 0
    for token in tokens:
        start = text.find(token, cursor)
        if start < 0:
            start = text.find(token)
            if start < 0:
                raise ValueError(f"Token '{token}' not found in text while reconstructing offsets.")
        end = start + len(token)
        offsets.append((start, end))
        cursor = end
    return offsets


def predict_word_labels(
    model: AutoModelForTokenClassification,
    tokenizer: AutoTokenizer,
    tokens: list[str],
    label_list: list[str],
    max_seq_length: int,
) -> tuple[list[str], list[float]]:
    encoded = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )

    # Save word_ids before converting to plain dict (BatchEncoding method)
    word_ids = encoded.word_ids(batch_index=0)

    model_device = next(model.parameters()).device
    encoded = {key: value.to(model_device) for key, value in encoded.items()}

    with torch.no_grad():
        logits = model(**encoded).logits[0]

    probabilities = torch.softmax(logits, dim=-1).detach().cpu().numpy()
    predicted_ids = np.argmax(probabilities, axis=-1)

    word_labels = ["O"] * len(tokens)
    word_confidences = [1.0] * len(tokens)
    seen_word_indices = set()

    for token_index, word_index in enumerate(word_ids):
        if word_index is None or word_index in seen_word_indices:
            continue
        seen_word_indices.add(word_index)
        label_id = int(predicted_ids[token_index])
        label_id = max(0, min(label_id, len(label_list) - 1))
        label_name = label_list[label_id]
        confidence = float(probabilities[token_index][label_id])
        word_labels[word_index] = label_name
        word_confidences[word_index] = confidence

    return word_labels, word_confidences


def parse_bio(label: str) -> tuple[str, str]:
    if label == "O":
        return "O", ""
    if "-" not in label:
        return "O", ""
    prefix, entity_type = label.split("-", 1)
    if prefix not in {"B", "I"}:
        return "O", ""
    return prefix, entity_type


def extract_entities_from_words(
    text: str,
    tokens: list[str],
    offsets: list[tuple[int, int]],
    labels: list[str],
    confidences: list[float],
    threshold: float,
) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []

    current_type = None
    current_start = None
    current_end = None
    current_confidences: list[float] = []

    def flush_current() -> None:
        nonlocal current_type, current_start, current_end, current_confidences
        if current_type is None or current_start is None or current_end is None:
            current_type = None
            current_start = None
            current_end = None
            current_confidences = []
            return

        entity_text = text[current_start:current_end]
        avg_confidence = float(sum(current_confidences) / len(current_confidences))
        entities.append(
            {
                "label": current_type,
                "text": entity_text,
                "confidence": round(avg_confidence, 6),
                "start": current_start,
                "end": current_end,
            }
        )
        current_type = None
        current_start = None
        current_end = None
        current_confidences = []

    for index, (label, confidence) in enumerate(zip(labels, confidences)):
        prefix, entity_type = parse_bio(label)
        token_start, token_end = offsets[index]

        if prefix == "O" or confidence < threshold:
            flush_current()
            continue

        if prefix == "B":
            flush_current()
            current_type = entity_type
            current_start = token_start
            current_end = token_end
            current_confidences = [confidence]
            continue

        if prefix == "I":
            if current_type == entity_type and current_start is not None:
                current_end = token_end
                current_confidences.append(confidence)
            else:
                flush_current()
                current_type = entity_type
                current_start = token_start
                current_end = token_end
                current_confidences = [confidence]

    flush_current()
    return entities


def top_entity_text(entities: list[dict[str, Any]], entity_label: str) -> str | None:
    candidates = [entity for entity in entities if entity.get("label") == entity_label]
    if not candidates:
        return None
    candidates.sort(key=lambda item: (float(item["confidence"]), len(item["text"])), reverse=True)
    return str(candidates[0]["text"])


def unique_entity_texts(entities: list[dict[str, Any]], entity_label: str) -> list[str]:
    candidates = [entity for entity in entities if entity.get("label") == entity_label]
    candidates.sort(key=lambda item: (float(item["confidence"]), len(item["text"])), reverse=True)

    results: list[str] = []
    seen = set()
    for entity in candidates:
        text = str(entity["text"]).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        results.append(text)
    return results


def map_entities_to_rag_payload(
    text: str,
    entities: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    query = ""
    query_priority = config.get("query_priority", ["BRAND", "DRUG"])
    for entity_label in query_priority:
        candidate = top_entity_text(entities, entity_label)
        if candidate:
            query = candidate
            break

    manufacturer_filter = top_entity_text(entities, "MANUFACTURER")
    forms = unique_entity_texts(entities, "FORM")
    strengths = unique_entity_texts(entities, "STRENGTH")

    return {
        "input_text": text,
        "query": query,
        "manufacturer_filter": manufacturer_filter,
        "entities": entities,
        "auxiliary": {
            "forms": forms,
            "strengths": strengths,
        },
    }


def run_inference(
    model: AutoModelForTokenClassification,
    tokenizer: AutoTokenizer,
    text: str,
    config: dict[str, Any],
    threshold_override: float | None,
    max_seq_length_override: int,
) -> dict[str, Any]:
    tokens = tokenize_text(text)
    if not tokens:
        raise ValueError("No tokens produced from input text.")

    offsets = compute_token_offsets(text, tokens)
    label_list = list(model.config.id2label.values())
    threshold = float(threshold_override) if threshold_override is not None else float(config["entity_threshold"])
    max_seq_length = max_seq_length_override if max_seq_length_override > 0 else int(config["max_seq_length"])

    labels, confidences = predict_word_labels(
        model=model,
        tokenizer=tokenizer,
        tokens=tokens,
        label_list=label_list,
        max_seq_length=max_seq_length,
    )

    entities = extract_entities_from_words(
        text=text,
        tokens=tokens,
        offsets=offsets,
        labels=labels,
        confidences=confidences,
        threshold=threshold,
    )
    payload = map_entities_to_rag_payload(text, entities, config)

    payload["debug"] = {
        "tokenizer_engine": "pythainlp_newmm" if thai_word_tokenize else "regex_fallback",
        "threshold": threshold,
        "max_seq_length": max_seq_length,
        "tokens": tokens,
        "word_labels": labels,
    }
    return payload


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir).resolve()
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    config = load_inference_config(model_dir)
    if not args.text.strip():
        raise ValueError("Please provide non-empty text using --text.")

    tokenizer = load_tokenizer_with_fallback(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(str(model_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    payload = run_inference(
        model=model,
        tokenizer=tokenizer,
        text=args.text,
        config=config,
        threshold_override=args.threshold,
        max_seq_length_override=args.max_seq_length,
    )

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print("=== NER Inference Result ===")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
