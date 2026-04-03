import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence as HFSequence, Value
from seqeval.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)


RUN_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = RUN_DIR.parent
DATA_DIR = EXPERIMENT_DIR / "data"
ARTIFACTS_DIR = EXPERIMENT_DIR / "artifacts"
BASE_DIR = DATA_DIR
DATA_FILES = {
    "train": DATA_DIR / "train.json",
    "validation": DATA_DIR / "validation.json",
    "test": DATA_DIR / "test.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Thai NER model for GraphRAG entity extraction.",
    )
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default="airesearch/wangchanberta-base-att-spm-uncased",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ARTIFACTS_DIR / "ner_model_output"),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=6)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=0)
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-validation-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--entity-threshold", type=float, default=0.60)
    parser.add_argument("--fp16", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--run-test-eval", action="store_true", default=True)
    parser.add_argument("--no-test-eval", dest="run_test_eval", action="store_false")

    args = parser.parse_args()

    import torch
    use_cuda = torch.cuda.is_available()
    args.use_fp16 = args.fp16 == "true" or (args.fp16 == "auto" and use_cuda)

    return args


def to_python_types(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_python_types(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_python_types(v) for v in value]
    if isinstance(value, tuple):
        return [to_python_types(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_json(path: Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_python_types(payload), f, ensure_ascii=False, indent=2)


def load_manifest(base_dir: Path) -> dict[str, Any]:
    manifest_path = base_dir / "dataset_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if "label_list" not in manifest or not isinstance(manifest["label_list"], list):
        raise ValueError("dataset_manifest.json must contain a list field 'label_list'.")
    return manifest


def maybe_limit_records(records: list[dict[str, Any]], max_samples: int, seed: int) -> list[dict[str, Any]]:
    if max_samples <= 0 or max_samples >= len(records):
        return records
    rng = random.Random(seed)
    return rng.sample(records, max_samples)


def normalize_ner_tags(
    tags: list[Any],
    label2id: dict[str, int],
    split_name: str,
    row_index: int,
) -> list[int]:
    if not isinstance(tags, list):
        raise ValueError(f"{split_name}[{row_index}] ner_tags must be a list.")
    if not tags:
        raise ValueError(f"{split_name}[{row_index}] ner_tags cannot be empty.")

    if all(isinstance(tag, int) for tag in tags):
        max_id = len(label2id) - 1
        for tag in tags:
            if tag < 0 or tag > max_id:
                raise ValueError(f"{split_name}[{row_index}] contains out-of-range label id: {tag}")
        return tags

    if all(isinstance(tag, str) for tag in tags):
        unknown = [tag for tag in tags if tag not in label2id]
        if unknown:
            raise ValueError(
                f"{split_name}[{row_index}] contains unknown labels: {sorted(set(unknown))}"
            )
        return [label2id[tag] for tag in tags]

    raise ValueError(
        f"{split_name}[{row_index}] ner_tags must be all int ids or all string labels."
    )


def validate_and_normalize_records(
    records: list[dict[str, Any]],
    split_name: str,
    label2id: dict[str, int],
) -> list[dict[str, Any]]:
    if not isinstance(records, list) or not records:
        raise ValueError(f"{split_name} split must be a non-empty list.")

    normalized_records: list[dict[str, Any]] = []
    for index, row in enumerate(records):
        if not isinstance(row, dict):
            raise ValueError(f"{split_name}[{index}] must be an object.")

        for required_key in ("id", "tokens", "ner_tags"):
            if required_key not in row:
                raise ValueError(f"{split_name}[{index}] missing key '{required_key}'.")

        tokens = row["tokens"]
        if not isinstance(tokens, list) or not tokens:
            raise ValueError(f"{split_name}[{index}] tokens must be a non-empty list.")

        tokens = [str(token) for token in tokens]
        ner_tag_ids = normalize_ner_tags(row["ner_tags"], label2id, split_name, index)

        if len(tokens) != len(ner_tag_ids):
            raise ValueError(
                f"{split_name}[{index}] length mismatch: tokens={len(tokens)}, ner_tags={len(ner_tag_ids)}"
            )

        normalized_records.append(
            {
                "id": str(row["id"]),
                "tokens": tokens,
                "ner_tags": ner_tag_ids,
            }
        )

    return normalized_records


def load_split_records(path: Path, split_name: str) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f"{split_name} split file must contain a list.")
    return records


def load_dataset_dict(
    label_list: list[str],
    label2id: dict[str, int],
    args: argparse.Namespace,
) -> DatasetDict:
    split_limits = {
        "train": args.max_train_samples,
        "validation": args.max_validation_samples,
        "test": args.max_test_samples,
    }
    split_seeds = {
        "train": args.seed + 11,
        "validation": args.seed + 17,
        "test": args.seed + 23,
    }

    data_splits: dict[str, Dataset] = {}
    for split_name, path in DATA_FILES.items():
        raw_records = load_split_records(path, split_name)
        raw_records = maybe_limit_records(raw_records, split_limits[split_name], split_seeds[split_name])
        normalized_records = validate_and_normalize_records(raw_records, split_name, label2id)

        columns = {"id": [], "tokens": [], "ner_tags": []}
        for item in normalized_records:
            columns["id"].append(item["id"])
            columns["tokens"].append(item["tokens"])
            columns["ner_tags"].append(item["ner_tags"])

        data_splits[split_name] = Dataset.from_dict(columns)

    features = Features(
        {
            "id": Value("string"),
            "tokens": HFSequence(Value("string")),
            "ner_tags": HFSequence(ClassLabel(num_classes=len(label_list), names=label_list)),
        }
    )

    for split_name in data_splits:
        data_splits[split_name] = data_splits[split_name].cast(features)

    return DatasetDict(data_splits)


def tokenize_and_align_labels(
    examples: dict[str, list[Any]],
    tokenizer: AutoTokenizer,
    max_seq_length: int,
) -> dict[str, Any]:
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        max_length=max_seq_length,
        is_split_into_words=True,
    )

    aligned_labels = []
    for batch_index, sentence_labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=batch_index)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(sentence_labels[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        aligned_labels.append(label_ids)

    tokenized["labels"] = aligned_labels
    return tokenized


def decode_sequences(
    predictions: np.ndarray,
    labels: np.ndarray,
    label_list: list[str],
) -> tuple[list[list[str]], list[list[str]]]:
    pred_ids = np.argmax(predictions, axis=2)

    decoded_predictions: list[list[str]] = []
    decoded_labels: list[list[str]] = []
    for pred_seq, label_seq in zip(pred_ids, labels):
        seq_pred: list[str] = []
        seq_label: list[str] = []
        for pred_id, label_id in zip(pred_seq, label_seq):
            if label_id == -100:
                continue
            seq_pred.append(label_list[pred_id])
            seq_label.append(label_list[label_id])
        decoded_predictions.append(seq_pred)
        decoded_labels.append(seq_label)

    return decoded_predictions, decoded_labels


def build_compute_metrics(label_list: list[str]):
    entity_types = sorted({label.split("-", 1)[1] for label in label_list if label != "O"})

    def compute_metrics(eval_prediction: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
        predictions, labels = eval_prediction
        decoded_predictions, decoded_labels = decode_sequences(predictions, labels, label_list)

        if not decoded_labels:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "accuracy": 0.0,
            }

        metrics = {
            "precision": float(precision_score(decoded_labels, decoded_predictions, zero_division=0)),
            "recall": float(recall_score(decoded_labels, decoded_predictions, zero_division=0)),
            "f1": float(f1_score(decoded_labels, decoded_predictions, zero_division=0)),
            "accuracy": float(accuracy_score(decoded_labels, decoded_predictions)),
        }

        report = classification_report(
            decoded_labels,
            decoded_predictions,
            output_dict=True,
            zero_division=0,
        )

        per_type_f1 = []
        for entity_type in entity_types:
            entity_stats = report.get(entity_type, {})
            entity_f1 = float(entity_stats.get("f1-score", 0.0))
            metrics[f"f1_{entity_type.lower()}"] = entity_f1
            per_type_f1.append(entity_f1)

        metrics["f1_macro_entity"] = float(sum(per_type_f1) / len(per_type_f1)) if per_type_f1 else 0.0
        return metrics

    return compute_metrics


def create_training_arguments(training_kwargs: dict[str, Any]) -> TrainingArguments:
    try:
        return TrainingArguments(**training_kwargs)
    except TypeError:
        fallback = dict(training_kwargs)
        if "eval_strategy" in fallback:
            fallback["evaluation_strategy"] = fallback.pop("eval_strategy")
        return TrainingArguments(**fallback)


def build_trainer(
    model: AutoModelForTokenClassification,
    tokenizer: AutoTokenizer,
    tokenized_datasets: DatasetDict,
    compute_metrics,
    args: argparse.Namespace,
) -> Trainer:
    output_dir = Path(args.output_dir)
    logging_dir = output_dir / "logs"

    training_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.grad_accum_steps,
        "num_train_epochs": args.num_epochs,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "logging_dir": str(logging_dir),
        "logging_steps": args.logging_steps,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "save_total_limit": 2,
        "report_to": "none",
        "fp16": bool(args.use_fp16),
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "dataloader_num_workers": args.dataloader_num_workers,
        "seed": args.seed,
    }

    if args.eval_steps and args.eval_steps > 0:
        training_kwargs["eval_strategy"] = "steps"
        training_kwargs["save_strategy"] = "steps"
        training_kwargs["eval_steps"] = args.eval_steps
        training_kwargs["save_steps"] = args.eval_steps
    else:
        training_kwargs["eval_strategy"] = "epoch"
        training_kwargs["save_strategy"] = "epoch"

    training_args = create_training_arguments(training_kwargs)
    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )


def build_inference_config(
    label_list: list[str],
    label2id: dict[str, int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    id2label = {str(index): label for index, label in enumerate(label_list)}
    return {
        "label_list": label_list,
        "id2label": id2label,
        "label2id": label2id,
        "entity_threshold": args.entity_threshold,
        "query_priority": ["BRAND", "DRUG"],
        "mapping_rules": {
            "manufacturer_filter": "MANUFACTURER",
        },
        "auxiliary_entities": ["FORM", "STRENGTH"],
        "max_seq_length": args.max_seq_length,
    }


def save_training_artifacts(
    output_dir: Path,
    args: argparse.Namespace,
    label_list: list[str],
    label2id: dict[str, int],
    test_metrics: dict[str, Any],
    class_report: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    train_config = {
        "model_checkpoint": args.model_checkpoint,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "num_epochs": args.num_epochs,
        "max_seq_length": args.max_seq_length,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "seed": args.seed,
        "fp16_mode": args.fp16,
        "fp16_enabled": args.use_fp16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "early_stopping_patience": args.early_stopping_patience,
        "run_test_eval": args.run_test_eval,
    }

    inference_config = build_inference_config(label_list, label2id, args)
    save_json(output_dir / "train_config.json", train_config)
    save_json(output_dir / "ner_inference_config.json", inference_config)
    save_json(output_dir / "metrics_test.json", test_metrics)
    save_json(output_dir / "classification_report_test.json", class_report)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    import torch

    use_cuda = torch.cuda.is_available()

    print("=== Fine-tune NER for GraphRAG Entity Extraction ===")
    print(f"Model checkpoint: {args.model_checkpoint}")
    print(f"CUDA available: {use_cuda}")
    print(f"FP16 enabled: {args.use_fp16}")
    print(f"Output dir: {args.output_dir}")

    manifest = load_manifest(DATA_DIR)
    label_list = manifest["label_list"]
    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {idx: label for idx, label in enumerate(label_list)}

    print("Loading dataset splits...")
    datasets = load_dataset_dict(label_list, label2id, args)
    print(datasets)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)

    print("Tokenizing datasets...")
    tokenized_datasets = datasets.map(
        lambda batch: tokenize_and_align_labels(batch, tokenizer, args.max_seq_length),
        batched=True,
        remove_columns=datasets["train"].column_names,
        desc="Tokenizing and aligning labels",
    )

    print("Loading model...")
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_checkpoint,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    if args.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass
        if getattr(model.config, "use_cache", None) is not None:
            model.config.use_cache = False

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        tokenized_datasets=tokenized_datasets,
        compute_metrics=build_compute_metrics(label_list),
        args=args,
    )

    print("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    print(f"Training finished. Steps: {train_result.global_step}")

    test_metrics: dict[str, Any] = {}
    class_report: dict[str, Any] = {}
    if args.run_test_eval:
        print("Running evaluation on test split...")
        prediction_output = trainer.predict(tokenized_datasets["test"], metric_key_prefix="test")
        decoded_predictions, decoded_labels = decode_sequences(
            prediction_output.predictions,
            prediction_output.label_ids,
            label_list,
        )
        class_report = classification_report(
            decoded_labels,
            decoded_predictions,
            output_dict=True,
            zero_division=0,
        )
        test_metrics = dict(prediction_output.metrics)
        print(f"Test metrics: {test_metrics}")
    else:
        print("Skipping test evaluation by config.")

    output_dir = Path(args.output_dir)
    final_model_dir = output_dir / "final_model"
    final_model_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving model to: {final_model_dir}")
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    print("Saving config and metrics artifacts...")
    save_training_artifacts(output_dir, args, label_list, label2id, test_metrics, class_report)
    print("Done.")


if __name__ == "__main__":
    main()
