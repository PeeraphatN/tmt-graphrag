"""
NER inference service for IntentV2 slot extraction.

Loads the fine-tuned token classification model lazily and provides
a lightweight API for query-time entity extraction.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config import NER_CONFIDENCE_THRESHOLD, NER_MAX_SEQ_LENGTH, NER_MODEL_DIR

try:
    import torch
    from transformers import AutoModelForTokenClassification
except Exception:  # pragma: no cover - optional runtime dependency
    torch = None
    AutoModelForTokenClassification = None

try:
    from src.services import ner_inference_helper as helper
except Exception:  # pragma: no cover - optional runtime dependency
    helper = None


class NerInferenceService:
    def __init__(self, model_dir: str | None = None):
        self.model_dir = Path(model_dir or NER_MODEL_DIR).resolve()
        self.threshold = float(NER_CONFIDENCE_THRESHOLD)
        self.max_seq_length = int(NER_MAX_SEQ_LENGTH)
        self._config: dict[str, Any] = {}
        self._model = None
        self._tokenizer = None
        self._load_error: str | None = None
        self._device_name = "cpu"

    def _ensure_loaded(self) -> bool:
        if self._model is not None and self._tokenizer is not None:
            return True

        if helper is None or torch is None or AutoModelForTokenClassification is None:
            self._load_error = "NER dependencies unavailable (torch/transformers/helper import failed)."
            return False

        if not self.model_dir.exists():
            self._load_error = f"NER model directory not found: {self.model_dir}"
            return False

        try:
            self._config = helper.load_inference_config(self.model_dir)
            self._tokenizer = helper.load_tokenizer_with_fallback(self.model_dir)
            self._model = AutoModelForTokenClassification.from_pretrained(str(self.model_dir))
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model.to(device)
            self._model.eval()
            self._device_name = str(device)
            self._load_error = None
            return True
        except Exception as exc:  # pragma: no cover - runtime loading path
            self._load_error = str(exc)
            self._model = None
            self._tokenizer = None
            return False

    def is_available(self) -> bool:
        return self._ensure_loaded()

    def _top_entity_text(self, entities: list[dict[str, Any]], label: str) -> str | None:
        if helper is not None:
            return helper.top_entity_text(entities, label)
        candidates = [e for e in entities if e.get("label") == label]
        if not candidates:
            return None
        candidates.sort(key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
        return str(candidates[0].get("text", "")).strip() or None

    def extract(self, text: str) -> dict[str, Any]:
        """
        Returns extraction payload compatible with AQT integration:
        {
          "available": bool,
          "query": str,
          "manufacturer_filter": str | None,
          "slots": {...},
          "entities": [...]
        }
        """
        if not text.strip():
            return {"available": False, "query": "", "manufacturer_filter": None, "slots": {}, "entities": []}

        if not self._ensure_loaded():
            return {
                "available": False,
                "query": "",
                "manufacturer_filter": None,
                "slots": {},
                "entities": [],
                "error": self._load_error,
            }

        threshold = self.threshold
        if self._config.get("entity_threshold") is not None:
            threshold = float(self._config["entity_threshold"])

        try:
            payload = helper.run_inference(
                model=self._model,
                tokenizer=self._tokenizer,
                text=text,
                config=self._config,
                threshold_override=threshold,
                max_seq_length_override=self.max_seq_length,
            )
        except Exception as exc:  # pragma: no cover - runtime inference path
            return {
                "available": False,
                "query": "",
                "manufacturer_filter": None,
                "slots": {},
                "entities": [],
                "error": str(exc),
            }

        entities = payload.get("entities", [])
        slots: dict[str, Any] = {}
        if payload.get("query"):
            slots["query"] = payload["query"]
        if payload.get("manufacturer_filter"):
            slots["manufacturer"] = payload["manufacturer_filter"]

        # Keep raw entity-derived slots for downstream topic/slot routing.
        brand = self._top_entity_text(entities, "BRAND")
        drug = self._top_entity_text(entities, "DRUG")
        strength = self._top_entity_text(entities, "STRENGTH")
        form = self._top_entity_text(entities, "FORM")
        if brand:
            slots["brand"] = brand
        if drug:
            slots["drug"] = drug
        if strength:
            slots["strength"] = strength
        if form:
            slots["form"] = form

        return {
            "available": True,
            "query": str(payload.get("query", "")),
            "manufacturer_filter": payload.get("manufacturer_filter"),
            "slots": slots,
            "entities": entities,
            "auxiliary": payload.get("auxiliary", {}),
            "device": self._device_name,
        }


_ner_service_instance: NerInferenceService | None = None


def get_ner_service() -> NerInferenceService:
    global _ner_service_instance
    if _ner_service_instance is None:
        _ner_service_instance = NerInferenceService()
    return _ner_service_instance


