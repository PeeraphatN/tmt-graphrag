"""
Intent Classification Service using Centroid-based Vector Similarity.
Uses bge-m3 embeddings via Ollama.
"""
import json
import logging
import os
import pickle
import hashlib
import numpy as np
from collections import defaultdict
from pathlib import Path

import ollama

from src.config import EMBED_MODEL

logger = logging.getLogger(__name__)

# Configuration
INTENT_DATASET_PATH = Path(__file__).parent.parent / "api" / "intent_dataset.json"
INTENT_CENTROID_CACHE_VERSION = 1


def _as_bool(value: str | None, default: bool = True) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


INTENT_CENTROID_CACHE_ENABLED = _as_bool(os.getenv("INTENT_CENTROID_CACHE_ENABLED"), default=True)
INTENT_CENTROID_CACHE_DIR = Path(
    os.getenv(
        "INTENT_CENTROID_CACHE_DIR",
        str(Path(__file__).resolve().parents[2] / "cache" / "intent_classifier"),
    )
)


class IntentClassifier:
    """
    Centroid-based Intent Classifier with two levels:
    1) Fine intent labels (e.g. manufacturer_find, nlem_check)
    2) Target intent labels (e.g. manufacturer, nlem)
    """

    ACTION_SUFFIXES = {"find", "check", "count"}

    def __init__(
        self,
        dataset_path: str = None,
        confidence_threshold: float = 0.55,
        ambiguity_margin: float = 0.03,
    ):
        self.dataset_path = dataset_path or str(INTENT_DATASET_PATH)
        self.confidence_threshold = confidence_threshold
        self.ambiguity_margin = ambiguity_margin
        self.fine_centroids = {}  # fine_intent -> np.array
        self.target_centroids = {}  # target_type -> np.array
        self.intent_names = []
        self.target_names = []
        self._initialized = False
        self._cache_enabled = INTENT_CENTROID_CACHE_ENABLED
        self._cache_dir = INTENT_CENTROID_CACHE_DIR

    def _resolve_dataset_path(self) -> Path:
        return Path(self.dataset_path).resolve()

    def _compute_dataset_hash(self, dataset_path: Path) -> str:
        content = dataset_path.read_bytes()
        return hashlib.sha256(content).hexdigest()

    def _build_cache_file_path(self, dataset_hash: str) -> Path:
        key_source = (
            f"v{INTENT_CENTROID_CACHE_VERSION}|"
            f"model={EMBED_MODEL}|"
            f"dataset={self._resolve_dataset_path()}|"
            f"hash={dataset_hash}"
        )
        cache_key = hashlib.sha256(key_source.encode("utf-8")).hexdigest()[:16]
        return self._cache_dir / f"intent_centroids_{cache_key}.pkl"

    @staticmethod
    def _coerce_centroid_map(raw_map: dict) -> dict[str, np.ndarray]:
        centroids: dict[str, np.ndarray] = {}
        for label, value in (raw_map or {}).items():
            arr = np.asarray(value, dtype=np.float32)
            if arr.size == 0:
                continue
            centroids[str(label)] = arr
        return centroids

    def _load_centroids_from_cache(self, dataset_hash: str) -> bool:
        if not self._cache_enabled:
            return False

        cache_file = self._build_cache_file_path(dataset_hash)
        if not cache_file.exists():
            return False

        try:
            with open(cache_file, "rb") as f:
                payload = pickle.load(f)
        except Exception as e:
            logger.warning("   Warning: Failed to read intent centroid cache: %s", e)
            return False

        if not isinstance(payload, dict):
            return False
        if int(payload.get("cache_version", -1)) != INTENT_CENTROID_CACHE_VERSION:
            return False
        if str(payload.get("embed_model", "")) != str(EMBED_MODEL):
            return False
        if str(payload.get("dataset_hash", "")) != dataset_hash:
            return False

        fine_centroids = self._coerce_centroid_map(payload.get("fine_centroids", {}))
        target_centroids = self._coerce_centroid_map(payload.get("target_centroids", {}))
        if not fine_centroids or not target_centroids:
            return False

        self.fine_centroids = fine_centroids
        self.target_centroids = target_centroids
        self.intent_names = list(payload.get("intent_names", list(fine_centroids.keys())))
        self.target_names = list(payload.get("target_names", list(target_centroids.keys())))

        logger.info(
            "   Loaded intent centroid cache (fine=%d, target=%d)",
            len(self.fine_centroids),
            len(self.target_centroids),
        )
        return True

    def _save_centroids_to_cache(self, dataset_hash: str) -> None:
        if not self._cache_enabled:
            return
        if not self.fine_centroids or not self.target_centroids:
            return

        cache_file = self._build_cache_file_path(dataset_hash)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "cache_version": INTENT_CENTROID_CACHE_VERSION,
            "embed_model": EMBED_MODEL,
            "dataset_path": str(self._resolve_dataset_path()),
            "dataset_hash": dataset_hash,
            "fine_centroids": self.fine_centroids,
            "target_centroids": self.target_centroids,
            "intent_names": self.intent_names,
            "target_names": self.target_names,
        }

        try:
            with open(cache_file, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("   Saved intent centroid cache -> %s", cache_file)
        except Exception as e:
            logger.warning("   Warning: Failed to save intent centroid cache: %s", e)

    @staticmethod
    def _parse_intent_name(intent_name: str) -> tuple[str, str]:
        """
        Convert fine label into (target, action) pair.
        Example: 'manufacturer_find' -> ('manufacturer', 'find')
        """
        if "_" not in intent_name:
            return intent_name, "find"

        base, suffix = intent_name.rsplit("_", 1)
        if suffix in IntentClassifier.ACTION_SUFFIXES:
            return base, suffix
        return intent_name, "find"

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec if norm == 0 else (vec / norm)

    def _embed_text(self, text: str) -> np.ndarray | None:
        try:
            response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
            return np.array(response["embedding"], dtype=np.float32)
        except Exception as e:
            logger.warning("   Warning: Failed to embed '%s...': %s", text[:30], e)
            return None

    def _score_against(self, query_embedding: np.ndarray, centroids: dict[str, np.ndarray]) -> list[tuple[str, float]]:
        """
        Return cosine similarity scores sorted descending.
        """
        q = self._normalize(query_embedding)
        scores = []
        for name, centroid in centroids.items():
            c = self._normalize(centroid)
            score = float(np.dot(q, c))
            scores.append((name, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def initialize(self):
        """Load dataset and compute centroids. Call once at startup."""
        if self._initialized:
            return

        dataset_path = self._resolve_dataset_path()
        if not dataset_path.exists():
            raise FileNotFoundError(f"Intent dataset not found: {dataset_path}")

        dataset_hash = self._compute_dataset_hash(dataset_path)
        if self._load_centroids_from_cache(dataset_hash):
            self._initialized = True
            return

        logger.info("   Loading intent dataset...")
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        logger.info("   Computing centroids for %d fine intents...", len(dataset))

        fine_vectors: dict[str, list[np.ndarray]] = defaultdict(list)
        target_vectors: dict[str, list[np.ndarray]] = defaultdict(list)

        for fine_intent, examples in dataset.items():
            target, _ = self._parse_intent_name(fine_intent)
            for example in examples:
                emb = self._embed_text(example)
                if emb is None:
                    continue
                fine_vectors[fine_intent].append(emb)
                target_vectors[target].append(emb)

        for fine_intent, vectors in fine_vectors.items():
            if vectors:
                self.fine_centroids[fine_intent] = np.mean(vectors, axis=0)
                self.intent_names.append(fine_intent)

        for target, vectors in target_vectors.items():
            if vectors:
                self.target_centroids[target] = np.mean(vectors, axis=0)
                self.target_names.append(target)

        self._save_centroids_to_cache(dataset_hash)
        self._initialized = True
        logger.info(
            "   Intent Classifier initialized (fine=%d, target=%d)",
            len(self.fine_centroids),
            len(self.target_centroids),
        )

    def classify(self, query: str, query_embedding: np.ndarray = None) -> dict:
        """
        Classify query into both fine intent and target intent.
        """
        if not self._initialized:
            self.initialize()

        if query_embedding is None:
            query_embedding = self._embed_text(query)
            if query_embedding is None:
                return {
                    "intent": "general_find",
                    "confidence": 0.0,
                    "base_intent": "general",
                    "action": "find",
                    "target_type": "general",
                    "target_confidence": 0.0,
                    "target_margin": 0.0,
                    "is_ambiguous": True,
                }
        else:
            query_embedding = np.array(query_embedding, dtype=np.float32)

        fine_scores = self._score_against(query_embedding, self.fine_centroids)
        target_scores = self._score_against(query_embedding, self.target_centroids)

        best_intent, best_score = fine_scores[0] if fine_scores else ("general_find", 0.0)
        parsed_target, parsed_action = self._parse_intent_name(best_intent)

        best_target, best_target_score = target_scores[0] if target_scores else ("general", 0.0)
        second_target_score = target_scores[1][1] if len(target_scores) > 1 else -1.0
        target_margin = best_target_score - second_target_score

        is_ambiguous = (best_target_score < self.confidence_threshold) or (target_margin < self.ambiguity_margin)
        if is_ambiguous:
            resolved_target = "general"
        else:
            resolved_target = best_target or parsed_target

        return {
            "intent": best_intent,
            "confidence": float(best_score),
            "base_intent": resolved_target,
            "action": parsed_action,
            "target_type": resolved_target,
            "target_confidence": float(best_target_score),
            "target_margin": float(target_margin),
            "is_ambiguous": is_ambiguous,
            "top_targets": [{"target": name, "score": score} for name, score in target_scores[:3]],
        }

    def get_top_k(
        self,
        query: str,
        query_embedding: np.ndarray = None,
        k: int = 3,
        level: str = "target",
    ) -> list[dict]:
        """
        Get top-k intent predictions with scores.
        Args:
            level: "target" or "fine"
        """
        if not self._initialized:
            self.initialize()

        if query_embedding is None:
            query_embedding = self._embed_text(query)
            if query_embedding is None:
                return []
        else:
            query_embedding = np.array(query_embedding, dtype=np.float32)

        centroids = self.target_centroids if level == "target" else self.fine_centroids
        scores = self._score_against(query_embedding, centroids)
        return [{"label": label, "score": score} for label, score in scores[:k]]


# Global instance (singleton pattern)
_classifier_instance: IntentClassifier = None


def get_intent_classifier() -> IntentClassifier:
    """Get or create the global IntentClassifier instance."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = IntentClassifier()
    return _classifier_instance

