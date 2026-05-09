import logging
import torch

from src.config import RERANKER_DEVICE

logger = logging.getLogger(__name__)


def _resolve_device(device_cfg: str) -> str:
    """Resolve the RERANKER_DEVICE config value to an actual torch device string."""
    if device_cfg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "RERANKER_DEVICE=cuda was requested but CUDA is not available on this system."
            )
        return "cuda"
    if device_cfg == "cpu":
        return "cpu"
    # "auto" or anything unrecognised — pick CUDA if present, else CPU
    return "cuda" if torch.cuda.is_available() else "cpu"


def _log_device(device: str) -> None:
    """Emit a structured log line about the resolved reranker device."""
    if device == "cuda":
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            free_gb = free_bytes / (1024 ** 3)
            total_gb = total_bytes / (1024 ** 3)
            logger.info(
                "Reranker device: cuda (VRAM free: %.1f GB / %.1f GB)",
                free_gb,
                total_gb,
            )
        except Exception:
            # Fallback if mem_get_info is unavailable
            try:
                props = torch.cuda.get_device_properties(0)
                total_gb = props.total_memory / (1024 ** 3)
                logger.info(
                    "Reranker device: cuda (total VRAM: %.1f GB)", total_gb
                )
            except Exception:
                logger.info("Reranker device: cuda")
    else:
        forced = RERANKER_DEVICE == "cpu"
        suffix = " (forced via RERANKER_DEVICE=cpu)" if forced else ""
        logger.info("Reranker device: cpu%s", suffix)


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the Reranker with a CrossEncoder model.
        Using a lightweight model by default for speed.
        """
        self.device = _resolve_device(RERANKER_DEVICE)
        _log_device(self.device)
        logger.info("Loading Reranker model: %s on %s...", model_name, self.device)
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name, device=self.device)
            logger.info("Reranker model loaded successfully.")
        except Exception as e:
            logger.error("Failed to load Reranker model: %s", e)
            self.model = None

    def rerank(self, query: str, candidates: list[dict], top_k: int = 10) -> list[dict]:
        """
        Rerank a list of candidate nodes based on relevance to the query.
        candidates: List of dicts, each must have a 'node' key.
        """
        if not self.model or not candidates:
            return candidates[:top_k]

        # Prepare pairs for CrossEncoder: (query, document_text)
        pairs = []
        for item in candidates:
            node = item.get("node")
            # Construct a text representation of the node
            props = dict(node)
            # Prioritize semantic fields
            text = f"{props.get('fsn', '')} {props.get('common_name', '')} {props.get('trade_name', '')} {props.get('generic_name', '')}"
            pairs.append([query, text.strip()])

        # Predict scores
        scores = self.model.predict(pairs)

        # Attach scores and sort
        for i, item in enumerate(candidates):
            item["rerank_score"] = float(scores[i])

        # Sort by rerank_score descending
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        return candidates[:top_k]


_reranker_instance: Reranker | None = None


def get_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> Reranker:
    """
    Singleton accessor for reranker model.
    Ensures model is loaded once per process.
    """
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = Reranker(model_name=model_name)
    return _reranker_instance
