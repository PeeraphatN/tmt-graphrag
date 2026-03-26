import torch

class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the Reranker with a CrossEncoder model.
        Using a lightweight model by default for speed.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Reranker model: {model_name} on {self.device}...")
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name, device=self.device)
            print("Reranker model loaded successfully.")
        except Exception as e:
            print(f"Failed to load Reranker model (Import Error or other): {e}")
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
