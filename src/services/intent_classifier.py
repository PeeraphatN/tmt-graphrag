"""
Intent Classification Service using Centroid-based Vector Similarity.
Uses bge-m3 embeddings via Ollama.
"""
import json
import numpy as np
import os
from pathlib import Path

import ollama

# Configuration
EMBED_MODEL = "bge-m3:latest"
INTENT_DATASET_PATH = Path(__file__).parent.parent / "api" / "intent_dataset.json"


class IntentClassifier:
    """
    Centroid-based Intent Classifier.
    
    1. Loads intent dataset (intent -> list of example queries)
    2. Computes centroid (mean) embedding for each intent
    3. Classifies new queries by finding nearest centroid (cosine similarity)
    """
    
    def __init__(self, dataset_path: str = None):
        self.dataset_path = dataset_path or str(INTENT_DATASET_PATH)
        self.centroids = {}  # intent_name -> np.array (embedding)
        self.intent_names = []
        self._initialized = False
    
    def initialize(self):
        """Load dataset and compute centroids. Call once at startup."""
        if self._initialized:
            return
        
        print("   Loading intent dataset...")
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        
        print(f"   Computing centroids for {len(dataset)} intents...")
        
        for intent_name, examples in dataset.items():
            embeddings = []
            for example in examples:
                try:
                    response = ollama.embeddings(model=EMBED_MODEL, prompt=example)
                    embeddings.append(response["embedding"])
                except Exception as e:
                    print(f"   Warning: Failed to embed '{example[:30]}...': {e}")
                    continue
            
            if embeddings:
                # Compute centroid (mean of all embeddings)
                centroid = np.mean(embeddings, axis=0)
                self.centroids[intent_name] = centroid
                self.intent_names.append(intent_name)
        
        self._initialized = True
        print(f"   ✅ Intent Classifier initialized with {len(self.centroids)} intents")
    
    def classify(self, query: str, query_embedding: np.ndarray = None) -> dict:
        """
        Classify a query into one of the known intents.
        
        Args:
            query: The user question
            query_embedding: Pre-computed embedding (optional, to avoid re-embedding)
        
        Returns:
            dict with:
                - intent: The predicted intent name
                - confidence: Cosine similarity score (0-1)
                - base_intent: The base intent category (e.g., 'manufacturer' from 'manufacturer_find')
                - action: The action type (e.g., 'find', 'check', 'count')
        """
        if not self._initialized:
            self.initialize()
        
        # Get query embedding if not provided
        if query_embedding is None:
            try:
                response = ollama.embeddings(model=EMBED_MODEL, prompt=query)
                query_embedding = np.array(response["embedding"])
            except Exception as e:
                print(f"   Error embedding query: {e}")
                return {"intent": "general_find", "confidence": 0.0, "base_intent": "general", "action": "find"}
        else:
            query_embedding = np.array(query_embedding)
        
        # Normalize query embedding
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        
        # Compute cosine similarity with all centroids
        best_intent = None
        best_score = -1.0
        
        for intent_name, centroid in self.centroids.items():
            centroid_norm = centroid / np.linalg.norm(centroid)
            similarity = np.dot(query_norm, centroid_norm)
            
            if similarity > best_score:
                best_score = similarity
                best_intent = intent_name
        
        # Parse intent into base_intent and action
        parts = best_intent.rsplit("_", 1) if best_intent else ["general", "find"]
        base_intent = parts[0] if len(parts) == 2 else best_intent
        action = parts[1] if len(parts) == 2 else "find"
        
        return {
            "intent": best_intent,
            "confidence": float(best_score),
            "base_intent": base_intent,
            "action": action
        }
    
    def get_top_k(self, query: str, query_embedding: np.ndarray = None, k: int = 3) -> list:
        """
        Get top-k intent predictions with scores.
        Useful for debugging and confidence thresholding.
        """
        if not self._initialized:
            self.initialize()
        
        if query_embedding is None:
            response = ollama.embeddings(model=EMBED_MODEL, prompt=query)
            query_embedding = np.array(response["embedding"])
        else:
            query_embedding = np.array(query_embedding)
        
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        
        scores = []
        for intent_name, centroid in self.centroids.items():
            centroid_norm = centroid / np.linalg.norm(centroid)
            similarity = np.dot(query_norm, centroid_norm)
            scores.append((intent_name, float(similarity)))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


# Global instance (singleton pattern)
_classifier_instance: IntentClassifier = None


def get_intent_classifier() -> IntentClassifier:
    """Get or create the global IntentClassifier instance."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = IntentClassifier()
    return _classifier_instance


def classify_intent(query: str, query_embedding: np.ndarray = None) -> dict:
    """
    Convenience function to classify intent.
    
    Args:
        query: User question
        query_embedding: Pre-computed embedding (optional)
    
    Returns:
        dict with intent, confidence, base_intent, action
    """
    classifier = get_intent_classifier()
    return classifier.classify(query, query_embedding)
