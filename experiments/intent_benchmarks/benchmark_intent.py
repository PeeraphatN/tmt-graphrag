
import json
import os
import time
import psutil
import numpy as np
import pprint
import ollama
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.metrics import accuracy_score, classification_report, f1_score, matthews_corrcoef, confusion_matrix, precision_score, recall_score

# Configuration
TRAIN_PATH = "experiments/intent_benchmarks/benchmark_train.json"
TEST_PATH = "experiments/intent_benchmarks/benchmark_test.json"
MODELS = ["all-minilm", "nomic-embed-text:latest", "bge-m3", "mxbai-embed-large:latest","qllama/bge-small-en-v1.5:latest","paraphrase-multilingual","llama3"]

def load_data(filepath):
    """Load and flatten dataset from JSON."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    X_text = []
    y_labels = []
    for label, sentences in data.items():
        for sent in sentences:
            X_text.append(sent)
            y_labels.append(label)
    
    return X_text, np.array(y_labels)

def get_ollama_embeddings(texts, model_name):
    """Fetch embeddings from Ollama API."""
    embeddings = []
    
    # Process one by one or in small batches
    for i, text in enumerate(texts):
        if i % 10 == 0:
            print(f"      Processing {i}/{len(texts)}...", end="\r")
        
        try:
            response = ollama.embeddings(model=model_name, prompt=text)
            embeddings.append(response["embedding"])
        except Exception as e:
            print(f"\n❌ Error fetching embedding for '{text}': {e}")
            raise e
            
    print(f"      Processing {len(texts)}/{len(texts)}... Done.")
    return np.array(embeddings)

def compute_centroids(X_vectors, y_labels):
    """Compute centroid (mean) vector for each class."""
    unique_labels = np.unique(y_labels)
    centroids = {}
    for label in unique_labels:
        mask = y_labels == label
        centroids[label] = np.mean(X_vectors[mask], axis=0)
    return centroids

def predict_centroid(X_test, centroids):
    """Predict class by finding nearest centroid using cosine similarity."""
    labels = list(centroids.keys())
    centroid_matrix = np.array([centroids[label] for label in labels])
    
    # Normalize for cosine similarity
    X_norm = X_test / np.linalg.norm(X_test, axis=1, keepdims=True)
    C_norm = centroid_matrix / np.linalg.norm(centroid_matrix, axis=1, keepdims=True)
    
    # Cosine similarity
    similarities = X_norm @ C_norm.T
    
    pred_indices = np.argmax(similarities, axis=1)
    return np.array([labels[i] for i in pred_indices])

def run_model_benchmark(model_name, X_train_text, y_train, X_test_text, y_test):
    print(f"\n{'='*40}")
    print(f" 🧠 Benchmarking Model: {model_name}")
    print(f"{'='*40}")
    
    # 1. Generate Embeddings
    print(f"   Connecting to Ollama: {model_name}")
    start_time = time.perf_counter()
    
    try:
        print("      Embedding Train Set...")
        X_train_vectors = get_ollama_embeddings(X_train_text, model_name)
        
        print("      Embedding Test Set...")
        X_test_vectors = get_ollama_embeddings(X_test_text, model_name)

    except Exception as e:
        print(f"\n❌ Failed to connect to Ollama. Ensure 'ollama serve' is running and '{model_name}' is pulled.")
        print(f"Error: {e}")
        return None
    
    embed_time = time.perf_counter() - start_time
    dims = X_train_vectors.shape[1]
    print(f"   Dimensions: {dims}")
    print(f"   Embedding Time: {embed_time:.2f}s")
    
    model_results = {}

    # Method 1: k-NN Sweep (K=3 to 15)
    best_knn_acc = -1
    best_k = -1
    best_knn_results = {}
    
    print(f"   > Sweeping k-NN (k=3..15)...")
    
    for k in range(3, 16):
        # Use 'distance' weights and 'cosine' metric for best standard results
        knn = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='cosine')
        knn.fit(X_train_vectors, y_train)
        
        eval_start = time.perf_counter()
        y_pred_knn = knn.predict(X_test_vectors)
        knn_latency = (time.perf_counter() - eval_start) / len(y_test) * 1000
        
        acc = accuracy_score(y_test, y_pred_knn)
        
        if acc > best_knn_acc:
            best_knn_acc = acc
            best_k = k
            best_knn_results = {
                'k': k,
                'accuracy': acc,
                'f1': f1_score(y_test, y_pred_knn, average='macro', zero_division=0),
                'precision': precision_score(y_test, y_pred_knn, average='macro', zero_division=0),
                'recall': recall_score(y_test, y_pred_knn, average='macro', zero_division=0),
                'latency': knn_latency,
                'cm': confusion_matrix(y_test, y_pred_knn)
            }
    
    model_results['knn'] = best_knn_results
    
    # Method 2: Centroid
    centroids = compute_centroids(X_train_vectors, y_train)
    
    eval_start = time.perf_counter()
    y_pred_centroid = predict_centroid(X_test_vectors, centroids)
    centroid_latency = (time.perf_counter() - eval_start) / len(y_test) * 1000
    
    model_results['centroid'] = {
        'accuracy': accuracy_score(y_test, y_pred_centroid),
        'f1': f1_score(y_test, y_pred_centroid, average='macro', zero_division=0),
        'precision': precision_score(y_test, y_pred_centroid, average='macro', zero_division=0),
        'recall': recall_score(y_test, y_pred_centroid, average='macro', zero_division=0),
        'latency': centroid_latency,
        'cm': confusion_matrix(y_test, y_pred_centroid)
    }
    
    print(f"   > Best k-NN (K={best_knn_results['k']}) Acc: {best_knn_results['accuracy']:.4f} | Prec: {best_knn_results['precision']:.4f} | Rec: {best_knn_results['recall']:.4f}")
    print(f"     Confusion Matrix:\n{best_knn_results['cm']}")
    print(f"   > Centroid Acc: {model_results['centroid']['accuracy']:.4f}")

    return {
        'model': model_name,
        'dims': dims,
        'results': model_results
    }

def run_benchmark():
    print("=== Multi-Model Vector Similarity Benchmark (K-Sweep 3..15) ===")
    
    # Load Data
    print(f"📂 Loading Data...")
    X_train_text, y_train = load_data(TRAIN_PATH)
    X_test_text, y_test = load_data(TEST_PATH)
    print(f"   Train: {len(X_train_text)} | Test: {len(X_test_text)}")

    all_benchmarks = []
    
    for model in MODELS:
        res = run_model_benchmark(model, X_train_text, y_train, X_test_text, y_test)
        if res:
            all_benchmarks.append(res)
            
    # Final Summary Table
    print(f"\n{'='*100}")
    print(f" 📊 FINAL COMPARISON SUMMARY")
    print(f"{'='*100}")
    print(f"{'Model':<25} {'Dims':<6} | {'Best K':<6} {'k-NN Acc':<10} {'k-NN F1':<10} | {'Centroid Acc':<12}")
    print("-" * 100)
    
    for b in all_benchmarks:
        m = b['model']
        d = b['dims']
        k_res = b['results']['knn']
        c_res = b['results']['centroid']
        
        print(f"{m:<25} {d:<6} | {k_res['k']:<6} {k_res['accuracy']:<10.4f} {k_res['f1']:<10.4f} | {c_res['accuracy']:<12.4f}")

if __name__ == "__main__":
    run_benchmark()
