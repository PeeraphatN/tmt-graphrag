
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

class Logger:
    def __init__(self):
        self.logs = []
    
    def log(self, message=""):
        print(message)
        self.logs.append(message)
        
    def get_content(self):
        return "\n".join(self.logs)

def save_results_to_file(content):
    """Save benchmark results to a timestamped file."""
    results_dir = "experiments/intent_benchmarks/results"
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_intent_{timestamp}.txt"
    filepath = os.path.join(results_dir, filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"\n✅ Results saved to: {filepath}")

def run_model_benchmark(model_name, X_train_text, y_train, X_test_text, y_test, logger):
    logger.log(f"\n{'='*40}")
    logger.log(f" 🧠 Benchmarking Model: {model_name}")
    logger.log(f"{'='*40}")
    
    # 1. Generate Embeddings
    logger.log(f"   Connecting to Ollama: {model_name}")
    start_time = time.perf_counter()
    
    try:
        logger.log("      Embedding Train Set...")
        X_train_vectors = get_ollama_embeddings(X_train_text, model_name)
        
        logger.log("      Embedding Test Set...")
        X_test_vectors = get_ollama_embeddings(X_test_text, model_name)

    except Exception as e:
        logger.log(f"\n❌ Failed to connect to Ollama. Ensure 'ollama serve' is running and '{model_name}' is pulled.")
        logger.log(f"Error: {e}")
        return None
    
    embed_time = time.perf_counter() - start_time
    dims = X_train_vectors.shape[1]
    logger.log(f"   Dimensions: {dims}")
    logger.log(f"   Embedding Time: {embed_time:.2f}s")
    
    model_results = {}

    # Method 1: k-NN Sweep (K=3 to 15)
    best_knn_acc = -1
    best_k = -1
    best_knn_results = {}
    
    logger.log(f"   > Sweeping k-NN (k=3..15)...")
    
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
    
    # Print Detailed Stats for BOTH k-NN and Centroid
    logger.log(f"   > Best k-NN (K={best_knn_results['k']}) Acc: {best_knn_results['accuracy']:.4f} | Prec: {best_knn_results['precision']:.4f} | Rec: {best_knn_results['recall']:.4f}")
    logger.log(f"     Confusion Matrix:\n{best_knn_results['cm']}")
    
    c_res = model_results['centroid']
    logger.log(f"   > Centroid Acc: {c_res['accuracy']:.4f} | Prec: {c_res['precision']:.4f} | Rec: {c_res['recall']:.4f}")
    logger.log(f"     Confusion Matrix:\n{c_res['cm']}")

    return {
        'model': model_name,
        'dims': dims,
        'results': model_results
    }

def run_benchmark():
    logger = Logger()
    logger.log("=== Multi-Model Vector Similarity Benchmark (K-Sweep 3..15) ===")
    
    # Load Data
    logger.log(f"📂 Loading Data...")
    X_train_text, y_train = load_data(TRAIN_PATH)
    X_test_text, y_test = load_data(TEST_PATH)
    logger.log(f"   Train: {len(X_train_text)} | Test: {len(X_test_text)}")

    all_benchmarks = []
    
    for model in MODELS:
        res = run_model_benchmark(model, X_train_text, y_train, X_test_text, y_test, logger)
        if res:
            all_benchmarks.append(res)
            
    # Final Summary Table
    logger.log(f"\n{'='*100}")
    logger.log(f" 📊 FINAL COMPARISON SUMMARY")
    logger.log(f"{'='*100}")
    logger.log(f"{'Model':<25} {'Dims':<6} | {'Best K':<6} {'k-NN Acc':<10} {'k-NN F1':<10} | {'Centroid Acc':<12}")
    logger.log("-" * 100)
    
    for b in all_benchmarks:
        m = b['model']
        d = b['dims']
        k_res = b['results']['knn']
        c_res = b['results']['centroid']
        
        logger.log(f"{m:<25} {d:<6} | {k_res['k']:<6} {k_res['accuracy']:<10.4f} {k_res['f1']:<10.4f} | {c_res['accuracy']:<12.4f}")
        
    save_results_to_file(logger.get_content())

if __name__ == "__main__":
    run_benchmark()
