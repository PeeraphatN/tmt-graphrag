"""
Result Cache for GraphRAG System.
Supports both Exact Match and Semantic Caching.
"""
import hashlib
import numpy as np
from typing import Optional, Any, Tuple
from cachetools import TTLCache

# ==============================
# CACHE INSTANCES
# ==============================

# TTL = 3600 seconds (1 hour)
_query_cache: TTLCache = TTLCache(maxsize=1000, ttl=3600)
_answer_cache: TTLCache = TTLCache(maxsize=500, ttl=3600)

# Semantic cache stores: {hash_key: (question_text, embedding, answer)}
_semantic_cache: dict[str, Tuple[str, list[float], str]] = {}

# Stats tracking
_cache_stats = {
    "query_hits": 0,
    "query_misses": 0,
    "answer_hits_exact": 0,
    "answer_hits_semantic": 0,
    "answer_misses": 0,
}

# Semantic similarity threshold
SIMILARITY_THRESHOLD = 0.80


# ==============================
# UTILITY FUNCTIONS
# ==============================

def _hash_key(text: str) -> str:
    """สร้าง MD5 Hash Key จาก Text"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """คำนวณ Cosine Similarity ระหว่าง 2 vectors"""
    a = np.array(vec1)
    b = np.array(vec2)
    
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return float(dot_product / (norm_a * norm_b))


# ==============================
# QUERY TRANSFORM CACHE (Layer 1) - Exact Match
# ==============================

def get_cached_query(question: str) -> Optional[dict]:
    """ดึง Query Transform result จาก cache (Exact Match)"""
    key = _hash_key(question.strip().lower())
    result = _query_cache.get(key)
    
    if result is not None:
        _cache_stats["query_hits"] += 1
    else:
        _cache_stats["query_misses"] += 1
    
    return result


def set_cached_query(question: str, query_obj: Any):
    """บันทึก Query Transform result ลง cache"""
    key = _hash_key(question.strip().lower())
    _query_cache[key] = {
        "query": query_obj.query,
        "target_type": query_obj.target_type,
        "nlem_filter": query_obj.nlem_filter,
        "nlem_category": query_obj.nlem_category,
        "manufacturer_filter": query_obj.manufacturer_filter,
    }


# ==============================
# ANSWER CACHE (Layer 3) - Semantic Match
# ==============================

def get_cached_answer_semantic(question: str, question_embedding: list[float]) -> Tuple[Optional[str], bool]:
    """
    ดึงคำตอบจาก cache โดยใช้ Semantic Similarity.
    
    Args:
        question: คำถาม User
        question_embedding: Embedding ของคำถาม
    
    Returns:
        Tuple of (answer or None, is_semantic_hit)
    """
    # 1. Try exact match first (fastest)
    key = _hash_key(question.strip().lower())
    exact_result = _answer_cache.get(key)
    
    if exact_result is not None:
        _cache_stats["answer_hits_exact"] += 1
        return exact_result, False  # False = exact match, not semantic
    
    # 2. Try semantic match
    best_similarity = 0.0
    best_answer = None
    best_question = None
    
    for cached_key, (cached_q, cached_emb, cached_answer) in _semantic_cache.items():
        similarity = _cosine_similarity(question_embedding, cached_emb)
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_answer = cached_answer
            best_question = cached_q
    
    if best_similarity >= SIMILARITY_THRESHOLD:
        _cache_stats["answer_hits_semantic"] += 1
        print(f"   🔍 Semantic match: '{best_question[:30]}...' (similarity: {best_similarity:.2%})")
        return best_answer, True  # True = semantic match
    
    print(f"   Debug: Best semantic match was {best_similarity:.2%} (Threshold: {SIMILARITY_THRESHOLD})")
    _cache_stats["answer_misses"] += 1
    return None, False


def set_cached_answer_semantic(question: str, question_embedding: list[float], answer: str):
    """บันทึกคำตอบพร้อม Embedding ลง cache"""
    key = _hash_key(question.strip().lower())
    
    # Store in exact match cache (for fast lookup)
    _answer_cache[key] = answer
    
    # Store in semantic cache (for similarity search)
    _semantic_cache[key] = (question, question_embedding, answer)


# ==============================
# LEGACY FUNCTIONS (Backward Compatibility)
# ==============================

def get_cached_answer(question: str) -> Optional[str]:
    """Legacy: Exact match only (for backward compatibility)"""
    key = _hash_key(question.strip().lower())
    return _answer_cache.get(key)


def set_cached_answer(question: str, answer: str):
    """Legacy: Set without embedding (exact match only)"""
    key = _hash_key(question.strip().lower())
    _answer_cache[key] = answer


# ==============================
# CACHE MANAGEMENT
# ==============================

def get_cache_stats() -> dict:
    """ดึงสถิติการใช้งาน cache"""
    total_answer_hits = _cache_stats["answer_hits_exact"] + _cache_stats["answer_hits_semantic"]
    total_answer_requests = total_answer_hits + _cache_stats["answer_misses"]
    
    return {
        "query_cache": {
            "size": len(_query_cache),
            "maxsize": _query_cache.maxsize,
            "hits": _cache_stats["query_hits"],
            "misses": _cache_stats["query_misses"],
        },
        "answer_cache": {
            "size": len(_answer_cache),
            "semantic_entries": len(_semantic_cache),
            "hits_exact": _cache_stats["answer_hits_exact"],
            "hits_semantic": _cache_stats["answer_hits_semantic"],
            "misses": _cache_stats["answer_misses"],
            "hit_rate": f"{(total_answer_hits / total_answer_requests * 100):.1f}%" if total_answer_requests > 0 else "N/A",
        },
    }


def clear_all_caches():
    """ล้าง cache ทั้งหมด"""
    _query_cache.clear()
    _answer_cache.clear()
    _semantic_cache.clear()
    
    for key in _cache_stats:
        _cache_stats[key] = 0
    
    print("🗑️ All caches cleared.")

