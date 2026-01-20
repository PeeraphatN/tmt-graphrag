"""
Result Cache for GraphRAG System.
Uses TTLCache for time-based expiration of cached results.
"""
import hashlib
from typing import Optional, Any
from cachetools import TTLCache

# ==============================
# CACHE INSTANCES
# ==============================

# TTL = 3600 seconds (1 hour)
# Max 1000 query transforms, 500 search results, 500 answers

_query_cache: TTLCache = TTLCache(maxsize=1000, ttl=3600)
_search_cache: TTLCache = TTLCache(maxsize=500, ttl=3600)
_answer_cache: TTLCache = TTLCache(maxsize=500, ttl=3600)

# Stats tracking
_cache_stats = {
    "query_hits": 0,
    "query_misses": 0,
    "search_hits": 0,
    "search_misses": 0,
    "answer_hits": 0,
    "answer_misses": 0,
}


# ==============================
# UTILITY FUNCTIONS
# ==============================

def _hash_key(text: str) -> str:
    """สร้าง MD5 Hash Key จาก Text"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


# ==============================
# QUERY TRANSFORM CACHE (Layer 1)
# ==============================

def get_cached_query(question: str) -> Optional[dict]:
    """
    ดึง Query Transform result จาก cache.
    
    Args:
        question: คำถาม User
    
    Returns:
        GraphRAGQuery dict หรือ None ถ้าไม่มี
    """
    key = _hash_key(question.strip().lower())
    result = _query_cache.get(key)
    
    if result is not None:
        _cache_stats["query_hits"] += 1
    else:
        _cache_stats["query_misses"] += 1
    
    return result


def set_cached_query(question: str, query_obj: Any):
    """
    บันทึก Query Transform result ลง cache.
    
    Args:
        question: คำถาม User
        query_obj: GraphRAGQuery object
    """
    key = _hash_key(question.strip().lower())
    # Store as dict for serialization
    _query_cache[key] = {
        "query": query_obj.query,
        "target_type": query_obj.target_type,
        "nlem_filter": query_obj.nlem_filter,
        "nlem_category": query_obj.nlem_category,
        "manufacturer_filter": query_obj.manufacturer_filter,
    }


# ==============================
# SEARCH RESULT CACHE (Layer 2)
# ==============================

def get_cached_search(query_key: str) -> Optional[dict]:
    """
    ดึง Search result จาก cache.
    
    Args:
        query_key: Hash key ของ query object
    
    Returns:
        Search results dict หรือ None
    """
    result = _search_cache.get(query_key)
    
    if result is not None:
        _cache_stats["search_hits"] += 1
    else:
        _cache_stats["search_misses"] += 1
    
    return result


def set_cached_search(query_key: str, results: dict):
    """บันทึก Search result ลง cache."""
    _search_cache[query_key] = results


# ==============================
# FINAL ANSWER CACHE (Layer 3)
# ==============================

def get_cached_answer(question: str) -> Optional[str]:
    """
    ดึงคำตอบสุดท้ายจาก cache.
    
    Args:
        question: คำถาม User
    
    Returns:
        Answer string หรือ None
    """
    key = _hash_key(question.strip().lower())
    result = _answer_cache.get(key)
    
    if result is not None:
        _cache_stats["answer_hits"] += 1
    else:
        _cache_stats["answer_misses"] += 1
    
    return result


def set_cached_answer(question: str, answer: str):
    """บันทึกคำตอบสุดท้ายลง cache."""
    key = _hash_key(question.strip().lower())
    _answer_cache[key] = answer


# ==============================
# CACHE MANAGEMENT
# ==============================

def get_cache_stats() -> dict:
    """
    ดึงสถิติการใช้งาน cache.
    
    Returns:
        dict พร้อม hit/miss stats และ current size
    """
    return {
        "query_cache": {
            "size": len(_query_cache),
            "maxsize": _query_cache.maxsize,
            "hits": _cache_stats["query_hits"],
            "misses": _cache_stats["query_misses"],
        },
        "search_cache": {
            "size": len(_search_cache),
            "maxsize": _search_cache.maxsize,
            "hits": _cache_stats["search_hits"],
            "misses": _cache_stats["search_misses"],
        },
        "answer_cache": {
            "size": len(_answer_cache),
            "maxsize": _answer_cache.maxsize,
            "hits": _cache_stats["answer_hits"],
            "misses": _cache_stats["answer_misses"],
        },
    }


def clear_all_caches():
    """ล้าง cache ทั้งหมด."""
    _query_cache.clear()
    _search_cache.clear()
    _answer_cache.clear()
    
    # Reset stats
    for key in _cache_stats:
        _cache_stats[key] = 0
    
    print("🗑️ All caches cleared.")
