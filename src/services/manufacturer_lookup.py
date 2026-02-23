"""
Manufacturer Lookup Service.
Loads manufacturer names from JSON and provides efficient lookup.
"""
import json
import re
from pathlib import Path

# Path to manufacturer data
MANUFACTURER_JSON = Path(__file__).parent.parent.parent / "manufacturers.json"

# Singleton cache
_manufacturers_set: set[str] = None
_manufacturers_lower: set[str] = None


def _clean_manufacturer(name: str) -> str:
    """Clean manufacturer name for matching."""
    # Remove country suffix: "ABBOTT, U.S.A." -> "ABBOTT"
    cleaned = re.sub(r',\s*[A-Z\.]+$', '', name)
    # Remove parentheses content: "สยาม (Thailand)" -> "สยาม"
    cleaned = re.sub(r'\s*\([^)]*\)\s*', ' ', cleaned)
    # Normalize whitespace
    cleaned = ' '.join(cleaned.split())
    return cleaned.strip()


def load_manufacturers() -> set[str]:
    """Load manufacturers from JSON file (once at startup)."""
    global _manufacturers_set, _manufacturers_lower
    
    if _manufacturers_set is not None:
        return _manufacturers_set
    
    try:
        with open(MANUFACTURER_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        raw_list = data.get("manufacturers", [])
        
        # Clean and store both original and cleaned versions
        _manufacturers_set = set()
        _manufacturers_lower = set()
        
        for name in raw_list:
            if not name or not name.strip():
                continue
            
            # Skip hospitals (not real manufacturers)
            if "โรงพยาบาล" in name:
                continue
            
            # Store original
            _manufacturers_set.add(name)
            _manufacturers_lower.add(name.lower())
            
            # Also store cleaned version
            cleaned = _clean_manufacturer(name)
            if cleaned and cleaned != name:
                _manufacturers_set.add(cleaned)
                _manufacturers_lower.add(cleaned.lower())
        
        print(f"   Loaded {len(_manufacturers_set)} manufacturers")
        return _manufacturers_set
        
    except FileNotFoundError:
        print(f"   Manufacturer file not found: {MANUFACTURER_JSON}")
        _manufacturers_set = set()
        _manufacturers_lower = set()
        return _manufacturers_set


def find_manufacturer(query: str) -> str:
    """
    Find manufacturer name in query using exact token match.
    
    Args:
        query: User's question
    
    Returns:
        Matched manufacturer name or None
    """
    global _manufacturers_lower
    
    if _manufacturers_set is None:
        load_manufacturers()
    
    q_lower = query.lower()
    
    # 1. Try exact token match (fast)
    # Split on spaces and common separators
    tokens = re.split(r'[\s,./()]+', q_lower)
    
    for token in tokens:
        if len(token) < 2:
            continue
        if token in _manufacturers_lower:
            # Return original case version
            for manu in _manufacturers_set:
                if manu.lower() == token:
                    return manu
    
    # 2. Try substring match for multi-word manufacturers
    # Sort by length (longer first) to match "องค์การเภสัชกรรม" before "องค์การ"
    for manu in sorted(_manufacturers_set, key=len, reverse=True):
        if len(manu) > 3 and manu.lower() in q_lower:
            return manu
    
    return None


# Common aliases mapping
MANUFACTURER_ALIASES = {
    "gpo": "องค์การเภสัชกรรม",
    "องค์การ": "องค์การเภสัชกรรม",
    "ไฟเซอร์": "Pfizer",
    "โนวาร์ตีส": "Novartis",
}


def find_manufacturer_with_alias(query: str) -> str:
    """Find manufacturer with alias support."""
    q_lower = query.lower()
    
    # Check aliases first
    for alias, canonical in MANUFACTURER_ALIASES.items():
        if alias in q_lower:
            return canonical
    
    # Fall back to regular lookup
    return find_manufacturer(query)


if __name__ == "__main__":
    # Test
    load_manufacturers()
    
    test_queries = [
        "ยา Paracetamol ของ Bayer",
        "ยาขององค์การเภสัชกรรม",
        "GPO ผลิตยาอะไรบ้าง",
        "ไฟเซอร์มียาอะไร",
    ]
    
    print("\nTesting manufacturer lookup:")
    for q in test_queries:
        result = find_manufacturer_with_alias(q)
        print(f"   '{q}' -> {result}")
