import re

# ==============================
# Query Preprocessing & Classification
# ==============================

# Thai stopwords (obvious ones)
THAI_STOPWORDS = {
    'อะไร', 'บ้าง', 'ใคร', 'ผลิต', 'เป็น', 'ยังไง', 'อย่างไร', 'ไหน', 'ทำไม',
    'มี', 'ได้', 'และ', 'หรือ', 'ที่', 'ของ', 'ใน', 'จาก', 'ให้', 'กับ',
    'แสดง', 'บอก', 'หา', 'ค้นหา', 'ดู', 'ต้องการ', 'ขอ', 'ช่วย'
}

# Query framing tokens that should not dominate fulltext intent terms.
FULLTEXT_QUERY_NOISE_EN = {
    "what", "which", "how", "many", "list", "show", "find", "search", "lookup",
    "overview", "information", "info", "data", "about", "does", "is", "are",
    "do", "did", "by", "of", "the", "a", "an", "products", "product",
    "drugs", "drug", "medications", "medication", "medicine", "medicines",
    "manufacturer", "manufacturers", "containing", "contain",
}
FULLTEXT_QUERY_NOISE_TH = {
    "ข้อมูล", "รายการ", "รายชื่อ", "แสดง", "ค้นหา", "หา", "ขอ", "ช่วย",
    "อะไร", "อย่างไร", "ยังไง", "กี่", "จำนวน", "เปรียบเทียบ", "ระหว่าง",
    "ยา",
}

# Lucene special characters that need escaping
LUCENE_SPECIAL_CHAR_SET = set(list(r'+-!(){}[]^"~*?:\/'))

def escape_lucene_char(char: str) -> str:
    return f'\\{char}' if char in LUCENE_SPECIAL_CHAR_SET else char

def extract_keywords(question: str) -> list[str]:
    """
    Extract high-signal tokens from question.
    Returns tokens that are:
    - English words/drug names
    - Numbers and units (mg, g, mL, etc.)
    - Patterns like "1 g/20 mL"
    """
    keywords = []
    
    # Pattern for units with numbers: "1 g/20 mL", "500 mg"
    unit_pattern = r'\d+\.?\d*\s*(?:mg|g|mL|mcg|IU|%|ml|MG|G|ML)(?:/\d+\.?\d*\s*(?:mg|g|mL|mcg|IU|%|ml|MG|G|ML))?'
    unit_matches = re.findall(unit_pattern, question, re.IGNORECASE)
    keywords.extend(unit_matches)
    
    # Pattern for English words (drug names, etc.)
    english_pattern = r'[A-Za-z][A-Za-z0-9\-]*[A-Za-z0-9]|[A-Za-z]'
    english_matches = re.findall(english_pattern, question)
    # Filter out very short words (likely not drug names)
    keywords.extend([w for w in english_matches if len(w) >= 2])
    
    # Pattern for numbers that might be important (e.g., dosage)
    number_pattern = r'\d+\.?\d*'
    numbers = re.findall(number_pattern, question)
    # Only include if not already part of unit pattern
    for num in numbers:
        if not any(num in unit for unit in unit_matches):
            keywords.append(num)
    
    # Extract Thai words that are not stopwords
    thai_pattern = r'[\u0E00-\u0E7F]+'
    thai_words = re.findall(thai_pattern, question)
    meaningful_thai = [w for w in thai_words if w not in THAI_STOPWORDS and len(w) > 1]
    keywords.extend(meaningful_thai)
    
    # Deduplicate while preserving order
    seen = set()
    unique_keywords = []
    for kw in keywords:
        kw_lower = kw.lower() if kw.isascii() else kw
        if kw_lower not in seen:
            seen.add(kw_lower)
            unique_keywords.append(kw)
    
    return unique_keywords


def has_english_or_numbers(question: str) -> bool:
    """Check if question contains English letters or numbers."""
    return bool(re.search(r'[A-Za-z0-9]', question))


def _is_noise_keyword(keyword: str) -> bool:
    token = keyword.strip().lower()
    if not token:
        return True
    if token in FULLTEXT_QUERY_NOISE_EN:
        return True
    if token in FULLTEXT_QUERY_NOISE_TH:
        return True
    return False


def _keyword_score(keyword: str) -> int:
    """
    Rank fulltext terms by specificity.
    Higher score = better anchor candidate.
    """
    token = keyword.strip().lower()
    score = 0
    if any(ch.isdigit() for ch in token):
        score += 3
    if re.search(r'[A-Za-z]', token):
        score += 2
    if re.search(r'[\u0E00-\u0E7F]', token):
        score += 2
    score += min(4, len(token) // 4)
    return score


def sanitize_fulltext_query(question: str) -> tuple[str, bool]:
    """
    Sanitize question for Lucene fulltext query.
    
    Returns:
        tuple: (sanitized_query, should_use_fulltext)
        - sanitized_query: AND-joined escaped tokens
        - should_use_fulltext: False if pure Thai with no English/numbers
    """
    # Extract and clean candidate keywords.
    keywords = []
    seen = set()
    for raw_kw in extract_keywords(question):
        kw = str(raw_kw or "").strip()
        if not kw:
            continue
        if _is_noise_keyword(kw):
            continue
        key = kw.lower()
        if key in seen:
            continue
        seen.add(key)
        keywords.append(kw)

    # Fallback: recover lexical terms directly from text if extraction was too strict.
    if not keywords:
        raw_tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-\./%]*|[\u0E00-\u0E7F]{2,}", question)
        for token in raw_tokens:
            if _is_noise_keyword(token):
                continue
            key = token.lower()
            if key in seen:
                continue
            seen.add(key)
            keywords.append(token)

    if not keywords:
        return "", False

    # Keep the most informative terms to avoid over-constrained lucene queries.
    keywords.sort(key=_keyword_score, reverse=True)
    selected = keywords[:4]

    escaped_keywords = []
    for kw in selected:
        escaped = ''.join(escape_lucene_char(c) for c in kw).strip()
        if not escaped:
            continue
        if " " in escaped:
            escaped = f"\"{escaped}\""
        escaped_keywords.append(escaped)

    if not escaped_keywords:
        return "", False

    # Use OR for recall; downstream ranking + filters handle precision.
    sanitized = ' OR '.join(escaped_keywords)
    return sanitized, True
