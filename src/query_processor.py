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


def sanitize_fulltext_query(question: str) -> tuple[str, bool]:
    """
    Sanitize question for Lucene fulltext query.
    
    Returns:
        tuple: (sanitized_query, should_use_fulltext)
        - sanitized_query: AND-joined escaped tokens
        - should_use_fulltext: False if pure Thai with no English/numbers
    """
    # Check if we should skip fulltext (pure Thai)
    if not has_english_or_numbers(question):
        return "", False
    
    # Extract keywords
    keywords = extract_keywords(question)
    
    if not keywords:
        return "", False
    
    # Escape special characters in each keyword
    escaped_keywords = []
    for kw in keywords:
        escaped = ''.join(escape_lucene_char(c) for c in kw)
        if escaped.strip():
            escaped_keywords.append(escaped)
    
    if not escaped_keywords:
        return "", False
    
    # Join with AND
    sanitized = ' AND '.join(escaped_keywords)
    return sanitized, True


def classify_question(question: str) -> str:
    """
    Classify question type using rule-based approach (no LLM).
    
    Returns one of:
    - 'manufacturer': บริษัท/ผู้ผลิต queries
    - 'ingredient': ส่วนผสม/active ingredient queries
    - 'formula': สูตร/formula comparison queries
    - 'hierarchy': ลำดับ/hierarchy queries
    - 'general': other queries
    """
    q_lower = question.lower()
    
    # Manufacturer patterns
    manufacturer_patterns = [
        r'บริษัท.*ผลิต', r'ผู้ผลิต', r'ใครผลิต', r'ผลิตโดย',
        r'manufacturer', r'บริษัท.*ยา', r'ยา.*บริษัท'
    ]
    for pattern in manufacturer_patterns:
        if re.search(pattern, question, re.IGNORECASE):
            return 'manufacturer'
    
    # Ingredient patterns
    ingredient_patterns = [
        r'ส่วนผสม', r'สาร.*ออกฤทธิ์', r'active.?ingredient',
        r'ประกอบด้วย', r'มี.*อะไร.*บ้าง', r'ingredient'
    ]
    for pattern in ingredient_patterns:
        if re.search(pattern, question, re.IGNORECASE):
            return 'ingredient'
    
    # Formula patterns
    formula_patterns = [
        r'สูตร.*เดียวกัน', r'เปรียบเทียบ.*สูตร', r'formula',
        r'สูตร.*ต่างกัน', r'same.*formula'
    ]
    for pattern in formula_patterns:
        if re.search(pattern, question, re.IGNORECASE):
            return 'formula'
    
    
    # Hierarchy patterns
    hierarchy_patterns = [
        r'hierarchy', r'ลำดับ', r'มาจาก.*สาร', r'vtm', r'gp', r'tp',
        r'subs', r'gpu', r'tpu', r'ระดับ'
    ]
    for pattern in hierarchy_patterns:
        if re.search(pattern, question, re.IGNORECASE):
            return 'hierarchy'
    
    return 'general'
