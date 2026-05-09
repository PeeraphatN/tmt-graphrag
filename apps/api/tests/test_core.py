"""
Unit tests for pure utility functions in the GraphRAG pipeline.
No Neo4j, Ollama, or model weights required.
"""
import re
import sys
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Patch module-level side effects before importing aqt / pipeline
# ---------------------------------------------------------------------------

# load_manufacturers reads a JSON file at import time — stub it out
with patch("src.services.manufacturer_lookup.load_manufacturers", return_value=None), \
     patch("src.services.manufacturer_lookup.find_manufacturer_with_alias", return_value=None):
    from src.services.aqt import (
        detect_strategy,
        detect_compare,
        detect_unknown,
        extract_nlem_filter,
        extract_nlem_category,
        extract_tmtid,
        _sanitize_ner_slot_value,
    )


# ---------------------------------------------------------------------------
# detect_strategy
# ---------------------------------------------------------------------------

class TestDetectStrategy:
    def test_count_english(self):
        assert detect_strategy("How many drugs does Pfizer make?") == "count"

    def test_count_thai(self):
        assert detect_strategy("จำนวนยาของ Pfizer มีกี่ตัว") == "count"

    def test_list_english(self):
        assert detect_strategy("List all drugs by AstraZeneca") == "list"

    def test_list_thai(self):
        assert detect_strategy("รายการยาทั้งหมด") == "list"

    def test_verify_english(self):
        assert detect_strategy("Is paracetamol in the NLEM list?") == "verify"

    def test_verify_thai(self):
        assert detect_strategy("ยานี้อยู่ในบัญชียาหลักไหม") == "verify"

    def test_retrieve_default(self):
        assert detect_strategy("paracetamol 500mg") == "retrieve"

    def test_count_takes_priority_over_list(self):
        # "count" should win when both patterns match
        result = detect_strategy("how many items are in this list")
        assert result == "count"


# ---------------------------------------------------------------------------
# detect_compare
# ---------------------------------------------------------------------------

class TestDetectCompare:
    def test_vs_abbreviation(self):
        assert detect_compare("amoxicillin vs ampicillin") is True

    def test_compare_keyword(self):
        assert detect_compare("compare paracetamol and ibuprofen") is True

    def test_thai_compare(self):
        assert detect_compare("เปรียบเทียบยาสองชนิด") is True

    def test_no_compare(self):
        assert detect_compare("what is paracetamol?") is False


# ---------------------------------------------------------------------------
# extract_nlem_filter
# ---------------------------------------------------------------------------

class TestExtractNlemFilter:
    def test_nlem_keyword(self):
        assert extract_nlem_filter("is this drug in the NLEM list?") is True

    def test_thai_keyword(self):
        assert extract_nlem_filter("ยานี้อยู่ในบัญชียาหลักไหม") is True

    def test_reimburse_keyword(self):
        assert extract_nlem_filter("can this be reimbursed?") is True

    def test_no_nlem(self):
        assert extract_nlem_filter("tell me about paracetamol") is None


# ---------------------------------------------------------------------------
# extract_nlem_category
# ---------------------------------------------------------------------------

class TestExtractNlemCategory:
    def test_category_thai_ก(self):
        assert extract_nlem_category("ยาในบัญชี ก") == "ก"

    def test_category_english_b(self):
        assert extract_nlem_category("drugs in category B") == "ข"

    def test_no_category(self):
        assert extract_nlem_category("paracetamol") is None


# ---------------------------------------------------------------------------
# extract_tmtid
# ---------------------------------------------------------------------------

class TestExtractTmtid:
    def test_explicit_prefix(self):
        assert extract_tmtid("TMTID: 1234567") == "1234567"

    def test_bare_id(self):
        assert extract_tmtid("drug 123456") == "123456"

    def test_no_id(self):
        assert extract_tmtid("what is paracetamol?") is None

    def test_short_number_ignored(self):
        # 5-digit numbers should not match the bare ID pattern (min 6 digits)
        assert extract_tmtid("There are 12345 drugs") is None


# ---------------------------------------------------------------------------
# _sanitize_ner_slot_value
# ---------------------------------------------------------------------------

class TestSanitizeNerSlotValue:
    def test_empty_value_rejected(self):
        result, reason = _sanitize_ner_slot_value("test question", "drug", "")
        assert result is None
        assert reason == "empty"

    def test_noise_term_rejected(self):
        result, reason = _sanitize_ner_slot_value("test question", "drug", "drug")
        assert result is None
        assert reason == "noise_term"

    def test_english_question_fragment_rejected(self):
        result, reason = _sanitize_ner_slot_value(
            "how many tablets", "drug", "how many tablets"
        )
        assert result is None
        assert reason == "english_question"

    def test_valid_drug_name_accepted(self):
        question = "what is paracetamol 500mg?"
        result, reason = _sanitize_ner_slot_value(question, "drug", "paracetamol")
        assert result == "paracetamol"
