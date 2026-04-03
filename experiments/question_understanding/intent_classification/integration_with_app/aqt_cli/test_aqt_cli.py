#!/usr/bin/env python3
"""
AQT CLI testing tool for the current app runtime.
This script intentionally depends on src.services.aqt.
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
API_APP_ROOT = PROJECT_ROOT / "apps" / "api"
INTEGRATION_DIR = Path(__file__).resolve().parents[1]
INTENT_EXPERIMENT_DIR = Path(__file__).resolve().parents[2]
SYNTHETIC_CASES_PATH = INTENT_EXPERIMENT_DIR / "data" / "aqt_synthetic_phase1_cases.json"
OUTPUT_DIR = INTEGRATION_DIR / "results" / "aqt_cli"

# Add the canonical backend app to path.
sys.path.insert(0, str(API_APP_ROOT))


def _configure_stdout_utf8() -> None:
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    try:
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


_configure_stdout_utf8()

from src.services.aqt import transform_query



def _load_synthetic_cases() -> list[dict]:
    if not SYNTHETIC_CASES_PATH.exists():
        return []
    try:
        data = json.loads(SYNTHETIC_CASES_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Warning: failed to load synthetic cases: {exc}")
        return []
    if not isinstance(data, list):
        print(f"Warning: synthetic dataset must be a list: {SYNTHETIC_CASES_PATH}")
        return []

    normalized: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        if not question:
            continue
        normalized.append(
            {
                "id": str(item.get("id", "")),
                "source": str(item.get("source", "synthetic")),
                "question": question,
                "description": str(item.get("description", "Synthetic case")),
            }
        )
    return normalized


def test_aqt_scenarios():
    """Test AQT with various query scenarios"""
    
    test_cases = [
        # Drug information queries
        {
            "question": "Paracetamol 500mg Ã Â¸Â­Ã Â¸Â¢Ã Â¸Â¹Ã Â¹Ë†Ã Â¹Æ’Ã Â¸â„¢Ã Â¸Å¡Ã Â¸Â±Ã Â¸ÂÃ Â¸Å Ã Â¸ÂµÃ Â¸Â¢Ã Â¸Â²Ã Â¸Â«Ã Â¸Â¥Ã Â¸Â±Ã Â¸ÂÃ Â¹â€žÃ Â¸Â«Ã Â¸Â¡",
            "description": "Drug verification with NLEM"
        },
        {
            "question": "Ã Â¸â€šÃ Â¹â€°Ã Â¸Â­Ã Â¸Â¡Ã Â¸Â¹Ã Â¸Â¥Ã Â¸Â¢Ã Â¸Â² Paracetamol",
            "description": "General drug information"
        },
        {
            "question": "TMTID 662401 Ã Â¸â€žÃ Â¸Â·Ã Â¸Â­Ã Â¸Â¢Ã Â¸Â²Ã Â¸Â­Ã Â¸Â°Ã Â¹â€žÃ Â¸Â£",
            "description": "Exact TMTID lookup"
        },
        
        # Manufacturer queries
        {
            "question": "MACROPHAR Ã Â¸Å“Ã Â¸Â¥Ã Â¸Â´Ã Â¸â€¢Ã Â¸Â¢Ã Â¸Â²Ã Â¸Å¾Ã Â¸Â²Ã Â¸Â£Ã Â¸Â²Ã Â¹â€žÃ Â¸Â«Ã Â¸Â¡",
            "description": "Manufacturer verification"
        },
        {
            "question": "Ã Â¸Â£Ã Â¸Â²Ã Â¸Â¢Ã Â¸Å Ã Â¸Â·Ã Â¹Ë†Ã Â¸Â­Ã Â¸Â¢Ã Â¸Â²Ã Â¸â€šÃ Â¸Â­Ã Â¸â€¡Ã Â¸Å“Ã Â¸Â¹Ã Â¹â€°Ã Â¸Å“Ã Â¸Â¥Ã Â¸Â´Ã Â¸â€¢ GPO",
            "description": "Manufacturer listing"
        },
        
        # Count queries
        {
            "question": "Ã Â¸Ë†Ã Â¸Â³Ã Â¸â„¢Ã Â¸Â§Ã Â¸â„¢Ã Â¸Â¢Ã Â¸Â²Ã Â¸â€”Ã Â¸ÂµÃ Â¹Ë†Ã Â¸Â¡Ã Â¸ÂµÃ Â¸ÂªÃ Â¹Ë†Ã Â¸Â§Ã Â¸â„¢Ã Â¸Å“Ã Â¸ÂªÃ Â¸Â¡ Paracetamol",
            "description": "Count with substance"
        },
        {
            "question": "how many drugs does TMT produce",
            "description": "English count query"
        },
        
        # Abstract queries
        {
            "question": "Ã Â¸â€šÃ Â¹â€°Ã Â¸Â­Ã Â¸Â¡Ã Â¸Â¹Ã Â¸Â¥Ã Â¸Â¢Ã Â¸Â²Ã Â¸â€”Ã Â¸Â±Ã Â¹Ë†Ã Â¸Â§Ã Â¹â€žÃ Â¸â€º",
            "description": "Abstract general query"
        },
        {
            "question": "overview of pain medications",
            "description": "English abstract query"
        },
        
        # Complex queries
        {
            "question": "Paracetamol 500mg Ã Â¸â€šÃ Â¸Â­Ã Â¸â€¡Ã Â¸Å“Ã Â¸Â¹Ã Â¹â€°Ã Â¸Å“Ã Â¸Â¥Ã Â¸Â´Ã Â¸â€¢Ã Â¹Æ’Ã Â¸â€Ã Â¸Å¡Ã Â¹â€°Ã Â¸Â²Ã Â¸â€¡",
            "description": "Complex: drug + strength + manufacturer"
        },
        {
            "question": "Ã Â¹â‚¬Ã Â¸â€ºÃ Â¸Â£Ã Â¸ÂµÃ Â¸Â¢Ã Â¸Å¡Ã Â¹â‚¬Ã Â¸â€”Ã Â¸ÂµÃ Â¸Â¢Ã Â¸Å¡Ã Â¸â€šÃ Â¹â€°Ã Â¸Â­Ã Â¸Â¡Ã Â¸Â¹Ã Â¸Â¥Ã Â¸Â£Ã Â¸Â°Ã Â¸Â«Ã Â¸Â§Ã Â¹Ë†Ã Â¸Â²Ã Â¸â€¡ Paracetamol Ã Â¸ÂÃ Â¸Â±Ã Â¸Å¡ Ibuprofen",
            "description": "Comparison query"
        }
    ]

    synthetic_cases = _load_synthetic_cases()
    if synthetic_cases:
        print(f"Loaded synthetic test cases: {len(synthetic_cases)} from {SYNTHETIC_CASES_PATH}")
        test_cases.extend(synthetic_cases)

    print("=" * 80)
    print("AQT CLI Testing Tool")
    print("=" * 80)
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['description']}")
        if test_case.get("source"):
            case_id = test_case.get("id", "")
            source = test_case.get("source", "base")
            print(f"Source: {source}" + (f" | Case ID: {case_id}" if case_id else ""))
        print(f"Question: {test_case['question']}")
        print("-" * 60)
        
        try:
            # Transform query using AQT
            result = transform_query(test_case['question'])
            
            # Get IntentBundle from result
            bundle = result.intent_bundle
            if not bundle:
                print("No IntentBundle generated (INTENT_V2_ENABLED may be False)")
                continue
            
            # Convert to dict for JSON serialization
            metadata = bundle.get('metadata', {})
            result_dict = {
                "question": test_case['question'],
                "description": test_case['description'],
                "source": test_case.get('source', 'base'),
                "id": test_case.get('id', ''),
                "action_intent": bundle.get('action_intent', 'unknown'),
                "topics_intents": bundle.get('topics_intents', []),
                "slots": bundle.get('slots', []),
                "slots_multi": metadata.get('ner_slots_multi', {}),
                "retrieval_plan": bundle.get('adaptive_retrieval_weights', {}),
                "metadata": metadata
            }
            
            # Print results
            print(f"Action Intent: {bundle.get('action_intent', 'unknown')}")
            print(f"Topics Intents: {bundle.get('topics_intents', [])}")
            
            slots = bundle.get('slots', [])
            if slots:
                print("Extracted Slots:")
                for slot in slots:
                    print(f"   - {slot.get('name', 'unknown')}: '{slot.get('value', '')}' (confidence: {slot.get('confidence', 0):.2f}, source: {slot.get('source', 'unknown')})")
            else:
                print("Extracted Slots: None")
            
            retrieval_plan = bundle.get('adaptive_retrieval_weights', {})
            print(f"Retrieval Weights: Vector={retrieval_plan.get('vector_weight', 0):.2f}, Fulltext={retrieval_plan.get('fulltext_weight', 0):.2f}")
            print(f"Retrieval Mode: {retrieval_plan.get('retrieval_mode', 'unknown')}")
            
            # Get control features for entity metrics
            control_features = bundle.get('control_features', {})
            print(f"Entity Ratio: {control_features.get('entity_ratio', 0):.2f}")
            print(f"Is Abstract: {metadata.get('is_abstract', False)}")
            print(f"Token Count: {control_features.get('token_count', 0)}")
            print(f"Entity Token Count: {control_features.get('entity_token_count', 0)}")
            
            # Get legacy intent metadata
            print(f"Intent Confidence: {metadata.get('legacy_intent_confidence', 0):.2f}")
            print(f"Target Confidence: {metadata.get('legacy_target_confidence', 0):.2f}")
            print(f"Target Margin: {metadata.get('legacy_target_margin', 0):.2f}")
            print(f"Raw Intent: {metadata.get('legacy_raw_intent', 'unknown')}")
            print(f"Top Targets: {metadata.get('legacy_intent_top_targets', [])}")
            print(f"NER Available: {metadata.get('ner_available', False)}")
            if metadata.get('ner_error'):
                print(f"NER Error: {metadata.get('ner_error')}")
            
            # Print NER detail summary if available
            ner_slots_multi = metadata.get('ner_slots_multi', {})
            ner_sanitized = metadata.get('ner_sanitized', {})
            
            if metadata.get('ner_available', False):
                # Show multi-entity slots (new primary view)
                if ner_slots_multi:
                    print("NER Multi-Entity Slots:")
                    for key, values in ner_slots_multi.items():
                        if values:
                            print(f"   - {key}: {values}")
                else:
                    print("NER Multi-Entity Slots: {}")
                
                # Show sanitized stats
                if ner_sanitized:
                    stats = ner_sanitized.get('slot_multi_count', {})
                    if stats:
                        print(f"NER Multi-Entity Counts: {stats}")
                        
                # Show entities
                if ner_sanitized.get('entity_count_after', 0) > 0:
                    print(f"NER Entities Found: {ner_sanitized['entity_count_after']} items")
            else:
                print("NER: Not available")
            
            results.append(result_dict)
            
        except Exception as e:
            print(f"Error: {str(e)}")
            results.append({
                "question": test_case['question'],
                "description": test_case['description'],
                "source": test_case.get('source', 'base'),
                "id": test_case.get('id', ''),
                "error": str(e)
            })
    
    # Save results to file
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "aqt_out.json"
    output_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    
    print("\n" + "=" * 80)
    print(f"Results saved to: {output_file}")
    print(f"Total test cases: {len(test_cases)}")
    print("=" * 80)

if __name__ == "__main__":
    test_aqt_scenarios()




