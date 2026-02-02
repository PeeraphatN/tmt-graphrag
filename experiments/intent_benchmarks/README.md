# Intent Classification Benchmark

Evaluating different approaches for intent classification in the TMT chatbot.

## Quick Results

| Method | Accuracy | F1 | Latency | Status |
|--------|----------|----|---------| ------ |
| **Vector Sim (bge-m3)** | 93.33% | 0.94 | 0.2ms | ✅ Done |
| Zero-shot LLM | - | - | - | ⏳ Pending |
| Few-shot LLM | - | - | - | ⏳ Pending |

## Dataset

- **Train**: 90 samples (`benchmark_train.json`)
- **Test**: 30 samples (`benchmark_test.json`)
- **Classes**: manufacturer, ingredient, nlem, hierarchy, formula, general

## Files

| File | Description |
|------|-------------|
| `benchmark_intent.py` | Vector Similarity benchmark script |
| `benchmark_train.json` | Training data |
| `benchmark_test.json` | Evaluation data (diversified) |
| `RESULTS_FINAL_VECTOR_SIM.txt` | Detailed results |

## Run Benchmark

```bash
python experiments/intent_benchmarks/benchmark_intent.py
```

## Key Findings

1. **Vector Similarity achieves 93%+ accuracy** with clean, well-structured data
2. **Data quality matters**: +13% improvement after removing ambiguous questions
3. **Ingredient class** acts as "catch-all" for ambiguous queries
