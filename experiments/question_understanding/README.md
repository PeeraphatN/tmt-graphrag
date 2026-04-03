# Question Understanding

This area is the canonical home for experiments that replace LLM-only question understanding with intent classification and NER.

Current split:

- `intent_classification/`: standalone intent benchmarks, dataset builders, and app-integration checks
- `ner_finetuning/`: standalone NER data generation/fine-tuning flow plus optional app-integration checks

Independence rule:

- standalone experiment scripts must not import `src/` or rely on the main app runtime
- anything that intentionally exercises AQT or other app logic must live under each experiment's `integration_with_app/` subtree
