# NER App Integration Checks

The scripts in this subtree intentionally exercise the canonical backend app in `apps/api` after standalone NER data generation has already completed.

Use this area for:

- AQT sanity checks over generated NER-style questions
- future end-to-end checks that combine generated prompts with app runtime behavior

Keep the core standalone flow in `run/` free from `src/` imports.
