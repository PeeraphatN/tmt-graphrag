# Intent Classification App Integration Checks

This subtree is intentionally allowed to import the canonical backend app under `apps/api/src/` so we can validate how the standalone intent work behaves when routed through the current application stack.

Keep here:

- AQT CLI smoke checks
- shadow comparisons against runtime intent/AQT behavior
- synthetic regression scripts that depend on app schemas or runtime services

Do not keep here:

- standalone dataset builders
- embedding benchmarks
- pure centroid or LLM baselines

Outputs from this subtree should go into `integration_with_app/results/` and stay out of Git.

Path rule:

- insert `apps/api` into `sys.path`
- then import backend modules as `from src...`
- do not point these scripts back to the legacy repo-root `src/` runtime
