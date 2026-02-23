# Intent V2 Branch Strategy

## Goals

1. Keep `master` as frozen baseline reference.
2. Use `integration/intent-v2-hybrid` as the main development line for this initiative.
3. Use short-lived `task/*` branches, merge back to `integration/*` quickly.
4. Prefer feature flags over long-lived feature branches.
5. Keep `experiment/*` as benchmark/evidence branches only (not primary product development).

## Branch Roles

- `master`
  - Baseline reference only.
  - No direct experiment development.
- `integration/intent-v2-hybrid`
  - Main branch for Intent V2 + adaptive hybrid implementation.
  - Receives frequent merges from `task/*`.
- `task/*`
  - Small scoped work units (1 feature/fix each).
  - Open PR to `integration/intent-v2-hybrid`.
  - Delete branch after merge.
- `experiment/*`
  - Reproducible benchmark and research history.
  - Not the primary integration path.

## Working Rules

1. Create task branch from integration:
   - `git checkout integration/intent-v2-hybrid`
   - `git checkout -b task/<short-scope>`
2. Keep tasks small and reviewable.
3. Merge task -> integration as soon as tests pass.
4. Rebase/merge integration into task frequently to avoid drift.
5. Do not merge experiment branches directly into `master`.

## Feature Flag Policy

Use runtime flags to stage rollout safely inside integration:

- `INTENT_V2_ENABLED`
- `INTENT_V2_USE_NER`
- `INTENT_V2_ADAPTIVE_PLANNER`

Recommended rollout:

1. Shadow mode (`enabled=false`, logging only)
2. Partial path (`enabled=true` for selected traffic/tests)
3. Full path (default true after metrics pass)

## Current Seed Branches

- Integration branch: `integration/intent-v2-hybrid`
- Task starter branch: `task/intent-v2-feature-flag-scaffold`

