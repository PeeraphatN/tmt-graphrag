# Contributing

Thanks for picking up TMT GraphRAG. The repo is set up for trunk-based
solo work with the rails kept enforceable by tooling — please follow
the conventions below so the history stays clean enough for the next
person (or future you).

## Branch model

`master` is the only long-lived branch and is protected on origin
(require PR + block force pushes). All work happens on short-lived
feature branches that merge via PR + squash. Delete the branch
afterwards.

| Prefix | Use for |
|---|---|
| `feat/<topic>` | new features |
| `fix/<topic>`  | bug fixes |
| `chore/<topic>`| tooling, dependency bumps, refactors with no behaviour change |
| `docs/<topic>` | documentation only |
| `test/<topic>` | tests only |
| `exp/<topic>`  | research; canonical results merge to master under `experiments/...` |

Lifetime ≤ ~1 week. Long-lived feature branches diverge and get
expensive to merge — prefer feature flags (e.g. `INTENT_V2_ENABLED`,
`NLEM_QA_ENABLED`) over keeping incomplete work on a branch.

## Commits

Conventional Commits format with a body that explains *why*, not
*what* (the diff already shows what):

    <type>(<scope>): <subject>

    <why this change is needed; tradeoffs; constraints; prior incidents>

Types: `feat`, `fix`, `chore`, `docs`, `test`, `refactor`, `perf`,
`build`, `ci`, `style`. No emoji.

For multi-line bodies on Windows PowerShell, use a heredoc so the
newlines survive shell quoting:

```powershell
git commit -m "$(cat <<'EOF'
feat(scope): subject

Body line 1.
Body line 2.
EOF
)"
```

## Pull requests

```powershell
git checkout -b feat/x
# edits + commits
git push -u origin feat/x
# open a PR via the GitHub UI (no gh CLI in this env)
# squash-merge after review
git checkout master && git pull origin master --ff-only
git branch -d feat/x
```

GitHub Settings → General → "Automatically delete head branches" is
on, so the remote branch goes away on merge. Local cleanup is up
to you.

## Tags

Releases only, semver:

- `v0.1.0-lab-handoff`
- `v1.0.0`, `v1.0.0-rc1`

No snapshot or experiment tags. Use commit SHAs in release notes if
you need to reference an experiment run.

## Local setup

```powershell
pip install pre-commit
pre-commit install
```

This activates four hooks on every commit:

- `detect-secrets` (with `.secrets.baseline` baseline)
- `check-added-large-files` (50 MB cap — model weights belong on
  HuggingFace, not git)
- `check-merge-conflict`
- `end-of-file-fixer` + `trailing-whitespace`

Don't bypass with `--no-verify` or `SKIP=...` without writing a
reason in the commit body.

## What does not belong in git

- `.env` (secrets — use `infra/.env.example` as the template)
- Model weights (`*.safetensors`, `*.pt`, `*.pth`, `*.onnx`, `*.pkl`,
  `training_args.bin`) — upload to HuggingFace and document the
  download path in the relevant experiment README
- Optimizer / scheduler / RNG state from training runs
- Vendored dependencies (`node_modules/`, `.venv/`)
- Local IDE/agent state (`.vscode/`, `.idea/`, `.claude/worktrees/`,
  `.claude/settings.local.json`)
- Generated logs, caches, raw eval result dumps (`*.jsonl` under
  `experiments/**/results/`)

`.gitignore` covers all of the above; `.gitattributes` declares
binary types so git diff doesn't try to text-merge fonts or weights.

## Auditing the repo

Run `bash scripts/audit_history.sh` after any history rewrite and
before any visibility change to public. Six checks; require 6/6 PASS.

For the full hygiene-test playbook (Layer 1 audit + Layer 2
pre-commit canaries + Layer 3 GitHub-side canaries), see
[`docs/security/hygiene-test-plan.md`](docs/security/hygiene-test-plan.md).

## Backups before destructive git ops

If a session calls for `git filter-repo`, force-push to a shared
branch, batch branch deletion, or `git reset --hard` past unpushed
work, take a backup first:

```powershell
# Per-branch bundle (recoverable forever)
git for-each-ref --format='%(refname:short)' refs/heads/ |
  ForEach-Object { git bundle create "../tmt-bundles/$($_.Replace('/','-')).bundle" $_ }

# Full mirror clone
git clone --mirror . ../tmt-backup-(Get-Date -Format yyyyMMdd-HHmm).git
```

Use `--force-with-lease` instead of plain `--force` even after the
backup, to catch concurrent updates.
