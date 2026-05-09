# Hygiene gate test plan

End-to-end checks that the secret/large-file rails set up after the
2026-05 history rewrite actually block bad commits before they reach
the public repo.

Run all tests **on a throwaway test branch**, never on `master`. Delete
the branch after each session.

---

## Layer 1 — Local history audit (automated)

```bash
bash scripts/audit_history.sh
```

Exit 0 if all six checks pass. Re-run before each public release and
after any history rewrite. See the script for what is verified.

---

## Layer 2 — Pre-commit gates (automated, run locally)

Pre-requisite: `pip install pre-commit && pre-commit install` once per
clone.

### 2.1 detect-secrets blocks credential-shaped strings

```bash
git checkout -b test/canary-secret
cat > canary.py <<'EOF'
AWS_ACCESS_KEY_ID = "AKIAIOSFODNN7EXAMPLE"
AWS_SECRET_ACCESS_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
GITHUB_TOKEN = "ghp_1234567890abcdefghijklmnopqrstuvwxyzAB"
EOF
git add canary.py
git commit -m "canary: should be blocked"  # expect REJECT
git reset HEAD canary.py && rm canary.py
git checkout master && git branch -D test/canary-secret
```

Pass: commit aborted with detect-secrets reporting "AWS Access Key" /
"GitHub Token" / "Base64 High Entropy String".

### 2.2 check-added-large-files blocks files > 50 MB

```bash
git checkout -b test/canary-bigfile
head -c 62914560 /dev/urandom > big.bin
git add -f big.bin
git commit -m "canary: should be blocked"  # expect REJECT
git reset HEAD big.bin && rm big.bin
git checkout master && git branch -D test/canary-bigfile
```

Pass: commit aborted, message `big.bin (61440 KB) exceeds 51200 KB`.

---

## Layer 3 — GitHub-side gates (manual)

These tests require touching the public origin and **must not push
real secrets**. Use the canary patterns below (synthetic but
GitHub-recognised shape — actual scanning behaviour may vary because
GitHub Push Protection is signed-pattern-driven).

### 3.1 Push Protection blocks push of credential-shaped strings

```bash
git checkout -b test/push-protection
# Use a clearly fake test pattern — DO NOT use a real key
cat > canary.py <<'EOF'
GITHUB_PAT = "ghp_TESTONLYxxxxxxxxxxxxxxxxxxxxxxxxxxxxAB"
EOF
git add canary.py
SKIP=detect-secrets git commit -m "canary: testing GH push protection"
git push origin test/push-protection
```

Pass: GitHub server rejects the push with a "Secret Detected" message
referencing the line of `canary.py`. The commit may sit locally but
never reaches origin.

Cleanup:

```bash
git reset HEAD~1 && rm canary.py
git checkout master && git branch -D test/push-protection
```

If the push is **not** rejected, Push Protection is not configured
correctly — go to **Settings → Code security → Secret scanning** and
re-enable.

### 3.2 Branch protection blocks direct push to master

```bash
git checkout master
git commit --allow-empty -m "should not be pushable directly"
git push origin master
git reset --hard origin/master   # undo the local empty commit
```

Pass: push rejected with a message referencing the protected-branch
rule (e.g. "must be merged via pull request" or "required status
checks").

If the push **is** accepted, branch protection is missing — go to
**Settings → Branches → Branch protection rules** and add a rule for
`master` requiring PRs.

---

## Cadence

| When                                  | Run                          |
|---------------------------------------|------------------------------|
| After any history rewrite             | Layer 1                      |
| Before each public release            | Layer 1 + spot-check Layer 2 |
| When GitHub repo settings are changed | Layer 3                      |
| When `.pre-commit-config.yaml` edited | Layer 2.1 + 2.2              |
