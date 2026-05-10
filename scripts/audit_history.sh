#!/usr/bin/env bash
#
# audit_history.sh — repo-history smoke check
#
# Verifies that this repository is free of:
#   - committed credentials (the historical Neo4j password)
#   - the original leaked blob in the object DB
#   - files > 25 MB in any commit
#   - binary model artifacts (*.safetensors, *.pt, *.pth, *.onnx, *.pkl)
#   - vendored node_modules trees
#   - non-master branches on origin
#
# Run after history rewrites or before public release.
# Exit code: 0 if all pass, 1 if any fail.

set -uo pipefail

PASS=0
FAIL=0

ok()   { echo "  PASS — $1"; PASS=$((PASS+1)); }
bad()  { echo "  FAIL — $1"; FAIL=$((FAIL+1)); }

cd "$(git rev-parse --show-toplevel)"
echo "Auditing $(pwd)"
echo

# ---- 1. No leaked password literal anywhere in history ------------------
# Two anti-self-reference measures:
# 1. Build the search literal via concatenation so this script's source
#    does not itself contain the contiguous bytes.
# 2. Exclude this script's path from the search corpus so historical
#    versions of the script are never counted.
echo "[1] Leaked Neo4j password not in any commit"
LEAK="quarter-contour-watch""-signal-honey"
hits=$(git rev-list --all 2>/dev/null \
       | xargs -n 50 git grep -l "$LEAK" -- ':(exclude)scripts/audit_history.sh' 2>/dev/null \
       | wc -l)
[ "$hits" -eq 0 ] && ok "no commit contains the historical Neo4j password literal" \
                 || bad "$hits commit(s) still contain the historical Neo4j password literal"

# ---- 2. Original leaking blob no longer reachable -----------------------
echo "[2] Original leaking blob no longer in object DB"
BAD_BLOB="ce8702630eed63564b7126306019dce044c8d232"
if git cat-file -e "$BAD_BLOB" 2>/dev/null; then
    bad "blob $BAD_BLOB still resolvable"
else
    ok "blob $BAD_BLOB unreachable"
fi

# ---- 3. No file > 25 MB in any reachable commit -------------------------
echo "[3] No reachable blob exceeds 25 MB"
big=$(git rev-list --all --objects 2>/dev/null \
      | awk 'NF==2 {print $1}' \
      | git cat-file --batch-check='%(objectsize) %(rest)' 2>/dev/null \
      | awk '$1 > 25000000 {print $1, $2}' \
      | head -5)
if [ -z "$big" ]; then
    ok "max blob size <= 25 MB"
else
    bad "blobs > 25 MB still reachable:"
    echo "$big" | sed 's/^/         /'
fi

# ---- 4. No binary model artifacts in history ----------------------------
echo "[4] No binary model artifacts (*.safetensors|pt|pth|onnx|pkl) in history"
binhits=$(git rev-list --all --objects 2>/dev/null \
          | grep -cE '\.(safetensors|pt|pth|onnx|pkl)$' || true)
[ "$binhits" -eq 0 ] && ok "no binary weight/state files reachable" \
                    || bad "$binhits binary artifact path(s) reachable"

# ---- 5. No vendored node_modules in history -----------------------------
echo "[5] No node_modules/ tree in history"
nm=$(git rev-list --all --objects 2>/dev/null | grep -c 'node_modules/' || true)
[ "$nm" -eq 0 ] && ok "no node_modules paths" \
              || bad "$nm node_modules entries still reachable"

# ---- 6. origin has only refs/heads/master -------------------------------
echo "[6] origin exposes only the master branch"
if ! git remote get-url origin >/dev/null 2>&1; then
    bad "no 'origin' remote configured"
else
    nbranches=$(git ls-remote --heads origin 2>/dev/null | wc -l)
    [ "$nbranches" -eq 1 ] && ok "origin exposes exactly 1 branch" \
                          || bad "origin exposes $nbranches branches (expected 1)"
fi

# ------------------------------------------------------------------------
echo
echo "----------------------------------"
echo "Result: $PASS passed, $FAIL failed"
echo "----------------------------------"
[ "$FAIL" -eq 0 ]
