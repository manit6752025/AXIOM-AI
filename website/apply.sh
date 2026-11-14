#!/usr/bin/env bash
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(git -C "$here" rev-parse --show-toplevel)"
cd "$root" || exit 1
branch="$(git branch --show-current)"
git pull --rebase --autostash origin "$branch" || echo "Pull skipped"
mkdir -p website/assets
cp "$here/website/index.html" website/index.html
cp "$here/website/assets/axiom-mark.svg" website/assets/axiom-mark.svg
if python3 -m pytest -q; then
  git add website
  if git diff --cached --quiet; then
    echo "Nothing changed"
    exit 0
  fi
  git commit -m "Redesign website with typed AXIOM hero and Saturn mark"
  git push origin "$branch" && echo "Pushed to $branch"
else
  echo "Tests failed. Restoring old website files."
  git checkout -- website
  grep -n "axiom-mark\|brand-mark\|website/" tests/test_release_contract.py | head -40
  exit 1
fi
