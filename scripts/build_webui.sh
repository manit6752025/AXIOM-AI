#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

npm ci
npm run build

rm -rf axiom/webui
mkdir -p axiom/webui
cp -R dist/. axiom/webui/

test -f axiom/webui/index.html
echo "Web UI bundled into axiom/webui"
