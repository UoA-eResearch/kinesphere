#!/usr/bin/env bash
# Builds the small random-weight Altlando model the browser tests use (test-results/altlando-tiny):
# the real architecture from dance-ai-research-project, exported with ai-partner/export_onnx.py.
# Needs Python 3 with PyTorch and ai-partner/requirements-export.txt installed.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
SRC=test-results/dance-ai-research-project
COMMIT=3315184f44b4c46a2fbb79d40ef6a3455b8e9033  # jzho987/dance-ai-research-project, the version this was written against
if [ ! -d "$SRC/.git" ]; then
  git clone -q https://github.com/jzho987/dance-ai-research-project "$SRC"
fi
git -C "$SRC" fetch -q --depth 1 origin "$COMMIT" 2>/dev/null || true
git -C "$SRC" checkout -q "$COMMIT"
"$PY" ai-partner/export_onnx.py --altlando-app "$SRC/src/bailandopp/app" --random --tiny --out test-results/altlando-tiny 2>&1 \
  | grep -v -E "Warning|^  |^DEVICE|We use bottleneck|No motion regularization" || true
test -f test-results/altlando-tiny/reference.json
