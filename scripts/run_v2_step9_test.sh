#!/usr/bin/env bash
# V2 Step 9: docs/cleanup + visualize helpers smoke.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/.venv/bin/activate"
fi
python -m unittest tests.test_v2_09_cleanup_docs -v
