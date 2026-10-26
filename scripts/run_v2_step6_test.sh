#!/usr/bin/env bash
# V2 Step 6 smoke (trainer pixel + resume).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/.venv/bin/activate"
fi
python -m unittest tests.test_v2_06_train_smoke tests.test_v2_01_scaffold -v
