#!/usr/bin/env bash
# V2 Step 3 smoke (identity + quality losses).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/.venv/bin/activate"
fi
python -m unittest \
  tests.test_v2_01_scaffold \
  tests.test_v2_02_dtcwt_module \
  tests.test_v2_02b_roi_compose_pixel \
  tests.test_v2_03_loss_quality \
  -v
