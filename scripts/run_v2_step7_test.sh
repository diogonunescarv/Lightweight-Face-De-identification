#!/usr/bin/env bash
# V2 Step 7: wavelet compose + regression of pixel suites (Steps 2b / 6).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/.venv/bin/activate"
fi
python -m unittest \
  tests.test_v2_02_dtcwt_module \
  tests.test_v2_02b_roi_compose_pixel \
  tests.test_v2_06_train_smoke \
  tests.test_v2_07_roi_compose_wavelet \
  -v
