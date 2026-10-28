#!/usr/bin/env bash
# V2 treino campeão enxuto: Protego + FR preset small + roi_compose=pixel.
# FR_DB local: third_party/protego_fr (setup automático; gitignored).
# Smoke sem FR_DB: adicione --smoke-fr.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/.venv/bin/activate"
fi

# Garante código + pesos small em third_party/protego_fr (não sobe ao GitHub).
if [[ "${SKIP_FR_SETUP:-0}" != "1" ]] && [[ "$*" != *"--smoke-fr"* ]]; then
  bash "${ROOT}/scripts/run_v2_setup_protego_fr.sh"
  export PROTEGO_FR_ROOT="${PROTEGO_FR_ROOT:-${ROOT}/third_party/protego_fr}"
fi

OUT_ROOT="${OUT_ROOT:-/mnt/study-data/dcarvalho/tests}"
DEVICE="${DEVICE:-cuda:0}"
EXP_NAME="${EXP_NAME:-merged_p33_protego_fullface}"

python deid_optimize.py train-v2 \
  --dataset merged_lfw_celeba \
  --out-root "${OUT_ROOT}" \
  --exp-name "${EXP_NAME}" \
  --device "${DEVICE}" \
  --image-size 224 \
  --batch-size 2 \
  --epochs 5 \
  --optimizer protego \
  --epsilon 0.062745098 \
  --min-ssim 0.95 \
  --wavelet-J 3 \
  --editable-levels 2,3 \
  --fr-preset small \
  --eval-fr-models inception_facenet_vgg,inception_facenet_casia \
  --celeba-train-frac 0.5 \
  --data-seed 42 \
  --seed 42 \
  --roi-compose pixel \
  --mask-mode landmarks \
  --mask-shape ellipse \
  --mask-regions full \
  --ssim-region full-landmarks \
  --early-stopping \
  --early-stopping-metric cos \
  --patience 2 \
  --val-max-samples 200 \
  --ckpt-every 1000 \
  --preview-every 100 \
  --init-std 1e-3 \
  "$@"
