#!/usr/bin/env bash
# Ablação V2: full vs hybrid × pixel vs wavelet (SignSGD, níveis 2–3, FR small).
# Treinos longos — rode no seu tmux. Não dispara GPU em smoke CI.
#
# Uso (defaults já estão abaixo; não precisa exportar nada):
#   tmux new -s v2_abl
#   bash scripts/run_v2_ablation_mask_roi.sh
#   # só eval (já treinou): SKIP_TRAIN=1 bash scripts/run_v2_ablation_mask_roi.sh
# Override opcional: DEVICE=cuda:1 N_EVAL=200 bash scripts/run_v2_ablation_mask_roi.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/.venv/bin/activate"
fi

# Defaults da ablação (env var sobrescreve se definida)
OUT_ROOT="${OUT_ROOT:-/mnt/study-data/dcarvalho/tests}"
DEVICE="${DEVICE:-cuda:0}"
N_EVAL="${N_EVAL:-500}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
HOLD_OUT="${HOLD_OUT:-${OUT_ROOT}/holdout_merged_seed42}"
FR_EVAL="${FR_EVAL:-inception_facenet_vgg,inception_facenet_casia,ir50_arcface_casia}"
SUMMARY_CSV="${SUMMARY_CSV:-${OUT_ROOT}/v2_ablation_mask_roi_summary.csv}"

# FR local neste repo (gitignored). Setup baixa código + pesos via gdown se faltar.
export PROTEGO_FR_ROOT="${PROTEGO_FR_ROOT:-${ROOT}/third_party/protego_fr}"

# EXP_NAME | roi | mask_shape | mask_regions
CONFIGS=(
  "v2_abl_full_pixel|pixel|ellipse|full"
  "v2_abl_hybrid_pixel|pixel|band|eyes-nose-mouth-hybrid"
  "v2_abl_full_wavelet|wavelet|ellipse|full"
  "v2_abl_hybrid_wavelet|wavelet|band|eyes-nose-mouth-hybrid"
)

echo "[ablation] OUT_ROOT=${OUT_ROOT} DEVICE=${DEVICE} N_EVAL=${N_EVAL}"
echo "[ablation] PROTEGO_FR_ROOT=${PROTEGO_FR_ROOT}"

# Preflight: third_party/protego_fr + pesos small (gdown; pasta gitignored).
if [[ "${SKIP_SETUP_FR:-0}" != "1" ]]; then
  echo "[ablation] setup FR (código + prefetch pesos small)…"
  bash "${ROOT}/scripts/run_v2_setup_protego_fr.sh"
  export PROTEGO_FR_ROOT="${ROOT}/third_party/protego_fr"
fi
if [[ ! -f "${PROTEGO_FR_ROOT}/protego/FacialRecognition.py" ]] || [[ ! -d "${PROTEGO_FR_ROOT}/FR_DB" ]]; then
  cat >&2 <<'EOF'
[ablation] ERRO: third_party/protego_fr incompleto.
  Rode: bash scripts/run_v2_setup_protego_fr.sh
  (código + pesos ficam só em third_party/ — gitignored)
EOF
  exit 1
fi
# Evita re-setup em cada run_v2_train_small.sh chamado abaixo.
export SKIP_FR_SETUP=1

run_train() {
  local exp="$1" roi="$2" shape="$3" regions="$4"
  echo "========== TRAIN ${exp} (roi=${roi} mask=${shape}/${regions}) =========="
  # Capture "[train-v2] done → <path>" from trainer_v2
  local log
  log="$(mktemp)"
  set +e
  EXP_NAME="${exp}" DEVICE="${DEVICE}" OUT_ROOT="${OUT_ROOT}" \
    bash "${ROOT}/scripts/run_v2_train_small.sh" \
      --roi-compose "${roi}" \
      --mask-mode landmarks \
      --mask-shape "${shape}" \
      --mask-regions "${regions}" \
      --ssim-region full-landmarks \
    2>&1 | tee "${log}"
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ "${rc}" -ne 0 ]]; then
    echo "[ablation] TRAIN failed for ${exp} (exit ${rc})" >&2
    rm -f "${log}"
    return "${rc}"
  fi
  local run_dir
  run_dir="$(grep -oE '\[train-v2\] done → .+' "${log}" | tail -n1 | sed 's/.*→ //')"
  rm -f "${log}"
  if [[ -n "${run_dir}" && -d "${run_dir}" ]]; then
    echo "${run_dir}" > "${OUT_ROOT}/LATEST_${exp}.txt"
    echo "[ablation] LATEST_${exp} → ${run_dir}"
  fi
}

find_run_dir() {
  local exp="$1"
  local ptr="${OUT_ROOT}/LATEST_${exp}.txt"
  if [[ -f "${ptr}" ]]; then
    local p
    p="$(tr -d '[:space:]' < "${ptr}")"
    if [[ -d "${p}" ]]; then
      echo "${p}"
      return 0
    fi
  fi
  # Fallback: newest directory matching *v2_<exp>* (trainer: {stamp}_v2_{exp_name})
  local found
  found="$(find "${OUT_ROOT}" -maxdepth 2 -type d -name "*_v2_${exp}" 2>/dev/null | sort | tail -n 1 || true)"
  echo "${found}"
}

find_best_ckpt() {
  local run_dir="$1"
  if [[ -f "${run_dir}/meta_mask_best.pt" ]]; then
    echo "${run_dir}/meta_mask_best.pt"
  elif [[ -f "${run_dir}/meta_mask.pt" ]]; then
    echo "${run_dir}/meta_mask.pt"
  else
    echo ""
  fi
}

if [[ "${SKIP_TRAIN}" != "1" ]]; then
  for row in "${CONFIGS[@]}"; do
    IFS='|' read -r exp roi shape regions <<<"${row}"
    run_train "${exp}" "${roi}" "${shape}" "${regions}"
  done
else
  echo "[ablation] SKIP_TRAIN=1 — pulando treinos"
fi

if [[ "${SKIP_EVAL}" == "1" ]]; then
  echo "[ablation] SKIP_EVAL=1 — fim"
  exit 0
fi

echo "========== Materialize holdout → ${HOLD_OUT} =========="
python "${ROOT}/scripts/run_v2_materialize_holdout.py" \
  --out-dir "${HOLD_OUT}" \
  --celeba-train-frac 0.5 \
  --data-seed 42

echo "exp,roi_compose,mask_shape,mask_regions,run_dir,checkpoint,cos_mean,ssim_mean,euclid_mean,time_ms_mean,time_ms_e2e_mean,n_images" \
  > "${SUMMARY_CSV}"

for row in "${CONFIGS[@]}"; do
  IFS='|' read -r exp roi shape regions <<<"${row}"
  run_dir="$(find_run_dir "${exp}")"
  if [[ -z "${run_dir}" ]]; then
    echo "[ablation] WARN: run dir não encontrado para ${exp}" >&2
    continue
  fi
  ckpt="$(find_best_ckpt "${run_dir}")"
  if [[ -z "${ckpt}" ]]; then
    echo "[ablation] WARN: checkpoint ausente em ${run_dir}" >&2
    continue
  fi
  echo "========== EVAL ${exp} ← ${ckpt} =========="
  python "${ROOT}/deid_optimize.py" evaluate-v2 \
    --checkpoint "${ckpt}" \
    --data "${HOLD_OUT}" \
    --run-dir "${run_dir}" \
    --device "${DEVICE}" \
    --batch-size 4 \
    --max-samples "${N_EVAL}" \
    --fr-models "${FR_EVAL}" \
    --mask-mode landmarks \
    --mask-shape "${shape}" \
    --mask-regions "${regions}" \
    --ssim-region full-landmarks \
    --no-lpips

  summary="${run_dir}/eval/eval_summary.json"
  if [[ -f "${summary}" ]]; then
    python - "${summary}" "${exp}" "${roi}" "${shape}" "${regions}" "${run_dir}" "${ckpt}" "${SUMMARY_CSV}" <<'PY'
import json, sys
path, exp, roi, shape, regions, run_dir, ckpt, csv_path = sys.argv[1:9]
s = json.loads(open(path, encoding="utf-8").read())
row = [
    exp, roi, shape, regions, run_dir, ckpt,
    f"{s.get('cos_mean', float('nan')):.6f}",
    f"{s.get('ssim_mean', float('nan')):.6f}",
    f"{s.get('euclid_mean', float('nan')):.6f}",
    f"{s.get('time_ms_mean', float('nan')):.4f}",
    f"{s.get('time_ms_e2e_mean', float('nan')):.4f}",
    str(s.get("n_images", "")),
]
with open(csv_path, "a", encoding="utf-8") as f:
    f.write(",".join(row) + "\n")
print("[ablation] appended", exp, "→", csv_path)
PY
  fi
done

echo "[ablation] done. summary → ${SUMMARY_CSV}"
