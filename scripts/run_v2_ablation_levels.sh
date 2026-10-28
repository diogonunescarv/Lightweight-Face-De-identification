#!/usr/bin/env bash
# Ablação V2 — níveis DT-CWT editáveis × Ideia 1/2.
#
# Base (melhor da grade máscaras): ROI full + ellipse + landmarks + SignSGD + FR small.
# Combinações de --editable-levels (J=3):
#   1 | 2 | 3 | 1,2 | 1,3 | 2,3 | 1,2,3
# Para cada combinação: pixel e wavelet.
#
# Nível 2,3 reutiliza os runs da ablação anterior (v2_abl_full_pixel / full_wavelet)
# se LATEST_* existir (REUSE_LV23=1, default).
#
# Uso:
#   bash scripts/run_v2_ablation_levels.sh
#   SKIP_TRAIN=1 bash scripts/run_v2_ablation_levels.sh   # só eval
#   REUSE_LV23=0 bash scripts/run_v2_ablation_levels.sh   # retreina também 2,3
#
# Idempotente: pula runs com meta_mask.pt; retoma checkpoint_latest.pt se existir.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/.venv/bin/activate"
fi

OUT_ROOT="${OUT_ROOT:-/mnt/study-data/dcarvalho/tests}"
DEVICE="${DEVICE:-cuda:0}"
N_EVAL="${N_EVAL:-500}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
REUSE_LV23="${REUSE_LV23:-1}"
HOLD_OUT="${HOLD_OUT:-${OUT_ROOT}/holdout_merged_seed42}"
FR_EVAL="${FR_EVAL:-inception_facenet_vgg,inception_facenet_casia,ir50_arcface_casia}"
SUMMARY_CSV="${SUMMARY_CSV:-${OUT_ROOT}/v2_ablation_levels_summary.csv}"

# Máscara fixa = melhor da 1ª bateria (full landmarks)
MASK_MODE="landmarks"
MASK_SHAPE="ellipse"
MASK_REGIONS="full"
SSIM_REGION="full-landmarks"

export PROTEGO_FR_ROOT="${PROTEGO_FR_ROOT:-${ROOT}/third_party/protego_fr}"

# exp | roi | editable_levels | tag
CONFIGS=(
  "v2_abl_lv1_pixel|pixel|1|lv1"
  "v2_abl_lv2_pixel|pixel|2|lv2"
  "v2_abl_lv3_pixel|pixel|3|lv3"
  "v2_abl_lv12_pixel|pixel|1,2|lv12"
  "v2_abl_lv13_pixel|pixel|1,3|lv13"
  "v2_abl_lv23_pixel|pixel|2,3|lv23"
  "v2_abl_lv123_pixel|pixel|1,2,3|lv123"
  "v2_abl_lv1_wavelet|wavelet|1|lv1"
  "v2_abl_lv2_wavelet|wavelet|2|lv2"
  "v2_abl_lv3_wavelet|wavelet|3|lv3"
  "v2_abl_lv12_wavelet|wavelet|1,2|lv12"
  "v2_abl_lv13_wavelet|wavelet|1,3|lv13"
  "v2_abl_lv23_wavelet|wavelet|2,3|lv23"
  "v2_abl_lv123_wavelet|wavelet|1,2,3|lv123"
)

echo "[ablation-levels] OUT_ROOT=${OUT_ROOT} DEVICE=${DEVICE} N_EVAL=${N_EVAL}"
echo "[ablation-levels] base mask=${MASK_MODE}/${MASK_SHAPE}/${MASK_REGIONS}"
echo "[ablation-levels] REUSE_LV23=${REUSE_LV23}  n_configs=${#CONFIGS[@]}"

if [[ "${SKIP_SETUP_FR:-0}" != "1" ]]; then
  bash "${ROOT}/scripts/run_v2_setup_protego_fr.sh"
  export PROTEGO_FR_ROOT="${ROOT}/third_party/protego_fr"
fi
if [[ ! -f "${PROTEGO_FR_ROOT}/protego/FacialRecognition.py" ]] || [[ ! -d "${PROTEGO_FR_ROOT}/FR_DB" ]]; then
  echo "[ablation-levels] ERRO: third_party/protego_fr incompleto" >&2
  exit 1
fi
export SKIP_FR_SETUP=1

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
  find "${OUT_ROOT}" -maxdepth 2 -type d -name "*_v2_${exp}" 2>/dev/null | sort | tail -n 1 || true
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

# Run completo = tem meta_mask.pt (final) ou meta_mask_best + summary do trainer.
is_train_done() {
  local exp="$1"
  local d
  d="$(find_run_dir "${exp}")"
  [[ -n "${d}" && -f "${d}/meta_mask.pt" ]]
}

# Run parcial interrompido = checkpoint_latest sem meta_mask.pt final.
find_resume_ckpt() {
  local exp="$1"
  local d
  d="$(find "${OUT_ROOT}" -maxdepth 2 -type d -name "*_v2_${exp}" 2>/dev/null | sort | tail -n 1 || true)"
  if [[ -n "${d}" && -f "${d}/checkpoint_latest.pt" && ! -f "${d}/meta_mask.pt" ]]; then
    echo "${d}/checkpoint_latest.pt"
    return 0
  fi
  echo ""
}

# Reusa baseline 2,3 da ablação de máscaras (full_pixel / full_wavelet).
maybe_reuse_lv23() {
  local exp="$1" roi="$2"
  if [[ "${REUSE_LV23}" != "1" ]]; then
    return 1
  fi
  local src_exp
  if [[ "${roi}" == "pixel" ]]; then
    src_exp="v2_abl_full_pixel"
  else
    src_exp="v2_abl_full_wavelet"
  fi
  local src
  src="$(find_run_dir "${src_exp}")"
  if [[ -z "${src}" || ! -d "${src}" ]]; then
    echo "[ablation-levels] REUSE_LV23: ${src_exp} não encontrado — vai treinar ${exp}"
    return 1
  fi
  echo "${src}" > "${OUT_ROOT}/LATEST_${exp}.txt"
  echo "[ablation-levels] REUSE ${exp} ← ${src_exp} (${src})"
  return 0
}

run_train() {
  local exp="$1" roi="$2" levels="$3"
  if is_train_done "${exp}"; then
    echo "[ablation-levels] SKIP (já completo): ${exp} → $(find_run_dir "${exp}")"
    return 0
  fi
  if [[ "${levels}" == "2,3" ]] && maybe_reuse_lv23 "${exp}" "${roi}"; then
    return 0
  fi

  local resume_ckpt=""
  resume_ckpt="$(find_resume_ckpt "${exp}")"
  local extra=()
  if [[ -n "${resume_ckpt}" ]]; then
    echo "[ablation-levels] RESUME ${exp} ← ${resume_ckpt}"
    extra+=(--resume "${resume_ckpt}")
  fi

  echo "========== TRAIN ${exp} (roi=${roi} levels=${levels}) =========="
  local log
  log="$(mktemp)"
  set +e
  EXP_NAME="${exp}" DEVICE="${DEVICE}" OUT_ROOT="${OUT_ROOT}" \
    bash "${ROOT}/scripts/run_v2_train_small.sh" \
      --roi-compose "${roi}" \
      --editable-levels "${levels}" \
      --mask-mode "${MASK_MODE}" \
      --mask-shape "${MASK_SHAPE}" \
      --mask-regions "${MASK_REGIONS}" \
      --ssim-region "${SSIM_REGION}" \
      "${extra[@]}" \
    2>&1 | tee "${log}"
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ "${rc}" -ne 0 ]]; then
    echo "[ablation-levels] TRAIN failed for ${exp} (exit ${rc})" >&2
    rm -f "${log}"
    return "${rc}"
  fi
  local run_dir
  run_dir="$(grep -oE '\[train-v2\] done → .+' "${log}" | tail -n1 | sed 's/.*→ //')"
  rm -f "${log}"
  if [[ -z "${run_dir}" && -n "${resume_ckpt}" ]]; then
    run_dir="$(dirname "${resume_ckpt}")"
  fi
  if [[ -n "${run_dir}" && -d "${run_dir}" ]]; then
    echo "${run_dir}" > "${OUT_ROOT}/LATEST_${exp}.txt"
    echo "[ablation-levels] LATEST_${exp} → ${run_dir}"
  fi
}

if [[ "${SKIP_TRAIN}" != "1" ]]; then
  for row in "${CONFIGS[@]}"; do
    IFS='|' read -r exp roi levels tag <<<"${row}"
    run_train "${exp}" "${roi}" "${levels}"
  done
else
  echo "[ablation-levels] SKIP_TRAIN=1"
fi

if [[ "${SKIP_EVAL}" == "1" ]]; then
  echo "[ablation-levels] SKIP_EVAL=1 — fim"
  exit 0
fi

echo "========== Materialize holdout → ${HOLD_OUT} =========="
python "${ROOT}/scripts/run_v2_materialize_holdout.py" \
  --out-dir "${HOLD_OUT}" \
  --celeba-train-frac 0.5 \
  --data-seed 42

echo "exp,roi_compose,editable_levels,mask_shape,mask_regions,run_dir,checkpoint,cos_mean,ssim_mean,euclid_mean,time_ms_mean,time_ms_e2e_mean,n_images" \
  > "${SUMMARY_CSV}"

for row in "${CONFIGS[@]}"; do
  IFS='|' read -r exp roi levels tag <<<"${row}"
  run_dir="$(find_run_dir "${exp}")"
  if [[ -z "${run_dir}" ]]; then
    echo "[ablation-levels] WARN: run dir não encontrado para ${exp}" >&2
    continue
  fi
  ckpt="$(find_best_ckpt "${run_dir}")"
  if [[ -z "${ckpt}" ]]; then
    echo "[ablation-levels] WARN: checkpoint ausente em ${run_dir}" >&2
    continue
  fi
  echo "========== EVAL ${exp} levels=${levels} ← ${ckpt} =========="
  python "${ROOT}/deid_optimize.py" evaluate-v2 \
    --checkpoint "${ckpt}" \
    --data "${HOLD_OUT}" \
    --run-dir "${run_dir}" \
    --device "${DEVICE}" \
    --batch-size 4 \
    --max-samples "${N_EVAL}" \
    --fr-models "${FR_EVAL}" \
    --mask-mode "${MASK_MODE}" \
    --mask-shape "${MASK_SHAPE}" \
    --mask-regions "${MASK_REGIONS}" \
    --ssim-region "${SSIM_REGION}" \
    --no-lpips

  summary="${run_dir}/eval/eval_summary.json"
  if [[ -f "${summary}" ]]; then
    python - "${summary}" "${exp}" "${roi}" "${levels}" "${MASK_SHAPE}" "${MASK_REGIONS}" "${run_dir}" "${ckpt}" "${SUMMARY_CSV}" <<'PY'
import json, sys
path, exp, roi, levels, shape, regions, run_dir, ckpt, csv_path = sys.argv[1:10]
s = json.loads(open(path, encoding="utf-8").read())
row = [
    exp, roi, levels, shape, regions, run_dir, ckpt,
    f"{s.get('cos_mean', float('nan')):.6f}",
    f"{s.get('ssim_mean', float('nan')):.6f}",
    f"{s.get('euclid_mean', float('nan')):.6f}",
    f"{s.get('time_ms_mean', float('nan')):.4f}",
    f"{s.get('time_ms_e2e_mean', float('nan')):.4f}",
    str(s.get("n_images", "")),
]
with open(csv_path, "a", encoding="utf-8") as f:
    f.write(",".join(row) + "\n")
print("[ablation-levels] appended", exp, "→", csv_path)
PY
  fi
done

echo "[ablation-levels] done. summary → ${SUMMARY_CSV}"
echo "[ablation-levels] notebook: notebooks/v2_ablation_levels_visualize.ipynb"
