#!/usr/bin/env bash
# Instala código FR (FacialRecognition + FR_DB) em third_party/protego_fr/ neste repo.
# Pesos .pth NÃO vêm no GitHub — baixam sob demanda via gdown (como no Protego).
# Pasta third_party/protego_fr/ está no .gitignore.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DEST="${PROTEGO_FR_DEST:-${ROOT}/third_party/protego_fr}"
REPO_URL="${PROTEGO_FR_REPO:-https://github.com/diogonunescarv/dtcwt_mask_optimization.git}"
PREFETCH="${PREFETCH_FR:-1}"  # 1 = dispara download dos pesos do preset small

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/.venv/bin/activate"
fi
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

need_code=0
if [[ ! -f "${DEST}/protego/FacialRecognition.py" ]] || [[ ! -d "${DEST}/FR_DB" ]]; then
  need_code=1
fi

if [[ "${need_code}" -eq 1 ]]; then
  echo "[setup-fr] Populando ${DEST} a partir de ${REPO_URL} (sparse: protego + FR_DB)…"
  mkdir -p "${ROOT}/third_party"
  TMP="$(mktemp -d)"
  cleanup() { rm -rf "${TMP}"; }
  trap cleanup EXIT
  git clone --depth 1 --filter=blob:none --sparse "${REPO_URL}" "${TMP}/repo"
  (
    cd "${TMP}/repo"
    git sparse-checkout set protego_base/protego protego_base/FR_DB
  )
  rm -rf "${DEST}"
  mkdir -p "${DEST}"
  cp -a "${TMP}/repo/protego_base/protego" "${DEST}/"
  cp -a "${TMP}/repo/protego_base/FR_DB" "${DEST}/"
  # Garantir que pesos acidentais não fiquem versionáveis (já gitignored).
  find "${DEST}/FR_DB" -type d -name pretrained -exec rm -rf {} + 2>/dev/null || true
  echo "[setup-fr] Código OK → ${DEST}"
else
  echo "[setup-fr] Código já presente → ${DEST}"
fi

# gdown necessário para download on-demand dos .pth
python - <<'PY'
import importlib.util
import subprocess
import sys
if importlib.util.find_spec("gdown") is None:
    print("[setup-fr] instalando gdown…")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gdown"])
print("[setup-fr] gdown OK")
PY

export PROTEGO_FR_ROOT="${DEST}"
echo "[setup-fr] PROTEGO_FR_ROOT=${PROTEGO_FR_ROOT}"

if [[ "${PREFETCH}" == "1" ]]; then
  echo "[setup-fr] Prefetch pesos do preset small (gdown; pode demorar)…"
  python - <<'PY'
from models.fr.presets import SMALL_TRAIN_FR_NAMES
from models.fr.ensemble import load_fr_ensemble
print("[setup-fr] loading:", SMALL_TRAIN_FR_NAMES)
load_fr_ensemble(SMALL_TRAIN_FR_NAMES, device="cpu", keep_on_device=False)
print("[setup-fr] pesos small OK")
PY
fi

echo "[setup-fr] pronto. Use: export PROTEGO_FR_ROOT=${DEST}"
