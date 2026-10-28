# Contexto V2 — status dos experimentos (handoff entre servidores / chats)

> **Leia este arquivo no início de sessões Cursor** sobre ablações V2.
> Spec canônica: [`V2.md`](V2.md). Decisões: [`DECISIONS.md`](DECISIONS.md).
> Atualizado: **2026-10-28** (interrupção no server5 → retomar no server3).

---

## 1. Onde estamos

Branch: `feature/v2-dtcwt-meta-mask`  
Repo: `git@github.com:diogonunescarv/Lightweight-Face-De-identification.git`  
Artefatos de treino (NFS compartilhado tipicamente): `/mnt/study-data/dcarvalho/tests`  
Bases: `/mnt/databases/faces/lfw_frontal`, `/mnt/databases/faces/celebahq_pp_frontal`  
FR local (gitignored): `third_party/protego_fr/` via `bash scripts/run_v2_setup_protego_fr.sh`

A V2 (Steps 0–9) está **treinável** (Ideias `pixel` / `wavelet`). Estamos na fase de **ablações experimentais**.

---

## 2. Bateria 1 — máscaras × ideias (CONCLUÍDA)

Script: `scripts/run_v2_ablation_mask_roi.sh`  
Notebook: `notebooks/v2_ablation_mask_roi_visualize.ipynb`  
Summary: `/mnt/study-data/dcarvalho/tests/v2_ablation_mask_roi_summary.csv`

Constantes: SignSGD (`protego`), FR preset `small`, níveis **2,3**, J=3, `min_ssim=0.95`, ε=16/255.

| EXP | roi | máscara | cos↓ | ssim | time_ms | time_e2e_ms |
|-----|-----|---------|------|------|---------|-------------|
| `v2_abl_full_pixel` | pixel | ellipse/full | **0.447** | 0.936 | 3.5 | 23.6 |
| `v2_abl_hybrid_pixel` | pixel | band/hybrid | 0.514 | 0.942 | 3.5 | 21.5 |
| `v2_abl_full_wavelet` | wavelet | ellipse/full | **0.467** | 0.933 | 3.0 | 18.3 |
| `v2_abl_hybrid_wavelet` | wavelet | band/hybrid | 0.545 | 0.943 | 3.0 | 18.6 |

**Conclusão:** melhor por ideia = **full + landmarks/ellipse** (não hybrid).  
Eval FR: `inception_facenet_vgg,inception_facenet_casia,ir50_arcface_casia` (2 held-out + 1 do treino).  
Holdout materializado: `/mnt/study-data/dcarvalho/tests/holdout_merged_seed42` (~14390 symlinks).

LATEST pointers:
- `LATEST_v2_abl_full_pixel.txt` → `20261027_133449_v2_v2_abl_full_pixel`
- `LATEST_v2_abl_hybrid_pixel.txt` → `20261027_174032_v2_v2_abl_hybrid_pixel`
- `LATEST_v2_abl_full_wavelet.txt` → `20261027_214901_v2_v2_abl_full_wavelet`
- `LATEST_v2_abl_hybrid_wavelet.txt` → `20261028_001606_v2_v2_abl_hybrid_wavelet`

---

## 3. Bateria 2 — níveis DT-CWT (EM ANDAMENTO)

Script: `scripts/run_v2_ablation_levels.sh`  
Notebook (pronto, rodar após treinos): `notebooks/v2_ablation_levels_visualize.ipynb`

**Base fixa:** máscara full (melhor da bateria 1) + FR small + SignSGD.  
**Varredura:** `--editable-levels` ∈ `{1}`, `{2}`, `{3}`, `{1,2}`, `{1,3}`, `{2,3}`, `{1,2,3}` × `pixel`/`wavelet` (14 configs).  
`{2,3}` **reutiliza** os runs `full_*` da bateria 1 (`REUSE_LV23=1`).

### Progresso ao interromper (bip-server5, 2026-10-28)

| Config | Status |
|--------|--------|
| `v2_abl_lv1_pixel` | **completo** → `20261028_121837_v2_v2_abl_lv1_pixel` (`meta_mask.pt`) |
| `v2_abl_lv2_pixel` | **parcial** → `20261028_161428_v2_v2_abl_lv2_pixel` (`checkpoint_latest.pt`, epoch≈1 / step≈13000, interrompido no epoch 2 ~33%) |
| lv3, lv12, lv13, lv23, lv123 (pixel) | pendente |
| todos wavelet | pendente |
| eval / summary CSV níveis | pendente |

Motivo da interrupção: outras instâncias no servidor (GPU/CPU); Ctrl-C no meio de `lv2_pixel`.

### Como retomar (mesmo NFS `/mnt/study-data`)

O script agora:
1. **SKIP** configs com `meta_mask.pt` final;
2. **RESUME** automático se existir `checkpoint_latest.pt` sem `meta_mask.pt`;
3. Reusa lv23 dos `full_*`.

```bash
# no server3 (ou onde a GPU estiver livre), na pasta do clone:
source .venv/bin/activate
bash scripts/run_v2_setup_protego_fr.sh          # se third_party/ ainda não existir
bash scripts/run_v2_ablation_levels.sh           # retoma lv2_pixel e segue a fila
```

Só eval depois de todos os treinos:
```bash
SKIP_TRAIN=1 bash scripts/run_v2_ablation_levels.sh
```

---

## 4. Contratos importantes (não quebrar)

- Inferência: meta-máscara **universal** ∘ \(M(\mathrm{landmarks})\) — nunca forçar elipse fixa na eval/viz desta linha.
- Loss Phase 3.3: cos médio ensemble + SSIM hinge com \(w\) adaptativo (`losses/identity.py`, `losses/quality.py`).
- Checkpoints V1 ≠ V2.
- `third_party/` e `*.pt` estão no `.gitignore` — pesos FR baixam via gdown, não vão ao GitHub.
- Notebooks em `notebooks/` também gitignored; helpers canônicos em `evaluation/` + `scripts/`.

---

## 5. Próximos passos sugeridos (ordem)

1. Clone/setup no **server3** (ver §6).
2. Retomar `bash scripts/run_v2_ablation_levels.sh` até os 14 configs + eval.
3. Abrir `notebooks/v2_ablation_levels_visualize.ipynb` (histos por FR, heatmap níveis×ideia, diff).
4. Registrar métricas em `DECISIONS.md`.
5. Só então planejar próxima ablação (se houver).

---

## 6. Setup rápido em máquina nova (server3)

```bash
# 1) Código
git clone git@github.com:diogonunescarv/Lightweight-Face-De-identification.git
cd Lightweight-Face-De-identification
git checkout feature/v2-dtcwt-meta-mask
git pull

# 2) Ambiente
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
# gdown vem no setup FR; insightface/onnxruntime conforme requirements

# 3) FR (gitignored) + pesos small
bash scripts/run_v2_setup_protego_fr.sh

# 4) Conferir mounts
ls /mnt/study-data/dcarvalho/tests/LATEST_v2_abl_* 
ls /mnt/databases/faces/lfw_frontal/faces_train | head

# 5) Retomar ablação de níveis
tmux new -s v2_lv
bash scripts/run_v2_ablation_levels.sh
```

Se `/mnt/study-data` **não** for o mesmo NFS do server5, copiar pelo menos:
- pasta `20261028_161428_v2_v2_abl_lv2_pixel/` (resume)
- pasta `20261028_121837_v2_v2_abl_lv1_pixel/`
- LATEST dos `full_*` (para REUSE lv23)
- opcional: `holdout_merged_seed42/`

---

## 7. Arquivos-chave desta linha experimental

| Papel | Path |
|-------|------|
| Treino V2 | `training/trainer_v2.py`, `deid_optimize.py train-v2` |
| Módulo | `models/dtcwt_meta_mask.py` |
| Ablação máscaras | `scripts/run_v2_ablation_mask_roi.sh` |
| Ablação níveis | `scripts/run_v2_ablation_levels.sh` |
| Setup FR | `scripts/run_v2_setup_protego_fr.sh` |
| Eval | `evaluation/evaluate_v2.py` |
| Viz | `evaluation/visualize_v2.py` |
