# Experimentos — resultados por checkpoint

Registro objetivo de números, separado do DECISIONS.md (que registra o
"por quê"). Preencha após cada `evaluate` / `evaluate-single` relevante.

| Data | Checkpoint | transform-type | target-cos | lambda-id | tau-ssim | steps | cos (val) | euclid (val) | ssim (val) | Observação |
|------|-----------|-----------------|------------|-----------|----------|-------|-----------|--------------|------------|------------|

## Comparação de regiões de máscara (2026-08-30)

Hiperparâmetros fixos (todos os runs abaixo): `dtcwt`, `wavelet-J=3`, `max-wavelet-amp=0.20`, `lambda-wavelet-mag/phase/smooth=0.01/0.01/0.001`, `lambda-id=15`, `target-cos=0.12`, `max-dct-amp=0.12`, `max-flow-px=4.5`, `max-photo-amp=0.15`, `tau-ssim=0.88`, `lambda-ssim=20`, `lambda-pixel=6`, `lr=0.008`, `seed=42`, `steps=5000`, `--lr-scheduler onecycle`. Treino: LFW train; métricas: LFW test (`evaluate-single`, 200 imgs).

| Data | Checkpoint | transform-type | target-cos | lambda-id | tau-ssim | steps | cos (val) | euclid (val) | ssim (val) | Observação |
|------|-----------|-----------------|------------|-----------|----------|-------|-----------|--------------|------------|------------|
| 2026-08-30 | nopxloss/mid_combo_amp20_flow45_nopx/transform.pt | dtcwt | 0.12 | 15 | 0.88 | 5000 | 0.456 | 1.026 | 0.817 | **Baseline reaproveitado (não retreinado).** `--mask-mode fixed` — elipse completa centrada (máscara antiga). |
| 2026-08-30 | mask_comparison/mid_combo_amp20_flow45__mask-full-landmarks/transform.pt | dtcwt | 0.12 | 15 | 0.88 | 5000 | 0.423 | 1.074 | 0.817 | `--mask-mode landmarks --mask-regions full` — máscara completa montada via SCRFD; **versão nova comparável ao baseline.** |
| 2026-08-30 | mask_comparison/mid_combo_amp20_flow45__mask-eyes/transform.pt | dtcwt | 0.12 | 15 | 0.88 | 5000 | 0.874 | 0.502 | 0.866 | `--mask-regions eyes` |
| 2026-08-30 | mask_comparison/mid_combo_amp20_flow45__mask-mouth/transform.pt | dtcwt | 0.12 | 15 | 0.88 | 5000 | 0.971 | 0.241 | 0.900 | `--mask-regions mouth` |
| 2026-08-30 | mask_comparison/mid_combo_amp20_flow45__mask-nose/transform.pt | dtcwt | 0.12 | 15 | 0.88 | 5000 | 0.958 | 0.291 | 0.893 | `--mask-regions nose` |
| 2026-08-30 | mask_comparison/mid_combo_amp20_flow45__mask-eyes+mouth/transform.pt | dtcwt | 0.12 | 15 | 0.88 | 5000 | 0.818 | 0.603 | 0.863 | `--mask-regions eyes,mouth` |
| 2026-08-30 | mask_comparison/mid_combo_amp20_flow45__mask-eyes+nose/transform.pt | dtcwt | 0.12 | 15 | 0.88 | 5000 | 0.804 | 0.627 | 0.854 | `--mask-regions eyes,nose` |
| 2026-08-30 | mask_comparison/mid_combo_amp20_flow45__mask-mouth+nose/transform.pt | dtcwt | 0.12 | 15 | 0.88 | 5000 | 0.907 | 0.430 | 0.889 | `--mask-regions nose,mouth` |
| 2026-08-30 | mask_comparison/mid_combo_amp20_flow45__mask-eyes+mouth+nose/transform.pt | dtcwt | 0.12 | 15 | 0.88 | 5000 | 0.753 | 0.703 | 0.852 | `--mask-regions eyes,nose,mouth` |

Checkpoints completos em `/mnt/study-data/dcarvalho/tests/nopxloss/` (baseline) e `/mnt/study-data/dcarvalho/tests/mask_comparison/` (8 runs novos). Avaliação detalhada: `/mnt/study-data/dcarvalho/metrics/mask_regions/summary_all.csv`.

**Nota:** SSIM das linhas acima medido com escopo restrito à máscara de treino (comportamento pré-tarefa 8). Valores corrigidos para os 5 casos representativos estão na seção [Re-treino SSIM corrigido (2026-09-01)](#re-treino-ssim-corrigido-2026-09-01).

## Comparação elipse vs. faixa (2026-08-31)

Mesmos hiperparâmetros fixos da seção anterior. Treino: `--mask-mode landmarks`, `--mask-shape ellipse|band`, 7 regiões/combinações (sem `full` em band). Métricas: LFW test (`evaluate-single`, 200 imgs). Checkpoints band em `/mnt/study-data/dcarvalho/tests/mask_band_comparison/`; elipse reaproveitados de `/mnt/study-data/dcarvalho/tests/mask_comparison/`. Avaliação consolidada: `/mnt/study-data/dcarvalho/metrics/mask_band_regions/summary_all.csv`.

| Data | Checkpoint | forma | região | transform-type | target-cos | lambda-id | tau-ssim | steps | cos (val) | euclid (val) | ssim (val) | Observação |
|------|-----------|-------|--------|-----------------|------------|-----------|----------|-------|-----------|--------------|------------|------------|
| 2026-08-31 | mask_band_comparison/mid_combo_amp20_flow45__maskband-eyes/transform.pt | band | eyes | dtcwt | 0.12 | 15 | 0.88 | 5000 | 0.728 | 0.719 | 0.861 | `--mask-shape band --mask-regions eyes` |
| 2026-08-30 | mask_comparison/mid_combo_amp20_flow45__mask-eyes/transform.pt | ellipse | eyes | dtcwt | 0.12 | 15 | 0.88 | 5000 | 0.872 | 0.485 | 0.866 | `--mask-shape ellipse --mask-regions eyes` |
| 2026-08-31 | mask_band_comparison/mid_combo_amp20_flow45__maskband-nose/transform.pt | band | nose | dtcwt | 0.12 | 15 | 0.88 | 5000 | 0.926 | 0.363 | 0.889 | `--mask-shape band --mask-regions nose` |
| 2026-08-30 | mask_comparison/mid_combo_amp20_flow45__mask-nose/transform.pt | ellipse | nose | dtcwt | 0.12 | 15 | 0.88 | 5000 | 0.959 | 0.261 | 0.896 | `--mask-shape ellipse --mask-regions nose` |
| 2026-08-31 | mask_band_comparison/mid_combo_amp20_flow45__maskband-mouth/transform.pt | band | mouth | dtcwt | 0.12 | 15 | 0.88 | 5000 | 0.890 | 0.446 | 0.884 | `--mask-shape band --mask-regions mouth` |
| 2026-08-30 | mask_comparison/mid_combo_amp20_flow45__mask-mouth/transform.pt | ellipse | mouth | dtcwt | 0.12 | 15 | 0.88 | 5000 | 0.972 | 0.221 | 0.902 | `--mask-shape ellipse --mask-regions mouth` |
| 2026-08-31 | mask_band_comparison/mid_combo_amp20_flow45__maskband-eyes+mouth/transform.pt | band | eyes+mouth | dtcwt | 0.12 | 15 | 0.88 | 5000 | 0.469 | 1.012 | 0.811 | `--mask-shape band --mask-regions eyes,mouth` |
| 2026-08-30 | mask_comparison/mid_combo_amp20_flow45__mask-eyes+mouth/transform.pt | ellipse | eyes+mouth | dtcwt | 0.12 | 15 | 0.88 | 5000 | 0.817 | 0.583 | 0.863 | `--mask-shape ellipse --mask-regions eyes,mouth` |
| 2026-08-31 | mask_band_comparison/mid_combo_amp20_flow45__maskband-eyes+nose/transform.pt | band | eyes+nose | dtcwt | 0.12 | 15 | 0.88 | 5000 | 0.660 | 0.806 | 0.848 | `--mask-shape band --mask-regions eyes,nose` |
| 2026-08-30 | mask_comparison/mid_combo_amp20_flow45__mask-eyes+nose/transform.pt | ellipse | eyes+nose | dtcwt | 0.12 | 15 | 0.88 | 5000 | 0.809 | 0.596 | 0.855 | `--mask-shape ellipse --mask-regions eyes,nose` |
| 2026-08-31 | mask_band_comparison/mid_combo_amp20_flow45__maskband-nose+mouth/transform.pt | band | nose+mouth | dtcwt | 0.12 | 15 | 0.88 | 5000 | 0.796 | 0.619 | 0.878 | `--mask-shape band --mask-regions nose,mouth` |
| 2026-08-30 | mask_comparison/mid_combo_amp20_flow45__mask-mouth+nose/transform.pt | ellipse | nose+mouth | dtcwt | 0.12 | 15 | 0.88 | 5000 | 0.911 | 0.401 | 0.891 | `--mask-shape ellipse --mask-regions nose,mouth` |
| 2026-08-31 | mask_band_comparison/mid_combo_amp20_flow45__maskband-eyes+nose+mouth/transform.pt | band | eyes+nose+mouth | dtcwt | 0.12 | 15 | 0.88 | 5000 | 0.485 | 0.995 | 0.825 | `--mask-shape band --mask-regions eyes,nose,mouth` |
| 2026-08-30 | mask_comparison/mid_combo_amp20_flow45__mask-eyes+mouth+nose/transform.pt | ellipse | eyes+nose+mouth | dtcwt | 0.12 | 15 | 0.88 | 5000 | 0.761 | 0.670 | 0.853 | `--mask-shape ellipse --mask-regions eyes,nose,mouth` |

Checkpoints band em `/mnt/study-data/dcarvalho/tests/mask_band_comparison/`; elipse em `/mnt/study-data/dcarvalho/tests/mask_comparison/`. Avaliação consolidada: `/mnt/study-data/dcarvalho/metrics/mask_band_regions/summary_all.csv`.

**Nota:** SSIM das linhas acima com escopo restrito à máscara (pré-tarefa 8). Substituições corrigidas na seção de 2026-09-01 para: `eyes+nose+mouth` (ellipse), `eyes`, `eyes+mouth`, `eyes+nose+mouth` (band).

## Re-treino SSIM corrigido (2026-09-01)

Mesmos hiperparâmetros fixos das seções anteriores + `--ssim-region full-landmarks` no treino. Métricas: LFW test (`evaluate-single`, 200 imgs, `--ssim-region full-landmarks`). Scripts: `run_ssim_fix_retrain.sh`, `compare_ssim_fix.sh`.

| Data | Checkpoint | forma | região | transform-type | target-cos | lambda-id | tau-ssim | steps | cos (val) | euclid (val) | ssim (val) | Observação |
|------|-----------|-------|--------|-----------------|------------|-----------|----------|-------|-----------|--------------|------------|------------|
| 2026-09-01 | mask_comparison/mid_combo_amp20_flow45__mask-full-landmarks/transform.pt | ellipse | full | dtcwt | 0.12 | 15 | 0.88 | 5000 | 0.411 | 1.068 | 0.818 | **Reaproveitado (não retreinado).** Treino tarefa 5; métricas reavaliadas com `--ssim-region full-landmarks`. |
| 2026-09-01 | ssim_fix_retrain/mid_combo_amp20_flow45__mask-eyes+mouth+nose__ssim-fixed/transform.pt | ellipse | eyes+nose+mouth | dtcwt | 0.12 | 15 | 0.88 | 5000 | 0.665 | 0.802 | 0.883 | **Substitui** `mask-eyes+mouth+nose` (tarefa 5: cos=0.761, ssim=0.853 com SSIM restrito à máscara). |
| 2026-09-01 | ssim_fix_retrain/mid_combo_amp20_flow45__maskband-eyes__ssim-fixed/transform.pt | band | eyes | dtcwt | 0.12 | 15 | 0.88 | 5000 | 0.652 | 0.818 | 0.872 | **Substitui** `maskband-eyes` (tarefa 7: cos=0.728, ssim=0.861 com SSIM restrito à máscara). |
| 2026-09-01 | ssim_fix_retrain/mid_combo_amp20_flow45__maskband-eyes+mouth__ssim-fixed/transform.pt | band | eyes+mouth | dtcwt | 0.12 | 15 | 0.88 | 5000 | 0.434 | 1.046 | 0.825 | **Substitui** `maskband-eyes+mouth` (tarefa 7: cos=0.469, ssim=0.811 com SSIM restrito à máscara). |
| 2026-09-01 | ssim_fix_retrain/mid_combo_amp20_flow45__maskband-eyes+nose+mouth__ssim-fixed/transform.pt | band | eyes+nose+mouth | dtcwt | 0.12 | 15 | 0.88 | 5000 | 0.437 | 1.045 | 0.820 | **Substitui** `maskband-eyes+nose+mouth` (tarefa 7: cos=0.485, ssim=0.825 com SSIM restrito à máscara). |

Checkpoints em `/mnt/study-data/dcarvalho/tests/mask_comparison/` (full-landmarks reaproveitado) e `/mnt/study-data/dcarvalho/tests/ssim_fix_retrain/` (4 runs novos). Avaliação consolidada: `/mnt/study-data/dcarvalho/metrics/ssim_fix/summary_all.csv`.

## Re-treino SSIM-fixed + early stopping (2026-09-03, item 15)

Mesmos 4 casos da seção anterior + early stopping (`--steps 10000` teto, `--eval-every 100`, `--patience 3`, `--val-max-samples 200`, `--early-stopping-metric score`). Scripts: `run_ssim_scope_earlystop.sh`, `compare_ssim_scope_earlystop.sh`. Métricas: LFW test (`evaluate-single`, 200 imgs, `--ssim-region full-landmarks`). Coluna `steps` = `stopped_step` (treino parou); `best_step` entre parênteses.

| Data | Checkpoint | forma | região | transform-type | target-cos | lambda-id | tau-ssim | steps | cos (val) | euclid (val) | ssim (val) | Observação |
|------|-----------|-------|--------|-----------------|------------|-----------|----------|-------|-----------|--------------|------------|------------|
| 2026-09-03 | ssim_scope_earlystop/mid_combo_amp20_flow45__mask-eyes+mouth+nose__ssim-fixed__earlystop10000/transform.pt | ellipse | eyes+nose+mouth | dtcwt | 0.12 | 15 | 0.88 | 4300 (best 4000) | 0.667 | 0.800 | 0.883 | Equivalente à tarefa 9 (cos=0.665, ssim=0.883). early_stopped=True. ~1148 s. |
| 2026-09-03 | ssim_scope_earlystop/mid_combo_amp20_flow45__maskband-eyes__ssim-fixed__earlystop10000/transform.pt | band | eyes | dtcwt | 0.12 | 15 | 0.88 | 3600 (best 3300) | 0.655 | 0.813 | 0.872 | Equivalente à tarefa 9 (cos=0.652, ssim=0.872). early_stopped=True. ~950 s. |
| 2026-09-03 | ssim_scope_earlystop/mid_combo_amp20_flow45__maskband-eyes+mouth__ssim-fixed__earlystop10000/transform.pt | band | eyes+mouth | dtcwt | 0.12 | 15 | 0.88 | 4300 (best 4000) | 0.435 | 1.045 | 0.825 | Equivalente à tarefa 9 (cos=0.434, ssim=0.825). early_stopped=True. ~1145 s. |
| 2026-09-03 | ssim_scope_earlystop/mid_combo_amp20_flow45__maskband-eyes+nose+mouth__ssim-fixed__earlystop10000/transform.pt | band | eyes+nose+mouth | dtcwt | 0.12 | 15 | 0.88 | 4200 (best 3900) | 0.446 | 1.037 | 0.822 | ≈ tarefa 9 (cos=0.437, ssim=0.820); cos +0.009. early_stopped=True. ~1223 s. |

Checkpoints em `/mnt/study-data/dcarvalho/tests/ssim_scope_earlystop/`. Avaliação: `/mnt/study-data/dcarvalho/metrics/ssim_scope_earlystop/summary_all.csv`.

## Top-3 hiperparâmetros vs baseline (2026-09-04, item 16)

Treino completo dos top 3 trials da busca Optuna (tarefa 14), com flags fixos do comando sugerido + early stopping (`--steps 5000` teto, `--eval-every 100`, `--patience 3`, `--val-max-samples 200`, `--early-stopping-metric score`). Máscara default `fixed` (elipse completa), `--ssim-region full-landmarks`. Baseline `mid_combo_amp20_flow45_nopx` reaproveitado (não retreinado); métricas abaixo reavaliadas com `--ssim-region full-landmarks`. Scripts: `run_best_hparams.sh`, `compare_best_hparams.sh`. Score = ssim-cos.

| Data | Checkpoint | lr | lambda-id | max-flow-px | tau-ssim | lambda-ssim | steps | cos (val) | euclid (val) | ssim (val) | Observação |
|------|-----------|-----|-----------|-------------|----------|-------------|-------|-----------|--------------|------------|------------|
| 2026-09-04 | nopxloss/mid_combo_amp20_flow45_nopx/transform.pt | 0.008 | 15 | 4.5 | 0.88 | 20.0 | 5000 | 0.456 | 1.026 | 0.862 | **Baseline reaproveitado** (reavaliado full-landmarks). Score=0.406. |
| 2026-09-04 | best_hparams/hparam_search__trial26/transform.pt | 0.0144483 | 22.78 | 5.18 | 0.922 | 21.2 | 1800 (best 1500) | 0.360 | 1.119 | 0.802 | Busca curta score=0.4345; full score=0.442. early_stopped. ~613 s. |
| 2026-09-04 | best_hparams/hparam_search__trial25/transform.pt | 0.0104433 | 22.41 | 5.23 | 0.922 | 21.3 | 2000 (best 1700) | 0.356 | 1.122 | 0.800 | Busca curta score=0.4322; full score=0.444 (**melhor score**). early_stopped. ~547 s. |
| 2026-09-04 | best_hparams/hparam_search__trial28/transform.pt | 0.0146674 | 24.96 | 4.58 | 0.920 | 29.5 | 1900 (best 1600) | 0.379 | 1.100 | 0.816 | Busca curta score=0.4288; full score=0.437. early_stopped. ~551 s. |

Checkpoints novos em `/mnt/study-data/dcarvalho/tests/best_hparams/`. Avaliação consolidada: `/mnt/study-data/dcarvalho/metrics/best_hparams/summary_all.csv`.

## Polar complex vs legacy — freq-only (2026-09-09)

Protocolo: `--disable-flow --disable-photo`, `dtcwt` J=3, máscara **full-landmarks** (`--mask-mode landmarks --mask-regions full --mask-shape ellipse`), `--ssim-region full-landmarks`, early stopping (`eval-every 100`, `patience 3`, métrica `score`), `max-wavelet-amp=0.20`. Scripts: `run_wavelet_polar_ab.sh`, `compare_wavelet_polar_ab.sh`. **Não comparar** com ckpts combo flow+photo sem ressalva.

| Data | Checkpoint | polar-mode | target-cos | lambda-id | tau-ssim | steps | cos (val) | euclid (val) | ssim (val) | Observação |
|------|-----------|------------|------------|-----------|----------|-------|-----------|--------------|------------|------------|
| 2026-09-09 | wavelet_polar_ab/freq_legacy/transform.pt | legacy | 0.12 | 15 | 0.88 | 4000 (best 3700) | 0.996 | 0.090 | 0.998 | early_stopped. Params mag saturados (tanh≈1); desid. quase nula. |
| 2026-09-09 | wavelet_polar_ab/freq_complex/transform.pt | complex | 0.12 | 15 | 0.88 | 3000 (best 2700) | 0.994 | 0.103 | 0.997 | early_stopped. Ligeiramente melhor que legacy (Δcos −0.002), mas ainda cos≈1. **Baseline freq-only** para fases seguintes. |

Outs: `/mnt/study-data/dcarvalho/tests/wavelet_polar_ab/` · metrics: `/mnt/study-data/dcarvalho/metrics/wavelet_polar_ab/summary_all.csv`.

**Leitura:** com amp≤0.20 e só wavelet, o otimizador satura os 36 params e o embedding quase não muda. Próximo: Fase 2 com amps maiores (mag/phase desacoplados).

## Amp mag/phase split — freq-only (2026-09-12)

Scripts: `run_wavelet_amp_split.sh`, `compare_wavelet_amp_split.sh`. Polar=`complex`, full-landmarks, freq-only. Early stopping. Score = ssim−cos.

| Data | Checkpoint | mag-amp | phase-amp | steps | cos (val) | euclid (val) | ssim (val) | Observação |
|------|-----------|---------|-----------|-------|-----------|--------------|------------|------------|
| 2026-09-12 | wavelet_amp_split/freq_cx_mag030_ph040/transform.pt | 0.30 | 0.40 | 4000 (best 3700) | 0.984 | 0.171 | 0.990 | score≈0.006 |
| 2026-09-12 | wavelet_amp_split/freq_cx_mag040_ph020/transform.pt | 0.40 | 0.20 | 7700 (best 7400) | 0.981 | 0.186 | 0.991 | score≈0.010 |
| 2026-09-12 | wavelet_amp_split/freq_cx_mag040_ph040/transform.pt | 0.40 | 0.40 | 5500 (best 5200) | 0.976 | 0.209 | 0.988 | score≈0.011 |
| 2026-09-12 | wavelet_amp_split/freq_cx_mag040_ph080/transform.pt | 0.40 | 0.80 | 4200 (best 3900) | 0.959 | 0.275 | 0.975 | score≈0.015 |
| 2026-09-12 | wavelet_amp_split/freq_cx_mag060_ph040/transform.pt | 0.60 | 0.40 | 4700 (best 4400) | 0.954 | 0.291 | 0.979 | score≈0.025 |
| 2026-09-12 | wavelet_amp_split/freq_cx_mag080_ph080/transform.pt | 0.80 | 0.80 | 4700 (best 4400) | 0.914 | 0.397 | 0.961 | **Melhor score≈0.047** e menor cos do grid. SSIM ainda alto. |

Outs: `/mnt/study-data/dcarvalho/tests/wavelet_amp_split/` · metrics: `/mnt/study-data/dcarvalho/metrics/wavelet_amp_split/summary_all.csv`.

**Leitura:** subir amp reduz cos de forma monotônica, mas o piso com 36 params globais fica em cos≳0.91 (longe de &lt;0.2). SSIM permanece &gt;0.96. **Baseline amp para Fase 3:** mag=0.80, phase=0.80 (`freq_cx_mag080_ph080`). Próximo: expressividade (`--wavelet-per-channel`).

## Expressividade wavelet — freq-only (fase 3, sequencial)

HPs base: polar `complex`, mag=0.80, phase=0.80, full-landmarks, freq-only. Scripts: `run_wavelet_per_channel.sh`, `run_wavelet_spatial.sh`, `run_wavelet_yl_gain.sh` (+ `compare_*`).

| Data | Checkpoint | modo | params approx | steps | cos (val) | euclid (val) | ssim (val) | Observação |
|------|-----------|------|---------------|-------|-----------|--------------|------------|------------|
| 2026-09-12 | wavelet_amp_split/freq_cx_mag080_ph080/transform.pt | global (ref) | 36 | 4700 (best 4400) | 0.914 | 0.397 | 0.961 | Baseline amp W2. |
| 2026-09-12 | wavelet_per_channel/freq_cx_perchannel/transform.pt | per-channel | 108 | 4300 (best 4000) | 0.914 | 0.398 | 0.961 | Sem ganho vs global. |
| 2026-09-12 | wavelet_spatial/freq_cx_spatial_g8/transform.pt | spatial g=8 | 2304 | 10000 (best 9700) | 0.816 | 0.583 | 0.951 | **Melhor isolado** da linha (Δcos −0.10 vs global). |
| 2026-09-12 | wavelet_yl_gain/freq_cx_yl_gain/transform.pt | Yl gain | 36+1 | 5500 (best 5200) | 0.906 | 0.417 | 0.959 | Ganho leve vs global (Δcos −0.008); Yl satura em escala≈0.95. **Pior que spatial.** |

Outs: spatial `/mnt/study-data/dcarvalho/tests/wavelet_spatial/` · Yl `/mnt/study-data/dcarvalho/tests/wavelet_yl_gain/` · metrics espelhadas em `/mnt/study-data/dcarvalho/metrics/wavelet_{spatial,yl_gain}/`.

**Síntese fase 3:** per-channel inútil; Yl ajuda pouco; **spatial g=8 domina**. Cos ainda ≫0.2. Próximo natural (fora da ordem estrita do plano): **spatial g=8 + Yl** e/ou `g>8`.

## Spatial + Yl — freq-only (2026-09-12, W4)

Script: `run_wavelet_spatial_yl.sh` / `compare_wavelet_spatial_yl.sh`. Polar complex, mag/phase 0.80, `--wavelet-spatial-grid 8 --wavelet-yl-gain` (yl-amp 0.05), full-landmarks, freq-only.

| Data | Checkpoint | modo | steps | cos (val) | euclid (val) | ssim (val) | Observação |
|------|-----------|------|-------|-----------|--------------|------------|------------|
| 2026-09-12 | wavelet_spatial/freq_cx_spatial_g8/transform.pt | spatial only | 10000 (best 9700) | 0.816 | 0.583 | 0.951 | Ref W3b. |
| 2026-09-12 | wavelet_spatial_yl/freq_cx_spatial_g8_yl/transform.pt | spatial+Yl | 10000 (best 9700) | 0.793 | 0.619 | 0.947 | **Melhor da linha** (Δcos −0.023 vs spatial). Yl satura em escala≈0.95. early_stopped=False. |

Outs: `/mnt/study-data/dcarvalho/tests/wavelet_spatial_yl/` · metrics: `/mnt/study-data/dcarvalho/metrics/wavelet_spatial_yl/summary_all.csv`.

**Leitura W4:** Yl é aditivo sobre spatial (sem colapso de SSIM). Bound `max-wavelet-yl-amp=0.05` satura → próximo: subir yl-amp e/ou `g=12`.

## Spatial+Yl amp / g sweep — freq-only (2026-09-14, W5)

Scripts: `run_wavelet_spatial_yl_sweep.sh` / `compare_wavelet_spatial_yl_sweep.sh`. Polar complex, mag/phase 0.80, full-landmarks, freq-only. Score = ssim−cos. Ref W4: g=8 yl=0.05 → cos=0.793, ssim=0.947, score≈0.155.

| Data | Checkpoint | g | yl-amp | steps | cos (val) | euclid (val) | ssim (val) | Observação |
|------|-----------|---|--------|-------|-----------|--------------|------------|------------|
| 2026-09-12 | wavelet_spatial_yl/freq_cx_spatial_g8_yl/transform.pt | 8 | 0.05 | 10000 (best 9700) | 0.793 | 0.619 | 0.947 | Ref W4. score≈0.155 |
| 2026-09-14 | wavelet_spatial_yl_sweep/freq_cx_g12_yl005/transform.pt | 12 | 0.05 | 10000 (best 9900) | 0.779 | 0.641 | 0.947 | score≈0.168; g↑ ajuda pouco com yl baixo |
| 2026-09-14 | wavelet_spatial_yl_sweep/freq_cx_g8_yl010/transform.pt | 8 | 0.10 | 10000 (best 9700) | 0.766 | 0.660 | 0.941 | score≈0.175 |
| 2026-09-14 | wavelet_spatial_yl_sweep/freq_cx_g12_yl010/transform.pt | 12 | 0.10 | 10000 (best 9700) | 0.753 | 0.679 | 0.940 | score≈0.187 |
| 2026-09-14 | wavelet_spatial_yl_sweep/freq_cx_g8_yl015/transform.pt | 8 | 0.15 | 9900 (best 9600) | 0.736 | 0.703 | 0.930 | **Melhor score≈0.194** e menor cos do sweep. Yl satura em escala=0.85. |

Outs: `/mnt/study-data/dcarvalho/tests/wavelet_spatial_yl_sweep/` · metrics: `/mnt/study-data/dcarvalho/metrics/wavelet_spatial_yl_sweep/summary_all.csv`.

**Leitura W5:** subir `yl-amp` é a alavanca dominante; `g=12` vs `g=8` no mesmo yl dá ganho menor. **Baseline freq-only final:** `g=8`, `yl-amp=0.15`, mag/phase=0.80 (cos=0.736, ssim=0.930). Cos ainda ≫0.2 → retornos decrescentes na linha só-frequência.
