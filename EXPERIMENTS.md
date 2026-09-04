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
