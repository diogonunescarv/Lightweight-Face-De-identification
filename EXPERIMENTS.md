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
