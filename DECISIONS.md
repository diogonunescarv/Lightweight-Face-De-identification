# Decisões do projeto

Ordem cronológica, mais recente no topo. Referencie este arquivo com `@DECISIONS.md` no início de sessões do Cursor que dependam de contexto
histórico. Toda decisão de design não óbvia, resultado de experimento relevante, ou motivo de rejeitar uma abordagem deve virar uma entrada aqui.

Formato sugerido por entrada:
- Data
- Contexto (o problema/dúvida)
- Decisão (o que foi feito)
- Resultado (métricas ou observação, mesmo que "a validar")

---

## 2026-10-26 — Início V2 — DT-CWT meta-máscara + ROI (`feature/v2-dtcwt-meta-mask`)
Contexto: a V1 (DCT/wavelet global + flow/photo) saturava o trade-off cos×SSIM e máscaras parciais falhavam em desidentificar. O projeto irmão `dtcwt_mask_optimization` (Phase 3.3) validou meta-máscara espacial DT-CWT + SignSGD + loss cos+SSIM hinge.
Decisão:
- Branch `feature/v2-dtcwt-meta-mask`; fonte da verdade: `V2.md`.
- Substituir warp/photo/DCT no path V2 por **DT-CWT meta-máscara** (Δmag/Δphase espaciais); **manter** máscaras ROI (`models/masks.py`, mesmas flags).
- Ordem fixa de composição ROI via `--roi-compose`: **Ideia 1 `pixel` primeiro** (ΔW full → IDT-CWT → residual⊙M → L∞); **Ideia 2 `wavelet` depois** (ΔW⊙M_ℓ; sem regressão no path `pixel`).
- Sem warm-start online nesta V2 (→ V2.1). Notebooks fora do escopo até Step 9 (só viz).
- Código V1 permanece atrás de CLI legada; checkpoints V1 não carregam no módulo V2.
Resultado: Step 0 concluído (branch + higiene). Próximo: Step 1 (scaffold + inventário).

## 2026-09-21 — Filtro frontal pose≤25° + oclusões (TODO item 6)
Contexto: treinos/avaliações precisam de faces quase frontais sem oclusões grosseiras (óculos escuros, máscaras). Alinhamento por landmarks (itens 1–2) é ortogonal; aqui só **seleção**. LFW tem dois splits (`faces_train` / `faces_test`) a preservar.
Decisão:
- Notebook canônico: `notebooks/filter_frontal_datasets.ipynb` (inspeção LFW → helpers → QA visual → batch retomável com JSONL). Espelho CLI opcional: `scripts/filter_frontal_datasets.py`.
- Pose: InsightFace `buffalo_m` + `landmark_3d_68` (`1k3d68.onnx`), `det_size=(224,224)`; keep se `max(|pitch|,|yaw|,|roll|) ≤ 25`. **Não** altera `models/face_detector.py` (treino continua só com detection).
- Oclusão (heurística, sem dep nova): óculos escuros se `eye_mean < 55` ou (`eye_mean < 65` e `eye_std < 22`); máscara se `mouth_std < 16` ou (`mouth_std < 22` e `|mouth−cheek| < 18`). Thresholds calibráveis no notebook.
- Saídas irmãs (sem sobrescrever fontes): `celebahq_pp_frontal`, `lfw_frontal/{faces_train,faces_test}`.
Resultado (batch completo; `notebooks/outputs/filter_frontal/summary.json`):
- **celebahq_pp:** kept **24722 / 29995** (desc. 17.6%) — pose 3466, sunglasses 1082, mask 721, no_face 4
- **lfw/faces_train:** kept **8226 / 10586** (desc. 22.3%) — mask 1456, sunglasses 456, pose 440, no_face 8
- **lfw/faces_test:** kept **2029 / 2647** (desc. 23.3%) — mask 383, sunglasses 114, pose 118, no_face 3
- Observação: em LFW a razão `mask` é alta (possível falso positivo da heurística de boca lisa) — validar visualmente na seção QA do notebook antes de confiar cegamente; se necessário, afrouxar `MOUTH_*` e re-filtrar.
- Scripts de treino ainda apontam aos paths não-frontais; trocar `--data` para `*_frontal` quando desejado.

## 2026-09-15 — Inversão local de fase φ+π sob máscara full (TODO item 5)
Contexto: segunda referência agressiva em fase DT-CWT — determinística (cristas↔vales / “negativo estrutural”), mag intacta, mesmas imagens e máscara full do item 4.
Decisão: seção (g) em `notebooks/dtcwt_modifications_tests.ipynb` — `invert_phase_yh` (φ+π ≡ −c) reusando `dtcwt_*` / blend / métricas da (f); tabela `rows_phase_invert` + comparativo `rows_compare` vs `rows_phase_random`.
Resultado (n=5, mesmas imgs do item 4):
- SSIM_mask / SSIM_full φ+π (média): **0.2812 / 0.6869**
- cos / euclid φ+π (média): **0.7388 / 0.7178**
- Demo 1 img: cos=0.7145, ssim_full=0.7487, ssim_mask=0.4788
- vs rand (médias): cos_rnd **0.8141** vs cos_inv **0.7388**; SSIM_mask_rnd **0.6733** vs SSIM_mask_inv **0.2812**
- Qual derruba mais identidade (menor cos)? **inv φ+π** (Δcos ≈ −0.075 vs rand)
- Qual derruba mais SSIM_mask? **inv φ+π** (quase 2.4× mais dano estrutural que rand)
- Observação: φ+π gera artefato mais “negativo estrutural” / contraste invertido na máscara (SSIM_mask cai a ~0.28); rand parece ruído textural com SSIM_mask mais alto (~0.67). Em ambos os casos FaceNet ainda fica com cos ≫ 0.5 (inv ~0.74) — longe do alvo cos&lt;0.5 / &lt;0.2. Conclusão: só mexer em fase de Yh (rand ou +π), com mag+Yl intactos, **não basta** para desidentificação útil; próximo é mexer em mag/Yl ou reintroduzir flow/photo (W6).

## 2026-09-15 — Fase aleatória U[-π,π] sob máscara full (TODO item 4)
Contexto: referência de agressividade máxima em fase DT-CWT (congruência de fase destruída), mag intacta, só dentro da máscara full.
Decisão: seção (f) no notebook `dtcwt_modifications_tests.ipynb` — helpers `dtcwt_forward/inverse`, `randomize_phase_yh`, blend com máscara; métricas FaceNet + `ssim_index` / `ssim_index_masked`. Dataset: 5 imgs LFW (`PHASE_N=5`).
Resultado (n=5):
- SSIM_mask recon fiel (média): **0.9986** (IDT-CWT fiel ≈1, OK)
- SSIM_mask / SSIM_full fase rand (média): **0.6733 / 0.8567**
- cos / euclid fase rand (média): **0.8141 / 0.6012**
- Demo 1 img: cos=0.750, ssim_full=0.893, ssim_mask=0.777
- Hipótese “SSIM_mask ≈ 0 e identidade destruída”: **rejeitada**. Face visualmente bem danificada (ruído/textura na máscara), mas FaceNet mantém cos ≫ 0.5 (média 0.81; min observado ~0.73). Longe do alvo cos&lt;0.5 e do alvo prático cos&lt;0.2.
- Interpretação: destruir só a fase dos detalhes (Yh), preservando magnitude e Yl, **não basta** para quebrar embedding — identidade FaceNet parece ancorada em energia/baixa frequência ainda intactas. Próximo: item 5 (inversão φ+π) para contraste; depois estratégias que mexam em mag/Yl ou reintroduzam flow/photo (W6).

## 2026-09-15 — Notebook base DT-CWT (TODO item 3)
Contexto: exploração manual de modificações DT-CWT sob máscara full; item 3 pedia células (a)–(e) sem dependência nova.
Decisão:
- Notebook `notebooks/dtcwt_modifications_tests.ipynb`: imports do projeto (`DTCWTDecimated`/`pytorch_wavelets`, `FaceNetEmbedder`, SCRFD, `build_region_mask`).
- Datasets: preferir `celebahq_pp_norm` / `lfw_norm/faces_train`; se ausentes, fallback para `celebahq_pp` e `lfw/faces_train` (paths dos scripts wavelet).
- LBP via `skimage.feature.local_binary_pattern` (P=8, R=1); `scikit-image` adicionado ao `requirements.txt`.
Resultado: **a validar** pelo usuário (célula a célula); sem execução pelo agente.

## 2026-09-14 — Sweep spatial×Yl (W5) + fechamento da linha freq-only
Contexto: W4 (g=8, yl=0.05, cos=0.793) saturava Yl; sweep g∈{8,12} × yl∈{0.05,0.10,0.15}.
Decisão: **baseline freq-only final = g=8, yl-amp=0.15, mag/phase=0.80** (`freq_cx_g8_yl015`). Encerrar ablações só-frequência: cos caiu ~0.99→0.74 com ssim≥0.93, mas longe do alvo cos&lt;0.2; próximo salto útil é **reintroduzir flow/photo** em cima desse espectro (ou aceitar freq-only como contribuição/ablation da dissertação).
Resultado (evaluate-single, 200 imgs; score=ssim−cos):
- g8 yl0.05 (W4): cos=0.793, ssim=0.947, score≈0.155
- g12 yl0.05: cos=0.779, ssim=0.947, score≈0.168
- g8 yl0.10: cos=0.766, ssim=0.941, score≈0.175
- g12 yl0.10: cos=0.753, ssim=0.940, score≈0.187
- **g8 yl0.15: cos=0.736, ssim=0.930, score≈0.194** (melhor)
- Yl sempre satura no bound (`scale=1−yl_amp`). g=12 não supera g=8 no mesmo yl-amp alto.
- Opcional residual (1 run): g=8 yl=0.20 — senão fechar a linha.

## 2026-09-12 — Spatial g=8 + Yl (freq-only, W4)
Contexto: combinar melhor isolado (spatial) com Yl; mag/phase 0.80, yl-amp 0.05, full-landmarks, freq-only.
Decisão: **spatial+Yl é o novo melhor** da linha (cos=0.793). Yl continua saturando no bound 0.05 → vale subir `max-wavelet-yl-amp` e testar `g=12`.
Resultado (evaluate-single, 200 imgs):
- spatial+Yl: cos=0.793, euclid=0.619, ssim=0.947 (10000 steps, best 9700, early_stopped=False); score≈0.155
- vs spatial only: Δcos −0.023, Δssim −0.004
- Ainda cos≫0.2. Próximo: sweep yl-amp ∈ {0.10,0.15} e g ∈ {8,12}.

## 2026-09-12 — Yl gain isolado (freq-only, W3c) + fechamento fase 3
Contexto: último item isolado do plano; mag/phase 0.80 + `--wavelet-yl-gain` (amp 0.05) sem spatial.
Decisão: fase 3 **encerrada**. Ranking freq-only: **spatial g=8 ≫ Yl ≈ global ≈ per-channel**. Próximo experimento: **spatial+Yl** (e opcionalmente g maior), não mais ablações isoladas pequenas.
Resultado (evaluate-single, 200 imgs):
- Yl: cos=0.906, euclid=0.417, ssim=0.959 (stopped 5500 / best 5200); ganho Yl satura em escala≈0.95
- vs global: Δcos −0.008; vs spatial g=8 (cos=0.816): bem inferior
- Plano original (polar → amp → per-ch → spatial → Yl) concluído em código+métricas.

## 2026-09-12 — Spatial grid g=8 (freq-only, W3b)
Contexto: per-channel não ganhou; testar mapas espaciais `(1,6,J,8,8)×2` ≈2304 params, mag/phase 0.80, polar complex, full-landmarks, freq-only.
Decisão: **spatial g=8 é o melhor checkpoint freq-only até agora**; vira referência para evoluções seguintes (combo com Yl / g maior).
Resultado (evaluate-single, 200 imgs):
- spatial g=8: cos=0.816, euclid=0.583, ssim=0.951 (steps 10000, best 9700, early_stopped=False)
- vs global 0.80/0.80: Δcos −0.098, Δeuclid +0.186, Δssim −0.010
- Ainda cos≫0.2, mas primeira alavanca estrutural útil. Próximo no plano: Yl gain isolado (`run_wavelet_yl_gain.sh`).

## 2026-09-12 — Per-channel wavelet (freq-only, W3a)
Contexto: 108 params `(1,3,6,J)×2` com mag/phase 0.80, polar complex, full-landmarks, freq-only.
Decisão: **não** adotar per-channel como upgrade — métricas idênticas ao global 36-param.
Resultado (evaluate-single, 200 imgs):
- per-channel: cos=0.914, euclid=0.398, ssim=0.961 (stopped 4300 / best 4000)
- ref mag080/ph080: cos=0.914, euclid=0.397, ssim=0.961
- Próximo: `--wavelet-spatial-grid 8` (~2304 params), mesmos amps 0.80/0.80.

## 2026-09-12 — Amp mag/phase split (freq-only)
Contexto: polar `complex` com amp 0.20 deixava cos≈0.99; grid com mag∈{0.3…0.8} e phase∈{0.2…0.8}, full-landmarks, `--disable-flow --disable-photo`.
Decisão: baseline amp da linha = **mag=0.80, phase=0.80** (`freq_cx_mag080_ph080`) — melhor score e menor cos do grid, SSIM ainda 0.961.
Resultado (evaluate-single, 200 imgs):
- cos cai de ~0.98 (amps baixos) até **0.914** (0.80/0.80); euclid até 0.397; ssim sempre ≥0.961.
- Ainda **longe** do alvo prático (cos&lt;0.2). 36 params globais parecem insuficientes → seguir Fase 3 com expressividade, partindo de 0.80/0.80.
- Ordem: per-channel → spatial g=8 → Yl gain.

## 2026-09-09 — Resultado A/B polar freq-only (amp 0.20)
Contexto: treinos `freq_legacy` vs `freq_complex` com full-landmarks, `--disable-flow --disable-photo`, early stopping.
Decisão: adotar **`complex` como baseline** da linha espectral (ligeira vantagem em cos/euclid); manter `legacy` só para ckpts antigos.
Resultado (evaluate-single, 200 imgs, full-landmarks):
- legacy: cos=0.996, euclid=0.090, ssim=0.998 (stopped 4000 / best 3700)
- complex: cos=0.994, euclid=0.103, ssim=0.997 (stopped 3000 / best 2700)
- Ambos **falham** em desidentificação (cos≈1). Pesos mag saturam em ±tanh≈1 → o bound `max-wavelet-amp=0.20` é o gargalo, não o modo polar.
- Próximo: Fase 2 (`run_wavelet_amp_split.sh`) com amps maiores e mag/phase desacoplados, ainda freq-only + complex.

## 2026-09-09 — Polar DT-CWT complexo + linha freq-only
Contexto: a perturbação wavelet pretendia operar em forma polar nos coeficientes complexos, mas o pytorch-wavelets devolve `Yh[j]` como `[B,C,6,H,W,2]` (real/imag). `torch.abs`/`torch.angle` no tensor real tratavam real e imag separadamente (modo legado). Além disso, a evolução espectral precisa ser isolada de flow/photo.
Decisão:
- `--wavelet-polar-mode complex|legacy` (default treino: `complex`); ckpts sem metadado carregam como `legacy`.
- Polar complexo: `view_as_complex` → escala/fase → `view_as_real` antes da inversa.
- Amps separados: `--max-wavelet-mag-amp` / `--max-wavelet-phase-amp` (+ `--wavelet-level-amp-scales`).
- Expressividade (um modo por vez): `--wavelet-per-channel`, `--wavelet-spatial-grid g`, `--wavelet-yl-gain`.
- **Treinos desta linha:** sempre `--disable-flow --disable-photo` (freq-only); máscara de transformação `--mask-mode landmarks --mask-regions full` (full-landmarks) + `--ssim-region full-landmarks`. HPs ajustáveis visando SSIM alto.
- Notebook: `notebooks/wavelet_polar_debug.ipynb`. Scripts: `scripts/run_wavelet_polar_ab.sh`, `run_wavelet_amp_split.sh`, `run_wavelet_per_channel.sh`, `run_wavelet_spatial.sh`, `run_wavelet_yl_gain.sh` (+ `compare_*`).
Resultado: código e scripts prontos; A/B polar preenchido em EXPERIMENTS (amp 0.20 insuficiente — ver entrada acima).

## 2026-09-04 — Normalização por landmarks (celebahq_pp, item 1)
Contexto: `celebahq_pp` já está em 224×224 via `norm_crop`, mas landmarks ainda variam entre imagens; falta refine ao template ArcFace fixo. Um template alternativo (face mais alta/maior, típico de pipeline de template visual em 112×112) em cima de `celebahq_pp` força scale≈1.12 e ty≈−50; com `BORDER_REFLECT_101` gerava espelhamento na base.
Decisão:
- Notebook `notebooks/normalize_landmarks.ipynb` reusa `models/face_detector.py` (SCRFD); `TEMPLATE_NAME` escolhe o destino.
- **Default `arcface`:** `arcface_dst * (224/112)` — mesmo destino do `norm_crop` → `celebahq_pp_norm/`.
- **Opcional `visual`:** template do pipeline externo (olhos ~y=68) — só comparação / célula de teste no notebook (`plot_before_after(..., template="visual")`); não é o padrão do projeto. Batch opcional em `celebahq_pp_norm_visual/`.
- Similaridade: `cv2.estimateAffinePartial2D(..., method=LMEDS)` → `warpAffine` 224×224, `BORDER_CONSTANT` (borderValue=0), JPEG q=95.
- `_to_bgr_uint8`: ndarray tratado como BGR (sem heurística RGB) — evita inversão de cores no refine.
Resultado (batch `arcface`):
- processadas / descartadas: **29989 / 6** (entrada 29995)
- taxa de descarte: **0.020%**
- tamanho final: **29989** JPEGs em `/mnt/study-data/dcarvalho/datasets/celebahq_pp_norm/`
- validação visual 3 exemplos (`arcface`): OK (refine fino, cores preservadas)

## 2026-09-04 — Normalização por landmarks (LFW, item 2)
Contexto: repetir o refine da tarefa 1 no LFW. Fontes do projeto em `lfw/faces_{train,test}` já estão em 224×224; o TODO prevê fontes 256×256 com center crop antes do alinhamento.
Decisão:
- Mesmo notebook `normalize_landmarks.ipynb`, seção Item 2.
- `center_crop_to(256→224)` sem resize; se já 224×224, crop é identidade.
- Template default = `TEMPLATE_NAME` (`arcface`); saída `/mnt/study-data/dcarvalho/datasets/lfw_norm/{faces_train,faces_test}`.
- Validação 3 exemplos + batch manual (não disparar automaticamente).
Resultado: **a validar** após run manual — preencher:
- processadas / descartadas: ___ / ___
- taxa de descarte: ___%
- tamanho final (train+test): ___
- crop aplicado (256→224 vs identidade em 224): ___

## 2026-09-04 — Top-3 HP da busca vs base fixa (item 16)
Contexto: busca Optuna (tarefa 14) encontrou trials com score > baseline curto; falta treino completo e comparação justa com `mid_combo_amp20_flow45_nopx` antes de trocar a base dos próximos experimentos de máscara.
Decisão:
- Scripts: `scripts/run_best_hparams.sh` e `scripts/compare_best_hparams.sh`.
- Treinar só os top 3: trial26 / trial25 / trial28 (nomes `hparam_search__trial*`).
- Flags fixos = comando sugerido pela tarefa 14; variam só lr, lambda-id, max-flow-px, tau-ssim, lambda-ssim.
- Máscara `fixed` (default) + `--ssim-region full-landmarks` — comparável ao baseline nopx.
- Baseline `mid_combo_amp20_flow45_nopx` só reavaliado (não retreinado).
- Outputs: `/mnt/study-data/dcarvalho/tests/best_hparams/` e `/mnt/study-data/dcarvalho/metrics/best_hparams/summary_all.csv`.
Resultado (evaluate-single, 200 imgs, full-landmarks; score = ssim−cos):
- **Melhor score:** trial25 (score=0.444; cos=0.356, ssim=0.800), seguido de trial26 (0.442; cos=0.360, ssim=0.802) e trial28 (0.437; cos=0.379, ssim=0.816). Baseline nopx: score=0.406 (cos=0.456, ssim=0.862).
- Os 3 trials **superam a base em score e cos**, mas **caem o SSIM para ~0.80–0.82** (abaixo do limiar prático 0.85; baseline fica em 0.862).
- Early stopping: pararam em 1800–2000 steps (best ~1500–1700), bem abaixo do teto 5000.
- **Não substitui** `mid_combo_amp20_flow45` como base padrão para comparações de máscara (tarefas futuras no estilo 5/7/9): a qualidade perceptual piora demais. Os HP otimizados ficam como variante **agressiva** (quando o objetivo for maximizar score/cos). Tarefas 5/7/9 **não** são invalidadas.
- Se no futuro priorizar só desidentificação (aceitando ssim≈0.80), partir do trial25.

## 2026-09-03 — Re-treino SSIM-fixed com early stopping (item 15)
Contexto: tarefa 9 treinou 4 máscaras com `--ssim-region full-landmarks` e `--steps 5000` fixos (sem early stopping). Com a tarefa 13 disponível, repetir os mesmos 4 casos com teto `--steps 10000` + early stopping para ver se cos/euclid/ssim se mantêm com menos (ou pelo menos ≤) custo de GPU.
Decisão:
- Scripts: `scripts/run_ssim_scope_earlystop.sh` e `scripts/compare_ssim_scope_earlystop.sh`.
- Mesmos HP base da tarefa 9 + `--ssim-region full-landmarks`.
- Early stopping: `--eval-every 100 --patience 3 --val-max-samples 200 --early-stopping-metric score` (alinhado à busca de HP).
- Nomes com sufixo `__earlystop10000` (não sobrescreve `__ssim-fixed` da tarefa 9).
- Outputs: `/mnt/study-data/dcarvalho/tests/ssim_scope_earlystop/` e `/mnt/study-data/dcarvalho/metrics/ssim_scope_earlystop/summary_all.csv` (só os 4 early-stop; comparação com tarefa 9 via EXPERIMENTS.md / metrics/ssim_scope).
Resultado:
- stopped / best / elapsed: eyes+nose+mouth ellipse 4300/4000/~1148s; maskband-eyes 3600/3300/~950s; maskband-eyes+mouth 4300/4000/~1145s; maskband-eyes+nose+mouth 4200/3900/~1223s. Todos `early_stopped=True`.
- cos/euclid/ssim **equivalentes** à tarefa 9 (diferenças ≤ ~0.01); o caso band eyes+nose+mouth teve cos ligeiramente pior (0.446 vs 0.437).
- Economia vs 5000 fixos: ~14–28% menos steps (parou ~3600–4300).
- **Sim — early stopping deve virar padrão** nos próximos treinos de comparação de máscara (mesmo critério score, eval-every/patience alinhados à busca), com teto generoso (≥5000 ou 10000).

## 2026-09-02 — Busca automática de hiperparâmetros (Optuna, item 14)
Contexto: hiperparâmetros ajustados manualmente entre experimentos; base `mid_combo_amp20_flow45_nopx` fixada "na mão".
Decisão:
- Script `scripts/run_hparam_search.py` com Optuna (TPE + MedianPruner).
- Espaço de busca v1: `--lr` log [2e-3, 2e-2], `--lambda-id` [5, 25], `--max-flow-px` [2, 6], `--tau-ssim` [0.85, 0.95], `--lambda-ssim` [10, 40].
- Demais HP fixos (DT-CWT, wavelet, target-cos 0.12, lambda-pixel 6.0, etc.).
- Trials curtos (--steps 1000 default) com early stopping; objetivo = score = ssim_mean - cos_mean.
- CSV incremental `hparam_search_results.csv`; study SQLite `optuna_study.db`; wrapper `scripts/run_hparam_search.sh`.
- `train()` retorna dict com métricas do melhor checkpoint (para caller programático).
Resultado (study em `/mnt/study-data/dcarvalho/tests/hparam_search/`):
- n_trials: 30 (22 complete, 8 pruned); steps/trial teto: 1000.
- Melhor (trial 26): lr=0.0144483, lambda-id=22.78, max-flow-px=5.18, tau-ssim=0.922, lambda-ssim=21.2 → score=0.4345, cos=0.374, ssim=0.808.
- Top-3: trial26 (0.4345), trial25 (0.4322), trial28 (0.4288) — promovidos ao treino completo (item 16).
- **Não substitui** sozinho a base `mid_combo_amp20_flow45` (decisão final no item 16 após treino completo).

## 2026-09-02 — Early stopping por métricas de validação (`--early-stopping`)
Contexto: treinos fixos em 5000 steps desperdiçam GPU após convergência e não protegem contra piora nas últimas etapas (item 13).
Decisão:
- Validação periódica via `evaluation/validate.py` → `compute_validation_metrics` (mesmas métricas de `evaluate`: cos, euclid, ssim).
- Score padrão: `score = ssim_mean - cos_mean` (maximizar). Alternativas: `--early-stopping-metric euclid|ssim|cos`.
- CLI opt-in: `--early-stopping`, `--val-data` (obrigatório), `--eval-every 500`, `--patience 5`, `--val-max-samples 200`, `--early-stopping-min-delta 0`.
- Melhor checkpoint em `transform_best.pt`; ao fim, `transform.pt` recebe os pesos do melhor step + metadados (`early_stopped`, `best_step`, `best_score`, `stopped_step`).
- Log `val.csv` por run. Script de teste: `scripts/run_early_stopping_test.sh` (LFW, STEPS=500, 3 runs).
- Default off — scripts existentes inalterados.
Resultado (teste curto, STEPS=500): `early_stop__weak-hp` cos=0.753/ssim=0.925; `early_stop__base-hp` cos=0.577/ssim=0.861; `no_early_stop__baseline` cos=0.575/ssim=0.861. Nos runs curtos o teto 500 foi atingido (`early_stopped=False` no ckpt, best_step 400–450). Validação mais forte veio na tarefa 15 (paradas reais ~3600–4300 com teto 10000).

## 2026-09-01 — Região do SSIM configurável (`--ssim-region`)
Contexto: o SSIM da loss era calculado sobre a mesma máscara da transformação (`--mask-regions`/`--mask-shape`), tornando comparações entre experimentos com máscaras parciais injustas — ex.: `mask-eyes` media SSIM só nos olhos, enquanto `mask-full-landmarks` media no rosto todo.
Decisão: nova flag `--ssim-region full-landmarks|mask` (default `full-landmarks` no treino). A loss de SSIM usa máscara independente (`resolve_ssim_mask` em `models/masks.py`): `full-landmarks` = elipse `full` via SCRFD (sempre `ellipse`, independente de `--mask-shape`); `mask` = comportamento antigo. Identity, wavelet, flow e photo continuam restritos à máscara de treino. Metadado `ssim_region` salvo no checkpoint; `state_dict` inalterado.
Resultado:
- **Checkpoints antigos** continuam carregando; na avaliação sem flag, checkpoints sem `ssim_region` usam `mask` (legado) com aviso.
- **Equivalente ao comportamento antigo** (não precisa re-treino por escopo SSIM): `mask-full-landmarks` (máscara de treino = full); aproximadamente `mid_combo_amp20_flow45_nopx` (elipse fixa ≈ full face).
- **SSIM em EXPERIMENTS.md (tarefas 5 e 7) medido com escopo `mask`** — não comparável diretamente a novos runs com default `full-landmarks`. Tarefa 9 re-treina 4 casos representativos com escopo corrigido.
- Validação: treino rápido `--mask-regions eyes` com `full-landmarks` vs `mask` deve mostrar SSIM menor no primeiro (mede área não perturbada fora dos olhos).

## 2026-08-31 — Comparação elipse vs. faixa (7 runs band + elipse reaproveitada)
Contexto: elipses unitárias (tarefa 5) cobriam área pequena; tarefa 6 adicionou `--mask-shape band` (faixas soft-rect proporcionais ao IOD, compostos = união das unitárias). Tarefa 7 treinou 7 configs band com hiperparâmetros idênticos à tarefa 5. Elipse reaproveitada da tarefa 5 (não retreinada). Scripts: `run_mask_band_comparison.sh`, `compare_mask_band_regions.sh`.
Decisão: **faixa melhora desidentificação em todas as regiões/combinações testadas**, especialmente quando duas ou mais regiões são unidas — a maior cobertura expõe mais pixels à perturbação e reduz o sinal de identidade preservado fora da máscara.
Resultado (LFW test, 200 imgs; faixa vs. elipse por região):
| Região | cos band | cos ellipse | ssim band | ssim ellipse | Vencedor trade-off |
|--------|----------|-------------|-----------|--------------|-------------------|
| eyes | 0.728 | 0.872 | 0.861 | 0.866 | **band** (Δcos −0.14, ssim ≈) |
| nose | 0.926 | 0.959 | 0.889 | 0.896 | **band** (Δcos −0.03, ssim ≈) |
| mouth | 0.890 | 0.972 | 0.884 | 0.902 | **band** (Δcos −0.08, ssim −0.02) |
| eyes+mouth | 0.469 | 0.817 | 0.811 | 0.863 | **band** (Δcos −0.35; ssim cai abaixo de 0.85) |
| eyes+nose | 0.660 | 0.809 | 0.848 | 0.855 | **band** (Δcos −0.15, ssim ≈) |
| nose+mouth | 0.796 | 0.911 | 0.878 | 0.891 | **band** (Δcos −0.12, ssim −0.01) |
| eyes+nose+mouth | 0.485 | 0.761 | 0.825 | 0.853 | **band** (Δcos −0.28, ssim −0.03) |

- **Melhor resultado band:** `eyes+mouth` (cos=0.469, euclid=1.012, ssim=0.811) — desidentificação comparável ao baseline fixo (cos=0.456) e próxima de `full-landmarks` (cos=0.411), mas com SSIM abaixo do limiar 0.85. Segundo: `eyes+nose+mouth` (cos=0.485, ssim=0.825).
- **Regiões unitárias isoladas continuam insuficientes** mesmo em band (cos > 0.73 para olhos; > 0.89 para boca/nariz).
- **Observações qualitativas** (previews em `/mnt/study-data/dcarvalho/metrics/mask_band_regions/`): faixas concentram artefatos em retângulos mais largos (olhos = faixa horizontal sobre sobrancelhas/pálpebras; boca/nariz = faixas verticais mais contidas após ajuste da tarefa 6). Compostos band cobrem faixa T superior + nariz + boca sem extrapolar para contorno da face — visualmente a perturbação ocupa mid-face, explicando o ganho de cos. Elipse deixa pele intacta nas bordas das faixas; band preenche os “vãos” entre regiões adjacentes quando combinadas.
- **Conclusão prática:** preferir `--mask-shape band` para combinações (eyes+mouth, eyes+nose+mouth); para cobertura facial completa, `full-landmarks` (ellipse) ainda oferece melhor equilíbrio cos×ssim (cos=0.411, ssim=0.818 vs. cos=0.469, ssim=0.811 de eyes+mouth band).

## 2026-08-31 — Ajuste fino das máscaras band (após revisão visual)
Contexto: validação em `mask_bands_view` — `full` band extrapolava demais; nariz invadia a boca; boca grande; compostos AABB (`eyes-mouth`) ficavam quase mid-face inteiro.
Decisão:
- **Removido** `full` em `--mask-shape band` (erro explícito; `full` permanece só no path ellipse).
- Eyes: faixa assimétrica hw=0.90·IOD, hh_up=0.42, hh_down=0.28 (menos extensão inferior).
- Nose: ancorado em landmarks — topo ≈ mid-olhos+0.18·IOD, base = nariz+0.28·(boca−nariz), hw=0.30·IOD (não chega à boca).
- Mouth: hw=0.55 / hh=0.28 ·IOD (menor).
- Compostos com hífen = **união (`max`) das unitárias band** (não AABB).
- Novo token `eyes-nose-mouth-hybrid`: max(eyes_rect, nose_rect, mouth_ellipse).
Resultado: regenerar figuras no notebook; a validar visualmente antes do item 7.

## 2026-08-31 — Máscaras em faixa (band) + compostos (versão inicial)
Contexto: elipses por landmark (tarefa 3) cobriam área pequena demais; item 6 pede faixas selecionáveis via CLI sem remover elipses.
Decisão (inicial, depois ajustada na entrada acima): `--mask-shape ellipse|band`; path elipse intacto; band = soft-rect L∞ por IOD; `mask_shape` no checkpoint.
Resultado: implementação base + notebook; paddings AABB iniciais rejeitados na revisão visual e substituídos pela união das unitárias.

## 2026-08-30 — Comparação de regiões de máscara (8 runs + baseline)
Contexto: isolar o efeito da região de transformação mantendo hiperparâmetros idênticos a `mid_combo_amp20_flow45_nopx` (DT-CWT, flow 4.5 px, amp 0.20, lambda-id 15, target-cos 0.12, 5000 steps). Baseline **reaproveitado sem retreino** (`--mask-mode fixed`, elipse completa centrada). Runs novos com `--mask-mode landmarks` e `--mask-regions` variando (8 configs). Scripts: `run_mask_comparison.sh`, `compare_mask_regions.sh`.
Decisão: para desidentificação efetiva, a transformação precisa cobrir praticamente toda a face — máscaras parciais preservam sinal de identidade fora da região mascarada e o embedder (FaceNet) continua reconhecendo a pessoa.
Resultado (LFW test, 200 imgs):
- **Melhor trade-off cos × ssim entre máscaras completas:** `full-landmarks` (cos=0.411, euclid=1.068, ssim=0.818) vs baseline fixo `mid_combo_amp20_flow45_nopx` (cos=0.456, euclid=1.026, ssim=0.817). A versão landmark-based cobre a face de forma anatômica e desidentifica ligeiramente melhor que a elipse fixa antiga, com SSIM equivalente — **`full-landmarks` é a contraparte nova comparável ao baseline.**
- **Regiões parciais falham em desidentificar:** olhos (cos=0.872), boca (0.972), nariz (0.959) — cos >> 0.2, apesar de SSIM alto (0.87–0.90). Combinações parciais melhoram pouco (melhor parcial: eyes+mouth+nose, cos=0.761, ssim=0.853).
- **Observações qualitativas** (previews em `/mnt/study-data/dcarvalho/metrics/mask_regions/`): regiões parciais concentram artefatos só na área mascarada (olhos/boca/nariz), deixando pele e contorno intactos — visualmente a face parece quase original fora da elipse. Máscara completa (fixed ou landmarks) distribui a perturbação por toda a região facial; diferença visual entre fixed e full-landmarks é sutil (contorno da elipse landmark segue IOD), mas métricas confirmam vantagem numérica de full-landmarks.
- Nenhum run atingiu cos < 0.2 com ssim > 0.85; objetivo `--target-cos 0.12` permanece distante — região de máscara não era o gargalo principal (baseline full já tinha cos≈0.46).

## 2026-08-28 — Máscaras dinâmicas por landmarks
Contexto: comparar efeito da região de transformação (olhos, nariz, boca, face) sem alterar pesos do modelo.
Decisão: módulo `models/masks.py` com elipses suaves proporcionais à distância interocular (IOD = ||RE−LE||). União por região via `max`. CLI: `--mask-mode fixed|landmarks` (default `fixed`, elipse centrada atual) e `--mask-regions eyes,nose,mouth,full`. Detecção on-the-fly via SCRFD quando `landmarks`; fallback para elipse fixa se detecção falhar.
Fórmula: olho rx=0.32·IOD ry=0.22·IOD; nariz 0.20/0.26; boca centro=(LM+RM)/2 rx=0.38/ry=0.18; full centro=olhos→nariz (40%) rx=0.90/ry=1.20 (escala se landmarks escaparem). Borda: exp(−4·max(val−1, 0)), val=(dx/rx)²+(dy/ry)².
Resultado: treino teste 200 steps com `--mask-mode landmarks --mask-regions eyes` concluído; previews em `runs/mask_test__landmarks-eyes/previews/`. Metadados `mask_mode`/`mask_regions` no checkpoint; `state_dict` inalterado (checkpoints antigos compatíveis).

## 2026-08-28 — Pré-processamento CelebA-HQ (celebahq_pp)
Contexto: dataset original em `/mnt/study-data/dcarvalho/datasets/celebahq` (30 000 imagens) precisava de faces detectadas, alinhadas e centralizadas em 224×224 para uso direto com `FaceImageFolder`.
Decisão: notebook `notebooks/preprocess_celebahq.ipynb` — detecção via `models/face_detector.py` (SCRFD, `det_thresh=0.5`), alinhamento com `insightface.utils.face_align.norm_crop`, saída JPEG quality=95 em `/mnt/study-data/dcarvalho/datasets/celebahq_pp`.
Resultado: 29 995/30 000 imagens salvas (taxa de descarte 0,017%). 5 descartadas por falha de detecção: `img_03543`, `img_10818`, `img_14208`, `img_20701`, `img_23797`. `FaceImageFolder` carrega o dataset sem alterações.

## 2026-08-28 — Detector facial SCRFD 2.5G KPS (buffalo_m)
Contexto: tarefas de pré-processamento CelebA-HQ e máscaras dinâmicas exigem bbox + 5 landmarks (olhos, nariz, cantos da boca).
Decisão: módulo `models/face_detector.py` usando `FaceAnalysis(name='buffalo_m', allowed_modules=['detection'])`. ONNX de detecção: `det_2.5g.onnx` (SCRFD-2.5GF com 5 KPS embutidos no head de detecção). Threshold default `det_thresh=0.5`, `det_size=Auto` (128×128 + 640×640, insightface 1.0.1). Download do pack via `insightface.utils.ensure_available('models', 'buffalo_m')` (release v0.7 GitHub).
Licença: código insightface MIT; modelos pré-treinados — uso não-comercial/pesquisa (README insightface).
Resultado: validado visualmente em 8/8 imagens CelebA-HQ + sanity check `t1.jpg` (notebook `notebooks/validate_face_detector.ipynb`, figuras em `notebooks/outputs/detector_validation/`).



<!--
Exemplo real de entrada de decisão técnica, para referência:

## 2026-08-20 — Por que DCT em vez de FFT complexa
Contexto: precisávamos escolher a transformada espectral base para a
perturbação de frequência.
Decisão: DCT (não FFT complexa) porque é real-valued e mais barato
computacionalmente, alinhado ao objetivo "lightweight" do projeto. DT-CWT
foi adicionado depois como alternativa mais expressiva para comparação
experimental (trade-off custo x qualidade).
Resultado: DCT roda ~Xms mais rápido por batch que DT-CWT no mesmo hardware
(medir e preencher).
-->
