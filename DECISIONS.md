# Decisões do projeto

Ordem cronológica, mais recente no topo. Referencie este arquivo com `@DECISIONS.md` no início de sessões do Cursor que dependam de contexto
histórico. Toda decisão de design não óbvia, resultado de experimento relevante, ou motivo de rejeitar uma abordagem deve virar uma entrada aqui.

Formato sugerido por entrada:
- Data
- Contexto (o problema/dúvida)
- Decisão (o que foi feito)
- Resultado (métricas ou observação, mesmo que "a validar")

---

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
