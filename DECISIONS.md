# Decisões do projeto

Ordem cronológica, mais recente no topo. Referencie este arquivo com `@DECISIONS.md` no início de sessões do Cursor que dependam de contexto
histórico. Toda decisão de design não óbvia, resultado de experimento relevante, ou motivo de rejeitar uma abordagem deve virar uma entrada aqui.

Formato sugerido por entrada:
- Data
- Contexto (o problema/dúvida)
- Decisão (o que foi feito)
- Resultado (métricas ou observação, mesmo que "a validar")

---

---

## 2026-08-30 — Comparação de regiões de máscara (8 runs + baseline)
Contexto: isolar o efeito da região de transformação mantendo hiperparâmetros idênticos a `mid_combo_amp20_flow45_nopx` (DT-CWT, flow 4.5 px, amp 0.20, lambda-id 15, target-cos 0.12, 5000 steps). Baseline **reaproveitado sem retreino** (`--mask-mode fixed`, elipse completa centrada). Runs novos com `--mask-mode landmarks` e `--mask-regions` variando (8 configs). Scripts: `run_mask_comparison.sh`, `compare_mask_regions.sh`.
Decisão: para desidentificação efetiva, a transformação precisa cobrir praticamente toda a face — máscaras parciais preservam sinal de identidade fora da região mascarada e o embedder (FaceNet) continua reconhecendo a pessoa.
Resultado (LFW test, 200 imgs):
- **Melhor trade-off cos × ssim entre máscaras completas:** `full-landmarks` (cos=0.423, euclid=1.074, ssim=0.817) vs baseline fixo `mid_combo_amp20_flow45_nopx` (cos=0.456, euclid=1.026, ssim=0.817). A versão landmark-based cobre a face de forma anatômica e desidentifica ligeiramente melhor que a elipse fixa antiga, com SSIM equivalente — **`full-landmarks` é a contraparte nova comparável ao baseline.**
- **Regiões parciais falham em desidentificar:** olhos (cos=0.874), boca (0.971), nariz (0.958) — cos >> 0.2, apesar de SSIM alto (0.87–0.90). Combinações parciais melhoram pouco (melhor parcial: eyes+mouth+nose, cos=0.753, ssim=0.852).
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
