# TODO — Backlog de melhorias

Status: [ ] pendente · [~] em andamento · [x] concluído
Ao concluir um item: marque [x] aqui e mova um resumo do resultado para DECISIONS.md.
Referencie este arquivo com `@TODO.md` ao iniciar uma tarefa no Cursor.

## Alta prioridade

- [x] **W0. Notebook polar debug (freq-only)**
  `notebooks/wavelet_polar_debug.ipynb` + entrada DECISIONS (polar legado ≠ complexo).

- [x] **W1. Polar complexo + CLI/checkpoint**
  `wavelet_polar_mode`, mag/phase amps, level scales, per-channel / spatial / Yl flags.
  Scripts A/B: `scripts/run_wavelet_polar_ab.sh`, `compare_wavelet_polar_ab.sh`.
  **Feito (2026-09-09):** A/B freq-only full-landmarks — complex vence por pouco, mas cos≈0.994 com amp 0.20 (insuficiente). Ver EXPERIMENTS/DECISIONS.

- [x] **W2. Amp-split**
  Grid mag/phase (2026-09-12): melhor = mag0.80/phase0.80 (cos=0.914, ssim=0.961). Ainda cos alto com 36 params.
  Ver EXPERIMENTS / DECISIONS.

- [x] **W3a. Per-channel**
  108 params, mag/phase 0.80: cos/ssim **iguais** ao global (0.914 / 0.961). Sem ganho.

- [x] **W3b. Spatial g=8**
  ~2304 params: cos **0.816**, ssim 0.951 (melhor isolado da linha).

- [x] **W3c. Yl gain**
  cos=0.906, ssim=0.959 — ganho leve vs global; **pior que spatial**. Fase 3 fechada.

- [x] **W4. Spatial + Yl**
  cos **0.793**, ssim 0.947 — melhor da linha (Δcos −0.023 vs spatial). Yl-amp 0.05 satura.

- [x] **W5. Sweep yl-amp / g**
  Melhor: **g=8, yl=0.15** → cos=0.736, ssim=0.930, score≈0.194.
  Linha freq-only: **fechada** (retornos decrescentes vs alvo cos&lt;0.2).

- [ ] **W6 (opcional / nova frente):** combo espectral vencedor + flow/photo
  Partir de `--wavelet-spatial-grid 8 --wavelet-yl-gain --max-wavelet-yl-amp 0.15`
  `--max-wavelet-mag-amp 0.80 --max-wavelet-phase-amp 0.80` **sem** `--disable-flow/--disable-photo`.
  Ou 1 run residual freq-only: g=8 yl=0.20.

- [x] **1. Normalização de imagens via landmarks — dataset celebahq_pp**
  Contexto: as imagens de `celebahq_pp` (tarefa 2 do backlog anterior) já estão centralizadas em 224x224, mas os landmarks não estão necessariamente na mesma posição exata entre imagens diferentes (a centralização foi feita por bbox/crop, não por alinhamento de landmarks). Quero uma normalização adicional: usar os 5 landmarks (via `models/face_detector.py`, tarefa 1 do backlog anterior) para calcular uma transformação de similaridade entre os landmarks detectados em cada imagem e o template oficial InsightFace `arcface_dst` escalado 2× (canvas 224×224; mesmo destino do `norm_crop` usado em `celebahq_pp`):

  ```python
    from insightface.utils.face_align import arcface_dst
    src = (arcface_dst * 2.0).astype(np.float32)
    # ≈ [[76.59, 103.39], [147.06, 103.00], [112.05, 143.47],
    #    [83.10, 184.73], [141.46, 184.41]]  # LE, RE, N, LM, RM
  ```

  Default do notebook: `TEMPLATE_NAME="arcface"`. O template alternativo de pipeline externo (`"visual"`, olhos ~y=68) fica disponível no mesmo notebook para comparação (`plot_compare_templates`) e, se quiser, batch em `celebahq_pp_norm_visual/` — não é o padrão do projeto. Sempre `BORDER_CONSTANT` (borderValue=0).

  Critério de pronto:
    - notebook novo (ex.: `notebooks/normalize_landmarks.ipynb`), reusando `models/face_detector.py` (não duplicar lógica de detecção)
    - função que detecta os 5 landmarks da imagem de entrada e calcula a matriz de transformação de similaridade entre os landmarks detectados e o template `src` acima (ex.: `cv2.estimateAffinePartial2D(landmarks_detectados, src)` ou `skimage.transform.SimilarityTransform`), aplicando `warpAffine`/`warp` para gerar a saída 224x224
    - **primeiro**, rodar em 3 imagens de exemplo de `celebahq_pp` e plotar, na própria célula, um grid comparando antes/depois (com os landmarks originais marcados na imagem "antes" e os landmarks resultantes marcados na imagem "depois", para validar visualmente o alinhamento contra o template `src`)
    - só prosseguir para o dataset completo depois de confirmar visualmente que os 3 exemplos ficaram bem alinhados
    - rodar a normalização em todo o `celebahq_pp`, salvando o resultado em pasta separada (ex.: `/mnt/study-data/dcarvalho/datasets/celebahq_pp_norm`), carregável sem alterações por `data/dataset.py`
    - log de quantas imagens foram processadas com sucesso vs. descartadas (ex.: falha na (re)detecção de landmarks nessa segunda passada)
    - registrar em DECISIONS.md: método de transformação usado, taxa de descarte, tamanho final do dataset
  Arquivos prováveis: `notebooks/normalize_landmarks.ipynb` (novo)
  Depende de: tarefa 1 (detecção facial) e tarefa 2 (celebahq_pp) do backlog anterior

- [X] **2. Repetir normalização por landmarks para o dataset LFW (com crop inicial 256→224)**
  Contexto: mesma normalização da tarefa 1, agora aplicada ao LFW. Diferença importante: as imagens do LFW são nativamente 256x256, e os pontos de destino (`dst`) foram definidos para um canvas 224x224 — portanto é necessário um crop central de 256x256 para 224x224 **antes** de detectar landmarks e normalizar, para não distorcer a escala assumida pelos pontos de destino.

  Critério de pronto:
    - de preferência, adicionar como uma nova seção no mesmo notebook da tarefa 1 (`notebooks/normalize_landmarks.ipynb`), reaproveitando as funções de transformação e plotagem já validadas ali, em vez de duplicar lógica em um notebook separado
    - aplicar center crop 256x256 → 224x224 (sem redimensionamento/resize, só corte) antes da detecção de landmarks
    - mesmo processo de validação da tarefa 1: rodar primeiro em 3 imagens de exemplo do LFW, plotar grid antes/depois, confirmar visualmente o alinhamento
    - só então rodar no dataset completo, salvando em pasta separada (ex.: `/mnt/study-data/dcarvalho/datasets/lfw_norm`), carregável sem alterações por `data/dataset.py`
    - log de imagens processadas vs. descartadas
    - registrar em DECISIONS.md: o crop aplicado, método de transformação, taxa de descarte, tamanho final do dataset
  Arquivos prováveis: `notebooks/normalize_landmarks.ipynb` (mesmo notebook da
    tarefa 1, seção nova)
  Depende de: tarefa 1 (deste backlog)

- [x] **3. Notebook base de testes DT-CWT — setup, landmarks, máscara full e extração de textura**
  `notebooks/dtcwt_modifications_tests.ipynb` com células (a)–(e) incrementais.
  Libs = projeto (`pytorch_wavelets`, FaceNet, SCRFD/cv2) + `skimage` para LBP.
  Datasets: prefer `*_norm`, fallback `celebahq_pp` / `lfw/faces_train` (scripts).
  **Validação:** rodar célula a célula manualmente (agente não executou).
  Depende de: tarefas de detecção facial, máscara full e normalização já concluídas no backlog anterior

- [x] **4. Teste: aleatorização total de fase (phase congruency destruction)**
  Seção (f) em `notebooks/dtcwt_modifications_tests.ipynb`: φ'~U[-π,π] sob máscara full, mag intacta.
  **Feito (2026-09-15):** n=5 LFW — cos médio **0.814**, SSIM_mask **0.673**, SSIM_full **0.857**.
  Imagem danificada visualmente, mas cos longe de &lt;0.5 (e de &lt;0.2). Ver DECISIONS.
  Depende de: tarefa 3

- [x] **5. Teste: inversão local de fase (φ' = φ + π)**
  Seção (g) em `notebooks/dtcwt_modifications_tests.ipynb`: `invert_phase_yh` (φ+π ≡ −c).
  **Feito (2026-09-15):** n=5 — cos **0.739**, SSIM_mask **0.281**, SSIM_full **0.687**.
  Vence rand em cos e em dano SSIM, mas cos ainda ≫ 0.5. Ver DECISIONS.
  Depende de: tarefas 3 e 4

- [x] **6. Filtrar LFW e CelebA-HQ (pose ≤25° + oclusões)**
  Notebook `notebooks/filter_frontal_datasets.ipynb` (QA visual + batch retomável) e espelho CLI `scripts/filter_frontal_datasets.py`.
  Saídas: `celebahq_pp_frontal` (24722/29995) e `lfw_frontal/{faces_train,faces_test}` (8226+2029).
  Pose via InsightFace `1k3d68`; oclusão por heurísticas (olhos escuros / boca lisa). Ver DECISIONS.
  