# TODO — Backlog de melhorias

Status: [ ] pendente · [~] em andamento · [x] concluído
Ao concluir um item: marque [x] aqui e mova um resumo do resultado para DECISIONS.md.
Referencie este arquivo com `@TODO.md` ao iniciar uma tarefa no Cursor.

## Alta prioridade

- [x] **1. Integrar SCRFD 2.5G KPS (via insightface) para detecção facial**
  Contexto: precisamos de bbox + landmarks (olhos, nariz, cantos da boca) para viabilizar o pré-processamento do CelebA-HQ (tarefa 2) e a futura criação de máscaras dinâmicas por região (tarefa 3).
  Critério de pronto:
    - módulo novo (ex.: models/face_detector.py) com função tipo detect_face(image) -> (bbox, landmarks_5pts, score)
    - usa insightface.app.FaceAnalysis com modelo SCRFD 2.5G KPS
    - testado em 5+ imagens de teste do repo, landmarks plotados validados visualmente
    - dependência insightface (e onnxruntime) documentada em requirements.txt
    - registrar em DECISIONS.md: bundle/modelo exato usado, versão, licença
  Arquivos prováveis: models/face_detector.py (novo), requirements.txt
  Depende de: nada

- [x] **2. Notebook de pré-processamento do CelebA-HQ (novo dataset de faces centralizadas)**
  Contexto: já tenho o CelebA-HQ baixado (está na rota: "/mnt/study-data/dcarvalho/datasets/celebahq"). Quero usar o SCRFD (tarefa 1) para detectar e alinhar cada face, gerando um novo dataset já centralizado no padrão esperado por data/dataset.py (ex. 224x224), via notebook — seguindo o padrão/estrutura já usado no notebook existente do CelebA em notebooks/.
  Critério de pronto:
    - notebook novo (ex.: notebooks/preprocess_celebahq.ipynb), reusando models/face_detector.py (não duplica lógica de detecção)
    - percorre o CelebA-HQ local, descarta imagens sem detecção confiável (log de quantas foram processadas vs. descartadas)
    - salva o novo dataset em pasta separada ("/mnt/study-data/dcarvalho/datasets/celebahq_pp"), carregável sem alterações por data/dataset.py
    - registrar em DECISIONS.md: taxa de descarte, tamanho final do dataset
  Arquivos prováveis: notebooks/preprocess_celebahq.ipynb (novo)
  Depende de: tarefa 1

- [x] **3. Máscara dinâmica por landmarks, restringindo a transformação à região de interesse**
  Contexto: hoje existe --no-face-mask (máscara elíptica fixa, tudo ou nada). Por isso, preciso de máscaras construídas a partir dos 5 landmarks do SCRFD (face por completo, elipses ao redor de olhos/nariz/boca, raio proporcional à distância interocular, etc), configuráveis via CLI, para testes comparativos futuros (ex.: só olhos, olhos + nariz, face por completa) sem mexer em código.
  Nota de precisão: 5 landmarks dão máscaras heurísticas (elipses), não contornos anatômicos exatos — suficiente para o objetivo atual; landmarks
  densos (106 pts) ficam como possível melhoria futura (ver item exploratório).
  Critério de pronto:
    - módulo novo (ex.: models/masks.py) com função que recebe landmarks e retorna máscara para um conjunto de regiões
    - nova flag de CLI (ex.: --mask-regions eyes,nose,mouth; default "full" mantém comportamento atual, sem quebrar experimentos existentes)
    - integrado em LightweightDeIdentifier / training/trainer.py sem quebrar checkpoints antigos quando --mask-regions não é usado
    - teste rápido: treinar poucos steps com --mask-regions eyes e conferir no preview que só a região dos olhos muda. Conferir códigos presentes na pasta script para entender rotas e criar uma nova pasta dentro da pasta em que os testes são salvos por padrão.
    - registrar em DECISIONS.md: como a máscara é construída (fórmula do raio, elipse vs. círculo, etc.)
  Arquivos prováveis: models/masks.py (novo), models/transformer.py, training/trainer.py, deid_optimize.py (nova flag)
  Depende de: tarefa 1

## Baixa prioridade / exploratório

- [x] **4. Notebook de visualização de máscaras por região (para relatórios)**
  Contexto: gerar figuras mostrando as máscaras em regiões específicas (olhos, olhos+nariz, etc.) lado a lado, para apresentações/relatórios.
  Critério de pronto:
    - notebooks/mask_regions_showcase.ipynb reusa models/masks.py e models/face_detector.py (não duplica lógica)
    - gera grid: original | máscara região A | máscara região B ...
    - salva figuras em `/mnt/study-data/dcarvalho/tests/new_masks_view` para reuso na dissertação
  Arquivos prováveis: notebooks/mask_regions_showcase.ipynb (novo)
  Depende de: tarefa 3

- [ ] **5. Treinar modelos com diferentes regiões de máscara (hiperparâmetros fixos) e comparar**
  Contexto: com --mask-regions implementado (tarefa 3), quero isolar o efeito da região da máscara nos resultados, mantendo todos os demais  hiperparâmetros idênticos. Configuração de referência (fixa para todos os runs, variando apenas --mask-regions):
      --transform-type dtcwt --wavelet-J 3 --max-wavelet-amp 0.20 \
      --lambda-wavelet-mag 0.01 --lambda-wavelet-phase 0.01 --lambda-wavelet-smooth 0.001 \
      --lambda-id 15 --target-cos 0.12 --max-dct-amp 0.12 \
      --max-flow-px 4.5 --max-photo-amp 0.15 \
      --tau-ssim 0.88 --lambda-ssim 20.0 --lambda-pixel 6.0 --lr 0.008
 
  Regiões a comparar: full (baseline atual), full (feita a partir dos landmarks) eyes, eyes+nose, nose+mouth, eyes+nose+mouth (ajustar à nomenclatura final da tarefa 3).

  Critério de pronto:
    - script/wrapper que dispara um run por região usando exatamente os hiperparâmetros acima, variando só --mask-regions (ex.: estender o padrão de "run <nome>" já usado, com nome tipo mid_combo_amp20_flow45__mask-<regiao>)
    - todos os runs completados até o mesmo número de --steps
    - `evaluate` (avaliação resumida) rodado em cada checkpoint resultante, sobre o mesmo conjunto de validação
    - uma linha nova por run em EXPERIMENTS.md: hiperparâmetros fixos + coluna de região + cos/euclid/ssim
    - comparação visual lado a lado (mesmas imagens de entrada, uma coluna por região) — pode reusar/estender o notebook da tarefa 4
    - registrar em DECISIONS.md um resumo comparativo: qual região teve melhor trade-off cos x ssim, e observações qualitativas (ex.: alguma região gerando artefato visível)
  Arquivos prováveis: script de execução dos runs (novo ou existente
  estendido), EXPERIMENTS.md, DECISIONS.md
  Depende de: tarefa 3 (máscaras) — tarefa 4 é opcional/útil, não bloqueante

- [ ] **6. Avaliar upgrade para landmarks densos (106 pts) se máscaras elípticas se mostrarem insuficientes**
  Contexto: só abrir esta se a avaliação visual da tarefa 3/4 mostrar que elipses grosseiras prejudicam a qualidade da desidentificação ou a credibilidade da figura na dissertação.