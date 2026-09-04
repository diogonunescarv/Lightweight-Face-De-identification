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

- [x] **5. Treinar modelos com diferentes regiões de máscara (hiperparâmetros fixos) e comparar**
  Contexto: com --mask-regions implementado (tarefa 3), quero isolar o efeito da região da máscara nos resultados, mantendo todos os demais hiperparâmetros idênticos ao experimento de referência já treinado (mid_combo_amp20_flow45_nopx). Dois scripts novos, seguindo o padrão dos scripts existentes scripts/run_nopxloss.sh e scripts/comparar_nopx.sh:
 
  **scripts/run_mask_comparison.sh** (baseado em run_nopxloss.sh)
    - mesma estrutura (CODE_DIR, DATA_TRAIN, DATA_VAL, OUTPUT_BASE, função run(), STEPS=5000, SEED=42, --lr-scheduler onecycle)
    - hiperparâmetros base fixos para todos os runs (idênticos ao experimento de referência abaixo), variando apenas --mask-regions:
          --transform-type dtcwt --wavelet-J 3 --max-wavelet-amp 0.20 \
          --lambda-wavelet-mag 0.01 --lambda-wavelet-phase 0.01 --lambda-wavelet-smooth 0.001 \
          --lambda-id 15 --target-cos 0.12 --max-dct-amp 0.12 \
          --max-flow-px 4.5 --max-photo-amp 0.15 \
          --tau-ssim 0.88 --lambda-ssim 20.0 --lambda-pixel 6.0 --lr 0.008
 
    - NÃO retreina o baseline "máscara completa (antiga)" — o experimento mid_combo_amp20_flow45_nopx já existe e deve ser reaproveitado (checkpoint em $OUTPUT_BASE/mid_combo_amp20_flow45_nopx/transform.pt); 
    
    o script só precisa treinar os 8 experimentos novos abaixo, um run() por região, com nome mid_combo_amp20_flow45__mask-<regiao>:
        1. full-landmarks   (máscara completa, mas montada a partir dos
           landmarks do SCRFD — não a elipse fixa antiga)
        2. eyes
        3. mouth
        4. nose
        5. eyes+mouth
        6. eyes+nose
        7. mouth+nose
        8. eyes+mouth+nose
      (nomes exatos de --mask-regions a ajustar conforme a nomenclatura final da tarefa 3)

  **scripts/compare_mask_regions.sh** (baseado em comparar_nopx.sh)
    - array EXPERIMENTS com os 9 nomes (baseline + 8 novos)
    - roda `evaluate-single` com --save-images para cada checkpoint, mesmo DATA_TEST/MAX_SAMPLES/MAX_VISUAL do script de referência
    - consolida summary_all.csv (modelo, cos_mean, euclid_mean, ssim_mean, time_ms_mean, n_images) igual ao padrão de comparar_nopx.sh
  Critério de pronto:
    - os dois scripts criados e executáveis, seguindo exatamente os padrões acima (paths configuráveis no topo do arquivo, como nos originais)
    - os 8 runs novos completados (baseline reaproveitado, não retreinado)
    - summary_all.csv gerado com as 9 linhas
    - linhas correspondentes copiadas/importadas para EXPERIMENTS.md
    - registrar em DECISIONS.md um resumo comparativo: qual região teve melhor trade-off cos x ssim, e observações qualitativas (ex.: alguma região gerando artefato visível), citando explicitamente que o baseline "máscara completa" é o mesmo experimento antigo (não retreinado) e que "full-landmarks" é a versão nova comparável a ele

  Arquivos prováveis: scripts/run_mask_comparison.sh (novo), scripts/compare_mask_regions.sh (novo), EXPERIMENTS.md, DECISIONS.md Depende de: tarefa 3 (máscaras) — tarefa 4 é opcional/útil, não bloqueante 
  
  ⚠️ São 8 treinos completos (custo de GPU/tempo real). Não deixe o agente disparar os runs automaticamente — peça os scripts prontos, revise os comandos, e execute/confirme cada run manualmente.

- [x] **6. Implementar máscaras em faixa (banda) por região e validar em notebook**
  Contexto: as máscaras atuais (tarefa 3, elipses ao redor dos landmarks) parecem cobrir uma área pequena demais — suspeita de que isso está limitando a eficácia da desidentificação. Quero uma nova forma de máscara em "faixa" (banda/retângulo largo ao redor da região, não só um círculo justo no ponto), cobrindo mais área ao redor de cada região e suas combinações. As máscaras elípticas atuais (tarefa 3) NÃO devem ser removidas nem alteradas — a nova forma é adicionada ao lado da existente, selecionável via CLI. Regiões/combinações (mesmas da tarefa 3, agora em formato faixa): eyes, nose, mouth, eyes+nose, eyes+mouth, nose+mouth, eyes+nose+mouth.
  
  Critério de pronto:
    - models/masks.py estendido com uma nova função de máscara em faixa por região (ex.: retângulo/banda ao redor do(s) landmark(s), com largura/ altura configuráveis, maior cobertura que a elipse atual) — função(ões) elípticas existentes permanecem intactas e utilizáveis
    - nova flag de CLI (ex.: --mask-shape ellipse|band, default "ellipse" para não quebrar experimentos existentes; --mask-regions continua valendo para escolher a(s) região(ões) dentro da forma escolhida) 
    - notebook novo (ex.: notebooks/mask_bands_validation.ipynb), reusando models/masks.py e models/face_detector.py, gerando lado a lado: original | máscara elipse (atual) | máscara faixa (nova), para cada região/combinação listada — usado só para validar visualmente se as novas máscaras fazem sentido antes de qualquer treino
    - só avançar para o item 7 depois de confirmar visualmente no notebook que as faixas cobrem a área esperada sem extrapolar demais para fora do rosto
    - registrar em DECISIONS.md: como a faixa é definida (dimensões, proporção em relação à distância interocular ou ao bbox), e a confirmação visual do notebook
  Arquivos prováveis: models/masks.py (estendido, não substituído), deid_optimize.py (nova flag), notebooks/mask_bands_validation.ipynb (novo)
  Depende de: tarefa 3

- [x] **7. Treinar modelos com máscaras em faixa e comparar (hiperparâmetros fixos)**
  Contexto: com --mask-shape band validado visualmente (tarefa 6), repetir o mesmo processo de comparação da tarefa 5, agora para a forma faixa, para verificar se a cobertura maior melhora a desidentificação (cos/euclid) sem destruir a qualidade (ssim). Mesmos hiperparâmetros base da tarefa 5, variando --mask-shape band e --mask-regions pelas 7 regiões/combinações. Dois scripts novos, seguindo o mesmo padrão de scripts/run_mask_comparison.sh e scripts/compare_mask_regions.sh (tarefa 5):

  **scripts/run_mask_band_comparison.sh**
    - mesma estrutura/hiperparâmetros base da tarefa 5, adicionando --mask-shape band em cada run
    - um run() por região/combinação: eyes, nose, mouth, eyes+mouth, eyes+nose, nose+mouth, eyes+nose+mouth (7 runs — sem "full-landmarks" aqui, já coberto na tarefa 5)
    - nomes tipo mid_combo_amp20_flow45__maskband-<regiao>
  **scripts/compare_mask_band_regions.sh**
    - mesmo padrão de scripts/compare_mask_regions.sh, array EXPERIMENTS com os 7 novos + os já existentes da tarefa 5 (para comparar faixa vs. elipse lado a lado na mesma tabela)
    - gera summary_all.csv consolidado (elipse + faixa juntos)

  Critério de pronto:
    - os dois scripts criados, seguindo exatamente os padrões da tarefa 5
    - os 7 runs completados
    - summary_all.csv com elipse + faixa juntos
    - linhas novas adicionadas em EXPERIMENTS.md (com coluna indicando a forma: ellipse/band)
    - eu rodo os treinos e o script de comparação manualmente, e envio o resultado (summary_all.csv) para você
    - ao receber o resultado: atualizar EXPERIMENTS.md com as linhas novas e adicionar entrada em DECISIONS.md comparando faixa vs. elipse por região (qual forma teve melhor trade-off cos x ssim), e marcar este item como [x] no TODO.md

  Arquivos prováveis: scripts/run_mask_band_comparison.sh (novo), scripts/compare_mask_band_regions.sh (novo), EXPERIMENTS.md, DECISIONS.md
  Depende de: tarefa 6  
  
  ⚠️ São 7 treinos completos (custo de GPU/tempo real). Não deixe o agente disparar os runs automaticamente — peça os scripts prontos, revise os comandos, e execute/confirme cada run manualmente.

- [x] **8. Tornar a região do SSIM configurável (full-landmarks vs. máscara atual)**
  Contexto: hoje o SSIM da loss é calculado sobre a região da máscara ativa no momento (--mask-regions/--mask-shape), o que torna as comparações entre experimentos com máscaras diferentes injustas — um experimento com máscara pequena (ex.: só olhos) tem o SSIM medido só nos olhos, enquanto um com máscara cheia é medido no rosto todo. Preciso de dois comportamentos selecionáveis por flag, decididos ANTES do treino:
    1. Padrão: SSIM calculado sobre a região da máscara "full" montada a partir dos landmarks (independente de qual máscara está sendo treinada/usada na transformação)
    2. Opcional: SSIM calculado sobre a mesma região da máscara em uso naquele treino (comportamento atual)
  Critério de pronto:
    - nova flag de CLI (ex.: --ssim-region full-landmarks|mask, default "full-landmarks" — ATENÇÃO: isso muda o comportamento padrão em relação ao que já foi treinado até aqui; ver nota abaixo)
    - losses/losses.py e training/trainer.py ajustados para computar o SSIM sobre a região selecionada, desacoplado da região usada para a transformação/demais losses (identity, wavelet, etc., que continuam restritas à máscara escolhida em --mask-regions/--mask-shape)
    - teste rápido: treinar poucos steps com --mask-regions eyes --ssim-region full-landmarks e confirmar (via log/inspeção) que o SSIM é computado sobre o rosto todo; repetir com --ssim-region mask e confirmar que fica restrito aos olhos (comportamento antigo)
    - registrar em DECISIONS.md: motivação da mudança, novo default, e o caveat explícito de que TODOS os resultados em EXPERIMENTS.md gerados antes desta mudança (tarefas 5 e 7) usaram SSIM restrito à máscara — não são diretamente comparáveis aos novos runs com --ssim-region full-landmarks (default).
  Arquivos prováveis: losses/losses.py, training/trainer.py, deid_optimize.py (nova flag)
  Depende de: tarefa 3

- [x] **9. Re-treinar um subconjunto de máscaras com --ssim-region full-landmarks (SSIM corrigido)**
  Contexto: com --ssim-region implementado (tarefa 8), quero re-treinar um subconjunto representativo usando o escopo correto de SSIM (rosto todo, via landmarks) — não é uma comparação entre os dois escopos, é a correção do comportamento anterior (SSIM restrito à máscara) nesses casos:
    1. mid_combo_amp20_flow45__mask-full-landmarks
    2. mid_combo_amp20_flow45__mask-eyes+mouth+nose
    3. mid_combo_amp20_flow45__maskband-eyes
    4. mid_combo_amp20_flow45__maskband-eyes+mouth
    5. mid_combo_amp20_flow45__maskband-eyes+nose+mouth 
  Todos os runs usam --ssim-region full-landmarks. Nenhum run deve usar --ssim-region mask (essa variante não deve ser treinada nesta tarefa).
  Otimização de custo:
    - Caso 1 (full-landmarks): quando a máscara já é o rosto todo, --ssim-region full-landmarks é equivalente ao comportamento antigo — NÃO retreinar; reaproveitar o checkpoint já existente da tarefa 5.
    - Casos 2–5: retreinar, pois o comportamento anterior (SSIM restrito à máscara) é diferente do novo default. 4 runs novos no total.
  Dois scripts novos, seguindo o mesmo padrão de scripts/run_mask_comparison.sh e scripts/compare_mask_regions.sh (tarefa 5):
  **scripts/run_ssim_fix_retrain.sh**
    - treina apenas os 4 runs novos, cada um com os mesmos hiperparâmetros base + --mask-shape/--mask-regions do caso original correspondente + --ssim-region full-landmarks:
        mid_combo_amp20_flow45__mask-eyes+mouth+nose__ssim-fixed
        mid_combo_amp20_flow45__maskband-eyes__ssim-fixed
        mid_combo_amp20_flow45__maskband-eyes+mouth__ssim-fixed
        mid_combo_amp20_flow45__maskband-eyes+nose+mouth__ssim-fixed
  **scripts/compare_ssim_fix.sh**
    - array EXPERIMENTS com 5 entradas: o baseline full-landmarks (reaproveitado, caso 1) + os 4 runs novos (casos 2–5)
    - gera summary_all.csv consolidado, com coluna indicando região e forma (ellipse/band)
  Critério de pronto:
    - os dois scripts criados, seguindo exatamente os padrões da tarefa 5
    - os 4 runs novos completados
    - summary_all.csv com as 5 entradas
    - linhas dos casos 2–5 em EXPERIMENTS.md atualizadas/substituídas pelos novos resultados (deixando claro que superam os valores antigos medidos com SSIM restrito à máscara — não são um ponto de comparação adicional, são a correção)
    - eu rodo os treinos e o script de comparação manualmente, e envio o resultado (summary_all.csv) para você
    - ao receber o resultado: atualizar EXPERIMENTS.md e adicionar entrada em DECISIONS.md explicando que esses 4 resultados substituem os equivalentes das tarefas 5/7 (SSIM agora medido sobre o rosto todo via landmarks, não mais restrito à máscara), e marcar este item como [x] no TODO.md
  Arquivos prováveis: scripts/run_ssim_fix_retrain.sh (novo),
  scripts/compare_ssim_fix.sh (novo), EXPERIMENTS.md, DECISIONS.md
  Depende de: tarefa 8
  ⚠️ São 4 treinos completos (custo de GPU/tempo real). Não deixe o agente disparar os runs automaticamente — peça os scripts prontos, revise os comandos, e execute/confirme cada run manualmente.

- [x] **10. Notebook de comparação final: imagens com máscaras + resultados (elipse vs. faixa)**
  **Contexto:** após a conclusão dos treinos das tarefas 5, 7 e 9, preciso de um notebook único que centralize a análise visual e quantitativa, facilitando a extração de conclusões para a dissertação e para apresentações. O notebook deve ser auto‑contido, reutilizar os módulos existentes e gerar material de alta qualidade (figuras, tabelas, métricas).
  **Requisitos adicionais (além do já especificado no backlog):**
  - Para cada região/combinação, exibir **três colunas** lado a lado:
    1. **Imagem original** (com anotação dos landmarks para referência).
    2. **Máscara utilizada** (elipse ou faixa, sobreposta à imagem original) – para que se veja claramente a área que está sendo modificada.
    3. **Imagem transformada** (resultado da desidentificação) com a respectiva máscara aplicada.
  - Abaixo de cada trio, incluir um **painel de métricas** (cosine, euclidean, SSIM) **e o tempo de inferência** (em ms) daquele exemplo, para que se possa correlacionar visualmente qualidade e desempenho.
  - Incluir **visualizações agregadas** (gráficos de barras/radar) comparando as formas (elipse vs. faixa) para cada região, destacando os valores corrigidos da tarefa 9 (SSIM sobre a máscara `full-landmarks`) e, quando não disponíveis, os valores originais das tarefas 5/7.
  - O notebook deve ser **reprodutível** e bem comentado, com células que expliquem cada etapa (carregamento dos checkpoints, aplicação da transformação, cálculo de métricas e geração das figuras).
  - **Não salvar** figuras em `notebooks/outputs/`; utilizar os mesmos padrões de caminho dos scripts anteriores (ex.: `/mnt/study-data/dcarvalho/tests/final_comparison/` ou `/mnt/study-data/dcarvalho/metrics/final_report/`). O caminho exato deve ser verificado olhando os `OUTPUT_BASE` usados nas tarefas 5, 7 e 9.

  **Critério de pronto:**
  - Notebook criado (`notebooks/mask_shape_comparison_report.ipynb`), reusando `models/masks.py` e `models/face_detector.py` (não duplicar lógica).
  - Gera grids de imagens por região, com as três colunas (original, máscara, transformada) para **ambas as formas** (elipse e faixa), sempre que houver checkpoint disponível.
  - Lê os `summary_all.csv` gerados pelas tarefas 5, 7 e 9 (ou diretamente `EXPERIMENTS.md`) e monta uma **tabela única** consolidada com:
    - Região
    - Forma (elipse/faixa)
    - Cos médio, Euclid médio, SSIM médio
    - Tempo de inferência médio (ms)
    - Número de imagens avaliadas
  - Gera **gráficos comparativos** (barras agrupadas) para cada métrica, separando os valores corrigidos (tarefa 9) dos demais, com legenda clara.
  - Gera **exemplos individuais** (pelo menos 2 imagens por região) que serão salvos como PNG em alta resolução para uso em slides, com anotações das métricas.
  - O notebook **não salva** arquivos dentro de `notebooks/outputs/`; segue o padrão já utilizado (ex.: `/mnt/study-data/dcarvalho/tests/final_comparison/`). O caminho é definido no início do notebook, com uma célula de configuração que permite ajuste fácil.
  - Ao final, o notebook deve **imprimir um resumo** das principais conclusões (ex.: "A faixa supera a elipse em SSIM para regiões com olhos, mas com leve piora no cos") para facilitar a redação da dissertação.

  **Arquivos prováveis:** `notebooks/mask_shape_comparison_report.ipynb` (novo)
  **Depende de:** tarefas 5, 7 e 9 (checkpoints e summaries gerados)


- [ ] **12. Avaliar upgrade para landmarks densos (106 pts) se máscaras elípticas se mostrarem insuficientes**
  Contexto: só abrir esta se a avaliação visual da tarefa 3/4 mostrar que elipses grosseiras prejudicam a qualidade da desidentificação ou a credibilidade da figura na dissertação.

- [x] **13. Implementar early stopping baseado em métricas de validação**
  Contexto: Atualmente, os treinos rodam por um número fixo de STEPS (ex.: 5000) independentemente de o modelo já ter convergido ou começado a piorar (overfitting na identidade ou degradação do SSIM). Isso desperdiça tempo de GPU em treinos que já estabilizaram, e não protege contra treinos que pioram nas últimas etapas. Quero adicionar early stopping monitorando uma métrica de validação calculada periodicamente durante o treino, parando automaticamente se não houver melhora após N avaliações consecutivas.

  Critério de pronto:
    - nova função/lógica em training/trainer.py que, a cada --eval-every steps, roda uma validação leve (subconjunto de DATA_VAL) e computa a métrica de referência (ex.: combinação de cos e ssim, ou a mesma métrica já usada em compare_mask_regions.sh — ajustar conforme o que já existe)
    - nova(s) flag(s) de CLI: --early-stopping (bool, default False para não quebrar scripts existentes), --patience (int, nº de avaliações sem melhora antes de parar), --eval-every (int, intervalo de steps entre validações)
    - ao acionar o early stopping, salvar o checkpoint correspondente à melhor métrica observada (não necessariamente o último step), e logar claramente em qual step o treino parou e por quê
    - teste rápido: rodar um treino curto com --early-stopping --patience 2 e confirmar que ele para antes do STEPS total quando a métrica não melhora (pode-se forçar isso com hiperparâmetros ruins de propósito para o teste)
    - registrar em DECISIONS.md: métrica escolhida como critério de parada e por quê, valores default de --patience e --eval-every
    - Comportamento atual, deve ser mantido como possibilidade para futuros treinamentos.
  Arquivos prováveis: training/trainer.py, deid_optimize.py (novas flags)
  Depende de: nada

- [x] **14. Otimização automática de hiperparâmetros**
  Contexto: Atualmente, os hiperparâmetros (--lr, --lambda-id, --max-flow-px, --tau-ssim, --lambda-ssim, etc.) são ajustados manualmente entre experimentos, como visto nos scripts run_mask_comparison.sh e run_mask_band_comparison.sh, onde o conjunto base foi fixado "na mão" a partir do experimento de referência mid_combo_amp20_flow45_nopx. Quero automatizar essa busca para encontrar combinações melhores de hiperparâmetros antes de fixá-los como base para as próximas rodadas de comparação de máscaras.

  Critério de pronto:
    - script novo (ex.: scripts/run_hparam_search.py) que varre um espaço de busca definido (ex.: --lr, --lambda-id, --max-flow-px, --tau-ssim, --lambda-ssim) usando [a definir: grid search simples / Optuna / Ray Tune — sugestão: Optuna, por ser leve e não exigir infraestrutura extra]
    - cada trial roda um treino curto (menos steps que um treino completo, ex.: --steps 1000) para viabilizar a busca em tempo razoável, usando a mesma métrica de validação da tarefa 13 (reaproveitar a lógica de eval, não duplicar)
    - resultados de todos os trials salvos em CSV (ex.: hparam_search_results.csv), com colunas para cada hiperparâmetro testado + métrica final
    - ao final, o script imprime a melhor combinação encontrada e como usá-la (ex.: linha de comando pronta para copiar em um treino completo)
    - registrar em DECISIONS.md: espaço de busca usado, ferramenta escolhida, melhor combinação encontrada e se ela substitui o conjunto base atual (mid_combo_amp20_flow45) nos próximos experimentos
  Arquivos prováveis: scripts/run_hparam_search.py (novo), training/trainer.py
    (se precisar expor a métrica de validação de forma programática), requirements.txt
    (nova dependência, ex.: optuna)
  Depende de: tarefa 13 (reaproveita a métrica/lógica de validação)

  ⚠️ Cada trial é um treino real (ainda que curto) — mesmo cuidado dos itens 5, 7 e 9: não deixe o agente disparar a busca completa automaticamente. Peça o script pronto, revise o espaço de busca e o número de trials, e execute manualmente.

- [x] **15. Repetir treinos da tarefa 9 (SSIM corrigido) com early stopping (teto de 10000 steps)**
  Contexto: na tarefa 9, os 4 experimentos com `--ssim-region full-landmarks` foram
  treinados com `--steps 5000` fixos, sem early stopping (tarefa 13 ainda não existia).
  Agora que early stopping está implementado, quero repetir esses mesmos 4 experimentos
  com `--steps 10000` como teto e `--early-stopping` habilitado, para verificar se a
  mesma qualidade (cos/euclid/ssim) é atingida com uma fração do custo de GPU.

  Critério de pronto:
    - script novo `scripts/run_ssim_fix_earlystop.sh` (baseado em
      `run_ssim_fix_retrain.sh`), retreinando os mesmos 4 casos da tarefa 9:
        - `mid_combo_amp20_flow45__mask-eyes+mouth+nose__ssim-fixed`
        - `mid_combo_amp20_flow45__maskband-eyes__ssim-fixed`
        - `mid_combo_amp20_flow45__maskband-eyes+mouth__ssim-fixed`
        - `mid_combo_amp20_flow45__maskband-eyes+nose+mouth__ssim-fixed`
      mesmos hiperparâmetros base + `--ssim-region full-landmarks`, mas agora com
      `--steps 10000 --early-stopping --eval-every 100 --patience 3 --val-max-samples 200
      --early-stopping-metric score` (mesmos valores usados no treino de referência
      da busca de hiperparâmetros)
    - nomes dos novos experimentos com sufixo `__earlystop1000`, para não sobrescrever
      os checkpoints da tarefa 9 (ex.: `..._ssim-fixed__earlystop1000`)
    - `scripts/compare_ssim_fix_earlystop.sh` (baseado em `compare_ssim_fix.sh`),
      juntando os 4 novos + os 4 originais da tarefa 9 no mesmo `summary_all.csv`,
      com colunas extras: steps efetivos rodados antes da parada e tempo total de treino
    - registrar em DECISIONS.md: quantos steps cada run usou antes de parar, se
      cos/euclid/ssim ficaram equivalentes aos da tarefa 9 apesar do teto de 1000 steps,
      e se early stopping deve virar padrão nos próximos treinos de comparação de máscara
  Arquivos prováveis: scripts/run_ssim_fix_earlystop.sh (novo),
    scripts/compare_ssim_fix_earlystop.sh (novo), EXPERIMENTS.md, DECISIONS.md
  Depende de: tarefa 9 (experimentos de referência), tarefa 13 (early stopping)

  ⚠️ São 4 treinos completos (ainda que possam parar antes do teto). Não deixe o
  agente disparar os runs automaticamente — peça os scripts prontos, revise os
  comandos, e execute/confirme cada run manualmente.

- [X] **16. Treinar modelos com os hiperparâmetros ótimos da busca (tarefa 14) e comparar com a base atual**
  Contexto: a busca de hiperparâmetros (tarefa 14) retornou os seguintes top trials
  (ordenados por score):
    - trial 26 (score=0.4345): lr=0.0144483, lambda_id=22.78, max_flow_px=5.18, tau_ssim=0.922, lambda_ssim=21.2
    - trial 25 (score=0.4322): lr=0.0104433, lambda_id=22.41, max_flow_px=5.23, tau_ssim=0.922, lambda_ssim=21.3
    - trial 28 (score=0.4288): lr=0.0146674, lambda_id=24.96, max_flow_px=4.58, tau_ssim=0.920, lambda_ssim=29.5
  Os demais flags (wavelet, target-cos, max-dct-amp, max-photo-amp, lambda-pixel etc.)
  não fizeram parte do espaço de busca e devem seguir o comando de treino completo
  sugerido pela tarefa 14. Quero treinar os top 3 trials por completo e compará-los
  entre si e contra a base fixa atual (`mid_combo_amp20_flow45`: lr=0.008,
  lambda_id=15, max_flow_px=4.5, tau_ssim=0.88, lambda_ssim=20.0) para decidir se os
  hiperparâmetros otimizados devem substituir a base usada em experimentos futuros
  de máscara (sucessores das tarefas 5/7/9).

  Critério de pronto:
    - script novo `scripts/run_best_hparams.sh`, treinando os 3 trials acima com os
      flags fixos do comando sugerido pela tarefa 14 (`--transform-type dtcwt
      --wavelet-J 3 --max-wavelet-amp 0.20 --lambda-wavelet-mag 0.01
      --lambda-wavelet-phase 0.01 --lambda-wavelet-smooth 0.001 --target-cos 0.12
      --max-dct-amp 0.12 --max-photo-amp 0.15 --lambda-pixel 6.0 --early-stopping
      --eval-every 100 --patience 3 --val-max-samples 200
      --early-stopping-metric score`), variando apenas lr/lambda-id/max-flow-px/
      tau-ssim/lambda-ssim conforme cada trial
    - nomes dos experimentos: `hparam_search__trial26`, `hparam_search__trial25`,
      `hparam_search__trial28`
    - `scripts/compare_best_hparams.sh`: roda `evaluate-single` nos 3 novos +
      no baseline atual `mid_combo_amp20_flow45_nopx` (reaproveitado, não retreinar),
      gerando `summary_all.csv` com as 4 linhas
    - registrar em DECISIONS.md: qual trial teve melhor trade-off cos x ssim, se
      algum supera a base fixa atual, e — em caso positivo — a partir de qual tarefa
      do backlog a nova base passa a valer (deixando claro que tarefas 5/7/9 já
      concluídas usaram a base antiga e não são retroativamente invalidadas)
  Arquivos prováveis: scripts/run_best_hparams.sh (novo),
    scripts/compare_best_hparams.sh (novo), EXPERIMENTS.md, DECISIONS.md
  Depende de: tarefa 14 (busca de hiperparâmetros)

  ⚠️ São 3 treinos completos reais (early stopping pode parar antes, mas ainda
  representa custo de GPU). Não deixe o agente disparar os runs automaticamente —
  peça os scripts prontos, revise os comandos, e execute/confirme cada run manualmente.


