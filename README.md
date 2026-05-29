# Lightweight Face De-identification

Este projeto implementa uma transformação facial leve para desidentificação visual, preservando a qualidade da imagem. Durante o treinamento, o modelo ajusta uma transformação paramétrica (DCT + fluxo óptico + ajuste fotométrico) para reduzir a similaridade entre embeddings faciais, mantendo a aparência natural.

## Sumário

- [Requisitos](#requisitos)
- [Instalação](#instalação)
- [Estrutura do código](#estrutura-do-código)
- [Preparação dos dados](#preparação-dos-dados)
- [Treinamento](#treinamento)
  - [Flags principais](#flags-principais)
  - [Flags de regularização e qualidade](#flags-de-regularização-e-qualidade)
  - [Flags da transformação](#flags-da-transformação)
  - [Flags do agendador de taxa de aprendizado (scheduler)](#flags-do-agendador-de-taxa-de-aprendizado-scheduler)
  - [Exemplo completo de treinamento](#exemplo-completo-de-treinamento)
- [Monitoramento do treinamento](#monitoramento-do-treinamento)
- [Aplicação do modelo (de-identificação)](#aplicação-do-modelo-de-identificação)
- [Interpretando os resultados](#interpretando-os-resultados)
- [Dicas para ajuste fino](#dicas-para-ajuste-fino)

---

## Requisitos

- Python 3.8+
- PyTorch 1.9+
- torchvision
- facenet-pytorch (para o embedder facial)
- numpy, Pillow, tqdm (opcional)

Instale as dependências com:

```bash
pip install torch torchvision facenet-pytorch numpy Pillow
```

## Instalação

Clone o repositório e coloque os arquivos `deid_optimize.py` e `face_embedder.py` no mesmo diretório.

Certifique-se de que o arquivo `face_embedder.py` contenha a função `build_face_embedders(device)`, que retorna uma lista de modelos de embedding facial (ex.: FaceNet). Um exemplo é fornecido no repositório.

## Estrutura do código

- `deid_optimize.py`: contém a classe do transformador (`LightweightDeIdentifier`), o loop de treinamento e a função de aplicação.
- `face_embedder.py`: responsável por carregar os modelos de extração de embeddings (ex.: FaceNet). **Você deve implementar ou ajustar este arquivo conforme seus modelos autorizados.**

## Preparação dos dados

Crie uma pasta com imagens de faces **alinhadas e centralizadas** (ex.: cortadas para 224x224). Não é necessário ter labels – o treinamento é auto-supervisionado usando apenas as imagens originais.

Estrutura esperada:

```
faces_train/
   img001.jpg
   img002.png
   ...
```

O dataset carrega imagens em RGB e as redimensiona para `--image-size` (padrão 224).

## Treinamento

O comando principal é:

```bash
python deid_optimize.py train --data /caminho/para/faces --out /pasta/de/saida [opções]
```

Todos os argumentos estão descritos abaixo.

### Flags principais

- `--data`: obrigatório. Pasta com as imagens de treino.
- `--out`: obrigatório. Pasta onde serão salvos os checkpoints, logs e previews.
- `--device`: padrão `cpu`. Pode ser `cpu` ou `cuda` (se disponível).
- `--image-size`: padrão `224`. Tamanho das imagens (largura=altura).
- `--batch-size`: padrão `8`. Tamanho do batch.
- `--num-workers`: padrão `0`. Número de subprocessos para carregar dados.
- `--steps`: padrão `2000`. Número total de iterações (passos) de treinamento.
- `--lr`: padrão `0.001`. Taxa de aprendizado inicial (para Adam).
- `--lr-scheduler`: padrão `none`. Tipo de agendador: `none`, `cosine`, `step`, `onecycle`.
- `--lr-warmup-steps`: padrão `0`. Passos de warmup linear (aumenta LR de 0 até `--lr`).
- `--lr-eta-min`: padrão `0.0`. LR mínimo para o scheduler `cosine`.
- `--lr-step-size`: padrão `500`. Intervalo (passos) para redução no scheduler `step`.
- `--lr-gamma`: padrão `0.5`. Fator de multiplicação no scheduler `step`.
- `--log-every`: padrão `25`. A cada N passos, exibe e salva métricas.
- `--preview-every`: padrão `100`. A cada N passos, salva uma imagem comparativa (original vs transformada).
- `--save-every`: padrão `500`. A cada N passos, salva um checkpoint do modelo.

### Flags de regularização e qualidade

- `--target-cos`: padrão `0.25`. Similaridade cosseno alvo para a perda de identidade (quanto menor, mais agressiva a desidentificação).
- `--tau-ssim`: padrão `0.95`. Limiar de SSIM abaixo do qual aplicamos penalidade.
- `--lambda-id`: padrão `1.0`. Peso da perda de identidade.
- `--lambda-ssim`: padrão `20.0`. Peso da perda de similaridade estrutural (SSIM).
- `--lambda-pixel`: padrão `2.0`. Peso da perda L2 entre original e transformada.
- `--lambda-tv`: padrão `0.2`. Peso da perda de variação total (suavidade).
- `--lambda-flow-smooth`: padrão `5.0`. Peso da regularização do fluxo geométrico.
- `--lambda-dct-l2`: padrão `0.01`. Regularização da magnitude dos coeficientes DCT.
- `--lambda-photo-l2`: padrão `0.01`. Regularização da magnitude do ajuste fotométrico.

### Flags da transformação

- `--dct-k`: padrão `32`. Resolução da grade DCT (base de frequência).
- `--dct-fmin`: padrão `2`. Frequência mínima a ser perturbada (exclui DC).
- `--dct-fmax`: padrão `18`. Frequência máxima a ser perturbada.
- `--flow-grid`: padrão `12`. Resolução do campo de fluxo (baixa resolução).
- `--photo-grid`: padrão `12`. Resolução do ajuste fotométrico local.
- `--max-dct-amp`: padrão `0.035`. Amplitude máxima da perturbação DCT (em escala de pixel).
- `--max-flow-px`: padrão `2.0`. Deslocamento máximo em pixels para o fluxo.
- `--max-photo-amp`: padrão `0.035`. Amplitude máxima do ajuste fotométrico.
- `--no-face-mask`: (não ativado por padrão). Se ativado, desabilita a máscara elíptica (a transformação afeta toda a imagem).

### Flags do agendador de taxa de aprendizado (scheduler)

- `none` (padrão): LR constante.
- `cosine`: Decaimento suave seguindo um cosseno. Use com `--lr-eta-min` (ex.: `1e-6`) e `--lr-warmup-steps`.
- `step`: Reduz o LR por um fator `--lr-gamma` a cada `--lr-step-size` passos.
- `onecycle`: Aumenta o LR até `--lr` durante a fração `pct_start` (warmup) e depois decai. Ideal para treinos curtos.

**Recomendação para 2000 passos**: `--lr 0.02 --lr-scheduler cosine --lr-warmup-steps 200 --lr-eta-min 1e-6`

### Exemplo completo de treinamento

```bash
python deid_optimize.py train \
    --data ./faces_train \
    --out ./runs/deid_v1 \
    --device cuda \
    --batch-size 16 \
    --steps 2000 \
    --lr 0.02 \
    --lr-scheduler cosine \
    --lr-warmup-steps 200 \
    --lr-eta-min 1e-6 \
    --target-cos 0.15 \
    --lambda-id 5.0 \
    --max-flow-px 2.5 \
    --max-photo-amp 0.06 \
    --max-dct-amp 0.06 \
    --tau-ssim 0.92 \
    --lambda-ssim 20.0 \
    --lambda-pixel 3.0
```

Após a execução, a pasta `./runs/deid_v1` conterá:

- `transform.pt` – checkpoint final do modelo.
- `steps.txt` – arquivo com métricas por passo (loss, id, cos, ssim, pix, tv, tempo, distância euclidiana, lr).
- `previews/` – imagens comparativas a cada `--preview-every` passos.
- checkpoints intermediários (ex.: `transform_step_0500.pt`).

## Monitoramento do treinamento

O arquivo `steps.txt` pode ser visualizado com `cat` ou carregado em uma planilha. As colunas são:

    step loss id cos ssim pix tv elapsed euclid lr

- `cos`: similaridade cosseno média entre os embeddings da face original e transformada. Desejamos valores baixos (próximos de `--target-cos`).
- `euclid`: distância euclidiana correspondente (com embeddings normalizados). Quanto maior, melhor a desidentificação. A relação é `euclid = sqrt(2*(1-cos))`.
- `ssim`: similaridade estrutural média (1 = idêntica). Valores típicos > 0.85.
- `lr`: taxa de aprendizado efetiva no passo (útil para debug do scheduler).

A cada `--preview-every` passos, uma imagem lado a lado (original vs transformada) é salva em `previews/`. Isso ajuda a avaliar visualmente a qualidade.

## Aplicação do modelo (de-identificação)

Após o treinamento, use o modo `apply` para processar novas imagens:

```bash
python deid_optimize.py apply \
    --checkpoint ./runs/deid_v1/transform.pt \
    --input ./faces_test \
    --output ./faces_deid \
    --device cuda \
    --image-size 224 \
    --batch-size 8
```

Argumentos do modo `apply`:

- `--checkpoint`: obrigatório. Caminho do `.pt` gerado no treinamento.
- `--input`: obrigatório. Pasta com imagens a serem transformadas.
- `--output`: obrigatório. Pasta de destino (será criada).
- `--device`: padrão `cpu`. `cpu` ou `cuda`.
- `--image-size`: padrão `224`. Redimensionar imagens para esse tamanho antes da transformação.
- `--batch-size`: padrão `8`. Tamanho do batch (processamento em lote).
- `--num-workers`: padrão `0`. Número de workers para carregamento.

O script processa todas as imagens suportadas (jpg, png, bmp, webp) e salva com o mesmo nome na pasta de saída. A qualidade de salvamento é 95 (JPEG).

## Interpretando os resultados

- **Desidentificação bem‑sucedida**: `cos` < 0.2 e `euclid` > 1.2, mantendo `ssim` > 0.85.
- **Desidentificação muito agressiva**: `cos` próximo de 0 ou negativo, `euclid` > 1.5 – pode causar artefatos visuais. Nesse caso, reduza `--lambda-id` ou aumente `--tau-ssim`.
- **Pouca desidentificação**: `cos` > 0.4 – aumente `--lambda-id`, diminua `--target-cos` ou aumente `--max-flow-px` / `--max-dct-amp`.
- **Qualidade visual baixa**: `ssim` abaixo de 0.8 – aumente `--lambda-ssim`, `--lambda-pixel` ou reduza `--max-flow-px`, `--max-photo-amp`.

## Dicas para ajuste fino

1. Sempre monitore o `steps.txt`: plote as curvas de loss, cos e ssim para entender a convergência.
2. Comece com LR fixo (0.001) e `--target-cos 0.25` para ver o comportamento base.
3. Se a perda de identidade não diminuir: aumente `--lambda-id` (ex.: 5.0) ou `--lr`.
4. Se a qualidade cair muito: aumente `--lambda-ssim` (ex.: 30.0) ou `--tau-ssim` (ex.: 0.92).
5. Use o scheduler `onecycle` quando tiver poucos passos (≤2000) – geralmente converge mais rápido.
6. Warmup é quase sempre benéfico para evitar instabilidade inicial. Use 5–10% do total de steps.
7. Para manter naturalidade, evite `--max-flow-px` > 3.0 ou `--max-photo-amp` > 0.1.

