# Lightweight Face De-identification

Este projeto implementa uma transformação facial leve para desidentificação visual, preservando a qualidade da imagem. O modelo ajusta uma transformação paramétrica que pode operar no domínio da **DCT** (transformada cosseno) ou da **DT‑CWT** (wavelet complexa de duas árvores), combinada com deformação geométrica suave e ajuste fotométrico local.

## Sumário

- [Requisitos](#requisitos)
- [Instalação](#instalação)
- [Estrutura do código](#estrutura-do-código)
- [Preparação dos dados](#preparação-dos-dados)
- [Treinamento](#treinamento)
  - [Flags principais](#flags-principais)
  - [Flags de regularização e qualidade](#flags-de-regularização-e-qualidade)
  - [Flags da transformação espectral (DCT / DT‑CWT)](#flags-da-transformação-espectral-dct--dt-cwt)
  - [Flags do agendador de taxa de aprendizado](#flags-do-agendador-de-taxa-de-aprendizado)
  - [Exemplo completo de treinamento](#exemplo-completo-de-treinamento)
- [Monitoramento do treinamento](#monitoramento-do-treinamento)
- [Aplicação do modelo (de‑identificação)](#aplicação-do-modelo-de-identificação)
- [Avaliação de um modelo](#avaliação-de-um-modelo)
  - [Avaliação detalhada (com métricas e imagens)](#avaliação-detalhada-com-métricas-e-imagens)
  - [Avaliação resumida](#avaliação-resumida)
- [Interpretando os resultados](#interpretando-os-resultados)
- [Dicas para ajuste fino](#dicas-para-ajuste-fino)

---

## Requisitos

- Python 3.8+
- PyTorch 1.9+
- torchvision
- facenet-pytorch (para o embedder facial)
- pytorch-wavelets (para DT‑CWT)
- numpy, Pillow, tqdm

Instale as dependências com:

```bash
pip install -r requirements.txt
```

## Instalação

Clone o repositório e instale o pacote em modo editável (recomendado) ou apenas garanta que os módulos estão acessíveis.

```bash
git clone <seu_repositorio>
cd Lightweight-Face-De-identification
pip install -e .
```

Caso prefira não instalar, mantenha a estrutura de pastas e execute python `deid_optimize.py` a partir da raiz.

## Estrutura do código

```bash
.
├── deid_optimize.py               # Ponto de entrada (CLI)
├── models/
│   ├── transformer.py             # Classe LightweightDeIdentifier
│   ├── embedders.py               # FaceNetEmbedder e load_authorized_embedders
│   └── wavelet_transform.py       # DT-CWT
├── data/
│   ├── dataset.py                 # FaceImageFolder, tensor_to_pil, save_preview
│   └── transforms.py              # (opcional) aumentos de dados
├── losses/
│   └── losses.py                  # identity_loss_ensemble, SSIM, TV, L2
├── utils/
│   ├── checkpoint.py              # load_transformer_from_checkpoint
│   ├── scheduler.py               # WarmupScheduler
│   └── helpers.py                
├── evaluation/
│   ├── evaluate.py                # Avaliação simplificada
│   └── evaluate_single.py         # Avaliação detalhada (métricas + imagens)
├── training/
│   └── trainer.py                 # Treinamento do modelo
├── inference/
│   └── apply.py                   # Aplicação do modelo em lote
```

## Preparação dos dados

Crie uma pasta com imagens de faces **alinhadas e centralizadas** (ex.: cortadas para 224x224). Não é necessário ter labels, o treinamento é auto-supervisionado usando apenas as imagens originais.

Estrutura esperada:

```bash
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

| Flag | Padrão | Descrição |
|------|--------|------------|
| `--data` | obrigatório | Pasta com as imagens de treino. |
| `--out` | obrigatório | Pasta onde serão salvos os checkpoints, logs e previews. |
| `--device` | `cpu` | `cpu` ou `cuda` (se disponível). |
| `--image-size` | `224` | Tamanho das imagens (largura=altura). |
| `--batch-size` | `8` | Tamanho do batch. |
| `--num-workers` | `0` | Número de subprocessos para carregar dados. |
| `--steps` | `2000` | Número total de iterações (passos) de treinamento. |
| `--lr` | `0.001` | Taxa de aprendizado inicial (Adam). |
| `--log-every` | `25` | A cada N passos, exibe e salva métricas. |
| `--preview-every` | `100` | A cada N passos, salva uma imagem comparativa (original vs transformada). |
| `--save-every` | `500` | A cada N passos, salva um checkpoint do modelo. |
| `--seed` | `42` | Semente para reprodutibilidade. |

### Flags de regularização e qualidade

| Flag | Padrão | Descrição |
|------|--------|-----------|
| `--target-cos` | `0.25` | Similaridade cosseno alvo para a perda de identidade (quanto menor, mais agressiva a desidentificação). |
| `--tau-ssim` | `0.95` | Limiar de SSIM abaixo do qual aplicamos penalidade. |
| `--lambda-id` | `1.0` | Peso da perda de identidade. |
| `--lambda-ssim` | `20.0` | Peso da perda de similaridade estrutural (SSIM). |
| `--lambda-pixel` | `2.0` | Peso da perda L2 entre original e transformada. |
| `--lambda-tv` | `0.2` | Peso da perda de variação total (suavidade). |
| `--lambda-flow-smooth` | `5.0` | Peso da regularização do fluxo geométrico. |
| `--lambda-dct-l2` | `0.01` | Regularização da magnitude dos coeficientes DCT. |
| `--lambda-photo-l2` | `0.01` | Regularização da magnitude do ajuste fotométrico. |

### Flags da transformação

| Flag | Padrão | Descrição |
|------|--------|-----------|
| `--transform-type` | `dct` | Tipo de transformada espectral: `dct` ou `dtcwt`. |
| `--dct-k` | `32` | Resolução da grade DCT (base de frequência). |
| `--dct-fmin` | `2` | Frequência mínima a ser perturbada (exclui DC). |
| `--dct-fmax` | `18` | Frequência máxima a ser perturbada. |
| `--wavelet-J` | `3` | Número de níveis (escalas) da DT‑CWT. |
| `--max-wavelet-amp` | `0.2` | Amplitude máxima para modificação de magnitude e fase (escala = 1 ± amp, deslocamento de fase = ± amp rad). |
| `--lambda-wavelet-mag` | `0.01` | Regularização L2 dos parâmetros de escala da wavelet. |
| `--lambda-wavelet-phase` | `0.01` | Regularização L2 dos parâmetros de fase da wavelet. |
| `--lambda-wavelet-smooth` | `0.001` | Suavidade entre níveis da wavelet. |
| `--flow-grid` | `12` | Resolução do campo de fluxo (baixa resolução). |
| `--photo-grid` | `12` | Resolução do ajuste fotométrico local. |
| `--max-dct-amp` | `0.035` | Amplitude máxima da perturbação DCT (em escala de pixel). |
| `--max-flow-px` | `2.0` | Deslocamento máximo em pixels para o fluxo. |
| `--max-photo-amp` | `0.035` | Amplitude máxima do ajuste fotométrico. |
| `--no-face-mask` | (não ativado) | Se ativado, desabilita a máscara elíptica (a transformação afeta toda a imagem). |

### Flags do agendador de taxa de aprendizado (scheduler)

| Flag | Padrão | Descrição |
|------|--------|-----------|
| `--lr-scheduler` | `none` | Tipo de agendador: `none`, `cosine`, `step`, `onecycle`. |
| `--lr-warmup-steps` | `0` | Número de passos de warmup linear (aumenta de 0 até `--lr`). |
| `--lr-eta-min` | `0.0` | LR mínimo para o scheduler `cosine`. |
| `--lr-step-size` | `500` | Intervalo (passos) para redução no scheduler `step`. |
| `--lr-gamma` | `0.5` | Fator de multiplicação no scheduler `step`. |
| `--onecycle-pct-start` | `0.3` | Fração do total de passos usada para warmup no scheduler `onecycle`. |
| `--onecycle-anneal-strategy` | `cos` | Estratégia de decaimento: `cos` (cosseno) ou `linear`. |

**Recomendação para 2000 passos**: `--lr 0.02 --lr-scheduler cosine --lr-warmup-steps 200 --lr-eta-min 1e-6`

### Exemplo completo de treinamento

```bash
python deid_optimize.py train \
    --data ./faces_train \
    --out ./runs/deid_dct \
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

Exemplo para *DT‑CWT* (substitua `--transform-type dtcwt` e adicione as flags específicas):

```bash
python deid_optimize.py train \
    --data ./faces_train \
    --out ./runs/deid_dtcwt \
    --transform-type dtcwt \
    --wavelet-J 3 \
    --max-wavelet-amp 0.12 \
    --lambda-wavelet-mag 0.01 \
    --lambda-wavelet-phase 0.01 \
    --lambda-wavelet-smooth 0.001 \
    --max-dct-amp 0.09 \           # ainda usado como referência, mas não afeta a wavelet
    --lambda-id 18 \
    --target-cos 0.12 \
    --max-flow-px 3.5 \
    --max-photo-amp 0.12 \
    --tau-ssim 0.88 \
    --lambda-ssim 30.0 \
    --lambda-pixel 6.0 \
    --lr 0.008 \
    --lr-scheduler onecycle \
    --steps 5000
```

Após a execução, a pasta `./runs/deid_v1` conterá:

- `transform.pt` – checkpoint final do modelo.
- `steps.txt` – arquivo com métricas por passo (loss, id, cos, ssim, pix, tv, tempo, distância euclidiana, lr).
- `previews/` – imagens comparativas a cada `--preview-every` passos.
- checkpoints intermediários (ex.: `transform_step_XXXX.pt`).

## Monitoramento do treinamento

O arquivo `steps.txt` pode ser visualizado com `cat` ou carregado em uma planilha. As colunas são:

```bash
step loss id cos ssim pix tv elapsed euclid lr
```

- `cos`: similaridade cosseno média entre os embeddings da face original e transformada. Desejamos valores baixos (próximos de `--target-cos`).
- `euclid`: distância euclidiana correspondente (com embeddings normalizados). Quanto maior, melhor a desidentificação. A relação é `euclid = sqrt(2*(1-cos))`.
- `ssim`: similaridade estrutural média (1 = idêntica).
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

| Flag | Padrão | Descrição |
|------|--------|-----------|
| `--checkpoint` | obrigatório | Caminho do `.pt` gerado no treinamento. |
| `--input` | obrigatório | Pasta com imagens a serem transformadas. |
| `--output` | obrigatório | Pasta de destino (será criada). |
| `--device` | `cpu` | `cpu` ou `cuda`. |
| `--image-size` | `224` | Redimensionar imagens para esse tamanho antes da transformação. |
| `--batch-size` | `8` | Tamanho do batch (processamento em lote). |
| `--num-workers` | `0` | Número de workers para carregamento. |

O script processa todas as imagens suportadas (jpg, png, bmp, webp) e salva com o mesmo nome na pasta de saída. A qualidade de salvamento é 95 (JPEG).

## Avaliação de um modelo

### Avaliação detalhada (com métricas e imagens)

O subcomando `evaluate-single` calcula métricas (cos, distância euclidiana, SSIM, tempo de inferência) e opcionalmente salva as imagens transformadas para inspeção visual.

```bash
python deid_optimize.py evaluate-single \
    --checkpoint ./runs/deid_v1/transform.pt \
    --data ./faces_val \
    --output ./metrics/meu_modelo \
    --device cuda \
    --max-samples 200 \
    --save-images \
    --max-visual-samples 10
```

Argumentos:

 Flag | Padrão | Descrição |
|------|--------|-----------|
| `--checkpoint` | obrigatório | Caminho do `.pt` gerado no treinamento. |
| `--input` | obrigatório | Pasta com imagens a serem transformadas. |
| `--output` | obrigatório | Pasta de destino (será criada). |
| `--device` | `cpu` | `cpu` ou `cuda`. |
| `--batch-size` | `8` | Tamanho do batch (processamento em lote). |
| `--max-samples` | `0` | Número máximo de imagens a processar (0 = todas). |
| `--save-images` | (não ativado) | Se presente, salva as imagens transformadas e comparações lado a lado. |
| `-max-visual-samples` | `10` | Quantas imagens salvar visualmente (quando `--save-images` é usado). |

Saídas:

- `output.json` e `output.csv` com as métricas (média e desvio padrão).

- Pasta `output_transformed/` contendo:

  - `nome_original_transformed.jpg` – imagem transformada.

  - `nome_original_compare.jpg` – original e transformada lado a lado.

### Avaliação resumida

- Carrega o transformador e os mesmos embeddings usados no treinamento (FaceNet).

- Processa todas as imagens da pasta --data sem embaralhar (shuffle=False).

- Para cada imagem original x e transformada y:

- Calcula a similaridade cosseno entre os embeddings normalizados.

- Calcula o SSIM (média dos canais).

- Ao final, exibe a média da similaridade cosseno, da distância euclidiana (sqrt(2*(1-cos))) e do SSIM.

- Opcionalmente salva um resumo em arquivo texto.

- Essa avaliação é determinística e não envolve treinamento, permitindo comparar diferentes checkpoints (completo, sem DCT, sem flow, etc.) de forma justa.

```bash
python deid_optimize.py evaluate \
  --checkpoint ./runs/deid_v1/transform.pt \
  --data ./faces_val \
  --image-size 224 \
  --batch-size 16 \
  --device cuda \
  --output-summary ./runs/deid_v1/eval_results.txt
```

## Interpretando os resultados 

(REPENSAR SOBRE)

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

