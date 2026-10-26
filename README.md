# Lightweight Face De-identification

Este projeto implementa uma transformação facial leve para desidentificação visual, preservando a qualidade da imagem. O modelo ajusta uma transformação paramétrica que pode operar no domínio da **DCT** (transformada cosseno) ou da **DT‑CWT** decimated (wavelet complexa de duas árvores, criticamente amostrada), combinada com deformação geométrica suave e ajuste fotométrico local.

## Sumário

- [Status V2](#status-v2)
- [Requisitos](#requisitos)
- [Instalação](#instalação)
- [Estrutura do código](#estrutura-do-código)
- [Preparação dos dados](#preparação-dos-dados)
- [Treinamento V1 (legado)](#treinamento)
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
- [Busca de hiperparâmetros](#busca-de-hiperparâmetros)
- [Interpretando os resultados](#interpretando-os-resultados)
- [Dicas para ajuste fino](#dicas-para-ajuste-fino)

---

## Status V2

**V2 treinável** (Ideias 1 e 2) na branch `feature/v2-dtcwt-meta-mask`. Spec: [`V2.md`](V2.md). Decisões: [`DECISIONS.md`](DECISIONS.md).  
Motor: meta-máscara DT-CWT espacial + ROI (`models/masks.py`); **sem** warp/photo/DCT. Artefatos (logs, ckpts, eval, viz) ficam **só** na pasta do run.

### Treino rápido

```bash
bash scripts/run_v2_train_small.sh          # campeão: protego + FR small + --roi-compose pixel
bash scripts/run_v2_train_wavelet.sh        # Ideia 2: --roi-compose wavelet
bash scripts/run_v2_train_full.sh           # ensemble full (VRAM alto)

python deid_optimize.py evaluate-v2 \
  --checkpoint /path/to/run/meta_mask_best.pt \
  --data /path/to/faces --max-samples 16 --no-lpips   # → <run>/eval/

python scripts/run_v2_visualize_meta_mask.py \
  --checkpoint /path/to/run/meta_mask.pt --image face.jpg   # → <run>/viz/
```

Smoke sem FR_DB: acrescente `--smoke-fr` (FaceNets built-in). Produção `small`/`full`: `PROTEGO_FR_ROOT` ou `third_party/protego_fr/`.

### `--roi-compose` (Ideia 1 vs 2)

| Valor | Pipeline | Quando |
|-------|----------|--------|
| `pixel` (**default**) | ΔW full → IDT-CWT → residual⊙M → L∞ | baseline / regressão |
| `wavelet` | ΔW⊙M_ℓ → IDT-CWT → L∞ full-frame | ablação Ideia 2 |

Cos e SSIM **sempre** sobre \(x_{\mathrm{prot}}\) final. Flags ROI iguais à V1: `--mask-mode`, `--mask-shape`, `--mask-regions`, `--no-face-mask`, `--ssim-region`.

### Flags `train-v2` (essenciais)

| Flag | Default | Nota |
|------|---------|------|
| `--roi-compose` | `pixel` | `pixel` \| `wavelet` |
| `--optimizer` | `protego` | SignSGD; ou `adam` |
| `--fr-preset` | `small` | `full` = VRAM alto |
| `--editable-levels` | `2,3` | nível 1 só ablação (warning) |
| `--epsilon` | `16/255` | orçamento L∞ |
| `--min-ssim` | `0.95` | hinge perceptual |
| `--mask-regions` | `full` | ROI landmarks; ablação: `band` / `--no-face-mask` |
| `--early-stopping` | on | métrica default `cos` |

Checkpoints V1 **não** carregam no módulo V2. SSIM canônico: `ssim_index` / `ssim_index_masked` em `losses/losses.py` (V1 e V2).

| | V1 (`train`) | V2 (`train-v2`) |
|--|--------------|-----------------|
| Motor | DCT/DT-CWT global + flow + photo | DT-CWT meta-máscara espacial |
| ROI | ellipse/band | mesmas flags + `--roi-compose` |
| Loss ID | hinge `target_cos` | mean cos (Phase 3.3) |
| Otimizador | Adam (+ schedulers) | `protego` \| `adam` |

### Fora da V2 → **V2.1**

Warm-start + 1–5 steps online por imagem; eval Phase 5 em massa; dispersão-K / Gram. A V2 só deixa o checkpoint e o `forward(x, face_mask, roi_compose=...)` prontos.

O restante deste README descreve o caminho **V1 (legado)**.

## Requisitos

- Python 3.8+
- PyTorch 1.9+
- torchvision
- facenet-pytorch (para o embedder facial)
- insightface + onnxruntime (para detecção facial SCRFD)
- opencv-python-headless (conversão de imagem para o detector)
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

### Modelo de detecção facial (buffalo_m)

O detector facial (`models/face_detector.py`) usa o pack **buffalo_m** do insightface (SCRFD-2.5GF, ONNX `det_2.5g.onnx`). Baixe o pack na primeira utilização:

```bash
python -c "from insightface.utils import ensure_available; ensure_available('models', 'buffalo_m')"
```

Os arquivos são extraídos em `~/.insightface/models/buffalo_m/`. Para inferência acelerada em GPU, instale `onnxruntime-gpu` no lugar de `onnxruntime`.

Uso básico:

```python
from models.face_detector import detect_face

result = detect_face("caminho/para/imagem.jpg")
if result is not None:
    bbox, landmarks_5pts, score = result
```

Validação visual: execute `notebooks/validate_face_detector.ipynb` (lê imagens do CelebA-HQ local e salva figuras em `notebooks/outputs/detector_validation/`).

## Estrutura do código

```bash
.
├── V2.md / DECISIONS.md           # Spec V2 + decisões
├── deid_optimize.py               # CLI: train-v2 | evaluate-v2 | train (V1) | …
├── models/
│   ├── dtcwt_meta_mask.py         # V2: DTCWTMetaMaskModule (pixel|wavelet)
│   ├── fr/                        # V2: ensemble FR (presets + FaceNet smoke)
│   ├── masks.py / face_detector.py  # ROI compartilhada V1/V2
│   ├── transformer.py             # V1: LightweightDeIdentifier
│   ├── embedders.py               # FaceNet (V1 / smoke)
│   └── wavelet_transform.py       # V1 DT-CWT decimated
├── data/
│   ├── merged_split.py            # V2: LFW+CelebA frac split
│   ├── dataset.py / transforms.py
├── losses/
│   ├── identity.py / quality.py   # V2: cos + SSIM hinge / adapt_w
│   └── losses.py                  # V1 + SSIM canônico (reusado pela V2)
├── training/
│   ├── trainer_v2.py              # V2: épocas + SignSGD/Adam + ES
│   └── trainer.py                 # V1
├── evaluation/
│   ├── evaluate_v2.py / validate_v2.py / visualize_v2.py
│   └── evaluate.py / evaluate_single.py   # V1
├── inference/apply.py             # V1
└── scripts/run_v2_*.sh            # runners V2 (treino / testes / viz)
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
| `--save-every` | `500` | A cada N passos, salva um checkpoint do modelo (ignorado se `--early-stopping`). |
| `--early-stopping` | (desativado) | Validação periódica + parada antecipada (requer `--val-data`). |
| `--val-data` | — | Pasta de validação para early stopping. |
| `--eval-every` | `500` | Intervalo em steps entre avaliações de validação. |
| `--patience` | `5` | Avaliações consecutivas sem melhora antes de parar. |
| `--val-max-samples` | `200` | Limite de imagens na validação durante treino (0 = todas). |
| `--early-stopping-metric` | `score` | `score` (= ssim−cos, max), `euclid`/`ssim` (max), `cos` (min). |
| `--early-stopping-min-delta` | `0.0` | Melhora mínima absoluta para contar como progresso. |
| `--seed` | `42` | Semente para reprodutibilidade. |

### Flags de regularização e qualidade

| Flag | Padrão | Descrição |
|------|--------|-----------|
| `--target-cos` | `0.25` | Similaridade cosseno alvo para a perda de identidade (quanto menor, mais agressiva a desidentificação). |
| `--tau-ssim` | `0.95` | Limiar de SSIM abaixo do qual aplicamos penalidade. |
| `--ssim-region` | `full-landmarks` | Escopo do SSIM na loss: `full-landmarks` = elipse `full` via SCRFD (independente da máscara de treino); `mask` = mesma região de `--mask-regions`/`--mask-shape` (comportamento antigo). |
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
| `--transform-type` | `dct` | Tipo de transformada espectral: `dct` ou `dtcwt` (DT‑CWT criticamente amostrada / decimated). |
| `--dct-k` | `32` | Resolução da grade DCT (base de frequência). |
| `--dct-fmin` | `2` | Frequência mínima a ser perturbada (exclui DC). |
| `--dct-fmax` | `18` | Frequência máxima a ser perturbada. |
| `--wavelet-J` | `3` | Número de níveis (escalas) da DT‑CWT decimated; cada nível reduz a resolução espacial pela metade. |
| `--max-wavelet-amp` | `0.2` | Fallback de amplitude quando mag/phase não são passados (escala = 1 ± amp, fase = ± amp rad). |
| `--max-wavelet-mag-amp` | `(=max-wavelet-amp)` | Amplitude máxima da escala de magnitude dos detalhes. |
| `--max-wavelet-phase-amp` | `(=max-wavelet-amp)` | Amplitude máxima do deslocamento de fase (radianos). |
| `--wavelet-level-amp-scales` | `1,1,...,1` | Multiplicadores por nível (lista CSV de comprimento J). |
| `--wavelet-polar-mode` | `complex` | `complex` = polar verdadeiro via `view_as_complex`; `legacy` = abs/angle elemento a elemento no stack real/imag (ckpts antigos). |
| `--wavelet-per-channel` | off | Mag/fase por canal RGB `(1,3,6,J)`. Exclusivo com spatial grid. |
| `--wavelet-spatial-grid` | `0` | Se `>0`, mapas espaciais `g×g` de mag/fase `(1,6,J,g,g)`. |
| `--wavelet-yl-gain` | off | Ganho aprendível na aproximação `Yl`. |
| `--max-wavelet-yl-amp` | `0.05` | Amplitude máxima do ganho em `Yl`. |
| `--lambda-wavelet-mag` | `0.01` | Regularização L2 dos parâmetros de escala da wavelet. |
| `--lambda-wavelet-phase` | `0.01` | Regularização L2 dos parâmetros de fase da wavelet. |
| `--lambda-wavelet-smooth` | `0.001` | Suavidade entre níveis da wavelet. |
| `--lambda-wavelet-yl` | `0.01` | Regularização L2 do ganho `Yl` (se `--wavelet-yl-gain`). |
| `--flow-grid` | `12` | Resolução do campo de fluxo (baixa resolução). |
| `--photo-grid` | `12` | Resolução do ajuste fotométrico local. |
| `--max-dct-amp` | `0.035` | Amplitude máxima da perturbação DCT (em escala de pixel). |
| `--max-flow-px` | `2.0` | Deslocamento máximo em pixels para o fluxo. |
| `--max-photo-amp` | `0.035` | Amplitude máxima do ajuste fotométrico. |
| `--no-face-mask` | (não ativado) | Se ativado, desabilita a máscara (a transformação afeta toda a imagem). |
| `--mask-mode` | `fixed` | `fixed` = elipse centrada original; `landmarks` = máscaras por região SCRFD. |
| `--mask-shape` | `ellipse` | `ellipse` = elipses por landmark; `band` = faixa suave (unitária ou união). Só com `--mask-mode landmarks`. |
| `--mask-regions` | `full` | Unitários: `eyes`, `nose`, `mouth`, `full` (`full` só com `ellipse`). Compostos `band` (união): `eyes-nose`, `eyes-mouth`, `nose-mouth`, `eyes-nose-mouth`. Híbrido: `eyes-nose-mouth-hybrid` (retângulos + elipse na boca). |

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

Exemplo para *DT‑CWT* **freq-only** (isola o módulo espectral — linha experimental atual):

```bash
python deid_optimize.py train \
    --data ./faces_train \
    --out ./runs/deid_dtcwt_freq \
    --transform-type dtcwt \
    --wavelet-J 3 \
    --wavelet-polar-mode complex \
    --max-wavelet-mag-amp 0.20 \
    --max-wavelet-phase-amp 0.20 \
    --lambda-wavelet-mag 0.01 \
    --lambda-wavelet-phase 0.01 \
    --lambda-wavelet-smooth 0.001 \
    --disable-flow \
    --disable-photo \
    --mask-mode landmarks --mask-regions full \
    --ssim-region full-landmarks \
    --lambda-id 15 \
    --target-cos 0.12 \
    --tau-ssim 0.88 \
    --lambda-ssim 20.0 \
    --lambda-pixel 6.0 \
    --lr 0.008 \
    --lr-scheduler onecycle \
    --steps 5000
```

Exemplo combo (DT‑CWT + flow + photo), fora da linha freq-only:

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
    --max-dct-amp 0.09 \
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

## Busca de hiperparâmetros

Script Optuna para explorar combinações de `--lr`, `--lambda-id`, `--max-flow-px`, `--tau-ssim` e `--lambda-ssim` em treinos curtos com early stopping. Demais hiperparâmetros ficam fixos em torno de `mid_combo_amp20_flow45_nopx`. Cada trial maximiza `score = ssim_mean - cos_mean` (mesma métrica do early stopping).

```bash
python scripts/run_hparam_search.py \
  --data /caminho/faces_train \
  --val-data /caminho/faces_test \
  --out ./runs/hparam_search \
  --n-trials 30 --steps 1000 --device cuda --seed 42
```

Wrapper com paths LFW: `scripts/run_hparam_search.sh` (revisar `N_TRIALS`/`STEPS` antes de executar).

| Flag | Default | Descrição |
|------|---------|-----------|
| `--data` | — | Pasta de treino (obrigatório). |
| `--val-data` | — | Pasta de validação (obrigatório). |
| `--out` | — | Diretório base; trials em `trial_NNNN/`, CSV e SQLite aqui. |
| `--n-trials` | `30` | Número de trials Optuna. |
| `--steps` | `1000` | Steps máximos por trial. |
| `--eval-every` | `100` | Intervalo de validação por trial. |
| `--patience` | `3` | Early stopping por trial. |
| `--fresh` | (desativado) | Apaga `optuna_study.db` e recomeça do zero. |
| `--resume` | (obsoleto) | Retomar é automático se o DB existir; flag mantida por compatibilidade. |

Reexecução no mesmo `--out`: retoma trials pendentes até completar `--n-trials` no total. Use `--fresh` para apagar o study e iniciar outro.

Saídas: `hparam_search_results.csv` (HP + métricas por trial), `optuna_study.db`, e ao final um comando `deid_optimize.py train ...` pronto para treino completo. Coluna `status`: `complete`, `pruned`, `failed`, `no_eval`.

## Interpretando os resultados

**Escopo do SSIM:** por padrão (`--ssim-region full-landmarks`), o SSIM mede qualidade no rosto todo via landmarks, independente da máscara de transformação. Resultados em `EXPERIMENTS.md` anteriores a set/2026 usaram SSIM restrito à máscara de treino (`mask`) — compare SSIM entre experimentos só se o escopo for o mesmo.

**DT‑CWT / freq-only:** a linha experimental de evolução espectral treina com `--disable-flow --disable-photo` para isolar o módulo de frequência. Não compare cos/ssim desses runs diretamente com checkpoints combo (flow+photo) sem ressalva. Com `--wavelet-polar-mode complex` (default em treinos novos), magnitude/fase atuam no coeficiente complexo; checkpoints antigos sem metadado `wavelet_polar_mode` carregam como `legacy`.

(REPENSAR SOBRE)

- **Desidentificação bem‑sucedida**: `cos` < 0.2 e `euclid` > 1.2, mantendo `ssim` > 0.85.
- **Desidentificação muito agressiva**: `cos` próximo de 0 ou negativo, `euclid` > 1.5 – pode causar artefatos visuais. Nesse caso, reduza `--lambda-id` ou aumente `--tau-ssim`.
- **Pouca desidentificação**: `cos` > 0.4 – aumente `--lambda-id`, diminua `--target-cos` ou aumente `--max-flow-px` / `--max-dct-amp` / `--max-wavelet-mag-amp` (freq-only).
- **Qualidade visual baixa**: `ssim` abaixo de 0.8 – aumente `--lambda-ssim`, `--lambda-pixel` ou reduza amplitudes wavelet / flow / photo.

## Dicas para ajuste fino

1. Sempre monitore o `steps.txt`: plote as curvas de loss, cos e ssim para entender a convergência.
2. Comece com LR fixo (0.001) e `--target-cos 0.25` para ver o comportamento base.
3. Se a perda de identidade não diminuir: aumente `--lambda-id` (ex.: 5.0) ou `--lr`.
4. Se a qualidade cair muito: aumente `--lambda-ssim` (ex.: 30.0) ou `--tau-ssim` (ex.: 0.92).
5. Use o scheduler `onecycle` quando tiver poucos passos (≤2000) – geralmente converge mais rápido.
6. Warmup é quase sempre benéfico para evitar instabilidade inicial. Use 5–10% do total de steps.
7. Para manter naturalidade, evite `--max-flow-px` > 3.0 ou `--max-photo-amp` > 0.1.

