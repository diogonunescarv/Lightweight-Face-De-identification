#!/bin/bash
# Define os caminhos e parâmetros
CODE_DIR="/home/dcarvalho/Lightweight-Face-De-identification"
DATA_DIR="/mnt/study-data/dcarvalho/datasets/lfw/faces_train"
OUTPUT_BASE="/mnt/study-data/dcarvalho/seed_test"
STEPS=2000
SEED=42
DEVICE="cuda"   # ou "cpu"

cd "$CODE_DIR" || exit 1

# Hiperparâmetros do V10 (sem nenhuma flag de desabilitação)
V10_ARGS="--lambda-id 6.0 --target-cos 0.1 --max-dct-amp 0.06 --max-flow-px 2.75 --max-photo-amp 0.06 --tau-ssim 0.90 --lambda-ssim 20.0 --lambda-pixel 3.0 --lr 0.006"

# Primeira execução
echo "=== Execução 1 (seed=$SEED) ==="
python deid_optimize.py train \
    --data "$DATA_DIR" \
    --out "$OUTPUT_BASE/run1" \
    --steps $STEPS \
    --device "$DEVICE" \
    $V10_ARGS \
    --seed $SEED

# Avaliação
python deid_optimize.py evaluate \
    --checkpoint "$OUTPUT_BASE/run1/transform.pt" \
    --data "$DATA_DIR" \
    --output-summary "$OUTPUT_BASE/run1/eval.txt" \
    --device "$DEVICE"

# Segunda execução (mesmo seed)
echo "=== Execução 2 (seed=$SEED) ==="
python deid_optimize.py train \
    --data "$DATA_DIR" \
    --out "$OUTPUT_BASE/run2" \
    --steps $STEPS \
    --device "$DEVICE" \
    $V10_ARGS \
    --seed $SEED

python deid_optimize.py evaluate \
    --checkpoint "$OUTPUT_BASE/run2/transform.pt" \
    --data "$DATA_DIR" \
    --output-summary "$OUTPUT_BASE/run2/eval.txt" \
    --device "$DEVICE"

echo "Treinamentos concluídos. Agora compare os arquivos CSV e checkpoints."