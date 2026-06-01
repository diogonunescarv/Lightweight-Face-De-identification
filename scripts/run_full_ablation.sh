#!/bin/bash

# -------------------------------
# Configurações fixas
# -------------------------------
CODE_DIR="/home/dcarvalho/Lightweight-Face-De-identification"
DATA_DIR="/mnt/study-data/dcarvalho/datasets/lfw/faces_train"
OUTPUT_BASE="/mnt/study-data/dcarvalho/ablation_tests/fixed_lr"
STEPS=2000

# Navega até o código
cd "$CODE_DIR" || exit 1

# Verifica se a GPU está disponível (opcional)
if python -c "import torch; print(torch.cuda.is_available())" | grep -q "False"; then
    echo "AVISO: GPU não detectada. Usando CPU."
    DEVICE="cpu"
else
    DEVICE="cuda"
fi

echo "Iniciando experimentos em lote. Device: $DEVICE"
echo "Diretório de saída base: $OUTPUT_BASE"

# -------------------------------
# Definição das configurações (V5, V10, V11)
# Cada entrada: nome e lista de argumentos específicos
# -------------------------------
declare -A CONFIGS
CONFIGS["V5_original"]="--lambda-id 4.0 --target-cos 0.15 --max-dct-amp 0.06 --max-flow-px 2.75 --max-photo-amp 0.06 --tau-ssim 0.92 --lambda-ssim 20.0 --lambda-pixel 3.0 --lr 0.008"
CONFIGS["V10__original"]="--lambda-id 6.0 --target-cos 0.1 --max-dct-amp 0.06 --max-flow-px 2.75 --max-photo-amp 0.06 --tau-ssim 0.90 --lambda-ssim 20.0 --lambda-pixel 3.0 --lr 0.006"
CONFIGS["V11_original"]="--lambda-id 5.0 --target-cos 0.15 --max-dct-amp 0.06 --max-flow-px 2.5 --max-photo-amp 0.06 --tau-ssim 0.92 --lambda-ssim 20.0 --lambda-pixel 3.0 --lr 0.008"

# -------------------------------
# Lista de ablações: (nome, flags adicionais)
# -------------------------------
ABLATIONS=(
    "baseline"
    "no_dct --disable-dct"
    "no_flow --disable-flow"
    "no_photo --disable-photo"
)

# -------------------------------
# Função para executar treino e avaliação
# -------------------------------
run_experiment() {
    local config_name=$1
    local config_args=$2
    local ablation_name=$3
    local ablation_flags=$4

    local out_dir="$OUTPUT_BASE/$config_name/$ablation_name"
    echo "===================================================="
    echo "Executando: $config_name / $ablation_name"
    echo "Saída: $out_dir"
    echo "Flags extras: $ablation_flags"
    echo "===================================================="

    # Treinamento
    python deid_optimize.py train \
        --data "$DATA_DIR" \
        --out "$out_dir" \
        --steps $STEPS \
        --device "$DEVICE" \
        $config_args \
        $ablation_flags

    # Avaliação (opcional, mas recomendada)
    # Certifique-se de que o comando evaluate está implementado
    echo "Avaliando $config_name / $ablation_name ..."
    python deid_optimize.py evaluate \
        --checkpoint "$out_dir/transform.pt" \
        --data "$DATA_DIR" \
        --output-summary "$out_dir/eval.txt" \
        --device "$DEVICE"

    echo "Concluído: $config_name / $ablation_name"
    echo ""
}

# -------------------------------
# Loop principal: para cada configuração e cada ablação
# -------------------------------
for config_name in "${!CONFIGS[@]}"; do
    config_args="${CONFIGS[$config_name]}"
    for ablation in "${ABLATIONS[@]}"; do
        # Divide a string em nome e flags
        set -- $ablation
        ablation_name=$1
        shift
        ablation_flags="$*"
        run_experiment "$config_name" "$config_args" "$ablation_name" "$ablation_flags"
    done
done

echo "===================================================="
echo "Todos os experimentos de ablação foram concluídos!"
echo "Resultados salvos em: $OUTPUT_BASE"
echo "===================================================="