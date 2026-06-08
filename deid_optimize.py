"""
    deid_optimizer.py - CLI para treinar, aplicar e avaliar transformação de desidentificação.
"""

import argparse
from training.trainer import train
from inference.apply import apply_transform
from evaluation.evaluate import evaluate

def build_parser():
    parser = argparse.ArgumentParser(
        description="Otimização de transformação leve para desidentificação facial autorizada."
    )

    sub = parser.add_subparsers(dest="mode", required=True)

    # -------------------------
    # train
    # -------------------------
    p_train = sub.add_parser("train")

    p_train.add_argument("--data", required=True, help="Pasta com faces normalizadas/cortadas para treinamento.")
    p_train.add_argument("--out", required=True, help="Pasta de saída.")
    p_train.add_argument("--device", default="cpu", help="cpu ou cuda.")
    p_train.add_argument("--image-size", type=int, default=224)
    p_train.add_argument("--batch-size", type=int, default=8)
    p_train.add_argument("--num-workers", type=int, default=0)
    p_train.add_argument("--steps", type=int, default=2000)
    p_train.add_argument("--lr", type=float, default=2e-2)

    p_train.add_argument("--dct-k", type=int, default=32)
    p_train.add_argument("--dct-fmin", type=int, default=2)
    p_train.add_argument("--dct-fmax", type=int, default=18)
    p_train.add_argument("--flow-grid", type=int, default=12)
    p_train.add_argument("--photo-grid", type=int, default=12)

    p_train.add_argument("--max-dct-amp", type=float, default=0.035)
    p_train.add_argument("--max-flow-px", type=float, default=2.0)
    p_train.add_argument("--max-photo-amp", type=float, default=0.035)
    p_train.add_argument("--no-face-mask", action="store_true")

    p_train.add_argument("--target-cos", type=float, default=0.25)
    p_train.add_argument("--tau-ssim", type=float, default=0.95)

    p_train.add_argument("--lambda-id", type=float, default=1.0)
    p_train.add_argument("--lambda-ssim", type=float, default=20.0)
    p_train.add_argument("--lambda-pixel", type=float, default=2.0)
    p_train.add_argument("--lambda-tv", type=float, default=0.2)
    p_train.add_argument("--lambda-flow-smooth", type=float, default=5.0)
    p_train.add_argument("--lambda-dct-l2", type=float, default=0.01)
    p_train.add_argument("--lambda-photo-l2", type=float, default=0.01)

    p_train.add_argument("--log-every", type=int, default=25)
    p_train.add_argument("--preview-every", type=int, default=100)
    p_train.add_argument("--save-every", type=int, default=500)

    p_train.add_argument("--transform-type", default="dct", choices=["dct", "dtcwt"])
    p_train.add_argument("--wavelet-J", type=int, default=3)
    p_train.add_argument("--max-wavelet-amp", type=float, default=0.2)
    p_train.add_argument("--lambda-wavelet-mag", type=float, default=0.01)
    p_train.add_argument("--lambda-wavelet-phase", type=float, default=0.01)
    p_train.add_argument("--lambda-wavelet-smooth", type=float, default=0.001)

    p_train.add_argument("--lr-scheduler", type=str, default="none",
                     choices=["none", "cosine", "step", "onecycle"],
                     help="Tipo de agendador: none (fixo), cosine, step, onecycle")

    p_train.add_argument("--onecycle-pct-start", type=float, default=0.3, help="pct_start para OneCycleLR")
    p_train.add_argument("--onecycle-anneal-strategy", type=str, default="cos", choices=["cos", "linear"])

    p_train.add_argument("--lr-warmup-steps", type=int, default=0,
                     help="Número de passos para warmup linear (só cosine/onecycle)")
    p_train.add_argument("--lr-eta-min", type=float, default=0.0,
                        help="LR mínimo para cosine annealing")
    p_train.add_argument("--lr-step-size", type=int, default=500,
                        help="Intervalo de steps para StepLR")
    p_train.add_argument("--lr-gamma", type=float, default=0.5,
                        help="Fator de decaimento para StepLR")

    p_train.add_argument("--disable-dct", action="store_true", help="Desabilita perturbação DCT")
    p_train.add_argument("--disable-flow", action="store_true", help="Desabilita deformação geométrica")
    p_train.add_argument("--disable-photo", action="store_true", help="Desabilita ajuste fotométrico")

    p_train.add_argument("--seed", type=int, default=42, help="Semente para reprodutibilidade")

    # -------------------------
    # apply
    # -------------------------
    p_apply = sub.add_parser("apply")

    p_apply.add_argument("--checkpoint", required=True)
    p_apply.add_argument("--input", required=True)
    p_apply.add_argument("--output", required=True)
    p_apply.add_argument("--device", default="cpu")
    p_apply.add_argument("--image-size", type=int, default=224)
    p_apply.add_argument("--batch-size", type=int, default=8)
    p_apply.add_argument("--num-workers", type=int, default=0)

    # -------------------------
    # evaluate
    # -------------------------
    p_evaluate = sub.add_parser("evaluate", help="Avalia um checkpoint em um conjunto de validação")

    p_evaluate.add_argument("--checkpoint", required=True, help="Arquivo .pt do transformador treinado")
    p_evaluate.add_argument("--data", required=True, help="Pasta com imagens de validação (faces)")
    p_evaluate.add_argument("--device", default="cpu", help="cpu ou cuda")
    p_evaluate.add_argument("--image-size", type=int, default=224)
    p_evaluate.add_argument("--batch-size", type=int, default=8)
    p_evaluate.add_argument("--num-workers", type=int, default=0)
    p_evaluate.add_argument("--output-summary", type=str, default=None,
                            help="Opcional: caminho para salvar um resumo em CSV/txt")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "apply":
        apply_transform(args)
    elif args.mode == "evaluate":
        evaluate(args)
    else:
        raise RuntimeError(f"Modo desconhecido: {args.mode}")


if __name__ == "__main__":
    main()