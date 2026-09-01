"""
    deid_optimizer.py - CLI para treinar, aplicar e avaliar transformação de desidentificação.
"""

import argparse
from training.trainer import train
from inference.apply import apply_transform
from evaluation.evaluate import evaluate
from evaluation.evaluate_single import evaluate_single_model
from models.masks import (
    parse_mask_regions,
    parse_mask_shape,
    validate_mask_shape_regions,
)


_MASK_REGIONS_HELP = (
    "Unitários: eyes,nose,mouth,full (full só com --mask-shape ellipse). "
    "Compostos band (união das unitárias): eyes-nose,eyes-mouth,nose-mouth,eyes-nose-mouth. "
    "Híbrido band: eyes-nose-mouth-hybrid (retângulos olhos+nariz + elipse na boca). "
    "Vírgula = lista/união; hífen = token composto."
)


def add_mask_args(parser: argparse.ArgumentParser, *, for_train: bool = False) -> None:
    if for_train:
        parser.add_argument(
            "--mask-mode",
            default="fixed",
            choices=["fixed", "landmarks"],
            help="fixed=elipse centrada original; landmarks=máscaras por região SCRFD.",
        )
        parser.add_argument(
            "--mask-shape",
            default="ellipse",
            choices=["ellipse", "band"],
            help="ellipse=elipses por landmark; band=faixa/retângulo suave (default: ellipse).",
        )
        parser.add_argument(
            "--mask-regions",
            default="full",
            help=_MASK_REGIONS_HELP + " Só usado com --mask-mode landmarks.",
        )
    else:
        parser.add_argument(
            "--mask-mode",
            default=None,
            choices=["fixed", "landmarks"],
            help="Override do modo de máscara (default: metadados do checkpoint).",
        )
        parser.add_argument(
            "--mask-shape",
            default=None,
            choices=["ellipse", "band"],
            help="Override da forma de máscara (default: metadados do checkpoint).",
        )
        parser.add_argument(
            "--mask-regions",
            default=None,
            help="Override das regiões (default: metadados do checkpoint). " + _MASK_REGIONS_HELP,
        )


def add_ssim_region_arg(parser: argparse.ArgumentParser, *, for_train: bool = False) -> None:
    parser.add_argument(
        "--ssim-region",
        default="full-landmarks" if for_train else None,
        choices=["full-landmarks", "mask"],
        help=(
            "Escopo do SSIM: full-landmarks=elipse full via SCRFD; "
            "mask=mesma região de --mask-regions/--mask-shape. "
            + ("Default: full-landmarks." if for_train else "Default: metadado do checkpoint ou mask (legado).")
        ),
    )


def validate_mask_args(args) -> None:
    spec = getattr(args, "mask_regions", None)
    shape = getattr(args, "mask_shape", None)
    regions = None
    if spec is not None:
        regions = parse_mask_regions(spec)
    if shape is not None:
        parse_mask_shape(shape)
    if regions is not None and shape is not None:
        validate_mask_shape_regions(shape, regions)

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
    add_mask_args(p_train, for_train=True)

    p_train.add_argument("--target-cos", type=float, default=0.25)
    p_train.add_argument("--tau-ssim", type=float, default=0.95)
    add_ssim_region_arg(p_train, for_train=True)

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
    add_mask_args(p_apply)

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
    add_mask_args(p_evaluate)
    add_ssim_region_arg(p_evaluate)

    

    # -------------------------
    # evaluate-single
    # -------------------------  
    p_eval_single = sub.add_parser("evaluate-single", help="Avalia um único modelo e salva JSON/CSV + imagens")
    p_eval_single.add_argument("--checkpoint", required=True)
    p_eval_single.add_argument("--data", required=True)
    p_eval_single.add_argument("--output", required=True, help="Caminho base para saída (ex: ./metrics/meu_modelo)")
    p_eval_single.add_argument("--device", default="cpu")
    p_eval_single.add_argument("--batch-size", type=int, default=8)
    p_eval_single.add_argument("--num-workers", type=int, default=4)
    p_eval_single.add_argument("--max-samples", type=int, default=0, help="Limite de imagens para métricas (0=todas)")
    p_eval_single.add_argument("--save-images", action="store_true", help="Salvar imagens transformadas")
    p_eval_single.add_argument("--max-visual-samples", type=int, default=10, help="Nº máximo de imagens visuais")
    add_mask_args(p_eval_single)
    add_ssim_region_arg(p_eval_single)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "train":
        validate_mask_args(args)
        train(args)
    elif args.mode == "apply":
        validate_mask_args(args)
        apply_transform(args)
    elif args.mode == "evaluate":
        validate_mask_args(args)
        evaluate(args)
    elif args.mode == "evaluate-single":
        validate_mask_args(args)
        evaluate_single_model(
            checkpoint_path=args.checkpoint,
            data_dir=args.data,
            output_file=args.output,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_samples=args.max_samples,
            save_images=args.save_images,
            max_visual_samples=args.max_visual_samples,
            mask_mode=args.mask_mode,
            mask_regions=args.mask_regions,
            mask_shape=args.mask_shape,
            ssim_region=args.ssim_region,
        )
    else:
        raise RuntimeError(f"Modo desconhecido: {args.mode}")


if __name__ == "__main__":
    main()