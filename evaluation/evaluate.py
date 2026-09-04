import torch
from torch.utils.data import DataLoader
from pathlib import Path

from models.masks import (
    parse_mask_regions,
    resolve_ssim_region_for_eval,
)
from utils.checkpoint import load_transformer_from_checkpoint
from data.dataset import FaceImageFolder
from evaluation.validate import compute_validation_metrics

@torch.no_grad()
def evaluate(args):
    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ssim_region = resolve_ssim_region_for_eval(args.ssim_region, ckpt)

    transformer = load_transformer_from_checkpoint(args.checkpoint, device)

    mask_mode = getattr(args, "mask_mode", None)
    mask_regions = getattr(args, "mask_regions", None)
    mask_shape = getattr(args, "mask_shape", None)
    effective_mode = mask_mode or transformer.mask_mode
    if mask_regions is not None:
        regions = parse_mask_regions(mask_regions)
    else:
        regions = transformer.mask_regions
    shape = mask_shape or getattr(transformer, "mask_shape", "ellipse")

    landmark_detector = None
    need_landmarks = transformer.use_face_mask and (
        effective_mode == "landmarks" or ssim_region == "full-landmarks"
    )
    if need_landmarks:
        from models.face_detector import get_default_detector

        landmark_detector = get_default_detector()

    from models.embedders import load_authorized_embedders
    embedders = load_authorized_embedders(device)

    dataset = FaceImageFolder(args.data, args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    metrics = compute_validation_metrics(
        transformer,
        embedders,
        loader,
        device,
        mask_mode=effective_mode,
        mask_regions=regions,
        mask_shape=shape,
        ssim_region=ssim_region,
        use_face_mask=transformer.use_face_mask,
        landmark_detector=landmark_detector,
        max_samples=0,
    )

    mean_cos = metrics["cos_mean"]
    mean_ssim = metrics["ssim_mean"]
    mean_euclid = metrics["euclid_mean"]
    total_images = int(metrics["n_images"])

    print("\n========== Evaluation Results ==========")
    print(f"Total images evaluated: {total_images}")
    print(f"SSIM region: {ssim_region}")
    print(f"Mean cosine similarity: {mean_cos:.4f}")
    print(f"Mean Euclidean distance: {mean_euclid:.4f}")
    print(f"Mean SSIM: {mean_ssim:.4f}")
    print("========================================\n")

    if args.output_summary:
        out_path = Path(args.output_summary)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            f.write(f"images,{total_images}\n")
            f.write(f"cos_mean,{mean_cos:.6f}\n")
            f.write(f"euclid_mean,{mean_euclid:.6f}\n")
            f.write(f"ssim_mean,{mean_ssim:.6f}\n")
        print(f"Summary saved to {out_path}")
