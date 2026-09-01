import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path

from models.masks import (
    detect_landmarks_batch,
    parse_mask_regions,
    resolve_face_mask_for_model,
    resolve_ssim_mask,
    resolve_ssim_region_for_eval,
)
from utils.checkpoint import load_transformer_from_checkpoint
from data.dataset import FaceImageFolder
from losses.losses import ssim_index_masked

@torch.no_grad()
def evaluate(args):
    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ssim_region = resolve_ssim_region_for_eval(args.ssim_region, ckpt)

    transformer = load_transformer_from_checkpoint(args.checkpoint, device)
    transformer.eval()

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

    all_cos = []
    all_ssim = []
    total_images = 0

    for x, paths in loader:
        x = x.to(device)

        landmarks_batch = None
        if need_landmarks:
            landmarks_batch = detect_landmarks_batch(paths, landmark_detector)

        face_mask = resolve_face_mask_for_model(
            transformer,
            paths,
            device,
            mask_mode=mask_mode,
            mask_regions=mask_regions,
            mask_shape=mask_shape,
            detector=landmark_detector,
            landmarks_batch=landmarks_batch,
        )
        y = transformer(x, face_mask=face_mask)

        batch_cos = []
        for model in embedders:
            e0 = F.normalize(model(x), dim=1)
            e1 = F.normalize(model(y), dim=1)
            cos = (e0 * e1).sum(dim=1)
            batch_cos.append(cos)
        mean_cos = torch.stack(batch_cos).mean(dim=0)
        all_cos.append(mean_cos)

        ssim_mask = resolve_ssim_mask(
            transformer,
            paths,
            device,
            ssim_region=ssim_region,
            face_mask=face_mask,
            mask_mode=effective_mode,
            mask_regions=regions,
            mask_shape=shape,
            landmarks_batch=landmarks_batch,
            detector=landmark_detector,
        )
        ssim_vals = ssim_index_masked(x, y, ssim_mask)
        all_ssim.append(ssim_vals)

        total_images += x.shape[0]

    cos_all = torch.cat(all_cos)
    ssim_all = torch.cat(all_ssim)

    mean_cos = cos_all.mean().item()
    mean_ssim = ssim_all.mean().item()
    mean_euclid = math.sqrt(2.0 * (1.0 - mean_cos))

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
