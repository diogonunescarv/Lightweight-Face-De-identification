"""V2 validation metrics over meta-mask + ROI (surrogate or held-out FRs)."""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F

from losses.quality import compute_ssim
from models.masks import (
    detect_landmarks_batch,
    resolve_face_mask_by_size,
    resolve_ssim_mask_by_size,
)


@torch.no_grad()
def validate_v2(
    module: torch.nn.Module,
    embedders: Sequence[torch.nn.Module],
    val_loader,
    device: torch.device,
    *,
    image_size: int,
    roi_compose: str = "pixel",
    use_face_mask: bool = True,
    mask_mode: str = "landmarks",
    mask_regions: tuple[str, ...] = ("full",),
    mask_shape: str = "ellipse",
    ssim_region: str = "full-landmarks",
    landmark_detector=None,
    max_samples: int = 0,
    offload_fr: bool = True,
) -> dict[str, float]:
    module.eval()
    all_cos: list[float] = []
    all_ssim: list[float] = []
    total = 0

    for batch in val_loader:
        if max_samples > 0 and total >= max_samples:
            break
        if len(batch) == 3:
            x, paths, _source_hw = batch
        else:
            x, paths = batch
        if max_samples > 0:
            remaining = max_samples - total
            if x.shape[0] > remaining:
                x = x[:remaining]
                paths = paths[:remaining]
        x = x.to(device)

        landmarks_batch = None
        need_lm = use_face_mask and (
            mask_mode == "landmarks" or ssim_region == "full-landmarks"
        )
        if need_lm:
            if landmark_detector is None:
                from models.face_detector import get_default_detector

                landmark_detector = get_default_detector()
            landmarks_batch = detect_landmarks_batch(list(paths), landmark_detector)

        face_mask = resolve_face_mask_by_size(
            x.shape[0],
            list(paths),
            device,
            image_size=image_size,
            use_face_mask=use_face_mask,
            mask_mode=mask_mode,
            mask_regions=mask_regions,
            mask_shape=mask_shape,
            detector=landmark_detector,
            landmarks_batch=landmarks_batch,
        )
        x_prot = module(x, face_mask=face_mask, roi_compose=roi_compose)

        cos_acc = 0.0
        for fr in embedders:
            if offload_fr and device.type == "cuda":
                fr.to(device)
            f0 = F.normalize(fr(x), dim=1)
            f1 = F.normalize(fr(x_prot), dim=1)
            cos_acc += float((f0 * f1).sum(dim=1).mean().cpu())
            if offload_fr and device.type == "cuda":
                fr.cpu()
        all_cos.append(cos_acc / max(len(embedders), 1))

        ssim_mask = resolve_ssim_mask_by_size(
            list(paths),
            device,
            image_size=image_size,
            ssim_region=ssim_region,
            face_mask=face_mask,
            use_face_mask=use_face_mask,
            landmarks_batch=landmarks_batch,
            detector=landmark_detector,
        )
        all_ssim.append(float(compute_ssim(x_prot, x, ssim_mask=ssim_mask).cpu()))
        total += x.shape[0]

    if not all_cos:
        return {
            "cos_mean": float("nan"),
            "euclid_mean": float("nan"),
            "ssim_mean": float("nan"),
            "n_images": 0,
        }
    mean_cos = sum(all_cos) / len(all_cos)
    mean_ssim = sum(all_ssim) / len(all_ssim)
    return {
        "cos_mean": mean_cos,
        "euclid_mean": math.sqrt(2.0 * (1.0 - mean_cos)),
        "ssim_mean": mean_ssim,
        "n_images": total,
    }


def early_stopping_score_v2(metrics: dict[str, float], metric_name: str) -> float:
    if metric_name == "score":
        return metrics["ssim_mean"] - metrics["cos_mean"]
    if metric_name == "cos":
        return metrics["cos_mean"]
    if metric_name == "ssim":
        return metrics["ssim_mean"]
    if metric_name == "euclid":
        return metrics["euclid_mean"]
    raise ValueError(f"Unknown early_stopping_metric: {metric_name}")


def is_improvement_v2(
    score: float,
    best: float,
    metric_name: str,
    min_delta: float,
) -> bool:
    # Minimize cos; maximize score/ssim/euclid.
    if metric_name == "cos":
        return score < (best - min_delta)
    return score > (best + min_delta)
