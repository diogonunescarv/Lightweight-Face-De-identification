from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F

from losses.losses import ssim_index_masked
from models.masks import detect_landmarks_batch, resolve_face_mask, resolve_ssim_mask


@torch.no_grad()
def compute_validation_metrics(
    transformer,
    embedders: Sequence[torch.nn.Module],
    val_loader,
    device: torch.device,
    *,
    mask_mode: str,
    mask_regions: tuple[str, ...],
    mask_shape: str,
    ssim_region: str,
    use_face_mask: bool,
    landmark_detector=None,
    max_samples: int = 0,
) -> dict[str, float]:
    transformer.eval()

    all_cos: list[torch.Tensor] = []
    all_ssim: list[torch.Tensor] = []
    total_images = 0

    for x, paths in val_loader:
        if max_samples > 0 and total_images >= max_samples:
            break

        if max_samples > 0:
            remaining = max_samples - total_images
            if x.shape[0] > remaining:
                x = x[:remaining]
                paths = paths[:remaining]

        x = x.to(device)

        landmarks_batch = None
        need_landmarks = use_face_mask and (
            mask_mode == "landmarks" or ssim_region == "full-landmarks"
        )
        if need_landmarks:
            if landmark_detector is None:
                from models.face_detector import get_default_detector

                landmark_detector = get_default_detector()
            landmarks_batch = detect_landmarks_batch(paths, landmark_detector)

        face_mask = resolve_face_mask(
            transformer,
            x.shape[0],
            paths,
            device,
            use_face_mask=use_face_mask,
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
            batch_cos.append((e0 * e1).sum(dim=1))
        all_cos.append(torch.stack(batch_cos).mean(dim=0))

        ssim_mask = resolve_ssim_mask(
            transformer,
            paths,
            device,
            ssim_region=ssim_region,
            face_mask=face_mask,
            mask_mode=mask_mode,
            mask_regions=mask_regions,
            mask_shape=mask_shape,
            landmarks_batch=landmarks_batch,
            detector=landmark_detector,
        )
        all_ssim.append(ssim_index_masked(x, y, ssim_mask))
        total_images += x.shape[0]

    if not all_cos:
        return {
            "cos_mean": float("nan"),
            "euclid_mean": float("nan"),
            "ssim_mean": float("nan"),
            "n_images": 0,
        }

    cos_all = torch.cat(all_cos)
    ssim_all = torch.cat(all_ssim)
    mean_cos = cos_all.mean().item()
    mean_ssim = ssim_all.mean().item()

    return {
        "cos_mean": mean_cos,
        "euclid_mean": math.sqrt(2.0 * (1.0 - mean_cos)),
        "ssim_mean": mean_ssim,
        "n_images": total_images,
    }


def early_stopping_score(
    metrics: dict[str, float],
    metric_name: str,
) -> float:
    if metric_name == "score":
        return metrics["ssim_mean"] - metrics["cos_mean"]
    if metric_name == "cos":
        return metrics["cos_mean"]
    if metric_name == "ssim":
        return metrics["ssim_mean"]
    if metric_name == "euclid":
        return metrics["euclid_mean"]
    raise ValueError(f"Métrica de early stopping desconhecida: {metric_name!r}")


def is_improvement(
    current: float,
    best: float,
    *,
    metric_name: str,
    min_delta: float,
) -> bool:
    if metric_name == "cos":
        return current < best - min_delta
    return current > best + min_delta
