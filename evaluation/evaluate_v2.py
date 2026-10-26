"""V2 evaluation: cos / SSIM / optional LPIPS; artifacts stay under the run folder."""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.merged_split import PathListFaceDataset, list_image_paths
from losses.losses import ssim_index, ssim_index_masked
from models.dtcwt_meta_mask import DEFAULT_EPSILON, DTCWTMetaMaskModule
from models.fr import SMOKE_FR_NAMES, load_fr_ensemble
from models.masks import (
    detect_landmarks_batch,
    parse_mask_regions,
    parse_mask_shape,
    parse_ssim_region,
    resolve_face_mask_by_size,
    resolve_ssim_mask_by_size,
)


def _try_lpips(device: torch.device):
    try:
        import lpips  # type: ignore

        net = lpips.LPIPS(net="alex").to(device)
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        return net
    except Exception:
        return None


def load_meta_mask_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
    *,
    roi_compose_override: Optional[str] = None,
) -> tuple[DTCWTMetaMaskModule, dict[str, Any]]:
    """Load ``meta_mask.pt`` / ``meta_mask_best.pt`` / ``checkpoint_latest.pt``."""
    path = Path(checkpoint_path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(ckpt.get("config") or {})
    image_size = int(ckpt.get("img_size") or cfg.get("image_size") or 224)
    J = int(ckpt.get("J") or cfg.get("J") or 3)
    editable = ckpt.get("editable_levels") or cfg.get("editable_levels") or (2, 3)
    epsilon = float(cfg.get("epsilon", DEFAULT_EPSILON))
    roi_compose = (
        roi_compose_override
        or ckpt.get("roi_compose")
        or cfg.get("roi_compose")
        or "pixel"
    )
    if roi_compose_override and roi_compose_override != (
        ckpt.get("roi_compose") or cfg.get("roi_compose")
    ):
        print(
            f"[evaluate-v2] roi_compose override CLI={roi_compose_override!r} "
            f"(checkpoint had {ckpt.get('roi_compose') or cfg.get('roi_compose')!r})"
        )

    module = DTCWTMetaMaskModule(
        channels=3,
        img_size=image_size,
        J=J,
        editable_levels=editable,
        init_std=0.0,
        epsilon=epsilon,
    )
    module.load_meta_mask_state(ckpt["meta_mask"])
    module.to(device).eval()

    meta = {
        "checkpoint": str(path.resolve()),
        "image_size": image_size,
        "J": J,
        "editable_levels": list(editable),
        "epsilon": epsilon,
        "roi_compose": roi_compose,
        "mask_mode": ckpt.get("mask_mode", cfg.get("mask_mode", "landmarks")),
        "mask_shape": ckpt.get("mask_shape", cfg.get("mask_shape", "ellipse")),
        "mask_regions": ckpt.get("mask_regions", cfg.get("mask_regions", ["full"])),
        "ssim_region": ckpt.get("ssim_region", cfg.get("ssim_region", "full-landmarks")),
        "use_face_mask": bool(
            ckpt.get("use_face_mask", cfg.get("use_face_mask", True))
        ),
        "config": cfg,
    }
    return module, meta


@torch.no_grad()
def evaluate_v2(
    checkpoint_path: str,
    data_dir: str | None = None,
    *,
    image_paths: Sequence[str] | None = None,
    run_dir: str | None = None,
    device: str = "cpu",
    batch_size: int = 4,
    num_workers: int = 0,
    max_samples: int = 0,
    fr_models: str | None = None,
    smoke_fr: bool = False,
    roi_compose: str | None = None,
    mask_mode: str | None = None,
    mask_shape: str | None = None,
    mask_regions: str | None = None,
    ssim_region: str | None = None,
    no_face_mask: bool | None = None,
    compute_lpips: bool = True,
) -> dict[str, Any]:
    """
    Evaluate a V2 meta-mask checkpoint.

    Writes under ``<run_dir>/eval/`` (run_dir defaults to checkpoint parent).
    """
    device_t = torch.device(device)
    module, meta = load_meta_mask_from_checkpoint(
        checkpoint_path, device_t, roi_compose_override=roi_compose
    )
    image_size = meta["image_size"]
    roi = meta["roi_compose"]

    use_face_mask = meta["use_face_mask"] if no_face_mask is None else (not no_face_mask)
    mode = mask_mode or meta["mask_mode"]
    shape = parse_mask_shape(mask_shape or meta["mask_shape"])
    if mask_regions is not None:
        regions = parse_mask_regions(mask_regions)
    else:
        raw = meta["mask_regions"]
        regions = parse_mask_regions(",".join(raw) if isinstance(raw, list) else str(raw))
    ssim_reg = parse_ssim_region(ssim_region or meta["ssim_region"])

    out_dir = Path(run_dir) if run_dir else Path(checkpoint_path).resolve().parent
    eval_dir = out_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    if image_paths is None:
        if not data_dir:
            raise ValueError("Provide data_dir or image_paths")
        image_paths = list_image_paths(data_dir)
    paths = list(image_paths)
    if max_samples and max_samples > 0:
        paths = paths[: int(max_samples)]

    ds = PathListFaceDataset(paths, image_size=image_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    fr_csv = fr_models
    if smoke_fr or fr_csv is None:
        fr_csv = ",".join(SMOKE_FR_NAMES)
    frs = load_fr_ensemble(fr_csv.split(","), device=device_t, keep_on_device=True)

    lpips_net = _try_lpips(device_t) if compute_lpips else None

    landmark_detector = None
    need_lm = use_face_mask and (mode == "landmarks" or ssim_reg == "full-landmarks")
    if need_lm:
        try:
            from models.face_detector import get_default_detector

            landmark_detector = get_default_detector()
        except Exception as exc:
            print(f"[evaluate-v2] detector unavailable ({exc}); fixed ellipse fallback")
            mode = "fixed"

    rows: list[dict[str, Any]] = []
    cos_all: list[float] = []
    ssim_all: list[float] = []
    lpips_all: list[float] = []
    times_ms: list[float] = []

    for batch in loader:
        x, batch_paths, _hw = batch
        x = x.to(device_t)
        batch_paths = list(batch_paths)

        landmarks_batch = None
        if need_lm and landmark_detector is not None:
            landmarks_batch = detect_landmarks_batch(batch_paths, landmark_detector)

        face_mask = resolve_face_mask_by_size(
            x.shape[0],
            batch_paths,
            device_t,
            image_size=image_size,
            use_face_mask=use_face_mask,
            mask_mode=mode,
            mask_regions=regions,
            mask_shape=shape,
            detector=landmark_detector,
            landmarks_batch=landmarks_batch,
        )

        t0 = time.perf_counter()
        x_prot = module(x, face_mask=face_mask, roi_compose=roi)
        dt = (time.perf_counter() - t0) * 1000.0 / max(x.shape[0], 1)

        ssim_mask = resolve_ssim_mask_by_size(
            batch_paths,
            device_t,
            image_size=image_size,
            ssim_region=ssim_reg,
            face_mask=face_mask,
            use_face_mask=use_face_mask,
            landmarks_batch=landmarks_batch,
            detector=landmark_detector,
        )
        if float(ssim_mask.mean().cpu()) >= 1.0 - 1e-6:
            ssim_b = ssim_index(x, x_prot).detach().cpu()
        else:
            ssim_b = ssim_index_masked(x, x_prot, ssim_mask).detach().cpu()

        cos_acc = 0.0
        for fr in frs:
            f0 = F.normalize(fr(x), dim=1)
            f1 = F.normalize(fr(x_prot), dim=1)
            cos_acc += (f0 * f1).sum(dim=1)
        cos_b = (cos_acc / max(len(frs), 1)).detach().cpu()

        lpips_b = None
        if lpips_net is not None:
            # LPIPS expects [-1,1]
            lpips_b = (
                lpips_net(x * 2 - 1, x_prot * 2 - 1).view(-1).detach().cpu()
            )

        for i, p in enumerate(batch_paths):
            cos_i = float(cos_b[i])
            ssim_i = float(ssim_b[i])
            euclid_i = math.sqrt(2.0 * max(0.0, 1.0 - cos_i))
            row = {
                "path": p,
                "cos": cos_i,
                "euclid": euclid_i,
                "ssim": ssim_i,
                "time_ms": dt,
                "roi_compose": roi,
                "image_size": image_size,
            }
            if lpips_b is not None:
                row["lpips"] = float(lpips_b[i])
                lpips_all.append(row["lpips"])
            rows.append(row)
            cos_all.append(cos_i)
            ssim_all.append(ssim_i)
            times_ms.append(dt)

    def _mean(xs: list[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else float("nan")

    summary = {
        "n_images": len(rows),
        "cos_mean": _mean(cos_all),
        "euclid_mean": math.sqrt(2.0 * max(0.0, 1.0 - _mean(cos_all))) if cos_all else float("nan"),
        "ssim_mean": _mean(ssim_all),
        "time_ms_mean": _mean(times_ms),
        "lpips_mean": _mean(lpips_all) if lpips_all else None,
        "lpips_available": lpips_net is not None,
        "roi_compose": roi,
        "image_size": image_size,
        "mask_mode": mode,
        "mask_shape": shape,
        "mask_regions": list(regions),
        "ssim_region": ssim_reg,
        "use_face_mask": use_face_mask,
        "fr_models": fr_csv,
        "checkpoint": meta["checkpoint"],
        "batch_hw": f"{image_size}x{image_size}",
    }

    metrics_csv = eval_dir / "eval_metrics.csv"
    fieldnames = list(rows[0].keys()) if rows else [
        "path",
        "cos",
        "euclid",
        "ssim",
        "time_ms",
        "roi_compose",
        "image_size",
    ]
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    summary_path = eval_dir / "eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"[evaluate-v2] n={summary['n_images']} cos={summary['cos_mean']:.4f} "
        f"ssim={summary['ssim_mean']:.4f} lpips={summary['lpips_mean']} "
        f"roi={roi} → {eval_dir}"
    )
    return summary
