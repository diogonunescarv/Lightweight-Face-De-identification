"""
V2 visualization helpers: orig / protected / residual / ROI overlays.

Writes only under ``out_dir`` (typically ``<run>/viz/``). Notebooks and CLI
scripts should call these helpers rather than duplicating forward logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from evaluation.evaluate_v2 import load_meta_mask_from_checkpoint
from models.dtcwt_meta_mask import ROI_COMPOSE_PIXEL, ROI_COMPOSE_WAVELET
from models.masks import (
    parse_mask_regions,
    parse_mask_shape,
    resolve_face_mask_by_size,
)


def _tensor_to_uint8(x: torch.Tensor) -> np.ndarray:
    """``[C,H,W]`` or ``[H,W]`` float [0,1] → uint8 HWC/HW."""
    t = x.detach().float().cpu().clamp(0, 1)
    if t.ndim == 3:
        arr = (t.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    else:
        arr = (t.numpy() * 255.0).round().astype(np.uint8)
    return arr


def _save_rgb(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.ndim == 2:
        Image.fromarray(arr, mode="L").save(path)
    else:
        Image.fromarray(arr, mode="RGB").save(path)


def _residual_heatmap(residual: torch.Tensor) -> np.ndarray:
    """Map |residual| mean over channels to a simple red overlay base (uint8 RGB)."""
    mag = residual.detach().float().abs().mean(dim=0).cpu()
    # Scale relative to typical L∞ budget (~16/255) for visibility.
    scaled = (mag / (16.0 / 255.0)).clamp(0, 1)
    h, w = scaled.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[..., 0] = (scaled.numpy() * 255.0).round().astype(np.uint8)
    return rgb


def _overlay_mask(rgb: np.ndarray, mask: torch.Tensor, alpha: float = 0.35) -> np.ndarray:
    m = mask.detach().float().cpu()
    if m.ndim == 3:
        m = m[0]
    m_np = m.numpy().clip(0, 1)
    out = rgb.astype(np.float32).copy()
    green = np.zeros_like(out)
    green[..., 1] = 255.0
    a = (m_np * alpha)[..., None]
    out = out * (1.0 - a) + green * a
    return out.round().clip(0, 255).astype(np.uint8)


def load_image_tensor(path: str | Path, image_size: int) -> torch.Tensor:
    """Load RGB image as ``[1,3,H,W]`` float in [0,1], resized to ``image_size``."""
    img = Image.open(path).convert("RGB")
    img = img.resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t


@torch.no_grad()
def render_meta_mask_panel(
    checkpoint_path: str | Path,
    image_path: str | Path,
    out_dir: str | Path,
    *,
    device: str = "cpu",
    roi_compose: Optional[str] = None,
    compare_compose: bool = True,
    no_face_mask: bool = False,
    mask_mode: Optional[str] = None,
    mask_shape: Optional[str] = None,
    mask_regions: Optional[str] = None,
    stem: Optional[str] = None,
) -> dict[str, Any]:
    """
    Save visualization panels under ``out_dir`` (created if needed).

    Files (prefix = stem or image stem):
      ``{stem}_orig.png``, ``{stem}_prot_{compose}.png``, ``{stem}_resid_{compose}.png``,
      ``{stem}_mask.png``, ``{stem}_panel.png`` (side-by-side).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    module, meta = load_meta_mask_from_checkpoint(
        checkpoint_path, device=device, roi_compose_override=roi_compose
    )
    image_size = int(meta["image_size"])
    compose = str(meta["roi_compose"])
    x = load_image_tensor(image_path, image_size).to(device)
    paths_list = [str(Path(image_path).resolve())]

    use_face_mask = not no_face_mask and bool(meta.get("use_face_mask", True))
    mm = mask_mode or meta.get("mask_mode") or "fixed"
    # Detectors may be unavailable in smoke; prefer fixed ellipse for viz.
    if mm == "landmarks":
        mm = "fixed"
    ms = parse_mask_shape(mask_shape or meta.get("mask_shape") or "ellipse")
    mr = parse_mask_regions(
        mask_regions
        if mask_regions is not None
        else ",".join(meta.get("mask_regions") or ["full"])
    )
    face_mask = resolve_face_mask_by_size(
        1,
        paths_list,
        torch.device(device),
        image_size=image_size,
        use_face_mask=use_face_mask,
        mask_mode=mm,
        mask_regions=mr,
        mask_shape=ms if ms != "band" else "ellipse",
    ).to(device)

    prefix = stem or Path(image_path).stem
    saved: dict[str, str] = {}

    orig_u8 = _tensor_to_uint8(x[0])
    _save_rgb(out / f"{prefix}_orig.png", orig_u8)
    saved["orig"] = str(out / f"{prefix}_orig.png")

    mask_u8 = _tensor_to_uint8(face_mask[0, 0])
    _save_rgb(out / f"{prefix}_mask.png", mask_u8)
    saved["mask"] = str(out / f"{prefix}_mask.png")
    _save_rgb(out / f"{prefix}_mask_overlay.png", _overlay_mask(orig_u8, face_mask[0]))
    saved["mask_overlay"] = str(out / f"{prefix}_mask_overlay.png")

    composes: list[str] = [compose]
    if compare_compose:
        other = (
            ROI_COMPOSE_WAVELET
            if compose == ROI_COMPOSE_PIXEL
            else ROI_COMPOSE_PIXEL
        )
        if other not in composes:
            composes.append(other)

    panels: list[np.ndarray] = [orig_u8]
    for rc in composes:
        x_prot = module(x, face_mask=face_mask, roi_compose=rc)
        resid = (x_prot - x).squeeze(0)
        prot_u8 = _tensor_to_uint8(x_prot[0])
        heat = _residual_heatmap(resid)
        _save_rgb(out / f"{prefix}_prot_{rc}.png", prot_u8)
        _save_rgb(out / f"{prefix}_resid_{rc}.png", heat)
        saved[f"prot_{rc}"] = str(out / f"{prefix}_prot_{rc}.png")
        saved[f"resid_{rc}"] = str(out / f"{prefix}_resid_{rc}.png")
        panels.append(prot_u8)
        panels.append(heat)

    # Side-by-side panel: orig | prot_a | resid_a [| prot_b | resid_b]
    h = panels[0].shape[0]
    gap = np.full((h, 4, 3), 255, dtype=np.uint8)
    row = panels[0]
    for p in panels[1:]:
        row = np.concatenate([row, gap, p], axis=1)
    panel_path = out / f"{prefix}_panel.png"
    _save_rgb(panel_path, row)
    saved["panel"] = str(panel_path)

    # Mag of ΔW for first editable level (diagnostic).
    try:
        level = int(module.editable_levels[0])
        dm = module.delta_mag[str(level)]
        mag = dm.detach().float().abs().mean(dim=(0, 1))  # → [6,H,W] or less
        while mag.ndim > 2:
            mag = mag.mean(dim=0)
        mag = mag - mag.min()
        if float(mag.max()) > 0:
            mag = mag / mag.max()
        mag_t = mag.unsqueeze(0).unsqueeze(0)
        mag_t = F.interpolate(
            mag_t, size=(image_size, image_size), mode="bilinear", align_corners=False
        )[0, 0]
        heat_dw = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        heat_dw[..., 2] = (mag_t.cpu().numpy() * 255.0).round().astype(np.uint8)
        dw_path = out / f"{prefix}_delta_mag_L{level}.png"
        _save_rgb(dw_path, heat_dw)
        saved[f"delta_mag_L{level}"] = str(dw_path)
    except Exception:
        pass

    manifest = {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "image": str(Path(image_path).resolve()),
        "out_dir": str(out.resolve()),
        "roi_compose_ckpt": compose,
        "composes_rendered": composes,
        "files": saved,
        "image_size": image_size,
    }
    (out / f"{prefix}_manifest.json").write_text(
        __import__("json").dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def visualize_batch(
    checkpoint_path: str | Path,
    image_paths: Sequence[str | Path],
    out_dir: str | Path,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Render panels for several images; all under ``out_dir``."""
    results = []
    for p in image_paths:
        results.append(
            render_meta_mask_panel(checkpoint_path, p, out_dir, stem=Path(p).stem, **kwargs)
        )
    return results
