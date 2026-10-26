import torch
from pathlib import Path
from models.transformer import LightweightDeIdentifier
from models.masks import parse_mask_regions, parse_mask_shape

def load_transformer_from_checkpoint(path: str | Path, device: torch.device) -> LightweightDeIdentifier:
    ckpt = torch.load(path, map_location=device)

    mask_regions = ckpt.get("mask_regions", "full")
    if isinstance(mask_regions, str):
        mask_regions = parse_mask_regions(mask_regions)

    mask_shape = parse_mask_shape(ckpt.get("mask_shape", "ellipse"))

    max_wavelet_amp = ckpt.get("max_wavelet_amp", 0.2)
    max_wavelet_mag_amp = ckpt.get("max_wavelet_mag_amp", max_wavelet_amp)
    max_wavelet_phase_amp = ckpt.get("max_wavelet_phase_amp", max_wavelet_amp)

    # Old checkpoints without polar mode metadata reproduce legacy behavior.
    wavelet_polar_mode = ckpt.get("wavelet_polar_mode", "legacy")
    wavelet_per_channel = bool(ckpt.get("wavelet_per_channel", False))
    wavelet_spatial_grid = int(ckpt.get("wavelet_spatial_grid", 0))
    wavelet_yl_gain = bool(ckpt.get("wavelet_yl_gain", False))

    # Prefer state_dict shapes when present (authoritative for expressivity modes).
    sd = ckpt.get("state_dict", {})
    mag = sd.get("raw_wavelet_mag_scale")
    if mag is not None:
        if mag.dim() == 5:
            wavelet_spatial_grid = int(mag.shape[-1])
            wavelet_per_channel = False
        elif mag.dim() == 4:
            wavelet_per_channel = True
            wavelet_spatial_grid = 0
        elif mag.dim() == 3:
            wavelet_per_channel = False
            wavelet_spatial_grid = 0
        else:
            raise RuntimeError(
                f"Unsupported raw_wavelet_mag_scale rank {mag.dim()} in checkpoint {path}"
            )
    if "raw_wavelet_yl_gain" in sd:
        wavelet_yl_gain = True

    # Legacy mode cannot combine with expressivity flags.
    if wavelet_polar_mode == "legacy" and (
        wavelet_per_channel or wavelet_spatial_grid > 0 or wavelet_yl_gain
    ):
        raise RuntimeError(
            f"Checkpoint {path} requests legacy polar mode with expressivity flags; "
            "refuse to load silently. Re-save with wavelet_polar_mode=complex or "
            "disable per-channel/spatial/Yl features."
        )

    level_scales = ckpt.get("wavelet_level_amp_scales")
    if isinstance(level_scales, list):
        level_scales = tuple(level_scales)

    transformer = LightweightDeIdentifier(
        image_size=ckpt["image_size"],
        transform_type=ckpt.get("transform_type", "dct"),
        dct_k=ckpt["dct_k"],
        dct_fmin=ckpt["dct_fmin"],
        dct_fmax=ckpt["dct_fmax"],
        wavelet_J=ckpt.get("wavelet_J", 3),
        max_wavelet_amp=max_wavelet_amp,
        max_wavelet_mag_amp=max_wavelet_mag_amp,
        max_wavelet_phase_amp=max_wavelet_phase_amp,
        wavelet_level_amp_scales=level_scales,
        wavelet_polar_mode=wavelet_polar_mode,
        wavelet_per_channel=wavelet_per_channel,
        wavelet_spatial_grid=wavelet_spatial_grid,
        wavelet_yl_gain=wavelet_yl_gain,
        max_wavelet_yl_amp=ckpt.get("max_wavelet_yl_amp", 0.05),
        flow_grid=ckpt["flow_grid"],
        photo_grid=ckpt["photo_grid"],
        max_dct_amp=ckpt["max_dct_amp"],
        max_flow_px=ckpt["max_flow_px"],
        max_photo_amp=ckpt["max_photo_amp"],
        use_face_mask=ckpt["use_face_mask"],
        mask_mode=ckpt.get("mask_mode", "fixed"),
        mask_regions=mask_regions,
        mask_shape=mask_shape,
        disable_dct=ckpt["disable_dct"],
        disable_flow=ckpt["disable_flow"],
        disable_photo=ckpt["disable_photo"],
    ).to(device)

    try:
        transformer.load_state_dict(ckpt["state_dict"])
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint wavelet/model shapes do not match. "
            "Check wavelet_per_channel / wavelet_spatial_grid / wavelet_yl_gain."
        ) from exc

    transformer.eval()
    return transformer
