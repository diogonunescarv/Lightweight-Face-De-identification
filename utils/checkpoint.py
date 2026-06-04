import torch
from pathlib import Path
from models.transformer import LightweightDeIdentifier

def load_transformer_from_checkpoint(path: str | Path, device: torch.device) -> LightweightDeIdentifier:
    ckpt = torch.load(path, map_location=device)

    transformer = LightweightDeIdentifier(
        image_size=ckpt["image_size"],
        dct_k=ckpt["dct_k"],
        dct_fmin=ckpt["dct_fmin"],
        dct_fmax=ckpt["dct_fmax"],
        flow_grid=ckpt["flow_grid"],
        photo_grid=ckpt["photo_grid"],
        max_dct_amp=ckpt["max_dct_amp"],
        max_flow_px=ckpt["max_flow_px"],
        max_photo_amp=ckpt["max_photo_amp"],
        use_face_mask=ckpt["use_face_mask"],
        disable_dct=ckpt["disable_dct"],
        disable_flow=ckpt["disable_flow"],
        disable_photo=ckpt["disable_photo"],
    ).to(device)

    transformer.load_state_dict(ckpt["state_dict"])
    transformer.eval()

    return transformer