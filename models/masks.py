from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch

if TYPE_CHECKING:
    from models.face_detector import FaceDetector
    from models.transformer import LightweightDeIdentifier

VALID_REGIONS = frozenset({"eyes", "nose", "mouth", "full"})

IOD_EPS = 1.0
MASK_SOFTNESS = 4.0

EYE_RX = 0.32
EYE_RY = 0.22
NOSE_RX = 0.20
NOSE_RY = 0.26
MOUTH_RX = 0.38
MOUTH_RY = 0.18
FULL_RX = 0.90
FULL_RY = 1.20
FULL_CENTER_SHIFT = 0.40


def parse_mask_regions(spec: str) -> tuple[str, ...]:
    tokens = tuple(t.strip().lower() for t in spec.split(",") if t.strip())
    if not tokens:
        raise ValueError("mask-regions não pode ser vazio.")

    unknown = set(tokens) - VALID_REGIONS
    if unknown:
        raise ValueError(
            f"Regiões desconhecidas: {sorted(unknown)}. "
            f"Válidas: {sorted(VALID_REGIONS)}"
        )
    return tokens


def _interocular_distance(landmarks: np.ndarray) -> float:
    iod = float(np.linalg.norm(landmarks[1] - landmarks[0]))
    return max(iod, IOD_EPS)


def _ellipse_mask(
    height: int,
    width: int,
    center_x: float,
    center_y: float,
    rx: float,
    ry: float,
    device: torch.device,
    softness: float = MASK_SOFTNESS,
) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    dx = xx - center_x
    dy = yy - center_y
    val = (dx / rx) ** 2 + (dy / ry) ** 2
    mask = torch.exp(-softness * torch.clamp(val - 1.0, min=0.0))
    return mask.clamp(0.0, 1.0)


def _full_face_center(landmarks: np.ndarray) -> tuple[float, float]:
    eye_center = (landmarks[0] + landmarks[1]) * 0.5
    eye_to_nose = landmarks[2] - eye_center
    center = eye_center + FULL_CENTER_SHIFT * eye_to_nose
    return float(center[0]), float(center[1])


def _scale_full_axes_to_cover_landmarks(
    landmarks: np.ndarray,
    center_x: float,
    center_y: float,
    rx: float,
    ry: float,
) -> tuple[float, float]:
    scale = 1.0
    for x, y in landmarks:
        dx = (x - center_x) / max(rx, IOD_EPS)
        dy = (y - center_y) / max(ry, IOD_EPS)
        val = dx * dx + dy * dy
        if val > 1.1 * 1.1:
            scale = max(scale, (val ** 0.5) / 1.1)
    return rx * scale, ry * scale


def build_region_mask(
    landmarks: np.ndarray,
    regions: tuple[str, ...],
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    landmarks = np.asarray(landmarks, dtype=np.float32).reshape(5, 2)
    iod = _interocular_distance(landmarks)

    mask = torch.zeros(height, width, device=device, dtype=torch.float32)

    if "full" in regions:
        cx, cy = _full_face_center(landmarks)
        rx, ry = _scale_full_axes_to_cover_landmarks(
            landmarks, cx, cy, FULL_RX * iod, FULL_RY * iod
        )
        mask = torch.max(mask, _ellipse_mask(height, width, cx, cy, rx, ry, device))

    if "eyes" in regions:
        for idx in (0, 1):
            cx, cy = float(landmarks[idx, 0]), float(landmarks[idx, 1])
            eye = _ellipse_mask(height, width, cx, cy, EYE_RX * iod, EYE_RY * iod, device)
            mask = torch.max(mask, eye)

    if "nose" in regions:
        cx, cy = float(landmarks[2, 0]), float(landmarks[2, 1])
        nose = _ellipse_mask(height, width, cx, cy, NOSE_RX * iod, NOSE_RY * iod, device)
        mask = torch.max(mask, nose)

    if "mouth" in regions:
        mouth_center = (landmarks[3] + landmarks[4]) * 0.5
        cx, cy = float(mouth_center[0]), float(mouth_center[1])
        mouth = _ellipse_mask(height, width, cx, cy, MOUTH_RX * iod, MOUTH_RY * iod, device)
        mask = torch.max(mask, mouth)

    return mask.unsqueeze(0).unsqueeze(0)


def build_batch_region_masks(
    landmarks_batch: Sequence[np.ndarray | None],
    regions: tuple[str, ...],
    height: int,
    width: int,
    device: torch.device,
    fallback_mask: torch.Tensor,
) -> torch.Tensor:
    masks = []
    for landmarks in landmarks_batch:
        if landmarks is None:
            masks.append(fallback_mask.squeeze(0))
        else:
            masks.append(
                build_region_mask(landmarks, regions, height, width, device).squeeze(0)
            )
    return torch.stack(masks, dim=0)


def detect_landmarks_batch(
    paths: Sequence[str],
    detector: FaceDetector,
) -> list[np.ndarray | None]:
    results: list[np.ndarray | None] = []
    for path in paths:
        detection = detector.detect_best(path)
        if detection is None:
            results.append(None)
        else:
            results.append(detection.landmarks.astype(np.float32))
    return results


def resolve_face_mask(
    transformer: LightweightDeIdentifier,
    batch_size: int,
    paths: Sequence[str],
    device: torch.device,
    *,
    use_face_mask: bool,
    mask_mode: str,
    mask_regions: tuple[str, ...],
    detector: FaceDetector | None = None,
) -> torch.Tensor:
    h = w = transformer.image_size

    if not use_face_mask:
        return torch.ones(batch_size, 1, h, w, device=device)

    fallback = transformer.get_fixed_face_mask(device)

    if mask_mode == "fixed":
        return fallback.expand(batch_size, -1, -1, -1)

    if mask_mode != "landmarks":
        raise ValueError(f"mask_mode desconhecido: {mask_mode!r}")

    if detector is None:
        from models.face_detector import get_default_detector

        detector = get_default_detector()

    landmarks_batch = detect_landmarks_batch(paths, detector)
    return build_batch_region_masks(
        landmarks_batch,
        mask_regions,
        h,
        w,
        device,
        fallback,
    )


def resolve_face_mask_for_model(
    transformer: LightweightDeIdentifier,
    paths: Sequence[str],
    device: torch.device,
    *,
    mask_mode: str | None = None,
    mask_regions: str | None = None,
    detector: FaceDetector | None = None,
) -> torch.Tensor:
    mode = mask_mode or transformer.mask_mode
    if mask_regions is not None:
        regions = parse_mask_regions(mask_regions)
    else:
        regions = transformer.mask_regions

    return resolve_face_mask(
        transformer,
        len(paths),
        paths,
        device,
        use_face_mask=transformer.use_face_mask,
        mask_mode=mode,
        mask_regions=regions,
        detector=detector,
    )
