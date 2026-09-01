from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch

if TYPE_CHECKING:
    from models.face_detector import FaceDetector
    from models.transformer import LightweightDeIdentifier

UNITARY_REGIONS = frozenset({"eyes", "nose", "mouth", "full"})
COMPOSITE_UNION_REGIONS = frozenset({
    "eyes-nose",
    "eyes-mouth",
    "nose-mouth",
    "eyes-nose-mouth",
})
HYBRID_REGION = "eyes-nose-mouth-hybrid"
COMPOSITE_REGIONS = COMPOSITE_UNION_REGIONS | frozenset({HYBRID_REGION})
VALID_REGIONS = UNITARY_REGIONS | COMPOSITE_REGIONS
VALID_SHAPES = frozenset({"ellipse", "band"})
VALID_SSIM_REGIONS = frozenset({"full-landmarks", "mask"})

_COMPOSITE_TO_UNITARIES = {
    "eyes-nose": ("eyes", "nose"),
    "eyes-mouth": ("eyes", "mouth"),
    "nose-mouth": ("nose", "mouth"),
    "eyes-nose-mouth": ("eyes", "nose", "mouth"),
}

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

# Band unitárias (frações de IOD). Eyes: assimétrica (menos extensão inferior).
BAND_EYE_HW = 0.90
BAND_EYE_HH_UP = 0.42
BAND_EYE_HH_DOWN = 0.28

# Nariz ancorado entre olhos e boca (não half-height fixo centrado no tip).
BAND_NOSE_HW = 0.30
BAND_NOSE_TOP_FROM_EYES = 0.18  # abaixo do mid-olhos
BAND_NOSE_TOWARD_MOUTH = 0.28  # fração do gap nariz→boca (para de antes da boca)

BAND_MOUTH_HW = 0.55
BAND_MOUTH_HH = 0.28

# Híbrido: boca como elipse (base); olhos/nariz permanecem faixas retangulares.
BAND_HYBRID_MOUTH_RX = 0.55
BAND_HYBRID_MOUTH_RY = 0.28


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


def parse_mask_shape(spec: str) -> str:
    shape = spec.strip().lower()
    if shape not in VALID_SHAPES:
        raise ValueError(
            f"mask-shape desconhecido: {spec!r}. Válidos: {sorted(VALID_SHAPES)}"
        )
    return shape


def parse_ssim_region(spec: str) -> str:
    region = spec.strip().lower()
    if region not in VALID_SSIM_REGIONS:
        raise ValueError(
            f"ssim-region desconhecido: {spec!r}. Válidos: {sorted(VALID_SSIM_REGIONS)}"
        )
    return region


def resolve_ssim_region_for_eval(
    cli_value: str | None,
    checkpoint: dict,
) -> str:
    """Resolve escopo SSIM na avaliação (CLI > metadado > legado mask)."""
    if cli_value is not None:
        return parse_ssim_region(cli_value)
    if "ssim_region" in checkpoint:
        return parse_ssim_region(checkpoint["ssim_region"])
    print(
        "Aviso: checkpoint sem ssim_region; usando 'mask' (métricas legadas). "
        "Passe --ssim-region full-landmarks para o escopo novo."
    )
    return "mask"


def validate_mask_shape_regions(mask_shape: str, regions: tuple[str, ...]) -> None:
    shape = parse_mask_shape(mask_shape)
    composites = [r for r in regions if r in COMPOSITE_REGIONS]
    if composites and shape == "ellipse":
        joined = ",".join(composites)
        raise ValueError(
            f"Regiões compostas ({joined}) exigem --mask-shape band "
            "(ou use a lista unitária com vírgula, ex.: eyes,nose)."
        )
    if shape == "band" and "full" in regions:
        raise ValueError(
            "--mask-regions full não é suportado com --mask-shape band "
            "(use --mask-shape ellipse --mask-regions full, ou "
            "eyes-nose-mouth / eyes-nose-mouth-hybrid)."
        )


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


def _soft_rect_mask(
    height: int,
    width: int,
    center_x: float,
    center_y: float,
    half_w: float,
    half_h: float,
    device: torch.device,
    softness: float = MASK_SOFTNESS,
) -> torch.Tensor:
    return _soft_rect_mask_asymmetric(
        height,
        width,
        center_x,
        center_y,
        half_w,
        half_h,
        half_h,
        device,
        softness=softness,
    )


def _soft_rect_mask_asymmetric(
    height: int,
    width: int,
    center_x: float,
    center_y: float,
    half_w: float,
    half_h_up: float,
    half_h_down: float,
    device: torch.device,
    softness: float = MASK_SOFTNESS,
) -> torch.Tensor:
    hw = max(float(half_w), IOD_EPS)
    hu = max(float(half_h_up), IOD_EPS)
    hd = max(float(half_h_down), IOD_EPS)
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    ux = torch.abs(xx - center_x) / hw
    uy = torch.where(
        yy < center_y,
        (center_y - yy) / hu,
        (yy - center_y) / hd,
    )
    val = torch.maximum(ux, uy)
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


def _unitary_band_mask(
    landmarks: np.ndarray,
    region: str,
    iod: float,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    if region == "eyes":
        eye_center = (landmarks[0] + landmarks[1]) * 0.5
        return _soft_rect_mask_asymmetric(
            height,
            width,
            float(eye_center[0]),
            float(eye_center[1]),
            BAND_EYE_HW * iod,
            BAND_EYE_HH_UP * iod,
            BAND_EYE_HH_DOWN * iod,
            device,
        )
    if region == "nose":
        eye_mid = (landmarks[0] + landmarks[1]) * 0.5
        nose = landmarks[2]
        mouth_mid = (landmarks[3] + landmarks[4]) * 0.5
        top_y = float(eye_mid[1] + BAND_NOSE_TOP_FROM_EYES * iod)
        gap = max(float(mouth_mid[1] - nose[1]), IOD_EPS)
        bot_y = float(nose[1] + BAND_NOSE_TOWARD_MOUTH * gap)
        # Garante ordem e que o tip do nariz fique no interior.
        top_y = min(top_y, float(nose[1]) - IOD_EPS)
        bot_y = max(bot_y, float(nose[1]) + IOD_EPS)
        cy = 0.5 * (top_y + bot_y)
        return _soft_rect_mask_asymmetric(
            height,
            width,
            float(nose[0]),
            cy,
            BAND_NOSE_HW * iod,
            max(cy - top_y, IOD_EPS),
            max(bot_y - cy, IOD_EPS),
            device,
        )
    if region == "mouth":
        mouth_center = (landmarks[3] + landmarks[4]) * 0.5
        return _soft_rect_mask(
            height,
            width,
            float(mouth_center[0]),
            float(mouth_center[1]),
            BAND_MOUTH_HW * iod,
            BAND_MOUTH_HH * iod,
            device,
        )
    if region == "full":
        raise ValueError(
            "Região 'full' não possui máscara band (formato descartado). "
            "Use ellipse/full ou eyes-nose-mouth[-hybrid]."
        )
    raise ValueError(f"Região unitária de banda desconhecida: {region!r}")


def _hybrid_midface_band(
    landmarks: np.ndarray,
    iod: float,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    """Retângulos suaves em olhos+nariz; elipse suave na boca."""
    eyes = _unitary_band_mask(landmarks, "eyes", iod, height, width, device)
    nose = _unitary_band_mask(landmarks, "nose", iod, height, width, device)
    mouth_center = (landmarks[3] + landmarks[4]) * 0.5
    mouth = _ellipse_mask(
        height,
        width,
        float(mouth_center[0]),
        float(mouth_center[1]),
        BAND_HYBRID_MOUTH_RX * iod,
        BAND_HYBRID_MOUTH_RY * iod,
        device,
    )
    return torch.maximum(torch.maximum(eyes, nose), mouth)


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


def build_band_region_mask(
    landmarks: np.ndarray,
    regions: tuple[str, ...],
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    landmarks = np.asarray(landmarks, dtype=np.float32).reshape(5, 2)
    iod = _interocular_distance(landmarks)
    validate_mask_shape_regions("band", regions)

    composites = [r for r in regions if r in COMPOSITE_REGIONS]
    unitaries = [r for r in regions if r in UNITARY_REGIONS]

    if composites and unitaries:
        raise ValueError(
            "Não misture regiões compostas e unitárias no mesmo --mask-regions "
            f"(recebido: {regions})."
        )
    if len(composites) > 1:
        raise ValueError(
            "Use um único token composto por vez "
            f"(recebido: {composites})."
        )
    if len(unitaries) > 1:
        warnings.warn(
            f"--mask-shape band com múltiplas regiões unitárias {unitaries}: "
            "aplicando união (max) das faixas unitárias; "
            "prefira o token com hífen (ex.: eyes-nose).",
            stacklevel=2,
        )

    mask = torch.zeros(height, width, device=device, dtype=torch.float32)

    if composites:
        token = composites[0]
        if token == HYBRID_REGION:
            mask = _hybrid_midface_band(landmarks, iod, height, width, device)
        else:
            for part in _COMPOSITE_TO_UNITARIES[token]:
                band = _unitary_band_mask(landmarks, part, iod, height, width, device)
                mask = torch.max(mask, band)
    else:
        for region in unitaries:
            band = _unitary_band_mask(landmarks, region, iod, height, width, device)
            mask = torch.max(mask, band)

    return mask.unsqueeze(0).unsqueeze(0)


def build_batch_region_masks(
    landmarks_batch: Sequence[np.ndarray | None],
    regions: tuple[str, ...],
    height: int,
    width: int,
    device: torch.device,
    fallback_mask: torch.Tensor,
    mask_shape: str = "ellipse",
) -> torch.Tensor:
    shape = parse_mask_shape(mask_shape)
    validate_mask_shape_regions(shape, regions)
    builder = build_band_region_mask if shape == "band" else build_region_mask

    masks = []
    for landmarks in landmarks_batch:
        if landmarks is None:
            masks.append(fallback_mask.squeeze(0))
        else:
            masks.append(builder(landmarks, regions, height, width, device).squeeze(0))
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
    mask_shape: str = "ellipse",
    detector: FaceDetector | None = None,
    landmarks_batch: Sequence[np.ndarray | None] | None = None,
) -> torch.Tensor:
    h = w = transformer.image_size

    if not use_face_mask:
        return torch.ones(batch_size, 1, h, w, device=device)

    fallback = transformer.get_fixed_face_mask(device)

    if mask_mode == "fixed":
        return fallback.expand(batch_size, -1, -1, -1)

    if mask_mode != "landmarks":
        raise ValueError(f"mask_mode desconhecido: {mask_mode!r}")

    shape = parse_mask_shape(mask_shape)
    validate_mask_shape_regions(shape, mask_regions)

    if landmarks_batch is None:
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
        mask_shape=shape,
    )


def resolve_ssim_mask(
    transformer: LightweightDeIdentifier,
    paths: Sequence[str],
    device: torch.device,
    *,
    ssim_region: str,
    face_mask: torch.Tensor,
    mask_mode: str,
    mask_regions: tuple[str, ...],
    mask_shape: str,
    landmarks_batch: Sequence[np.ndarray | None] | None = None,
    detector: FaceDetector | None = None,
) -> torch.Tensor:
    region = parse_ssim_region(ssim_region)

    if not transformer.use_face_mask:
        return torch.ones_like(face_mask)

    if region == "mask":
        return face_mask

    h = w = transformer.image_size
    fallback = transformer.get_fixed_face_mask(device)

    if landmarks_batch is None:
        if detector is None:
            from models.face_detector import get_default_detector

            detector = get_default_detector()
        landmarks_batch = detect_landmarks_batch(paths, detector)

    return build_batch_region_masks(
        landmarks_batch,
        ("full",),
        h,
        w,
        device,
        fallback,
        mask_shape="ellipse",
    )


def resolve_face_mask_for_model(
    transformer: LightweightDeIdentifier,
    paths: Sequence[str],
    device: torch.device,
    *,
    mask_mode: str | None = None,
    mask_regions: str | None = None,
    mask_shape: str | None = None,
    detector: FaceDetector | None = None,
    landmarks_batch: Sequence[np.ndarray | None] | None = None,
) -> torch.Tensor:
    mode = mask_mode or transformer.mask_mode
    if mask_regions is not None:
        regions = parse_mask_regions(mask_regions)
    else:
        regions = transformer.mask_regions
    shape = mask_shape or getattr(transformer, "mask_shape", "ellipse")

    return resolve_face_mask(
        transformer,
        len(paths),
        paths,
        device,
        use_face_mask=transformer.use_face_mask,
        mask_mode=mode,
        mask_regions=regions,
        mask_shape=shape,
        detector=detector,
        landmarks_batch=landmarks_batch,
    )
