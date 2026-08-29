from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
from insightface.app import FaceAnalysis
from PIL import Image

ImageInput = Union[np.ndarray, Image.Image, str, Path]

# 0=left eye, 1=right eye, 2=nose, 3=left mouth, 4=right mouth
LANDMARK_NAMES = ("LE", "RE", "N", "LM", "RM")

DEFAULT_MODEL_PACK = "buffalo_m"
DEFAULT_DET_THRESH = 0.5


@dataclass(frozen=True)
class FaceDetection:
    bbox: np.ndarray       # (4,) float32 — [x1, y1, x2, y2]
    landmarks: np.ndarray  # (5, 2) float32
    score: float


def _resolve_onnx_providers() -> list[str]:
    if torch.cuda.is_available():
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _resolve_ctx_id(ctx_id: int | None) -> int:
    if ctx_id is not None:
        return ctx_id
    return 0 if torch.cuda.is_available() else -1


def _to_bgr_uint8(image: ImageInput) -> np.ndarray:
    if isinstance(image, (str, Path)):
        bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Não foi possível ler a imagem: {image}")
        return bgr

    if isinstance(image, Image.Image):
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if not isinstance(image, np.ndarray):
        raise TypeError(f"Tipo de imagem não suportado: {type(image)!r}")

    arr = image
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)

    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f"Esperado array HxWx3/4, recebido shape={arr.shape}")

    if arr.shape[2] == 4:
        arr = arr[:, :, :3]

    # Heurística simples: se canal 0 parece vermelho dominante, assume RGB.
    if arr[0, 0, 0] > arr[0, 0, 2]:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr


class FaceDetector:
    def __init__(
        self,
        model_pack: str = DEFAULT_MODEL_PACK,
        det_thresh: float = DEFAULT_DET_THRESH,
        ctx_id: int | None = None,
        providers: list[str] | None = None,
    ):
        self.model_pack = model_pack
        self.det_thresh = det_thresh
        self.ctx_id = _resolve_ctx_id(ctx_id)
        self.providers = providers or _resolve_onnx_providers()

        try:
            self.app = FaceAnalysis(
                name=model_pack,
                allowed_modules=["detection"],
                providers=self.providers,
            )
        except AssertionError as exc:
            raise RuntimeError(
                f"Modelo '{model_pack}' não encontrado em ~/.insightface/models/. "
                f"Baixe o pack manualmente ou rode: "
                f"python -c \"from insightface.utils import ensure_available; "
                f"ensure_available('models', '{model_pack}')\""
            ) from exc

        self.app.prepare(ctx_id=self.ctx_id, det_thresh=det_thresh)

    def detect(self, image: ImageInput, max_faces: int = 0) -> list[FaceDetection]:
        bgr = _to_bgr_uint8(image)
        faces = self.app.get(bgr, max_num=max_faces)

        detections: list[FaceDetection] = []
        for face in faces:
            if face.kps is None:
                continue
            detections.append(
                FaceDetection(
                    bbox=face.bbox.astype(np.float32),
                    landmarks=face.kps.astype(np.float32),
                    score=float(face.det_score),
                )
            )
        return detections

    def detect_best(self, image: ImageInput) -> FaceDetection | None:
        detections = self.detect(image)
        if not detections:
            return None
        return max(detections, key=lambda d: d.score)


_default_detector: FaceDetector | None = None


def get_default_detector(**kwargs) -> FaceDetector:
    global _default_detector
    if _default_detector is None:
        _default_detector = FaceDetector(**kwargs)
    return _default_detector


def detect_face(
    image: ImageInput,
    *,
    detector: FaceDetector | None = None,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """
    Detecta a face de maior score.

    Returns:
        (bbox, landmarks_5pts, score) ou None se nenhuma face for encontrada.
    """
    det = detector or get_default_detector()
    result = det.detect_best(image)
    if result is None:
        return None
    return result.bbox, result.landmarks, result.score
