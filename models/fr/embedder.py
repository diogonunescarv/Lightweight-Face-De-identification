"""Frozen FR embedder wrappers (input ``[0,1]`` RGB ``(N,3,H,W)`` → features)."""

from __future__ import annotations

from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F


class FREmbedder(Protocol):
    name: str

    def to(self, device: torch.device | str) -> "FREmbedder": ...

    def cpu(self) -> "FREmbedder": ...

    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...


class FaceNetFR(nn.Module):
    """``inception_facenet_{vgg,casia}`` via facenet-pytorch (built-in, no FR_DB)."""

    def __init__(self, name: str, device: torch.device | str = "cpu") -> None:
        super().__init__()
        if name == "inception_facenet_vgg":
            pretrained = "vggface2"
        elif name == "inception_facenet_casia":
            pretrained = "casia-webface"
        else:
            raise ValueError(f"FaceNetFR não cobre {name!r}")

        from facenet_pytorch import InceptionResnetV1

        self.name = name
        self.device = torch.device(device)
        self.model = InceptionResnetV1(classify=False, pretrained=pretrained)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.to(self.device)

    def to(self, device: torch.device | str) -> "FaceNetFR":  # type: ignore[override]
        self.device = torch.device(device)
        self.model.to(self.device)
        return self

    def cpu(self) -> "FaceNetFR":  # type: ignore[override]
        return self.to("cpu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N,3,H,W) in [0,1] — differentiable resize + FaceNet norm.
        x = x.to(self.device)
        x = F.interpolate(x, size=(160, 160), mode="bilinear", align_corners=False)
        x = (x * 255.0 - 127.5) / 128.0
        return self.model(x)


class ProtegoFRAdapter(nn.Module):
    """Thin adapter around sibling ``FacialRecognition.FR`` (optional external FR_DB)."""

    def __init__(self, fr_obj: object, name: str, device: torch.device | str) -> None:
        super().__init__()
        self.name = name
        self.device = torch.device(device)
        self._fr = fr_obj
        # Freeze underlying torch modules if present.
        model = getattr(fr_obj, "fr_model", None)
        inner = getattr(model, "model", model)
        if isinstance(inner, nn.Module):
            inner.eval()
            for p in inner.parameters():
                p.requires_grad_(False)

    def to(self, device: torch.device | str) -> "ProtegoFRAdapter":  # type: ignore[override]
        self.device = torch.device(device)
        model = getattr(self._fr, "fr_model", None)
        inner = getattr(model, "model", model)
        if isinstance(inner, nn.Module):
            inner.to(self.device)
        if hasattr(self._fr, "device"):
            self._fr.device = self.device
        return self

    def cpu(self) -> "ProtegoFRAdapter":  # type: ignore[override]
        return self.to("cpu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Protego FR expects float RGB; preprocess handles [0,1] vs [0,255].
        return self._fr(x.to(self.device))


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)


def l2_normalize(feat: torch.Tensor) -> torch.Tensor:
    return F.normalize(feat, p=2, dim=-1)
