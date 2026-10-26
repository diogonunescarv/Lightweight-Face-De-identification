"""
V2 differentiable DT-CWT spatial meta-mask (Phase 3.3 parity + ROI compose).

``roi_compose="pixel"`` (Ideia 1 — default):
  x → DT-CWT → yh[ℓ] += ΔW_ℓ · e^{iφ} (editable levels, full-frame)
    → IDT-CWT → residual=(recon−x)⊙M → L∞(ε) → x_prot

``roi_compose="wavelet"`` (Ideia 2):
  x → DT-CWT → yh[ℓ] += (ΔW_ℓ ⊙ M_ℓ) · e^{iφ} → IDT-CWT → L∞(ε) full-frame → x_prot
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from pytorch_wavelets import DTCWTForward, DTCWTInverse
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "pytorch_wavelets is required for V2 DT-CWT meta-mask. "
        "Install with: pip install pytorch-wavelets PyWavelets"
    ) from exc

DEFAULT_EDITABLE_LEVELS: Tuple[int, ...] = (2, 3)
DEFAULT_EPSILON: float = 16.0 / 255.0
ROI_COMPOSE_PIXEL = "pixel"
ROI_COMPOSE_WAVELET = "wavelet"
VALID_ROI_COMPOSE = frozenset({ROI_COMPOSE_PIXEL, ROI_COMPOSE_WAVELET})


def parse_editable_levels(spec: str | Sequence[int], J: int) -> Tuple[int, ...]:
    """Parse ``'2,3'`` or a sequence into a validated non-empty 1-based level tuple."""
    if isinstance(spec, str):
        parts = [p.strip() for p in spec.split(",") if p.strip()]
        if not parts:
            raise ValueError("editable_levels não pode ser vazio.")
        levels = tuple(int(p) for p in parts)
    else:
        levels = tuple(int(l) for l in spec)
    if not levels:
        raise ValueError("editable_levels não pode ser vazio.")
    if len(set(levels)) != len(levels):
        raise ValueError(f"editable_levels com duplicatas: {levels}")
    if any(l < 1 or l > J for l in levels):
        raise ValueError(f"editable_levels deve estar em {{1..{J}}}, got {levels}")
    if 1 in levels:
        warnings.warn(
            "editable_levels inclui nível 1 (alta frequência / sensível a JPEG). "
            "Use apenas para ablação; default recomendado: 2,3.",
            UserWarning,
            stacklevel=2,
        )
    return levels


def _spatial_size_for_level(img_size: int, level: int) -> Tuple[int, int]:
    s = img_size // (2**level)
    if s < 1:
        raise ValueError(
            f"img_size={img_size} too small for DT-CWT level {level} (got spatial {s})."
        )
    return s, s


def coeffs_to_mag_phase(
    band: torch.Tensor, eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert real/imag band ``(..., 2)`` to magnitude and phase."""
    real = band[..., 0]
    imag = band[..., 1]
    mag = torch.sqrt(real * real + imag * imag + eps)
    phase = torch.atan2(imag, real)
    return mag, phase


def mag_phase_to_coeffs(mag: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """Convert magnitude/phase back to real/imag band ``(..., 2)``."""
    return torch.stack([mag * torch.cos(phase), mag * torch.sin(phase)], dim=-1)


def polar_to_real_imag(
    mag: torch.Tensor, phase: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    return mag * torch.cos(phase), mag * torch.sin(phase)


def rotate_real_imag(
    real: torch.Tensor, imag: torch.Tensor, phi: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    c, s = torch.cos(phi), torch.sin(phi)
    return real * c - imag * s, real * s + imag * c


def add_complex_meta_mask(
    band: torch.Tensor,
    delta_mag: torch.Tensor,
    delta_phase: torch.Tensor,
    phi: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply ``W ← W + ΔW · e^{iφ}`` on one high-pass band (real/imag last dim)."""
    dw_r, dw_i = polar_to_real_imag(delta_mag, delta_phase)
    if phi is not None:
        while phi.ndim < dw_r.ndim:
            phi = phi.unsqueeze(-1)
        dw_r, dw_i = rotate_real_imag(dw_r, dw_i, phi)
    out = band.clone()
    out[..., 0] = out[..., 0] + dw_r
    out[..., 1] = out[..., 1] + dw_i
    return out


def apply_linf_residual(
    x: torch.Tensor,
    recon: torch.Tensor,
    face_mask: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Ideia 1: ``x_prot = clamp(x + clamp((recon-x)⊙M, ±ε), 0, 1)``."""
    residual = (recon - x) * face_mask
    residual = residual.clamp(-epsilon, epsilon)
    return (x + residual).clamp(0.0, 1.0)


class DTCWTMetaMaskModule(nn.Module):
    """
    Image → DT-CWT →(+ spatial ΔW on editable levels) → IDT-CWT → ROI residual⊙M → L∞.

    Parameters match Phase 3.3 of the sibling project, plus ``epsilon`` / ``roi_compose``.
    """

    def __init__(
        self,
        channels: int = 3,
        img_size: int = 224,
        J: int = 3,
        editable_levels: Sequence[int] | str = DEFAULT_EDITABLE_LEVELS,
        biort: str = "near_sym_b",
        qshift: str = "qshift_b",
        init_std: float = 0.0,
        epsilon: float = DEFAULT_EPSILON,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.img_size = int(img_size)
        self.J = int(J)
        self.biort = biort
        self.qshift = qshift
        self.epsilon = float(epsilon)
        self.editable_levels = parse_editable_levels(editable_levels, self.J)

        self.xfm = DTCWTForward(J=self.J, biort=biort, qshift=qshift)
        self.ifm = DTCWTInverse(biort=biort, qshift=qshift)
        for p in list(self.xfm.parameters()) + list(self.ifm.parameters()):
            p.requires_grad_(False)

        self.delta_mag = nn.ParameterDict()
        self.delta_phase = nn.ParameterDict()
        for level in self.editable_levels:
            h, w = _spatial_size_for_level(self.img_size, level)
            key = str(level)
            mag = torch.zeros(1, self.channels, 6, h, w)
            phase = torch.zeros(1, self.channels, 6, h, w)
            if init_std > 0:
                mag.normal_(0.0, init_std)
                phase.normal_(0.0, init_std)
            self.delta_mag[key] = nn.Parameter(mag)
            self.delta_phase[key] = nn.Parameter(phase)

    @property
    def frozen_levels(self) -> Tuple[int, ...]:
        return tuple(l for l in range(1, self.J + 1) if l not in self.editable_levels)

    def meta_mask_state(self) -> Dict[str, torch.Tensor]:
        state: Dict[str, torch.Tensor] = {}
        for level in self.editable_levels:
            key = str(level)
            state[f"delta_mag_{key}"] = self.delta_mag[key].detach().cpu().clone()
            state[f"delta_phase_{key}"] = self.delta_phase[key].detach().cpu().clone()
        return state

    def load_meta_mask_state(self, state: Dict[str, torch.Tensor]) -> None:
        for level in self.editable_levels:
            key = str(level)
            self.delta_mag[key].data.copy_(
                state[f"delta_mag_{key}"].to(self.delta_mag[key].device)
            )
            self.delta_phase[key].data.copy_(
                state[f"delta_phase_{key}"].to(self.delta_phase[key].device)
            )

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected x shaped (N,C,H,W), got {tuple(x.shape)}")
        if x.shape[-2] != self.img_size or x.shape[-1] != self.img_size:
            return F.interpolate(
                x,
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            )
        return x

    def _apply_mask_to_yh(
        self,
        yh: List[torch.Tensor],
        phi: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Ideia 1: apply full-frame ΔW on editable levels."""
        yh_out: List[torch.Tensor] = []
        for j, band in enumerate(yh):
            level = j + 1
            if level not in self.editable_levels:
                yh_out.append(band)
                continue
            key = str(level)
            yh_out.append(
                add_complex_meta_mask(
                    band,
                    self.delta_mag[key],
                    self.delta_phase[key],
                    phi=phi,
                )
            )
        return yh_out

    def _downsample_roi_to_level(
        self, face_mask: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        """Bilinear downsample of ROI mask to YH spatial size; always detached."""
        m = F.interpolate(
            face_mask,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        return m.detach()

    def _apply_mask_to_yh_wavelet(
        self,
        yh: List[torch.Tensor],
        face_mask: torch.Tensor,
        phi: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Ideia 2: apply ``ΔW ⊙ M_ℓ`` on editable levels only."""
        yh_out: List[torch.Tensor] = []
        for j, band in enumerate(yh):
            level = j + 1
            if level not in self.editable_levels:
                yh_out.append(band)
                continue
            key = str(level)
            # band: (N, C, 6, H, W, 2)
            h_l, w_l = int(band.shape[-3]), int(band.shape[-2])
            m_l = self._downsample_roi_to_level(face_mask, h_l, w_l)  # (N,1,H,W)
            m_b = m_l.unsqueeze(2)  # (N,1,1,H,W) → broadcast vs (1,C,6,H,W)
            d_mag = self.delta_mag[key] * m_b
            d_phase = self.delta_phase[key] * m_b
            yh_out.append(add_complex_meta_mask(band, d_mag, d_phase, phi=phi))
        return yh_out

    def _resolve_face_mask(
        self,
        x: torch.Tensor,
        face_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        n, _, h, w = x.shape
        if face_mask is None:
            return torch.ones(n, 1, h, w, device=x.device, dtype=x.dtype)
        m = face_mask
        if m.ndim == 3:
            m = m.unsqueeze(1)
        if m.shape[-2:] != (h, w):
            m = F.interpolate(m, size=(h, w), mode="bilinear", align_corners=False)
        return m.to(device=x.device, dtype=x.dtype).detach()

    def forward(
        self,
        x: torch.Tensor,
        face_mask: Optional[torch.Tensor] = None,
        roi_compose: str = ROI_COMPOSE_PIXEL,
        extra_phase: Optional[torch.Tensor] = None,
        return_coeffs: bool = False,
    ):
        """
        Protect ``x`` with the meta-mask and optional ROI compose.

        cos/SSIM (later steps) must always be computed on the returned ``x_prot``.
        """
        if roi_compose == ROI_COMPOSE_PIXEL:
            # --- Ideia 1 (baseline; do not refactor body casually) ---
            x_in = self._prepare_input(x)
            yl, yh = self.xfm(x_in)
            yh_out = self._apply_mask_to_yh(yh, phi=extra_phase)
            recon = self.ifm((yl, yh_out))
            if torch.is_complex(recon):
                recon = recon.real

            m = self._resolve_face_mask(x_in, face_mask)
            x_prot = apply_linf_residual(x_in, recon, m, self.epsilon)

            if x_prot.shape[-2:] != x.shape[-2:]:
                x_prot = F.interpolate(
                    x_prot, size=x.shape[-2:], mode="bilinear", align_corners=False
                )

            if return_coeffs:
                return x_prot, yl, yh_out
            return x_prot

        if roi_compose == ROI_COMPOSE_WAVELET:
            # --- Ideia 2: restrict ΔW in wavelet domain; L∞ on full residual ---
            x_in = self._prepare_input(x)
            m = self._resolve_face_mask(x_in, face_mask)
            yl, yh = self.xfm(x_in)
            yh_out = self._apply_mask_to_yh_wavelet(yh, m, phi=extra_phase)
            recon = self.ifm((yl, yh_out))
            if torch.is_complex(recon):
                recon = recon.real
            residual = (recon - x_in).clamp(-self.epsilon, self.epsilon)
            x_prot = (x_in + residual).clamp(0.0, 1.0)

            if x_prot.shape[-2:] != x.shape[-2:]:
                x_prot = F.interpolate(
                    x_prot, size=x.shape[-2:], mode="bilinear", align_corners=False
                )

            if return_coeffs:
                return x_prot, yl, yh_out
            return x_prot

        raise ValueError(
            f"roi_compose inválido: {roi_compose!r}. "
            f"Válidos: {ROI_COMPOSE_PIXEL!r}, {ROI_COMPOSE_WAVELET!r}."
        )
