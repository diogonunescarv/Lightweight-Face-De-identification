"""
V2 quality: L∞ budget, SSIM hinge, adaptive percep weight (Phase 3.3).

SSIM uses the project’s canonical ``ssim_index`` / ``ssim_index_masked`` (no
``pytorch_msssim`` dependency) so ``--ssim-region`` matches V1 semantics.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch

from losses.losses import ssim_index, ssim_index_masked

DEFAULT_MIN_SSIM = 0.95
DEFAULT_W_MIN = 1.0 / 129.0
DEFAULT_W_MAX = 129.0
DEFAULT_EPSILON = 16.0 / 255.0


def apply_linf_budget(
    recon: torch.Tensor,
    orig: torch.Tensor,
    epsilon: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Clamp spatial residual to ``[-epsilon, epsilon]`` then pixels to ``[0, 1]``.

    Note: Ideia 1 ROI path uses ``(recon-x)⊙M`` inside ``DTCWTMetaMaskModule``;
    this helper is the plain (full-frame) L∞ clamp for tests / ``--no-face-mask``.
    """
    if epsilon < 0:
        raise ValueError(f"epsilon must be >= 0, got {epsilon}")
    if recon.shape != orig.shape:
        raise ValueError(
            f"shape mismatch recon={tuple(recon.shape)} orig={tuple(orig.shape)}"
        )
    residual = (recon - orig).clamp(-epsilon, epsilon)
    protected = (orig + residual).clamp(0.0, 1.0)
    return protected, residual


def compute_ssim(
    protected: torch.Tensor,
    orig: torch.Tensor,
    ssim_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Batch-mean SSIM scalar with grad.

    If ``ssim_mask`` is None or effectively all-ones, uses full-image SSIM;
    otherwise weighted ``ssim_index_masked`` (``--ssim-region mask`` / scoped).
    """
    if ssim_mask is None:
        return ssim_index(orig, protected).mean()
    # Treat near-full coverage as full SSIM (numerically stable + matches M≡1).
    if float(ssim_mask.mean().detach()) >= 1.0 - 1e-6:
        return ssim_index(orig, protected).mean()
    return ssim_index_masked(orig, protected, ssim_mask).mean()


def ssim_hinge_loss(
    protected: torch.Tensor,
    orig: torch.Tensor,
    min_ssim: float,
    weight: float,
    ssim_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Protego-style perceptual hinge: ``weight * max(min_ssim - SSIM, 0)``.

    Returns ``(percep_term, ssim_value)``; both scalars (ssim may be used for logging).
    """
    ssim_value = compute_ssim(protected, orig, ssim_mask=ssim_mask)
    zero = torch.zeros((), device=ssim_value.device, dtype=ssim_value.dtype)
    percep_term = float(weight) * torch.max(
        torch.as_tensor(min_ssim, device=ssim_value.device, dtype=ssim_value.dtype)
        - ssim_value,
        zero,
    )
    return percep_term, ssim_value


def adapt_percep_weight(
    ssim_value: Union[float, torch.Tensor],
    min_ssim: float,
    weight: float,
    min_weight: float = DEFAULT_W_MIN,
    max_weight: float = DEFAULT_W_MAX,
) -> float:
    """
    Protego adaptive weight: ×2 if SSIM too low, ÷2 if too high, else reset to 1.

    Thresholds: ``min_ssim * 0.99`` / ``min_ssim * 1.01``.
    """
    s = (
        float(ssim_value.detach().cpu())
        if isinstance(ssim_value, torch.Tensor)
        else float(ssim_value)
    )
    w = float(weight)
    if s >= min_ssim * 1.01 and w > min_weight:
        return w / 2.0
    if s <= min_ssim * 0.99 and w <= max_weight:
        return w * 2.0
    if min_ssim * 0.99 < s < min_ssim * 1.01:
        return 1.0
    return w
