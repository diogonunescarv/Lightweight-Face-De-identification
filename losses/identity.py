"""V2 identity loss — mean cosine over FR features (Phase 3.3; no target hinge)."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def identity_cosine_loss(
    feat_orig: torch.Tensor,
    feat_prot: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Mean cosine similarity between original and protected embeddings.

    Minimize this value (lower = stronger de-identification). No ``target_cos`` hinge.

    Args:
        feat_orig: ``(N, D)`` FR features of clean faces.
        feat_prot: ``(N, D)`` FR features of protected faces.

    Returns:
        ``(loss, cos_detached)`` — both scalars; loss has grad through ``feat_prot``.
    """
    if feat_orig.shape != feat_prot.shape:
        raise ValueError(
            f"feat shape mismatch orig={tuple(feat_orig.shape)} "
            f"prot={tuple(feat_prot.shape)}"
        )
    if feat_orig.ndim != 2:
        raise ValueError(f"features must be (N,D), got {tuple(feat_orig.shape)}")

    o = F.normalize(feat_orig, p=2, dim=-1)
    p = F.normalize(feat_prot, p=2, dim=-1)
    cos_sim = (o * p).sum(dim=-1).mean()
    return cos_sim, cos_sim.detach()


def mean_identity_cosine(
    cos_per_fr: list[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Average per-FR cosine losses (Phase 3.3 ensemble: mean over surrogates).

    Each entry should already be a scalar cosine (e.g. from :func:`identity_cosine_loss`).
    """
    if not cos_per_fr:
        raise ValueError("cos_per_fr must be non-empty")
    stacked = torch.stack([c if c.ndim == 0 else c.mean() for c in cos_per_fr])
    mean_cos = stacked.mean()
    return mean_cos, mean_cos.detach()
