"""Load FR ensembles, CPU↔GPU offload, and train/eval name resolution."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from models.fr.embedder import FaceNetFR, ProtegoFRAdapter, l2_normalize
from models.fr.presets import (
    BUILTIN_FACENET_NAMES,
    assert_train_eval_disjoint,
    resolve_eval_fr_names,
    resolve_train_fr_names,
)
from models.fr.protego_bridge import load_protego_fr, protego_available


def load_fr_model(name: str, device: torch.device | str = "cpu") -> nn.Module:
    """Load a single frozen FR by Protego-style name."""
    device = torch.device(device)
    if name in BUILTIN_FACENET_NAMES:
        return FaceNetFR(name, device=device)
    fr_obj = load_protego_fr(name, device)
    return ProtegoFRAdapter(fr_obj, name=name, device=device)


def load_fr_ensemble(
    names: Sequence[str] | None = None,
    *,
    preset: str = "small",
    fr_models: str | Sequence[str] | None = None,
    device: torch.device | str = "cpu",
    keep_on_device: bool = False,
) -> list[nn.Module]:
    """
    Load a list of frozen FRs.

    By default models are placed on CPU (``keep_on_device=False``) so callers can
    offload one-at-a-time onto GPU during the forward loop (Phase 3.2/3.3 style).
    """
    if names is None:
        names = resolve_train_fr_names(preset=preset, fr_models=fr_models)
    if not names:
        raise ValueError("FR ensemble names must be non-empty")

    target = torch.device(device) if keep_on_device else torch.device("cpu")
    models: list[nn.Module] = []
    for name in names:
        m = load_fr_model(name, device=target)
        models.append(m)
    return models


def load_train_and_eval_ensembles(
    *,
    preset: str = "small",
    fr_models: str | Sequence[str] | None = None,
    eval_fr_models: str | Sequence[str] | None = None,
    device: torch.device | str = "cpu",
    keep_on_device: bool = False,
) -> tuple[list[nn.Module], list[nn.Module], tuple[str, ...], tuple[str, ...]]:
    """Resolve names, assert disjointness, load train + held-out eval ensembles."""
    train_names = resolve_train_fr_names(preset=preset, fr_models=fr_models)
    eval_names = resolve_eval_fr_names(eval_fr_models)
    assert_train_eval_disjoint(train_names, eval_names)
    train = load_fr_ensemble(train_names, device=device, keep_on_device=keep_on_device)
    eval_frs = load_fr_ensemble(eval_names, device=device, keep_on_device=keep_on_device)
    return train, eval_frs, train_names, eval_names


def fr_to_device(model: nn.Module, device: torch.device | str) -> nn.Module:
    if hasattr(model, "to"):
        return model.to(device)  # type: ignore[return-value]
    return model


def fr_to_cpu(model: nn.Module) -> nn.Module:
    return fr_to_device(model, "cpu")


@torch.no_grad()
def embed_no_grad(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return model(x)


def embed_with_grad(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Forward allowing grads w.r.t. ``x`` (FR weights stay frozen)."""
    return model(x)


def ensemble_cosine_mean(
    models: Sequence[nn.Module],
    x_orig: torch.Tensor,
    x_prot: torch.Tensor,
    device: torch.device | str,
    *,
    offload: bool = True,
    normalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Mean cosine over FRs. Orig embeddings without grad; prot with grad.

    When ``offload=True``, each FR is moved to ``device`` for its forward then
    returned to CPU (VRAM-friendly multi-FR).
    """
    device = torch.device(device)
    cos_list: list[torch.Tensor] = []
    for model in models:
        if offload:
            fr_to_device(model, device)
        try:
            with torch.no_grad():
                f0 = model(x_orig.to(device))
            f1 = model(x_prot.to(device))
            if normalize:
                f0 = l2_normalize(f0)
                f1 = l2_normalize(f1)
            cos_list.append((f0 * f1).sum(dim=-1).mean())
        finally:
            if offload and device.type == "cuda":
                fr_to_cpu(model)
                torch.cuda.empty_cache()
    stacked = torch.stack(cos_list)
    mean_cos = stacked.mean()
    return mean_cos, mean_cos.detach()


def fr_requires_protego(names: Sequence[str]) -> bool:
    return any(n not in BUILTIN_FACENET_NAMES for n in names)


def describe_fr_backend() -> str:
    if protego_available():
        return "builtin-facenet+protego"
    return "builtin-facenet-only"
