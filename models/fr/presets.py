"""FR ensemble presets (Phase 3.3 / Protego naming parity)."""

from __future__ import annotations

from typing import Sequence

# Ten training-surrogate models (Protego Table 2). Default intruder: ir50_adaface_casia.
BASIC_POOL: tuple[str, ...] = (
    "ir50_softmax_casia",
    "ir50_cosface_casia",
    "ir50_arcface_casia",
    "mobilenet_arcface_casia",
    "mobilefacenet_arcface_casia",
    "ir18_adaface_webface",
    "ir50_adaface_ms1mv2",
    "ir50_adaface_casia",
    "ir50_adaface_webface",
    "ir101_adaface_webface",
)

# Held-out black-box models (different algo/arch).
SPECIAL_POOL: tuple[str, ...] = (
    "inception_facenet_vgg",
    "inception_facenet_casia",
    "ir50_magface_ms1mv2",
    "ir100_magface_ms1mv2",
    "vit_cosface_ms1mv2",
    "vits_cosface_ms1mv2",
)

# Protego train surrogates: BASIC_POOL without the usual held-out intruder.
FULL_TRAIN_FR_NAMES: tuple[str, ...] = tuple(
    n for n in BASIC_POOL if n != "ir50_adaface_casia"
)

# Compact ensemble for ~11GB GPUs.
SMALL_TRAIN_FR_NAMES: tuple[str, ...] = (
    "ir50_arcface_casia",
    "mobilefacenet_arcface_casia",
    "ir18_adaface_webface",
)

# Default held-out eval (never in small/full train presets).
DEFAULT_EVAL_FR_NAMES: tuple[str, ...] = (
    "inception_facenet_vgg",
    "inception_facenet_casia",
)

# Built-in backends available without Protego FR_DB (facenet-pytorch).
BUILTIN_FACENET_NAMES: frozenset[str] = frozenset(
    {"inception_facenet_vgg", "inception_facenet_casia"}
)

SMOKE_FR_NAMES: tuple[str, ...] = (
    "inception_facenet_vgg",
    "inception_facenet_casia",
)


def parse_fr_csv(spec: str | Sequence[str] | None) -> tuple[str, ...]:
    if spec is None:
        return ()
    if isinstance(spec, str):
        parts = tuple(p.strip() for p in spec.split(",") if p.strip())
    else:
        parts = tuple(str(p).strip() for p in spec if str(p).strip())
    if len(set(parts)) != len(parts):
        raise ValueError(f"FR names with duplicates: {parts}")
    return parts


def resolve_train_fr_names(
    preset: str = "small",
    fr_models: str | Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Resolve train surrogate names. ``fr_models`` CSV overrides ``preset``."""
    override = parse_fr_csv(fr_models)
    if override:
        return override
    preset = preset.strip().lower()
    if preset == "small":
        return SMALL_TRAIN_FR_NAMES
    if preset == "full":
        return FULL_TRAIN_FR_NAMES
    raise ValueError(f"fr_preset desconhecido: {preset!r}. Use 'small' ou 'full'.")


def resolve_eval_fr_names(
    eval_fr_models: str | Sequence[str] | None = None,
) -> tuple[str, ...]:
    override = parse_fr_csv(eval_fr_models)
    return override if override else DEFAULT_EVAL_FR_NAMES


def assert_train_eval_disjoint(
    train_names: Sequence[str],
    eval_names: Sequence[str],
) -> None:
    overlap = set(train_names) & set(eval_names)
    if overlap:
        raise ValueError(
            "train FR ∩ eval FR must be empty; overlap="
            f"{sorted(overlap)}. Ajuste --fr-models / --eval-fr-models."
        )
