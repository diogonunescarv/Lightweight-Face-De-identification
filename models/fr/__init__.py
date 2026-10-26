"""V2 facial-recognition ensembles (Phase 3.3 presets + optional Protego FR_DB)."""

from models.fr.ensemble import (
    describe_fr_backend,
    ensemble_cosine_mean,
    fr_requires_protego,
    fr_to_cpu,
    fr_to_device,
    load_fr_ensemble,
    load_fr_model,
    load_train_and_eval_ensembles,
)
from models.fr.presets import (
    BASIC_POOL,
    BUILTIN_FACENET_NAMES,
    DEFAULT_EVAL_FR_NAMES,
    FULL_TRAIN_FR_NAMES,
    SMALL_TRAIN_FR_NAMES,
    SMOKE_FR_NAMES,
    SPECIAL_POOL,
    assert_train_eval_disjoint,
    resolve_eval_fr_names,
    resolve_train_fr_names,
)

__all__ = [
    "BASIC_POOL",
    "BUILTIN_FACENET_NAMES",
    "DEFAULT_EVAL_FR_NAMES",
    "FULL_TRAIN_FR_NAMES",
    "SMALL_TRAIN_FR_NAMES",
    "SMOKE_FR_NAMES",
    "SPECIAL_POOL",
    "assert_train_eval_disjoint",
    "describe_fr_backend",
    "ensemble_cosine_mean",
    "fr_requires_protego",
    "fr_to_cpu",
    "fr_to_device",
    "load_fr_ensemble",
    "load_fr_model",
    "load_train_and_eval_ensembles",
    "resolve_eval_fr_names",
    "resolve_train_fr_names",
]
