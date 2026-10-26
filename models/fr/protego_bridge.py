"""
Optional bridge to the sibling Protego ``FacialRecognition.FR`` + ``FR_DB``.

Weights / FR_DB are **not** vendored in this repo (keeps GitHub lean). Set either:
  - env ``PROTEGO_FR_ROOT`` → directory containing ``FacialRecognition.py`` and ``FR_DB/``
  - or ``third_party/protego_fr/`` relative to the repo root

Without that path, only built-in FaceNet names load.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_THIRD_PARTY = _REPO_ROOT / "third_party" / "protego_fr"


def resolve_protego_root() -> Path | None:
    env = os.environ.get("PROTEGO_FR_ROOT", "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if (p / "FacialRecognition.py").is_file() or (p / "protego" / "FacialRecognition.py").is_file():
            return p
    if (_DEFAULT_THIRD_PARTY / "FacialRecognition.py").is_file():
        return _DEFAULT_THIRD_PARTY.resolve()
    # Sibling layout: .../protego_base/protego/FacialRecognition.py
    nested = _DEFAULT_THIRD_PARTY / "protego" / "FacialRecognition.py"
    if nested.is_file():
        return (_DEFAULT_THIRD_PARTY / "protego").resolve()
    return None


def protego_available() -> bool:
    return resolve_protego_root() is not None


def _load_fr_class(root: Path) -> type:
    """Import ``FR`` from FacialRecognition.py without requiring package install."""
    candidates = [
        root / "FacialRecognition.py",
        root / "protego" / "FacialRecognition.py",
    ]
    path = next((c for c in candidates if c.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"FacialRecognition.py not found under {root}")

    # Ensure FR_DB sibling imports resolve (FacialRecognition imports FR_DB.*).
    parent = str(path.parent)
    grand = str(path.parent.parent)
    for entry in (parent, grand):
        if entry not in sys.path:
            sys.path.insert(0, entry)

    spec = importlib.util.spec_from_file_location("protego_facial_recognition_v2", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fr_cls = getattr(mod, "FR", None)
    if fr_cls is None:
        raise ImportError(f"{path} does not define FR")
    return fr_cls


def load_protego_fr(name: str, device: str | Any) -> Any:
    root = resolve_protego_root()
    if root is None:
        raise FileNotFoundError(
            f"FR '{name}' requires Protego FR_DB. Set PROTEGO_FR_ROOT or place "
            f"FacialRecognition.py + FR_DB under {_DEFAULT_THIRD_PARTY}. "
            "Built-in without FR_DB: inception_facenet_vgg, inception_facenet_casia."
        )
    fr_cls = _load_fr_class(root)
    import torch

    return fr_cls(model_name=name, device=torch.device(device))
