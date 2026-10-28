"""
Bridge to Protego ``FacialRecognition.FR`` + ``FR_DB`` **neste repositório**.

Layout (via ``scripts/run_v2_setup_protego_fr.sh``)::

    third_party/protego_fr/
      protego/FacialRecognition.py
      FR_DB/   # arquiteturas; pesos em */pretrained/ baixados com gdown

``third_party/protego_fr/`` está no ``.gitignore`` — código local + pesos não sobem
ao GitHub. Override opcional: ``PROTEGO_FR_ROOT`` = essa base (ou a pasta ``protego/``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_THIRD_PARTY = _REPO_ROOT / "third_party" / "protego_fr"


def resolve_protego_base() -> Path | None:
    """Diretório com ``protego/`` + ``FR_DB/`` (BASE_PATH do FacialRecognition)."""
    candidates: list[Path] = []
    env = os.environ.get("PROTEGO_FR_ROOT", "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        candidates.append(p)
        candidates.append(p.parent)  # se apontar para .../protego
    candidates.append(_DEFAULT_THIRD_PARTY.resolve())

    seen: set[Path] = set()
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        if (c / "protego" / "FacialRecognition.py").is_file() and (c / "FR_DB").is_dir():
            return c
    return None


def resolve_protego_root() -> Path | None:
    """Alias compatível com testes antigos."""
    return resolve_protego_base()


def protego_available() -> bool:
    return resolve_protego_base() is not None


def _load_fr_class(base: Path) -> type:
    """Importa ``FR`` como ``protego.FacialRecognition`` (imports relativos OK)."""
    base_s = str(base.resolve())
    if base_s not in sys.path:
        sys.path.insert(0, base_s)
    from protego.FacialRecognition import FR  # type: ignore[import-not-found]

    return FR


def load_protego_fr(name: str, device: str | Any) -> Any:
    base = resolve_protego_base()
    if base is None:
        raise FileNotFoundError(
            f"FR '{name}' requires Protego FR_DB under {_DEFAULT_THIRD_PARTY}. "
            "Rode: bash scripts/run_v2_setup_protego_fr.sh "
            "(código + pesos via gdown; pasta gitignored). "
            "Built-in sem FR_DB: inception_facenet_vgg, inception_facenet_casia."
        )
    fr_cls = _load_fr_class(base)
    import torch

    return fr_cls(model_name=name, device=torch.device(device))
