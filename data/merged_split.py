"""
V2 merged LFW + CelebA fractional split (Phase 3.3 parity).

Default roots (override via env or function args):
  /mnt/databases/faces/lfw_frontal/{faces_train,faces_test}
  /mnt/databases/faces/celebahq_pp_frontal/
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

MERGED_DATASET = "merged_lfw_celeba"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

DEFAULT_LFW_ROOT = "/mnt/databases/faces/lfw_frontal"
DEFAULT_CELEBA_ROOT = "/mnt/databases/faces/celebahq_pp_frontal"

DATASET_ROOTS: Dict[str, str] = {
    "lfw_frontal": os.environ.get("V2_LFW_ROOT", DEFAULT_LFW_ROOT),
    "celebahq_pp_frontal": os.environ.get("V2_CELEBA_ROOT", DEFAULT_CELEBA_ROOT),
}


def list_image_paths(directory: str | Path, recursive: bool = False) -> List[str]:
    """Return sorted absolute paths of usable images under ``directory``."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory not found: {directory}")

    paths: List[str] = []
    if recursive:
        for root, _dirs, files in os.walk(directory):
            for name in files:
                if name.startswith((".", "_")):
                    continue
                if name.lower().endswith(IMAGE_EXTENSIONS):
                    paths.append(str(Path(root, name).resolve()))
    else:
        for name in os.listdir(directory):
            if name.startswith((".", "_")):
                continue
            if name.lower().endswith(IMAGE_EXTENSIONS):
                paths.append(str((directory / name).resolve()))
    paths.sort()
    if not paths:
        raise FileNotFoundError(f"No images found in: {directory}")
    return paths


def resolve_split_dir(
    dataset_name: str,
    split: str = "train",
    *,
    roots: Optional[Dict[str, str]] = None,
) -> str:
    roots = roots or DATASET_ROOTS
    if dataset_name not in roots:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Supported: {sorted(roots)}"
        )
    root = roots[dataset_name]

    if dataset_name == "lfw_frontal":
        if split == "train":
            return os.path.join(root, "faces_train")
        if split == "test":
            return os.path.join(root, "faces_test")
        raise ValueError(f"Unsupported split for lfw_frontal: {split}")

    # celebahq_pp_frontal: flat folder
    return root


def build_merged_frac_split(
    celeba_train_frac: float = 0.5,
    data_seed: int = 42,
    max_train: Optional[int] = None,
    *,
    lfw_root: Optional[str] = None,
    celeba_root: Optional[str] = None,
) -> Tuple[List[str], List[str], Dict[str, int]]:
    """
    Phase 3.3 on-the-fly split (no new dataset on disk).

    Train = all LFW ``faces_train`` + seeded CelebA fraction.
    Eval  = LFW ``faces_test`` ∪ CelebA holdout.
    """
    if not 0.0 < celeba_train_frac <= 1.0:
        raise ValueError(
            f"celeba_train_frac must be in (0, 1], got {celeba_train_frac}"
        )

    roots = {
        "lfw_frontal": lfw_root or DATASET_ROOTS["lfw_frontal"],
        "celebahq_pp_frontal": celeba_root or DATASET_ROOTS["celebahq_pp_frontal"],
    }

    lfw_train_dir = resolve_split_dir("lfw_frontal", "train", roots=roots)
    lfw_test_dir = resolve_split_dir("lfw_frontal", "test", roots=roots)
    celeba_dir = resolve_split_dir("celebahq_pp_frontal", "all", roots=roots)

    for label, d in (
        ("LFW train", lfw_train_dir),
        ("LFW test", lfw_test_dir),
        ("CelebA", celeba_dir),
    ):
        if not os.path.isdir(d):
            raise FileNotFoundError(
                f"{label} directory missing: {d}. "
                "Set V2_LFW_ROOT / V2_CELEBA_ROOT or pass lfw_root/celeba_root."
            )

    lfw_train = list_image_paths(lfw_train_dir)
    lfw_test = list_image_paths(lfw_test_dir)
    celeba = list_image_paths(celeba_dir)

    rng = np.random.default_rng(data_seed)
    celeba_shuffled = list(celeba)
    rng.shuffle(celeba_shuffled)
    n_celeba_train = int(np.floor(celeba_train_frac * len(celeba_shuffled)))
    celeba_train = celeba_shuffled[:n_celeba_train]
    celeba_eval = celeba_shuffled[n_celeba_train:]

    train_paths = list(lfw_train) + list(celeba_train)
    rng.shuffle(train_paths)
    if max_train is not None:
        train_paths = train_paths[: int(max_train)]

    eval_paths = list(lfw_test) + list(celeba_eval)
    stats = {
        "n_lfw_train": len(lfw_train),
        "n_lfw_test": len(lfw_test),
        "n_celeba_total": len(celeba),
        "n_celeba_train": len(celeba_train),
        "n_celeba_eval": len(celeba_eval),
        "n_train_total": len(train_paths),
        "n_eval_total": len(eval_paths),
    }
    return train_paths, eval_paths, stats


def build_flat_data_dict(
    dataset_name: str = MERGED_DATASET,
    *,
    celeba_train_frac: float = 0.5,
    data_seed: int = 42,
    max_train: Optional[int] = None,
    max_eval: Optional[int] = None,
    lfw_root: Optional[str] = None,
    celeba_root: Optional[str] = None,
) -> Dict[str, Any]:
    """Minimal Phase 3.3 data dict: ``{train, eval, split_stats}``."""
    if dataset_name != MERGED_DATASET:
        raise ValueError(
            f"V2 Step 5 supports only '{MERGED_DATASET}', got {dataset_name!r}"
        )
    train_paths, eval_paths, split_stats = build_merged_frac_split(
        celeba_train_frac=celeba_train_frac,
        data_seed=data_seed,
        max_train=max_train,
        lfw_root=lfw_root,
        celeba_root=celeba_root,
    )
    if max_eval is not None:
        eval_paths = eval_paths[: int(max_eval)]
        split_stats = dict(split_stats)
        split_stats["n_eval_total"] = len(eval_paths)
    return {
        "train": train_paths,
        "eval": eval_paths,
        "split_stats": split_stats,
        "dataset": MERGED_DATASET,
        "celeba_train_frac": float(celeba_train_frac),
        "data_seed": int(data_seed),
    }


class PathListFaceDataset(Dataset):
    """Flat face images from an explicit path list → ``(tensor[0,1], path, source_hw)``."""

    def __init__(
        self,
        image_paths: Sequence[str],
        image_size: int = 224,
        transform: Optional[Callable] = None,
    ) -> None:
        self.image_paths = [str(Path(p).resolve()) for p in image_paths]
        self.image_size = int(image_size)
        self.transform = transform
        if not self.image_paths:
            raise ValueError("PathListFaceDataset received an empty path list.")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str, Tuple[int, int]]:
        path = self.image_paths[index]
        img = Image.open(path).convert("RGB")
        source_hw = (int(img.height), int(img.width))
        if self.image_size > 0 and (
            img.width != self.image_size or img.height != self.image_size
        ):
            img = img.resize(
                (self.image_size, self.image_size), Image.Resampling.LANCZOS
            )
        if self.transform is not None:
            tensor = self.transform(img)
        else:
            array = np.asarray(img, dtype=np.float32)
            tensor = torch.from_numpy(array).permute(2, 0, 1) / 255.0
        return tensor, path, source_hw


def make_face_dataloader(
    image_paths: Sequence[str],
    *,
    batch_size: int = 2,
    image_size: int = 224,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    dataset = PathListFaceDataset(image_paths=image_paths, image_size=image_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def observe_batch_hw(batch_x: torch.Tensor) -> Dict[str, Any]:
    """Log fields for config / train header from a ``(N,C,H,W)`` batch."""
    if batch_x.ndim != 4:
        raise ValueError(f"Expected (N,C,H,W), got {tuple(batch_x.shape)}")
    _, _, h, w = batch_x.shape
    return {
        "batch_h": int(h),
        "batch_w": int(w),
        "batch_hw": f"{h}x{w}",
        "image_size": int(h) if h == w else None,
    }


def sample_native_hw(
    image_paths: Sequence[str],
    max_samples: int = 8,
) -> List[str]:
    """Short list of native ``HxW`` strings for config logging."""
    out: List[str] = []
    for path in list(image_paths)[:max_samples]:
        with Image.open(path) as img:
            out.append(f"{img.height}x{img.width}")
    return out


def build_split_config_block(
    split_stats: Dict[str, int],
    *,
    image_size: int,
    data_seed: int,
    celeba_train_frac: float,
    dataset_native_hw_sample: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Fields intended for ``config.json`` / run summary."""
    block: Dict[str, Any] = {
        "dataset": MERGED_DATASET,
        "image_size": int(image_size),
        "data_seed": int(data_seed),
        "celeba_train_frac": float(celeba_train_frac),
        "n_train": int(split_stats.get("n_train_total", 0)),
        "n_val": int(split_stats.get("n_eval_total", 0)),
        "split_stats": dict(split_stats),
    }
    if dataset_native_hw_sample is not None:
        block["dataset_native_hw_sample"] = list(dataset_native_hw_sample)
    return block
