#!/usr/bin/env python3
"""Materialize merged LFW-test + CelebA holdout as a flat symlink folder for evaluate-v2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.merged_split import build_merged_frac_split


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir",
        required=True,
        help="Pasta flat de destino (symlinks para as imagens holdout).",
    )
    p.add_argument("--celeba-train-frac", type=float, default=0.5)
    p.add_argument("--data-seed", type=int, default=42)
    p.add_argument("--max-eval", type=int, default=0, help="0 = todas as holdout")
    p.add_argument("--lfw-root", default=None)
    p.add_argument("--celeba-root", default=None)
    p.add_argument(
        "--copy",
        action="store_true",
        help="Copiar arquivos em vez de symlink (default: symlink).",
    )
    args = p.parse_args()

    _train, eval_paths, stats = build_merged_frac_split(
        celeba_train_frac=args.celeba_train_frac,
        data_seed=args.data_seed,
        lfw_root=args.lfw_root,
        celeba_root=args.celeba_root,
    )
    if args.max_eval and args.max_eval > 0:
        eval_paths = eval_paths[: int(args.max_eval)]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    import shutil

    n_ok = 0
    for i, src in enumerate(eval_paths):
        src_p = Path(src)
        # Unique names: index + original stem/suffix (avoid collisions across datasets).
        dst = out / f"{i:06d}_{src_p.stem}{src_p.suffix}"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if args.copy:
            shutil.copy2(src_p, dst)
        else:
            dst.symlink_to(src_p.resolve())
        n_ok += 1

    meta = {
        "out_dir": str(out.resolve()),
        "n_links": n_ok,
        "celeba_train_frac": args.celeba_train_frac,
        "data_seed": args.data_seed,
        "max_eval": args.max_eval or None,
        "split_stats": stats,
        "mode": "copy" if args.copy else "symlink",
    }
    (out / "holdout_manifest.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(
        f"[holdout] {n_ok} imagens → {out} "
        f"(lfw_test={stats['n_lfw_test']} celeba_eval={stats['n_celeba_eval']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
