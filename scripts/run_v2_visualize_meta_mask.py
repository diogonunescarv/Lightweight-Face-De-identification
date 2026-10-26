#!/usr/bin/env python3
"""CLI: visualize V2 meta-mask (orig / prot / residual / ROI). Outputs under <run>/viz/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.merged_split import list_image_paths
from evaluation.visualize_v2 import render_meta_mask_panel, visualize_batch


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="meta_mask.pt / best / latest")
    p.add_argument("--image", default=None, help="Uma imagem")
    p.add_argument("--data", default=None, help="Pasta flat de imagens")
    p.add_argument(
        "--out-dir",
        default=None,
        help="Destino (default: <checkpoint_dir>/viz)",
    )
    p.add_argument("--max-images", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--roi-compose",
        choices=["pixel", "wavelet"],
        default=None,
        help="Override do checkpoint",
    )
    p.add_argument(
        "--no-compare",
        action="store_true",
        help="Não renderiza o outro roi_compose lado a lado",
    )
    p.add_argument("--no-face-mask", action="store_true")
    p.add_argument("--mask-mode", default=None)
    p.add_argument("--mask-shape", default=None)
    p.add_argument("--mask-regions", default=None)
    args = p.parse_args()

    ckpt = Path(args.checkpoint).resolve()
    out_dir = Path(args.out_dir) if args.out_dir else ckpt.parent / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    if args.image:
        paths.append(Path(args.image))
    if args.data:
        paths.extend(Path(p) for p in list_image_paths(args.data)[: args.max_images])
    if not paths:
        print("Informe --image e/ou --data.", file=sys.stderr)
        return 2

    kwargs = dict(
        device=args.device,
        roi_compose=args.roi_compose,
        compare_compose=not args.no_compare,
        no_face_mask=args.no_face_mask,
        mask_mode=args.mask_mode,
        mask_shape=args.mask_shape,
        mask_regions=args.mask_regions,
    )
    if len(paths) == 1:
        man = render_meta_mask_panel(ckpt, paths[0], out_dir, **kwargs)
        print(f"[viz] panel → {man['files'].get('panel')}")
    else:
        results = visualize_batch(ckpt, paths[: args.max_images], out_dir, **kwargs)
        print(f"[viz] {len(results)} painéis → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
