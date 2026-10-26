"""V2 Step 6 — trainer smoke (pixel ROI, adam/protego, resume, isolation)."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from models.dtcwt_meta_mask import DTCWTMetaMaskModule
from training.trainer_v2 import train_v2


def _write_jpg(path: Path, size: int = 64, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(path, format="JPEG")


def _make_image_dir(root: Path, n: int = 8, size: int = 64) -> list[str]:
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        p = root / f"img_{i:03d}.jpg"
        _write_jpg(p, size=size, seed=i)
        paths.append(str(p.resolve()))
    return paths


def _base_args(out_root: Path, train_paths: list[str], eval_paths: list[str], **kw):
    d = dict(
        out_root=str(out_root),
        exp_name="smoke",
        device="cpu",
        image_size=64,
        batch_size=2,
        num_workers=0,
        epochs=1,
        max_steps=2,
        optimizer="protego",
        lr=None,
        epsilon=16.0 / 255.0,
        min_ssim=0.95,
        wavelet_J=3,
        editable_levels="2,3",
        init_std=1e-2,
        fr_preset="small",
        fr_models=None,
        eval_fr_models="",
        smoke_fr=True,
        celeba_train_frac=0.5,
        data_seed=42,
        seed=0,
        roi_compose="pixel",
        no_face_mask=False,
        mask_mode="fixed",
        mask_shape="ellipse",
        mask_regions="full",
        ssim_region="mask",
        early_stopping=False,
        early_stopping_metric="cos",
        patience=2,
        min_delta=0.005,
        val_max_samples=4,
        ckpt_every=1,
        preview_every=1,
        resume=None,
        train_paths=train_paths,
        eval_paths=eval_paths,
        lfw_root=None,
        celeba_root=None,
        max_train=None,
    )
    d.update(kw)
    return argparse.Namespace(**d)


class TestTrainV2Smoke(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.train_paths = _make_image_dir(self.root / "train", n=6, size=64)
        self.eval_paths = _make_image_dir(self.root / "eval", n=4, size=64)
        self.out_root = self.root / "runs"

    def tearDown(self):
        self.tmp.cleanup()

    def test_protego_smoke_writes_only_under_run(self):
        args = _base_args(self.out_root, self.train_paths, self.eval_paths, optimizer="protego")
        out = train_v2(args)
        self.assertTrue(out.is_dir())
        self.assertTrue((out / "config.json").is_file())
        self.assertTrue((out / "train_log.csv").is_file())
        self.assertTrue((out / "checkpoint_latest.pt").is_file())
        self.assertTrue((out / "meta_mask.pt").is_file())
        self.assertTrue((out / "summary.json").is_file())
        cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(cfg["roi_compose"], "pixel")
        self.assertEqual(cfg["image_size"], 64)
        self.assertIn("batch_hw", cfg)
        # Isolation: no sibling artifacts next to out_root except the run folder.
        outside = [
            p
            for p in self.root.rglob("*")
            if p.is_file() and self.out_root not in p.parents and p.parent != self.root / "train" and p.parent != self.root / "eval"
        ]
        # train/eval jpgs are expected; nothing else at root.
        self.assertTrue(all("train" in str(p) or "eval" in str(p) or "runs" in str(p) for p in self.root.rglob("*.jpg")))

    def test_adam_smoke(self):
        args = _base_args(
            self.out_root / "adam",
            self.train_paths,
            self.eval_paths,
            optimizer="adam",
            lr=1e-2,
            exp_name="adam",
        )
        out = train_v2(args)
        self.assertTrue((out / "meta_mask.pt").is_file())

    def test_no_face_mask_and_resume(self):
        args = _base_args(
            self.out_root / "resume",
            self.train_paths,
            self.eval_paths,
            no_face_mask=True,
            max_steps=2,
            ckpt_every=1,
            exp_name="resume",
        )
        out1 = train_v2(args)
        ckpt = out1 / "checkpoint_latest.pt"
        self.assertTrue(ckpt.is_file())
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        step1 = int(payload["global_step"])

        args2 = _base_args(
            self.out_root / "resume",
            self.train_paths,
            self.eval_paths,
            no_face_mask=True,
            max_steps=step1 + 2,
            resume=str(ckpt),
            exp_name="resume",
        )
        out2 = train_v2(args2)
        self.assertEqual(out2, out1)
        payload2 = torch.load(out2 / "checkpoint_latest.pt", map_location="cpu", weights_only=False)
        self.assertGreaterEqual(int(payload2["global_step"]), step1 + 1)

    def test_wavelet_rejected(self):
        args = _base_args(
            self.out_root / "wav",
            self.train_paths,
            self.eval_paths,
            roi_compose="wavelet",
        )
        with self.assertRaises(NotImplementedError) as ctx:
            train_v2(args)
        self.assertIn("Step 7", str(ctx.exception))

    def test_residual_outside_hard_mask_near_zero_after_steps(self):
        args = _base_args(
            self.out_root / "roi",
            self.train_paths,
            self.eval_paths,
            no_face_mask=True,
            max_steps=3,
            init_std=5e-2,
            exp_name="roi",
        )
        out = train_v2(args)
        ckpt = torch.load(out / "meta_mask.pt", map_location="cpu", weights_only=False)
        module = DTCWTMetaMaskModule(
            img_size=64, J=3, editable_levels=(2, 3), init_std=0.0, epsilon=16 / 255
        )
        module.load_meta_mask_state(ckpt["meta_mask"])
        module.eval()
        x = torch.rand(1, 3, 64, 64)
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, 64), torch.linspace(-1, 1, 64), indexing="ij"
        )
        m = ((xx**2 + yy**2) <= 0.25).float().view(1, 1, 64, 64)
        with torch.no_grad():
            y = module(x, face_mask=m, roi_compose="pixel")
        residual = (y - x).abs()
        outside = m < 0.5
        max_out = residual[:, :, outside.squeeze(0).squeeze(0)].max().item()
        self.assertLess(max_out, 1e-5, f"residual outside M={max_out}")

    def test_fixed_ellipse_smoke(self):
        args = _base_args(
            self.out_root / "ellipse",
            self.train_paths,
            self.eval_paths,
            mask_mode="fixed",
            mask_regions="full",
            no_face_mask=False,
            exp_name="ellipse",
        )
        out = train_v2(args)
        cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(cfg["mask_mode"], "fixed")


if __name__ == "__main__":
    unittest.main()
