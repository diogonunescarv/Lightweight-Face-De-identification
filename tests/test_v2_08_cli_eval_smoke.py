"""V2 Step 8 — e2e smoke: train pixel + wavelet + evaluate-v2 (16 imgs)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from training.trainer_v2 import train_v2

ROOT = Path(__file__).resolve().parents[1]


def _write_jpgs(folder: Path, n: int = 16, size: int = 64) -> list[str]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        rng = np.random.default_rng(i)
        arr = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
        p = folder / f"img_{i:03d}.jpg"
        Image.fromarray(arr, mode="RGB").save(p)
        paths.append(str(p.resolve()))
    return paths


def _train_args(out_root: Path, paths: list[str], roi: str, exp: str) -> argparse.Namespace:
    return argparse.Namespace(
        out_root=str(out_root),
        exp_name=exp,
        device="cpu",
        image_size=64,
        batch_size=4,
        num_workers=0,
        epochs=1,
        max_steps=2,
        optimizer="protego",
        lr=None,
        epsilon=16 / 255,
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
        roi_compose=roi,
        no_face_mask=False,
        mask_mode="fixed",
        mask_shape="ellipse",
        mask_regions="full",
        ssim_region="mask",
        early_stopping=False,
        early_stopping_metric="cos",
        patience=2,
        min_delta=0.005,
        val_max_samples=8,
        ckpt_every=1,
        preview_every=0,
        resume=None,
        train_paths=paths,
        eval_paths=paths[:4],
        lfw_root=None,
        celeba_root=None,
        max_train=None,
    )


class TestV2Step8E2E(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.paths = _write_jpgs(self.root / "faces", n=16, size=64)
        self.out_root = self.root / "runs"

    def tearDown(self):
        self.tmp.cleanup()

    def test_pixel_wavelet_and_evaluate(self):
        out_pix = train_v2(_train_args(self.out_root, self.paths, "pixel", "e2e_pix"))
        out_wav = train_v2(_train_args(self.out_root, self.paths, "wavelet", "e2e_wav"))

        for out, roi in ((out_pix, "pixel"), (out_wav, "wavelet")):
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "deid_optimize.py"),
                    "evaluate-v2",
                    "--checkpoint",
                    str(out / "meta_mask.pt"),
                    "--data",
                    str(self.root / "faces"),
                    "--run-dir",
                    str(out),
                    "--device",
                    "cpu",
                    "--batch-size",
                    "4",
                    "--max-samples",
                    "16",
                    "--no-lpips",
                    "--mask-mode",
                    "fixed",
                    "--ssim-region",
                    "mask",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
            summary = json.loads((out / "eval" / "eval_summary.json").read_text())
            self.assertEqual(summary["n_images"], 16)
            self.assertEqual(summary["roi_compose"], roi)
            self.assertIn("cos_mean", summary)
            self.assertIn("ssim_mean", summary)
            self.assertTrue((out / "eval" / "eval_metrics.csv").is_file())

    def test_cli_help_evaluate_v2(self):
        proc = subprocess.run(
            [sys.executable, str(ROOT / "deid_optimize.py"), "evaluate-v2", "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("evaluate-v2", proc.stdout)


if __name__ == "__main__":
    unittest.main()
