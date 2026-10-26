"""V2 Step 7 — Ideia 2 roi_compose=wavelet (no regression on pixel)."""

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from models.dtcwt_meta_mask import DTCWTMetaMaskModule
from training.trainer_v2 import train_v2


def _hard_center_mask(size: int, radius: float = 0.25) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, size),
        torch.linspace(-1, 1, size),
        indexing="ij",
    )
    return ((xx**2 + yy**2) <= radius).float().view(1, 1, size, size)


def _write_jpgs(root: Path, n: int = 6, size: int = 64) -> list[str]:
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        p = root / f"img_{i:03d}.jpg"
        rng = np.random.default_rng(i)
        arr = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
        Image.fromarray(arr, mode="RGB").save(p, format="JPEG")
        paths.append(str(p.resolve()))
    return paths


class TestRoiComposeWavelet(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.size = 64
        self.x = torch.rand(1, 3, self.size, self.size)
        self.module = DTCWTMetaMaskModule(
            img_size=self.size,
            J=3,
            editable_levels=(2, 3),
            init_std=5e-2,
            epsilon=16.0 / 255.0,
        )

    def test_delta_energy_outside_m_level_near_zero(self):
        m = _hard_center_mask(self.size, radius=0.2)
        with torch.no_grad():
            yl0, yh0 = self.module.xfm(self.x)
            _prot, _yl, yh_out = self.module(
                self.x, face_mask=m, roi_compose="wavelet", return_coeffs=True
            )
        for j, (b0, b1) in enumerate(zip(yh0, yh_out)):
            level = j + 1
            if level not in self.module.editable_levels:
                self.assertTrue(torch.allclose(b0, b1, atol=1e-6))
                continue
            h_l, w_l = int(b0.shape[-3]), int(b0.shape[-2])
            m_l = self.module._downsample_roi_to_level(m, h_l, w_l)
            delta = (b1 - b0).abs()  # (N,C,6,H,W,2)
            # Bilinear downsample softens edges; assert only where M_ℓ≈0.
            outside = m_l < 1e-3
            inside = m_l > 0.5
            mask_out = outside[:, :, None, :, :, None]
            mask_in = inside[:, :, None, :, :, None]
            if mask_out.any():
                max_out = delta[mask_out.expand_as(delta)].max().item()
                self.assertLess(
                    max_out,
                    1e-5,
                    f"level {level}: ΔW energy where M_ℓ≈0 max={max_out}",
                )
            if mask_out.any() and mask_in.any():
                mean_out = delta[mask_out.expand_as(delta)].mean().item()
                mean_in = delta[mask_in.expand_as(delta)].mean().item()
                self.assertLess(
                    mean_out,
                    max(mean_in * 0.05, 1e-6),
                    f"level {level}: mean_out={mean_out} not ≪ mean_in={mean_in}",
                )

    def test_no_face_mask_pixel_approx_wavelet(self):
        m = DTCWTMetaMaskModule(
            img_size=self.size,
            J=3,
            editable_levels=(2, 3),
            init_std=1e-2,
            epsilon=16.0 / 255.0,
        )
        # Same weights for both compose modes.
        with torch.no_grad():
            y_pix = m(self.x, face_mask=None, roi_compose="pixel")
            y_wav = m(self.x, face_mask=None, roi_compose="wavelet")
        # M≡1: pixel residual⊙M ≡ full residual path ≈ wavelet path (same ΔW).
        self.assertTrue(torch.allclose(y_pix, y_wav, atol=1e-5, rtol=1e-4))

    def test_wavelet_grads_flow(self):
        m = _hard_center_mask(self.size)
        y = self.module(self.x, face_mask=m, roi_compose="wavelet")
        y.mean().backward()
        g = self.module.delta_mag["2"].grad
        self.assertIsNotNone(g)
        self.assertGreater(g.abs().sum().item(), 0.0)

    def test_invalid_compose_raises(self):
        with self.assertRaises(ValueError):
            self.module(self.x, roi_compose="nope")


class TestWaveletTrainSmoke(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.train_paths = _write_jpgs(self.root / "train", n=6)
        self.eval_paths = _write_jpgs(self.root / "eval", n=4)
        self.out_root = self.root / "runs"

    def tearDown(self):
        self.tmp.cleanup()

    def _args(self, **kw):
        d = dict(
            out_root=str(self.out_root),
            exp_name="wav",
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
            roi_compose="wavelet",
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
            preview_every=0,
            resume=None,
            train_paths=self.train_paths,
            eval_paths=self.eval_paths,
            lfw_root=None,
            celeba_root=None,
            max_train=None,
        )
        d.update(kw)
        return argparse.Namespace(**d)

    def test_protego_wavelet_smoke(self):
        out = train_v2(self._args(optimizer="protego"))
        self.assertTrue((out / "meta_mask.pt").is_file())
        cfg = (out / "config.json").read_text(encoding="utf-8")
        self.assertIn('"roi_compose": "wavelet"', cfg)

    def test_adam_wavelet_smoke(self):
        out = train_v2(self._args(optimizer="adam", lr=1e-2, exp_name="adam_wav"))
        self.assertTrue((out / "meta_mask.pt").is_file())


if __name__ == "__main__":
    unittest.main()
