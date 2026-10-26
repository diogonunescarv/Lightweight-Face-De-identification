"""V2 Step 2b — Ideia 1: residual ⊙ M (pixel ROI compose) + L∞."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from models.dtcwt_meta_mask import DTCWTMetaMaskModule
from models.masks import (
    build_band_region_mask,
    build_region_mask,
    fixed_ellipse_face_mask,
    parse_mask_regions,
    parse_mask_shape,
    resolve_face_mask_by_size,
    validate_mask_shape_regions,
)


def _synthetic_landmarks(size: int = 64) -> np.ndarray:
    """Rough 5-point face landmarks for a ``size×size`` crop."""
    s = float(size)
    return np.array(
        [
            [0.35 * s, 0.38 * s],
            [0.65 * s, 0.38 * s],
            [0.50 * s, 0.52 * s],
            [0.38 * s, 0.70 * s],
            [0.62 * s, 0.70 * s],
        ],
        dtype=np.float32,
    )


class TestV2RoiComposePixel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.size = 64
        self.x = torch.rand(1, 3, self.size, self.size)
        self.module = DTCWTMetaMaskModule(
            img_size=self.size,
            J=3,
            editable_levels=(2, 3),
            init_std=5e-2,
            epsilon=16.0 / 255.0,
        )

    def test_residual_outside_mask_near_zero(self):
        # Hard circular ROI in the center.
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, self.size),
            torch.linspace(-1, 1, self.size),
            indexing="ij",
        )
        m = ((xx**2 + yy**2) <= 0.25).float().view(1, 1, self.size, self.size)
        outside = m < 0.5

        y = self.module(self.x, face_mask=m, roi_compose="pixel")
        residual = (y - self.x).abs()
        max_out = residual[:, :, outside.squeeze(0).squeeze(0)].max().item()
        self.assertLess(max_out, 1e-5, f"residual outside M max={max_out}")

    def test_no_face_mask_equals_full_ones(self):
        m_ones = torch.ones(1, 1, self.size, self.size)
        y_none = self.module(self.x, face_mask=None, roi_compose="pixel")
        y_ones = self.module(self.x, face_mask=m_ones, roi_compose="pixel")
        self.assertTrue(torch.allclose(y_none, y_ones, atol=1e-6))

    def test_linf_budget_respected(self):
        m = torch.ones(1, 1, self.size, self.size)
        y = self.module(self.x, face_mask=m, roi_compose="pixel")
        linf = (y - self.x).abs().max().item()
        self.assertLessEqual(linf, self.module.epsilon + 1e-5)

    def _assert_soft_roi_bound(self, mask: torch.Tensor, y: torch.Tensor, label: str):
        """Soft masks: M≈0 ⇒ residual≈0; mean |Δ| outside ≪ inside."""
        residual = (y - self.x).abs()
        near_zero = mask < 1e-3
        if near_zero.any():
            max_out = residual[near_zero.expand_as(residual)].max().item()
            self.assertLess(max_out, 1e-4, f"{label} residual where M≈0: {max_out}")
        outside = mask < 0.1
        inside = mask > 0.5
        if outside.any() and inside.any():
            mean_out = residual[outside.expand_as(residual)].mean().item()
            mean_in = residual[inside.expand_as(residual)].mean().item()
            self.assertLess(
                mean_out,
                max(mean_in * 0.25, 1e-5),
                f"{label}: mean_out={mean_out} not ≪ mean_in={mean_in}",
            )

    def test_ellipse_full_landmarks_restricts_residual(self):
        lm = _synthetic_landmarks(self.size)
        mask = build_region_mask(lm, ("full",), self.size, self.size, torch.device("cpu"))
        mask = mask.view(1, 1, self.size, self.size)
        y = self.module(self.x, face_mask=mask, roi_compose="pixel")
        self._assert_soft_roi_bound(mask, y, "ellipse-full")

    def test_band_eyes_nose_mouth_restricts_residual(self):
        lm = _synthetic_landmarks(self.size)
        regions = parse_mask_regions("eyes-nose-mouth")
        validate_mask_shape_regions("band", regions)
        mask = build_band_region_mask(
            lm, regions, self.size, self.size, torch.device("cpu")
        )
        mask = mask.view(1, 1, self.size, self.size)
        y = self.module(self.x, face_mask=mask, roi_compose="pixel")
        self._assert_soft_roi_bound(mask, y, "band-eyes-nose-mouth")

    def test_resolve_face_mask_by_size_fixed_no_transformer(self):
        mask = resolve_face_mask_by_size(
            batch_size=2,
            paths=["a.jpg", "b.jpg"],
            device=torch.device("cpu"),
            image_size=self.size,
            use_face_mask=True,
            mask_mode="fixed",
            mask_regions=("full",),
            mask_shape="ellipse",
        )
        self.assertEqual(tuple(mask.shape), (2, 1, self.size, self.size))
        fallback = fixed_ellipse_face_mask(self.size, torch.device("cpu"))
        self.assertTrue(torch.allclose(mask[0:1], fallback))

    def test_mask_flags_parse_for_roi_compose_pixel(self):
        parse_mask_shape("ellipse")
        parse_mask_shape("band")
        parse_mask_regions("full")
        parse_mask_regions("eyes-nose-mouth")
        validate_mask_shape_regions("ellipse", ("full",))
        validate_mask_shape_regions("band", ("eyes-nose-mouth",))


if __name__ == "__main__":
    unittest.main()
