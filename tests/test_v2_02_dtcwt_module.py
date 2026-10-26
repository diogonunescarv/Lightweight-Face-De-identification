"""V2 Step 2 — DTCWTMetaMaskModule: reconstruction, grads, editable_levels."""

from __future__ import annotations

import unittest
import warnings

import torch

from models.dtcwt_meta_mask import (
    DEFAULT_EDITABLE_LEVELS,
    DTCWTMetaMaskModule,
    parse_editable_levels,
)


class TestV2DtcwtMetaMask(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.img_size = 64  # smaller for speed; J=3 → L3 spatial = 8
        self.x = torch.rand(2, 3, self.img_size, self.img_size)

    def test_parse_editable_levels(self):
        self.assertEqual(parse_editable_levels("2,3", 3), (2, 3))
        self.assertEqual(parse_editable_levels((2,), 3), (2,))
        with self.assertRaises(ValueError):
            parse_editable_levels("", 3)
        with self.assertRaises(ValueError):
            parse_editable_levels("2,2", 3)
        with self.assertRaises(ValueError):
            parse_editable_levels("4", 3)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            levels = parse_editable_levels("1,2,3", 3)
        self.assertEqual(levels, (1, 2, 3))
        self.assertTrue(any("nível 1" in str(w.message) for w in caught))

    def test_delta_shapes_for_editable_levels(self):
        for levels in ((2,), (3,), (2, 3)):
            m = DTCWTMetaMaskModule(
                img_size=self.img_size, J=3, editable_levels=levels, init_std=0.0
            )
            self.assertEqual(m.editable_levels, levels)
            self.assertEqual(tuple(sorted(m.delta_mag.keys())), tuple(str(l) for l in levels))
            for level in levels:
                h = self.img_size // (2**level)
                self.assertEqual(
                    tuple(m.delta_mag[str(level)].shape),
                    (1, 3, 6, h, h),
                )

    def test_zero_delta_near_identity(self):
        m = DTCWTMetaMaskModule(
            img_size=self.img_size, J=3, editable_levels=(2, 3), init_std=0.0
        )
        with torch.no_grad():
            y = m(self.x, face_mask=None, roi_compose="pixel")
        # Perfect reconstruction of DT-CWT is near-exact; L∞ path with ΔW=0 → identity.
        err = (y - self.x).abs().max().item()
        self.assertLess(err, 1e-4, f"max |y-x|={err}")

    def test_gradients_flow_to_deltas_only_editable(self):
        m = DTCWTMetaMaskModule(
            img_size=self.img_size,
            J=3,
            editable_levels=(2, 3),
            init_std=1e-2,
        )
        y = m(self.x, face_mask=None, roi_compose="pixel")
        loss = y.mean()
        loss.backward()
        for level in (2, 3):
            g_mag = m.delta_mag[str(level)].grad
            g_ph = m.delta_phase[str(level)].grad
            self.assertIsNotNone(g_mag)
            self.assertIsNotNone(g_ph)
            self.assertGreater(g_mag.abs().sum().item() + g_ph.abs().sum().item(), 0.0)
        self.assertEqual(m.frozen_levels, (1,))
        self.assertNotIn("1", m.delta_mag)

    def test_wavelet_compose_not_implemented(self):
        m = DTCWTMetaMaskModule(img_size=self.img_size, J=3, init_std=0.0)
        with self.assertRaises(NotImplementedError) as ctx:
            m(self.x, roi_compose="wavelet")
        self.assertIn("Step 7", str(ctx.exception))

    def test_default_editable_levels(self):
        self.assertEqual(DEFAULT_EDITABLE_LEVELS, (2, 3))

    def test_meta_mask_state_roundtrip(self):
        m = DTCWTMetaMaskModule(
            img_size=self.img_size, J=3, editable_levels=(2,), init_std=1e-2
        )
        state = m.meta_mask_state()
        m2 = DTCWTMetaMaskModule(
            img_size=self.img_size, J=3, editable_levels=(2,), init_std=0.0
        )
        m2.load_meta_mask_state(state)
        self.assertTrue(torch.allclose(m.delta_mag["2"], m2.delta_mag["2"]))
        self.assertTrue(torch.allclose(m.delta_phase["2"], m2.delta_phase["2"]))


if __name__ == "__main__":
    unittest.main()
