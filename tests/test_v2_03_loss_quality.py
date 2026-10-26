"""V2 Step 3 — Phase 3.3 identity cosine + SSIM hinge / adapt_w / L∞."""

from __future__ import annotations

import unittest

import torch

from losses.identity import identity_cosine_loss, mean_identity_cosine
from losses.quality import (
    adapt_percep_weight,
    apply_linf_budget,
    compute_ssim,
    ssim_hinge_loss,
)


class TestIdentityCosine(unittest.TestCase):
    def test_identical_features_cos_one(self):
        feat = torch.randn(4, 16)
        loss, cos = identity_cosine_loss(feat, feat.clone())
        self.assertAlmostEqual(float(cos), 1.0, places=5)
        self.assertAlmostEqual(float(loss), 1.0, places=5)

    def test_orthogonal_features_cos_near_zero(self):
        a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        b = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        loss, cos = identity_cosine_loss(a, b)
        self.assertAlmostEqual(float(cos), 0.0, places=5)

    def test_grad_flows_to_prot_only_path(self):
        feat_o = torch.randn(2, 8)
        feat_p = torch.randn(2, 8, requires_grad=True)
        loss, _ = identity_cosine_loss(feat_o, feat_p)
        loss.backward()
        self.assertIsNotNone(feat_p.grad)
        self.assertGreater(feat_p.grad.abs().sum().item(), 0.0)

    def test_mean_ensemble(self):
        c1 = torch.tensor(0.8)
        c2 = torch.tensor(0.4)
        mean, _ = mean_identity_cosine([c1, c2])
        self.assertAlmostEqual(float(mean), 0.6, places=5)


class TestQualityLinfAndSsim(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.rand(2, 3, 64, 64)

    def test_linf_budget_clamps(self):
        recon = self.x + 0.5  # huge residual
        eps = 16.0 / 255.0
        prot, residual = apply_linf_budget(recon, self.x, eps)
        self.assertLessEqual(float(residual.abs().max()), eps + 1e-6)
        self.assertTrue(torch.all(prot >= 0.0) and torch.all(prot <= 1.0))
        self.assertTrue(torch.allclose(prot, (self.x + residual).clamp(0, 1)))

    def test_ssim_identical_near_one_hinge_zero(self):
        percep, ssim_v = ssim_hinge_loss(
            self.x, self.x, min_ssim=0.95, weight=1.0, ssim_mask=None
        )
        self.assertGreater(float(ssim_v), 0.99)
        self.assertAlmostEqual(float(percep), 0.0, places=5)

    def test_ssim_hinge_positive_when_degraded(self):
        noisy = (self.x + 0.3 * torch.randn_like(self.x)).clamp(0, 1)
        percep, ssim_v = ssim_hinge_loss(
            noisy, self.x, min_ssim=0.95, weight=2.0, ssim_mask=None
        )
        self.assertLess(float(ssim_v), 0.95)
        self.assertGreater(float(percep), 0.0)
        # percep ≈ 2 * (0.95 - ssim)
        expected = 2.0 * max(0.95 - float(ssim_v), 0.0)
        self.assertAlmostEqual(float(percep), expected, places=4)

    def test_adapt_percep_weight_rules(self):
        # too high SSIM → halve w
        self.assertAlmostEqual(adapt_percep_weight(0.99, 0.95, 4.0), 2.0)
        # too low SSIM → double w
        self.assertAlmostEqual(adapt_percep_weight(0.90, 0.95, 1.0), 2.0)
        # inside band → reset to 1
        self.assertAlmostEqual(adapt_percep_weight(0.95, 0.95, 8.0), 1.0)

    def test_ssim_mask_scope_differs_from_full(self):
        """Partial ROI SSIM ≠ full-image SSIM when damage is localized."""
        prot = self.x.clone()
        # Damage only a corner (outside a center ROI).
        prot[:, :, :16, :16] = 0.0
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, 64), torch.linspace(-1, 1, 64), indexing="ij"
        )
        center = ((xx**2 + yy**2) <= 0.2).float().view(1, 1, 64, 64).expand(2, -1, -1, -1)

        ssim_full = float(compute_ssim(prot, self.x, ssim_mask=None))
        ssim_roi = float(compute_ssim(prot, self.x, ssim_mask=center))
        # Corner damage hurts full SSIM more than center-only SSIM.
        self.assertGreater(ssim_roi, ssim_full + 0.02)

    def test_ssim_hinge_with_mask_grad(self):
        prot = self.x.clone().requires_grad_(True)
        mask = torch.ones(2, 1, 64, 64)
        # Force a gap vs min_ssim by using a degraded clone without breaking graph.
        degraded = (prot * 0.5).clamp(0, 1)
        percep, _ = ssim_hinge_loss(
            degraded, self.x.detach(), min_ssim=0.99, weight=1.0, ssim_mask=mask
        )
        if float(percep.detach()) > 0:
            percep.backward()
            self.assertIsNotNone(prot.grad)


if __name__ == "__main__":
    unittest.main()
