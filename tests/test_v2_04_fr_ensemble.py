"""V2 Step 4 — FR presets, disjoint train/eval, FaceNet smoke (+ skip heavy)."""

from __future__ import annotations

import unittest

import torch

from models.fr import (
    BASIC_POOL,
    BUILTIN_FACENET_NAMES,
    DEFAULT_EVAL_FR_NAMES,
    FULL_TRAIN_FR_NAMES,
    SMALL_TRAIN_FR_NAMES,
    SMOKE_FR_NAMES,
    SPECIAL_POOL,
    assert_train_eval_disjoint,
    ensemble_cosine_mean,
    fr_requires_protego,
    load_fr_ensemble,
    load_fr_model,
    resolve_eval_fr_names,
    resolve_train_fr_names,
)
from models.fr.protego_bridge import protego_available


class TestFRPresets(unittest.TestCase):
    def test_small_has_three_expected_names(self):
        names = resolve_train_fr_names("small")
        self.assertEqual(names, SMALL_TRAIN_FR_NAMES)
        self.assertEqual(
            names,
            (
                "ir50_arcface_casia",
                "mobilefacenet_arcface_casia",
                "ir18_adaface_webface",
            ),
        )

    def test_full_is_basic_without_intruder(self):
        names = resolve_train_fr_names("full")
        self.assertEqual(names, FULL_TRAIN_FR_NAMES)
        self.assertEqual(len(names), 9)
        self.assertNotIn("ir50_adaface_casia", names)
        self.assertTrue(set(names).issubset(set(BASIC_POOL)))

    def test_override_csv(self):
        names = resolve_train_fr_names(
            "full", fr_models="inception_facenet_vgg,inception_facenet_casia"
        )
        self.assertEqual(names, SMOKE_FR_NAMES)

    def test_default_eval_disjoint_from_small_and_full(self):
        eval_names = resolve_eval_fr_names(None)
        self.assertEqual(eval_names, DEFAULT_EVAL_FR_NAMES)
        assert_train_eval_disjoint(SMALL_TRAIN_FR_NAMES, eval_names)
        assert_train_eval_disjoint(FULL_TRAIN_FR_NAMES, eval_names)
        self.assertTrue(set(eval_names).issubset(set(SPECIAL_POOL)))

    def test_overlap_raises(self):
        with self.assertRaises(ValueError) as ctx:
            assert_train_eval_disjoint(
                ("inception_facenet_vgg", "ir50_arcface_casia"),
                ("inception_facenet_vgg",),
            )
        self.assertIn("overlap", str(ctx.exception).lower())

    def test_small_requires_protego_backend(self):
        self.assertTrue(fr_requires_protego(SMALL_TRAIN_FR_NAMES))
        self.assertFalse(fr_requires_protego(SMOKE_FR_NAMES))


class TestFaceNetSmoke(unittest.TestCase):
    """Always-on smoke: facenet-pytorch only (not a production preset)."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.x = torch.rand(2, 3, 160, 160)
        cls.device = torch.device("cpu")

    def test_load_builtin_facenet_frozen(self):
        for name in BUILTIN_FACENET_NAMES:
            m = load_fr_model(name, device=self.device)
            self.assertEqual(m.name, name)
            for p in m.parameters():
                self.assertFalse(p.requires_grad)
            feat = m(self.x)
            self.assertEqual(feat.ndim, 2)
            self.assertEqual(feat.shape[0], 2)

    def test_ensemble_smoke_forward_and_grad(self):
        models = load_fr_ensemble(SMOKE_FR_NAMES, device=self.device, keep_on_device=True)
        self.assertEqual(len(models), 2)
        x_orig = self.x.detach()
        x_prot = self.x.clone().requires_grad_(True)
        # Mild perturbation so cos < 1 but graph exists.
        x_prot_in = (x_prot + 0.05).clamp(0, 1)
        mean_cos, _ = ensemble_cosine_mean(
            models, x_orig, x_prot_in, self.device, offload=False
        )
        self.assertTrue(torch.isfinite(mean_cos))
        mean_cos.backward()
        self.assertIsNotNone(x_prot.grad)
        self.assertGreater(x_prot.grad.abs().sum().item(), 0.0)

    def test_missing_protego_raises_clear_error(self):
        if protego_available():
            self.skipTest("PROTEGO_FR_ROOT present; skip missing-backend check")
        with self.assertRaises(FileNotFoundError) as ctx:
            load_fr_model("ir50_arcface_casia", device="cpu")
        msg = str(ctx.exception)
        self.assertIn("PROTEGO_FR_ROOT", msg)
        self.assertIn("inception_facenet", msg)


@unittest.skipUnless(protego_available(), "Protego FR_DB not configured")
class TestProtegoHeavy(unittest.TestCase):
    def test_load_one_small_member(self):
        m = load_fr_model(SMALL_TRAIN_FR_NAMES[0], device="cpu")
        x = torch.rand(1, 3, 112, 112)
        feat = m(x)
        self.assertEqual(feat.shape[0], 1)


if __name__ == "__main__":
    unittest.main()
