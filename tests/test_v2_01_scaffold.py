"""V2 Step 1 — scaffold: stubs importable; CLI recognizes train-v2."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestV2Step1Scaffold(unittest.TestCase):
    def test_v2_stub_imports(self):
        from models.dtcwt_meta_mask import DTCWTMetaMaskModule, add_complex_meta_mask
        from models.fr import load_fr_ensemble
        from losses.identity import identity_cosine_loss
        from losses.quality import adapt_percep_weight, apply_linf_budget, ssim_hinge_loss
        from data.merged_split import build_merged_frac_split
        from training.trainer_v2 import train_v2
        from evaluation.validate_v2 import validate_v2

        self.assertIsNotNone(DTCWTMetaMaskModule)
        self.assertTrue(callable(add_complex_meta_mask))
        self.assertTrue(callable(load_fr_ensemble))
        self.assertTrue(callable(identity_cosine_loss))
        self.assertTrue(callable(ssim_hinge_loss))
        self.assertTrue(callable(adapt_percep_weight))
        self.assertTrue(callable(apply_linf_budget))
        self.assertTrue(callable(build_merged_frac_split))
        self.assertTrue(callable(train_v2))
        self.assertTrue(callable(validate_v2))

    def test_v2_remaining_stubs_raise_not_implemented(self):
        from training.trainer_v2 import train_v2

        with self.assertRaises(NotImplementedError):
            train_v2()

    def test_mask_token_coverage_v1_parity(self):
        """ROI tokens from V2.md §3.1.1 must parse with existing masks.py."""
        from models.masks import (
            COMPOSITE_REGIONS,
            UNITARY_REGIONS,
            VALID_REGIONS,
            VALID_SHAPES,
            parse_mask_regions,
            parse_mask_shape,
            validate_mask_shape_regions,
        )

        unitary = ("eyes", "nose", "mouth", "full")
        composite = (
            "eyes-nose",
            "eyes-mouth",
            "nose-mouth",
            "eyes-nose-mouth",
            "eyes-nose-mouth-hybrid",
        )
        self.assertEqual(set(unitary), set(UNITARY_REGIONS))
        self.assertTrue(set(composite) <= set(COMPOSITE_REGIONS))
        self.assertEqual(set(unitary) | set(composite), set(VALID_REGIONS))
        self.assertEqual(VALID_SHAPES, frozenset({"ellipse", "band"}))

        for tok in unitary + composite:
            parse_mask_regions(tok)
        parse_mask_shape("ellipse")
        parse_mask_shape("band")
        validate_mask_shape_regions("ellipse", ("full",))
        validate_mask_shape_regions("band", ("eyes-nose-mouth",))
        with self.assertRaises(ValueError):
            validate_mask_shape_regions("band", ("full",))
        with self.assertRaises(ValueError):
            validate_mask_shape_regions("ellipse", ("eyes-nose",))

    def test_cli_train_v2_recognized_and_exits_nonzero(self):
        proc = subprocess.run(
            [sys.executable, str(ROOT / "deid_optimize.py"), "train-v2", "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue("train-v2" in proc.stdout or "V2" in proc.stdout)

        proc2 = subprocess.run(
            [sys.executable, str(ROOT / "deid_optimize.py"), "train-v2"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(proc2.returncode, 0)
        combined = (proc2.stderr + proc2.stdout).lower()
        self.assertIn("not implemented", combined)


if __name__ == "__main__":
    unittest.main()
