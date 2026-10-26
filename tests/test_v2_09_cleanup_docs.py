"""V2 Step 9 — cleanup/docs: viz helpers, artefatos só no run, status docs."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from evaluation.visualize_v2 import render_meta_mask_panel
from training.trainer_v2 import train_v2

ROOT = Path(__file__).resolve().parents[1]


def _write_jpgs(folder: Path, n: int = 4, size: int = 64) -> list[str]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        rng = np.random.default_rng(i + 90)
        arr = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
        p = folder / f"img_{i:03d}.jpg"
        Image.fromarray(arr, mode="RGB").save(p)
        paths.append(str(p.resolve()))
    return paths


def _train_args(out_root: Path, paths: list[str]) -> argparse.Namespace:
    return argparse.Namespace(
        out_root=str(out_root),
        exp_name="s9_viz",
        device="cpu",
        image_size=64,
        batch_size=2,
        num_workers=0,
        epochs=1,
        max_steps=1,
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
        val_max_samples=2,
        ckpt_every=1,
        preview_every=0,
        resume=None,
        train_paths=paths,
        eval_paths=paths[:2],
        lfw_root=None,
        celeba_root=None,
        max_train=None,
    )


class TestV2Step9CleanupDocs(unittest.TestCase):
    def test_module_docstring_mentions_wavelet(self):
        text = (ROOT / "models" / "dtcwt_meta_mask.py").read_text(encoding="utf-8")
        self.assertIn('roi_compose="wavelet"', text)
        self.assertNotIn("intentionally unimplemented until Step 7", text)

    def test_readme_v2_status_and_roi(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("## Status V2", readme)
        self.assertIn("--roi-compose", readme)
        self.assertIn("train-v2", readme)
        self.assertIn("V2.1", readme)

    def test_decisions_has_step9_and_ablations(self):
        dec = (ROOT / "DECISIONS.md").read_text(encoding="utf-8")
        self.assertIn("V2 Step 9", dec)
        self.assertIn("abla", dec.lower())
        self.assertIn("V2.1", dec)

    def test_visualize_writes_only_under_run_viz(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _write_jpgs(root / "faces", n=4, size=64)
            out_root = root / "runs"
            run_dir = train_v2(_train_args(out_root, paths))
            viz_dir = run_dir / "viz"
            outside = root / "outside_should_stay_empty"
            outside.mkdir()

            man = render_meta_mask_panel(
                run_dir / "meta_mask.pt",
                paths[0],
                viz_dir,
                device="cpu",
                compare_compose=True,
                mask_mode="fixed",
            )
            self.assertTrue(Path(man["files"]["panel"]).is_file())
            self.assertTrue(Path(man["files"]["prot_pixel"]).is_file())
            self.assertTrue(Path(man["files"]["prot_wavelet"]).is_file())
            # All saved files under viz_dir
            for p in man["files"].values():
                self.assertTrue(
                    str(Path(p).resolve()).startswith(str(viz_dir.resolve())),
                    f"artifact escaped run: {p}",
                )
            # No new files outside the run
            self.assertEqual(list(outside.iterdir()), [])
            # Manifest present
            manifests = list(viz_dir.glob("*_manifest.json"))
            self.assertTrue(manifests)
            data = json.loads(manifests[0].read_text(encoding="utf-8"))
            self.assertEqual(data["out_dir"], str(viz_dir.resolve()))

    def test_visualize_script_exists(self):
        script = ROOT / "scripts" / "run_v2_visualize_meta_mask.py"
        self.assertTrue(script.is_file())
        text = script.read_text(encoding="utf-8")
        self.assertIn("render_meta_mask_panel", text)


if __name__ == "__main__":
    unittest.main()
