"""V2 Step 5 — merged LFW+CelebA frac split + image-size logging."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from data.merged_split import (
    MERGED_DATASET,
    build_flat_data_dict,
    build_merged_frac_split,
    build_split_config_block,
    list_image_paths,
    make_face_dataloader,
    observe_batch_hw,
    sample_native_hw,
)


def _write_fake_jpg(path: Path, size: tuple[int, int] = (80, 64), seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    h, w = size
    arr = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(path, format="JPEG")


class TestMergedFracSplit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.lfw = root / "lfw_frontal"
        self.celeba = root / "celebahq_pp_frontal"
        (self.lfw / "faces_train").mkdir(parents=True)
        (self.lfw / "faces_test").mkdir(parents=True)
        self.celeba.mkdir(parents=True)

        for i in range(6):
            _write_fake_jpg(self.lfw / "faces_train" / f"tr_{i:02d}.jpg", seed=i)
        for i in range(4):
            _write_fake_jpg(self.lfw / "faces_test" / f"te_{i:02d}.jpg", seed=100 + i)
        for i in range(10):
            # Vary native sizes for logging sample.
            hw = (64 + (i % 3) * 8, 64 + (i % 2) * 16)
            _write_fake_jpg(self.celeba / f"c_{i:02d}.jpg", size=hw, seed=200 + i)

    def tearDown(self):
        self.tmp.cleanup()

    def test_reproducible_same_seed(self):
        kwargs = dict(
            celeba_train_frac=0.5,
            data_seed=42,
            lfw_root=str(self.lfw),
            celeba_root=str(self.celeba),
        )
        t1, e1, s1 = build_merged_frac_split(**kwargs)
        t2, e2, s2 = build_merged_frac_split(**kwargs)
        self.assertEqual(t1, t2)
        self.assertEqual(e1, e2)
        self.assertEqual(s1, s2)

    def test_different_seed_changes_celeba_partition(self):
        t_a, e_a, _ = build_merged_frac_split(
            celeba_train_frac=0.5,
            data_seed=1,
            lfw_root=str(self.lfw),
            celeba_root=str(self.celeba),
        )
        t_b, e_b, _ = build_merged_frac_split(
            celeba_train_frac=0.5,
            data_seed=2,
            lfw_root=str(self.lfw),
            celeba_root=str(self.celeba),
        )
        # Same LFW counts but CelebA assignment / train shuffle differ.
        self.assertNotEqual(t_a, t_b)

    def test_stats_and_frac(self):
        train, eval_paths, stats = build_merged_frac_split(
            celeba_train_frac=0.5,
            data_seed=42,
            lfw_root=str(self.lfw),
            celeba_root=str(self.celeba),
        )
        self.assertEqual(stats["n_lfw_train"], 6)
        self.assertEqual(stats["n_lfw_test"], 4)
        self.assertEqual(stats["n_celeba_total"], 10)
        self.assertEqual(stats["n_celeba_train"], 5)
        self.assertEqual(stats["n_celeba_eval"], 5)
        self.assertEqual(stats["n_train_total"], 6 + 5)
        self.assertEqual(stats["n_eval_total"], 4 + 5)
        self.assertEqual(len(train), stats["n_train_total"])
        self.assertEqual(len(eval_paths), stats["n_eval_total"])
        # Absolute paths
        self.assertTrue(all(Path(p).is_absolute() for p in train + eval_paths))
        # Disjoint celeba partitions
        celeba_train = [p for p in train if "/celebahq_pp_frontal/" in p or str(self.celeba) in p]
        celeba_eval = [p for p in eval_paths if str(self.celeba) in p]
        self.assertEqual(len(set(celeba_train) & set(celeba_eval)), 0)

    def test_max_train_cap(self):
        train, _, stats = build_merged_frac_split(
            celeba_train_frac=1.0,
            data_seed=0,
            max_train=3,
            lfw_root=str(self.lfw),
            celeba_root=str(self.celeba),
        )
        self.assertEqual(len(train), 3)
        self.assertEqual(stats["n_train_total"], 3)

    def test_missing_dir_raises_clear_error(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            build_merged_frac_split(
                lfw_root="/tmp/does_not_exist_lfw_v2",
                celeba_root=str(self.celeba),
            )
        self.assertIn("missing", str(ctx.exception).lower())

    def test_invalid_frac(self):
        with self.assertRaises(ValueError):
            build_merged_frac_split(
                celeba_train_frac=0.0,
                lfw_root=str(self.lfw),
                celeba_root=str(self.celeba),
            )

    def test_flat_data_dict_and_config_block(self):
        data = build_flat_data_dict(
            MERGED_DATASET,
            celeba_train_frac=0.4,
            data_seed=7,
            max_eval=3,
            lfw_root=str(self.lfw),
            celeba_root=str(self.celeba),
        )
        self.assertEqual(data["dataset"], MERGED_DATASET)
        self.assertIn("split_stats", data)
        self.assertEqual(len(data["eval"]), 3)
        native = sample_native_hw(data["train"], max_samples=4)
        self.assertEqual(len(native), 4)
        self.assertTrue(all("x" in s for s in native))
        cfg = build_split_config_block(
            data["split_stats"],
            image_size=224,
            data_seed=7,
            celeba_train_frac=0.4,
            dataset_native_hw_sample=native,
        )
        self.assertEqual(cfg["image_size"], 224)
        self.assertEqual(cfg["n_train"], data["split_stats"]["n_train_total"])
        self.assertEqual(cfg["dataset_native_hw_sample"], native)

    def test_dataloader_image_size_logging(self):
        train, _, _ = build_merged_frac_split(
            celeba_train_frac=0.5,
            data_seed=42,
            max_train=4,
            lfw_root=str(self.lfw),
            celeba_root=str(self.celeba),
        )
        loader = make_face_dataloader(train, batch_size=2, image_size=32, shuffle=False)
        batch = next(iter(loader))
        x, paths, source_hw = batch
        # DataLoader collates source_hw as list/tuple of tensors or nested — accept both.
        self.assertEqual(tuple(x.shape), (2, 3, 32, 32))
        hw = observe_batch_hw(x)
        self.assertEqual(hw["batch_hw"], "32x32")
        self.assertEqual(hw["batch_h"], 32)
        self.assertEqual(hw["image_size"], 32)
        self.assertEqual(len(paths), 2)


@unittest.skipUnless(
    Path("/mnt/databases/faces/lfw_frontal/faces_train").is_dir()
    and Path("/mnt/databases/faces/celebahq_pp_frontal").is_dir(),
    "real frontal datasets not mounted",
)
class TestMergedSplitRealRoots(unittest.TestCase):
    def test_real_roots_smoke_stats(self):
        train, eval_paths, stats = build_merged_frac_split(
            celeba_train_frac=0.5,
            data_seed=42,
            max_train=16,
        )
        self.assertGreater(stats["n_lfw_train"], 0)
        self.assertGreater(stats["n_celeba_total"], 0)
        self.assertEqual(len(train), 16)
        self.assertTrue(Path(train[0]).is_file())
        # Reproducibility on real data
        train2, _, _ = build_merged_frac_split(
            celeba_train_frac=0.5, data_seed=42, max_train=16
        )
        self.assertEqual(train, train2)


if __name__ == "__main__":
    unittest.main()
