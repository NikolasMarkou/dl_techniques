"""Tests for ``dl_techniques.datasets.vision.image_folder_burst_dp``.

Covers:
  * File discovery for DIV2K + VGG-Face2 layouts.
  * Shape / dtype contract of ``__getitem__`` (zero seg labels included).
  * ``max_images`` slicing via factory.
  * ``aux_mask`` variability under ``sample_n_per_sample=True``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dl_techniques.datasets.vision.coco_burst_dp import (
    DistortionSpec,
    default_aux_spec,
)
from dl_techniques.datasets.vision.image_folder_burst_dp import (
    ImageFolderBurstDPConfig,
    ImageFolderBurstDPLoader,
    build_div2k_burst_dp_datasets,
    build_vggface2_burst_dp_datasets,
    discover_div2k_paths,
    discover_vggface2_paths,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_image(path: Path, h: int = 64, w: int = 64, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    Image.fromarray(arr).save(str(path))


@pytest.fixture
def div2k_root(tmp_path: Path) -> Path:
    root = tmp_path / "div2k"
    (root / "train").mkdir(parents=True)
    (root / "validation").mkdir(parents=True)
    for i in range(4):
        _make_image(root / "train" / f"{i:04d}.png", seed=i)
    for i in range(2):
        _make_image(root / "validation" / f"{i:04d}.png", seed=100 + i)
    return root


@pytest.fixture
def vggface2_root(tmp_path: Path) -> Path:
    root = tmp_path / "vggface2"
    train_dir = root / "train"
    test_dir = root / "test"
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    # Two identities x two files each, train + test.
    train_rels = []
    for ident in ("n000001", "n000002"):
        (train_dir / ident).mkdir()
        for k in range(2):
            rel = f"{ident}/{k:04d}_01.jpg"
            _make_image(train_dir / ident / f"{k:04d}_01.jpg", seed=hash(rel) % 1000)
            train_rels.append(rel)
    test_rels = []
    for ident in ("n000010", "n000011"):
        (test_dir / ident).mkdir()
        for k in range(2):
            rel = f"{ident}/{k:04d}_01.jpg"
            _make_image(test_dir / ident / f"{k:04d}_01.jpg", seed=hash(rel) % 1000)
            test_rels.append(rel)
    (root / "train_list.txt").write_text("\n".join(train_rels) + "\n", encoding="utf-8")
    (root / "test_list.txt").write_text("\n".join(test_rels) + "\n", encoding="utf-8")
    return root


@pytest.fixture
def flat_paths(tmp_path: Path) -> list:
    d = tmp_path / "flat"
    d.mkdir()
    paths = []
    for i in range(8):
        p = d / f"img_{i:02d}.png"
        _make_image(p, seed=i)
        paths.append(str(p))
    return paths


# ---------------------------------------------------------------------------
# Discovery tests
# ---------------------------------------------------------------------------


class TestDiscoverDiv2k:
    def test_discovers_train_and_val(self, div2k_root: Path) -> None:
        train, val = discover_div2k_paths(str(div2k_root))
        assert len(train) == 4
        assert len(val) == 2
        # Sorted.
        assert train == sorted(train)
        assert val == sorted(val)
        for p in train + val:
            assert os.path.isfile(p)

    def test_raises_on_empty_root(self, tmp_path: Path) -> None:
        empty = tmp_path / "nope"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            discover_div2k_paths(str(empty))

    def test_raises_on_missing_val(self, tmp_path: Path) -> None:
        root = tmp_path / "div2k_partial"
        (root / "train").mkdir(parents=True)
        _make_image(root / "train" / "0001.png")
        # No validation/ subdir at all.
        with pytest.raises(FileNotFoundError):
            discover_div2k_paths(str(root))


class TestDiscoverVggface2:
    def test_discovers_from_lists(self, vggface2_root: Path) -> None:
        train, val = discover_vggface2_paths(str(vggface2_root))
        assert len(train) == 4
        assert len(val) == 4
        for p in train + val:
            assert os.path.isfile(p), p

    def test_raises_on_missing_list(self, tmp_path: Path) -> None:
        root = tmp_path / "vggf2_partial"
        root.mkdir()
        with pytest.raises(FileNotFoundError):
            discover_vggface2_paths(str(root))


# ---------------------------------------------------------------------------
# Loader contract
# ---------------------------------------------------------------------------


class TestImageFolderLoader:
    def test_shape_and_dtype_contract(self, flat_paths: list) -> None:
        cfg = ImageFolderBurstDPConfig(
            image_paths=flat_paths,
            image_size=64,
            batch_size=2,
            n_max=3,
            n_min=3,  # deterministic n=3 for shape assertion
            sample_n_per_sample=False,
            shuffle=False,
            workers=1,
            use_multiprocessing=False,
            seed=0,
        )
        loader = ImageFolderBurstDPLoader(cfg)
        x, y = loader[0]
        assert x["ref"].shape == (2, 64, 64, 3)
        assert x["aux"].shape == (2, 3, 64, 64, 3)
        assert x["aux_mask"].shape == (2, 3)
        assert y["recon"].shape == (2, 64, 64, 3)
        assert y["segmentation"].shape == (2, 64, 64)
        assert y["segmentation"].dtype == np.int32
        assert (y["segmentation"] == 0).all()
        # Image data is in [0, 1].
        assert x["ref"].min() >= 0.0 and x["ref"].max() <= 1.0
        assert y["recon"].min() >= 0.0 and y["recon"].max() <= 1.0
        # All aux slots active when n_min=n_max=3.
        assert (x["aux_mask"] == 1.0).all()

    def test_empty_paths_raises(self) -> None:
        cfg = ImageFolderBurstDPConfig(
            image_paths=[],
            image_size=32,
            batch_size=1,
            workers=1,
            use_multiprocessing=False,
        )
        with pytest.raises(FileNotFoundError):
            ImageFolderBurstDPLoader(cfg)

    def test_aux_mask_variability_under_sampling(self, flat_paths: list) -> None:
        cfg = ImageFolderBurstDPConfig(
            image_paths=flat_paths,
            image_size=32,
            batch_size=2,
            n_max=3,
            n_min=1,
            sample_n_per_sample=True,
            shuffle=False,
            workers=1,
            use_multiprocessing=False,
            seed=1234,
        )
        loader = ImageFolderBurstDPLoader(cfg)
        seen = set()
        for i in range(len(loader)):
            _, _ = loader[i]
            x, _ = loader[i]
            per_sample = x["aux_mask"].sum(axis=-1).astype(int).tolist()
            for v in per_sample:
                assert v in (1, 2, 3)
                seen.add(v)
        # Across multiple batches we should see more than one value.
        assert len(seen) >= 2, f"aux n was constant across batches: {seen}"

    def test_aux_spec_propagates_div2k(self, div2k_root: Path) -> None:
        custom = DistortionSpec(noise_sigma_range=(0.5, 0.5))
        train, val = build_div2k_burst_dp_datasets(
            div2k_root=str(div2k_root),
            image_size=32,
            batch_size=2,
            n_max=2,
            n_min=1,
            max_train_images=2,
            max_val_images=1,
            workers=1,
            aux_spec=custom,
            seed=0,
        )
        assert train.cfg.aux_spec.noise_sigma_range == (0.5, 0.5)
        assert val.cfg.aux_spec.noise_sigma_range == (0.5, 0.5)
        # Sanity: passing None preserves default.
        train2, _ = build_div2k_burst_dp_datasets(
            div2k_root=str(div2k_root),
            image_size=32,
            batch_size=2,
            n_max=2,
            n_min=1,
            max_train_images=2,
            max_val_images=1,
            workers=1,
            aux_spec=None,
            seed=0,
        )
        assert train2.cfg.aux_spec.noise_sigma_range == default_aux_spec().noise_sigma_range

    def test_aux_spec_propagates_vggface2(self, vggface2_root: Path) -> None:
        custom = DistortionSpec(noise_sigma_range=(0.5, 0.5))
        train, val = build_vggface2_burst_dp_datasets(
            vggface2_root=str(vggface2_root),
            image_size=32,
            batch_size=2,
            n_max=2,
            n_min=1,
            workers=1,
            aux_spec=custom,
            seed=0,
        )
        assert train.cfg.aux_spec.noise_sigma_range == (0.5, 0.5)
        assert val.cfg.aux_spec.noise_sigma_range == (0.5, 0.5)

    def test_max_train_images_slicing_via_factory(self, div2k_root: Path) -> None:
        train, val = build_div2k_burst_dp_datasets(
            div2k_root=str(div2k_root),
            image_size=32,
            batch_size=2,
            n_max=2,
            n_min=1,
            max_train_images=2,
            max_val_images=1,
            workers=1,
            seed=0,
        )
        # mp off after construction; PyDataset stores config — we set workers=1.
        assert len(train.image_paths) == 2
        assert len(val.image_paths) == 1
        # __len__ = ceil(N / batch_size).
        assert len(train) == 1  # 2 / 2
        assert len(val) == 1    # 1 / 2 → 1
