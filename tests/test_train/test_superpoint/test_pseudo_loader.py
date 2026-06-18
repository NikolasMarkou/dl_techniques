"""Unit tests for the SuperPoint stage-4 pseudo-label loader.

Covers ``_pseudo_pair_generator`` and ``SuperPointConfig`` validation (plan
SC2/SC3). A tiny self-contained pseudo set is built in ``tmp_path`` that mirrors
the REAL stage-3 HA output layout:

- ``<exp>/manifest.json`` with top-level ``input_size``/``cell`` and per-entry
  ``source_path`` (absolute) + ``npz`` (relative to the experiment dir, e.g.
  ``pseudo_labels/<id>.npz``), exactly as
  ``homographic_adaptation._build_manifest_entry`` writes them.
- ``<exp>/images/<id>.png`` grayscale source images at ``input_size``.
- ``<exp>/pseudo_labels/<id>.npz`` each carrying a pre-encoded
  ``grid_label (Hc, Wc) i32 [0..64]`` + ``keypoints (N, 2) f32``.

With ``input_size=64`` and ``cell=8`` the detector grid is ``Hc=Wc=8`` and the
correspondence matrix is ``(N, N)`` with ``N = Hc*Wc = 64``.
"""

import json
import itertools
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from train.superpoint.train_superpoint import (
    SuperPointConfig,
    _pseudo_pair_generator,
)


INPUT_SIZE = 64
CELL = 8
HC = INPUT_SIZE // CELL  # 8
N = HC * HC  # 64 correspondence dim


# ---------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------


def _write_png(path: Path, input_size: int) -> None:
    """Write a tiny grayscale PNG at ``input_size x input_size``."""
    arr = np.random.default_rng(abs(hash(path.name)) % (2**31)).integers(
        0, 256, size=(input_size, input_size, 1), dtype=np.uint8
    )
    png = tf.io.encode_png(arr).numpy()
    path.write_bytes(png)


def _write_npz(path: Path, grid_label: np.ndarray) -> None:
    """Write a pseudo-label npz mirroring the HA format."""
    keypoints = np.array([[3.0, 4.0], [10.0, 12.0]], dtype=np.float32)
    np.savez_compressed(path, keypoints=keypoints, grid_label=grid_label)


def _build_pseudo_set(
    root: Path,
    input_size: int = INPUT_SIZE,
    cell: int = CELL,
    manifest_input_size: int = None,
    manifest_cell: int = None,
    drop_source_path: bool = False,
):
    """Create a self-contained pseudo-label set under ``root``.

    Returns a dict ``image_id -> grid_label`` for round-trip assertions.
    """
    images_dir = root / "images"
    labels_dir = root / "pseudo_labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    grid_labels = {}
    entries = []
    for i in range(2):
        image_id = f"img{i}"
        png_path = images_dir / f"{image_id}.png"
        npz_path = labels_dir / f"{image_id}.npz"
        _write_png(png_path, input_size)

        grid_label = rng.integers(0, 65, size=(HC, HC), dtype=np.int32)
        grid_labels[image_id] = grid_label
        _write_npz(npz_path, grid_label)

        entry = {
            "image_id": image_id,
            "dataset_name": "synthetic",
            "source_path": str(png_path.resolve()),
            "npz": str(npz_path.relative_to(root)),
            "num_keypoints": 2,
        }
        if drop_source_path:
            del entry["source_path"]
        entries.append(entry)

    manifest = {
        "input_size": manifest_input_size or input_size,
        "cell": manifest_cell or cell,
        "n_homographies": 4,
        "entries": entries,
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return grid_labels


def _make_config(root: Path, input_size: int = INPUT_SIZE) -> SuperPointConfig:
    return SuperPointConfig(
        data_mode="pseudo",
        pseudo_labels_dir=str(root),
        input_size=input_size,
        cell=CELL,
        channels=1,
        batch_size=2,
        epochs=1,
        steps_per_epoch=2,
        variant="tiny",
        seed=42,
    )


# ---------------------------------------------------------------------
# Generator contract (SC3)
# ---------------------------------------------------------------------


class TestPseudoPairGeneratorContract:

    def test_generator_yields_correct_contract(self, tmp_path):
        _build_pseudo_set(tmp_path)
        config = _make_config(tmp_path)

        items = list(itertools.islice(_pseudo_pair_generator(config), 4))
        assert len(items) == 4
        for image, warped, grid_label, corr in items:
            assert image.shape == (INPUT_SIZE, INPUT_SIZE, 1)
            assert image.dtype == np.float32
            assert warped.shape == (INPUT_SIZE, INPUT_SIZE, 1)
            assert warped.dtype == np.float32
            assert grid_label.shape == (HC, HC)
            assert grid_label.dtype == np.int32
            assert corr.shape == (N, N)
            assert corr.dtype == np.float32

    def test_grid_label_loaded_verbatim(self, tmp_path):
        grid_labels = _build_pseudo_set(tmp_path)
        config = _make_config(tmp_path)

        # First yielded entry is img0 (idx starts at 0, entries in write order).
        first = next(_pseudo_pair_generator(config))
        _, _, grid_label, _ = first
        assert np.array_equal(grid_label, grid_labels["img0"])

    def test_wraparound_cycles_entries(self, tmp_path):
        _build_pseudo_set(tmp_path)
        config = _make_config(tmp_path)

        # 2 entries -> third pull wraps back to the first grid_label.
        items = list(itertools.islice(_pseudo_pair_generator(config), 3))
        assert np.array_equal(items[0][2], items[2][2])


# ---------------------------------------------------------------------
# Fail-loud guards (SC2/SC3)
# ---------------------------------------------------------------------


class TestPseudoPairGeneratorGuards:

    def test_input_size_mismatch_raises(self, tmp_path):
        # Manifest stores input_size=64; config requests 128 -> mismatch.
        _build_pseudo_set(tmp_path, manifest_input_size=INPUT_SIZE)
        config = _make_config(tmp_path, input_size=128)
        with pytest.raises(ValueError) as exc:
            next(_pseudo_pair_generator(config))
        msg = str(exc.value)
        assert "64" in msg and "128" in msg

    def test_missing_source_path_raises(self, tmp_path):
        _build_pseudo_set(tmp_path, drop_source_path=True)
        config = _make_config(tmp_path)
        with pytest.raises(ValueError) as exc:
            next(_pseudo_pair_generator(config))
        assert "source_path" in str(exc.value)


# ---------------------------------------------------------------------
# Config validation (SC2)
# ---------------------------------------------------------------------


class TestSuperPointConfigValidation:

    def test_bad_data_mode_raises(self):
        with pytest.raises(ValueError):
            SuperPointConfig(data_mode="bogus")

    def test_pseudo_without_dir_raises(self):
        with pytest.raises(ValueError):
            SuperPointConfig(data_mode="pseudo", pseudo_labels_dir=None)

    def test_synthetic_default_ok(self):
        # Default path must construct cleanly (regression-safe).
        config = SuperPointConfig()
        assert config.data_mode == "synthetic"
        assert config.pseudo_labels_dir is None
