"""Tests for the LightGlue homography pair pipeline (train.lightglue.data)."""

import ast
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from train.lightglue import data as pair_data
from train.lightglue.data import list_images, make_pair_dataset

SIZE = 64
SRC = 20  # dot position in the source (x, y) = (SRC, SRC + 8)
ZERO = {"rotation": 0.0, "scale": (1.0, 1.0), "perspective": 0.0,
        "translation": 0.0, "shear": 0.0}


def _write(path: Path, arr: np.ndarray) -> None:
    enc = tf.io.encode_png if path.suffix == ".png" else tf.io.encode_jpeg
    path.write_bytes(enc(tf.constant(arr)).numpy())


def _texture(rng: np.random.RandomState, h: int, w: int) -> np.ndarray:
    base = rng.randint(30, 120, size=(h, w, 3)).astype(np.uint8)
    return base


def _dot_image(h: int, w: int, x: int, y: int) -> np.ndarray:
    img = np.zeros((h, w, 3), np.uint8)
    img[y - 2:y + 3, x - 2:x + 3] = 255
    return img


@pytest.fixture()
def folder(tmp_path: Path) -> Path:
    rng = np.random.RandomState(0)
    for i in range(5):
        _write(tmp_path / f"img{i}.jpg", _texture(rng, 90, 120))
    _write(tmp_path / "img5.png", _texture(rng, 100, 80))
    (tmp_path / "notes.txt").write_text("x")
    return tmp_path


def _first(ds):
    return next(iter(ds))


def test_list_images_sorted_and_filtered(folder):
    paths = list_images(str(folder))
    assert paths == sorted(paths)
    assert len(paths) == 6
    assert all(p.endswith((".jpg", ".png")) for p in paths)


def test_list_images_errors(tmp_path):
    with pytest.raises(ValueError, match="no image files"):
        list_images(str(tmp_path))
    with pytest.raises(FileNotFoundError):
        list_images(str(tmp_path / "missing"))


def test_shapes_dtypes_ranges(folder):
    ds = make_pair_dataset(list_images(str(folder)), SIZE, 2, seed=1)
    b = _first(ds)
    assert set(b) == {"image0", "image1", "H0to1", "image_size0", "image_size1"}
    assert b["image0"].shape == (2, SIZE, SIZE, 1)
    assert b["image1"].shape == (2, SIZE, SIZE, 1)
    assert b["H0to1"].shape == (2, 3, 3)
    for k in b:
        assert b[k].dtype == tf.float32
    for k in ("image0", "image1"):
        a = b[k].numpy()
        assert a.min() >= 0.0 and a.max() <= 1.0
    np.testing.assert_allclose(b["H0to1"].numpy()[:, 2, 2], 1.0, atol=1e-5)


def test_rectangular_image_size_outputs(folder):
    ds = make_pair_dataset(list_images(str(folder)), (48, 80), 3, seed=1)
    b = _first(ds)
    assert b["image0"].shape == (3, 48, 80, 1)
    np.testing.assert_array_equal(
        b["image_size0"].numpy(), np.tile([80.0, 48.0], (3, 1)))
    np.testing.assert_array_equal(b["image_size1"].numpy(), b["image_size0"].numpy())


def test_determinism_and_seed_sensitivity(folder):
    paths = list_images(str(folder))
    a = _first(make_pair_dataset(paths, SIZE, 4, seed=3))
    b = _first(make_pair_dataset(paths, SIZE, 4, seed=3))
    c = _first(make_pair_dataset(paths, SIZE, 4, seed=4))
    for k in a:
        np.testing.assert_array_equal(a[k].numpy(), b[k].numpy())
    assert not np.allclose(a["H0to1"].numpy(), c["H0to1"].numpy())


def test_element_index_changes_homography(folder):
    ds = make_pair_dataset(list_images(str(folder)), SIZE, 4, seed=3, shuffle=False)
    h = _first(ds)["H0to1"].numpy()
    for i in range(4):
        for j in range(i + 1, 4):
            assert not np.allclose(h[i], h[j])


def test_epochs_differ_with_repeat(folder):
    paths = list_images(str(folder))[:2]
    ds = make_pair_dataset(paths, SIZE, 2, seed=3, shuffle=False, repeat=True)
    it = iter(ds)
    e0, e1 = next(it), next(it)
    np.testing.assert_array_equal(e0["image0"].numpy(), e1["image0"].numpy())
    assert not np.allclose(e0["H0to1"].numpy(), e1["H0to1"].numpy())


def test_marked_pixel_maps_through_h0to1(tmp_path):
    for i in range(3):
        _write(tmp_path / f"d{i}.png", _dot_image(SIZE, SIZE, SRC, SRC + 8))
    params = dict(ZERO, rotation=0.3, scale=(0.9, 1.1), translation=0.05,
                  perspective=0.001)
    ds = make_pair_dataset(list_images(str(tmp_path)), SIZE, 3, seed=5,
                           homography_params=params, photometric_jitter=False,
                           shuffle=False)
    b = _first(ds)
    checked = 0
    for k in range(3):
        h = b["H0to1"].numpy()[k].astype(np.float64)
        q = h @ np.array([SRC, SRC + 8, 1.0])
        q = q[:2] / q[2]
        if not (4 <= q[0] < SIZE - 4 and 4 <= q[1] < SIZE - 4):
            continue
        img = b["image1"].numpy()[k, ..., 0]
        ys, xs = np.mgrid[0:SIZE, 0:SIZE]
        w = img * (img > 0.2)
        cen = np.array([(w * xs).sum(), (w * ys).sum()]) / w.sum()
        assert np.abs(cen - q).max() < 1.0, (cen, q)
        checked += 1
    assert checked >= 1


def test_identity_params_give_identity_pair(folder):
    ds = make_pair_dataset(list_images(str(folder)), SIZE, 2, seed=1,
                           homography_params=ZERO, photometric_jitter=False)
    b = _first(ds)
    np.testing.assert_allclose(
        b["H0to1"].numpy(), np.tile(np.eye(3), (2, 1, 1)), atol=1e-5)
    np.testing.assert_allclose(b["image1"].numpy(), b["image0"].numpy(), atol=2e-3)


def test_jitter_only_on_image1_and_out_of_frame_zero(folder):
    paths = list_images(str(folder))
    off = _first(make_pair_dataset(paths, SIZE, 2, seed=1, shuffle=False,
                                   homography_params=ZERO, photometric_jitter=False))
    on = _first(make_pair_dataset(paths, SIZE, 2, seed=1, shuffle=False,
                                  homography_params=ZERO, photometric_jitter=True))
    np.testing.assert_array_equal(off["image0"].numpy(), on["image0"].numpy())
    assert np.abs(off["image1"].numpy() - on["image1"].numpy()).max() > 1e-3
    shift = make_pair_dataset(paths, SIZE, 2, seed=1, shuffle=False,
                              homography_params=dict(ZERO, translation=0.3))
    b = _first(shift)
    assert (b["image1"].numpy() == 0.0).mean() > 0.05


def test_jitter_image0_flag(folder):
    paths = list_images(str(folder))
    a = _first(make_pair_dataset(paths, SIZE, 2, seed=1, shuffle=False))
    b = _first(make_pair_dataset(paths, SIZE, 2, seed=1, shuffle=False,
                                 jitter_image0=True))
    assert not np.allclose(a["image0"].numpy(), b["image0"].numpy())


def test_cardinality_semantics(folder):
    paths = list_images(str(folder))  # 6 files
    ds = make_pair_dataset(paths, SIZE, 4, seed=1)
    assert sum(1 for _ in ds) == 1
    ds = make_pair_dataset(paths, SIZE, 4, seed=1, drop_remainder=False)
    assert [int(b["image0"].shape[0]) for b in ds] == [4, 2]
    ds = make_pair_dataset(paths, SIZE, 2, seed=1, repeat=True)
    assert sum(1 for _ in ds.take(10)) == 10
    ds = make_pair_dataset(paths, SIZE, 2, seed=1, shuffle=False)
    assert sum(1 for _ in ds) == 3


def test_bad_arguments(folder):
    paths = list_images(str(folder))
    with pytest.raises(ValueError):
        make_pair_dataset([], SIZE, 2)
    with pytest.raises(ValueError, match="unknown homography_params"):
        make_pair_dataset(paths, SIZE, 2, homography_params={"bogus": 1.0})
    with pytest.raises(ValueError):
        make_pair_dataset(paths, SIZE, 0)


def test_import_is_side_effect_free():
    root = Path(__file__).resolve().parents[3]
    code = ("import sys; sys.path.insert(0, 'src'); "
            "import train.lightglue.data")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", TF_CPP_MIN_LOG_LEVEL="3")
    r = subprocess.run([sys.executable, "-c", code], cwd=root, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    tree = ast.parse(Path(pair_data.__file__).read_text())
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.Import, ast.ImportFrom)):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name):
                assert sub.value.id != "tf", ast.dump(node)[:200]
