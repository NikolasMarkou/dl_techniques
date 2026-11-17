"""The depth data library (``train.common.depth_data``) on a synthetic MegaDepth tree.

The tree is written into ``tmp_path`` with h5py and PIL: nine scenes of six pairs, 40 x 40 px PNG images and
HDF5 depth maps, one ordinal map (one constant valid depth) and one under-valid map (3 percent valid) in a train
scene, a selection scene and a test scene. ``train_split=0.6`` deals 4 validation scenes, which the library
deals alternately into 2 selection and 2 test scenes; MegaDepth itself is never read here.

What is held: the scene groups are pairwise disjoint; the selection and test crops are a function of
``(seed, salt, pool index)`` (bit-equal across builds, stream seeds, a resume and a fresh interpreter); a train
batch is a function of ``(stream seed, global batch index)`` and ``fit(steps_per_epoch=)`` hands the stream a
global index that carries across epochs; the target is ``[disparity, mask]``; the input is ImageNet-standardized;
a horizontal flip is the only augmentation; the layout map is the mask-weighted mean of the layout targets.
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import keras
import numpy as np
import pytest
from PIL import Image

import dl_techniques
import train.common.depth_data as depth_data
from train.common.depth_data import (
    LAYOUT_SALT,
    MAX_TRAIN_TRIES,
    SELECT_SALT,
    TEST_SALT,
    DepthStream,
    deal_megadepth_scenes,
    load_depth_batches,
)
from train.common.megadepth import (
    load_and_process_pair,
    split_megadepth_pairs,
    split_megadepth_scenes,
    standardize_imagenet,
)

SCENES = 9
PAIRS = 6
SIDE = 40
SIZE = 28
BATCH = 4
TRAIN_SPLIT = 0.6
#: ``(scene, pair)`` of the files no crop of which may qualify: an ordinal map and an under-valid map.
ORDINAL = {(0, 0), (1, 0)}
UNDER_VALID = {(2, 1), (3, 0)}
KEYS = dict(seed=7, batch_size=BATCH, image_size=SIZE, steps_per_epoch=3, n_select=8, n_test=8,
            layout_samples=16, min_valid_ratio=0.1, max_train_files=None, augment=True, workers=0,
            train_split=TRAIN_SPLIT)


def write_megadepth_tree(root: Path, scenes: int = SCENES, pairs: int = PAIRS) -> Path:
    """A MegaDepth-shaped tree: ``<root>/s<NN>/dense0/{imgs/*.png, depths/*.h5}``; returns ``root``."""
    rng = np.random.default_rng(2026)
    yy, xx = np.mgrid[0:SIDE, 0:SIDE].astype(np.float32)
    for s in range(scenes):
        imgs = root / f"s{s:02d}" / "dense0" / "imgs"
        depths = root / f"s{s:02d}" / "dense0" / "depths"
        imgs.mkdir(parents=True)
        depths.mkdir(parents=True)
        for p in range(pairs):
            rgb = rng.integers(0, 256, size=(SIDE, SIDE, 3), dtype=np.uint8)
            depth = (1.0 + 0.05 * yy + 0.03 * xx + rng.uniform(0.0, 0.5, size=(SIDE, SIDE))).astype(np.float32)
            depth[rng.random((SIDE, SIDE)) < 0.15] = 0.0
            if (s, p) in ORDINAL:
                depth = np.where(depth > 0, 2.0, 0.0).astype(np.float32)
            if (s, p) in UNDER_VALID:
                depth = np.where(rng.random((SIDE, SIDE)) < 0.03, depth, 0.0).astype(np.float32)
            Image.fromarray(rgb).save(imgs / f"{p:03d}.png")
            with h5py.File(depths / f"{p:03d}.h5", "w") as f:
                f.create_dataset("depth", data=depth)
    return root


def scene_and_pair(path: str):
    parts = Path(path).parts
    return int(parts[-4][1:]), int(Path(path).stem)


BAD = ORDINAL | UNDER_VALID


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    return str(write_megadepth_tree(tmp_path_factory.mktemp("megadepth")))


@pytest.fixture(scope="module")
def data(tree):
    return load_depth_batches(tree, **KEYS)


def digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(f"{a.dtype}{a.shape}".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def fixed_digest(d):
    return digest(d.x_select, d.y_select, d.x_test, d.y_test, d.layout)


# ---------------------------------------------------------------------
# The split
# ---------------------------------------------------------------------


def test_the_extracted_scene_split_is_the_one_split_megadepth_pairs_uses(tree):
    train_scenes, val_scenes = split_megadepth_scenes(tree, TRAIN_SPLIT)
    train_rgb, train_depth, val_rgb, val_depth = split_megadepth_pairs(tree, TRAIN_SPLIT)
    assert [p for _, rgb, _ in train_scenes for p in rgb] == train_rgb
    assert [p for _, _, dep in train_scenes for p in dep] == train_depth
    assert [p for _, rgb, _ in val_scenes for p in rgb] == val_rgb
    assert [p for _, _, dep in val_scenes for p in dep] == val_depth


def test_the_validation_scenes_are_dealt_alternately_into_selection_and_test(tree):
    _, val_scenes = split_megadepth_scenes(tree, TRAIN_SPLIT)
    names = sorted(name for name, _, _ in val_scenes)
    deal = deal_megadepth_scenes(tree, TRAIN_SPLIT)
    assert [name for name, _, _ in deal.select] == names[0::2]
    assert [name for name, _, _ in deal.test] == names[1::2]
    assert len(deal.select) >= 1 and len(deal.test) >= 1


def test_train_selection_and_test_scenes_are_pairwise_disjoint(data):
    train, select, test = set(data.train_scenes), set(data.select_scenes), set(data.test_scenes)
    assert train and select and test
    assert not train & select and not train & test and not select & test
    assert len(train | select | test) == SCENES


def test_every_file_comes_from_its_own_groups_scenes(data):
    assert {Path(p).parts[-4] for p in data.train_rgb} <= set(data.train_scenes)
    assert {Path(p).parts[-4] for p in data.select_files} <= set(data.select_scenes)
    assert {Path(p).parts[-4] for p in data.test_files} <= set(data.test_scenes)


def test_fewer_than_three_validation_scenes_is_a_named_error(tmp_path):
    root = write_megadepth_tree(tmp_path / "small", scenes=4, pairs=2)
    with pytest.raises(ValueError, match=r"2 validation scenes.*at least 3"):
        load_depth_batches(str(root), **{**KEYS, "train_split": 0.5})


# ---------------------------------------------------------------------
# The fixed selection and test crops
# ---------------------------------------------------------------------


def test_the_fixed_sets_are_bit_equal_across_builds_and_stream_seeds(tree, data):
    again = load_depth_batches(tree, **KEYS)
    other_stream = load_depth_batches(tree, **KEYS, stream_seed=991)
    assert fixed_digest(again) == fixed_digest(data) == fixed_digest(other_stream)
    assert other_stream.stream_seed == 991 and data.stream_seed == KEYS["seed"]
    assert other_stream.fingerprint == data.fingerprint


_FRESH = """
import hashlib, json, sys
import numpy as np
from train.common.depth_data import load_depth_batches

def digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(f"{a.dtype}{a.shape}".encode())
        h.update(a.tobytes())
    return h.hexdigest()

d = load_depth_batches(sys.argv[1], **json.loads(sys.argv[2]))
print("DIGEST", digest(d.x_select, d.y_select, d.x_test, d.y_test, d.layout))
"""


def test_a_fresh_interpreter_builds_the_same_fixed_sets(tree, data, tmp_path):
    src = str(Path(dl_techniques.__file__).resolve().parents[1])
    env = dict(os.environ, PYTHONPATH=src, CUDA_VISIBLE_DEVICES="", TF_CPP_MIN_LOG_LEVEL="3")
    env.pop("PYTHONHASHSEED", None)
    out = subprocess.run([sys.executable, "-c", _FRESH, tree, json.dumps(KEYS)], capture_output=True, text=True,
                         env=env, cwd=str(tmp_path), timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert [l for l in out.stdout.splitlines() if l.startswith("DIGEST ")] == [f"DIGEST {fixed_digest(data)}"]


def test_a_selection_crop_is_the_loaders_crop_keyed_on_seed_salt_and_pool_index(tree, data):
    for x, y, path, key in zip(data.x_select, data.y_select, data.select_files, data.select_keys):
        rgb, y_ref = load_and_process_pair(path, path.replace("/imgs/", "/depths/")[:-4] + ".h5", SIZE,
                                           min_valid_ratio=KEYS["min_valid_ratio"], augment=False,
                                           rng=np.random.default_rng([KEYS["seed"], SELECT_SALT, key]))
        np.testing.assert_array_equal(x, standardize_imagenet(rgb))
        np.testing.assert_array_equal(y, y_ref)
    for x, y, path, key in zip(data.x_test, data.y_test, data.test_files, data.test_keys):
        rgb, y_ref = load_and_process_pair(path, path.replace("/imgs/", "/depths/")[:-4] + ".h5", SIZE,
                                           min_valid_ratio=KEYS["min_valid_ratio"], augment=False,
                                           rng=np.random.default_rng([KEYS["seed"], TEST_SALT, key]))
        np.testing.assert_array_equal(x, standardize_imagenet(rgb))
        np.testing.assert_array_equal(y, y_ref)


def test_the_target_is_disparity_and_mask_on_the_last_axis(data):
    for y in (data.y_select, data.y_test):
        assert y.shape[1:] == (SIZE, SIZE, 2) and y.dtype == np.float32
        mask, disparity = y[..., 1], y[..., 0]
        assert set(np.unique(mask)) <= {0.0, 1.0}
        assert (mask.reshape(len(y), -1).mean(axis=1) >= KEYS["min_valid_ratio"]).all()
        assert (disparity[mask == 0] == 0).all()
        assert disparity.min() >= 0 and disparity.max() <= 1
        # Every crop has spread: its farthest valid pixel is 0 and its nearest is 1.
        assert (disparity.reshape(len(y), -1).max(axis=1) == 1.0).all()


def test_no_ordinal_or_under_valid_file_is_a_fixed_sample(data):
    used = {scene_and_pair(p) for p in data.select_files + data.test_files}
    assert not used & BAD
    # The bad files are in the selection and test pools, so the test has something to refuse.
    assert {(1, 0), (3, 0)} <= {scene_and_pair(p) for p in data.select_pool + data.test_pool}


def test_the_streams_state_their_length(data):
    assert data.n_select == (KEYS["n_select"] // BATCH) * BATCH and data.n_test == (KEYS["n_test"] // BATCH) * BATCH
    assert int(data.select_ds.cardinality()) == KEYS["n_select"] // BATCH
    assert int(data.test_ds.cardinality()) == KEYS["n_test"] // BATCH
    assert data.train_ds.num_batches is None, "the train stream is infinite; fit takes steps_per_epoch"
    assert data.steps_per_epoch == KEYS["steps_per_epoch"]
    assert data.n_train == data.steps_per_epoch * BATCH
    assert data.input_shape == (SIZE, SIZE, 3) and data.num_classes is None
    # One pass of the selection stream is the selection arrays, in order, whole batches only.
    xs, ys = zip(*[(x.numpy(), y.numpy()) for x, y in data.select_ds])
    np.testing.assert_array_equal(np.concatenate(xs), data.x_select)
    np.testing.assert_array_equal(np.concatenate(ys), data.y_select)
    assert all(x.shape[0] == BATCH for x in xs)
    assert data.select_ds.element_spec[0].shape[0] == BATCH


def test_a_count_that_is_not_a_multiple_of_the_batch_keeps_whole_batches(tree):
    d = load_depth_batches(tree, **{**KEYS, "n_select": 7, "layout_samples": 2})
    assert d.n_select == 4 and len(d.x_select) == 4 and int(d.select_ds.cardinality()) == 1


def test_a_batch_above_a_fixed_set_is_a_named_error(tree):
    with pytest.raises(ValueError, match=r"batch_size 16.*n_select 8"):
        load_depth_batches(tree, **{**KEYS, "batch_size": 16})


def test_too_few_qualifying_crops_is_a_named_error_naming_the_group_and_counts(tree):
    # The two selection scenes hold 12 files, one ordinal: 11 qualify.
    with pytest.raises(ValueError, match=r"selection.*12 crops.*11 of the 12 files"):
        load_depth_batches(tree, **{**KEYS, "n_select": 12})


def test_the_fingerprint_changes_with_the_listing_and_with_n_select(tree, data, tmp_path):
    assert load_depth_batches(tree, **{**KEYS, "n_select": 4}).fingerprint != data.fingerprint
    copy = write_megadepth_tree(tmp_path / "copy")
    assert load_depth_batches(str(copy), **KEYS).fingerprint == data.fingerprint, "relative listing, not the root"
    extra = copy / "s08" / "dense0"
    Image.fromarray(np.zeros((SIDE, SIDE, 3), np.uint8)).save(extra / "imgs" / "999.png")
    with h5py.File(extra / "depths" / "999.h5", "w") as f:
        f.create_dataset("depth", data=np.ones((SIDE, SIDE), np.float32))
    assert load_depth_batches(str(copy), **KEYS).fingerprint != data.fingerprint


# ---------------------------------------------------------------------
# The train stream
# ---------------------------------------------------------------------


def stream_of(data, stream_seed, augment=True, rgb=None, depth=None):
    return DepthStream(data.train_rgb if rgb is None else rgb, data.train_depth if depth is None else depth,
                       stream_seed=stream_seed, batch_size=BATCH, image_size=SIZE,
                       min_valid_ratio=KEYS["min_valid_ratio"], augment=augment, workers=0)


def test_a_train_batch_is_a_function_of_the_stream_seed_and_the_index(data):
    one, same, other = stream_of(data, 5), stream_of(data, 5), stream_of(data, 6)
    for index in (0, 3, 17):
        a, b, c = one[index], same[index], other[index]
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])
        assert not np.array_equal(a[0], c[0])
    assert not np.array_equal(one[0][0], one[1][0])
    # Batch i is the samples of counters i*B .. i*B+B-1, read in any order.
    x, y = one[2]
    for j in range(BATCH):
        sx, sy, _ = one.sample(2 * BATCH + j)
        np.testing.assert_array_equal(x[j], sx)
        np.testing.assert_array_equal(y[j], sy)
    np.testing.assert_array_equal(stream_of(data, 5)[17][0], one[17][0])


def test_a_train_sample_reads_only_train_files_and_never_an_ordinal_or_under_valid_one(data):
    stream = stream_of(data, 3)
    tried_bad = 0
    for counter in range(120):
        _, y, tried = stream.sample(counter)
        assert 1 <= len(tried) <= MAX_TRAIN_TRIES
        kept = data.train_rgb[tried[-1]]
        assert Path(kept).parts[-4] in data.train_scenes
        assert scene_and_pair(kept) not in BAD
        tried_bad += sum(scene_and_pair(data.train_rgb[k]) in BAD for k in tried[:-1])
        assert y.shape == (SIZE, SIZE, 2) and (y[..., 1].mean() >= KEYS["min_valid_ratio"])
    assert tried_bad > 0, "the bad train files were drawn and skipped, so the test has power"


def test_a_counter_whose_files_all_fail_raises_a_named_error_within_the_bound(data, monkeypatch):
    bad = [i for i, p in enumerate(data.train_rgb) if scene_and_pair(p) in BAD]
    assert bad
    reads = []
    real = depth_data.load_and_process_pair

    def counting(*args, **kwargs):
        reads.append(args[0])
        return real(*args, **kwargs)

    monkeypatch.setattr(depth_data, "load_and_process_pair", counting)
    stream = stream_of(data, 1, rgb=[data.train_rgb[i] for i in bad], depth=[data.train_depth[i] for i in bad])
    with pytest.raises(RuntimeError, match=rf"counter 4 of stream 1.*{MAX_TRAIN_TRIES} files"):
        stream.sample(4)
    assert len(reads) == MAX_TRAIN_TRIES


def test_a_horizontal_flip_is_the_only_augmentation(data):
    flipped = unchanged = 0
    plain, augmented = stream_of(data, 11, augment=False), stream_of(data, 11, augment=True)
    for counter in range(40):
        px, py, pt = plain.sample(counter)
        ax, ay, at = augmented.sample(counter)
        assert pt == at, "the file and the crop do not depend on the flip"
        if np.array_equal(ax, px):
            np.testing.assert_array_equal(ay, py)
            unchanged += 1
        else:
            np.testing.assert_array_equal(ax, px[:, ::-1])
            np.testing.assert_array_equal(ay, py[:, ::-1])
            flipped += 1
    assert flipped and unchanged


def test_the_train_input_is_imagenet_standardized(data):
    stream = stream_of(data, 2, augment=False)
    x, _, tried = stream.sample(9)
    rng = np.random.default_rng([2, depth_data.TRAIN_SALT, 9])
    for _ in range(len(tried)):
        j = int(rng.integers(len(data.train_rgb)))
        out = load_and_process_pair(data.train_rgb[j], data.train_depth[j], SIZE,
                                    min_valid_ratio=KEYS["min_valid_ratio"], augment=False, rng=rng)
    assert j == tried[-1] and out is not None
    np.testing.assert_array_equal(x, standardize_imagenet(out[0]))


def test_with_stream_keeps_the_fixed_sets_and_moves_the_train_stream(data):
    from train.common.pretrain_data import derive_stream_seed

    resumed = data.with_stream(300)
    assert resumed.stream_seed == derive_stream_seed(KEYS["seed"], 300) != data.stream_seed
    assert fixed_digest(resumed) == fixed_digest(data)
    assert resumed.select_ds is data.select_ds and resumed.test_ds is data.test_ds
    for field in ("steps_per_epoch", "n_train", "n_select", "n_test", "fingerprint"):
        assert getattr(resumed, field) == getattr(data, field)
    assert not np.array_equal(resumed.train_ds[0][0], data.train_ds[0][0])


def test_fit_hands_the_stream_a_global_index_that_carries_across_epochs(data):
    """D-014: with ``steps_per_epoch`` and worker processes, epoch e trains batches e*S .. e*S+S-1 of the stream."""

    class YSum(keras.metrics.Metric):
        def __init__(self):
            super().__init__(name="y_sum")
            self.total = self.add_weight(name="total", initializer="zeros")

        def update_state(self, y_true, y_pred, sample_weight=None):
            self.total.assign(keras.ops.sum(y_true))

        def result(self):
            return self.total

    class Seen(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.sums = []

        def on_train_batch_end(self, batch, logs=None):
            self.sums.append(float(logs["y_sum"]))

    stream = DepthStream(data.train_rgb, data.train_depth, stream_seed=13, batch_size=BATCH, image_size=SIZE,
                         min_valid_ratio=KEYS["min_valid_ratio"], augment=True, workers=2)
    assert stream.use_multiprocessing and stream.workers == 2
    model = keras.Sequential([keras.layers.Input((SIZE, SIZE, 3)), keras.layers.Conv2D(2, 1)])
    model.compile(optimizer="sgd", loss="mse", metrics=[YSum()], jit_compile=False)
    seen = Seen()
    epochs, steps = 3, 2
    model.fit(stream, epochs=epochs, steps_per_epoch=steps, verbose=0, callbacks=[seen])
    reference = stream_of(data, 13)
    expected = [float(keras.ops.sum(reference[i][1])) for i in range(epochs * steps)]
    np.testing.assert_allclose(seen.sums, expected, rtol=1e-6, atol=0)
    assert len(set(expected)) == len(expected), "distinct batches, so the order is visible"


# ---------------------------------------------------------------------
# The average layout
# ---------------------------------------------------------------------


def test_the_layout_is_the_mask_weighted_mean_of_the_layout_targets(data):
    total = np.zeros((SIZE, SIZE), np.float64)
    weight = np.zeros((SIZE, SIZE), np.float64)
    for i in range(KEYS["layout_samples"]):
        rng = np.random.default_rng([KEYS["seed"], LAYOUT_SALT, i])
        for _ in range(MAX_TRAIN_TRIES):
            j = int(rng.integers(len(data.train_rgb)))
            out = load_and_process_pair(data.train_rgb[j], data.train_depth[j], SIZE,
                                        min_valid_ratio=KEYS["min_valid_ratio"], augment=False, rng=rng)
            if out is not None:
                break
        y = out[1].astype(np.float64)
        total += y[..., 0] * y[..., 1]
        weight += y[..., 1]
    oracle = np.where(weight > 0, total / np.maximum(weight, 1e-300), 0.0)
    assert data.layout.shape == (SIZE, SIZE, 1) and data.layout.dtype == np.float32
    np.testing.assert_allclose(data.layout[..., 0], oracle, rtol=0, atol=1e-6)
    # The plain mean is another map here (an invalid pixel holds 0 in the disparity channel), so the test can
    # tell the two apart.
    plain = total / KEYS["layout_samples"]
    assert np.abs(plain - oracle).max() > 1e-2
