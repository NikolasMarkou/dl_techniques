"""Coverage for `PushTHDF5Dataset`, previously an untested skeleton.

Three properties pinned, each on a synthetic HDF5 fixture written to
`tmp_path` (real h5py file, same reader code path as production, no staged
corpus required):

1. `.as_tf_dataset()` yields batches with the documented shapes/dtypes, and
   pixels are genuinely ImageNet-normalized (some values below the
   per-channel mean, not raw `[0,255]`/`[0,1]`).
2. `_load_metadata()` replaces every NaN action value with `0.0`
   (`pusht_hdf5.py:167`) — episode-break markers in the upstream format.
3. `_window_starts()` never emits a window that crosses an `episode_ends`
   boundary (`pusht_hdf5.py:170-181`).

Case 4 (`/episode_ends` absent -> fallback to `[n_pixels]`) is included as a
one-line variant of the shared helper.
"""

import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.datasets.pusht_hdf5 import PushTHDF5Dataset

h5py = pytest.importorskip(
    "h5py", reason="PushTHDF5Dataset requires h5py; it is in the 'data' extra"
)


def _write_pusht_h5(path, pixels, action, episode_ends=None):
    """Write a minimal PushT-schema HDF5 file: `/pixels`, `/action`, optional
    `/episode_ends`."""
    with h5py.File(path, "w") as f:
        f.create_dataset("pixels", data=pixels)
        f.create_dataset("action", data=action)
        if episode_ends is not None:
            f.create_dataset(
                "episode_ends", data=np.asarray(episode_ends, dtype=np.int64)
            )


def test_as_tf_dataset_yields_documented_shapes_and_normalizes_pixels(tmp_path):
    rng = np.random.default_rng(0)
    n, h0, w0, action_dim = 20, 48, 48, 2
    pixels = rng.integers(0, 256, size=(n, h0, w0, 3), dtype=np.uint8)
    action = rng.standard_normal((n, action_dim)).astype(np.float32)
    h5_path = tmp_path / "pusht.h5"
    _write_pusht_h5(str(h5_path), pixels, action, episode_ends=[n])

    dataset = PushTHDF5Dataset(
        str(h5_path),
        img_size=16,
        action_dim=action_dim,
        history_size=2,
        num_preds=1,
        batch_size=2,
    )
    tf_dataset = dataset.as_tf_dataset()
    x, y = next(iter(tf_dataset))

    assert x["pixels"].shape == (2, 3, 16, 16, 3)
    assert x["action"].shape == (2, 2, 2)
    assert x["pixels"].dtype == tf.float32
    assert x["action"].dtype == tf.float32
    assert y.shape == (2,)
    assert np.array_equal(y.numpy(), np.zeros((2,), dtype=np.float32))
    # ImageNet normalization must have actually run: raw uint8 -> [0,1] ->
    # subtract per-channel mean -> divide by per-channel std produces values
    # below zero for any pixel darker than the channel mean. A bug that
    # skipped normalization (or left pixels in [0,1]/[0,255]) would never
    # produce a negative value.
    assert np.any(x["pixels"].numpy() < 0)


def test_load_metadata_replaces_nan_actions_with_zero(tmp_path):
    rng = np.random.default_rng(0)
    n, h0, w0, action_dim = 20, 48, 48, 2
    pixels = rng.integers(0, 256, size=(n, h0, w0, 3), dtype=np.uint8)
    action = rng.standard_normal((n, action_dim)).astype(np.float32)
    nan_row = 5
    action[nan_row, :] = np.nan
    h5_path = tmp_path / "pusht.h5"
    _write_pusht_h5(str(h5_path), pixels, action, episode_ends=[n])

    dataset = PushTHDF5Dataset(str(h5_path), action_dim=action_dim)
    loaded_action, _ends, n_pixels = dataset._load_metadata()

    assert n_pixels == n
    assert not np.any(np.isnan(loaded_action))
    assert np.array_equal(loaded_action[nan_row], np.array([0.0, 0.0], dtype=np.float32))


def test_window_starts_never_crosses_an_episode_boundary():
    # Two episodes: [0, 10) and [10, 20). history_size=3, num_preds=1 -> T=4.
    # h5_path is never opened by `_window_starts` (it operates purely on the
    # `ends` array and instance config), so a nonexistent path is fine here.
    dataset = PushTHDF5Dataset(
        "unused.h5", history_size=3, num_preds=1, frameskip=1
    )
    ends = np.array([10, 20])
    starts = dataset._window_starts(ends)

    # Independently re-derive the expected start set with the same
    # per-episode "while i + T <= end" rule described in the module
    # docstring, rather than trusting a hand-traced literal set copied from
    # the design notes.
    T = 3 + 1
    expected = []
    prev_end = 0
    for end in ends:
        i = prev_end
        while i + T <= end:
            expected.append(i)
            i += 1
        prev_end = int(end)
    expected = np.asarray(expected, dtype=np.int64)

    assert np.array_equal(np.sort(starts), np.sort(expected))
    # Structural cross-check, independent of the re-derivation above: no
    # window may straddle the episode-1/episode-2 boundary at index 10, and
    # every window must fit entirely inside its own episode.
    assert all(s + T <= 10 or s >= 10 for s in starts)
    assert all(s + T <= 20 for s in starts)


def test_load_metadata_falls_back_to_n_pixels_when_episode_ends_absent(tmp_path):
    rng = np.random.default_rng(0)
    n, h0, w0, action_dim = 20, 48, 48, 2
    pixels = rng.integers(0, 256, size=(n, h0, w0, 3), dtype=np.uint8)
    action = rng.standard_normal((n, action_dim)).astype(np.float32)
    h5_path = tmp_path / "pusht_no_ends.h5"
    _write_pusht_h5(str(h5_path), pixels, action, episode_ends=None)

    dataset = PushTHDF5Dataset(str(h5_path), action_dim=action_dim)
    _action, ends, n_pixels = dataset._load_metadata()

    assert n_pixels == n
    assert np.array_equal(ends, np.array([n], dtype=np.int64))
