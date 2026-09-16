"""Coverage for `PushTHDF5Dataset`, previously an untested skeleton.

Properties pinned, each on a synthetic HDF5 fixture written to `tmp_path`
(real h5py file, same reader code path as production, no staged corpus
required):

1. `.as_tf_dataset()` yields batches with the documented shapes/dtypes, and
   pixels are genuinely ImageNet-normalized (some values below the
   per-channel mean, not raw `[0,255]`/`[0,1]`).
1b. A constant-valued fixture frame normalizes to the EXACT per-channel
   `-mean/std` value (`pusht_hdf5.py:41-42`) — strong enough to distinguish
   correct normalization from a wrong shift or swapped mean/std constants,
   which `np.any(pixels < 0)` alone cannot (REFLECT D-011).
2. `_load_metadata()` replaces every NaN action value with `0.0`
   (`pusht_hdf5.py:167`) — episode-break markers in the upstream format —
   AND leaves every non-NaN action row byte-for-byte unchanged (D-011: a
   test that only checks "no NaN remains" cannot tell `np.where(isnan, 0,
   action)` from `np.zeros_like(action)`).
3. `_window_starts()` never emits a window that crosses an `episode_ends`
   boundary (`pusht_hdf5.py:170-181`).
4. End-to-end (D-011): pulled through the REAL `.as_tf_dataset()` pipeline
   (not `_window_starts()` in isolation) with an index-encoded, 2-episode
   fixture, every emitted window's frames are consecutive, no window
   straddles the episode boundary, and each window's actions match its own
   frame indices — the property the plan calls load-bearing ("a spliced
   window is silent corruption, with no shape/dtype symptom").

Case 5 (`/episode_ends` absent -> fallback to `[n_pixels]`) is included as a
one-line variant of the shared helper.

Only Cases 1, 1b and 4 touch TF ops (`tf.image.resize` inside
`as_tf_dataset()`); they are pinned to CPU via `_pin_cpu_only` since none of
them need a GPU and a reviewer reproduced a genuine CUDA OOM here under real
external GPU contention on this machine (D-011).
"""

import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.datasets.pusht_hdf5 import (
    PushTHDF5Dataset,
    _IMAGENET_MEAN,
    _IMAGENET_STD,
)

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


def _pin_cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force this test process's TF ops onto CPU, best-effort.

    Two layers, in order of strength: (1) ``CUDA_VISIBLE_DEVICES=""`` is set
    before TF's lazy CUDA device enumeration runs — authoritative when this
    is the first TF op in the process, which is the exact scenario a
    reviewer reproduced a genuine ``CUDA_ERROR_OUT_OF_MEMORY`` crash in under
    real external GPU contention (`findings/review-iter-3.md` Concern #6).
    (2) ``tf.config.set_visible_devices([], "GPU")`` is attempted as a
    second, TF-native mechanism, tolerating the ``RuntimeError`` TF raises
    when GPUs were already initialized earlier in the same pytest session —
    that fallback case is outside a single test's control. Callers: any test
    in this module invoking `PushTHDF5Dataset.as_tf_dataset()`.

    :param monkeypatch: the test's `monkeypatch` fixture.
    :return: None (mutates process/TF global device state for this test).
    """
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError:
        pass


def test_as_tf_dataset_yields_documented_shapes_and_normalizes_pixels(tmp_path, monkeypatch):
    _pin_cpu_only(monkeypatch)
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


def test_pixel_normalization_matches_imagenet_stats_exactly(tmp_path, monkeypatch):
    # A constant-valued (all-zero) fixture frame has a single, exactly
    # predictable normalized value per channel: (0/255 - mean) / std =
    # -mean/std, approx [-2.118, -2.036, -1.804]. Unlike
    # `np.any(pixels < 0)`, this distinguishes correct ImageNet
    # normalization from a wrong shift (e.g. `- 0.5`) or swapped mean/std
    # constants, both of which produce a different constant value
    # (REFLECT D-011, findings/review-iter-3.md Concern #3).
    _pin_cpu_only(monkeypatch)
    n, h0, w0, action_dim = 4, 16, 16, 2
    pixels = np.zeros((n, h0, w0, 3), dtype=np.uint8)
    action = np.zeros((n, action_dim), dtype=np.float32)
    h5_path = tmp_path / "pusht_const.h5"
    _write_pusht_h5(str(h5_path), pixels, action, episode_ends=[n])

    dataset = PushTHDF5Dataset(
        str(h5_path),
        img_size=16,
        action_dim=action_dim,
        history_size=2,
        num_preds=1,
        batch_size=1,
    )
    x, _y = next(iter(dataset.as_tf_dataset()))

    expected = (-_IMAGENET_MEAN / _IMAGENET_STD).astype(np.float32)
    got = x["pixels"].numpy()[0, 0, 0, 0, :]  # first window, first frame, one pixel
    assert np.allclose(got, expected, atol=1e-5)


def test_load_metadata_replaces_nan_actions_with_zero(tmp_path):
    rng = np.random.default_rng(0)
    n, h0, w0, action_dim = 20, 48, 48, 2
    pixels = rng.integers(0, 256, size=(n, h0, w0, 3), dtype=np.uint8)
    action = rng.standard_normal((n, action_dim)).astype(np.float32)
    action_before_nan = action.copy()
    nan_row = 5
    action[nan_row, :] = np.nan
    h5_path = tmp_path / "pusht.h5"
    _write_pusht_h5(str(h5_path), pixels, action, episode_ends=[n])

    dataset = PushTHDF5Dataset(str(h5_path), action_dim=action_dim)
    loaded_action, _ends, n_pixels = dataset._load_metadata()

    assert n_pixels == n
    assert not np.any(np.isnan(loaded_action))
    assert np.array_equal(loaded_action[nan_row], np.array([0.0, 0.0], dtype=np.float32))
    # Non-NaN rows must survive unchanged. Without this, `_load_metadata`
    # replacing ALL actions (e.g. `np.zeros_like(action)`) would still pass
    # every prior assertion here (REFLECT D-011, review Concern #1: M1).
    non_nan_mask = np.ones(n, dtype=bool)
    non_nan_mask[nan_row] = False
    assert np.allclose(loaded_action[non_nan_mask], action_before_nan[non_nan_mask])


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


def test_as_tf_dataset_windows_never_straddle_episode_boundary_end_to_end(
    tmp_path, monkeypatch
):
    # `_window_starts()` alone (Case 3) cannot see whether `as_tf_dataset()`'s
    # generator actually pairs `pixels[i:i+T]` with `action[i:i+T-1]` for the
    # starts it computes — a pipeline-level misalignment is a silent
    # corruption with no shape/dtype symptom (REFLECT D-011, review
    # Concern #2: M7). Index-encode every frame (`pixels[i, ...] = i`,
    # `action[i] = [i, -i]`) across 2 real episodes so each emitted window
    # can be decoded back to the frame indices it actually contains and
    # checked against what it SHOULD contain.
    _pin_cpu_only(monkeypatch)
    n, h0, w0, action_dim = 20, 8, 8, 2
    episode_ends = [10, 20]
    pixels = np.zeros((n, h0, w0, 3), dtype=np.uint8)
    for i in range(n):
        pixels[i, ...] = i
    action = np.stack(
        [np.arange(n, dtype=np.float32), -np.arange(n, dtype=np.float32)], axis=1
    )
    h5_path = tmp_path / "pusht_e2e.h5"
    _write_pusht_h5(str(h5_path), pixels, action, episode_ends=episode_ends)

    history_size, num_preds = 3, 1
    T = history_size + num_preds
    dataset = PushTHDF5Dataset(
        str(h5_path),
        img_size=h0,  # == input size: resize is a no-op, preserving the index
        action_dim=action_dim,
        history_size=history_size,
        num_preds=num_preds,
        batch_size=1,
    )
    expected_starts = dataset._window_starts(
        np.asarray(episode_ends, dtype=np.int64)
    )
    tf_dataset = dataset.as_tf_dataset()

    seen_starts = []
    for x, _y in tf_dataset.take(len(expected_starts)):
        win_pixels = x["pixels"].numpy()[0]  # (T, h0, w0, 3)
        win_action = x["action"].numpy()[0]  # (T - 1, action_dim)
        # Invert the resize (no-op here) + ImageNet normalization to recover
        # the encoded frame index from one pixel per frame.
        decoded = np.round(
            (win_pixels[:, 0, 0, 0] * _IMAGENET_STD[0] + _IMAGENET_MEAN[0]) * 255.0
        ).astype(int)
        start = int(decoded[0])
        seen_starts.append(start)
        # (a) every window's frame indices are consecutive
        assert np.array_equal(decoded, np.arange(start, start + T))
        # (b) no window straddles the episode boundary at index 10
        assert start + T <= 10 or start >= 10
        # (c) actions match the expected frame indices for this window
        frame_idx = np.arange(start, start + T - 1, dtype=np.float32)
        expected_action = np.stack([frame_idx, -frame_idx], axis=1)
        assert np.allclose(win_action, expected_action)

    # Every valid window was emitted exactly once across the one epoch pulled.
    assert sorted(seen_starts) == sorted(expected_starts.tolist())


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
