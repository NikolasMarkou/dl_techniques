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

6. Non-square input frames (`h0 != w0`, e.g. 48x32): every prior fixture uses
   a square frame, so a bug that swaps axes inside `_preprocess_pair`'s
   `tf.image.resize(pixels_f, [img_size, img_size])` call — or drops the
   resize entirely — would be invisible, since a square input already
   produces a square output by coincidence. This case forces the resize to
   actually run and checks the output shape (still `img_size x img_size`,
   not `h0 x w0`-shaped). The fixture content is the per-pixel formula
   `(3*y + 7*x) % 256` — spatially ASYMMETRIC, not a constant — so it
   distinguishes a correct resize from an axis-swapped one because a
   transpose inserted before the resize reads a different raw value at any
   coordinate where `y != x`. The exact post-normalization value is checked
   at such a coordinate (on the resize's untouched edge column, where
   bilinear interpolation is exact and the expected value stays
   hand-computable). An earlier revision used a spatially-INVARIANT
   (constant) fixture here, which is a no-op under any spatial permutation
   and provably does NOT catch this bug class (REFLECT D-004).
7. Three episodes of deliberately UNEQUAL, non-round length (7, 15, 9): the
   existing boundary tests (Cases 3-4) use only 2 episodes, so a mutant that
   validates only the FIRST episode boundary (rather than iterating every
   boundary in `ends`) is indistinguishable from correct code there — with
   one internal boundary, "check the first" and "check every" are the same
   code path. Three unequal episodes create 2 distinct internal boundaries,
   strong enough to catch a first-boundary-only mutant that would silently
   admit a straddling window at the second boundary.
8. Spatially AND temporally correlated pixel content (a per-frame linear
   ramp, `(i + 2*y + 3*x + 5*c) % 256` — distinct, asymmetric coefficients
   per axis so the raw value is not invariant under a y<->x transpose either
   — rather than uniform random noise or a flat constant): prior fixtures
   are either mutually indistinguishable across frames in aggregate (uniform
   random) or numerically identical across all frames in a window
   (constant), so neither can catch a mutant that applies correct
   normalization to the WRONG frame within a window (e.g. an off-by-one
   window shift in the generator). This fixture makes every (frame, y, x, c)
   coordinate map to a unique raw value; combined with a window-start-set
   assertion (mirroring Case 4's `sorted(seen) == sorted(expected)` over a
   full epoch), a one-frame shift is caught because the emitted start set no
   longer matches `_window_starts()`'s own output — REFLECT D-004 corrects
   an earlier revision's inline comment, which misattributed this catch to
   the decoded per-value formula alone; that formula's `start` is read back
   from the (possibly shifted) data itself, so a uniform shift is invisible
   to it in isolation.

Only Cases 1, 1b, 4, 6 and 8 touch TF ops (`tf.image.resize` inside
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

    # Standard ImageNet normalization constants, hard-coded here
    # independently of the module under test — do not import from
    # pusht_hdf5. REFLECT D-004: every expected value in this file was
    # computed by importing `_IMAGENET_MEAN`/`_IMAGENET_STD` from
    # `pusht_hdf5` itself, so a wrong-constant bug in the module (e.g. both
    # replaced with `[0.5, 0.5, 0.5]`) left this test green (measured: 9/9
    # pass under that substitution). The two assertions below are the fix
    # for THIS test specifically — other tests in this file import the
    # constants only to compute OTHER things (window content, boundaries),
    # not to verify the constants' own values, so they don't need this
    # treatment.
    literal_imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    literal_imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    assert np.array_equal(_IMAGENET_MEAN.astype(np.float32), literal_imagenet_mean)
    assert np.array_equal(_IMAGENET_STD.astype(np.float32), literal_imagenet_std)

    expected = (-literal_imagenet_mean / literal_imagenet_std).astype(np.float32)
    got = x["pixels"].numpy()[0, 0, 0, 0, :]  # first window, first frame, one pixel
    assert np.allclose(got, expected, atol=1e-5)


def test_as_tf_dataset_handles_non_square_frames(tmp_path, monkeypatch):
    # Every existing fixture in this file uses a SQUARE input frame
    # ((48,48), (16,16), (8,8)), so a bug that swaps axes inside
    # `_preprocess_pair`'s `tf.image.resize(pixels_f, [img_size, img_size])`
    # call — or drops the resize entirely — would be invisible: a square
    # input already produces a square output by coincidence. Using
    # `h0 != w0` here forces the resize to actually run and produce a
    # SQUARE `img_size x img_size` output from a non-square input, so
    # either mutant (no resize, or an axis-swapped resize target) would
    # leave `x["pixels"].shape` at `(h0, w0)`-shaped, not `(img_size,
    # img_size)`-shaped — assertion (a) below would catch it directly.
    #
    # `img_size == h0` (per plan.md's Assumptions: keep the resize math
    # tractable by hand) means the height axis is a no-op resize and only
    # the width axis (32 -> 48) is actually rescaled. REFLECT D-004: an
    # earlier revision used a per-frame CONSTANT pixel value here, which is
    # spatially INVARIANT — an H/W transpose inserted before the resize call
    # is a no-op on every value of a constant field, so that fixture cannot
    # distinguish a correct resize from an axis-swapped one (measured: all 9
    # tests stayed GREEN under that exact mutant). The fixture below is
    # spatially ASYMMETRIC instead: `pixels[i, y, x, c] = (3*y + 7*x) % 256`
    # varies differently along y than along x, so a transpose reads a
    # different raw value at any `y != x` coordinate.
    #
    # Assertion (b) below checks the exact value at `(y=5, x=0)` — `y != x`,
    # satisfying the requirement above. `x=0` is deliberately the resize's
    # EDGE column: `tf.image.resize`'s bilinear sampling clamps out-of-range
    # source coordinates at the edges, so column 0 (and the last column) of
    # the resized output equals column 0 (last column) of the input EXACTLY,
    # regardless of scale factor (verified empirically: resizing a
    # `(3*y + 7*x) % 256` field from width 32 to 48 leaves `out[:, 0, :]` and
    # `out[:, -1, :]` bit-identical to the corresponding input column). The
    # height axis is already a no-op (`img_size == h0`). This keeps the
    # expected value hand-computable — `(3*5 + 7*0) % 256 == 15` — without
    # simulating the bilinear kernel for interior columns.
    _pin_cpu_only(monkeypatch)
    n, h0, w0, action_dim = 4, 48, 32, 2
    img_size = h0  # one of h0/w0, per plan.md Assumptions
    y_check, x_check = 5, 0  # y != x; x_check=0 is the exact-edge column
    yy = np.arange(h0, dtype=np.int64).reshape(1, h0, 1, 1)
    xx = np.arange(w0, dtype=np.int64).reshape(1, 1, w0, 1)
    pixels = np.broadcast_to((3 * yy + 7 * xx) % 256, (n, h0, w0, 3)).astype(
        np.uint8
    )
    action = np.zeros((n, action_dim), dtype=np.float32)
    h5_path = tmp_path / "pusht_nonsquare.h5"
    _write_pusht_h5(str(h5_path), pixels, action, episode_ends=[n])

    dataset = PushTHDF5Dataset(
        str(h5_path),
        img_size=img_size,
        action_dim=action_dim,
        history_size=2,
        num_preds=1,
        batch_size=1,
    )
    x, _y = next(iter(dataset.as_tf_dataset()))

    # (a) output is SQUARE post-resize even though the input was not —
    # would fail under either mutant named above.
    assert x["pixels"].shape == (1, 3, img_size, img_size, 3)

    # (b) exact normalized value at a `y != x` coordinate on the resize's
    # exact edge column — independently re-derived from the fixture's own
    # construction formula; a transpose mutant reads a different raw value
    # here (see comment above).
    raw = (3 * y_check + 7 * x_check) % 256
    expected = ((raw / 255.0) - _IMAGENET_MEAN) / _IMAGENET_STD
    got = x["pixels"].numpy()[0, 0, y_check, x_check, :]  # first window, first frame
    assert np.allclose(got, expected.astype(np.float32), atol=1e-5)


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


def test_window_starts_and_dataset_respect_three_unequal_length_episodes(
    tmp_path, monkeypatch
):
    # The existing boundary tests (Cases 3-4) use exactly 2 episodes of
    # ROUND, equal-ish length (10, 10). A boundary-check bug that only
    # validates the FIRST episode transition (e.g. hardcoding the single
    # boundary rather than iterating `ends`) would be invisible there: with
    # only one internal boundary, "check the first boundary" and "check
    # every boundary" are the same code path. Using 3 episodes of
    # DELIBERATELY unequal, non-round lengths (7, 15, 9 — none equal to each
    # other, to T, or to a round number) creates 2 DISTINCT internal
    # boundaries, so a mutant that only validates the first one would pass
    # the boundary-1 check but silently admit a straddling window at
    # boundary 2.
    _pin_cpu_only(monkeypatch)
    episode_lengths = [7, 15, 9]
    n = sum(episode_lengths)
    episode_ends = list(np.cumsum(episode_lengths))  # [7, 22, 31]
    h0, w0, action_dim = 8, 8, 2
    # Index-encode every frame, same pattern as the Case-4 end-to-end test,
    # so each emitted window can be decoded back to the frame indices it
    # actually contains.
    pixels = np.zeros((n, h0, w0, 3), dtype=np.uint8)
    for i in range(n):
        pixels[i, ...] = i
    action = np.stack(
        [np.arange(n, dtype=np.float32), -np.arange(n, dtype=np.float32)], axis=1
    )
    h5_path = tmp_path / "pusht_three_episodes.h5"
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

    # Independently re-derive the expected per-episode window-start set with
    # the SAME "while i + T <= end" rule as `test_window_starts_never_crosses_an_episode_boundary`
    # — regenerated fresh for this 3-episode, unequal-length case, not copied.
    expected_starts = []
    prev_end = 0
    for end in episode_ends:
        i = prev_end
        while i + T <= end:
            expected_starts.append(i)
            i += 1
        prev_end = end
    expected_starts = sorted(expected_starts)

    actual_starts = dataset._window_starts(np.asarray(episode_ends, dtype=np.int64))
    assert sorted(actual_starts.tolist()) == expected_starts

    # (b) the window count matches the independently-derived count EXACTLY.
    assert len(expected_starts) == 22  # hand check: 4 (ep1) + 12 (ep2) + 6 (ep3)

    # (a) no window straddles EITHER internal boundary (7 and 22). REFLECT
    # D-004: an earlier revision iterated `expected_starts` here — the
    # test's OWN re-derived value — making this true BY CONSTRUCTION
    # regardless of what `_window_starts` actually returns (it can never
    # fail). Iterating `actual_starts` (the real, `_window_starts`-derived
    # value asserted equal to `expected_starts` above) is the assertion a
    # "first-boundary-only" mutant would actually fail on boundary 2.
    for s in actual_starts.tolist():
        for boundary in (episode_ends[0], episode_ends[1]):
            assert s + T <= boundary or s >= boundary

    # (c) drive the REAL end-to-end pipeline (not `_window_starts()` alone)
    # and confirm the emitted window-start set is exactly the expected set,
    # and that no window straddles either boundary when read from actual
    # decoded frame content.
    tf_dataset = dataset.as_tf_dataset()
    seen_starts = []
    for x, _y in tf_dataset.take(len(expected_starts)):
        win_pixels = x["pixels"].numpy()[0]  # (T, h0, w0, 3)
        win_action = x["action"].numpy()[0]  # (T - 1, action_dim)
        decoded = np.round(
            (win_pixels[:, 0, 0, 0] * _IMAGENET_STD[0] + _IMAGENET_MEAN[0]) * 255.0
        ).astype(int)
        start = int(decoded[0])
        seen_starts.append(start)
        assert np.array_equal(decoded, np.arange(start, start + T))
        for boundary in (episode_ends[0], episode_ends[1]):
            assert start + T <= boundary or start >= boundary
        frame_idx = np.arange(start, start + T - 1, dtype=np.float32)
        expected_action = np.stack([frame_idx, -frame_idx], axis=1)
        assert np.allclose(win_action, expected_action)

    assert sorted(seen_starts) == expected_starts


def test_as_tf_dataset_with_correlated_pixel_content_preserves_temporal_structure(
    tmp_path, monkeypatch
):
    # Every fixture so far is either uniform random noise (Cases 1/4/5 —
    # frames are mutually indistinguishable in aggregate, so a
    # wrong-frame-within-window bug wouldn't shift any single-value
    # assertion) or a per-frame CONSTANT (Cases 1b, non-square, 3-episode —
    # all frames in a window are numerically identical, so which frame a
    # given assertion "actually" reads is unobservable). Neither extreme can
    # catch a mutant that applies the correct normalization formula to the
    # WRONG frame within a window (e.g. reading `pixels_ds[i+1 : i+1+T]`
    # instead of `pixels_ds[i : i+T]` in `as_tf_dataset`'s generator — a
    # window silently shifted by one). A per-frame linear ramp
    # (`(i + 2*y + 3*x + 5*c) % 256`, distinct asymmetric coefficients per
    # axis) is spatially AND temporally distinct: every (frame, y, x, c)
    # coordinate maps to a unique raw value.
    #
    # REFLECT D-004 corrects an earlier revision's claim here. That revision
    # said a one-frame window shift is caught because "the decoded `start`
    # value from frame 0 no longer lines up with the formula used at frames
    # t=2 and t=3" — this is FALSE: `start` below is decoded FROM the
    # window's own frame-0 pixel, so a uniform shift of the whole window is
    # algebraically invisible to a check that only ever compares the window
    # against a `start` read back from itself (whatever the window's actual
    # content is, `start` is redefined to match it, and the per-value
    # assertions below hold by construction regardless of which real frame
    # the window started at). The one-frame-shift mutant the earlier revision
    # ran DID go RED, but for an unrelated reason: with only one window
    # pulled via `next(iter(...))`, shifting `win_pixels = pixels_ds[i+1 :
    # i+1+T]` at the sequence's END (`i == n - T`, the max valid start) reads
    # past the end of the `pixels` dataset, and `tf.data`'s `output_signature`
    # rejects the resulting short/misshapen array before this test's own
    # assertions even run — a shuffle-order accident, not the claimed
    # mechanism.
    #
    # The window-start-set assertion below, added in this fix and mirroring
    # Case 4's `sorted(seen) == sorted(expected)` pattern, is what actually,
    # robustly catches a window-shift mutant: shifting every window's read
    # by one changes which `start` values the generator's OUTPUT decodes to
    # (each window still decodes internally-consistently, but the SET of
    # decoded starts no longer matches `_window_starts()`'s own output),
    # independent of any single window's per-value formula.
    _pin_cpu_only(monkeypatch)
    n, h0, w0, action_dim = 6, 8, 8, 2
    img_size = h0  # no-op resize (== h0 == w0), per plan.md Assumptions
    history_size, num_preds = 3, 1
    T = history_size + num_preds  # 4; valid starts in [0, n - T] = [0, 2]

    idx = np.arange(n, dtype=np.int64).reshape(n, 1, 1, 1)
    yy = np.arange(h0, dtype=np.int64).reshape(1, h0, 1, 1)
    xx = np.arange(w0, dtype=np.int64).reshape(1, 1, w0, 1)
    cc = np.arange(3, dtype=np.int64).reshape(1, 1, 1, 3)
    pixels = np.broadcast_to(
        (idx + 2 * yy + 3 * xx + 5 * cc) % 256, (n, h0, w0, 3)
    ).astype(np.uint8)
    action = np.zeros((n, action_dim), dtype=np.float32)
    h5_path = tmp_path / "pusht_correlated.h5"
    _write_pusht_h5(str(h5_path), pixels, action, episode_ends=[n])

    dataset = PushTHDF5Dataset(
        str(h5_path),
        img_size=img_size,
        action_dim=action_dim,
        history_size=history_size,
        num_preds=num_preds,
        batch_size=1,
    )
    expected_starts = dataset._window_starts(np.asarray([n], dtype=np.int64))
    tf_dataset = dataset.as_tf_dataset()

    # Decode a window's start index from frame 0's (y=0, x=0, c=0) pixel: by
    # construction its raw value is `(start + 2*0 + 3*0 + 5*0) % 256 ==
    # start` (no wraparound since `start <= n - T = 2 < 256`). This makes the
    # assertions below robust to `as_tf_dataset`'s window-order shuffle — we
    # don't assume `start == 0`, we recover it from the data itself, same
    # discipline as the end-to-end tests above.
    def _decode_raw(value: float, channel: int) -> int:
        return int(
            round(
                (float(value) * _IMAGENET_STD[channel] + _IMAGENET_MEAN[channel])
                * 255
            )
        )

    seen_starts = []
    for x, _y in tf_dataset.take(len(expected_starts)):
        got = x["pixels"].numpy()[0]  # (T, h0, w0, 3)
        start = _decode_raw(got[0, 0, 0, 0], 0)
        assert 0 <= start <= n - T
        seen_starts.append(start)

        # (2+) specific (frame, y, x, c) coordinates, spanning distinct
        # frames AND distinct spatial positions, each checked against an
        # independently-computed expected value derived from the fixture's
        # own construction formula plus the decoded `start`.
        for t, y_coord, x_coord, c_coord in [(0, 0, 0, 0), (2, 3, 5, 1), (3, 7, 7, 2)]:
            raw = (start + t + 2 * y_coord + 3 * x_coord + 5 * c_coord) % 256
            expected = (
                raw / 255.0 - _IMAGENET_MEAN[c_coord]
            ) / _IMAGENET_STD[c_coord]
            assert np.isclose(
                got[t, y_coord, x_coord, c_coord], expected, atol=1e-5
            ), f"mismatch at (start={start}, t={t}, y={y_coord}, x={x_coord}, c={c_coord})"

    # The catching mechanism for a window-shift mutant, per the comment
    # above: the SET of decoded starts across a full epoch must match
    # `_window_starts()`'s own output exactly.
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
