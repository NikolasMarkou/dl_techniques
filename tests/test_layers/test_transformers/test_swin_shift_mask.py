"""SW-MSA cross-region leakage guard for :class:`SwinTransformerBlock`.

Before ``e25d2bac`` (plan ``plan-2026-07-31-ddc92265``, finding F-01) the block
applied the cyclic roll and the window partition of shifted-window attention but
never built or applied the SW-MSA attention mask.  Every shifted block therefore
performed *full* attention over each physical window, so tokens that the roll
brought together from opposite edges of the feature map attended to one another.
That mask now exists, and this module is what keeps it honest.

This module proved that defect with a perturbation-isolation probe, and it
derives the ground truth for "which token pairs are allowed" **twice, by two
independent routes**, so that a wrong oracle cannot silently define the bug
away:

1. *Wrap status* (first-principles).  Roll an index grid holding the flat
   original coordinate of every token, partition it with the very same
   :func:`window_partition` the block uses, then recover each slot's original
   ``(row, col)``.  A rolled row wrapped around the top edge iff its original
   row is ``< shift_size``; likewise for columns.  Two tokens sharing a physical
   window may attend iff their ``(row_wrapped, col_wrapped)`` pair is equal.
2. *Reference 3x3 region image* (canonical Swin).  The standard ``img_mask``
   slice-counter construction from the reference Swin implementation (mirrored
   in-repo at ``layers/attention/progressive_focused_attention.py`` in
   ``_compute_attention_mask``), partitioned with the same function.

The two labellings are **not** identical globally -- the reference splits the
non-wrapped rows into two slices while wrap status does not -- but the pairwise
"same class?" relation they induce **inside each physical window** must agree
exactly.  :func:`TestSwMsaOracle.test_wrap_status_matches_reference_regions`
asserts that; if it ever fails, the oracle is wrong and the leakage probe below
proves nothing (plan Pre-Mortem #1: STOP, do not tune the oracle to match).

Test inventory (kept in step with the file -- it was stale once already):

* ``TestSwMsaOracle`` -- the oracle self-check described above, plus a
  non-vacuity check that the probe config really does mix wrap statuses inside
  a window.
* ``TestSwMsaLeakage`` -- the F-01 leakage probe.  It was RED at the pre-fix
  commit (perturbing token ``(2, 6)`` moved same-window token ``(2, 0)`` by
  ``1.001258e-01`` on 32/32 channels) and is green at ``rtol=0, atol=0`` now
  that the mask is built and applied.  A second, parametrized test repeats the
  same isolation assertion under ``mixed_float16`` and ``mixed_bfloat16``,
  where the additive mask bias is ``-inf`` and this package has shipped a
  ``0 * -inf = NaN`` defect before.
* ``TestSwinShiftMaskControls`` -- controls that were green both before and
  after that fix: window isolation at ``shift_size=0`` (the probe harness
  itself works) and a full ``.keras`` save/load round-trip of a shifted block.
* ``TestSwMsaMaskConstruction`` -- the shipped mask against the oracle, the
  D-006 single-window fallback (``H == window_size`` reduces to
  ``shift_size=0``), the strictly-below-``window_size`` raise, that the raise is
  specific to ``shift_size > 0``, and dynamic spatial dims (never raise,
  supported at several resolutions, dynamic-vs-static build agreement).
* ``TestShapeIndependenceAtTraceTime`` -- the D-012 guard: the same block on
  the same tensor must answer identically whether or not the tracer knows the
  spatial extent, plus a self-check that the guard can see a wrong runtime
  shift.

The two *shipped consumers* of this block are guarded numerically elsewhere, and
not here: ``tests/test_models/test_scunet/test_golden_values.py`` and
``tests/test_models/test_swin_transformer/test_golden_values.py``.
"""

import contextlib
import os
import tempfile

import keras
import numpy as np
import pytest
import tensorflow as tf
from keras import ops

from dl_techniques.layers.attention.window_attention import WindowAttention
from dl_techniques.layers.ffn.swin_mlp import SwinMLP
from dl_techniques.layers.stochastic_depth import StochasticDepth
from dl_techniques.layers.transformers.swin_transformer_block import (
    SwinTransformerBlock,
)
from dl_techniques.utils.logger import logger
from dl_techniques.utils.tensors import window_partition

# --- Frozen probe configuration (plan step 1). -------------------------------
DIM = 32
NUM_HEADS = 4
WINDOW_SIZE = 4
SHIFT_SIZE = 2
HEIGHT = 8
WIDTH = 8
BATCH = 2
SEED = 1234


# ---------------------------------------------------------------------------
# Oracles
# ---------------------------------------------------------------------------


def wrap_status_windows(
    height: int, width: int, window_size: int, shift_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Label every window slot by the wrap status of the token it holds.

    Derived from first principles and *independently of any mask
    implementation*: an index grid carrying each token's flat original
    coordinate is rolled by ``(-shift_size, -shift_size)`` and partitioned with
    the same :func:`window_partition` the block itself calls, so the returned
    arrays are in exactly the block's window order (assumption A3: B-major /
    window-minor).

    Args:
        height: Feature-map height. Must be divisible by ``window_size``.
        width: Feature-map width. Must be divisible by ``window_size``.
        window_size: Window edge length.
        shift_size: Cyclic shift applied before partitioning.

    Returns:
        A ``(status, orig_row, orig_col)`` triple of int arrays, each of shape
        ``(num_windows, window_size ** 2)``. ``status`` encodes
        ``2 * row_wrapped + col_wrapped``; ``orig_row``/``orig_col`` give the
        pre-roll coordinates of the token occupying each slot.
    """
    flat_index = np.arange(height * width, dtype="float32").reshape(
        1, height, width, 1
    )
    rolled = ops.roll(
        ops.convert_to_tensor(flat_index),
        shift=(-shift_size, -shift_size),
        axis=(1, 2),
    )
    windows = window_partition(rolled, window_size)
    windows = np.asarray(
        ops.convert_to_numpy(
            ops.reshape(windows, (-1, window_size * window_size))
        )
    ).astype("int64")

    orig_row = windows // width
    orig_col = windows % width

    # A rolled row i holds original row (i + shift) % height; it wrapped around
    # the top edge iff i + shift >= height, i.e. iff its original row < shift.
    row_wrapped = (orig_row < shift_size).astype("int64")
    col_wrapped = (orig_col < shift_size).astype("int64")
    status = 2 * row_wrapped + col_wrapped
    return status, orig_row, orig_col


def reference_region_windows(
    height: int, width: int, window_size: int, shift_size: int
) -> np.ndarray:
    """Build the canonical Swin 3x3-region label image, partitioned to windows.

    This is the standard ``img_mask`` slice-counter construction of the
    reference Swin implementation (and of the in-repo
    ``ProgressiveFocusedAttention._compute_attention_mask``), reproduced here
    with no dependency on :func:`wrap_status_windows`.

    Args:
        height: Feature-map height.
        width: Feature-map width.
        window_size: Window edge length.
        shift_size: Cyclic shift applied before partitioning.

    Returns:
        Int array of shape ``(num_windows, window_size ** 2)`` holding the
        region index (0..8) of every window slot.
    """
    img_mask = np.zeros((1, height, width, 1), dtype="float32")
    h_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    w_slices = h_slices
    count = 0
    for h_slice in h_slices:
        for w_slice in w_slices:
            img_mask[:, h_slice, w_slice, :] = count
            count += 1

    windows = window_partition(ops.convert_to_tensor(img_mask), window_size)
    return np.asarray(
        ops.convert_to_numpy(
            ops.reshape(windows, (-1, window_size * window_size))
        )
    ).astype("int64")


def _same_class(labels: np.ndarray) -> np.ndarray:
    """Return the per-window pairwise "same label" relation.

    Args:
        labels: Int array of shape ``(num_windows, tokens_per_window)``.

    Returns:
        Bool array of shape ``(num_windows, tokens_per_window,
        tokens_per_window)``.
    """
    return labels[:, :, None] == labels[:, None, :]


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------


def _make_block(
    shift_size: int, build_shape: tuple = (None, HEIGHT, WIDTH, DIM)
) -> SwinTransformerBlock:
    """Build a deterministic probe block (no dropout, no stochastic depth).

    Args:
        shift_size: Cyclic shift for the block.
        build_shape: Shape passed to ``build()``. Pass ``(None, None, None,
            DIM)`` to reproduce the ``models/thera`` construction, whose
            spatial dims are unknown at build time.

    Returns:
        A built ``SwinTransformerBlock`` with seeded weights.
    """
    keras.utils.set_random_seed(SEED)
    block = SwinTransformerBlock(
        dim=DIM,
        num_heads=NUM_HEADS,
        window_size=WINDOW_SIZE,
        shift_size=shift_size,
        dropout_rate=0.0,
        attention_dropout_rate=0.0,
        stochastic_depth_rate=0.0,
    )
    block.build(build_shape)
    return block


def _probe_input() -> np.ndarray:
    """Fixed-seed float32 probe input of shape ``(B, H, W, C)``."""
    rng = np.random.default_rng(SEED)
    return rng.standard_normal((BATCH, HEIGHT, WIDTH, DIM)).astype("float32")


def _perturbation() -> np.ndarray:
    """A per-channel *non-uniform* perturbation vector of shape ``(C,)``.

    Non-uniformity is load-bearing. The block's ``norm1`` is a
    ``LayerNormalization`` over the channel axis, so adding the *same* scalar to
    every channel of a token is mean-centred away and reaches the attention
    almost unchanged -- a probe built on a constant shift moves downstream
    tokens by ~1e-7 (pure float reassociation) and would report a bogus RED.
    """
    rng = np.random.default_rng(SEED + 1)
    return (3.0 * rng.standard_normal(DIM)).astype("float32")


def _forward(block: SwinTransformerBlock, x: np.ndarray) -> np.ndarray:
    """Deterministic inference-mode forward pass returning a numpy array."""
    return np.asarray(
        ops.convert_to_numpy(block(ops.convert_to_tensor(x), training=False))
    )


@contextlib.contextmanager
def global_dtype_policy(name: str):
    """Run a block under a global Keras dtype policy, restoring it afterwards.

    Restoration is in a ``finally`` so a failing assertion cannot leak
    ``mixed_float16`` into the rest of the session -- a leaked policy would
    silently change every later test in the module.
    """
    previous = keras.mixed_precision.dtype_policy()
    keras.mixed_precision.set_dtype_policy(name)
    try:
        yield
    finally:
        keras.mixed_precision.set_dtype_policy(previous)


def _pick_leakage_pair(status: np.ndarray) -> tuple[int, int, int, int]:
    """Find a window holding two wrap statuses, and pick a probe triple in it.

    Returns ``(window_index, slot_a, slot_b, slot_same)`` where ``slot_a`` and
    ``slot_b`` carry DIFFERENT wrap statuses (so they must not attend to each
    other) and ``slot_same`` shares ``slot_a``'s status (so it must move, which
    is the live-probe control that stops the isolation assertion from being
    satisfiable by a dead probe).
    """
    for window_index in range(status.shape[0]):
        row = status[window_index]
        differing = np.argwhere(row[:, None] != row[None, :])
        if not differing.size:
            continue
        slot_a, slot_b = (int(v) for v in differing[0])
        same = [int(s) for s in np.argwhere(row == row[slot_a]).ravel() if int(s) != slot_a]
        if not same:
            continue
        return window_index, slot_a, slot_b, same[0]
    raise AssertionError("No mixed-status window in the probe config.")


# ---------------------------------------------------------------------------
# 1. Oracle self-check
# ---------------------------------------------------------------------------


class TestSwMsaOracle:
    """Cross-validate the wrap-status oracle against the canonical regions."""

    def test_wrap_status_matches_reference_regions(self):
        """The two independent labellings must induce the same per-window relation.

        They are deliberately *not* compared as labels (the reference is a
        strictly finer partition globally); what must match is the pairwise
        "may these two tokens attend?" relation restricted to each physical
        window.
        """
        status, _, _ = wrap_status_windows(
            HEIGHT, WIDTH, WINDOW_SIZE, SHIFT_SIZE
        )
        regions = reference_region_windows(
            HEIGHT, WIDTH, WINDOW_SIZE, SHIFT_SIZE
        )

        assert status.shape == regions.shape

        keep_from_wrap = _same_class(status)
        keep_from_regions = _same_class(regions)

        for window_index in range(status.shape[0]):
            np.testing.assert_array_equal(
                keep_from_wrap[window_index],
                keep_from_regions[window_index],
                err_msg=(
                    f"Oracle disagreement in window {window_index}: the "
                    f"wrap-status labelling {status[window_index].tolist()} "
                    f"does not induce the same pairwise relation as the "
                    f"reference regions {regions[window_index].tolist()}. "
                    f"Plan Pre-Mortem #1 trigger: STOP, do not tune the "
                    f"oracle to match."
                ),
            )

        logger.info(
            "SW-MSA oracle cross-check passed for H=%d W=%d ws=%d shift=%d "
            "over %d windows.",
            HEIGHT,
            WIDTH,
            WINDOW_SIZE,
            SHIFT_SIZE,
            status.shape[0],
        )

    def test_probe_config_has_a_mixed_window(self):
        """At least one physical window must mix two wrap statuses.

        Without such a window the leakage probe below would be vacuous: it
        could only ever compare tokens that are legitimately allowed to attend
        to each other.
        """
        status, _, _ = wrap_status_windows(
            HEIGHT, WIDTH, WINDOW_SIZE, SHIFT_SIZE
        )
        distinct_per_window = [len(np.unique(row)) for row in status]
        assert max(distinct_per_window) >= 2, (
            "Probe config produces no window containing more than one wrap "
            f"status: {distinct_per_window}"
        )


# ---------------------------------------------------------------------------
# 2. The leakage probe
# ---------------------------------------------------------------------------


class TestSwMsaLeakage:
    """Cross-region leakage inside a shifted window.

    Historically this class was the plan's RED probe and carried an
    ``xfail(strict=True)`` marker: before ``e25d2bac`` the block never built an
    SW-MSA mask, so a token in wrap-region A influenced same-window tokens in
    wrap-region B (measured: perturbing ``(2, 6)`` moved ``(2, 0)`` by
    ``1.001258e-01`` on 32/32 channels). The mask landed, the marker was
    removed, and both tests below are ordinary green guards asserting exact
    isolation -- one at the session default precision, one under mixed
    precision.
    """

    def test_no_cross_region_leakage_within_window(self):
        """Perturbing a region-A token must not move any same-window region-B token.

        The comparison is bit-exact (``rtol=0, atol=0``): with a correct
        SW-MSA mask the disallowed key's softmax weight is exactly ``0.0``, so
        its value contributes an exact zero to the reduction and the region-B
        outputs are unchanged to the last bit.
        """
        status, orig_row, orig_col = wrap_status_windows(
            HEIGHT, WIDTH, WINDOW_SIZE, SHIFT_SIZE
        )

        # Pick a window that mixes wrap statuses, then a disallowed (A, B) pair
        # inside it (plus a same-status partner as the live-probe control).
        window_index, slot_a, slot_b, slot_same = _pick_leakage_pair(status)

        def _coord(slot: int) -> tuple[int, int]:
            return (
                int(orig_row[window_index, slot]),
                int(orig_col[window_index, slot]),
            )

        coord_a = _coord(slot_a)
        coord_b = _coord(slot_b)
        coord_same = _coord(slot_same)
        assert coord_a != coord_b
        logger.info(
            "Leakage probe: window %d, A=%s (status %d), B=%s (status %d), "
            "same-status control=%s.",
            window_index,
            coord_a,
            int(status[window_index, slot_a]),
            coord_b,
            int(status[window_index, slot_b]),
            coord_same,
        )

        block = _make_block(shift_size=SHIFT_SIZE)
        x_base = _probe_input()
        x_perturbed = x_base.copy()
        x_perturbed[0, coord_a[0], coord_a[1], :] += _perturbation()

        out_base = _forward(block, x_base)
        out_perturbed = _forward(block, x_perturbed)

        # Live-probe control: a token that IS allowed to attend to A (same wrap
        # status, same window) must move by a large, unmistakably non-roundoff
        # amount. Without this the leakage assertion could "fire" on 1e-7 of
        # float reassociation and prove nothing. This stays true after the fix.
        moved = float(
            np.abs(
                out_perturbed[0, coord_same[0], coord_same[1], :]
                - out_base[0, coord_same[0], coord_same[1], :]
            ).max()
        )
        assert moved > 1e-2, (
            f"Probe is dead: the same-window, same-region token {coord_same} "
            f"moved by only {moved:.3e} when {coord_a} was perturbed, so a "
            f"bit-identical region-B token would prove nothing."
        )

        delta_b = np.abs(
            out_perturbed[0, coord_b[0], coord_b[1], :]
            - out_base[0, coord_b[0], coord_b[1], :]
        )
        np.testing.assert_allclose(
            out_perturbed[0, coord_b[0], coord_b[1], :],
            out_base[0, coord_b[0], coord_b[1], :],
            rtol=0,
            atol=0,
            err_msg=(
                f"SW-MSA cross-region leakage: perturbing token {coord_a} "
                f"(wrap status {int(status[window_index, slot_a])}) changed "
                f"the output at same-window token {coord_b} (wrap status "
                f"{int(status[window_index, slot_b])}) by up to "
                f"{float(delta_b.max()):.6e}. These tokens are not spatially "
                f"adjacent before the cyclic roll and must not attend to each "
                f"other."
            ),
        )

    @pytest.mark.parametrize("policy", ["mixed_float16", "mixed_bfloat16"])
    def test_no_cross_region_leakage_under_mixed_precision(self, policy):
        """The SW-MSA mask must still isolate regions at reduced precision.

        The rest of this module runs at the session default (float32). That is
        NOT where this mask is most fragile: the additive bias is
        ``MASK_BIAS_VALUE``, which is ``-inf`` in float16, and the arithmetic
        form ``attn + (1 - keep) * -1e9`` that ``apply_attention_mask``
        replaced produced ``0 * -inf = NaN`` at every UNMASKED position under
        ``mixed_float16``. This repo has shipped that exact defect before, in
        this very attention primitive, so the property gets a standing guard
        instead of a one-off hand probe.

        Bit-identity is still the honest bar: a correctly masked key's softmax
        weight is exactly ``0.0`` in any precision, so its value contributes an
        exact zero. The finiteness assertion comes first because a NaN output
        would otherwise fail the equality check for entirely the wrong reason.
        """
        status, orig_row, orig_col = wrap_status_windows(
            HEIGHT, WIDTH, WINDOW_SIZE, SHIFT_SIZE
        )
        window_index, slot_a, slot_b, slot_same = _pick_leakage_pair(status)

        def _coord(slot: int) -> tuple[int, int]:
            return (
                int(orig_row[window_index, slot]),
                int(orig_col[window_index, slot]),
            )

        coord_a, coord_b, coord_same = (
            _coord(slot_a),
            _coord(slot_b),
            _coord(slot_same),
        )

        with global_dtype_policy(policy):
            block = _make_block(shift_size=SHIFT_SIZE)
            x_base = _probe_input()
            x_perturbed = x_base.copy()
            x_perturbed[0, coord_a[0], coord_a[1], :] += _perturbation()
            out_base = _forward(block, x_base)
            out_perturbed = _forward(block, x_perturbed)

        assert np.isfinite(out_base).all() and np.isfinite(out_perturbed).all(), (
            f"[{policy}] the masked shifted block produced non-finite output "
            f"({int((~np.isfinite(out_base)).sum())} / {out_base.size} in the "
            "base run) -- this is the fp16 mask-NaN failure mode"
        )

        moved = float(
            np.abs(
                out_perturbed[0, coord_same[0], coord_same[1], :]
                - out_base[0, coord_same[0], coord_same[1], :]
            ).max()
        )
        assert moved > 1e-2, (
            f"[{policy}] probe is dead: the same-region token {coord_same} "
            f"moved by only {moved:.3e}"
        )

        delta_b = float(
            np.abs(
                out_perturbed[0, coord_b[0], coord_b[1], :]
                - out_base[0, coord_b[0], coord_b[1], :]
            ).max()
        )
        assert delta_b == 0.0, (
            f"[{policy}] SW-MSA cross-region leakage: perturbing {coord_a} "
            f"moved same-window token {coord_b} by up to {delta_b:.6e} "
            "(must be exactly 0.0)"
        )


# ---------------------------------------------------------------------------
# 3. Controls -- green before and after the fix
# ---------------------------------------------------------------------------


class TestSwinShiftMaskControls:
    """Behaviour that must not move when the SW-MSA mask lands."""

    def test_unshifted_block_isolates_windows(self):
        """With ``shift_size=0``, a perturbation must not cross a window edge.

        This is the harness control: it proves the perturbation-isolation
        machinery can produce a bit-identical verdict at all, so a RED leakage
        probe is evidence about the mask and not about the probe.
        """
        block = _make_block(shift_size=0)
        x_base = _probe_input()

        coord_a = (0, 0)  # window (0, 0)
        coord_b = (HEIGHT - 1, WIDTH - 1)  # window (1, 1)
        coord_same = (1, 1)  # window (0, 0), same window as A
        x_perturbed = x_base.copy()
        x_perturbed[0, coord_a[0], coord_a[1], :] += _perturbation()

        out_base = _forward(block, x_base)
        out_perturbed = _forward(block, x_perturbed)

        # Live-probe control, same rationale as the leakage test.
        moved = float(
            np.abs(
                out_perturbed[0, coord_same[0], coord_same[1], :]
                - out_base[0, coord_same[0], coord_same[1], :]
            ).max()
        )
        assert moved > 1e-2, f"Probe is dead: in-window token moved {moved:.3e}"
        np.testing.assert_allclose(
            out_perturbed[0, coord_b[0], coord_b[1], :],
            out_base[0, coord_b[0], coord_b[1], :],
            rtol=0,
            atol=0,
            err_msg=(
                "Unshifted windowed attention leaked across a window "
                "boundary."
            ),
        )

    def test_unshifted_block_leaves_other_batch_element_untouched(self):
        """A perturbation in sample 0 must never reach sample 1."""
        block = _make_block(shift_size=SHIFT_SIZE)
        x_base = _probe_input()
        x_perturbed = x_base.copy()
        x_perturbed[0, 0, 0, :] += _perturbation()

        out_base = _forward(block, x_base)
        out_perturbed = _forward(block, x_perturbed)

        assert not np.array_equal(out_base[0], out_perturbed[0])
        np.testing.assert_allclose(
            out_perturbed[1], out_base[1], rtol=0, atol=0
        )

    def test_shifted_block_save_load_round_trip(self):
        """A shifted block survives a ``.keras`` round-trip by value."""
        keras.utils.set_random_seed(SEED)
        inputs = keras.Input(shape=(HEIGHT, WIDTH, DIM))
        outputs = SwinTransformerBlock(
            dim=DIM,
            num_heads=NUM_HEADS,
            window_size=WINDOW_SIZE,
            shift_size=SHIFT_SIZE,
            name="swin_shift_block",
        )(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)

        x = _probe_input()
        original = model.predict(x, verbose=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.keras")
            model.save(path)
            reloaded = keras.models.load_model(
                path,
                custom_objects={
                    "SwinMLP": SwinMLP,
                    "StochasticDepth": StochasticDepth,
                    "WindowAttention": WindowAttention,
                    "SwinTransformerBlock": SwinTransformerBlock,
                },
            )
            restored = reloaded.predict(x, verbose=0)

            assert isinstance(
                reloaded.get_layer("swin_shift_block"), SwinTransformerBlock
            )

        # Values, not just shapes: a shape-only check passes with zero weights.
        np.testing.assert_allclose(original, restored, rtol=1e-6, atol=1e-6)
        assert np.abs(original).max() > 0.0


# ---------------------------------------------------------------------------
# 4. Mask-construction contract (plan step 3)
# ---------------------------------------------------------------------------


class TestSwMsaMaskConstruction:
    """The shipped mask, its degenerate-geometry guard and its dynamic shapes."""

    def test_shipped_mask_equals_the_independent_oracle(self):
        """``_build_swmsa_keep_mask`` must reproduce the wrap-status relation.

        This compares the tensor the block actually feeds to its attention
        against the oracle derived at the top of this module, window by window
        and for both batch elements -- so a window-order (assumption A3) or
        batch-tiling mistake, both of which are otherwise silent, shows up here.
        """
        block = _make_block(shift_size=SHIFT_SIZE)
        mask = np.asarray(
            ops.convert_to_numpy(
                block._build_swmsa_keep_mask(
                    BATCH, HEIGHT, WIDTH, SHIFT_SIZE
                )
            )
        )

        status, _, _ = wrap_status_windows(
            HEIGHT, WIDTH, WINDOW_SIZE, SHIFT_SIZE
        )
        expected = _same_class(status).astype(mask.dtype)
        num_windows = expected.shape[0]

        assert mask.shape == (
            BATCH * num_windows,
            WINDOW_SIZE**2,
            WINDOW_SIZE**2,
        )
        # B-major / window-minor: sample b occupies rows [b*nw, (b+1)*nw).
        for sample in range(BATCH):
            np.testing.assert_array_equal(
                mask[sample * num_windows: (sample + 1) * num_windows],
                expected,
                err_msg=(
                    f"Shipped SW-MSA mask disagrees with the independent "
                    f"wrap-status oracle for batch element {sample}."
                ),
            )

        # The mask must actually forbid something, and never forbid a token
        # from attending to itself (which would hand the fully-masked-row
        # rescue live work to do).
        assert mask.min() == 0, "Mask forbids nothing -- it is dead."
        eye = np.eye(WINDOW_SIZE**2, dtype=mask.dtype)
        assert (mask * eye == eye).all(), "Mask forbids a self-attention pair."

    def test_single_window_geometry_matches_shift_size_zero(self):
        """A statically single-window feature map must behave as ``shift_size=0``.

        This is decision D-006 and the reference Swin rule
        (``if min(input_resolution) <= window_size: shift_size = 0``): at
        ``H == W == window_size`` there is exactly one window, so the roll and
        the mask are both dropped and the block degenerates to plain W-MSA.

        The assertion is the sharpest one available -- bit-identity against an
        independently constructed ``shift_size=0`` block carrying the *same*
        weights -- rather than a weaker "does not raise". A block that rolled
        but did not mask (the F-01 bug), or masked but did not roll, would
        differ here.
        """
        rng = np.random.default_rng(SEED)
        x = rng.standard_normal(
            (BATCH, WINDOW_SIZE, WINDOW_SIZE, DIM)
        ).astype("float32")

        def _single_window_block(shift_size: int) -> SwinTransformerBlock:
            keras.utils.set_random_seed(SEED)
            block = SwinTransformerBlock(
                dim=DIM,
                num_heads=NUM_HEADS,
                window_size=WINDOW_SIZE,
                shift_size=shift_size,
                dropout_rate=0.0,
                attention_dropout_rate=0.0,
                stochastic_depth_rate=0.0,
            )
            block.build((None, WINDOW_SIZE, WINDOW_SIZE, DIM))
            return block

        shifted = _single_window_block(SHIFT_SIZE)
        unshifted = _single_window_block(0)

        # Same weight layout is a precondition for the comparison to mean
        # anything; assert it rather than assuming it.
        assert [w.name for w in shifted.weights] == [
            w.name for w in unshifted.weights
        ]
        for target, source in zip(shifted.weights, unshifted.weights):
            target.assign(source)

        np.testing.assert_allclose(
            _forward(shifted, x), _forward(unshifted, x), rtol=0, atol=0,
            err_msg=(
                "A statically single-window shifted block must be bit-"
                "identical to the same block with shift_size=0 (D-006)."
            ),
        )

    def test_geometry_below_window_size_raises(self):
        """A statically-known ``H``/``W`` *strictly* below ``window_size`` must fail."""
        block = SwinTransformerBlock(
            dim=DIM,
            num_heads=NUM_HEADS,
            window_size=WINDOW_SIZE,
            shift_size=SHIFT_SIZE,
        )
        too_small = ops.convert_to_tensor(
            np.zeros((1, WINDOW_SIZE - 2, WINDOW_SIZE - 2, DIM), dtype="float32")
        )
        with pytest.raises(ValueError, match="smaller than window_size"):
            block(too_small)

    def test_dynamic_spatial_dims_never_raise(self):
        """A dynamic (``None``) spatial dim must never hit the raise (D-002 / A2).

        ``models/thera`` builds every Swin block with ``(B, None, None, C)``,
        so a static-shape check that fired on ``None`` would kill it. Feeding
        the single-window resolution through a dynamic graph exercises the one
        case where the static path downgrades the shift; since D-012 the
        dynamic path downgrades it too, at runtime. This test asserts only that
        nothing raises -- the *numeric* agreement of the two paths is
        :class:`TestShapeIndependenceAtTraceTime`'s job.
        """
        keras.utils.set_random_seed(SEED)
        inputs = keras.Input(shape=(None, None, DIM))
        outputs = SwinTransformerBlock(
            dim=DIM,
            num_heads=NUM_HEADS,
            window_size=WINDOW_SIZE,
            shift_size=SHIFT_SIZE,
            name="dynamic_single_window_swin",
        )(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)

        rng = np.random.default_rng(SEED)
        x = rng.standard_normal(
            (1, WINDOW_SIZE, WINDOW_SIZE, DIM)
        ).astype("float32")
        y = model.predict(x, verbose=0)
        assert y.shape == x.shape
        assert np.isfinite(y).all()

    def test_single_window_geometry_is_allowed_without_shift(self):
        """The raise is specific to ``shift_size > 0`` and must not fire otherwise."""
        block = SwinTransformerBlock(
            dim=DIM,
            num_heads=NUM_HEADS,
            window_size=WINDOW_SIZE,
            shift_size=0,
        )
        small = ops.convert_to_tensor(
            np.zeros((1, WINDOW_SIZE, WINDOW_SIZE, DIM), dtype="float32")
        )
        assert tuple(block(small).shape) == (1, WINDOW_SIZE, WINDOW_SIZE, DIM)

    def test_dynamic_spatial_dims_are_supported(self):
        """A block built with ``(B, None, None, C)`` must run at two resolutions.

        This is the ``models/thera`` contract (plan assumption A2): the ``pro``
        tail builds every Swin block with dynamic spatial dims and reflect-pads
        H/W to a window-size multiple with symbolic amounts at call time. The
        guard above must therefore stay silent on a ``None`` dim, and the mask
        must be derived from the runtime shape.
        """
        keras.utils.set_random_seed(SEED)
        inputs = keras.Input(shape=(None, None, DIM))
        outputs = SwinTransformerBlock(
            dim=DIM,
            num_heads=NUM_HEADS,
            window_size=WINDOW_SIZE,
            shift_size=SHIFT_SIZE,
            name="dynamic_swin",
        )(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)

        rng = np.random.default_rng(SEED)
        for size in (HEIGHT, 3 * WINDOW_SIZE):
            x = rng.standard_normal((1, size, size, DIM)).astype("float32")
            y = model.predict(x, verbose=0)
            assert y.shape == x.shape
            assert np.isfinite(y).all()

    @pytest.mark.parametrize(
        "height,width",
        [
            (HEIGHT, WIDTH),                    # 2 x window_size
            (WINDOW_SIZE, WINDOW_SIZE),         # exactly one window
            (HEIGHT, WINDOW_SIZE),              # one window wide only
        ],
        ids=["two_windows", "single_window", "mixed"],
    )
    def test_dynamic_build_matches_static_build(self, height, width):
        """Dynamic-shape *build* execution must produce the same numbers as static.

        A mask derived from a *wrong* runtime shape would still be finite and
        correctly shaped; only a comparison against the statically-shaped
        reference can see it.

        **This test cannot see the D-012 defect and is not meant to.** Both
        blocks are called *eagerly* here, so both see concrete integers in
        ``x.shape`` regardless of what shape they were BUILT with, and both
        therefore take the same ``_resolve_shift_size`` branch. Build-time
        shape independence and call-time shape independence are different
        properties; the second one is
        :class:`TestShapeIndependenceAtTraceTime`'s job.
        """
        rng = np.random.default_rng(SEED)
        x = rng.standard_normal((BATCH, height, width, DIM)).astype("float32")

        keras.utils.set_random_seed(SEED)
        static_block = SwinTransformerBlock(
            dim=DIM,
            num_heads=NUM_HEADS,
            window_size=WINDOW_SIZE,
            shift_size=SHIFT_SIZE,
        )
        static_block.build((None, height, width, DIM))

        keras.utils.set_random_seed(SEED)
        dynamic_block = SwinTransformerBlock(
            dim=DIM,
            num_heads=NUM_HEADS,
            window_size=WINDOW_SIZE,
            shift_size=SHIFT_SIZE,
        )
        dynamic_block.build((None, None, None, DIM))

        for target, source in zip(
            dynamic_block.weights, static_block.weights
        ):
            target.assign(source)

        np.testing.assert_allclose(
            _forward(dynamic_block, x), _forward(static_block, x),
            rtol=0, atol=0,
        )


# ---------------------------------------------------------------------------
# 5. Shape independence at TRACE time (plan step 10, decision D-012)
# ---------------------------------------------------------------------------


def _traced_forward(
    block: SwinTransformerBlock, x: np.ndarray, spec_shape
) -> np.ndarray:
    """Run ``block`` inside a graph traced against ``spec_shape``.

    Both comparands in this section go through ``tf.function``, so the only
    difference between them is what the tracer *knew about the shape* --
    graph-vs-eager float reassociation is held constant and cannot be mistaken
    for a branch divergence. (That reassociation is real and is not small: at
    these weight scales an eager call and a traced call of the *same* block on
    the *same* tensor differ by ~1e-3 at ``H = 2 * window_size``, where no
    branch is in dispute at all.)

    Args:
        block: A built ``SwinTransformerBlock``.
        x: Concrete input array.
        spec_shape: The shape given to ``tf.TensorSpec``; ``None`` entries are
            the dims the trace must treat as unknown.

    Returns:
        The traced forward pass as a numpy array.
    """
    fn = tf.function(
        lambda t: block(t, training=False),
        input_signature=[tf.TensorSpec(spec_shape, tf.float32)],
    )
    return np.asarray(fn(tf.convert_to_tensor(x)))


class TestShapeIndependenceAtTraceTime:
    """The same block, weights and tensor must give one answer (D-012).

    Before plan step 10 the D-006 single-window fallback was a *static-shape*
    decision: ``_resolve_shift_size`` read ``x.shape[1:3]`` and skipped any
    ``None``. Traced against ``TensorSpec([None, None, None, C])`` at
    ``H == W == window_size`` the fallback therefore could not fire, and the
    block kept the roll and the region mask while an eager call on the very
    same tensor dropped both.

    Measured RED at the pre-fix commit, config ``dim=32, num_heads=4,
    window_size=4, shift_size=2``, ``(1, 4, 4, 32)``, weights assigned from a
    seeded RNG: **512/512 elements differed, max |diff| 251.4, 97% relative.**
    ``models/thera/tails.py`` builds every Swin block with
    ``(B, None, None, C)``, so it was the one consumer taking the branch that
    ``models/swin_transformer`` never takes at identical geometry.
    """

    @pytest.mark.parametrize(
        "height,width",
        [
            (WINDOW_SIZE, WINDOW_SIZE),
            (HEIGHT, WIDTH),
            (HEIGHT, WINDOW_SIZE),
            (WINDOW_SIZE, WIDTH),
            (3 * WINDOW_SIZE, 3 * WINDOW_SIZE),
        ],
        ids=["1x1", "2x2", "2x1", "1x2", "3x3"],
    )
    def test_unknown_spatial_dims_give_the_static_answer(self, height, width):
        """A trace that does not know ``H``/``W`` must answer like one that does.

        ``rtol=0, atol=0``: the two graphs execute the identical op sequence
        once the shift agrees, so anything other than bit-identity means the
        runtime fallback did not reproduce the static one.
        """
        block = _make_block(shift_size=SHIFT_SIZE, build_shape=(None, None, None, DIM))
        rng = np.random.default_rng(SEED)
        x = rng.standard_normal((BATCH, height, width, DIM)).astype("float32")

        dynamic = _traced_forward(block, x, [None, None, None, DIM])
        static = _traced_forward(block, x, [BATCH, height, width, DIM])

        np.testing.assert_allclose(
            dynamic, static, rtol=0, atol=0,
            err_msg=(
                f"Shape-dependent output at H={height}, W={width}: the same "
                "block, weights and tensor answered differently depending on "
                "whether the tracer knew the spatial extent (D-012)."
            ),
        )

    def test_the_probe_can_see_a_wrong_runtime_shift(self):
        """Non-vacuity: the comparison above must be able to fail.

        A block whose *static* answer is a shifted one (``H = 2 * ws``) and a
        block whose static answer is unshifted (``H == ws``) must produce
        visibly different numbers on the same tensor -- otherwise the geometry
        this suite compares is not observable and the bit-identity assertion
        above would hold for a block that ignored the shift entirely.
        """
        block = _make_block(shift_size=SHIFT_SIZE, build_shape=(None, None, None, DIM))
        rng = np.random.default_rng(SEED)
        x = rng.standard_normal(
            (BATCH, WINDOW_SIZE, WINDOW_SIZE, DIM)
        ).astype("float32")

        unshifted = _traced_forward(block, x, [BATCH, WINDOW_SIZE, WINDOW_SIZE, DIM])

        forced = _make_block(shift_size=0, build_shape=(None, None, None, DIM))
        for target, source in zip(forced.weights, block.weights):
            target.assign(source)
        control = _traced_forward(
            forced, x, [BATCH, WINDOW_SIZE, WINDOW_SIZE, DIM]
        )

        # The single-window geometry must reduce to the shift_size=0 block...
        np.testing.assert_allclose(unshifted, control, rtol=0, atol=0)

        # ...while a genuinely shifted geometry must NOT be the same numbers,
        # so "bit-identical" above is a real constraint and not a tautology.
        bigger = rng.standard_normal((BATCH, HEIGHT, WIDTH, DIM)).astype("float32")
        shifted = _traced_forward(block, bigger, [None, None, None, DIM])
        plain = _traced_forward(forced, bigger, [None, None, None, DIM])
        assert float(np.abs(shifted - plain).max()) > 1e-2

    def test_the_runtime_shift_survives_xla(self):
        """``ops.roll`` with a *tensor* shift must still compile under XLA.

        The dynamic path feeds ``ops.roll`` a runtime scalar rather than a
        Python int. That is the one construct in this fix that a backend could
        refuse, and refusing it would be a hard outage for any consumer using
        ``jit_compile=True`` -- so it is pinned rather than assumed.
        """
        block = _make_block(shift_size=SHIFT_SIZE, build_shape=(None, None, None, DIM))
        rng = np.random.default_rng(SEED)
        fn = tf.function(
            lambda t: block(t, training=False),
            input_signature=[tf.TensorSpec([None, None, None, DIM], tf.float32)],
            jit_compile=True,
        )
        for size in (WINDOW_SIZE, HEIGHT, 3 * WINDOW_SIZE):
            x = rng.standard_normal((1, size, size, DIM)).astype("float32")
            y = np.asarray(fn(tf.convert_to_tensor(x)))
            assert y.shape == x.shape
            assert np.isfinite(y).all()
