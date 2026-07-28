"""Tests for the Sparsemax activation layer."""

import os
import tempfile

import numpy as np
import keras
from keras import ops
import pytest

from dl_techniques.layers.activations.sparsemax import Sparsemax
from dl_techniques.layers.attention.multi_head_cross_attention import (
    MultiHeadCrossAttention,
)


def _x() -> np.ndarray:
    rng = np.random.default_rng(3)
    return rng.standard_normal((4, 8)).astype("float32")


# ---------------------------------------------------------------------
# float64 reference oracle for `-inf`-masked sparsemax.
#
# Test-local helper on purpose: it is an independent re-implementation of the
# Martins & Astudillo projection used to CHECK the layer, so it must NOT share
# code with `src/`. It is not an abstraction the library gains.
# ---------------------------------------------------------------------


def _sparsemax_reference(z: np.ndarray) -> np.ndarray:
    """Sparsemax projection over the FINITE entries of each row, in float64.

    Masked (non-finite) positions are exactly ``0.0`` by construction; finite
    positions receive the standard sparsemax projection computed over ONLY the
    finite sub-vector.

    :param z: 2-D array of logits, possibly containing ``-inf`` / ``+inf``.
    :type z: np.ndarray

    :return: float64 array of the same shape as ``z``.
    :rtype: np.ndarray
    """
    z = np.asarray(z, dtype=np.float64)
    finite = np.isfinite(z)
    out = np.zeros_like(z)
    for i, row in enumerate(z):
        idx = np.where(finite[i])[0]
        assert idx.size > 0, f"row {i} is fully masked; that case is out of scope"
        vals = row[idx]
        sorted_vals = np.sort(vals)[::-1]
        k = np.arange(1, sorted_vals.size + 1)
        cssv = np.cumsum(sorted_vals)
        cond = 1.0 + k * sorted_vals > cssv
        k_z = int(np.count_nonzero(cond))
        tau = (cssv[k_z - 1] - 1.0) / k_z
        out[i, idx] = np.maximum(vals - tau, 0.0)
    return out


def _partially_masked_row_batch(
    width: int,
    masked_fraction: float,
    seed: int,
    rows: int = 4,
) -> np.ndarray:
    """Build a seeded ``(rows, width)`` logit batch with `-inf` at a random subset.

    Includes deliberate exact-tie patterns, and NEVER masks an entire row
    (fully-masked rows are out of scope: `apply_attention_mask`'s rescue_axis
    handles them upstream and they never reach Sparsemax).

    :param width: Row width.
    :type width: int
    :param masked_fraction: Fraction of each row to set to ``-inf`` (``0.0``
        yields the no-``inf`` control).
    :type masked_fraction: float
    :param seed: RNG seed.
    :type seed: int
    :param rows: Number of rows in the batch.
    :type rows: int

    :return: float32 array of shape ``(rows, width)``.
    :rtype: np.ndarray
    """
    rng = np.random.default_rng(seed)
    scale = float(rng.choice([0.5, 1.0, 2.0]))
    z = (rng.uniform(-5.0, 5.0, size=(rows, width)) * scale).astype(np.float32)

    # Deliberate exact ties: duplicate one value into a couple of positions.
    if width >= 3:
        for r in range(rows):
            src = int(rng.integers(width))
            dst = rng.choice(width, size=min(2, width - 1), replace=False)
            z[r, dst] = z[r, src]

    n_masked = int(round(masked_fraction * width))
    # Precondition: never mask an entire row.
    n_masked = min(n_masked, width - 1)
    assert n_masked < width, (
        f"generator would mask an entire row (width={width}, "
        f"masked_fraction={masked_fraction})"
    )
    if n_masked > 0:
        for r in range(rows):
            pos = rng.choice(width, size=n_masked, replace=False)
            z[r, pos] = -np.inf

    assert np.isfinite(z).any(axis=-1).all(), "a generated row is fully masked"
    return z


class TestSparsemax:

    def test_construction(self) -> None:
        assert Sparsemax(axis=-1).axis == -1

    def test_invalid_axis(self) -> None:
        with pytest.raises(ValueError):
            Sparsemax(axis=1.5)

    def test_forward_sums_to_one(self) -> None:
        y = ops.convert_to_numpy(Sparsemax()(_x()))
        assert y.shape == (4, 8)
        np.testing.assert_allclose(y.sum(axis=-1), np.ones(4), atol=1e-5)
        assert np.all(y >= -1e-6)

    def test_compute_output_shape(self) -> None:
        layer = Sparsemax()
        x = _x()
        assert tuple(layer.compute_output_shape(x.shape)) == tuple(layer(x).shape)

    def test_serialization_round_trip(self) -> None:
        inp = keras.Input(shape=(8,))
        out = Sparsemax()(inp)
        model = keras.Model(inp, out)
        x = _x()
        y0 = model(x)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sparsemax.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
        y1 = loaded(x)
        np.testing.assert_allclose(
            ops.convert_to_numpy(y0), ops.convert_to_numpy(y1), atol=1e-6
        )

    # -----------------------------------------------------------------
    # `-inf` (attention-mask) coverage.
    #
    # Tolerance is FIXED at atol=rtol=1e-3: that is the documented TF32
    # precision floor (see `test_linear_attention.py:37`, which disables TF32
    # process-globally at import). Tightening it would make these tests
    # collection-order dependent.
    # -----------------------------------------------------------------

    @pytest.mark.parametrize("compute_dtype", ["float32", "mixed_float16"])
    @pytest.mark.parametrize("n", [4, 512, 4096])
    def test_partial_mask_neg_inf_no_nan(self, n: int, compute_dtype: str) -> None:
        """A partially `-inf`-masked row must stay finite and match the oracle."""
        previous = keras.mixed_precision.global_policy().name
        keras.mixed_precision.set_global_policy(compute_dtype)
        try:
            z = np.full((2, n), -np.inf, dtype=np.float32)
            # Three surviving keys (not one): a single survivor is degenerate.
            z[:, 0] = 2.0
            z[:, 1] = 1.0
            z[:, 2] = 0.5

            out = ops.convert_to_numpy(Sparsemax()(z))

            # (a) THE primary, intended-RED assertion. Must be first.
            nan_count = int(np.isnan(out).sum())
            assert not np.isnan(out).any(), (
                "Sparsemax produced NaN for a partially-masked -inf row "
                f"(n={n}, dtype={compute_dtype}); nan_count={nan_count}/{out.size}"
            )

            # (b) Secondary correctness check against the float64 oracle.
            ref = _sparsemax_reference(z)
            np.testing.assert_allclose(
                out.astype(np.float64), ref, atol=1e-3, rtol=1e-3
            )
        finally:
            keras.mixed_precision.set_global_policy(previous)

    def test_sparsemax_property_random_masks(self, dtype_policy: str) -> None:
        """Randomized mask/width/tie grid, in every supported dtype policy.

        Uses the shared `dtype_policy` fixture from `tests/test_layers/conftest.py`
        (float32 / mixed_float16 / float64), which restores the process-global
        policy in its teardown.

        The ``masked_fraction == 0.0`` grid points carry NO ``inf`` at all and are
        the deliberate no-`inf` control inside this property test.

        Assertion (a) is checked over the WHOLE grid before any (b) is checked, so
        the NaN assertion is what fires first regardless of grid order.
        """
        widths = (3, 17, 128, 512)
        fractions = (0.0, 0.25, 0.5, 0.9)
        seeds = tuple(range(8))

        collected = []
        for width in widths:
            for frac in fractions:
                for seed in seeds:
                    z = _partially_masked_row_batch(width, frac, seed)
                    out = ops.convert_to_numpy(Sparsemax()(z))
                    label = (
                        f"policy={dtype_policy} width={width} "
                        f"masked_fraction={frac} seed={seed}"
                    )

                    # (a) primary, intended-RED assertion — first, for every point.
                    nan_count = int(np.isnan(out).sum())
                    assert not np.isnan(out).any(), (
                        f"Sparsemax produced NaN ({label}); "
                        f"nan_count={nan_count}/{out.size}"
                    )
                    collected.append((label, out.astype(np.float64), z))

        # (b) secondary correctness check, only reachable once (a) held everywhere.
        for label, out, z in collected:
            np.testing.assert_allclose(
                out,
                _sparsemax_reference(z),
                atol=1e-3,
                rtol=1e-3,
                err_msg=f"sparsemax != float64 oracle ({label})",
            )

    def test_attention_integration_sparsemax_fp16_no_nan(self) -> None:
        """End-to-end repro: sparsemax attention under fp16 with a partial mask."""
        previous = keras.mixed_precision.global_policy().name
        keras.mixed_precision.set_global_policy("mixed_float16")
        try:
            rng = np.random.default_rng(0)
            x = rng.standard_normal((2, 4, 16)).astype("float32")
            mask = np.ones((2, 4, 4), dtype="float32")
            mask[:, :, 2:] = 0.0

            layer = MultiHeadCrossAttention(
                dim=16, num_heads=2, probability_type="sparsemax"
            )
            out = ops.convert_to_numpy(layer(x, attention_mask=mask))

            nan_count = int(np.isnan(out).sum())
            assert not np.isnan(out).any(), (
                "MultiHeadCrossAttention(probability_type='sparsemax') produced NaN "
                f"under mixed_float16 with a partial mask; "
                f"nan_count={nan_count}/{out.size}"
            )
        finally:
            keras.mixed_precision.set_global_policy(previous)
