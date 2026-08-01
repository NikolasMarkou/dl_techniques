"""Tests for ContinuousRoPE (continuous multi-dimensional rotary phase angles)."""

import os
import hashlib
import numpy as np
import pytest
import keras
import tensorflow as tf

from dl_techniques.layers.embedding.continuous_rope_embedding import ContinuousRoPE

# ---------------------------------------------------------------------
# dtype-policy corpus + tolerances (D-007 — the THIRD site of the G-10 shape)
# ---------------------------------------------------------------------

# The four policies this layer must survive. `tests/test_layers/conftest.py::dtype_policy`
# is the house fixture (restore-safe global-policy set/teardown in ONE place); its own
# params cover only three of them, so the fourth is supplied by INDIRECT parametrization
# rather than by a second copy of the set/restore dance in this module. `mixed_bfloat16`
# is not optional here: it is one of the three policies measured DEAD at HEAD.
_D7_POLICIES = ("float32", "mixed_float16", "float64", "mixed_bfloat16")

_D7_SEED = 20260801

# COORD REGIME — a tolerance is meaningless without the corpus it was measured on.
# Coordinates are drawn uniformly from [0, 64). NOT near-zero: this layer's output is
# the raw phase `coord * omega`, so at coords in [0, 1e-3) every phase is <= 1e-3 and a
# `call()` returning zeros would be nearly invisible to a value assertion. At [0, 64)
# the largest reference phase is 61.90, so the zeros injection is RED by ~250x even
# against the loosest (bfloat16) tolerance below.
_D7_COORD_HI = 64.0

# Max abs error vs a float64 numpy reference, MEASURED at the [0, 64) regime above on
# CPU with dim=64, ndim=3: float32 9.26e-07, float64 5.18e-07, mixed_float16 1.42e-02,
# mixed_bfloat16 1.23e-01. Each entry carries ~2x headroom over its measurement.
# A SINGLE policy-wide tolerance is rejected on purpose: the span is 5 orders of
# magnitude, so a bound loose enough for bfloat16 is vacuous at float32.
# The fp16/bf16 rows are NOT a claim that this layer "works" at those policies — the
# error is dominated by the fp16/bf16 REPRESENTATION of the phase itself (resolution
# 0.03 / 0.25 at magnitude 62), which no in-layer widening can recover.
_D7_TOL = {
    "float32": 2.0e-06,
    "float64": 1.0e-06,
    "mixed_float16": 3.0e-02,
    "mixed_bfloat16": 2.5e-01,
}

# I-B (this layer's copy): float32 output must stay BIT-IDENTICAL to HEAD. Captured on
# CPU at HEAD 2597a65b (policy float32) as the sha256 of the C-contiguous float32 output
# bytes -- byte equality of a float32 buffer is exactly uint32-view equality of the same
# buffer. Three corpora, including the `self.padding > 0` branch.
# ANY movement here is a STOP, never a tolerance to widen.
_D7_BITID_CORPORA = {
    "unit_0_1": dict(
        dim=64, ndim=3, shape=(2, 5, 3), scale=1.0,
        sha256="5ae521686bd0de970b75e429f58da14167c835dcf617f3de6f77e069e2667a5e",
        first4_uint32=[1054468030, 1043169233, 1032463490, 1021049716],
    ),
    "large_pos_0_2000": dict(
        dim=64, ndim=3, shape=(2, 5, 3), scale=2000.0,
        sha256="f7556d94a88ed676d51c481abbc1bd58b17a7415200246e7c8e39c7f7c0caf53",
        first4_uint32=[1146408015, 1135177425, 1124525991, 1112986510],
    ),
    "padded_dim66_ndim4": dict(
        dim=66, ndim=4, shape=(2, 5, 4), scale=64.0,
        sha256="785e451d8f2b2797e611f46da858a5684189744878905362a023496f67f603cb",
        first4_uint32=[1104799678, 1091162300, 1076777727, 1063026991],
    ),
}


def _d7_coords(shape, scale):
    """Deterministic float32 coordinates in ``[0, scale)``."""
    rng = np.random.default_rng(_D7_SEED)
    return (rng.random(shape) * scale).astype("float32")


def _d7_reference_f64(coords_f64, dim, ndim, max_wavelength=10000.0):
    """Independent float64 numpy oracle for the continuous-RoPE phase angles.

    Reimplements the published formula (``phi_k = p_k * omega``, concatenated across
    coordinate axes, zero-padded to the phase width) rather than calling the layer, so
    a reversed frequency progression or a zeroed output is visible.
    """
    ndim_padding = dim % ndim
    dim_per_ndim = (dim - ndim_padding) // ndim
    padding = ndim_padding + (dim_per_ndim % 2) * ndim
    eff = (dim - padding) // ndim
    omega = 1.0 / (max_wavelength ** (np.arange(0, eff, 2, dtype=np.float64) / eff))
    phases = coords_f64[..., None] * omega
    phases = phases.reshape(*coords_f64.shape[:-1], -1)
    if padding > 0:
        phases = np.concatenate(
            [phases, np.zeros((*phases.shape[:-1], padding // 2))], axis=-1)
    return phases


class TestContinuousRoPE:

    # ---- constructor validation -------------------------------------

    def test_ctor_rejects_bad_args(self):
        with pytest.raises(ValueError):
            ContinuousRoPE(dim=0, ndim=2)
        with pytest.raises(ValueError):
            ContinuousRoPE(dim=64, ndim=0)
        with pytest.raises(ValueError):
            ContinuousRoPE(dim=64, ndim=2, max_wavelength=0.0)
        with pytest.raises(ValueError):
            # dim too small for ndim
            ContinuousRoPE(dim=2, ndim=4)

    # ---- forward / shape --------------------------------------------

    @pytest.mark.parametrize("shape,ndim", [((2, 5, 3), 3), ((5, 3), 3), ((2, 7, 2), 2)])
    def test_forward_shape(self, shape, ndim):
        layer = ContinuousRoPE(dim=64, ndim=ndim)
        x = keras.ops.convert_to_tensor(np.random.rand(*shape).astype("float32"))
        out = layer(x)
        # phase width is dim/2 for divisible dim
        assert tuple(out.shape) == tuple(shape[:-1]) + (32,)

    def test_compute_output_shape_matches_actual(self):
        # This is the regression for the prior dim-vs-dim/2 bug.
        for dim, ndim, in_shape in [(64, 3, (2, 5, 3)), (66, 4, (2, 5, 4)), (60, 3, (2, 5, 3))]:
            layer = ContinuousRoPE(dim=dim, ndim=ndim)
            x = keras.ops.convert_to_tensor(np.random.rand(*in_shape).astype("float32"))
            actual = int(layer(x).shape[-1])
            declared = layer.compute_output_shape(in_shape)[-1]
            assert actual == declared, f"dim={dim} ndim={ndim}: actual {actual} != declared {declared}"

    # ---- graph safety (locks the removed eager convert_to_numpy) -----

    def test_graph_trace_no_eager(self):
        layer = ContinuousRoPE(dim=64, ndim=3)  # assert_positive=True default
        x = tf.constant(np.random.rand(2, 5, 3).astype("float32"))
        eager = keras.ops.convert_to_numpy(layer(x))
        f = tf.function(lambda t: layer(t),
                        input_signature=[tf.TensorSpec([None, None, 3], tf.float32)])
        graph = f(x).numpy()
        np.testing.assert_allclose(eager, graph, atol=1e-6)

    def test_graph_trace_with_padding(self):
        layer = ContinuousRoPE(dim=66, ndim=4)
        x = tf.constant(np.random.rand(2, 5, 4).astype("float32"))
        f = tf.function(lambda t: layer(t),
                        input_signature=[tf.TensorSpec([None, None, 4], tf.float32)])
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(layer(x)), f(x).numpy(), atol=1e-6)

    # ---- build idempotency ------------------------------------------

    def test_double_build_stable(self):
        layer = ContinuousRoPE(dim=64, ndim=3)
        layer.build((2, 5, 3))
        n1 = len(layer.weights)
        layer.build((2, 5, 3))
        assert len(layer.weights) == n1

    # ---- serialization ----------------------------------------------

    def test_get_config_round_trip(self):
        layer = ContinuousRoPE(dim=48, ndim=3, max_wavelength=5000.0, assert_positive=False)
        rebuilt = ContinuousRoPE.from_config(layer.get_config())
        assert rebuilt.dim == 48 and rebuilt.ndim == 3
        assert rebuilt.max_wavelength == 5000.0 and rebuilt.assert_positive is False

    # ---- dtype policies (D-007) --------------------------------------

    @pytest.mark.parametrize("dtype_policy", _D7_POLICIES, indirect=True)
    def test_forward_matches_float64_reference_under_all_policies(self, dtype_policy):
        """The layer must RUN and be ACCURATE at every dtype policy, coords in [0, 64).

        At HEAD this raises `InvalidArgumentError [Op:Mul]` at three of the four
        policies: `call()` forces `coords` to a literal "float32" while Keras
        autocasts the `omega` weight to the compute dtype, so the two operands of
        the phase multiply disagree. This is the same defect as G-10 in the sibling
        `ContinuousSinCosEmbed`, in the same package (decisions.md D-006/D-007).
        """
        with tf.device("/CPU:0"):
            layer = ContinuousRoPE(dim=64, ndim=3)
            coords = _d7_coords((2, 5, 3), _D7_COORD_HI)
            # Cast through keras.ops: numpy has no bfloat16.
            x = keras.ops.cast(
                keras.ops.convert_to_tensor(coords), layer.compute_dtype)
            out = layer(x)

        # NOT a literal "float32": that assertion PASSES under mixed_float16 for a
        # fix that forgets the final cast back, i.e. it would pin the bug.
        assert keras.backend.standardize_dtype(out.dtype) == layer.compute_dtype, (
            f"policy {dtype_policy}: expected compute_dtype {layer.compute_dtype}, "
            f"got {keras.backend.standardize_dtype(out.dtype)}"
        )

        reference = _d7_reference_f64(coords.astype(np.float64), dim=64, ndim=3)
        actual = np.asarray(
            keras.ops.convert_to_numpy(tf.cast(out, tf.float64)), dtype=np.float64)
        # rtol=0 is mandatory: assert_allclose's default rtol=1e-7 otherwise
        # silently contributes to a nominally-atol bound.
        np.testing.assert_allclose(
            actual, reference, atol=_D7_TOL[dtype_policy], rtol=0,
            err_msg=(f"policy {dtype_policy}, coords in [0, {_D7_COORD_HI})"),
        )

    @pytest.mark.parametrize("corpus", sorted(_D7_BITID_CORPORA))
    @pytest.mark.parametrize("dtype_policy", ["float32"], indirect=True)
    def test_float32_output_is_bit_identical_to_head(self, dtype_policy, corpus):
        """The float32 bytes must not move. Device-pinned to CPU.

        Any movement is a STOP for the D-007 change, not a tolerance to widen.
        """
        spec = _D7_BITID_CORPORA[corpus]
        with tf.device("/CPU:0"):
            layer = ContinuousRoPE(dim=spec["dim"], ndim=spec["ndim"])
            x = keras.ops.convert_to_tensor(_d7_coords(spec["shape"], spec["scale"]))
            out = np.ascontiguousarray(keras.ops.convert_to_numpy(layer(x)))

        assert out.dtype == np.float32
        assert out.reshape(-1)[:4].view(np.uint32).tolist() == spec["first4_uint32"], (
            f"{corpus}: leading float32 words moved from the HEAD reference"
        )
        assert hashlib.sha256(out.tobytes()).hexdigest() == spec["sha256"], (
            f"{corpus}: float32 output is NOT bit-identical to HEAD"
        )

    @pytest.mark.parametrize("dtype_policy", _D7_POLICIES, indirect=True)
    def test_graph_trace_matches_eager_under_all_policies(self, dtype_policy):
        """Graph safety must survive the widening at every policy.

        Guards the `plan_2026-06-15_9dbb87c1/D-001` constraint that this `call()`
        carries verbatim: no eager host materialization may creep back in.
        """
        with tf.device("/CPU:0"):
            layer = ContinuousRoPE(dim=64, ndim=3)
            tf_dtype = tf.as_dtype(layer.compute_dtype)
            x = tf.cast(tf.constant(_d7_coords((2, 5, 3), _D7_COORD_HI)), tf_dtype)
            eager = layer(x)
            traced = tf.function(
                lambda t: layer(t),
                input_signature=[tf.TensorSpec([None, None, 3], tf_dtype)],
            )(x)
            # bfloat16 has no plain-numpy view, so compare the VALUES at float32.
            eager_np = keras.ops.convert_to_numpy(tf.cast(eager, tf.float32))
            traced_np = keras.ops.convert_to_numpy(tf.cast(traced, tf.float32))
        assert keras.backend.standardize_dtype(eager.dtype) == layer.compute_dtype
        assert keras.backend.standardize_dtype(traced.dtype) == layer.compute_dtype
        np.testing.assert_array_equal(eager_np, traced_np)

    # ---- serialization (continued) -----------------------------------

    def test_keras_round_trip(self, tmp_path):
        inp = keras.Input(shape=(5, 3), dtype="float32")
        out = ContinuousRoPE(dim=64, ndim=3)(inp)
        model = keras.Model(inp, out)
        x = np.random.rand(2, 5, 3).astype("float32")
        before = keras.ops.convert_to_numpy(model(x))
        path = os.path.join(tmp_path, "crope.keras")
        model.save(path)
        after = keras.ops.convert_to_numpy(keras.models.load_model(path)(x))
        np.testing.assert_allclose(before, after, atol=1e-6)
