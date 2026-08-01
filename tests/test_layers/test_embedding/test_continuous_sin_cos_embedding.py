"""Tests for ContinuousSinCosEmbed (fixed sin/cos embedding of continuous coords)."""

import os
import hashlib
import numpy as np
import pytest
import keras
import tensorflow as tf

from dl_techniques.layers.embedding.continuous_sin_cos_embedding import ContinuousSinCosEmbed

# ---------------------------------------------------------------------
# dtype-policy corpus + tolerances (G-10)
# ---------------------------------------------------------------------

# The four policies this layer must survive. `tests/test_layers/conftest.py::dtype_policy`
# is the house fixture (restore-safe global-policy set/teardown in ONE place); its own
# params cover only three of them, so the fourth is supplied by INDIRECT parametrization
# rather than by a second copy of the set/restore dance in this module. `mixed_bfloat16`
# is not optional here: it is one of the three policies measured DEAD at HEAD.
_G10_POLICIES = ("float32", "mixed_float16", "float64", "mixed_bfloat16")

_G10_SEED = 20260801

# COORD REGIME — a tolerance is meaningless without the corpus it was measured on.
# Coordinates are drawn uniformly from [0, 64). This regime is chosen deliberately:
#   * NOT near-zero. At coords in [0, 1e-3) the sin half maxes out at 0.0010, so a
#     `call()` that returned zeros would be nearly invisible to a value assertion.
#     At [0, 64) the sin half reaches 0.9999 and the full reference 0.9999991, so the
#     zeros injection is RED by ~7x even against the loosest (bfloat16) tolerance here.
#   * NOT large positions. At positions up to 2000 the fp16/bf16 error reaches 0.17/0.63,
#     which is an autocast-boundary limitation of the CALLER's coordinate dtype, not
#     something this layer can fix. It is documented separately, not smuggled into a
#     tolerance here.
_G10_COORD_HI = 64.0

# Max abs error vs a float64 numpy reference, MEASURED at the [0, 64) regime above on
# CPU with dim=64, ndim=3: float32 8.34e-07, float64 4.36e-07, mixed_float16 1.28e-02,
# mixed_bfloat16 9.81e-02. Each entry carries ~1.5-2.5x headroom over its measurement.
# The fp16/bf16 rows are a property OF THIS COORD REGIME and are emphatically NOT a
# claim that this layer "works" at those policies -- at coords in [0, 2000) the same
# configuration is wrong by 0.47 / 1.99 on a [-1, 1]-ranged output. That limitation is
# pinned executably by `test_fp16_large_position_error_bound_is_pinned` below and
# documented in the layer's own class docstring.
# A SINGLE policy-wide tolerance is rejected on purpose: the span is 5 orders of
# magnitude, so a bound loose enough for bfloat16 is vacuous at float32.
# (float64 does not reach ~1e-16 because `omega` is still computed at float32 —
# a separate, known precision ceiling.)
_G10_TOL = {
    "float32": 2.0e-06,
    "float64": 1.0e-06,
    "mixed_float16": 3.0e-02,
    "mixed_bfloat16": 1.5e-01,
}

# H-10: the LARGE-POSITION (coords in [0, 2000)) error band, MEASURED against the
# shipped code on CPU at dim=64, ndim=3: mixed_float16 0.4745, mixed_bfloat16 1.9859
# (8-seed sweep 0.446-0.483 and 1.964-1.997). The bands below sit either side of those
# measurements. NOTE these are ~2.8x and ~3.1x WORSE than the figures an EXPLORE
# prototype produced (0.17 / 0.63): the prototype did not narrow `coords` at the
# autocast boundary, so it measured only the `omega` narrowing. A plan's predicted
# figure is not a measurement of the shipped path.
_G10_LARGE_POS_BOUND = {
    "mixed_float16": (0.30, 0.60),
    "mixed_bfloat16": (1.50, 2.00),
}

# I-B: float32 output must stay BIT-IDENTICAL to HEAD. Captured on CPU at HEAD 5b1a966e
# (policy float32) as the sha256 of the C-contiguous float32 output bytes -- byte
# equality of a float32 buffer is exactly uint32-view equality of the same buffer.
# Three corpora, including the `self.padding > 0` branch.
# ANY movement here is a STOP, never a tolerance to widen.
_G10_BITID_CORPORA = {
    "unit_0_1": dict(
        dim=64, ndim=3, shape=(2, 5, 3), scale=1.0,
        sha256="0698ae23cfc543cc5be52838033d6de1419ea70a522e9e4d0002ae4c7b22c196",
        first4_uint32=[1054040804, 1043114908, 1032456626, 1021047983],
    ),
    "large_pos_0_2000": dict(
        dim=64, ndim=3, shape=(2, 5, 3), scale=2000.0,
        sha256="e0325428a52f653178ac11973de9e8d4688a6fc1fe6f2c99cf4b9098d2cfc657",
        first4_uint32=[1043766199, 3201544536, 1044050347, 3197571705],
    ),
    "padded_dim66_ndim4": dict(
        dim=66, ndim=4, shape=(2, 5, 4), scale=64.0,
        sha256="8e480891c0c51019245b02278fa04fd15d5788a979d9fe5ad7c1b672aa128b46",
        first4_uint32=[1063012181, 1060742752, 1053801419, 1061305212],
    ),
}


def _g10_coords(shape, scale):
    """Deterministic float32 coordinates in ``[0, scale)``."""
    rng = np.random.default_rng(_G10_SEED)
    return (rng.random(shape) * scale).astype("float32")


def _g10_reference_f64(coords_f64, dim, ndim, max_wavelength=10000.0):
    """Independent float64 numpy oracle for the sin/cos embedding.

    Reimplements the published formula (sin then cos per coordinate, concatenated
    across coordinates, zero-padded to ``dim``) rather than calling the layer, so a
    swapped sin/cos half or a zeroed output is visible.
    """
    ndim_padding = dim % ndim
    dim_per_ndim = (dim - ndim_padding) // ndim
    padding = ndim_padding + (dim_per_ndim % 2) * ndim
    eff = (dim - padding) // ndim
    omega = 1.0 / (max_wavelength ** (np.arange(0, eff, 2, dtype=np.float64) / eff))
    freqs = coords_f64[..., None] * omega
    emb = np.concatenate([np.sin(freqs), np.cos(freqs)], axis=-1)
    emb = emb.reshape(*coords_f64.shape[:-1], -1)
    if padding > 0:
        emb = np.concatenate([emb, np.zeros((*emb.shape[:-1], padding))], axis=-1)
    return emb


class TestContinuousSinCosEmbed:

    # ---- constructor validation -------------------------------------

    def test_ctor_rejects_bad_args(self):
        with pytest.raises(ValueError):
            ContinuousSinCosEmbed(dim=0, ndim=2)
        with pytest.raises(ValueError):
            ContinuousSinCosEmbed(dim=64, ndim=0)
        with pytest.raises(ValueError):
            ContinuousSinCosEmbed(dim=64, ndim=2, max_wavelength=0.0)
        with pytest.raises(ValueError):
            ContinuousSinCosEmbed(dim=2, ndim=4)

    # ---- forward / shape --------------------------------------------

    @pytest.mark.parametrize("shape,ndim", [((2, 5, 3), 3), ((5, 3), 3), ((2, 7, 2), 2)])
    def test_forward_shape(self, shape, ndim):
        layer = ContinuousSinCosEmbed(dim=64, ndim=ndim)
        x = keras.ops.convert_to_tensor(np.random.rand(*shape).astype("float32"))
        out = layer(x)
        # full embedding width == dim
        assert tuple(out.shape) == tuple(shape[:-1]) + (64,)

    def test_compute_output_shape_matches_actual(self):
        for dim, ndim, in_shape in [(64, 3, (2, 5, 3)), (66, 4, (2, 5, 4)), (60, 3, (2, 5, 3))]:
            layer = ContinuousSinCosEmbed(dim=dim, ndim=ndim)
            x = keras.ops.convert_to_tensor(np.random.rand(*in_shape).astype("float32"))
            actual = int(layer(x).shape[-1])
            declared = layer.compute_output_shape(in_shape)[-1]
            assert actual == declared == dim

    # ---- graph safety (locks the removed eager convert_to_numpy) -----

    def test_graph_trace_no_eager(self):
        layer = ContinuousSinCosEmbed(dim=64, ndim=3)  # assert_positive=True default
        x = tf.constant(np.random.rand(2, 5, 3).astype("float32"))
        eager = keras.ops.convert_to_numpy(layer(x))
        f = tf.function(lambda t: layer(t),
                        input_signature=[tf.TensorSpec([None, None, 3], tf.float32)])
        np.testing.assert_allclose(eager, f(x).numpy(), atol=1e-6)

    def test_graph_trace_with_padding(self):
        layer = ContinuousSinCosEmbed(dim=66, ndim=4)
        x = tf.constant(np.random.rand(2, 5, 4).astype("float32"))
        f = tf.function(lambda t: layer(t),
                        input_signature=[tf.TensorSpec([None, None, 4], tf.float32)])
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(layer(x)), f(x).numpy(), atol=1e-6)

    # ---- build idempotency ------------------------------------------

    def test_double_build_stable(self):
        layer = ContinuousSinCosEmbed(dim=64, ndim=3)
        layer.build((2, 5, 3))
        n1 = len(layer.weights)
        layer.build((2, 5, 3))
        assert len(layer.weights) == n1

    # ---- serialization ----------------------------------------------

    def test_get_config_round_trip(self):
        layer = ContinuousSinCosEmbed(dim=48, ndim=3, max_wavelength=5000.0, assert_positive=False)
        rebuilt = ContinuousSinCosEmbed.from_config(layer.get_config())
        assert rebuilt.dim == 48 and rebuilt.ndim == 3
        assert rebuilt.max_wavelength == 5000.0 and rebuilt.assert_positive is False

    # ---- dtype policies (G-10) --------------------------------------

    @pytest.mark.parametrize("dtype_policy", _G10_POLICIES, indirect=True)
    def test_forward_matches_float64_reference_under_all_policies(self, dtype_policy):
        """The layer must RUN and be ACCURATE at every dtype policy, coords in [0, 64).

        At HEAD this raises `InvalidArgumentError [Op:Mul]` at three of the four
        policies: `call()` forces `coords` to a literal "float32" while Keras
        autocasts the `omega` weight to the compute dtype, so the two operands of
        the frequency multiply disagree.
        """
        with tf.device("/CPU:0"):
            layer = ContinuousSinCosEmbed(dim=64, ndim=3)
            coords = _g10_coords((2, 5, 3), _G10_COORD_HI)
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

        reference = _g10_reference_f64(coords.astype(np.float64), dim=64, ndim=3)
        actual = np.asarray(keras.ops.convert_to_numpy(out), dtype=np.float64)
        # rtol=0 is mandatory: assert_allclose's default rtol=1e-7 otherwise
        # silently contributes to a nominally-atol bound.
        np.testing.assert_allclose(
            actual, reference, atol=_G10_TOL[dtype_policy], rtol=0,
            err_msg=(f"policy {dtype_policy}, coords in [0, {_G10_COORD_HI})"),
        )

    @pytest.mark.parametrize("corpus", sorted(_G10_BITID_CORPORA))
    @pytest.mark.parametrize("dtype_policy", ["float32"], indirect=True)
    def test_float32_output_is_bit_identical_to_head(self, dtype_policy, corpus):
        """I-B: the float32 bytes must not move. Device-pinned to CPU.

        Any movement is a STOP for the G-10 change, not a tolerance to widen.
        """
        spec = _G10_BITID_CORPORA[corpus]
        with tf.device("/CPU:0"):
            layer = ContinuousSinCosEmbed(dim=spec["dim"], ndim=spec["ndim"])
            x = keras.ops.convert_to_tensor(_g10_coords(spec["shape"], spec["scale"]))
            out = np.ascontiguousarray(keras.ops.convert_to_numpy(layer(x)))

        assert out.dtype == np.float32
        assert out.reshape(-1)[:4].view(np.uint32).tolist() == spec["first4_uint32"], (
            f"{corpus}: leading float32 words moved from the HEAD reference"
        )
        assert hashlib.sha256(out.tobytes()).hexdigest() == spec["sha256"], (
            f"{corpus}: float32 output is NOT bit-identical to HEAD"
        )

    @pytest.mark.parametrize("dtype_policy", _G10_POLICIES, indirect=True)
    def test_graph_trace_matches_eager_under_all_policies(self, dtype_policy):
        """Graph safety must survive the widening at every policy.

        Guards the `plan_2026-06-15_9dbb87c1/D-001` constraint: no eager host
        materialization may creep back into `call()`.
        """
        with tf.device("/CPU:0"):
            layer = ContinuousSinCosEmbed(dim=64, ndim=3)
            tf_dtype = tf.as_dtype(layer.compute_dtype)
            x = tf.cast(tf.constant(_g10_coords((2, 5, 3), _G10_COORD_HI)), tf_dtype)
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

    @pytest.mark.parametrize(
        "dtype_policy", ["mixed_float16", "mixed_bfloat16"], indirect=True)
    def test_fp16_large_position_error_bound_is_pinned(self, dtype_policy):
        """H-10: the reduced-precision LARGE-POSITION limitation, made executable.

        REGIME (a bound without its corpus is meaningless): coords drawn uniformly
        from [0, 2000), `dim=64, ndim=3, max_wavelength=10000`, CPU, compared against
        the float64 oracle built from the ORIGINAL float32 coordinates.

        The bound is TWO-SIDED on purpose. The LOWER edge is the non-vacuous half: it
        asserts the limitation is REAL and would go red if anyone ever claimed (or
        made) this layer accurate at fp16/bf16 for large positions -- including via
        the partial `autocast=False` mitigation, which measures 0.16 / 0.56 here and
        trips the lower edge. The UPPER edge pins the magnitude, and goes red for a
        dead `call()` (a zeros return measures ~1.0 at both policies).

        This is NOT a claim that fp16 works. It is the opposite: the documented
        failure is now something the suite executes rather than something a docstring
        merely asserts. The cause is the AUTOCAST BOUNDARY, before `call()` runs.
        """
        lo, hi = _G10_LARGE_POS_BOUND[dtype_policy]
        with tf.device("/CPU:0"):
            layer = ContinuousSinCosEmbed(dim=64, ndim=3)
            coords = _g10_coords((2, 5, 3), 2000.0)
            # FLOAT32 coords ON PURPOSE, not pre-cast to compute_dtype. This is the
            # test's whole point: the caller hands over full float32 precision and
            # Keras narrows it at the AUTOCAST BOUNDARY anyway. Pre-casting here would
            # make the lower edge unfalsifiable -- the error would be pinned by the
            # test's own input, so a layer that had somehow avoided the narrowing
            # would still pass. (Measured: it does not help; float32-in gives exactly
            # the same 0.4745 as fp16-in.)
            x = keras.ops.convert_to_tensor(coords)
            out = layer(x)
            actual = np.asarray(
                keras.ops.convert_to_numpy(tf.cast(out, tf.float64)), dtype=np.float64)

        reference = _g10_reference_f64(coords.astype(np.float64), dim=64, ndim=3)
        err = float(np.max(np.abs(actual - reference)))
        assert lo < err <= hi, (
            f"policy {dtype_policy}, coords in [0, 2000): max abs error {err:.6f} "
            f"left the pinned band ({lo}, {hi}]. Below the band means the documented "
            f"limitation no longer holds as written (update the class docstring "
            f"note); above it means something else broke."
        )

    # ---- serialization (continued) ----------------------------------

    def test_keras_round_trip(self, tmp_path):
        inp = keras.Input(shape=(5, 3), dtype="float32")
        out = ContinuousSinCosEmbed(dim=64, ndim=3)(inp)
        model = keras.Model(inp, out)
        x = np.random.rand(2, 5, 3).astype("float32")
        before = keras.ops.convert_to_numpy(model(x))
        path = os.path.join(tmp_path, "csincos.keras")
        model.save(path)
        after = keras.ops.convert_to_numpy(keras.models.load_model(path)(x))
        np.testing.assert_allclose(before, after, atol=1e-6)
