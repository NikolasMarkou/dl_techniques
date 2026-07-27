"""Test suite for RPCAttention layer.

This module contains comprehensive tests for the RPCAttention layer,
focusing on the robustness mechanisms, PCP decomposition, and SVD stability
in addition to standard Keras layer functionality.
"""

import pytest
import numpy as np
import tensorflow as tf
import keras
import tempfile
import os
from keras import ops

from dl_techniques.layers.attention.common import MASK_BIAS_VALUE
from dl_techniques.layers.attention.rpc_attention import RPCAttention


class TestRPCAttention:
    """Test suite for RPCAttention layer."""

    @pytest.fixture
    def input_tensor(self):
        """Create a test input tensor."""
        return keras.random.normal((2, 10, 64))  # (batch_size, seq_len, dim)

    @pytest.fixture
    def layer_instance(self):
        """Create a default layer instance for testing."""
        return RPCAttention(dim=64, num_heads=8)

    @pytest.fixture
    def different_configs(self):
        """Provide different layer configurations for testing."""
        return [
            {"dim": 32, "num_heads": 4, "lambda_sparse": 0.01},
            {"dim": 128, "num_heads": 8, "max_pcp_iter": 5},
            {"dim": 256, "num_heads": 16, "svd_threshold": 0.5},
            {"dim": 64, "num_heads": 8, "dropout_rate": 0.1, "qkv_bias": True},
        ]

    # ==================== Initialization Tests ====================

    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        layer = RPCAttention(dim=64, num_heads=8)

        assert layer.dim == 64
        assert layer.num_heads == 8
        assert layer.lambda_sparse == 0.1
        assert layer.max_pcp_iter == 10
        assert layer.svd_threshold == 1.0
        assert layer.dropout_rate == 0.0
        assert isinstance(layer.kernel_initializer, keras.initializers.GlorotUniform)

        # Check computed attributes
        assert layer.head_dim == 8
        np.testing.assert_allclose(layer.attention_scale, 1.0 / np.sqrt(8))

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        custom_reg = keras.regularizers.L2(1e-4)

        layer = RPCAttention(
            dim=128,
            num_heads=8,
            lambda_sparse=0.5,
            max_pcp_iter=20,
            svd_threshold=0.1,
            qkv_bias=True,
            kernel_regularizer=custom_reg
        )

        assert layer.dim == 128
        assert layer.lambda_sparse == 0.5
        assert layer.max_pcp_iter == 20
        assert layer.svd_threshold == 0.1
        assert layer.qkv_bias is True
        assert layer.kernel_regularizer == custom_reg

    def test_invalid_dim_mismatch(self):
        """Test that invalid dim/head ratio raises ValueError."""
        with pytest.raises(ValueError, match="dim \\(63\\) must be divisible by num_heads \\(8\\)"):
            RPCAttention(dim=63, num_heads=8)

    def test_invalid_pcp_params(self):
        """Test validation of PCP-specific parameters."""
        with pytest.raises(ValueError, match="lambda_sparse must be positive"):
            RPCAttention(dim=64, lambda_sparse=-0.1)

        with pytest.raises(ValueError, match="max_pcp_iter must be positive"):
            RPCAttention(dim=64, max_pcp_iter=0)

        with pytest.raises(ValueError, match="svd_threshold must be positive"):
            RPCAttention(dim=64, svd_threshold=-1.0)

    # ==================== Build Process Tests ====================

    def test_build_process(self, input_tensor):
        """Test that the layer builds properly."""
        layer = RPCAttention(dim=64, num_heads=8)
        layer(input_tensor)  # Forward pass triggers build

        assert layer.built is True
        assert layer.to_qkv.built is True
        assert layer.to_out.built is True

        # Verify weight shapes
        # to_qkv: (dim, 3*dim) + bias if used
        expected_kernel_shape = (64, 64 * 3)
        assert layer.to_qkv.kernel.shape == expected_kernel_shape

    def test_build_input_shape_validation(self):
        """Test input shape validation in build."""
        layer = RPCAttention(dim=64, num_heads=8)

        # Test invalid input shape (2D instead of 3D)
        with pytest.raises(ValueError, match="Expected 3D input"):
            layer.build((32, 64))

        # Test dimension mismatch
        with pytest.raises(ValueError, match="Last dimension of input"):
            layer.build((None, 10, 32))

    # ==================== Forward Pass & Computation Tests ====================

    def test_forward_pass_basic(self, input_tensor):
        """Test basic forward pass functionality."""
        layer = RPCAttention(dim=64, num_heads=8)
        output = layer(input_tensor)

        assert not tf.reduce_any(tf.math.is_nan(output))
        assert not tf.reduce_any(tf.math.is_inf(output))
        assert output.shape == input_tensor.shape

    def test_return_attention_scores(self, input_tensor):
        """Test return_attention_scores=True behavior."""
        layer = RPCAttention(dim=64, num_heads=8)
        output, weights = layer(input_tensor, return_attention_scores=True)

        assert output.shape == input_tensor.shape
        # Weights shape: (batch, num_heads, seq_len, seq_len)
        expected_weights_shape = (2, 8, 10, 10)
        assert weights.shape == expected_weights_shape

        # Weights should sum to 1 on the last axis (softmax)
        sums = tf.reduce_sum(weights, axis=-1)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(sums),
            np.ones(sums.shape),
            rtol=1e-5, atol=1e-5
        )

    def test_pcp_determinism(self, input_tensor):
        """
        Test that PCP decomposition is deterministic for the same input
        (SVD results should be consistent).
        """
        layer = RPCAttention(dim=64, num_heads=8, max_pcp_iter=5)

        # Run twice in inference mode
        out1 = layer(input_tensor, training=False)
        out2 = layer(input_tensor, training=False)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(out1),
            keras.ops.convert_to_numpy(out2),
            rtol=1e-6, atol=1e-6,
            err_msg="RPC execution should be deterministic during inference"
        )

    # ==================== SVD & Numerical Stability Tests ====================

    def test_svd_stability_zeros(self):
        """Test stability when attention matrix is all zeros."""
        layer = RPCAttention(dim=64, num_heads=8)

        # Zero input -> Zero Q, K, V -> Zero Attention Matrix
        zeros_input = tf.zeros((2, 10, 64))
        output = layer(zeros_input)

        assert not tf.reduce_any(tf.math.is_nan(output))
        # With zero input, output should be zero (biases are zero by default)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output),
            np.zeros((2, 10, 64)),
            atol=1e-6
        )

    def test_svd_stability_random(self):
        """Test stability with random normal inputs (checking for SVD convergence issues)."""
        # Use a larger matrix to stress the SVD
        layer = RPCAttention(dim=32, num_heads=4, max_pcp_iter=5)

        for _ in range(5):
            inp = keras.random.normal((2, 20, 32)) # (batch, seq, dim)
            output = layer(inp)
            assert not tf.reduce_any(tf.math.is_nan(output)), "NaN found in RPC output"

    def test_pcp_convergence_logic(self, input_tensor):
        """
        Indirectly test that increasing iterations or changing lambda affects output.
        This verifies the internal loops are actually running and using params.
        """
        # Layer with 1 iteration
        layer_fast = RPCAttention(dim=64, num_heads=8, max_pcp_iter=1, lambda_sparse=0.1)
        layer_fast.build(input_tensor.shape)
        # Force same weights
        weights = layer_fast.get_weights()

        # Layer with 10 iterations
        layer_deep = RPCAttention(dim=64, num_heads=8, max_pcp_iter=10, lambda_sparse=0.1)
        layer_deep.build(input_tensor.shape)
        layer_deep.set_weights(weights)

        out_fast = layer_fast(input_tensor, training=False)
        out_deep = layer_deep(input_tensor, training=False)

        # Outputs should be slightly different due to more refinement steps
        diff = tf.reduce_mean(tf.abs(out_fast - out_deep))
        assert diff > 0.0, "More PCP iterations should change the result"

    # ==================== Masking Tests ====================

    def test_mask_handling(self, input_tensor):
        """Test that attention masks are correctly applied before PCP."""
        layer = RPCAttention(dim=64, num_heads=8)

        seq_len = input_tensor.shape[1]
        # Mask the last 5 tokens
        mask = tf.concat([
            tf.ones((2, seq_len - 5)),
            tf.zeros((2, 5))
        ], axis=1)
        # Expand for attention logic usually handled by framework,
        # but here we pass raw mask to layer which expects (batch, seq, seq) or broadcastable
        mask_expanded = mask[:, None, :] * mask[:, :, None] # (batch, seq, seq)

        output_masked = layer(input_tensor, mask=mask_expanded)
        output_nomask = layer(input_tensor)

        assert not tf.reduce_all(tf.equal(output_masked, output_nomask))
        assert not tf.reduce_any(tf.math.is_nan(output_masked))

    # ==================== Gradient Flow Tests ====================

    def test_gradient_flow_through_svd(self, input_tensor):
        """
        Critical test: Ensure gradients propagate through SVD and the iterative loop.
        SVD gradients can be tricky in some backends.
        """
        layer = RPCAttention(dim=64, num_heads=8, max_pcp_iter=2) # Keep iter low for speed

        with tf.GradientTape() as tape:
            inputs = tf.Variable(input_tensor)
            outputs = layer(inputs)
            loss = tf.reduce_mean(tf.square(outputs))

        grads = tape.gradient(loss, layer.trainable_variables)

        # Check gradients exist
        assert len(grads) > 0
        assert all(g is not None for g in grads)

        # Check gradients are non-zero (implies flow through SVD)
        # We check the kernel weights of the projection layers
        for g in grads:
            assert tf.reduce_max(tf.abs(g)) > 0.0

    # ==================== Serialization Tests ====================

    def test_serialization(self, input_tensor):
        """Test complete serialization cycle."""
        # Note: dim=64 to match input_tensor fixture
        layer = RPCAttention(
            dim=64,
            num_heads=8,
            lambda_sparse=0.2,
            max_pcp_iter=5,
            svd_threshold=0.5,
            qkv_bias=True
        )

        # Build first
        layer(input_tensor)

        config = layer.get_config()
        recreated = RPCAttention.from_config(config)

        assert recreated.dim == 64
        assert recreated.lambda_sparse == 0.2
        assert recreated.max_pcp_iter == 5
        assert recreated.svd_threshold == 0.5
        assert recreated.qkv_bias is True

    def test_model_save_load(self, input_tensor):
        """Test saving/loading within a full model."""
        inputs = keras.Input(shape=(10, 64))
        x = RPCAttention(dim=64, num_heads=8)(inputs)
        outputs = keras.layers.Dense(1)(x)
        model = keras.Model(inputs, outputs)

        # Run prediction
        pred_orig = model.predict(input_tensor, verbose=0)

        with tempfile.TemporaryDirectory() as tmpdirname:
            path = os.path.join(tmpdirname, "model.keras")
            model.save(path)

            loaded_model = keras.models.load_model(
                path,
                custom_objects={"RPCAttention": RPCAttention}
            )

            pred_loaded = loaded_model.predict(input_tensor, verbose=0)

            np.testing.assert_allclose(
                pred_orig, pred_loaded,
                rtol=1e-5, atol=1e-5
            )

    # ==================== Edge Case Tests ====================

    def test_single_token_sequence(self):
        """
        Test behavior with sequence length 1.
        SVD on 1x1 matrices should degenerate gracefully.
        """
        layer = RPCAttention(dim=64, num_heads=8)
        input_one = tf.random.normal((2, 1, 64))
        output = layer(input_one)

        assert output.shape == (2, 1, 64)
        assert not tf.reduce_any(tf.math.is_nan(output))

    def test_batch_consistency(self):
        """Test that different batch sizes work."""
        layer = RPCAttention(dim=64, num_heads=8)

        # Batch size 1
        out1 = layer(tf.random.normal((1, 10, 64)))
        assert out1.shape == (1, 10, 64)

        # Batch size large
        out16 = layer(tf.random.normal((16, 10, 64)))
        assert out16.shape == (16, 10, 64)


# =====================================================================
# Mixed-precision masking: the `-1e9` -> `ops.svd` hazard
# (plan-2026-07-27-b4ef45f0, step 2)
# =====================================================================
#
# What is under guard here, and why it is NOT just "another fp16 NaN test":
#
# `RPCAttention` masks with `ops.where`, so it can never form `0 * -inf = NaN`
# the way its multiply-form siblings do. Its defect is entirely DOWNSTREAM: the
# masked scores are fed to `_pcp_decomposition`, whose `ops.svd` is
#   (a) NaN-poisoned by a single non-finite entry, and
#   (b) MEASURED to have no float16 kernel at all on this backend
#       (`Could not find device for node: Svd[T=DT_HALF]`, TF 2.18 / CUDA;
#        registered kernels are CPU float/double/complex and GPU float/double).
# So under `mixed_float16` the masked forward pass does not merely return NaN —
# it raises. Both symptoms have the same cure: the masked score matrix must
# reach the SVD in >= float32, which is exactly what
# `common.apply_attention_mask`'s DEFAULT `out_dtype` guarantees.
#
# Anti-vacuity note on sizes. The reduction-size trap (`plans/LESSONS.md`:
# `N = 7` once hid an fp16 `-inf` that only appeared at `N >= 512`) does not
# transfer to this site: the hazard is per-ELEMENT (a dtype overflow of a
# constant) plus a missing kernel, neither of which is a reduction. It is
# nevertheless proven reachable rather than assumed — see
# `TestRPCAttentionMaskHazardIsReal`, which asserts that the compute dtype
# really is float16 under the policy, that `float16(MASK_BIAS_VALUE)` really is
# `-inf`, and that the mask really does mask a nonzero number of positions at
# the shapes used below (16384 of 32768 score entries).

_MP_B, _MP_N, _MP_D, _MP_H = 2, 64, 32, 4
_MP_ITER = 3                     # keep the SVD cost sane; 3 sweeps still call `ops.svd`
_MP_KEEP = _MP_N // 2            # first half kept, second half masked
_MP_SEED = 1234

# Absolute tolerance for "this policy's forward agrees with the float32 control".
#
# These numbers are NOT an fp16 error budget — they are this layer's own DTYPE
# CONDITIONING, measured on unmodified HEAD with byte-identical weights and no
# fp16 anywhere in sight (float32 forward vs float64 forward):
#
#     mask       max|float32 - float64|      output absmax
#     no mask         0.0021                     2.57
#     all_ones        0.0021                     2.57
#     padding         0.1216                     1.88
#     causal          0.1636                     3.23
#
# Merely APPLYING a `-1e9` mask makes the output ~6% dtype-sensitive, and the
# figure does not shrink with fewer PCP sweeps (probed at 1, 2, 3 and 10). The
# reason is structural: the masked score matrix spans nine orders of magnitude
# (`-1e9` next to O(10) logits), and a float32 SVD of such a matrix carries an
# absolute error of order `1e9 * eps_f32`. So an `allclose`-at-fp16-tolerance
# criterion is simply not available at this site for a masked input, and pretending
# otherwise would produce a tolerance tuned until it passed.
#
# The tolerances below are therefore per-mask-kind, and `TestRPCAttentionConditioning`
# pins the justification: it re-measures the float32-vs-float64 divergence and fails
# if the loose entries stop being NECESSARY (someone improves the conditioning) or
# stop being SUFFICIENT (it gets worse). The load-bearing guards for this step are
# finiteness, the SVD-path observation and the polarity test — not this comparison.
_MP_ATOL = {
    "all_ones": {"float32": 1e-5, "mixed_float16": 0.05, "float64": 0.05},
    "padding": {"float32": 1e-5, "mixed_float16": 0.5, "float64": 0.5},
    "causal": {"float32": 1e-5, "mixed_float16": 0.5, "float64": 0.5},
}

# Largest change permitted on an UNMASKED query row when a MASKED position is
# perturbed. MEASURED, not guessed:
#
#     float32        0.0        in isolation; 1.1e-06 inside a full-suite session
#     mixed_float16  0.000977   (= 2**-10, one fp16 ulp of the output — quantization)
#     float64        0.0254
#
# against a KEPT-position perturbation of 17.0 in all three. The float32 entry is
# the reason its tolerance is 1e-3 rather than "exact": the masked score is
# `logit + (-1e9)`, which rounds back to exactly `-1e9` in float32, so the two
# score matrices ARE bit-identical — but the batched `ops.svd` that consumes them
# is not bit-reproducible across a session (observed 0.0 running this file alone,
# 1.1e-06 running the whole attention directory). 1e-3 sits three orders above
# that noise and four orders below the 17.0 signal. The float64 entry is
# the interesting one and is a real, if small, consequence of adopting the shared
# helper: `common.apply_attention_mask` ADDS `MASK_BIAS_VALUE` where this site
# previously REPLACED the score with it. In float32 the two are indistinguishable —
# one ulp at 1e9 is 64, so `-1e9 + O(10)` rounds back to exactly `-1e9` and the
# masked entry is a constant again — but float64 resolves the sum, so a masked
# score varies with its (masked) token and the global SVD leaks ~1.5% of that back
# into the unmasked rows. Recorded rather than papered over; the separation from a
# genuine polarity inversion (which measures ~4.4, i.e. 170x larger, and is what
# this test exists to catch) is not remotely threatened by it.
_POLARITY_TOL = {"float32": 1e-3, "mixed_float16": 2e-3, "float64": 0.05}

_F32_REFERENCE = {}


def _mp_input():
    """Deterministic ``(B, N, D)`` float32 input, shared by every test below."""
    return np.random.default_rng(7).standard_normal(
        (_MP_B, _MP_N, _MP_D)
    ).astype("float32")


def _mp_mask(kind):
    """One of the three masks success criterion SC1 requires, as float32 numpy.

    ``'all_ones'`` masks nothing (the shape of the hopfield catastrophe),
    ``'padding'`` is a rank-2 ``(B, N)`` key-axis mask (exercises the layer's
    rank-2 expand branch), ``'causal'`` is a rank-3 ``(B, N, N)`` lower-triangular
    mask. None of the three produces a fully-masked query row — that degenerate
    case belongs to a different mechanism (see the polarity test's docstring).
    """
    if kind == "all_ones":
        return np.ones((_MP_B, _MP_N), dtype="float32")
    if kind == "padding":
        m = np.ones((_MP_B, _MP_N), dtype="float32")
        m[:, _MP_KEEP:] = 0.0
        return m
    if kind == "causal":
        return np.broadcast_to(
            np.tril(np.ones((_MP_N, _MP_N), dtype="float32")),
            (_MP_B, _MP_N, _MP_N),
        ).copy()
    raise ValueError(f"unknown mask kind {kind!r}")


def _mp_weights():
    """Explicit float32 kernels for the two Dense sub-layers (``qkv_bias=False``).

    Seeding alone is NOT enough here, and that is not a detail: MEASURED, a
    ``glorot_uniform`` draw under a ``float64`` policy differs from the same-seed
    draw under ``float32`` by up to 0.597 — the initializer samples in the
    variable dtype. Comparing a float64 forward against a float32 control on
    *differently initialized* weights would report a 0.67 "deviation" that has
    nothing to do with the code under test.
    """
    rng = np.random.default_rng(_MP_SEED)
    return [
        (rng.standard_normal((_MP_D, 3 * _MP_D)) * 0.2).astype("float32"),
        (rng.standard_normal((_MP_D, _MP_D)) * 0.2).astype("float32"),
    ]


def _mp_layer():
    """A layer whose weights are byte-identical under every dtype policy."""
    layer = RPCAttention(dim=_MP_D, num_heads=_MP_H, max_pcp_iter=_MP_ITER)
    layer.build((_MP_B, _MP_N, _MP_D))
    layer.set_weights(_mp_weights())
    return layer


def _float32_reference(kind):
    """Masked float32 output for ``kind``, memoized, computed under an explicit policy.

    This is the CONTROL every mixed-precision assertion is compared against. It is
    computed under a policy this function sets and restores itself, so it is valid
    no matter which parametrization of ``dtype_policy`` happens to call it first.
    """
    if kind not in _F32_REFERENCE:
        previous = keras.mixed_precision.global_policy().name
        keras.mixed_precision.set_global_policy("float32")
        try:
            layer = _mp_layer()
            out = layer(
                ops.convert_to_tensor(_mp_input()),
                mask=ops.convert_to_tensor(_mp_mask(kind)),
            )
            _F32_REFERENCE[kind] = ops.convert_to_numpy(out).astype("float32")
        finally:
            keras.mixed_precision.set_global_policy(previous)
    return _F32_REFERENCE[kind]


class _PCPSpy:
    """Records every tensor that enters and leaves ``_pcp_decomposition``.

    This is what makes the SVD-path test independent of how the mask expression
    is *spelled*: it observes the actual tensor handed to ``ops.svd``, so it stays
    meaningful even if the masking code is rewritten again later.
    """

    def __init__(self, layer):
        self._original = layer._pcp_decomposition
        self.inputs = []
        self.outputs = []
        layer._pcp_decomposition = self

    def __call__(self, attention_matrix):
        self.inputs.append(attention_matrix)
        result = self._original(attention_matrix)
        self.outputs.append(result)
        return result


def _numpy(tensor):
    return ops.convert_to_numpy(tensor).astype("float32")


class TestRPCAttentionMaskHazardIsReal:
    """Anti-vacuity. If these stop holding, every fp16 test below is worthless."""

    def test_policy_really_selects_float16_compute(self, dtype_policy):
        expected = {
            "float32": "float32",
            "mixed_float16": "float16",
            "float64": "float64",
        }[dtype_policy]
        assert keras.mixed_precision.global_policy().compute_dtype == expected

    def test_mask_bias_value_overflows_in_the_compute_dtype(self):
        with np.errstate(over="ignore"):
            assert np.isneginf(np.float16(MASK_BIAS_VALUE)), (
                "anti-vacuity FAILED: float16(MASK_BIAS_VALUE) is not -inf, so the "
                "hazard this module guards is not reproducible here."
            )

    def test_the_padding_mask_actually_masks_something(self):
        mask = _mp_mask("padding")
        # Each masked KEY blanks one column of every (N, N) score matrix, and there
        # are H such matrices per batch element.
        masked_scores = int((mask == 0).sum()) * _MP_H * _MP_N
        assert (mask == 0).sum() > 0
        assert masked_scores == 16384, (
            f"expected 16384 masked score entries at "
            f"(B={_MP_B}, H={_MP_H}, N={_MP_N}), got {masked_scores}"
        )


class TestRPCAttentionMixedPrecisionMask:
    """SC1 / SC2: masked forward is finite and agrees with the float32 control."""

    @pytest.mark.parametrize("kind", ["all_ones", "padding", "causal"])
    def test_masked_forward_is_finite_and_matches_float32(self, dtype_policy, kind):
        layer = _mp_layer()
        x = ops.convert_to_tensor(_mp_input())
        mask = ops.convert_to_tensor(_mp_mask(kind))

        try:
            out = _numpy(layer(x, mask=mask))
        except Exception as exc:                      # noqa: BLE001 - reported verbatim
            pytest.fail(
                f"masked forward ({kind}) RAISED under policy {dtype_policy!r}: "
                f"{type(exc).__name__}: {str(exc)[:300]}"
            )

        n_bad = int((~np.isfinite(out)).sum())
        assert n_bad == 0, (
            f"{n_bad}/{out.size} non-finite output entries for a {kind!r} mask under "
            f"policy {dtype_policy!r}"
        )

        reference = _float32_reference(kind)
        atol = _MP_ATOL[kind][dtype_policy]
        max_dev = float(np.abs(out - reference).max())
        assert max_dev <= atol, (
            f"{kind!r} mask under {dtype_policy!r} deviates from the float32 control "
            f"by {max_dev:.4g} > {atol:.4g}"
        )
        # Scale check: a tolerance this loose could pass on a collapsed output.
        assert float(np.abs(out).max()) > 0.5 * float(np.abs(reference).max()), (
            f"{kind!r} mask under {dtype_policy!r}: output absmax "
            f"{np.abs(out).max():.4g} collapsed relative to the control "
            f"{np.abs(reference).max():.4g}"
        )


class TestRPCAttentionConditioning:
    """Pins the JUSTIFICATION for the loose tolerances in :data:`_MP_ATOL`.

    This test manages its own policies (it needs two at once) and touches no fp16.
    It exists so that the loose masked-path tolerances above cannot silently become
    either unnecessary or insufficient: it re-measures the float32-vs-float64
    divergence of the same forward pass and asserts the tolerance brackets it.
    """

    @pytest.mark.parametrize("kind", ["all_ones", "padding", "causal"])
    def test_masked_pcp_path_is_dtype_sensitive_by_this_much(self, kind):
        def forward(policy):
            previous = keras.mixed_precision.global_policy().name
            keras.mixed_precision.set_global_policy(policy)
            try:
                layer = _mp_layer()
                return ops.convert_to_numpy(
                    layer(
                        ops.convert_to_tensor(_mp_input()),
                        mask=ops.convert_to_tensor(_mp_mask(kind)),
                    )
                ).astype("float64")
            finally:
                keras.mixed_precision.set_global_policy(previous)

        divergence = float(np.abs(forward("float32") - forward("float64")).max())
        budget = _MP_ATOL[kind]["float64"]

        assert divergence <= budget, (
            f"the float32-vs-float64 divergence of a {kind!r}-masked forward is "
            f"{divergence:.4g}, which EXCEEDS the tolerance {budget:.4g} that the "
            "agreement test above relies on; re-derive the tolerances"
        )
        if kind == "all_ones":
            assert divergence < 0.05, (
                f"an unmasked-in-effect forward should be well conditioned; measured "
                f"{divergence:.4g}"
            )
        else:
            assert divergence > 0.01, (
                f"a {kind!r}-masked forward is now well conditioned ({divergence:.4g}); "
                f"the loose {budget:.4g} tolerance above is no longer justified and "
                "must be tightened"
            )


class TestRPCAttentionSVDPath:
    """SC3: verified at the SVD boundary itself, not at the mask expression.

    The mask expression could be "fixed" and this layer would still be broken if
    anything non-finite — or anything in a dtype ``ops.svd`` has no kernel for —
    still reached ``_pcp_decomposition``. That is precisely the pre-mortem
    "the ``-inf`` merely moves downstream", so it gets its own observation point.
    """

    def test_pcp_decomposition_never_sees_a_non_finite_or_fp16_score(self, dtype_policy):
        layer = _mp_layer()
        spy = _PCPSpy(layer)
        x = ops.convert_to_tensor(_mp_input())
        mask = ops.convert_to_tensor(_mp_mask("padding"))

        raised = None
        try:
            output = layer(x, mask=mask)
        except Exception as exc:                      # noqa: BLE001
            output, raised = None, exc

        assert spy.inputs, (
            f"`_pcp_decomposition` was never reached under policy {dtype_policy!r}"
        )

        for index, scores in enumerate(spy.inputs):
            scores_np = _numpy(scores)
            n_bad = int((~np.isfinite(scores_np)).sum())
            assert n_bad == 0, (
                f"{n_bad}/{scores_np.size} non-finite score entries reached "
                f"`_pcp_decomposition` (call {index}) / `ops.svd` under policy "
                f"{dtype_policy!r}; a single one NaN-poisons the whole decomposition"
            )
            dtype = keras.backend.standardize_dtype(scores.dtype)
            assert dtype != "float16", (
                f"`_pcp_decomposition` received a float16 tensor under policy "
                f"{dtype_policy!r}; `ops.svd` has NO float16 kernel on this backend "
                "(measured: Svd[T=DT_HALF] -> NotFoundError)"
            )

        assert raised is None, (
            f"masked forward RAISED under policy {dtype_policy!r}: "
            f"{type(raised).__name__}: {str(raised)[:300]}"
        )

        for low_rank, sparse in spy.outputs:
            for name, part in (("L", low_rank), ("S", sparse)):
                part_np = _numpy(part)
                n_bad = int((~np.isfinite(part_np)).sum())
                assert n_bad == 0, (
                    f"{n_bad}/{part_np.size} non-finite entries in the PCP {name} "
                    f"component under policy {dtype_policy!r}"
                )

        out = _numpy(output)
        assert np.isfinite(out).all(), (
            "PCP produced finite components but the LAYER OUTPUT is not finite — the "
            "`-inf` moved downstream instead of being removed"
        )


class TestRPCAttentionMaskPolarity:
    """SC6: the mask must suppress the MASKED positions, not the kept ones.

    A polarity inversion (passing ``mask == 0`` where ``mask != 0`` is meant)
    raises nothing, changes no shape, and leaves the output perfectly finite —
    finiteness tests cannot see it. Only an influence test can.

    Why the mask here is the rank-3 OUTER PRODUCT ``keep_i * keep_j`` rather than
    the rank-2 key-axis mask used above: ``_pcp_decomposition`` is a GLOBAL matrix
    factorization, so perturbing token ``p`` perturbs the unmasked query ROW ``p``
    of the score matrix, and the SVD spreads that change over every row. Masking
    the row as well as the column makes the whole score matrix invariant to the
    perturbation, which is what turns "a masked key has no influence" into an
    exactly-measurable statement (measured 0.0 on unmodified float32 HEAD).
    Its side effect is that the masked query rows are fully masked — a degenerate
    softmax row, a DIFFERENT mechanism owned by another step — so the assertions
    below are restricted to the unmasked query rows.
    """

    def test_a_masked_position_has_no_influence_on_unmasked_query_rows(
        self, dtype_policy
    ):
        keep = np.ones((_MP_N,), dtype="float32")
        keep[_MP_KEEP:] = 0.0
        mask = np.broadcast_to(
            np.einsum("i,j->ij", keep, keep), (_MP_B, _MP_N, _MP_N)
        ).copy()

        layer = _mp_layer()
        mask_tensor = ops.convert_to_tensor(mask)

        def forward(array):
            return _numpy(layer(ops.convert_to_tensor(array), mask=mask_tensor))

        base = _mp_input()
        perturbed_masked = base.copy()
        perturbed_masked[:, _MP_KEEP + 3, :] += 5.0          # a MASKED position
        perturbed_kept = base.copy()
        perturbed_kept[:, 3, :] += 5.0                       # a KEPT position

        try:
            out_base = forward(base)
            out_masked = forward(perturbed_masked)
            out_kept = forward(perturbed_kept)
        except Exception as exc:                              # noqa: BLE001
            pytest.fail(
                f"masked forward RAISED under policy {dtype_policy!r}: "
                f"{type(exc).__name__}: {str(exc)[:300]}"
            )

        rows = slice(0, _MP_KEEP)
        assert np.isfinite(out_base[:, rows]).all(), (
            "unmasked query rows are not finite; the influence comparison below "
            "would be meaningless"
        )

        delta_masked = float(np.abs(out_masked[:, rows] - out_base[:, rows]).max())
        delta_kept = float(np.abs(out_kept[:, rows] - out_base[:, rows]).max())
        tolerance = _POLARITY_TOL[dtype_policy]

        assert delta_masked <= tolerance, (
            f"perturbing a MASKED position changed the unmasked query rows by "
            f"{delta_masked:.6g} > {tolerance:.6g} under policy {dtype_policy!r} — the "
            "mask polarity is INVERTED (the layer is attending to the padding)"
        )
        assert delta_kept > 0.1, (
            f"perturbing a KEPT position changed the output by only {delta_kept:.6g}; "
            "the test is vacuous — the layer is ignoring its input, so the "
            "no-influence assertion above proves nothing"
        )
        assert delta_kept > 50.0 * max(delta_masked, tolerance), (
            f"masked influence {delta_masked:.6g} is not decisively smaller than kept "
            f"influence {delta_kept:.6g}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])