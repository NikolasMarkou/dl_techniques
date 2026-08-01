import pytest
import itertools
import numpy as np
import tensorflow as tf
import keras
import os
import tempfile

from dl_techniques.layers.geometric.supernode_pooling import SupernodePooling

# ---------------------------------------------------------------------
# dtype-policy corpus (G-10 / H-08)
# ---------------------------------------------------------------------

# The four policies this layer must survive. `tests/test_layers/conftest.py::dtype_policy`
# is the house fixture (the restore-safe global-policy set/teardown lives there, once);
# its own params cover three of them, so `mixed_bfloat16` is supplied by INDIRECT
# parametrization instead of a second copy of the set/restore dance in this module.
_G10_POLICIES = ("float32", "mixed_float16", "float64", "mixed_bfloat16")

# A 3x3x3 unit grid, NOT random points. The neighbour predicate is a DISCRETE
# comparison (`sq_distance <= radius**2`), so a corpus whose distances sit near the
# radius makes the whole test a coin flip at reduced precision: with 60 random points
# the closest squared distance to the boundary was 1.7e-04, which bfloat16 rounding
# alone can flip. On this grid the squared distances are exactly {0,1,2,3,4,...} and
# the margin to `radius**2 = 1.44` is 0.44 — far outside bfloat16 resolution — so a
# tolerance failure means the ARITHMETIC moved, not that the neighbour set changed.
_SP_POSITIONS = np.array(
    list(itertools.product([0.0, 1.0, 2.0], repeat=3)), dtype="float32")
_SP_INDICES = np.array([0, 4, 13, 22, 26], dtype="int32")
_SP_RADIUS = 1.2  # neighbour counts per supernode: 4, 6, 7, 6, 4
_SP_HIDDEN = 16

# Max abs deviation of the policy-P output from the SAME-WEIGHTS float32 output,
# MEASURED on CPU with the corpus above (float32 0.0 / float64 2.4e-07 /
# mixed_float16 6.5e-04 / mixed_bfloat16 1.14e-02), each with ~1.5-3x headroom.
# The reference output's max magnitude is ~1.24, so a `call()` returning zeros, or an
# `_aggregate_messages` that dropped its divisor (values would scale by the 4..7
# neighbour count), is RED by two orders of magnitude against every row here.
_SP_TOL = {
    "float32": 1.0e-06,
    "float64": 1.0e-06,
    "mixed_float16": 2.0e-03,
    "mixed_bfloat16": 3.0e-02,
}


def _sp_inputs(compute_dtype="float32"):
    return {
        "positions": keras.ops.cast(
            keras.ops.convert_to_tensor(_SP_POSITIONS), compute_dtype),
        "supernode_indices": keras.ops.convert_to_tensor(_SP_INDICES),
    }


def _sp_float32_reference():
    """Weights + output of a float32 SupernodePooling, per mode.

    Computed at IMPORT time, i.e. during pytest collection, before any test body has
    touched the global dtype policy. Sub-layers capture the global policy in their own
    ``__init__``, so a ``dtype='float32'`` kwarg on the parent would NOT pin them — the
    reference has to be taken while the ambient policy really is float32.
    """
    assert keras.mixed_precision.global_policy().name == "float32", (
        "the float32 reference must be captured under the float32 policy"
    )
    ref = {}
    for mode in ("abspos", "relpos"):
        with tf.device("/CPU:0"):
            layer = SupernodePooling(
                hidden_dim=_SP_HIDDEN, ndim=3, radius=_SP_RADIUS, mode=mode)
            out = keras.ops.convert_to_numpy(layer(_sp_inputs()))
        ref[mode] = (layer.get_weights(), out.astype(np.float64))
    return ref


_SP_REFERENCE = _sp_float32_reference()


# ===========================================================================
# TestSupernodePooling
# ===========================================================================


class TestSupernodePooling:
    """Test suite for SupernodePooling."""

    @pytest.fixture
    def hidden_dim(self) -> int:
        return 32

    @pytest.fixture
    def ndim(self) -> int:
        return 3

    @pytest.fixture
    def num_points(self) -> int:
        return 100

    @pytest.fixture
    def num_supernodes(self) -> int:
        return 5

    @pytest.fixture
    def dict_inputs(self, num_points, ndim, num_supernodes) -> dict:
        positions = tf.random.normal([num_points, ndim])
        supernode_indices = tf.constant([0, 10, 20, 30, 40])
        return {"positions": positions, "supernode_indices": supernode_indices}

    @pytest.fixture
    def layer_instance(self, hidden_dim, ndim) -> SupernodePooling:
        return SupernodePooling(hidden_dim=hidden_dim, ndim=ndim, radius=2.0)

    # ------------------------------------------------------------------

    def test_initialization_defaults(self, hidden_dim, ndim):
        """Test initialization with default parameters."""
        layer = SupernodePooling(hidden_dim=hidden_dim, ndim=ndim, radius=1.0)
        assert layer.hidden_dim == hidden_dim
        assert layer.ndim == ndim
        assert layer.radius == 1.0
        assert layer.k_neighbors is None
        assert layer.max_neighbors == 32
        assert layer.mode == "relpos"
        assert layer.use_bias is True

    def test_initialization_custom(self, hidden_dim, ndim):
        """Test initialization with custom parameters."""
        layer = SupernodePooling(
            hidden_dim=hidden_dim,
            ndim=ndim,
            k_neighbors=8,
            max_neighbors=16,
            mode="abspos",
            activation="relu",
            use_bias=False,
            name="custom_sp",
        )
        assert layer.k_neighbors == 8
        assert layer.radius is None
        assert layer.max_neighbors == 16
        assert layer.mode == "abspos"
        assert layer.use_bias is False
        assert layer.name == "custom_sp"

    def test_invalid_hidden_dim(self, ndim):
        """Test that non-positive hidden_dim raises ValueError."""
        with pytest.raises(ValueError, match="hidden_dim"):
            SupernodePooling(hidden_dim=0, ndim=ndim, radius=1.0)

    def test_invalid_ndim(self, hidden_dim):
        """Test that non-positive ndim raises ValueError."""
        with pytest.raises(ValueError, match="ndim"):
            SupernodePooling(hidden_dim=hidden_dim, ndim=0, radius=1.0)

    def test_invalid_max_neighbors(self, hidden_dim, ndim):
        """Test that non-positive max_neighbors raises ValueError."""
        with pytest.raises(ValueError, match="max_neighbors"):
            SupernodePooling(hidden_dim=hidden_dim, ndim=ndim, radius=1.0, max_neighbors=0)

    def test_invalid_mode(self, hidden_dim, ndim):
        """Test that unknown mode raises ValueError."""
        with pytest.raises(ValueError, match="mode"):
            SupernodePooling(hidden_dim=hidden_dim, ndim=ndim, radius=1.0, mode="bad")

    def test_invalid_both_radius_and_k_neighbors(self, hidden_dim, ndim):
        """Test that specifying both radius and k_neighbors raises ValueError."""
        with pytest.raises(ValueError, match="Exactly one"):
            SupernodePooling(hidden_dim=hidden_dim, ndim=ndim, radius=1.0, k_neighbors=5)

    def test_invalid_neither_radius_nor_k_neighbors(self, hidden_dim, ndim):
        """Test that specifying neither radius nor k_neighbors raises ValueError."""
        with pytest.raises(ValueError, match="Exactly one"):
            SupernodePooling(hidden_dim=hidden_dim, ndim=ndim)

    def test_invalid_negative_radius(self, hidden_dim, ndim):
        """Test that negative radius raises ValueError."""
        with pytest.raises(ValueError, match="radius"):
            SupernodePooling(hidden_dim=hidden_dim, ndim=ndim, radius=-1.0)

    def test_invalid_negative_k_neighbors(self, hidden_dim, ndim):
        """Test that negative k_neighbors raises ValueError."""
        with pytest.raises(ValueError, match="k_neighbors"):
            SupernodePooling(hidden_dim=hidden_dim, ndim=ndim, k_neighbors=-1)

    def test_build(self, layer_instance, ndim):
        """Test that the layer builds correctly with dict input_shape."""
        layer_instance.build({"positions": (100, ndim), "supernode_indices": (5,)})
        assert layer_instance.built is True
        assert layer_instance.pos_embed.built is True
        assert layer_instance.message_mlp.built is True
        assert layer_instance.proj_layer.built is True

    def test_forward_pass_radius(self, hidden_dim, ndim, dict_inputs):
        """Test forward pass with radius mode."""
        layer = SupernodePooling(hidden_dim=hidden_dim, ndim=ndim, radius=2.0)
        output = layer(dict_inputs)
        assert output.shape == (1, 5, hidden_dim)
        assert not np.any(np.isnan(output.numpy()))

    @pytest.mark.xfail(reason="Known bug in _knn_neighbors: take_along_axis shape mismatch")
    def test_forward_pass_k_neighbors(self, hidden_dim, ndim):
        """Test forward pass with k_neighbors mode."""
        layer = SupernodePooling(hidden_dim=hidden_dim, ndim=ndim, k_neighbors=3)
        positions = tf.random.normal([50, ndim])
        supernode_indices = tf.constant([0, 10, 20, 30, 40])
        inputs = {"positions": positions, "supernode_indices": supernode_indices}
        output = layer(inputs)
        assert output.shape == (1, 5, hidden_dim)
        assert not np.any(np.isnan(output.numpy()))

    def test_forward_pass_abspos_mode(self, hidden_dim, ndim, dict_inputs):
        """Test forward pass with abspos mode."""
        layer = SupernodePooling(hidden_dim=hidden_dim, ndim=ndim, radius=2.0, mode="abspos")
        output = layer(dict_inputs)
        assert output.shape == (1, 5, hidden_dim)
        assert not np.any(np.isnan(output.numpy()))

    def test_forward_pass_relpos_mode(self, hidden_dim, ndim, dict_inputs):
        """Test forward pass with relpos mode."""
        layer = SupernodePooling(hidden_dim=hidden_dim, ndim=ndim, radius=2.0, mode="relpos")
        output = layer(dict_inputs)
        assert output.shape == (1, 5, hidden_dim)
        assert not np.any(np.isnan(output.numpy()))

    def test_output_shape(self, layer_instance, dict_inputs, hidden_dim):
        """Test that output shape matches expected (1, num_supernodes, hidden_dim)."""
        output = layer_instance(dict_inputs)
        assert output.shape == (1, 5, hidden_dim)

    def test_compute_output_shape(self, layer_instance, ndim, hidden_dim):
        """Test compute_output_shape returns correct shape."""
        input_shape = {"positions": (100, ndim), "supernode_indices": (5,)}
        computed = layer_instance.compute_output_shape(input_shape)
        assert computed == (1, None, hidden_dim)

    def test_compute_output_shape_invalid_input(self, layer_instance):
        """Test compute_output_shape raises on invalid input."""
        with pytest.raises(ValueError):
            layer_instance.compute_output_shape((100, 3))

    def test_serialization(self, hidden_dim, ndim):
        """get_config / from_config round-trip preserves attributes."""
        original = SupernodePooling(
            hidden_dim=hidden_dim,
            ndim=ndim,
            radius=2.5,
            max_neighbors=16,
            mode="relpos",
            activation="relu",
            use_bias=False,
            name="sp_s",
        )
        config = original.get_config()
        restored = SupernodePooling.from_config(config)

        assert restored.hidden_dim == original.hidden_dim
        assert restored.ndim == original.ndim
        assert restored.radius == original.radius
        assert restored.k_neighbors == original.k_neighbors
        assert restored.max_neighbors == original.max_neighbors
        assert restored.mode == original.mode
        assert restored.use_bias == original.use_bias

    def test_serialization_k_neighbors(self, hidden_dim, ndim):
        """get_config / from_config round-trip preserves k_neighbors mode."""
        original = SupernodePooling(
            hidden_dim=hidden_dim,
            ndim=ndim,
            k_neighbors=10,
            mode="abspos",
            name="sp_k",
        )
        config = original.get_config()
        restored = SupernodePooling.from_config(config)

        assert restored.k_neighbors == original.k_neighbors
        assert restored.radius is None
        assert restored.mode == original.mode

    def test_gradient_flow(self, hidden_dim, ndim):
        """Gradients propagate through the layer."""
        layer = SupernodePooling(hidden_dim=hidden_dim, ndim=ndim, radius=2.0)
        positions = tf.Variable(tf.random.normal([50, ndim]))
        supernode_indices = tf.constant([0, 10, 20, 30])
        inputs = {"positions": positions, "supernode_indices": supernode_indices}

        with tf.GradientTape() as tape:
            out = layer(inputs)
            loss = tf.reduce_mean(tf.square(out))
        grads = tape.gradient(loss, positions)
        assert grads is not None
        assert np.any(grads.numpy() != 0)

    def test_numerical_stability(self, hidden_dim, ndim):
        """No NaN / Inf with extreme input values."""
        layer = SupernodePooling(hidden_dim=hidden_dim, ndim=ndim, radius=5.0)
        supernode_indices = tf.constant([0, 5, 10])
        for scale in [1e-6, 1e6]:
            positions = tf.ones([20, ndim]) * scale
            # Add small offsets so points are not all identical
            offsets = tf.random.normal([20, ndim], stddev=scale * 0.01)
            positions = positions + offsets
            inputs = {"positions": positions, "supernode_indices": supernode_indices}
            out = layer(inputs)
            assert not np.any(np.isnan(out.numpy())), f"NaN at scale {scale}"
            assert not np.any(np.isinf(out.numpy())), f"Inf at scale {scale}"

    def test_2d_points(self, hidden_dim):
        """Test with 2D point cloud."""
        layer = SupernodePooling(hidden_dim=hidden_dim, ndim=2, radius=3.0)
        positions = tf.random.normal([50, 2])
        supernode_indices = tf.constant([0, 10, 20])
        inputs = {"positions": positions, "supernode_indices": supernode_indices}
        output = layer(inputs)
        assert output.shape == (1, 3, hidden_dim)

    def test_different_supernode_counts(self, hidden_dim, ndim):
        """Layer handles different numbers of supernodes."""
        layer = SupernodePooling(hidden_dim=hidden_dim, ndim=ndim, radius=2.0)
        positions = tf.random.normal([50, ndim])
        for n_super in [1, 3, 10]:
            indices = tf.constant(list(range(0, n_super * 5, 5))[:n_super])
            inputs = {"positions": positions, "supernode_indices": indices}
            output = layer(inputs)
            assert output.shape == (1, n_super, hidden_dim)

    @pytest.mark.xfail(reason="Known bug in _knn_neighbors: take_along_axis shape mismatch")
    def test_k_neighbors_mode_output_shape(self, hidden_dim, ndim):
        """Test k_neighbors mode produces correct output shape."""
        layer = SupernodePooling(hidden_dim=hidden_dim, ndim=ndim, k_neighbors=3)
        positions = tf.random.normal([30, ndim])
        supernode_indices = tf.constant([0, 10, 20])
        inputs = {"positions": positions, "supernode_indices": supernode_indices}
        output = layer(inputs)
        assert output.shape == (1, 3, hidden_dim)

    def test_relpos_has_rel_pos_embed(self, hidden_dim, ndim):
        """relpos mode should create rel_pos_embed layer."""
        layer = SupernodePooling(hidden_dim=hidden_dim, ndim=ndim, radius=1.0, mode="relpos")
        assert layer.rel_pos_embed is not None

    def test_abspos_no_rel_pos_embed(self, hidden_dim, ndim):
        """abspos mode should not create rel_pos_embed layer."""
        layer = SupernodePooling(hidden_dim=hidden_dim, ndim=ndim, radius=1.0, mode="abspos")
        assert layer.rel_pos_embed is None

    # ---- dtype policies (G-10 / H-08) --------------------------------

    @pytest.mark.parametrize("mode", ["abspos", "relpos"])
    @pytest.mark.parametrize("dtype_policy", _G10_POLICIES, indirect=True)
    def test_forward_matches_float32_reference_under_all_policies(self, dtype_policy, mode):
        """Same weights, every policy: the layer must RUN and agree with float32.

        This is the SECOND half of G-10. Widening `ContinuousSinCosEmbed` alone is not
        enough — `_aggregate_messages` normalises by a neighbour count it hard-casts to
        "float32" while the numerator is at the compute dtype, so at three of the four
        policies this layer still died, with a DIFFERENT exception from the embedding's
        (`TypeError: 'x' and 'y' must have the same dtype`).
        """
        ref_weights, ref_out = _SP_REFERENCE[mode]
        with tf.device("/CPU:0"):
            layer = SupernodePooling(
                hidden_dim=_SP_HIDDEN, ndim=3, radius=_SP_RADIUS, mode=mode)
            x = _sp_inputs(layer.compute_dtype)
            _ = layer(x)                     # build, so the weights exist
            layer.set_weights(ref_weights)   # identical parameters at every policy
            out = layer(x)
            # bfloat16 has no plain-numpy view; compare the VALUES at float32.
            out_np = keras.ops.convert_to_numpy(
                tf.cast(out, tf.float32)).astype(np.float64)

        assert keras.backend.standardize_dtype(out.dtype) == layer.compute_dtype, (
            f"policy {dtype_policy}, mode {mode}: expected {layer.compute_dtype}, "
            f"got {keras.backend.standardize_dtype(out.dtype)}"
        )
        assert out_np.shape == ref_out.shape
        np.testing.assert_allclose(
            out_np, ref_out, atol=_SP_TOL[dtype_policy], rtol=0,
            err_msg=f"policy {dtype_policy}, mode {mode}, 3x3x3 unit grid, radius 1.2",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
