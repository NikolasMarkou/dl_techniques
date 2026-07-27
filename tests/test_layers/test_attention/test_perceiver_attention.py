"""Test suite for PerceiverAttention layer.

First-ever regression coverage for PerceiverAttention, the asymmetric
cross-attention block from the Perceiver / Perceiver IO architectures. It wraps
``MultiHeadCrossAttention`` so a small latent query array attends to a large
data (key/value) array. Covers:
1. Initialization & Configuration
2. Forward pass (asymmetric: distinct query / kv sequence lengths)
3. Self-attention mode (kv_input=None)
4. get_config / from_config round-trip
5. Full `.keras` model save/load round-trip (functional API; single tensor out)
"""

import os
import tempfile

import pytest
import numpy as np
import tensorflow as tf
import keras

from dl_techniques.layers.attention.perceiver_attention import PerceiverAttention


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def minimal_config():
    return {"dim": 32, "num_heads": 4}


@pytest.fixture
def query_input():
    """Small latent query array: [B, Q_seq, dim]."""
    return keras.random.normal((2, 8, 32))


@pytest.fixture
def kv_input():
    """Large data array (keys/values): [B, KV_seq, dim] — distinct seq len."""
    return keras.random.normal((2, 40, 32))


# ==============================================================================
# 1. Initialization & Configuration
# ==============================================================================

class TestInitialization:

    def test_defaults(self, minimal_config):
        layer = PerceiverAttention(**minimal_config)
        assert layer.dim == 32
        assert layer.num_heads == 4
        assert layer.dropout_rate == 0.0
        assert layer.use_bias is True
        assert layer.cross_attention is not None

    def test_invalid_dim_negative(self):
        with pytest.raises(ValueError, match="dim must be positive"):
            PerceiverAttention(dim=-10, num_heads=2)

    def test_invalid_heads_negative(self):
        with pytest.raises(ValueError, match="num_heads must be positive"):
            PerceiverAttention(dim=32, num_heads=0)

    def test_invalid_divisibility(self):
        with pytest.raises(ValueError, match="must be divisible"):
            PerceiverAttention(dim=30, num_heads=4)

    def test_invalid_dropout_range(self):
        with pytest.raises(ValueError, match="dropout_rate"):
            PerceiverAttention(dim=32, num_heads=4, dropout_rate=1.5)


# ==============================================================================
# 2. Forward Pass (asymmetric cross-attention)
# ==============================================================================

class TestForwardPass:

    def test_asymmetric_output_shape(self, minimal_config, query_input, kv_input):
        """Output sequence length matches the QUERY length, not the KV length."""
        layer = PerceiverAttention(**minimal_config)
        out = layer(query_input, kv_input=kv_input)
        # [B, Q_seq, dim] — bottleneck preserves the small latent length (8).
        assert tuple(out.shape) == (2, 8, 32)

    def test_self_attention_mode(self, minimal_config, query_input):
        """kv_input=None falls back to self-attention over the query array."""
        layer = PerceiverAttention(**minimal_config)
        out = layer(query_input)
        assert tuple(out.shape) == (2, 8, 32)

    def test_forward_no_nans(self, minimal_config, query_input, kv_input):
        layer = PerceiverAttention(**minimal_config)
        out = layer(query_input, kv_input=kv_input)
        assert not np.any(np.isnan(np.asarray(out)))

    def test_determinism_inference(self, minimal_config, query_input, kv_input):
        layer = PerceiverAttention(**minimal_config)
        out1 = layer(query_input, kv_input=kv_input, training=False)
        out2 = layer(query_input, kv_input=kv_input, training=False)
        np.testing.assert_allclose(
            np.asarray(out1), np.asarray(out2), atol=1e-6
        )

    def test_gradient_flow(self, minimal_config, query_input, kv_input):
        layer = PerceiverAttention(**minimal_config)
        with tf.GradientTape() as tape:
            out = layer(query_input, kv_input=kv_input)
            loss = tf.reduce_mean(out)
        grads = tape.gradient(loss, layer.trainable_variables)
        assert len(grads) > 0
        assert all(g is not None for g in grads)


# ==============================================================================
# 3. Serialization & Persistence
# ==============================================================================

class TestSerialization:

    def test_get_config(self, minimal_config):
        layer = PerceiverAttention(**minimal_config, dropout_rate=0.2)
        config = layer.get_config()
        assert config["dim"] == 32
        assert config["num_heads"] == 4
        assert config["dropout_rate"] == 0.2

    def test_from_config(self, minimal_config):
        original = PerceiverAttention(**minimal_config)
        rebuilt = PerceiverAttention.from_config(original.get_config())
        assert rebuilt.dim == original.dim
        assert rebuilt.num_heads == original.num_heads
        assert rebuilt.dropout_rate == original.dropout_rate

    def test_model_save_load_loop(self, minimal_config):
        """Full `.keras` round-trip via the functional API (two-input model)."""
        query_in = keras.Input(shape=(8, 32), name="query")
        kv_in = keras.Input(shape=(40, 32), name="kv")
        # Pass kv positionally so the functional API registers it as a second
        # input (the layer's call signature is call(query_input, kv_input, ...)).
        out = PerceiverAttention(**minimal_config)(query_in, kv_in)
        model = keras.Model([query_in, kv_in], out)

        q_data = np.random.normal(size=(2, 8, 32)).astype("float32")
        kv_data = np.random.normal(size=(2, 40, 32)).astype("float32")
        pred_orig = model.predict([q_data, kv_data], verbose=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "perceiver_model.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
            pred_load = loaded.predict([q_data, kv_data], verbose=0)

            np.testing.assert_allclose(pred_orig, pred_load, atol=1e-6)


# ==============================================================================
# 4. Misc
# ==============================================================================

class TestMisc:

    def test_compute_output_shape_single(self, minimal_config):
        layer = PerceiverAttention(**minimal_config)
        assert layer.compute_output_shape((2, 8, 32)) == (2, 8, 32)

    def test_compute_output_shape_list(self, minimal_config):
        layer = PerceiverAttention(**minimal_config)
        shape = layer.compute_output_shape([(2, 8, 32), (2, 40, 32)])
        assert shape == (2, 8, 32)

    def test_kwargs_passthrough(self, minimal_config):
        layer = PerceiverAttention(**minimal_config, name="perceiver_block")
        assert layer.name == "perceiver_block"


# ==============================================================================
# 5. compute_output_shape / build predicate agreement
#    (defect 6, findings/non-mask-defects.md; plan step 6a)
# ==============================================================================

class TestComputeOutputShapeUsesTheBuildPredicate:
    """``compute_output_shape`` must classify shapes exactly as ``build`` does.

    ``build`` uses the ``plan_2026-06-14_7734bacd/D-003`` predicate
    ``_is_list_of_shapes`` (a container whose FIRST ELEMENT is itself a shape).
    ``compute_output_shape`` used a bare ``isinstance(input_shape, list)``, so a
    single 3D shape that Keras had serialized to a plain list — ``[None, 8, 32]``,
    exactly what ``build_from_config`` hands back on a ``.keras`` reload — was
    misread as a list of three inputs and the method returned element 0
    (``None``) instead of the shape.

    These tests pin AGREEMENT between the two methods, which is the actual
    invariant; a finiteness-style "it returns something" test would not.
    """

    # -- the defect itself -------------------------------------------------

    def test_serialized_single_shape_as_a_flat_list(self, minimal_config):
        """The reported failure case: a flat list of scalars is ONE shape."""
        layer = PerceiverAttention(**minimal_config)
        assert layer.compute_output_shape([None, 8, 32]) == (None, 8, 32)

    def test_flat_list_with_concrete_batch(self, minimal_config):
        layer = PerceiverAttention(**minimal_config)
        assert layer.compute_output_shape([2, 8, 32]) == (2, 8, 32)

    # -- agreement with build() -------------------------------------------

    @pytest.mark.parametrize(
        "input_shape",
        [
            (2, 8, 32),                    # single shape, tuple container
            [2, 8, 32],                    # single shape, list container (the defect)
            [None, 8, 32],                 # single shape, list container, dynamic batch
            [(2, 8, 32), (2, 40, 32)],     # list of shapes
            [[2, 8, 32], [2, 40, 32]],     # list of shapes, list elements
        ],
    )
    def test_agrees_with_build_on_every_container_form(self, minimal_config, input_shape):
        """Whatever ``build`` treats as the QUERY shape is what must come out.

        ``build`` is the reference: it is the method carrying the D-003 anchor,
        and it is the one Keras drives on a ``.keras`` reload.
        """
        build_layer = PerceiverAttention(**minimal_config)
        build_layer.build(input_shape)
        # build() stores the query shape it selected on the wrapped layer's
        # weights; re-derive it the same way build() does instead of guessing.
        if isinstance(input_shape[0], (list, tuple)):
            expected = tuple(input_shape[0])
        else:
            expected = tuple(input_shape)

        shape_layer = PerceiverAttention(**minimal_config)
        assert shape_layer.compute_output_shape(input_shape) == expected

    def test_a_flat_list_that_build_accepts_is_not_read_as_three_inputs(self, minimal_config):
        """Anti-vacuity: prove ``build`` really accepts the flat-list form.

        If ``build`` rejected ``[None, 8, 32]`` the agreement test above would be
        comparing against nothing.
        """
        layer = PerceiverAttention(**minimal_config)
        layer.build([None, 8, 32])  # must NOT raise "Expected 2 inputs"
        assert layer.built is True

    def test_tuple_of_shapes_is_classified_the_same_way_build_classifies_it(
        self, minimal_config
    ):
        """A TUPLE container holding two shapes is a list-of-shapes here.

        This layer's ``_is_list_of_shapes`` is deliberately the most permissive
        of the three sibling spellings (see the R13 cross-reference in
        ``perceiver_attention.py::build``) — it accepts a ``tuple`` container,
        while ``multi_head_cross_attention`` / ``multi_head_latent_attention``
        accept only a ``list``. ``compute_output_shape`` must follow THIS
        module's predicate.

        SEPARATE, PRE-EXISTING, OUT OF SCOPE (measured while writing this test):
        ``build(((2, 8, 32), (2, 40, 32)))`` raises
        ``ValueError: Query input must be 3D, got shape ((2, 8, 32), (2, 40, 32))``
        from inside ``MultiHeadCrossAttention.build``, because ``build`` forwards
        the RAW container to the wrapped layer whose own predicate rejects a
        tuple container. That is a defect in the forwarding, not in the
        predicate, and unifying the two predicates is explicitly forbidden by
        the R13 note. Reported, not fixed here.
        """
        layer = PerceiverAttention(**minimal_config)
        assert layer.compute_output_shape(((2, 8, 32), (2, 40, 32))) == (2, 8, 32)

    def test_return_type_is_always_a_tuple(self, minimal_config):
        """A shape returned to Keras must be a tuple, not the caller's list."""
        layer = PerceiverAttention(**minimal_config)
        assert isinstance(layer.compute_output_shape([None, 8, 32]), tuple)
        assert isinstance(layer.compute_output_shape((2, 8, 32)), tuple)
        assert isinstance(
            layer.compute_output_shape([(2, 8, 32), (2, 40, 32)]), tuple
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
