"""Tests for attention_routing_capsule — AttentionRoutingCapsule and CapsuleBlockV2."""

import os
import tempfile

import numpy as np
import keras
import pytest
import tensorflow as tf

from dl_techniques.layers.attention.attention_routing_capsule import (
    AttentionRoutingCapsule,
    CapsuleBlockV2,
)


# ---------------------------------------------------------------------


class TestAttentionRoutingCapsule:
    """AttentionRoutingCapsule: single-step attention routing."""

    @pytest.fixture
    def input_tensor(self):
        # (batch, num_input_capsules, input_dim_capsules)
        return keras.random.normal([4, 32, 8])

    @pytest.fixture
    def layer_instance(self):
        return AttentionRoutingCapsule(num_capsules=10, dim_capsules=16)

    # ---- init / validation ----

    def test_initialization_defaults(self):
        layer = AttentionRoutingCapsule(num_capsules=10, dim_capsules=16)
        assert layer.num_capsules == 10
        assert layer.dim_capsules == 16
        assert layer.softmax_axis == "output"
        assert layer.top_k is None
        assert layer.use_bias is True
        assert layer.use_load_balancing is False

    def test_initialization_custom(self):
        layer = AttentionRoutingCapsule(
            num_capsules=5,
            dim_capsules=8,
            softmax_axis="input",
            top_k=4,
            use_bias=False,
            use_load_balancing=True,
            load_balancing_weight=0.05,
            eps=1e-6,
        )
        assert layer.softmax_axis == "input"
        assert layer.top_k == 4
        assert layer.use_bias is False
        assert layer.use_load_balancing is True
        assert layer.load_balancing_weight == 0.05
        assert layer.eps == 1e-6

    def test_invalid_parameters(self):
        with pytest.raises(ValueError, match="num_capsules must be positive"):
            AttentionRoutingCapsule(num_capsules=0, dim_capsules=16)
        with pytest.raises(ValueError, match="dim_capsules must be positive"):
            AttentionRoutingCapsule(num_capsules=10, dim_capsules=-1)
        with pytest.raises(ValueError, match="softmax_axis"):
            AttentionRoutingCapsule(num_capsules=10, dim_capsules=16, softmax_axis="bogus")
        with pytest.raises(ValueError, match="top_k must be positive"):
            AttentionRoutingCapsule(num_capsules=10, dim_capsules=16, top_k=0)
        with pytest.raises(ValueError, match="load_balancing_weight"):
            AttentionRoutingCapsule(
                num_capsules=10, dim_capsules=16, load_balancing_weight=-0.1
            )

    # ---- build / forward ----

    def test_build(self, input_tensor, layer_instance):
        out = layer_instance(input_tensor)
        assert layer_instance.built is True
        assert layer_instance.W is not None
        assert layer_instance.q is not None
        assert layer_instance.bias is not None
        assert out.shape == (4, 10, 16)

    def test_output_shape(self):
        configs = [
            (4, 32, 8, 10, 16),
            (2, 100, 4, 5, 8),
            (1, 50, 16, 20, 32),
        ]
        for B, N_in, D_in, N_out, D_out in configs:
            x = keras.random.normal([B, N_in, D_in])
            layer = AttentionRoutingCapsule(num_capsules=N_out, dim_capsules=D_out)
            out = layer(x)
            assert out.shape == (B, N_out, D_out)
            assert layer.compute_output_shape(x.shape) == (B, N_out, D_out)

    def test_forward_pass_no_nan(self, input_tensor, layer_instance):
        out = layer_instance(input_tensor)
        assert not np.any(np.isnan(out.numpy()))
        assert not np.any(np.isinf(out.numpy()))

    def test_lengths_in_unit_interval(self, input_tensor, layer_instance):
        """sigmoid magnitude → ||v|| ∈ (0, 1)."""
        out = layer_instance(input_tensor)
        lengths = np.sqrt(np.sum(np.square(out.numpy()), axis=-1))
        assert np.all(lengths > 0.0)
        assert np.all(lengths < 1.0)

    def test_lengths_show_variance(self, input_tensor, layer_instance):
        out = layer_instance(input_tensor)
        lengths = np.sqrt(np.sum(np.square(out.numpy()), axis=-1))
        assert lengths.std() > 1e-3, "decoupled magnitude collapsed to constant"

    # ---- routing variants ----

    def test_softmax_axis_input(self, input_tensor):
        layer = AttentionRoutingCapsule(
            num_capsules=10, dim_capsules=16, softmax_axis="input"
        )
        out = layer(input_tensor)
        assert out.shape == (4, 10, 16)
        assert not np.any(np.isnan(out.numpy()))

    def test_top_k_masking(self, input_tensor):
        layer = AttentionRoutingCapsule(num_capsules=10, dim_capsules=16, top_k=3)
        out = layer(input_tensor)
        assert out.shape == (4, 10, 16)
        assert not np.any(np.isnan(out.numpy()))

    def test_top_k_with_input_axis(self, input_tensor):
        layer = AttentionRoutingCapsule(
            num_capsules=10, dim_capsules=16, top_k=5, softmax_axis="input"
        )
        out = layer(input_tensor)
        assert out.shape == (4, 10, 16)
        assert not np.any(np.isnan(out.numpy()))

    def test_top_k_clamped_to_axis_size(self, input_tensor):
        # top_k larger than num_capsules along the soft-maxed axis -> clamp.
        layer = AttentionRoutingCapsule(num_capsules=10, dim_capsules=16, top_k=99)
        out = layer(input_tensor)
        assert out.shape == (4, 10, 16)

    # ---- load-balancing ----

    def test_load_balancing_aux_loss_in_training(self, input_tensor):
        layer = AttentionRoutingCapsule(
            num_capsules=10, dim_capsules=16, use_load_balancing=True
        )
        # Trigger build with training=False to get baseline.
        _ = layer(input_tensor, training=False)
        assert len(layer.losses) == 0
        # Training=True should attach the aux loss.
        _ = layer(input_tensor, training=True)
        assert len(layer.losses) >= 1
        # The aux loss must be a non-negative scalar.
        aux = float(layer.losses[-1].numpy())
        assert aux >= 0.0

    def test_load_balancing_disabled_no_aux_loss(self, input_tensor):
        layer = AttentionRoutingCapsule(
            num_capsules=10, dim_capsules=16, use_load_balancing=False
        )
        _ = layer(input_tensor, training=True)
        # No aux losses contributed by the layer when load-balancing is off.
        assert len(layer.losses) == 0

    # ---- gradients ----

    def test_gradient_flow(self, input_tensor):
        layer = AttentionRoutingCapsule(num_capsules=10, dim_capsules=16)
        with tf.GradientTape() as tape:
            out = layer(input_tensor, training=True)
            loss = tf.reduce_mean(tf.square(out))
        grads = tape.gradient(loss, layer.trainable_variables)
        assert all(g is not None for g in grads)
        for g, v in zip(grads, layer.trainable_variables):
            assert not np.any(np.isnan(g.numpy())), f"NaN gradient on {v.name}"
            # At least some gradients should be non-trivial.
        # The W matrix and the query vector q must receive non-zero gradient.
        named = {v.name: g for v, g in zip(layer.trainable_variables, grads)}
        for name, g in named.items():
            if "transformation_weights" in name or "routing_query" in name:
                assert np.max(np.abs(g.numpy())) > 1e-8, f"zero grad on {name}"

    # ---- serialization ----

    def test_get_config_round_trip(self, input_tensor):
        original = AttentionRoutingCapsule(
            num_capsules=10,
            dim_capsules=16,
            softmax_axis="input",
            top_k=5,
            use_load_balancing=True,
            load_balancing_weight=0.05,
        )
        _ = original(input_tensor)
        config = original.get_config()
        recreated = AttentionRoutingCapsule.from_config(config)
        assert recreated.num_capsules == original.num_capsules
        assert recreated.dim_capsules == original.dim_capsules
        assert recreated.softmax_axis == original.softmax_axis
        assert recreated.top_k == original.top_k
        assert recreated.use_load_balancing == original.use_load_balancing


# ---------------------------------------------------------------------


class TestCapsuleBlockV2:
    """CapsuleBlockV2: routing + dropout + length-preserving direction LN."""

    @pytest.fixture
    def input_tensor(self):
        return keras.random.normal([4, 32, 8])

    def test_initialization_defaults(self):
        block = CapsuleBlockV2(num_capsules=10, dim_capsules=16)
        assert block.num_capsules == 10
        assert block.dropout_rate == 0.0
        assert block.direction_only_norm is False

    def test_invalid_dropout(self):
        with pytest.raises(ValueError, match="dropout_rate"):
            CapsuleBlockV2(num_capsules=10, dim_capsules=16, dropout_rate=1.5)

    def test_invalid_direction_only_norm(self):
        with pytest.raises(TypeError, match="direction_only_norm"):
            CapsuleBlockV2(num_capsules=10, dim_capsules=16, direction_only_norm="yes")

    def test_forward_pass_default(self, input_tensor):
        block = CapsuleBlockV2(num_capsules=10, dim_capsules=16)
        out = block(input_tensor)
        assert out.shape == (4, 10, 16)
        assert not np.any(np.isnan(out.numpy()))

    def test_forward_pass_with_dropout(self, input_tensor):
        block = CapsuleBlockV2(num_capsules=10, dim_capsules=16, dropout_rate=0.3)
        out = block(input_tensor, training=True)
        assert out.shape == (4, 10, 16)

    def test_direction_only_norm_preserves_length(self, input_tensor):
        """Direction-only LN must preserve capsule magnitudes."""
        block_ln = CapsuleBlockV2(
            num_capsules=10,
            dim_capsules=16,
            direction_only_norm=True,
            kernel_initializer=keras.initializers.RandomNormal(seed=42),
        )
        block_no_ln = CapsuleBlockV2(
            num_capsules=10,
            dim_capsules=16,
            direction_only_norm=False,
            kernel_initializer=keras.initializers.RandomNormal(seed=42),
        )
        # Build via forward pass.
        out_ln = block_ln(input_tensor)
        out_no_ln = block_no_ln(input_tensor)

        # Sync the routing weights so both pathways start identical.
        block_ln.routing.set_weights(block_no_ln.routing.get_weights())
        out_ln = block_ln(input_tensor)
        out_no_ln = block_no_ln(input_tensor)

        len_ln = np.sqrt(np.sum(np.square(out_ln.numpy()), axis=-1))
        len_no_ln = np.sqrt(np.sum(np.square(out_no_ln.numpy()), axis=-1))
        assert np.allclose(len_ln, len_no_ln, atol=1e-5), (
            f"direction_only_norm rescaled magnitudes; "
            f"max abs diff = {np.max(np.abs(len_ln - len_no_ln))}"
        )

    def test_serialization_round_trip(self, input_tensor):
        block = CapsuleBlockV2(
            num_capsules=10,
            dim_capsules=16,
            dropout_rate=0.2,
            direction_only_norm=True,
            top_k=8,
            use_load_balancing=True,
        )
        _ = block(input_tensor)
        config = block.get_config()
        recreated = CapsuleBlockV2.from_config(config)
        assert recreated.num_capsules == 10
        assert recreated.dropout_rate == 0.2
        assert recreated.direction_only_norm is True
        assert recreated.top_k == 8
        assert recreated.use_load_balancing is True

    def test_full_model_save_load_round_trip(self, input_tensor):
        """End-to-end: wrap the V2 layer in a Model, save, load, compare."""
        inp = keras.Input(shape=(32, 8))
        x = AttentionRoutingCapsule(num_capsules=10, dim_capsules=16)(inp)
        model = keras.Model(inputs=inp, outputs=x)

        # Reference forward pass.
        ref_out = model(input_tensor).numpy()

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "v2_model.keras")
            model.save(path)
            reloaded = keras.models.load_model(path)
            # Architecture ok: forward pass shape unchanged.
            new_out = reloaded(input_tensor).numpy()
            assert new_out.shape == ref_out.shape
            assert np.allclose(new_out, ref_out, atol=1e-5)
