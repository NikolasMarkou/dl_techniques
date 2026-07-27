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


class TestInitializerAndRegularizerPlumbing:
    """Defect 14 (findings/non-mask-defects.md; plan step 6c).

    Two constructor arguments were not plumbed through:

    * ``kernel_initializer`` reached ``W`` and ``q`` but the internal
      ``prob_head`` ``Dense`` hardcoded ``"glorot_uniform"``, so a caller who
      asked for a specific initialization silently did not get it on one of the
      layer's three weight groups.
    * ``kernel_regularizer`` was stored RAW (``self.kernel_regularizer = arg``)
      while ``get_config()`` calls ``keras.regularizers.serialize`` on it — so a
      string spec such as ``"l2"`` was serialized as the bare string instead of
      the canonical dict every sibling layer produces.

    ``get_config()`` KEYS are unchanged by the fix (invariant I5); only the
    stored OBJECT and hence the serialized VALUE become canonical.
    """

    _EXPECTED_OWN_CONFIG_KEYS = {
        "num_capsules",
        "dim_capsules",
        "softmax_axis",
        "top_k",
        "use_bias",
        "use_load_balancing",
        "load_balancing_weight",
        "eps",
        "kernel_initializer",
        "kernel_regularizer",
    }

    @pytest.fixture
    def input_tensor(self):
        return keras.random.normal([4, 32, 8])

    # -- kernel_initializer reaches prob_head -----------------------------

    def test_prob_head_uses_the_constructor_initializer_object(self):
        init = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=1234)
        layer = AttentionRoutingCapsule(
            num_capsules=4, dim_capsules=6, kernel_initializer=init
        )
        assert layer.prob_head.kernel_initializer is layer.kernel_initializer

    def test_prob_head_uses_the_constructor_initializer_string(self):
        layer = AttentionRoutingCapsule(
            num_capsules=4, dim_capsules=6, kernel_initializer="he_normal"
        )
        assert isinstance(
            layer.prob_head.kernel_initializer, keras.initializers.HeNormal
        )

    def test_prob_head_default_is_still_glorot_uniform(self):
        """Anti-regression: the DEFAULT must be unchanged for existing callers."""
        layer = AttentionRoutingCapsule(num_capsules=4, dim_capsules=6)
        assert isinstance(
            layer.prob_head.kernel_initializer, keras.initializers.GlorotUniform
        )

    def test_the_initializer_actually_reaches_the_prob_head_weights(self, input_tensor):
        """Not just the attribute: the built kernel must reflect the choice.

        A constant initializer makes this observable without depending on RNG
        seeding, which is the trap a ``is``-identity check alone would leave open.
        """
        layer = AttentionRoutingCapsule(
            num_capsules=4,
            dim_capsules=6,
            kernel_initializer=keras.initializers.Constant(0.25),
        )
        _ = layer(input_tensor)
        kernel = np.array(layer.prob_head.kernel)
        assert np.allclose(kernel, 0.25), (
            "prob_head kernel was not initialized with the constructor's "
            f"kernel_initializer; got min={kernel.min()} max={kernel.max()}"
        )

    # -- kernel_regularizer canonicalization ------------------------------

    def test_string_regularizer_is_canonicalized_at_construction(self):
        layer = AttentionRoutingCapsule(
            num_capsules=4, dim_capsules=6, kernel_regularizer="l2"
        )
        assert isinstance(layer.kernel_regularizer, keras.regularizers.Regularizer), (
            "kernel_regularizer was stored raw as "
            f"{type(layer.kernel_regularizer).__name__}; it must go through "
            "keras.regularizers.get() so get_config() serializes it canonically"
        )

    def test_object_regularizer_is_stored_as_given(self):
        reg = keras.regularizers.L2(0.03)
        layer = AttentionRoutingCapsule(
            num_capsules=4, dim_capsules=6, kernel_regularizer=reg
        )
        assert layer.kernel_regularizer is reg

    def test_none_regularizer_stays_none(self):
        layer = AttentionRoutingCapsule(num_capsules=4, dim_capsules=6)
        assert layer.kernel_regularizer is None

    def test_an_unknown_string_spec_now_fails_loudly_at_construction(self):
        """Deliberate consequence of routing through ``keras.regularizers.get``.

        A bogus spec used to be accepted and stored verbatim, only to surface
        later (or never, if the weight it guards was never regularized). It now
        raises where the caller wrote it.
        """
        with pytest.raises(ValueError, match="Could not interpret regularizer"):
            AttentionRoutingCapsule(
                num_capsules=4, dim_capsules=6, kernel_regularizer="not_a_regularizer"
            )

    @pytest.mark.parametrize("spec", ["l2", "l1"])
    def test_string_regularizer_serializes_to_the_canonical_dict(self, spec):
        layer = AttentionRoutingCapsule(
            num_capsules=4, dim_capsules=6, kernel_regularizer=spec
        )
        serialized = layer.get_config()["kernel_regularizer"]
        assert isinstance(serialized, dict), (
            f"kernel_regularizer={spec!r} serialized as {serialized!r} "
            "(a bare string) instead of the canonical dict every sibling emits"
        )

    # -- round trips -------------------------------------------------------

    @pytest.mark.parametrize(
        "spec",
        ["l2", keras.regularizers.L1L2(l1=0.01, l2=0.02), None],
        ids=["string_spec", "object_spec", "none"],
    )
    def test_regularizer_survives_get_config_from_config(self, spec):
        original = AttentionRoutingCapsule(
            num_capsules=4, dim_capsules=6, kernel_regularizer=spec
        )
        recreated = AttentionRoutingCapsule.from_config(original.get_config())

        if spec is None:
            assert recreated.kernel_regularizer is None
            return

        assert isinstance(original.kernel_regularizer, keras.regularizers.Regularizer)
        assert isinstance(recreated.kernel_regularizer, keras.regularizers.Regularizer)
        assert type(recreated.kernel_regularizer) is type(original.kernel_regularizer)
        assert (
            recreated.kernel_regularizer.get_config()
            == original.kernel_regularizer.get_config()
        )

    @pytest.mark.parametrize(
        "spec",
        ["l2", keras.regularizers.L1L2(l1=0.01, l2=0.02), None],
        ids=["string_spec", "object_spec", "none"],
    )
    def test_regularizer_survives_a_full_keras_model_round_trip(self, spec, input_tensor):
        inp = keras.Input(shape=(32, 8))
        layer = AttentionRoutingCapsule(
            num_capsules=10, dim_capsules=16, kernel_regularizer=spec
        )
        model = keras.Model(inputs=inp, outputs=layer(inp))
        ref_out = model(input_tensor).numpy()

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "arc_reg.keras")
            model.save(path)
            reloaded = keras.models.load_model(path)

        loaded_layer = [
            l for l in reloaded.layers if isinstance(l, AttentionRoutingCapsule)
        ][0]

        assert np.allclose(reloaded(input_tensor).numpy(), ref_out, atol=1e-5)

        if spec is None:
            assert loaded_layer.kernel_regularizer is None
            return

        assert isinstance(
            loaded_layer.kernel_regularizer, keras.regularizers.Regularizer
        ), (
            "after a .keras round trip kernel_regularizer came back as "
            f"{type(loaded_layer.kernel_regularizer).__name__}"
        )
        assert type(loaded_layer.kernel_regularizer) is type(layer.kernel_regularizer)
        assert (
            loaded_layer.kernel_regularizer.get_config()
            == layer.kernel_regularizer.get_config()
        )

    def test_initializer_survives_a_full_keras_model_round_trip(self, input_tensor):
        inp = keras.Input(shape=(32, 8))
        layer = AttentionRoutingCapsule(
            num_capsules=10, dim_capsules=16, kernel_initializer="he_normal"
        )
        model = keras.Model(inputs=inp, outputs=layer(inp))
        ref_out = model(input_tensor).numpy()

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "arc_init.keras")
            model.save(path)
            reloaded = keras.models.load_model(path)

        loaded_layer = [
            l for l in reloaded.layers if isinstance(l, AttentionRoutingCapsule)
        ][0]
        assert isinstance(
            loaded_layer.kernel_initializer, keras.initializers.HeNormal
        )
        assert isinstance(
            loaded_layer.prob_head.kernel_initializer, keras.initializers.HeNormal
        )
        assert np.allclose(reloaded(input_tensor).numpy(), ref_out, atol=1e-5)

    # -- invariant I5: the get_config() KEY SET is frozen -------------------

    @pytest.mark.parametrize(
        "spec", ["l2", keras.regularizers.L2(0.01), None], ids=["str", "obj", "none"]
    )
    def test_get_config_key_set_is_unchanged(self, spec):
        layer = AttentionRoutingCapsule(
            num_capsules=4,
            dim_capsules=6,
            kernel_initializer="he_normal",
            kernel_regularizer=spec,
        )
        keys = set(layer.get_config().keys())
        missing = self._EXPECTED_OWN_CONFIG_KEYS - keys
        assert not missing, f"get_config() lost keys: {sorted(missing)}"
        # Only base-Layer keys may be present beyond this layer's own set.
        extra = keys - self._EXPECTED_OWN_CONFIG_KEYS
        assert extra <= {"name", "trainable", "dtype"}, (
            f"get_config() gained unexpected keys: {sorted(extra)}"
        )

    # -- the identical twin one class down ---------------------------------

    @pytest.mark.parametrize(
        "spec",
        ["l2", keras.regularizers.L1L2(l1=0.01, l2=0.02), None],
        ids=["string_spec", "object_spec", "none"],
    )
    def test_capsule_block_v2_canonicalizes_its_own_regularizer(self, spec):
        """``CapsuleBlockV2`` carried the identical raw-store defect.

        It has its OWN ``kernel_regularizer`` attribute and its OWN
        ``keras.regularizers.serialize`` call in ``get_config()``, so fixing only
        ``AttentionRoutingCapsule`` would have left the wrapper inconsistent with
        the thing it wraps.
        """
        block = CapsuleBlockV2(
            num_capsules=4, dim_capsules=6, kernel_regularizer=spec
        )
        if spec is None:
            assert block.kernel_regularizer is None
        else:
            assert isinstance(
                block.kernel_regularizer, keras.regularizers.Regularizer
            )
        # The wrapper forwards the RESOLVED object to the routing capsule.
        assert block.routing.kernel_regularizer is block.kernel_regularizer

        recreated = CapsuleBlockV2.from_config(block.get_config())
        if spec is None:
            assert recreated.kernel_regularizer is None
        else:
            assert isinstance(
                recreated.kernel_regularizer, keras.regularizers.Regularizer
            )
            assert (
                recreated.kernel_regularizer.get_config()
                == block.kernel_regularizer.get_config()
            )


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
