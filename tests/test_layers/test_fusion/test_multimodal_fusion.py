import pytest
import keras
from keras import ops
import numpy as np
import tensorflow as tf
import tempfile
import os
from typing import List

from dl_techniques.layers.fusion.multimodal_fusion import MultiModalFusion, FusionStrategy

SINGLE_OUTPUT_STRATEGIES: List[FusionStrategy] = [
    'concatenation',
    'addition',
    'multiplication',
    'gated',
    'attention_pooling',
    'bilinear',
    'tensor_fusion'
]


@keras.saving.register_keras_serializable(package="test_custom")
class _CustomActivationLayer(keras.layers.Layer):
    """A minimal custom (non-``keras.layers``) activation Layer, for D-004.

    Deliberately registered under its own ``test_custom`` package rather than
    left unregistered, and never decorated with the project's own
    ``@register_dl_technique`` -- this class exists purely to prove
    ``from_config``'s dispatch predicate does not depend on WHERE the class
    is registered. See ``test_custom_layer_activation_round_trips``.
    """

    def __init__(self, scale: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale

    def call(self, x):
        return keras.ops.relu(x) * self.scale

    def get_config(self):
        config = super().get_config()
        config.update({'scale': self.scale})
        return config


class TestMultiModalFusion:
    """Comprehensive test suite for the MultiModalFusion layer."""

    @pytest.fixture
    def dim(self) -> int:
        """Provides a standard dimension for testing."""
        return 64

    @pytest.fixture
    def sample_input(self, dim: int) -> List[keras.KerasTensor]:
        """Provides a sample multi-modal input (list of 2 tensors)."""
        batch_size, seq_len = 4, 16
        return [
            keras.random.normal(shape=(batch_size, seq_len, dim)),
            keras.random.normal(shape=(batch_size, seq_len, dim)),
        ]

    def test_initialization(self, dim: int):
        """Test layer initialization with default parameters."""
        layer = MultiModalFusion(dim=dim, fusion_strategy='concatenation')
        assert layer.dim == dim
        assert layer.fusion_strategy == 'concatenation'
        assert not layer.built

    @pytest.mark.parametrize("strategy", SINGLE_OUTPUT_STRATEGIES)
    def test_forward_pass_single_output(self, strategy: FusionStrategy, sample_input: List[keras.KerasTensor],
                                        dim: int):
        """Test forward pass and building for single-output strategies."""
        layer = MultiModalFusion(dim=dim, fusion_strategy=strategy)
        output = layer(sample_input)

        assert layer.built, f"Layer should be built after forward pass for strategy '{strategy}'"

        batch_size, seq_len = ops.shape(sample_input[0])[:2]

        if strategy == 'attention_pooling':
            # Special case: attention pooling reduces the sequence dimension
            expected_shape = (batch_size, dim)
        else:
            expected_shape = (batch_size, seq_len, dim)

        assert output.shape == expected_shape, f"Output shape mismatch for strategy '{strategy}'"

    def test_forward_pass_cross_attention(self, sample_input: List[keras.KerasTensor], dim: int):
        """Test forward pass for the cross-attention strategy which returns multiple outputs."""
        layer = MultiModalFusion(
            dim=dim,
            fusion_strategy='cross_attention',
            num_fusion_layers=2  # Test iterative fusion
        )
        outputs = layer(sample_input)

        assert layer.built
        assert isinstance(outputs, tuple)
        assert len(outputs) == len(sample_input)

        for i, out_tensor in enumerate(outputs):
            assert out_tensor.shape == sample_input[i].shape

    @pytest.mark.parametrize("strategy", SINGLE_OUTPUT_STRATEGIES)
    def test_serialization_cycle_single_output(self, strategy: FusionStrategy, sample_input: List[keras.KerasTensor],
                                               dim: int):
        """CRITICAL TEST: Full serialization cycle for single-output strategies."""
        layer_instance = MultiModalFusion(dim=dim, fusion_strategy=strategy)

        # Create a model with the custom layer
        inputs = [keras.Input(shape=s.shape[1:]) for s in sample_input]
        outputs = layer_instance(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)

        # Get original prediction
        original_pred = model(sample_input)

        # Save and load the model
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, f'test_model_{strategy}.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_input)

            # Verify identical predictions
            np.testing.assert_allclose(
                ops.convert_to_numpy(original_pred),
                ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg=f"Predictions differ after serialization for strategy '{strategy}'"
            )

    def test_serialization_cycle_cross_attention(self, sample_input: List[keras.KerasTensor], dim: int):
        """CRITICAL TEST: Full serialization cycle for cross-attention strategy."""
        layer_instance = MultiModalFusion(dim=dim, fusion_strategy='cross_attention')

        inputs = [keras.Input(shape=s.shape[1:]) for s in sample_input]
        outputs = layer_instance(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)

        original_preds = model(sample_input)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model_cross_attention.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_preds = loaded_model(sample_input)

            # Verify identical predictions for each output
            for orig, loaded in zip(original_preds, loaded_preds):
                np.testing.assert_allclose(
                    ops.convert_to_numpy(orig),
                    ops.convert_to_numpy(loaded),
                    rtol=1e-6, atol=1e-6,
                    err_msg="Predictions differ after serialization for cross-attention strategy"
                )

    def test_config_completeness(self, dim: int):
        """Test that get_config contains all __init__ parameters."""
        config_params = {
            'dim': dim,
            'fusion_strategy': 'cross_attention',
            'num_fusion_layers': 2,
            'attention_config': {'num_heads': 4, 'dropout_rate': 0.2},
            'ffn_type': 'mlp',
            'ffn_config': {'hidden_dim': dim * 2},
            'norm_type': 'layer_norm',
            'norm_config': {'epsilon': 1e-6},
            'dropout_rate': 0.15,
            'use_residual': False,
        }
        layer = MultiModalFusion(**config_params)
        config = layer.get_config()

        for key, value in config_params.items():
            assert key in config, f"Missing '{key}' in get_config()"
            if isinstance(value, dict):
                assert config[key] == value, f"Config mismatch for nested dict '{key}'"

        assert config['attention_config']['num_heads'] == 4
        assert config['ffn_config']['hidden_dim'] == dim * 2

    def test_layer_instance_activation_round_trips(
        self, sample_input: List[keras.KerasTensor], dim: int
    ):
        """A Layer-instance activation must survive get_config/from_config.

        D-011 (plan-2026-09-15T034909-a7edc8da): before the fix,
        ``from_config`` unconditionally routed the serialized activation
        through ``keras.activations.deserialize``, which silently returns the
        bare class-name STRING for a Layer-shaped dict instead of
        reconstructing the Layer -- a subsequent forward pass would then try
        to call a string, or ``keras.activations.get`` on that string would
        raise. This guards against that regressing.
        """
        layer = MultiModalFusion(
            dim=dim,
            fusion_strategy='concatenation',
            activation=keras.layers.LeakyReLU(negative_slope=0.2),
        )
        config = layer.get_config()
        assert isinstance(config['activation'], dict)
        assert config['activation'].get('module') == 'keras.layers'

        rebuilt = MultiModalFusion.from_config(config)
        assert isinstance(rebuilt.activation, keras.layers.LeakyReLU)
        assert rebuilt.activation.negative_slope == pytest.approx(0.2)

        # And the rebuilt layer actually runs.
        output = rebuilt(sample_input)
        assert output.shape == sample_input[0].shape

    def test_unregistered_custom_function_activation_round_trips_in_a_scope(
        self, sample_input: List[keras.KerasTensor], dim: int
    ):
        """An UNREGISTERED custom activation function round-trips IN a scope.

        MEASURED (completion-fix step 13.1, plan-2026-09-15T094955-31fbe3db):
        an unregistered plain Python function serializes to a
        ``{'module': 'builtins', 'class_name': 'function', ...}`` dict that
        carries no resolvable module path -- Keras can only reconstruct it
        given an explicit name->object mapping, either
        ``keras.saving.custom_object_scope`` (used here) or
        ``keras.models.load_model(..., custom_objects=...)``. This is a
        documented Keras contract (see
        ``utils/activation_serialization.py``'s own module-docstring table),
        not something this class can work around -- so this test exercises
        the REALISTIC, supported round trip, which both the pre-fix (D-005)
        and post-fix code already pass (this call path was never the
        regression; see the sibling test below for what was).
        """
        def my_custom_activation(x):
            return x * 2.0

        layer = MultiModalFusion(
            dim=dim,
            fusion_strategy='concatenation',
            activation=my_custom_activation,
        )
        config = layer.get_config()
        assert isinstance(config['activation'], dict)
        assert config['activation'].get('module') != 'keras.layers'

        with keras.saving.custom_object_scope({'my_custom_activation': my_custom_activation}):
            rebuilt = MultiModalFusion.from_config(config)
        assert rebuilt.activation is my_custom_activation

        # And the rebuilt layer actually runs (weights differ from `layer`'s
        # own random init, so only shape -- not value -- is compared here,
        # matching test_layer_instance_activation_round_trips's own strength).
        output = rebuilt(sample_input)
        assert output.shape == sample_input[0].shape

    def test_unregistered_custom_function_activation_fails_clearly_without_a_scope(
        self, dim: int
    ):
        """Outside any custom_objects scope, the failure must be a clear ValueError.

        RED-PROOF (completion-fix step 13.1, plan-2026-09-15T094955-31fbe3db):
        commit 8fa35229f (D-005) routed a non-Layer-shaped activation dict
        through ``deserialize_activation(..., allow_layer=True)``
        unconditionally, which -- for an unregistered custom FUNCTION dict
        with no active ``custom_objects`` mapping -- raises a confusing
        internal ``TypeError: Could not locate function '<name>'`` straight
        out of ``keras.saving.serialization_lib``. This test fails against
        that pre-fix code (wrong exception TYPE: ``TypeError``, not
        ``ValueError``) and passes against the fix, which restores this
        class's original, pre-D-011 dispatch for the function-dict branch
        (``keras.activations.deserialize`` in ``from_config`` feeding
        ``keras.activations.get()`` in ``__init__``), raising the same
        ``ValueError: Could not interpret activation function identifier``
        this class always raised for this exact unsupported case -- both
        before D-011 ever existed and after this fix. This is NOT a claim
        that the bare round trip now succeeds (it structurally cannot,
        without a custom_objects mapping); it is a claim that the failure
        mode is restored to a clear, expected exception rather than a
        Keras-internal deserialization error.
        """
        def my_custom_activation(x):
            return x * 2.0

        layer = MultiModalFusion(
            dim=dim,
            fusion_strategy='concatenation',
            activation=my_custom_activation,
        )
        config = layer.get_config()

        with pytest.raises(ValueError, match="Could not interpret activation"):
            MultiModalFusion.from_config(config)

    def test_custom_layer_activation_round_trips(
        self, sample_input: List[keras.KerasTensor], dim: int
    ):
        """A CUSTOM (non-``keras.layers``) Layer-instance activation round-trips.

        D-004 (plan-2026-09-15T135450-e083ae85): before this fix,
        ``from_config`` dispatched on
        ``activation_config.get('module') == 'keras.layers'``, which only
        matches a BUILT-IN ``keras.layers.Layer`` (see
        ``test_layer_instance_activation_round_trips`` above, using
        ``LeakyReLU``). A custom Layer subclass registered under its own
        package serializes with a *different* ``module`` value (MEASURED:
        ``None`` for this locally-defined, ``register_keras_serializable``-
        decorated class, via ``keras.saving.serialize_keras_object`` --
        never the literal string ``'keras.layers'``), so the OLD predicate
        would misroute it into the function-deserialization branch below and
        fail. This test proves the fix's structural predicate (dispatch on
        whether the serialized dict's ``'config'`` value is itself a dict,
        not on which package registered the class) reconstructs a custom
        Layer correctly.

        RED-PROOF (manual, one-off interpreter check, not shipped as a
        mutation test): for this exact serialized dict,
        ``serialized.get('module') == 'keras.layers'`` evaluates ``False``
        (the OLD predicate would have routed to the function branch and
        raised/corrupted), while the FIXED predicate
        (``isinstance(serialized, dict) and
        isinstance(serialized.get('config'), dict)``) evaluates ``True``.
        """
        activation_layer = _CustomActivationLayer(scale=2.0)
        layer = MultiModalFusion(
            dim=dim,
            fusion_strategy='concatenation',
            activation=activation_layer,
        )
        config = layer.get_config()
        assert isinstance(config['activation'], dict)
        # The custom class is registered under its OWN package, never under
        # 'keras.layers' -- confirming the old module-string predicate could
        # never have matched this case.
        assert config['activation'].get('module') != 'keras.layers'
        assert isinstance(config['activation'].get('config'), dict)

        rebuilt = MultiModalFusion.from_config(config)
        assert isinstance(rebuilt.activation, _CustomActivationLayer)
        assert rebuilt.activation.scale == pytest.approx(2.0)

        # And the rebuilt layer actually runs.
        output = rebuilt(sample_input)
        assert output.shape == sample_input[0].shape

    def test_keras_roundtrip_bit_for_bit_with_layer_activation(
        self, sample_input: List[keras.KerasTensor], dim: int
    ):
        """.keras save/load round trip with a Layer-valued activation (Item 5a).

        Following `tests/test_layers/test_ffn/test_mlp.py:627-728`'s template:
        a functional `keras.Model` wraps `MultiModalFusion` constructed with
        `activation=keras.layers.LeakyReLU()` and `dropout_rate=0.0` (this
        class's `'concatenation'` strategy builds a `Dropout` layer from
        `dropout_rate`, so 0.0 keeps `call()` deterministic). Every existing
        round-trip test for this class (`test_serialization_cycle_*`) only
        exercises the default string activation, so none of them proves the
        FILE-based `.keras` mechanism itself survives a Layer-valued
        activation, per plan.md Item 5/D-007.

        Expectation: BIT-FOR-BIT equality (`np.testing.assert_array_equal`),
        matching the stronger claim `test_mlp.py`'s template makes over the
        `rtol=1e-6, atol=1e-6` general-purpose serialization smoke tests
        elsewhere in this file.

        RED-proof: see
        `test_keras_roundtrip_detects_weight_perturbation_with_layer_activation`
        below, a permanent sibling proving this comparison has the power to
        detect a real difference.
        """
        inputs = [keras.Input(shape=s.shape[1:]) for s in sample_input]
        outputs = MultiModalFusion(
            dim=dim,
            fusion_strategy='concatenation',
            activation=keras.layers.LeakyReLU(),
            dropout_rate=0.0,
        )(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)

        original_prediction = ops.convert_to_numpy(
            model(sample_input, training=False)
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "model.keras")
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_prediction = ops.convert_to_numpy(
                loaded_model(sample_input, training=False)
            )

        np.testing.assert_array_equal(
            original_prediction,
            loaded_prediction,
            err_msg="Reloaded model's forward pass is not bit-for-bit identical",
        )

    def test_keras_roundtrip_detects_weight_perturbation_with_layer_activation(
        self, sample_input: List[keras.KerasTensor], dim: int
    ):
        """RED-proof for `test_keras_roundtrip_bit_for_bit_with_layer_activation`.

        Repeats the same save/load round trip, then perturbs the reloaded
        `MultiModalFusion` sublayer's output-projection kernel by a known,
        clearly-detectable amount before comparing. Asserts the bit-for-bit
        comparison DOES raise `AssertionError` against the perturbed reload,
        proving the comparison above is not vacuously passing.
        """
        inputs = [keras.Input(shape=s.shape[1:]) for s in sample_input]
        fusion_layer = MultiModalFusion(
            dim=dim,
            fusion_strategy='concatenation',
            activation=keras.layers.LeakyReLU(),
            dropout_rate=0.0,
        )
        outputs = fusion_layer(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)

        original_prediction = ops.convert_to_numpy(
            model(sample_input, training=False)
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "model.keras")
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)

            # Locate the reloaded MultiModalFusion sublayer and perturb one
            # of its trainable weights by a known, clearly-detectable amount.
            loaded_fusion_layer = loaded_model.layers[-1]
            assert isinstance(loaded_fusion_layer, MultiModalFusion)
            perturbed_weight = loaded_fusion_layer.trainable_weights[0]
            perturbed_weight.assign(perturbed_weight + 1.0)

            perturbed_prediction = ops.convert_to_numpy(
                loaded_model(sample_input, training=False)
            )

        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(
                original_prediction,
                perturbed_prediction,
                err_msg="Perturbation should have been detected but was not",
            )

    @pytest.mark.parametrize("strategy", SINGLE_OUTPUT_STRATEGIES)
    def test_gradients_flow_single_output(self, strategy: FusionStrategy, sample_input: List[keras.KerasTensor],
                                          dim: int):
        """Test that gradients can be computed for single-output strategies."""
        layer = MultiModalFusion(dim=dim, fusion_strategy=strategy)

        with tf.GradientTape() as tape:
            tape.watch(sample_input)
            output = layer(sample_input)
            loss = ops.mean(ops.square(output))

        gradients = tape.gradient(loss, layer.trainable_variables)

        assert all(g is not None for g in gradients), f"Gradient is None for strategy '{strategy}'"
        assert len(gradients) > 0, f"No trainable variables found for strategy '{strategy}'"

    def test_gradients_flow_cross_attention(self, sample_input: List[keras.KerasTensor], dim: int):
        """Test gradient computation for cross-attention strategy."""
        layer = MultiModalFusion(dim=dim, fusion_strategy='cross_attention')

        with tf.GradientTape() as tape:
            tape.watch(sample_input)
            outputs = layer(sample_input)
            # Sum losses from all outputs
            loss = sum(ops.mean(ops.square(o)) for o in outputs)

        gradients = tape.gradient(loss, layer.trainable_variables)

        assert all(g is not None for g in gradients)
        assert len(gradients) > 0

    @pytest.mark.parametrize("training", [True, False])
    @pytest.mark.parametrize("strategy", ['cross_attention', 'concatenation', 'gated'])
    def test_training_modes(self, strategy: FusionStrategy, sample_input: List[keras.KerasTensor], dim: int,
                            training: bool):
        """Test behavior in different training modes for strategies with dropout."""
        layer = MultiModalFusion(dim=dim, fusion_strategy=strategy, dropout_rate=0.5)

        # This test just ensures the call doesn't crash
        _ = layer(sample_input, training=training)
        assert True  # If we reach here, the call was successful

    def test_edge_cases_and_errors(self, sample_input, dim):
        """Test for expected error conditions."""
        # Invalid parameters
        with pytest.raises(ValueError):
            MultiModalFusion(dim=0)
        with pytest.raises(ValueError):
            MultiModalFusion(dim=dim, num_fusion_layers=0)
        with pytest.raises(ValueError):
            MultiModalFusion(dim=dim, dropout_rate=1.1)

        # num_fusion_layers > 1 for non-iterative strategy
        with pytest.raises(ValueError):
            MultiModalFusion(dim=dim, fusion_strategy='concatenation', num_fusion_layers=2)

        # Bilinear with != 2 inputs
        with pytest.raises(ValueError, match="Bilinear fusion requires exactly 2 modalities, got 3"):
            layer = MultiModalFusion(dim=dim, fusion_strategy='bilinear')
            three_inputs = sample_input + [keras.random.normal(shape=(4, 16, dim))]
            layer(three_inputs)

        # Input with < 2 modalities
        with pytest.raises(ValueError, match="Expected at least 2 modalities, got 1"):
            layer = MultiModalFusion(dim=dim)
            layer([sample_input[0]])

        # Dimension mismatch
        with pytest.raises(ValueError, match=f"Modality 1 dimension {dim * 2} doesn't match expected dim {dim}"):
            layer = MultiModalFusion(dim=dim)
            mismatched_input = [
                sample_input[0],
                keras.random.normal(shape=(4, 16, dim * 2))
            ]
            # Must build the layer to trigger the shape check
            layer.build([s.shape for s in mismatched_input])