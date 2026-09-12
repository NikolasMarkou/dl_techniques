"""Comprehensive test suite for the Zamba2 model package.

Grows incrementally alongside ``src/dl_techniques/models/language/zamba2/``;
see ``plans/plan-2026-09-12T075714-035fd488/plan.md`` for the build order.
Step 1 covers :class:`LoRAAdapter`; step 2 adds
:class:`Zamba2SharedAttentionBlock`; step 3 adds
:class:`Zamba2SharedMLPBlock`.
"""

import os
import tempfile
from typing import Any, Dict

import numpy as np
import pytest
import tensorflow as tf
import keras

from dl_techniques.models.language.zamba2.layers import (
    LoRAAdapter,
    Zamba2SharedAttentionBlock,
    Zamba2SharedMLPBlock,
)


class TestLoRAAdapter:
    """Comprehensive test suite for the LoRAAdapter layer."""

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard configuration for testing."""
        return {
            "output_dim": 32,
            "rank": 4,
            "alpha": 8.0,
            "num_occurrences": 3,
        }

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Sample 3D input (sequence data)."""
        return keras.random.normal(shape=(2, 10, 16))

    def test_initialization(self, layer_config: Dict[str, Any]) -> None:
        """Test layer initialization stores all params."""
        layer = LoRAAdapter(**layer_config)

        assert layer.output_dim == layer_config["output_dim"]
        assert layer.rank == layer_config["rank"]
        assert layer.alpha == layer_config["alpha"]
        assert layer.num_occurrences == layer_config["num_occurrences"]
        assert layer.scale == pytest.approx(
            layer_config["alpha"] / layer_config["rank"]
        )
        assert not layer.built
        assert layer.a is None
        assert layer.b is None

    def test_edge_cases(self) -> None:
        """Every positional hyperparameter must be validated as positive."""
        with pytest.raises(ValueError, match="output_dim must be positive"):
            LoRAAdapter(output_dim=0, rank=4, alpha=8.0, num_occurrences=2)

        with pytest.raises(ValueError, match="rank must be positive"):
            LoRAAdapter(output_dim=32, rank=0, alpha=8.0, num_occurrences=2)

        with pytest.raises(ValueError, match="alpha must be positive"):
            LoRAAdapter(output_dim=32, rank=4, alpha=0.0, num_occurrences=2)

        with pytest.raises(ValueError, match="num_occurrences must be positive"):
            LoRAAdapter(output_dim=32, rank=4, alpha=8.0, num_occurrences=0)

    def test_forward_pass_shape_and_zero_init(
        self, layer_config: Dict[str, Any], sample_input: keras.KerasTensor
    ) -> None:
        """Output shape is correct; at construction every delta is exactly zero.

        ``B`` is zero-initialized (standard LoRA convention), so every
        occurrence's delta must be exactly zero before any training step.
        """
        layer = LoRAAdapter(**layer_config)

        for occurrence_idx in range(layer_config["num_occurrences"]):
            output = layer(sample_input, occurrence_idx=occurrence_idx)
            assert output.shape == (*sample_input.shape[:-1], layer_config["output_dim"])

            output_numpy = keras.ops.convert_to_numpy(output)
            assert np.isfinite(output_numpy).all()
            np.testing.assert_allclose(
                output_numpy, np.zeros_like(output_numpy), rtol=0, atol=0,
                err_msg="Delta must be exactly zero before training (B is zero-init)",
            )

        assert layer.built
        assert layer.a.shape == (
            layer_config["num_occurrences"], sample_input.shape[-1], layer_config["rank"]
        )
        assert layer.b.shape == (
            layer_config["num_occurrences"], layer_config["rank"], layer_config["output_dim"]
        )

    def test_occurrence_idx_out_of_range_raises(
        self, layer_config: Dict[str, Any], sample_input: keras.KerasTensor
    ) -> None:
        """call() must reject an occurrence index outside [0, num_occurrences)."""
        layer = LoRAAdapter(**layer_config)

        with pytest.raises(ValueError, match="occurrence_idx must be in"):
            layer(sample_input, occurrence_idx=layer_config["num_occurrences"])

        with pytest.raises(ValueError, match="occurrence_idx must be in"):
            layer(sample_input, occurrence_idx=-1)

    def test_different_occurrences_differ_after_training_step(
        self, layer_config: Dict[str, Any], sample_input: keras.KerasTensor
    ) -> None:
        """Anti-vacuity: two occurrence indices must produce DIFFERENT deltas.

        At construction every delta is identically zero (by design, see
        test_forward_pass_shape_and_zero_init), so this guard first takes one
        optimizer step to move B away from zero, THEN asserts the two
        occurrences' outputs differ on identical input -- proving the
        per-occurrence A/B pairs are independently parameterized, not the
        same pair selected twice by coincidence.
        """
        layer = LoRAAdapter(**layer_config)
        optimizer = keras.optimizers.Adam(learning_rate=1e-1)

        # One training step per occurrence actually exercised, with a
        # different target each time so the occurrences diverge from
        # each other, not just away from zero in lockstep.
        targets = {
            0: keras.ops.ones((2, 10, layer_config["output_dim"])),
            1: keras.ops.ones((2, 10, layer_config["output_dim"])) * -1.0,
        }
        for occurrence_idx, target in targets.items():
            with tf.GradientTape() as tape:
                output = layer(sample_input, occurrence_idx=occurrence_idx)
                loss = keras.ops.mean(keras.ops.square(output - target))
            grads = tape.gradient(loss, layer.trainable_variables)
            optimizer.apply_gradients(zip(grads, layer.trainable_variables))

        output_0 = keras.ops.convert_to_numpy(layer(sample_input, occurrence_idx=0))
        output_1 = keras.ops.convert_to_numpy(layer(sample_input, occurrence_idx=1))

        assert not np.allclose(output_0, output_1), (
            "Two different occurrence_idx values produced identical output "
            "after diverging training steps -- LoRA pairs are not independent"
        )

    def test_gradients_flow_to_every_exercised_occurrence(
        self, layer_config: Dict[str, Any], sample_input: keras.KerasTensor
    ) -> None:
        """Gradients reach both A and B for every occurrence actually called.

        An occurrence never selected in this call sequence (index 2 here)
        must NOT receive a nonzero gradient -- proving gradients are scoped
        to the exercised slice, not broadcast across the whole stacked
        weight by the indexing op.

        ``B`` starts at zero (standard LoRA init), so ``dL/dA`` is
        identically zero at construction -- the chain rule through
        ``delta = (x @ A) @ B`` routes zero back onto ``A`` whenever ``B``
        is exactly zero, which is mathematically correct, not a defect. This
        guard first takes one optimizer step (as in
        test_different_occurrences_differ_after_training_step) to move ``B``
        away from zero, then checks both ``A`` and ``B``.
        """
        layer = LoRAAdapter(**layer_config)
        exercised = (0, 1)

        optimizer = keras.optimizers.Adam(learning_rate=1e-1)
        with tf.GradientTape() as warmup_tape:
            warmup_loss = 0.0
            for occurrence_idx in exercised:
                warmup_output = layer(sample_input, occurrence_idx=occurrence_idx)
                warmup_loss = warmup_loss + keras.ops.mean(keras.ops.square(warmup_output - 1.0))
        warmup_grads = warmup_tape.gradient(warmup_loss, layer.trainable_variables)
        optimizer.apply_gradients(zip(warmup_grads, layer.trainable_variables))

        with tf.GradientTape() as tape:
            total_loss = 0.0
            for occurrence_idx in exercised:
                output = layer(sample_input, occurrence_idx=occurrence_idx)
                total_loss = total_loss + keras.ops.mean(keras.ops.square(output))
        grads = tape.gradient(total_loss, [layer.a, layer.b])

        assert grads[0] is not None
        assert grads[1] is not None
        a_grad = keras.ops.convert_to_numpy(grads[0])
        b_grad = keras.ops.convert_to_numpy(grads[1])

        for occurrence_idx in exercised:
            assert not np.allclose(a_grad[occurrence_idx], 0.0), (
                f"Occurrence {occurrence_idx} (exercised) has a zero gradient on A"
            )
            # B is zero-initialized; its gradient w.r.t. a squared-error-style
            # loss is ``2 * mean(output) * (A_i @ x)``-shaped and nonzero
            # whenever the upstream activation is nonzero, which holds here
            # since A is randomly initialized and the input is nonzero.
            assert not np.allclose(b_grad[occurrence_idx], 0.0), (
                f"Occurrence {occurrence_idx} (exercised) has a zero gradient on B"
            )

        unexercised_idx = 2
        assert np.allclose(a_grad[unexercised_idx], 0.0), (
            f"Occurrence {unexercised_idx} (never called) got a nonzero gradient on A"
        )
        assert np.allclose(b_grad[unexercised_idx], 0.0), (
            f"Occurrence {unexercised_idx} (never called) got a nonzero gradient on B"
        )

    def test_serialization_cycle(
        self, layer_config: Dict[str, Any], sample_input: keras.KerasTensor
    ) -> None:
        """CRITICAL: full .keras serialization cycle with prediction comparison.

        The layer is trained one step first so B is away from zero --
        comparing two all-zero tensors would pass regardless of whether
        serialization actually preserved the weights.
        """
        inputs = keras.Input(shape=sample_input.shape[1:])
        layer = LoRAAdapter(**layer_config)
        outputs = layer(inputs, occurrence_idx=1)
        model = keras.Model(inputs, outputs)

        optimizer = keras.optimizers.Adam(learning_rate=1e-1)
        with tf.GradientTape() as tape:
            pred = model(sample_input)
            loss = keras.ops.mean(keras.ops.square(pred - 1.0))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        original_prediction = model(sample_input)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_model.keras")
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_prediction = loaded_model(sample_input)

            reloaded_layer = next(
                lyr for lyr in loaded_model.layers if isinstance(lyr, LoRAAdapter)
            )
            assert reloaded_layer.output_dim == layer_config["output_dim"]
            assert reloaded_layer.rank == layer_config["rank"]
            assert reloaded_layer.alpha == layer_config["alpha"]
            assert reloaded_layer.num_occurrences == layer_config["num_occurrences"]

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_prediction),
                keras.ops.convert_to_numpy(loaded_prediction),
                rtol=0, atol=1e-6,
                err_msg="Predictions differ after serialization",
            )

    def test_config_completeness(self, layer_config: Dict[str, Any]) -> None:
        """get_config() must contain every __init__ param."""
        layer = LoRAAdapter(**layer_config)
        config = layer.get_config()

        required_keys = {"output_dim", "rank", "alpha", "num_occurrences", "kernel_initializer"}
        for key in required_keys:
            assert key in config, f"Missing {key} in get_config()"

        assert config["output_dim"] == layer_config["output_dim"]
        assert config["rank"] == layer_config["rank"]
        assert config["alpha"] == layer_config["alpha"]
        assert config["num_occurrences"] == layer_config["num_occurrences"]


class TestZamba2SharedAttentionBlock:
    """Comprehensive test suite for the Zamba2SharedAttentionBlock layer."""

    @pytest.fixture
    def block_config(self) -> Dict[str, Any]:
        """Standard configuration for testing."""
        return {
            "d_model": 32,
            "num_heads": 4,
            "max_seq_len": 64,
        }

    @pytest.fixture
    def sample_inputs(self) -> Dict[str, keras.KerasTensor]:
        """Sample hidden_state / original_embedding pair."""
        return {
            "hidden_state": keras.random.normal(shape=(2, 12, 32)),
            "original_embedding": keras.random.normal(shape=(2, 12, 32)),
        }

    def test_initialization(self, block_config: Dict[str, Any]) -> None:
        """Test layer initialization stores all params and builds sub-layers."""
        block = Zamba2SharedAttentionBlock(**block_config)

        assert block.d_model == block_config["d_model"]
        assert block.num_heads == block_config["num_heads"]
        assert block.head_dim == block_config["d_model"] // block_config["num_heads"]
        assert block.max_seq_len == block_config["max_seq_len"]
        assert not block.built
        assert isinstance(block.norm, keras.layers.Layer)
        assert isinstance(block.input_proj, keras.layers.Dense)
        assert isinstance(block.rope, keras.layers.Layer)
        assert isinstance(block.attention, keras.layers.Layer)

    def test_edge_cases(self) -> None:
        """Every positional hyperparameter must be validated."""
        with pytest.raises(ValueError, match="d_model must be positive"):
            Zamba2SharedAttentionBlock(d_model=0, num_heads=4, max_seq_len=64)

        with pytest.raises(ValueError, match="num_heads must be positive"):
            Zamba2SharedAttentionBlock(d_model=32, num_heads=0, max_seq_len=64)

        with pytest.raises(ValueError, match="must be divisible by num_heads"):
            Zamba2SharedAttentionBlock(d_model=32, num_heads=5, max_seq_len=64)

        with pytest.raises(ValueError, match="max_seq_len must be positive"):
            Zamba2SharedAttentionBlock(d_model=32, num_heads=4, max_seq_len=0)

        with pytest.raises(ValueError, match="norm_epsilon must be positive"):
            Zamba2SharedAttentionBlock(d_model=32, num_heads=4, max_seq_len=64, norm_epsilon=0.0)

    def test_forward_pass_shape_and_finiteness(
        self, block_config: Dict[str, Any], sample_inputs: Dict[str, keras.KerasTensor]
    ) -> None:
        """Output shape matches hidden_state's shape and contains no NaN/Inf."""
        block = Zamba2SharedAttentionBlock(**block_config)
        output = block(sample_inputs["hidden_state"], sample_inputs["original_embedding"])

        assert output.shape == sample_inputs["hidden_state"].shape
        output_numpy = keras.ops.convert_to_numpy(output)
        assert np.isfinite(output_numpy).all()
        assert block.built

    def test_same_instance_called_twice_shares_weight_objects(
        self, block_config: Dict[str, Any], sample_inputs: Dict[str, keras.KerasTensor]
    ) -> None:
        """The same block instance, called at two different depths, shares
        identical weight ``Variable`` objects (``is``-level identity).

        This is the mechanical heart of Zamba2's mem-block reuse -- see
        ``test_shared_weights_identical_across_depth.py`` for the dedicated
        sentence-named guard; this is a lighter in-suite check alongside the
        rest of the comprehensive coverage.
        """
        block = Zamba2SharedAttentionBlock(**block_config)

        _ = block(sample_inputs["hidden_state"], sample_inputs["original_embedding"])
        weights_at_first_call = list(block.weights)

        _ = block(sample_inputs["hidden_state"], sample_inputs["original_embedding"])
        weights_at_second_call = list(block.weights)

        assert len(weights_at_first_call) == len(weights_at_second_call)
        for w1, w2 in zip(weights_at_first_call, weights_at_second_call):
            assert w1 is w2, "Calling the same instance twice must not create new weights"

    def test_gradient_flow(
        self, block_config: Dict[str, Any], sample_inputs: Dict[str, keras.KerasTensor]
    ) -> None:
        """Gradients reach every trainable weight."""
        block = Zamba2SharedAttentionBlock(**block_config)

        with tf.GradientTape() as tape:
            output = block(sample_inputs["hidden_state"], sample_inputs["original_embedding"])
            loss = keras.ops.mean(keras.ops.square(output))
        grads = tape.gradient(loss, block.trainable_variables)

        assert len(grads) == len(block.trainable_variables)
        assert len(grads) > 0
        for grad, variable in zip(grads, block.trainable_variables):
            assert grad is not None, f"No gradient for {variable.name}"

    def test_serialization_cycle(
        self, block_config: Dict[str, Any], sample_inputs: Dict[str, keras.KerasTensor]
    ) -> None:
        """Full .keras serialization cycle with prediction comparison."""
        hidden_input = keras.Input(shape=sample_inputs["hidden_state"].shape[1:])
        embedding_input = keras.Input(shape=sample_inputs["original_embedding"].shape[1:])
        layer = Zamba2SharedAttentionBlock(**block_config)
        outputs = layer(hidden_input, embedding_input)
        model = keras.Model([hidden_input, embedding_input], outputs)

        original_prediction = model(
            [sample_inputs["hidden_state"], sample_inputs["original_embedding"]]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_model.keras")
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_prediction = loaded_model(
                [sample_inputs["hidden_state"], sample_inputs["original_embedding"]]
            )

            reloaded_layer = next(
                lyr for lyr in loaded_model.layers if isinstance(lyr, Zamba2SharedAttentionBlock)
            )
            assert reloaded_layer.d_model == block_config["d_model"]
            assert reloaded_layer.num_heads == block_config["num_heads"]
            assert reloaded_layer.max_seq_len == block_config["max_seq_len"]

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_prediction),
                keras.ops.convert_to_numpy(loaded_prediction),
                rtol=0, atol=1e-5,
                err_msg="Predictions differ after serialization",
            )

    def test_config_completeness(self, block_config: Dict[str, Any]) -> None:
        """get_config() must contain every __init__ param."""
        layer = Zamba2SharedAttentionBlock(**block_config)
        config = layer.get_config()

        required_keys = {
            "d_model", "num_heads", "max_seq_len", "rope_theta", "rope_percentage",
            "attention_dropout_rate", "norm_epsilon", "use_bias", "kernel_initializer",
        }
        for key in required_keys:
            assert key in config, f"Missing {key} in get_config()"

        assert config["d_model"] == block_config["d_model"]
        assert config["num_heads"] == block_config["num_heads"]
        assert config["max_seq_len"] == block_config["max_seq_len"]

    @pytest.mark.parametrize("dtype_policy", ["float32", "mixed_float16"])
    def test_mixed_float16_no_nan(
        self,
        block_config: Dict[str, Any],
        sample_inputs: Dict[str, keras.KerasTensor],
        dtype_policy: str,
    ) -> None:
        """The causal-masked attention path must not produce NaN under
        ``mixed_float16`` -- the repo-wide fp16 additive-mask NaN trap.
        """
        original_policy = keras.mixed_precision.global_policy()
        try:
            keras.mixed_precision.set_global_policy(dtype_policy)
            block = Zamba2SharedAttentionBlock(**block_config)
            hidden = keras.ops.cast(sample_inputs["hidden_state"], block.compute_dtype)
            embedding = keras.ops.cast(sample_inputs["original_embedding"], block.compute_dtype)
            output = block(hidden, embedding)
            output_numpy = keras.ops.convert_to_numpy(output)
            assert np.isfinite(output_numpy).all(), f"NaN/Inf under {dtype_policy}"
        finally:
            keras.mixed_precision.set_global_policy(original_policy)


class TestZamba2SharedMLPBlock:
    """Comprehensive test suite for the Zamba2SharedMLPBlock layer."""

    @pytest.fixture
    def block_config(self) -> Dict[str, Any]:
        """Standard configuration for testing."""
        return {
            "d_model": 32,
            "num_occurrences": 3,
            "hidden_dim": 64,
            "lora_rank": 4,
            "lora_alpha": 8.0,
        }

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Sample 3D hidden-state input."""
        return keras.random.normal(shape=(2, 12, 32))

    def test_initialization(self, block_config: Dict[str, Any]) -> None:
        """Test layer initialization stores all params and builds sub-layers."""
        block = Zamba2SharedMLPBlock(**block_config)

        assert block.d_model == block_config["d_model"]
        assert block.num_occurrences == block_config["num_occurrences"]
        assert block.hidden_dim == block_config["hidden_dim"]
        assert not block.built
        assert isinstance(block.norm, keras.layers.Layer)
        assert isinstance(block.gate_proj, keras.layers.Dense)
        assert isinstance(block.up_proj, keras.layers.Dense)
        assert isinstance(block.lora, LoRAAdapter)
        assert isinstance(block.down_proj, keras.layers.Dense)
        assert block.lora.num_occurrences == block_config["num_occurrences"]
        assert block.lora.output_dim == block_config["hidden_dim"]

    def test_edge_cases(self) -> None:
        """Every positional hyperparameter must be validated."""
        with pytest.raises(ValueError, match="d_model must be positive"):
            Zamba2SharedMLPBlock(d_model=0, num_occurrences=2)

        with pytest.raises(ValueError, match="num_occurrences must be positive"):
            Zamba2SharedMLPBlock(d_model=32, num_occurrences=0)

        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            Zamba2SharedMLPBlock(d_model=32, num_occurrences=2, hidden_dim=0)

        with pytest.raises(ValueError, match="ffn_expansion_factor must be positive"):
            Zamba2SharedMLPBlock(d_model=32, num_occurrences=2, ffn_expansion_factor=0)

        with pytest.raises(ValueError, match="ffn_multiple_of must be positive"):
            Zamba2SharedMLPBlock(d_model=32, num_occurrences=2, ffn_multiple_of=0)

        with pytest.raises(ValueError, match="lora_rank must be positive"):
            Zamba2SharedMLPBlock(d_model=32, num_occurrences=2, lora_rank=0)

        with pytest.raises(ValueError, match="lora_alpha must be positive"):
            Zamba2SharedMLPBlock(d_model=32, num_occurrences=2, lora_alpha=0.0)

        with pytest.raises(ValueError, match="norm_epsilon must be positive"):
            Zamba2SharedMLPBlock(d_model=32, num_occurrences=2, norm_epsilon=0.0)

    def test_forward_pass_shape_and_finiteness(
        self, block_config: Dict[str, Any], sample_input: keras.KerasTensor
    ) -> None:
        """Output shape matches the input's shape and contains no NaN/Inf."""
        block = Zamba2SharedMLPBlock(**block_config)
        output = block(sample_input, occurrence_idx=0)

        assert output.shape == sample_input.shape
        output_numpy = keras.ops.convert_to_numpy(output)
        assert np.isfinite(output_numpy).all()
        assert block.built

    def test_same_instance_called_twice_shares_weight_objects(
        self, block_config: Dict[str, Any], sample_input: keras.KerasTensor
    ) -> None:
        """The same block instance, called at two different depths (here,
        twice with the same ``occurrence_idx``), shares identical weight
        ``Variable`` objects (``is``-level identity).

        See ``test_lora_differs_per_occurrence.py`` for the decisive guard
        that this identity holds EVEN ACROSS two different occurrence
        indices, while the LoRA delta itself still diverges.
        """
        block = Zamba2SharedMLPBlock(**block_config)

        _ = block(sample_input, occurrence_idx=0)
        weights_at_first_call = list(block.weights)

        _ = block(sample_input, occurrence_idx=0)
        weights_at_second_call = list(block.weights)

        assert len(weights_at_first_call) == len(weights_at_second_call)
        for w1, w2 in zip(weights_at_first_call, weights_at_second_call):
            assert w1 is w2, "Calling the same instance twice must not create new weights"

    def test_gradient_flow(
        self, block_config: Dict[str, Any], sample_input: keras.KerasTensor
    ) -> None:
        """Gradients reach every trainable weight, including the owned
        LoRAAdapter's ``A``/``B`` for the exercised occurrence."""
        block = Zamba2SharedMLPBlock(**block_config)

        with tf.GradientTape() as tape:
            output = block(sample_input, occurrence_idx=0)
            loss = keras.ops.mean(keras.ops.square(output))
        grads = tape.gradient(loss, block.trainable_variables)

        assert len(grads) == len(block.trainable_variables)
        assert len(grads) > 0
        for grad, variable in zip(grads, block.trainable_variables):
            assert grad is not None, f"No gradient for {variable.name}"

    def test_serialization_cycle(
        self, block_config: Dict[str, Any], sample_input: keras.KerasTensor
    ) -> None:
        """Full .keras serialization cycle with prediction comparison."""
        inputs = keras.Input(shape=sample_input.shape[1:])
        layer = Zamba2SharedMLPBlock(**block_config)
        outputs = layer(inputs, occurrence_idx=1)
        model = keras.Model(inputs, outputs)

        original_prediction = model(sample_input)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_model.keras")
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_prediction = loaded_model(sample_input)

            reloaded_layer = next(
                lyr for lyr in loaded_model.layers if isinstance(lyr, Zamba2SharedMLPBlock)
            )
            assert reloaded_layer.d_model == block_config["d_model"]
            assert reloaded_layer.num_occurrences == block_config["num_occurrences"]
            assert reloaded_layer.hidden_dim == block_config["hidden_dim"]

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_prediction),
                keras.ops.convert_to_numpy(loaded_prediction),
                rtol=0, atol=1e-5,
                err_msg="Predictions differ after serialization",
            )

    def test_config_completeness(self, block_config: Dict[str, Any]) -> None:
        """get_config() must contain every __init__ param."""
        layer = Zamba2SharedMLPBlock(**block_config)
        config = layer.get_config()

        required_keys = {
            "d_model", "num_occurrences", "hidden_dim", "ffn_expansion_factor",
            "ffn_multiple_of", "lora_rank", "lora_alpha", "norm_epsilon",
            "use_bias", "kernel_initializer",
        }
        for key in required_keys:
            assert key in config, f"Missing {key} in get_config()"

        assert config["d_model"] == block_config["d_model"]
        assert config["num_occurrences"] == block_config["num_occurrences"]
        assert config["hidden_dim"] == block_config["hidden_dim"]
        assert config["lora_rank"] == block_config["lora_rank"]
        assert config["lora_alpha"] == block_config["lora_alpha"]

    def test_hidden_dim_derived_when_not_given(self) -> None:
        """``hidden_dim=None`` derives via the 2/3 rule, matching
        ``SwiGLUFFN``'s own arithmetic."""
        block = Zamba2SharedMLPBlock(d_model=768, num_occurrences=2)
        raw = int(768 * 4 * 2 / 3)
        expected = 256 * ((raw + 256 - 1) // 256)
        assert block.hidden_dim == expected

    @pytest.mark.parametrize("dtype_policy", ["float32", "mixed_float16"])
    def test_mixed_float16_no_nan(
        self,
        block_config: Dict[str, Any],
        sample_input: keras.KerasTensor,
        dtype_policy: str,
    ) -> None:
        """The gated-SiLU + additive-LoRA path must not produce NaN/Inf
        under ``mixed_float16``."""
        original_policy = keras.mixed_precision.global_policy()
        try:
            keras.mixed_precision.set_global_policy(dtype_policy)
            block = Zamba2SharedMLPBlock(**block_config)
            hidden = keras.ops.cast(sample_input, block.compute_dtype)
            output = block(hidden, occurrence_idx=0)
            output_numpy = keras.ops.convert_to_numpy(output)
            assert np.isfinite(output_numpy).all(), f"NaN/Inf under {dtype_policy}"
        finally:
            keras.mixed_precision.set_global_policy(original_policy)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
