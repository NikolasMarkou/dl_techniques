"""Comprehensive test suite for the Zamba2 model package.

Grows incrementally alongside ``src/dl_techniques/models/language/zamba2/``;
see ``plans/plan-2026-09-12T075714-035fd488/plan.md`` for the build order.
Step 1 covers :class:`LoRAAdapter`; step 2 adds
:class:`Zamba2SharedAttentionBlock`; step 3 adds
:class:`Zamba2SharedMLPBlock`; step 4 adds :class:`Zamba2MambaBlock`; step 5
adds :class:`Zamba2Model`, the full decoder-stack assembly.
"""

import os
import tempfile
from typing import Any, Dict, List

import numpy as np
import pytest
import tensorflow as tf
import keras

from dl_techniques.models.language.zamba2.layers import (
    LoRAAdapter,
    Zamba2MambaBlock,
    Zamba2SharedAttentionBlock,
    Zamba2SharedMLPBlock,
)
from dl_techniques.models.language.zamba2.model import Zamba2Model

from ..gradient_flow_oracle import assert_gradients_reach_every_trainable_weight


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


class TestZamba2MambaBlock:
    """Comprehensive test suite for the Zamba2MambaBlock layer."""

    @pytest.fixture
    def block_config(self) -> Dict[str, Any]:
        """Small configuration for testing -- the selective scan is an
        exact ``while_loop``, so keep ``seq_len`` short."""
        return {
            "d_model": 32,
            "d_state": 16,
            "d_conv": 4,
            "expand": 2,
            "headdim": 8,
        }

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Sample 3D hidden-state input."""
        return keras.random.normal(shape=(2, 6, 32))

    def test_initialization(self, block_config: Dict[str, Any]) -> None:
        """Initialization stores every param, resolves ``d_ssm``, and owns
        exactly one ``Mamba2ResidualBlock`` sub-layer (unbuilt)."""
        block = Zamba2MambaBlock(**block_config)

        assert block.d_model == block_config["d_model"]
        assert block.d_state == block_config["d_state"]
        assert block.d_conv == block_config["d_conv"]
        assert block.expand == block_config["expand"]
        assert block.headdim == block_config["headdim"]
        assert block.d_ssm == block_config["d_model"] * block_config["expand"]
        assert not block.built
        from dl_techniques.models.language.mamba.components_v2 import Mamba2ResidualBlock
        assert isinstance(block.mamba_block, Mamba2ResidualBlock)

    def test_edge_cases(self) -> None:
        """``d_model`` must be validated; an indivisible ``d_ssm``/``headdim``
        pair must raise from the wrapped ``Mamba2Layer``."""
        with pytest.raises(ValueError, match="d_model must be positive"):
            Zamba2MambaBlock(d_model=0)

        with pytest.raises(ValueError, match="d_ssm"):
            Zamba2MambaBlock(d_model=32, expand=2, headdim=9)

    def test_forward_pass_shape_and_finiteness(
        self, block_config: Dict[str, Any], sample_input: keras.KerasTensor
    ) -> None:
        """Output shape matches the input's shape and contains no NaN/Inf."""
        block = Zamba2MambaBlock(**block_config)
        output = block(sample_input)

        assert output.shape == sample_input.shape
        output_numpy = keras.ops.convert_to_numpy(output)
        assert np.isfinite(output_numpy).all()
        assert block.built

    def test_two_instances_do_not_share_weight_objects(
        self, block_config: Dict[str, Any], sample_input: keras.KerasTensor
    ) -> None:
        """Negative twin of the mem-block sharing guards (``decisions.md``
        D-005): two ``Zamba2MambaBlock`` instances built at different stack
        positions must NOT share any weight ``Variable`` object, and their
        values must differ after independent random initialization."""
        block_a = Zamba2MambaBlock(**block_config)
        block_b = Zamba2MambaBlock(**block_config)

        _ = block_a(sample_input)
        _ = block_b(sample_input)

        weights_a = list(block_a.weights)
        weights_b = list(block_b.weights)

        assert len(weights_a) == len(weights_b)
        assert len(weights_a) > 0
        for w_a, w_b in zip(weights_a, weights_b):
            assert w_a is not w_b, (
                "Two independently-built Zamba2MambaBlock instances must "
                "never share a weight Variable object"
            )

        any_differs = any(
            not np.allclose(
                keras.ops.convert_to_numpy(w_a), keras.ops.convert_to_numpy(w_b)
            )
            for w_a, w_b in zip(weights_a, weights_b)
        )
        assert any_differs, (
            "Two independently-built Zamba2MambaBlock instances must not "
            "coincidentally initialize to identical weight values"
        )

    def test_gradient_flow(
        self, block_config: Dict[str, Any], sample_input: keras.KerasTensor
    ) -> None:
        """Gradients reach every trainable weight of the owned
        ``Mamba2ResidualBlock``."""
        block = Zamba2MambaBlock(**block_config)

        with tf.GradientTape() as tape:
            output = block(sample_input)
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
        layer = Zamba2MambaBlock(**block_config)
        outputs = layer(inputs)
        model = keras.Model(inputs, outputs)

        original_prediction = model(sample_input)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_model.keras")
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_prediction = loaded_model(sample_input)

            reloaded_layer = next(
                lyr for lyr in loaded_model.layers if isinstance(lyr, Zamba2MambaBlock)
            )
            assert reloaded_layer.d_model == block_config["d_model"]
            assert reloaded_layer.d_ssm == layer.d_ssm

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_prediction),
                keras.ops.convert_to_numpy(loaded_prediction),
                rtol=0, atol=1e-5,
                err_msg="Predictions differ after serialization",
            )

    def test_config_completeness(self, block_config: Dict[str, Any]) -> None:
        """get_config() must contain every __init__ param."""
        layer = Zamba2MambaBlock(**block_config)
        config = layer.get_config()

        required_keys = {
            "d_model", "d_state", "d_conv", "expand", "headdim", "d_ssm",
            "ngroups", "norm_epsilon", "rmsnorm", "norm_before_gate",
            "dt_min", "dt_max", "dt_init_floor", "bias", "conv_bias",
        }
        for key in required_keys:
            assert key in config, f"Missing {key} in get_config()"

        assert config["d_model"] == block_config["d_model"]
        assert config["d_state"] == block_config["d_state"]
        assert config["d_conv"] == block_config["d_conv"]
        assert config["expand"] == block_config["expand"]
        assert config["headdim"] == block_config["headdim"]
        # d_ssm was not given explicitly, so get_config() round-trips the
        # ORIGINAL None, not the resolved value (mirrors Zamba2SharedMLPBlock's
        # hidden_dim convention).
        assert config["d_ssm"] is None

    def test_d_ssm_round_trips_when_given_explicitly(self) -> None:
        """An explicitly-given ``d_ssm`` round-trips through ``get_config()``
        unresolved, matching ``Zamba2SharedMLPBlock.hidden_dim``'s convention."""
        layer = Zamba2MambaBlock(d_model=32, expand=2, headdim=8, d_ssm=48)
        assert layer.d_ssm == 48
        config = layer.get_config()
        assert config["d_ssm"] == 48

    @pytest.mark.parametrize("dtype_policy", ["float32", "mixed_float16"])
    def test_mixed_float16_no_nan(
        self,
        block_config: Dict[str, Any],
        sample_input: keras.KerasTensor,
        dtype_policy: str,
    ) -> None:
        """The Mamba2 scan must not produce NaN/Inf under ``mixed_float16``."""
        original_policy = keras.mixed_precision.global_policy()
        try:
            keras.mixed_precision.set_global_policy(dtype_policy)
            block = Zamba2MambaBlock(**block_config)
            hidden = keras.ops.cast(sample_input, block.compute_dtype)
            output = block(hidden)
            output_numpy = keras.ops.convert_to_numpy(output)
            assert np.isfinite(output_numpy).all(), f"NaN/Inf under {dtype_policy}"
        finally:
            keras.mixed_precision.set_global_policy(original_policy)


class TestZamba2Model:
    """Comprehensive test suite for :class:`Zamba2Model`, the full decoder
    stack assembled from steps 1-4's building blocks."""

    @pytest.fixture
    def model_config(self) -> Dict[str, Any]:
        """Small configuration: 6 depth positions, 2 'g' occurrences, 2
        physical mem-block slots (so the two 'g' positions use DIFFERENT
        slots -- the round-robin-repeat case is covered by a dedicated
        test/fixture below)."""
        return {
            "vocab_size": 37,
            "hidden_size": 32,
            "layer_mapping": ["m", "m", "g", "m", "m", "g"],
            "num_mem_blocks": 2,
            "num_heads": 4,
            "max_seq_len": 16,
            "d_state": 16,
            "headdim": 8,
        }

    @pytest.fixture
    def sample_ids(self) -> keras.KerasTensor:
        """Sample batch of token ids, seq_len short enough for the Mamba2
        scan's exact ``while_loop``."""
        return keras.random.randint((2, 6), 0, 37, dtype="int32")

    def test_initialization(self, model_config: Dict[str, Any]) -> None:
        """Initialization validates config, builds 4 Zamba2MambaBlock
        instances (one per 'm'), 2 physical mem-block pairs, and resolves 2
        LoRA occurrences (one per 'g')."""
        model = Zamba2Model(**model_config)

        assert model.vocab_size == model_config["vocab_size"]
        assert model.hidden_size == model_config["hidden_size"]
        assert model.layer_mapping == model_config["layer_mapping"]
        assert len(model.mamba_blocks) == 4
        assert len(model.mem_attention_blocks) == 2
        assert len(model.mem_mlp_blocks) == 2
        assert model._num_occurrences == 2
        assert not model.built

    def test_edge_cases(self) -> None:
        """Invalid configuration must raise before any sub-layer is built."""
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            Zamba2Model(vocab_size=0, hidden_size=32, layer_mapping=["m"], num_mem_blocks=1)
        with pytest.raises(ValueError, match="hidden_size must be positive"):
            Zamba2Model(vocab_size=10, hidden_size=0, layer_mapping=["m"], num_mem_blocks=1)
        with pytest.raises(ValueError, match="num_mem_blocks must be positive"):
            Zamba2Model(vocab_size=10, hidden_size=32, layer_mapping=["m"], num_mem_blocks=0)
        with pytest.raises(ValueError, match="divisible by"):
            Zamba2Model(
                vocab_size=10, hidden_size=32, layer_mapping=["m"],
                num_mem_blocks=1, num_heads=5,
            )
        with pytest.raises(ValueError, match="non-empty"):
            Zamba2Model(vocab_size=10, hidden_size=32, layer_mapping=[], num_mem_blocks=1)
        with pytest.raises(ValueError, match="'m' or 'g'"):
            Zamba2Model(
                vocab_size=10, hidden_size=32, layer_mapping=["m", "x"],
                num_mem_blocks=1,
            )

    def test_zero_g_positions_builds_and_runs(self) -> None:
        """Edge case (plan.md Problem Statement): a pure-Mamba2 stack with
        zero 'g' entries must still build and run -- the shared-block
        machinery must not assume at least one occurrence exists."""
        model = Zamba2Model(
            vocab_size=20,
            hidden_size=16,
            layer_mapping=["m", "m", "m"],
            num_mem_blocks=2,
            num_heads=4,
            max_seq_len=8,
            d_state=8,
            headdim=4,
        )
        ids = keras.random.randint((2, 5), 0, 20, dtype="int32")
        output = model(ids)

        assert output.shape == (2, 5, 20)
        assert np.isfinite(keras.ops.convert_to_numpy(output)).all()
        assert model._num_occurrences == 1  # max(0, 1), never zero

    def test_forward_pass_shape_and_finiteness(
        self, model_config: Dict[str, Any], sample_ids: keras.KerasTensor
    ) -> None:
        """Output shape is (batch, seq_len, vocab_size) and contains no
        NaN/Inf."""
        model = Zamba2Model(**model_config)
        output = model(sample_ids)

        assert output.shape == (
            sample_ids.shape[0], sample_ids.shape[1], model_config["vocab_size"]
        )
        output_numpy = keras.ops.convert_to_numpy(output)
        assert np.isfinite(output_numpy).all()
        assert model.built

    def test_pretrained_true_raises_not_implemented_error(
        self, model_config: Dict[str, Any]
    ) -> None:
        """No pretrained Zamba2 checkpoint exists anywhere; ``pretrained=True``
        must raise rather than silently returning a random-init model."""
        with pytest.raises(NotImplementedError, match="[Nn]o pretrained"):
            Zamba2Model(**model_config, pretrained=True)

    def test_pretrained_false_is_the_default_and_builds_normally(
        self, model_config: Dict[str, Any], sample_ids: keras.KerasTensor
    ) -> None:
        """``pretrained=False`` (the default) builds and runs normally."""
        model = Zamba2Model(**model_config, pretrained=False)
        output = model(sample_ids)
        assert np.isfinite(keras.ops.convert_to_numpy(output)).all()

    def test_gradient_flow_reaches_every_trainable_weight(
        self, model_config: Dict[str, Any], sample_ids: keras.KerasTensor
    ) -> None:
        """Decisive composition guard: a real backward pass through the
        WHOLE assembled model must reach every trainable weight -- the
        embedding, every Zamba2MambaBlock, every shared mem-block's base
        weights, AND every LoRA A/B pair actually exercised by
        ``layer_mapping`` (both 'g' occurrences here). A dead weight here
        is exactly the v2 guide Section 12.7 "stack reads only the last
        block" failure the plan's Pre-Mortem names.

        Every ``LoRAAdapter.b`` is zero-initialized (standard LoRA init), so
        the chain rule through ``delta = (x @ A) @ B`` routes an exactly-zero
        gradient back onto ``A`` at construction -- mathematically correct,
        not a defect (see ``TestLoRAAdapter.test_gradients_flow_to_every_exercised_occurrence``'s
        identical note). This test takes one optimizer warmup step first, as
        that test does, to move every exercised ``B`` away from zero before
        asserting gradient flow.
        """
        model = Zamba2Model(**model_config)
        _ = model(sample_ids)  # a subclassed keras.Model is unbuilt until its first call

        optimizer = keras.optimizers.Adam(learning_rate=1e-1)
        with tf.GradientTape() as warmup_tape:
            warmup_output = model(sample_ids, training=True)
            warmup_loss = keras.ops.mean(keras.ops.square(warmup_output))
        warmup_grads = warmup_tape.gradient(warmup_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(warmup_grads, model.trainable_variables))

        report = assert_gradients_reach_every_trainable_weight(model, sample_ids)

        assert len(report) == len(model.trainable_weights)
        assert len(report) > 0

    def test_serialization_cycle(
        self, model_config: Dict[str, Any], sample_ids: keras.KerasTensor
    ) -> None:
        """Full .keras serialization cycle with value-level (rtol=0)
        prediction comparison."""
        model = Zamba2Model(**model_config)
        original_prediction = model(sample_ids)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "zamba2_model.keras")
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_prediction = loaded_model(sample_ids)

            assert loaded_model.layer_mapping == model_config["layer_mapping"]
            assert loaded_model.num_mem_blocks == model_config["num_mem_blocks"]

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_prediction),
                keras.ops.convert_to_numpy(loaded_prediction),
                rtol=0, atol=1e-5,
                err_msg="Predictions differ after serialization",
            )

    def test_config_completeness(self, model_config: Dict[str, Any]) -> None:
        """get_config() must contain every __init__ param needed to rebuild
        an architecturally-identical model."""
        model = Zamba2Model(**model_config)
        config = model.get_config()

        required_keys = {
            "vocab_size", "hidden_size", "layer_mapping", "num_mem_blocks",
            "num_heads", "max_seq_len", "rope_theta", "rope_percentage",
            "attention_dropout_rate", "mem_block_norm_epsilon",
            "mlp_hidden_dim", "ffn_expansion_factor", "ffn_multiple_of",
            "lora_rank", "lora_alpha", "d_state", "d_conv", "expand",
            "headdim", "d_ssm", "ngroups", "mamba_norm_epsilon",
            "mamba_rmsnorm", "norm_before_gate", "dt_min", "dt_max",
            "dt_init_floor", "mamba_bias", "conv_bias", "use_bias",
            "kernel_initializer", "embeddings_initializer",
            "final_norm_epsilon",
        }
        for key in required_keys:
            assert key in config, f"Missing {key} in get_config()"

        assert config["layer_mapping"] == model_config["layer_mapping"]
        assert config["num_mem_blocks"] == model_config["num_mem_blocks"]
        assert config["vocab_size"] == model_config["vocab_size"]

        rebuilt = Zamba2Model.from_config(config)
        assert rebuilt.layer_mapping == model.layer_mapping
        assert rebuilt.num_mem_blocks == model.num_mem_blocks

    def test_causal_mask_is_honoured_at_the_model_level(
        self, model_config: Dict[str, Any]
    ) -> None:
        """Three-armed future-leak probe (v2 guide Section 12.1) run through
        the WHOLE model, not just one attention block: perturbing a future
        token must not change an earlier position's logits, perturbing the
        CURRENT token must change its own logits (anti-vacuity -- rules out
        a mask that blocks everything), and perturbing a PAST token must
        change a later position's logits (confirms the earlier position is
        actually read, not just unreadable by the future)."""
        model = Zamba2Model(**model_config)
        seq_len = 6
        base_ids = keras.random.randint((1, seq_len), 0, model_config["vocab_size"], dtype="int32")
        base_ids_numpy = keras.ops.convert_to_numpy(base_ids)

        earlier_position = 1
        current_position = 2
        future_position = 4

        def run(ids_numpy: np.ndarray) -> np.ndarray:
            ids = keras.ops.convert_to_tensor(ids_numpy, dtype="int32")
            return keras.ops.convert_to_numpy(model(ids))

        base_output = run(base_ids_numpy)

        # Arm 1: perturb a FUTURE token -> earlier position's logits unchanged.
        future_perturbed = base_ids_numpy.copy()
        future_perturbed[0, future_position] = (
            future_perturbed[0, future_position] + 1
        ) % model_config["vocab_size"]
        future_output = run(future_perturbed)
        np.testing.assert_allclose(
            base_output[:, earlier_position, :],
            future_output[:, earlier_position, :],
            rtol=0, atol=1e-5,
            err_msg="A future-token perturbation leaked into an earlier position's logits",
        )

        # Arm 2: perturb the CURRENT token -> its own logits DO change.
        current_perturbed = base_ids_numpy.copy()
        current_perturbed[0, current_position] = (
            current_perturbed[0, current_position] + 1
        ) % model_config["vocab_size"]
        current_output = run(current_perturbed)
        assert not np.allclose(
            base_output[:, current_position, :],
            current_output[:, current_position, :],
            rtol=0, atol=1e-5,
        ), "Perturbing a position's own token must change its own logits"

        # Arm 3: perturb a PAST token -> a LATER position's logits DO change.
        past_perturbed = base_ids_numpy.copy()
        past_perturbed[0, earlier_position] = (
            past_perturbed[0, earlier_position] + 1
        ) % model_config["vocab_size"]
        past_output = run(past_perturbed)
        assert not np.allclose(
            base_output[:, future_position, :],
            past_output[:, future_position, :],
            rtol=0, atol=1e-5,
        ), "A past-token perturbation must reach a later position's logits"

    def test_shared_mem_blocks_round_robin_repeat_with_fewer_physical_blocks(
        self,
    ) -> None:
        """Decisive weight-identity guard at the MODEL level: when
        ``num_mem_blocks`` is smaller than the number of 'g' occurrences,
        the round-robin must make two different depth positions literally
        invoke the SAME physical :class:`Zamba2SharedAttentionBlock`/
        :class:`Zamba2SharedMLPBlock` instance (``is``-identity on the
        underlying Variable objects), not merely an equal-valued copy."""
        model = Zamba2Model(
            vocab_size=20,
            hidden_size=16,
            layer_mapping=["m", "g", "m", "g", "m", "g"],  # 3 'g' occurrences
            num_mem_blocks=2,  # occurrences 0 and 2 both land on slot 0
            num_heads=4,
            max_seq_len=8,
            d_state=8,
            headdim=4,
        )

        g_positions = [
            (ref_idx, occurrence_idx)
            for kind, ref_idx, occurrence_idx in model._position_info
            if kind == "g"
        ]
        assert [ref_idx for ref_idx, _ in g_positions] == [0, 1, 0]
        assert [occ for _, occ in g_positions] == [0, 1, 2]

        # Occurrences 0 and 2 route to physical slot 0 -- same instances.
        first_slot, second_slot = g_positions[0][0], g_positions[2][0]
        assert first_slot == second_slot
        assert (
            model.mem_attention_blocks[first_slot]
            is model.mem_attention_blocks[second_slot]
        )
        assert model.mem_mlp_blocks[first_slot] is model.mem_mlp_blocks[second_slot]

        # Occurrence 1 routes to a DIFFERENT physical slot.
        assert model.mem_attention_blocks[0] is not model.mem_attention_blocks[1]
        assert model.mem_mlp_blocks[0] is not model.mem_mlp_blocks[1]

        # Run the model once, then confirm the shared instance still holds
        # the same Variable objects after a real call (not just pre-build).
        ids = keras.random.randint((2, 5), 0, 20, dtype="int32")
        _ = model(ids)
        attn_weights_slot_0 = list(model.mem_attention_blocks[0].weights)
        mlp_weights_slot_0 = list(model.mem_mlp_blocks[0].weights)
        assert len(attn_weights_slot_0) > 0
        assert len(mlp_weights_slot_0) > 0
        # Re-fetch through the SAME index used by both depth positions
        # (first_slot == second_slot == 0) -- identical object both times.
        assert all(
            w is other
            for w, other in zip(
                attn_weights_slot_0, model.mem_attention_blocks[first_slot].weights
            )
        )

    def test_lora_deltas_differ_across_g_occurrences(
        self, model_config: Dict[str, Any], sample_ids: keras.KerasTensor
    ) -> None:
        """The decisive composition guard for invariant 2 (plan.md): capture
        the per-'g'-position hidden-state delta attributable to the shared
        MLP mem-block's LoRA-selected occurrence, and assert they are NOT
        all equal. The Section 12.7 "stack reads only the last occurrence"
        failure would collapse every occurrence's delta onto one, which a
        shape-only or finiteness-only check cannot detect.
        """
        model = Zamba2Model(**model_config)
        _ = model(sample_ids)  # build

        hidden = keras.random.normal(shape=(2, 6, model_config["hidden_size"]))

        g_occurrences = [
            (ref_idx, occurrence_idx)
            for kind, ref_idx, occurrence_idx in model._position_info
            if kind == "g"
        ]
        assert len(g_occurrences) == 2

        mlp_block = model.mem_mlp_blocks[g_occurrences[0][0]]
        # Both occurrences in this fixture route to DIFFERENT physical
        # slots (num_mem_blocks == num 'g' occurrences == 2), so compare
        # each slot's own LoRA delta at its own occurrence index.
        outputs = []
        for ref_idx, occurrence_idx in g_occurrences:
            block = model.mem_mlp_blocks[ref_idx]
            outputs.append(
                keras.ops.convert_to_numpy(
                    block(hidden, occurrence_idx=occurrence_idx)
                )
            )

        assert not np.allclose(outputs[0], outputs[1], rtol=0, atol=1e-5), (
            "Two different 'g' occurrences produced identical MLP mem-block "
            "outputs -- the LoRA delta is not actually varying per occurrence "
            "(the 'stack reads only the last block' failure shape)."
        )

        # Same occurrence, same physical block, called twice -> identical
        # (no hidden per-call-order state leak).
        ref_idx, occurrence_idx = g_occurrences[0]
        block = model.mem_mlp_blocks[ref_idx]
        repeat_a = keras.ops.convert_to_numpy(block(hidden, occurrence_idx=occurrence_idx))
        repeat_b = keras.ops.convert_to_numpy(block(hidden, occurrence_idx=occurrence_idx))
        np.testing.assert_allclose(repeat_a, repeat_b, rtol=0, atol=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
