"""Comprehensive test suite for the Zamba2 model package.

Grows incrementally alongside ``src/dl_techniques/models/language/zamba2/``;
see ``plans/plan-2026-09-12T075714-035fd488/plan.md`` for the build order.
Step 1 covers :class:`LoRAAdapter` only.
"""

import os
import tempfile
from typing import Any, Dict

import numpy as np
import pytest
import tensorflow as tf
import keras

from dl_techniques.models.language.zamba2.layers import LoRAAdapter


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
