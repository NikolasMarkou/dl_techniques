"""Model-level tests for `dl_techniques.models.tiny_recursive_model`.

Ported from the former `tests/test_models/test_trm/test_model.py` during the
iter-1 test-dir consolidation onto the src-mirroring `test_tiny_recursive_model/`.

Coverage here is the UNIQUE state/config/integration coverage not already in the
canonical `test_trm.py` (carry structure, single-step state changes, state reset
on halt, layer-config variants, Bellman target presence, end-to-end training).

NOTE: the stale `TestTRMModelBasic::test_multi_step_loop_simulation` was
INTENTIONALLY DROPPED, not ported. Its assertion that inference never halts
early contradicts the live, decision-anchored TRM semantics
(`DECISION plan_2026-05-10_e6309bd5/D-001`, model.py:365-382: inference halts on
`is_last_step | (q_halt > 0)` BY DESIGN). Its only valid concern — that the model
halts by `halt_max_steps` — is already covered by canonical
`test_trm.py::test_inference_early_halt` (+ `test_inference_halt_max_steps_1`,
`test_inference_halt_qlearning`). See plan_2026-06-17_b38768bf D-001.
Redundant tests (`test_model_creation`, serialization-cycle/config-reconstruction)
were also dropped in favour of the canonical `test_creation` /
`test_get_config_roundtrip` / `test_save_load_roundtrip`.
"""

import pytest
import numpy as np
from typing import Dict, Any

import keras
import tensorflow as tf

from dl_techniques.models.tiny_recursive_model import TRM, TRMInner, TRMReasoningModule


class TestTRMModelBasic:
    """Test basic TRM model functionality and state management."""

    @pytest.fixture(scope="class")
    def tiny_config(self) -> Dict[str, Any]:
        """Create a tiny model configuration for fast testing."""
        return {
            "vocab_size": 100,
            "hidden_size": 32,
            "seq_len": 16,
            "expansion": 2.0,
            "num_heads": 4,
            "l_layers": 2,
            "h_layers": 2,
            "puzzle_emb_len": 4,
            "halt_max_steps": 5,
            "halt_exploration_prob": 0.1,
            "no_act_continue": True,
        }

    @pytest.fixture(scope="class")
    def sample_batch(self, tiny_config: Dict[str, Any]) -> Dict[str, keras.KerasTensor]:
        """Create a sample input batch for testing."""
        batch_size = 2
        return {
            "inputs": keras.ops.convert_to_tensor(
                np.random.randint(0, tiny_config['vocab_size'], size=(batch_size, tiny_config['seq_len'])),
                dtype='int32'
            ),
        }

    def test_initial_carry(self, tiny_config: Dict[str, Any], sample_batch: Dict[str, keras.KerasTensor]):
        """Test the initial_carry method for correct state structure and shapes."""
        model = TRM(**tiny_config)
        carry = model.initial_carry(sample_batch)

        batch_size = sample_batch["inputs"].shape[0]
        full_len = tiny_config['seq_len'] + tiny_config['puzzle_emb_len']

        # Check carry structure
        assert "inner_carry" in carry
        assert "steps" in carry
        assert "halted" in carry
        assert "current_data" in carry

        # Check shapes
        assert carry["inner_carry"]["z_H"].shape == (batch_size, full_len, tiny_config['hidden_size'])
        assert carry["inner_carry"]["z_L"].shape == (batch_size, full_len, tiny_config['hidden_size'])
        assert carry["steps"].shape == (batch_size,)
        assert carry["halted"].shape == (batch_size,)
        assert carry["current_data"]["inputs"].shape == sample_batch["inputs"].shape

        # Check initial values
        assert keras.ops.all(carry["halted"])  # Should start halted to trigger reset

    def test_single_forward_step(self, tiny_config: Dict[str, Any], sample_batch: Dict[str, keras.KerasTensor]):
        """Test a single forward pass (one ACT step) through the model."""
        # Make the test deterministic by forcing exploration, which prevents immediate halting.
        config = tiny_config.copy()
        config["halt_exploration_prob"] = 1.0
        model = TRM(**config)
        initial_carry = model.initial_carry(sample_batch)

        # Run one step
        new_carry, outputs = model(initial_carry, sample_batch, training=True)

        # Check output shapes
        assert "logits" in outputs
        assert "q_halt_logits" in outputs
        assert "q_continue_logits" in outputs
        assert outputs["logits"].shape == (sample_batch["inputs"].shape[0], config['seq_len'],
                                           config['vocab_size'])
        assert outputs["q_halt_logits"].shape == (sample_batch["inputs"].shape[0],)

        # Check new carry shapes and state changes
        assert new_carry["steps"].shape == (sample_batch["inputs"].shape[0],)
        assert keras.ops.all(new_carry["steps"] == 1)  # Steps should be 1
        assert not keras.ops.all(new_carry["halted"])  # Should not be halted after one step in training

    def test_state_reset_on_halt(self, tiny_config: Dict[str, Any], sample_batch: Dict[str, keras.KerasTensor]):
        """Verify that z_H and z_L are reset when an item is halted."""
        model = TRM(**tiny_config)
        # Build the model to access inner weights
        _ = model(model.initial_carry(sample_batch), sample_batch)

        carry = model.initial_carry(sample_batch)
        # Manually set one item to be halted
        halted_mask = keras.ops.convert_to_tensor([True, False], dtype="bool")
        carry["halted"] = halted_mask

        # Perform one step
        new_carry, _ = model(carry, sample_batch, training=False)

        # The model's call method checks for the *previous* halt state to reset.
        # We can't directly inspect the intermediate `z_H`, but we can check the new carry's `z_H`
        # and know that it was computed from a reset state for the first item.
        # A simpler check is to see if the `current_data` was updated correctly.
        data_item0_before = carry["current_data"]["inputs"][0]
        data_item0_after = new_carry["current_data"]["inputs"][0]

        # Since item 0 was halted, its data should be updated from the new batch.
        assert not np.array_equal(
            keras.ops.convert_to_numpy(data_item0_before),
            keras.ops.convert_to_numpy(data_item0_after)
        )
        assert np.array_equal(
            keras.ops.convert_to_numpy(sample_batch["inputs"][0]),
            keras.ops.convert_to_numpy(data_item0_after)
        )

        # Since item 1 was not halted, its data should NOT be updated.
        data_item1_before = carry["current_data"]["inputs"][1]
        data_item1_after = new_carry["current_data"]["inputs"][1]
        assert np.array_equal(
            keras.ops.convert_to_numpy(data_item1_before),
            keras.ops.convert_to_numpy(data_item1_after)
        )


class TestTRMModelConfigurations:
    """Test various model configurations."""

    def test_different_layer_configs(self):
        """Test the model with different h_layers and l_layers."""
        configs_to_test = [
            {"h_layers": 1, "l_layers": 1},
            {"h_layers": 3, "l_layers": 1},
            {"h_layers": 1, "l_layers": 3},
        ]
        base_config = {
            "vocab_size": 50, "hidden_size": 16, "seq_len": 8,
            "expansion": 2.0, "num_heads": 2,
            "puzzle_emb_len": 2, "halt_max_steps": 3,
            "no_act_continue": True, "halt_exploration_prob": 0.1,
        }

        for layer_config in configs_to_test:
            config = {**base_config, **layer_config}
            model = TRM(**config)
            batch = {
                "inputs": keras.ops.zeros((1, config['seq_len']), dtype='int32'),
            }
            carry = model.initial_carry(batch)
            _, outputs = model(carry, batch)

            # Check that it runs and produces valid output shapes
            assert outputs["logits"].shape == (1, config['seq_len'], config['vocab_size'])

    def test_bellman_update_config(self):
        """Test that target_q_continue is produced when no_act_continue is False."""
        config = {
            "vocab_size": 50, "hidden_size": 16, "seq_len": 8,
            "expansion": 2.0, "num_heads": 2, "l_layers": 1, "h_layers": 1,
            "puzzle_emb_len": 2,
            "halt_max_steps": 3, "no_act_continue": False, "halt_exploration_prob": 0.0,
        }
        model = TRM(**config)
        batch = {
            "inputs": keras.ops.zeros((2, config['seq_len']), dtype='int32'),
        }
        carry = model.initial_carry(batch)

        # In training mode with no_act_continue=False, we expect a Bellman target
        _, outputs = model(carry, batch, training=True)
        assert "target_q_continue" in outputs
        assert outputs["target_q_continue"].shape == (2,)

        # In inference mode, it should not be present
        _, outputs = model(carry, batch, training=False)
        assert "target_q_continue" not in outputs


class TestTRMIntegration:
    """Test integration and end-to-end functionality."""

    @pytest.fixture(scope="class")
    def tiny_config(self) -> Dict[str, Any]:
        """Create a tiny model configuration for fast testing."""
        return {
            "vocab_size": 100,
            "hidden_size": 32,
            "seq_len": 16,
            "expansion": 2.0,
            "num_heads": 4,
            "l_layers": 2,
            "h_layers": 2,
            "puzzle_emb_len": 4,
            "halt_max_steps": 5,
            "halt_exploration_prob": 0.1,
            "no_act_continue": True,
        }

    @pytest.fixture(scope="class")
    def sample_batch(self, tiny_config: Dict[str, Any]) -> Dict[str, keras.KerasTensor]:
        """Create a sample input batch for testing."""
        batch_size = 2
        return {
            "inputs": keras.ops.convert_to_tensor(
                np.random.randint(0, tiny_config['vocab_size'], size=(batch_size, tiny_config['seq_len'])),
                dtype='int32'
            ),
        }

    def test_end_to_end_training_simulation(self, tiny_config: Dict[str, Any], sample_batch: Dict[str, Any]):
        """Test a complete simulated training step with gradient flow."""
        model = TRM(**tiny_config)
        optimizer = keras.optimizers.Adam(learning_rate=1e-3)
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        # Create dummy labels
        labels = keras.ops.convert_to_tensor(
            np.random.randint(0, tiny_config['vocab_size'], size=(2, tiny_config['seq_len'])),
            dtype='int32'
        )

        # Build the model by running a dummy forward pass before accessing weights
        _ = model(model.initial_carry(sample_batch), sample_batch, training=True)

        # Get initial weights to check for updates
        initial_weights = [tf.identity(w) for w in model.trainable_weights]
        assert len(initial_weights) > 0

        with tf.GradientTape() as tape:
            carry = model.initial_carry(sample_batch)
            total_loss = 0.0

            # Simulate a few steps of the ACT loop
            for _ in range(tiny_config["halt_max_steps"]):
                carry, outputs = model(carry, sample_batch, training=True)
                # For simplicity, we just add the loss from each step
                step_loss = loss_fn(labels, outputs["logits"])
                total_loss += step_loss

        # Check that loss is a valid tensor
        assert keras.ops.shape(total_loss) == ()
        assert not np.isnan(keras.ops.convert_to_numpy(total_loss))

        # Apply gradients
        grads = tape.gradient(total_loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # Check if weights have been updated
        weights_updated = False
        for initial_w, final_w in zip(initial_weights, model.trainable_weights):
            if not np.allclose(keras.ops.convert_to_numpy(initial_w), keras.ops.convert_to_numpy(final_w)):
                weights_updated = True
                break

        assert weights_updated, "Model weights were not updated after a training step."


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
