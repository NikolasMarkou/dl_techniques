"""Model-level tests for `dl_techniques.models.language.ptrm`.

Tests cover model creation, forward pass, serialization, and PTRM-specific
inference functionality.
"""

import pytest
import numpy as np
from typing import Dict, Any

import keras
import tensorflow as tf

from dl_techniques.models.language.ptrm import (
    PTRM, PTRMInference, create_ptrm, create_ptrm_from_variant,
    get_ppbench_config, get_sudoku_extreme_config
)

from ..knob_sensitivity_oracle import (
    as_array,
    assert_structural_knob_changes_weights,
    build_seeded,
)


class TestPTRMModelBasic:
    """Test basic PTRM model functionality and state management."""

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
        model = PTRM(**tiny_config)
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
        model = PTRM(**config)
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
        model = PTRM(**tiny_config)
        # Build the model to access inner weights
        _ = model(model.initial_carry(sample_batch), sample_batch)

        carry = model.initial_carry(sample_batch)
        # Manually set one item to be halted
        halted_mask = keras.ops.convert_to_tensor([True, False], dtype="bool")
        carry["halted"] = halted_mask

        # Perform one step
        new_carry, _ = model(carry, sample_batch, training=False)

        # Check that current_data was updated correctly for halted item
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


class TestPTRMModelConfigurations:
    """Test various model configurations and variants."""

    def test_mlp_variants(self):
        """Test MLP variant configurations."""
        variants = ["mlp_tiny", "mlp_small", "mlp_base"]
        for variant in variants:
            model = create_ptrm_from_variant(
                variant=variant,
                vocab_size=100,
                seq_len=50,
            )
            assert isinstance(model, PTRM)
            assert model.built

    def test_att_variants(self):
        """Test Attention variant configurations."""
        variants = ["att_small", "att_base", "att_large"]
        for variant in variants:
            model = create_ptrm_from_variant(
                variant=variant,
                vocab_size=100,
                seq_len=50,
            )
            assert isinstance(model, PTRM)
            assert model.built

    def test_invalid_variant_raises(self):
        """Test that invalid variant raises ValueError."""
        with pytest.raises(ValueError, match="Unknown variant"):
            create_ptrm_from_variant("nonexistent", vocab_size=100, seq_len=50)

    def test_different_layer_configs(self):
        """Test that h_layers/l_layers build the requested layer stacks."""
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

        def _build(layer_config):
            config = {**base_config, **layer_config}
            model = PTRM(**config)
            batch = {"inputs": keras.ops.zeros((1, config['seq_len']), dtype='int32')}
            model(model.initial_carry(batch), batch)
            return model

        def _sig(cfg):
            return assert_structural_knob_changes_weights(
                {"base": (lambda: _build({"h_layers": 1, "l_layers": 1})),
                 "swept": (lambda: _build(cfg))},
                knob=f"h_layers={cfg['h_layers']}, l_layers={cfg['l_layers']}",
            )

        sig_h = _sig({"h_layers": 3, "l_layers": 1})
        sig_l = _sig({"h_layers": 1, "l_layers": 3})
        assert len(sig_h["base"]) < len(sig_h["swept"])
        assert len(sig_l["base"]) < len(sig_l["swept"])

        # (3, 1) and (1, 3) are bit-identical in weights under one seed
        outs = []
        for cfg in ({"h_layers": 3, "l_layers": 1}, {"h_layers": 1, "l_layers": 3}):
            model = build_seeded(lambda cfg=cfg: PTRM(**{**base_config, **cfg}))
            batch = {"inputs": keras.ops.zeros((1, base_config['seq_len']), dtype='int32')}
            _, out = model(model.initial_carry(batch), batch)
            outs.append(as_array(out["logits"]))
        delta = float(np.max(np.abs(outs[0] - outs[1])))
        assert delta > 1e-5, (
            "h_layers and l_layers are interchangeable: (3, 1) and (1, 3) hold "
            f"identical weights AND produce logits differing by only {delta:.3e}"
        )

    def test_bellman_update_config(self):
        """Test that target_q_continue is produced when no_act_continue is False."""
        config = {
            "vocab_size": 50, "hidden_size": 16, "seq_len": 8,
            "expansion": 2.0, "num_heads": 2, "l_layers": 1, "h_layers": 1,
            "puzzle_emb_len": 2,
            "halt_max_steps": 3, "no_act_continue": False, "halt_exploration_prob": 0.0,
        }
        model = PTRM(**config)
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


class TestPTRMSerialization:
    """Test serialization round-trip."""

    def test_get_config_roundtrip(self, tiny_config):
        """Test that get_config() returns all constructor args."""
        model = PTRM(**tiny_config)
        config = model.get_config()
        
        # Check all constructor args are in config
        for key in tiny_config:
            assert key in config, f"Missing config key: {key}"
            assert config[key] == tiny_config[key], f"Mismatch for {key}"

    def test_save_load_roundtrip(self, tiny_config, sample_batch, tmp_path):
        """Test full .keras save/load roundtrip with value identity."""
        model = PTRM(**tiny_config)
        
        # Build by running once
        carry = model.initial_carry(sample_batch)
        _ = model(carry, sample_batch, training=True)
        
        # Save
        save_path = tmp_path / "ptrm_model.keras"
        model.save(str(save_path))
        
        # Load
        loaded_model = keras.models.load_model(str(save_path))
        
        # Verify config matches
        assert loaded_model.vocab_size == model.vocab_size
        assert loaded_model.hidden_size == model.hidden_size
        assert loaded_model.num_heads == model.num_heads
        assert loaded_model.halt_max_steps == model.halt_max_steps
        
        # Verify weights match (before any forward pass on loaded model)
        for w1, w2 in zip(model.weights, loaded_model.weights):
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(w1),
                keras.ops.convert_to_numpy(w2),
                atol=0.0,
                err_msg=f"Weight mismatch: {w1.name}"
            )
        
        # Verify forward pass matches
        carry = model.initial_carry(sample_batch)
        _, outputs1 = model(carry, sample_batch, training=False)
        
        carry = loaded_model.initial_carry(sample_batch)
        _, outputs2 = loaded_model(carry, sample_batch, training=False)
        
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(outputs1["logits"]),
            keras.ops.convert_to_numpy(outputs2["logits"]),
            atol=1e-6, rtol=0,
        )


class TestPTRMIntegration:
    """Test integration and end-to-end functionality."""

    @pytest.fixture(scope="class")
    def tiny_config(self) -> Dict[str, Any]:
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
        batch_size = 2
        return {
            "inputs": keras.ops.convert_to_tensor(
                np.random.randint(0, tiny_config['vocab_size'], size=(batch_size, tiny_config['seq_len'])),
                dtype='int32'
            ),
        }

    def test_end_to_end_training_simulation(self, tiny_config: Dict[str, Any], sample_batch: Dict[str, Any]):
        """Test a complete simulated training step with gradient flow."""
        model = PTRM(**tiny_config)
        optimizer = keras.optimizers.Adam(learning_rate=1e-3)
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        labels = keras.ops.convert_to_tensor(
            np.random.randint(0, tiny_config['vocab_size'], size=(2, tiny_config['seq_len'])),
            dtype='int32'
        )

        # Build the model
        _ = model(model.initial_carry(sample_batch), sample_batch, training=True)

        initial_weights = [tf.identity(w) for w in model.trainable_weights]
        assert len(initial_weights) > 0

        with tf.GradientTape() as tape:
            carry = model.initial_carry(sample_batch)
            total_loss = 0.0

            for _ in range(tiny_config["halt_max_steps"]):
                carry, outputs = model(carry, sample_batch, training=True)
                step_loss = loss_fn(labels, outputs["logits"])
                total_loss += step_loss

        assert keras.ops.shape(total_loss) == ()
        assert not np.isnan(keras.ops.convert_to_numpy(total_loss))

        grads = tape.gradient(total_loss, model.trainable_weights)
        with pytest.warns(UserWarning, match="Gradients do not exist"):
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

        weights_updated = False
        for initial_w, final_w in zip(initial_weights, model.trainable_weights):
            if not np.allclose(keras.ops.convert_to_numpy(initial_w), keras.ops.convert_to_numpy(final_w)):
                weights_updated = True
                break

        assert weights_updated, "Model weights were not updated after a training step."


class TestPTRMPresets:
    """Test preset configurations match paper specifications."""

    def test_ppbench_config(self):
        """Test PPBench preset configuration."""
        config = get_ppbench_config()
        assert config["variant"] == "att_base"
        assert config["vocab_size"] == 294
        assert config["seq_len"] == 100
        assert "inference" in config
        assert config["inference"]["num_rollouts"] == 100
        assert config["inference"]["supervision_steps"] == 48
        assert config["inference"]["noise_scale"] == 0.2

    def test_sudoku_config(self):
        """Test Sudoku-Extreme preset configuration."""
        config = get_sudoku_extreme_config()
        assert config["variant"] == "mlp_base"
        assert config["vocab_size"] == 20
        assert config["seq_len"] == 81
        assert config["inference"]["num_rollouts"] == 100
        assert config["inference"]["supervision_steps"] == 64
        assert config["inference"]["noise_scale"] == 0.3

    def test_model_creation_from_preset(self):
        """Test creating model from preset configs."""
        for preset_fn in [get_ppbench_config, get_sudoku_extreme_config]:
            config = preset_fn()
            model_config = {k: v for k, v in config.items() if k != "inference"}
            model = create_ptrm_from_variant(**model_config)
            assert isinstance(model, PTRM)
            assert model.built


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])