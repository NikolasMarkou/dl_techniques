import pytest
import numpy as np
import tempfile
import os
from typing import Dict, Any, Tuple

import keras
import tensorflow as tf

from dl_techniques.models.trm.components import TRMReasoningModule, TRMInner


@pytest.fixture(scope="module")
def common_config() -> Dict[str, Any]:
    """Provides a common base configuration for TRM components."""
    return {
        "hidden_size": 32,
        "num_heads": 4,
        "expansion": 2.0,
        "seq_len": 16,
        "puzzle_emb_len": 4,
    }


class TestTRMReasoningModule:
    """Tests for the TRMReasoningModule layer."""

    @pytest.fixture(scope="class")
    def reasoning_config(self, common_config: Dict[str, Any]) -> Dict[str, Any]:
        """Configuration specific to TRMReasoningModule."""
        return {**common_config, "num_layers": 2}

    @pytest.fixture(scope="class")
    def sample_input(self, reasoning_config: Dict[str, Any]) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Provides sample input tensors for the reasoning module."""
        batch_size = 2
        full_seq_len = reasoning_config["seq_len"] + reasoning_config["puzzle_emb_len"]
        hidden_size = reasoning_config["hidden_size"]
        shape = (batch_size, full_seq_len, hidden_size)

        hidden_states = keras.ops.convert_to_tensor(np.random.randn(*shape), dtype='float32')
        input_injection = keras.ops.convert_to_tensor(np.random.randn(*shape), dtype='float32')
        return hidden_states, input_injection

    def test_creation_and_config(self, reasoning_config: Dict[str, Any]):
        """Test layer creation and sub-layer count."""
        layer = TRMReasoningModule(**reasoning_config)
        assert len(layer.layers_list) == reasoning_config["num_layers"]

        # Test get_config and from_config cycle
        config = layer.get_config()
        reconstructed_layer = TRMReasoningModule.from_config(config)
        assert reconstructed_layer.get_config() == config

    def test_build_method(self, reasoning_config: Dict[str, Any], sample_input: Tuple[keras.KerasTensor, ...]):
        """Test that the build method correctly builds the layer and its sub-layers."""
        layer = TRMReasoningModule(**reasoning_config)
        input_shape = sample_input[0].shape

        assert not layer.built
        layer.build(input_shape)
        assert layer.built

        # Check that all internal transformer layers are also built
        for sub_layer in layer.layers_list:
            assert sub_layer.built

    def test_call_method(self, reasoning_config: Dict[str, Any], sample_input: Tuple[keras.KerasTensor, ...]):
        """Test the forward pass of the layer."""
        layer = TRMReasoningModule(**reasoning_config)
        hidden_states, input_injection = sample_input

        # Test in inference mode
        output = layer(hidden_states, input_injection, training=False)
        assert output.shape == hidden_states.shape

        # Test in training mode
        output_train = layer(hidden_states, input_injection, training=True)
        assert output_train.shape == hidden_states.shape

    def test_serialization(self, reasoning_config: Dict[str, Any], sample_input: Tuple[keras.KerasTensor, ...]):
        """Test saving and loading the layer."""
        layer = TRMReasoningModule(**reasoning_config)
        hidden_states, input_injection = sample_input
        original_output = layer(hidden_states, input_injection)

        # Wrap in a Functional API model to save
        input1 = keras.Input(shape=hidden_states.shape[1:])
        input2 = keras.Input(shape=input_injection.shape[1:])
        outputs = layer(input1, input2)
        model = keras.Model(inputs=[input1, input2], outputs=outputs)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "reasoning_module.keras")
            model.save(model_path)
            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={"TRMReasoningModule": TRMReasoningModule}
            )

        loaded_output = loaded_model([hidden_states, input_injection])

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(original_output),
            keras.ops.convert_to_numpy(loaded_output),
            rtol=1e-6, atol=1e-6
        )


class TestTRMInner:
    """Tests for the TRMInner layer."""

    @pytest.fixture(scope="class")
    def inner_config(self, common_config: Dict[str, Any]) -> Dict[str, Any]:
        """Configuration specific to TRMInner."""
        return {
            **common_config,
            "vocab_size": 100,
            "h_layers": 2,
            "l_layers": 3,
        }

    @pytest.fixture(scope="class")
    def sample_inner_input(self, inner_config: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Provides sample carry and data dicts for the inner module."""
        batch_size = 2
        full_seq_len = inner_config["seq_len"] + inner_config["puzzle_emb_len"]
        hidden_size = inner_config["hidden_size"]
        shape = (batch_size, full_seq_len, hidden_size)

        carry = {
            "z_H": keras.ops.convert_to_tensor(np.random.randn(*shape), dtype='float32'),
            "z_L": keras.ops.convert_to_tensor(np.random.randn(*shape), dtype='float32'),
        }
        data = {
            "inputs": keras.ops.convert_to_tensor(
                np.random.randint(0, inner_config["vocab_size"], size=(batch_size, inner_config["seq_len"])),
                dtype='int32'
            )
        }
        return carry, data

    def test_creation_and_config(self, inner_config: Dict[str, Any]):
        """Test layer creation and structure."""
        layer = TRMInner(**inner_config)
        assert isinstance(layer.H_level, TRMReasoningModule)
        assert isinstance(layer.L_level, TRMReasoningModule)
        assert isinstance(layer.token_emb, keras.layers.Embedding)
        assert isinstance(layer.lm_head, keras.layers.Dense)
        assert isinstance(layer.q_head, keras.layers.Dense)
        assert len(layer.H_level.layers_list) == inner_config["h_layers"]
        assert len(layer.L_level.layers_list) == inner_config["l_layers"]

        # Test get_config and from_config cycle
        config = layer.get_config()
        reconstructed_layer = TRMInner.from_config(config)
        assert reconstructed_layer.get_config() == config

    def test_build_method(self, inner_config: Dict[str, Any]):
        """Test that build creates initial state weights correctly."""
        layer = TRMInner(**inner_config)
        assert not layer.built

        layer.build()
        assert layer.built

        full_seq_len = inner_config["seq_len"] + inner_config["puzzle_emb_len"]
        expected_shape = (1, full_seq_len, inner_config["hidden_size"])
        assert hasattr(layer, "H_init")
        assert hasattr(layer, "L_init")
        assert layer.H_init.shape == expected_shape
        assert layer.L_init.shape == expected_shape

    def test_call_method(self, inner_config: Dict[str, Any], sample_inner_input: Tuple[Dict, Dict]):
        """Test the forward pass of the TRMInner layer and gradient stopping."""
        layer = TRMInner(**inner_config)
        carry, data = sample_inner_input

        # --- Part 1: Test forward pass shapes ---
        new_carry_check, logits_check, (q_halt, q_continue) = layer(carry, data, training=False)

        batch_size = data["inputs"].shape[0]

        assert new_carry_check["z_H"].shape == carry["z_H"].shape
        assert new_carry_check["z_L"].shape == carry["z_L"].shape
        assert logits_check.shape == (batch_size, inner_config["seq_len"], inner_config["vocab_size"])
        assert q_halt.shape == (batch_size,)
        assert q_continue.shape == (batch_size,)

        # --- Part 2: Test gradient stopping logic ---
        z_H_var = tf.Variable(carry["z_H"])
        z_L_var = tf.Variable(carry["z_L"])
        carry_vars = {"z_H": z_H_var, "z_L": z_L_var}

        with tf.GradientTape(persistent=True) as tape:
            tape.watch([z_H_var, z_L_var])
            new_carry, logits, _ = layer(carry_vars, data, training=True)
            loss_from_logits = tf.reduce_sum(logits)
            loss_from_new_carry = tf.reduce_sum(new_carry["z_H"])

        # Gradient from logits should exist because the computation path is intact
        grad_from_logits = tape.gradient(loss_from_logits, z_H_var)
        assert grad_from_logits is not None, "Gradient should flow from logits to initial carry."

        # Gradient from new_carry should be None because of tf.stop_gradient
        grad_from_new_carry = tape.gradient(loss_from_new_carry, z_H_var)
        assert grad_from_new_carry is None, "Gradient should be stopped from new_carry to initial carry."

        del tape  # Explicitly release persistent tape resources

    def test_serialization(self, inner_config: Dict[str, Any], sample_inner_input: Tuple[Dict, Dict]):
        """Test saving and loading the inner layer, including initial state weights."""
        layer = TRMInner(**inner_config)
        carry, data = sample_inner_input

        # Run a forward pass to build the layer
        original_outputs = layer(carry, data)
        original_h_init_val = keras.ops.convert_to_numpy(layer.H_init)

        # Wrap in a model to save
        inputs = [
            keras.Input(shape=carry["z_H"].shape[1:], name="z_H"),
            keras.Input(shape=carry["z_L"].shape[1:], name="z_L"),
            keras.Input(shape=data["inputs"].shape[1:], dtype='int32', name="inputs")
        ]
        carry_in = {"z_H": inputs[0], "z_L": inputs[1]}
        data_in = {"inputs": inputs[2]}
        outputs = layer(carry_in, data_in)
        model = keras.Model(inputs=inputs, outputs=outputs)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "trm_inner.keras")
            model.save(model_path)
            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={"TRMInner": TRMInner, "TRMReasoningModule": TRMReasoningModule}
            )

        # loaded_model is the functional wrapper, not the layer itself.
        # Find the layer inside the loaded model to check its weights.
        loaded_layer = None
        for lyr in loaded_model.layers:
            if isinstance(lyr, TRMInner):
                loaded_layer = lyr
                break
        assert loaded_layer is not None

        # Check that initial state weights were loaded correctly
        loaded_h_init_val = keras.ops.convert_to_numpy(loaded_layer.H_init)
        np.testing.assert_array_equal(original_h_init_val, loaded_h_init_val)

        # Check that forward pass gives the same result
        loaded_outputs = loaded_layer(carry, data)

        # Compare new_carry, logits, and q_values
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(original_outputs[0]["z_H"]),
            keras.ops.convert_to_numpy(loaded_outputs[0]["z_H"]),
            rtol=1e-6, atol=1e-6,
            err_msg="z_H mismatch after serialization"
        )
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(original_outputs[1]),
            keras.ops.convert_to_numpy(loaded_outputs[1]),
            rtol=1e-6, atol=1e-6,
            err_msg="Logits mismatch after serialization"
        )
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(original_outputs[2][0]),  # q_halt
            keras.ops.convert_to_numpy(loaded_outputs[2][0]),
            rtol=1e-6, atol=1e-6,
            err_msg="q_halt mismatch after serialization"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])