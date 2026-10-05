"""Shared fixtures for PTRM tests."""

import pytest
import numpy as np
import keras

from dl_techniques.models.language.ptrm import PTRM


@pytest.fixture(scope="class")
def tiny_config() -> dict:
    """Create a tiny model configuration for fast testing."""
    return {
        "vocab_size": 100,
        "hidden_size": 32,
        "seq_len": 16,
        "expansion": 2.0,
        "num_heads": 4,
        "l_layers": 1,
        "h_layers": 1,
        "puzzle_emb_len": 4,
        "halt_max_steps": 4,
        "halt_exploration_prob": 0.1,
        "no_act_continue": True,
    }


@pytest.fixture(scope="class")
def tiny_model(tiny_config) -> PTRM:
    """Create a tiny PTRM model for testing."""
    model = PTRM(**tiny_config)
    # Build the model
    batch = {"inputs": keras.ops.zeros((2, tiny_config['seq_len']), dtype='int32')}
    carry = model.initial_carry(batch)
    _ = model(carry, batch, training=False)
    return model


@pytest.fixture(scope="class")
def sample_batch(tiny_config) -> dict:
    """Create a sample input batch."""
    return {
        "inputs": keras.ops.convert_to_tensor(
            np.random.randint(0, tiny_config['vocab_size'], size=(4, tiny_config['seq_len'])),
            dtype='int32'
        ),
    }