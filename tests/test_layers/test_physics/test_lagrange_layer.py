"""Tests for the physics Lagrangian Neural Network layers.

Covers both:
  - ``LagrangianNeuralNetworkLayer`` (gradient-tape based; TensorFlow-only
    forward path — documented accepted exception), and
  - ``ApproximatedLNNLayer`` (gradient-free, fully ``keras.ops`` forward path).

Each layer is exercised for construction (incl. ``ValueError`` paths), a forward
pass, ``compute_output_shape`` agreement, and a full ``.keras`` serialization
round-trip.
"""

import os
import keras
import numpy as np
import pytest

from dl_techniques.layers.physics.lagrange_layer import (
    LagrangianNeuralNetworkLayer,
)
from dl_techniques.layers.physics.approximate_lagrange_layer import (
    ApproximatedLNNLayer,
)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

COORD_DIM = 2
BATCH = 4


@pytest.fixture
def sample_inputs():
    """Two random (batch, coord_dim) tensors for [q, q_dot]."""
    rng = np.random.default_rng(42)
    q = rng.standard_normal((BATCH, COORD_DIM)).astype("float32")
    q_dot = rng.standard_normal((BATCH, COORD_DIM)).astype("float32")
    return q, q_dot


# ---------------------------------------------------------------------
# ApproximatedLNNLayer
# ---------------------------------------------------------------------

class TestApproximatedLNNLayer:
    """Tests for the gradient-free approximated LNN layer."""

    def test_construction(self):
        layer = ApproximatedLNNLayer(hidden_dims=[16, 16])
        assert layer.hidden_dims == [16, 16]
        assert layer.activation == "softplus"

    def test_invalid_hidden_dims_empty(self):
        with pytest.raises(ValueError):
            ApproximatedLNNLayer(hidden_dims=[])

    def test_invalid_hidden_dims_nonpositive(self):
        with pytest.raises(ValueError):
            ApproximatedLNNLayer(hidden_dims=[16, 0])

    def test_forward_pass(self, sample_inputs):
        q, q_dot = sample_inputs
        layer = ApproximatedLNNLayer(hidden_dims=[16])
        out = layer([q, q_dot])
        assert out.shape == (BATCH, COORD_DIM)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(out)))

    def test_compute_output_shape(self):
        layer = ApproximatedLNNLayer(hidden_dims=[16])
        out_shape = layer.compute_output_shape([(None, COORD_DIM), (None, COORD_DIM)])
        assert out_shape == (None, COORD_DIM)

    def test_compute_output_shape_matches_call(self, sample_inputs):
        q, q_dot = sample_inputs
        layer = ApproximatedLNNLayer(hidden_dims=[16])
        out = layer([q, q_dot])
        computed = layer.compute_output_shape([q.shape, q_dot.shape])
        assert tuple(out.shape) == tuple(computed)

    def test_serialization_round_trip(self, sample_inputs, tmp_path):
        q, q_dot = sample_inputs

        q_in = keras.Input(shape=(COORD_DIM,), name="q")
        q_dot_in = keras.Input(shape=(COORD_DIM,), name="q_dot")
        out = ApproximatedLNNLayer(hidden_dims=[16], name="lnn")([q_in, q_dot_in])
        model = keras.Model([q_in, q_dot_in], out)

        y_before = model([q, q_dot])

        path = os.path.join(tmp_path, "approx_lnn.keras")
        model.save(path)
        reloaded = keras.models.load_model(
            path, custom_objects={"ApproximatedLNNLayer": ApproximatedLNNLayer}
        )
        y_after = reloaded([q, q_dot])

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y_before),
            keras.ops.convert_to_numpy(y_after),
            rtol=1e-6, atol=1e-6,
        )

    def test_get_config(self):
        layer = ApproximatedLNNLayer(hidden_dims=[8, 4], activation="tanh")
        config = layer.get_config()
        assert config["hidden_dims"] == [8, 4]
        assert config["activation"] == "tanh"
        rebuilt = ApproximatedLNNLayer.from_config(config)
        assert rebuilt.hidden_dims == [8, 4]
        assert rebuilt.activation == "tanh"


# ---------------------------------------------------------------------
# LagrangianNeuralNetworkLayer (TensorFlow-backend forward path)
# ---------------------------------------------------------------------

class TestLagrangianNeuralNetworkLayer:
    """Tests for the gradient-tape-based LNN layer."""

    def test_construction(self):
        layer = LagrangianNeuralNetworkLayer(hidden_dims=[16])
        assert layer.hidden_dims == [16]
        assert layer.activation == "softplus"

    def test_invalid_hidden_dims_empty(self):
        with pytest.raises(ValueError):
            LagrangianNeuralNetworkLayer(hidden_dims=[])

    def test_invalid_hidden_dims_nonpositive(self):
        with pytest.raises(ValueError):
            LagrangianNeuralNetworkLayer(hidden_dims=[-3])

    def test_forward_pass(self, sample_inputs):
        import tensorflow as tf
        q, q_dot = sample_inputs
        layer = LagrangianNeuralNetworkLayer(hidden_dims=[16])
        out = layer([tf.convert_to_tensor(q), tf.convert_to_tensor(q_dot)])
        assert tuple(out.shape) == (BATCH, COORD_DIM)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(out)))

    def test_compute_output_shape(self):
        layer = LagrangianNeuralNetworkLayer(hidden_dims=[16])
        out_shape = layer.compute_output_shape([(None, COORD_DIM), (None, COORD_DIM)])
        assert out_shape == (None, COORD_DIM)

    def test_get_config(self):
        layer = LagrangianNeuralNetworkLayer(hidden_dims=[8], activation="softplus")
        config = layer.get_config()
        assert config["hidden_dims"] == [8]
        assert config["activation"] == "softplus"
        rebuilt = LagrangianNeuralNetworkLayer.from_config(config)
        assert rebuilt.hidden_dims == [8]

    def test_serialization_round_trip(self, sample_inputs, tmp_path):
        """Full save/load round-trip with identical outputs.

        The forward path relies on ``tf.GradientTape``; the functional model is
        traced and saved/reloaded on the TensorFlow backend.
        """
        import tensorflow as tf
        q, q_dot = sample_inputs
        q_t, q_dot_t = tf.convert_to_tensor(q), tf.convert_to_tensor(q_dot)

        q_in = keras.Input(shape=(COORD_DIM,), name="q")
        q_dot_in = keras.Input(shape=(COORD_DIM,), name="q_dot")
        out = LagrangianNeuralNetworkLayer(hidden_dims=[16], name="lnn")([q_in, q_dot_in])
        model = keras.Model([q_in, q_dot_in], out)

        y_before = model([q_t, q_dot_t])

        path = os.path.join(tmp_path, "lnn.keras")
        model.save(path)
        reloaded = keras.models.load_model(
            path,
            custom_objects={
                "LagrangianNeuralNetworkLayer": LagrangianNeuralNetworkLayer
            },
        )
        y_after = reloaded([q_t, q_dot_t])

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y_before),
            keras.ops.convert_to_numpy(y_after),
            rtol=1e-6, atol=1e-6,
        )
