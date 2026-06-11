"""
Test suite for the Temporal Convolutional Network layers.

Covers both ``TemporalBlock`` and ``TemporalConvNet``:
- initialization + every constructor ``ValueError`` validation case
- forward-pass output shape ``(batch, time, filters)``
- explicit ``build()`` materializes ALL sublayer weights, including the inner
  ``Conv1D`` children of every ``TemporalBlock`` (the canonical guard)
- ``compute_output_shape`` matches the actual forward output shape
- ``.keras`` save/load round-trip (wrapped in a ``keras.Model``) with numeric
  match
- gradient flow (all trainable grads non-None)
- training-mode dropout path
- edge cases: channels != filters (downsample), channels == filters (no
  downsample), and ``num_levels == 1``
"""

import pytest
import numpy as np
import keras
import tensorflow as tf

from dl_techniques.layers.time_series.temporal_convolutional_network import (
    TemporalBlock,
    TemporalConvNet,
)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def sample_inputs() -> tf.Tensor:
    """Sample input tensor of shape (batch, time, channels)."""
    tf.random.set_seed(42)
    return tf.random.normal((4, 16, 3))


@pytest.fixture
def block_params() -> dict:
    """Default parameters for a TemporalBlock."""
    return {
        "filters": 8,
        "kernel_size": 2,
        "dilation_rate": 1,
        "dropout_rate": 0.1,
        "activation": "relu",
    }


@pytest.fixture
def tcn_params() -> dict:
    """Default parameters for a TemporalConvNet."""
    return {
        "filters": 8,
        "kernel_size": 2,
        "num_levels": 3,
        "dropout_rate": 0.1,
        "activation": "relu",
    }


# ---------------------------------------------------------------------
# TemporalBlock — initialization + validation
# ---------------------------------------------------------------------

def test_block_initialization(block_params: dict) -> None:
    block = TemporalBlock(**block_params)
    assert block.filters == block_params["filters"]
    assert block.kernel_size == block_params["kernel_size"]
    assert block.dilation_rate == block_params["dilation_rate"]
    assert block.dropout_rate == block_params["dropout_rate"]
    assert block.activation == block_params["activation"]
    assert block.kernel_initializer == "he_normal"
    # Activation must live on the instance, not be created in call()
    assert block.act is not None


def test_block_invalid_filters() -> None:
    with pytest.raises(ValueError, match="filters must be positive"):
        TemporalBlock(filters=0, kernel_size=2, dilation_rate=1)
    with pytest.raises(ValueError, match="filters must be positive"):
        TemporalBlock(filters=-3, kernel_size=2, dilation_rate=1)


def test_block_invalid_kernel_size() -> None:
    with pytest.raises(ValueError, match="kernel_size must be positive"):
        TemporalBlock(filters=8, kernel_size=0, dilation_rate=1)


def test_block_invalid_dilation_rate() -> None:
    with pytest.raises(ValueError, match="dilation_rate must be positive"):
        TemporalBlock(filters=8, kernel_size=2, dilation_rate=0)


def test_block_invalid_dropout_rate() -> None:
    with pytest.raises(ValueError, match="dropout_rate must be in"):
        TemporalBlock(filters=8, kernel_size=2, dilation_rate=1, dropout_rate=-0.1)
    with pytest.raises(ValueError, match="dropout_rate must be in"):
        TemporalBlock(filters=8, kernel_size=2, dilation_rate=1, dropout_rate=1.5)


# ---------------------------------------------------------------------
# TemporalBlock — build / forward / shapes
# ---------------------------------------------------------------------

def test_block_forward_shape_downsample(sample_inputs: tf.Tensor, block_params: dict) -> None:
    """channels (3) != filters (8) -> downsample path."""
    block = TemporalBlock(**block_params)
    out = block(sample_inputs)
    assert out.shape == (4, 16, block_params["filters"])
    assert block.downsample is not None


def test_block_forward_shape_no_downsample(block_params: dict) -> None:
    """channels == filters -> no downsample (identity residual)."""
    params = dict(block_params)
    params["filters"] = 3
    block = TemporalBlock(**params)
    inputs = tf.random.normal((4, 16, 3))
    out = block(inputs)
    assert out.shape == (4, 16, 3)
    assert block.downsample is None


def test_block_build_materializes_convs(sample_inputs: tf.Tensor, block_params: dict) -> None:
    """build() must create the inner Conv1D kernels (the whole point)."""
    block = TemporalBlock(**block_params)
    assert len(block.weights) == 0
    block.build(sample_inputs.shape)
    assert block.built
    # Inner convs must have materialized kernels at build time
    assert len(block.conv1.weights) > 0
    assert len(block.conv2.weights) > 0
    assert block.conv1.built
    assert block.conv2.built
    # Downsample built since channels (3) != filters (8)
    assert block.downsample is not None
    assert len(block.downsample.weights) > 0
    assert len(block.weights) > 0


def test_block_compute_output_shape(block_params: dict) -> None:
    block = TemporalBlock(**block_params)
    input_shape = (4, 16, 3)
    cos = block.compute_output_shape(input_shape)
    assert cos == (4, 16, block_params["filters"])
    # Matches actual forward output
    out = block(tf.random.normal(input_shape))
    assert tuple(out.shape) == cos


# ---------------------------------------------------------------------
# TemporalConvNet — initialization + validation
# ---------------------------------------------------------------------

def test_tcn_initialization(tcn_params: dict) -> None:
    tcn = TemporalConvNet(**tcn_params)
    assert tcn.filters == tcn_params["filters"]
    assert tcn.kernel_size == tcn_params["kernel_size"]
    assert tcn.num_levels == tcn_params["num_levels"]
    assert tcn.dropout_rate == tcn_params["dropout_rate"]
    assert tcn.activation == tcn_params["activation"]
    assert len(tcn.blocks) == tcn_params["num_levels"]
    # Exponentially increasing dilations
    assert [b.dilation_rate for b in tcn.blocks] == [1, 2, 4]


def test_tcn_invalid_filters() -> None:
    with pytest.raises(ValueError, match="filters must be positive"):
        TemporalConvNet(filters=0)


def test_tcn_invalid_kernel_size() -> None:
    with pytest.raises(ValueError, match="kernel_size must be positive"):
        TemporalConvNet(filters=8, kernel_size=-1)


def test_tcn_invalid_num_levels() -> None:
    with pytest.raises(ValueError, match="num_levels must be positive"):
        TemporalConvNet(filters=8, num_levels=0)


def test_tcn_invalid_dropout_rate() -> None:
    with pytest.raises(ValueError, match="dropout_rate must be in"):
        TemporalConvNet(filters=8, dropout_rate=2.0)
    with pytest.raises(ValueError, match="dropout_rate must be in"):
        TemporalConvNet(filters=8, dropout_rate=-0.5)


# ---------------------------------------------------------------------
# TemporalConvNet — build / forward / shapes
# ---------------------------------------------------------------------

def test_tcn_forward_shape(sample_inputs: tf.Tensor, tcn_params: dict) -> None:
    tcn = TemporalConvNet(**tcn_params)
    out = tcn(sample_inputs)
    assert out.shape == (4, 16, tcn_params["filters"])


def test_tcn_build_materializes_all_block_convs(sample_inputs: tf.Tensor, tcn_params: dict) -> None:
    """The canonical guard: every block's inner Conv1Ds must build."""
    tcn = TemporalConvNet(**tcn_params)
    assert len(tcn.weights) == 0
    tcn.build(sample_inputs.shape)
    assert tcn.built
    assert len(tcn.weights) > 0
    for block in tcn.blocks:
        assert block.built
        assert block.conv1.built
        assert block.conv2.built
        assert len(block.conv1.weights) > 0
        assert len(block.conv2.weights) > 0


def test_tcn_compute_output_shape(tcn_params: dict) -> None:
    tcn = TemporalConvNet(**tcn_params)
    input_shape = (4, 16, 3)
    cos = tcn.compute_output_shape(input_shape)
    assert cos == (4, 16, tcn_params["filters"])
    out = tcn(tf.random.normal(input_shape))
    assert tuple(out.shape) == cos


def test_tcn_num_levels_one(sample_inputs: tf.Tensor) -> None:
    """num_levels=1 (single block) must build and run."""
    tcn = TemporalConvNet(filters=8, kernel_size=2, num_levels=1, dropout_rate=0.0)
    tcn.build(sample_inputs.shape)
    assert len(tcn.blocks) == 1
    out = tcn(sample_inputs)
    assert out.shape == (4, 16, 8)


# ---------------------------------------------------------------------
# Gradient flow + training mode
# ---------------------------------------------------------------------

def test_tcn_gradient_flow(sample_inputs: tf.Tensor, tcn_params: dict) -> None:
    tcn = TemporalConvNet(**tcn_params)
    with tf.GradientTape() as tape:
        out = tcn(sample_inputs, training=True)
        loss = tf.reduce_mean(out)
    grads = tape.gradient(loss, tcn.trainable_variables)
    assert len(tcn.trainable_variables) > 0
    assert all(g is not None for g in grads)


def test_tcn_training_mode_dropout(sample_inputs: tf.Tensor, tcn_params: dict) -> None:
    """The training=True dropout path must run without error."""
    params = dict(tcn_params)
    params["dropout_rate"] = 0.5
    tcn = TemporalConvNet(**params)
    out_train = tcn(sample_inputs, training=True)
    out_infer = tcn(sample_inputs, training=False)
    assert out_train.shape == out_infer.shape == (4, 16, params["filters"])


# ---------------------------------------------------------------------
# get_config key stability (additive-only invariant)
# ---------------------------------------------------------------------

def test_block_get_config_keys(block_params: dict) -> None:
    block = TemporalBlock(**block_params)
    config = block.get_config()
    expected = {
        "filters", "kernel_size", "dilation_rate",
        "dropout_rate", "activation", "kernel_initializer",
    }
    assert expected.issubset(set(config.keys()))


def test_tcn_get_config_keys(tcn_params: dict) -> None:
    tcn = TemporalConvNet(**tcn_params)
    config = tcn.get_config()
    expected = {"filters", "kernel_size", "num_levels", "dropout_rate", "activation"}
    assert expected.issubset(set(config.keys()))
    # kernel_initializer must NOT be a TCN key (D-004)
    assert "kernel_initializer" not in config


# ---------------------------------------------------------------------
# .keras save / load round-trip (canonical guard)
# ---------------------------------------------------------------------

def test_tcn_keras_roundtrip(sample_inputs: tf.Tensor, tcn_params: dict, tmp_path) -> None:
    inputs = keras.Input(shape=(16, 3))
    out = TemporalConvNet(**tcn_params)(inputs)
    model = keras.Model(inputs=inputs, outputs=out)

    # Forward pass BEFORE save (never save a never-called subclass)
    original = model(sample_inputs, training=False)

    save_path = str(tmp_path / "tcn_model.keras")
    model.save(save_path)
    loaded = keras.models.load_model(save_path)

    reloaded = loaded(sample_inputs, training=False)
    np.testing.assert_allclose(
        original.numpy(), reloaded.numpy(), atol=1e-6
    )


def test_tcn_keras_roundtrip_no_downsample(tmp_path) -> None:
    """channels == filters: residual identity path round-trips."""
    inputs = keras.Input(shape=(16, 8))
    out = TemporalConvNet(filters=8, kernel_size=2, num_levels=2, dropout_rate=0.0)(inputs)
    model = keras.Model(inputs=inputs, outputs=out)

    x = tf.random.normal((4, 16, 8))
    original = model(x, training=False)

    save_path = str(tmp_path / "tcn_model_no_ds.keras")
    model.save(save_path)
    loaded = keras.models.load_model(save_path)

    reloaded = loaded(x, training=False)
    np.testing.assert_allclose(original.numpy(), reloaded.numpy(), atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])
