"""
Test suite for ZeroCenteredBandRMSNorm layer.

Mirrors test_zero_centered_rms_norm.py structurally but adapts for the
band-constrained scalar scale (single trainable `band_param`) and the
combined zero-centering + band-RMS-scaling math.
"""

import pytest
import tensorflow as tf
import numpy as np
import keras
from typing import Dict, Any, Union, Tuple

from dl_techniques.layers.norms.zero_centered_band_rms_norm import (
    ZeroCenteredBandRMSNorm,
)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def sample_inputs_2d() -> tf.Tensor:
    tf.random.set_seed(42)
    return tf.random.normal((4, 512))


@pytest.fixture
def sample_inputs_3d() -> tf.Tensor:
    tf.random.set_seed(42)
    return tf.random.normal((2, 128, 768))


@pytest.fixture
def sample_inputs_4d() -> tf.Tensor:
    tf.random.set_seed(42)
    return tf.random.normal((2, 8, 8, 64))


@pytest.fixture
def sample_inputs_non_zero_mean() -> tf.Tensor:
    tf.random.set_seed(42)
    return tf.random.normal((4, 512)) + 2.5


@pytest.fixture
def default_params() -> Dict[str, Any]:
    return {
        "max_band_width": 0.1,
        "axis": -1,
        "epsilon": 1e-7,
        "band_initializer": "zeros",
        "band_regularizer": None,
    }


# ---------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------


def test_initialization_defaults() -> None:
    norm = ZeroCenteredBandRMSNorm()
    assert norm.max_band_width == 0.1
    assert norm.axis == -1
    assert norm.epsilon == 1e-7
    assert isinstance(norm.band_initializer, keras.initializers.Zeros)
    # Default regularizer is L2(1e-5) when None is passed
    assert norm.band_regularizer is not None


def test_initialization_custom(default_params: Dict[str, Any]) -> None:
    norm = ZeroCenteredBandRMSNorm(**default_params)
    assert norm.max_band_width == default_params["max_band_width"]
    assert norm.axis == default_params["axis"]
    assert norm.epsilon == default_params["epsilon"]
    # None override is replaced by default L2(1e-5)
    assert norm.band_regularizer is not None


def test_initialization_multi_axis() -> None:
    norm = ZeroCenteredBandRMSNorm(max_band_width=0.2, axis=(-2, -1))
    assert norm.axis == (-2, -1)


def test_invalid_max_band_width() -> None:
    with pytest.raises(ValueError, match="max_band_width must be between 0 and 1"):
        ZeroCenteredBandRMSNorm(max_band_width=0.0)
    with pytest.raises(ValueError, match="max_band_width must be between 0 and 1"):
        ZeroCenteredBandRMSNorm(max_band_width=1.0)
    with pytest.raises(ValueError, match="max_band_width must be between 0 and 1"):
        ZeroCenteredBandRMSNorm(max_band_width=-0.1)


def test_invalid_epsilon() -> None:
    with pytest.raises(ValueError, match="epsilon must be positive"):
        ZeroCenteredBandRMSNorm(epsilon=0.0)
    with pytest.raises(ValueError, match="epsilon must be positive"):
        ZeroCenteredBandRMSNorm(epsilon=-1e-6)


def test_invalid_axis_type() -> None:
    with pytest.raises(TypeError, match="axis must be int or tuple of ints"):
        ZeroCenteredBandRMSNorm(axis="invalid")
    with pytest.raises(TypeError, match="All elements in axis must be integers"):
        ZeroCenteredBandRMSNorm(axis=(1, "x"))


# ---------------------------------------------------------------------
# Shape preservation
# ---------------------------------------------------------------------


def test_output_shape_2d(sample_inputs_2d, default_params) -> None:
    norm = ZeroCenteredBandRMSNorm(**default_params)
    out = norm(sample_inputs_2d)
    assert out.shape == sample_inputs_2d.shape


def test_output_shape_3d(sample_inputs_3d, default_params) -> None:
    norm = ZeroCenteredBandRMSNorm(**default_params)
    out = norm(sample_inputs_3d)
    assert out.shape == sample_inputs_3d.shape


def test_output_shape_4d_multi_axis(sample_inputs_4d) -> None:
    norm = ZeroCenteredBandRMSNorm(axis=(-2, -1), max_band_width=0.2)
    out = norm(sample_inputs_4d)
    assert out.shape == sample_inputs_4d.shape


def test_compute_output_shape(sample_inputs_3d, default_params) -> None:
    norm = ZeroCenteredBandRMSNorm(**default_params)
    assert norm.compute_output_shape(sample_inputs_3d.shape) == sample_inputs_3d.shape


# ---------------------------------------------------------------------
# Weight contract
# ---------------------------------------------------------------------


def test_single_scalar_band_param(sample_inputs_2d, default_params) -> None:
    """BandRMS variants must have exactly ONE trainable scalar weight regardless of input dim."""
    norm = ZeroCenteredBandRMSNorm(**default_params)
    _ = norm(sample_inputs_2d)
    # band_param is shape ()
    assert norm.band_param.shape == ()
    # Exactly 1 trainable weight
    trainable_param_count = sum(int(np.prod(w.shape)) for w in norm.trainable_weights)
    assert trainable_param_count == 1


def test_param_count_independent_of_feature_dim() -> None:
    for d in (8, 64, 512, 4096):
        norm = ZeroCenteredBandRMSNorm()
        x = tf.zeros((2, d))
        _ = norm(x)
        count = sum(int(np.prod(w.shape)) for w in norm.trainable_weights)
        assert count == 1, f"d={d} should yield 1 param, got {count}"


# ---------------------------------------------------------------------
# Mathematical correctness
# ---------------------------------------------------------------------


def test_zero_mean_output_property(sample_inputs_non_zero_mean) -> None:
    """Output along normalized axis must have ~zero mean even with non-zero-mean input."""
    norm = ZeroCenteredBandRMSNorm(max_band_width=0.1, epsilon=1e-7)
    out = norm(sample_inputs_non_zero_mean)
    out_np = keras.ops.convert_to_numpy(out)
    mean_along_axis = np.mean(out_np, axis=-1)
    np.testing.assert_allclose(
        mean_along_axis,
        np.zeros_like(mean_along_axis),
        atol=1e-5,
        err_msg="Output mean along normalized axis must be ~0.",
    )


def test_band_bound_property(sample_inputs_non_zero_mean) -> None:
    """Output per-sample RMS must lie in [1-alpha, 1] band."""
    alpha = 0.1
    norm = ZeroCenteredBandRMSNorm(max_band_width=alpha, epsilon=1e-7)
    _ = norm(sample_inputs_non_zero_mean)  # build
    # At init band_param=0 → sigmoid(0)=0.5 → scale = (1-α) + α*0.5 = 1 - α/2
    out = norm(sample_inputs_non_zero_mean)
    out_np = keras.ops.convert_to_numpy(out)
    per_sample_rms = np.sqrt(np.mean(out_np ** 2, axis=-1))
    # Allow small numerical slack
    assert np.all(per_sample_rms >= (1.0 - alpha) - 1e-4), (
        f"Output RMS below lower band: min={per_sample_rms.min()}, expected >= {1.0 - alpha}"
    )
    assert np.all(per_sample_rms <= 1.0 + 1e-4), (
        f"Output RMS above upper band: max={per_sample_rms.max()}, expected <= 1.0"
    )


def test_mathematical_correctness_at_init(sample_inputs_non_zero_mean) -> None:
    """At init (band_param=0): scale = (1 - α/2). Test exact match vs manual computation."""
    alpha = 0.1
    norm = ZeroCenteredBandRMSNorm(
        max_band_width=alpha,
        epsilon=1e-7,
        band_initializer="zeros",
    )

    x_np = sample_inputs_non_zero_mean.numpy().astype(np.float32)
    mean = np.mean(x_np, axis=-1, keepdims=True)
    centered = x_np - mean
    rms = np.maximum(np.sqrt(np.mean(centered ** 2, axis=-1, keepdims=True) + 1e-7), 1e-7)
    normalized = centered / rms
    # band_param=0 → sigmoid(0)=0.5
    scale = (1.0 - alpha) + alpha * 0.5
    expected = normalized * scale

    out = norm(sample_inputs_non_zero_mean)
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(out),
        expected,
        rtol=1e-5,
        atol=1e-6,
    )


def test_max_band_width_extremes() -> None:
    """Very small alpha → output ~= unit-norm; large alpha → wider band possible."""
    x = tf.random.normal((4, 128))

    tight = ZeroCenteredBandRMSNorm(max_band_width=0.01, epsilon=1e-7)
    out_tight = keras.ops.convert_to_numpy(tight(x))
    rms_tight = np.sqrt(np.mean(out_tight ** 2, axis=-1))
    # Tight band: rms in [0.99, 1.0]
    assert np.all((rms_tight >= 0.99 - 1e-4) & (rms_tight <= 1.0 + 1e-4))

    wide = ZeroCenteredBandRMSNorm(max_band_width=0.5, epsilon=1e-7)
    out_wide = keras.ops.convert_to_numpy(wide(x))
    rms_wide = np.sqrt(np.mean(out_wide ** 2, axis=-1))
    # Wide band: rms in [0.5, 1.0]
    assert np.all((rms_wide >= 0.5 - 1e-4) & (rms_wide <= 1.0 + 1e-4))


# ---------------------------------------------------------------------
# Training / inference
# ---------------------------------------------------------------------


def test_training_vs_inference(sample_inputs_3d, default_params) -> None:
    norm = ZeroCenteredBandRMSNorm(**default_params)
    out_train = norm(sample_inputs_3d, training=True)
    out_eval = norm(sample_inputs_3d, training=False)
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(out_train),
        keras.ops.convert_to_numpy(out_eval),
        rtol=1e-6,
    )


def test_deterministic(sample_inputs_2d, default_params) -> None:
    norm = ZeroCenteredBandRMSNorm(**default_params)
    a = norm(sample_inputs_2d)
    b = norm(sample_inputs_2d)
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(a),
        keras.ops.convert_to_numpy(b),
        rtol=1e-6,
    )


# ---------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------


def test_get_config(default_params) -> None:
    norm = ZeroCenteredBandRMSNorm(**default_params)
    cfg = norm.get_config()
    for k in ("max_band_width", "axis", "epsilon", "band_initializer", "band_regularizer"):
        assert k in cfg
    assert cfg["max_band_width"] == default_params["max_band_width"]
    assert cfg["epsilon"] == default_params["epsilon"]


def test_from_config(default_params) -> None:
    original = ZeroCenteredBandRMSNorm(**default_params)
    cfg = original.get_config()
    restored = ZeroCenteredBandRMSNorm.from_config(cfg)
    assert restored.max_band_width == original.max_band_width
    assert restored.axis == original.axis
    assert restored.epsilon == original.epsilon


def test_save_load_model_roundtrip(sample_inputs_3d, default_params, tmp_path) -> None:
    inputs = keras.Input(shape=sample_inputs_3d.shape[1:])
    outputs = ZeroCenteredBandRMSNorm(**default_params)(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)

    original = model(sample_inputs_3d, training=False).numpy()
    path = str(tmp_path / "zcbrms.keras")
    model.save(path)

    loaded = keras.models.load_model(path)
    reloaded = loaded(sample_inputs_3d, training=False).numpy()
    np.testing.assert_allclose(original, reloaded, rtol=1e-6, atol=1e-7)


def test_save_load_weights(sample_inputs_2d, default_params, tmp_path) -> None:
    inputs = keras.Input(shape=sample_inputs_2d.shape[1:])
    outputs = ZeroCenteredBandRMSNorm(**default_params)(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)

    before = model(sample_inputs_2d, training=False).numpy()
    path = str(tmp_path / "w.weights.h5")
    model.save_weights(path)
    model.load_weights(path)
    after = model(sample_inputs_2d, training=False).numpy()
    np.testing.assert_allclose(before, after, rtol=1e-6, atol=1e-7)


# ---------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------


def test_gradient_flow(sample_inputs_3d, default_params) -> None:
    norm = ZeroCenteredBandRMSNorm(**default_params)
    with tf.GradientTape() as tape:
        out = norm(sample_inputs_3d, training=True)
        loss = keras.ops.mean(out ** 2)
    grads = tape.gradient(loss, norm.trainable_variables)
    assert len(grads) == 1
    assert grads[0] is not None
    assert not keras.ops.any(keras.ops.isnan(grads[0]))


def test_gradient_flow_through_input(sample_inputs_2d) -> None:
    norm = ZeroCenteredBandRMSNorm(max_band_width=0.1, epsilon=1e-7)
    var_input = tf.Variable(sample_inputs_2d)
    with tf.GradientTape() as tape:
        out = norm(var_input, training=True)
        loss = keras.ops.mean(out)
    g = tape.gradient(loss, var_input)
    assert g is not None
    assert not keras.ops.any(keras.ops.isnan(g))


# ---------------------------------------------------------------------
# Mixed precision / numerical
# ---------------------------------------------------------------------


def test_mixed_precision_fp16(sample_inputs_2d) -> None:
    original_policy = keras.mixed_precision.global_policy()
    keras.mixed_precision.set_global_policy("mixed_float16")
    try:
        norm = ZeroCenteredBandRMSNorm(max_band_width=0.1, epsilon=1e-5)
        x_fp16 = tf.cast(sample_inputs_2d, tf.float16)
        out = norm(x_fp16)
        assert out.dtype == tf.float16
        assert not keras.ops.any(keras.ops.isnan(out))
        assert keras.ops.all(keras.ops.isfinite(out))
    finally:
        keras.mixed_precision.set_global_policy(original_policy)


def test_numerical_stability_extremes() -> None:
    norm = ZeroCenteredBandRMSNorm(max_band_width=0.1, epsilon=1e-6)
    for vals in [1e-10, 1e10, 0.0]:
        x = tf.constant([[vals, vals, vals]], dtype=tf.float32)
        out = norm(x)
        assert not keras.ops.any(keras.ops.isnan(out))
        assert keras.ops.all(keras.ops.isfinite(out))


def test_zero_input_safe() -> None:
    norm = ZeroCenteredBandRMSNorm(max_band_width=0.1, epsilon=1e-6)
    out = norm(tf.zeros((2, 8), dtype=tf.float32))
    assert not keras.ops.any(keras.ops.isnan(out))


# ---------------------------------------------------------------------
# Parametrized sweeps
# ---------------------------------------------------------------------


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2, 0.5, 0.9])
def test_various_band_widths(alpha: float, sample_inputs_2d) -> None:
    norm = ZeroCenteredBandRMSNorm(max_band_width=alpha, epsilon=1e-7)
    out = norm(sample_inputs_2d)
    assert out.shape == sample_inputs_2d.shape
    assert not keras.ops.any(keras.ops.isnan(out))


@pytest.mark.parametrize("epsilon", [1e-8, 1e-7, 1e-6, 1e-5])
def test_various_epsilons(epsilon: float, sample_inputs_2d) -> None:
    norm = ZeroCenteredBandRMSNorm(max_band_width=0.1, epsilon=epsilon)
    out = norm(sample_inputs_2d)
    assert out.shape == sample_inputs_2d.shape


# ---------------------------------------------------------------------
# Model integration
# ---------------------------------------------------------------------


def test_model_integration(sample_inputs_3d, default_params) -> None:
    inputs = keras.Input(shape=sample_inputs_3d.shape[1:])
    x = ZeroCenteredBandRMSNorm(**default_params)(inputs)
    model = keras.Model(inputs=inputs, outputs=x)
    out = model(sample_inputs_3d)
    assert out.shape == sample_inputs_3d.shape


def test_train_one_step(sample_inputs_2d, default_params) -> None:
    inputs = keras.Input(shape=sample_inputs_2d.shape[1:])
    x = ZeroCenteredBandRMSNorm(**default_params)(inputs)
    x = keras.layers.Dense(64, activation="relu")(x)
    outputs = keras.layers.Dense(10, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    labels = tf.random.uniform((sample_inputs_2d.shape[0],), 0, 10, dtype=tf.int32)
    history = model.fit(sample_inputs_2d, labels, epochs=1, verbose=0)
    assert len(history.history["loss"]) == 1


# ---------------------------------------------------------------------
# Comparison with sibling layers
# ---------------------------------------------------------------------


def test_differs_from_band_rms_on_nonzero_mean(sample_inputs_non_zero_mean) -> None:
    """ZeroCenteredBandRMSNorm must produce different output than BandRMS on biased input."""
    from dl_techniques.layers.norms.band_rms import BandRMS

    zc = ZeroCenteredBandRMSNorm(max_band_width=0.1, epsilon=1e-7, band_initializer="zeros")
    band = BandRMS(max_band_width=0.1, epsilon=1e-7, band_initializer="zeros")

    out_zc = keras.ops.convert_to_numpy(zc(sample_inputs_non_zero_mean))
    out_band = keras.ops.convert_to_numpy(band(sample_inputs_non_zero_mean))

    assert not np.allclose(out_zc, out_band, rtol=1e-3), (
        "ZeroCenteredBandRMSNorm must differ from BandRMS for non-zero-mean input."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
