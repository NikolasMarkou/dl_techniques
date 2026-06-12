"""Scoped tests for the THERA EDSR-baseline feature backbone (step 4)."""

import os
import tempfile

import keras
import numpy as np
import pytest

from dl_techniques.models.thera.edsr_backbone import EDSRBackbone, EDSRResidualBlock


# ---------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def baseline_kwargs():
    """edsr-baseline defaults (THERA: num_feats=64, num_blocks=16)."""
    return dict(num_feats=64, num_blocks=16)


@pytest.fixture
def small_backbone():
    """A tiny backbone for fast round-trip / config tests."""
    return EDSRBackbone(num_feats=8, num_blocks=2, kernel_size=3, res_scale=0.1)


# ---------------------------------------------------------------------
# 1. shape-preserving forward
# ---------------------------------------------------------------------


def test_forward_shape_preserving(baseline_kwargs):
    backbone = EDSRBackbone(**baseline_kwargs)
    x = keras.random.normal((2, 24, 24, 3))
    y = backbone(x)
    assert tuple(y.shape) == (2, 24, 24, 64)
    assert np.all(np.isfinite(keras.ops.convert_to_numpy(y)))


def test_compute_output_shape(baseline_kwargs):
    backbone = EDSRBackbone(**baseline_kwargs)
    assert backbone.compute_output_shape((2, 24, 24, 3)) == (2, 24, 24, 64)
    assert backbone.compute_output_shape((None, None, None, 3)) == (None, None, None, 64)


# ---------------------------------------------------------------------
# 2. non-square input
# ---------------------------------------------------------------------


def test_non_square_input():
    backbone = EDSRBackbone(num_feats=64, num_blocks=4)
    x = keras.random.normal((1, 16, 24, 3))
    y = backbone(x)
    assert tuple(y.shape) == (1, 16, 24, 64)
    assert np.all(np.isfinite(keras.ops.convert_to_numpy(y)))


# ---------------------------------------------------------------------
# 3. .keras round-trip (via a tiny functional Model)
# ---------------------------------------------------------------------


def test_keras_roundtrip():
    inp = keras.Input(shape=(None, None, 3))
    backbone = EDSRBackbone(num_feats=16, num_blocks=3, res_scale=0.1)
    out = backbone(inp)
    model = keras.Model(inp, out)

    x = keras.random.normal((2, 20, 20, 3))
    y_before = keras.ops.convert_to_numpy(model(x))

    # trainable params must exist (weights actually created).
    n_params = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    assert n_params > 0

    # capture a sample conv kernel for the weight-survival assertion.
    kernel_before = keras.ops.convert_to_numpy(backbone.head_conv.kernel)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "edsr.keras")
        model.save(path)
        reloaded = keras.models.load_model(path)

    y_after = keras.ops.convert_to_numpy(reloaded(x))
    np.testing.assert_allclose(y_before, y_after, atol=1e-5, rtol=1e-5)

    # find the reloaded backbone layer and compare its head kernel.
    reloaded_backbone = next(
        l for l in reloaded.layers if isinstance(l, EDSRBackbone)
    )
    kernel_after = keras.ops.convert_to_numpy(reloaded_backbone.head_conv.kernel)
    np.testing.assert_allclose(kernel_before, kernel_after, atol=1e-6)


# ---------------------------------------------------------------------
# 4. get_config / from_config round-trip
# ---------------------------------------------------------------------


def test_config_roundtrip():
    backbone = EDSRBackbone(
        num_feats=32, num_blocks=5, kernel_size=5, res_scale=0.1, activation="gelu"
    )
    config = backbone.get_config()
    rebuilt = EDSRBackbone.from_config(config)
    assert rebuilt.num_feats == 32
    assert rebuilt.num_blocks == 5
    assert rebuilt.kernel_size == 5
    assert rebuilt.res_scale == 0.1
    # activation is now stored as a resolved Keras activation object; compare by
    # its serialized name rather than a bare string.
    assert keras.activations.serialize(rebuilt.activation) == "gelu"
    assert len(rebuilt.res_blocks) == 5


def test_residual_block_config_roundtrip():
    blk = EDSRResidualBlock(num_feats=16, kernel_size=3, res_scale=0.1, activation="relu")
    rebuilt = EDSRResidualBlock.from_config(blk.get_config())
    assert rebuilt.num_feats == 16
    assert rebuilt.kernel_size == 3
    assert rebuilt.res_scale == 0.1
    assert keras.activations.serialize(rebuilt.activation) == "relu"


# ---------------------------------------------------------------------
# 5. res_scale gating: res_scale=0.0 => output == head(x)
# ---------------------------------------------------------------------


def test_res_scale_zero_is_head_passthrough():
    # With res_scale=0.0 every residual block becomes identity, so the body is
    # body_conv(head(x)) but the residual-block contribution is zeroed; the
    # overall out = head(x) + body_conv(head(x)). To isolate the res_scale=>0
    # branch-zeroing on the *blocks*, compare each block in isolation.
    blk = EDSRResidualBlock(num_feats=8, res_scale=0.0)
    x = keras.random.normal((1, 6, 6, 8))
    y = blk(x)
    # residual branch fully zeroed => block is identity.
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(y), keras.ops.convert_to_numpy(x), atol=1e-6
    )


def test_res_scale_multiplies_body_branch():
    # Backbone with res_scale=0.0: every block is identity, so the feature
    # passed into body_conv equals head(x). Verify out == head(x) + body_conv(head(x)).
    backbone = EDSRBackbone(num_feats=8, num_blocks=3, res_scale=0.0)
    x = keras.random.normal((1, 6, 6, 3))
    backbone(x)  # build
    h = backbone.head_conv(x)
    expected = h + backbone.body_conv(h)
    y = backbone(x)
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(y),
        keras.ops.convert_to_numpy(expected),
        atol=1e-6,
    )


# ---------------------------------------------------------------------
# 6. param count sanity for the baseline
# ---------------------------------------------------------------------


def test_baseline_param_count_reasonable(baseline_kwargs):
    backbone = EDSRBackbone(**baseline_kwargs)
    backbone.build((None, None, None, 3))
    n_params = int(sum(np.prod(w.shape) for w in backbone.trainable_weights))
    assert n_params > 100_000


def test_import_smoke():
    from dl_techniques.models.thera.edsr_backbone import EDSRBackbone as _E  # noqa: F401


# ---------------------------------------------------------------------
# 7. iter-2 compliance: callable-activation serialization round-trip
# ---------------------------------------------------------------------


def test_activation_callable_roundtrip():
    # Build with a CALLABLE activation (not a string).
    backbone = EDSRBackbone(
        num_feats=32, num_blocks=2, activation=keras.activations.relu
    )
    x = keras.random.normal((1, 8, 8, 3))
    y_before = backbone(x)
    assert tuple(y_before.shape) == (1, 8, 8, 32)

    config = backbone.get_config()
    # get_config must serialize the activation to a serialized form (string name
    # for a built-in), NOT leak a bare python function object.
    assert config["activation"] == "relu"
    assert not callable(config["activation"])

    # serialize_keras_object round-trip exercises get_config + from_config.
    serialized = keras.saving.serialize_keras_object(backbone)
    rebuilt = keras.saving.deserialize_keras_object(serialized)

    y_after = rebuilt(x)
    assert tuple(y_after.shape) == (1, 8, 8, 32)
    # the reconstructed activation resolves back to relu.
    assert keras.activations.serialize(rebuilt.activation) == "relu"


def test_kernel_size_validation():
    with pytest.raises(ValueError):
        EDSRBackbone(kernel_size=0)


def test_compute_output_shape_edsr():
    backbone = EDSRBackbone(num_feats=32)
    assert backbone.compute_output_shape((2, 16, 16, 3)) == (2, 16, 16, 32)
