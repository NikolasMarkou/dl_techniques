"""
Test suite for GMMLayer (differentiable Gaussian Mixture Model layer).

Covers initialization (incl. the default 'orthonormal' regression for D-001),
build/shape, both output modes, properties, serialization round-trip, the
isometric-kernel add_loss, and a training smoke test. Backend-agnostic
(keras.ops only) per repo convention.
"""

import os
import tempfile
import pytest
import numpy as np
import keras
from keras import ops
from typing import Dict, Any

from dl_techniques.layers.mixtures.gmm import GMMLayer


# --------------------------------------------------------------------- fixtures

@pytest.fixture
def random_seed() -> int:
    return 42


@pytest.fixture
def basic_config() -> Dict[str, Any]:
    """Basic config using a standard initializer for general tests."""
    return {
        "n_components": 4,
        "temperature": 1.0,
        "isometric_regularizer_strength": 0.01,
        "output_mode": "assignments",
        "cluster_axis": -1,
        "mean_initializer": "glorot_normal",  # standard initializer for tests
    }


@pytest.fixture
def sample_data_2d() -> keras.KerasTensor:
    np.random.seed(42)
    data = np.random.normal(0, 1, (32, 64)).astype(np.float32)
    return keras.ops.convert_to_tensor(data)


@pytest.fixture
def sample_data_4d() -> keras.KerasTensor:
    np.random.seed(42)
    data = np.random.normal(0, 1, (8, 28, 28, 3)).astype(np.float32)
    return keras.ops.convert_to_tensor(data)


# --------------------------------------------------------------- initialization

class TestGMMLayerInitialization:

    def test_valid_initialization(self, basic_config: Dict[str, Any]) -> None:
        layer = GMMLayer(**basic_config)
        assert layer.n_components == 4
        assert layer.temperature == 1.0
        assert layer.isometric_regularizer_strength == 0.01
        assert layer.output_mode == "assignments"
        assert layer.built is False
        assert layer.means is None

    def test_default_orthonormal_initialization(self) -> None:
        """Regression (D-001): default 'orthonormal' must NOT raise in __init__."""
        layer = GMMLayer(n_components=4)
        # 'orthonormal' is kept as a string and resolved lazily in build()
        assert layer.mean_initializer == "orthonormal"
        assert layer.built is False

    @pytest.mark.parametrize("param,value,msg", [
        ("n_components", 0, "n_components must be a positive integer"),
        ("n_components", -3, "n_components must be a positive integer"),
        ("temperature", 0.0, "temperature must be positive"),
        ("temperature", -1.0, "temperature must be positive"),
        ("isometric_regularizer_strength", -0.1, "isometric_regularizer_strength must be non-negative"),
        ("variance_floor", 0.0, "variance_floor must be positive"),
        ("variance_floor", -1e-3, "variance_floor must be positive"),
        ("output_mode", "bogus", "output_mode must be"),
    ])
    def test_invalid_initialization(self, param: str, value: Any, msg: str) -> None:
        kwargs = {"n_components": 4, param: value}
        with pytest.raises(ValueError, match=msg):
            GMMLayer(**kwargs)


# ----------------------------------------------------------------- build/shapes

class TestGMMLayerShapes:

    def test_weight_shapes_after_build(self, basic_config: Dict[str, Any], sample_data_2d) -> None:
        layer = GMMLayer(**basic_config)
        _ = layer(sample_data_2d)  # triggers build
        assert layer.built is True
        feat = int(sample_data_2d.shape[-1])
        assert tuple(layer.means.shape) == (4, feat)
        assert tuple(layer.log_variances.shape) == (4, feat)
        assert tuple(layer.mixture_logits.shape) == (4,)

    def test_compute_output_shape_assignments(self, basic_config: Dict[str, Any]) -> None:
        layer = GMMLayer(**basic_config)
        assert layer.compute_output_shape((None, 64)) == (None, 4)

    def test_compute_output_shape_mixture(self) -> None:
        layer = GMMLayer(n_components=4, output_mode="mixture", mean_initializer="glorot_normal")
        assert layer.compute_output_shape((None, 64)) == (None, 64)


# ---------------------------------------------------------------- forward pass

class TestGMMLayerForwardPass:

    def test_assignments_output(self, basic_config: Dict[str, Any], sample_data_2d) -> None:
        layer = GMMLayer(**basic_config)
        out = layer(sample_data_2d)
        assert tuple(out.shape) == (32, 4)
        out_np = ops.convert_to_numpy(out)
        # responsibilities are a softmax -> rows sum to 1, all in [0, 1]
        np.testing.assert_allclose(out_np.sum(axis=-1), np.ones(32), rtol=1e-5, atol=1e-5)
        assert out_np.min() >= 0.0 and out_np.max() <= 1.0

    def test_mixture_output(self, sample_data_2d) -> None:
        layer = GMMLayer(n_components=4, output_mode="mixture", mean_initializer="glorot_normal")
        out = layer(sample_data_2d)
        assert tuple(out.shape) == tuple(sample_data_2d.shape)
        assert np.all(np.isfinite(ops.convert_to_numpy(out)))

    def test_4d_assignments(self, basic_config: Dict[str, Any], sample_data_4d) -> None:
        layer = GMMLayer(**basic_config)
        out = layer(sample_data_4d)
        # cluster_axis=-1 (3 channels) -> last dim becomes n_components
        assert tuple(out.shape) == (8, 28, 28, 4)

    def test_training_flag_both(self, basic_config: Dict[str, Any], sample_data_2d) -> None:
        layer = GMMLayer(**basic_config)
        o_train = layer(sample_data_2d, training=True)
        o_inf = layer(sample_data_2d, training=False)
        assert tuple(o_train.shape) == tuple(o_inf.shape) == (32, 4)


# ------------------------------------------------------------------- properties

class TestGMMLayerProperties:

    def test_properties_after_build(self, basic_config: Dict[str, Any], sample_data_2d) -> None:
        layer = GMMLayer(**basic_config)
        _ = layer(sample_data_2d)
        feat = int(sample_data_2d.shape[-1])
        assert tuple(layer.component_means.shape) == (4, feat)
        var = ops.convert_to_numpy(layer.component_variances)
        assert var.shape == (4, feat)
        assert np.all(var >= layer.variance_floor - 1e-9)  # floored
        w = ops.convert_to_numpy(layer.mixture_weights)
        np.testing.assert_allclose(w.sum(), 1.0, rtol=1e-6, atol=1e-6)

    def test_reset_parameters(self, basic_config: Dict[str, Any], sample_data_2d) -> None:
        layer = GMMLayer(**basic_config)
        _ = layer(sample_data_2d)
        layer.reset_parameters()
        # log_variances reset to zeros, mixture_logits to zeros (uniform)
        np.testing.assert_allclose(ops.convert_to_numpy(layer.log_variances),
                                   np.zeros_like(ops.convert_to_numpy(layer.log_variances)),
                                   atol=1e-7)


# ---------------------------------------------------------------- serialization

class TestGMMLayerSerialization:

    def test_get_config_keys(self, basic_config: Dict[str, Any]) -> None:
        layer = GMMLayer(**basic_config)
        config = layer.get_config()
        for key in ["n_components", "temperature", "isometric_regularizer_strength",
                    "variance_floor", "output_mode", "cluster_axis",
                    "mean_initializer", "log_variance_initializer",
                    "mean_regularizer", "random_seed"]:
            assert key in config

    def test_from_config_roundtrip(self, basic_config: Dict[str, Any]) -> None:
        layer = GMMLayer(**basic_config)
        config = layer.get_config()
        layer2 = GMMLayer.from_config(config)
        assert layer2.n_components == layer.n_components
        assert layer2.temperature == layer.temperature
        assert layer2.output_mode == layer.output_mode

    def _roundtrip_model(self, layer_kwargs: Dict[str, Any], sample) -> None:
        inp = keras.Input(shape=(sample.shape[-1],))
        out = GMMLayer(**layer_kwargs, name="gmm")(inp)
        model = keras.Model(inp, out)
        y0 = model(sample)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "gmm.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
        np.testing.assert_allclose(
            ops.convert_to_numpy(y0),
            ops.convert_to_numpy(loaded(sample)),
            rtol=1e-6, atol=1e-6,
        )

    def test_model_save_load_glorot(self, sample_data_2d) -> None:
        """SC4: serialization round-trip with a standard initializer."""
        self._roundtrip_model(
            {"n_components": 4, "mean_initializer": "glorot_normal"}, sample_data_2d
        )

    def test_model_save_load_default_orthonormal(self, sample_data_2d) -> None:
        """SC4 + D-002: round-trip with the default 'orthonormal' initializer."""
        self._roundtrip_model({"n_components": 4}, sample_data_2d)


# -------------------------------------------------------------------- add_loss

class TestGMMLayerIntegration:

    def test_losses_non_empty_on_training(self, sample_data_2d) -> None:
        """SC5: isometric add_loss registered + finite when training=True."""
        inp = keras.Input(shape=(sample_data_2d.shape[-1],))
        out = GMMLayer(
            n_components=4,
            isometric_regularizer_strength=0.1,
            mean_initializer="glorot_normal",
            log_variance_initializer=keras.initializers.RandomNormal(stddev=1.0, seed=7),
            name="gmm",
        )(inp)
        model = keras.Model(inp, out)
        _ = model(sample_data_2d, training=True)
        assert len(model.losses) > 0
        loss_vals = [float(ops.convert_to_numpy(l)) for l in model.losses]
        assert all(np.isfinite(v) for v in loss_vals)
        # with random (anisotropic) log-variances the penalty is strictly positive
        assert sum(loss_vals) > 0.0

    def test_no_loss_when_strength_zero(self, sample_data_2d) -> None:
        inp = keras.Input(shape=(sample_data_2d.shape[-1],))
        out = GMMLayer(
            n_components=4,
            isometric_regularizer_strength=0.0,
            mean_initializer="glorot_normal",
            name="gmm",
        )(inp)
        model = keras.Model(inp, out)
        _ = model(sample_data_2d, training=True)
        assert len(model.losses) == 0

    def test_training_step_runs(self, sample_data_2d) -> None:
        """Backend-agnostic gradient-flow smoke: one fit step completes."""
        inp = keras.Input(shape=(sample_data_2d.shape[-1],))
        out = GMMLayer(n_components=4, mean_initializer="glorot_normal", name="gmm")(inp)
        model = keras.Model(inp, out)
        model.compile(optimizer="adam", loss="mse")
        target = np.random.RandomState(0).uniform(size=(32, 4)).astype(np.float32)
        history = model.fit(ops.convert_to_numpy(sample_data_2d), target,
                            epochs=1, batch_size=16, verbose=0)
        assert np.isfinite(history.history["loss"][-1])
