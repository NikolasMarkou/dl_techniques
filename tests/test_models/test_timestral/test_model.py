"""Tests for Timestral model: forward pass, shapes, serialization, gradients."""

import keras
import numpy as np
import pytest

from dl_techniques.models.time_series.timestral import (
    TimestralConfig,
    TimestralModel,
    create_timestral_model,
    create_timestral_by_variant,
)


class TestTimestralModel:
    """Comprehensive test suite for TimestralModel."""

    @pytest.fixture
    def tiny_config(self):
        """Minimal config for fast testing."""
        return TimestralConfig(
            patch_size=8,
            d_model=64,
            num_heads=4,
            mlp_hidden_dim=128,
            block_schedule=("time", "time_variate", "time", "time_variate"),
            quantile_knots=(0.1, 0.5, 0.9),
            block_size=4,
            context_length=128,
        )

    @pytest.fixture
    def model(self, tiny_config):
        """Model instance for testing."""
        model = TimestralModel(tiny_config)
        model.build((None, 64, 1))  # (batch, seq_len, features)
        return model

    def test_model_creation(self, tiny_config):
        """Test model instantiation and build."""
        model = TimestralModel(tiny_config)
        model.build((None, 64, 1))
        assert model.built
        assert model.config.patch_size == 8
        assert model.config.d_model == 64

    def test_model_variants(self):
        """Test all predefined variants can be created."""
        for variant in ["5M", "28M", "123M", "209M"]:
            model = create_timestral_model(variant, prediction_length=24)
            model.build((None, 64, 1))
            assert model.built
            assert model.config.d_model > 0

    def test_create_by_variant(self):
        """Test create_timestral_by_variant alias."""
        model = create_timestral_by_variant("5M", prediction_length=24)
        model.build((None, 64, 1))
        assert model.built

    def test_invalid_variant(self):
        """Test that invalid variant raises error."""
        with pytest.raises(ValueError, match="Unknown variant"):
            create_timestral_model("invalid")

    def test_forward_shape(self, model):
        """Test forward pass output shape."""
        batch_size = 2
        seq_len = 64
        x = np.random.randn(batch_size, seq_len, 1).astype(np.float32)

        # The scaler on its own; call() and predict_block are in test_training.py.
        scaled, loc, scale = model.scale(
            keras.ops.convert_to_tensor(x[:, :, 0]),
            keras.ops.convert_to_tensor(np.ones((batch_size, seq_len), dtype=bool))
        )
        assert scaled.shape == (batch_size, seq_len)

    def test_predict_block_context_only(self, model):
        """Test single-block prediction without covariates."""
        batch_size = 1
        seq_len = 64  # 8 patches of size 8
        r = batch_size  # packed rows = batch for univariate
        inputs = keras.ops.convert_to_tensor(np.random.randn(r, seq_len).astype(np.float32))
        observed = keras.ops.convert_to_tensor(np.ones((r, seq_len), dtype=bool))
        group_ids = keras.ops.convert_to_tensor(np.zeros(r, dtype=np.int32))

        preds = model.predict_block(inputs, observed, group_ids, horizon_patches=1)

        # Output: (R, horizon_patches * patch_size, num_quantiles)
        assert preds.shape == (r, 8, 3)  # 1 patch * 8 = 8 time steps, 3 quantiles
        # Quantiles should be monotonic
        for i in range(preds.shape[0]):
            for t in range(preds.shape[1]):
                q = preds[i, t]
                assert np.all(np.diff(q) >= -1e-6), f"Quantiles not monotonic: {q}"

    def test_get_config_roundtrip(self, model):
        """Test get_config/from_config roundtrip."""
        config = model.get_config()
        restored = TimestralModel.from_config(config)
        restored.build((None, 64, 1))

        # Check key config values preserved
        assert restored.config.patch_size == model.config.patch_size
        assert restored.config.d_model == model.config.d_model
        assert restored.config.num_heads == model.config.num_heads
        assert list(restored.config.quantile_knots) == list(model.config.quantile_knots)

    def test_serialization(self, model, tmp_path):
        """Test model save/load roundtrip."""
        path = tmp_path / "timestral_test.keras"
        model.save(str(path))
        loaded = keras.models.load_model(str(path))
        assert isinstance(loaded, TimestralModel)
        assert loaded.config.patch_size == model.config.patch_size

    def test_gradient_flow(self, tiny_config):
        """Test gradients flow through the model (scale path)."""
        import tensorflow as tf

        model = TimestralModel(tiny_config)
        model.build((None, 64, 1))

        x = tf.convert_to_tensor(np.random.randn(1, 64).astype(np.float32))
        observed = tf.ones((1, 64), dtype=tf.bool)

        with tf.GradientTape() as tape:
            tape.watch(x)
            scaled, _, _ = model.scale(x, observed)
            loss = tf.reduce_mean(scaled ** 2)
        grads = tape.gradient(loss, x)
        assert grads is not None
        assert tf.reduce_any(grads != 0).numpy(), "No gradients flowing"

    def test_quantile_monotonicity(self, model):
        """Test that quantile head produces monotonic quantiles."""
        batch_size = 1
        seq_len = 64
        r = batch_size
        inputs = keras.ops.convert_to_tensor(np.random.randn(r, seq_len).astype(np.float32))
        observed = keras.ops.convert_to_tensor(np.ones((r, seq_len), dtype=bool))
        group_ids = keras.ops.convert_to_tensor(np.zeros(r, dtype=np.int32))

        preds = model.predict_block(inputs, observed, group_ids, horizon_patches=1)
        # preds: (R, H, Q) - check monotonicity along Q axis
        diffs = keras.ops.diff(preds, axis=-1)
        assert keras.ops.all(diffs >= -1e-5), "Quantiles must be non-decreasing"

    def test_median_at_center(self, model):
        """Test that median (0.5) is at the center of quantile knots."""
        assert model.config.quantile_knots[model.config.median_index] == 0.5

    def test_different_quantile_levels(self):
        """Test model with custom quantile levels."""
        config = TimestralConfig(
            patch_size=8,
            d_model=32,
            num_heads=4,
            mlp_hidden_dim=64,
            block_schedule=("time", "variate"),
            quantile_knots=(0.1, 0.25, 0.5, 0.75, 0.9),
            block_size=4,
            context_length=64,
        )
        model = TimestralModel(config)
        model.build((None, 64, 1))
        assert model.config.num_quantiles == 5
        assert model.config.median_index == 2


def _tiny_model():
    return create_timestral_model(
        "5M", prediction_length=16, patch_size=8,
        quantile_levels=(0.1, 0.5, 0.9), d_model=32, num_heads=4,
        mlp_hidden_dim=64, block_schedule=("time", "time_variate"),
        block_size=4, context_length=64,
    )


def _predict(model):
    rng = np.random.RandomState(0)
    context = np.cumsum(rng.randn(3, 32), axis=-1).astype("float32")
    observed = np.ones((3, 32), dtype=bool)
    group_ids = np.array([0, 0, 1], dtype="int32")
    return keras.ops.convert_to_numpy(
        model.predict_block(context, observed, group_ids, horizon_patches=2))


def _relative_paths(model):
    """Weight paths without the model's own prefix (an explicit build adds it)."""
    return [w.path.split("/", 1)[1] if w.path.startswith(model.name + "/")
            else w.path for w in model.weights]


class TestTimestralBuildAndRoundTrip:
    """``build()`` creates every weight, so a ``.keras`` round trip is exact.

    Until 2026-10-19 (plan-2026-10-14T062616-81938e81 D-162) the sublayers
    created their Dense layers in ``build()`` without building them, so
    ``model.build`` materialized 1 of 30 weights and a saved model reloaded
    with 29 of 30 weights fresh.
    """

    def test_build_creates_every_weight_a_forward_creates(self):
        lazy = _tiny_model()
        _predict(lazy)
        built = _tiny_model()
        built.build((None, 32))
        assert len(lazy.weights) == 30
        assert sorted(_relative_paths(built)) == sorted(_relative_paths(lazy))

    def test_keras_round_trip_restores_every_weight_and_the_forecast(self, tmp_path):
        keras.utils.set_random_seed(0)
        model = _tiny_model()
        model.build((None, 32))
        before = _predict(model)
        path = tmp_path / "timestral.keras"
        model.save(str(path))
        loaded = keras.models.load_model(str(path), compile=False)
        donor = dict(zip(_relative_paths(model), model.weights))
        restored = dict(zip(_relative_paths(loaded), loaded.weights))
        assert sorted(restored) == sorted(donor)
        for name, weight in donor.items():
            np.testing.assert_array_equal(
                keras.ops.convert_to_numpy(restored[name]),
                keras.ops.convert_to_numpy(weight), err_msg=name)
        np.testing.assert_array_equal(_predict(loaded), before)

    @pytest.mark.parametrize("explicit_build, digest", [
        (False, "61f1075e8f0eb4dd566432936e81ec042c8df1dd39c412af1ee4a17949ec31fe"),
        (True, "5c798e9184d6ce7e3c491967f90e4ac81b238af3ec24226e6d2dd8c7098b222b"),
    ])
    def test_initial_weights_are_the_ports(self, explicit_build, digest):
        """The build fix moved no initial weight, built lazily or explicitly.

        Digests are sha256 over every weight in ``model.weights`` order at seed
        1234, measured on the port as committed (6d757c6a6), identical on CPU and
        GPU 1. Building in call order instead of head first changes the explicit
        digest (D-162). A Keras/TF upgrade that changes initializer streams also
        moves them; re-measure then, do not loosen.
        """
        import hashlib

        keras.utils.set_random_seed(1234)
        model = _tiny_model()
        if explicit_build:
            model.build((None, 32))
        else:
            _predict(model)
        sha = hashlib.sha256()
        for weight in model.weights:
            sha.update(np.ascontiguousarray(
                keras.ops.convert_to_numpy(weight), dtype="float32").tobytes())
        assert sha.hexdigest() == digest

    def test_the_factories_are_from_variant(self):
        expected = TimestralModel.from_variant("5M", prediction_length=24).config
        assert create_timestral_model("5M", prediction_length=24).config == expected
        assert create_timestral_by_variant("5M", prediction_length=24).config == expected
        assert TimestralModel.MODEL_VARIANTS["5M"]["d_model"] == expected.d_model


class TestTimestralConfig:
    """Tests for TimestralConfig validation and serialization."""

    def test_default_config(self):
        """Test default config values."""
        config = TimestralConfig()
        assert config.patch_size == 32
        assert config.d_model == 512
        assert config.num_heads == 8
        assert len(config.quantile_knots) == 21
        assert config.quantile_knots[10] == 0.5  # median at index 10

    def test_config_validation_d_model_divisible(self):
        """Test d_model must be divisible by num_heads."""
        with pytest.raises(ValueError, match="not divisible"):
            TimestralConfig(d_model=513, num_heads=8)

    def test_config_validation_rope_dim(self):
        """Test rope_partial_factor produces valid rope_dim."""
        with pytest.raises(ValueError, match="invalid rope_dim|invalid "):
            TimestralConfig(d_model=512, num_heads=8, rope_partial_factor=0.3)  # 512/8*0.3 = 19.2

    def test_config_validation_block_schedule(self):
        """Test block_schedule must use valid kinds."""
        with pytest.raises(ValueError, match="Unknown layer kind"):
            TimestralConfig(block_schedule=("invalid_kind",))

    def test_config_validation_attention_gates(self):
        """Test attention gates must be valid."""
        with pytest.raises(ValueError, match="Unknown attention gate"):
            TimestralConfig(time_attn_gate="invalid")

    def test_config_validation_quantile_knots(self):
        """Test quantile_knots must contain 0.5 and be increasing."""
        with pytest.raises(ValueError, match="must contain 0.5"):
            TimestralConfig(quantile_knots=(0.1, 0.2, 0.3))
        with pytest.raises(ValueError, match="must be strictly increasing"):
            TimestralConfig(quantile_knots=(0.5, 0.1, 0.9))

    def test_config_validation_stabilize_with_global(self):
        """Test stabilize_with_global must be False."""
        with pytest.raises(ValueError, match="advancing refine frame"):
            TimestralConfig(stabilize_with_global=True)

    def test_config_serialization(self):
        """Test config to_dict/from_dict roundtrip."""
        config = TimestralConfig(
            patch_size=16,
            d_model=256,
            num_heads=8,
            block_schedule=("time", "variate"),
        )
        d = config.to_dict()
        restored = TimestralConfig.from_dict(d)
        assert restored.patch_size == config.patch_size
        assert restored.d_model == config.d_model
        assert list(restored.block_schedule) == list(config.block_schedule)

    def test_config_save_load(self, tmp_path):
        """Test config save/load from file."""
        config = TimestralConfig(patch_size=16, d_model=256)
        path = tmp_path / "config.json"
        config.save(path)
        loaded = TimestralConfig.load(path)
        assert loaded.patch_size == config.patch_size
        assert loaded.d_model == config.d_model

    def test_supports_known_covariates(self):
        """Test supports_known_covariates property."""
        # Schedule with time followed by time_variate should support known covariates
        config = TimestralConfig(
            block_schedule=("time", "time_variate"),
            covariate_enabled=True,
        )
        assert config.supports_known_covariates is True

        # Schedule without time->time_variate transition should not
        config2 = TimestralConfig(
            block_schedule=("variate", "time"),
            covariate_enabled=True,
        )
        assert config2.supports_known_covariates is False

        # Disabled covariates
        config3 = TimestralConfig(
            block_schedule=("time", "time_variate"),
            covariate_enabled=False,
        )
        assert config3.supports_known_covariates is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])