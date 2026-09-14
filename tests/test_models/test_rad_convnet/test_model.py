"""Tests for :class:`RADConvNet`."""

import numpy as np
import pytest
import keras
import tensorflow as tf

from dl_techniques.models.vision.rad_convnet.model import RADConvNet, create_rad_convnet


class TestRADConvNetVariants:
    @pytest.mark.parametrize("variant", ["tiny", "small", "base"])
    def test_variant_builds_and_predicts(self, variant):
        model = RADConvNet.from_variant(variant, num_classes=5, input_shape=(32, 32, 3))
        x = keras.random.normal((2, 32, 32, 3))
        y = model(x)
        assert y.shape == (2, 5)

    def test_unknown_variant_raises(self):
        with pytest.raises(ValueError):
            RADConvNet.from_variant("does-not-exist")

    def test_create_rad_convnet_factory(self):
        model = create_rad_convnet(variant="tiny", num_classes=3, input_shape=(32, 32, 3))
        assert isinstance(model, RADConvNet)
        y = model(keras.random.normal((1, 32, 32, 3)))
        assert y.shape == (1, 3)


class TestRADConvNetHeadModes:
    def test_include_top_false_returns_feature_map(self):
        model = RADConvNet.from_variant("tiny", include_top=False, input_shape=(32, 32, 3))
        y = model(keras.random.normal((1, 32, 32, 3)))
        assert len(y.shape) == 4
        assert y.shape[-1] == model.dims[-1]

    def test_num_classes_zero_returns_pooled_features(self):
        model = RADConvNet.from_variant("tiny", num_classes=0, input_shape=(32, 32, 3))
        y = model(keras.random.normal((1, 32, 32, 3)))
        assert y.shape == (1, model.dims[-1])


class TestRADConvNetGradients:
    def test_all_trainable_variables_receive_gradients(self):
        model = RADConvNet.from_variant("tiny", num_classes=4, input_shape=(24, 24, 3))
        x = tf.Variable(keras.random.normal((2, 24, 24, 3)))
        with tf.GradientTape() as tape:
            y = model(x, training=True)
            loss = tf.reduce_sum(tf.square(y))
        grads = tape.gradient(loss, model.trainable_variables)
        assert len(grads) == len(model.trainable_variables)
        none_vars = [v.name for v, g in zip(model.trainable_variables, grads) if g is None]
        assert not none_vars, f"variables with no gradient: {none_vars}"


class TestRADConvNetSerialization:
    def test_get_config_round_trip(self):
        model = RADConvNet.from_variant("tiny", num_classes=7, input_shape=(32, 32, 3))
        config = model.get_config()
        restored = RADConvNet.from_config(config)
        assert restored.num_classes == 7
        assert restored.dims == model.dims
        assert restored.depths == model.depths

    def test_save_and_load_round_trip_produces_identical_output(self, tmp_path):
        model = RADConvNet.from_variant("tiny", num_classes=5, input_shape=(32, 32, 3))
        x = keras.random.normal((2, 32, 32, 3))
        y_before = model(x, training=False)

        save_path = tmp_path / "rad_convnet.keras"
        model.save(save_path)
        loaded = keras.models.load_model(save_path)
        y_after = loaded(x, training=False)

        np.testing.assert_allclose(
            np.array(y_before), np.array(y_after), atol=1e-5, rtol=0
        )
