"""Tests for :class:`RADConvNet`."""

import numpy as np
import pytest
import keras
import tensorflow as tf

from dl_techniques.layers.heads.vision import ClassificationHead
from dl_techniques.models.vision.rad_convnet.model import RADConvNet, create_rad_convnet

TINY = dict(depths=[1, 2], dims=[8, 16], strides=2, input_shape=(32, 32, 3))


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

    def test_num_classes_zero_with_a_head_is_refused_by_value(self):
        with pytest.raises(ValueError, match=r"num_classes must be positive.*got 0"):
            RADConvNet.from_variant("tiny", num_classes=0, input_shape=(32, 32, 3))

    def test_num_classes_does_not_matter_without_a_head(self):
        model = RADConvNet.from_variant(
            "tiny", num_classes=0, include_top=False, input_shape=(32, 32, 3))
        assert model(keras.random.normal((1, 32, 32, 3))).shape[-1] == model.dims[-1]

    def test_a_headless_model_defines_its_head_attribute_as_none(self):
        model = RADConvNet(include_top=False, **TINY)
        assert model.classification_head is None


def _head_owners(model):
    """The names under ``classification_head/`` that own a weight."""
    return {path.split("/")[path.split("/").index("classification_head") + 1]
            for path in (w.path for w in model.weights) if "classification_head" in path}


def _weights_by_relative_path(model):
    return {w.path.split("/", 1)[1]: w for w in model.weights}


class TestRADConvNetHead:
    def test_the_head_is_the_shared_classification_head(self):
        model = RADConvNet(num_classes=5, **TINY)
        model(keras.random.normal((1, 32, 32, 3)))
        head = model.classification_head
        assert isinstance(head, ClassificationHead)
        assert head.name == "classification_head"
        assert head.block_dims == ()
        assert head.pooled_norm.epsilon == pytest.approx(1e-6)
        assert _head_owners(model) == {"pooled_norm", "classifier"}

    def test_the_head_owns_the_old_head_weights_and_nothing_else(self):
        model = RADConvNet(num_classes=5, **TINY)
        model(keras.random.normal((1, 32, 32, 3)))
        head_paths = sorted(w.path.split("classification_head/")[1]
                            for w in model.weights if "classification_head/" in w.path)
        assert head_paths == ["classifier/bias", "classifier/kernel",
                              "pooled_norm/beta", "pooled_norm/gamma"]

    @pytest.mark.parametrize("use_bias", [True, False])
    def test_use_bias_decides_which_head_weights_exist(self, use_bias):
        model = RADConvNet(num_classes=5, use_bias=use_bias, **TINY)
        model(keras.random.normal((1, 32, 32, 3)))
        paths = [w.path for w in model.weights if "classification_head/" in w.path]
        assert any(p.endswith("classifier/kernel") for p in paths)
        assert any(p.endswith("pooled_norm/gamma") for p in paths)
        assert any(p.endswith("classifier/bias") for p in paths) is use_bias
        assert any(p.endswith("pooled_norm/beta") for p in paths) is use_bias

    @pytest.mark.parametrize("use_bias", [True, False])
    def test_the_head_is_pool_then_layer_norm_then_one_linear_layer(self, use_bias):
        """The logits equal the reference head computed by hand from the trunk's features."""
        keras.utils.set_random_seed(0)
        model = RADConvNet(num_classes=5, use_bias=use_bias, **TINY)
        trunk = RADConvNet(include_top=False, use_bias=use_bias, **TINY)
        x = keras.random.normal((3, 32, 32, 3))
        model(x)
        trunk(x)
        rng = np.random.default_rng(1)
        for w in model.weights:
            if "classification_head/" in w.path:
                w.assign(rng.normal(size=w.shape).astype("float32"))
        donor = _weights_by_relative_path(model)
        for relative, w in _weights_by_relative_path(trunk).items():
            w.assign(donor[relative])
        features = keras.ops.convert_to_numpy(trunk(x)).astype("float64")
        head = {w.path.split("classification_head/")[1]:
                keras.ops.convert_to_numpy(w).astype("float64")
                for w in model.weights if "classification_head/" in w.path}
        pooled = features.mean(axis=(1, 2))
        normed = (pooled - pooled.mean(-1, keepdims=True)) / np.sqrt(
            pooled.var(-1, keepdims=True) + 1e-6) * head["pooled_norm/gamma"]
        if use_bias:
            normed = normed + head["pooled_norm/beta"]
        expected = normed @ head["classifier/kernel"]
        if use_bias:
            expected = expected + head["classifier/bias"]
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(model(x)), expected, rtol=1e-4, atol=1e-5)

    def test_the_classifier_kernel_is_the_draw_a_plain_dense_makes_at_that_point(self):
        """The shared head draws the classifier as ``Dense(kernel_initializer="truncated_normal")`` would.

        The trunk makes the same default-seed draws with or without a head, so a plain Dense
        created right after a headless trunk takes the seed the classifier must have taken.
        """
        x = keras.random.normal((1, 32, 32, 3))
        keras.utils.set_random_seed(11)
        model = RADConvNet(num_classes=5, **TINY)
        model(x)
        keras.utils.set_random_seed(11)
        RADConvNet(include_top=False, **TINY)
        reference = keras.layers.Dense(5, kernel_initializer="truncated_normal")
        reference.build((None, 16))
        np.testing.assert_array_equal(
            keras.ops.convert_to_numpy(model.classification_head.classifier.kernel),
            keras.ops.convert_to_numpy(reference.kernel))

    def test_head_config_overrides_reach_the_head(self):
        model = RADConvNet(num_classes=5, head_config_overrides={"classifier_dropout_rate": 0.25},
                           **TINY)
        assert model.classification_head.classifier_dropout_rate == pytest.approx(0.25)
        assert model.get_config()["head_config_overrides"] == {"classifier_dropout_rate": 0.25}

    def test_a_live_object_in_head_config_overrides_is_refused_by_key(self):
        with pytest.raises(ValueError, match=r"head_config_overrides\['kernel_regularizer'\]"):
            RADConvNet(num_classes=5, head_config_overrides={
                "kernel_regularizer": keras.regularizers.L2(1e-4)}, **TINY)


class TestRADConvNetNames:
    def test_blocks_and_downsamplers_follow_the_naming_table(self):
        model = RADConvNet(num_classes=5, dropout_rate=0.1, **TINY)
        names = {layer.name for layer in model.layers}
        assert {"stem", "classification_head", "mlp_activation"} <= names
        assert {f"stage{s}_block{b}_{role}"
                for s, b in ((1, 1), (2, 1), (2, 2))
                for role in ("rad_conv", "norm", "expand", "reduce", "dropout")} <= names
        assert {"downsample_0_norm", "downsample_0_conv"} <= names
        old = {"gap", "head_norm", "classifier", "downsample_norm_0", "stage0_block0_norm"}
        assert not old & names

    def test_the_constructor_arguments_are_stored_under_their_own_names(self):
        model = RADConvNet(num_classes=5, activation="relu", **TINY)
        assert model.activation == "relu"
        assert len(model.stages) == 2 and [len(s) for s in model.stages] == [1, 2]
        assert len(model.downsamplers) == 1


class TestRADConvNetDropoutRate:
    @pytest.mark.parametrize("rate", [-1.0, 1.0, 2.0])
    def test_an_out_of_range_dropout_rate_is_refused_by_value(self, rate):
        with pytest.raises(ValueError, match=rf"dropout_rate must be in \[0, 1\), got {rate}"):
            RADConvNet(num_classes=5, dropout_rate=rate, **TINY)


class TestRADConvNetPretrained:
    def test_a_checkpoint_loads_into_a_model_with_the_default_unknown_spatial_size(self, tmp_path):
        donor = RADConvNet(num_classes=4, depths=[1, 1], dims=[8, 16], strides=2,
                           input_shape=(32, 32, 3))
        x = keras.random.normal((2, 32, 32, 3))
        expected = keras.ops.convert_to_numpy(donor(x))
        path = tmp_path / "donor.keras"
        donor.save(path)
        model = RADConvNet.from_variant("tiny", num_classes=4, depths=[1, 1], dims=[8, 16],
                                        strides=2, pretrained=str(path))
        assert model.input_shape_config == (None, None, 3)
        np.testing.assert_array_equal(keras.ops.convert_to_numpy(model(x)), expected)

    def test_from_variant_overrides_an_architecture_field_of_the_variant(self):
        model = RADConvNet.from_variant("tiny", depths=[1, 1], dims=[8, 16], strides=2)
        assert (model.depths, model.dims) == ([1, 1], [8, 16])


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
