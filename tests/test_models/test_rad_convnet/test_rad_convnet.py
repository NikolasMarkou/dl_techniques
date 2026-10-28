"""Tests for :class:`RADConvNet`."""

import numpy as np
import pytest
import keras
import tensorflow as tf

from dl_techniques.layers.heads.vision import ClassificationHead
from dl_techniques.models.vision.rad_convnet.model import RADConvNet, create_rad_convnet
from tests.test_models.initializer_replay_oracle import replayed_pairs

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


def _draw_after_the_trunk(initializer, units=5, in_features=16):
    """The kernel a plain ``Dense(initializer)`` draws right after a headless TINY trunk."""
    keras.utils.set_random_seed(11)
    RADConvNet(include_top=False, **TINY)
    reference = keras.layers.Dense(units, kernel_initializer=initializer)
    reference.build((None, in_features))
    return keras.ops.convert_to_numpy(reference.kernel)


class TestRADConvNetHeadInitializer:
    @pytest.mark.parametrize("block_dims", [[16, 16], [32]])
    def test_hidden_head_blocks_draw_independent_kernels(self, block_dims):
        """The default head takes its seed once so the classifier equals the pre-head-move draw;
        with hidden blocks that one explicit seed would hand every Dense the same draw (three
        bit-identical (16, 16) kernels for ``block_dims=[16, 16]``)."""
        model = RADConvNet(num_classes=16, depths=[1, 1], dims=[8, 16], strides=2,
                           input_shape=(32, 32, 3), head_config_overrides={"block_dims": block_dims})
        model(keras.random.normal((1, 32, 32, 3)))
        kernels = [w for w in model.weights if "classification_head" in w.path and w.path.endswith("kernel")]
        assert len(kernels) == len(block_dims) + 1
        pairs, compared = replayed_pairs(model)
        assert compared > 0
        assert not pairs, pairs

    @pytest.mark.parametrize("initializer", ["truncated_normal", "glorot_uniform", "he_normal", "zeros"])
    def test_the_head_initializer_constant_is_honored_whole(self, initializer):
        """A subclass that names another ``HEAD_INITIALIZER`` gets that initializer: the constant
        used to supply only a seed (``glorot_uniform`` still drew a TruncatedNormal, ``zeros``
        raised ``AttributeError: 'Zeros' object has no attribute 'seed'``)."""

        class Variant(RADConvNet):
            HEAD_INITIALIZER = initializer

        expected = _draw_after_the_trunk(initializer)
        keras.utils.set_random_seed(11)
        model = Variant(num_classes=5, **TINY)
        model(keras.random.normal((1, 32, 32, 3)))
        np.testing.assert_array_equal(
            keras.ops.convert_to_numpy(model.classification_head.classifier.kernel), expected)


class TestRADConvNetBiases:
    @staticmethod
    def _bias_paths(model):
        return {w.path.split("/", 1)[1] for w in model.weights
                if w.path.endswith(("/bias", "/beta"))}

    def test_use_bias_false_keeps_only_the_rad_conv_predictor_biases(self):
        """``RADConv2D`` has no knob for the biases of its offset and weight predictors: the offset
        bias is the initial region size (``offset_bias_init``), part of the mechanism. So
        ``use_bias=False`` removes every other bias and beta, and the two predictor biases of
        each block remain."""
        model = RADConvNet(num_classes=5, use_bias=False, **TINY)
        model(keras.random.normal((1, 32, 32, 3)))
        blocks = [f"stage{s}_block{b}" for s, b in ((1, 1), (2, 1), (2, 2))]
        assert self._bias_paths(model) == {
            f"{block}_rad_conv/{predictor}/bias"
            for block in blocks for predictor in ("offset_conv", "weight_conv")}

    def test_use_bias_true_adds_the_other_biases_and_betas(self):
        model = RADConvNet(num_classes=5, use_bias=True, **TINY)
        model(keras.random.normal((1, 32, 32, 3)))
        present = self._bias_paths(model)
        assert {"stem/stem_conv1/bias", "stem/stem_norm1/beta", "downsample_0_conv/bias",
                "downsample_0_norm/beta", "stage1_block1_rad_conv/bias", "stage1_block1_norm/beta",
                "stage1_block1_expand/bias", "stage1_block1_reduce/bias",
                "classification_head/pooled_norm/beta", "classification_head/classifier/bias",
                "stage1_block1_rad_conv/offset_conv/bias"} <= present


def _l2_of_every_regularized_kernel(model, factor):
    """``factor * sum(w ** 2)`` over every kernel the model's ``kernel_regularizer`` reaches.

    Every ``kernel`` weight except the zero-initialized offset and weight predictors inside
    ``RADConv2D``: the layer regularizes its static mixing kernel only.
    """
    total = 0.0
    for w in model.weights:
        if w.path.endswith("/kernel") and "offset_conv" not in w.path and "weight_conv" not in w.path:
            total += factor * float(np.sum(np.square(keras.ops.convert_to_numpy(w).astype("float64"))))
    return total


class TestRADConvNetKernelRegularizer:
    def test_the_default_is_no_regularizer_and_no_loss(self):
        model = RADConvNet(num_classes=5, **TINY)
        model(keras.random.normal((1, 32, 32, 3)))
        assert model.kernel_regularizer is None
        assert len(model.losses) == 0
        assert model.get_config()["kernel_regularizer"] is None

    @pytest.mark.parametrize("include_top", [True, False])
    def test_a_regularizer_reaches_every_kernel_site(self, include_top):
        """Stem, downsampler, the RAD-Conv mixing kernel, both MLP denses of every block, and the
        classifier: 1 + 1 + 3 * 3 blocks + 1 = 12 losses (11 without the head), summing to the
        L2 of every such kernel."""
        model = RADConvNet(num_classes=5, include_top=include_top,
                           kernel_regularizer=keras.regularizers.L2(1e-3), **TINY)
        model(keras.random.normal((1, 32, 32, 3)))
        assert len(model.losses) == (12 if include_top else 11)
        np.testing.assert_allclose(
            float(sum(keras.ops.convert_to_numpy(loss) for loss in model.losses)),
            _l2_of_every_regularized_kernel(model, 1e-3), rtol=1e-4)

    def test_the_regularizer_survives_get_config_and_a_saved_model(self, tmp_path):
        import json

        model = RADConvNet(num_classes=5, kernel_regularizer=keras.regularizers.L2(1e-3), **TINY)
        config = json.loads(json.dumps(model.get_config()))
        assert config["kernel_regularizer"]["class_name"] == "L2"
        assert config["kernel_regularizer"]["config"]["l2"] == pytest.approx(1e-3)
        rebuilt = RADConvNet.from_config(model.get_config())
        assert isinstance(rebuilt.kernel_regularizer, keras.regularizers.L2)
        x = keras.random.normal((1, 32, 32, 3))
        model(x)
        path = tmp_path / "regularized.keras"
        model.save(path)
        loaded = keras.models.load_model(path)
        loaded(x)
        assert len(loaded.losses) == 12

    def test_a_regularizer_given_as_a_name_is_resolved(self):
        model = RADConvNet(num_classes=5, kernel_regularizer="l2", **TINY)
        assert isinstance(model.kernel_regularizer, keras.regularizers.L2)

    def test_a_regularizer_leaves_the_seeded_draws_alone(self):
        keras.utils.set_random_seed(5)
        plain = RADConvNet(num_classes=5, **TINY)
        plain(keras.random.normal((1, 32, 32, 3)))
        keras.utils.set_random_seed(5)
        regularized = RADConvNet(num_classes=5, kernel_regularizer=keras.regularizers.L2(1e-3), **TINY)
        regularized(keras.random.normal((1, 32, 32, 3)))
        for a, b in zip(plain.weights, regularized.weights):
            np.testing.assert_array_equal(keras.ops.convert_to_numpy(a), keras.ops.convert_to_numpy(b))


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
