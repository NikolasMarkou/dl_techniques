"""Tests for the SiamFC tracker."""

import json

import keras
import numpy as np
import pytest

from dl_techniques.layers.conv_blocks.stem import ConvStem
from dl_techniques.models.vision.siamfc import (
    SiamFC,
    SiamFCBackbone,
    create_siamfc,
    create_hann_window,
    siamfc_score_size,
)

#: The names a BatchNormalization weight carries. A path test on the layer name cannot
#: tell: the stem's norm is ``stem_norm1`` and contains no "bn".
BATCH_NORM_WEIGHTS = ("gamma", "beta", "moving_mean", "moving_variance")


def _batch_norm_weights(model):
    return [w.path for w in model.weights if w.path.rsplit("/", 1)[-1] in BATCH_NORM_WEIGHTS]


def _pair(batch=1):
    return (
        np.zeros((batch, 127, 127, 3), dtype="float32"),
        np.zeros((batch, 255, 255, 3), dtype="float32"),
    )


class TestSiamFC:
    def test_score_helper_matches_paper_geometry(self):
        assert siamfc_score_size(127, 255) == 17

    def test_invalid_sizes_raise(self):
        with pytest.raises(ValueError):
            SiamFC(exemplar_size=255, search_size=127)
        with pytest.raises(ValueError):
            SiamFC(exemplar_size=0, search_size=255)

    def test_forward_shape_and_finite(self):
        model = create_siamfc()
        z = np.zeros((1, 127, 127, 3), dtype="float32")
        x = np.zeros((1, 255, 255, 3), dtype="float32")
        y = model((z, x), training=False)
        assert tuple(y.shape) == (1, 17, 17, 1)
        assert bool(np.all(np.isfinite(keras.ops.convert_to_numpy(y))))

    def test_template_shift_moves_response(self):
        # An untrained trunk does not guarantee that the peak sits at the template's
        # offset: 2 of the first 40 seeds (17 and 29) put it at (8, 2) and (9, 0). The
        # weights are seeded so the check is the same on every run.
        keras.utils.set_random_seed(0)
        model = create_siamfc()
        rng = np.random.RandomState(0)
        x = rng.randn(1, 255, 255, 3).astype("float32")
        z = x[:, 64:191, 64:191, :]
        y = keras.ops.convert_to_numpy(model((z, x), training=False))
        peak = np.unravel_index(int(np.argmax(y[0, :, :, 0])), y.shape[1:3])
        assert abs(peak[0] - 8) <= 3 and abs(peak[1] - 8) <= 3

    def test_explicit_build_matches_lazy_build(self):
        def _relative(m):
            return sorted(w.path.split("/", 1)[-1] for w in m.weights)

        explicit = create_siamfc()
        explicit.build([(None, 127, 127, 3), (None, 255, 255, 3)])
        lazy = create_siamfc()
        lazy(
            (
                np.zeros((1, 127, 127, 3), dtype="float32"),
                np.zeros((1, 255, 255, 3), dtype="float32"),
            )
        )
        assert _relative(explicit) == _relative(lazy)
        assert len(explicit.weights) > 0

    def test_serialization_round_trip_values(self, tmp_path):
        model = create_siamfc()
        z = np.ones((1, 127, 127, 3), dtype="float32")
        x = np.ones((1, 255, 255, 3), dtype="float32")
        original = keras.ops.convert_to_numpy(model((z, x), training=False))
        path = str(tmp_path / "siamfc.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        restored = keras.ops.convert_to_numpy(loaded((z, x), training=False))
        np.testing.assert_allclose(original, restored, atol=1e-6, rtol=0)

    def test_get_config_round_trip(self):
        model = create_siamfc(exemplar_size=127, search_size=255)
        config = model.get_config()
        rebuilt = SiamFC.from_config(dict(config))
        assert rebuilt.exemplar_size == 127
        assert rebuilt.search_size == 255

    def test_pretrained_true_raises(self):
        with pytest.raises(NotImplementedError):
            create_siamfc(pretrained=True)

    def test_no_batchnorm_builds_no_batchnorm_weights(self):
        """The BN flag gates weights, not just the forward path: no weight of the
        BatchNormalization kind exists, whatever its layer is called."""
        model = create_siamfc(use_batch_norm=False)
        model(_pair())
        assert _batch_norm_weights(model) == []

    def test_batchnorm_builds_four_weights_in_each_of_four_norms(self):
        """The twin of the test above, so a guard that finds nothing is not blind: the
        stem's norm and ``bn2`` to ``bn4`` hold sixteen BatchNormalization weights."""
        model = create_siamfc(use_batch_norm=True)
        model(_pair())
        paths = _batch_norm_weights(model)
        assert len(paths) == 16
        assert sum("/stem/stem_norm1/" in p for p in paths) == 4

    def test_hann_window_shape(self):
        w = create_hann_window(17)
        assert w.shape == (17, 17)
        assert float(w.max()) > 0

    def test_gradient_flows_to_backbone(self):
        import tensorflow as tf

        model = create_siamfc()
        z = tf.zeros((1, 127, 127, 3))
        x = tf.zeros((1, 255, 255, 3))
        with tf.GradientTape() as tape:
            y = model((z, x), training=True)
            loss = keras.ops.sum(y)
        grads = tape.gradient(loss, model.trainable_variables)
        assert len(grads) > 0
        assert any(
            g is not None and bool(np.any(keras.ops.convert_to_numpy(g) != 0))
            for g in grads
        )


class TestSiamFCStem:
    """Stage 1 (11x11/2 convolution, BatchNorm, ReLU, 3x3/2 pool) is one ``ConvStem``."""

    def test_the_stem_is_one_conv_stem_with_the_reference_settings(self):
        backbone = create_siamfc().backbone
        stem = backbone.stem
        assert isinstance(stem, ConvStem) and stem.name == "stem"
        assert (stem.num_convs, stem.filters, stem.kernel_size, stem.strides) == (1, 96, 11, 2)
        assert stem.padding == "valid" and stem.use_bias is True
        assert stem.activation_type == "relu"
        assert (stem.pool_type, stem.pool_size, stem.pool_strides, stem.pool_padding) == (
            "max", 3, 2, "valid")
        assert stem.norm1.momentum == pytest.approx(0.9)
        assert stem.norm1.epsilon == pytest.approx(1e-5)

    def test_the_stem_takes_the_backbones_batch_norm_values(self):
        backbone = SiamFCBackbone(bn_momentum=0.8, bn_epsilon=1e-3)
        assert backbone.stem.norm1.momentum == pytest.approx(0.8)
        assert backbone.stem.norm1.epsilon == pytest.approx(1e-3)

    def test_without_batch_norm_the_stem_holds_no_norm(self):
        backbone = SiamFCBackbone(use_batch_norm=False)
        assert backbone.stem.norm1 is None and backbone.stem.normalization_type is None

    def test_the_backbone_owns_no_loose_first_stage_layer(self):
        backbone = SiamFCBackbone()
        for old in ("conv1", "bn1", "pool1"):
            assert not hasattr(backbone, old), f"{old} survived the migration"

    def test_the_weights_keep_the_reference_counts(self):
        backbone = SiamFCBackbone()
        backbone.build((None, 127, 127, 3))
        assert (len(backbone.weights), len(backbone.trainable_weights)) == (26, 18)
        assert backbone.count_params() == 3_751_680
        plain = SiamFCBackbone(use_batch_norm=False)
        plain.build((None, 127, 127, 3))
        assert (len(plain.weights), len(plain.trainable_weights)) == (10, 10)
        assert plain.count_params() == 3_747_200

    def test_the_stem_weights_have_exactly_these_paths(self):
        model = create_siamfc()
        model.build([(None, 127, 127, 3), (None, 255, 255, 3)])
        stem_paths = sorted(w.path.split("/", 1)[-1] for w in model.weights if "/stem/" in w.path)
        assert stem_paths == sorted(
            f"backbone/stem/stem_{layer}/{weight}"
            for layer, names in (
                ("conv1", ("kernel", "bias")),
                ("norm1", ("gamma", "beta", "moving_mean", "moving_variance")))
            for weight in names)

    def test_an_explicit_build_builds_the_whole_stem(self):
        backbone = SiamFCBackbone()
        backbone.build((None, 127, 127, 3))
        assert backbone.stem.built and all(
            layer.built for layer in backbone.stem._flatten_layers(include_self=False))

    def test_training_updates_the_stem_moving_statistics(self):
        model = create_siamfc()
        rng = np.random.default_rng(0)
        pair = (rng.normal(size=(2, 127, 127, 3)).astype("float32"),
                rng.normal(size=(2, 255, 255, 3)).astype("float32"))
        model(pair, training=False)
        before = keras.ops.convert_to_numpy(model.backbone.stem.norm1.moving_mean).copy()
        model(pair, training=True)
        assert not np.array_equal(
            before, keras.ops.convert_to_numpy(model.backbone.stem.norm1.moving_mean))

    def test_a_saved_model_reloads_with_the_stem(self, tmp_path):
        model = create_siamfc()
        rng = np.random.default_rng(1)
        pair = (rng.normal(size=(1, 127, 127, 3)).astype("float32"),
                rng.normal(size=(1, 255, 255, 3)).astype("float32"))
        original = keras.ops.convert_to_numpy(model(pair, training=False))
        path = str(tmp_path / "siamfc.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        assert isinstance(loaded.backbone.stem, ConvStem)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(loaded(pair, training=False)), original, atol=0, rtol=0)


class TestSiamFCInitializer:
    """The knob is stored as a spec and every convolution draws from its own copy."""

    def test_an_instance_is_stored_as_a_json_safe_spec(self):
        model = SiamFC(kernel_initializer=keras.initializers.HeNormal())
        assert isinstance(model.kernel_initializer, dict)
        assert isinstance(model.backbone.kernel_initializer, dict)
        json.dumps(model.get_config())
        json.dumps(model.backbone.get_config())

    def test_a_name_is_stored_as_the_name(self):
        assert SiamFC(kernel_initializer="he_uniform").kernel_initializer == "he_uniform"

    def test_a_config_with_the_old_serialized_dict_still_builds(self):
        old = keras.initializers.serialize(keras.initializers.GlorotUniform())
        model = SiamFC.from_config({"exemplar_size": 127, "search_size": 255,
                                    "use_batch_norm": True, "kernel_initializer": old,
                                    "name": "old"})
        assert isinstance(model.backbone.stem.conv1.kernel_initializer,
                          keras.initializers.GlorotUniform)

    def test_the_knob_reaches_every_convolution(self):
        backbone = SiamFC(kernel_initializer="he_uniform").backbone
        convs = [backbone.stem.conv1, backbone.conv2, backbone.conv3, backbone.conv4,
                 backbone.conv5]
        assert all(isinstance(c.kernel_initializer, keras.initializers.HeUniform) for c in convs)

    def test_no_two_convolutions_share_one_initializer_object(self):
        backbone = SiamFC(kernel_initializer=keras.initializers.HeNormal()).backbone
        convs = [backbone.stem.conv1, backbone.conv2, backbone.conv3, backbone.conv4,
                 backbone.conv5]
        assert len({id(c.kernel_initializer) for c in convs}) == len(convs)

    def test_a_backbone_given_an_instance_stores_a_spec_and_gives_each_convolution_a_copy(self):
        """The backbone is public and takes the knob on its own, not only through the model."""
        backbone = SiamFCBackbone(kernel_initializer=keras.initializers.HeNormal())
        assert isinstance(backbone.kernel_initializer, dict)
        json.dumps(backbone.get_config())
        convs = [backbone.stem.conv1, backbone.conv2, backbone.conv3, backbone.conv4,
                 backbone.conv5]
        assert len({id(c.kernel_initializer) for c in convs}) == len(convs)

    def test_a_seedless_instance_draws_independent_kernels(self):
        keras.utils.set_random_seed(0)
        backbone = SiamFC(kernel_initializer=keras.initializers.HeNormal()).backbone
        backbone.build((None, 127, 127, 3))
        first = keras.ops.convert_to_numpy(backbone.stem.conv1.kernel).ravel()
        second = keras.ops.convert_to_numpy(backbone.conv2.kernel).ravel()
        # A shared instance gives the shorter draw as the prefix of the longer one.
        assert not np.array_equal(first[:1000], second[:1000])


class TestSiamFCArguments:
    @pytest.mark.parametrize("make", [
        lambda **kw: SiamFC(**kw), lambda **kw: SiamFCBackbone(**kw)], ids=["model", "backbone"])
    def test_a_shape_keyword_is_refused_not_dropped_with_a_warning(self, make):
        for key, value in (("input_shape", (127, 127, 3)), ("input_dim", 3)):
            with pytest.raises(ValueError, match=key):
                make(**{key: value})

    @pytest.mark.parametrize("bad", [0, -127, 2.5, True, None])
    def test_a_size_is_a_positive_integer_named_in_the_message(self, bad):
        with pytest.raises(ValueError, match="exemplar_size"):
            SiamFC(exemplar_size=bad)
        with pytest.raises(ValueError, match="search_size"):
            SiamFC(search_size=bad)

    @pytest.mark.parametrize("bad", [0.0, -1e-5, float("nan"), None])
    def test_a_backbone_epsilon_is_positive(self, bad):
        with pytest.raises(ValueError, match="bn_epsilon"):
            SiamFCBackbone(bn_epsilon=bad)

    @pytest.mark.parametrize("bad", [0.0, 1.0, 1.5])
    def test_a_backbone_momentum_is_inside_the_open_unit_interval(self, bad):
        with pytest.raises(ValueError, match="bn_momentum"):
            SiamFCBackbone(bn_momentum=bad)
