"""Tests for the DaSiamRPN tracker."""

import json

import keras
import numpy as np
import pytest

from dl_techniques.layers.conv_blocks.stem import ConvStem
from dl_techniques.models.vision.dasiamrpn import (
    DaSiamRPN,
    SiamRPNBackbone,
    create_dasiamrpn,
    decode_dasiamrpn_boxes,
    generate_dasiamrpn_anchors,
)

#: The names a BatchNormalization weight carries. A path test on the layer name cannot
#: tell: the stem's norm is ``stem_norm1`` and contains no "bn".
BATCH_NORM_WEIGHTS = ("gamma", "beta", "moving_mean", "moving_variance")


def _batch_norm_weights(model):
    return [w.path for w in model.weights if w.path.rsplit("/", 1)[-1] in BATCH_NORM_WEIGHTS]


def _pair(batch=1):
    return (
        np.zeros((batch, 127, 127, 3), dtype="float32"),
        np.zeros((batch, 271, 271, 3), dtype="float32"),
    )


class TestDaSiamRPN:
    def test_unknown_variant_raises_listing_keys(self):
        with pytest.raises(ValueError, match="big"):
            DaSiamRPN.from_variant("nope")

    def test_forward_shapes_and_finite(self):
        model = create_dasiamrpn("otb")
        z = np.zeros((1, 127, 127, 3), dtype="float32")
        x = np.zeros((1, 271, 271, 3), dtype="float32")
        out = model((z, x), training=False)
        assert tuple(out["cls"].shape) == (1, 19, 19, 10)
        assert tuple(out["reg"].shape) == (1, 19, 19, 20)
        assert bool(np.all(np.isfinite(keras.ops.convert_to_numpy(out["cls"]))))
        assert bool(np.all(np.isfinite(keras.ops.convert_to_numpy(out["reg"]))))

    def test_big_variant_wider(self):
        big = create_dasiamrpn("big")
        otb = create_dasiamrpn("otb")
        assert big.feature_out > otb.feature_out
        assert big.backbone.out_channels > otb.backbone.out_channels
        assert big.variant_config["width_scale"] == 2

    def test_explicit_build_matches_lazy_build(self):
        def _relative(m):
            return sorted(w.path.split("/", 1)[-1] for w in m.weights)

        explicit = create_dasiamrpn("otb")
        explicit.build([(None, 127, 127, 3), (None, 271, 271, 3)])
        lazy = create_dasiamrpn("otb")
        lazy(
            (
                np.zeros((1, 127, 127, 3), dtype="float32"),
                np.zeros((1, 271, 271, 3), dtype="float32"),
            )
        )
        assert _relative(explicit) == _relative(lazy)
        assert len(explicit.weights) > 0

    def test_serialization_round_trip_values(self, tmp_path):
        model = create_dasiamrpn("otb")
        z = np.ones((1, 127, 127, 3), dtype="float32")
        x = np.ones((1, 271, 271, 3), dtype="float32")
        original = {k: keras.ops.convert_to_numpy(v) for k, v in model((z, x), training=False).items()}
        path = str(tmp_path / "dasiamrpn.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        restored = {k: keras.ops.convert_to_numpy(v) for k, v in loaded((z, x), training=False).items()}
        for k in original:
            np.testing.assert_allclose(original[k], restored[k], atol=1e-6, rtol=0)

    def test_anchors_and_decode(self):
        anchors = generate_dasiamrpn_anchors(19)
        assert anchors.shape == (19 * 19 * 5, 4)
        assert bool(np.all(anchors[:, 2:] > 0))
        # Pinned against the reference `generate_anchor` (bit-identical):
        # anchor-major rows, origin -(19/2)*8 = -76.
        np.testing.assert_allclose(anchors[0], [-76, -76, 104, 32], atol=0, rtol=0)
        np.testing.assert_allclose(anchors[361], [-76, -76, 88, 40], atol=0, rtol=0)
        # Same anchor shape across positions; new shape across anchor blocks.
        np.testing.assert_allclose(anchors[1, 2:], anchors[0, 2:], atol=0, rtol=0)
        assert not np.allclose(anchors[361, 2:], anchors[0, 2:])
        zeros = np.zeros((4, anchors.shape[0]), dtype="float32")
        decoded = decode_dasiamrpn_boxes(anchors, zeros)
        np.testing.assert_allclose(decoded, anchors, atol=0, rtol=0)

    def test_no_batchnorm_builds_no_batchnorm_weights(self):
        """The BN flag gates weights, not just the forward path: no weight of the
        BatchNormalization kind exists, whatever its layer is called."""
        model = create_dasiamrpn(
            "otb", use_batch_norm=False, exemplar_size=127, search_size=271
        )
        model(_pair())
        assert _batch_norm_weights(model) == []

    def test_batchnorm_builds_four_weights_in_each_of_four_norms(self):
        """The twin of the test above, so a guard that finds nothing is not blind: the
        stem's norm and ``bn2`` to ``bn4`` hold sixteen BatchNormalization weights."""
        model = create_dasiamrpn("otb", use_batch_norm=True)
        model(_pair())
        paths = _batch_norm_weights(model)
        assert len(paths) == 16
        assert sum("/stem/stem_norm1/" in p for p in paths) == 4

    def test_pretrained_true_raises(self):
        with pytest.raises(NotImplementedError):
            create_dasiamrpn("otb", pretrained=True)

    def test_gradient_flows_to_backbone(self):
        import tensorflow as tf

        model = create_dasiamrpn("otb")
        z = tf.zeros((1, 127, 127, 3))
        x = tf.zeros((1, 271, 271, 3))
        with tf.GradientTape() as tape:
            out = model((z, x), training=True)
            loss = keras.ops.sum(out["cls"]) + keras.ops.sum(out["reg"])
        grads = tape.gradient(loss, model.trainable_variables)
        assert len(grads) > 0
        assert any(
            g is not None and bool(np.any(keras.ops.convert_to_numpy(g) != 0))
            for g in grads
        )


class TestDaSiamRPNStem:
    """Stage 1 (11x11/2 convolution, BatchNorm, ReLU, 3x3/2 pool) is one ``ConvStem``."""

    @pytest.mark.parametrize("variant,width", [("otb", 1), ("vot", 1), ("big", 2)])
    def test_the_stem_is_one_conv_stem_scaled_by_the_width(self, variant, width):
        stem = create_dasiamrpn(variant).backbone.stem
        assert isinstance(stem, ConvStem) and stem.name == "stem"
        assert (stem.num_convs, stem.filters, stem.kernel_size, stem.strides) == (
            1, 96 * width, 11, 2)
        assert stem.padding == "valid" and stem.use_bias is True
        assert stem.activation_type == "relu"
        assert (stem.pool_type, stem.pool_size, stem.pool_strides, stem.pool_padding) == (
            "max", 3, 2, "valid")
        assert stem.norm1.momentum == pytest.approx(0.9)
        assert stem.norm1.epsilon == pytest.approx(1e-5)

    def test_the_stem_takes_the_backbones_batch_norm_values(self):
        backbone = SiamRPNBackbone(bn_momentum=0.8, bn_epsilon=1e-3)
        assert backbone.stem.norm1.momentum == pytest.approx(0.8)
        assert backbone.stem.norm1.epsilon == pytest.approx(1e-3)

    def test_without_batch_norm_the_stem_holds_no_norm(self):
        backbone = SiamRPNBackbone(use_batch_norm=False)
        assert backbone.stem.norm1 is None and backbone.stem.normalization_type is None

    def test_the_backbone_owns_no_loose_first_stage_layer(self):
        backbone = SiamRPNBackbone()
        for old in ("conv1", "bn1", "pool1"):
            assert not hasattr(backbone, old), f"{old} survived the migration"

    @pytest.mark.parametrize("width,total", [(1, 3_751_680), (2, 14_925_312)])
    def test_the_weights_keep_the_reference_counts(self, width, total):
        backbone = SiamRPNBackbone(width_scale=width)
        backbone.build((None, 127, 127, 3))
        assert (len(backbone.weights), len(backbone.trainable_weights)) == (26, 18)
        assert backbone.count_params() == total

    def test_the_stem_weights_have_exactly_these_paths(self):
        model = create_dasiamrpn("otb")
        model.build([(None, 127, 127, 3), (None, 271, 271, 3)])
        stem_paths = sorted(w.path.split("/", 1)[-1] for w in model.weights if "/stem/" in w.path)
        assert stem_paths == sorted(
            f"backbone/stem/stem_{layer}/{weight}"
            for layer, names in (
                ("conv1", ("kernel", "bias")),
                ("norm1", ("gamma", "beta", "moving_mean", "moving_variance")))
            for weight in names)

    def test_an_explicit_build_builds_the_whole_stem(self):
        backbone = SiamRPNBackbone()
        backbone.build((None, 127, 127, 3))
        assert backbone.stem.built and all(
            layer.built for layer in backbone.stem._flatten_layers(include_self=False))

    def test_training_updates_the_stem_moving_statistics(self):
        model = create_dasiamrpn("otb")
        rng = np.random.default_rng(0)
        pair = (rng.normal(size=(2, 127, 127, 3)).astype("float32"),
                rng.normal(size=(2, 271, 271, 3)).astype("float32"))
        model(pair, training=False)
        before = keras.ops.convert_to_numpy(model.backbone.stem.norm1.moving_mean).copy()
        model(pair, training=True)
        assert not np.array_equal(
            before, keras.ops.convert_to_numpy(model.backbone.stem.norm1.moving_mean))

    def test_a_saved_model_reloads_with_the_stem(self, tmp_path):
        model = create_dasiamrpn("otb")
        rng = np.random.default_rng(1)
        pair = (rng.normal(size=(1, 127, 127, 3)).astype("float32"),
                rng.normal(size=(1, 271, 271, 3)).astype("float32"))
        original = {k: keras.ops.convert_to_numpy(v) for k, v in model(pair, training=False).items()}
        path = str(tmp_path / "dasiamrpn.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        assert isinstance(loaded.backbone.stem, ConvStem)
        restored = {k: keras.ops.convert_to_numpy(v) for k, v in loaded(pair, training=False).items()}
        for k in original:
            np.testing.assert_allclose(original[k], restored[k], atol=0, rtol=0)


class TestDaSiamRPNInitializer:
    """The knob is stored as a spec and every convolution draws from its own copy."""

    def _convs(self, model):
        backbone = model.backbone
        return [backbone.stem.conv1, backbone.conv2, backbone.conv3, backbone.conv4,
                backbone.conv5, model.conv_r1, model.conv_r2, model.conv_cls1,
                model.conv_cls2, model.regress_adjust]

    def test_an_instance_is_stored_as_a_json_safe_spec(self):
        model = create_dasiamrpn("otb", kernel_initializer=keras.initializers.HeNormal())
        assert isinstance(model.kernel_initializer, dict)
        assert isinstance(model.backbone.kernel_initializer, dict)
        json.dumps(model.get_config())
        json.dumps(model.backbone.get_config())

    def test_a_name_is_stored_as_the_name(self):
        model = create_dasiamrpn("otb", kernel_initializer="he_uniform")
        assert model.kernel_initializer == "he_uniform"

    def test_a_config_with_the_old_serialized_dict_still_builds(self):
        old = keras.initializers.serialize(keras.initializers.GlorotUniform())
        config = create_dasiamrpn("otb").get_config()
        config["kernel_initializer"] = old
        model = DaSiamRPN.from_config(config)
        assert isinstance(model.backbone.stem.conv1.kernel_initializer,
                          keras.initializers.GlorotUniform)

    def test_a_backbone_given_an_instance_stores_a_spec_and_gives_each_convolution_a_copy(self):
        """The backbone is public and takes the knob on its own, not only through the model."""
        backbone = SiamRPNBackbone(kernel_initializer=keras.initializers.HeNormal())
        assert isinstance(backbone.kernel_initializer, dict)
        json.dumps(backbone.get_config())
        convs = [backbone.stem.conv1, backbone.conv2, backbone.conv3, backbone.conv4,
                 backbone.conv5]
        assert len({id(c.kernel_initializer) for c in convs}) == len(convs)

    def test_the_knob_reaches_all_ten_convolutions(self):
        model = create_dasiamrpn("otb", kernel_initializer="he_uniform")
        convs = self._convs(model)
        assert len(convs) == 10
        assert all(isinstance(c.kernel_initializer, keras.initializers.HeUniform) for c in convs)

    def test_no_two_convolutions_share_one_initializer_object(self):
        model = create_dasiamrpn("otb", kernel_initializer=keras.initializers.HeNormal())
        convs = self._convs(model)
        assert len({id(c.kernel_initializer) for c in convs}) == len(convs)


class TestDaSiamRPNArguments:
    @pytest.mark.parametrize("make", [
        lambda **kw: DaSiamRPN(**kw), lambda **kw: SiamRPNBackbone(**kw)],
        ids=["model", "backbone"])
    def test_a_shape_keyword_is_refused_not_dropped_with_a_warning(self, make):
        for key, value in (("input_shape", (127, 127, 3)), ("input_dim", 3)):
            with pytest.raises(ValueError, match=key):
                make(**{key: value})

    @pytest.mark.parametrize("bad", [0, -127, 2.5, True, None])
    def test_a_size_is_a_positive_integer_named_in_the_message(self, bad):
        with pytest.raises(ValueError, match="exemplar_size"):
            DaSiamRPN(exemplar_size=bad)
        with pytest.raises(ValueError, match="search_size"):
            DaSiamRPN(search_size=bad)

    @pytest.mark.parametrize("bad", [0, -1, 2.5, True])
    def test_feature_out_is_a_positive_integer(self, bad):
        with pytest.raises(ValueError, match="feature_out"):
            DaSiamRPN(variant_config={"width_scale": 1, "feature_out": bad})

    @pytest.mark.parametrize("bad", [0.0, -1e-5, float("nan"), None])
    def test_a_backbone_epsilon_is_positive(self, bad):
        with pytest.raises(ValueError, match="bn_epsilon"):
            SiamRPNBackbone(bn_epsilon=bad)
