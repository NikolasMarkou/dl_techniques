"""Tests for the DaSiamRPN tracker."""

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.models.vision.dasiamrpn import (
    DaSiamRPN,
    create_dasiamrpn,
    decode_dasiamrpn_boxes,
    generate_dasiamrpn_anchors,
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
        assert bool(np.all(np.isfinite(ops.convert_to_numpy(out["cls"]))))
        assert bool(np.all(np.isfinite(ops.convert_to_numpy(out["reg"]))))

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
        original = {k: ops.convert_to_numpy(v) for k, v in model((z, x), training=False).items()}
        path = str(tmp_path / "dasiamrpn.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        restored = {k: ops.convert_to_numpy(v) for k, v in loaded((z, x), training=False).items()}
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
        """Anti-vacuity: the BN flag gates weights, not just the forward path."""
        model = create_dasiamrpn(
            "otb", use_batch_norm=False, exemplar_size=127, search_size=271
        )
        model(
            (
                np.zeros((1, 127, 127, 3), dtype="float32"),
                np.zeros((1, 271, 271, 3), dtype="float32"),
            )
        )
        assert not [w for w in model.weights if "bn" in w.path]

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
            loss = ops.sum(out["cls"]) + ops.sum(out["reg"])
        grads = tape.gradient(loss, model.trainable_variables)
        assert len(grads) > 0
        assert any(
            g is not None and bool(np.any(ops.convert_to_numpy(g) != 0))
            for g in grads
        )
