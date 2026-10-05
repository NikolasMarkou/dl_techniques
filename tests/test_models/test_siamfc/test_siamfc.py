"""Tests for the SiamFC tracker."""

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.models.vision.siamfc import (
    SiamFC,
    create_siamfc,
    create_hann_window,
    siamfc_score_size,
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
        assert bool(np.all(np.isfinite(ops.convert_to_numpy(y))))

    def test_template_shift_moves_response(self):
        model = create_siamfc()
        rng = np.random.RandomState(0)
        x = rng.randn(1, 255, 255, 3).astype("float32")
        z = x[:, 64:191, 64:191, :]
        y = ops.convert_to_numpy(model((z, x), training=False))
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
        original = ops.convert_to_numpy(model((z, x), training=False))
        path = str(tmp_path / "siamfc.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        restored = ops.convert_to_numpy(loaded((z, x), training=False))
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
        """Anti-vacuity: the BN flag gates weights, not just the forward path."""
        model = create_siamfc(use_batch_norm=False)
        model(
            (
                np.zeros((1, 127, 127, 3), dtype="float32"),
                np.zeros((1, 255, 255, 3), dtype="float32"),
            )
        )
        assert not [w for w in model.weights if "bn" in w.path]

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
            loss = ops.sum(y)
        grads = tape.gradient(loss, model.trainable_variables)
        assert len(grads) > 0
        assert any(
            g is not None and bool(np.any(ops.convert_to_numpy(g) != 0))
            for g in grads
        )
