"""Tests for PositionEmbeddingSine2D (fixed 2D sinusoidal positional encoding)."""

import os
import numpy as np
import pytest
import keras
import tensorflow as tf

from dl_techniques.layers.embedding.positional_embedding_sine_2d import PositionEmbeddingSine2D


class TestPositionEmbeddingSine2D:

    # ---- constructor validation -------------------------------------

    def test_ctor_rejects_bad_args(self):
        with pytest.raises(ValueError):
            PositionEmbeddingSine2D(num_pos_feats=0)
        with pytest.raises(ValueError):
            PositionEmbeddingSine2D(temperature=0.0)

    # ---- forward / shape (channels-first NCHW) ----------------------

    def test_forward_shape_channels_first(self):
        layer = PositionEmbeddingSine2D(num_pos_feats=8)
        x = keras.ops.convert_to_tensor(np.random.rand(2, 6, 5, 3).astype("float32"))
        out = layer(x)
        # (B, 2*num_pos_feats, H, W)
        assert tuple(out.shape) == (2, 16, 6, 5)

    def test_compute_output_shape(self):
        layer = PositionEmbeddingSine2D(num_pos_feats=8)
        assert layer.compute_output_shape((2, 6, 5, 3)) == (2, 16, 6, 5)

    def test_normalize_false(self):
        layer = PositionEmbeddingSine2D(num_pos_feats=8, normalize=False)
        x = keras.ops.convert_to_tensor(np.random.rand(2, 6, 5, 3).astype("float32"))
        out = keras.ops.convert_to_numpy(layer(x))
        assert out.shape == (2, 16, 6, 5)
        assert np.all(np.isfinite(out))

    # ---- graph safety -----------------------------------------------

    def test_graph_trace(self):
        layer = PositionEmbeddingSine2D(num_pos_feats=8)
        x = tf.constant(np.random.rand(2, 6, 5, 3).astype("float32"))
        eager = keras.ops.convert_to_numpy(layer(x))
        f = tf.function(lambda t: layer(t),
                        input_signature=[tf.TensorSpec([None, None, None, 3], tf.float32)])
        np.testing.assert_allclose(eager, f(x).numpy(), atol=1e-6)

    # ---- serialization ----------------------------------------------

    def test_get_config_round_trip(self):
        layer = PositionEmbeddingSine2D(num_pos_feats=16, temperature=5000.0, normalize=False)
        rebuilt = PositionEmbeddingSine2D.from_config(layer.get_config())
        assert rebuilt.num_pos_feats == 16
        assert rebuilt.temperature == 5000.0
        assert rebuilt.normalize is False

    def test_keras_round_trip(self, tmp_path):
        inp = keras.Input(shape=(6, 5, 3), dtype="float32")
        out = PositionEmbeddingSine2D(num_pos_feats=8)(inp)
        model = keras.Model(inp, out)
        x = np.random.rand(2, 6, 5, 3).astype("float32")
        before = keras.ops.convert_to_numpy(model(x))
        path = os.path.join(tmp_path, "pe2d.keras")
        model.save(path)
        after = keras.ops.convert_to_numpy(keras.models.load_model(path)(x))
        np.testing.assert_allclose(before, after, atol=1e-6)


class TestOddNumPosFeatsIsRejectedAtConstruction:
    """SC4/§3.5: `num_pos_feats` must be EVEN.

    The class's own docstring already said so, but nothing enforced it, so an
    odd value survived construction and died at CALL time with
    `InvalidArgumentError: Shapes of all inputs must match:
    values[0].shape=[2,8,8,4] != values[1].shape=[2,8,8,3] [Op:Pack]`.
    This layer owns no weights and has no `build()`, so `__init__` is the only
    possible site for the check.
    """

    def test_ctor_rejects_odd_num_pos_feats(self):
        with pytest.raises(ValueError) as excinfo:
            PositionEmbeddingSine2D(num_pos_feats=7)
        assert "num_pos_feats" in str(excinfo.value)
        assert "even" in str(excinfo.value).lower()

    def test_even_values_still_accepted(self):
        assert PositionEmbeddingSine2D(num_pos_feats=8).num_pos_feats == 8

    def test_zero_still_rejected_for_being_non_positive(self):
        with pytest.raises(ValueError, match="positive"):
            PositionEmbeddingSine2D(num_pos_feats=0)


class TestOddNumPosFeatsStoredConfigStillLoads:
    """SC4/§6.3: the new raise must not brick a stored config.

    A config written before the raise existed can carry an odd
    `num_pos_feats`. `from_config` must substitute-and-warn so the archive
    still deserializes; only fresh construction is required to be correct.
    """

    def test_from_config_substitutes_and_warns(self, caplog):
        config = PositionEmbeddingSine2D(num_pos_feats=8).get_config()
        config["num_pos_feats"] = 7  # what an old archive can contain
        with caplog.at_level("WARNING"):
            layer = PositionEmbeddingSine2D.from_config(config)
        assert layer.num_pos_feats == 8
        assert layer.num_pos_feats % 2 == 0
        warnings_ = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
        assert any("num_pos_feats" in m for m in warnings_), warnings_

    def test_from_config_leaves_a_valid_even_value_alone(self):
        config = PositionEmbeddingSine2D(num_pos_feats=16, temperature=5000.0).get_config()
        layer = PositionEmbeddingSine2D.from_config(config)
        assert layer.num_pos_feats == 16
        assert layer.temperature == 5000.0

    def test_from_config_does_not_mutate_the_caller_dict(self):
        config = PositionEmbeddingSine2D(num_pos_feats=8).get_config()
        config["num_pos_feats"] = 7
        PositionEmbeddingSine2D.from_config(config)
        assert config["num_pos_feats"] == 7

    def test_the_substituted_layer_actually_runs(self):
        config = PositionEmbeddingSine2D(num_pos_feats=8).get_config()
        config["num_pos_feats"] = 7
        layer = PositionEmbeddingSine2D.from_config(config)
        x = keras.ops.convert_to_tensor(np.random.rand(2, 6, 5, 3).astype("float32"))
        assert tuple(layer(x).shape) == (2, 16, 6, 5)
