"""
Test suite for the rank-generalized :class:`GlobalSumPooling` layer.

Every expected value in this file is HAND-COMPUTED in the test source from the
literal input it asserts on. No expectation is ever read back from the layer
under test -- a self-referential oracle passes against any implementation,
including a broken one, and is a recorded defect class in this repository.

The layer assumes the Keras default ``channels_last`` layout, i.e. inputs of
shape ``(batch, *spatial, channels)``. There is no ``data_format`` parameter --
the layout is a deliberate, documented assumption -- so this file covers the
spatial rank axis (3/4/5) instead of a layout axis.
"""

import os
import tempfile
from typing import Any, Dict

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.layers.pooling.global_sum_pool import GlobalSumPooling


class TestGlobalSumPoolingForward:
    """Forward-pass values and shapes at ranks 3, 4 and 5."""

    @pytest.fixture
    def rank4_block(self) -> np.ndarray:
        """A rank-4 input whose per-channel sums are trivially checkable by hand.

        Shape ``(1, 2, 2, 2)`` holding ``1..8`` in row-major order, so channel 0
        carries ``1, 3, 5, 7`` and channel 1 carries ``2, 4, 6, 8``.

        :return: The input array, float32.
        :rtype: np.ndarray
        """
        return np.array(
            [[[[1.0, 2.0], [3.0, 4.0]],
              [[5.0, 6.0], [7.0, 8.0]]]],
            dtype="float32",
        )

    def test_rank3_default_sums_all_spatial_axes(self):
        """Rank-3 input, default axes: one spatial axis is summed away."""
        x = np.array(
            [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
             [[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]]],
            dtype="float32",
        )
        layer = GlobalSumPooling()

        out = keras.ops.convert_to_numpy(layer(x))

        # sample 0: ch0 = 1+3+5 = 9, ch1 = 2+4+6 = 12
        # sample 1: ch0 = 0+0+0 = 0, ch1 = 1+2+3 = 6
        assert out.shape == (2, 2)
        np.testing.assert_allclose(out, np.array([[9.0, 12.0], [0.0, 6.0]]))

    def test_rank4_default_sums_all_spatial_axes(self, rank4_block):
        """Rank-4 input, default axes: both spatial axes are summed away."""
        layer = GlobalSumPooling()

        out = keras.ops.convert_to_numpy(layer(rank4_block))

        # ch0 = 1+3+5+7 = 16, ch1 = 2+4+6+8 = 20
        assert out.shape == (1, 2)
        np.testing.assert_allclose(out, np.array([[16.0, 20.0]]))

    def test_rank4_uniform_input_sums_to_the_spatial_area(self):
        """``ones((2, 4, 5, 3))`` sums to exactly 20.0 in every channel."""
        x = np.ones((2, 4, 5, 3), dtype="float32")
        layer = GlobalSumPooling()

        out = keras.ops.convert_to_numpy(layer(x))

        assert out.shape == (2, 3)
        np.testing.assert_allclose(out, np.full((2, 3), 20.0))

    def test_rank5_default_sums_all_spatial_axes(self):
        """Rank-5 input, default axes: all three spatial axes are summed away."""
        x = np.array(
            [[[[[1.0, 2.0]]],
              [[[3.0, 4.0]]]]],
            dtype="float32",
        )
        layer = GlobalSumPooling()

        out = keras.ops.convert_to_numpy(layer(x))

        # shape (1, 2, 1, 1, 2): ch0 = 1+3 = 4, ch1 = 2+4 = 6
        assert out.shape == (1, 2)
        np.testing.assert_allclose(out, np.array([[4.0, 6.0]]))

    def test_rank5_uniform_input_sums_to_the_spatial_volume(self):
        """``ones((2, 2, 3, 4, 3))`` sums to exactly 24.0 in every channel."""
        x = np.ones((2, 2, 3, 4, 3), dtype="float32")
        layer = GlobalSumPooling()

        out = keras.ops.convert_to_numpy(layer(x))

        assert out.shape == (2, 3)
        np.testing.assert_allclose(out, np.full((2, 3), 24.0))

    def test_keepdims_preserves_reduced_axes_as_size_one(self, rank4_block):
        """``keepdims=True`` keeps the summed axes with extent 1, same values."""
        layer = GlobalSumPooling(keepdims=True)

        out = keras.ops.convert_to_numpy(layer(rank4_block))

        assert out.shape == (1, 1, 1, 2)
        np.testing.assert_allclose(out, np.array([[[[16.0, 20.0]]]]))

    def test_axes_as_single_int_sums_only_that_axis(self, rank4_block):
        """``axes=1`` sums the first spatial axis and leaves the second intact."""
        layer = GlobalSumPooling(axes=1)

        out = keras.ops.convert_to_numpy(layer(rank4_block))

        # w=0: ch0 = 1+5 = 6,  ch1 = 2+6 = 8
        # w=1: ch0 = 3+7 = 10, ch1 = 4+8 = 12
        assert out.shape == (1, 2, 2)
        np.testing.assert_allclose(
            out, np.array([[[6.0, 8.0], [10.0, 12.0]]])
        )

    def test_negative_axis_resolves_from_the_end(self, rank4_block):
        """``axes=-2`` is the last spatial axis of a rank-4 input (axis 2)."""
        layer = GlobalSumPooling(axes=-2)

        out = keras.ops.convert_to_numpy(layer(rank4_block))

        # h=0: ch0 = 1+3 = 4,  ch1 = 2+4 = 6
        # h=1: ch0 = 5+7 = 12, ch1 = 6+8 = 14
        assert out.shape == (1, 2, 2)
        np.testing.assert_allclose(
            out, np.array([[[4.0, 6.0], [12.0, 14.0]]])
        )

    def test_batch_independence(self):
        """Output row ``i`` depends only on input row ``i``."""
        x = keras.random.normal(shape=(4, 6, 7, 5), seed=1234)
        layer = GlobalSumPooling()

        full = keras.ops.convert_to_numpy(layer(x))
        rows = [
            keras.ops.convert_to_numpy(layer(x[i:i + 1]))
            for i in range(x.shape[0])
        ]

        np.testing.assert_allclose(
            full, np.concatenate(rows, axis=0), rtol=1e-5, atol=1e-5
        )


class TestGlobalSumPoolingValidation:
    """The six ways an invalid axis specification is refused."""

    def test_batch_axis_is_rejected(self):
        """``axes=0`` names the batch axis, which is never summable."""
        layer = GlobalSumPooling(axes=0)

        with pytest.raises(ValueError, match="is not a spatial axis"):
            layer.build((2, 4, 5, 3))

    def test_channel_axis_is_rejected(self):
        """``axes=-1`` names the channel axis, which is never summable."""
        layer = GlobalSumPooling(axes=-1)

        with pytest.raises(ValueError, match="is not a spatial axis"):
            layer.build((2, 4, 5, 3))

    def test_rank2_input_is_rejected(self):
        """A rank-2 input has no spatial axes at all."""
        layer = GlobalSumPooling()

        with pytest.raises(ValueError, match=r"rank >= 3"):
            layer.build((2, 3))

    def test_empty_axes_is_rejected(self):
        """An empty ``axes`` sequence selects nothing and is refused."""
        with pytest.raises(ValueError, match="axes must not be empty"):
            GlobalSumPooling(axes=())

    def test_duplicate_axes_are_rejected(self):
        """A literal duplicate is refused at construction time."""
        with pytest.raises(ValueError, match="must not contain duplicates"):
            GlobalSumPooling(axes=(1, 1))

    def test_duplicate_axes_via_negative_index_are_rejected(self):
        """``(1, -3)`` both resolve to axis 1 of a rank-4 input."""
        layer = GlobalSumPooling(axes=(1, -3))

        with pytest.raises(ValueError, match="duplicate axes"):
            layer.build((2, 4, 5, 3))


class TestGlobalSumPoolingShapeInference:
    """``compute_output_shape`` must be correct WITHOUT building the layer."""

    @pytest.mark.parametrize(
        "input_shape,expected",
        [
            ((2, 4, 3), (2, 3)),
            ((2, 4, 5, 3), (2, 3)),
            ((2, 2, 3, 4, 3), (2, 3)),
        ],
    )
    def test_compute_output_shape_on_unbuilt_layer(self, input_shape, expected):
        """Default axes, ranks 3/4/5, no build call first."""
        layer = GlobalSumPooling()

        assert not layer.built
        assert layer.compute_output_shape(input_shape) == expected
        assert not layer.built

    def test_compute_output_shape_on_unbuilt_layer_with_keepdims(self):
        """``keepdims=True`` replaces the summed axes with 1 rather than dropping them."""
        layer = GlobalSumPooling(keepdims=True)

        assert not layer.built
        assert layer.compute_output_shape((2, 4, 5, 3)) == (2, 1, 1, 3)

    def test_compute_output_shape_on_unbuilt_layer_with_partial_axes(self):
        """A partial ``axes`` selection drops only the selected axis."""
        layer = GlobalSumPooling(axes=1)

        assert not layer.built
        assert layer.compute_output_shape((2, 4, 5, 3)) == (2, 5, 3)


class TestGlobalSumPoolingSerialization:
    """Config completeness, ``from_config`` and a full ``.keras`` round-trip."""

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """A non-default configuration exercising both constructor arguments.

        :return: Constructor keyword arguments.
        :rtype: Dict[str, Any]
        """
        return {"axes": (1,), "keepdims": True}

    def test_get_config_lists_every_constructor_argument(self, layer_config):
        """Every ``__init__`` parameter appears in ``get_config()``."""
        layer = GlobalSumPooling(**layer_config)

        config = layer.get_config()

        assert config["axes"] == [1]
        assert config["keepdims"] is True

    def test_from_config_reproduces_the_layer(self, layer_config):
        """``from_config(get_config())`` reproduces both attributes and the values."""
        layer = GlobalSumPooling(**layer_config)
        x = np.ones((1, 2, 3, 2), dtype="float32")

        restored = GlobalSumPooling.from_config(layer.get_config())

        assert restored.axes == (1,)
        assert restored.keepdims is True
        # axes=1 sums the height axis of extent 2 over an all-ones input, and
        # keepdims=True holds that axis at extent 1 instead of dropping it.
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(restored(x)), np.full((1, 1, 3, 2), 2.0)
        )

    def test_serialization_round_trip_preserves_values(self):
        """Save and reload a model; the OUTPUT VALUES must match, not just shapes."""
        x = keras.random.normal(shape=(3, 5, 6, 4), seed=99)
        inputs = keras.Input(shape=(5, 6, 4))
        outputs = GlobalSumPooling()(inputs)
        model = keras.Model(inputs, outputs)
        original = keras.ops.convert_to_numpy(model(x))

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "global_sum_pool.keras")
            model.save(filepath)
            loaded = keras.models.load_model(filepath)
            reloaded = keras.ops.convert_to_numpy(loaded(x))

        np.testing.assert_allclose(original, reloaded, rtol=1e-6, atol=1e-6)

    def test_registered_under_the_package_qualified_key(self):
        """The layer resolves under ``dl_techniques.layers.pooling.global_sum_pool``."""
        key = "dl_techniques.layers.pooling.global_sum_pool>GlobalSumPooling"

        assert keras.saving.get_registered_name(GlobalSumPooling) == key
        assert keras.saving.get_registered_object(key) is GlobalSumPooling


class TestGlobalSumPoolingIntegration:
    """Gradient flow and use inside Sequential / Functional models."""

    def test_gradients_flow_to_an_upstream_dense(self):
        """A trainable layer BEFORE the pooling still receives a gradient."""
        x = keras.random.normal(shape=(2, 4, 5, 3), seed=7)
        inputs = keras.Input(shape=(4, 5, 3))
        h = keras.layers.Dense(6)(inputs)
        outputs = GlobalSumPooling()(h)
        model = keras.Model(inputs, outputs)

        with tf.GradientTape() as tape:
            loss = keras.ops.mean(keras.ops.square(model(x)))
        grads = tape.gradient(loss, model.trainable_variables)

        assert len(grads) == 2
        assert all(g is not None for g in grads)
        assert all(
            float(keras.ops.sum(keras.ops.abs(g))) > 0.0 for g in grads
        )

    def test_layer_in_sequential_model(self):
        """The layer composes inside ``keras.Sequential``."""
        model = keras.Sequential([
            keras.Input(shape=(8, 8, 3)),
            keras.layers.Conv2D(16, 3, activation="relu"),
            GlobalSumPooling(),
            keras.layers.Dense(1),
        ])

        out = model(keras.random.normal(shape=(4, 8, 8, 3), seed=5))

        assert tuple(out.shape) == (4, 1)

    def test_layer_in_functional_model(self):
        """The layer composes inside a functional graph."""
        inputs = keras.Input(shape=(8, 8, 3))
        h = keras.layers.Conv2D(16, 3, activation="relu")(inputs)
        h = GlobalSumPooling()(h)
        outputs = keras.layers.Dense(1)(h)
        model = keras.Model(inputs, outputs)

        out = model(keras.random.normal(shape=(4, 8, 8, 3), seed=5))

        assert tuple(out.shape) == (4, 1)
