"""
Tests for Per-Channel Loss Wrapper and Multi-Label Segmentation Losses.
"""

import pytest
import numpy as np
import keras
from keras import ops

from dl_techniques.losses.multi_labels_loss import (
    DiceLossPerChannel,
    PerChannelBinaryLoss,
    WeightedBinaryCrossEntropyLoss,
    WeightedBinaryFocalLoss,
    create_multilabel_segmentation_loss,
)
from dl_techniques.losses.segmentation_metrics import dice_scalar
from tests.optimizer_state import build_optimizer_state
from tests.test_losses.keras_mask_probe import (
    assert_a_per_sample_mask_is_left_to_keras,
    assert_refuses_a_keras_mask,
    assert_refuses_per_timestep_weights,
    with_keras_mask,
)


class TestWeightedBinaryFocalLoss:
    """Tests for WeightedBinaryFocalLoss."""

    def test_output_shape_matches_input(self):
        """Verify element-wise loss has same shape as input."""
        loss_fn = WeightedBinaryFocalLoss(alpha=0.25, gamma=2.0)
        y_true = keras.random.uniform((2, 8, 8, 3), minval=0, maxval=1)
        y_true = ops.cast(y_true > 0.5, "float32")
        y_pred = keras.random.uniform((2, 8, 8, 3), minval=0.1, maxval=0.9)

        loss_tensor = loss_fn.call(y_true, y_pred)

        assert ops.shape(loss_tensor) == (2, 8, 8, 3)

    def test_loss_is_positive(self):
        """Loss values should always be non-negative."""
        loss_fn = WeightedBinaryFocalLoss(alpha=0.25, gamma=2.0)
        y_true = keras.random.uniform((4, 16, 16, 5), minval=0, maxval=1)
        y_true = ops.cast(y_true > 0.5, "float32")
        y_pred = keras.random.uniform((4, 16, 16, 5), minval=0.01, maxval=0.99)

        loss_tensor = loss_fn.call(y_true, y_pred)

        assert ops.all(loss_tensor >= 0)

    def test_perfect_prediction_low_loss(self):
        """Perfect predictions should yield very low loss."""
        loss_fn = WeightedBinaryFocalLoss(alpha=0.25, gamma=2.0)
        y_true = ops.ones((2, 8, 8, 2))
        y_pred = ops.ones((2, 8, 8, 2)) * 0.999

        loss_tensor = loss_fn.call(y_true, y_pred)
        mean_loss = ops.mean(loss_tensor)

        assert float(mean_loss) < 0.01

    def test_wrong_prediction_high_loss(self):
        """Wrong predictions should yield higher loss."""
        loss_fn = WeightedBinaryFocalLoss(alpha=0.25, gamma=2.0)
        y_true = ops.ones((2, 8, 8, 2))
        y_pred = ops.ones((2, 8, 8, 2)) * 0.01

        loss_tensor = loss_fn.call(y_true, y_pred)
        mean_loss = ops.mean(loss_tensor)

        assert float(mean_loss) > 1.0

    def test_from_logits(self):
        """Test logits input mode."""
        loss_fn = WeightedBinaryFocalLoss(alpha=0.25, gamma=2.0, from_logits=True)
        y_true = ops.ones((2, 4, 4, 1))
        y_pred_logits = ops.ones((2, 4, 4, 1)) * 5.0  # High logits -> high prob

        loss_tensor = loss_fn.call(y_true, y_pred_logits)
        mean_loss = ops.mean(loss_tensor)

        assert float(mean_loss) < 0.01

    def test_gamma_effect(self):
        """Higher gamma should reduce loss for easy examples."""
        y_true = ops.ones((2, 8, 8, 2))
        y_pred = ops.ones((2, 8, 8, 2)) * 0.9  # Easy example

        loss_low_gamma = WeightedBinaryFocalLoss(alpha=0.25, gamma=0.5)
        loss_high_gamma = WeightedBinaryFocalLoss(alpha=0.25, gamma=4.0)

        loss_val_low = float(ops.mean(loss_low_gamma.call(y_true, y_pred)))
        loss_val_high = float(ops.mean(loss_high_gamma.call(y_true, y_pred)))

        assert loss_val_high < loss_val_low

    def test_serialization(self):
        """Test get_config and from_config."""
        loss_fn = WeightedBinaryFocalLoss(alpha=0.3, gamma=1.5, from_logits=True)
        config = loss_fn.get_config()

        restored = WeightedBinaryFocalLoss.from_config(config)

        assert restored.alpha == 0.3
        assert restored.gamma == 1.5
        assert restored.from_logits is True


class TestPerChannelBinaryLoss:
    """Tests for PerChannelBinaryLoss wrapper."""

    def test_output_is_one_value_per_sample(self):
        """Wrapped loss returns one value per sample (plan-2026-11-11T223825-64d1b401 step 2)."""
        loss_fn = PerChannelBinaryLoss(base_loss='binary_crossentropy')
        y_true = keras.random.uniform((2, 8, 8, 3), minval=0, maxval=1)
        y_true = ops.cast(y_true > 0.5, "float32")
        y_pred = keras.random.uniform((2, 8, 8, 3), minval=0.1, maxval=0.9)

        loss_val = loss_fn.call(y_true, y_pred)

        assert tuple(loss_val.shape) == (2,)

    def test_loss_is_positive(self):
        """Loss should be non-negative."""
        loss_fn = PerChannelBinaryLoss(base_loss='binary_crossentropy')
        y_true = keras.random.uniform((4, 16, 16, 5), minval=0, maxval=1)
        y_true = ops.cast(y_true > 0.5, "float32")
        y_pred = keras.random.uniform((4, 16, 16, 5), minval=0.1, maxval=0.9)

        loss_val = loss_fn.call(y_true, y_pred)

        assert float(ops.min(loss_val)) >= 0

    def test_with_focal_loss_base(self):
        """Test with WeightedBinaryFocalLoss as base."""
        focal = WeightedBinaryFocalLoss(alpha=0.25, gamma=2.0)
        loss_fn = PerChannelBinaryLoss(base_loss=focal)
        y_true = keras.random.uniform((2, 8, 8, 3), minval=0, maxval=1)
        y_true = ops.cast(y_true > 0.5, "float32")
        y_pred = keras.random.uniform((2, 8, 8, 3), minval=0.1, maxval=0.9)

        loss_val = loss_fn.call(y_true, y_pred)

        assert tuple(loss_val.shape) == (2,)
        assert float(ops.min(loss_val)) >= 0

    def test_channel_weights(self):
        """Test static channel weights are applied."""
        weights = [1.0, 2.0, 0.5]
        loss_fn = PerChannelBinaryLoss(
            base_loss='binary_crossentropy',
            channel_weights=weights
        )
        y_true = ops.ones((2, 8, 8, 3))
        y_pred = ops.ones((2, 8, 8, 3)) * 0.5

        loss_val = loss_fn.call(y_true, y_pred)

        assert tuple(loss_val.shape) == (2,)
        assert float(ops.min(loss_val)) > 0

    def test_adaptive_weighting_rare_class(self):
        """Rare class should get higher adaptive weight."""
        loss_fn = PerChannelBinaryLoss(base_loss='binary_crossentropy')

        # Channel 0: all positive, Channel 1: mostly negative (rare positive)
        y_true_np = np.zeros((2, 8, 8, 2), dtype=np.float32)
        y_true_np[:, :, :, 0] = 1.0
        y_true_np[:, 0, 0, 1] = 1.0  # Only one pixel positive in channel 1
        y_true = ops.convert_to_tensor(y_true_np)

        y_pred = ops.ones((2, 8, 8, 2)) * 0.5

        loss_val = loss_fn.call(y_true, y_pred)

        assert tuple(loss_val.shape) == (2,)

    def test_serialization_with_string_base(self):
        """Test serialization with string base loss."""
        loss_fn = PerChannelBinaryLoss(
            base_loss='binary_crossentropy',
            channel_weights=[1.0, 2.0]
        )
        config = loss_fn.get_config()

        restored = PerChannelBinaryLoss.from_config(config)

        assert restored.base_loss_name == 'binary_crossentropy'
        assert restored.channel_weights == [1.0, 2.0]

    def test_serialization_with_loss_instance(self):
        """Test serialization with Loss instance as base."""
        focal = WeightedBinaryFocalLoss(alpha=0.3, gamma=1.5)
        loss_fn = PerChannelBinaryLoss(base_loss=focal)
        config = loss_fn.get_config()

        restored = PerChannelBinaryLoss.from_config(config)

        assert restored.base_loss is not None
        assert restored.base_loss.alpha == 0.3
        assert restored.base_loss.gamma == 1.5

    def test_model_compile_and_fit(self):
        """Test loss works with model.compile and fit."""
        loss_fn = PerChannelBinaryLoss(base_loss='binary_crossentropy')

        inputs = keras.layers.Input(shape=(16, 16, 3))
        x = keras.layers.Conv2D(8, 3, padding='same', activation='relu')(inputs)
        outputs = keras.layers.Conv2D(2, 1, activation='sigmoid')(x)
        model = keras.Model(inputs, outputs)

        model.compile(optimizer='adam', loss=loss_fn)

        x_train = np.random.rand(4, 16, 16, 3).astype(np.float32)
        y_train = (np.random.rand(4, 16, 16, 2) > 0.5).astype(np.float32)

        history = model.fit(x_train, y_train, epochs=1, verbose=0)

        assert 'loss' in history.history
        assert len(history.history['loss']) == 1


class TestCreateMultilabelSegmentationLoss:
    """Tests for the factory function."""

    def test_focal_loss_creation(self):
        """Test focal loss type."""
        loss_fn = create_multilabel_segmentation_loss(
            loss_type='focal',
            alpha=0.75,
            gamma=2.0
        )

        assert isinstance(loss_fn, PerChannelBinaryLoss)
        assert isinstance(loss_fn.base_loss, WeightedBinaryFocalLoss)

    def test_dice_loss_creation(self):
        """Test dice loss type."""
        loss_fn = create_multilabel_segmentation_loss(
            loss_type='dice',
            smooth=0.5
        )

        # New implementation uses a thin wrapper around shared backend
        assert hasattr(loss_fn, 'smooth')
        assert loss_fn.smooth == 0.5

    def test_bce_loss_creation(self):
        """Test BCE loss type."""
        loss_fn = create_multilabel_segmentation_loss(loss_type='bce')

        assert isinstance(loss_fn, PerChannelBinaryLoss)
        assert loss_fn.base_loss_name == 'binary_crossentropy'

    def test_weighted_bce_loss_creation(self):
        """Test weighted BCE loss type."""
        loss_fn = create_multilabel_segmentation_loss(
            loss_type='weighted_bce',
            alpha=0.8
        )

        assert isinstance(loss_fn, PerChannelBinaryLoss)

    def test_channel_weights_passed(self):
        """Test channel weights are passed through."""
        weights = [1.0, 2.0, 3.0]
        loss_fn = create_multilabel_segmentation_loss(
            loss_type='focal',
            channel_weights=weights
        )

        assert loss_fn.channel_weights == weights

    def test_unknown_loss_type_raises(self):
        """Test unknown loss type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown loss_type"):
            create_multilabel_segmentation_loss(loss_type='unknown')

    def test_all_loss_types_forward_pass(self):
        """Test all loss types produce valid output."""
        loss_types = ['focal', 'dice', 'bce', 'weighted_bce']
        y_true = keras.random.uniform((2, 8, 8, 3), minval=0, maxval=1)
        y_true = ops.cast(y_true > 0.5, "float32")
        y_pred = keras.random.uniform((2, 8, 8, 3), minval=0.1, maxval=0.9)

        for loss_type in loss_types:
            loss_fn = create_multilabel_segmentation_loss(loss_type=loss_type)
            loss_val = loss_fn(y_true, y_pred)

            assert not np.isnan(float(loss_val)), f"{loss_type} produced NaN"
            assert float(loss_val) >= 0, f"{loss_type} produced negative loss"

    def test_weighted_bce_callable_works(self):
        """The weighted_bce factory builds a working registered base loss."""
        loss_fn = create_multilabel_segmentation_loss(loss_type='weighted_bce')
        y_true = keras.random.uniform((2, 8, 8, 3), minval=0, maxval=1)
        y_true = ops.cast(y_true > 0.5, "float32")
        y_pred = keras.random.uniform((2, 8, 8, 3), minval=0.1, maxval=0.9)

        loss_val = loss_fn(y_true, y_pred)
        assert not np.isnan(float(loss_val))
        assert float(loss_val) >= 0


class TestModelSaveLoad:
    """Tests for model save/load with custom losses."""

    def test_save_load_with_focal_loss(self, tmp_path):
        """Test model save/load with focal loss."""
        focal = WeightedBinaryFocalLoss(alpha=0.3, gamma=1.5)
        loss_fn = PerChannelBinaryLoss(base_loss=focal, channel_weights=[1.0, 2.0])

        inputs = keras.layers.Input(shape=(16, 16, 3))
        x = keras.layers.Conv2D(8, 3, padding='same', activation='relu')(inputs)
        outputs = keras.layers.Conv2D(2, 1, activation='sigmoid')(x)
        model = keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss=loss_fn)

        model_path = tmp_path / "model.keras"
        # The optimizer's slot variables are allocated lazily, so a compiled-but-
        # unfitted model would otherwise save an optimizer the reload cannot match.
        # See tests/optimizer_state.py (D-016).
        build_optimizer_state(model)
        model.save(model_path)

        loaded_model = keras.models.load_model(model_path)

        x_test = np.random.rand(2, 16, 16, 3).astype(np.float32)
        original_pred = model.predict(x_test, verbose=0)
        loaded_pred = loaded_model.predict(x_test, verbose=0)

        np.testing.assert_allclose(
            original_pred,
            loaded_pred,
            rtol=1e-6, atol=1e-6,
            err_msg="Predictions should match after save/load"
        )

    def test_save_load_with_dice_loss(self, tmp_path):
        """Test model save/load with dice loss."""
        loss_fn = create_multilabel_segmentation_loss(loss_type='dice', smooth=0.5)

        inputs = keras.layers.Input(shape=(16, 16, 3))
        x = keras.layers.Conv2D(8, 3, padding='same', activation='relu')(inputs)
        outputs = keras.layers.Conv2D(2, 1, activation='sigmoid')(x)
        model = keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss=loss_fn)

        model_path = tmp_path / "model_dice.keras"
        # The optimizer's slot variables are allocated lazily, so a compiled-but-
        # unfitted model would otherwise save an optimizer the reload cannot match.
        # See tests/optimizer_state.py (D-016).
        build_optimizer_state(model)
        model.save(model_path)

        loaded_model = keras.models.load_model(model_path)

        assert loaded_model.loss.smooth == 0.5


class TestNumericalStability:
    """Tests for numerical edge cases."""

    def test_focal_loss_extreme_predictions(self):
        """Test focal loss with predictions near 0 and 1."""
        loss_fn = WeightedBinaryFocalLoss(alpha=0.25, gamma=2.0)
        y_true = ops.ones((2, 4, 4, 2))
        y_pred_extreme = ops.concatenate([
            ops.ones((2, 4, 4, 1)) * 1e-7,
            ops.ones((2, 4, 4, 1)) * (1.0 - 1e-7)
        ], axis=-1)

        loss_val = loss_fn.call(y_true, y_pred_extreme)

        assert not np.isnan(float(ops.mean(loss_val)))
        assert not np.isinf(float(ops.mean(loss_val)))

    def test_per_channel_loss_single_channel(self):
        """Test with single channel input."""
        loss_fn = PerChannelBinaryLoss(base_loss='binary_crossentropy')
        y_true = keras.random.uniform((2, 8, 8, 1), minval=0, maxval=1)
        y_true = ops.cast(y_true > 0.5, "float32")
        y_pred = keras.random.uniform((2, 8, 8, 1), minval=0.1, maxval=0.9)

        loss_val = loss_fn.call(y_true, y_pred)

        assert tuple(loss_val.shape) == (2,)
        assert not np.isnan(float(ops.mean(loss_val)))

    def test_dice_loss_all_zeros_prediction(self):
        """Test dice loss with all zeros prediction."""
        loss_fn = create_multilabel_segmentation_loss(loss_type='dice', smooth=1.0)
        y_true = ops.ones((2, 8, 8, 2))
        y_pred = ops.zeros((2, 8, 8, 2))

        loss_val = loss_fn.call(y_true, y_pred)

        # One value per sample since plan-2026-11-11T223825-64d1b401 step 1.
        assert tuple(loss_val.shape) == (2,)
        assert not np.isnan(float(keras.ops.mean(loss_val)))

    def test_large_batch_size(self):
        """Test with larger batch size."""
        loss_fn = PerChannelBinaryLoss(base_loss='binary_crossentropy')
        y_true = keras.random.uniform((32, 64, 64, 8), minval=0, maxval=1)
        y_true = ops.cast(y_true > 0.5, "float32")
        y_pred = keras.random.uniform((32, 64, 64, 8), minval=0.1, maxval=0.9)

        loss_val = loss_fn.call(y_true, y_pred)

        assert tuple(loss_val.shape) == (32,)
        assert not np.isnan(float(ops.mean(loss_val)))


class TestGradientFlow:
    """Tests to verify gradients flow correctly."""

    def test_focal_loss_gradient_exists(self):
        """Verify gradients flow through focal loss."""
        import tensorflow as tf

        loss_fn = WeightedBinaryFocalLoss(alpha=0.25, gamma=2.0)
        y_true = ops.ones((2, 4, 4, 2))
        y_pred = tf.Variable(ops.ones((2, 4, 4, 2)) * 0.5)

        with tf.GradientTape() as tape:
            loss = ops.mean(loss_fn.call(y_true, y_pred))

        grads = tape.gradient(loss, y_pred)

        assert grads is not None
        assert not np.any(np.isnan(keras.ops.convert_to_numpy(grads)))

    def test_dice_loss_gradient_exists(self):
        """Verify gradients flow through dice loss."""
        import tensorflow as tf

        loss_fn = create_multilabel_segmentation_loss(loss_type='dice', smooth=1.0)
        y_true = ops.ones((2, 4, 4, 2))
        y_pred = tf.Variable(ops.ones((2, 4, 4, 2)) * 0.5)

        with tf.GradientTape() as tape:
            loss = loss_fn.call(y_true, y_pred)

        grads = tape.gradient(loss, y_pred)

        assert grads is not None
        assert not np.any(np.isnan(keras.ops.convert_to_numpy(grads)))

    def test_per_channel_loss_gradient_exists(self):
        """Verify gradients flow through per-channel wrapper."""
        import tensorflow as tf

        loss_fn = PerChannelBinaryLoss(base_loss='binary_crossentropy')
        y_true = ops.ones((2, 4, 4, 2))
        y_pred = tf.Variable(ops.ones((2, 4, 4, 2)) * 0.5)

        with tf.GradientTape() as tape:
            loss = loss_fn.call(y_true, y_pred)

        grads = tape.gradient(loss, y_pred)

        assert grads is not None
        assert not np.any(np.isnan(keras.ops.convert_to_numpy(grads)))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# DECISION plan-2026-11-11T131349-13b28b37/D-009 (tests): PerChannelBinaryLoss only
# accepts bases that return one value per element; a base that reduces the channel
# axis collapses the loss to a scalar and silently disables channel_weights.
class TestPerChannelBinaryLossRefusesABaseThatCollapsesChannels:
    """A string or stock Loss base reduced the channel axis and made weights a no-op."""

    @pytest.mark.parametrize("name", ["mse", "binary_focal_crossentropy", "mae"])
    def test_a_string_base_other_than_binary_crossentropy_raises(self, name):
        with pytest.raises(ValueError, match="binary_crossentropy"):
            PerChannelBinaryLoss(name)

    def test_a_stock_loss_that_reduces_the_channel_axis_raises_at_call(self):
        loss_fn = PerChannelBinaryLoss(keras.losses.BinaryCrossentropy())
        y_true = ops.cast(keras.random.uniform((2, 8, 8, 3)) > 0.5, "float32")
        y_pred = keras.random.uniform((2, 8, 8, 3), minval=0.1, maxval=0.9)
        with pytest.raises(ValueError, match="rank"):
            loss_fn.call(y_true, y_pred)

    def test_a_base_that_keeps_a_size_one_channel_axis_raises_at_call(self):
        """Rank (B, H, W, 1) passes the rank check but broadcasts against the weights.

        Measured before the width check: channel_weights [10, 1, .1] and None gave
        the same value for such a base, so the weights were a silent no-op.
        """

        @keras.saving.register_keras_serializable(package="test_multi_labels_loss")
        class SizeOneChannelLoss(keras.losses.Loss):
            def call(self, y_true, y_pred):
                err = ops.square(y_true - y_pred)
                return ops.mean(err, axis=-1, keepdims=True)

        y_true = ops.cast(keras.random.uniform((2, 8, 8, 3), seed=1) > 0.5, "float32")
        y_pred = keras.random.uniform((2, 8, 8, 3), minval=0.1, maxval=0.9, seed=2)
        loss_fn = PerChannelBinaryLoss(SizeOneChannelLoss(), channel_weights=[10.0, 1.0, 0.1])
        with pytest.raises(ValueError, match="channel axis"):
            loss_fn.call(y_true, y_pred)

    def test_a_per_channel_base_with_a_symbolic_channel_width_still_works(self):
        """The width check compares only static sizes: a per-channel base under fit passes."""
        loss_fn = PerChannelBinaryLoss(
            WeightedBinaryFocalLoss(alpha=0.5, gamma=1.0), channel_weights=[10.0, 1.0, 0.1]
        )
        inputs = keras.layers.Input(shape=(8, 8, 3))
        outputs = keras.layers.Conv2D(3, 1, activation="sigmoid")(inputs)
        model = keras.Model(inputs, outputs)
        model.compile(optimizer="adam", loss=loss_fn)
        x = keras.random.uniform((4, 8, 8, 3), seed=3)
        y = ops.cast(keras.random.uniform((4, 8, 8, 3), seed=4) > 0.5, "float32")
        history = model.fit(x, y, epochs=1, batch_size=2, verbose=0)
        assert np.isfinite(history.history["loss"][0])

    def test_channel_weights_change_the_value_with_an_elementwise_base(self):
        y_true = ops.cast(keras.random.uniform((2, 8, 8, 3), seed=1) > 0.5, "float32")
        y_pred = keras.random.uniform((2, 8, 8, 3), minval=0.1, maxval=0.9, seed=2)
        base = WeightedBinaryFocalLoss(alpha=0.5, gamma=1.0)
        plain = float(ops.mean(PerChannelBinaryLoss(base).call(y_true, y_pred)))
        weighted = float(ops.mean(
            PerChannelBinaryLoss(base, channel_weights=[10.0, 1.0, 0.1]).call(y_true, y_pred)
        ))
        assert abs(plain - weighted) > 1e-4, (plain, weighted)


class TestWeightedBinaryCrossEntropyLoss:
    """The registered class that replaces the unserializable ``weighted_bce_fn`` closure."""

    def test_is_exported_from_the_package(self):
        import dl_techniques.losses as losses_pkg

        assert "WeightedBinaryCrossEntropyLoss" in losses_pkg.__all__
        assert hasattr(losses_pkg, "WeightedBinaryCrossEntropyLoss")

    def test_pos_weight_one_is_elementwise_bce(self):
        y_true = ops.cast(keras.random.uniform((2, 4, 4, 3), seed=3) > 0.5, "float32")
        y_pred = keras.random.uniform((2, 4, 4, 3), minval=0.1, maxval=0.9, seed=4)
        got = ops.convert_to_numpy(WeightedBinaryCrossEntropyLoss(pos_weight=1.0).call(y_true, y_pred))
        yt, yp = ops.convert_to_numpy(y_true), ops.convert_to_numpy(y_pred)
        want = -(yt * np.log(yp) + (1 - yt) * np.log(1 - yp))
        assert got.shape == (2, 4, 4, 3)
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)

    def test_pos_weight_scales_only_the_positive_term(self):
        y_true = ops.ones((1, 2, 2, 1))
        y_pred = ops.ones((1, 2, 2, 1)) * 0.5
        one = ops.convert_to_numpy(WeightedBinaryCrossEntropyLoss(pos_weight=1.0).call(y_true, y_pred))
        three = ops.convert_to_numpy(WeightedBinaryCrossEntropyLoss(pos_weight=3.0).call(y_true, y_pred))
        np.testing.assert_allclose(three, 3.0 * one, rtol=1e-5)
        neg = ops.zeros((1, 2, 2, 1))
        a = ops.convert_to_numpy(WeightedBinaryCrossEntropyLoss(pos_weight=1.0).call(neg, y_pred))
        b = ops.convert_to_numpy(WeightedBinaryCrossEntropyLoss(pos_weight=3.0).call(neg, y_pred))
        np.testing.assert_allclose(a, b, rtol=1e-6)

    def test_config_round_trip(self):
        loss = WeightedBinaryCrossEntropyLoss(pos_weight=4.0)
        restored = WeightedBinaryCrossEntropyLoss.from_config(loss.get_config())
        assert restored.pos_weight == 4.0

    def test_rejects_a_nonpositive_pos_weight(self):
        with pytest.raises(ValueError, match="pos_weight"):
            WeightedBinaryCrossEntropyLoss(pos_weight=0.0)

    def test_factory_weighted_bce_uses_the_alpha_odds_as_pos_weight(self):
        loss_fn = create_multilabel_segmentation_loss(loss_type="weighted_bce", alpha=0.75)
        assert isinstance(loss_fn.base_loss, WeightedBinaryCrossEntropyLoss)
        assert loss_fn.base_loss.pos_weight == pytest.approx(3.0)
        capped = create_multilabel_segmentation_loss(loss_type="weighted_bce", alpha=1.0)
        assert capped.base_loss.pos_weight == 10.0

    def test_factory_weighted_bce_config_round_trips(self):
        loss_fn = create_multilabel_segmentation_loss(
            loss_type="weighted_bce", alpha=0.8, channel_weights=[1.0, 2.0, 3.0]
        )
        restored = PerChannelBinaryLoss.from_config(loss_fn.get_config())
        y_true = ops.cast(keras.random.uniform((2, 8, 8, 3), seed=5) > 0.5, "float32")
        y_pred = keras.random.uniform((2, 8, 8, 3), minval=0.1, maxval=0.9, seed=6)
        assert isinstance(restored.base_loss, WeightedBinaryCrossEntropyLoss)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(restored.call(y_true, y_pred)),
            keras.ops.convert_to_numpy(loss_fn.call(y_true, y_pred)),
            rtol=1e-6,
        )

    def test_factory_weighted_bce_survives_a_model_save_and_load(self, tmp_path):
        loss_fn = create_multilabel_segmentation_loss(loss_type="weighted_bce", alpha=0.8)
        inputs = keras.layers.Input(shape=(8, 8, 3))
        outputs = keras.layers.Conv2D(2, 1, activation="sigmoid")(inputs)
        model = keras.Model(inputs, outputs)
        model.compile(optimizer="adam", loss=loss_fn)
        path = str(tmp_path / "m.keras")
        build_optimizer_state(model)  # lazy slot variables; see tests/optimizer_state.py
        model.save(path)
        loaded = keras.models.load_model(path)
        assert isinstance(loaded.loss.base_loss, WeightedBinaryCrossEntropyLoss)
        assert loaded.loss.base_loss.pos_weight == pytest.approx(4.0)


# ---------------------------------------------------------------------
# DiceLossPerChannel returns one value per sample
# (plan-2026-11-11T223825-64d1b401, step 1)
# ---------------------------------------------------------------------

DICE_BATCH = 4
# float64 reference vs a float32 sum over 25 pixels: measured max error 6e-8.
DICE_ROW_ATOL = 2e-7
# Default-reduction value of `DiceLossPerChannel()(y, p)` on `_ragged_seg_batch()`
# measured at HEAD b4831ec33, BEFORE the return became `(batch,)`
# (plans/plan-2026-11-11T223825-64d1b401/scratch_golden_head.json, case dice/ragged).
DICE_HEAD_GOLDEN = 0.5451357364654541
DICE_SMOOTH = 1.0


def _ragged_seg_batch():
    """Binary masks with a DIFFERENT foreground density per row (0.15 to 0.8).

    Rows must differ: if every row scored the same, a scalar broadcast to every
    row would be indistinguishable from a per-row value.
    """
    r = np.random.default_rng(7)
    density = np.array([0.15, 0.35, 0.55, 0.8])[:, None, None, None]
    y_true = (r.random((DICE_BATCH, 5, 5, 3)) < density).astype("float32")
    y_pred = r.uniform(0.05, 0.95, (DICE_BATCH, 5, 5, 3)).astype("float32")
    return y_true, y_pred


def _dice_rows_float64(y_true, y_pred, smooth=DICE_SMOOTH):
    """Independent reference: 1 - mean_c (2 TP + s) / (2 TP + FP + FN + s), per image."""
    yt = y_true.astype("float64")
    yp = y_pred.astype("float64")
    tp = (yt * yp).sum(axis=(1, 2))
    fp = ((1.0 - yt) * yp).sum(axis=(1, 2))
    fn = (yt * (1.0 - yp)).sum(axis=(1, 2))
    return 1.0 - ((2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)).mean(axis=-1)


def _drop_last_row_gap(make_loss):
    """``|loss(w=[1,..,1,0]) * B/(B-1) - loss(batch[:B-1])|`` for a loss factory.

    Zero for a loss whose rows are independent; the scalar-broadcast defect leaves
    it at the batch-versus-subset difference.
    """
    y_true, y_pred = _ragged_seg_batch()
    w = np.array([1.0, 1.0, 1.0, 0.0], "float32")
    weighted = float(make_loss()(y_true, y_pred, sample_weight=w))
    dropped = float(make_loss()(y_true[:-1], y_pred[:-1]))
    return abs(weighted * DICE_BATCH / (DICE_BATCH - 1) - dropped)


class _ScalarDice(DiceLossPerChannel):
    """The defect re-injected: the batch mean returned instead of `(batch,)`."""

    def call(self, y_true, y_pred):
        return keras.ops.mean(super().call(y_true, y_pred))


class TestDiceLossPerChannelIsPerSample:
    """`DiceLossPerChannel` charges each image its own Dice loss.

    ``keras.losses.Loss.__call__`` multiplies ``call()`` by ``sample_weight`` and only
    then reduces, so a scalar return discards WHICH rows were weighted.
    """

    def test_call_returns_one_value_per_sample(self):
        y_true, y_pred = _ragged_seg_batch()
        out = DiceLossPerChannel(smooth=DICE_SMOOTH).call(y_true, y_pred)
        assert tuple(out.shape) == (DICE_BATCH,)

    def test_reduction_none_returns_one_value_per_sample(self):
        y_true, y_pred = _ragged_seg_batch()
        out = DiceLossPerChannel(smooth=DICE_SMOOTH, reduction="none")(y_true, y_pred)
        assert tuple(out.shape) == (DICE_BATCH,)

    def test_rows_equal_an_independent_float64_reference(self):
        y_true, y_pred = _ragged_seg_batch()
        got = keras.ops.convert_to_numpy(
            DiceLossPerChannel(smooth=DICE_SMOOTH, reduction="none")(y_true, y_pred)
        )
        ref = _dice_rows_float64(y_true, y_pred)
        # The rows must differ, or a broadcast scalar would pass this test.
        assert np.ptp(ref) > 0.05, ref
        np.testing.assert_allclose(got, ref, atol=DICE_ROW_ATOL, rtol=0.0)

    def test_default_value_is_unchanged(self):
        """The default reduction is the batch mean it always was (passes before AND after)."""
        y_true, y_pred = _ragged_seg_batch()
        value = float(DiceLossPerChannel(smooth=DICE_SMOOTH)(y_true, y_pred))
        assert abs(value - DICE_HEAD_GOLDEN) < 1e-6, (value, DICE_HEAD_GOLDEN)
        assert abs(value - _dice_rows_float64(y_true, y_pred).mean()) < 1e-6

    def test_default_gradient_is_unchanged(self):
        """Gradient of the default value equals that of `dice_scalar`, the old return."""
        import tensorflow as tf

        y_true, y_pred = _ragged_seg_batch()
        grads = []
        for fn in (
            lambda v: DiceLossPerChannel(smooth=DICE_SMOOTH)(y_true, v),
            lambda v: dice_scalar(y_true, v, from_logits=False, smooth=DICE_SMOOTH),
        ):
            v = tf.Variable(y_pred)
            with tf.GradientTape() as tape:
                loss = fn(v)
            grads.append(tape.gradient(loss, v).numpy())
        np.testing.assert_allclose(grads[0], grads[1], atol=1e-9, rtol=0.0)

    def test_sample_weight_charges_each_row_its_own_weight(self):
        y_true, y_pred = _ragged_seg_batch()
        w = np.array([1.0, 0.5, 2.0, 0.0], "float32")
        got = float(DiceLossPerChannel(smooth=DICE_SMOOTH)(y_true, y_pred, sample_weight=w))
        want = float((w * _dice_rows_float64(y_true, y_pred)).sum() / DICE_BATCH)
        assert abs(got - want) < 1e-6, (got, want)

    def test_zeroing_a_sample_weight_equals_dropping_the_row(self):
        """No batch term: `loss(w=[1,1,1,0]) * B/(B-1) == loss(batch[:3])`."""
        gap = _drop_last_row_gap(lambda: DiceLossPerChannel(smooth=DICE_SMOOTH))
        assert gap < 1e-6, gap

    def test_sum_reduction_is_batch_times_the_default(self):
        y_true, y_pred = _ragged_seg_batch()
        default = float(DiceLossPerChannel(smooth=DICE_SMOOTH)(y_true, y_pred))
        total = float(DiceLossPerChannel(smooth=DICE_SMOOTH, reduction="sum")(y_true, y_pred))
        assert abs(total - DICE_BATCH * default) < 1e-6, (total, default)

    def test_a_zero_weighted_row_gets_exactly_zero_gradient(self):
        import tensorflow as tf

        y_true, y_pred = _ragged_seg_batch()
        w = np.array([1.0, 1.0, 1.0, 0.0], "float32")
        v = tf.Variable(y_pred)
        with tf.GradientTape() as tape:
            loss = DiceLossPerChannel(smooth=DICE_SMOOTH)(y_true, v, sample_weight=w)
        grad = tape.gradient(loss, v).numpy()
        assert np.abs(grad[:3]).max() > 0.0
        assert np.abs(grad[3]).max() == 0.0

    def test_the_scalar_return_is_caught_by_these_arms(self):
        """ANTI-VACUITY: the defect re-injected must fail the shape and identity arms."""
        y_true, y_pred = _ragged_seg_batch()
        scalar = _ScalarDice(smooth=DICE_SMOOTH)
        assert tuple(scalar.call(y_true, y_pred).shape) == ()
        assert tuple(_ScalarDice(smooth=DICE_SMOOTH, reduction="none")(y_true, y_pred).shape) == ()
        assert _drop_last_row_gap(lambda: _ScalarDice(smooth=DICE_SMOOTH)) > 1e-3


def _seg_batch_of(batch):
    """`_ragged_seg_batch()` tiled or cut to ``batch`` images (5 makes batch == the 5-pixel axes)."""
    y_true, y_pred = _ragged_seg_batch()
    index = np.arange(batch) % DICE_BATCH
    return y_true[index], y_pred[index]


class TestPerSampleSegmentationLossesRefuseWhatTheyCannotCombine:
    """A Keras mask or a per-timestep weight cannot be combined with ``(batch,)`` rows.

    Keras multiplies the mask of ``y_pred`` (here ``(batch, 5, 5)``) into the rows. With
    ``batch == 5`` that broadcast succeeds and silently mis-weights the rows; with any
    other batch it is an opaque shape error. Dice and ``PerChannelBinaryLoss`` give the
    mask no meaning, so ``__call__`` refuses it with a ``ValueError`` that says so
    (plan-2026-11-11T223825-64d1b401, D-014).
    """

    MAKERS = {
        "dice": lambda **k: DiceLossPerChannel(smooth=DICE_SMOOTH, **k),
        "pcb": lambda **k: PerChannelBinaryLoss("binary_crossentropy", **k),
    }

    @pytest.mark.parametrize("name", sorted(MAKERS))
    @pytest.mark.parametrize("batch", [4, 5])
    @pytest.mark.parametrize("axes", [(5, 5), (5,)])
    def test_a_keras_mask_on_y_pred_raises_a_clear_error(self, name, batch, axes):
        y_true, y_pred = _seg_batch_of(batch)
        assert_refuses_a_keras_mask(self.MAKERS[name], y_true, y_pred, (batch,) + axes)

    @pytest.mark.parametrize("name", sorted(MAKERS))
    @pytest.mark.parametrize("batch", [4, 5])
    def test_a_per_timestep_sample_weight_raises_a_clear_error(self, name, batch):
        y_true, y_pred = _seg_batch_of(batch)
        assert_refuses_per_timestep_weights(self.MAKERS[name], y_true, y_pred, (batch, 5))
        assert_refuses_per_timestep_weights(self.MAKERS[name], y_true, y_pred, (batch, 5, 5))

    @pytest.mark.parametrize("name", sorted(MAKERS))
    def test_a_per_sample_mask_and_a_column_weight_still_work(self, name):
        y_true, y_pred = _seg_batch_of(4)
        assert_a_per_sample_mask_is_left_to_keras(self.MAKERS[name], y_true, y_pred)
        w = np.array([1.0, 0.5, 2.0, 0.0], "float32")
        flat = float(self.MAKERS[name]()(y_true, y_pred, sample_weight=w))
        column = float(self.MAKERS[name]()(y_true, y_pred, sample_weight=w[:, None]))
        assert abs(flat - column) < 1e-7

    @pytest.mark.parametrize("name", sorted(MAKERS))
    def test_the_refusal_names_the_class_and_the_way_out(self, name):
        y_true, y_pred = _seg_batch_of(4)
        loss = self.MAKERS[name]()
        with pytest.raises(ValueError) as error:
            loss(y_true, with_keras_mask(y_pred, np.ones((4, 5, 5), bool)))
        text = str(error.value)
        assert type(loss).__name__ in text
        assert "(4, 5, 5)" in text and "sample_weight" in text

    @pytest.mark.parametrize("cls", [DiceLossPerChannel, PerChannelBinaryLoss])
    def test_the_docstrings_state_the_mask_rule(self, cls):
        doc = " ".join((cls.__doc__ or "").split())
        assert "Keras mask" in doc and "ValueError" in doc


# ---------------------------------------------------------------------
# PerChannelBinaryLoss returns one value per sample, batch-coupled weights
# (plan-2026-11-11T223825-64d1b401, step 2)
# ---------------------------------------------------------------------

PCB_ROW_ATOL = 5e-7
PCB_CHANNEL_WEIGHTS = [3.0, 1.0, 0.2]
# Default-reduction values measured at HEAD b4831ec33, BEFORE the return became
# `(batch,)`, on `_ragged_seg_batch()`
# (plans/plan-2026-11-11T223825-64d1b401/scratch_golden_head.json, cases pcb_*/ragged).
PCB_HEAD_GOLDEN = {
    "bce": 0.8499828577041626,
    "bce_channel_weights": 0.8297531008720398,
    "weighted_bce_3": 1.5535728931427002,
    "focal": 0.22829985618591309,
}


def _bce_elements(yt, yp):
    return -(yt * np.log(yp) + (1.0 - yt) * np.log(1.0 - yp))


def _weighted_bce_elements(yt, yp):
    return -3.0 * yt * np.log(yp) - (1.0 - yt) * np.log(1.0 - yp)


def _focal_elements(yt, yp, alpha=0.25, gamma=2.0):
    pos = -alpha * (1.0 - yp) ** gamma * np.log(yp)
    neg = -(1.0 - alpha) * yp ** gamma * np.log(1.0 - yp)
    return yt * pos + (1.0 - yt) * neg


# name -> (loss factory, float64 element loss, static channel weights)
PCB_CASES = {
    "bce": (lambda: PerChannelBinaryLoss("binary_crossentropy"), _bce_elements, None),
    "bce_channel_weights": (
        lambda: PerChannelBinaryLoss("binary_crossentropy", channel_weights=PCB_CHANNEL_WEIGHTS),
        _bce_elements,
        PCB_CHANNEL_WEIGHTS,
    ),
    "weighted_bce_3": (
        lambda: PerChannelBinaryLoss(WeightedBinaryCrossEntropyLoss(3.0)),
        _weighted_bce_elements,
        None,
    ),
    "focal": (lambda: PerChannelBinaryLoss(WeightedBinaryFocalLoss()), _focal_elements, None),
}


def _pcb_rows_float64(y_true, y_pred, element, channel_weights=None):
    """Independent reference: ``sum_c wn_c * mean_hw(l_ihwc)`` with ``wn`` from THIS batch.

    ``wn_c`` is the normalized ``(sqrt(count_c / (B*H*W)) + 0.1) * channel_weights_c``;
    ``count_c`` is the positive count over every row passed in, so passing a subset of
    the rows recomputes the weights over that subset.
    """
    yt = y_true.astype("float64")
    yp = y_pred.astype("float64")
    b, h, w, _ = yt.shape
    raw = np.sqrt(yt.sum(axis=(0, 1, 2)) / (b * h * w)) + 0.1
    if channel_weights is not None:
        raw = raw * np.asarray(channel_weights, dtype="float64")
    wn = raw / raw.sum()
    return (element(yt, yp).mean(axis=(1, 2)) * wn).sum(axis=-1)


def _coupled_seg_batch():
    """`_ragged_seg_batch()` whose LAST row is all-positive in channel 0.

    Dropping that row changes the class-presence counts, so the channel weights
    computed over the 3 kept rows differ from the ones computed over all 4 rows.
    """
    y_true, y_pred = _ragged_seg_batch()
    y_true = y_true.copy()
    y_true[-1, :, :, 0] = 1.0
    return y_true, y_pred


class _ScalarPerChannelBinaryLoss(PerChannelBinaryLoss):
    """The defect re-injected: the batch mean returned instead of `(batch,)`."""

    def call(self, y_true, y_pred):
        return keras.ops.mean(super().call(y_true, y_pred))


class TestPerChannelBinaryLossIsPerSample:
    """`PerChannelBinaryLoss` charges each image its own weighted channel mean.

    ``keras.losses.Loss.__call__`` multiplies ``call()`` by ``sample_weight`` and only
    then reduces, so a scalar return discards WHICH rows were weighted. The channel
    weights are a batch statistic and stay one; see the coupling test below.
    """

    @pytest.mark.parametrize("case", sorted(PCB_CASES))
    def test_call_returns_one_value_per_sample(self, case):
        y_true, y_pred = _ragged_seg_batch()
        out = PCB_CASES[case][0]().call(y_true, y_pred)
        assert tuple(out.shape) == (DICE_BATCH,)

    @pytest.mark.parametrize("case", sorted(PCB_CASES))
    def test_reduction_none_returns_one_value_per_sample(self, case):
        y_true, y_pred = _ragged_seg_batch()
        loss_fn = PCB_CASES[case][0]()
        loss_fn.reduction = "none"
        assert tuple(loss_fn(y_true, y_pred).shape) == (DICE_BATCH,)

    @pytest.mark.parametrize("case", sorted(PCB_CASES))
    def test_rows_equal_an_independent_float64_reference(self, case):
        make, element, channel_weights = PCB_CASES[case]
        y_true, y_pred = _ragged_seg_batch()
        got = keras.ops.convert_to_numpy(make().call(y_true, y_pred))
        ref = _pcb_rows_float64(y_true, y_pred, element, channel_weights)
        # The rows must differ, or a broadcast scalar would pass this test.
        assert np.ptp(ref) > 0.02, ref
        np.testing.assert_allclose(got, ref, atol=PCB_ROW_ATOL, rtol=0.0)

    @pytest.mark.parametrize("case", sorted(PCB_CASES))
    def test_default_value_is_unchanged(self, case):
        """The default reduction is the scalar it always was (passes before AND after)."""
        make, element, channel_weights = PCB_CASES[case]
        y_true, y_pred = _ragged_seg_batch()
        value = float(make()(y_true, y_pred))
        assert abs(value - PCB_HEAD_GOLDEN[case]) < PCB_ROW_ATOL, (value, PCB_HEAD_GOLDEN[case])
        ref = _pcb_rows_float64(y_true, y_pred, element, channel_weights).mean()
        assert abs(value - ref) < PCB_ROW_ATOL, (value, ref)

    def test_default_gradient_is_unchanged(self):
        """The gradient of the default value equals that of the old scalar (float64 reference)."""
        import tensorflow as tf

        y_true, y_pred = _ragged_seg_batch()
        v = tf.Variable(y_pred)
        with tf.GradientTape() as tape:
            loss = PerChannelBinaryLoss("binary_crossentropy")(y_true, v)
        got = tape.gradient(loss, v).numpy()

        b, h, w, c = y_true.shape
        yt = y_true.astype("float64")
        yp = y_pred.astype("float64")
        raw = np.sqrt(yt.sum(axis=(0, 1, 2)) / (b * h * w)) + 0.1
        wn = raw / raw.sum()
        # d/dp of the old scalar sum_c wn_c * mean_{bhw} l, with l = -(y log p + (1-y) log(1-p)).
        want = wn * (-(yt / yp) + (1.0 - yt) / (1.0 - yp)) / (b * h * w)
        # float32 gradients reach 0.064 (one ulp 7.5e-9): measured max error 8.7e-9.
        np.testing.assert_allclose(got, want, atol=5e-8, rtol=0.0)

    @pytest.mark.parametrize("case", sorted(PCB_CASES))
    def test_sample_weight_charges_each_row_its_own_weight(self, case):
        make, element, channel_weights = PCB_CASES[case]
        y_true, y_pred = _ragged_seg_batch()
        w = np.array([1.0, 0.5, 2.0, 0.0], "float32")
        got = float(make()(y_true, y_pred, sample_weight=w))
        want = (w * _pcb_rows_float64(y_true, y_pred, element, channel_weights)).sum() / DICE_BATCH
        assert abs(got - want) < PCB_ROW_ATOL, (got, want)

    def test_sum_reduction_is_batch_times_the_default(self):
        y_true, y_pred = _ragged_seg_batch()
        default = float(PerChannelBinaryLoss("binary_crossentropy")(y_true, y_pred))
        total = float(
            PerChannelBinaryLoss("binary_crossentropy", reduction="sum")(y_true, y_pred)
        )
        assert abs(total - DICE_BATCH * default) < 4 * PCB_ROW_ATOL, (total, default)

    def test_a_zero_weighted_row_gets_exactly_zero_gradient(self):
        """The row's OWN loss leaves the gradient; its class counts (a constant) do not matter."""
        import tensorflow as tf

        y_true, y_pred = _ragged_seg_batch()
        w = np.array([1.0, 1.0, 1.0, 0.0], "float32")
        v = tf.Variable(y_pred)
        with tf.GradientTape() as tape:
            loss = PerChannelBinaryLoss("binary_crossentropy")(y_true, v, sample_weight=w)
        grad = tape.gradient(loss, v).numpy()
        assert np.abs(grad[:3]).max() > 0.0
        assert np.abs(grad[3]).max() == 0.0

    # DECISION plan-2026-11-11T223825-64d1b401/D-002 (test): the guard for the
    # batch-level channel weights in PerChannelBinaryLoss.call. It fails if the
    # weights become per-sample or are recomputed over the rows a sample_weight keeps.
    def test_a_zero_sample_weight_keeps_the_row_s_class_counts_in_the_weights(self):
        """BATCH-COUPLED: zeroing a row does NOT equal dropping it, and the gap is pinned.

        The channel weights come from the class-presence counts of the WHOLE batch and
        ``call()`` never sees ``sample_weight``, so a zero weight removes the row's loss
        but not its counts. The last row is all-positive in channel 0 here, so keeping or
        removing it moves the weights; measured gap 1.55e-3.
        """
        y_true, y_pred = _coupled_seg_batch()
        w = np.array([1.0, 1.0, 1.0, 0.0], "float32")
        weighted = float(PerChannelBinaryLoss("binary_crossentropy")(y_true, y_pred, sample_weight=w))
        dropped = float(PerChannelBinaryLoss("binary_crossentropy")(y_true[:-1], y_pred[:-1]))

        full = _pcb_rows_float64(y_true, y_pred, _bce_elements)
        kept = _pcb_rows_float64(y_true[:-1], y_pred[:-1], _bce_elements)
        # With the weights from the FULL batch the zero-weighted value is exact ...
        assert abs(weighted * DICE_BATCH / (DICE_BATCH - 1) - full[:-1].mean()) < PCB_ROW_ATOL
        # ... and it is NOT the loss of the batch without that row.
        gap = abs(weighted * DICE_BATCH / (DICE_BATCH - 1) - dropped)
        assert gap > 5e-4, gap
        assert abs(dropped - kept.mean()) < PCB_ROW_ATOL

    def test_the_docstrings_say_batch_coupled_and_one_value_per_sample(self):
        doc = " ".join((PerChannelBinaryLoss.__doc__ or "").split())
        assert "batch-coupled" in doc
        assert "one value per sample" in " ".join(
            (PerChannelBinaryLoss.call.__doc__ or "").split()
        ).lower()

    def test_the_scalar_return_is_caught_by_these_arms(self):
        """ANTI-VACUITY: the defect re-injected must fail the shape and reference arms."""
        y_true, y_pred = _ragged_seg_batch()
        scalar = _ScalarPerChannelBinaryLoss("binary_crossentropy")
        assert tuple(scalar.call(y_true, y_pred).shape) == ()
        scalar.reduction = "none"
        assert tuple(scalar(y_true, y_pred).shape) == ()
        w = np.array([1.0, 1.0, 1.0, 0.0], "float32")
        broken = float(
            _ScalarPerChannelBinaryLoss("binary_crossentropy")(y_true, y_pred, sample_weight=w)
        )
        ref = (w * _pcb_rows_float64(y_true, y_pred, _bce_elements)).sum() / DICE_BATCH
        assert abs(broken - ref) > 1e-3, (broken, ref)
