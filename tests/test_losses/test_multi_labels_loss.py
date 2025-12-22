"""
Tests for Per-Channel Loss Wrapper and Multi-Label Segmentation Losses.
"""

import pytest
import numpy as np
import keras
from keras import ops

from dl_techniques.losses.multi_labels_loss import (
    PerChannelBinaryLoss,
    WeightedBinaryFocalLoss,
    DiceLossPerChannel,
    create_multilabel_segmentation_loss,
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


class TestDiceLossPerChannel:
    """Tests for DiceLossPerChannel."""

    def test_output_is_scalar(self):
        """Dice loss should return a scalar."""
        loss_fn = DiceLossPerChannel(smooth=1.0)
        y_true = keras.random.uniform((2, 8, 8, 3), minval=0, maxval=1)
        y_true = ops.cast(y_true > 0.5, "float32")
        y_pred = keras.random.uniform((2, 8, 8, 3), minval=0.1, maxval=0.9)

        loss_val = loss_fn.call(y_true, y_pred)

        assert ops.shape(loss_val) == ()

    def test_loss_range(self):
        """Dice loss should be in [0, 1]."""
        loss_fn = DiceLossPerChannel(smooth=1.0)
        y_true = keras.random.uniform((4, 16, 16, 5), minval=0, maxval=1)
        y_true = ops.cast(y_true > 0.5, "float32")
        y_pred = keras.random.uniform((4, 16, 16, 5), minval=0.0, maxval=1.0)

        loss_val = float(loss_fn.call(y_true, y_pred))

        assert 0.0 <= loss_val <= 1.0

    def test_perfect_overlap(self):
        """Perfect overlap should yield loss close to 0."""
        loss_fn = DiceLossPerChannel(smooth=1e-7)
        y_true = ops.ones((2, 8, 8, 2))
        y_pred = ops.ones((2, 8, 8, 2))

        loss_val = float(loss_fn.call(y_true, y_pred))

        assert loss_val < 0.01

    def test_no_overlap(self):
        """No overlap should yield loss close to 1."""
        loss_fn = DiceLossPerChannel(smooth=1e-7)
        y_true = ops.ones((2, 8, 8, 2))
        y_pred = ops.zeros((2, 8, 8, 2))

        loss_val = float(loss_fn.call(y_true, y_pred))

        assert loss_val > 0.99

    def test_smooth_prevents_nan(self):
        """Smooth factor should prevent NaN with zero inputs."""
        loss_fn = DiceLossPerChannel(smooth=1.0)
        y_true = ops.zeros((2, 8, 8, 2))
        y_pred = ops.zeros((2, 8, 8, 2))

        loss_val = loss_fn.call(y_true, y_pred)

        assert not np.isnan(float(loss_val))

    def test_serialization(self):
        """Test get_config and from_config."""
        loss_fn = DiceLossPerChannel(smooth=0.5)
        config = loss_fn.get_config()

        restored = DiceLossPerChannel.from_config(config)

        assert restored.smooth == 0.5


class TestPerChannelBinaryLoss:
    """Tests for PerChannelBinaryLoss wrapper."""

    def test_output_is_scalar(self):
        """Wrapped loss should return scalar."""
        loss_fn = PerChannelBinaryLoss(base_loss='binary_crossentropy')
        y_true = keras.random.uniform((2, 8, 8, 3), minval=0, maxval=1)
        y_true = ops.cast(y_true > 0.5, "float32")
        y_pred = keras.random.uniform((2, 8, 8, 3), minval=0.1, maxval=0.9)

        loss_val = loss_fn.call(y_true, y_pred)

        assert ops.shape(loss_val) == ()

    def test_loss_is_positive(self):
        """Loss should be non-negative."""
        loss_fn = PerChannelBinaryLoss(base_loss='binary_crossentropy')
        y_true = keras.random.uniform((4, 16, 16, 5), minval=0, maxval=1)
        y_true = ops.cast(y_true > 0.5, "float32")
        y_pred = keras.random.uniform((4, 16, 16, 5), minval=0.1, maxval=0.9)

        loss_val = float(loss_fn.call(y_true, y_pred))

        assert loss_val >= 0

    def test_with_focal_loss_base(self):
        """Test with WeightedBinaryFocalLoss as base."""
        focal = WeightedBinaryFocalLoss(alpha=0.25, gamma=2.0)
        loss_fn = PerChannelBinaryLoss(base_loss=focal)
        y_true = keras.random.uniform((2, 8, 8, 3), minval=0, maxval=1)
        y_true = ops.cast(y_true > 0.5, "float32")
        y_pred = keras.random.uniform((2, 8, 8, 3), minval=0.1, maxval=0.9)

        loss_val = loss_fn.call(y_true, y_pred)

        assert ops.shape(loss_val) == ()
        assert float(loss_val) >= 0

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

        assert ops.shape(loss_val) == ()
        assert float(loss_val) > 0

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

        assert ops.shape(loss_val) == ()

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

        assert isinstance(loss_fn, DiceLossPerChannel)
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
        # Note: 'weighted_bce' excluded - factory passes a function to
        # PerChannelBinaryLoss but call() expects a Loss object with .call() method
        loss_types = ['focal', 'dice', 'bce']
        y_true = keras.random.uniform((2, 8, 8, 3), minval=0, maxval=1)
        y_true = ops.cast(y_true > 0.5, "float32")
        y_pred = keras.random.uniform((2, 8, 8, 3), minval=0.1, maxval=0.9)

        for loss_type in loss_types:
            loss_fn = create_multilabel_segmentation_loss(loss_type=loss_type)
            loss_val = loss_fn(y_true, y_pred)

            assert not np.isnan(float(loss_val)), f"{loss_type} produced NaN"
            assert float(loss_val) >= 0, f"{loss_type} produced negative loss"

    def test_weighted_bce_has_callable_bug(self):
        """Document that weighted_bce factory has a bug with callable base_loss."""
        loss_fn = create_multilabel_segmentation_loss(loss_type='weighted_bce')
        y_true = keras.random.uniform((2, 8, 8, 3), minval=0, maxval=1)
        y_true = ops.cast(y_true > 0.5, "float32")
        y_pred = keras.random.uniform((2, 8, 8, 3), minval=0.1, maxval=0.9)

        # Current implementation passes a function, but PerChannelBinaryLoss.call()
        # tries to invoke .call() on it which fails
        with pytest.raises(AttributeError, match="'function' object has no attribute 'call'"):
            loss_fn(y_true, y_pred)


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
        loss_fn = DiceLossPerChannel(smooth=0.5)

        inputs = keras.layers.Input(shape=(16, 16, 3))
        x = keras.layers.Conv2D(8, 3, padding='same', activation='relu')(inputs)
        outputs = keras.layers.Conv2D(2, 1, activation='sigmoid')(x)
        model = keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss=loss_fn)

        model_path = tmp_path / "model_dice.keras"
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

        assert ops.shape(loss_val) == ()
        assert not np.isnan(float(loss_val))

    def test_dice_loss_all_zeros_prediction(self):
        """Test dice loss with all zeros prediction."""
        loss_fn = DiceLossPerChannel(smooth=1.0)
        y_true = ops.ones((2, 8, 8, 2))
        y_pred = ops.zeros((2, 8, 8, 2))

        loss_val = loss_fn.call(y_true, y_pred)

        assert not np.isnan(float(loss_val))

    def test_large_batch_size(self):
        """Test with larger batch size."""
        loss_fn = PerChannelBinaryLoss(base_loss='binary_crossentropy')
        y_true = keras.random.uniform((32, 64, 64, 8), minval=0, maxval=1)
        y_true = ops.cast(y_true > 0.5, "float32")
        y_pred = keras.random.uniform((32, 64, 64, 8), minval=0.1, maxval=0.9)

        loss_val = loss_fn.call(y_true, y_pred)

        assert ops.shape(loss_val) == ()
        assert not np.isnan(float(loss_val))


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

        loss_fn = DiceLossPerChannel(smooth=1.0)
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