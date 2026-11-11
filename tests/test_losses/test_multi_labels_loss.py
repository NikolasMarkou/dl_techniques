"""
Tests for Per-Channel Loss Wrapper and Multi-Label Segmentation Losses.
"""

import pytest
import numpy as np
import keras
from keras import ops

from dl_techniques.losses.multi_labels_loss import (
    PerChannelBinaryLoss,
    WeightedBinaryCrossEntropyLoss,
    WeightedBinaryFocalLoss,
    create_multilabel_segmentation_loss,
)
from tests.optimizer_state import build_optimizer_state


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

        assert ops.shape(loss_val) == ()
        assert not np.isnan(float(loss_val))

    def test_dice_loss_all_zeros_prediction(self):
        """Test dice loss with all zeros prediction."""
        loss_fn = create_multilabel_segmentation_loss(loss_type='dice', smooth=1.0)
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
        plain = float(PerChannelBinaryLoss(base).call(y_true, y_pred))
        weighted = float(
            PerChannelBinaryLoss(base, channel_weights=[10.0, 1.0, 0.1]).call(y_true, y_pred)
        )
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
            float(restored.call(y_true, y_pred)), float(loss_fn.call(y_true, y_pred)), rtol=1e-6
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
