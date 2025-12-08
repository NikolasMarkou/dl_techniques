"""
Tests for Masked Language Model (MLM) Pre-training Framework
=============================================================

Comprehensive tests for the MaskedLanguageModel class and the apply_mlm_masking
strategy function.
"""

import keras
import numpy as np
import pytest
import tensorflow as tf
from typing import Dict, Any, Optional


# ---------------------------------------------------------------------
# Mock Encoder for Testing
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class MockEncoder(keras.Model):
    """A minimal mock encoder that satisfies the MaskedLanguageModel interface.

    :param hidden_size: The hidden dimension size.
    :type hidden_size: int
    :param vocab_size: The vocabulary size.
    :type vocab_size: int
    """

    def __init__(
            self,
            hidden_size: int = 64,
            vocab_size: int = 1000,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding = keras.layers.Embedding(vocab_size, hidden_size)
        self.dense = keras.layers.Dense(hidden_size)

    def call(
            self,
            inputs: Dict[str, keras.KerasTensor],
            training: Optional[bool] = False,
    ) -> Dict[str, keras.KerasTensor]:
        """Forward pass returning last_hidden_state."""
        input_ids = inputs["input_ids"]
        x = self.embedding(input_ids)
        x = self.dense(x)
        return {"last_hidden_state": x}

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
        })
        return config


# ---------------------------------------------------------------------
# Import modules under test (assumed to be importable)
# ---------------------------------------------------------------------

from dl_techniques.utils.masking.strategies import apply_mlm_masking
from dl_techniques.models.masked_language_model.mlm import MaskedLanguageModel

# ---------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def mock_encoder() -> MockEncoder:
    """Create a mock encoder for testing."""
    return MockEncoder(hidden_size=64, vocab_size=1000)


@pytest.fixture
def mlm_model(mock_encoder: MockEncoder) -> MaskedLanguageModel:
    """Create a basic MaskedLanguageModel for testing."""
    return MaskedLanguageModel(
        encoder=mock_encoder,
        vocab_size=1000,
        mask_ratio=0.15,
        mask_token_id=103,
        random_token_ratio=0.1,
        unchanged_ratio=0.1,
        special_token_ids=[0, 101, 102],  # PAD, CLS, SEP
        mlm_head_activation="gelu",
        initializer_range=0.02,
        mlm_head_dropout=0.1,
        layer_norm_eps=1e-12,
    )


@pytest.fixture
def sample_inputs() -> Dict[str, tf.Tensor]:
    """Create sample input data for testing."""
    batch_size = 4
    seq_len = 32
    vocab_size = 1000

    # Create input_ids with special tokens
    input_ids = tf.random.uniform(
        shape=(batch_size, seq_len),
        minval=3,  # Avoid special tokens
        maxval=vocab_size,
        dtype=tf.int32,
    )
    # Add CLS at start and SEP at end
    cls_tokens = tf.fill((batch_size, 1), 101)
    sep_tokens = tf.fill((batch_size, 1), 102)
    input_ids = tf.concat([cls_tokens, input_ids[:, 1:-1], sep_tokens], axis=1)

    # Create attention mask (all ones, no padding)
    attention_mask = tf.ones((batch_size, seq_len), dtype=tf.int32)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


@pytest.fixture
def sample_inputs_with_padding() -> Dict[str, tf.Tensor]:
    """Create sample input data with padding for testing."""
    batch_size = 4
    seq_len = 32
    vocab_size = 1000
    pad_token_id = 0

    input_ids = tf.random.uniform(
        shape=(batch_size, seq_len),
        minval=3,
        maxval=vocab_size,
        dtype=tf.int32,
    )

    # Add padding at the end (last 8 tokens)
    padding = tf.fill((batch_size, 8), pad_token_id)
    input_ids = tf.concat([input_ids[:, :24], padding], axis=1)

    # Create attention mask
    attention_mask = tf.concat([
        tf.ones((batch_size, 24), dtype=tf.int32),
        tf.zeros((batch_size, 8), dtype=tf.int32),
    ], axis=1)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


# ---------------------------------------------------------------------
# Tests for apply_mlm_masking function
# ---------------------------------------------------------------------


class TestApplyMLMMasking:
    """Tests for the apply_mlm_masking strategy function."""

    def test_output_shapes(self) -> None:
        """Test that output tensors have correct shapes."""
        batch_size, seq_len = 4, 32
        input_ids = tf.random.uniform(
            (batch_size, seq_len), minval=0, maxval=1000, dtype=tf.int32
        )
        attention_mask = tf.ones((batch_size, seq_len), dtype=tf.int32)

        masked_ids, labels, mask = apply_mlm_masking(
            input_ids=input_ids,
            attention_mask=attention_mask,
            vocab_size=1000,
            mask_ratio=0.15,
            mask_token_id=103,
            special_token_ids=[0, 101, 102],
        )

        assert masked_ids.shape == (batch_size, seq_len)
        assert labels.shape == (batch_size, seq_len)
        assert mask.shape == (batch_size, seq_len)

    def test_labels_preserve_original_ids(self) -> None:
        """Test that labels contain the original input IDs."""
        input_ids = tf.constant([[10, 20, 30, 40, 50]], dtype=tf.int32)
        attention_mask = tf.ones_like(input_ids, dtype=tf.int32)

        _, labels, _ = apply_mlm_masking(
            input_ids=input_ids,
            attention_mask=attention_mask,
            vocab_size=1000,
            mask_ratio=0.15,
            mask_token_id=103,
            special_token_ids=[],
        )

        np.testing.assert_array_equal(
            keras.ops.convert_to_numpy(labels),
            keras.ops.convert_to_numpy(input_ids),
            err_msg="Labels should preserve original input IDs"
        )

    def test_special_tokens_not_masked(self) -> None:
        """Test that special tokens are never masked."""
        # Create input with only special tokens
        input_ids = tf.constant([[101, 102, 0, 101, 102]], dtype=tf.int32)
        attention_mask = tf.ones_like(input_ids, dtype=tf.int32)

        masked_ids, _, mask = apply_mlm_masking(
            input_ids=input_ids,
            attention_mask=attention_mask,
            vocab_size=1000,
            mask_ratio=1.0,  # Try to mask everything
            mask_token_id=103,
            special_token_ids=[0, 101, 102],
        )

        # No tokens should be masked
        assert not keras.ops.any(mask)
        np.testing.assert_array_equal(
            keras.ops.convert_to_numpy(masked_ids),
            keras.ops.convert_to_numpy(input_ids),
            err_msg="Special tokens should not be modified"
        )

    def test_padding_not_masked(self) -> None:
        """Test that padding tokens are not masked."""
        input_ids = tf.constant([[10, 20, 30, 0, 0]], dtype=tf.int32)
        attention_mask = tf.constant([[1, 1, 1, 0, 0]], dtype=tf.int32)

        _, _, mask = apply_mlm_masking(
            input_ids=input_ids,
            attention_mask=attention_mask,
            vocab_size=1000,
            mask_ratio=1.0,  # Try to mask everything
            mask_token_id=103,
            special_token_ids=[],
        )

        # Check that padding positions are not masked
        padding_positions = attention_mask == 0
        masked_at_padding = keras.ops.logical_and(mask, padding_positions)
        assert not keras.ops.any(masked_at_padding)

    def test_mask_ratio_approximately_correct(self) -> None:
        """Test that approximately mask_ratio of tokens are masked."""
        batch_size, seq_len = 100, 100
        input_ids = tf.random.uniform(
            (batch_size, seq_len), minval=10, maxval=1000, dtype=tf.int32
        )
        attention_mask = tf.ones((batch_size, seq_len), dtype=tf.int32)
        mask_ratio = 0.15

        _, _, mask = apply_mlm_masking(
            input_ids=input_ids,
            attention_mask=attention_mask,
            vocab_size=1000,
            mask_ratio=mask_ratio,
            mask_token_id=103,
            special_token_ids=[],
        )

        actual_ratio = (
            keras.ops.sum(keras.ops.cast(mask, "float32")) /
            (batch_size * seq_len)
        )
        actual_ratio = float(keras.ops.convert_to_numpy(actual_ratio))

        # Allow 5% tolerance due to randomness
        assert abs(actual_ratio - mask_ratio) < 0.05

    def test_mask_token_replacement(self) -> None:
        """Test that approximately 80% of masked tokens become [MASK]."""
        batch_size, seq_len = 100, 100
        mask_token_id = 103
        input_ids = tf.random.uniform(
            (batch_size, seq_len), minval=200, maxval=1000, dtype=tf.int32
        )
        attention_mask = tf.ones((batch_size, seq_len), dtype=tf.int32)

        masked_ids, _, mask = apply_mlm_masking(
            input_ids=input_ids,
            attention_mask=attention_mask,
            vocab_size=1000,
            mask_ratio=0.5,  # Higher ratio for statistical stability
            mask_token_id=mask_token_id,
            special_token_ids=[],
            random_token_ratio=0.1,
            unchanged_ratio=0.1,
        )

        total_masked = keras.ops.sum(keras.ops.cast(mask, "float32"))
        mask_token_count = keras.ops.sum(keras.ops.cast(
            masked_ids == mask_token_id, "float32"
        ))

        mask_token_ratio = float(keras.ops.convert_to_numpy(
            mask_token_count / total_masked
        ))

        # Should be approximately 80% (allow 10% tolerance)
        assert 0.70 < mask_token_ratio < 0.90

    def test_without_attention_mask(self) -> None:
        """Test that masking works without attention mask."""
        input_ids = tf.constant([[10, 20, 30, 40, 50]], dtype=tf.int32)

        masked_ids, labels, mask = apply_mlm_masking(
            input_ids=input_ids,
            attention_mask=None,
            vocab_size=1000,
            mask_ratio=0.5,
            mask_token_id=103,
            special_token_ids=[],
        )

        assert masked_ids.shape == input_ids.shape
        assert labels.shape == input_ids.shape
        assert mask.shape == input_ids.shape

    def test_mask_dtype(self) -> None:
        """Test that mask has boolean dtype."""
        input_ids = tf.constant([[10, 20, 30]], dtype=tf.int32)
        attention_mask = tf.ones_like(input_ids)

        _, _, mask = apply_mlm_masking(
            input_ids=input_ids,
            attention_mask=attention_mask,
            vocab_size=1000,
            mask_ratio=0.15,
            mask_token_id=103,
            special_token_ids=[],
        )

        assert mask.dtype == tf.bool


# ---------------------------------------------------------------------
# Tests for MaskedLanguageModel initialization
# ---------------------------------------------------------------------


class TestMaskedLanguageModelInit:
    """Tests for MaskedLanguageModel initialization."""

    def test_basic_initialization(self, mock_encoder: MockEncoder) -> None:
        """Test basic model initialization."""
        model = MaskedLanguageModel(
            encoder=mock_encoder,
            vocab_size=1000,
            mask_ratio=0.15,
            mask_token_id=103,
        )

        assert model.vocab_size == 1000
        assert model.mask_ratio == 0.15
        assert model.mask_token_id == 103
        assert model.hidden_size == mock_encoder.hidden_size

    def test_default_values(self, mock_encoder: MockEncoder) -> None:
        """Test that default values are set correctly."""
        model = MaskedLanguageModel(
            encoder=mock_encoder,
            vocab_size=1000,
        )

        assert model.mask_ratio == 0.15
        assert model.mask_token_id == 103
        assert model.random_token_ratio == 0.1
        assert model.unchanged_ratio == 0.1
        assert model.special_token_ids == []
        assert model.mlm_head_activation == "gelu"
        assert model.initializer_range == 0.02
        assert model.mlm_head_dropout == 0.1
        assert model.layer_norm_eps == 1e-12

    def test_invalid_vocab_size(self, mock_encoder: MockEncoder) -> None:
        """Test that invalid vocab_size raises ValueError."""
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            MaskedLanguageModel(
                encoder=mock_encoder,
                vocab_size=0,
            )

        with pytest.raises(ValueError, match="vocab_size must be positive"):
            MaskedLanguageModel(
                encoder=mock_encoder,
                vocab_size=-100,
            )

    def test_invalid_mask_ratio(self, mock_encoder: MockEncoder) -> None:
        """Test that invalid mask_ratio raises ValueError."""
        with pytest.raises(ValueError, match="mask_ratio must be between 0 and 1"):
            MaskedLanguageModel(
                encoder=mock_encoder,
                vocab_size=1000,
                mask_ratio=0.0,
            )

        with pytest.raises(ValueError, match="mask_ratio must be between 0 and 1"):
            MaskedLanguageModel(
                encoder=mock_encoder,
                vocab_size=1000,
                mask_ratio=1.5,
            )

    def test_invalid_mask_token_id(self, mock_encoder: MockEncoder) -> None:
        """Test that invalid mask_token_id raises ValueError."""
        with pytest.raises(ValueError, match="mask_token_id must be in"):
            MaskedLanguageModel(
                encoder=mock_encoder,
                vocab_size=1000,
                mask_token_id=-1,
            )

        with pytest.raises(ValueError, match="mask_token_id must be in"):
            MaskedLanguageModel(
                encoder=mock_encoder,
                vocab_size=1000,
                mask_token_id=1000,  # Out of range
            )

    def test_invalid_random_token_ratio(self, mock_encoder: MockEncoder) -> None:
        """Test that invalid random_token_ratio raises ValueError."""
        with pytest.raises(
                ValueError, match="random_token_ratio must be between 0 and 1"
        ):
            MaskedLanguageModel(
                encoder=mock_encoder,
                vocab_size=1000,
                random_token_ratio=1.5,
            )

    def test_invalid_unchanged_ratio(self, mock_encoder: MockEncoder) -> None:
        """Test that invalid unchanged_ratio raises ValueError."""
        with pytest.raises(
                ValueError, match="unchanged_ratio must be between 0 and 1"
        ):
            MaskedLanguageModel(
                encoder=mock_encoder,
                vocab_size=1000,
                unchanged_ratio=-0.1,
            )

    def test_ratios_sum_exceeds_one(self, mock_encoder: MockEncoder) -> None:
        """Test that ratios summing > 1 raises ValueError."""
        with pytest.raises(
                ValueError, match="random_token_ratio \\+ unchanged_ratio cannot exceed"
        ):
            MaskedLanguageModel(
                encoder=mock_encoder,
                vocab_size=1000,
                random_token_ratio=0.6,
                unchanged_ratio=0.6,
            )

    def test_invalid_initializer_range(self, mock_encoder: MockEncoder) -> None:
        """Test that invalid initializer_range raises ValueError."""
        with pytest.raises(ValueError, match="initializer_range must be positive"):
            MaskedLanguageModel(
                encoder=mock_encoder,
                vocab_size=1000,
                initializer_range=0.0,
            )

    def test_invalid_mlm_head_dropout(self, mock_encoder: MockEncoder) -> None:
        """Test that invalid mlm_head_dropout raises ValueError."""
        with pytest.raises(
                ValueError, match="mlm_head_dropout must be between 0 and 1"
        ):
            MaskedLanguageModel(
                encoder=mock_encoder,
                vocab_size=1000,
                mlm_head_dropout=1.0,
            )

    def test_encoder_missing_hidden_size(self) -> None:
        """Test that encoder without hidden_size raises ValueError."""
        # Create an encoder without hidden_size
        encoder = keras.Sequential([keras.layers.Dense(64)])

        with pytest.raises(
                ValueError, match="encoder must have a 'hidden_size' attribute"
        ):
            MaskedLanguageModel(
                encoder=encoder,
                vocab_size=1000,
            )

    def test_mlm_head_layers_created(
            self, mlm_model: MaskedLanguageModel
    ) -> None:
        """Test that MLM head layers are created."""
        assert hasattr(mlm_model, "mlm_dense")
        assert hasattr(mlm_model, "mlm_dropout")
        assert hasattr(mlm_model, "mlm_norm")
        assert hasattr(mlm_model, "mlm_output")

    def test_metrics_created(self, mlm_model: MaskedLanguageModel) -> None:
        """Test that metrics are created."""
        assert hasattr(mlm_model, "loss_tracker")
        assert hasattr(mlm_model, "acc_metric")
        assert len(mlm_model.metrics) == 2


# ---------------------------------------------------------------------
# Tests for MaskedLanguageModel forward pass
# ---------------------------------------------------------------------


class TestMaskedLanguageModelCall:
    """Tests for MaskedLanguageModel call method."""

    def test_output_shape(
            self,
            mlm_model: MaskedLanguageModel,
            sample_inputs: Dict[str, tf.Tensor],
    ) -> None:
        """Test that output has correct shape."""
        output = mlm_model(sample_inputs, training=False)

        batch_size = sample_inputs["input_ids"].shape[0]
        seq_len = sample_inputs["input_ids"].shape[1]

        assert output.shape == (batch_size, seq_len, mlm_model.vocab_size)

    def test_output_dtype(
            self,
            mlm_model: MaskedLanguageModel,
            sample_inputs: Dict[str, tf.Tensor],
    ) -> None:
        """Test that output has float32 dtype."""
        output = mlm_model(sample_inputs, training=False)
        assert output.dtype == tf.float32

    def test_training_vs_inference_mode(
            self,
            mlm_model: MaskedLanguageModel,
            sample_inputs: Dict[str, tf.Tensor],
    ) -> None:
        """Test that training and inference modes produce valid outputs."""
        output_train = mlm_model(sample_inputs, training=True)
        output_infer = mlm_model(sample_inputs, training=False)

        # Both should have same shape
        assert output_train.shape == output_infer.shape

        # Outputs may differ due to dropout
        # Just check they are valid (no NaN/Inf)
        assert not keras.ops.any(keras.ops.isnan(output_train))
        assert not keras.ops.any(keras.ops.isnan(output_infer))


# ---------------------------------------------------------------------
# Tests for MaskedLanguageModel _mask_tokens
# ---------------------------------------------------------------------


class TestMaskedLanguageModelMaskTokens:
    """Tests for MaskedLanguageModel _mask_tokens method."""

    def test_mask_tokens_output_structure(
            self,
            mlm_model: MaskedLanguageModel,
            sample_inputs: Dict[str, tf.Tensor],
    ) -> None:
        """Test that _mask_tokens returns correct structure."""
        masked_inputs, labels, mask = mlm_model._mask_tokens(sample_inputs)

        assert "input_ids" in masked_inputs
        assert "attention_mask" in masked_inputs
        assert labels.shape == sample_inputs["input_ids"].shape
        assert mask.shape == sample_inputs["input_ids"].shape

    def test_attention_mask_preserved(
            self,
            mlm_model: MaskedLanguageModel,
            sample_inputs: Dict[str, tf.Tensor],
    ) -> None:
        """Test that attention mask is preserved after masking."""
        masked_inputs, _, _ = mlm_model._mask_tokens(sample_inputs)

        np.testing.assert_array_equal(
            keras.ops.convert_to_numpy(masked_inputs["attention_mask"]),
            keras.ops.convert_to_numpy(sample_inputs["attention_mask"]),
            err_msg="Attention mask should be preserved"
        )


# ---------------------------------------------------------------------
# Tests for MaskedLanguageModel compute_loss
# ---------------------------------------------------------------------


class TestMaskedLanguageModelComputeLoss:
    """Tests for MaskedLanguageModel compute_loss method."""

    def test_loss_without_sample_weight(
            self, mlm_model: MaskedLanguageModel
    ) -> None:
        """Test loss computation without sample weights."""
        batch_size, seq_len = 2, 10
        y = tf.random.uniform(
            (batch_size, seq_len), minval=0, maxval=1000, dtype=tf.int32
        )
        y_pred = tf.random.uniform(
            (batch_size, seq_len, 1000), dtype=tf.float32
        )

        loss = mlm_model.compute_loss(y=y, y_pred=y_pred, sample_weight=None)

        assert loss.shape == ()
        assert loss.dtype == tf.float32
        assert float(keras.ops.convert_to_numpy(loss)) > 0

    def test_loss_with_sample_weight(
            self, mlm_model: MaskedLanguageModel
    ) -> None:
        """Test loss computation with sample weights."""
        batch_size, seq_len = 2, 10
        y = tf.random.uniform(
            (batch_size, seq_len), minval=0, maxval=1000, dtype=tf.int32
        )
        y_pred = tf.random.uniform(
            (batch_size, seq_len, 1000), dtype=tf.float32
        )
        # Only compute loss on first 5 tokens
        sample_weight = tf.concat([
            tf.ones((batch_size, 5), dtype=tf.bool),
            tf.zeros((batch_size, 5), dtype=tf.bool),
        ], axis=1)

        loss = mlm_model.compute_loss(y=y, y_pred=y_pred, sample_weight=sample_weight)

        assert loss.shape == ()
        assert float(keras.ops.convert_to_numpy(loss)) > 0

    def test_loss_with_zero_sample_weight(
            self, mlm_model: MaskedLanguageModel
    ) -> None:
        """Test loss computation when all sample weights are zero."""
        batch_size, seq_len = 2, 10
        y = tf.random.uniform(
            (batch_size, seq_len), minval=0, maxval=1000, dtype=tf.int32
        )
        y_pred = tf.random.uniform(
            (batch_size, seq_len, 1000), dtype=tf.float32
        )
        # All zeros
        sample_weight = tf.zeros((batch_size, seq_len), dtype=tf.bool)

        loss = mlm_model.compute_loss(y=y, y_pred=y_pred, sample_weight=sample_weight)

        # Should not produce NaN or Inf due to division by zero protection
        assert not keras.ops.isnan(loss)
        assert not keras.ops.isinf(loss)


# ---------------------------------------------------------------------
# Tests for MaskedLanguageModel train_step and test_step
# ---------------------------------------------------------------------


class TestMaskedLanguageModelTraining:
    """Tests for MaskedLanguageModel training functionality."""

    def test_train_step_dict_input(
            self,
            mlm_model: MaskedLanguageModel,
            sample_inputs: Dict[str, tf.Tensor],
    ) -> None:
        """Test train_step with dictionary input."""
        mlm_model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4))

        metrics = mlm_model.train_step(sample_inputs)

        assert "loss" in metrics
        assert "accuracy" in metrics
        assert float(keras.ops.convert_to_numpy(metrics["loss"])) > 0
        assert 0.0 <= float(keras.ops.convert_to_numpy(metrics["accuracy"])) <= 1.0

    def test_train_step_tuple_input(
            self,
            mlm_model: MaskedLanguageModel,
            sample_inputs: Dict[str, tf.Tensor],
    ) -> None:
        """Test train_step with tuple input (x, y, sample_weight)."""
        mlm_model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4))

        # Pack as tuple
        data = (sample_inputs, None, None)
        metrics = mlm_model.train_step(data)

        assert "loss" in metrics
        assert "accuracy" in metrics

    def test_test_step_dict_input(
            self,
            mlm_model: MaskedLanguageModel,
            sample_inputs: Dict[str, tf.Tensor],
    ) -> None:
        """Test test_step with dictionary input."""
        mlm_model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4))

        # Build model first
        _ = mlm_model(sample_inputs, training=False)

        # Reset metrics
        mlm_model.loss_tracker.reset_state()
        mlm_model.acc_metric.reset_state()

        metrics = mlm_model.test_step(sample_inputs)

        assert "loss" in metrics
        assert "accuracy" in metrics

    def test_test_step_tuple_input(
            self,
            mlm_model: MaskedLanguageModel,
            sample_inputs: Dict[str, tf.Tensor],
    ) -> None:
        """Test test_step with tuple input."""
        mlm_model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4))

        # Build model
        _ = mlm_model(sample_inputs, training=False)

        # Reset metrics
        mlm_model.loss_tracker.reset_state()
        mlm_model.acc_metric.reset_state()

        data = (sample_inputs, None, None)
        metrics = mlm_model.test_step(data)

        assert "loss" in metrics
        assert "accuracy" in metrics

    def test_weights_updated_after_train_step(
            self,
            mlm_model: MaskedLanguageModel,
            sample_inputs: Dict[str, tf.Tensor],
    ) -> None:
        """Test that model weights are updated after train_step."""
        mlm_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.1))

        # Build model and get initial weights
        _ = mlm_model(sample_inputs, training=False)
        initial_weights = [w.numpy().copy() for w in mlm_model.trainable_weights]

        # Run train step
        mlm_model.train_step(sample_inputs)

        # Check weights changed
        final_weights = [w.numpy() for w in mlm_model.trainable_weights]

        at_least_one_changed = False
        for init_w, final_w in zip(initial_weights, final_weights):
            if not np.allclose(init_w, final_w):
                at_least_one_changed = True
                break

        assert at_least_one_changed, "At least one weight should change after training"


# ---------------------------------------------------------------------
# Tests for MaskedLanguageModel serialization
# ---------------------------------------------------------------------


class TestMaskedLanguageModelSerialization:
    """Tests for MaskedLanguageModel serialization."""

    def test_get_config(self, mlm_model: MaskedLanguageModel) -> None:
        """Test that get_config returns complete configuration."""
        config = mlm_model.get_config()

        assert "encoder" in config
        assert config["vocab_size"] == 1000
        assert config["mask_ratio"] == 0.15
        assert config["mask_token_id"] == 103
        assert config["random_token_ratio"] == 0.1
        assert config["unchanged_ratio"] == 0.1
        assert config["special_token_ids"] == [0, 101, 102]
        assert config["mlm_head_activation"] == "gelu"
        assert config["initializer_range"] == 0.02
        assert config["mlm_head_dropout"] == 0.1
        assert config["layer_norm_eps"] == 1e-12

    def test_from_config(
            self,
            mlm_model: MaskedLanguageModel,
            sample_inputs: Dict[str, tf.Tensor],
    ) -> None:
        """Test reconstruction from config."""
        # Build original model
        _ = mlm_model(sample_inputs, training=False)

        config = mlm_model.get_config()
        reconstructed = MaskedLanguageModel.from_config(config)

        assert reconstructed.vocab_size == mlm_model.vocab_size
        assert reconstructed.mask_ratio == mlm_model.mask_ratio
        assert reconstructed.mask_token_id == mlm_model.mask_token_id
        assert reconstructed.hidden_size == mlm_model.hidden_size

    def test_save_and_load(
            self,
            mlm_model: MaskedLanguageModel,
            sample_inputs: Dict[str, tf.Tensor],
            tmp_path,
    ) -> None:
        """Test model save and load."""
        # Build model
        output_before = mlm_model(sample_inputs, training=False)

        # Save
        save_path = tmp_path / "mlm_model.keras"
        mlm_model.save(save_path)

        # Load
        loaded_model = keras.models.load_model(save_path)

        # Compare outputs
        output_after = loaded_model(sample_inputs, training=False)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output_before),
            keras.ops.convert_to_numpy(output_after),
            rtol=1e-5, atol=1e-5,
            err_msg="Loaded model should produce same outputs"
        )

    def test_config_roundtrip(self, mlm_model: MaskedLanguageModel) -> None:
        """Test config roundtrip preserves all attributes."""
        config = mlm_model.get_config()
        reconstructed = MaskedLanguageModel.from_config(config.copy())
        new_config = reconstructed.get_config()

        # Compare all non-encoder keys
        for key in config:
            if key != "encoder":
                assert config[key] == new_config[key], f"Mismatch for key: {key}"


# ---------------------------------------------------------------------
# Tests for MaskedLanguageModel with padding
# ---------------------------------------------------------------------


class TestMaskedLanguageModelWithPadding:
    """Tests for MaskedLanguageModel behavior with padded sequences."""

    def test_padding_tokens_not_masked(
            self,
            mlm_model: MaskedLanguageModel,
            sample_inputs_with_padding: Dict[str, tf.Tensor],
    ) -> None:
        """Test that padding tokens are not masked during training."""
        masked_inputs, _, mask = mlm_model._mask_tokens(sample_inputs_with_padding)

        attention_mask = sample_inputs_with_padding["attention_mask"]
        padding_positions = attention_mask == 0

        # Check no padding positions are masked
        masked_at_padding = keras.ops.logical_and(mask, padding_positions)
        assert not keras.ops.any(masked_at_padding)

    def test_forward_pass_with_padding(
            self,
            mlm_model: MaskedLanguageModel,
            sample_inputs_with_padding: Dict[str, tf.Tensor],
    ) -> None:
        """Test forward pass works with padded inputs."""
        output = mlm_model(sample_inputs_with_padding, training=False)

        assert output.shape[0] == sample_inputs_with_padding["input_ids"].shape[0]
        assert output.shape[1] == sample_inputs_with_padding["input_ids"].shape[1]
        assert output.shape[2] == mlm_model.vocab_size


# ---------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------


class TestMaskedLanguageModelIntegration:
    """Integration tests for MaskedLanguageModel."""

    def test_fit_single_batch(
            self,
            mlm_model: MaskedLanguageModel,
            sample_inputs: Dict[str, tf.Tensor],
    ) -> None:
        """Test fitting model on a single batch."""
        mlm_model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4))

        # Create dataset
        dataset = tf.data.Dataset.from_tensors(sample_inputs).repeat(2)

        # Fit for 1 epoch
        history = mlm_model.fit(dataset, epochs=1, verbose=0)

        assert "loss" in history.history
        assert "accuracy" in history.history
        assert len(history.history["loss"]) == 1

    def test_evaluate(
            self,
            mlm_model: MaskedLanguageModel,
            sample_inputs: Dict[str, tf.Tensor],
    ) -> None:
        """Test model evaluation."""
        mlm_model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4))

        # Build model
        _ = mlm_model(sample_inputs, training=False)

        dataset = tf.data.Dataset.from_tensors(sample_inputs)

        results = mlm_model.evaluate(dataset, verbose=0, return_dict=True)

        assert "loss" in results
        assert "accuracy" in results

    def test_encoder_extraction(self, mlm_model: MaskedLanguageModel) -> None:
        """Test that encoder can be extracted from trained model."""
        encoder = mlm_model.encoder

        assert encoder is not None
        assert hasattr(encoder, "hidden_size")
        assert encoder.hidden_size == mlm_model.hidden_size

    def test_multiple_train_steps_decrease_loss(
            self,
            mock_encoder: MockEncoder,
            sample_inputs: Dict[str, tf.Tensor],
    ) -> None:
        """Test that multiple training steps decrease loss on same data."""
        # Use higher mask ratio for more signal
        model = MaskedLanguageModel(
            encoder=mock_encoder,
            vocab_size=1000,
            mask_ratio=0.3,
            mask_token_id=103,
            special_token_ids=[],
        )
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3))

        # Build
        _ = model(sample_inputs, training=False)

        # Train multiple steps
        losses = []
        for _ in range(10):
            metrics = model.train_step(sample_inputs)
            losses.append(float(keras.ops.convert_to_numpy(metrics["loss"])))

        # Loss should generally decrease (may not be monotonic due to stochastic masking)
        assert losses[-1] < losses[0], "Loss should decrease over training"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])