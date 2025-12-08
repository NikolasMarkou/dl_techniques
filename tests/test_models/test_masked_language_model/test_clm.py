"""
Tests for Causal Language Model (CLM) Pre-training Framework
=============================================================

Comprehensive tests for the CausalLanguageModel class.
"""

import keras
from keras import ops
import numpy as np
import pytest
import tensorflow as tf
from typing import Dict, Any, Optional

from dl_techniques.models.masked_language_model.clm import CausalLanguageModel


# ---------------------------------------------------------------------
# Mock Backbones (Corrected for Keras 3 Building)
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class MockCausalBackbone(keras.Model):
    """A mock backbone that correctly implements build() for Keras 3."""
    def __init__(self, hidden_size=64, vocab_size=1000, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.token_embeddings = keras.layers.Embedding(vocab_size, hidden_size)
        self.dense = keras.layers.Dense(hidden_size)

    def build(self, input_shape):
        if isinstance(input_shape, dict):
            shape = input_shape["input_ids"]
        else:
            shape = input_shape
        # Ensure we build with the last dimension (hidden size) where appropriate
        self.token_embeddings.build(shape)
        self.dense.build((None, shape[-1], self.hidden_size))
        super().build(input_shape)

    def call(self, inputs, training=False):
        x = self.token_embeddings(inputs["input_ids"])
        x = self.dense(x)
        return {"last_hidden_state": x}

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_size": self.hidden_size, "vocab_size": self.vocab_size})
        return config


@keras.saving.register_keras_serializable()
class MockBackboneNoEmbeddings(keras.Model):
    """A mock backbone with hidden embeddings."""
    def __init__(self, hidden_size=64, vocab_size=1000, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self._embedding = keras.layers.Embedding(vocab_size, hidden_size)
        self.dense = keras.layers.Dense(hidden_size)

    def build(self, input_shape):
        if isinstance(input_shape, dict):
            shape = input_shape["input_ids"]
        else:
            shape = input_shape
        self._embedding.build(shape)
        self.dense.build((None, shape[-1], self.hidden_size))
        super().build(input_shape)

    def call(self, inputs, training=False):
        x = self._embedding(inputs["input_ids"])
        x = self.dense(x)
        return {"last_hidden_state": x}

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_size": self.hidden_size, "vocab_size": self.vocab_size})
        return config


@keras.saving.register_keras_serializable()
class MockBackboneWithGetEmbedding(keras.Model):
    """A mock backbone with explicit getter."""
    def __init__(self, hidden_size=64, vocab_size=1000, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self._embedding = keras.layers.Embedding(vocab_size, hidden_size)
        self.dense = keras.layers.Dense(hidden_size)

    def build(self, input_shape):
        if isinstance(input_shape, dict):
            shape = input_shape["input_ids"]
        else:
            shape = input_shape
        self._embedding.build(shape)
        self.dense.build((None, shape[-1], self.hidden_size))
        super().build(input_shape)

    def call(self, inputs, training=False):
        x = self._embedding(inputs["input_ids"])
        x = self.dense(x)
        return {"last_hidden_state": x}

    def get_embedding_matrix(self):
        # Access variables only after build/call
        if self._embedding.built:
            return self._embedding.embeddings
        return None

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_size": self.hidden_size, "vocab_size": self.vocab_size})
        return config


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def mock_backbone():
    return MockCausalBackbone(hidden_size=64, vocab_size=1000)

@pytest.fixture
def mock_backbone_no_embeddings():
    return MockBackboneNoEmbeddings(hidden_size=64, vocab_size=1000)

@pytest.fixture
def mock_backbone_with_get_embedding():
    return MockBackboneWithGetEmbedding(hidden_size=64, vocab_size=1000)

@pytest.fixture
def clm_model(mock_backbone):
    return CausalLanguageModel(backbone=mock_backbone, vocab_size=1000, tie_weights=True)

@pytest.fixture
def clm_model_no_tying(mock_backbone_no_embeddings):
    return CausalLanguageModel(backbone=mock_backbone_no_embeddings, vocab_size=1000, tie_weights=False)

@pytest.fixture
def sample_inputs():
    input_ids = tf.random.uniform((4, 32), minval=1, maxval=1000, dtype=tf.int32)
    return {"input_ids": input_ids, "attention_mask": tf.ones((4, 32), dtype=tf.int32)}

@pytest.fixture
def sample_inputs_with_padding():
    input_ids = tf.random.uniform((4, 24), minval=1, maxval=1000, dtype=tf.int32)
    padding = tf.zeros((4, 8), dtype=tf.int32)
    input_ids = tf.concat([input_ids, padding], axis=1)
    mask = tf.concat([tf.ones((4, 24), dtype=tf.int32), tf.zeros((4, 8), dtype=tf.int32)], axis=1)
    return {"input_ids": input_ids, "attention_mask": mask}

# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

class TestCausalLanguageModelInit:
    def test_basic_initialization(self, mock_backbone):
        model = CausalLanguageModel(backbone=mock_backbone, vocab_size=1000, tie_weights=True)
        assert model.vocab_size == 1000
        assert model.tie_weights is True

    def test_default_values(self, mock_backbone):
        model = CausalLanguageModel(backbone=mock_backbone, vocab_size=1000)
        assert model.initializer_range == 0.02
        assert model.tie_weights is True

    def test_invalid_vocab_size(self, mock_backbone):
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            CausalLanguageModel(backbone=mock_backbone, vocab_size=0)

    def test_invalid_initializer_range(self, mock_backbone):
        with pytest.raises(ValueError):
            CausalLanguageModel(backbone=mock_backbone, vocab_size=1000, initializer_range=-0.1)

    def test_backbone_missing_hidden_size(self):
        backbone = keras.Sequential([keras.layers.Dense(64)])
        with pytest.raises(ValueError, match="hidden_size"):
            CausalLanguageModel(backbone=backbone, vocab_size=1000)

    def test_weight_tying_enabled(self, mock_backbone, sample_inputs):
        model = CausalLanguageModel(backbone=mock_backbone, vocab_size=1000, tie_weights=True)
        # Force build via forward pass to ensure variables exist
        _ = model(sample_inputs)
        assert model.use_weight_tying is True
        assert model.embedding_weights is not None
        assert model.output_bias is not None

    def test_weight_tying_disabled_explicitly(self, mock_backbone, sample_inputs):
        model = CausalLanguageModel(backbone=mock_backbone, vocab_size=1000, tie_weights=False)
        _ = model(sample_inputs)
        assert model.use_weight_tying is False
        assert model.output_layer is not None

    def test_weight_tying_fallback(self, mock_backbone_no_embeddings, sample_inputs):
        model = CausalLanguageModel(backbone=mock_backbone_no_embeddings, vocab_size=1000, tie_weights=True)
        # Should fallback to False because embeddings are private/hidden
        _ = model(sample_inputs)
        assert model.use_weight_tying is False
        assert model.output_layer is not None

    def test_weight_tying_with_get_embedding_matrix(self, mock_backbone_with_get_embedding, sample_inputs):
        model = CausalLanguageModel(backbone=mock_backbone_with_get_embedding, vocab_size=1000, tie_weights=True)
        _ = model(sample_inputs)
        assert model.use_weight_tying is True
        assert model.embedding_weights is not None

    def test_metrics_created(self, clm_model):
        assert len(clm_model.metrics) == 3

class TestCausalLanguageModelCall:
    def test_output_shape(self, clm_model, sample_inputs):
        output = clm_model(sample_inputs, training=False)
        assert output.shape == (4, 32, 1000)

    def test_output_shape_no_tying(self, clm_model_no_tying, sample_inputs):
        output = clm_model_no_tying(sample_inputs, training=False)
        assert output.shape == (4, 32, 1000)

    def test_output_dtype(self, clm_model, sample_inputs):
        output = clm_model(sample_inputs)
        assert output.dtype == tf.float32

    def test_training_vs_inference_mode(self, clm_model, sample_inputs):
        out_train = clm_model(sample_inputs, training=True)
        out_infer = clm_model(sample_inputs, training=False)
        assert out_train.shape == out_infer.shape

    def test_call_does_not_shift_tokens(self, clm_model, sample_inputs):
        output = clm_model(sample_inputs)
        # call() returns logic for full sequence (inference), train_step shifts
        assert output.shape[1] == 32

class TestPrepareInputsAndLabels:
    def test_token_shifting(self, clm_model):
        inputs = {
            "input_ids": tf.constant([[1, 2, 3, 4, 5]], dtype=tf.int32),
            "attention_mask": tf.ones((1, 5), dtype=tf.int32)
        }
        x, y, mask = clm_model._prepare_inputs_and_labels(inputs)

        # x: [1,2,3,4], y: [2,3,4,5]
        np.testing.assert_array_equal(x["input_ids"], [[1, 2, 3, 4]])
        np.testing.assert_array_equal(y, [[2, 3, 4, 5]])
        np.testing.assert_array_equal(mask, [[1, 1, 1, 1]])

    def test_attention_mask_shifted(self, clm_model):
        # Mask is [1, 1, 1, 0, 0] -> Shifted: [1, 1, 1, 0]
        inputs = {
            "input_ids": tf.constant([[1, 2, 3, 4, 5]], dtype=tf.int32),
            "attention_mask": tf.constant([[1, 1, 1, 0, 0]], dtype=tf.int32)
        }
        x, _, mask = clm_model._prepare_inputs_and_labels(inputs)
        np.testing.assert_array_equal(mask, [[1, 1, 1, 0]])

    def test_no_attention_mask(self, clm_model):
        inputs = {"input_ids": tf.constant([[1, 2, 3]], dtype=tf.int32)}
        x, y, mask = clm_model._prepare_inputs_and_labels(inputs)
        assert mask is None
        assert x["input_ids"].shape[1] == 2

    def test_batch_handling(self, clm_model, sample_inputs):
        x, y, _ = clm_model._prepare_inputs_and_labels(sample_inputs)
        assert x["input_ids"].shape == (4, 31)
        assert y.shape == (4, 31)

class TestApplyOutputHead:
    def test_output_head_with_weight_tying(self, clm_model, sample_inputs):
        # Ensure built
        _ = clm_model(sample_inputs)
        hidden = tf.random.normal((2, 5, 64))
        logits = clm_model._apply_output_head(hidden)
        assert logits.shape == (2, 5, 1000)

    def test_output_head_without_weight_tying(self, clm_model_no_tying, sample_inputs):
        _ = clm_model_no_tying(sample_inputs)
        hidden = tf.random.normal((2, 5, 64))
        logits = clm_model_no_tying._apply_output_head(hidden)
        assert logits.shape == (2, 5, 1000)

class TestCausalLanguageModelComputeLoss:
    def test_loss_basic(self, clm_model):
        y = tf.random.uniform((2, 10), maxval=1000, dtype=tf.int32)
        y_pred = tf.random.normal((2, 10, 1000))
        loss = clm_model.compute_loss(y=y, y_pred=y_pred)
        assert float(loss) > 0

    def test_loss_with_mask(self, clm_model):
        y = tf.random.uniform((2, 10), maxval=1000, dtype=tf.int32)
        y_pred = tf.random.normal((2, 10, 1000))
        mask = tf.concat([tf.ones((2, 5)), tf.zeros((2, 5))], axis=1)
        loss = clm_model.compute_loss(y=y, y_pred=y_pred, sample_weight=mask)
        assert float(loss) > 0

    def test_loss_zero_mask(self, clm_model):
        y = tf.zeros((2, 10), dtype=tf.int32)
        y_pred = tf.zeros((2, 10, 1000))
        mask = tf.zeros((2, 10))
        loss = clm_model.compute_loss(y=y, y_pred=y_pred, sample_weight=mask)
        assert not keras.ops.isnan(loss)

class TestCausalLanguageModelTraining:
    def test_train_step_dict(self, clm_model, sample_inputs):
        clm_model.compile(optimizer="adam")
        # Explicit build
        _ = clm_model(sample_inputs)
        metrics = clm_model.train_step(sample_inputs)
        assert "loss" in metrics
        assert "perplexity" in metrics

    def test_train_step_tuple(self, clm_model, sample_inputs):
        clm_model.compile(optimizer="adam")
        _ = clm_model(sample_inputs)
        metrics = clm_model.train_step((sample_inputs, None, None))
        assert "loss" in metrics

    def test_test_step(self, clm_model, sample_inputs):
        clm_model.compile(optimizer="adam")
        _ = clm_model(sample_inputs) # build
        metrics = clm_model.test_step(sample_inputs)
        assert "loss" in metrics

    def test_perplexity_logic(self, clm_model, sample_inputs):
        clm_model.compile(optimizer="adam")
        _ = clm_model(sample_inputs)
        metrics = clm_model.test_step(sample_inputs)
        loss = metrics["loss"]
        perp = metrics["perplexity"]
        np.testing.assert_allclose(perp, np.exp(loss), rtol=1e-4)

    def test_weights_update(self, clm_model, sample_inputs):
        clm_model.compile(optimizer=keras.optimizers.SGD(1.0))
        _ = clm_model(sample_inputs) # build
        w0 = [w.numpy() for w in clm_model.trainable_weights]
        clm_model.train_step(sample_inputs)
        w1 = [w.numpy() for w in clm_model.trainable_weights]

        changed = False
        for a, b in zip(w0, w1):
            if not np.allclose(a, b):
                changed = True
        assert changed

    def test_padding_train(self, clm_model, sample_inputs_with_padding):
        clm_model.compile(optimizer="adam")
        _ = clm_model(sample_inputs_with_padding)
        metrics = clm_model.train_step(sample_inputs_with_padding)
        assert not np.isnan(metrics["loss"])

class TestCausalLanguageModelSerialization:
    def test_get_config(self, clm_model):
        config = clm_model.get_config()
        assert config["vocab_size"] == 1000
        assert config["tie_weights"] is True

    def test_from_config(self, clm_model, sample_inputs):
        config = clm_model.get_config()
        model2 = CausalLanguageModel.from_config(config)
        assert model2.tie_weights is True

    def test_save_and_load(self, clm_model, sample_inputs, tmp_path):
        # Build and Run to initialize variables
        _ = clm_model(sample_inputs)

        path = tmp_path / "model.keras"
        clm_model.save(path)

        loaded = keras.models.load_model(path)
        # Verify outputs match
        out1 = clm_model(sample_inputs)
        out2 = loaded(sample_inputs)
        np.testing.assert_allclose(out1, out2, atol=1e-5)

    def test_save_and_load_no_tying(self, clm_model_no_tying, sample_inputs, tmp_path):
        _ = clm_model_no_tying(sample_inputs)
        path = tmp_path / "model_untied.keras"
        clm_model_no_tying.save(path)
        loaded = keras.models.load_model(path)
        out1 = clm_model_no_tying(sample_inputs)
        out2 = loaded(sample_inputs)
        np.testing.assert_allclose(out1, out2, atol=1e-5)

    def test_config_roundtrip(self, clm_model):
        config = clm_model.get_config()
        model2 = CausalLanguageModel.from_config(config)
        config2 = model2.get_config()
        for k in config:
            if k != "backbone":
                assert config[k] == config2[k]

class TestCausalLanguageModelWithPadding:
    def test_forward_pad(self, clm_model, sample_inputs_with_padding):
        out = clm_model(sample_inputs_with_padding)
        assert out.shape == (4, 32, 1000)

class TestCausalLanguageModelIntegration:
    def test_fit_batch(self, clm_model, sample_inputs):
        clm_model.compile(optimizer="adam")
        # Ensure build before fit for custom model robustly
        _ = clm_model(sample_inputs)
        ds = tf.data.Dataset.from_tensors(sample_inputs).repeat(2)
        hist = clm_model.fit(ds, epochs=1, verbose=0)
        assert len(hist.history["loss"]) == 1

    def test_evaluate(self, clm_model, sample_inputs):
        clm_model.compile(optimizer="adam")
        _ = clm_model(sample_inputs) # build
        ds = tf.data.Dataset.from_tensors(sample_inputs)
        res = clm_model.evaluate(ds, verbose=0, return_dict=True)
        assert "perplexity" in res

    def test_backbone_extract(self, clm_model):
        assert clm_model.backbone.hidden_size == 64

    def test_loss_decrease(self, mock_backbone, sample_inputs):
        model = CausalLanguageModel(backbone=mock_backbone, vocab_size=1000)
        model.compile(optimizer=keras.optimizers.Adam(0.01))
        _ = model(sample_inputs) # build

        l_start = model.train_step(sample_inputs)["loss"]
        for _ in range(5):
            l_end = model.train_step(sample_inputs)["loss"]

        assert l_end < l_start

    def test_weight_tying_shares_parameters(self, mock_backbone, sample_inputs):
        model = CausalLanguageModel(backbone=mock_backbone, vocab_size=1000, tie_weights=True)
        # Must build to tie
        _ = model(sample_inputs)

        # In Keras 3, model.embedding_weights will be the Variable itself
        emb_var = mock_backbone.token_embeddings.variables[0]

        np.testing.assert_array_equal(
            model.embedding_weights.numpy(),
            emb_var.numpy()
        )

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])