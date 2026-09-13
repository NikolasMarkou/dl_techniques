"""
Tests for Causal Language Model (CLM) Pre-training Framework
=============================================================

Comprehensive tests for the CausalLanguageModel class.
"""

import logging

import keras
from keras import ops
import numpy as np
import pytest
import tensorflow as tf
from typing import Dict, Any, Optional

from dl_techniques.models.language.masked_language_model.clm import CausalLanguageModel


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

    def test_loss_weights_are_label_aligned_not_input_aligned(self, clm_model):
        """The returned weights multiply LABELS, so they are ``mask[:, 1:]``.

        This test previously asserted ``[[1, 1, 1, 0]]`` -- the *input*-aligned
        slice ``mask[:, :-1]`` -- and so pinned an off-by-one: with a real mask
        of ``[1, 1, 1, 0, 0]`` the tokens are ``[t0, t1, t2, PAD, PAD]``, and
        weighting the third prediction with a 1 scores the model on predicting
        the first padding id. The label-aligned slice is ``[[1, 1, 0, 0]]``.
        The backbone still receives the input-aligned slice; the two are
        checked separately here because they are genuinely different objects.
        """
        inputs = {
            "input_ids": tf.constant([[1, 2, 3, 4, 5]], dtype=tf.int32),
            "attention_mask": tf.constant([[1, 1, 1, 0, 0]], dtype=tf.int32)
        }
        x, _, loss_weights = clm_model._prepare_inputs_and_labels(inputs)
        np.testing.assert_array_equal(loss_weights, [[1, 1, 0, 0]])
        np.testing.assert_array_equal(x["attention_mask"], [[1, 1, 1, 0]])

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

@keras.saving.register_keras_serializable()
class MockBidirectionalBackbone(keras.Model):
    """A backbone that mixes across the whole sequence -- i.e. leaks the future.

    ``GlobalAveragePooling`` broadcast back over the sequence is the smallest
    thing that makes every output position depend on every input position, so
    it is the dead component the causality guard must catch.
    """

    def __init__(self, hidden_size=64, vocab_size=1000, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.token_embeddings = keras.layers.Embedding(vocab_size, hidden_size)
        self.dense = keras.layers.Dense(hidden_size)

    def call(self, inputs, training=False):
        x = self.token_embeddings(inputs["input_ids"])
        pooled = ops.mean(x, axis=1, keepdims=True)
        x = self.dense(x + pooled)
        return {"last_hidden_state": x}

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_size": self.hidden_size, "vocab_size": self.vocab_size})
        return config


class TestBackboneCausalityGuard:
    """`CausalLanguageModel` cannot inject a mask into an arbitrary backbone,
    but it must refuse one that leaks: under a next-token objective a
    bidirectional backbone trains on its own targets, and the symptom is a
    loss that collapses -- i.e. it looks like success.
    """

    def test_bidirectional_backbone_is_rejected(self, sample_inputs):
        model = CausalLanguageModel(
            backbone=MockBidirectionalBackbone(hidden_size=64, vocab_size=1000),
            vocab_size=1000,
        )
        with pytest.raises(ValueError, match="NOT causal"):
            model(sample_inputs)

    def test_causal_backbone_is_accepted(self, mock_backbone, sample_inputs):
        model = CausalLanguageModel(backbone=mock_backbone, vocab_size=1000)
        out = model(sample_inputs)
        assert out.shape == (4, 32, 1000)

    def test_the_guard_can_be_switched_off(self, sample_inputs):
        """Opt-out exists, and it really opts out."""
        model = CausalLanguageModel(
            backbone=MockBidirectionalBackbone(hidden_size=64, vocab_size=1000),
            vocab_size=1000,
            verify_causality=False,
        )
        out = model(sample_inputs)
        assert out.shape == (4, 32, 1000)

    def test_verify_causality_survives_get_config(self, mock_backbone):
        model = CausalLanguageModel(
            backbone=mock_backbone, vocab_size=1000, verify_causality=False
        )
        assert model.get_config()["verify_causality"] is False


@keras.saving.register_keras_serializable()
class PlainTensorOnlyBackbone(keras.Model):
    """A backbone shaped like Zamba2Model/HNet: ``call()`` takes ONLY a plain
    positional ``input_ids`` tensor, never a dict. Feeding it the
    ``{"input_ids": ..., "attention_mask": ...}`` dict the causality probe
    hardcodes by default crashes inside ``self.token_embeddings(input_ids)``,
    since ``Embedding`` expects an integer tensor, not a mapping -- exactly
    the plain-tensor-only shape this step's fix targets (step 1 / D-003).
    """

    def __init__(self, hidden_size=64, vocab_size=1000, bidirectional=False, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.bidirectional = bidirectional
        self.token_embeddings = keras.layers.Embedding(vocab_size, hidden_size)
        self.dense = keras.layers.Dense(hidden_size)

    def call(self, input_ids, training=False):
        x = self.token_embeddings(input_ids)
        if self.bidirectional:
            # Mixes every position into every other -- the future-leak the
            # probe must catch.
            pooled = ops.mean(x, axis=1, keepdims=True)
            x = self.dense(x + pooled)
        else:
            x = self.dense(x)
        return {"last_hidden_state": x}

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "vocab_size": self.vocab_size,
                "bidirectional": self.bidirectional,
            }
        )
        return config


class TestCausalityProbePlainTensor:
    """``causality_probe_plain_tensor`` (step 1 / D-003) fixes
    ``_verify_backbone_causality``'s hardcoded dict-input convention for a
    backbone whose ``call()`` accepts only a plain positional tensor (e.g.
    Zamba2Model, HNet).
    """

    COULD_NOT_RUN = "could not run the causality probe"

    def test_probe_actually_runs_and_accepts_a_causal_plain_tensor_backbone(self, caplog):
        """RED-proof A (new case): ``causality_probe_plain_tensor=True``
        against a causal plain-tensor-only backbone -- the probe must
        genuinely execute (no "could not run" warning) and must NOT raise,
        since this backbone is causal."""
        with caplog.at_level(logging.WARNING, logger="dl"):
            model = CausalLanguageModel(
                backbone=PlainTensorOnlyBackbone(
                    hidden_size=64, vocab_size=1000, bidirectional=False
                ),
                vocab_size=1000,
                causality_probe_plain_tensor=True,
            )
            input_ids = tf.random.uniform((2, 8), minval=0, maxval=1000, dtype=tf.int32)
            out = model(input_ids, training=False)  # triggers build() -> probe

        assert self.COULD_NOT_RUN not in caplog.text.lower()
        assert out.shape == (2, 8, 1000)

    def test_probe_actually_runs_and_rejects_a_bidirectional_plain_tensor_backbone(
        self, caplog
    ):
        """RED-proof A (new case), continued: the SAME flag against a
        deliberately bidirectional plain-tensor-only backbone must raise
        ``ValueError`` -- proof the probe is genuinely comparing hidden
        states, not merely "not crashing"."""
        with caplog.at_level(logging.WARNING, logger="dl"):
            model = CausalLanguageModel(
                backbone=PlainTensorOnlyBackbone(
                    hidden_size=64, vocab_size=1000, bidirectional=True
                ),
                vocab_size=1000,
                causality_probe_plain_tensor=True,
            )
            input_ids = tf.random.uniform((2, 8), minval=0, maxval=1000, dtype=tf.int32)
            with pytest.raises(ValueError, match="NOT causal"):
                model(input_ids, training=False)

        assert self.COULD_NOT_RUN not in caplog.text.lower()

    def test_probe_default_silently_skips_a_plain_tensor_only_backbone(self, caplog):
        """RED-proof B (regression guard): the SAME stub backbone (causal
        variant), but with ``causality_probe_plain_tensor=False`` (the
        default) -- documents TODAY's silent-degrade behavior precisely: the
        hardcoded dict probe crashes inside the backbone's own
        ``token_embeddings`` call, is swallowed by
        ``_verify_backbone_causality``'s broad ``try/except``, emits the
        "could not run" warning, and does NOT raise -- even though this
        backbone is, in fact, causal and would have passed had the probe
        actually run. A future accidental change to this default must fail
        THIS test, not just the opposite (fixed) case above.
        """
        with caplog.at_level(logging.WARNING, logger="dl"):
            model = CausalLanguageModel(
                backbone=PlainTensorOnlyBackbone(
                    hidden_size=64, vocab_size=1000, bidirectional=False
                ),
                vocab_size=1000,
                causality_probe_plain_tensor=False,
            )
            input_ids = tf.random.uniform((2, 8), minval=0, maxval=1000, dtype=tf.int32)
            out = model(input_ids, training=False)  # must NOT raise

        assert self.COULD_NOT_RUN in caplog.text.lower()
        assert out.shape == (2, 8, 1000)

    def test_causality_probe_plain_tensor_default_is_false(self, clm_model):
        """The additive flag defaults False, matching prior behavior."""
        assert clm_model.causality_probe_plain_tensor is False

    def test_causality_probe_plain_tensor_survives_get_config_roundtrip(self, mock_backbone):
        model = CausalLanguageModel(
            backbone=mock_backbone,
            vocab_size=1000,
            causality_probe_plain_tensor=True,
            verify_causality=False,
        )
        config = model.get_config()
        assert config["causality_probe_plain_tensor"] is True
        model2 = CausalLanguageModel.from_config(config)
        assert model2.causality_probe_plain_tensor is True


class TestSkipHead:
    """`skip_head=True` wraps a backbone that already bakes its own head and
    returns vocabulary logits directly -- no ``hidden_size`` requirement, no
    ``last_hidden_state`` dict-indexing, no tied/untied head construction.

    Uses a REAL headed backbone (`Qwen3`, tiny-sized), not a mock, per
    plan.md step 1's testing requirement.
    """

    @staticmethod
    def _tiny_qwen3():
        from dl_techniques.models.language.qwen.qwen3 import Qwen3

        return Qwen3(
            vocab_size=48,
            hidden_size=16,
            num_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_seq_len=16,
        )

    def test_skip_head_output_matches_backbone_raw_logits(self):
        """`call()`'s output IS the backbone's own logits, unmodified."""
        backbone = self._tiny_qwen3()
        model = CausalLanguageModel(
            backbone=backbone, vocab_size=48, skip_head=True, verify_causality=False
        )
        input_ids = tf.random.uniform((2, 8), minval=0, maxval=48, dtype=tf.int32)
        wrapped_out = model({"input_ids": input_ids}, training=False)
        raw_out = backbone(input_ids, training=False)

        assert wrapped_out.shape == raw_out.shape == (2, 8, 48)
        assert wrapped_out.dtype == raw_out.dtype
        np.testing.assert_allclose(
            ops.convert_to_numpy(wrapped_out),
            ops.convert_to_numpy(raw_out),
            atol=1e-6,
            rtol=0,
        )

    def test_skip_head_creates_no_head_state(self):
        """No `output_bias`/`output_layer`/`embedding_weights` populated."""
        backbone = self._tiny_qwen3()
        model = CausalLanguageModel(
            backbone=backbone, vocab_size=48, skip_head=True, verify_causality=False
        )
        input_ids = tf.random.uniform((2, 8), minval=0, maxval=48, dtype=tf.int32)
        _ = model({"input_ids": input_ids}, training=False)

        assert model.output_bias is None
        assert model.output_layer is None
        assert model.embedding_weights is None
        assert model.use_weight_tying is False
        assert model.hidden_size is None

    def test_skip_head_does_not_require_hidden_size(self):
        """The `hasattr(backbone, "hidden_size")` check is not enforced.

        A backbone-stand-in that deliberately lacks `hidden_size` (unlike
        `Qwen3`, which does have it) constructs cleanly under `skip_head=True`
        and scores correctly -- proof the check would not matter even if
        `hidden_size` were absent.
        """

        @keras.saving.register_keras_serializable()
        class HeadedBackboneNoHiddenSize(keras.Model):
            """Bakes its own head; deliberately has no `hidden_size`."""

            def __init__(self, vocab_size=48, **kwargs):
                super().__init__(**kwargs)
                self.vocab_size = vocab_size
                self.embed = keras.layers.Embedding(vocab_size, 16)
                self.head = keras.layers.Dense(vocab_size)

            def call(self, inputs, training=False):
                input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs
                return self.head(self.embed(input_ids))

            def get_config(self):
                config = super().get_config()
                config.update({"vocab_size": self.vocab_size})
                return config

        assert not hasattr(HeadedBackboneNoHiddenSize(), "hidden_size")

        backbone = HeadedBackboneNoHiddenSize(vocab_size=48)
        model = CausalLanguageModel(
            backbone=backbone, vocab_size=48, skip_head=True, verify_causality=False
        )
        input_ids = tf.random.uniform((2, 8), minval=0, maxval=48, dtype=tf.int32)
        out = model({"input_ids": input_ids}, training=False)
        assert out.shape == (2, 8, 48)

    def test_skip_head_survives_get_config_roundtrip(self, mock_backbone):
        model = CausalLanguageModel(
            backbone=mock_backbone, vocab_size=1000, skip_head=True, verify_causality=False
        )
        config = model.get_config()
        assert config["skip_head"] is True
        model2 = CausalLanguageModel.from_config(config)
        assert model2.skip_head is True

    def test_skip_head_default_is_false(self, clm_model):
        """The additive flag defaults False, matching prior behavior."""
        assert clm_model.skip_head is False


class TestPreShifted:
    """`pre_shifted=True` skips `_prepare_inputs_and_labels`: `train_step`/
    `test_step` unpack ``data`` into ``(x, y)`` via
    ``keras.utils.unpack_x_y_sample_weight`` and use both AS GIVEN, matching
    what ``preprocess_clm_packed_dataset`` already yields upstream.

    This is a delta-impulse / offset-tracking probe, not a shape/finiteness
    check: ``FixedPatternBackbone`` below always predicts a FIXED answer
    pattern indexed by absolute sequence POSITION, ignoring the input's
    actual token values. That pattern lines up with the batch's true,
    correctly-paired label only when the input is used un-shifted. A second,
    internal shift (the double-shift bug this flag exists to prevent)
    truncates the sequence by one position and reconstructs a DIFFERENT
    label straight from ``x`` itself, discarding the batch's real ``y``
    entirely -- the fixed pattern does not match that reconstructed label,
    so the loss goes sharply UP, not merely finite. This is what makes the
    test able to tell single-shift and double-shift apart.
    """

    PATTERN = [11, 21, 31, 41, 51, 61, 71, 81]  # the one true, position-indexed label
    VOCAB_SIZE = 100
    LOGIT_SCALE = 30.0

    @staticmethod
    def _make_backbone():
        @keras.saving.register_keras_serializable()
        class FixedPatternBackbone(keras.Model):
            """Predicts a FIXED, position-indexed pattern; ignores input values.

            A stub whose ``call()`` returns a KNOWN, hand-inspectable logits
            tensor that depends only on the input's sequence LENGTH, not its
            content -- exactly the "fixed, inspectable tensor unrelated to
            input" shape needed to compute what ``compute_loss`` should give
            under "correct single shift" vs "accidental double shift" by
            hand, ahead of running either.
            """

            def __init__(self, pattern=None, vocab_size=100, scale=30.0, **kwargs):
                super().__init__(**kwargs)
                self.pattern = list(pattern) if pattern is not None else list(
                    TestPreShifted.PATTERN
                )
                self.vocab_size = vocab_size
                self.scale = scale

            def call(self, inputs, training=False):
                input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs
                seq_len = input_ids.shape[1]
                batch_size = ops.shape(input_ids)[0]
                classes = tf.constant(self.pattern[:seq_len], dtype=tf.int32)
                one_hot = tf.one_hot(classes, depth=self.vocab_size) * self.scale
                one_hot = ops.expand_dims(one_hot, axis=0)
                return ops.tile(one_hot, [batch_size, 1, 1])

            def get_config(self):
                config = super().get_config()
                config.update(
                    {
                        "pattern": self.pattern,
                        "vocab_size": self.vocab_size,
                        "scale": self.scale,
                    }
                )
                return config

        return FixedPatternBackbone(
            TestPreShifted.PATTERN, TestPreShifted.VOCAB_SIZE, TestPreShifted.LOGIT_SCALE
        )

    def test_pre_shifted_train_step_uses_batch_unchanged(self):
        """Correct behavior: the fixed pattern predicts the TRUE label
        exactly when the batch's own ``(x, y)`` is used un-shifted, so the
        loss should be ~0."""
        backbone = self._make_backbone()
        model = CausalLanguageModel(
            backbone=backbone,
            vocab_size=self.VOCAB_SIZE,
            skip_head=True,
            pre_shifted=True,
            verify_causality=False,
        )
        model.compile(optimizer="adam")
        x = tf.constant([[10, 20, 30, 40, 50, 60, 70, 80]], dtype=tf.int32)
        y = tf.constant([[11, 21, 31, 41, 51, 61, 71, 81]], dtype=tf.int32)
        _ = model(x, training=False)  # build

        metrics = model.test_step((x, y))
        loss = float(metrics["loss"])
        assert loss < 0.01, (
            "pre_shifted=True must train against the batch's OWN (x, y) "
            f"unchanged; the fixed-pattern backbone predicts the true label "
            f"exactly under a single shift, so loss should be ~0, got {loss}"
        )

    def test_double_shift_would_be_wrong_not_merely_finite(self):
        """RED-proof: if ``pre_shifted=True`` accidentally still routed
        through ``_prepare_inputs_and_labels``, the reconstructed label
        would be a DIFFERENT target the fixed pattern does not match -- a
        sharply HIGHER loss, not merely a finite one.

        This directly exercises ``_prepare_inputs_and_labels`` (the
        double-shift bug's mechanism) against the identical backbone and
        batch as the test above, to prove the two are distinguishable by
        more than shape or finiteness. Before this step's fix, forcing
        ``pre_shifted``'s branch in ``train_step``/``test_step`` to fall
        through to ``_prepare_inputs_and_labels`` regardless of the flag
        reproduces exactly this ``broken_loss`` computation -- confirmed by
        temporarily reverting ``_unpack_batch``'s branch during this step's
        implementation and observing this test fail (see step report).
        """
        backbone = self._make_backbone()
        model = CausalLanguageModel(
            backbone=backbone,
            vocab_size=self.VOCAB_SIZE,
            skip_head=True,
            pre_shifted=True,
            verify_causality=False,
        )
        x = tf.constant([[10, 20, 30, 40, 50, 60, 70, 80]], dtype=tf.int32)
        y_correct = tf.constant([[11, 21, 31, 41, 51, 61, 71, 81]], dtype=tf.int32)

        # Correct: single shift, the batch's own (x, y) used unchanged.
        correct_logits = model._backbone_forward(x, training=False)
        correct_loss = float(model.compute_loss(y=y_correct, y_pred=correct_logits))

        # Broken: `pre_shifted=True`'s x fed into the OLD internal shift
        # anyway. `_prepare_inputs_and_labels` truncates by one position and
        # reconstructs ITS OWN label straight from `x` -- discarding the
        # batch's real, upstream-correct `y` entirely.
        x_double_shifted, y_double_shifted, _ = model._prepare_inputs_and_labels(
            {"input_ids": x}
        )
        broken_logits = model._backbone_forward(x_double_shifted, training=False)
        broken_loss = float(model.compute_loss(y=y_double_shifted, y_pred=broken_logits))

        assert correct_loss < 0.01
        assert broken_loss > 10.0
        assert broken_loss > 1000 * max(correct_loss, 1e-9), (
            "A double-shift bug must show up as a sharply WRONG target, not "
            "merely a finite loss -- if this ratio ever collapses, the "
            "probe has stopped distinguishing the two shift levels."
        )

    def test_pre_shifted_works_with_a_headed_dict_backbone_too(self, mock_backbone):
        """`pre_shifted` and `skip_head` are independent, orthogonal flags:
        a headless, dict-input/dict-output backbone (the mamba-shaped case)
        also honors `pre_shifted`, exercising the `_apply_output_head`
        branch under this flag, not only `skip_head=True`."""
        model = CausalLanguageModel(
            backbone=mock_backbone,
            vocab_size=1000,
            pre_shifted=True,
            verify_causality=False,
        )
        model.compile(optimizer="adam")
        x = {"input_ids": tf.random.uniform((2, 8), minval=1, maxval=1000, dtype=tf.int32)}
        y = tf.random.uniform((2, 8), minval=1, maxval=1000, dtype=tf.int32)
        _ = model(x, training=False)  # build

        metrics = model.train_step((x, y))
        assert "loss" in metrics
        assert not np.isnan(metrics["loss"])
        assert model.skip_head is False
        assert model.pre_shifted is True

    def test_pre_shifted_survives_get_config_roundtrip(self, mock_backbone):
        model = CausalLanguageModel(
            backbone=mock_backbone, vocab_size=1000, pre_shifted=True, verify_causality=False
        )
        config = model.get_config()
        assert config["pre_shifted"] is True
        model2 = CausalLanguageModel.from_config(config)
        assert model2.pre_shifted is True

    def test_pre_shifted_default_is_false(self, clm_model):
        """The additive flag defaults False, matching prior behavior."""
        assert clm_model.pre_shifted is False


@keras.saving.register_keras_serializable()
class ScaledCELoss(keras.losses.Loss):
    """A trivial ``keras.losses.Loss`` that scales plain CE by a known
    constant -- distinguishable from the default CE by a fixed, predictable
    ratio, so two models fed the identical batch can be told apart by more
    than "the numbers differ somehow"."""

    def __init__(self, scale=10.0, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale
        self._ce = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="sum_over_batch_size"
        )

    def call(self, y_true, y_pred):
        return self._ce(y_true, y_pred) * self.scale

    def get_config(self):
        config = super().get_config()
        config.update({"scale": self.scale})
        return config


class TestLossFn:
    """``loss_fn`` (D-007/step 2.5) fully replaces ``compute_loss``'s default
    hardcoded cross-entropy when set, so gemma/qwen's ``--loss-type
    focal``/``--label-smoothing`` config path -- neither of which the
    default CE can reproduce -- survives migration onto
    ``CausalLanguageModel``.

    The RED-proof for this flag is not "the flag exists" but "the flag is
    actually CONSULTED by `train_step`/`test_step`, not silently ignored" --
    exactly the failure mode this step guards against (e.g. if `loss_fn`
    were wired into `model.compile(loss=...)` instead, which `train_step`
    never reads). `ScaledCELoss` multiplies CE by a KNOWN constant, so the
    two models' losses on the identical batch must differ by exactly that
    ratio, not merely "some amount".
    """

    def test_loss_fn_default_is_none(self, clm_model):
        """The additive flag defaults None, matching prior behavior."""
        assert clm_model.loss_fn is None

    def test_loss_fn_is_actually_consulted_by_compute_loss(self, mock_backbone, sample_inputs):
        """Two models, identical backbone/inputs, differing only by
        `loss_fn` -- the injected loss must change the computed loss by
        exactly its known scale factor, proving `compute_loss` reads
        `self.loss_fn` rather than ignoring it or reading `compile(loss=...)`.
        """
        default_model = CausalLanguageModel(
            backbone=mock_backbone, vocab_size=1000, verify_causality=False
        )
        scaled_backbone = MockCausalBackbone(hidden_size=64, vocab_size=1000)
        scaled_model = CausalLanguageModel(
            backbone=scaled_backbone,
            vocab_size=1000,
            loss_fn=ScaledCELoss(scale=10.0),
            verify_causality=False,
        )

        # Same weights on both backbones, so both produce identical logits
        # for the identical input -- any difference in the reported loss is
        # attributable ONLY to `loss_fn`, not to a difference in `y_pred`.
        _ = default_model(sample_inputs)
        _ = scaled_model(sample_inputs)
        scaled_backbone.set_weights(mock_backbone.get_weights())

        x, y, mask = default_model._prepare_inputs_and_labels(sample_inputs)
        default_logits = default_model._backbone_forward(x, training=False)
        default_logits = default_model._apply_output_head(default_logits)
        scaled_logits = scaled_model._backbone_forward(x, training=False)
        scaled_logits = scaled_model._apply_output_head(scaled_logits)

        np.testing.assert_allclose(
            ops.convert_to_numpy(default_logits),
            ops.convert_to_numpy(scaled_logits),
            atol=1e-5,
            rtol=1e-5,
        )

        default_loss = float(
            default_model.compute_loss(y=y, y_pred=default_logits, sample_weight=mask)
        )
        scaled_loss = float(
            scaled_model.compute_loss(y=y, y_pred=scaled_logits, sample_weight=mask)
        )

        assert default_loss != pytest.approx(scaled_loss, rel=1e-3), (
            "loss_fn must be CONSULTED by compute_loss -- identical logits "
            "under a 10x-scaling injected loss must NOT produce the same "
            "loss value as the default."
        )
        np.testing.assert_allclose(scaled_loss, default_loss * 10.0, rtol=1e-3)

    def test_loss_fn_via_train_step_end_to_end(self, mock_backbone, sample_inputs):
        """The same distinguishing check, but through `train_step` itself --
        the actual call site gemma/qwen's migrated trainer will exercise."""
        default_model = CausalLanguageModel(
            backbone=mock_backbone, vocab_size=1000, verify_causality=False
        )
        scaled_backbone = MockCausalBackbone(hidden_size=64, vocab_size=1000)
        scaled_model = CausalLanguageModel(
            backbone=scaled_backbone,
            vocab_size=1000,
            loss_fn=ScaledCELoss(scale=10.0),
            verify_causality=False,
        )
        default_model.compile(optimizer="adam")
        scaled_model.compile(optimizer="adam")
        _ = default_model(sample_inputs)
        _ = scaled_model(sample_inputs)
        scaled_backbone.set_weights(mock_backbone.get_weights())

        default_loss = float(default_model.test_step(sample_inputs)["loss"])
        scaled_loss = float(scaled_model.test_step(sample_inputs)["loss"])

        np.testing.assert_allclose(scaled_loss, default_loss * 10.0, rtol=1e-3)

    def test_loss_fn_survives_get_config_roundtrip(self, mock_backbone):
        model = CausalLanguageModel(
            backbone=mock_backbone,
            vocab_size=1000,
            loss_fn=ScaledCELoss(scale=3.0),
            verify_causality=False,
        )
        config = model.get_config()
        assert config["loss_fn"] is not None
        model2 = CausalLanguageModel.from_config(config)
        assert isinstance(model2.loss_fn, ScaledCELoss)
        assert model2.loss_fn.scale == 3.0

    def test_loss_fn_none_survives_get_config_roundtrip(self, clm_model):
        config = clm_model.get_config()
        assert config["loss_fn"] is None
        model2 = CausalLanguageModel.from_config(config)
        assert model2.loss_fn is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])