"""
Comprehensive pytest test suite for the ModernBERT foundation model.

This module provides extensive testing for the ModernBERT implementation including:
- Foundation model initialization and parameter validation for modern features.
- Architecture building (Pre-LN, GeGLU, Hybrid Attention) and consistent output shape.
- Forward pass functionality with a standardized dictionary output.
- Model variant creation and configuration.
- Serialization and deserialization of the pure encoder.
- Error handling and edge cases.
- The in-model task head (`include_top=True`) and the `create_modern_bert` factory.
- End-to-end integration testing for gradient flow and training.
"""

import pytest
import numpy as np
import keras
import tensorflow as tf
import tempfile
import os
from typing import Dict, Any


from dl_techniques.layers.heads.nlp import NLPTaskConfig, NLPTaskType
from dl_techniques.layers.embedding.modern_bert_embeddings import ModernBertEmbeddings
from dl_techniques.models.language.modern_bert.model import ModernBERT, create_modern_bert

# Define a small window size for all tests to avoid OOM errors.
TEST_WINDOW_SIZE = 16


class TestModernBERTModelInitialization:
    """Test ModernBERT model initialization and parameter validation."""

    def test_basic_initialization(self):
        """Test basic ModernBERT model initialization as a pure encoder."""
        model = ModernBERT(
            vocab_size=1000,
            hidden_size=256,
            num_layers=6,
            num_heads=8,
            intermediate_size=384,
            global_attention_interval=2,
            local_attention_window_size=TEST_WINDOW_SIZE, # FIX: Use small window size
            use_bias=False
        )

        assert model.hidden_size == 256
        assert model.num_layers == 6
        assert model.use_bias is False
        assert model.global_attention_interval == 2
        assert not model.built

        # Components should be created in __init__
        assert isinstance(model.embeddings, ModernBertEmbeddings)
        assert len(model.encoder_layers) == 6
        assert model.final_norm is not None

        # But should not be built yet
        assert not model.embeddings.built
        for layer in model.encoder_layers:
            assert not layer.built
        assert not model.final_norm.built

    def test_parameter_validation(self):
        """Test ModernBERT parameter validation for invalid values."""
        with pytest.raises(ValueError, match="hidden_size.*must be divisible by num_heads"):
            ModernBERT(vocab_size=1000, hidden_size=100, num_heads=12, num_layers=4)

        with pytest.raises(ValueError, match="Sizes and layer/head counts must be positive"):
            ModernBERT(vocab_size=1000, hidden_size=-100, num_layers=4, num_heads=8)

        with pytest.raises(ValueError, match="hidden_dropout_rate must be between 0 and 1"):
            ModernBERT(
                vocab_size=1000,
                hidden_size=256,
                num_layers=4,
                num_heads=8,
                hidden_dropout_rate=1.5
            )

        with pytest.raises(ValueError, match="global_attention_interval must be positive"):
            ModernBERT(global_attention_interval=0)

    def test_initialization_with_custom_config(self):
        """Test ModernBERT model initialization with custom configuration."""
        model = ModernBERT(
            vocab_size=25000,
            hidden_size=512,
            num_layers=8,
            num_heads=8,
            intermediate_size=1024,
            hidden_dropout_rate=0.2,
            global_attention_interval=4,
            local_attention_window_size=TEST_WINDOW_SIZE, # FIX: Use small window size
        )
        assert model.vocab_size == 25000
        assert model.hidden_size == 512
        assert model.num_layers == 8
        assert model.hidden_dropout_rate == 0.2
        assert model.global_attention_interval == 4


class TestModernBERTModelVariants:
    """Test ModernBERT model variants and factory methods."""

    def test_bert_base_variant(self):
        # FIX: Override default large window size to prevent OOM
        model = ModernBERT.from_variant("base", local_attention_window_size=TEST_WINDOW_SIZE)
        assert model.hidden_size == 768
        assert model.num_layers == 22
        assert model.num_heads == 12
        assert model.intermediate_size == 1152
        assert model.use_bias is False

    def test_bert_large_variant(self):
        # FIX: Override default large window size to prevent OOM
        model = ModernBERT.from_variant("large", local_attention_window_size=TEST_WINDOW_SIZE)
        assert model.hidden_size == 1024
        assert model.num_layers == 28
        assert model.num_heads == 16
        assert model.intermediate_size == 2624

    def test_invalid_variant(self):
        with pytest.raises(ValueError, match="Unknown variant 'invalid'"):
            ModernBERT.from_variant("invalid")

    def test_variant_with_custom_params(self):
        """Test creating variant with custom parameters."""
        model = ModernBERT.from_variant(
            "base",
            use_bias=True,
            num_layers=10,
            local_attention_window_size=TEST_WINDOW_SIZE # FIX: Use small window size
        )
        assert model.hidden_size == 768
        assert model.num_layers == 10
        assert model.use_bias is True


class TestModernBERTModelBuilding:
    """Test ModernBERT model building and architecture creation."""

    @pytest.fixture
    def basic_config(self) -> Dict[str, Any]:
        return {
            "vocab_size": 1000, "hidden_size": 256, "num_layers": 4,
            "num_heads": 8, "intermediate_size": 384, "global_attention_interval": 2,
            "local_attention_window_size": TEST_WINDOW_SIZE # FIX: Use small window size
        }

    def test_build_basic_functionality(self, basic_config):
        """Test basic building functionality and output contract."""
        model = ModernBERT(**basic_config)
        batch_size, seq_length = 2, 32 # Seq length can be larger now
        input_ids = keras.ops.cast(
            keras.random.uniform((batch_size, seq_length), maxval=basic_config['vocab_size']),
            dtype='int32'
        )

        outputs = model(input_ids, training=False)
        assert model.built
        assert isinstance(outputs, dict)
        assert "last_hidden_state" in outputs
        assert outputs["last_hidden_state"].shape == (
            batch_size, seq_length, basic_config['hidden_size']
        )
        assert "attention_mask" in outputs
        # DECISION plan-2026-08-18T140459-7991552f/D-031: an OMITTED mask is
        # echoed back as all-ones, not as `None`. This line asserted `is None`
        # until 2026-08-19 and was therefore pinning the defect that made
        # `predict({"input_ids": ...})` raise "Structures don't have the same
        # nested structure". The mask still does not reach the encoder --
        # see the D-031 anchor in `modern_bert/model.py`.
        echoed = keras.ops.convert_to_numpy(outputs["attention_mask"])
        assert echoed.shape == (batch_size, seq_length)
        assert (echoed == 1).all()

    def test_transformer_layers_configuration(self, basic_config):
        model = ModernBERT(**basic_config)
        input_ids = keras.ops.cast(
            keras.random.uniform((1, 16), maxval=basic_config['vocab_size']), dtype='int32'
        )
        _ = model(input_ids, training=False)
        for i, layer in enumerate(model.encoder_layers):
            assert layer.hidden_size == basic_config['hidden_size']
            assert layer.num_heads == basic_config['num_heads']
            assert layer.name == f"encoder_layer_{i}"
            assert layer.normalization_position == 'pre'
            assert layer.ffn_type == 'geglu'


class TestModernBERTModelForwardPass:
    """Test ModernBERT model forward pass functionality."""

    @pytest.fixture
    def built_model(self) -> ModernBERT:
        """Create a built ModernBERT model for testing."""
        model = ModernBERT(
            vocab_size=1000, hidden_size=256, num_layers=3, num_heads=8,
            intermediate_size=384,
            local_attention_window_size=TEST_WINDOW_SIZE # FIX: Use small window size
        )
        sample_input = keras.ops.cast(
            keras.random.uniform((1, 16), maxval=1000), dtype='int32'
        )
        _ = model(sample_input, training=False)
        return model

    def test_forward_pass_input_ids_only(self, built_model):
        batch_size, seq_length = 4, 32
        input_ids = keras.ops.cast(
            keras.random.uniform((batch_size, seq_length), maxval=built_model.vocab_size),
            dtype='int32'
        )
        outputs = built_model(input_ids, training=False)
        assert isinstance(outputs, dict)
        assert outputs["last_hidden_state"].shape == (
            batch_size, seq_length, built_model.hidden_size
        )

    def test_forward_pass_dict_input(self, built_model):
        batch_size, seq_length = 3, 24
        inputs = {
            'input_ids': keras.ops.cast(
                keras.random.uniform((batch_size, seq_length), maxval=built_model.vocab_size),
                dtype='int32'
            ),
            'attention_mask': keras.ops.ones((batch_size, seq_length), dtype='int32'),
            'token_type_ids': keras.ops.zeros((batch_size, seq_length), dtype='int32')
        }
        outputs = built_model(inputs, training=False)
        assert isinstance(outputs, dict)
        assert outputs["last_hidden_state"].shape == (
            batch_size, seq_length, built_model.hidden_size
        )
        assert "attention_mask" in outputs
        assert outputs["attention_mask"] is not None

    def test_invalid_dict_input(self, built_model):
        inputs = {'invalid_key': keras.ops.ones((2, 16), dtype='int32')}
        with pytest.raises(ValueError, match="Dictionary input must contain 'input_ids' key"):
            built_model(inputs, training=False)


class TestModernBERTModelSerialization:
    """Test ModernBERT model serialization and deserialization."""

    def test_config_serialization(self):
        model = ModernBERT(
            vocab_size=25000, hidden_size=512, num_layers=8, num_heads=8,
            global_attention_interval=3, use_bias=False,
            local_attention_window_size=TEST_WINDOW_SIZE # FIX: Use small window size
        )
        model_config = model.get_config()
        assert model_config['vocab_size'] == 25000
        assert model_config['hidden_size'] == 512
        assert model_config['global_attention_interval'] == 3
        assert model_config['use_bias'] is False

    def test_model_from_config(self):
        original_model = ModernBERT(
            vocab_size=1000, hidden_size=256, num_layers=4,
            num_heads=8, global_attention_interval=2,
            local_attention_window_size=TEST_WINDOW_SIZE # FIX: Use small window size
        )
        config = original_model.get_config()
        new_model = ModernBERT.from_config(config)
        assert new_model.hidden_size == original_model.hidden_size
        assert new_model.global_attention_interval == original_model.global_attention_interval

    def test_model_save_load(self):
        model = ModernBERT(
            vocab_size=1000, hidden_size=256, num_layers=2, num_heads=8,
            local_attention_window_size=TEST_WINDOW_SIZE # FIX: Use small window size
        )
        input_ids = keras.ops.cast(keras.random.uniform((2, 16), maxval=1000), dtype='int32')
        original_outputs = model(input_ids, training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_modern_bert.keras')
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)
            loaded_outputs = loaded_model(input_ids, training=False)

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_outputs['last_hidden_state']),
                keras.ops.convert_to_numpy(loaded_outputs['last_hidden_state']),
                rtol=1e-5, atol=1e-6, err_msg="Hidden states should match"
            )


class TestModernBERTEdgeCases:
    """Test ModernBERT model edge cases."""

    def test_minimum_sequence_length(self):
        """L=1 still runs, and emits a Keras advisory that is a FALSE POSITIVE.

        Since 2026-08-25 the local layers are 1-D bands
        (plan-2026-08-25T053412-0f1fa04f, D-012) and pad nothing, so at L=1 the
        score tensor is genuinely ``(1, heads, 1, 1)`` and
        ``keras.layers.Softmax`` warns that "this axis has size 1 ... will
        always return the value 1". It is right about the mechanism and wrong
        about the intent: with one token, attention to oneself with weight 1.0
        IS the correct answer. This repo escalates warnings to errors under
        pytest, so the warning is captured explicitly rather than left to blow
        up the test.

        Not band-specific, and NOT a regression this change introduced.
        MEASURED 2026-08-25 at N=1, ``dim=64, num_heads=4``, under
        ``-W error::UserWarning``: ``'multi_head'`` RAISES the same warning and
        ``'window_band'`` RAISES it; ``'window'`` and ``'window_zigzag'`` do not,
        because they pad N=1 up to a full tile -- i.e. the only reason the old
        wiring was quiet here is the padding this plan removed. If
        ``keras.layers.Softmax`` ever stops warning, ``pytest.warns`` fails
        loudly and this docstring should be re-derived, not deleted.
        """
        model = ModernBERT(
            vocab_size=1000, hidden_size=128, num_layers=2, num_heads=8,
            local_attention_window_size=TEST_WINDOW_SIZE # FIX: Use small window size
        )
        input_ids = keras.ops.cast([[42]], dtype='int32')
        with pytest.warns(UserWarning, match="axis has size 1"):
            outputs = model(input_ids, training=False)
        assert outputs['last_hidden_state'].shape == (1, 1, 128)
        assert np.all(np.isfinite(
            keras.ops.convert_to_numpy(outputs['last_hidden_state'])
        ))

    def test_sequence_length_two_needs_no_warning_suppression(self):
        """The control for the test above: the advisory is about L=1 ONLY, so
        an ordinary short sequence must run clean. If this ever needs
        ``pytest.warns`` too, the band is collapsing something it should not."""
        model = ModernBERT(
            vocab_size=1000, hidden_size=128, num_layers=2, num_heads=8,
            local_attention_window_size=TEST_WINDOW_SIZE,
        )
        outputs = model(keras.ops.cast([[42, 7]], dtype='int32'), training=False)
        assert outputs['last_hidden_state'].shape == (1, 2, 128)


class TestModernBERTIntegrationFactory:
    """``create_modern_bert(..., include_top=True)``: the in-model task head.

    The functional ``create_modern_bert_with_head`` this replaced returned the
    head's raw dict; the in-model head returns the finalized output.
    """

    @staticmethod
    def _inputs(seq=32):
        return {
            'input_ids': keras.random.randint((2, seq), 0, 1000, dtype='int32'),
            'attention_mask': keras.ops.ones((2, seq), 'int32'),
            'token_type_ids': keras.ops.zeros((2, seq), 'int32'),
        }

    def test_create_with_classification_head(self):
        """Bare logits: ``probabilities`` is derived from them and dropped."""
        task_config = NLPTaskConfig(
            name="sentiment", task_type=NLPTaskType.SENTIMENT_ANALYSIS, num_classes=3
        )
        model = create_modern_bert(
            "tiny", include_top=True, task_config=task_config,
            local_attention_window_size=TEST_WINDOW_SIZE,
        )
        assert isinstance(model, ModernBERT)
        outputs = model(self._inputs(), training=False)
        assert not isinstance(outputs, dict)
        assert outputs.shape == (2, 3)

    def test_create_with_token_classification_head(self):
        task_config = NLPTaskConfig(
            name="ner", task_type=NLPTaskType.NAMED_ENTITY_RECOGNITION, num_classes=9
        )
        model = create_modern_bert(
            "tiny", include_top=True, task_config=task_config,
            local_attention_window_size=TEST_WINDOW_SIZE,
        )
        outputs = model(self._inputs(), training=False)
        assert not isinstance(outputs, dict)
        assert outputs.shape == (2, 32, 9)


class TestModernBERTIntegration:
    """Integration tests for the complete model (encoder + head)."""

    @pytest.fixture
    def classification_model(self) -> keras.Model:
        """Create a classification model for integration tests."""
        task_config = NLPTaskConfig(
            name="clf", task_type=NLPTaskType.TEXT_CLASSIFICATION, num_classes=3
        )
        return create_modern_bert(
            "tiny",
            include_top=True,
            task_config=task_config,
            hidden_size=64,
            intermediate_size=96,
            local_attention_window_size=TEST_WINDOW_SIZE,
        )

    def test_gradient_flow_integration(self, classification_model):
        """Test that gradients flow through the entire integrated model."""
        batch_size, seq_length = 2, 16
        inputs = {
            'input_ids': keras.ops.cast(keras.random.uniform((batch_size, seq_length), maxval=1000), 'int32'),
            'attention_mask': keras.ops.ones((batch_size, seq_length), 'int32'),
            'token_type_ids': keras.ops.zeros((batch_size, seq_length), 'int32'),
        }

        with tf.GradientTape() as tape:
            logits = classification_model(inputs, training=True)
            targets = keras.ops.one_hot(keras.ops.array([0, 2]), 3)
            loss = keras.ops.mean(keras.losses.categorical_crossentropy(
                targets, logits, from_logits=True))

        gradients = tape.gradient(loss, classification_model.trainable_weights)
        non_none_grads = [g for g in gradients if g is not None]
        assert len(non_none_grads) > 0
        assert len(non_none_grads) == len(classification_model.trainable_weights)


class TestModernBERTAdvancedFeatures:
    """Test advanced ModernBERT features."""

    def test_hybrid_attention_mechanism(self):
        """Verify that attention types alternate correctly."""
        interval = 3
        model = ModernBERT(
            vocab_size=1000, hidden_size=128, num_layers=7, num_heads=8,
            global_attention_interval=interval,
            local_attention_window_size=TEST_WINDOW_SIZE # FIX: Use small window size
        )
        input_ids = keras.ops.cast(keras.random.uniform((2, 16), maxval=1000), 'int32')
        _ = model(input_ids)  # Build the model

        for i, layer in enumerate(model.encoder_layers):
            is_global = (i + 1) % interval == 0
            # Global layers are 'group_query' with num_kv_heads == num_heads,
            # which is arithmetically plain MHA plus RoPE. See the D-007 anchor
            # in models/language/modern_bert/model.py: 'multi_head' cannot carry RoPE.
            # 'window_band' since 2026-08-25 (plan-2026-08-25T053412-0f1fa04f,
            # D-012): the local layers were 'window', a SPATIAL layer over a
            # synthetic ceil(sqrt(L)) grid. They are now a 1-D symmetric band.
            expected_type = "group_query" if is_global else "window_band"
            assert layer.attention_type == expected_type, f"Layer {i} has wrong attention type"
            if is_global:
                assert layer.attention_args["num_kv_heads"] == model.num_heads

    def test_pre_ln_and_geglu_config(self):
        """Verify that layers are configured for Pre-LN and GeGLU."""
        model = ModernBERT.from_variant("tiny", local_attention_window_size=TEST_WINDOW_SIZE)
        input_ids = keras.ops.cast(keras.random.uniform((2, 16), maxval=1000), 'int32')
        _ = model(input_ids)  # Build the model

        # Check a sample encoder layer
        sample_layer = model.encoder_layers[0]
        assert sample_layer.normalization_position == 'pre'
        assert sample_layer.ffn_type == 'geglu'

    def test_model_summary(self):
        """Test that model summary works without errors."""
        model = ModernBERT.from_variant("tiny", local_attention_window_size=TEST_WINDOW_SIZE)
        input_ids = keras.ops.cast(keras.random.uniform((1, 16), maxval=1000), 'int32')
        _ = model(input_ids, training=False)
        try:
            model.summary()
        except Exception as e:
            pytest.fail(f"Model summary raised an exception: {e}")



class TestPretrainedContract:
    """`pretrained=True` must RAISE, never return a random-init model.

    Before this contract, `ModernBERT.PRETRAINED_WEIGHTS` held placeholder URLs on a non-existent host,
    `from_variant` caught the download failure, logged a warning and continued with
    random initialization — so a caller asking for pretrained weights silently
    got untrained ones and no error. Do not reinstate that.
    """

    def test_from_variant_pretrained_true_raises(self):
        with pytest.raises(NotImplementedError, match="No pretrained ModernBERT weights"):
            ModernBERT.from_variant("tiny", pretrained=True)

    def test_pretrained_false_still_builds(self):
        model = ModernBERT.from_variant("tiny", vocab_size=128)
        assert isinstance(model, ModernBERT)

    def test_no_placeholder_weight_table(self):
        assert not hasattr(ModernBERT, "PRETRAINED_WEIGHTS")


class TestCreateModernBERT:
    """``create_modern_bert`` is the package's factory; the head wrapper is gone."""

    def test_public_api(self):
        import dl_techniques.models.language.modern_bert as pkg

        assert sorted(pkg.__all__) == ["ModernBERT", "create_modern_bert"]
        assert pkg.create_modern_bert is create_modern_bert
        # Replaced by `include_top` on the class (plan-2026-10-11T004932-b83935e1).
        assert not hasattr(pkg, "create_modern_bert_with_head")

    def test_pretrained_true_raises(self):
        with pytest.raises(NotImplementedError, match="No pretrained ModernBERT weights"):
            create_modern_bert("tiny", pretrained=True)

    def test_delegates_to_from_variant(self):
        model = create_modern_bert("tiny", vocab_size=128, max_position_embeddings=32)
        assert isinstance(model, ModernBERT)
        tiny = ModernBERT.MODEL_VARIANTS["tiny"]
        assert (model.hidden_size, model.num_layers) == (
            tiny["hidden_size"], tiny["num_layers"])
        assert model.vocab_size == 128 and model.max_position_embeddings == 32


# ---------------------------------------------------------------------
# include_top: the in-model task head
# ---------------------------------------------------------------------

#: Small enough that every arm below runs in about a second. The sub-128
#: ``max_position_embeddings`` also pins the load path: its old fixed-length
#: (1, 128) dummy forward made the global layers' RoPE raise on every
#: ``pretrained=<path>`` load into such a model.
_HEAD_KW: Dict[str, Any] = {
    "vocab_size": 64,
    "hidden_size": 32,
    "num_layers": 2,
    "num_heads": 2,
    "intermediate_size": 64,
    "global_attention_interval": 2,
    "local_attention_window_size": TEST_WINDOW_SIZE,
    "max_position_embeddings": 32,
}
_SEQ = 8
_CLF = NLPTaskConfig(
    name="clf", task_type=NLPTaskType.TEXT_CLASSIFICATION, num_classes=3
)
_NER = NLPTaskConfig(
    name="ner", task_type=NLPTaskType.TOKEN_CLASSIFICATION, num_classes=9
)
#: Same head class and weight shapes as ``_CLF``, a different task.
_SPAM = NLPTaskConfig(
    name="spam", task_type=NLPTaskType.SPAM_DETECTION, num_classes=3
)


def _np(tensor) -> np.ndarray:
    return np.asarray(keras.ops.convert_to_numpy(tensor))


def _head_batch() -> Dict[str, np.ndarray]:
    ids = np.random.default_rng(0).integers(1, 64, (2, _SEQ)).astype("int32")
    mask = np.ones_like(ids)
    mask[1, 5:] = 0
    return {"input_ids": ids, "attention_mask": mask}


def _relative_paths(model: keras.Model) -> list:
    """Weight paths without the model-root segment (guide 8.3)."""
    return sorted(w.path.split("/", 1)[-1] for w in model.weights)


def _backbone_hidden(model: ModernBERT, batch: Dict[str, np.ndarray]) -> np.ndarray:
    """``last_hidden_state`` of ``model``'s encoder, whatever its head.

    Read through a headless twin rebuilt from the model's own config and given
    the model's own encoder weights, layer by layer.
    """
    config = model.get_config()
    config.update(include_top=False, task_config=None, head_config_overrides=None)
    config.pop("name", None)
    twin = ModernBERT.from_config(config)
    twin(batch, training=False)
    for layer in twin.layers:
        layer.set_weights(model.get_layer(layer.name).get_weights())
    return _np(twin(batch, training=False)["last_hidden_state"])


def _randomized(model: ModernBERT, seed: int) -> ModernBERT:
    """Overwrite every weight, so a restored value cannot match by init luck."""
    assert model.weights, "build the model before randomizing it"
    rng = np.random.default_rng(seed)
    for w in model.weights:
        w.assign(rng.normal(0.0, 0.1, size=tuple(w.shape)).astype(w.dtype))
    return model


class TestModernBERTIncludeTop:
    """The ``include_top`` / ``task_config`` / ``head_config_overrides`` knobs."""

    def test_head_output_shapes(self):
        """Bare logits for single-output heads, a dict for QA."""
        batch = _head_batch()
        out = ModernBERT(**_HEAD_KW, include_top=True, task_config=_CLF)(
            batch, training=False)
        assert not isinstance(out, dict) and tuple(out.shape) == (2, 3)

        out = ModernBERT(**_HEAD_KW, include_top=True, task_config=_NER)(
            batch, training=False)
        assert not isinstance(out, dict) and tuple(out.shape) == (2, _SEQ, 9)

        qa = ModernBERT(**_HEAD_KW, include_top=True, task_config=NLPTaskConfig(
            name="qa", task_type=NLPTaskType.QUESTION_ANSWERING, num_classes=2))
        out = qa(batch, training=False)
        assert set(out) == {"start_logits", "end_logits"}
        assert tuple(out["start_logits"].shape) == (2, _SEQ)

    def test_symbolic_output_matches_the_eager_one(self):
        """A symbolic call reports what ``call`` returns, in both modes."""
        model = ModernBERT(**_HEAD_KW, include_top=True, task_config=_NER)
        symbolic = model(keras.Input(shape=(None,), dtype="int32"))
        assert tuple(symbolic.shape) == (None, None, 9)
        headless = ModernBERT(**_HEAD_KW)
        symbolic = headless(keras.Input(shape=(None,), dtype="int32"))
        assert set(symbolic) == {"last_hidden_state", "attention_mask"}

    def test_config_round_trip_with_a_head(self):
        """Every knob is serialized; ``task_config`` as a plain JSON dict."""
        import json

        overrides = {"pooling_type": "mean"}
        model = ModernBERT(**_HEAD_KW, include_top=True, task_config=_CLF,
                           head_config_overrides=overrides)
        config = model.get_config()
        assert config["include_top"] is True
        assert config["head_config_overrides"] == overrides
        assert isinstance(config["task_config"], dict)
        assert config["task_config"]["task_type"] == "text_classification"
        json.dumps(config)  # plain data, no enum or dataclass inside

        from_dict = ModernBERT(**_HEAD_KW, include_top=True,
                               task_config=dict(config["task_config"]),
                               head_config_overrides=overrides)
        assert from_dict.get_config()["task_config"] == config["task_config"]

        rebuilt = ModernBERT.from_config(config)
        assert rebuilt.get_config() == config
        assert rebuilt.task_head.pooling_type == "mean"

        batch = _head_batch()
        model(batch, training=False)
        rebuilt(batch, training=False)
        assert _relative_paths(rebuilt) == _relative_paths(model)
        rebuilt.set_weights(model.get_weights())
        np.testing.assert_array_equal(
            _np(rebuilt(batch, training=False)), _np(model(batch, training=False)))

    def test_a_config_without_the_head_keys_still_loads(self):
        """A config written before the knobs existed rebuilds the encoder."""
        config = ModernBERT(**_HEAD_KW).get_config()
        for key in ("include_top", "task_config", "head_config_overrides"):
            config.pop(key)
        model = ModernBERT.from_config(config)
        assert model.include_top is False and model.task_head is None

    def test_keras_round_trip_restores_the_head_by_value(self, tmp_path):
        """Weights compared BEFORE the first call on the loaded model (guide 8.4)."""
        model = ModernBERT(**_HEAD_KW, include_top=True, task_config=_CLF)
        batch = _head_batch()
        before = _np(model(batch, training=False))
        saved = [_np(w).copy() for w in model.weights]
        head_saved = [_np(w).copy() for w in model.task_head.weights]
        assert head_saved, "the head has no weights to compare"

        path = str(tmp_path / "modern_bert_head.keras")
        model.save(path)
        loaded = keras.models.load_model(path)

        assert len(loaded.weights) == len(saved)
        for s, w in zip(saved, loaded.weights):
            np.testing.assert_allclose(s, _np(w), atol=0.0, rtol=0.0,
                                       err_msg=f"{w.path} not restored")
        for s, w in zip(head_saved, loaded.task_head.weights):
            np.testing.assert_allclose(s, _np(w), atol=0.0, rtol=0.0)
        np.testing.assert_array_equal(_np(loaded(batch, training=False)), before)

    @pytest.mark.parametrize("include_top", [False, True])
    def test_explicit_build_matches_lazy_build(self, include_top):
        """``build()`` materializes exactly the tree ``call()`` runs (guide 8.1)."""
        kw = dict(_HEAD_KW, include_top=include_top,
                  task_config=_CLF if include_top else None)
        explicit = ModernBERT(**kw)
        explicit.build((None, _SEQ))
        lazy = ModernBERT(**kw)
        lazy(_head_batch(), training=False)
        assert _relative_paths(explicit) == _relative_paths(lazy)
        assert explicit.count_params() == lazy.count_params() > 0

    def test_a_disabled_head_builds_no_head_weights(self):
        """Anti-vacuity sibling of the parity arm (guide 8.3)."""
        headless = ModernBERT(**_HEAD_KW)
        headless(_head_batch(), training=False)
        assert headless.task_head is None
        assert not [w.path for w in headless.weights if "task_head" in w.path]

        headed = ModernBERT(**_HEAD_KW, include_top=True, task_config=_CLF)
        headed(_head_batch(), training=False)
        assert [w.path for w in headed.weights if "/task_head/" in w.path]

    def test_the_encoder_weight_paths_do_not_depend_on_the_head(self):
        """The head lives entirely under ``task_head/`` (D-005 of the plan)."""
        def paths(**kw):
            m = ModernBERT(**_HEAD_KW, name="modern_bert", **kw)
            m(_head_batch(), training=False)
            return [w.path for w in m.weights]

        headless = paths()
        for task in (_CLF, _NER):
            headed = paths(include_top=True, task_config=task)
            encoder_part = [
                p for p in headed if not p.startswith("modern_bert/task_head/")]
            assert encoder_part == headless, task.name
            assert len(headed) > len(headless)

    def test_backbone_weights_transfer_across_heads(self, tmp_path):
        """An encoder trained under one head is reused, same weights, under another.

        The supported route is a local ``pretrained`` path, which copies the
        encoder by layer name and never the checkpoint's ``task_head``
        (D-007 of plan-2026-10-11T004932-b83935e1). Exercised into a
        TOKEN_CLASSIFICATION model and into a headless one.
        """
        source = create_modern_bert("tiny", include_top=True, task_config=_CLF,
                                    **_HEAD_KW)
        batch = _head_batch()
        source(batch, training=False)
        _randomized(source, 1)
        path = str(tmp_path / "trained_under_clf.keras")
        source.save(path)
        reference = _backbone_hidden(source, batch)

        token = create_modern_bert("tiny", include_top=True, task_config=_NER,
                                   pretrained=path, **_HEAD_KW)
        headless = create_modern_bert("tiny", pretrained=path, **_HEAD_KW)

        np.testing.assert_array_equal(_backbone_hidden(token, batch), reference)
        np.testing.assert_array_equal(
            _np(headless(batch, training=False)["last_hidden_state"]), reference)
        assert tuple(token(batch, training=False).shape) == (2, _SEQ, 9)

        # Anti-vacuity: a fresh encoder does NOT reproduce the reference.
        fresh = create_modern_bert("tiny", **_HEAD_KW)
        assert not np.array_equal(
            _np(fresh(batch, training=False)["last_hidden_state"]), reference)

    def test_an_equal_shape_head_of_another_task_is_not_copied(self, tmp_path):
        """A SPAM head is not filled from a CLF checkpoint's head of equal shape.

        With the old ``skip_prefixes=()`` the transfer matched the head by
        layer name and copied it silently, because the shapes agree
        (plan-2026-10-11T004932-b83935e1 D-007).
        """
        source = create_modern_bert("tiny", include_top=True, task_config=_CLF,
                                    **_HEAD_KW)
        batch = _head_batch()
        source(batch, training=False)
        _randomized(source, 2)
        path = str(tmp_path / "clf.keras")
        source.save(path)

        spam = create_modern_bert("tiny", include_top=True, task_config=_SPAM,
                                  pretrained=path, **_HEAD_KW)
        src_head = [_np(w) for w in source.task_head.weights]
        dst_head = [_np(w) for w in spam.task_head.weights]
        assert [a.shape for a in src_head] == [b.shape for b in dst_head], (
            "fixture invalid: the two heads must have equal weight shapes")
        assert not any(np.array_equal(a, b) for a, b in zip(src_head, dst_head))
        np.testing.assert_array_equal(
            _backbone_hidden(spam, batch), _backbone_hidden(source, batch))

    def test_pretrained_headless_checkpoint_into_a_headed_model(self, tmp_path):
        """A headless checkpoint restores the encoder of a headed model by value."""
        encoder = create_modern_bert("tiny", **_HEAD_KW)
        batch = _head_batch()
        encoder(batch, training=False)
        _randomized(encoder, 3)
        expected = _np(encoder(batch, training=False)["last_hidden_state"])
        path = str(tmp_path / "encoder.keras")
        encoder.save(path)

        headed = create_modern_bert("tiny", include_top=True, task_config=_CLF,
                                    pretrained=path, **_HEAD_KW)
        np.testing.assert_array_equal(_backbone_hidden(headed, batch), expected)

    def test_include_top_without_task_config_raises(self):
        with pytest.raises(ValueError, match="include_top=True needs a task_config"):
            ModernBERT(**_HEAD_KW, include_top=True)

    @pytest.mark.parametrize("dead", [
        {"task_config": _CLF},
        {"head_config_overrides": {"pooling_type": "mean"}},
    ], ids=["task_config", "head_config_overrides"])
    def test_head_knobs_without_include_top_raise(self, dead):
        with pytest.raises(ValueError, match="only take effect with include_top=True"):
            ModernBERT(**_HEAD_KW, **dead)

    def test_pretrained_true_still_raises_with_a_head(self):
        with pytest.raises(NotImplementedError):
            create_modern_bert("tiny", pretrained=True, include_top=True,
                               task_config=_CLF)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
