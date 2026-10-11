"""
Comprehensive pytest test suite for the refactored BERT foundation model.

This module provides extensive testing for the decoupled BERT implementation including:
- Foundation model initialization and parameter validation.
- Architecture building and consistent output shape.
- Forward pass functionality with a standardized dictionary output.
- Model variant creation and configuration.
- Serialization and deserialization of the pure encoder.
- Error handling and edge cases.
- The in-model task head (``include_top=True``): output contract, config and
  ``.keras`` round trips, build parity, the knob validation, and reuse of one
  set of encoder weights under a different head.
- End-to-end integration testing for gradient flow and training.
"""

import pytest
import numpy as np
import keras
import tensorflow as tf
import tempfile
import os
from typing import Dict, Any

from dl_techniques.models.language.bert import BERT, create_bert
from dl_techniques.layers.heads.nlp import NLPTaskConfig, NLPTaskType

from ..knob_sensitivity_oracle import assert_structural_knob_changes_weights
from ..gradient_flow_oracle import assert_gradients_reach_every_trainable_weight


# R-038 closure -- plan-2026-08-22T035419-a11304c8 / D-251.
# Keras `ops/nn.py:907` advises that a softmax over a size-1 axis always returns
# exactly 1.0. Every site in this module feeds that axis a size of 1 ON PURPOSE
# -- single class, single token, single head, single anchor, single cluster,
# minimum sequence length -- so the advisory describes the test's own input, not
# a defect. Suppressed HERE rather than in `pyproject.toml` so an ACCIDENTAL
# size-1 softmax anywhere else still fails under `error::UserWarning`.
pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:You are using a softmax over axis:UserWarning"),
]


class TestBERTModelInitialization:
    """Test BERT model initialization and parameter validation."""

    def test_basic_initialization(self):
        """Test basic BERT model initialization as a pure encoder."""
        model = BERT(
            vocab_size=1000,
            hidden_size=256,
            num_layers=6,
            num_heads=8,
            intermediate_size=1024
        )

        assert model.hidden_size == 256
        assert model.num_layers == 6
        assert not hasattr(model, 'add_pooling_layer')
        assert not hasattr(model, 'pooler')
        assert not model.built

        # Components should be created in __init__
        assert model.embeddings is not None
        assert len(model.encoder_layers) == 6

        # But should not be built yet
        assert not model.embeddings.built
        for layer in model.encoder_layers:
            assert not layer.built

    def test_parameter_validation(self):
        """Test BERT parameter validation for invalid values."""
        with pytest.raises(ValueError, match="hidden_size.*must be divisible by num_heads"):
            BERT(vocab_size=1000, hidden_size=100, num_heads=12, num_layers=4)

        with pytest.raises(ValueError, match="hidden_size must be positive"):
            BERT(vocab_size=1000, hidden_size=-100, num_layers=4, num_heads=8)

        with pytest.raises(ValueError, match="hidden_dropout_rate must be between"):
            BERT(
                vocab_size=1000,
                hidden_size=256,
                num_layers=4,
                num_heads=8,
                hidden_dropout_rate=1.5
            )

    def test_initialization_with_custom_config(self):
        """Test BERT model initialization with custom configuration."""
        model = BERT(
            vocab_size=25000,
            hidden_size=512,
            num_layers=8,
            num_heads=8,
            intermediate_size=2048,
            hidden_dropout_rate=0.2,
            attention_probs_dropout_rate=0.1,
            normalization_type="rms_norm",
            normalization_position="pre"
        )
        assert model.vocab_size == 25000
        assert model.hidden_size == 512
        assert model.num_layers == 8
        assert model.hidden_dropout_rate == 0.2
        assert model.normalization_type == "rms_norm"


class TestBERTModelVariants:
    """Test BERT model variants and factory methods."""

    def test_bert_base_variant(self):
        model = BERT.from_variant("base")
        assert model.hidden_size == 768
        assert model.num_layers == 12
        assert model.num_heads == 12

    def test_bert_large_variant(self):
        model = BERT.from_variant("large")
        assert model.hidden_size == 1024
        assert model.num_layers == 24
        assert model.num_heads == 16

    def test_invalid_variant(self):
        with pytest.raises(ValueError, match="Unknown variant 'invalid'"):
            BERT.from_variant("invalid")

    def test_variant_with_custom_params(self):
        """Test creating variant with custom parameters."""
        model = BERT.from_variant("base", normalization_type="rms_norm")
        assert model.hidden_size == 768
        assert model.num_layers == 12
        assert model.normalization_type == "rms_norm"
        assert not hasattr(model, 'add_pooling_layer')


class TestBERTModelBuilding:
    """Test BERT model building and architecture creation."""

    @pytest.fixture
    def basic_config(self) -> Dict[str, Any]:
        return {
            "vocab_size": 1000, "hidden_size": 256, "num_layers": 4,
            "num_heads": 8, "intermediate_size": 1024, "max_position_embeddings": 128
        }

    def test_build_basic_functionality(self, basic_config):
        """Test basic building functionality and output contract."""
        model = BERT(**basic_config)
        batch_size, seq_length = 2, 32
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

    def test_transformer_layers_configuration(self, basic_config):
        model = BERT(**basic_config)
        input_ids = keras.ops.cast(
            keras.random.uniform((1, 16), maxval=basic_config['vocab_size']), dtype='int32'
        )
        _ = model(input_ids, training=False)
        for i, layer in enumerate(model.encoder_layers):
            assert layer.hidden_size == basic_config['hidden_size']
            assert layer.num_heads == basic_config['num_heads']
            assert layer.name == f"encoder_layer_{i}"


class TestBERTModelForwardPass:
    """Test BERT model forward pass functionality."""

    @pytest.fixture
    def built_model(self) -> BERT:
        """Create a built BERT model for testing."""
        model = BERT(
            vocab_size=1000, hidden_size=256, num_layers=3, num_heads=8,
            intermediate_size=1024, max_position_embeddings=128
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

    def test_forward_pass_with_attention_mask(self, built_model):
        batch_size, seq_length = 2, 16
        input_ids = keras.ops.cast(
            keras.random.uniform((batch_size, seq_length), maxval=built_model.vocab_size),
            dtype='int32'
        )
        attention_mask = keras.ops.ones((batch_size, seq_length), dtype='int32')
        attention_mask = keras.ops.slice_update(
            attention_mask, [0, seq_length - 4], keras.ops.zeros((batch_size, 4), dtype='int32')
        )
        outputs = built_model(input_ids, attention_mask=attention_mask, training=False)
        assert outputs["last_hidden_state"].shape == (
            batch_size, seq_length, built_model.hidden_size
        )

    def test_invalid_dict_input(self, built_model):
        inputs = {'invalid_key': keras.ops.ones((2, 16), dtype='int32')}
        with pytest.raises(ValueError, match="Dictionary input must contain 'input_ids' key"):
            built_model(inputs, training=False)


class TestBERTModelSerialization:
    """Test BERT model serialization and deserialization."""

    def test_config_serialization(self):
        model = BERT(
            vocab_size=25000, hidden_size=512, num_layers=8, num_heads=8,
            normalization_type="rms_norm"
        )
        model_config = model.get_config()
        assert model_config['vocab_size'] == 25000
        assert model_config['hidden_size'] == 512
        assert not hasattr(model_config, 'add_pooling_layer')

    def test_model_from_config(self):
        original_model = BERT(
            vocab_size=1000, hidden_size=256, num_layers=4,
            num_heads=8, normalization_type="rms_norm"
        )
        config = original_model.get_config()
        new_model = BERT.from_config(config)
        assert new_model.hidden_size == original_model.hidden_size
        assert new_model.normalization_type == original_model.normalization_type

    def test_model_save_load(self):
        model = BERT(vocab_size=1000, hidden_size=256, num_layers=2, num_heads=8)
        input_ids = keras.ops.cast(keras.random.uniform((2, 16), maxval=1000), dtype='int32')
        original_outputs = model(input_ids, training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_bert.keras')
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)
            loaded_outputs = loaded_model(input_ids, training=False)

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_outputs['last_hidden_state']),
                keras.ops.convert_to_numpy(loaded_outputs['last_hidden_state']),
                rtol=1e-5, atol=1e-6, err_msg="Hidden states should match"
            )


class TestBERTEdgeCases:
    """Test BERT model edge cases."""

    def test_minimum_sequence_length(self):
        model = BERT(vocab_size=1000, hidden_size=128, num_layers=2, num_heads=8)
        input_ids = keras.ops.cast([[42]], dtype='int32')
        outputs = model(input_ids, training=False)
        assert outputs['last_hidden_state'].shape == (1, 1, 128)

    def test_long_sequence(self):
        model = BERT(
            vocab_size=1000, hidden_size=256, num_layers=2, num_heads=8,
            max_position_embeddings=128
        )
        input_ids = keras.ops.cast(
            keras.random.uniform((1, 128), maxval=1000), dtype='int32'
        )
        outputs = model(input_ids, training=False)
        assert outputs['last_hidden_state'].shape == (1, 128, 256)


class TestBERTIntegrationFactory:
    """``create_bert(..., include_top=True, task_config=...)`` builds a headed BERT."""

    def test_create_with_classification_head(self):
        """Test creating a BERT model for sequence classification."""
        task_config = NLPTaskConfig(
            name="sentiment", task_type=NLPTaskType.SENTIMENT_ANALYSIS, num_classes=3
        )
        model = create_bert("tiny", include_top=True, task_config=task_config)
        assert isinstance(model, BERT)

        inputs = {
            'input_ids': keras.ops.cast(keras.random.uniform((2, 32), maxval=1000), 'int32'),
            'attention_mask': keras.ops.ones((2, 32), 'int32'),
            'token_type_ids': keras.ops.zeros((2, 32), 'int32'),
        }
        outputs = model(inputs, training=False)
        # The model drops the head's `probabilities` entry -- it is exactly
        # `softmax(logits)` -- and exposes the one surviving tensor bare. That
        # is what makes the model compilable; see
        # `test_the_head_model_output_is_compilable.py`, which owns the full
        # contract. These assertions pinned the dict-output defect until
        # 2026-08-24.
        assert not isinstance(outputs, dict)
        assert outputs.shape == (2, 3)

    def test_create_with_token_classification_head(self):
        """Test creating a BERT model for token classification."""
        task_config = NLPTaskConfig(
            name="ner", task_type=NLPTaskType.NAMED_ENTITY_RECOGNITION, num_classes=9
        )
        model = create_bert("tiny", include_top=True, task_config=task_config)
        assert isinstance(model, BERT)

        inputs = {
            'input_ids': keras.ops.cast(keras.random.uniform((2, 32), maxval=1000), 'int32'),
            'attention_mask': keras.ops.ones((2, 32), 'int32'),
            'token_type_ids': keras.ops.zeros((2, 32), 'int32'),
        }
        outputs = model(inputs, training=False)
        # `TokenClassificationHead` emits `{'logits'}` alone, so the single
        # surviving key is likewise exposed bare.
        assert not isinstance(outputs, dict)
        assert outputs.shape == (2, 32, 9)


class TestBERTIntegration:
    """Integration tests for the complete model (encoder + head)."""

    @pytest.fixture
    def classification_model(self) -> keras.Model:
        """Create a classification model for integration tests."""
        task_config = NLPTaskConfig(
            name="clf", task_type=NLPTaskType.TEXT_CLASSIFICATION, num_classes=3
        )
        model = create_bert(
            "tiny",
            include_top=True,
            task_config=task_config,
            hidden_size=64,
            intermediate_size=256,
        )
        # A subclassed model owns no weights until built; the functional
        # wrapper this replaced was built at construction.
        model.build((None, 16))
        return model

    def test_gradient_flow_integration(self, classification_model):
        """Test that gradients flow through the entire integrated model.

        The per-weight assertions live in
        ``tests/test_models/gradient_flow_oracle.py`` (this test's body was the
        pattern they were extracted from; its own RED proofs are in
        ``tests/test_models/test_gradient_flow_oracle.py``). What stays here is
        the part that is BERT-specific: the real labelled loss.
        """
        batch_size, seq_length = 2, 16
        inputs = {
            'input_ids': keras.ops.cast(keras.random.uniform((batch_size, seq_length), maxval=1000), 'int32'),
            'attention_mask': keras.ops.ones((batch_size, seq_length), 'int32'),
            'token_type_ids': keras.ops.zeros((batch_size, seq_length), 'int32'),
        }
        targets = keras.ops.one_hot(keras.ops.array([0, 2]), 3)

        # `from_logits=True` is REQUIRED here and was missing. With the
        # default `False`, Keras treats these logits as probabilities:
        # it renormalizes by their sum and CLIPS to [eps, 1-eps], and for
        # some input draws every element lands in the clipped region, where
        # the loss is constant and EVERY gradient is exactly 0.0. MEASURED
        # 2026-08-18: 61 of 61 trainable weights had identically-zero
        # gradients on one such draw, which is why this test was
        # order-dependent -- importing a sibling test module shifts the RNG
        # stream and changes the ids. The old `all(norm >= 0.0)` assertion
        # reported green either way; the oracle below convicts it by name.
        def loss_fn(outputs):
            return keras.ops.mean(
                keras.losses.categorical_crossentropy(
                    targets, outputs, from_logits=True
                )
            )

        report = assert_gradients_reach_every_trainable_weight(
            classification_model, inputs, loss_fn=loss_fn
        )
        # Anti-vacuity: the oracle raises on an empty weight set, but pin the
        # count here too so a future refactor that hands it a sub-model instead
        # of the full classification stack is visible.
        assert len(report) == len(classification_model.trainable_weights)
        assert len(report) > 0

    def test_training_integration(self, classification_model):
        """A 100-step loop on a fixed batch must drive the loss down.

        The original body ran the same 100 steps, assigned ``initial_loss`` and
        then asserted NOTHING -- it could not fail for any model, including one
        whose gradients were all zero. It also used ``learning_rate=1e-6``,
        which over 100 steps on a tiny model cannot move the loss measurably;
        that reads as a decrease assertion that was written, found flaky, and
        deleted rather than fixed.

        The bar is derived from the trivial baseline, not chosen: after 100 Adam
        steps at 1e-3 on ONE fixed batch of 4 examples, the model must be far
        better than the constant predictor that outputs the uniform
        distribution, whose loss is ``ln(3) = 1.0986``. MEASURED over 4
        independent runs of exactly this body: initial 1.1041 / 1.0639 /
        1.0646 / 1.0858 (i.e. the uniform value, as random logits should give)
        and final 0.0334 / 0.0363 / 0.0371 / 0.0272 -- the batch is memorised,
        a ~30x drop. The bar below is a QUARTER of the uniform baseline
        (0.2747), roughly 7x above the worst measurement.

        NOTE the ``from_logits=True`` above: without it this loop's gradients
        can be identically zero. See the comment at ``test_gradient_flow_integration``.
        """
        optimizer = keras.optimizers.Adam(learning_rate=1e-3)
        batch_size, seq_length = 4, 16
        inputs = {
            'input_ids': keras.ops.cast(keras.random.uniform((batch_size, seq_length), maxval=1000), 'int32'),
            'attention_mask': keras.ops.ones((batch_size, seq_length), 'int32'),
            'token_type_ids': keras.ops.zeros((batch_size, seq_length), 'int32'),
        }
        labels = keras.ops.cast(keras.random.uniform((batch_size,), maxval=3), 'int32')

        initial_loss = None
        loss_value = None
        for step in range(100):
            with tf.GradientTape() as tape:
                outputs = classification_model(inputs, training=True)
                loss = keras.ops.mean(
                    keras.losses.sparse_categorical_crossentropy(
                        labels, outputs, from_logits=True
                    )
                )
            loss_value = float(loss)
            assert np.isfinite(loss_value), f"loss became {loss_value} at step {step}"
            if initial_loss is None:
                initial_loss = loss_value
            gradients = tape.gradient(loss, classification_model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, classification_model.trainable_weights))

        uniform_baseline = float(np.log(3.0))
        assert loss_value < initial_loss, (
            f"100 Adam steps at 1e-3 on a single fixed batch did not reduce the "
            f"loss at all: {initial_loss:.4f} -> {loss_value:.4f}."
        )
        assert loss_value < 0.25 * uniform_baseline, (
            f"after 100 steps the loss is {loss_value:.4f}; the constant "
            f"uniform predictor scores {uniform_baseline:.4f} and a model that "
            f"memorises this 4-example batch measures ~0.03. Initial was "
            f"{initial_loss:.4f}."
        )


class TestBERTAdvancedFeatures:
    """Test advanced BERT features from dl-techniques framework."""

    def test_different_normalization_types(self):
        """``normalization_type`` must reach the normalization layers.

        ``last_hidden_state`` is ``(2, 16, 128)`` under both settings, so the
        old assertion held whether or not the kwarg was honoured. RMSNorm has no
        centering shift, so it holds strictly FEWER weights than LayerNorm --
        a structural difference the weight-shape signature pins directly. (An
        output-difference check would not: different signatures consume
        different RNG draws, so the outputs differ either way.)
        """
        base_config = {"vocab_size": 1000, "hidden_size": 128, "num_layers": 2, "num_heads": 8}

        def _build(norm_type):
            model = BERT(**base_config, normalization_type=norm_type)
            model(keras.ops.zeros((1, 16), dtype='int32'), training=False)
            return model

        builders = {n: (lambda n=n: _build(n)) for n in ["layer_norm", "rms_norm"]}
        sigs = assert_structural_knob_changes_weights(
            builders, knob="normalization_type")
        assert len(sigs["rms_norm"]) < len(sigs["layer_norm"]), (
            "rms_norm should drop one weight per norm layer (no beta), but held "
            f"{len(sigs['rms_norm'])} vs layer_norm's {len(sigs['layer_norm'])}"
        )

        for norm_type in ["layer_norm", "rms_norm"]:
            model = BERT(**base_config, normalization_type=norm_type)
            input_ids = keras.ops.cast(keras.random.uniform((2, 16), maxval=1000), 'int32')
            output = model(input_ids, training=False)
            assert output['last_hidden_state'].shape == (2, 16, 128)

    def test_stochastic_depth(self):
        model = BERT(
            vocab_size=1000, hidden_size=256, num_layers=4, num_heads=8,
            use_stochastic_depth=True, stochastic_depth_rate=0.1
        )
        input_ids = keras.ops.cast(keras.random.uniform((2, 16), maxval=1000), 'int32')
        output_train = model(input_ids, training=True)
        output_inference = model(input_ids, training=False)
        assert output_train['last_hidden_state'].shape == (2, 16, 256)
        assert output_inference['last_hidden_state'].shape == (2, 16, 256)

    def test_model_summary(self):
        """Test that model summary works without errors."""
        model = BERT.from_variant("tiny")
        input_ids = keras.ops.cast(keras.random.uniform((1, 16), maxval=1000), 'int32')
        _ = model(input_ids, training=False)
        try:
            model.summary()
        except Exception as e:
            pytest.fail(f"Model summary raised an exception: {e}")


class TestBERTIter1Refactor:
    """Lock-in tests for the iter-1 bert refactor (plan_2026-05-11_9357982a).

    Mirrors the ``TestTreeTransformerIter1Refactor`` pattern from
    ``tests/test_models/test_tree_transformer/test_model.py``.
    """

    def test_create_bert_factory(self):
        """``create_bert`` returns a configured ``BERT`` and runs a forward pass."""
        from dl_techniques.models.language.bert.model import create_bert

        model = create_bert("tiny", vocab_size=200)
        assert isinstance(model, BERT)
        assert model.vocab_size == 200

        batch_size, seq_len = 2, 8
        input_ids = keras.ops.cast(
            keras.random.uniform((batch_size, seq_len), maxval=200), "int32"
        )
        outputs = model(input_ids, training=False)
        assert "last_hidden_state" in outputs
        assert outputs["last_hidden_state"].shape == (
            batch_size, seq_len, model.hidden_size,
        )

    def test_from_variant_pretrained_true_raises(self):
        """``BERT.from_variant(..., pretrained=True)`` must raise ``NotImplementedError``.

        No public BERT weights are distributed by this library; the
        placeholder URL flow was silently random-initialising before the
        iter-1 refactor (see plan_2026-05-11_9357982a D-001).
        """
        with pytest.raises(NotImplementedError):
            BERT.from_variant("tiny", pretrained=True)

    def test_public_api_surface(self):
        """The bert package public API is exactly ``{BERT, create_bert}``.

        The head factory that used to be the third name was replaced by
        ``include_top`` on the class (plan-2026-10-11T004932-b83935e1).
        """
        import dl_techniques.models.language.bert as pkg

        assert set(pkg.__all__) == {"BERT", "create_bert"}
        assert not hasattr(pkg, "create_bert_with_head")
        assert not hasattr(pkg, "create_nlp_head"), (
            "create_nlp_head must no longer be re-exported from the bert "
            "package; import it from dl_techniques.layers.heads.nlp instead."
        )

        # BERT must remain importable from the implementation submodule path
        # (renamed .bert -> .model by commit 2c4a0ca7c; existing tests rely on this).
        from dl_techniques.models.language.bert.model import BERT as BERTViaSubmodule
        assert BERTViaSubmodule is pkg.BERT



# ---------------------------------------------------------------------
# include_top: the in-model task head
# ---------------------------------------------------------------------

#: Small enough that every arm below runs in about a second.
_HEAD_KW: Dict[str, Any] = {
    "vocab_size": 64,
    "hidden_size": 32,
    "num_layers": 2,
    "num_heads": 2,
    "intermediate_size": 64,
    "max_position_embeddings": 32,
}
_SEQ = 8
_CLF = NLPTaskConfig(
    name="clf", task_type=NLPTaskType.TEXT_CLASSIFICATION, num_classes=3
)
_NER = NLPTaskConfig(
    name="ner", task_type=NLPTaskType.TOKEN_CLASSIFICATION, num_classes=9
)


def _head_batch() -> Dict[str, np.ndarray]:
    ids = np.random.default_rng(0).integers(1, 64, (2, _SEQ)).astype("int32")
    mask = np.ones_like(ids)
    mask[1, 5:] = 0
    return {"input_ids": ids, "attention_mask": mask}


def _np(x) -> np.ndarray:
    return keras.ops.convert_to_numpy(x)


def _relative_paths(model: keras.Model) -> list:
    """Weight paths without the model-root segment (guide 8.3)."""
    return sorted(w.path.split("/", 1)[-1] for w in model.weights)


def _backbone_hidden(model: BERT, batch: Dict[str, np.ndarray]) -> np.ndarray:
    """``last_hidden_state`` of ``model``'s encoder, whatever its head.

    A headed model does not return its hidden states, so the encoder is read
    through a headless twin rebuilt from the model's own config and given the
    model's own encoder weights, layer by layer.
    """
    config = model.get_config()
    config.update(include_top=False, task_config=None, head_config_overrides=None)
    config.pop("name", None)
    twin = BERT.from_config(config)
    twin(batch, training=False)
    for layer in twin.layers:
        layer.set_weights(model.get_layer(layer.name).get_weights())
    return _np(twin(batch, training=False)["last_hidden_state"])


class TestBERTIncludeTop:
    """The ``include_top`` / ``task_config`` / ``head_config_overrides`` knobs."""

    def test_head_output_shapes(self):
        """Bare logits for single-output heads, a dict for QA."""
        batch = _head_batch()
        clf = BERT(**_HEAD_KW, include_top=True, task_config=_CLF)
        out = clf(batch, training=False)
        assert not isinstance(out, dict)
        assert tuple(out.shape) == (2, 3)

        ner = BERT(**_HEAD_KW, include_top=True, task_config=_NER)
        out = ner(batch, training=False)
        assert not isinstance(out, dict)
        assert tuple(out.shape) == (2, _SEQ, 9)

        qa = BERT(**_HEAD_KW, include_top=True, task_config=NLPTaskConfig(
            name="qa", task_type=NLPTaskType.QUESTION_ANSWERING, num_classes=2))
        out = qa(batch, training=False)
        assert set(out) == {"start_logits", "end_logits"}
        assert tuple(out["start_logits"].shape) == (2, _SEQ)

    def test_symbolic_output_matches_the_eager_one(self):
        """``compute_output_shape`` branches with ``call`` (Keras builds from it)."""
        model = BERT(**_HEAD_KW, include_top=True, task_config=_NER)
        symbolic = model(keras.Input(shape=(None,), dtype="int32"))
        assert tuple(symbolic.shape) == (None, None, 9)
        headless = BERT(**_HEAD_KW)
        symbolic = headless(keras.Input(shape=(None,), dtype="int32"))
        assert set(symbolic) == {"last_hidden_state", "attention_mask"}

    def test_config_round_trip_with_a_head(self):
        """Every knob is serialized; ``task_config`` as a plain JSON dict."""
        import json

        overrides = {"pooling_type": "mean"}
        model = BERT(**_HEAD_KW, include_top=True, task_config=_CLF,
                     head_config_overrides=overrides)
        config = model.get_config()
        assert config["include_top"] is True
        assert config["head_config_overrides"] == overrides
        assert isinstance(config["task_config"], dict)
        assert config["task_config"]["task_type"] == "text_classification"
        json.dumps(config)  # plain data, no enum or dataclass inside

        # A dict task_config and the dataclass serialize identically.
        from_dict = BERT(**_HEAD_KW, include_top=True,
                         task_config=dict(config["task_config"]),
                         head_config_overrides=overrides)
        assert from_dict.get_config()["task_config"] == config["task_config"]

        rebuilt = BERT.from_config(config)
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
        config = BERT(**_HEAD_KW).get_config()
        for key in ("include_top", "task_config", "head_config_overrides"):
            config.pop(key)
        model = BERT.from_config(config)
        assert model.include_top is False and model.task_head is None

    def test_keras_round_trip_restores_the_head_by_value(self, tmp_path):
        """Weights compared BEFORE the first call on the loaded model (guide 8.4)."""
        model = BERT(**_HEAD_KW, include_top=True, task_config=_CLF)
        batch = _head_batch()
        before = _np(model(batch, training=False))
        saved = [_np(w).copy() for w in model.weights]
        head_saved = [_np(w).copy() for w in model.task_head.weights]
        assert head_saved, "the head has no weights to compare"

        path = str(tmp_path / "bert_head.keras")
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
        explicit = BERT(**kw)
        explicit.build((None, _SEQ))
        lazy = BERT(**kw)
        lazy(_head_batch(), training=False)
        assert _relative_paths(explicit) == _relative_paths(lazy)
        assert explicit.count_params() == lazy.count_params() > 0

    def test_a_disabled_head_builds_no_head_weights(self):
        """Anti-vacuity sibling of the parity arm (guide 8.3)."""
        headless = BERT(**_HEAD_KW)
        headless(_head_batch(), training=False)
        assert headless.task_head is None
        assert not [w.path for w in headless.weights if "task_head" in w.path]

        headed = BERT(**_HEAD_KW, include_top=True, task_config=_CLF)
        headed(_head_batch(), training=False)
        assert [w.path for w in headed.weights if "/task_head/" in w.path]

    def test_the_encoder_weight_paths_do_not_depend_on_the_head(self):
        """The head lives entirely under ``task_head/`` (D-005 of the plan)."""
        def paths(**kw):
            m = BERT(**_HEAD_KW, name="bert", **kw)
            m(_head_batch(), training=False)
            return [w.path for w in m.weights]

        headless = paths()
        for task in (_CLF, _NER):
            headed = paths(include_top=True, task_config=task)
            encoder_part = [p for p in headed if not p.startswith("bert/task_head/")]
            assert encoder_part == headless, task.name
            assert len(headed) > len(headless)

    def test_backbone_weights_transfer_across_heads(self, tmp_path):
        """An encoder trained under one head is reused, same weights, under another.

        The supported route is a local ``pretrained`` path: ``load_pretrained_weights``
        copies by layer name and never copies the checkpoint's ``task_head``
        (D-007 of plan-2026-10-11T004932-b83935e1). Exercised here into a
        TOKEN_CLASSIFICATION model and into a headless one.
        """
        source = create_bert("tiny", include_top=True, task_config=_CLF, **_HEAD_KW)
        batch = _head_batch()
        source(batch, training=False)
        rng = np.random.default_rng(1)
        for w in source.weights:
            w.assign(rng.normal(0.0, 0.1, size=tuple(w.shape)).astype(w.dtype))
        path = str(tmp_path / "trained_under_clf.keras")
        source.save(path)
        reference = _backbone_hidden(source, batch)

        token = create_bert("tiny", include_top=True, task_config=_NER,
                            pretrained=path, **_HEAD_KW)
        headless = create_bert("tiny", pretrained=path, **_HEAD_KW)

        np.testing.assert_array_equal(_backbone_hidden(token, batch), reference)
        np.testing.assert_array_equal(
            _np(headless(batch, training=False)["last_hidden_state"]), reference)
        # The new head is the target's own: fresh, and of the new task's shape.
        assert tuple(token(batch, training=False).shape) == (2, _SEQ, 9)

        # Anti-vacuity: a fresh encoder does NOT reproduce the reference.
        fresh = create_bert("tiny", **_HEAD_KW)
        assert not np.array_equal(
            _np(fresh(batch, training=False)["last_hidden_state"]), reference)

    def test_pretrained_headless_checkpoint_into_a_headed_model(self, tmp_path):
        """A headless checkpoint restores the encoder of a headed model by value."""
        encoder = create_bert("tiny", **_HEAD_KW)
        batch = _head_batch()
        expected = _np(encoder(batch, training=False)["last_hidden_state"])
        path = str(tmp_path / "encoder.keras")
        encoder.save(path)

        headed = create_bert("tiny", include_top=True, task_config=_CLF,
                             pretrained=path, **_HEAD_KW)
        np.testing.assert_array_equal(_backbone_hidden(headed, batch), expected)

    def test_include_top_without_task_config_raises(self):
        with pytest.raises(ValueError, match="include_top=True needs a task_config"):
            BERT(**_HEAD_KW, include_top=True)

    @pytest.mark.parametrize("dead", [
        {"task_config": _CLF},
        {"head_config_overrides": {"pooling_type": "mean"}},
    ], ids=["task_config", "head_config_overrides"])
    def test_head_knobs_without_include_top_raise(self, dead):
        with pytest.raises(ValueError, match="only take effect with include_top=True"):
            BERT(**_HEAD_KW, **dead)

    def test_pretrained_true_still_raises_with_a_head(self):
        with pytest.raises(NotImplementedError):
            create_bert("tiny", pretrained=True, include_top=True, task_config=_CLF)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])