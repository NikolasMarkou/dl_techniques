"""
Comprehensive test suite for the FNet model implementation.

This test suite validates the FNet foundation model, its in-model task head
(``include_top=True``) and the ``create_fnet`` factory, using small memory
footprint configurations suitable for CPU testing.
"""

import pytest
import tempfile
import os
from typing import Any, Dict

import numpy as np
import keras
import tensorflow as tf

# Imports from the project structure, assuming it's available in the path
from dl_techniques.models.language.fnet.model import (
    FNet,
    create_fnet,
)
from dl_techniques.layers.heads.nlp import NLPTaskConfig, NLPTaskType
from dl_techniques.layers.regularization.stochastic_depth import StochasticDepth
from dl_techniques.utils.drop_path import linear_drop_path_rates


class TestFNetModel:
    """Test suite for the main FNet foundation model class."""

    @pytest.fixture
    def small_config(self) -> Dict[str, Any]:
        """Small configuration for CPU testing."""
        return {
            'vocab_size': 100,
            'hidden_size': 32,
            'num_layers': 2,
            'intermediate_size': 64,
            'max_position_embeddings': 16,
            'hidden_dropout_rate': 0.1,
        }

    @pytest.fixture
    def sample_inputs(self) -> Dict[str, keras.KerasTensor]:
        """Sample token inputs for testing."""
        batch_size = 2
        seq_len = 16
        return {
            'input_ids': keras.random.randint(minval=0, maxval=100, shape=(batch_size, seq_len)),
            'attention_mask': keras.ops.ones((batch_size, seq_len), dtype="int32"),
            'token_type_ids': keras.ops.zeros((batch_size, seq_len), dtype="int32"),
            'position_ids': keras.ops.repeat(keras.ops.arange(seq_len)[None, :], repeats=batch_size, axis=0)
        }

    def test_basic_model_creation(self, small_config):
        """Test basic FNet model creation."""
        model = FNet(**small_config)
        assert isinstance(model, keras.Model)
        assert model.vocab_size == 100
        assert model.hidden_size == 32
        assert model.num_layers == 2

    def test_model_forward_pass_dict_input(self, small_config, sample_inputs):
        """Test model forward pass with dictionary input."""
        model = FNet(**small_config)
        output = model(sample_inputs)

        # Output should be a dictionary with specific keys
        assert isinstance(output, dict)
        assert 'last_hidden_state' in output
        assert 'attention_mask' in output

        # Check shapes
        assert output['last_hidden_state'].shape == (2, 16, 32)
        assert output['attention_mask'].shape == (2, 16)

    def test_model_forward_pass_tensor_input(self, small_config):
        """Test model forward pass with a single tensor input."""
        model = FNet(**small_config)
        input_ids = keras.random.randint(minval=0, maxval=100, shape=(2, 16))
        output = model(input_ids)

        assert isinstance(output, dict)
        assert 'last_hidden_state' in output
        assert output['last_hidden_state'].shape == (2, 16, 32)

    def test_from_variant_method(self):
        """Test creating model from predefined variants."""
        variants = ['base', 'large', 'small', 'tiny']
        for variant in variants:
            model = FNet.from_variant(variant)
            assert isinstance(model, FNet)

            # Check variant-specific configurations
            expected_config = FNet.MODEL_VARIANTS[variant]
            assert model.hidden_size == expected_config['hidden_size']
            assert model.num_layers == expected_config['num_layers']

    def test_from_variant_with_overrides(self):
        """Test from_variant with configuration overrides."""
        model = FNet.from_variant(
            'tiny',
            hidden_dropout_rate=0.2,
            normalization_type='rms_norm'
        )

        assert model.hidden_dropout_rate == 0.2
        assert model.normalization_type == 'rms_norm'

    def test_model_serialization(self, small_config, sample_inputs):
        """Test model serialization and loading."""
        model = FNet(**small_config)
        original_pred = model(sample_inputs)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_fnet.keras')
            model.save(filepath)
            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_inputs)

            # Compare the main output tensor
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred["last_hidden_state"]),
                keras.ops.convert_to_numpy(loaded_pred["last_hidden_state"]),
                rtol=1e-6, atol=1e-6,
                err_msg="Loaded model outputs should match original"
            )

    def test_get_config_and_from_config(self, small_config):
        """Test configuration serialization."""
        model = FNet(**small_config)
        config = model.get_config()

        # Verify all expected keys are present
        expected_keys = {
            'vocab_size', 'hidden_size', 'num_layers', 'intermediate_size',
            'hidden_dropout_rate', 'max_position_embeddings'
        }
        assert expected_keys.issubset(set(config.keys()))

        # Test reconstruction
        new_model = FNet.from_config(config)
        assert new_model.vocab_size == model.vocab_size
        assert new_model.hidden_size == model.hidden_size
        assert new_model.num_layers == model.num_layers

    def test_invalid_variant(self):
        """Test error handling for invalid variant."""
        with pytest.raises(ValueError, match="Unknown variant"):
            FNet.from_variant('invalid_variant')

    def test_invalid_input_dict(self, small_config):
        """Test error handling for invalid input dictionary."""
        model = FNet(**small_config)
        with pytest.raises(ValueError, match="Dictionary input must contain 'input_ids' key"):
            model({'attention_mask': keras.ops.ones((2, 16))})

    def test_gradients_flow(self, small_config, sample_inputs):
        """Test that gradients flow through the model."""
        model = FNet(**small_config)

        with tf.GradientTape() as tape:
            output = model(sample_inputs)
            # Calculate loss based on the main output tensor
            loss = keras.ops.mean(keras.ops.square(output['last_hidden_state']))

        gradients = tape.gradient(loss, model.trainable_variables)

        # Check that all gradients are not None
        assert all(g is not None for g in gradients)

        # Check that at least some gradients have meaningful magnitudes
        grad_norms = [keras.ops.convert_to_numpy(keras.ops.norm(g)) for g in gradients if g is not None]
        assert any(norm > 1e-8 for norm in grad_norms)

    def test_model_with_different_input_lengths(self):
        """Test model with different sequence lengths by creating new instances."""
        base_config = {
            'vocab_size': 50,
            'hidden_size': 16,
            'num_layers': 1,
            'intermediate_size': 32,
        }

        for seq_len in [8, 16, 32]:
            # Create a new model instance for each sequence length
            config = base_config.copy()
            config['max_position_embeddings'] = seq_len
            model = FNet(**config)

            input_ids = keras.random.randint(minval=0, maxval=50, shape=(1, seq_len))
            output = model(input_ids)
            assert output['last_hidden_state'].shape == (1, seq_len, 16)

    def test_model_summary(self, small_config):
        """Test model summary functionality."""
        model = FNet(**small_config)
        # Build the model by calling it on some data
        input_ids = keras.random.randint(minval=0, maxval=100, shape=(1, 16))
        model(input_ids)
        # This should not raise any errors
        model.summary()


def _fnet_block_reference(block, x):
    """The pre-drop-path `FNetEncoderBlock.call` body, transcribed from 2e10c29bb.

    An ORACLE, not a re-implementation of the current code: the formula as it
    stood BEFORE stochastic depth was added, driven through the block's own
    already-built sublayers. An injection inside `FNetEncoderBlock.call()` or in
    FNet's block-construction loop moves the MODEL side only, never this side.
    (Deliberately duplicated from tests/test_layers/test_fnet_encoder_block.py:
    test oracles are not shared across suites, and this plan adds no new files.)
    """
    fourier_output = block.fourier_transform(x, training=False)
    fourier_output = block.fourier_layer_norm(x + fourier_output, training=False)
    ffn_output = block.ffn_layer(fourier_output, training=False)
    return block.output_layer_norm(fourier_output + ffn_output, training=False)


class TestFNetStochasticDepth:
    """The `use_stochastic_depth` / `stochastic_depth_rate` knob reaches blocks."""

    @pytest.fixture
    def small_config(self) -> Dict[str, Any]:
        """Small configuration with enough layers for a visible schedule."""
        return {
            'vocab_size': 100,
            'hidden_size': 32,
            'num_layers': 4,
            'intermediate_size': 64,
            'max_position_embeddings': 16,
            'hidden_dropout_rate': 0.0,
        }

    @pytest.fixture
    def sample_inputs(self) -> Dict[str, keras.KerasTensor]:
        """Fixed token inputs (seeded) so identity probes are repeatable."""
        return {
            'input_ids': keras.random.randint(
                minval=0, maxval=100, shape=(3, 16), seed=11
            ),
            'attention_mask': keras.ops.ones((3, 16), dtype="int32"),
        }

    def test_default_is_inert_and_bit_identical(self, small_config, sample_inputs):
        """Default `use_stochastic_depth=False` builds no drop-path at all.

        The model's output is compared against the pre-drop-path forward pass
        replayed through the model's OWN embeddings and block sublayers, so both
        sides share weights by construction. In-process (dodging cross-process
        GPU kernel nondeterminism) at explicit `training=False`; EXACT zero.
        """
        model = FNet(**small_config)
        out = model(sample_inputs, training=False)

        for block in model.encoder_layers:
            assert block.use_stochastic_depth is False
            assert block.fourier_stochastic_depth is None
            assert block.ffn_stochastic_depth is None

        ref = model.embeddings(
            input_ids=sample_inputs['input_ids'],
            token_type_ids=None,
            position_ids=None,
            training=False,
        )
        for block in model.encoder_layers:
            ref = _fnet_block_reference(block, ref)

        delta = float(
            keras.ops.max(keras.ops.abs(out['last_hidden_state'] - ref))
        )
        assert delta == 0.0, f"expected exact bit-identity, got max|delta|={delta!r}"

    def test_schedule_reaches_every_block(self, small_config, sample_inputs):
        """Each block carries two distinct StochasticDepth at its schedule rate.

        The rate is deliberately NON-ZERO (0.3): an all-zeros schedule is
        invariant under both shift and reversal, so a zero-rate wiring proof
        cannot distinguish a correct schedule from a reversed or constant one.
        """
        model = FNet(
            use_stochastic_depth=True, stochastic_depth_rate=0.3, **small_config
        )
        _ = model(sample_inputs, training=False)

        expected = linear_drop_path_rates(
            num_blocks=small_config['num_layers'], max_rate=0.3
        )
        # Guard the guard: the schedule itself must not be degenerate.
        assert len(set(expected)) == len(expected)
        assert max(expected) > 0.0

        assert len(model.encoder_layers) == len(expected)
        for i, (block, rate) in enumerate(zip(model.encoder_layers, expected)):
            assert isinstance(block.fourier_stochastic_depth, StochasticDepth), i
            assert isinstance(block.ffn_stochastic_depth, StochasticDepth), i
            assert (
                block.fourier_stochastic_depth is not block.ffn_stochastic_depth
            ), f"block {i} shares one StochasticDepth across both branches"
            assert block.fourier_stochastic_depth.drop_path_rate == rate, (
                f"block {i}: fourier branch rate "
                f"{block.fourier_stochastic_depth.drop_path_rate} != {rate}"
            )
            assert block.ffn_stochastic_depth.drop_path_rate == rate, (
                f"block {i}: ffn branch rate "
                f"{block.ffn_stochastic_depth.drop_path_rate} != {rate}"
            )

    def test_stochastic_depth_config_round_trip(self, small_config, sample_inputs):
        """Both keys survive FNet.get_config -> from_config and re-wire blocks."""
        model = FNet(
            use_stochastic_depth=True, stochastic_depth_rate=0.2, **small_config
        )
        config = model.get_config()
        assert config['use_stochastic_depth'] is True
        assert config['stochastic_depth_rate'] == 0.2

        restored = FNet.from_config(config)
        assert restored.use_stochastic_depth is True
        assert restored.stochastic_depth_rate == 0.2

        expected = linear_drop_path_rates(
            num_blocks=small_config['num_layers'], max_rate=0.2
        )
        for i, (block, rate) in enumerate(zip(restored.encoder_layers, expected)):
            assert block.fourier_stochastic_depth is not None, i
            assert block.ffn_stochastic_depth is not None, i
            assert block.fourier_stochastic_depth.drop_path_rate == rate, i
            assert block.ffn_stochastic_depth.drop_path_rate == rate, i

        _ = restored(sample_inputs, training=False)


class TestFNetWithHeadFactory:
    """``create_fnet(..., include_top=True)``: the in-model task head.

    The functional ``create_fnet_with_head`` this replaced returned the head's
    raw dict and took a ``sequence_length`` for its ``keras.Input``; the
    in-model head returns the finalized output and takes its length from the
    first call, as the headless encoder always has.
    """

    @pytest.fixture
    def sample_inputs(self) -> Dict[str, keras.KerasTensor]:
        """Sample full inputs for an end-to-end model."""
        batch_size = 2
        seq_len = 16
        return {
            'input_ids': keras.random.randint(minval=0, maxval=100, shape=(batch_size, seq_len)),
            'attention_mask': keras.ops.ones((batch_size, seq_len), dtype="int32"),
            'token_type_ids': keras.ops.zeros((batch_size, seq_len), dtype="int32"),
        }

    def test_create_with_sentiment_head(self, sample_inputs):
        """Sequence classification: bare logits, ``probabilities`` dropped."""
        task_config = NLPTaskConfig(
            name="sentiment",
            task_type=NLPTaskType.SENTIMENT_ANALYSIS,
            num_classes=3
        )
        model = create_fnet(
            "tiny", include_top=True, task_config=task_config,
            max_position_embeddings=16, vocab_size=100,
        )
        assert isinstance(model, FNet)

        output = model(sample_inputs)
        assert not isinstance(output, dict)
        assert output.shape == (2, 3)

    def test_create_with_ner_head(self, sample_inputs):
        """Token classification: bare logits, ``predictions`` dropped."""
        task_config = NLPTaskConfig(
            name="ner",
            task_type=NLPTaskType.NAMED_ENTITY_RECOGNITION,
            num_classes=9
        )
        model = create_fnet(
            "tiny", include_top=True, task_config=task_config,
            max_position_embeddings=16, vocab_size=100,
        )
        assert isinstance(model, FNet)

        output = model(sample_inputs)
        assert not isinstance(output, dict)
        assert output.shape == (2, 16, 9)


class TestPretrainedContract:
    """`pretrained=True` must RAISE, never return a random-init model.

    Before this contract, `FNet.PRETRAINED_WEIGHTS` held placeholder URLs on a non-existent host,
    `from_variant` caught the download failure, logged a warning and continued with
    random initialization — so a caller asking for pretrained weights silently
    got untrained ones and no error. Do not reinstate that.
    """

    def test_from_variant_pretrained_true_raises(self):
        with pytest.raises(NotImplementedError, match="No pretrained FNet weights"):
            FNet.from_variant("tiny", pretrained=True)

    def test_pretrained_false_still_builds(self):
        model = FNet.from_variant("tiny", vocab_size=128, max_position_embeddings=32)
        assert isinstance(model, FNet)

    def test_no_placeholder_weight_table(self):
        assert not hasattr(FNet, "PRETRAINED_WEIGHTS")



class TestCreateFNet:
    """``create_fnet`` is the package's factory; the head wrapper is gone."""

    def test_public_api(self):
        import dl_techniques.models.language.fnet as pkg

        assert sorted(pkg.__all__) == ["FNet", "create_fnet"]
        assert pkg.create_fnet is create_fnet
        # Replaced by `include_top` on the class (plan-2026-10-11T004932-b83935e1).
        assert not hasattr(pkg, "create_fnet_with_head")

    def test_pretrained_true_raises(self):
        with pytest.raises(NotImplementedError, match="No pretrained FNet weights"):
            create_fnet("tiny", pretrained=True)

    def test_delegates_to_from_variant(self):
        model = create_fnet("tiny", vocab_size=128, max_position_embeddings=32)
        assert isinstance(model, FNet)
        tiny = FNet.MODEL_VARIANTS["tiny"]
        assert (model.hidden_size, model.num_layers) == (
            tiny["hidden_size"], tiny["num_layers"])
        assert model.vocab_size == 128 and model.max_position_embeddings == 32


# ---------------------------------------------------------------------
# include_top: the in-model task head
# ---------------------------------------------------------------------

#: Small enough that every arm below runs in about a second. ``_SEQ`` is
#: deliberately NOT ``min(128, max_position_embeddings)``: a pretrained load
#: must build at the checkpoint's own length (D-010 of
#: plan-2026-10-11T004932-b83935e1), and a fixed-length build would drop every
#: encoder block of these checkpoints.
_HEAD_KW: Dict[str, Any] = {
    "vocab_size": 64,
    "hidden_size": 32,
    "num_layers": 2,
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


def _backbone_hidden(model: FNet, batch: Dict[str, np.ndarray]) -> np.ndarray:
    """``last_hidden_state`` of ``model``'s encoder, whatever its head.

    Read through a headless twin rebuilt from the model's own config and given
    the model's own encoder weights, layer by layer.
    """
    config = model.get_config()
    config.update(include_top=False, task_config=None, head_config_overrides=None)
    config.pop("name", None)
    twin = FNet.from_config(config)
    twin(batch, training=False)
    for layer in twin.layers:
        layer.set_weights(model.get_layer(layer.name).get_weights())
    return _np(twin(batch, training=False)["last_hidden_state"])


def _randomized(model: FNet, seed: int) -> FNet:
    """Overwrite every weight, so a restored value cannot match by init luck."""
    assert model.weights, "build the model before randomizing it"
    rng = np.random.default_rng(seed)
    for w in model.weights:
        w.assign(rng.normal(0.0, 0.1, size=tuple(w.shape)).astype(w.dtype))
    return model


class TestFNetIncludeTop:
    """The ``include_top`` / ``task_config`` / ``head_config_overrides`` knobs."""

    def test_head_output_shapes(self):
        """Bare logits for single-output heads, a dict for QA."""
        batch = _head_batch()
        out = FNet(**_HEAD_KW, include_top=True, task_config=_CLF)(
            batch, training=False)
        assert not isinstance(out, dict) and tuple(out.shape) == (2, 3)

        out = FNet(**_HEAD_KW, include_top=True, task_config=_NER)(
            batch, training=False)
        assert not isinstance(out, dict) and tuple(out.shape) == (2, _SEQ, 9)

        qa = FNet(**_HEAD_KW, include_top=True, task_config=NLPTaskConfig(
            name="qa", task_type=NLPTaskType.QUESTION_ANSWERING, num_classes=2))
        out = qa(batch, training=False)
        assert set(out) == {"start_logits", "end_logits"}
        assert tuple(out["start_logits"].shape) == (2, _SEQ)

    def test_symbolic_output_matches_the_eager_one(self):
        """A symbolic call at a static length reports what ``call`` returns."""
        model = FNet(**_HEAD_KW, include_top=True, task_config=_NER)
        symbolic = model(keras.Input(shape=(_SEQ,), dtype="int32"))
        assert tuple(symbolic.shape) == (None, _SEQ, 9)
        headless = FNet(**_HEAD_KW)
        symbolic = headless(keras.Input(shape=(_SEQ,), dtype="int32"))
        assert set(symbolic) == {"last_hidden_state", "attention_mask"}

    @pytest.mark.parametrize("include_top", [False, True], ids=["headless", "headed"])
    def test_the_sequence_length_is_static_with_or_without_a_head(self, include_top):
        """The DFT fixes the length at the first build; a head changes nothing.

        Why the old wrapper's ``sequence_length`` argument has no successor
        (D-010 of plan-2026-10-11T004932-b83935e1): the length comes from the
        first call or ``build((None, seq_len))``, as it always did headless.
        """
        kw = dict(_HEAD_KW, include_top=include_top,
                  task_config=_CLF if include_top else None)
        with pytest.raises(ValueError, match="must be known at build time"):
            FNet(**kw).build((None, None))
        model = FNet(**kw)
        model.build((None, 16))
        ids = np.ones((2, 16), dtype="int32")
        out = model(ids, training=False)
        out = out if include_top else out["last_hidden_state"]
        assert tuple(out.shape) == ((2, 3) if include_top else (2, 16, 32))
        with pytest.raises(tf.errors.InvalidArgumentError):
            model(np.ones((2, 12), dtype="int32"), training=False)

    def test_config_round_trip_with_a_head(self):
        """Every knob is serialized; ``task_config`` as a plain JSON dict."""
        import json

        overrides = {"pooling_type": "mean"}
        model = FNet(**_HEAD_KW, include_top=True, task_config=_CLF,
                     head_config_overrides=overrides)
        config = model.get_config()
        assert config["include_top"] is True
        assert config["head_config_overrides"] == overrides
        assert isinstance(config["task_config"], dict)
        assert config["task_config"]["task_type"] == "text_classification"
        json.dumps(config)  # plain data, no enum or dataclass inside

        from_dict = FNet(**_HEAD_KW, include_top=True,
                         task_config=dict(config["task_config"]),
                         head_config_overrides=overrides)
        assert from_dict.get_config()["task_config"] == config["task_config"]

        rebuilt = FNet.from_config(config)
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
        config = FNet(**_HEAD_KW).get_config()
        for key in ("include_top", "task_config", "head_config_overrides"):
            config.pop(key)
        model = FNet.from_config(config)
        assert model.include_top is False and model.task_head is None

    def test_keras_round_trip_restores_the_head_by_value(self, tmp_path):
        """Weights compared BEFORE the first call on the loaded model (guide 8.4)."""
        model = FNet(**_HEAD_KW, include_top=True, task_config=_CLF)
        batch = _head_batch()
        before = _np(model(batch, training=False))
        saved = [_np(w).copy() for w in model.weights]
        head_saved = [_np(w).copy() for w in model.task_head.weights]
        assert head_saved, "the head has no weights to compare"

        path = str(tmp_path / "fnet_head.keras")
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
        explicit = FNet(**kw)
        explicit.build((None, _SEQ))
        lazy = FNet(**kw)
        lazy(_head_batch(), training=False)
        assert _relative_paths(explicit) == _relative_paths(lazy)
        assert explicit.count_params() == lazy.count_params() > 0

    def test_a_disabled_head_builds_no_head_weights(self):
        """Anti-vacuity sibling of the parity arm (guide 8.3)."""
        headless = FNet(**_HEAD_KW)
        headless(_head_batch(), training=False)
        assert headless.task_head is None
        assert not [w.path for w in headless.weights if "task_head" in w.path]

        headed = FNet(**_HEAD_KW, include_top=True, task_config=_CLF)
        headed(_head_batch(), training=False)
        assert [w.path for w in headed.weights if "/task_head/" in w.path]

    def test_the_encoder_weight_paths_do_not_depend_on_the_head(self):
        """The head lives entirely under ``task_head/`` (D-005 of the plan)."""
        def paths(**kw):
            m = FNet(**_HEAD_KW, name="fnet", **kw)
            m(_head_batch(), training=False)
            return [w.path for w in m.weights]

        headless = paths()
        for task in (_CLF, _NER):
            headed = paths(include_top=True, task_config=task)
            encoder_part = [p for p in headed if not p.startswith("fnet/task_head/")]
            assert encoder_part == headless, task.name
            assert len(headed) > len(headless)

    def test_backbone_weights_transfer_across_heads(self, tmp_path):
        """An encoder trained under one head is reused, same weights, under another.

        The supported route is a local ``pretrained`` path, which copies the
        encoder by layer name and never the checkpoint's ``task_head``
        (D-007 of plan-2026-10-11T004932-b83935e1). Exercised into a
        TOKEN_CLASSIFICATION model and into a headless one.
        """
        source = create_fnet("tiny", include_top=True, task_config=_CLF, **_HEAD_KW)
        batch = _head_batch()
        source(batch, training=False)
        _randomized(source, 1)
        path = str(tmp_path / "trained_under_clf.keras")
        source.save(path)
        reference = _backbone_hidden(source, batch)

        token = create_fnet("tiny", include_top=True, task_config=_NER,
                            pretrained=path, **_HEAD_KW)
        headless = create_fnet("tiny", pretrained=path, **_HEAD_KW)

        np.testing.assert_array_equal(_backbone_hidden(token, batch), reference)
        np.testing.assert_array_equal(
            _np(headless(batch, training=False)["last_hidden_state"]), reference)
        assert tuple(token(batch, training=False).shape) == (2, _SEQ, 9)

        # Anti-vacuity: a fresh encoder does NOT reproduce the reference.
        fresh = create_fnet("tiny", **_HEAD_KW)
        assert not np.array_equal(
            _np(fresh(batch, training=False)["last_hidden_state"]), reference)

    def test_an_equal_shape_head_of_another_task_is_not_copied(self, tmp_path):
        """A SPAM head is not filled from a CLF checkpoint's head of equal shape.

        With the old ``skip_prefixes=()`` the transfer matched the head by
        layer name and copied it silently, because the shapes agree
        (plan-2026-10-11T004932-b83935e1 D-007).
        """
        source = create_fnet("tiny", include_top=True, task_config=_CLF, **_HEAD_KW)
        batch = _head_batch()
        source(batch, training=False)
        _randomized(source, 2)
        path = str(tmp_path / "clf.keras")
        source.save(path)

        spam = create_fnet("tiny", include_top=True, task_config=_SPAM,
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
        encoder = create_fnet("tiny", **_HEAD_KW)
        batch = _head_batch()
        encoder(batch, training=False)
        _randomized(encoder, 3)
        expected = _np(encoder(batch, training=False)["last_hidden_state"])
        path = str(tmp_path / "encoder.keras")
        encoder.save(path)

        headed = create_fnet("tiny", include_top=True, task_config=_CLF,
                             pretrained=path, **_HEAD_KW)
        np.testing.assert_array_equal(_backbone_hidden(headed, batch), expected)

    def test_a_pretrained_load_builds_at_the_checkpoint_length(self, tmp_path):
        """The loaded model runs at the length its checkpoint was built at.

        ``vocab_size`` 64 differs from the variant default, so ``from_variant``
        loads with ``skip_mismatch=True``: a length-mismatched block would be
        skipped, not raised, and only this by-value check would see it.
        """
        encoder = create_fnet("tiny", **_HEAD_KW)
        batch = _head_batch()
        encoder(batch, training=False)
        _randomized(encoder, 4)
        path = str(tmp_path / "encoder.keras")
        encoder.save(path)

        loaded = create_fnet("tiny", pretrained=path, **_HEAD_KW)
        assert loaded.built
        fourier = loaded.encoder_layers[0].fourier_transform
        assert fourier.dft_matrix_seq.shape[0] == _SEQ
        for i, block in enumerate(loaded.encoder_layers):
            for a, b in zip(block.weights, encoder.encoder_layers[i].weights):
                np.testing.assert_array_equal(_np(a), _np(b), err_msg=a.path)

    def test_include_top_without_task_config_raises(self):
        with pytest.raises(ValueError, match="include_top=True needs a task_config"):
            FNet(**_HEAD_KW, include_top=True)

    @pytest.mark.parametrize("dead", [
        {"task_config": _CLF},
        {"head_config_overrides": {"pooling_type": "mean"}},
    ], ids=["task_config", "head_config_overrides"])
    def test_head_knobs_without_include_top_raise(self, dead):
        with pytest.raises(ValueError, match="only take effect with include_top=True"):
            FNet(**_HEAD_KW, **dead)

    def test_pretrained_true_still_raises_with_a_head(self):
        with pytest.raises(NotImplementedError):
            create_fnet("tiny", pretrained=True, include_top=True, task_config=_CLF)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
