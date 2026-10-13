"""Tests for ``heads/nlp/model_head.py``, the in-model task head helpers.

Covers:

* ``task_config`` given as a dataclass, as the dict ``task_config_to_dict``
  writes, or as a dict holding the enum member, builds the same head.
* ``MULTIPLE_CHOICE`` is refused with a ``ValueError``.
* ``validate_task_head_knobs`` accepts the two consistent knob sets and
  refuses the two inconsistent ones.
* ``default_pooling`` applies only when the overrides set no ``pooling_type``.
* The head carries the requested layer name, and every weight sits under it.
* ``finalize_head_outputs`` gives a bare tensor for classification, token
  classification, generation and similarity heads, and keeps the QA dict.
* The returned head round-trips through ``get_config`` / ``from_config`` by
  value.
* ``head_attention_mask`` returns the caller's mask when given, else
  ``input_ids != pad_token_id``, else (no pad id either) all ones, in the
  requested dtype.
* ``NLPTaskConfig`` refuses a classification class count that is not a
  positive integer (0, -1, a bool, a float), on the config and through a
  model's ``task_config`` dict; numpy integers are accepted.
"""

import json

import keras
import numpy as np
import pytest

from dl_techniques.layers.heads.nlp.model_head import (
    head_output_shape,
    ids_shape_of,
)
from dl_techniques.layers.heads.nlp import (
    NLPTaskConfig,
    NLPTaskType,
    build_model_task_head,
    finalize_head_outputs,
    head_attention_mask,
    task_config_to_dict,
    validate_task_head_knobs,
)
from dl_techniques.models.language.bert.model import create_bert

# ---------------------------------------------------------------------

B, S, D = 2, 5, 16


def _classification_config() -> NLPTaskConfig:
    return NLPTaskConfig(
        name="sst2",
        task_type=NLPTaskType.TEXT_CLASSIFICATION,
        num_classes=3,
        dropout_rate=0.2,
        max_length=128,
    )


def _head_inputs() -> dict:
    rng = np.random.default_rng(7)
    mask = np.ones((B, S), dtype="int32")
    mask[0, -2:] = 0
    return {
        "hidden_states": rng.standard_normal((B, S, D)).astype("float32"),
        "attention_mask": mask,
    }


# ---------------------------------------------------------------------
# task_config forms
# ---------------------------------------------------------------------


class TestTaskConfigForms:

    def test_dict_round_trips_to_an_equal_dataclass(self) -> None:
        cfg = _classification_config()
        as_dict = task_config_to_dict(cfg)
        assert as_dict["task_type"] == "text_classification"
        json.dumps(as_dict)  # plain data, no enum member left
        assert NLPTaskConfig(
            **{**as_dict, "task_type": NLPTaskType(as_dict["task_type"])}
        ) == cfg
        assert task_config_to_dict(as_dict) == as_dict

    def test_dict_and_dataclass_build_the_same_head(self) -> None:
        cfg = _classification_config()
        enum_dict = {
            "name": "sst2",
            "task_type": NLPTaskType.TEXT_CLASSIFICATION,
            "num_classes": 3,
            "dropout_rate": 0.2,
            "max_length": 128,
        }
        configs = [
            build_model_task_head(form, input_dim=D).get_config()
            for form in (cfg, task_config_to_dict(cfg), enum_dict)
        ]
        assert configs[0] == configs[1] == configs[2]
        assert configs[0]["task_config"]["dropout_rate"] == 0.2

    def test_the_callers_dict_is_not_modified(self) -> None:
        as_dict = task_config_to_dict(_classification_config())
        snapshot = dict(as_dict)
        build_model_task_head(as_dict, input_dim=D)
        assert as_dict == snapshot

    def test_a_non_config_is_refused(self) -> None:
        with pytest.raises(TypeError, match="task_config must be"):
            build_model_task_head("text_classification", input_dim=D)


# ---------------------------------------------------------------------
# refusals and knobs
# ---------------------------------------------------------------------


class TestBuildRules:

    @pytest.mark.parametrize("as_dict", [False, True], ids=["dataclass", "dict"])
    def test_multiple_choice_is_refused(self, as_dict: bool) -> None:
        cfg = NLPTaskConfig(
            name="mc", task_type=NLPTaskType.MULTIPLE_CHOICE, num_classes=4
        )
        form = task_config_to_dict(cfg) if as_dict else cfg
        with pytest.raises(ValueError, match="MULTIPLE_CHOICE"):
            build_model_task_head(form, input_dim=D)

    @pytest.mark.parametrize(
        "overrides, default_pooling, expected",
        [
            (None, None, "cls"),
            (None, "last", "last"),
            ({"use_intermediate": False}, "last", "last"),
            ({"pooling_type": "mean"}, "last", "mean"),
            ({"pooling_type": "mean"}, None, "mean"),
        ],
        ids=[
            "head-default",
            "default-applied",
            "default-applied-beside-other-overrides",
            "explicit-override-wins",
            "explicit-override-alone",
        ],
    )
    def test_default_pooling_applies_only_without_an_explicit_one(
            self, overrides, default_pooling, expected
    ) -> None:
        snapshot = None if overrides is None else dict(overrides)
        head = build_model_task_head(
            _classification_config(),
            input_dim=D,
            head_config_overrides=overrides,
            default_pooling=default_pooling,
        )
        assert head.pooling_type == expected
        assert overrides == snapshot

    @pytest.mark.parametrize(
        "include_top, task_config, overrides, match",
        [
            (True, None, None, "include_top=True needs a task_config"),
            (True, None, {"pooling_type": "mean"},
             "include_top=True needs a task_config"),
            (False, "cfg", None, "only take effect with include_top=True"),
            (False, None, {"pooling_type": "mean"},
             "only take effect with include_top=True"),
            (True, "cfg", None, None),
            (True, "cfg", {"pooling_type": "mean"}, None),
            (False, None, None, None),
        ],
        ids=["missing-config", "missing-config-with-overrides",
             "dead-config", "dead-overrides",
             "head", "head-with-overrides", "headless"],
    )
    def test_the_knob_set_is_validated(
            self, include_top, task_config, overrides, match
    ) -> None:
        cfg = _classification_config() if task_config == "cfg" else None
        if match is None:
            assert validate_task_head_knobs(include_top, cfg, overrides) is None
            return
        with pytest.raises(ValueError, match=match):
            validate_task_head_knobs(include_top, cfg, overrides)

    def test_overrides_reach_the_head(self) -> None:
        head = build_model_task_head(
            _classification_config(),
            input_dim=D,
            head_config_overrides={"use_intermediate": False},
        )
        assert head.use_intermediate is False
        assert head.intermediate is None

    def test_a_name_override_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must not set 'name'"):
            build_model_task_head(
                _classification_config(),
                input_dim=D,
                head_config_overrides={"name": "other"},
            )

    @pytest.mark.parametrize("name", ["task_head", "classifier_top"])
    def test_every_head_weight_lives_under_the_layer_name(self, name: str) -> None:
        kwargs = {} if name == "task_head" else {"name": name}
        head = build_model_task_head(_classification_config(), input_dim=D, **kwargs)
        head(_head_inputs())
        assert head.name == name
        assert head.weights
        stray = [w.path for w in head.weights if not w.path.startswith(f"{name}/")]
        assert stray == []


# ---------------------------------------------------------------------
# finalize_head_outputs
# ---------------------------------------------------------------------


_FINALIZE_CASES = [
    # (task_type, extra config fields, raw keys, expected bare shape or dict keys)
    (NLPTaskType.TEXT_CLASSIFICATION, {"num_classes": 3},
     {"logits", "probabilities"}, (B, 3)),
    (NLPTaskType.TOKEN_CLASSIFICATION, {"num_classes": 4},
     {"logits", "predictions"}, (B, S, 4)),
    (NLPTaskType.TEXT_GENERATION, {"vocabulary_size": 11},
     {"logits"}, (B, S, 11)),
    (NLPTaskType.TEXT_SIMILARITY, {},
     {"embeddings"}, None),
    (NLPTaskType.QUESTION_ANSWERING, {},
     {"start_logits", "end_logits"}, {"start_logits", "end_logits"}),
]


class TestFinalizeHeadOutputs:

    @pytest.mark.parametrize(
        "task_type, extra, raw_keys, expected",
        _FINALIZE_CASES,
        ids=[case[0].name for case in _FINALIZE_CASES],
    )
    def test_each_head_family_finalizes_to_its_contract(
            self, task_type, extra, raw_keys, expected
    ) -> None:
        cfg = NLPTaskConfig(name="t", task_type=task_type, **extra)
        head = build_model_task_head(cfg, input_dim=D)
        raw = head(_head_inputs(), training=False)
        assert set(raw) == raw_keys
        raw_snapshot = dict(raw)

        final = finalize_head_outputs(raw)

        assert dict(raw) == raw_snapshot  # input dict untouched
        if isinstance(expected, set):
            assert isinstance(final, dict)
            assert set(final) == expected
            return
        assert not isinstance(final, dict)
        source_key = "logits" if "logits" in raw_keys else next(iter(raw_keys))
        np.testing.assert_array_equal(
            keras.ops.convert_to_numpy(final),
            keras.ops.convert_to_numpy(raw[source_key]),
        )
        if expected is not None:
            assert tuple(final.shape) == expected

    def test_a_bare_tensor_passes_through(self) -> None:
        tensor = keras.ops.ones((B, 3))
        assert finalize_head_outputs(tensor) is tensor

    def test_a_multi_key_dict_without_logits_is_kept(self) -> None:
        raw = {"a": keras.ops.ones((B,)), "b": keras.ops.zeros((B,))}
        final = finalize_head_outputs(raw)
        assert set(final) == {"a", "b"}
        assert final is not raw

    def test_the_finalized_classifier_compiles_with_a_sparse_loss(self) -> None:
        head = build_model_task_head(_classification_config(), input_dim=D)
        inputs = {
            "hidden_states": keras.Input(shape=(None, D), name="hidden_states"),
            "attention_mask": keras.Input(
                shape=(None,), dtype="int32", name="attention_mask"
            ),
        }
        model = keras.Model(inputs, finalize_head_outputs(head(inputs)))
        model.compile(
            optimizer="sgd",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        result = model.evaluate(
            _head_inputs(), np.array([0, 2]), verbose=0, return_dict=True
        )
        assert np.isfinite(result["loss"])


# ---------------------------------------------------------------------
# a generation head's vocabulary is the model's
# ---------------------------------------------------------------------

_GENERATION_TYPES = [
    NLPTaskType.TEXT_GENERATION,
    NLPTaskType.MASKED_LANGUAGE_MODELING,
    NLPTaskType.TEXT_COMPLETION,
    NLPTaskType.TEXT_SUMMARIZATION,
]
_MODEL_VOCAB = 11


def _generation_dict(task_type, **extra) -> dict:
    return {"name": "g", "task_type": task_type, **extra}


@pytest.mark.parametrize("task_type", _GENERATION_TYPES, ids=lambda t: t.name)
class TestGenerationVocabulary:

    def test_an_unset_vocabulary_is_filled_from_the_model(self, task_type) -> None:
        head = build_model_task_head(
            _generation_dict(task_type), input_dim=D, vocab_size=_MODEL_VOCAB)
        assert head.task_config.vocabulary_size == _MODEL_VOCAB
        out = head(_head_inputs(), training=False)
        assert tuple(out["logits"].shape) == (B, S, _MODEL_VOCAB)

    def test_the_string_form_of_the_task_type_is_filled_too(self, task_type) -> None:
        head = build_model_task_head(
            _generation_dict(task_type.value), input_dim=D, vocab_size=_MODEL_VOCAB)
        assert head.task_config.vocabulary_size == _MODEL_VOCAB

    def test_the_same_explicit_vocabulary_is_accepted(self, task_type) -> None:
        head = build_model_task_head(
            _generation_dict(task_type, vocabulary_size=_MODEL_VOCAB),
            input_dim=D, vocab_size=_MODEL_VOCAB)
        assert head.task_config.vocabulary_size == _MODEL_VOCAB

    def test_a_different_explicit_vocabulary_raises(self, task_type) -> None:
        with pytest.raises(ValueError, match="vocabulary_size=7.*model's vocabulary is 11"):
            build_model_task_head(
                _generation_dict(task_type, vocabulary_size=7),
                input_dim=D, vocab_size=_MODEL_VOCAB)

    def test_a_dataclass_that_kept_the_default_raises(self, task_type) -> None:
        """The default 32000 cannot be told from an explicit 32000."""
        config = NLPTaskConfig(name="g", task_type=task_type)
        assert config.vocabulary_size == 32000
        with pytest.raises(ValueError, match="vocabulary_size=32000"):
            build_model_task_head(config, input_dim=D, vocab_size=_MODEL_VOCAB)
        # ...and is fine on a model whose vocabulary IS 32000.
        build_model_task_head(config, input_dim=D, vocab_size=32000)

    def test_no_model_vocabulary_means_no_check(self, task_type) -> None:
        head = build_model_task_head(
            NLPTaskConfig(name="g", task_type=task_type), input_dim=D)
        assert head.task_config.vocabulary_size == 32000

    def test_task_config_to_dict_fills_the_same_way(self, task_type) -> None:
        filled = task_config_to_dict(_generation_dict(task_type), vocab_size=_MODEL_VOCAB)
        assert filled["vocabulary_size"] == _MODEL_VOCAB
        explicit = task_config_to_dict(
            _generation_dict(task_type, vocabulary_size=5), vocab_size=_MODEL_VOCAB)
        assert explicit["vocabulary_size"] == 5  # converting never raises
        assert task_config_to_dict(_generation_dict(task_type))["vocabulary_size"] == 32000


def test_a_task_that_is_not_generation_ignores_the_model_vocabulary() -> None:
    head = build_model_task_head(
        _classification_config(), input_dim=D, vocab_size=_MODEL_VOCAB)
    out = head(_head_inputs(), training=False)
    assert tuple(out["logits"].shape) == (B, 3)


# ---------------------------------------------------------------------
# a classification class count must be a positive integer
# ---------------------------------------------------------------------

_BAD_CLASS_COUNTS = [0, -1, True, False, 3.0]
_BAD_IDS = ["zero", "minus_one", "true", "false", "float"]


class TestClassificationClassCount:
    """A count of 0 built an ``(N, 0)`` head silently; -1 failed only on the
    first call ("Negative dimensions"). The removed per-task Qwen/Gemma
    factories refused both; the config now refuses them for every model."""

    @pytest.mark.parametrize("count", _BAD_CLASS_COUNTS, ids=_BAD_IDS)
    @pytest.mark.parametrize("field", ["num_classes", "num_labels"])
    def test_the_config_refuses_it(self, count, field) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            NLPTaskConfig(
                name="clf", task_type=NLPTaskType.TEXT_CLASSIFICATION,
                **{field: count})

    @pytest.mark.parametrize("count", [0, -1], ids=["zero", "minus_one"])
    def test_a_model_refuses_it_at_construction(self, count) -> None:
        task_config = {
            "name": "clf", "task_type": "text_classification",
            "num_classes": count,
        }
        with pytest.raises(ValueError, match="positive integer"):
            create_bert("tiny", include_top=True, task_config=task_config)

    @pytest.mark.parametrize(
        "count", [1, 3, np.int32(3), np.int64(3)],
        ids=["one", "three", "np_int32", "np_int64"])
    def test_a_positive_integer_is_accepted(self, count) -> None:
        cfg = NLPTaskConfig(
            name="clf", task_type=NLPTaskType.NAMED_ENTITY_RECOGNITION,
            num_classes=count)
        assert cfg.num_classes == cfg.num_labels == count

    def test_a_task_that_is_not_classification_is_not_checked(self) -> None:
        cfg = NLPTaskConfig(
            name="qa", task_type=NLPTaskType.QUESTION_ANSWERING, num_classes=0)
        assert cfg.num_classes == 0


# ---------------------------------------------------------------------
# the shapes a headed model returns
# ---------------------------------------------------------------------


def _structure_of_shapes(outputs):
    if isinstance(outputs, dict):
        return {key: tuple(value.shape) for key, value in outputs.items()}
    return tuple(outputs.shape)


class TestHeadOutputShape:

    @pytest.mark.parametrize(
        "task_type, extra, raw_keys, expected",
        _FINALIZE_CASES,
        ids=[case[0].name for case in _FINALIZE_CASES],
    )
    def test_it_reports_the_shapes_finalized_outputs_have(
            self, task_type, extra, raw_keys, expected
    ) -> None:
        cfg = NLPTaskConfig(name="t", task_type=task_type, **extra)
        head = build_model_task_head(cfg, input_dim=D)
        actual = _structure_of_shapes(
            finalize_head_outputs(head(_head_inputs(), training=False)))
        reported = head_output_shape(head, (B, S), D)
        if isinstance(reported, dict):
            reported = {key: tuple(value) for key, value in reported.items()}
        else:
            reported = tuple(reported)
        assert reported == actual

    @pytest.mark.parametrize("form", ["tuple", "dict", "list of shapes"])
    def test_the_ids_shape_is_accepted_in_every_form_a_model_gets(self, form) -> None:
        shape = {"tuple": (B, S),
                 "dict": {"input_ids": (B, S), "attention_mask": (B, S)},
                 "list of shapes": [(B, S), (B, S)]}[form]
        assert ids_shape_of(shape) == (B, S)
        head = build_model_task_head(_classification_config(), input_dim=D)
        assert head_output_shape(head, shape, D) == (B, 3)


# ---------------------------------------------------------------------
# serialization of the returned head
# ---------------------------------------------------------------------


class TestHeadRoundTrip:

    def test_get_config_from_config_round_trip_by_value(self) -> None:
        head = build_model_task_head(
            _classification_config(),
            input_dim=D,
            head_config_overrides={"use_intermediate": False},
            default_pooling="last",
        )
        inputs = _head_inputs()
        original = keras.ops.convert_to_numpy(head(inputs, training=False)["logits"])

        config = head.get_config()
        rebuilt = type(head).from_config(dict(config))
        rebuilt(inputs, training=False)
        rebuilt.set_weights(head.get_weights())

        assert rebuilt.name == "task_head"
        assert rebuilt.pooling_type == "last"
        assert rebuilt.get_config() == config
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(rebuilt(inputs, training=False)["logits"]),
            original,
            atol=1e-6,
            rtol=0,
        )


# ---------------------------------------------------------------------
# the head's padding mask
# ---------------------------------------------------------------------


class TestHeadAttentionMask:

    def test_no_mask_derives_one_from_the_pad_id(self) -> None:
        ids = np.array([[5, 7, 3, 3], [3, 9, 1, 2]], dtype="int32")
        mask = keras.ops.convert_to_numpy(
            head_attention_mask(ids, None, pad_token_id=3, dtype="float32"))
        np.testing.assert_array_equal(
            mask, [[1, 1, 0, 0], [0, 1, 1, 1]])
        assert mask.dtype == np.float32

    def test_a_given_mask_wins_over_the_pad_id(self) -> None:
        ids = np.zeros((2, 4), dtype="int32")  # all "padding" by id
        given = np.array([[1, 1, 1, 0], [1, 0, 0, 0]], dtype="int32")
        mask = keras.ops.convert_to_numpy(
            head_attention_mask(ids, given, pad_token_id=0, dtype="float16"))
        np.testing.assert_array_equal(mask, given)
        assert mask.dtype == np.float16

    def test_no_mask_and_no_pad_id_is_all_ones(self) -> None:
        # Id 0 is a real token; with no pad id nothing is padding.
        ids = np.array([[5, 0, 3, 0], [0, 0, 0, 0]], dtype="int32")
        mask = keras.ops.convert_to_numpy(
            head_attention_mask(ids, None, pad_token_id=None, dtype="float32"))
        np.testing.assert_array_equal(mask, np.ones((2, 4)))
        assert mask.dtype == np.float32

    def test_a_given_mask_wins_when_there_is_no_pad_id(self) -> None:
        ids = np.zeros((2, 4), dtype="int32")
        given = np.array([[1, 1, 1, 0], [1, 0, 0, 0]], dtype="int32")
        mask = keras.ops.convert_to_numpy(
            head_attention_mask(ids, given, pad_token_id=None, dtype="float16"))
        np.testing.assert_array_equal(mask, given)
        assert mask.dtype == np.float16

    def test_the_all_ones_mask_has_the_input_shape_under_symbolic_ids(self) -> None:
        ids = keras.Input(shape=(7,), dtype="int32")
        mask = head_attention_mask(ids, None, None, "float32")
        assert tuple(mask.shape) == (None, 7)
