"""Tests for ``heads/nlp/model_head.py``, the in-model task head helpers.

Covers:

* ``task_config`` given as a dataclass, as the dict ``task_config_to_dict``
  writes, or as a dict holding the enum member, builds the same head.
* ``MULTIPLE_CHOICE`` is refused with a ``ValueError``.
* ``default_pooling`` applies only when the overrides set no ``pooling_type``.
* The head carries the requested layer name, and every weight sits under it.
* ``finalize_head_outputs`` gives a bare tensor for classification, token
  classification, generation and similarity heads, and keeps the QA dict.
* The returned head round-trips through ``get_config`` / ``from_config`` by
  value.
"""

import json

import keras
import numpy as np
import pytest

from dl_techniques.layers.heads.nlp import (
    NLPTaskConfig,
    NLPTaskType,
    build_model_task_head,
    finalize_head_outputs,
    task_config_to_dict,
)

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
