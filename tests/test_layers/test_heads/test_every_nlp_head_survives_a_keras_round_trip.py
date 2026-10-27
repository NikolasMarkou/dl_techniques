"""Every NLP head survives ``get_config`` and a ``.keras`` round trip by value.

Guard for plan-2026-10-14T062616-81938e81 step 11.3 (the user's heads review,
``research/reviews/20261015_heads_review.md``, NLP serialization gaps and the
"no NLP round-trip test" test gap). Measured at ``846589b7f``:

* ``BaseNLPHead.get_config`` wrote ten of the fourteen ``NLPTaskConfig``
  fields, and ``MultiTaskNLPHead.get_config`` six: a token head built with
  ``use_crf=True`` inside a multi-task head came back with ``use_crf=False``
  (and a ``'predictions'`` output it did not have). Reproduced. Seven of the
  fourteen fields were read by no head at all (``num_labels``, an alias of
  ``num_classes``; ``max_length``, ``loss_weight``, ``label_smoothing``,
  ``use_attention_pooling``, ``beam_size``, ``temperature``); they are deleted
  (D-029), and the config now holds every remaining field.
* Both ``from_config`` overrides ``pop``-ed ``'task_config(s)'`` out of the
  caller's dict and wrote a dataclass back in its place. Reproduced. The
  heads now take a dict ``task_config`` in ``__init__``, and the overrides
  are gone.
* ``task_heads``/``task_projections`` are plain dicts and
  ``similarity_layers`` a list; the review read this as a layout that does
  not round-trip. Refuted for the dicts: a functional model holding a
  ``MultiTaskNLPHead`` with projections saved and reloaded with all 20
  weights and every output bit-identical (Keras tracks a dict of layers by
  key). The list could not be measured at ``846589b7f``, because a pair given
  as a list never reached it (see ``test_the_nlp_heads_honour_padding_masks``);
  it is measured here and holds.

Each case wraps the head in a functional model, randomizes every weight (so no
value can match by a shared initializer), saves, reloads, and compares every
weight by path and every output, exactly.
"""

import copy
import dataclasses
import inspect

import keras
import numpy as np
import pytest

from dl_techniques.layers.heads.nlp import (
    MultiTaskNLPHead,
    NLPTaskConfig,
    NLPTaskType,
    create_multi_task_nlp_head,
    create_nlp_head,
)
from dl_techniques.layers.heads.nlp import BaseNLPHead

D = 8
B, S, C = 2, 5, 3
_RNG = np.random.default_rng(11)
SEQ = _RNG.normal(size=(B, S, D)).astype("float32")
MASK = np.ones((B, S), "float32")
MASK[1, 3:] = 0.0
CHOICES = _RNG.normal(size=(B, C, S, D)).astype("float32")


def _cfg(task, name="t", **kw):
    return NLPTaskConfig(name=name, task_type=task, **kw)


CLF = _cfg(NLPTaskType.TEXT_CLASSIFICATION, "clf", num_classes=3, hidden_size=16)
TOK = _cfg(NLPTaskType.NAMED_ENTITY_RECOGNITION, "ner", num_classes=4, use_crf=True)
QA = _cfg(NLPTaskType.QUESTION_ANSWERING, "qa")
SIM = _cfg(NLPTaskType.TEXT_SIMILARITY, "sim", hidden_size=12)
GEN = _cfg(NLPTaskType.TEXT_GENERATION, "gen", vocabulary_size=13)
MC = _cfg(NLPTaskType.MULTIPLE_CHOICE, "mc")


def _masked(x=SEQ, mask=MASK):
    return {"hidden_states": x, "attention_mask": mask}


#: id -> (head factory, input).
CASES = {
    "classification-cls": (lambda: create_nlp_head(CLF, D), _masked()),
    "classification-attention-pool-ffn": (
        lambda: create_nlp_head(CLF, D, pooling_type="attention", use_ffn=True), _masked()),
    "classification-last-no-intermediate": (
        lambda: create_nlp_head(CLF, D, pooling_type="last", use_intermediate=False), _masked()),
    "token-crf-attention-ffn": (
        lambda: create_nlp_head(TOK, D, use_task_attention=True, use_ffn=True), SEQ),
    "token-no-crf": (
        lambda: create_nlp_head(dataclasses.replace(TOK, use_crf=False), D), SEQ),
    "qa-attention": (lambda: create_nlp_head(QA, D, use_task_attention=True), _masked()),
    "similarity-single-mean": (lambda: create_nlp_head(SIM, D, pooling_type="mean"), _masked()),
    "similarity-pair-learned": (
        lambda: create_nlp_head(SIM, D, pooling_type="mean", similarity_function="learned"),
        [_masked(), _masked(SEQ[::-1].copy())]),
    "generation-ffn": (lambda: create_nlp_head(GEN, D, use_ffn=True), SEQ),
    "multiple-choice-masked": (
        lambda: create_nlp_head(MC, D, pooling_type="mean"),
        {"hidden_states": CHOICES, "attention_mask": np.ones((B, C, S), "float32")}),
    "multi-task-projections": (
        lambda: create_multi_task_nlp_head([CLF, TOK, QA], D,
                                           use_task_specific_projections=True),
        _masked()),
}


def _symbolic(x):
    if isinstance(x, dict):
        return {k: keras.Input(shape=v.shape[1:], name=k) for k, v in x.items()}
    if isinstance(x, list):
        return [{k: keras.Input(shape=v.shape[1:], name=f"{k}_{i}") for k, v in item.items()}
                for i, item in enumerate(x)]
    return keras.Input(shape=x.shape[1:])


def _feed(x):
    if isinstance(x, list):
        return [{k: v for k, v in item.items()} for item in x]
    return x


def _leaves(y):
    if isinstance(y, dict):
        return [leaf for k in sorted(y) for leaf in _leaves(y[k])]
    return [np.asarray(y)]


@pytest.mark.parametrize("case", sorted(CASES))
def test_a_saved_and_reloaded_head_is_the_same_head(case, tmp_path) -> None:
    make, x = CASES[case]
    inputs = _symbolic(x)
    model = keras.Model(inputs, make()(inputs))
    rng = np.random.default_rng(3)
    for w in model.weights:
        w.assign(rng.normal(0.0, 0.2, size=tuple(w.shape)).astype(w.dtype))
    before = model.predict(_feed(x), verbose=0)

    path = str(tmp_path / f"{case}.keras")
    model.save(path)
    loaded = keras.models.load_model(path)

    weights = {w.path: np.asarray(w) for w in model.weights}
    reloaded = {w.path: np.asarray(w) for w in loaded.weights}
    assert weights and weights.keys() == reloaded.keys()
    for key, value in weights.items():
        np.testing.assert_array_equal(reloaded[key], value, err_msg=key)
    after = loaded.predict(_feed(x), verbose=0)
    for a, b in zip(_leaves(before), _leaves(after), strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("case", sorted(CASES))
def test_from_config_rebuilds_the_same_config_and_leaves_its_input_alone(case) -> None:
    make, x = CASES[case]
    head = make()
    head(_feed(x) if not isinstance(x, list) else tuple(x))
    config = head.get_config()
    snapshot = copy.deepcopy(config)
    rebuilt = type(head).from_config(config)
    assert config == snapshot, "from_config modified the caller's dict"
    assert rebuilt.get_config() == snapshot


_EVERY_TASK_FIELD = {f.name for f in dataclasses.fields(NLPTaskConfig)}


@pytest.mark.parametrize("case", sorted(c for c in CASES if not c.startswith("multi-task")))
def test_get_config_holds_every_constructor_argument(case) -> None:
    head = CASES[case][0]()
    config = head.get_config()
    wanted = set()
    for klass in type(head).__mro__:
        if klass is keras.layers.Layer:
            break
        if "__init__" in klass.__dict__:
            wanted |= {p for p in inspect.signature(klass.__init__).parameters
                       if p not in ("self", "kwargs")}
    assert wanted <= set(config), f"missing {sorted(wanted - set(config))}"
    assert set(config["task_config"]) == _EVERY_TASK_FIELD


def test_the_multi_task_config_holds_every_task_field() -> None:
    head = create_multi_task_nlp_head([CLF, TOK], D)
    config = head.get_config()
    for name in ("clf", "ner"):
        assert set(config["task_configs"][name]) == _EVERY_TASK_FIELD


def test_use_crf_survives_a_multi_task_round_trip() -> None:
    head = create_multi_task_nlp_head({"ner": TOK, "clf": CLF}, D)
    rebuilt = MultiTaskNLPHead.from_config(copy.deepcopy(head.get_config()))
    assert rebuilt.task_heads["ner"].task_config.use_crf is True
    assert sorted(rebuilt({"hidden_states": SEQ}, task_name="ner")) == ["logits"]


def test_a_dict_task_config_is_accepted_by_the_heads() -> None:
    """What ``from_config`` hands ``__init__`` now that the overrides are gone."""
    as_dict = CLF.to_dict()
    head = create_nlp_head(CLF, D)
    assert type(head)(task_config=as_dict, input_dim=D).task_config == CLF
    assert isinstance(BaseNLPHead(task_config=as_dict, input_dim=D).task_config, NLPTaskConfig)
