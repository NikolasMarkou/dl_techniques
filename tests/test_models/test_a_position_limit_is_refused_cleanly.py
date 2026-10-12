"""A model with a position limit refuses an over-long sequence CLEANLY, on every path.

Subject: ``dl_techniques.utils.sequence_length_guard`` and every language model
wired onto it (``SUBJECTS`` below; plan ``plan-2026-10-11T170958-dc4d2202``,
D-006). "Cleanly" has four parts, and each is an arm because each failed
somewhere on its own:

``test_a_static_over_long_first_call_is_refused_before_any_variable_exists``
    ``model(ids)`` with a statically over-long length raises ``ValueError``
    from ``build``, and the model holds NO variable afterwards. A refusal that
    fires inside ``call`` (after ``build`` made the variables) is the
    precondition of the unusable-model defect below.
``test_a_refused_first_predict_leaves_the_model_usable``
    A ``ValueError`` raised during the first trace of Keras's cached
    ``predict_function`` leaves that function broken for every later call
    (measured in 9 of 13 classes before the guard). The refusal must drop the
    cached step functions, so the next legal ``predict`` works and is finite.
``test_the_declared_limit_itself_is_legal``
    Anti-vacuity: a ``<`` where ``<=`` belongs passes both raise arms while
    breaking every model run at exactly its declared maximum.
``test_a_relaxed_retrace_refuses_an_over_long_batch_and_stays_usable``
    After ``predict`` has seen two lengths, TensorFlow retraces with the
    sequence axis relaxed to ``None`` and never re-enters Python for a new
    length, so the static check is skipped. The run-time gate must fire there:
    ``InvalidArgumentError`` (a Python ``ValueError`` cannot cross a graph
    boundary). Run three ways: Keras's default on a CPU-only box, Keras's
    default with a GPU visible (XLA), and ``jit_compile=True``. Under XLA the
    assertion is dropped, and only the zero-sum term, which fails the XLA
    compile naming ``seq_len_exceeds_<limit_name>_<limit>``, refuses; on a
    gather-based position table the alternative is FINITE GARBAGE. A legal call
    afterwards must work again.
``test_the_model_reaches_the_shared_guard``
    The guard is only worth its name if a class cannot silently bypass it with
    a private copy. The model's module is patched to record the shared
    functions' calls: both must be reached, by the module-level names.

Why XLA is forced: a CPU-only suite never sees XLA, and Keras 3.8 resolves
``jit_compile="auto"`` through ``Trainer._resolve_auto_jit_compile``, which on
a GPU box returns ``model_supports_jit(model)``. The resolver is replaced so
every arm runs the same configuration on any machine;
``monkeypatch.setattr`` raises if Keras renames it, and each arm asserts the
resolved value, so neither can go vacuous.

Adding a subject is one ``Subject(...)`` row in ``SUBJECTS``.
"""

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Dict

import keras
import numpy as np
import pytest

from dl_techniques.layers.heads.nlp import NLPTaskConfig, NLPTaskType
from dl_techniques.models.language.bert import create_bert

# ---------------------------------------------------------------------

BATCH = 2
LEGAL_LENGTHS = (5, 12)
OVER_BY = 1

# ---------------------------------------------------------------------


@dataclass(frozen=True)
class Subject:
    """One model under test.

    :param name: Test id.
    :param module: Importable module whose namespace holds the model's
        references to the two shared guard functions (patched by the
        reachability arm).
    :param limit_name: Name of the limit attribute on the model; also the
        stem of the XLA guard node.
    :param factory: Builds a fresh, unbuilt, tiny model with a small limit.
    :param feed: ``(model, seq_len) -> model input`` of that sequence length.
    :param message_regex: Pattern (with ``{limit}``) the graph-path refusal
        must match.
    """

    name: str
    module: str
    limit_name: str
    factory: Callable[[], keras.Model]
    feed: Callable[[keras.Model, int], Any]
    message_regex: str


def _bert_factory() -> keras.Model:
    return create_bert(
        "tiny",
        include_top=True,
        task_config=NLPTaskConfig(
            name="sentiment",
            task_type=NLPTaskType.SENTIMENT_ANALYSIS,
            num_classes=3,
        ),
        max_position_embeddings=16,
    )


def _bert_feed(model: keras.Model, seq_len: int) -> Dict[str, np.ndarray]:
    ids = np.random.RandomState(7).randint(0, 100, (BATCH, seq_len)).astype("int32")
    return {
        "input_ids": ids,
        "attention_mask": np.ones_like(ids),
        "token_type_ids": np.zeros_like(ids),
    }


SUBJECTS = [
    Subject(
        name="bert",
        module="dl_techniques.models.language.bert.model",
        limit_name="max_position_embeddings",
        factory=_bert_factory,
        feed=_bert_feed,
        message_regex=r"exceeds max_position_embeddings={limit}(?![0-9])",
    ),
]

SUBJECT_IDS = [s.name for s in SUBJECTS]

# ---------------------------------------------------------------------


def _limit(subject: Subject, model: keras.Model) -> int:
    return int(getattr(model, subject.limit_name))


def _finite(out: Any) -> bool:
    return all(
        np.isfinite(np.asarray(leaf)).all() for leaf in keras.tree.flatten(out)
    )


_JIT_CONFIGS = ("cpu_default", "gpu_default", "jit_compile_true")


def _resolve_jit(monkeypatch: pytest.MonkeyPatch, config: str) -> bool:
    """Pin how ``"auto"`` resolves; return the ``jit_compile`` the model must end with.

    ``jit_compile_true`` leaves the resolver alone: the caller sets it.
    """
    # Keras exposes no public switch for how ``jit_compile="auto"`` resolves,
    # so the private resolver is the only seam; reached through ``keras``.
    keras_trainer = keras.src.trainers.trainer

    if config == "jit_compile_true":
        return True
    expected = config == "gpu_default"
    monkeypatch.setattr(
        keras_trainer.Trainer,
        "_resolve_auto_jit_compile",
        (lambda self: keras_trainer.model_supports_jit(self))
        if expected
        else (lambda self: False),
    )
    return expected


# ---------------------------------------------------------------------


@pytest.mark.parametrize("subject", SUBJECTS, ids=SUBJECT_IDS)
def test_a_static_over_long_first_call_is_refused_before_any_variable_exists(subject):
    model = subject.factory()
    limit = _limit(subject, model)
    with pytest.raises(ValueError, match=subject.limit_name):
        model(subject.feed(model, limit + OVER_BY))
    assert model.weights == [], [w.path for w in model.weights]


@pytest.mark.parametrize("subject", SUBJECTS, ids=SUBJECT_IDS)
def test_a_refused_first_predict_leaves_the_model_usable(subject):
    model = subject.factory()
    limit = _limit(subject, model)
    with pytest.raises(ValueError, match=subject.limit_name):
        model.predict(subject.feed(model, limit + OVER_BY), verbose=0)
    for seq_len in LEGAL_LENGTHS:
        out = model.predict(subject.feed(model, seq_len), verbose=0)
        assert _finite(out)


@pytest.mark.parametrize("subject", SUBJECTS, ids=SUBJECT_IDS)
def test_the_declared_limit_itself_is_legal(subject):
    model = subject.factory()
    limit = _limit(subject, model)
    assert _finite(model.predict(subject.feed(model, limit), verbose=0))


@pytest.mark.parametrize("config", _JIT_CONFIGS)
@pytest.mark.parametrize("subject", SUBJECTS, ids=SUBJECT_IDS)
def test_a_relaxed_retrace_refuses_an_over_long_batch_and_stays_usable(
    monkeypatch, subject, config
):
    import tensorflow as tf

    expected_jit = _resolve_jit(monkeypatch, config)
    model = subject.factory()
    limit = _limit(subject, model)
    if config == "jit_compile_true":
        model.jit_compile = True
    for seq_len in LEGAL_LENGTHS:
        assert _finite(model.predict(subject.feed(model, seq_len), verbose=0))
    assert model.jit_compile is expected_jit, (config, model.jit_compile)

    pattern = (
        rf"seq_len_exceeds_{subject.limit_name}_{limit}(?![0-9])"
        if expected_jit
        else subject.message_regex.format(limit=limit)
    )
    with pytest.raises(
        (ValueError, tf.errors.InvalidArgumentError), match=pattern
    ):
        model.predict(subject.feed(model, limit + OVER_BY), verbose=0)
    # And the model still serves a legal length afterwards.
    assert _finite(model.predict(subject.feed(model, limit), verbose=0))


@pytest.mark.parametrize("subject", SUBJECTS, ids=SUBJECT_IDS)
def test_the_model_reaches_the_shared_guard(monkeypatch, subject):
    """Both shared functions are called by the module-level names the model imports."""
    module = importlib.import_module(subject.module)
    calls = {"refuse": 0, "gate": 0}
    real_refuse = module.refuse_static_sequence_length
    real_gate = module.gate_dynamic_sequence_length

    def refuse(*args, **kwargs):
        calls["refuse"] += 1
        return real_refuse(*args, **kwargs)

    def gate(*args, **kwargs):
        calls["gate"] += 1
        return real_gate(*args, **kwargs)

    # `raising=True` (the default): a model that stopped importing the shared
    # name would make this attribute lookup fail instead of passing vacuously.
    monkeypatch.setattr(module, "refuse_static_sequence_length", refuse)
    monkeypatch.setattr(module, "gate_dynamic_sequence_length", gate)

    model = subject.factory()
    limit = _limit(subject, model)
    with pytest.raises(ValueError):
        model(subject.feed(model, limit + OVER_BY))
    assert calls["refuse"] >= 1, calls
    # Two lengths make Keras retrace `predict` with a relaxed (None) axis.
    for seq_len in LEGAL_LENGTHS:
        model.predict(subject.feed(model, seq_len), verbose=0)
    assert calls["gate"] >= 1, calls
