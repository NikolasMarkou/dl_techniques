"""A language model's ``compute_output_shape`` reports what its ``call`` returns.

Subject: ``compute_output_shape`` of the eight classes that had none
(DistilBERT, ModernBERT, FNet, Mamba, Mamba2, Qwen3, Qwen3Next, Gemma3;
plan-2026-10-11T170958-dc4d2202 step 8a; Keras guide 3.4). Keras builds a
symbolic call of a layer that overrides ``compute_output_shape`` from that
method, never from ``call``, so a wrong answer there is a wrong functional graph
(and the headed branch was never shape-checked at all).

``test_the_reported_shape_is_the_shape_call_returns``
    Run ``call`` for real and compare the structure and every shape with
    ``compute_output_shape`` given the ids shape, headless, with a
    classification head (a bare tensor) and with a question-answering head (a
    dict of two). The dict-of-shapes input form is accepted too.
``test_a_symbolic_call_gets_the_shape_call_returns``
    ``model(keras.Input((LENGTH,)))`` is built by Keras from
    ``compute_output_shape`` (batch unknown; the length is static because FNet
    builds its Fourier matrix for one known length), and its tensors must have
    the shapes a real ``call`` gives, with the batch axis unknown. This is the
    arm that shows a wrong answer reaching a functional graph.

Adding a class is one name in ``SUBJECTS``.
"""

import keras
import numpy as np
import pytest

from dl_techniques.layers.heads.nlp import NLPTaskConfig, NLPTaskType

from .tiny_language_models import TINY, VOCAB

# ---------------------------------------------------------------------

SUBJECTS = ("distilbert", "modern_bert", "fnet", "mamba", "mamba2", "qwen3",
            "qwen3_next", "gemma3")

TASKS = {
    "headless": None,
    "classification": NLPTaskConfig(
        name="clf", task_type=NLPTaskType.TEXT_CLASSIFICATION, num_classes=3),
    "question_answering": NLPTaskConfig(
        name="qa", task_type=NLPTaskType.QUESTION_ANSWERING),
}

BATCH, LENGTH = 2, 6


def _build(name, task):
    tiny = TINY[name]
    extra = {} if TASKS[task] is None else dict(
        include_top=True, task_config=TASKS[task])
    return tiny.cls()(**tiny.kwargs, **extra)


def _shapes(structure):
    """Tensors (or shapes) to a comparable structure of shape tuples."""
    if isinstance(structure, dict):
        return {key: _shapes(value) for key, value in structure.items()}
    shape = getattr(structure, "shape", structure)
    return tuple(shape)


def _unknown_batch(shapes):
    """The same structure with every leading (batch) axis replaced by None."""
    if isinstance(shapes, dict):
        return {key: _unknown_batch(value) for key, value in shapes.items()}
    return (None, *shapes[1:])


@pytest.fixture(params=SUBJECTS)
def subject(request):
    return request.param


@pytest.fixture(params=sorted(TASKS))
def task(request):
    return request.param


def test_the_reported_shape_is_the_shape_call_returns(subject, task):
    model = _build(subject, task)
    ids = np.random.default_rng(0).integers(
        1, VOCAB, (BATCH, LENGTH)).astype("int32")
    actual = _shapes(model(ids, training=False))

    assert _shapes(model.compute_output_shape((BATCH, LENGTH))) == actual
    as_dict = {"input_ids": (BATCH, LENGTH), "attention_mask": (BATCH, LENGTH)}
    assert _shapes(model.compute_output_shape(as_dict)) == actual


def test_a_symbolic_call_gets_the_shape_call_returns(subject, task):
    model = _build(subject, task)
    ids = np.random.default_rng(0).integers(
        1, VOCAB, (BATCH, LENGTH)).astype("int32")
    actual = _shapes(model(ids, training=False))
    symbolic = _shapes(model(keras.Input(shape=(LENGTH,), dtype="int32")))
    assert symbolic == _unknown_batch(actual)
