"""A vision classifier built on the shared ``ClassificationHead`` returns bare logits.

Census guard for plan-2026-10-14T062616-81938e81 (decisions D-002, D-010). Until
that plan the vision ``ClassificationHead`` ended in a softmax and returned the
same tensor under ``'logits'`` and ``'probabilities'``; eleven of its model
consumers unwrapped ``['probabilities']`` and seven returned the head's dict.
Each consumer now returns ``head_output['logits']`` as one bare tensor.

One row per consumer class, four arms per row:

``test_the_output_is_one_bare_tensor``
    ``call`` returns a single tensor of shape ``(batch, num_classes)``, not a
    dict, list or tuple.
``test_the_output_is_not_normalized``
    On a non-degenerate input (unit-normal scaled by 3, batch 3, 5 classes)
    some row misses 1 by more than ``reassociation_atol([C], 1, 1.0)``, the
    float32 rounding bound inside which a softmax row sums to 1. A model that
    re-applies a softmax, or unwraps ``'probabilities'``, sits inside it.
``test_the_symbolic_shape_is_the_eager_shape``
    ``model(keras.Input(...))`` has the eager shape with an unknown batch.
    Keras builds that symbolic call from ``compute_output_shape`` when the
    class overrides it and by tracing ``call`` otherwise, so the arm reaches a
    hand-written shape method wherever one exists. Where one exists it is also
    compared directly. ResNet, AlexNet and CBAMNet define none; Keras then
    raises ``NotImplementedError`` from ``compute_output_shape``, which reports
    nothing false.
``test_a_stock_from_logits_fit_step_moves_the_classifier``
    ``compile(loss=SparseCategoricalCrossentropy(from_logits=True))`` plus one
    ``fit`` step runs, gives a finite loss and changes the head's classifier
    kernel. ``pyproject.toml`` turns every ``UserWarning`` into an error, so
    Keras' "``from_logits=True`` but the output came from a softmax" warning
    fails this arm too.

Adding a consumer is one entry in :data:`SUBJECTS`.
"""

from typing import Callable, Dict, Tuple

import keras
import numpy as np
import pytest

from tests.numerics import reassociation_atol

NUM_CLASSES = 5
BATCH = 3


def _resnet() -> keras.Model:
    from dl_techniques.models.vision.resnet import ResNet
    return ResNet(num_classes=NUM_CLASSES, blocks_per_stage=[1, 1],
                  filters_per_stage=[8, 16], block_type="basic",
                  input_shape=(32, 48, 3))


def _convnext_v1() -> keras.Model:
    from dl_techniques.models.vision.convnext import ConvNeXtV1
    return ConvNeXtV1(num_classes=NUM_CLASSES, depths=[1, 1], dims=[8, 16],
                      input_shape=(32, 48, 3))


def _convnext_v2() -> keras.Model:
    from dl_techniques.models.vision.convnext import ConvNeXtV2
    return ConvNeXtV2(num_classes=NUM_CLASSES, depths=[1, 1], dims=[8, 16],
                      input_shape=(32, 48, 3))


def _alexnet() -> keras.Model:
    # Square: AlexNet's flatten width is 256 * extent ** 2 for ONE extent.
    # 67 is the smallest input above MIN_SPATIAL_EXTENT (55) that keeps the
    # fc6 kernel small.
    from dl_techniques.models.vision.alexnet import AlexNet
    return AlexNet(num_classes=NUM_CLASSES, input_shape=(67, 67, 3))


def _cbam() -> keras.Model:
    from dl_techniques.models.vision.cbam import CBAMNet
    return CBAMNet(num_classes=NUM_CLASSES, dims=[8, 16],
                   input_shape=(32, 48, 3))


#: class name -> (builder, per-sample input shape).
SUBJECTS: Dict[str, Tuple[Callable[[], keras.Model], Tuple[int, int, int]]] = {
    "ResNet": (_resnet, (32, 48, 3)),
    "ConvNeXtV1": (_convnext_v1, (32, 48, 3)),
    "ConvNeXtV2": (_convnext_v2, (32, 48, 3)),
    "AlexNet": (_alexnet, (67, 67, 3)),
    "CBAMNet": (_cbam, (32, 48, 3)),
}


def _images(shape: Tuple[int, int, int], seed: int = 0) -> np.ndarray:
    """A non-degenerate batch: unit-normal scaled by 3, fixed seed.

    :param shape: Per-sample input shape.
    :type shape: Tuple[int, int, int]
    :param seed: RNG seed.
    :type seed: int
    :return: ``(BATCH, *shape)`` float32.
    :rtype: np.ndarray
    """
    rng = np.random.default_rng(seed)
    return (3.0 * rng.standard_normal((BATCH, *shape))).astype("float32")


def _build(name: str) -> Tuple[keras.Model, Tuple[int, int, int]]:
    builder, shape = SUBJECTS[name]
    keras.utils.set_random_seed(0)
    return builder(), shape


@pytest.fixture(params=sorted(SUBJECTS))
def subject(request) -> str:
    return request.param


# ---------------------------------------------------------------------


def test_the_output_is_one_bare_tensor(subject: str) -> None:
    model, shape = _build(subject)
    out = model(_images(shape), training=False)
    assert not isinstance(out, (dict, list, tuple)), (
        f"{subject} returned {type(out).__name__}; a classifier returns its "
        f"logits as one tensor"
    )
    assert tuple(out.shape) == (BATCH, NUM_CLASSES), (
        f"{subject} returned shape {tuple(out.shape)}"
    )


def test_the_output_is_not_normalized(subject: str) -> None:
    model, shape = _build(subject)
    out = keras.ops.convert_to_numpy(model(_images(shape), training=False))
    assert np.isfinite(out).all(), f"{subject} produced non-finite logits"
    atol = reassociation_atol([NUM_CLASSES], 1, 1.0)
    rows = out.astype(np.float64).sum(axis=-1)
    assert np.any(np.abs(rows - 1.0) > atol), (
        f"{subject}: every output row sums to 1 within {atol:.2e} ({rows}); "
        f"the model returns probabilities, not logits"
    )


def test_the_symbolic_shape_is_the_eager_shape(subject: str) -> None:
    model, shape = _build(subject)
    eager = tuple(model(_images(shape), training=False).shape)

    symbolic = model(keras.Input(shape))
    assert not isinstance(symbolic, (dict, list, tuple)), (
        f"{subject}: the symbolic call returned {type(symbolic).__name__}"
    )
    assert tuple(symbolic.shape) == (None, *eager[1:]), (
        f"{subject}: symbolic {tuple(symbolic.shape)} vs eager {eager}"
    )

    overrides = type(model).compute_output_shape is not keras.Model.compute_output_shape
    if overrides:
        reported = model.compute_output_shape((BATCH, *shape))
        assert tuple(reported) == eager, (
            f"{subject}.compute_output_shape reports {reported}, call "
            f"returns {eager}"
        )


def test_a_stock_from_logits_fit_step_moves_the_classifier(subject: str) -> None:
    model, shape = _build(subject)
    x = _images(shape, seed=1)
    y = np.arange(BATCH) % NUM_CLASSES
    model(x[:1], training=False)
    kernel = model.classification_head.classifier.kernel
    before = keras.ops.convert_to_numpy(kernel).copy()

    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=0.1),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )
    history = model.fit(x, y, batch_size=BATCH, epochs=1, verbose=0)

    loss = history.history["loss"][0]
    assert np.isfinite(loss), f"{subject}: loss {loss}"
    after = keras.ops.convert_to_numpy(kernel)
    assert not np.array_equal(before, after), (
        f"{subject}: one fit step left the classifier kernel unchanged"
    )
