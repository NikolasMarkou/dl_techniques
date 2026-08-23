"""Guards for ``dl_techniques.utils.deep_supervision`` on a subclassed model.

``train.resnet.train_resnet.train_resnet_imagenet`` was dead on arrival for
every configuration: it calls ``get_model_output_info(model)`` on a ``ResNet``,
which is a subclassed ``keras.Model``. Keras 3 only populates
``.input``/``.output`` for Functional models, so against the pre-fix code the
exact sequence in ``test_get_model_output_info_works_on_a_subclassed_resnet``
raised (text observed, not predicted, by reverting the fix and re-running)::

    AttributeError: The layer res_net_4 has never been called and thus has no
    defined output.

(the ``_4`` suffix is Keras's per-process instance counter and varies.)

Two measured details worth keeping. First, the ``AttributeError`` is raised
*identically* before and after an **eager** forward pass, so "call the model
first" is not a workaround. Second, a **symbolic** call *does* populate
``.input``/``.output`` on a subclassed model -- which is why the helpers
resolve ``(inputs, outputs)`` exactly once, through ``_resolve_outputs``,
instead of each tracing separately: a second resolution would silently take
the Functional branch off the first trace's residue and leave the fallback
branch dead (measured: an injected wrong-output-index in a
separately-traced fallback was invisible to every arm here).
``create_inference_model_from_training_model`` -- which
``models/resnet/__init__.py`` re-exports publicly -- had the same defect on
both ``.output`` and ``.input``.

Why each arm is not satisfied by construction:

``test_get_model_output_info_works_on_a_subclassed_resnet``
    The RED arm. Reverting either helper to the bare ``model.output`` read
    makes it raise the ``AttributeError`` quoted above.
``test_output_info_reports_both_deep_supervision_branches``
    Pins the *values*, not merely the absence of a raise: a fallback that
    returned a single tensor for the deep-supervised model would pass a
    "does not raise" check and fail this one.
``test_inference_model_matches_the_primary_training_output``
    Pins forward-value equality with ``model(x)[0]``. A helper that built the
    inference model from the wrong output index has the same output *shape*
    (all four ResNet heads emit ``(None, num_classes)``), so only a value
    comparison can see it.
``test_functional_path_is_untouched``
    The invariant that the three Functional consumers (``bfunet``,
    ``bfconvunext``, ``convunext``) are unaffected: with ``input_shape`` NOT
    passed, both helpers must still work off ``model.output``/``model.input``.
``test_subclassed_model_without_input_shape_raises_actionable_value_error``
    Pins the actionable failure, so a caller who forgets the keyword gets a
    message naming the class and the keyword rather than the framework's
    opaque ``AttributeError``.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.resnet import create_resnet
from dl_techniques.utils.deep_supervision import (
    get_model_output_info,
    create_inference_model_from_training_model,
)

# GPU fp32 round-trip invariant used repo-wide (tests/.../test_round_trip.py).
ATOL = 1e-4

INPUT_SHAPE = (32, 32, 3)
NUM_CLASSES = 10


def _resnet(enable_deep_supervision: bool) -> keras.Model:
    return create_resnet(
        variant="resnet18",
        num_classes=NUM_CLASSES,
        input_shape=INPUT_SHAPE,
        enable_deep_supervision=enable_deep_supervision,
    )


def _functional_two_output_model() -> keras.Model:
    inputs = keras.Input(shape=(4,))
    primary = keras.layers.Dense(3, name="primary")(inputs)
    auxiliary = keras.layers.Dense(3, name="auxiliary")(inputs)
    return keras.Model(inputs, [primary, auxiliary], name="functional_ds")


def test_get_model_output_info_works_on_a_subclassed_resnet():
    """RED arm: the exact trainer sequence create_resnet -> summary -> info."""
    model = _resnet(enable_deep_supervision=True)
    model.summary()

    info = get_model_output_info(model, input_shape=INPUT_SHAPE)

    assert info["num_outputs"] == 4
    assert info["primary_output_index"] == 0


@pytest.mark.parametrize(
    "enable_deep_supervision, expected_outputs, expected_flag",
    [(True, 4, True), (False, 1, False)],
)
def test_output_info_reports_both_deep_supervision_branches(
    enable_deep_supervision, expected_outputs, expected_flag
):
    model = _resnet(enable_deep_supervision=enable_deep_supervision)

    info = get_model_output_info(model, input_shape=INPUT_SHAPE)

    assert info["num_outputs"] == expected_outputs
    assert info["has_deep_supervision"] is expected_flag
    assert len(info["output_shapes"]) == expected_outputs
    for shape in info["output_shapes"]:
        assert tuple(shape) == (None, NUM_CLASSES)


def test_inference_model_matches_the_primary_training_output():
    model = _resnet(enable_deep_supervision=True)
    x = np.random.default_rng(0).random((2, *INPUT_SHAPE)).astype("float32")
    training_outputs = model(x, training=False)
    assert isinstance(training_outputs, list) and len(training_outputs) == 4

    inference_model = create_inference_model_from_training_model(
        model, input_shape=INPUT_SHAPE
    )

    assert len(inference_model.outputs) == 1
    assert tuple(inference_model.output.shape) == (None, NUM_CLASSES)

    inference_output = np.asarray(inference_model(x, training=False))
    primary = np.asarray(training_outputs[0])
    assert inference_output.shape == primary.shape
    assert np.max(np.abs(inference_output - primary)) <= ATOL
    # The auxiliary heads are genuinely different tensors, so the value match
    # above is discriminating and not a shape coincidence.
    assert np.max(np.abs(primary - np.asarray(training_outputs[1]))) > ATOL


def test_functional_path_is_untouched():
    """Both helpers must work on a Functional model with no `input_shape`."""
    model = _functional_two_output_model()

    info = get_model_output_info(model)
    assert info == {
        "num_outputs": 2,
        "has_deep_supervision": True,
        "output_shapes": [model.output[0].shape, model.output[1].shape],
        "primary_output_index": 0,
    }

    inference_model = create_inference_model_from_training_model(model)
    assert len(inference_model.outputs) == 1
    assert tuple(inference_model.output.shape) == (None, 3)
    assert inference_model.name == "functional_ds_inference"

    x = np.random.default_rng(1).random((2, 4)).astype("float32")
    expected = np.asarray(model(x, training=False)[0])
    assert np.max(np.abs(np.asarray(inference_model(x, training=False)) - expected)) <= ATOL

    # Single-output Functional model: returned as-is, unchanged.
    single = keras.Model(model.input, model.output[0], name="functional_single")
    assert get_model_output_info(single)["has_deep_supervision"] is False
    assert create_inference_model_from_training_model(single) is single


def test_subclassed_model_without_input_shape_raises_actionable_value_error():
    model = _resnet(enable_deep_supervision=True)

    with pytest.raises(ValueError) as excinfo:
        get_model_output_info(model)
    assert "ResNet" in str(excinfo.value)
    assert "input_shape" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        create_inference_model_from_training_model(model)
    assert "ResNet" in str(excinfo.value)
    assert "input_shape" in str(excinfo.value)
