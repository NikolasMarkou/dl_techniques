"""Deep-supervision plumbing helpers.

Utilities for working with multi-output training models that emit auxiliary
predictions for deep supervision. The primary inference output is always at
index 0; auxiliary outputs follow.

Both helpers accept Functional models (whose ``.input``/``.output`` attributes
Keras populates at construction time) and subclassed ``keras.Model`` instances
(for which Keras never populates them). For a subclassed model the caller must
supply ``input_shape``, which is used to trace the model symbolically.
"""

from typing import Any, Dict, Optional, Tuple

import keras

from dl_techniques.utils.logger import logger


def _trace_symbolic_outputs(
    model: keras.Model,
    input_shape: Optional[Tuple[int, ...]],
) -> Tuple[keras.KerasTensor, Any]:
    """Symbolically trace a subclassed model to recover its input/output structure.

    Keras 3 only populates ``model.input``/``model.output`` for Functional models.
    For a subclassed model the structure has to be recovered by calling the model
    on a symbolic placeholder; this is side-effect free (no BatchNorm moving
    statistic is updated, no loss is added) because no eager op runs.

    :param model: Subclassed model to trace.
    :param input_shape: Per-sample input shape (no batch dimension).
    :return: Tuple of ``(symbolic_input, symbolic_output)``, where the output is
        whatever structure the model's ``call`` returns (a list for a
        deep-supervised model, a single tensor otherwise).
    :raises ValueError: If ``input_shape`` is ``None``.
    """
    # DECISION plan-2026-08-23T203721-009b7ccf/D-001
    # Do NOT read the shape off the model instead of taking it as a parameter
    # (e.g. ResNet's `input_shape_config`): that couples this generic helper to
    # one model class's attribute name and every other subclassed consumer would
    # need its own special case. Do NOT do an eager forward pass to populate the
    # structure either -- a real call runs BatchNorm moving-statistic updates and
    # `add_loss` side effects for real, and (measured) it does not populate
    # `.output` anyway: Keras 3 raises the same AttributeError before AND after
    # an eager call. See decisions.md D-001.
    if input_shape is None:
        raise ValueError(
            f"{type(model).__name__} is a subclassed keras.Model, so Keras never "
            f"populates its `.input`/`.output` attributes. Pass the `input_shape` "
            f"keyword (the per-sample shape, without the batch dimension) so the "
            f"model can be traced symbolically."
        )

    symbolic_input = keras.Input(shape=input_shape)
    return symbolic_input, model(symbolic_input)


def _output_info(outputs: Any) -> Dict[str, Any]:
    """Derive output metadata from a model's output structure.

    :param outputs: A model's output tensor, or a list of them.
    :return: Dict with ``num_outputs``, ``has_deep_supervision``,
        ``output_shapes``, and ``primary_output_index`` (always 0).
    """
    if isinstance(outputs, list):
        num_outputs = len(outputs)
        output_shapes = [output.shape for output in outputs]
        has_deep_supervision = True
    else:
        num_outputs = 1
        output_shapes = [outputs.shape]
        has_deep_supervision = False

    return {
        "num_outputs": num_outputs,
        "has_deep_supervision": has_deep_supervision,
        "output_shapes": output_shapes,
        "primary_output_index": 0,
    }


def _resolve_outputs(
    model: keras.Model,
    input_shape: Optional[Tuple[int, ...]],
) -> Tuple[Any, Any]:
    """Return a model's ``(inputs, outputs)``, tracing it symbolically if needed.

    :param model: Functional or subclassed Keras model.
    :param input_shape: Per-sample input shape (no batch dimension). Required
        only for a subclassed model; ignored for a Functional one.
    :return: Tuple of ``(inputs, outputs)``.
    :raises ValueError: If ``model`` is subclassed and ``input_shape`` is ``None``.
    """
    try:
        return model.input, model.output
    except AttributeError:
        return _trace_symbolic_outputs(model, input_shape)


def get_model_output_info(
    model: keras.Model,
    input_shape: Optional[Tuple[int, ...]] = None,
) -> Dict[str, Any]:
    """Return output metadata for a (possibly deep-supervised) model.

    :param model: Keras model to analyze.
    :param input_shape: Per-sample input shape (no batch dimension). Required
        only when ``model`` is a subclassed ``keras.Model``, whose ``.output``
        Keras never populates; ignored for Functional models.
    :return: Dict with ``num_outputs``, ``has_deep_supervision``,
        ``output_shapes``, and ``primary_output_index`` (always 0).
    :raises ValueError: If ``model`` is subclassed and ``input_shape`` is ``None``.
    """
    _, outputs = _resolve_outputs(model, input_shape)
    return _output_info(outputs)


def create_inference_model_from_training_model(
    training_model: keras.Model,
    input_shape: Optional[Tuple[int, ...]] = None,
) -> keras.Model:
    """Build a single-output inference model from a deep-supervised training model.

    :param training_model: Multi-output training model.
    :param input_shape: Per-sample input shape (no batch dimension). Required
        only when ``training_model`` is a subclassed ``keras.Model``; ignored for
        Functional models.
    :return: Single-output model exposing only the primary output (index 0).
        Returned unchanged if it already has a single output.
    :raises ValueError: If ``training_model`` is subclassed and ``input_shape``
        is ``None``.
    """
    inputs, outputs = _resolve_outputs(training_model, input_shape)
    model_info = _output_info(outputs)

    if not model_info["has_deep_supervision"]:
        logger.info("Model already has single output, returning as-is")
        return training_model

    primary_output = outputs[model_info["primary_output_index"]]

    inference_model = keras.Model(
        inputs=inputs,
        outputs=primary_output,
        name=f"{training_model.name}_inference",
    )

    logger.info(
        f"Created inference model with single output shape: {primary_output.shape}"
    )

    return inference_model
