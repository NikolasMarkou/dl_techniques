"""Deep-supervision plumbing helpers.

Utilities for working with multi-output training models that emit auxiliary
predictions for deep supervision. The primary inference output is always at
index 0; auxiliary outputs follow.
"""

from typing import Any, Dict

import keras

from dl_techniques.utils.logger import logger


def get_model_output_info(model: keras.Model) -> Dict[str, Any]:
    """Return output metadata for a (possibly deep-supervised) model.

    :param model: Keras model to analyze.
    :return: Dict with ``num_outputs``, ``has_deep_supervision``,
        ``output_shapes``, and ``primary_output_index`` (always 0).
    """
    if isinstance(model.output, list):
        num_outputs = len(model.output)
        output_shapes = [output.shape for output in model.output]
        has_deep_supervision = True
    else:
        num_outputs = 1
        output_shapes = [model.output.shape]
        has_deep_supervision = False

    return {
        "num_outputs": num_outputs,
        "has_deep_supervision": has_deep_supervision,
        "output_shapes": output_shapes,
        "primary_output_index": 0,
    }


def create_inference_model_from_training_model(
    training_model: keras.Model,
) -> keras.Model:
    """Build a single-output inference model from a deep-supervised training model.

    :param training_model: Multi-output training model.
    :return: Single-output model exposing only the primary output (index 0).
        Returned unchanged if it already has a single output.
    """
    model_info = get_model_output_info(training_model)

    if not model_info["has_deep_supervision"]:
        logger.info("Model already has single output, returning as-is")
        return training_model

    primary_output = training_model.output[model_info["primary_output_index"]]

    inference_model = keras.Model(
        inputs=training_model.input,
        outputs=primary_output,
        name=f"{training_model.name}_inference",
    )

    logger.info(
        f"Created inference model with single output shape: {primary_output.shape}"
    )

    return inference_model
