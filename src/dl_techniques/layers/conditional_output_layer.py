"""
ConditionalOutputLayer, a per-sample switch between two same-shape tensors.

For each sample in the batch, the layer inspects `ground_truth`. If every
element of that sample is zero, the matching `inference` sample is routed to
the output; otherwise the `ground_truth` sample is used. Feeding this output
into a loss function masks the loss for labeled samples automatically: when
the loss target is also `ground_truth`, a labeled sample scores `L(gt, gt) = 0`
and stops contributing gradient, while an unlabeled (all-zero) sample trains
through the inference path. This lets one batch mix labeled and unlabeled
samples under a single loss call, useful for semi-supervised training and for
inpainting-style tasks where only the missing region should contribute loss.

The layer takes exactly two same-shape tensors, `[ground_truth, inference]`,
and has no learnable parameters.
"""

import keras
from typing import List, Tuple, Optional, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.conditional_output_layer")
class ConditionalOutputLayer(keras.layers.Layer):
    """Batch-wise conditional tensor selector for semi-supervised training.

    For each sample in the batch, inspects ``ground_truth``: if every element
    is zero the corresponding ``inference`` sample is routed to output;
    otherwise the ``ground_truth`` sample is used. This implements
    ``output_i = where(all(gt_i == 0), inference_i, gt_i)``, enabling
    loss masking where labeled samples contribute zero loss via
    ``L(gt, gt) = 0`` while unlabeled samples train through the inference path.

    Architecture:

    .. code-block:: text

        ┌────────────────────┐  ┌──────────────────────┐
        │  ground_truth      │  │  inference           │
        │  (B, ...)          │  │  (B, ...)            │
        └─────────┬──────────┘  └──────────┬───────────┘
                  │                        │
                  ▼                        │
        ┌──────────────────┐               │
        │  all(gt == 0) ?  │               │
        │  per-sample      │               │
        └────┬────────┬────┘               │
             │ True   │ False              │
             ▼        ▼                    ▼
        ┌─────────┐ ┌──────────┐   ┌──────────┐
        │ select  │ │ select   │   │          │
        │ infer   │ │ gt       │   │          │
        └────┬────┘ └────┬─────┘   └──────────┘
             └─────┬─────┘
                   ▼
        ┌──────────────────┐
        │  Output (B, ...) │
        └──────────────────┘

    :param kwargs: Additional keyword arguments for the Layer base class.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the ConditionalOutputLayer.

        :param kwargs: Additional keyword arguments for the Layer base class.
        """
        super().__init__(**kwargs)
        self.supports_masking = True

    def call(
        self,
        inputs: List[keras.KerasTensor],
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Select each batch item from ground_truth or inference.

        :param inputs: List of ``[ground_truth, inference]`` tensors with identical shapes.
        :type inputs: List[keras.KerasTensor]
        :param training: Unused, kept for Layer API consistency.
        :type training: Optional[bool]
        :return: Tensor with each sample selected from ground_truth or inference.
        :rtype: keras.KerasTensor
        """
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError(
                f"BatchConditionalOutputLayer expects a list of exactly 2 tensors, "
                f"got {type(inputs)} with length {len(inputs) if hasattr(inputs, '__len__') else 'unknown'}"
            )

        ground_truth, inference = inputs

        if ground_truth.shape != inference.shape:
            raise ValueError(
                f"Input tensor shapes must match exactly. "
                f"Got ground_truth: {ground_truth.shape}, inference: {inference.shape}"
            )

        reduction_axes = list(range(1, len(ground_truth.shape)))
        is_all_zeros = keras.ops.all(
            keras.ops.equal(ground_truth, 0.0),
            axis=reduction_axes if reduction_axes else None
        )

        # Broadcast the per-sample flag back to the input's rank.
        broadcast_shape = [-1] + [1] * (len(ground_truth.shape) - 1)
        is_all_zeros_broadcasted = keras.ops.reshape(is_all_zeros, broadcast_shape)

        output = keras.ops.where(is_all_zeros_broadcasted, inference, ground_truth)

        return output

    def compute_output_shape(
        self,
        input_shape: List[Tuple[Optional[int], ...]]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape.

        :param input_shape: List of ``[ground_truth, inference]`` input shapes.
        :type input_shape: List[Tuple[Optional[int], ...]]
        :return: Output shape, identical to each input shape.
        :rtype: Tuple[Optional[int], ...]
        """
        if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 2:
            raise ValueError(
                f"Expected list of 2 input shapes, got {type(input_shape)} "
                f"with length {len(input_shape) if hasattr(input_shape, '__len__') else 'unknown'}"
            )

        ground_truth_shape, inference_shape = input_shape

        if ground_truth_shape != inference_shape:
            raise ValueError(
                f"Input shapes must be identical. "
                f"Got ground_truth: {ground_truth_shape}, inference: {inference_shape}"
            )

        return ground_truth_shape

    def get_config(self) -> dict[str, Any]:
        """Return the layer configuration.

        :return: Configuration dictionary for serialization.
        :rtype: dict[str, Any]
        """
        config = super().get_config()

        return config

# ---------------------------------------------------------------------
