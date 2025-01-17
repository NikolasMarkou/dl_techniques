"""
Batch Conditional Output Layer Module.

This module implements a custom Keras layer that conditionally selects outputs
based on batch-wise ground truth values.

Example:
    >>> layer = BatchConditionalOutputLayer()
    >>> output = layer([ground_truth, inference])
"""

import keras
import numpy as np
import tensorflow as tf
from typing import List, Tuple, Optional, Union, Sequence


@tf.keras.utils.register_keras_serializable(package="custom_layers")
class BatchConditionalOutputLayer(keras.layers.Layer):
    """
    A custom Keras layer for conditional output selection based on ground truth values.

    This layer implements a batch-wise selection mechanism where for each item in the batch:
    - If the ground truth tensor item contains all zeros, output the inference tensor item
    - Otherwise, output the ground truth tensor item

    The layer maintains the same shape and dtype as the input tensors.

    Attributes:
        supports_masking (bool): Indicates if the layer supports masking.

    Note:
        Both input tensors must have identical shapes and compatible dtypes.
    """

    def __init__(
            self,
            name: Optional[str] = None,
            **kwargs: dict
    ) -> None:
        """
        Initialize the BatchConditionalOutputLayer.

        Args:
            name: Optional name for the layer
            **kwargs: Additional layer arguments

        """
        super().__init__(name=name, **kwargs)
        self.supports_masking = True

    def build(
            self,
            input_shape: List[tf.TensorShape]
    ) -> None:
        """
        Build the layer.

        Args:
            input_shape: List of shapes for [ground_truth, inference]

        Raises:
            ValueError: If input shapes don't match or aren't valid
        """
        if len(input_shape) != 2:
            raise ValueError(
                f"Layer expects exactly 2 input shapes, got {len(input_shape)}"
            )

        if input_shape[0] != input_shape[1]:
            raise ValueError(
                f"Input shapes must match. Got {input_shape[0]} and {input_shape[1]}"
            )

        super().build(input_shape)

    def call(
            self,
            inputs: Sequence[tf.Tensor],
            training: Optional[bool] = None,
            mask: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        """
        Forward pass of the layer.

        Args:
            inputs: List containing [ground_truth, inference] tensors
            training: Boolean indicating training mode (unused)
            mask: Optional tensor for masking (unused)

        Returns:
            tf.Tensor: Output tensor with values selected from either ground truth
                      or inference based on ground truth values

        Raises:
            ValueError: If inputs aren't valid or compatible
            TypeError: If inputs aren't tensors
        """
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError(f"Layer expects list of 2 tensors, got {inputs}")

        ground_truth, inference = inputs

        # Type checking
        if not (isinstance(ground_truth, tf.Tensor) and
                isinstance(inference, tf.Tensor)):
            raise TypeError("Inputs must be TensorFlow tensors")

        # Shape validation
        if ground_truth.shape != inference.shape:
            raise ValueError(
                f"Input shapes must match. Got {ground_truth.shape} "
                f"and {inference.shape}"
            )

        # Check for all zeros per batch item
        # Reduce across all dimensions except batch (axis 0)
        reduction_axes = list(range(1, len(ground_truth.shape)))
        is_zeros = tf.reduce_all(
            tf.equal(ground_truth, 0),
            axis=reduction_axes
        )

        # Reshape mask for broadcasting
        broadcast_shape = [-1] + [1] * (len(ground_truth.shape) - 1)
        is_zeros = tf.reshape(is_zeros, broadcast_shape)

        # Select output values using mask
        return tf.where(is_zeros, inference, ground_truth)

    def compute_output_shape(
            self,
            input_shape: List[Tuple[Optional[int], ...]]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute the layer's output shape.

        Args:
            input_shape: List of shapes for [ground_truth, inference]

        Returns:
            Tuple representing output shape (same as input shapes)

        Raises:
            ValueError: If input shapes aren't valid
        """
        if len(input_shape) != 2:
            raise ValueError(
                f"Expected 2 input shapes, got {len(input_shape)}"
            )

        if input_shape[0] != input_shape[1]:
            raise ValueError(
                f"Input shapes must match. Got {input_shape[0]} "
                f"and {input_shape[1]}"
            )

        return input_shape[0]

    def get_config(self) -> dict:
        """
        Get layer configuration.

        Returns:
            dict: Layer configuration dictionary
        """
        return super().get_config()

