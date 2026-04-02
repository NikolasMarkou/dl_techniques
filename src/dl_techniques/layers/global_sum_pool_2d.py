"""
Pool features globally by summing over spatial dimensions.

Performs global sum pooling on 2D spatial feature maps, reducing a 4D tensor
(batch, H, W, C) to (batch, C) by summing over spatial dimensions per channel:
y_c = sum_{h,w} x_{h,w,c}. Unlike average pooling (which normalizes by area)
or max pooling (which identifies peak response), sum pooling measures the total
magnitude of feature activation, making it suited for tasks where total feature
quantity matters such as object counting and density estimation.

References:
    - Lempitsky, V. and Zisserman, A. "Learning To Count Objects in Images".
      https://www.robots.ox.ac.uk/~vgg/publications/2010/Lempitsky10/
"""

import keras
from keras import ops
from typing import Optional, Any, Dict, Tuple

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class GlobalSumPooling2D(keras.layers.Layer):
    """
    Global sum pooling operation for 2D spatial data.

    Sums over spatial dimensions (height and width) to reduce 4D tensors to 2D
    channel descriptors: y_c = sum_{h=1..H, w=1..W} x_{h,w,c}. Preserves the
    total activation magnitude rather than averaging or taking the maximum,
    making it useful for object counting and density estimation where the
    integral of a learned density map corresponds to a count.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────────────┐
        │  Input [batch, H, W, C]                  │
        └──────────────────┬───────────────────────┘
                           ▼
        ┌──────────────────────────────────────────┐
        │  Sum over spatial dims (H, W)            │
        │  y_c = Σ_{h,w} x_{h,w,c}                 │
        └──────────────────┬───────────────────────┘
                           ▼
        ┌──────────────────────────────────────────┐
        │  Output [batch, C]                       │
        │  (or [batch, 1, 1, C] if keepdims)       │
        └──────────────────────────────────────────┘

    :param keepdims: Whether to keep the spatial dimensions as size 1.
        Defaults to False.
    :type keepdims: bool
    :param data_format: Either "channels_last" or "channels_first". If None,
        uses Keras default. Defaults to None.
    :type data_format: Optional[str]
    :param kwargs: Additional keyword arguments for the Layer base class.
    """

    def __init__(
            self,
            keepdims: bool = False,
            data_format: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Store configuration parameters
        self.keepdims = keepdims

        # Set data format - use provided value or default from Keras config
        if data_format is None:
            data_format = keras.backend.image_data_format()

        # Validate data format
        if data_format not in ("channels_last", "channels_first"):
            raise ValueError(
                f"data_format must be 'channels_last' or 'channels_first', "
                f"got {data_format}"
            )

        self.data_format = data_format

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass computation.

        :param inputs: Input tensor of rank 4.
        :type inputs: keras.KerasTensor
        :param training: Boolean indicating training mode (unused).
        :type training: Optional[bool]
        :return: Tensor with spatial dimensions summed out.
        :rtype: keras.KerasTensor
        """
        # Determine which axes to sum over based on data format
        if self.data_format == "channels_last":
            # For channels_last: (batch, height, width, channels)
            # Sum over height (axis 1) and width (axis 2)
            sum_axes = [1, 2]
        else:
            # For channels_first: (batch, channels, height, width)
            # Sum over height (axis 2) and width (axis 3)
            sum_axes = [2, 3]

        return ops.sum(inputs, axis=sum_axes, keepdims=self.keepdims)

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple.
        :rtype: Tuple[Optional[int], ...]
        """
        # Convert to list for manipulation
        input_shape_list = list(input_shape)

        if self.data_format == "channels_last":
            # Input: (batch, height, width, channels)
            if self.keepdims:
                # Output: (batch, 1, 1, channels)
                output_shape_list = [input_shape_list[0], 1, 1, input_shape_list[3]]
            else:
                # Output: (batch, channels)
                output_shape_list = [input_shape_list[0], input_shape_list[3]]
        else:
            # Input: (batch, channels, height, width)
            if self.keepdims:
                # Output: (batch, channels, 1, 1)
                output_shape_list = [input_shape_list[0], input_shape_list[1], 1, 1]
            else:
                # Output: (batch, channels)
                output_shape_list = [input_shape_list[0], input_shape_list[1]]

        return tuple(output_shape_list)

    def get_config(self) -> Dict[str, Any]:
        """Get the layer configuration for serialization.

        :return: Dictionary containing the layer configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "keepdims": self.keepdims,
            "data_format": self.data_format,
        })
        return config

# ---------------------------------------------------------------------
