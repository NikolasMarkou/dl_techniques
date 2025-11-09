"""
Pool features globally by summing over spatial dimensions.

This layer performs a global reduction operation on spatial feature maps,
transforming a 4D tensor `(batch, height, width, channels)` into a 2D
tensor `(batch, channels)` by summing all elements across the spatial
dimensions (height and width) for each channel independently.

Architectural and Mathematical Foundations:
Global pooling layers are essential components in modern convolutional neural
networks, typically used near the output to aggregate spatial features into a
fixed-size vector representation before a final classification or regression
head. While `GlobalAveragePooling2D` and `GlobalMaxPooling2D` are more
common, `GlobalSumPooling2D` serves a distinct and critical purpose.

The mathematical operation for each channel `c` in the output vector `y` is:
    `y_c = Σ_{h=1..H, w=1..W} x_{h,w,c}`
where `x` is the input feature map of shape `(H, W)`.

The key intuition lies in what this summation represents. Unlike average
pooling, which normalizes by the spatial area and measures the average
feature presence, or max pooling, which identifies the peak feature response,
sum pooling measures the *total magnitude* or *integral* of the feature
activation across the entire spatial map.

This property makes it uniquely suited for tasks where the total quantity of
a feature is more important than its average intensity or peak presence. A
prominent application is in object counting and density estimation. For
instance, if a convolutional channel is trained to activate in the presence
of a specific object (e.g., a cell), the sum of its activation map provides
a direct estimate of the total number of objects in the image. The model
effectively learns to produce a density map, where the integral (sum)
corresponds to the count.

By preserving the total activation, this layer avoids the ambiguity present
in average pooling, where a large, weakly-activated area can produce the same
output as a small, strongly-activated area.

References:
    - Lempitsky, V. and Zisserman, A. "Learning To Count Objects in Images".
      While this paper uses density maps and integral images, the core concept
      of summing a feature map to obtain a count is foundational to the use
      case for global sum pooling.
      https://www.robots.ox.ac.uk/~vgg/publications/2010/Lempitsky10/
"""

import keras
from keras import ops
from typing import Optional, Any, Dict, Tuple

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class GlobalSumPooling2D(keras.layers.Layer):
    """Global sum pooling operation for 2D spatial data.

    This layer performs global sum pooling on 2D spatial data by summing over
    the spatial dimensions (height and width). For 4D tensors, it reduces from
    shape `(batch_size, height, width, channels)` to `(batch_size, channels)`
    or `(batch_size, 1, 1, channels)` if `keepdims=True`.

    Unlike average or max pooling, sum pooling preserves the total magnitude
    of activations across spatial dimensions, making it useful for tasks where
    the total activation strength is meaningful (e.g., object counting, density
    estimation).

    Args:
        keepdims: Boolean, whether to keep the spatial dimensions as size 1.
            If `True`, the output will have shape `(batch_size, 1, 1, channels)`
            for channels_last format. If `False`, the output will have shape
            `(batch_size, channels)`. Defaults to `False`.
        data_format: String, either `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs. `"channels_last"`
            corresponds to inputs with shape `(batch, height, width, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch, channels, height, width)`. If `None`, uses the default
            `image_data_format` from Keras config. Defaults to `None`.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        - If `data_format="channels_last"`:
            4D tensor with shape `(batch_size, height, width, channels)`.
        - If `data_format="channels_first"`:
            4D tensor with shape `(batch_size, channels, height, width)`.

    Output shape:
        - If `data_format="channels_last"` and `keepdims=False`:
            2D tensor with shape `(batch_size, channels)`.
        - If `data_format="channels_last"` and `keepdims=True`:
            4D tensor with shape `(batch_size, 1, 1, channels)`.
        - If `data_format="channels_first"` and `keepdims=False`:
            2D tensor with shape `(batch_size, channels)`.
        - If `data_format="channels_first"` and `keepdims=True`:
            4D tensor with shape `(batch_size, channels, 1, 1)`.

    Returns:
        A tensor with the spatial dimensions summed out.

    Example:
        ```python
        # Basic usage
        inputs = keras.Input(shape=(32, 32, 64))
        layer = GlobalSumPooling2D()
        outputs = layer(inputs)
        print(outputs.shape)  # (batch_size, 64)

        # Keep spatial dimensions
        layer_keepdims = GlobalSumPooling2D(keepdims=True)
        outputs_keepdims = layer_keepdims(inputs)
        print(outputs_keepdims.shape)  # (batch_size, 1, 1, 64)

        # Channels first format
        inputs_cf = keras.Input(shape=(64, 32, 32))
        layer_cf = GlobalSumPooling2D(data_format="channels_first")
        outputs_cf = layer_cf(inputs_cf)
        print(outputs_cf.shape)  # (batch_size, 64)
        ```

    Note:
        This layer is particularly useful for tasks where the total activation
        magnitude matters, such as object counting or when you want to preserve
        the sum of feature responses across spatial locations.
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

        Args:
            inputs: Input tensor of rank 4.
            training: Boolean indicating training mode (unused for this layer).

        Returns:
            Tensor with spatial dimensions summed out.
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

        Args:
            input_shape: Shape tuple of the input tensor.

        Returns:
            Output shape tuple.
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

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "keepdims": self.keepdims,
            "data_format": self.data_format,
        })
        return config

# ---------------------------------------------------------------------