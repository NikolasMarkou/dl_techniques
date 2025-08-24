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