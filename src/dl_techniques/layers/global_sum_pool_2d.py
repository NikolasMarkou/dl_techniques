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

    Args:
        keepdims: Boolean, whether to keep the spatial dimensions as size 1.
            If `True`, the output will have shape `(batch_size, 1, 1, channels)`.
            If `False`, the output will have shape `(batch_size, channels)`.
            Defaults to `False`.
        data_format: String, either `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs. `"channels_last"`
            corresponds to inputs with shape `(batch, height, width, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch, channels, height, width)`. It defaults to the
            `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json`. If you never set it, then it will be
            `"channels_last"`.
        **kwargs: Additional keyword arguments to pass to the Layer base class.

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
        >>> import numpy as np
        >>> x = np.random.rand(4, 32, 32, 64)
        >>> layer = GlobalSumPooling2D()
        >>> output = layer(x)
        >>> print(output.shape)
        (4, 64)

        >>> layer_keepdims = GlobalSumPooling2D(keepdims=True)
        >>> output_keepdims = layer_keepdims(x)
        >>> print(output_keepdims.shape)
        (4, 1, 1, 64)
    """

    def __init__(
            self,
            keepdims: bool = False,
            data_format: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.keepdims = keepdims
        self.data_format = keras.backend.image_data_format()

        # Will be set in build()
        self._build_input_shape = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer."""
        # Store input shape for serialization
        self._build_input_shape = input_shape
        super().build(input_shape)

    def call(self, inputs: Any, training: Optional[bool] = None) -> Any:
        """Forward pass."""
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

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer."""
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
        """Get the layer configuration."""
        config = super().get_config()
        config.update({
            "keepdims": self.keepdims,
            "data_format": self.data_format,
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get the build configuration."""
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build the layer from configuration."""
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

# ---------------------------------------------------------------------
