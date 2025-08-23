"""
This module provides a `MovingStd` layer that applies a 2D moving standard deviation
filter to input images using a sliding window approach.

The layer computes local standard deviation by applying the mathematical formula:
`sqrt(E[X^2] - (E[X])^2)` where E represents the expectation (mean) over a sliding
window. This is efficiently implemented using average pooling operations to compute
both E[X] and E[X^2].

This layer is particularly valuable for:
- **Texture analysis**: Capturing local texture patterns and roughness
- **Edge detection**: Highlighting regions with high local variability  
- **Feature extraction**: Providing variance-based features for classification
- **Noise characterization**: Analyzing spatial noise patterns
- **Medical imaging**: Detecting tissue boundaries and abnormalities

The implementation processes each channel independently, making it suitable for
both grayscale and multi-channel images while maintaining computational efficiency
through vectorized operations.
"""

import keras
from keras import ops
from typing import Tuple, Union, List, Optional, Any, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class MovingStd(keras.layers.Layer):
    """
    Applies a 2D moving standard deviation filter to input images for texture analysis.

    This layer computes the local standard deviation over sliding windows using the
    mathematically stable formula: ``std = sqrt(E[X²] - (E[X])²)``. It leverages
    average pooling operations to efficiently compute expectations, processing each
    channel independently for multi-channel inputs.

    The layer is designed for computer vision applications where local variability
    information is crucial, such as texture classification, edge detection, and
    feature extraction for medical imaging. It provides a differentiable alternative
    to traditional sliding window standard deviation implementations.

    Key computational features:
    - **Efficient implementation**: Uses average pooling for fast expectation computation
    - **Numerical stability**: Includes epsilon term and non-negative variance clamping
    - **Channel independence**: Processes each channel separately for multi-channel inputs
    - **Flexible windowing**: Configurable window size and stride patterns
    - **Memory efficient**: Vectorized operations minimize memory overhead

    Mathematical formulation:
        For a sliding window W over input X:

        - ``μ = E[X] = (1/|W|) Σ(x ∈ W) x`` (local mean)
        - ``σ² = E[X²] - (E[X])² = (1/|W|) Σ(x ∈ W) x² - μ²`` (local variance)  
        - ``σ = sqrt(max(0, σ² + ε))`` (local standard deviation with stability)

    Args:
        pool_size: Tuple[int, int], size of the 2D pooling window as (height, width).
            Larger windows capture broader texture patterns but reduce spatial resolution.
            Must contain exactly 2 positive integers. Defaults to (3, 3).
        strides: Union[Tuple[int, int], List[int]], strides for the pooling operation
            as (height_stride, width_stride). Controls output spatial resolution and
            computational overlap. Must contain exactly 2 positive integers.
            Defaults to (1, 1).
        padding: str, padding mode for the pooling operation. Either:
            - 'valid': No padding, output size depends on window size and strides
            - 'same': Pad input to preserve spatial dimensions when stride=1
            Case-insensitive. Defaults to 'same'.
        data_format: Optional[str], data layout format. Either:
            - 'channels_last': (batch, height, width, channels) - TensorFlow format
            - 'channels_first': (batch, channels, height, width) - PyTorch format
            If None, uses keras.config.image_data_format(). Defaults to None.
        epsilon: float, small positive value added to variance before square root
            to prevent numerical instabilities from floating-point precision errors.
            Should be small but non-zero. Defaults to 1e-7.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        4D tensor with format determined by data_format:
        - If data_format='channels_last': ``(batch_size, height, width, channels)``
        - If data_format='channels_first': ``(batch_size, channels, height, width)``

    Output shape:
        4D tensor with same format as input:
        - If data_format='channels_last': ``(batch_size, new_height, new_width, channels)``
        - If data_format='channels_first': ``(batch_size, channels, new_height, new_width)``

        Where new_height and new_width depend on pool_size, strides, and padding.

    Attributes:
        pooler: keras.layers.AveragePooling2D instance used for computing local means
            and expectations. Configured with the same spatial parameters as the layer.

    Example:
        ```python
        # Basic texture analysis
        layer = MovingStd(pool_size=(5, 5), padding='same')
        inputs = keras.Input(shape=(224, 224, 3))
        texture_features = layer(inputs)

        # Edge detection with larger windows
        edge_detector = MovingStd(
            pool_size=(7, 7),
            strides=(2, 2),  # Downsample for efficiency
            padding='valid'
        )

        # Multi-scale texture analysis
        inputs = keras.Input(shape=(256, 256, 1))
        fine_texture = MovingStd(pool_size=(3, 3))(inputs)
        coarse_texture = MovingStd(pool_size=(9, 9))(inputs)
        combined = keras.layers.Concatenate()([fine_texture, coarse_texture])

        # Medical imaging application
        medical_inputs = keras.Input(shape=(512, 512, 1))
        tissue_boundaries = MovingStd(
            pool_size=(5, 5),
            epsilon=1e-6,  # Higher precision for medical data
            padding='same'
        )(medical_inputs)

        # Complete texture classification model
        model_inputs = keras.Input(shape=(128, 128, 3))
        std_features = MovingStd(pool_size=(7, 7))(model_inputs)
        pooled = keras.layers.GlobalAveragePooling2D()(std_features)
        outputs = keras.layers.Dense(10, activation='softmax')(pooled)
        model = keras.Model(model_inputs, outputs)
        ```

    Raises:
        ValueError: If pool_size is not a tuple/list of exactly 2 positive integers.
        ValueError: If strides is not a tuple/list of exactly 2 positive integers.
        ValueError: If padding is not 'valid' or 'same' (case-insensitive).
        ValueError: If data_format is not 'channels_first' or 'channels_last'.
        ValueError: If epsilon is negative.
        ValueError: If input tensor is not 4-dimensional.

    Note:
        This layer is particularly effective when combined with other texture analysis
        techniques such as local binary patterns or Gabor filters. The epsilon parameter
        should be chosen based on the expected dynamic range of your input data -
        smaller values provide higher precision but may be less numerically stable.

        For very large images, consider using larger strides to reduce computational
        cost while maintaining texture characterization capability.
    """

    def __init__(
            self,
            pool_size: Tuple[int, int] = (3, 3),
            strides: Union[Tuple[int, int], List[int]] = (1, 1),
            padding: str = "same",
            data_format: Optional[str] = None,
            epsilon: float = 1e-7,
            **kwargs: Any
    ) -> None:
        """Initialize the MovingStd layer."""
        super().__init__(**kwargs)

        # Validate and store pool size
        if not isinstance(pool_size, (tuple, list)) or len(pool_size) != 2:
            raise ValueError(
                f"pool_size must be a tuple or list of length 2, got {pool_size}"
            )
        if not all(isinstance(x, int) and x > 0 for x in pool_size):
            raise ValueError(
                f"pool_size values must be positive integers, got {pool_size}"
            )
        self.pool_size = tuple(pool_size)

        # Validate and store strides
        if not isinstance(strides, (tuple, list)) or len(strides) != 2:
            raise ValueError(
                f"strides must be a tuple or list of length 2, got {strides}"
            )
        if not all(isinstance(x, int) and x > 0 for x in strides):
            raise ValueError(
                f"strides values must be positive integers, got {strides}"
            )
        self.strides = tuple(strides)

        # Process and validate padding
        if not isinstance(padding, str):
            raise ValueError(f"padding must be a string, got {type(padding)}")
        self.padding = padding.lower()
        if self.padding not in {"valid", "same"}:
            raise ValueError(
                f"padding must be 'valid' or 'same', got '{padding}'"
            )

        # Process and validate data_format
        if data_format is None:
            self.data_format = keras.config.image_data_format()
        else:
            self.data_format = data_format.lower()

        if self.data_format not in {"channels_first", "channels_last"}:
            raise ValueError(
                f"data_format must be 'channels_first' or 'channels_last', "
                f"got '{data_format}'"
            )

        # Validate epsilon
        if not isinstance(epsilon, (int, float)) or epsilon < 0:
            raise ValueError(f"epsilon must be a non-negative number, got {epsilon}")
        self.epsilon = float(epsilon)

        # CREATE the average pooling sub-layer in __init__ (modern Keras 3 pattern)
        self.pooler = keras.layers.AveragePooling2D(
            pool_size=self.pool_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dtype=self.compute_dtype,
            name='internal_pooler'
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and its internal average pooling component.

        Args:
            input_shape: Shape tuple of the input tensor.

        Raises:
            ValueError: If input is not a 4D tensor.
        """
        # Validate input shape
        if len(input_shape) != 4:
            raise ValueError(
                f"Input must be a 4D tensor, got shape {input_shape}"
            )

        # BUILD the internal pooling layer (critical for serialization)
        self.pooler.build(input_shape)

        logger.debug(
            f"MovingStd layer built with pool_size={self.pool_size}, "
            f"strides={self.strides}, padding={self.padding}, "
            f"data_format={self.data_format}"
        )

        # Always call parent build at the end
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply the moving standard deviation filter to the input tensor.

        Args:
            inputs: Input tensor with shape determined by data_format.
            training: Boolean indicating whether in training mode. Included for
                API consistency but does not affect layer behavior.

        Returns:
            Tensor representing local standard deviation at each spatial location.
            Same format as input with potentially different spatial dimensions
            based on pooling parameters.
        """
        # Compute E[X] - local mean over the pooling window
        mean_x = self.pooler(inputs, training=training)

        # Compute E[X²] - local mean of squared values over the pooling window
        mean_x_sq = self.pooler(ops.square(inputs), training=training)

        # Calculate local variance: Var(X) = E[X²] - (E[X])²
        variance = mean_x_sq - ops.square(mean_x)

        # Ensure variance is non-negative for numerical stability
        # This handles potential floating-point precision issues
        variance = ops.maximum(variance, 0.0)

        # Compute local standard deviation: Std(X) = sqrt(Var(X) + epsilon)
        stddev = ops.sqrt(variance + self.epsilon)

        return stddev

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Shape tuple of the input tensor.

        Returns:
            Output shape tuple. Same as the output shape of the internal
            average pooling operation.
        """
        # Use the pooling layer's compute_output_shape method
        # Create a temporary pooling layer if not built yet
        if self.pooler is None or not hasattr(self.pooler, '_build_input_shape'):
            temp_pooler = keras.layers.AveragePooling2D(
                pool_size=self.pool_size,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format
            )
            output_shape = temp_pooler.compute_output_shape(input_shape)
        else:
            output_shape = self.pooler.compute_output_shape(input_shape)

        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        Returns:
            Dictionary containing all layer configuration parameters needed
            for reconstruction during model loading.
        """
        config = super().get_config()
        config.update({
            "pool_size": self.pool_size,
            "strides": self.strides,
            "padding": self.padding,
            "data_format": self.data_format,
            "epsilon": self.epsilon,
        })
        return config

# ---------------------------------------------------------------------
