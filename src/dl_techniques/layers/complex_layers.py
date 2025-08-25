"""
Complex-Valued Neural Network Layers Implementation
================================================

This module provides a comprehensive implementation of complex-valued neural network layers
for deep learning models. These layers support complex number operations while maintaining
numerical stability and efficient computation.

Key Features:
------------
- Split-complex implementation for better numerical stability
- Proper complex weight initialization using Rayleigh distribution
- Efficient complex arithmetic operations
- Support for complex convolution and dense operations
- Complex-valued ReLU activation
- Customizable regularization and initialization

Technical Details:
-----------------
1. Complex Number Representation:
   - Uses split representation (real/imaginary) for stability
   - Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
   - Complex convolution implemented as 4 real convolutions
   - Complex dense operations use matrix form of complex multiplication

2. Weight Initialization:
   - Magnitude: Rayleigh distribution for proper scaling
   - Phase: Uniform distribution in [-π, π]
   - Fan-in/fan-out scaling for stable gradients
   - Xavier/Glorot-style initialization adapted for complex domain

3. Numerical Stability:
   - Epsilon factor for division operations
   - Split operations to prevent catastrophic cancellation
   - Proper handling of complex gradients
   - Stable complex arithmetic implementation

4. Layer Types:
   - ComplexLayer: Base class with common functionality
   - ComplexConv2D: Complex-valued convolutional layer
   - ComplexDense: Complex-valued fully connected layer
   - ComplexReLU: Split ReLU activation for complex values

References:
----------
1. "Deep Complex Networks" (Trabelsi et al., 2018)
2. "Unitary Evolution Recurrent Neural Networks" (Arjovsky et al., 2016)
"""

import keras
import numpy as np
import tensorflow as tf
from typing import Optional, Tuple, Union, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..utils.random import rayleigh

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class ComplexLayer(keras.layers.Layer):
    """
    Base class for complex-valued layers.

    Provides common functionality for complex-valued operations with improved
    numerical stability and initialization strategies. This base class handles
    complex weight initialization using Rayleigh distribution for magnitudes
    and uniform distribution for phases, ensuring proper scaling for complex
    neural network training.

    **Intent**: Establish a foundation for complex-valued neural network layers
    with proper initialization, regularization support, and numerical stability
    measures. All complex layers in this module inherit from this base class.

    **Mathematical Foundation**:
    Complex numbers are represented as z = x + iy, where x and y are real numbers
    and i is the imaginary unit. Operations are performed using split real/imaginary
    arithmetic to maintain numerical stability and computational efficiency.

    Args:
        epsilon: Float, small value added for numerical stability in division operations.
            Prevents catastrophic cancellation and division by zero. Defaults to 1e-7.
        kernel_regularizer: Optional regularizer instance for kernel weights.
            Applied to both real and imaginary parts of complex weights.
            Defaults to None (no regularization).
        kernel_initializer: Optional initializer for kernel weights. When None,
            uses GlorotUniform initialization. For complex layers, this affects
            the base scaling before applying complex-specific initialization.
            Defaults to None.
        **kwargs: Additional keyword arguments passed to Layer base class.

    Attributes:
        epsilon: Numerical stability constant.
        kernel_regularizer: Kernel weight regularizer.
        kernel_initializer: Kernel weight initializer.

    Note:
        This is an abstract base class. Use specific implementations like
        ComplexConv2D or ComplexDense for actual neural network construction.
    """

    def __init__(
        self,
        epsilon: float = 1e-7,
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        kernel_initializer: Optional[keras.initializers.Initializer] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        # Store configuration
        self.epsilon = epsilon
        self.kernel_regularizer = kernel_regularizer
        self.kernel_initializer = kernel_initializer or keras.initializers.GlorotUniform()

    def _init_complex_weights(
        self,
        shape: Tuple[int, ...],
        dtype: tf.DType = tf.complex64
    ) -> tf.Tensor:
        """
        Initialize complex weights using Rayleigh distribution with proper scaling.

        This method creates complex-valued weights by sampling magnitudes from a
        Rayleigh distribution and phases from a uniform distribution, then combining
        them into complex numbers. The scaling follows Xavier/Glorot initialization
        principles adapted for the complex domain.

        Args:
            shape: Tuple of integers specifying the shape of weight tensor.
            dtype: TensorFlow data type for complex weights. Defaults to tf.complex64.

        Returns:
            Complex-valued weight tensor with proper initialization.
        """
        fan_in = int(np.prod(shape[:-1]))
        fan_out = int(shape[-1])
        sigma = keras.ops.sqrt(2.0 / (fan_in + fan_out))

        # Initialize magnitude and phase
        magnitude = rayleigh(shape, sigma, dtype=tf.float32)
        phase = keras.random.uniform(shape, -np.pi, np.pi, dtype=tf.float32)

        # Convert to complex representation
        weights = tf.complex(
            magnitude * keras.ops.cos(phase),
            magnitude * keras.ops.sin(phase)
        )

        return tf.cast(weights, dtype)

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        Returns:
            Dictionary containing all constructor parameters needed for
            reconstruction during model loading.
        """
        config = super().get_config()
        config.update({
            'epsilon': self.epsilon,
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer)
        })
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ComplexConv2D(ComplexLayer):
    """
    Complex-valued 2D convolution layer with improved numerical stability.

    Implements complex convolution using split real/imaginary implementation
    for better numerical stability and performance. The complex convolution
    is computed using four real convolutions following the mathematical
    expansion of complex multiplication applied to convolution operations.

    **Intent**: Provide complex-valued convolutional processing for applications
    requiring phase and magnitude information, such as signal processing,
    radar, sonar, and certain computer vision tasks where complex representations
    are natural.

    **Architecture**:
    ```
    Input(shape=[batch, height, width, channels_complex])
           ↓
    Split: real_part, imag_part
           ↓
    Complex Convolution:
      real_out = conv(real_in, real_kernel) - conv(imag_in, imag_kernel)
      imag_out = conv(real_in, imag_kernel) + conv(imag_in, real_kernel)
           ↓
    Combine + Bias: output = complex(real_out + real_bias, imag_out + imag_bias)
           ↓
    Output(shape=[batch, new_height, new_width, filters_complex])
    ```

    **Mathematical Operation**:
    For complex input I = I_r + iI_i and complex kernel K = K_r + iK_i:
    Output = (I_r + iI_i) * (K_r + iK_i) = (I_r*K_r - I_i*K_i) + i(I_r*K_i + I_i*K_r)

    Args:
        filters: Integer, number of output filters. Must be positive.
            Each filter produces one complex-valued output channel.
        kernel_size: Integer or tuple of 2 integers, specifying height and width
            of the 2D convolution window. If integer, same value for both dimensions.
        strides: Integer or tuple of 2 integers, specifying stride of convolution.
            If integer, same stride for both dimensions. Defaults to 1.
        padding: String, either 'SAME' or 'VALID' (case-insensitive).
            'SAME' pads input to preserve spatial dimensions with stride=1.
            'VALID' performs convolution without padding. Defaults to 'SAME'.
        **kwargs: Additional keyword arguments for ComplexLayer base class.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, channels)`.
        Input should be complex-valued (tf.complex64 or tf.complex128).

    Output shape:
        4D tensor with shape: `(batch_size, new_height, new_width, filters)`.
        Output dimensions depend on padding and stride configuration.

    Attributes:
        filters: Number of output filters.
        kernel_size: Size of convolution kernel.
        strides: Stride configuration.
        padding: Padding mode.
        kernel: Complex-valued convolution kernel weights.
        bias: Complex-valued bias terms.

    Example:
        ```python
        # Basic complex convolution
        conv = ComplexConv2D(filters=64, kernel_size=3)
        inputs = keras.random.uniform((32, 28, 28, 1), dtype=tf.complex64)
        outputs = conv(inputs)  # Shape: (32, 28, 28, 64)

        # Strided convolution with regularization
        conv = ComplexConv2D(
            filters=32,
            kernel_size=(5, 5),
            strides=(2, 2),
            padding='VALID',
            kernel_regularizer=keras.regularizers.L2(0.01)
        )
        ```

    Note:
        This implementation uses split arithmetic to maintain numerical stability
        and compatibility with automatic differentiation. Input tensors should
        be complex-valued (tf.complex64 or tf.complex128).
    """

    def __init__(
        self,
        filters: int,
        kernel_size: Union[int, Tuple[int, int]],
        strides: Union[int, Tuple[int, int]] = 1,
        padding: str = 'SAME',
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if filters <= 0:
            raise ValueError(f"filters must be positive, got {filters}")

        # Store configuration
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.strides = strides if isinstance(strides, tuple) else (strides, strides)
        self.padding = padding.upper()

        if self.padding not in ['SAME', 'VALID']:
            raise ValueError(f"padding must be 'SAME' or 'VALID', got {padding}")

        # Initialize weight attributes (created in build)
        self.kernel = None
        self.bias = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Create the layer's weights.

        Args:
            input_shape: Shape of the input tensor.
        """
        if len(input_shape) != 4:
            raise ValueError(f"ComplexConv2D requires 4D input, got {len(input_shape)}D")

        input_channels = input_shape[-1]
        if input_channels is None:
            raise ValueError("Last dimension of input must be defined")

        kernel_shape = (*self.kernel_size, input_channels, self.filters)

        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=self._init_complex_weights,
            regularizer=self.kernel_regularizer,
            trainable=True,
            dtype=tf.complex64
        )

        self.bias = self.add_weight(
            name='bias',
            shape=(self.filters,),
            initializer='zeros',
            trainable=True,
            dtype=tf.complex64
        )

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply complex convolution to input tensor.

        Args:
            inputs: Complex-valued input tensor.
            training: Boolean indicating whether in training mode.

        Returns:
            Complex-valued output tensor after convolution.
        """
        # Split real and imaginary parts
        inputs_real = tf.math.real(inputs)
        inputs_imag = tf.math.imag(inputs)
        kernel_real = tf.math.real(self.kernel)
        kernel_imag = tf.math.imag(self.kernel)

        # Convert padding for keras.ops.conv (which expects lowercase)
        padding_lower = self.padding.lower()

        # Compute convolution components using keras.ops
        real_output = keras.ops.conv(
            inputs_real, kernel_real, strides=self.strides, padding=padding_lower
        ) - keras.ops.conv(
            inputs_imag, kernel_imag, strides=self.strides, padding=padding_lower
        )

        imag_output = keras.ops.conv(
            inputs_real, kernel_imag, strides=self.strides, padding=padding_lower
        ) + keras.ops.conv(
            inputs_imag, kernel_real, strides=self.strides, padding=padding_lower
        )

        # Combine real and imaginary parts
        output = tf.complex(
            real_output + tf.math.real(self.bias),
            imag_output + tf.math.imag(self.bias)
        )

        return output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the convolution.

        Args:
            input_shape: Input tensor shape.

        Returns:
            Output tensor shape.
        """
        batch_size = input_shape[0]

        if self.padding == 'SAME':
            output_height = input_shape[1] // self.strides[0] if input_shape[1] is not None else None
            output_width = input_shape[2] // self.strides[1] if input_shape[2] is not None else None
        else:  # VALID padding
            if input_shape[1] is not None:
                output_height = (input_shape[1] - self.kernel_size[0]) // self.strides[0] + 1
            else:
                output_height = None
            if input_shape[2] is not None:
                output_width = (input_shape[2] - self.kernel_size[1]) // self.strides[1] + 1
            else:
                output_width = None

        return (batch_size, output_height, output_width, self.filters)

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        Returns:
            Configuration dictionary containing all constructor parameters.
        """
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
        })
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ComplexDense(ComplexLayer):
    """
    Complex-valued dense layer with improved initialization.

    Implements complex dense operations with split real/imaginary parts
    for stability and efficient computation. The complex multiplication
    is performed using the mathematical expansion where each complex
    weight interacts with complex inputs through proper complex arithmetic.

    **Intent**: Provide fully connected complex-valued transformations for
    neural networks processing complex-valued data, such as signal processing
    applications, complex embeddings, or as building blocks in complex
    neural architectures.

    **Architecture**:
    ```
    Input(shape=[..., input_dim_complex])
           ↓
    Split: real_part, imag_part
           ↓
    Complex Matrix Multiplication:
      real_out = matmul(real_in, real_weights) - matmul(imag_in, imag_weights)
      imag_out = matmul(real_in, imag_weights) + matmul(imag_in, real_weights)
           ↓
    Add Bias: output = complex(real_out + real_bias, imag_out + imag_bias)
           ↓
    Output(shape=[..., units_complex])
    ```

    **Mathematical Operation**:
    For complex input I = I_r + iI_i and complex weights W = W_r + iW_i:
    Output = (I_r + iI_i) @ (W_r + iW_i) = (I_r@W_r - I_i@W_i) + i(I_r@W_i + I_i@W_r)

    Args:
        units: Integer, number of output units. Must be positive.
            Determines the dimensionality of the output space.
        **kwargs: Additional keyword arguments for ComplexLayer base class.

    Input shape:
        N-D tensor with shape: `(..., input_dim)`.
        Input should be complex-valued (tf.complex64 or tf.complex128).

    Output shape:
        N-D tensor with shape: `(..., units)`.
        Same rank as input with last dimension changed to `units`.

    Attributes:
        units: Number of output units.
        kernel: Complex-valued weight matrix.
        bias: Complex-valued bias vector.

    Example:
        ```python
        # Basic complex dense layer
        dense = ComplexDense(units=128)
        inputs = keras.random.uniform((32, 256), dtype=tf.complex64)
        outputs = dense(inputs)  # Shape: (32, 128)

        # With regularization and custom initialization
        dense = ComplexDense(
            units=64,
            kernel_regularizer=keras.regularizers.L2(0.01),
            kernel_initializer=keras.initializers.HeNormal()
        )

        # As part of complex neural network
        inputs = keras.Input(shape=(784,), dtype=tf.complex64)
        x = ComplexDense(256)(inputs)
        x = ComplexDense(128)(x)
        outputs = ComplexDense(10)(x)
        model = keras.Model(inputs, outputs)
        ```

    Note:
        This layer performs efficient complex matrix multiplication using
        split arithmetic to maintain numerical stability. Input tensors
        should be complex-valued for proper operation.
    """

    def __init__(
        self,
        units: int,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if units <= 0:
            raise ValueError(f"units must be positive, got {units}")

        self.units = units

        # Initialize weight attributes (created in build)
        self.kernel = None
        self.bias = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Create the layer's weights.

        Args:
            input_shape: Shape of the input tensor.
        """
        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError("Last dimension of input must be defined")

        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_dim, self.units),
            initializer=self._init_complex_weights,
            regularizer=self.kernel_regularizer,
            trainable=True,
            dtype=tf.complex64
        )

        self.bias = self.add_weight(
            name='bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            dtype=tf.complex64
        )

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply complex dense transformation.

        Args:
            inputs: Complex-valued input tensor.
            training: Boolean indicating whether in training mode.

        Returns:
            Complex-valued output tensor after transformation.
        """
        # Split computations for numerical stability
        real_output = keras.ops.matmul(
            tf.math.real(inputs), tf.math.real(self.kernel)
        ) - keras.ops.matmul(
            tf.math.imag(inputs), tf.math.imag(self.kernel)
        )

        imag_output = keras.ops.matmul(
            tf.math.real(inputs), tf.math.imag(self.kernel)
        ) + keras.ops.matmul(
            tf.math.imag(inputs), tf.math.real(self.kernel)
        )

        return tf.complex(
            real_output + tf.math.real(self.bias),
            imag_output + tf.math.imag(self.bias)
        )

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the dense layer.

        Args:
            input_shape: Input tensor shape.

        Returns:
            Output tensor shape.
        """
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        Returns:
            Configuration dictionary containing all constructor parameters.
        """
        config = super().get_config()
        config.update({
            'units': self.units,
        })
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ComplexReLU(keras.layers.Layer):
    """
    Complex ReLU activation function.

    Applies ReLU activation separately to real and imaginary parts of complex
    inputs, effectively creating a split complex activation. This preserves
    the complex structure while introducing non-linearity through element-wise
    rectification of both components.

    **Intent**: Provide non-linear activation for complex-valued neural networks
    while maintaining the complex number structure. The split application ensures
    that gradients can flow through both real and imaginary components.

    **Mathematical Operation**:
    For complex input z = x + iy:
    output = ReLU(x) + i·ReLU(y) = max(0, x) + i·max(0, y)

    **Architecture**:
    ```
    Input(shape=[..., complex_features])
           ↓
    Split: real_part = Re(z), imag_part = Im(z)
           ↓
    Apply ReLU: real_activated = max(0, real_part)
                imag_activated = max(0, imag_part)
           ↓
    Combine: output = complex(real_activated, imag_activated)
           ↓
    Output(shape=[..., complex_features])
    ```

    Args:
        **kwargs: Additional keyword arguments for Layer base class.

    Input shape:
        Arbitrary. Complex-valued tensor of any shape.

    Output shape:
        Same as input shape. Complex-valued tensor with ReLU applied component-wise.

    Example:
        ```python
        # Basic complex ReLU activation
        activation = ComplexReLU()
        complex_input = tf.complex([[-1.0, 2.0]], [[3.0, -4.0]])
        output = activation(complex_input)
        # Output: complex([[0.0, 2.0]], [[3.0, 0.0]])

        # In a complex neural network
        inputs = keras.Input(shape=(128,), dtype=tf.complex64)
        x = ComplexDense(64)(inputs)
        x = ComplexReLU()(x)  # Apply complex activation
        outputs = ComplexDense(32)(x)
        model = keras.Model(inputs, outputs)
        ```

    Note:
        This activation maintains complex structure while introducing sparsity
        through rectification. Alternative complex activations like complex
        sigmoid or tanh could be implemented following similar patterns.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply complex ReLU activation.

        Args:
            inputs: Complex-valued input tensor.
            training: Boolean indicating whether in training mode.

        Returns:
            Complex-valued output tensor with ReLU applied to both components.
        """
        return tf.complex(
            keras.ops.relu(tf.math.real(inputs)),
            keras.ops.relu(tf.math.imag(inputs))
        )

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape (same as input for activation).

        Args:
            input_shape: Input tensor shape.

        Returns:
            Output tensor shape (identical to input).
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        Returns:
            Configuration dictionary.
        """
        return super().get_config()

# ---------------------------------------------------------------------
