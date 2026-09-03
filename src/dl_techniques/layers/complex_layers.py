"""
Complex-valued neural network layers: ``ComplexConv2D``, ``ComplexDense``,
``ComplexReLU``, and complex-aware pooling and dropout layers.

Each layer keeps a complex tensor as its actual dtype and computes with the
4-real-multiply expansion of complex arithmetic, rather than threading two
parallel real tensors through the public API. Weights are initialized with
magnitude drawn from a Rayleigh distribution and phase drawn uniformly, an
Xavier/Glorot-style scheme adapted to the complex domain.

The forward paths use raw TensorFlow operations (``tf.complex``,
``tf.math.real``/``imag``) instead of ``keras.ops``, because ``keras.ops`` has
no complex dtype or complex-tensor constructor. These layers are therefore
TensorFlow-backend-only.

References:
    - Trabelsi et al., 2018. Deep Complex Networks.
    - Arjovsky et al., 2016. Unitary Evolution Recurrent Neural Networks.
"""

import keras
import numpy as np
import tensorflow as tf
from typing import Optional, Tuple, Union, Dict, Any

from ..utils.random import rayleigh
from dl_techniques.utils.keras_registration import register_dl_technique


@register_dl_technique("dl_techniques.layers.complex_layers")
class ComplexLayer(keras.layers.Layer):
    """Base class for complex-valued layers.

    Handles complex weight initialization: magnitude from a Rayleigh
    distribution, phase from a uniform distribution over ``[-pi, pi]``.
    Complex numbers are represented as ``z = x + iy`` throughout, computed
    with split real/imaginary arithmetic.

    :param epsilon: Accepted and serialized for config compatibility but read
        by no computation in this module. Measured on CoShNet: values from
        ``1e-30`` to ``1e+3`` move the output by exactly 0. Kept because
        existing ``.keras`` files pass it through ``from_config``; removing it
        would raise a ``TypeError`` loading those checkpoints. Defaults to
        ``1e-7``.
    :param kernel_regularizer: Regularizer applied to both real and imaginary
        parts of complex weights. Defaults to ``None``.
    :param kernel_initializer: Initializer affecting the base scaling before
        the complex-specific initialization is applied. Defaults to
        ``GlorotUniform``.
    """

    def __init__(
        self,
        epsilon: float = 1e-7,
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        kernel_initializer: Optional[keras.initializers.Initializer] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        # DECISION plan-2026-08-19T163559-499b6f0e/D-053: epsilon is inert -- no
        # division in this module reads it. Removing it breaks from_config on every saved .keras checkpoint. See decisions.md.
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

        :param shape: Shape of the weight tensor.
        :type shape: Tuple[int, ...]
        :param dtype: TensorFlow dtype for the complex weights.
        :type dtype: tf.DType
        :return: Complex-valued weight tensor.
        :rtype: tf.Tensor
        """
        fan_in = int(np.prod(shape[:-1]))
        fan_out = int(shape[-1])
        sigma = keras.ops.sqrt(2.0 / (fan_in + fan_out))

        magnitude = rayleigh(shape, sigma, dtype=tf.float32)
        phase = keras.random.uniform(shape, -np.pi, np.pi, dtype=tf.float32)

        weights = tf.complex(
            magnitude * keras.ops.cos(phase),
            magnitude * keras.ops.sin(phase)
        )

        return tf.cast(weights, dtype)

    def get_config(self) -> Dict[str, Any]:
        """Return the layer configuration for serialization."""
        config = super().get_config()
        config.update({
            'epsilon': self.epsilon,
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer)
        })
        return config


@register_dl_technique("dl_techniques.layers.complex_layers")
class ComplexConv2D(ComplexLayer):
    """Complex-valued 2D convolution layer.

    Computes complex convolution as four real convolutions, following the
    algebraic expansion of complex multiplication.

    Architecture:

    .. code-block:: text

        Input [B, H, W, C] (complex)
                    │
                    ▼
        ┌───────────────────────────┐
        │ split: real, imag         │
        └─────────────┬─────────────┘
                       ▼
        ┌───────────────────────────┐
        │ 4 real convolutions       │
        │  real_out = conv(re,Kre)  │
        │           - conv(im,Kim)  │
        │  imag_out = conv(re,Kim)  │
        │           + conv(im,Kre)  │
        └─────────────┬─────────────┘
                       ▼
        combine + bias, complex(real_out, imag_out)
                       │
                       ▼
        Output [B, H', W', filters] (complex)

    For complex input ``I = I_r + iI_i`` and kernel ``K = K_r + iK_i``:
    ``output = (I_r*K_r - I_i*K_i) + i(I_r*K_i + I_i*K_r)``

    :param filters: Number of output filters. Each filter produces one
        complex-valued output channel.
    :type filters: int
    :param kernel_size: Height and width of the convolution window.
    :type kernel_size: int or Tuple[int, int]
    :param strides: Stride of the convolution. Defaults to ``1``.
    :type strides: int or Tuple[int, int]
    :param padding: ``'SAME'`` or ``'VALID'`` (case-insensitive). Defaults to
        ``'SAME'``.
    :type padding: str

    Example:

    .. code-block:: python

        conv = ComplexConv2D(
            filters=32, kernel_size=(5, 5), strides=(2, 2),
            padding='VALID', kernel_regularizer=keras.regularizers.L2(0.01),
        )
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
        """Create the layer's weights.

        :param input_shape: Shape of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
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
        """Apply complex convolution to the input tensor.

        :param inputs: Complex-valued input tensor.
        :type inputs: keras.KerasTensor
        :param training: Whether the layer is in training mode.
        :type training: Optional[bool]
        :return: Complex-valued output tensor.
        :rtype: keras.KerasTensor
        """
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

        :param input_shape: Input tensor shape.

        :return: Output tensor shape.
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

        :return: Configuration dictionary containing all constructor parameters.
        """
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
        })
        return config


@register_dl_technique("dl_techniques.layers.complex_layers")
class ComplexDense(ComplexLayer):
    """Complex-valued fully connected layer.

    Computes complex matrix multiplication as four real matmuls, following
    the algebraic expansion of complex multiplication.

    Architecture:

    .. code-block:: text

        Input [..., D] (complex)
                    │
                    ▼
        ┌───────────────────────────┐
        │ split: real, imag         │
        └─────────────┬─────────────┘
                       ▼
        ┌───────────────────────────┐
        │ 4 real matmuls            │
        │  real_out = re@Wre - im@Wim│
        │  imag_out = re@Wim + im@Wre│
        └─────────────┬─────────────┘
                       ▼
        combine + bias, complex(real_out, imag_out)
                       │
                       ▼
        Output [..., units] (complex)

    For complex input ``I = I_r + iI_i`` and weights ``W = W_r + iW_i``:
    ``output = (I_r@W_r - I_i@W_i) + i(I_r@W_i + I_i@W_r)``

    :param units: Number of output units.
    :type units: int

    Example:

    .. code-block:: python

        dense = ComplexDense(
            units=64, kernel_regularizer=keras.regularizers.L2(0.01),
            kernel_initializer=keras.initializers.HeNormal(),
        )
        inputs = keras.Input(shape=(784,), dtype=tf.complex64)
        x = ComplexDense(256)(inputs)
        outputs = ComplexDense(10)(x)
        model = keras.Model(inputs, outputs)
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

        :param input_shape: Shape of the input tensor.
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

        :param inputs: Complex-valued input tensor.
        :param training: Boolean indicating whether in training mode.

        :return: Complex-valued output tensor after transformation.
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

        :param input_shape: Input tensor shape.

        :return: Output tensor shape.
        """
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        :return: Configuration dictionary containing all constructor parameters.
        """
        config = super().get_config()
        config.update({
            'units': self.units,
        })
        return config


@register_dl_technique("dl_techniques.layers.complex_layers")
class ComplexReLU(keras.layers.Layer):
    """Complex ReLU activation.

    Applies ReLU to the real and imaginary parts independently: for
    ``z = x + iy``, ``output = max(0, x) + i*max(0, y)``.

    Architecture:

    .. code-block:: text

        Input [..., D] (complex)
                    │
                    ▼
        ┌───────────────────────────┐
        │ split: real, imag         │
        │ relu each independently   │
        └─────────────┬─────────────┘
                       ▼
        Output [..., D] (complex)

    Example:

    .. code-block:: python

        inputs = keras.Input(shape=(128,), dtype=tf.complex64)
        x = ComplexDense(64)(inputs)
        x = ComplexReLU()(x)
        outputs = ComplexDense(32)(x)
        model = keras.Model(inputs, outputs)
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

        :param inputs: Complex-valued input tensor.
        :param training: Boolean indicating whether in training mode.

        :return: Complex-valued output tensor with ReLU applied to both components.
        """
        return tf.complex(
            keras.ops.relu(tf.math.real(inputs)),
            keras.ops.relu(tf.math.imag(inputs))
        )

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape (same as input for activation).

        :param input_shape: Input tensor shape.

        :return: Output tensor shape (identical to input).
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        :return: Configuration dictionary.
        """
        return super().get_config()


@register_dl_technique("dl_techniques.layers.complex_layers")
class ComplexAveragePooling2D(keras.layers.Layer):
    """Complex-valued 2D average pooling layer.

    Applies standard average pooling independently to the real and
    imaginary components: for ``z = x + iy``, ``output = AvgPool(x) + i*AvgPool(y)``.

    Architecture:

    .. code-block:: text

        Input [B, H, W, C] (complex)
                    │
                    ▼
        ┌───────────────────────────┐
        │ split: real, imag         │
        │ avg-pool each independently│
        └─────────────┬─────────────┘
                       ▼
        Output [B, H', W', C] (complex)

    :param pool_size: Height and width of the pooling window. Defaults to
        ``(2, 2)``.
    :type pool_size: int or Tuple[int, int]
    :param strides: Stride of the pooling operation. Defaults to
        ``pool_size``.
    :type strides: int or Tuple[int, int], optional
    :param padding: ``'SAME'`` or ``'VALID'`` (case-insensitive). Defaults to
        ``'VALID'``.
    :type padding: str
    """

    def __init__(
            self,
            pool_size: Union[int, Tuple[int, int]] = (2, 2),
            strides: Optional[Union[int, Tuple[int, int]]] = None,
            padding: str = 'VALID',
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Store and validate configuration
        self.pool_size = pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
        self.strides = strides if strides is not None else self.pool_size
        self.strides = self.strides if isinstance(self.strides, tuple) else (self.strides, self.strides)
        self.padding = padding.upper()

        if self.padding not in ['SAME', 'VALID']:
            raise ValueError(f"padding must be 'SAME' or 'VALID', got {padding}")

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply complex average pooling.

        :param inputs: Complex-valued input tensor.
        :param training: Boolean indicating whether in training mode (unused).

        :return: Complex-valued output tensor after pooling.
        """
        # Split into real and imaginary components
        inputs_real = tf.math.real(inputs)
        inputs_imag = tf.math.imag(inputs)

        # Apply average pooling to each component
        # Note: keras.ops.average_pool expects lowercase padding
        padding_lower = self.padding.lower()

        pooled_real = keras.ops.average_pool(
            inputs_real,
            pool_size=self.pool_size,
            strides=self.strides,
            padding=padding_lower
        )

        pooled_imag = keras.ops.average_pool(
            inputs_imag,
            pool_size=self.pool_size,
            strides=self.strides,
            padding=padding_lower
        )

        # Recombine into a complex tensor
        return tf.complex(pooled_real, pooled_imag)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the pooling layer.

        :param input_shape: Input tensor shape.

        :return: Output tensor shape.
        """
        if len(input_shape) != 4:
            raise ValueError(f"ComplexAveragePooling2D requires 4D input, got {len(input_shape)}D")

        batch_size, height, width, channels = input_shape

        def _compute_dim(dim, pool, stride, padding):
            if dim is None:
                return None
            if padding == 'VALID':
                return (dim - pool + stride) // stride
            else:  # SAME
                return (dim + stride - 1) // stride

        output_height = _compute_dim(height, self.pool_size[0], self.strides[0], self.padding)
        output_width = _compute_dim(width, self.pool_size[1], self.strides[1], self.padding)

        return (batch_size, output_height, output_width, channels)

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        :return: Configuration dictionary containing all constructor parameters.
        """
        config = super().get_config()
        config.update({
            'pool_size': self.pool_size,
            'strides': self.strides,
            'padding': self.padding
        })
        return config


@register_dl_technique("dl_techniques.layers.complex_layers")
class ComplexDropout(keras.layers.Layer):
    """Complex-valued dropout layer.

    Generates one real-valued dropout mask and applies it to both the real
    and imaginary parts, so entire complex units drop together rather than
    their components dropping independently.

    Architecture:

    .. code-block:: text

        Input [..., D] (complex)
                    │
                    ▼ (training only)
        ┌───────────────────────────┐
        │ mask = Dropout(ones_like) │
        │ output = input * mask     │
        └─────────────┬─────────────┘
                       ▼
        Output [..., D] (complex)

    :param rate: Fraction of the input units to drop, in ``[0, 1)``.
    :type rate: float

    Example:

    .. code-block:: python

        inputs = keras.Input(shape=(256,), dtype=tf.complex64)
        x = ComplexDense(128)(inputs)
        x = ComplexReLU()(x)
        x = ComplexDropout(0.3)(x)
        outputs = ComplexDense(64)(x)
        model = keras.Model(inputs, outputs)
    """

    def __init__(self, rate: float, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        if not 0 <= rate < 1:
            raise ValueError(f"rate must be in the interval [0, 1), got {rate}")

        self.rate = rate
        self.dropout_layer = keras.layers.Dropout(self.rate)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply complex dropout.

        :param inputs: Complex-valued input tensor.
        :param training: Boolean indicating whether in training mode.

        :return: Complex-valued output tensor after dropout.
        """
        # Generate a real-valued mask by applying dropout to a tensor of ones.
        # The internal Dropout layer handles the `training` flag and scaling.
        mask = self.dropout_layer(
            tf.ones_like(tf.math.real(inputs)),
            training=training
        )

        # The mask is real-valued. Multiplying it with the complex input
        # correctly scales both the real and imaginary parts simultaneously.
        return inputs * tf.cast(mask, dtype=inputs.dtype)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape (same as input for dropout).

        :param input_shape: Input tensor shape.

        :return: Output tensor shape (identical to input).
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        :return: Configuration dictionary.
        """
        config = super().get_config()
        config.update({
            'rate': self.rate,
        })
        return config


@register_dl_technique("dl_techniques.layers.complex_layers")
class ComplexGlobalAveragePooling2D(keras.layers.Layer):
    """Complex-valued global 2D average pooling layer.

    Reduces each complex feature map to a single complex number by averaging
    over the spatial axes, independently on the real and imaginary parts.

    Architecture:

    .. code-block:: text

        Input [B, H, W, C] (complex)
                    │
                    ▼
        ┌───────────────────────────┐
        │ split: real, imag         │
        │ mean over axes [1, 2]     │
        └─────────────┬─────────────┘
                       ▼
        Output [B, C] (complex), or [B, 1, 1, C] if keepdims

    :param keepdims: If ``False`` (default), the output has shape
        ``(batch, channels)``. If ``True``, ``(batch, 1, 1, channels)``.
    :type keepdims: bool
    """
    def __init__(self, keepdims: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.keepdims = keepdims

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply complex global average pooling.

        :param inputs: Complex-valued input tensor.
        :param training: Boolean indicating whether in training mode (unused).

        :return: Complex-valued output tensor after pooling.
        """
        # Split into real and imaginary components
        inputs_real = tf.math.real(inputs)
        inputs_imag = tf.math.imag(inputs)

        # Apply global average pooling to each component
        # The spatial axes for a 4D tensor are 1 and 2.
        pooled_real = keras.ops.mean(inputs_real, axis=[1, 2], keepdims=self.keepdims)
        pooled_imag = keras.ops.mean(inputs_imag, axis=[1, 2], keepdims=self.keepdims)

        # Recombine into a complex tensor
        return tf.complex(pooled_real, pooled_imag)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the global pooling layer.

        :param input_shape: Input tensor shape.

        :return: Output tensor shape.
        """
        if len(input_shape) != 4:
            raise ValueError(f"ComplexGlobalAveragePooling2D requires 4D input, got {len(input_shape)}D")

        if self.keepdims:
            return (input_shape[0], 1, 1, input_shape[3])
        else:
            return (input_shape[0], input_shape[3])

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        :return: Configuration dictionary containing all constructor parameters.
        """
        config = super().get_config()
        config.update({
            'keepdims': self.keepdims,
        })
        return config

# ---------------------------------------------------------------------