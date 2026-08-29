"""
A complex-valued classifier over a fixed shearlet frontend, in six size variants.

The design question CoShNet answers is which part of a vision model actually has
to be learned. The first layers of a trained CNN reliably converge to oriented,
band-limited edge detectors -- a filter bank that harmonic analysis can write down
in closed form and that costs nothing to fit. Shearlets are the principled choice
for that bank: built from anisotropic scaling paired with shearing rather than
the isotropic dilation of wavelets, their elements are elongated and oriented,
and under parabolic scaling (support `~2^j` along one axis, `~2^(j/2)` along the
other) they are provably optimal at sparsifying cartoon-like images -- piecewise
smooth with curved discontinuities, which is a fair caricature of natural images.
Fixing the frontend removes those parameters from the optimization entirely and
hands the network a strong geometric prior instead of asking it to rediscover
one.

The transform is applied as a non-trainable filter bank in the frequency domain:
FFT, elementwise multiply against `1 + scales * (directions + 1)` filters, and an
inverse FFT per filter. At the defaults (`scales=4`, `directions=8`) that is 37
filters, so a 3-channel input leaves the frontend with 111 channels at unchanged
spatial resolution. The filter bank is built for one static `(height, width)`;
this frontend cannot be traced with unknown spatial dims.

What follows is complex-valued, and it is worth being precise about where the
complexity comes from, because the layer names invite the wrong reading. The
transform returns the *real part* of the inverse FFT, so the phase of the
shearlet coefficients is not propagated; the model casts that real tensor to
`complex64` with an identically zero imaginary part. The imaginary channel is
populated only by the first complex convolution, whose kernel is genuinely
complex -- so phase here is a learned quantity mixed by complex multiplication,
not the analytic phase of the transform. Complex multiplication is what makes the
layer more than two real convolutions: it couples the two components
(`(a + ib)(c + id)`), and that coupling is the architecture's stated source of
parameter efficiency.

The body is two (or three, at `large`/`imagenet`) complex convolutions with
stride 2, each followed by a split ReLU applied independently to the real and
imaginary parts -- the simplest complex activation, and not a modulus-based one
such as modReLU. Global average pooling over the spatial axes then collapses the
feature map before the dense stack, a deliberate substitution for flattening: it
drops the first dense layer's parameter count by the spatial area and makes the
dense widths independent of input resolution, at the cost of discarding where in
the image a response occurred. The dense stack alternates complex dense,
activation and complex dropout.

The classifier head takes `ops.abs` of the final complex vector before a real
`Dense`, so the phase learned through the network reaches the decision only
through the magnitude it produces. That Dense applies `softmax` itself: the model
emits probabilities, not logits, and must be compiled with `from_logits=False`.
With `include_top=False` the model returns the `complex64` convolutional feature
map instead, which most downstream real-valued Keras layers will refuse.

References:
    - Ko, Panchal, Andrade-Loarca & Mendez-Vazquez, 2022. CoShNet: A Hybrid
      Complex Valued Neural Network using Shearlets.
    - Trabelsi et al., 2018. Deep Complex Networks. ICLR 2018.
      (https://arxiv.org/abs/1705.09792)
    - Guo, Kutyniok & Labate, 2006. Sparse Multidimensional Representations
      using Anisotropic Dilation and Shear Operators.
    - Kutyniok & Labate, 2012. Shearlets: Multiscale Analysis for Multivariate
      Data. Birkhauser.
    - Reisenhofer et al., 2016. Shearlab 3D / CoShREM: Faithful Digital Shearlet
      Transforms Based on Compactly Supported Shearlets.
"""

import keras
from keras import layers, ops, initializers, regularizers
from typing import Optional, Tuple, List, Dict, Any, Sequence, Union

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.shearlet_transform import ShearletTransform
from dl_techniques.layers.complex_layers import (
    ComplexDense,
    ComplexConv2D,
    ComplexReLU,
    ComplexDropout,
    ComplexGlobalAveragePooling2D,
)
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.coshnet.model")
class CoShNet(keras.Model):
    """Complex Shearlet Network: a complex-valued classifier on a fixed frontend.

    CoShNet replaces the learned early layers of a CNN with a NON-TRAINABLE
    shearlet filter bank applied in the frequency domain, then processes the
    result with complex-valued layers. The transform emits
    ``1 + scales * (directions + 1)`` filter responses per input channel at
    unchanged spatial resolution; the real result is cast to ``complex64`` with
    a zero imaginary part, and the imaginary channel is populated only by the
    FIRST complex convolution, whose kernel is genuinely complex. The body is
    a stack of stride-2 :class:`ComplexConv2D` layers with split ReLU, then
    complex global average pooling (not flatten), then a complex dense stack
    with complex dropout. The head takes ``ops.abs`` and applies a real softmax
    ``Dense``, so the model emits PROBABILITIES, not logits.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────────┐
        │   Input [B, H, W, C_in]              │
        │   (H, W must be STATIC: the filter   │
        │    bank is built for one shape)      │
        └───────────────┬──────────────────────┘
                        │
                        ▼
        ┌──────────────────────────────────────┐
        │  ShearletTransform  (NON-TRAINABLE)  │
        │    FFT → × N filters → iFFT → Re{·}  │
        │    N = 1 + scales·(directions + 1)   │
        │    defaults 4, 8 → N = 37            │
        │    spatial resolution UNCHANGED      │
        └───────────────┬──────────────────────┘
                        │  real [B, H, W, C_in·N]
                        ▼
        ┌──────────────────────────────────────┐
        │  ops.cast → complex64                │
        │  imaginary part is IDENTICALLY ZERO  │
        └───────────────┬──────────────────────┘
                        │
                        ▼
        ┌──────────────────────────────────────┐
        │  ComplexConv2D(f₀, K, stride 2)      │
        │  → ComplexReLU (split: Re, Im apart) │
        │  first conv is where Im ≠ 0 begins   │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  ComplexConv2D(f₁, K, stride 2)      │
        │  → ComplexReLU                       │
        │  (large / imagenet add a third)      │
        └───────────────┬──────────────────────┘
                        │  complex64 [B, H', W', f_last]
                        │
                        ├──── include_top=False ──► returned AS IS
                        │      (complex64; most real Keras
                        │       layers downstream will refuse it)
                        ▼
        ┌──────────────────────────────────────┐
        │  ComplexGlobalAveragePooling2D       │
        │  (replaces flatten: dense widths     │
        │   become resolution-independent,     │
        │   spatial position is discarded)     │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  ComplexDense(u₀) → ComplexReLU      │
        │                   → ComplexDropout   │
        │  ComplexDense(u₁) → ComplexReLU      │
        │                   → ComplexDropout   │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  ops.abs  (phase reaches the decision│
        │            only via its magnitude)   │
        │  → Dense(num_classes, softmax)       │
        └───────────────┬──────────────────────┘
                        ▼
        ┌──────────────────────────────────────┐
        │  Output [B, num_classes]             │
        │  PROBABILITIES, not logits           │
        │  → compile with from_logits=False    │
        └──────────────────────────────────────┘

    **Complex multiplication (why this is not two real convolutions):**

    .. code-block:: text

        input  x = a + i·b        kernel  w = c + i·d

        x · w = (a·c - b·d) + i·(a·d + b·c)
                └── Re ──┘        └── Im ──┘

        the two components are COUPLED: each output part reads
        both input parts. This coupling is the stated source of
        the architecture's parameter efficiency.

        ComplexReLU is SPLIT, not modulus-based:
            relu(Re) + i·relu(Im)     ── not modReLU

    **Variants:**

    .. code-block:: text

        variant    conv_filters       dense_units          sc  dir  drop  K
        nano       [16, 24]           [128, 64]            3    4   0.15  3
        tiny       [16, 32]           [256, 128]           3    6   0.20  3
        base       [32, 64]           [1250, 500]          4    8   0.10  5
        large      [64, 128, 256]     [2048, 1024, 512]    5   12   0.15  5
        cifar10    [32, 64]           [800, 400]           4    8   0.10  5
        imagenet   [64, 128, 256]     [2048, 1024]         5   16   0.20  7

        sc = shearlet_scales, dir = shearlet_directions, K = conv_kernel_size
        `imagenet` defaults to input_shape (224, 224, 3); every other
        variant defaults to (32, 32, 3).

    :param num_classes: Number of output classes. Must be positive. Defaults
        to 10.
    :type num_classes: int
    :param input_shape: Input shape ``(height, width, channels)`` excluding the
        batch dimension. ``None`` resolves to ``(32, 32, 3)`` for CIFAR-10
        compatibility. The spatial dims must be STATIC and positive: the
        shearlet filter bank is constructed for one concrete ``(height,
        width)`` and cannot be traced with unknown spatial dims.
    :type input_shape: Optional[Tuple[int, int, int]]
    :param conv_filters: Filter count per complex convolutional layer; its
        length sets the number of layers. Must be non-empty and all positive.
        Defaults to ``(32, 64)``.
    :type conv_filters: Sequence[int]
    :param dense_units: Unit count per complex dense layer in the head. Must be
        non-empty and all positive when ``include_top=True``. Defaults to
        ``(1250, 500)``.
    :type dense_units: Sequence[int]
    :param shearlet_scales: Number of scales in the shearlet transform. Must be
        positive. Defaults to 4.
    :type shearlet_scales: int
    :param shearlet_directions: Number of directions per scale. Must be
        positive. Defaults to 8. Together with ``shearlet_scales`` this fixes
        the frontend's channel multiplier at
        ``1 + scales * (directions + 1)``.
    :type shearlet_directions: int
    :param conv_kernel_size: Kernel size for the complex convolutions. Must be
        positive. Defaults to 5.
    :type conv_kernel_size: int
    :param conv_strides: Stride for the complex convolutions. Must be positive.
        Defaults to 2.
    :type conv_strides: int
    :param conv_padding: Padding mode, one of ``"valid"``, ``"same"`` or
        ``"causal"``. Defaults to ``"same"``.
    :type conv_padding: str
    :param dropout_rate: Complex dropout rate in the dense stack; must be in
        ``[0, 1]``. Defaults to 0.1.
    :type dropout_rate: float
    :param kernel_regularizer: Optional regularizer applied to every complex
        and real kernel. Defaults to None.
    :type kernel_regularizer: Optional[Union[str, regularizers.Regularizer]]
    :param kernel_initializer: Initializer for every kernel. Defaults to
        ``'glorot_uniform'``.
    :type kernel_initializer: Union[str, initializers.Initializer]
    :param include_top: Whether to include the pooling + dense + classifier
        head. When False the ``complex64`` convolutional feature map is
        returned. Defaults to True.
    :type include_top: bool
    :param epsilon: **DEAD KNOB.** Validated, stored, serialized and forwarded
        to every complex layer, and never used in a computation. ``grep -n
        "epsilon" layers/complex_layers.py`` returns 6 hits, all of them
        docstring / signature / validation / ``get_config``; no complex layer
        performs a division at all, so there is no numerical-stability term for
        it to be. MEASURED 2026-08-18 on ONE built ``nano`` (so the weights are
        held fixed -- rebuilding under ``set_random_seed`` does NOT reproduce
        them here, and that invalid instrument reported a spurious 0.023):
        forward is deterministic self-vs-self at exactly 0.0, and after mutating
        all 4 live ``epsilon`` attributes from 1e-7 to 1e-1 the output changes by
        exactly **0.0**. Kept for config compatibility. Must still be positive.
    :type epsilon: float
    :param kwargs: Additional keyword arguments for the ``keras.Model`` base
        class. ``name`` defaults to ``"coshnet"`` but may be overridden.

    :raises ValueError: If ``input_shape`` is not a 3D tuple of positive
        dimensions, if any architecture parameter is non-positive or empty, if
        ``conv_padding`` is not a recognized mode, or if ``dropout_rate`` is
        outside ``[0, 1]``.

    Input shape:
        4D tensor with shape ``(batch_size, height, width, channels)``.

    Output shape:
        - ``include_top=True``: 2D tensor ``(batch_size, num_classes)`` of
          softmax probabilities.
        - ``include_top=False``: 4D ``complex64`` tensor
          ``(batch_size, H', W', conv_filters[-1])``.

    Example:
        .. code-block:: python

            # From a variant, for CIFAR-10
            model = CoShNet.from_variant("base", num_classes=10,
                                         input_shape=(32, 32, 3))

            # Custom configuration
            model = CoShNet(
                num_classes=100,
                input_shape=(64, 64, 3),
                conv_filters=[64, 128],
                dense_units=[800, 400]
            )

            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

    Note:
        The classifier applies softmax itself, so compile with
        ``from_logits=False``.
    """

    # Model variant configurations
    MODEL_VARIANTS = {
        "nano": {
            "conv_filters": [16, 24],
            "dense_units": [128, 64],
            "shearlet_scales": 3,
            "shearlet_directions": 4,
            "dropout_rate": 0.15,
            "conv_kernel_size": 3,
        },
        "tiny": {
            "conv_filters": [16, 32],
            "dense_units": [256, 128],
            "shearlet_scales": 3,
            "shearlet_directions": 6,
            "dropout_rate": 0.2,
            "conv_kernel_size": 3,
        },
        "base": {
            "conv_filters": [32, 64],
            "dense_units": [1250, 500],
            "shearlet_scales": 4,
            "shearlet_directions": 8,
            "dropout_rate": 0.1,
            "conv_kernel_size": 5,
        },
        "large": {
            "conv_filters": [64, 128, 256],
            "dense_units": [2048, 1024, 512],
            "shearlet_scales": 5,
            "shearlet_directions": 12,
            "dropout_rate": 0.15,
            "conv_kernel_size": 5,
        },
        "cifar10": {
            "conv_filters": [32, 64],
            "dense_units": [800, 400],
            "shearlet_scales": 4,
            "shearlet_directions": 8,
            "dropout_rate": 0.1,
            "conv_kernel_size": 5,
        },
        "imagenet": {
            "conv_filters": [64, 128, 256],
            "dense_units": [2048, 1024],
            "shearlet_scales": 5,
            "shearlet_directions": 16,
            "dropout_rate": 0.2,
            "conv_kernel_size": 7,
            "conv_strides": 2,
        },
    }

    # Architecture constants
    EPSILON_DEFAULT = 1e-7
    KERNEL_INITIALIZER_DEFAULT = "glorot_uniform"

    def __init__(
        self,
        # Core configuration
        num_classes: int = 10,
        input_shape: Optional[Tuple[int, int, int]] = None,
        # Architecture configuration
        conv_filters: Sequence[int] = (32, 64),
        dense_units: Sequence[int] = (1250, 500),
        # Shearlet transform configuration
        shearlet_scales: int = 4,
        shearlet_directions: int = 8,
        # Layer configuration
        conv_kernel_size: int = 5,
        conv_strides: int = 2,
        conv_padding: str = "same",
        # Regularization and training
        dropout_rate: float = 0.1,
        kernel_regularizer: Optional[Union[str, regularizers.Regularizer]] = None,
        # Initialization
        kernel_initializer: Union[str, initializers.Initializer] = "glorot_uniform",
        # Advanced options
        include_top: bool = True,
        epsilon: float = 1e-7,
        **kwargs: Any
    ) -> None:
        """Initialize and immediately build the CoShNet functional graph.

        :param num_classes: Number of output classes.
        :type num_classes: int
        :param input_shape: ``(height, width, channels)``; ``None`` resolves to
            ``(32, 32, 3)``.
        :type input_shape: Optional[Tuple[int, int, int]]
        :param conv_filters: Filter count per complex convolutional layer.
        :type conv_filters: Sequence[int]
        :param dense_units: Unit count per complex dense layer.
        :type dense_units: Sequence[int]
        :param shearlet_scales: Number of shearlet scales.
        :type shearlet_scales: int
        :param shearlet_directions: Number of directions per scale.
        :type shearlet_directions: int
        :param conv_kernel_size: Kernel size for the complex convolutions.
        :type conv_kernel_size: int
        :param conv_strides: Stride for the complex convolutions.
        :type conv_strides: int
        :param conv_padding: Padding mode for the complex convolutions.
        :type conv_padding: str
        :param dropout_rate: Complex dropout rate in the dense stack.
        :type dropout_rate: float
        :param kernel_regularizer: Optional kernel regularizer.
        :type kernel_regularizer: Optional[Union[str, regularizers.Regularizer]]
        :param kernel_initializer: Kernel initializer.
        :type kernel_initializer: Union[str, initializers.Initializer]
        :param include_top: Whether to build the classification head.
        :type include_top: bool
        :param epsilon: Dead knob; see the class docstring.
        :type epsilon: float
        :param kwargs: Additional keyword arguments for ``keras.Model``.
        :raises ValueError: If any configuration value is invalid.
        """

        # Set default input shape if not provided
        if input_shape is None:
            input_shape = (32, 32, 3)
            logger.info("Using default input_shape (32, 32, 3) for CIFAR-10 compatibility")

        # Store configuration before validation
        self.num_classes = num_classes
        self._input_shape = input_shape
        self.conv_filters = list(conv_filters)
        self.dense_units = list(dense_units)
        self.shearlet_scales = shearlet_scales
        self.shearlet_directions = shearlet_directions
        self.conv_kernel_size = conv_kernel_size
        self.conv_strides = conv_strides
        self.conv_padding = conv_padding
        self.dropout_rate = dropout_rate
        self.include_top = include_top
        self.epsilon = epsilon

        # Handle regularizer serialization
        if isinstance(kernel_regularizer, str):
            self.kernel_regularizer = regularizers.get(kernel_regularizer)
            self._kernel_regularizer_config = kernel_regularizer
        elif kernel_regularizer is None:
            self.kernel_regularizer = None
            self._kernel_regularizer_config = None
        else:
            self.kernel_regularizer = kernel_regularizer
            self._kernel_regularizer_config = regularizers.serialize(kernel_regularizer)

        # Handle initializer serialization
        if isinstance(kernel_initializer, str):
            self.kernel_initializer = initializers.get(kernel_initializer)
            self._kernel_initializer_config = kernel_initializer
        else:
            self.kernel_initializer = kernel_initializer
            self._kernel_initializer_config = initializers.serialize(kernel_initializer)

        # Validate configuration
        self._validate_config()

        # Store input shape properties
        self.input_height = input_shape[0]
        self.input_width = input_shape[1]
        self.input_channels = input_shape[2]

        # Create input layer
        inputs = keras.Input(shape=input_shape, name="input")

        # Build the complete model
        outputs = self._build_model(inputs)

        # Initialize the Model
        # DECISION plan-2026-08-19T163559-499b6f0e/D-066
        # `name` is a DEFAULT here, not a constant. Hard-coding it made two
        # things impossible at once: `CoShNet(name="x")` raised `TypeError: got
        # multiple values for keyword argument 'name'`, and -- now that
        # `get_config` calls `super().get_config()` -- a round trip would feed
        # the restored `name` straight back into this call and raise. Do NOT
        # restore the literal. See decisions.md D-066.
        kwargs.setdefault("name", "coshnet")
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)

        logger.info(
            f"Created CoShNet model for input {input_shape} "
            f"with {sum(self.conv_filters)} conv filters, "
            f"{sum(self.dense_units) if self.include_top else 0} dense units"
        )

    def _validate_config(self) -> None:
        """Validate every configuration parameter with a specific message.

        :raises ValueError: If ``input_shape`` is not 3D or has a non-positive
            dimension, if ``num_classes`` is not positive, if ``conv_filters``
            or ``dense_units`` is empty or contains a non-positive value, if
            any shearlet or convolution parameter is not positive, if
            ``conv_padding`` is unrecognized, if ``dropout_rate`` leaves
            ``[0, 1]``, or if ``epsilon`` is not positive.
        """
        # Input shape validation
        if len(self._input_shape) != 3:
            raise ValueError(
                f"input_shape must be 3D (height, width, channels), got {self._input_shape}"
            )

        height, width, channels = self._input_shape
        if any(dim <= 0 for dim in self._input_shape):
            raise ValueError(
                f"All dimensions in input_shape must be positive, got {self._input_shape}"
            )

        if channels not in [1, 3]:
            logger.warning(
                f"Unusual number of channels: {channels}. CoShNet typically uses 1 or 3 channels"
            )

        # Architecture validation
        if self.num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {self.num_classes}")

        if not self.conv_filters:
            raise ValueError("conv_filters cannot be empty")
        if any(f <= 0 for f in self.conv_filters):
            raise ValueError(f"All values in conv_filters must be positive, got {self.conv_filters}")

        if self.include_top:
            if not self.dense_units:
                raise ValueError("dense_units cannot be empty when include_top=True")
            if any(u <= 0 for u in self.dense_units):
                raise ValueError(f"All values in dense_units must be positive, got {self.dense_units}")

        # Shearlet validation
        if self.shearlet_scales <= 0:
            raise ValueError(f"shearlet_scales must be positive, got {self.shearlet_scales}")
        if self.shearlet_directions <= 0:
            raise ValueError(f"shearlet_directions must be positive, got {self.shearlet_directions}")

        # Layer configuration validation
        if self.conv_kernel_size <= 0:
            raise ValueError(f"conv_kernel_size must be positive, got {self.conv_kernel_size}")
        if self.conv_strides <= 0:
            raise ValueError(f"conv_strides must be positive, got {self.conv_strides}")
        if self.conv_padding not in ["valid", "same", "causal"]:
            raise ValueError(f"conv_padding must be 'valid', 'same', or 'causal', got {self.conv_padding}")

        # Regularization validation
        if not 0.0 <= self.dropout_rate <= 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1], got {self.dropout_rate}")

        # Numerical validation
        if self.epsilon <= 0.0:
            raise ValueError(f"epsilon must be positive, got {self.epsilon}")

    def _build_model(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Assemble the frontend, the complex body and (optionally) the head.

        :param inputs: Symbolic input tensor.
        :type inputs: keras.KerasTensor
        :return: Symbolic output tensor: softmax probabilities with
            ``include_top=True``, otherwise the ``complex64`` feature map.
        :rtype: keras.KerasTensor
        """
        x = inputs

        # Build shearlet transform frontend
        x = self._build_shearlet_frontend(x)

        # Build complex convolutional layers
        x = self._build_conv_layers(x)

        # Build classification head if requested
        if self.include_top:
            x = self._build_classification_head(x)

        return x

    def _build_shearlet_frontend(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Build and apply the fixed, non-trainable shearlet transform.

        :param x: Input tensor.
        :type x: keras.KerasTensor
        :return: Real-valued transform output with
            ``channels * (1 + scales * (directions + 1))`` channels at the
            input's spatial resolution.
        :rtype: keras.KerasTensor
        """
        self.shearlet = ShearletTransform(
            scales=self.shearlet_scales,
            directions=self.shearlet_directions,
            name='shearlet_transform'
        )

        x = self.shearlet(x)

        logger.debug(f"Shearlet transform output shape: {x.shape}")
        return x

    def _build_conv_layers(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Cast to ``complex64`` and apply the complex convolutional body.

        The cast gives an identically zero imaginary part; the first complex
        convolution is what makes it non-zero.

        :param x: Real-valued tensor from the shearlet frontend.
        :type x: keras.KerasTensor
        :return: ``complex64`` feature map after every conv + split-ReLU pair.
        :rtype: keras.KerasTensor
        """
        # Convert real shearlet output to complex
        x = ops.cast(x, 'complex64')

        # Complex ReLU activation (shared across layers)
        self.activation = ComplexReLU(name='complex_relu')

        # Create and apply complex convolutional layers
        self.conv_layers: List[ComplexConv2D] = []

        for i, filters in enumerate(self.conv_filters):
            # Complex convolution
            conv_layer = ComplexConv2D(
                filters=filters,
                kernel_size=self.conv_kernel_size,
                strides=self.conv_strides,
                padding=self.conv_padding,
                kernel_regularizer=self.kernel_regularizer,
                kernel_initializer=self.kernel_initializer,
                epsilon=self.epsilon,
                name=f'complex_conv_{i}'
            )
            self.conv_layers.append(conv_layer)

            # Apply convolution and activation
            x = conv_layer(x)
            x = self.activation(x)

            logger.debug(f"After conv layer {i}: {x.shape}")

        return x

    def _build_classification_head(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Build complex global pooling, the complex dense stack and the head.

        The final ``Dense`` applies softmax itself, so the returned tensor holds
        probabilities rather than logits.

        :param x: ``complex64`` feature tensor from the convolutional body.
        :type x: keras.KerasTensor
        :return: Real-valued ``(batch_size, num_classes)`` probability tensor.
        :rtype: keras.KerasTensor
        """
        # Global Average Pooling (replaces flatten for efficiency)
        self.global_avg_pool = ComplexGlobalAveragePooling2D(
            keepdims=False,
            name='global_avg_pool'
        )
        x = self.global_avg_pool(x)

        # Complex dense layers with dropout
        self.dense_layers: List[ComplexDense] = []
        self.dropout_layers: List[ComplexDropout] = []

        for i, units in enumerate(self.dense_units):
            # Complex dense layer
            dense_layer = ComplexDense(
                units=units,
                kernel_regularizer=self.kernel_regularizer,
                kernel_initializer=self.kernel_initializer,
                epsilon=self.epsilon,
                name=f'complex_dense_{i}'
            )
            self.dense_layers.append(dense_layer)

            # Dropout layer
            dropout_layer = ComplexDropout(
                rate=self.dropout_rate,
                name=f'dropout_{i}'
            )
            self.dropout_layers.append(dropout_layer)

            # Apply dense, activation, and dropout
            x = dense_layer(x)
            x = self.activation(x)
            x = dropout_layer(x)

            logger.debug(f"After dense layer {i}: {x.shape}")

        # Final real-valued classification layer
        self.classifier = layers.Dense(
            units=self.num_classes,
            activation="softmax",
            kernel_regularizer=self.kernel_regularizer,
            kernel_initializer=self.kernel_initializer,
            name='classifier'
        )

        # Convert complex to real by taking magnitude
        x = ops.abs(x)
        x = self.classifier(x)

        return x

    @classmethod
    def from_variant(
        cls,
        variant: str,
        num_classes: int = 10,
        input_shape: Optional[Tuple[int, int, int]] = None,
        **kwargs: Any
    ) -> "CoShNet":
        """Create a CoShNet model from a predefined variant.

        :param variant: One of ``"nano"``, ``"tiny"``, ``"base"``, ``"large"``,
            ``"cifar10"``, ``"imagenet"``.
        :type variant: str
        :param num_classes: Number of output classes. Defaults to 10.
        :type num_classes: int
        :param input_shape: Input shape. ``None`` resolves to
            ``(224, 224, 3)`` for the ``"imagenet"`` variant and
            ``(32, 32, 3)`` for every other.
        :type input_shape: Optional[Tuple[int, int, int]]
        :param kwargs: Additional arguments passed to the constructor; these
            override the variant's preset fields.
        :return: CoShNet model instance configured for the variant.
        :rtype: CoShNet
        :raises ValueError: If ``variant`` is not recognized.

        Example:
            >>> # CIFAR-10 model
            >>> model = CoShNet.from_variant("base", num_classes=10,
            ...                              input_shape=(32, 32, 3))
            >>> # MNIST model
            >>> model = CoShNet.from_variant("tiny", num_classes=10,
            ...                              input_shape=(28, 28, 3))
            >>> # ImageNet model
            >>> model = CoShNet.from_variant("imagenet", num_classes=1000,
            ...                              input_shape=(224, 224, 3))
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        # DECISION plan-2026-08-19T163559-499b6f0e/D-127
        # House style (`wave_field/model.py`): copy the preset, drop the
        # metadata key, then `config.update(kwargs)`. Do NOT go back to
        # splatting named preset fields alongside `**kwargs` -- every
        # documented override of one of those fields raised
        # `TypeError: got multiple values for keyword argument`
        # (MEASURED at all six sites). The `.copy()` is NOT optional and
        # NOT cosmetic: `config.update(kwargs)` on the shared
        # `MODEL_VARIANTS[variant]` dict would permanently poison the
        # class-level table for every later caller. See decisions.md D-127.
        config = cls.MODEL_VARIANTS[variant].copy()
        config.pop("description", None)
        config.update(kwargs)

        # Set default input shape based on variant if not provided
        if input_shape is None:
            if variant == "imagenet":
                input_shape = (224, 224, 3)
            else:
                input_shape = (32, 32, 3)  # Default for CIFAR-10

        logger.info(f"Creating CoShNet-{variant.upper()} model")
        logger.info(f"Input shape: {input_shape}, Classes: {num_classes}")

        return cls(
            num_classes=num_classes,
            input_shape=input_shape,
            **config
        )

    def get_config(self) -> Dict[str, Any]:
        """Return the model configuration for serialization.

        :return: Configuration dictionary containing every model parameter.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            # Core configuration
            "num_classes": self.num_classes,
            "input_shape": self._input_shape,
            # Architecture configuration
            "conv_filters": self.conv_filters,
            "dense_units": self.dense_units,
            # Shearlet transform configuration
            "shearlet_scales": self.shearlet_scales,
            "shearlet_directions": self.shearlet_directions,
            # Layer configuration
            "conv_kernel_size": self.conv_kernel_size,
            "conv_strides": self.conv_strides,
            "conv_padding": self.conv_padding,
            # Regularization and training
            "dropout_rate": self.dropout_rate,
            "kernel_regularizer": self._kernel_regularizer_config,
            # Initialization
            "kernel_initializer": self._kernel_initializer_config,
            # Advanced options
            "include_top": self.include_top,
            "epsilon": self.epsilon,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CoShNet":
        """Create a model instance from its configuration.

        :param config: Configuration dictionary from :meth:`get_config`.
        :type config: Dict[str, Any]
        :return: CoShNet model instance.
        :rtype: CoShNet
        """
        # Handle regularizer deserialization
        if config.get("kernel_regularizer"):
            config["kernel_regularizer"] = regularizers.deserialize(
                config["kernel_regularizer"]
            )

        # Handle initializer deserialization
        if config.get("kernel_initializer"):
            config["kernel_initializer"] = initializers.deserialize(
                config["kernel_initializer"]
            )

        return cls(**config)

    def summary(self, **kwargs) -> None:
        """Print the model summary with additional CoShNet-specific information.

        :param kwargs: Additional arguments passed to ``keras.Model.summary``.
        """
        super().summary(**kwargs)

        # Print additional model information
        total_conv_filters = sum(self.conv_filters)
        total_dense_units = sum(self.dense_units) if self.include_top else 0

        logger.info("CoShNet Configuration:")
        logger.info(f"  - Input shape: {self._input_shape}")
        logger.info(f"  - Shearlet scales: {self.shearlet_scales}")
        logger.info(f"  - Shearlet directions: {self.shearlet_directions}")
        logger.info(f"  - Conv layers: {len(self.conv_filters)}")
        logger.info(f"  - Total conv filters: {total_conv_filters}")
        if self.include_top:
            logger.info(f"  - Dense layers: {len(self.dense_units)}")
            logger.info(f"  - Total dense units: {total_dense_units}")
            logger.info(f"  - Number of classes: {self.num_classes}")
        logger.info(f"  - Dropout rate: {self.dropout_rate}")
        logger.info(f"  - Include top: {self.include_top}")


# ---------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------

def create_coshnet(
    variant: str = "base",
    num_classes: int = 10,
    input_shape: Optional[Tuple[int, int, int]] = None,
    **kwargs: Any
) -> CoShNet:
    """Convenience function to create CoShNet models from predefined variants.

    Thin wrapper around :meth:`CoShNet.from_variant` exposing the most common
    construction arguments at module level.

    :param variant: Model variant:

        - ``"nano"``: minimal model for resource-constrained environments
        - ``"tiny"``: small model
        - ``"base"``: standard model
        - ``"large"``: larger model for complex datasets
        - ``"cifar10"``: tuned for CIFAR-10 classification
        - ``"imagenet"``: scaled for ImageNet-style inputs

    :type variant: str
    :param num_classes: Number of output classes. Defaults to 10.
    :type num_classes: int
    :param input_shape: Input shape. ``None`` uses the variant-appropriate
        default.
    :type input_shape: Optional[Tuple[int, int, int]]
    :param kwargs: Additional arguments passed to the model constructor.
    :return: Configured CoShNet model.
    :rtype: CoShNet
    :raises ValueError: If ``variant`` is not recognized.

    Example:
        .. code-block:: python

            # Base CoShNet for CIFAR-10
            model = create_coshnet("base", num_classes=10,
                                   input_shape=(32, 32, 3))

            # Tiny CoShNet for MNIST
            model = create_coshnet("tiny", num_classes=10,
                                   input_shape=(28, 28, 3))

            # ImageNet CoShNet
            model = create_coshnet("imagenet", num_classes=1000)
    """
    logger.info(f"Creating CoShNet-{variant.upper()} model")

    return CoShNet.from_variant(
        variant=variant,
        num_classes=num_classes,
        input_shape=input_shape,
        **kwargs
    )

# ---------------------------------------------------------------------