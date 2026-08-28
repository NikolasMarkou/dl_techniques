"""
Convert a 2D image or a 1D sequence into patch embeddings.

This module holds two layers. ``PatchEmbedding2D`` turns an image into the
token sequence a Vision Transformer consumes. ``PatchEmbedding1D`` does the
same for a time series, and its patches may overlap.

Architecture:
    Both layers are a single strided convolution. The kernel size equals the
    patch size, so the filter sees exactly one patch at a time. The stride
    equals the patch size, so the filter steps from one patch to the next.
    The filter count equals the embedding dimension, so the same operation
    also projects each patch into its embedding.

    ``PatchEmbedding2D`` produces a 4D grid of shape
    ``(batch, H/P_h, W/P_w, embed_dim)``. It then folds the two spatial axes
    into one sequence axis and returns ``(batch, num_patches, embed_dim)``.
    Pass ``flatten=False`` to get the 4D grid instead. Window-attention
    backbones need that layout.

    ``PatchEmbedding1D`` always returns 3D. Its stride is configurable, so
    patches may overlap, and its padding mode may be ``'same'``, ``'valid'``
    or ``'causal'``.

Foundational Mathematics:
    The operation is a patching step followed by a linear projection. An
    input image ``X`` in ``R^(H x W x C)`` is cut into ``N`` flattened
    patches ``x_p`` in ``R^(P*P*C)``, where ``P`` is the patch size and
    ``N = (H*W)/P^2``.

    Each patch is projected by a learned matrix ``E`` in
    ``R^((P^2*C) x D)``, where ``D`` is the embedding dimension::

        z_i = x_p^(i) * E

    The convolution computes exactly this. Its kernel weights are ``E`` and
    its stride does the patching, so one fused operation replaces an
    explicit cut-then-project.

References:
    - Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X.,
      Unterthiner, T., ... & Houlsby, N. (2020). "An Image is Worth 16x16
      Words: Transformers for Image Recognition at Scale". Patch embedding
      is the input stage of the Vision Transformer described there.
"""

import keras
from keras import ops
from typing import Optional, Union, Tuple, Any, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class PatchEmbedding2D(keras.layers.Layer):
    """Cut an image into patches and project each patch to a vector.

    A single ``Conv2D`` does both jobs. Its kernel size and its stride both
    equal the patch size, so it reads one non-overlapping patch per step and
    projects that patch with the same weights. For an input image ``X`` in
    ``R^(H x W x C)`` the layer produces ``N = (H/P_h) * (W/P_w)`` vectors of
    dimension ``embed_dim``, where ``P_h`` and ``P_w`` are the patch height
    and width. The projection ``z_i = x_p^(i) * E`` is learned end to end,
    with ``E`` in ``R^((P_h*P_w*C) x D)``.

    The output layout depends on ``flatten``. The default ``True`` returns
    the 3D sequence a Transformer encoder expects. ``False`` returns the raw
    4D grid that window-attention backbones expect.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────────┐
        │  Input (batch, H, W, C)              │
        └──────────────────┬───────────────────┘
                           ▼
        ┌──────────────────────────────────────┐
        │  Conv2D(filters=embed_dim,           │
        │         kernel=patch_size,           │
        │         stride=patch_size,           │
        │         padding="valid")             │
        │  → (batch, H/P_h, W/P_w, embed_dim)  │
        └──────────────────┬───────────────────┘
                           │
             flatten=False │ flatten=True (default)
             ┌─────────────┴─────────────┐
             ▼                           ▼
        ┌──────────────────┐  ┌────────────────────────────┐
        │ return the 4D    │  │ Reshape → (batch, N, D)    │
        │ grid unchanged   │  │ N = (H/P_h) * (W/P_w)      │
        └──────────────────┘  └────────────────────────────┘

    :param patch_size: Patch size. An integer gives square patches, a tuple
        ``(height, width)`` gives rectangular ones. Every entry must be
        positive. Stored as a 2-tuple.
    :type patch_size: Union[int, Tuple[int, int]]
    :param embed_dim: Embedding dimension per patch. Must be positive.
    :type embed_dim: int
    :param kernel_initializer: Initializer for the projection matrix.
        Defaults to ``"glorot_normal"``.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for the projection
        matrix.
    :type kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param bias_initializer: Initializer for the bias vector. Defaults to
        ``"zeros"``.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param bias_regularizer: Optional regularizer for the bias vector.
    :type bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param activation: Activation applied by the projection convolution.
        Defaults to ``"linear"``, which is no activation.
    :type activation: Optional[Union[str, callable]]
    :param use_bias: Whether the projection has a bias. Defaults to ``True``.
    :type use_bias: bool
    :param flatten: Whether to fold the patch grid into a sequence. ``True``
        (default) returns ``(batch, num_patches, embed_dim)``. ``False``
        returns ``(batch, H/P_h, W/P_w, embed_dim)``, the layout a
        window-attention backbone such as the SAM image encoder consumes.
    :type flatten: bool
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    :ivar proj: The ``Conv2D`` that performs both the patching and the
        projection. Created in ``__init__`` and built in ``build()``.
    :vartype proj: keras.layers.Conv2D

    Input shape:
        4D tensor with shape ``(batch_size, height, width, channels)``.
        ``height`` and ``width`` must be divisible by the patch height and
        patch width.

    Output shape:
        3D tensor ``(batch_size, num_patches, embed_dim)`` when ``flatten``
        is ``True``, otherwise the 4D tensor
        ``(batch_size, height // P_h, width // P_w, embed_dim)``.

    :raises ValueError: If ``patch_size`` or ``embed_dim`` is not positive,
        or if ``patch_size`` is a sequence whose length is not 2. Raised from
        ``__init__``.
    :raises ValueError: If the input is not 4D, or if a statically known
        height or width is not divisible by the matching patch dimension.
        Raised from ``build()``, not from ``call()``.

    Example:

    .. code-block:: python

        import keras
        from dl_techniques.layers.embedding import (
            create_embedding_layer,
        )

        patch = create_embedding_layer(
            "patch_2d", patch_size=16, embed_dim=64,
        )
        images = keras.random.normal((2, 32, 32, 3))
        patch(images).shape  # (2, 4, 64)
    """

    def __init__(
        self,
        patch_size: Union[int, Tuple[int, int]],
        embed_dim: int,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_normal",
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
        bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        activation: Optional[Union[str, callable]] = "linear",
        use_bias: bool = True,
        flatten: bool = True,
        **kwargs: Any
    ) -> None:
        """Validate the configuration and create the projection convolution.

        ``patch_size`` is normalized to a 2-tuple here, so ``get_config()``
        always reports a tuple even when an integer was passed.

        :param patch_size: Patch size, integer or ``(height, width)``.
        :type patch_size: Union[int, Tuple[int, int]]
        :param embed_dim: Embedding dimension per patch.
        :type embed_dim: int
        :param kernel_initializer: Initializer for the projection matrix.
        :type kernel_initializer: Union[str, keras.initializers.Initializer]
        :param kernel_regularizer: Optional regularizer for the projection
            matrix.
        :type kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
        :param bias_initializer: Initializer for the bias vector.
        :type bias_initializer: Union[str, keras.initializers.Initializer]
        :param bias_regularizer: Optional regularizer for the bias vector.
        :type bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
        :param activation: Activation applied by the projection.
        :type activation: Optional[Union[str, callable]]
        :param use_bias: Whether the projection has a bias.
        :type use_bias: bool
        :param flatten: Whether to fold the patch grid into a sequence.
        :type flatten: bool
        :param kwargs: Additional keyword arguments for the ``Layer`` base
            class.
        :type kwargs: Any
        :raises ValueError: If ``patch_size`` or ``embed_dim`` is not
            positive, or if ``patch_size`` is a sequence of length other
            than 2.
        """
        super().__init__(**kwargs)

        # Validate inputs
        if isinstance(patch_size, int):
            if patch_size <= 0:
                raise ValueError(f"patch_size must be positive, got {patch_size}")
            self.patch_size = (patch_size, patch_size)
        else:
            if len(patch_size) != 2 or any(p <= 0 for p in patch_size):
                raise ValueError(f"patch_size must be positive integer or tuple of 2 positive integers, got {patch_size}")
            self.patch_size = tuple(patch_size)

        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")

        # Store ALL configuration parameters
        self.embed_dim = embed_dim
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.flatten = flatten

        # CREATE sub-layer in __init__ (modern Keras 3 pattern)
        self.proj = keras.layers.Conv2D(
            filters=self.embed_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bias_regularizer,
            use_bias=self.use_bias,
            activation=self.activation,
            padding="valid",
            name="projection"
        )

        logger.info(f"Initialized PatchEmbedding2D with patch_size={self.patch_size}, "
                    f"embed_dim={self.embed_dim}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Check the input shape and build the projection convolution.

        The divisibility check runs here, not in ``call()``. A ``None``
        height or width skips it, so a fully dynamic spatial shape reaches
        the convolution unchecked.

        :param input_shape: Shape of the input tensor
            ``(batch_size, height, width, channels)``.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the input is not 4D, or if a statically known
            height or width is not divisible by the matching patch
            dimension.
        """
        if self.built:
            return

        # Validate input shape
        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D input (batch_size, height, width, channels), "
                             f"got {len(input_shape)}D input with shape {input_shape}")

        # Validate that height and width are divisible by patch size (if known)
        height, width = input_shape[1], input_shape[2]
        if height is not None and height % self.patch_size[0] != 0:
            raise ValueError(f"Input height ({height}) must be divisible by "
                             f"patch height ({self.patch_size[0]})")
        if width is not None and width % self.patch_size[1] != 0:
            raise ValueError(f"Input width ({width}) must be divisible by "
                             f"patch width ({self.patch_size[1]})")

        # CRITICAL: Explicitly build sub-layers for robust serialization
        self.proj.build(input_shape)

        logger.info(f"Built PatchEmbedding2D with input_shape={input_shape}")

        # Always call parent build at the end
        super().build(input_shape)

    def call(self, inputs, training: Optional[bool] = None) -> keras.KerasTensor:
        """Project image patches into embedding vectors.

        :param inputs: Input tensor of shape
            ``(batch_size, height, width, channels)``.
        :type inputs: keras.KerasTensor
        :param training: Whether the layer should behave in training mode or
            inference mode.
        :type training: Optional[bool]
        :return: Embedded patches. Shape
            ``(batch_size, num_patches, embed_dim)`` when ``flatten`` is
            ``True``, otherwise the 4D grid
            ``(batch_size, h_patches, w_patches, embed_dim)``.
        :rtype: keras.KerasTensor
        """
        # Apply the convolution to extract and embed patches. The result is
        # (batch_size, h_patches, w_patches, embed_dim).
        x = self.proj(inputs, training=training)

        # DECISION plan_2026-06-16_6e8c78a3/D-009
        # When flatten=False, return the raw 4D spatial grid. Do NOT always
        # flatten: window-attention backbones consume the 4D layout and add a
        # 4D pos_embed to it, and the SAM image encoder is such a consumer.
        # The default flatten=True preserves the 3D sequence contract every
        # other caller depends on. Re-measured on 2026-08-28: src/ builds this
        # layer at 11 sites, 6 by direct construction and 5 through
        # create_embedding_layer('patch_2d', ...), and exactly one of them,
        # models/vision_language/sam/sam1/image_encoder.py, passes
        # flatten=False. The originating plan directory is gone, so this
        # comment is the only record of the rationale. Do not delete it.
        if not self.flatten:
            return x

        # Flatten the patch grid into a sequence. Let the backend infer the
        # patch-count axis with -1 (graph-safe; avoids multiplying two symbolic
        # ops.shape() scalars, which some backends reject as a reshape arg).
        batch_size = ops.shape(x)[0]
        x = ops.reshape(x, (batch_size, -1, self.embed_dim))

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        An unknown height or width propagates to an unknown patch count.

        :param input_shape: Shape of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: ``(batch_size, num_patches, embed_dim)`` when ``flatten`` is
            ``True``, otherwise
            ``(batch_size, h_patches, w_patches, embed_dim)``.
        :rtype: Tuple[Optional[int], ...]
        :raises ValueError: If ``input_shape`` is not 4D.
        """
        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D input shape, got {len(input_shape)}D")

        batch_size = input_shape[0]
        height, width = input_shape[1], input_shape[2]

        # Calculate number of patches per spatial axis
        if height is not None:
            h_patches = height // self.patch_size[0]
        else:
            h_patches = None
        if width is not None:
            w_patches = width // self.patch_size[1]
        else:
            w_patches = None

        if not self.flatten:
            # Return the raw 4D spatial grid (window-attention backbones).
            return (batch_size, h_patches, w_patches, self.embed_dim)

        if h_patches is not None and w_patches is not None:
            num_patches = h_patches * w_patches
        else:
            num_patches = None

        return (batch_size, num_patches, self.embed_dim)

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration of the layer for serialization.

        ``patch_size`` is reported as a 2-tuple, not as the integer a caller
        may have passed.

        :return: Dictionary holding every ``__init__`` parameter.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
            "activation": keras.activations.serialize(self.activation),
            "use_bias": self.use_bias,
            "flatten": self.flatten,
        })
        return config


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class PatchEmbedding1D(keras.layers.Layer):
    """Cut a 1D sequence into patches and project each patch to a vector.

    A single ``Conv1D`` does both jobs, as in ``PatchEmbedding2D``. The
    difference is the stride: it defaults to ``patch_size`` for
    non-overlapping patches but may be set smaller, in which case
    consecutive patches overlap. The kernel weights are the projection
    matrix ``E`` in ``R^(patch_size*F x embed_dim)``, where ``F`` is the
    feature count.

    NaN inputs are replaced by zero before the convolution. Without that a
    single missing sample would poison every patch that covers it.

    The output is always 3D. There is no ``flatten`` switch here.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────────┐
        │  Input (batch, seq_len, features)    │
        └──────────────────┬───────────────────┘
                           ▼
        ┌──────────────────────────────────────┐
        │  where(isnan(x), 0.0, x)             │
        └──────────────────┬───────────────────┘
                           ▼
        ┌──────────────────────────────────────┐
        │  Conv1D(filters=embed_dim,           │
        │         kernel=patch_size,           │
        │         stride=stride,               │
        │         padding=padding)             │
        └──────────────────┬───────────────────┘
                           ▼
        ┌──────────────────────────────────────┐
        │  Output (batch, out_len, embed_dim)  │
        └──────────────────────────────────────┘

    :param patch_size: Length of each patch. Must be positive.
    :type patch_size: int
    :param embed_dim: Embedding dimension per patch. Must be positive.
    :type embed_dim: int
    :param stride: Step between patch starts. ``None`` (default) means
        ``patch_size``, which gives non-overlapping patches. A smaller value
        makes patches overlap. Must be positive when given.
    :type stride: Optional[int]
    :param padding: Convolution padding mode, one of ``'same'``,
        ``'valid'`` or ``'causal'``. Defaults to ``'causal'``, which pads on
        the left only, so no output position reads a future sample.
    :type padding: str
    :param use_bias: Whether the projection has a bias. Defaults to
        ``True``.
    :type use_bias: bool
    :param kernel_initializer: Initializer for the kernel weights. Defaults
        to ``"glorot_uniform"``.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for the bias vector. Defaults to
        ``"zeros"``.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    :ivar stride: The resolved stride, never ``None``. A ``stride=None``
        argument is stored as ``patch_size``, and ``get_config()`` reports
        that resolved integer, so a saved and reloaded layer carries an
        explicit stride.
    :vartype stride: int
    :ivar embedding: The ``Conv1D`` that performs both the patching and the
        projection. Created in ``__init__`` and built in ``build()``.
    :vartype embedding: keras.layers.Conv1D

    Input shape:
        3D tensor with shape ``(batch_size, seq_len, features)``.

    Output shape:
        3D tensor with shape ``(batch_size, output_len, embed_dim)``. For
        ``'same'`` and ``'causal'`` padding ``output_len`` is
        ``ceil(seq_len / stride)``. For ``'valid'`` padding it is
        ``(seq_len - patch_size) // stride + 1``.

    :raises ValueError: If ``patch_size``, ``embed_dim`` or ``stride`` is
        not positive, or if ``padding`` is not one of the three allowed
        values. Raised from ``__init__``.
    :raises ValueError: If the input is not 3D. Raised from ``build()``.

    Example:

    .. code-block:: python

        import keras
        from dl_techniques.layers.embedding import (
            create_embedding_layer,
        )

        patch = create_embedding_layer(
            "patch_1d", patch_size=8, embed_dim=32,
        )
        series = keras.random.normal((2, 64, 3))
        patch(series).shape  # (2, 8, 32)
    """

    def __init__(
        self,
        patch_size: int,
        embed_dim: int,
        stride: Optional[int] = None,
        padding: str = 'causal',
        use_bias: bool = True,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
        **kwargs: Any
    ) -> None:
        """Validate the configuration and create the projection convolution.

        A ``stride`` of ``None`` is resolved to ``patch_size`` here, so the
        attribute and the serialized config always hold an integer.

        :param patch_size: Length of each patch.
        :type patch_size: int
        :param embed_dim: Embedding dimension per patch.
        :type embed_dim: int
        :param stride: Step between patch starts, or ``None`` for
            ``patch_size``.
        :type stride: Optional[int]
        :param padding: Convolution padding mode.
        :type padding: str
        :param use_bias: Whether the projection has a bias.
        :type use_bias: bool
        :param kernel_initializer: Initializer for the kernel weights.
        :type kernel_initializer: Union[str, keras.initializers.Initializer]
        :param bias_initializer: Initializer for the bias vector.
        :type bias_initializer: Union[str, keras.initializers.Initializer]
        :param kwargs: Additional keyword arguments for the ``Layer`` base
            class.
        :type kwargs: Any
        :raises ValueError: If ``patch_size``, ``embed_dim`` or ``stride``
            is not positive, or if ``padding`` is not ``'same'``,
            ``'valid'`` or ``'causal'``.
        """
        super().__init__(**kwargs)

        # Validate inputs
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if stride is not None and stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        if padding not in ['same', 'valid', 'causal']:
            raise ValueError(f"padding must be one of ['same', 'valid', 'causal'], got {padding}")

        # Store ALL configuration parameters
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.stride = stride if stride is not None else patch_size
        self.padding = padding
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)

        # CREATE sub-layer in __init__ (modern Keras 3 pattern)
        self.embedding = keras.layers.Conv1D(
            filters=self.embed_dim,
            kernel_size=self.patch_size,
            strides=self.stride,
            padding=self.padding,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            name="patch_embedding"
        )

        logger.info(f"Initialized PatchEmbedding1D with patch_size={self.patch_size}, "
                    f"embed_dim={self.embed_dim}, stride={self.stride}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Check the input shape and build the projection convolution.

        :param input_shape: Shape of the input tensor
            ``(batch_size, seq_len, features)``.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the input is not 3D.
        """
        if self.built:
            return

        # Validate input shape
        if len(input_shape) != 3:
            raise ValueError(f"Expected 3D input (batch_size, seq_len, features), "
                             f"got {len(input_shape)}D input with shape {input_shape}")

        # CRITICAL: Explicitly build sub-layers for robust serialization
        self.embedding.build(input_shape)

        logger.info(f"Built PatchEmbedding1D with input_shape={input_shape}")

        # Always call parent build at the end
        super().build(input_shape)

    def call(self, inputs, training: Optional[bool] = None) -> keras.KerasTensor:
        """Convert inputs to patches and embed them.

        :param inputs: Input tensor of shape
            ``(batch_size, seq_len, features)``.
        :type inputs: keras.KerasTensor
        :param training: Whether the layer should behave in training mode or
            inference mode.
        :type training: Optional[bool]
        :return: Embedded patches tensor of shape
            ``(batch_size, output_len, embed_dim)``.
        :rtype: keras.KerasTensor
        """
        # Replace NaN samples with zeros. One missing sample would otherwise
        # make every patch covering it NaN, and the projection would spread
        # that NaN across all embed_dim channels.
        x = ops.where(ops.isnan(inputs), 0.0, inputs)

        # Apply patch embedding
        embedded = self.embedding(x, training=training)

        return embedded

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape after patch embedding.

        An unknown ``seq_len`` propagates to an unknown ``output_len``.

        :param input_shape: Shape of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple ``(batch_size, output_len, embed_dim)``.
        :rtype: Tuple[Optional[int], ...]
        :raises ValueError: If ``input_shape`` is not 3D.
        """
        if len(input_shape) != 3:
            raise ValueError(f"Expected 3D input shape, got {len(input_shape)}D")

        batch_size = input_shape[0]
        seq_len = input_shape[1]

        if seq_len is None:
            output_len = None
        else:
            if self.padding == 'valid':
                output_len = (seq_len - self.patch_size) // self.stride + 1
            elif self.padding == 'same':
                output_len = (seq_len + self.stride - 1) // self.stride
            # 'causal': Conv1D pads (patch_size - 1) samples on the left and
            # then convolves as 'valid', so the output length matches 'same'
            # semantics, ceil(seq_len / stride).
            else:
                output_len = (seq_len + self.stride - 1) // self.stride

        return (batch_size, output_len, self.embed_dim)

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration of the layer for serialization.

        ``stride`` is reported as the resolved integer, not as the ``None``
        a caller may have passed.

        :return: Dictionary holding every ``__init__`` parameter.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
            "stride": self.stride,
            "padding": self.padding,
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
        })
        return config


# ---------------------------------------------------------------------
