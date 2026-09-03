"""
Convert a video clip into tubelet patch embeddings.

This module mirrors ``PatchEmbedding2D`` (``patch_embedding.py``) for a third,
temporal axis. It ports ``PatchEmbed3D`` from the LeVJEPA PyTorch reference,
which is channels-first ``(B, C, T, H, W)`` with a ``Conv3d``. This repo's
convention is channels-last, so the port is channels-last throughout:
``(batch, T, H, W, C)`` in, projected by a single ``Conv3D``.

Architecture:
    A single strided ``Conv3D`` does both the tubelet cut and the projection.
    Its kernel size is ``(tubelet_size, patch_h, patch_w)`` and its stride
    equals the kernel size, so the filter reads one non-overlapping tubelet at
    a time and projects it with the same weights. The filter count equals
    ``embed_dim``.

    **Architecture Overview:**

    .. code-block:: text

        ┌────────────────────────────────────────────┐
        │  Input (batch, T, H, W, C)                 │
        └──────────────────┬───────────────────────────┘
                           ▼
        ┌────────────────────────────────────────────┐
        │  Conv3D(filters=embed_dim,                 │
        │         kernel=(tubelet, P_h, P_w),         │
        │         stride=(tubelet, P_h, P_w),         │
        │         padding="valid")                    │
        │  → (batch, T/tubelet, H/P_h, W/P_w, D)      │
        └──────────────────┬───────────────────────────┘
                           │
             flatten=False │ flatten=True (default)
             ┌─────────────┴─────────────┐
             ▼                           ▼
        ┌──────────────────┐  ┌────────────────────────────┐
        │ return the 5D    │  │ Reshape → (batch, N, D)    │
        │ grid unchanged   │  │ N = (T/tub)*(H/P_h)*(W/P_w)│
        └──────────────────┘  └────────────────────────────┘

    ``tubelet_size=1`` degenerates the temporal kernel/stride to 1, which is a
    well-defined ``Conv3D`` configuration (no special-casing needed) and is
    the natural way to treat a single-frame "video" as a still image while
    keeping the same layer.

Foundational Mathematics:
    Identical in kind to ``PatchEmbedding2D``, extended with a third (time)
    axis. A clip ``X`` in ``R^(T x H x W x C)`` is cut into
    ``N = (T/T_t) * (H/P_h) * (W/P_w)`` flattened tubelets
    ``x_p`` in ``R^(T_t*P_h*P_w*C)``, each projected by a learned matrix
    ``E`` in ``R^((T_t*P_h*P_w*C) x D)``: ``z_i = x_p^(i) * E``. The
    convolution computes exactly this, with its kernel weights as ``E`` and
    its stride performing the tubelet cut.

References:
    - LeVJEPA PyTorch reference, ``module.py::PatchEmbed3D`` (pasted
      transcript; no public arXiv id in this plan's context).
    - Tong, Z., et al. (2022). "VideoMAE: Masked Autoencoders are
      Data-Efficient Learners for Self-Supervised Video Pre-Training".
      arXiv:2203.12602 (the tubelet-embedding idea this layer implements).
"""

import keras
from typing import Optional, Union, Tuple, Any, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.embedding.patch_embed_3d")
class PatchEmbed3D(keras.layers.Layer):
    """Cut a video clip into tubelets and project each tubelet to a vector.

    A single ``Conv3D`` does both jobs, exactly as ``PatchEmbedding2D`` does
    for images with a ``Conv2D``. Its kernel size and its stride both equal
    ``(tubelet_size, patch_h, patch_w)``, so it reads one non-overlapping
    spatio-temporal tubelet per step and projects that tubelet with the same
    weights. For a clip ``X`` in ``R^(T x H x W x C)`` the layer produces
    ``N = (T/T_t) * (H/P_h) * (W/P_w)`` vectors of dimension ``embed_dim``.

    The output layout depends on ``flatten``. The default ``True`` returns
    the 3D sequence a Transformer encoder expects. ``False`` returns the raw
    5D grid.

    :param patch_size: Spatial patch size. An integer gives square patches, a
        tuple ``(height, width)`` gives rectangular ones. Every entry must be
        positive. Stored as a 2-tuple.
    :type patch_size: Union[int, Tuple[int, int]]
    :param tubelet_size: Temporal patch size (number of frames per tubelet).
        Must be positive. ``1`` treats every frame as its own tubelet, the
        natural single-frame / image-as-video case.
    :type tubelet_size: int
    :param embed_dim: Embedding dimension per tubelet. Must be positive.
    :type embed_dim: int
    :param kernel_initializer: Initializer for the projection kernel.
        Defaults to ``"glorot_normal"``.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for the projection
        kernel.
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
    :param flatten: Whether to fold the tubelet grid into a sequence. ``True``
        (default) returns ``(batch, num_tubelets, embed_dim)``. ``False``
        returns ``(batch, T/T_t, H/P_h, W/P_w, embed_dim)``.
    :type flatten: bool
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    :ivar proj: The ``Conv3D`` that performs both the tubelet cut and the
        projection. Created in ``__init__`` and built in ``build()``.
    :vartype proj: keras.layers.Conv3D

    Input shape:
        5D tensor with shape ``(batch_size, T, height, width, channels)``.
        ``T``, ``height`` and ``width`` must be divisible by ``tubelet_size``,
        the patch height and the patch width respectively.

    Output shape:
        3D tensor ``(batch_size, num_tubelets, embed_dim)`` when ``flatten``
        is ``True``, otherwise the 5D tensor
        ``(batch_size, T // tubelet_size, height // P_h, width // P_w,
        embed_dim)``.

    :raises ValueError: If ``patch_size``, ``tubelet_size`` or ``embed_dim``
        is not positive, or if ``patch_size`` is a sequence whose length is
        not 2. Raised from ``__init__``.
    :raises ValueError: If the input is not 5D, or if a statically known
        ``T``, height or width is not divisible by the matching tubelet
        dimension. Raised from ``build()``, not from ``call()``.

    Example:

    .. code-block:: python

        import keras
        from dl_techniques.layers.embedding.patch_embed_3d import (
            PatchEmbed3D,
        )

        patch = PatchEmbed3D(patch_size=16, tubelet_size=2, embed_dim=768)
        clip = keras.random.normal((2, 16, 224, 224, 3))
        patch(clip).shape  # (2, 8 * 14 * 14, 768)
    """

    def __init__(
        self,
        patch_size: Union[int, Tuple[int, int]] = 16,
        tubelet_size: int = 2,
        embed_dim: int = 768,
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

        :param patch_size: Spatial patch size, integer or ``(height, width)``.
        :type patch_size: Union[int, Tuple[int, int]]
        :param tubelet_size: Temporal patch size.
        :type tubelet_size: int
        :param embed_dim: Embedding dimension per tubelet.
        :type embed_dim: int
        :param kernel_initializer: Initializer for the projection kernel.
        :type kernel_initializer: Union[str, keras.initializers.Initializer]
        :param kernel_regularizer: Optional regularizer for the projection
            kernel.
        :type kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
        :param bias_initializer: Initializer for the bias vector.
        :type bias_initializer: Union[str, keras.initializers.Initializer]
        :param bias_regularizer: Optional regularizer for the bias vector.
        :type bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
        :param activation: Activation applied by the projection.
        :type activation: Optional[Union[str, callable]]
        :param use_bias: Whether the projection has a bias.
        :type use_bias: bool
        :param flatten: Whether to fold the tubelet grid into a sequence.
        :type flatten: bool
        :param kwargs: Additional keyword arguments for the ``Layer`` base
            class.
        :type kwargs: Any
        :raises ValueError: If ``patch_size``, ``tubelet_size`` or
            ``embed_dim`` is not positive, or if ``patch_size`` is a sequence
            of length other than 2.
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

        if tubelet_size <= 0:
            raise ValueError(f"tubelet_size must be positive, got {tubelet_size}")

        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")

        # Store ALL configuration parameters
        self.tubelet_size = tubelet_size
        self.embed_dim = embed_dim
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.flatten = flatten

        self._kernel_size = (self.tubelet_size, self.patch_size[0], self.patch_size[1])

        # CREATE sub-layer in __init__ (modern Keras 3 pattern)
        self.proj = keras.layers.Conv3D(
            filters=self.embed_dim,
            kernel_size=self._kernel_size,
            strides=self._kernel_size,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bias_regularizer,
            use_bias=self.use_bias,
            activation=self.activation,
            padding="valid",
            name="projection"
        )

        logger.info(f"Initialized PatchEmbed3D with patch_size={self.patch_size}, "
                    f"tubelet_size={self.tubelet_size}, embed_dim={self.embed_dim}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Check the input shape and build the projection convolution.

        The divisibility check runs here, not in ``call()``. A ``None``
        frame count, height or width skips it, so a fully dynamic shape
        reaches the convolution unchecked.

        :param input_shape: Shape of the input tensor
            ``(batch_size, T, height, width, channels)``.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the input is not 5D, or if a statically known
            ``T``, height or width is not divisible by the matching tubelet
            dimension.
        """
        if self.built:
            return

        # Validate input shape
        if len(input_shape) != 5:
            raise ValueError(f"Expected 5D input (batch_size, T, height, width, channels), "
                             f"got {len(input_shape)}D input with shape {input_shape}")

        # Validate that T, height and width are divisible by the tubelet
        # dimensions (if known).
        num_frames, height, width = input_shape[1], input_shape[2], input_shape[3]
        if num_frames is not None and num_frames % self.tubelet_size != 0:
            raise ValueError(f"Input frame count ({num_frames}) must be divisible by "
                             f"tubelet_size ({self.tubelet_size})")
        if height is not None and height % self.patch_size[0] != 0:
            raise ValueError(f"Input height ({height}) must be divisible by "
                             f"patch height ({self.patch_size[0]})")
        if width is not None and width % self.patch_size[1] != 0:
            raise ValueError(f"Input width ({width}) must be divisible by "
                             f"patch width ({self.patch_size[1]})")

        # Build the sub-layer here rather than letting the first call do it.
        # A sub-layer built lazily inside `call()` has no weights when
        # `.keras` saving walks the tree, so its kernel reloads as fresh
        # values.
        self.proj.build(input_shape)

        logger.info(f"Built PatchEmbed3D with input_shape={input_shape}")

        # Always call parent build at the end
        super().build(input_shape)

    def call(self, inputs, training: Optional[bool] = None) -> keras.KerasTensor:
        """Project video tubelets into embedding vectors.

        :param inputs: Input tensor of shape
            ``(batch_size, T, height, width, channels)``.
        :type inputs: keras.KerasTensor
        :param training: Whether the layer should behave in training mode or
            inference mode.
        :type training: Optional[bool]
        :return: Embedded tubelets. Shape
            ``(batch_size, num_tubelets, embed_dim)`` when ``flatten`` is
            ``True``, otherwise the 5D grid
            ``(batch_size, t_patches, h_patches, w_patches, embed_dim)``.
        :rtype: keras.KerasTensor
        """
        # Apply the convolution to extract and embed tubelets. The result is
        # (batch_size, t_patches, h_patches, w_patches, embed_dim).
        x = self.proj(inputs, training=training)

        if not self.flatten:
            return x

        # Flatten the tubelet grid into a sequence. Let the backend infer the
        # tubelet-count axis with -1 (graph-safe; avoids multiplying three
        # symbolic keras.ops.shape() scalars).
        batch_size = keras.ops.shape(x)[0]
        x = keras.ops.reshape(x, (batch_size, -1, self.embed_dim))

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        An unknown ``T``, height or width propagates to an unknown tubelet
        count.

        :param input_shape: Shape of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: ``(batch_size, num_tubelets, embed_dim)`` when ``flatten`` is
            ``True``, otherwise
            ``(batch_size, t_patches, h_patches, w_patches, embed_dim)``.
        :rtype: Tuple[Optional[int], ...]
        :raises ValueError: If ``input_shape`` is not 5D.
        """
        if len(input_shape) != 5:
            raise ValueError(f"Expected 5D input shape, got {len(input_shape)}D")

        batch_size = input_shape[0]
        num_frames, height, width = input_shape[1], input_shape[2], input_shape[3]

        t_patches = num_frames // self.tubelet_size if num_frames is not None else None
        h_patches = height // self.patch_size[0] if height is not None else None
        w_patches = width // self.patch_size[1] if width is not None else None

        if not self.flatten:
            return (batch_size, t_patches, h_patches, w_patches, self.embed_dim)

        if t_patches is not None and h_patches is not None and w_patches is not None:
            num_tubelets = t_patches * h_patches * w_patches
        else:
            num_tubelets = None

        return (batch_size, num_tubelets, self.embed_dim)

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
            "tubelet_size": self.tubelet_size,
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
