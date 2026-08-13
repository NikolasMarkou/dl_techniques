"""
Reparameterizable conditional positional encoding (RepCPE) of the FastViT backbone.

Vision transformers need positional information, but a fixed learned table ties
the model to one input resolution. Conditional positional encodings sidestep
this: the "encoding" is produced *from the feature map itself* by a depthwise
convolution, so it is generated at whatever resolution the input happens to
have and is translation-equivariant by construction.

FastViT's variant is deliberately trivial in structure:

.. code-block:: text

    out = depthwise_conv_{k x k}(x) + x

The convolution is depthwise (one group per channel), stride 1, ``padding='same'``
and — unlike almost every other convolution in the backbone — **biased**. The
additive skip is what makes the whole thing reparameterizable: at inference the
identity can be folded into the convolution kernel by adding a centred Dirac
delta, collapsing the block into a single depthwise convolution. That fusion is
NOT implemented here (this port deliberately ships the train-time graph only),
but the skip connection is exactly the structure that makes it possible, and it
is load-bearing for the forward pass regardless.

References:
    - Chu et al., 2021. Conditional Positional Encodings for Vision
      Transformers. (https://arxiv.org/abs/2102.10882)
    - Vasu et al., 2023. FastViT: A Fast Hybrid Vision Transformer using
      Structural Reparameterization. (https://arxiv.org/abs/2303.14189)
"""

import keras
from typing import Optional, Union, Tuple, Dict, Any
from keras import ops, layers, initializers, regularizers

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class RepConditionalPosEnc(keras.layers.Layer):
    """Conditional positional encoding: ``out = depthwise_conv(x) + x``.

    Channels-last transcription of timm's ``RepConditionalPosEnc`` as used by
    FastViT / MobileCLIP2's MCi backbone. Preserves both spatial resolution and
    channel count.

    **Architecture**

    .. code-block:: text

        ┌──────────────────────────────────────────────┐
        │           Input [B, H, W, dim]               │
        └──────────┬────────────────────────┬──────────┘
                   │                        │
                   ▼                        │
        ┌───────────────────────────┐       │ identity
        │ DepthwiseConv2D           │       │
        │   kernel = spatial_shape  │       │
        │   stride 1, 'same'        │       │
        │   use_bias = True         │       │
        └───────────┬───────────────┘       │
                    │                       │
                    └──────────►  (+)  ◄────┘
                                  │
                                  ▼
        ┌──────────────────────────────────────────────┐
        │           Output [B, H, W, dim]              │
        └──────────────────────────────────────────────┘

    .. note::
       The reference writes the convolution as ``Conv2d(dim, dim_out, k,
       groups=dim)``, which is depthwise only when ``dim_out == dim``. Every
       real MCi call site satisfies that. Rather than silently building a
       different graph for ``dim_out != dim``, this layer raises
       :class:`ValueError`.

    .. note::
       ``padding='same'`` with a 7x7 kernel is well defined on feature maps
       smaller than the kernel (e.g. the 4x4 map of a 5-stage variant at 256px)
       — the kernel is simply clipped against the zero padding.

    :param dim: Number of input channels. Must be positive.
    :type dim: int
    :param dim_out: Number of output channels. Defaults to ``dim`` when ``None``.
        Must equal ``dim`` (see the note above).
    :type dim_out: Optional[int]
    :param spatial_shape: Kernel size of the depthwise convolution, either an
        ``int`` (broadcast to both spatial axes) or a 2-tuple. Every entry must
        be a positive odd integer — an even kernel under ``padding='same'``
        would shift the feature map. Defaults to ``(7, 7)``.
    :type spatial_shape: Union[int, Tuple[int, int]]
    :param kernel_initializer: Initializer for the depthwise kernel. Defaults to
        ``'he_normal'``.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for the bias vector. Defaults to ``'zeros'``.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for the depthwise kernel.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param bias_regularizer: Optional regularizer for the bias vector.
    :type bias_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional keyword arguments forwarded to ``keras.layers.Layer``.

    :raises ValueError: If ``dim`` is not positive, if ``dim_out`` differs from
        ``dim``, or if ``spatial_shape`` is not a positive odd int / 2-tuple of
        positive odd ints.

    Example:
        >>> import numpy as np
        >>> layer = RepConditionalPosEnc(dim=32)
        >>> y = layer(np.zeros((2, 4, 4, 32), dtype='float32'), training=False)
        >>> y.shape
        (2, 4, 4, 32)
    """

    def __init__(
            self,
            dim: int,
            dim_out: Optional[int] = None,
            spatial_shape: Union[int, Tuple[int, int]] = (7, 7),
            kernel_initializer: Union[str, initializers.Initializer] = 'he_normal',
            bias_initializer: Union[str, initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            bias_regularizer: Optional[regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # ---- validation -------------------------------------------------
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if dim_out is not None and dim_out <= 0:
            raise ValueError(f"dim_out must be positive, got {dim_out}")

        resolved_dim_out = dim if dim_out is None else dim_out
        if resolved_dim_out != dim:
            raise ValueError(
                f"dim_out must equal dim: the reference only ever uses the "
                f"dim_out == dim form of RepConditionalPosEnc, whose grouped "
                f"convolution is exactly a channels-last DepthwiseConv2D. "
                f"Got dim={dim}, dim_out={dim_out}."
            )

        resolved_spatial_shape = self._normalize_spatial_shape(spatial_shape)

        # ---- store configuration ---------------------------------------
        self.dim = dim
        self.dim_out = resolved_dim_out
        self.spatial_shape = resolved_spatial_shape
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        # ---- CREATE all sub-layers (unbuilt) ----------------------------
        self.pos_conv = layers.DepthwiseConv2D(
            kernel_size=self.spatial_shape,
            strides=1,
            padding='same',
            use_bias=True,
            depthwise_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            depthwise_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='pos_conv'
        )

    @staticmethod
    def _normalize_spatial_shape(
            spatial_shape: Union[int, Tuple[int, int]]
    ) -> Tuple[int, int]:
        """Normalize ``spatial_shape`` to a validated 2-tuple of odd positive ints.

        :param spatial_shape: An ``int`` or a 2-element sequence.
        :type spatial_shape: Union[int, Tuple[int, int]]
        :return: A 2-tuple ``(kh, kw)``.
        :rtype: Tuple[int, int]
        :raises ValueError: If the value is not a positive odd int / 2-tuple of
            positive odd ints.
        """
        if isinstance(spatial_shape, int):
            normalized = (spatial_shape, spatial_shape)
        else:
            try:
                normalized = tuple(spatial_shape)
            except TypeError:
                raise ValueError(
                    f"spatial_shape must be an int or a 2-tuple of ints, "
                    f"got {spatial_shape!r}"
                )
            if len(normalized) != 2:
                raise ValueError(
                    f"spatial_shape must have exactly 2 entries, "
                    f"got {spatial_shape!r}"
                )

        for value in normalized:
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(
                    f"spatial_shape entries must be ints, got {value!r}"
                )
            if value <= 0:
                raise ValueError(
                    f"spatial_shape entries must be positive, got {value}"
                )
            if value % 2 == 0:
                raise ValueError(
                    f"spatial_shape entries must be odd (an even kernel with "
                    f"padding='same' shifts the feature map), got {value}"
                )
        return normalized

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Explicitly build the depthwise convolution, then the layer itself.

        :param input_shape: Shape of the input tensor, ``(B, H, W, dim)``.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the input is not rank 4 or its channel count is
            not ``dim``.
        """
        if len(input_shape) != 4:
            raise ValueError(
                f"RepConditionalPosEnc expects a rank-4 (B, H, W, C) input, "
                f"got shape {input_shape}"
            )
        if input_shape[-1] is not None and input_shape[-1] != self.dim:
            raise ValueError(
                f"Input channel count must equal dim={self.dim}, "
                f"got {input_shape[-1]}"
            )

        self.pos_conv.build(input_shape)

        super().build(input_shape)

    def call(self, inputs, training: Optional[bool] = None):
        """Add the conditionally generated positional encoding to the input.

        :param inputs: Input tensor of shape ``(B, H, W, dim)``.
        :param training: Keras training flag (unused; the layer is deterministic).
        :type training: Optional[bool]
        :return: Output tensor of shape ``(B, H, W, dim)``.
        """
        return ops.add(self.pos_conv(inputs), inputs)

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape from stored config alone (works pre-build).

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple, identical to the input shape.
        :rtype: Tuple[Optional[int], ...]
        """
        input_shape = tuple(input_shape)
        return input_shape[:-1] + (self.dim_out,)

    def get_config(self) -> Dict[str, Any]:
        """Return the full layer configuration for serialization.

        :return: Dictionary containing every constructor parameter.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'dim_out': self.dim_out,
            'spatial_shape': self.spatial_shape,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RepConditionalPosEnc":
        """Rebuild the layer from a serialized configuration.

        :param config: Configuration dictionary produced by :meth:`get_config`.
        :type config: Dict[str, Any]
        :return: A new :class:`RepConditionalPosEnc` instance.
        :rtype: RepConditionalPosEnc
        """
        config = dict(config)
        if config.get('spatial_shape') is not None:
            config['spatial_shape'] = tuple(config['spatial_shape'])
        config['kernel_initializer'] = initializers.deserialize(
            config['kernel_initializer'])
        config['bias_initializer'] = initializers.deserialize(
            config['bias_initializer'])
        config['kernel_regularizer'] = regularizers.deserialize(
            config['kernel_regularizer'])
        config['bias_regularizer'] = regularizers.deserialize(
            config['bias_regularizer'])
        return cls(**config)

# ---------------------------------------------------------------------
