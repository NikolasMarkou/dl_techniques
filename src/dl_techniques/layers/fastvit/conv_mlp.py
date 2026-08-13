"""
Convolutional MLP (channel mixer) of the FastViT / MobileCLIP2 MCi backbone.

This module transcribes timm's ``ConvMlp`` as used by FastViT's ``RepMixerBlock``
and ``AttentionBlock``. It is the *channel-mixing* half of both blocks: a spatial
depthwise convolution followed by a two-layer, position-wise (1x1) MLP.

The design intent is to keep the token-mixing and channel-mixing responsibilities
separate while still injecting a small amount of local spatial context into the
channel mixer. The reference achieves this with:

1. A depthwise ``k x k`` convolution followed by BatchNormalization and **no**
   activation. Being depthwise, it mixes only within each channel, so it adds
   spatial context at negligible parameter cost and leaves channel mixing
   entirely to the two 1x1 convolutions that follow.
2. A biased 1x1 convolution expanding to ``hidden_dim`` (the inverted
   bottleneck), a GELU non-linearity, dropout, a biased 1x1 convolution
   projecting back to ``out_dim``, and dropout again.

Weight initialization is part of the reference contract, not a preference:
timm's ``_init_weights`` applies ``trunc_normal_(std=0.02)`` to every ``Conv2d``
kernel in the block and zeroes the biases. That is reproduced here as the
default ``kernel_initializer`` / ``bias_initializer``.

References:
    - Vasu et al., 2023. FastViT: A Fast Hybrid Vision Transformer using
      Structural Reparameterization. (https://arxiv.org/abs/2303.14189)
    - Vasu et al., 2024. MobileCLIP: Fast Image-Text Models through Multi-Modal
      Reinforced Training. (https://arxiv.org/abs/2311.17049)
"""

import keras
from typing import Optional, Union, Tuple, Dict, Any
from keras import layers, initializers, regularizers, activations

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..norms.factory import create_normalization_layer
from .reference import REFERENCE_NORM_EPSILON

# ---------------------------------------------------------------------

#: Single definition of the reference epsilon lives in :mod:`.reference`.
_REFERENCE_BN_EPSILON = REFERENCE_NORM_EPSILON


@keras.saving.register_keras_serializable()
class FastVitConvMlp(keras.layers.Layer):
    """FastViT convolutional MLP: depthwise ``k x k`` + BN, then a 1x1 inverted bottleneck.

    Channels-last transcription of timm's ``ConvMlp`` as instantiated by FastViT.
    The layer is purely position-wise after its leading depthwise convolution, so
    it preserves the spatial resolution of its input and only changes the channel
    count (which, for every MCi call site, it does not — see the ``out_dim``
    restriction below).

    **Architecture**

    .. code-block:: text

        ┌──────────────────────────────────────────────┐
        │           Input [B, H, W, dim]               │
        └───────────────────────┬──────────────────────┘
                                │
                                ▼
        ┌──────────────────────────────────────────────┐
        │  DepthwiseConv2D k×k, 'same', use_bias=False │
        │  BatchNormalization (eps 1e-5)  — NO act     │
        └───────────────────────┬──────────────────────┘
                                │  [B, H, W, dim]
                                ▼
        ┌──────────────────────────────────────────────┐
        │  Conv2D 1×1 → hidden_dim   (biased)          │
        │  activation (GELU)                           │
        │  Dropout                                     │
        └───────────────────────┬──────────────────────┘
                                │  [B, H, W, hidden_dim]
                                ▼
        ┌──────────────────────────────────────────────┐
        │  Conv2D 1×1 → out_dim      (biased)          │
        │  Dropout                                     │
        └───────────────────────┬──────────────────────┘
                                │
                                ▼
        ┌──────────────────────────────────────────────┐
        │          Output [B, H, W, out_dim]           │
        └──────────────────────────────────────────────┘

    .. note::
       The reference builds the first convolution as ``Conv2d(in_chs, out_chs,
       k, groups=in_chs)``, i.e. a grouped convolution that is depthwise *only*
       when ``out_chs == in_chs``. Every FastViT/MCi call site satisfies that,
       and channels-last Keras expresses it exactly as a ``DepthwiseConv2D``
       with ``depth_multiplier=1`` (kernel shape ``(k, k, C, 1)``). Rather than
       silently building a different graph for the unreachable ``out_dim != dim``
       case, this layer raises :class:`ValueError`.

    :param dim: Number of input channels. Must be positive.
    :type dim: int
    :param hidden_dim: Width of the inverted bottleneck. Defaults to ``dim``
        when ``None``. Must be positive when given.
    :type hidden_dim: Optional[int]
    :param out_dim: Number of output channels. Defaults to ``dim`` when ``None``.
        Must equal ``dim`` (see the note above).
    :type out_dim: Optional[int]
    :param kernel_size: Spatial size of the leading depthwise convolution. Must
        be a positive odd integer (an even kernel under ``padding='same'`` would
        shift the feature map). Defaults to 7.
    :type kernel_size: int
    :param activation: Activation applied after the expanding 1x1 convolution.
        Defaults to ``'gelu'``.
    :type activation: Union[str, callable]
    :param dropout_rate: Dropout rate applied after the activation and again
        after the projecting 1x1 convolution. Must be in ``[0, 1)``. Defaults to 0.0.
    :type dropout_rate: float
    :param kernel_initializer: Initializer for every convolution kernel.
        Defaults to ``TruncatedNormal(stddev=0.02)``, matching timm's
        ``_init_weights``.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for the 1x1 convolution biases.
        Defaults to ``'zeros'``.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for convolution kernels.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param bias_regularizer: Optional regularizer for bias terms.
    :type bias_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional keyword arguments forwarded to ``keras.layers.Layer``.

    :raises ValueError: If ``dim``, ``hidden_dim`` or ``kernel_size`` are not
        positive, if ``kernel_size`` is even, if ``dropout_rate`` is outside
        ``[0, 1)``, or if ``out_dim`` differs from ``dim``.

    Example:
        >>> import numpy as np
        >>> layer = FastVitConvMlp(dim=64, hidden_dim=256)
        >>> y = layer(np.zeros((2, 16, 16, 64), dtype='float32'), training=False)
        >>> y.shape
        (2, 16, 16, 64)
    """

    def __init__(
            self,
            dim: int,
            hidden_dim: Optional[int] = None,
            out_dim: Optional[int] = None,
            kernel_size: int = 7,
            activation: Union[str, callable] = 'gelu',
            dropout_rate: float = 0.0,
            kernel_initializer: Union[str, initializers.Initializer] = (
                    initializers.TruncatedNormal(stddev=0.02)
            ),
            bias_initializer: Union[str, initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            bias_regularizer: Optional[regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # ---- validation -------------------------------------------------
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if hidden_dim is not None and hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if out_dim is not None and out_dim <= 0:
            raise ValueError(f"out_dim must be positive, got {out_dim}")
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")
        if kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be odd (an even kernel with padding='same' "
                f"shifts the feature map), got {kernel_size}"
            )
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}")

        resolved_out_dim = dim if out_dim is None else out_dim
        if resolved_out_dim != dim:
            raise ValueError(
                f"out_dim must equal dim: the reference only ever uses the "
                f"out_dim == dim form of ConvMlp, whose leading grouped "
                f"convolution is exactly a channels-last DepthwiseConv2D. "
                f"Got dim={dim}, out_dim={out_dim}."
            )

        # ---- store configuration ---------------------------------------
        self.dim = dim
        self.hidden_dim = dim if hidden_dim is None else hidden_dim
        self.out_dim = resolved_out_dim
        self.kernel_size = kernel_size
        self.activation = activations.get(activation)
        self.dropout_rate = dropout_rate
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        # ---- CREATE all sub-layers (unbuilt) ----------------------------
        self.dw_conv = layers.DepthwiseConv2D(
            kernel_size=self.kernel_size,
            strides=1,
            padding='same',
            use_bias=False,
            depthwise_initializer=self.kernel_initializer,
            depthwise_regularizer=self.kernel_regularizer,
            name='dw_conv'
        )
        self.norm = create_normalization_layer(
            'batch_norm',
            epsilon=_REFERENCE_BN_EPSILON,
            name='norm'
        )
        self.fc1 = layers.Conv2D(
            filters=self.hidden_dim,
            kernel_size=1,
            strides=1,
            padding='valid',
            use_bias=True,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='fc1'
        )
        self.fc2 = layers.Conv2D(
            filters=self.out_dim,
            kernel_size=1,
            strides=1,
            padding='valid',
            use_bias=True,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='fc2'
        )
        self.dropout1 = layers.Dropout(self.dropout_rate, name='dropout1')
        self.dropout2 = layers.Dropout(self.dropout_rate, name='dropout2')

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Explicitly build every sub-layer, then the layer itself.

        :param input_shape: Shape of the input tensor, ``(B, H, W, dim)``.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the input is not rank 4 or its channel count is
            not ``dim``.
        """
        if len(input_shape) != 4:
            raise ValueError(
                f"FastVitConvMlp expects a rank-4 (B, H, W, C) input, "
                f"got shape {input_shape}"
            )
        if input_shape[-1] is not None and input_shape[-1] != self.dim:
            raise ValueError(
                f"Input channel count must equal dim={self.dim}, "
                f"got {input_shape[-1]}"
            )

        self.dw_conv.build(input_shape)
        dw_shape = self.dw_conv.compute_output_shape(input_shape)

        self.norm.build(dw_shape)

        self.fc1.build(dw_shape)
        fc1_shape = self.fc1.compute_output_shape(dw_shape)

        self.dropout1.build(fc1_shape)

        self.fc2.build(fc1_shape)
        fc2_shape = self.fc2.compute_output_shape(fc1_shape)

        self.dropout2.build(fc2_shape)

        super().build(input_shape)

    def call(self, inputs, training: Optional[bool] = None):
        """Apply the convolutional MLP.

        :param inputs: Input tensor of shape ``(B, H, W, dim)``.
        :param training: Keras training flag. Pass ``False`` explicitly for
            deterministic behaviour when ``dropout_rate > 0``.
        :type training: Optional[bool]
        :return: Output tensor of shape ``(B, H, W, out_dim)``.
        """
        x = self.dw_conv(inputs)
        x = self.norm(x, training=training)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x, training=training)
        x = self.fc2(x)
        x = self.dropout2(x, training=training)
        return x

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape from stored config alone (works pre-build).

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple, spatially identical to the input.
        :rtype: Tuple[Optional[int], ...]
        """
        input_shape = tuple(input_shape)
        return input_shape[:-1] + (self.out_dim,)

    def get_config(self) -> Dict[str, Any]:
        """Return the full layer configuration for serialization.

        :return: Dictionary containing every constructor parameter.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'hidden_dim': self.hidden_dim,
            'out_dim': self.out_dim,
            'kernel_size': self.kernel_size,
            'activation': activations.serialize(self.activation),
            'dropout_rate': self.dropout_rate,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FastVitConvMlp":
        """Rebuild the layer from a serialized configuration.

        :param config: Configuration dictionary produced by :meth:`get_config`.
        :type config: Dict[str, Any]
        :return: A new :class:`FastVitConvMlp` instance.
        :rtype: FastVitConvMlp
        """
        config = dict(config)
        config['activation'] = activations.deserialize(config['activation'])
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
