"""
RepMixer token mixer and RepMixer block of the FastViT / MobileCLIP2 MCi backbone.

This module transcribes timm's ``RepMixer`` and ``RepMixerBlock`` as used by the
FastViT MCi image tower.

The design idea behind RepMixer is that a residual token mixer of the usual form

.. code-block:: text

    y = x + LayerScale * TokenMixer(Norm(x))

can be re-arranged so that, at inference time, the *whole* branch collapses into a
single depthwise convolution. FastViT achieves this by making both the "norm" and
the "mixer" the same kind of object — a reparameterizable ``MobileOneBlock`` — and
subtracting one from the other:

.. code-block:: text

    y = x + LayerScale * (Mixer(x) - Norm(x))

The subtraction is what makes the identity recoverable: ``Norm`` is deliberately
degenerate. It is a ``MobileOneBlock`` configured with **zero** ``k x k`` conv
branches and **no** 1x1 scale branch, at ``stride=1`` with matching channel counts,
so the only branch that survives is the identity BatchNormalization. ``Mixer`` is
the same block *with* its depthwise ``k x k`` branch. Both are affine at inference,
so ``x + gamma * (Mixer(x) - Norm(x))`` is itself affine and fuses away entirely.

This port implements the **train-time** multi-branch form only; no structural
reparameterization (`reparameterize()` / branch fusion) is provided, matching the
reference weights shipped by MobileCLIP2 (which are always evaluated with
``inference_mode=False``).

Two details are load-bearing and are easy to get silently wrong:

1. **LayerScale must be allowed to go negative.** ``LearnableMultiplier`` defaults
   to ``constraint='non_neg'``, which would clamp a legitimately-negative LayerScale
   gamma to zero and silently halve the parameterization. ``constraint=None`` is
   passed explicitly (MEASURED).
2. **The ``norm`` block really must have zero conv branches.** Giving it even one
   ``k x k`` branch turns the subtraction into a difference of two conv branches,
   which trains — badly — while every shape assertion still passes.

References:
    - Vasu et al., 2023. FastViT: A Fast Hybrid Vision Transformer using
      Structural Reparameterization. (https://arxiv.org/abs/2303.14189)
    - Vasu et al., 2022. MobileOne: An Improved One millisecond Mobile Backbone.
      (https://arxiv.org/abs/2206.04040)
    - Touvron et al., 2021. Going Deeper with Image Transformers (LayerScale).
      (https://arxiv.org/abs/2103.17239)
    - Huang et al., 2016. Deep Networks with Stochastic Depth.
      (https://arxiv.org/abs/1603.09382)
"""

import keras
from keras import ops, initializers, regularizers, activations
from typing import Optional, Union, Tuple, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .conv_mlp import FastVitConvMlp
from ..layer_scale import LearnableMultiplier
from ..mobile_one_block import MobileOneBlock
from ..stochastic_depth import StochasticDepth

# ---------------------------------------------------------------------

#: Default LayerScale initialization value used by the reference for every
#: FastViT residual branch.
_REFERENCE_LAYER_SCALE_INIT = 1e-5


def _create_layer_scale(
        layer_scale_init_value: Optional[float],
        name: str,
) -> Optional[LearnableMultiplier]:
    """Build a FastViT ``LayerScale2d`` equivalent, or ``None`` when disabled.

    Shared by :class:`FastVitRepMixer` and :class:`FastVitRepMixerBlock` (and
    intended for the remaining FastViT blocks) so that the ``constraint=None``
    requirement is stated in exactly one place.

    :param layer_scale_init_value: Constant value the per-channel gamma is
        initialized to, or ``None`` to disable LayerScale entirely (the branch is
        then admitted unscaled, which is the reference's ``layer_scale_init_value
        is None`` path).
    :type layer_scale_init_value: Optional[float]
    :param name: Sub-layer name.
    :type name: str
    :return: A per-channel :class:`LearnableMultiplier` with ``constraint=None``,
        or ``None`` when ``layer_scale_init_value is None``.
    :rtype: Optional[LearnableMultiplier]

    .. warning::
       ``constraint=None`` is REQUIRED, not cosmetic. ``LearnableMultiplier``
       defaults to ``constraint='non_neg'``, which clamps gamma at zero; a
       LayerScale gamma must be free to take a negative value.
    """
    if layer_scale_init_value is None:
        return None
    return LearnableMultiplier(
        multiplier_type='CHANNEL',
        initializer=keras.initializers.Constant(layer_scale_init_value),
        constraint=None,
        name=name,
    )


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class FastVitRepMixer(keras.layers.Layer):
    """FastViT RepMixer token mixer: ``x + gamma * (mixer(x) - norm(x))``.

    Channels-last transcription of timm's ``RepMixer``. Both ``norm`` and
    ``mixer`` are :class:`MobileOneBlock` instances at ``stride=1`` with
    ``out_channels == dim``; ``norm`` is the degenerate single-BatchNormalization
    form (no conv branches, no scale branch), ``mixer`` additionally carries one
    depthwise ``k x k`` Conv-BN branch. Neither applies an activation.

    **Architecture**

    .. code-block:: text

        ┌──────────────────────────────────────────────┐
        │            Input [B, H, W, dim]              │
        └───────┬──────────────────────────┬───────────┘
                │                          │
                │              ┌───────────┴───────────┐
                │              │                       │
                │              ▼                       ▼
                │   ┌──────────────────────┐ ┌──────────────────────┐
                │   │ mixer: MobileOneBlock│ │ norm: MobileOneBlock │
                │   │  depthwise k×k + BN  │ │  identity BN ONLY    │
                │   │  + 1×1 scale + id BN │ │  (0 conv branches,   │
                │   │  use_act=False       │ │   no scale branch)   │
                │   └───────────┬──────────┘ └──────────┬───────────┘
                │               │                       │
                │               └────────── − ──────────┘
                │                           │
                │                           ▼
                │              ┌──────────────────────────┐
                │              │ LayerScale (per-channel) │
                │              │  gamma, constraint=None  │
                │              └────────────┬─────────────┘
                │                           │
                └────────────── + ──────────┘
                                │
                                ▼
        ┌──────────────────────────────────────────────┐
        │            Output [B, H, W, dim]             │
        └──────────────────────────────────────────────┘

    .. note::
       With ``layer_scale_init_value=0.0`` the layer is the EXACT identity at
       ``training=False``, and with a nonzero value it is not. Both halves matter:
       an identity-only assertion is also satisfied by a completely dead branch.

    :param dim: Number of channels. The layer preserves it. Must be positive.
    :type dim: int
    :param kernel_size: Spatial size of the mixer's depthwise convolution. Must be
        a positive odd integer (an even kernel under ``padding='same'`` would shift
        the feature map). Defaults to 3.
    :type kernel_size: int
    :param layer_scale_init_value: Constant initialization for the per-channel
        LayerScale gamma, or ``None`` to omit LayerScale altogether. Defaults to
        ``1e-5``.
    :type layer_scale_init_value: Optional[float]
    :param kernel_initializer: Initializer for the MobileOne convolution kernels.
        Defaults to ``'he_normal'``.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for the MobileOne convolution
        kernels.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional keyword arguments forwarded to ``keras.layers.Layer``.

    :raises ValueError: If ``dim`` or ``kernel_size`` are not positive, if
        ``kernel_size`` is even, or if ``layer_scale_init_value`` is not a real
        number or ``None``.

    Example:
        >>> import numpy as np
        >>> layer = FastVitRepMixer(dim=32)
        >>> y = layer(np.zeros((2, 8, 8, 32), dtype='float32'), training=False)
        >>> y.shape
        (2, 8, 8, 32)
    """

    def __init__(
            self,
            dim: int,
            kernel_size: int = 3,
            layer_scale_init_value: Optional[float] = _REFERENCE_LAYER_SCALE_INIT,
            kernel_initializer: Union[str, initializers.Initializer] = 'he_normal',
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # ---- validation -------------------------------------------------
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")
        if kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be odd (an even kernel with padding='same' "
                f"shifts the feature map), got {kernel_size}"
            )
        if layer_scale_init_value is not None:
            if isinstance(layer_scale_init_value, bool) or not isinstance(
                    layer_scale_init_value, (int, float)):
                raise ValueError(
                    f"layer_scale_init_value must be a real number or None, "
                    f"got {layer_scale_init_value!r}"
                )

        # ---- store configuration ---------------------------------------
        self.dim = dim
        self.kernel_size = kernel_size
        self.layer_scale_init_value = (
            None if layer_scale_init_value is None else float(layer_scale_init_value)
        )
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

        # ---- CREATE all sub-layers (unbuilt) ----------------------------
        # `norm` is the DEGENERATE MobileOneBlock: num_conv_branches=0 and
        # use_scale_branch=False leave only the identity BatchNormalization alive
        # (stride=1 and out_channels == in_channels guarantee it exists). Giving it
        # a conv branch would silently change the architecture while every shape
        # assertion still passes.
        self.norm = MobileOneBlock(
            out_channels=self.dim,
            kernel_size=self.kernel_size,
            stride=1,
            group_size=1,
            use_act=False,
            use_scale_branch=False,
            num_conv_branches=0,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='norm',
        )
        self.mixer = MobileOneBlock(
            out_channels=self.dim,
            kernel_size=self.kernel_size,
            stride=1,
            group_size=1,
            use_act=False,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='mixer',
        )
        self.layer_scale = _create_layer_scale(
            self.layer_scale_init_value, name='layer_scale')

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Explicitly build every sub-layer, then the layer itself.

        :param input_shape: Shape of the input tensor, ``(B, H, W, dim)``.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the input is not rank 4 or its channel count is not
            ``dim``.
        """
        if len(input_shape) != 4:
            raise ValueError(
                f"FastVitRepMixer expects a rank-4 (B, H, W, C) input, "
                f"got shape {input_shape}"
            )
        if input_shape[-1] is not None and input_shape[-1] != self.dim:
            raise ValueError(
                f"Input channel count must equal dim={self.dim}, "
                f"got {input_shape[-1]}"
            )

        self.norm.build(input_shape)
        self.mixer.build(input_shape)
        if self.layer_scale is not None:
            self.layer_scale.build(input_shape)

        super().build(input_shape)

    def call(self, inputs, training: Optional[bool] = None):
        """Apply the RepMixer token mixer.

        :param inputs: Input tensor of shape ``(B, H, W, dim)``.
        :param training: Keras training flag. Pass ``False`` explicitly for
            deterministic behaviour (the BatchNormalizations inside the two
            MobileOne blocks update their moving statistics otherwise).
        :type training: Optional[bool]
        :return: Output tensor of shape ``(B, H, W, dim)``.
        """
        residual = ops.subtract(
            self.mixer(inputs, training=training),
            self.norm(inputs, training=training),
        )
        if self.layer_scale is not None:
            residual = self.layer_scale(residual, training=training)
        return ops.add(inputs, residual)

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
        return input_shape[:-1] + (self.dim,)

    def get_config(self) -> Dict[str, Any]:
        """Return the full layer configuration for serialization.

        :return: Dictionary containing every constructor parameter.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'kernel_size': self.kernel_size,
            'layer_scale_init_value': self.layer_scale_init_value,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FastVitRepMixer":
        """Rebuild the layer from a serialized configuration.

        :param config: Configuration dictionary produced by :meth:`get_config`.
        :type config: Dict[str, Any]
        :return: A new :class:`FastVitRepMixer` instance.
        :rtype: FastVitRepMixer
        """
        config = dict(config)
        config['kernel_initializer'] = initializers.deserialize(
            config['kernel_initializer'])
        config['kernel_regularizer'] = regularizers.deserialize(
            config['kernel_regularizer'])
        return cls(**config)


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class FastVitRepMixerBlock(keras.layers.Layer):
    """FastViT RepMixer block: a RepMixer token mixer followed by a residual ConvMlp.

    Channels-last transcription of timm's ``RepMixerBlock``. The token-mixing half
    carries its residual connection internally (inside :class:`FastVitRepMixer`);
    the channel-mixing half is an explicit residual around
    :class:`FastVitConvMlp`, gated by its own LayerScale and guarded by
    stochastic depth.

    **Architecture**

    .. code-block:: text

        ┌──────────────────────────────────────────────┐
        │            Input [B, H, W, dim]              │
        └───────────────────────┬──────────────────────┘
                                ▼
        ┌──────────────────────────────────────────────┐
        │ token_mixer: FastVitRepMixer                 │
        │   x + gamma1 * (mixer(x) - norm(x))          │
        └───────┬──────────────────────────┬───────────┘
                │                          ▼
                │            ┌──────────────────────────────┐
                │            │ mlp: FastVitConvMlp          │
                │            │  dw 7×7 + BN → 1×1 → act →   │
                │            │  drop → 1×1 → drop           │
                │            └──────────────┬───────────────┘
                │                           ▼
                │            ┌──────────────────────────────┐
                │            │ LayerScale (per-channel)     │
                │            │  gamma2, constraint=None     │
                │            └──────────────┬───────────────┘
                │                           ▼
                │            ┌──────────────────────────────┐
                │            │ StochasticDepth (drop path)  │
                │            └──────────────┬───────────────┘
                │                           │
                └────────────── + ──────────┘
                                │
                                ▼
        ┌──────────────────────────────────────────────┐
        │            Output [B, H, W, dim]             │
        └──────────────────────────────────────────────┘

    .. note::
       ``StochasticDepth`` short-circuits to the identity only when ``training is
       False`` (or the rate is exactly 0.0); ``training=None`` runs the stochastic
       path. Deterministic tests must pass ``training=False`` EXPLICITLY.

    :param dim: Number of channels. The block preserves it. Must be positive.
    :type dim: int
    :param kernel_size: Spatial size of the token mixer's depthwise convolution.
        Must be a positive odd integer. Defaults to 3.
    :type kernel_size: int
    :param mlp_ratio: Expansion ratio of the ConvMlp bottleneck; the hidden width
        is ``int(dim * mlp_ratio)``. Must be positive and yield a positive hidden
        width. Defaults to 4.0.
    :type mlp_ratio: float
    :param dropout_rate: Dropout rate inside the ConvMlp. Must be in ``[0, 1)``.
        Defaults to 0.0.
    :type dropout_rate: float
    :param drop_path_rate: Per-sample stochastic-depth rate applied to the ConvMlp
        residual branch. Must be in ``[0, 1)`` — the value is validated by
        :class:`StochasticDepth`, which raises for ``>= 1.0``. Defaults to 0.0.
    :type drop_path_rate: float
    :param layer_scale_init_value: Constant initialization for BOTH LayerScale
        gammas (the token mixer's and the block's), or ``None`` to omit LayerScale.
        Defaults to ``1e-5``.
    :type layer_scale_init_value: Optional[float]
    :param activation: Activation used inside the ConvMlp. Defaults to ``'gelu'``.
    :type activation: Union[str, callable]
    :param kernel_initializer: Initializer for the token mixer's MobileOne
        convolution kernels. Defaults to ``'he_normal'``. The ConvMlp keeps its own
        reference default (``TruncatedNormal(stddev=0.02)``, per timm's
        ``_init_weights``).
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for every convolution kernel in
        the block.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional keyword arguments forwarded to ``keras.layers.Layer``.

    :raises ValueError: If ``dim`` or ``kernel_size`` are not positive, if
        ``kernel_size`` is even, if ``mlp_ratio`` is not positive or yields a
        zero-width bottleneck, if ``dropout_rate`` is outside ``[0, 1)``, if
        ``drop_path_rate`` is outside ``[0, 1)``, or if
        ``layer_scale_init_value`` is not a real number or ``None``.

    Example:
        >>> import numpy as np
        >>> block = FastVitRepMixerBlock(dim=32, mlp_ratio=3.0)
        >>> y = block(np.zeros((2, 8, 8, 32), dtype='float32'), training=False)
        >>> y.shape
        (2, 8, 8, 32)
    """

    def __init__(
            self,
            dim: int,
            kernel_size: int = 3,
            mlp_ratio: float = 4.0,
            dropout_rate: float = 0.0,
            drop_path_rate: float = 0.0,
            layer_scale_init_value: Optional[float] = _REFERENCE_LAYER_SCALE_INIT,
            activation: Union[str, callable] = 'gelu',
            kernel_initializer: Union[str, initializers.Initializer] = 'he_normal',
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # ---- validation -------------------------------------------------
        # `dim` / `kernel_size` / `layer_scale_init_value` are re-validated by
        # FastVitRepMixer below, but validating here keeps the raise site on the
        # block the caller actually constructed.
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")
        hidden_dim = int(dim * mlp_ratio)
        if hidden_dim <= 0:
            raise ValueError(
                f"mlp_ratio={mlp_ratio} with dim={dim} yields a zero-width "
                f"bottleneck (int(dim * mlp_ratio) == {hidden_dim})"
            )
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}")
        if layer_scale_init_value is not None:
            if isinstance(layer_scale_init_value, bool) or not isinstance(
                    layer_scale_init_value, (int, float)):
                raise ValueError(
                    f"layer_scale_init_value must be a real number or None, "
                    f"got {layer_scale_init_value!r}"
                )

        # ---- store configuration ---------------------------------------
        self.dim = dim
        self.kernel_size = kernel_size
        self.mlp_ratio = float(mlp_ratio)
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.drop_path_rate = drop_path_rate
        self.layer_scale_init_value = (
            None if layer_scale_init_value is None else float(layer_scale_init_value)
        )
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

        # ---- CREATE all sub-layers (unbuilt) ----------------------------
        self.token_mixer = FastVitRepMixer(
            dim=self.dim,
            kernel_size=self.kernel_size,
            layer_scale_init_value=self.layer_scale_init_value,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='token_mixer',
        )
        self.mlp = FastVitConvMlp(
            dim=self.dim,
            hidden_dim=self.hidden_dim,
            activation=self.activation,
            dropout_rate=self.dropout_rate,
            kernel_regularizer=self.kernel_regularizer,
            name='mlp',
        )
        self.layer_scale = _create_layer_scale(
            self.layer_scale_init_value, name='layer_scale')
        # Created UNCONDITIONALLY: StochasticDepth accepts a rate of exactly 0.0
        # and short-circuits to the identity for it, so there is no reason to gate
        # construction on the rate. `drop_path_rate >= 1.0` raises from here.
        self.drop_path = StochasticDepth(
            drop_path_rate=self.drop_path_rate, name='drop_path')

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Explicitly build every sub-layer, then the layer itself.

        :param input_shape: Shape of the input tensor, ``(B, H, W, dim)``.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the input is not rank 4 or its channel count is not
            ``dim``.
        """
        if len(input_shape) != 4:
            raise ValueError(
                f"FastVitRepMixerBlock expects a rank-4 (B, H, W, C) input, "
                f"got shape {input_shape}"
            )
        if input_shape[-1] is not None and input_shape[-1] != self.dim:
            raise ValueError(
                f"Input channel count must equal dim={self.dim}, "
                f"got {input_shape[-1]}"
            )

        self.token_mixer.build(input_shape)
        mixed_shape = self.token_mixer.compute_output_shape(input_shape)

        self.mlp.build(mixed_shape)
        mlp_shape = self.mlp.compute_output_shape(mixed_shape)

        if self.layer_scale is not None:
            self.layer_scale.build(mlp_shape)
        self.drop_path.build(mlp_shape)

        super().build(input_shape)

    def call(self, inputs, training: Optional[bool] = None):
        """Apply the RepMixer block.

        :param inputs: Input tensor of shape ``(B, H, W, dim)``.
        :param training: Keras training flag. Pass ``False`` explicitly for
            deterministic behaviour — ``StochasticDepth`` treats ``None`` as
            training.
        :type training: Optional[bool]
        :return: Output tensor of shape ``(B, H, W, dim)``.
        """
        x = self.token_mixer(inputs, training=training)

        residual = self.mlp(x, training=training)
        if self.layer_scale is not None:
            residual = self.layer_scale(residual, training=training)
        residual = self.drop_path(residual, training=training)

        return ops.add(x, residual)

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
        return input_shape[:-1] + (self.dim,)

    def get_config(self) -> Dict[str, Any]:
        """Return the full layer configuration for serialization.

        :return: Dictionary containing every constructor parameter.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'kernel_size': self.kernel_size,
            'mlp_ratio': self.mlp_ratio,
            'dropout_rate': self.dropout_rate,
            'drop_path_rate': self.drop_path_rate,
            'layer_scale_init_value': self.layer_scale_init_value,
            'activation': activations.serialize(self.activation),
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FastVitRepMixerBlock":
        """Rebuild the layer from a serialized configuration.

        :param config: Configuration dictionary produced by :meth:`get_config`.
        :type config: Dict[str, Any]
        :return: A new :class:`FastVitRepMixerBlock` instance.
        :rtype: FastVitRepMixerBlock
        """
        config = dict(config)
        config['activation'] = activations.deserialize(config['activation'])
        config['kernel_initializer'] = initializers.deserialize(
            config['kernel_initializer'])
        config['kernel_regularizer'] = regularizers.deserialize(
            config['kernel_regularizer'])
        return cls(**config)

# ---------------------------------------------------------------------
