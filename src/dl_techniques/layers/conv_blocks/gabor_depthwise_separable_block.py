"""
Gabor depthwise-separable convolution block.

A depthwise-separable convolution decomposes a standard convolution into
spatial filtering per channel (depthwise) followed by channel mixing (a 1x1
pointwise convolution). This block fixes the spatial half: the depthwise
kernel is a *frozen* bank of 2D Gabor filters, and only the pointwise
projection learns.

The composition is not new to this library. It is hand-written, twice and
independently, inside two shipped model builders --
``models/vision/convunext/model.py`` and
``models/vision/bias_free_denoisers/bfunet.py`` -- as a frozen Gabor
``DepthwiseConv2D`` followed by a mandatory bias-free 1x1 projection. This
module packages that convention as one registered layer; it does not
reimplement any Gabor math, which lives in
:class:`~dl_techniques.initializers.gabor_filters_initializer.GaborFiltersInitializer`
and is composed here through
:func:`~dl_techniques.initializers.gabor_filters_initializer.create_gabor_depthwise_conv2d`.

Why freeze the bank
-------------------
A Gabor bank is a deterministic, closed-form, orientation- and
frequency-selective front-end. Freezing it makes the block cheap (one
learnable weight tensor, the 1x1 kernel), removes the gradient path through a
synthesized kernel entirely, and -- with the shipped defaults, which are
bias-free and carry no normalization -- preserves positive homogeneity,
``D(a * x) == a * D(x)`` for ``a > 0``. That property is what the two existing
consumers, both bias-free denoisers, depend on.

References:
    - Ozbulak, G., & Ekenel, H. K. *Initialization of Convolutional Neural
      Networks by Gabor Filters*. 26th Signal Processing and Communications
      Applications Conference (SIU), 2018.
    - Howard, A. G., et al. (2017). "MobileNets: Efficient Convolutional
      Neural Networks for Mobile Vision Applications."
    - Chollet, F. (2017). "Xception: Deep Learning with Depthwise Separable
      Convolutions."
"""

import keras
from typing import Any, Dict, FrozenSet, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from ..norms import create_normalization_layer
from ..activations import resolve_activation_layer
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.initializers.gabor_filters_initializer import (
    RangeLike,
    create_gabor_depthwise_conv2d,
)

# ---------------------------------------------------------------------
# module constants
# ---------------------------------------------------------------------

# DECISION plan-2026-09-05T115518-e69163e4/D-007: keep this as a private COPY.
# Do NOT replace it with an import of
# ``models.vision.convunext.model.POSITIVELY_HOMOGENEOUS_ACTIVATIONS`` -- a
# layer in ``layers/`` must never depend upward on ``models/``. The contents are
# copied verbatim from ``src/dl_techniques/models/vision/convunext/model.py``
# lines 502-504, whose own anchor
# ``# DECISION plan-2026-08-14T092357-0e3d792d/D-012`` states that this is a
# narrow, INCOMPLETE allowlist and not a denylist: membership is not a
# homogeneity certificate, it only suppresses the warning below. See
# decisions.md D-007.
_POSITIVELY_HOMOGENEOUS_ACTIVATIONS: FrozenSet[Optional[str]] = frozenset(
    {None, 'linear', 'relu', 'leaky_relu'}
)


# ---------------------------------------------------------------------
# module-private helpers
# ---------------------------------------------------------------------

def _conv_output_length(
        input_length: Optional[int],
        kernel_size: int,
        stride: int,
        padding: str
) -> Optional[int]:
    """Compute one spatial output dimension of a convolution.

    This is the single home of the block's spatial arithmetic; both
    :meth:`GaborDepthwiseSeparableBlock.compute_output_shape` and the
    ``build``-time shape threading go through it, so the two can never drift.

    :param input_length: Input extent along one spatial axis, or ``None`` when
        the dimension is dynamic.
    :type input_length: Optional[int]
    :param kernel_size: Kernel extent along the same axis. Must be positive.
    :type kernel_size: int
    :param stride: Stride along the same axis. Must be positive.
    :type stride: int
    :param padding: Padding mode, ``'same'`` or ``'valid'``.
    :type padding: str
    :return: Output extent along that axis, or ``None`` if ``input_length``
        is ``None``.
    :rtype: Optional[int]
    """
    if input_length is None:
        return None

    if padding == 'same':
        # ceil(input_length / stride)
        return (input_length + stride - 1) // stride

    # 'valid': ceil((input_length - kernel_size + 1) / stride)
    return (input_length - kernel_size + stride) // stride


# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.conv_blocks.gabor_depthwise_separable_block")
class GaborDepthwiseSeparableBlock(keras.layers.Layer):
    """
    Frozen Gabor depthwise convolution followed by a learnable pointwise 1x1.

    The depthwise stage applies ``filters_per_channel`` fixed Gabor filters to
    every input channel independently, producing
    ``channels * filters_per_channel`` responses; the pointwise stage projects
    those to ``filters`` channels. Two stages between them -- normalization and
    activation -- are optional and are **absent by default**: when their knob is
    ``None`` no sub-layer is constructed at all, so the ON and OFF arms
    deliberately have different weight layouts.

    Architecture:

    .. code-block:: text

        ┌─────────────────────────────────────────────────────┐
        │  Input [batch, height, width, channels]             │
        └───────────────────────┬─────────────────────────────┘
                                ▼
        ┌─────────────────────────────────────────────────────┐
        │  DepthwiseConv2D (K x K, strides, padding)          │
        │  depth_multiplier = filters_per_channel             │
        │  GaborFiltersInitializer, use_bias=False            │
        │  trainable=False  ── frozen Gabor bank              │
        └───────────────────────┬─────────────────────────────┘
                                ▼ [batch, h', w', channels * filters_per_channel]
        ┌─────────────────────────────────────────────────────┐
        │  Normalization      (OPTIONAL, absent by default)   │
        └───────────────────────┬─────────────────────────────┘
                                ▼
        ┌─────────────────────────────────────────────────────┐
        │  Activation         (OPTIONAL, absent by default)   │
        └───────────────────────┬─────────────────────────────┘
                                ▼
        ┌─────────────────────────────────────────────────────┐
        │  Conv2D (1 x 1) ── pointwise projection, learnable  │
        │  use_bias=False by default                         │
        └───────────────────────┬─────────────────────────────┘
                                ▼
        ┌─────────────────────────────────────────────────────┐
        │  Output [batch, h', w', filters]                    │
        └─────────────────────────────────────────────────────┘

    Relation to the existing consumers
    ----------------------------------
    The default configuration -- no normalization, no activation,
    ``pointwise_use_bias=False`` -- is exactly the Gabor-stem convention already
    used by ``models/vision/convunext/model.py`` and
    ``models/vision/bias_free_denoisers/bfunet.py``. At those defaults the block
    is bias-free end to end and contains no normalization, so positive
    homogeneity ``D(a * x) == a * D(x)`` for ``a > 0`` is preserved. Neither
    consumer is migrated onto this class; both keep their inline stems.

    Positive homogeneity and ``activation``
    ---------------------------------------
    Any activation is accepted, but one that is not positively homogeneous
    (``'gelu'``, ``'tanh'``, ``'sigmoid'``, ...) destroys the property above and
    triggers a ``logger.warning``. The check is a narrow, incomplete
    **allowlist** (``None``, ``'linear'``, ``'relu'``, ``'leaky_relu'``), so
    passing it is not a homogeneity certificate -- it only means nobody has
    flagged that name. Enabling ``normalization_type`` or
    ``pointwise_use_bias`` also breaks homogeneity and is not warned about.

    Caveat -- ``activation_kwargs`` can be silently dropped
    ------------------------------------------------------
    The activation is built by
    :func:`~dl_techniques.layers.activations.factory.resolve_activation_layer`,
    which routes a key of ``ACTIVATION_REGISTRY`` to the activation factory and
    hands anything else to ``keras.layers.Activation``. On that fallback path
    ``activation_kwargs`` are **silently dropped**. Of the four allowlisted
    names only ``'relu'`` is a registry key; ``'linear'`` and ``'leaky_relu'``
    take the fallback path, so kwargs passed alongside them have no effect.

    Caveat -- the freeze does not survive a ``trainable`` toggle
    -----------------------------------------------------------
    The Gabor bank is frozen at construction (``trainable=False``) and its
    kernel is therefore absent from ``trainable_weights``. This freeze is
    **not** robust to a ``parent.trainable = False; parent.trainable = True``
    cycle on any enclosing layer or model: Keras 3's ``Layer.trainable`` setter
    recurses into sub-layers, so the second assignment unfreezes the bank and
    puts its kernel back into ``trainable_weights``. (Re-assigning the *same*
    value is a no-op only because TensorFlow's ``AutoTrackable.__setattr__``
    short-circuits on identity, which masks the effect in the common case.)
    This is standard Keras 3 semantics, not a defect of this block. A caller who
    toggles an enclosing model's ``trainable`` -- the canonical fine-tuning
    freeze/unfreeze cycle -- must re-assert
    ``block.gabor_depthwise.trainable = False`` afterwards.

    :param filters: Number of output channels of the pointwise projection.
        Required; the block has no no-projection mode. Must be positive.
    :type filters: int
    :param filters_per_channel: Number of Gabor filters applied to each input
        channel, i.e. the depthwise ``depth_multiplier``. Must be >= 1.
        Defaults to 4.
    :type filters_per_channel: int
    :param kernel_size: Spatial size of the Gabor window, an int or a
        ``(kh, kw)`` tuple. Defaults to 11, the Ozbulak & Ekenel first-layer
        size and the Gabor factory's own default.
    :type kernel_size: Union[int, Tuple[int, int]]
    :param strides: Stride of the depthwise stage, an int or an ``(sh, sw)``
        tuple. The pointwise stage is always stride 1. Defaults to 1.
    :type strides: Union[int, Tuple[int, int]]
    :param padding: Padding mode of the depthwise stage, ``'same'`` or
        ``'valid'``. Defaults to ``'same'``.
    :type padding: str
    :param sigma_range: ``(min, max)`` interval for the Gaussian envelope
        width, or ``None`` for the kernel-relative default.
    :type sigma_range: RangeLike
    :param theta_range: ``(min, max)`` interval for orientation, in DEGREES.
        Defaults to ``(0.0, 180.0)``.
    :type theta_range: RangeLike
    :param lambda_range: ``(min, max)`` interval for the sinusoid wavelength,
        or ``None`` for the kernel-relative default.
    :type lambda_range: RangeLike
    :param gamma_range: ``(min, max)`` interval for the spatial aspect ratio.
        Defaults to ``(0.5, 1.5)``.
    :type gamma_range: RangeLike
    :param psi_range: ``(min, max)`` interval for the phase offset, in DEGREES.
        Defaults to ``(0.0, 360.0)``.
    :type psi_range: RangeLike
    :param sweep: Bank construction strategy, ``'product'`` or ``'diagonal'``.
        Defaults to ``'product'``.
    :type sweep: str
    :param normalize: Whether each Gabor filter is DC-removed and
        energy-normalized. Defaults to ``True``.
    :type normalize: bool
    :param normalization_type: Key of the normalization factory for the
        optional stage between the two convolutions. ``None`` (the default)
        creates no sub-layer at all.
    :type normalization_type: Optional[str]
    :param normalization_kwargs: Extra arguments for the normalization factory.
    :type normalization_kwargs: Optional[Dict[str, Any]]
    :param activation: Activation applied between the two convolutions.
        ``None`` (the default) creates no sub-layer at all. See the positive
        homogeneity note above.
    :type activation: Optional[str]
    :param activation_kwargs: Extra arguments for the activation resolver.
        Silently dropped for non-registry activation names.
    :type activation_kwargs: Optional[Dict[str, Any]]
    :param pointwise_use_bias: Whether the pointwise convolution carries a
        bias. Defaults to ``False``, which keeps the block bias-free.
    :type pointwise_use_bias: bool
    :param kernel_initializer: Initializer for the pointwise kernel. Never
        applied to the depthwise stage, which always gets its own fresh Gabor
        initializer. Defaults to ``'he_normal'``.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for the pointwise kernel.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional arguments for the Layer base class.
    :raises ValueError: If ``filters`` is not positive, ``filters_per_channel``
        is below 1, ``kernel_size`` or ``strides`` is non-positive or not a
        pair, or ``padding`` is neither ``'same'`` nor ``'valid'``.
    """

    def __init__(
            self,
            filters: int,
            filters_per_channel: int = 4,
            kernel_size: Union[int, Tuple[int, int]] = 11,
            strides: Union[int, Tuple[int, int]] = 1,
            padding: str = 'same',
            sigma_range: RangeLike = None,
            theta_range: RangeLike = (0.0, 180.0),
            lambda_range: RangeLike = None,
            gamma_range: RangeLike = (0.5, 1.5),
            psi_range: RangeLike = (0.0, 360.0),
            sweep: str = "product",
            normalize: bool = True,
            normalization_type: Optional[str] = None,
            normalization_kwargs: Optional[Dict[str, Any]] = None,
            activation: Optional[str] = None,
            activation_kwargs: Optional[Dict[str, Any]] = None,
            pointwise_use_bias: bool = False,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        def _positive_pair(value: Union[int, Tuple[int, int]], arg_name: str) -> Tuple[int, int]:
            """Normalize an int-or-pair argument and reject non-positive extents."""
            if isinstance(value, int):
                pair = (value, value)
            else:
                pair = tuple(value)
                if len(pair) != 2:
                    raise ValueError(
                        f"{arg_name} must be an int or a pair of ints, got {value}"
                    )
            if any((not isinstance(v, int)) or v <= 0 for v in pair):
                raise ValueError(f"{arg_name} must be positive, got {value}")
            return pair

        # DECISION plan-2026-09-05T115518-e69163e4/D-009: `filters` is REQUIRED and
        # the pointwise stage is unconditional. Do NOT add the
        # `gabor_stem_projection=False` no-projection escape hatch the two model
        # consumers carry: it would add a second branch to `call()`, make
        # `compute_output_shape` depend on the input channel count rather than on
        # stored config, and a block with no pointwise stage is not a
        # depthwise-separable block. That mode is already served exactly by
        # calling `create_gabor_depthwise_conv2d` directly. See decisions.md D-009.
        if filters <= 0:
            raise ValueError(f"filters must be positive, got {filters}")
        if filters_per_channel < 1:
            raise ValueError(
                f"filters_per_channel must be >= 1, got {filters_per_channel}"
            )
        if padding not in ('same', 'valid'):
            raise ValueError(
                f"padding must be 'same' or 'valid', got {padding!r}"
            )

        self._kernel_size_hw = _positive_pair(kernel_size, 'kernel_size')
        self._strides_hw = _positive_pair(strides, 'strides')

        self.filters = filters
        self.filters_per_channel = filters_per_channel
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.sigma_range = sigma_range
        self.theta_range = theta_range
        self.lambda_range = lambda_range
        self.gamma_range = gamma_range
        self.psi_range = psi_range
        self.sweep = sweep
        self.normalize = normalize
        self.normalization_type = normalization_type
        self.normalization_kwargs = normalization_kwargs or {}
        self.activation = activation
        self.activation_kwargs = activation_kwargs or {}
        self.pointwise_use_bias = pointwise_use_bias
        # DECISION plan-2026-09-05T115518-e69163e4/D-010: resolve the initializer
        # HERE, not in `build()`. Do NOT move the sub-layer construction into
        # `build()` the way `convnext_v1_block.py` does in order to round-trip the
        # caller's literal string: the v2 guide's golden rule (sub-layers are
        # created unconditionally in `__init__`) is a HARD constraint and beats
        # that SOFT one. The cost is that `get_config()` emits the RESOLVED
        # initializer, so `"he_normal"` reads back as a serialized `HeNormal`
        # dict. See decisions.md D-010.
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        if activation is not None and activation not in _POSITIVELY_HOMOGENEOUS_ACTIVATIONS:
            logger.warning(
                f"GaborDepthwiseSeparableBlock activation={activation!r} is not in the "
                f"positively-homogeneous allowlist "
                f"{sorted(str(a) for a in _POSITIVELY_HOMOGENEOUS_ACTIVATIONS)}; "
                f"D(a*x) == a*D(x) will not hold. This allowlist is narrow and "
                f"incomplete, so membership would not have proven homogeneity either."
            )

        # Composing the factory guarantees a FRESH GaborFiltersInitializer instance,
        # distinct from `self.kernel_initializer`. `activation=None` here is
        # deliberate: the activation is a separate sub-layer so the optional
        # normalization can sit between the two stages.
        self.gabor_depthwise = create_gabor_depthwise_conv2d(
            filters_per_channel=filters_per_channel,
            kernel_size=kernel_size,
            activation=None,
            sigma_range=sigma_range,
            theta_range=theta_range,
            lambda_range=lambda_range,
            gamma_range=gamma_range,
            psi_range=psi_range,
            sweep=sweep,
            normalize=normalize,
            strides=strides,
            padding=padding,
            use_bias=False,
            trainable=False,
            name='gabor_depthwise'
        )

        self.gabor_norm = None
        if normalization_type is not None:
            self.gabor_norm = create_normalization_layer(
                normalization_type=normalization_type,
                name='gabor_norm',
                **self.normalization_kwargs
            )

        # DECISION plan-2026-09-05T115518-e69163e4/D-012: use
        # `resolve_activation_layer`, NOT `create_activation_layer`. The latter was
        # MEASURED to raise ValueError on 'linear' and 'leaky_relu' -- two of the
        # four names in the positive-homogeneity allowlist above -- because they are
        # not `ACTIVATION_REGISTRY` keys. Do not "simplify" this back to the
        # registry-only factory, and do not hand-roll a `keras.layers.Activation`
        # fallback: the resolver already is one. See decisions.md D-012.
        self.gabor_activation = None
        if activation is not None:
            self.gabor_activation = resolve_activation_layer(
                activation,
                name='gabor_activation',
                **self.activation_kwargs
            )

        self.pointwise_conv = keras.layers.Conv2D(
            filters=filters,
            kernel_size=1,
            strides=1,
            padding='same',
            use_bias=pointwise_use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='pointwise_conv'
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer and every sub-layer it actually runs.

        :param input_shape: Shape tuple including the batch dimension.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the input is not rank 4 or its channel
            dimension is undefined.
        """
        if len(input_shape) != 4:
            raise ValueError(
                f"Expected 4D input (batch, height, width, channels), "
                f"got shape with {len(input_shape)} dimensions: {input_shape}"
            )
        if input_shape[-1] is None:
            raise ValueError(
                f"The channel dimension of the input must be defined, "
                f"got input_shape={input_shape}"
            )

        self.gabor_depthwise.build(input_shape)
        depthwise_output_shape = self.gabor_depthwise.compute_output_shape(input_shape)

        if self.gabor_norm is not None:
            self.gabor_norm.build(depthwise_output_shape)
        if self.gabor_activation is not None:
            self.gabor_activation.build(depthwise_output_shape)

        self.pointwise_conv.build(depthwise_output_shape)
        pointwise_output_shape = self.pointwise_conv.compute_output_shape(
            depthwise_output_shape
        )

        logger.debug(
            f"Built GaborDepthwiseSeparableBlock: input_shape={input_shape} -> "
            f"depthwise={depthwise_output_shape} -> output={pointwise_output_shape}, "
            f"kernel_size={self.kernel_size}, strides={self.strides}, "
            f"padding={self.padding}, filters_per_channel={self.filters_per_channel}, "
            f"norm={self.normalization_type}, activation={self.activation}"
        )

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Run the depthwise-separable chain.

        :param inputs: Input tensor of shape ``(batch, height, width, channels)``.
        :type inputs: keras.KerasTensor
        :param training: Boolean or ``None`` indicating training mode.
        :type training: Optional[bool]
        :return: Output tensor of shape ``(batch, new_height, new_width, filters)``.
        :rtype: keras.KerasTensor
        :raises ValueError: If the input is not rank 4.
        """
        if len(inputs.shape) != 4:
            raise ValueError(
                f"Expected 4D input (batch, height, width, channels), "
                f"got shape {inputs.shape}"
            )

        x = self.gabor_depthwise(inputs, training=training)

        if self.gabor_norm is not None:
            x = self.gabor_norm(x, training=training)
        if self.gabor_activation is not None:
            x = self.gabor_activation(x)

        return self.pointwise_conv(x, training=training)

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape from stored configuration alone.

        This works on an unbuilt layer: it reads only constructor arguments,
        never a sub-layer's weights.

        :param input_shape: Shape tuple including the batch dimension.
        :type input_shape: Tuple[Optional[int], ...]
        :return: ``(batch, new_height, new_width, filters)``.
        :rtype: Tuple[Optional[int], ...]
        :raises ValueError: If the input shape is not rank 4.
        """
        if len(input_shape) != 4:
            raise ValueError(
                f"Expected 4D input shape, got {len(input_shape)}D: {input_shape}"
            )

        batch_size, height, width, _ = input_shape
        kernel_h, kernel_w = self._kernel_size_hw
        stride_h, stride_w = self._strides_hw

        # The pointwise stage is 1x1 / stride 1 / 'same', so it leaves the
        # spatial dimensions untouched and only the depthwise stage matters.
        new_height = _conv_output_length(height, kernel_h, stride_h, self.padding)
        new_width = _conv_output_length(width, kernel_w, stride_w, self.padding)

        return (batch_size, new_height, new_width, self.filters)

    def get_config(self) -> Dict[str, Any]:
        """Return every constructor argument, for serialization.

        :return: Dictionary containing all constructor parameters.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'filters_per_channel': self.filters_per_channel,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'sigma_range': self.sigma_range,
            'theta_range': self.theta_range,
            'lambda_range': self.lambda_range,
            'gamma_range': self.gamma_range,
            'psi_range': self.psi_range,
            'sweep': self.sweep,
            'normalize': self.normalize,
            'normalization_type': self.normalization_type,
            'normalization_kwargs': self.normalization_kwargs,
            'activation': self.activation,
            'activation_kwargs': self.activation_kwargs,
            'pointwise_use_bias': self.pointwise_use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer)
            if self.kernel_regularizer else None,
        })
        return config

# ---------------------------------------------------------------------
