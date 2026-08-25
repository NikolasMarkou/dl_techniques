"""
Root Mean Square Normalization Layer for Deep Neural Networks

This module implements RMS (Root Mean Square) Normalization, a normalization technique
that can help stabilize training and improve gradient flow in deep neural networks.
RMS normalization is particularly effective in transformer architectures and has been
shown to provide computational benefits over LayerNorm in certain scenarios.

Mathematical Formulation:
    Given an input tensor x with shape (..., d), RMS normalization computes:

    RMS(x) = sqrt(mean(x²) + ε)
    output = (x / RMS(x)) * γ

    Where:
    - mean(x²) is computed over specified axes (typically the feature dimension)
    - ε is a small epsilon for numerical stability
    - γ is an optional learnable scaling parameter

Key Differences from LayerNorm:
    - LayerNorm: (x - μ) / σ * γ + β (centers and scales)
    - RMSNorm: x / RMS(x) * γ (only scales, no centering)

This makes RMSNorm computationally more efficient as it doesn't require computing
the mean for centering, only the RMS for scaling.

Performance Benefits:
    - Reduced computational overhead (no mean subtraction)
    - Better gradient flow in some architectures
    - Maintains similar normalization benefits to LayerNorm
    - More stable in mixed precision training

    No throughput figure is quoted here. The only quantitative claim this module can
    make from its own code is structural: one reduction over the normalization axes
    instead of LayerNorm's two.

References:
    - Zhang, B., & Sennrich, R. (2019). "Root Mean Square Layer Normalization."
      Advances in Neural Information Processing Systems, 32.
      https://arxiv.org/abs/1910.07467
"""

import keras
from typing import Optional, Any, Dict, Union, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class RMSNorm(keras.layers.Layer):
    """
    Root Mean Square Normalization layer for stabilized training in deep networks.

    This layer implements root mean square normalization by normalizing inputs by their
    RMS (Root Mean Square) value. RMS normalization can help stabilize training and
    improve gradient flow in deep networks, particularly in transformer architectures.

    The normalization is computed as:

    .. math::
        \\text{rms} = \\sqrt{\\text{mean}(\\text{input}^2) + \\varepsilon}

    .. math::
        \\text{output} = \\frac{\\text{input}}{\\text{rms}} \\times \\text{scale}

    Where scale is a learnable parameter when ``use_scale=True``.

    RMSNorm avoids LayerNorm's mean subtraction, so it performs one reduction over
    the normalization axes where LayerNorm performs two. It is particularly effective
    in transformer architectures and large language models.

    Statistics are computed in ``keras.backend.result_type(input_dtype, "float32")``
    - float32 at minimum for numerical stability under mixed precision, and float64
    when the layer's own policy is float64 - and the result is cast back to the input
    dtype before it is returned.

    ``supports_masking`` is ``True``. With the default ``axis=-1`` each position's
    output depends only on that position: perturbing one ``(sample, token)`` slot
    moves no other position by more than ``0.0`` (measured, in both ``training=False``
    and ``training=True``), so a Keras mask stays valid on the output and is passed
    through unchanged. Normalizing over a token axis instead couples positions, and
    the flag does not describe that configuration.

    **Architecture Overview:**

    .. code-block:: text

        Input: x (batch, ..., features)
                │
                ▼
        ┌───────────────────────────┐
        │  Compute mean(x²) along   │
        │  normalization axis       │
        └────────────┬──────────────┘
                     │
                     ▼
        ┌───────────────────────────┐
        │  RMS = √(mean(x²) + ε)    │
        └────────────┬──────────────┘
                     │
                     ▼
        ┌───────────────────────────┐
        │  normalized = x / RMS     │
        └────────────┬──────────────┘
                     │
                     ▼
        ┌───────────────────────────┐
        │  output = normalized × γ  │
        │  (if use_scale=True)      │
        └────────────┬──────────────┘
                     │
                     ▼
        Output: (batch, ..., features)

    :param axis: Axis or axes along which to compute RMS statistics.
        The default (-1) computes RMS over the last dimension. For multi-axis
        normalization, pass a tuple (e.g., (-2, -1) for normalizing over last
        two dimensions).

        Non-trailing axes are supported together with ``use_scale=True``: the
        ``scale`` weight is stored with one dimension per normalized axis (the
        checkpoint-visible shape, deliberately unchanged) and is reshaped for
        broadcasting at call time only when the normalized axes are not the
        trailing ones, so the ``axis=-1`` path emits no reshape op.

        One spelling is deliberately carved out: an axis tuple that is not
        strictly ascending (e.g. ``(-1, -2)``) keeps the legacy broadcast rather
        than a corrected one, because ``build()`` orders the scale's dimensions
        by the order the axes were WRITTEN. Use an ascending tuple.
    :type axis: Union[int, Tuple[int, ...]]
    :param epsilon: Small constant added to denominator for numerical stability.
        Should be positive and typically in range [1e-8, 1e-5].
    :type epsilon: float
    :param use_scale: Whether to use a learnable scaling parameter after
        normalization. When True, adds a trainable parameter that can help the
        model learn appropriate scaling.
    :type use_scale: bool
    :param scale_initializer: Initializer for the scale parameter when
        ``use_scale=True``. Common choices include "ones" (default), "zeros",
        or custom initializers.
    :type scale_initializer: Union[str, keras.initializers.Initializer]
    :param kwargs: Additional keyword arguments passed to the parent Layer class.
    """

    def __init__(
        self,
        axis: Union[int, Tuple[int, ...]] = -1,
        epsilon: float = 1e-6,
        use_scale: bool = True,
        scale_initializer: Union[str, keras.initializers.Initializer] = "ones",
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs early
        self._validate_inputs(axis, epsilon)

        # Store ALL configuration parameters - required for get_config()
        self.axis = axis
        self.epsilon = epsilon
        self.use_scale = use_scale
        self.scale_initializer = keras.initializers.get(scale_initializer)

        # Initialize weight attributes - created in build()
        self.scale = None

        # Broadcast shape used to reshape 'scale' in call(); computed in build().
        # None means "no reshape needed" - see the DECISION anchor in build().
        # Initialized here so build()'s `if self.built: return` early exit can
        # never leave the attribute undefined.
        self._scale_broadcast_shape = None

        # Enable masking support
        self.supports_masking = True

        logger.debug(f"Initialized RMSNorm with axis={axis}, epsilon={epsilon}, use_scale={use_scale}")

    def _validate_inputs(self, axis: Union[int, Tuple[int, ...]], epsilon: float) -> None:
        """
        Validate initialization parameters.

        :param axis: Normalization axis/axes to validate.
        :type axis: Union[int, Tuple[int, ...]]
        :param epsilon: Epsilon value to validate.
        :type epsilon: float
        :raises ValueError: If epsilon is not positive.
        :raises TypeError: If axis is not int or tuple of ints.
        """
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        # Validate axis type
        if isinstance(axis, (list, tuple)):
            if not all(isinstance(ax, int) for ax in axis):
                raise TypeError(f"All elements in axis must be integers, got {axis}")
        elif not isinstance(axis, int):
            raise TypeError(f"axis must be int or tuple of ints, got {type(axis)}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Create the layer's own weights.

        This is called automatically when the layer first processes input.
        Following modern Keras 3 Pattern 1: Simple Layer (No Sub-layers).

        :param input_shape: Shape tuple indicating input tensor shape.
            First dimension (batch size) may be None.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If attempting to create scale parameter with dynamic
            shape along normalization axes.
        """
        if self.built:
            return

        if self.use_scale:
            # Determine the shape for the scale parameter
            # Scale parameter shape matches the input shape along normalization axes
            if isinstance(self.axis, int):
                param_axes = [self.axis]
            else:
                param_axes = list(self.axis)

            # Convert negative axes to positive for shape computation
            param_axes = [ax % len(input_shape) if ax < 0 else ax for ax in param_axes]

            param_shape = tuple(input_shape[i] for i in param_axes)

            # Check for dynamic dimensions along normalization axes
            if any(dim is None for dim in param_shape):
                raise ValueError(
                    f"Cannot create 'scale' parameter for RMSNorm. The "
                    f"normalization axis {self.axis} corresponds to dynamic "
                    f"dimensions in input_shape {input_shape}. "
                    f"Scale parameter shape would be {param_shape}."
                )

            # Create layer's own weights using add_weight()
            self.scale = self.add_weight(
                name="scale",
                shape=param_shape,
                initializer=self.scale_initializer,
                trainable=True,
            )

            logger.debug(f"Created scale parameter with shape {param_shape}")

            # DECISION plan-2026-08-25T195813-d5a035ab/D-004
            # The built 'scale' weight shape is checkpoint-visible: every saved
            # .keras file holding an RMSNorm stores exactly `param_shape`.
            # Widening it above to a full-rank shape carrying 1s at the
            # unnormalized axes would be the cleaner algebra, but it would make
            # every existing checkpoint unloadable. So do NOT touch
            # add_weight(shape=param_shape); the broadcast is instead done at
            # CALL time, and only when it is actually needed.
            # Record: plans/plan-2026-08-25T195813-d5a035ab/decisions.md D-004.
            rank = len(input_shape)
            if param_axes == list(range(rank - len(param_axes), rank)):
                # The normalized axes are exactly the trailing axes, in ascending
                # order. `param_shape` already broadcasts against the input, so
                # this path (axis=-1, i.e. 100% of the live consumers) must emit
                # no reshape op at all.
                self._scale_broadcast_shape = None
            elif any(b <= a for a, b in zip(param_axes, param_axes[1:])):
                # Not strictly ascending (e.g. axis=(-1, -2)): build() orders the
                # scale's dimensions by the order the axes were WRITTEN, so a
                # broadcast shape derived from ascending order would silently
                # reinterpret the stored weight. Deliberate: an unsorted 'axis'
                # tuple keeps today's behaviour verbatim and is an unsupported
                # spelling. Do NOT "fix" this by sorting param_axes.
                self._scale_broadcast_shape = None
            else:
                broadcast_shape = [1] * rank
                for ax, dim in zip(param_axes, param_shape):
                    broadcast_shape[ax] = dim
                self._scale_broadcast_shape = tuple(broadcast_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply RMS normalization to inputs.

        :param inputs: Input tensor of any shape. Normalization is applied along
            the axes specified during initialization.
        :type inputs: keras.KerasTensor
        :param training: Boolean indicating whether the layer should behave in
            training mode. Not used in RMSNorm but kept for consistency with
            other normalization layers.
        :type training: Optional[bool]
        :return: RMS normalized tensor with the same shape as inputs.
        :rtype: keras.KerasTensor
        """
        # Store original dtype for casting back
        original_dtype = inputs.dtype

        # Statistics dtype: float32 at minimum (numerical stability under
        # mixed precision), but float64 when the layer really is float64 -
        # a hardcoded "float32" here silently ran the statistics in float32
        # under a float64 policy (measured: the output matched a float32
        # reference exactly and missed the float64 one by 1.5e-8).
        stat_dtype = keras.backend.result_type(original_dtype, "float32")
        inputs_fp32 = keras.ops.cast(inputs, stat_dtype)

        # Compute RMS: sqrt(mean(x²) + ε)
        mean_square = keras.ops.mean(
            keras.ops.square(inputs_fp32),
            axis=self.axis,
            keepdims=True
        )

        # Add epsilon for numerical stability and compute RMS
        rms = keras.ops.sqrt(mean_square + self.epsilon)

        # Normalize by RMS
        normalized = inputs_fp32 / rms

        # Apply learnable scale if enabled
        if self.use_scale:
            scale = self.scale
            if self._scale_broadcast_shape is not None:
                # Non-trailing normalization axes only: the stored weight shape is
                # not broadcast-compatible with the input, so give it explicit 1s
                # at the unnormalized axes here rather than in build()
                # (DECISION plan-2026-08-25T195813-d5a035ab/D-004).
                scale = keras.ops.reshape(scale, self._scale_broadcast_shape)
            normalized = normalized * scale

        # Cast back to original dtype
        return keras.ops.cast(normalized, original_dtype)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple (same as input shape for normalization layers).
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration for serialization.

        Following modern Keras 3 patterns, this method returns ALL constructor
        arguments needed to recreate this layer instance.

        :return: Dictionary containing all constructor arguments.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "axis": self.axis,
            "epsilon": self.epsilon,
            "use_scale": self.use_scale,
            "scale_initializer": keras.initializers.serialize(self.scale_initializer),
        })
        return config

# ---------------------------------------------------------------------
