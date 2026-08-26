"""
Zero-Centered Root Mean Square Normalization Layer for Deep Neural Networks

This module implements Zero-Centered RMS (Root Mean Square) Normalization, an advanced
normalization technique that combines the computational efficiency of RMSNorm with the
stabilizing zero-mean property of LayerNorm. This variant addresses the "mean shift"
problem in standard RMSNorm while maintaining computational advantages.

Mathematical Formulation:
    Given an input tensor x with shape (..., d), Zero-Centered RMS normalization computes:

    mu = mean(x) over specified axes
    x_centered = x - mu
    RMS(x_centered) = sqrt(mean(x_centered^2) + epsilon)
    output = (x_centered / RMS(x_centered)) * gamma

    Where:
    - mu is the mean computed over specified axes (centering step)
    - mean(x_centered^2) is computed over the same specified axes
    - epsilon is a small epsilon for numerical stability
    - gamma is an optional learnable scaling parameter

Key Differences from Standard Normalization:
    - LayerNorm: (x - mu) / sigma * gamma + beta (centers, scales, and shifts)
    - RMSNorm: x / RMS(x) * gamma (only scales, no centering)
    - Zero-Centered RMSNorm: (x - mu) / RMS(x - mu) * gamma (centers and scales, no shift)

This makes Zero-Centered RMSNorm arithmetically IDENTICAL to LayerNorm without the bias
term, not merely similar to it. Because mean(x_centered) is zero by construction,
mean(x_centered^2) is exactly var(x) over the same axes - the denominator LayerNorm
computes - and epsilon sits in the same place (inside the sqrt, added to the second
moment). What differs is framing and implementation, not arithmetic: this layer is
presented as an enhancement to RMSNorm that prevents mean drift. If you want the same
function and nothing else, prefer keras.layers.LayerNormalization(center=False), which
may reach fused kernels this implementation cannot.

Performance Benefits:
    - Prevents abnormal growth of layer normalization weights
    - Maintains training stability through zero-mean outputs
    - Combines efficiency with stabilization
    - Better gradient flow compared to standard RMSNorm
    - Particularly effective in large language models like Qwen3-Next

References:
    - Used in Qwen3-Next model for solving abnormal growth issues in layer normalization weights
    - Builds upon concepts from both LayerNorm and RMSNorm literature
"""


import keras
from keras import ops, initializers
from typing import Optional, Union, Tuple, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.norms._masking import (
    normalizes_only_the_feature_axis,
)

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ZeroCenteredRMSNorm(keras.layers.Layer):
    """
    Zero-Centered Root Mean Square Normalization layer for enhanced training stability.

    This layer implements zero-centered root mean square normalization by first centering
    the inputs around zero, then normalizing by their RMS value. This approach combines
    the computational efficiency of RMSNorm with the stabilizing zero-mean property of
    LayerNorm, preventing mean drift and abnormal weight growth.

    The normalization is computed as:

    1. Centering: mu = E[x], x_centered = x - mu
    2. RMS Computation: rms = sqrt(E[x_centered^2] + epsilon)
    3. Normalization: x_hat = x_centered / rms
    4. Scaling: y = gamma * x_hat (if use_scale=True)

    Where mu is computed per feature across normalization axes, gamma (scale) is a
    learnable parameter if use_scale=True, and epsilon is a small constant for
    numerical stability.

    This layer is particularly beneficial for transformer architectures and large
    language models, preventing abnormal growth of layer normalization weights while
    maintaining computational efficiency.

    .. note::
        This is not merely *similar* to LayerNorm without a bias - it is the same
        function. Centering forces ``mean(x_centered) = 0``, so
        ``mean(x_centered**2)`` is exactly ``var(x)`` over the same axes, and the
        epsilon is placed identically (inside the square root, added to the second
        moment). ``keras.layers.LayerNormalization(center=False)`` computes the same
        thing and may reach fused kernels this implementation cannot; this class
        exists for the RMSNorm framing and for the band/zero-centered family it
        belongs to.

    Statistics are computed in ``keras.backend.result_type(input_dtype, "float32")``
    - float32 at minimum, float64 under a float64 policy - and cast back to the
    input dtype on return.

    ``supports_masking`` is decided from the RESOLVED normalization axis, not set
    unconditionally: it is ``True`` only while every normalized axis is the trailing
    (feature) axis of the input. At the default ``axis=-1`` a single ``(sample, token)``
    perturbation of a ``(3, 5, 8)`` input moves no other position by more than
    ``0.0`` (measured, both training regimes), so a Keras mask remains valid.
    Normalizing over the TOKEN axis instead couples positions - measured leak
    ``2.189`` at ``axis=1`` on the same input - and there the flag is ``False``, so
    Keras drops the mask and says so. The decision is made in ``__init__`` from the
    spelling (only ``-1`` is rank-independent) and made exact in ``build()``.

    **Architecture Overview:**

    .. code-block:: text

        Input: x (batch, ..., features)
                │
                ▼
        ┌───────────────────────────────┐
        │  μ = mean(x) along axis       │
        └────────────┬──────────────────┘
                     │
                     ▼
        ┌───────────────────────────────┐
        │  x_centered = x - μ           │
        └────────────┬──────────────────┘
                     │
                     ▼
        ┌───────────────────────────────┐
        │  RMS = √(mean(x_centered²)+ε) │
        └────────────┬──────────────────┘
                     │
                     ▼
        ┌───────────────────────────────┐
        │  normalized = x_centered / RMS│
        └────────────┬──────────────────┘
                     │
                     ▼
        ┌───────────────────────────────┐
        │  output = normalized × γ      │
        │  (if use_scale=True)          │
        └────────────┬──────────────────┘
                     │
                     ▼
        Output: (batch, ..., features)

    :param axis: Axis or axes along which to compute mean and RMS statistics.
        The default (-1) computes statistics over the last dimension. For multi-axis
        normalization, pass a tuple (e.g., (-2, -1) for normalizing over last two
        dimensions).

        Non-trailing axes are supported together with ``use_scale=True``: the
        ``scale`` weight keeps its checkpoint-visible shape (one dimension per
        normalized axis) and is reshaped for broadcasting at call time only when
        the normalized axes are not the trailing ones, so ``axis=-1`` emits no
        reshape op.

        Deliberate carve-out: an axis tuple that is not strictly ascending (e.g.
        ``(-1, -2)``) keeps the legacy broadcast, because ``build()`` orders the
        scale's dimensions by the order the axes were WRITTEN. Use an ascending
        tuple.
    :type axis: Union[int, Tuple[int, ...]]
    :param epsilon: Small constant added to denominator for numerical stability.
        Should be positive and typically in range [1e-8, 1e-5].
    :type epsilon: float
    :param use_scale: Whether to use a learnable scaling parameter after
        normalization. When True, adds a trainable parameter that can help the model
        learn appropriate scaling.
    :type use_scale: bool
    :param scale_initializer: Initializer for the scale parameter when
        ``use_scale=True``. Common choices include "ones" (default), "zeros",
        or custom initializers.
    :type scale_initializer: Union[str, initializers.Initializer]
    :param kwargs: Additional keyword arguments passed to the parent Layer class.

    :raises ValueError: If epsilon is not positive.
    :raises ValueError: If attempting to normalize along dynamic axes during build.
    :raises TypeError: If axis is not int or tuple of ints.
    """

    def __init__(
            self,
            axis: Union[int, Tuple[int, ...]] = -1,
            epsilon: float = 1e-6,
            use_scale: bool = True,
            scale_initializer: Union[str, initializers.Initializer] = "ones",
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs early
        self._validate_inputs(axis, epsilon)

        # Store ALL configuration parameters - required for get_config()
        self.axis = axis
        self.epsilon = epsilon
        self.use_scale = use_scale
        self.scale_initializer = initializers.get(scale_initializer)

        # Initialize weight attributes - created in build()
        self.scale = None

        # Broadcast shape used to reshape 'scale' in call(); computed in build().
        # None means "no reshape needed" - see the DECISION anchor in build().
        # Initialized here so build()'s `if self.built: return` early exit can
        # never leave the attribute undefined.
        self._scale_broadcast_shape = None

        # supports_masking is a promise about the AXIS, not about the class: it holds
        # only while the normalized axis is the trailing (feature) axis. Decided here
        # from the spelling alone - `-1` names the trailing axis at every rank - and
        # made exact in build(), where the input rank is finally known.
        self.supports_masking = normalizes_only_the_feature_axis(axis)

        logger.debug(
            f"Initialized ZeroCenteredRMSNorm with "
            f"axis={axis}, "
            f"epsilon={epsilon}, "
            f"use_scale={use_scale}"
        )

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

        # Refine the __init__ estimate now that the rank is known. Keras reads
        # supports_masking inside __call__, which runs build() first, so this is the
        # value that decides whether the mask actually survives.
        self.supports_masking = normalizes_only_the_feature_axis(
            self.axis, rank=len(input_shape)
        )

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
                    f"Cannot create 'scale' parameter for ZeroCenteredRMSNorm. The "
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
            # .keras file holding a ZeroCenteredRMSNorm stores exactly
            # `param_shape`. Widening it above to a full-rank shape carrying 1s
            # at the unnormalized axes would be the cleaner algebra, but it would
            # make every existing checkpoint unloadable. So do NOT touch
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
        Apply Zero-Centered RMS normalization to inputs.

        :param inputs: Input tensor of any shape. Normalization is applied along
            the axes specified during initialization.
        :type inputs: keras.KerasTensor
        :param training: Boolean indicating whether the layer should behave in
            training mode. Not used in Zero-Centered RMSNorm but kept for
            consistency with other normalization layers.
        :type training: Optional[bool]
        :return: Zero-centered RMS normalized tensor with the same shape as inputs.
        :rtype: keras.KerasTensor
        """
        # Store original dtype for casting back
        original_dtype = inputs.dtype

        # Statistics dtype: float32 at minimum (numerical stability under
        # mixed precision), but float64 when the layer really is float64 -
        # a hardcoded "float32" here silently ran the statistics in float32
        # under a float64 policy (measured: the centered tensor collapsed to
        # exactly zero on an input whose float64 answer is O(1)).
        stat_dtype = keras.backend.result_type(original_dtype, "float32")
        inputs_fp32 = ops.cast(inputs, stat_dtype)

        # Step 1: Compute mean and center the input
        mean = ops.mean(
            inputs_fp32,
            axis=self.axis,
            keepdims=True
        )

        centered_inputs = inputs_fp32 - mean

        # Step 2: Compute RMS of centered inputs: sqrt(mean(x_centered²) + ε)
        mean_square = ops.mean(
            ops.square(centered_inputs),
            axis=self.axis,
            keepdims=True
        )

        # Add epsilon for numerical stability and compute RMS
        rms = ops.sqrt(mean_square + self.epsilon)

        # Step 3: Normalize by RMS
        normalized = centered_inputs / rms

        # Apply learnable scale if enabled
        if self.use_scale:
            scale = self.scale
            if self._scale_broadcast_shape is not None:
                # Non-trailing normalization axes only: the stored weight shape is
                # not broadcast-compatible with the input, so give it explicit 1s
                # at the unnormalized axes here rather than in build()
                # (DECISION plan-2026-08-25T195813-d5a035ab/D-004).
                scale = ops.reshape(scale, self._scale_broadcast_shape)
            normalized = normalized * scale

        # Cast back to original dtype
        return ops.cast(normalized, original_dtype)

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
            "scale_initializer": initializers.serialize(self.scale_initializer),
        })
        return config

# ---------------------------------------------------------------------
