"""ZeroCenteredRMSNorm: center over the axis, then normalize by the root mean square.

``ZeroCenteredRMSNorm`` subtracts the mean over the normalization axes, divides
by the RMS of the centered values, and multiplies by an optional learnable
``scale``. Centering is what separates it from ``RMSNorm``: it stops the mean of
the activations drifting, which is the failure Qwen3-Next reports as abnormal
growth of normalization weights.

Computation
-----------

For an input ``x`` reduced over ``axis``::

    mean = mean(x)
    centered = x - mean
    rms = sqrt(mean(centered ** 2) + epsilon)
    output = (centered / rms) * scale

This is the same function as ``keras.layers.LayerNormalization(center=False)``,
not merely a similar one. Centering drives ``mean(centered)`` to zero, so
``mean(centered ** 2)`` is ``var(x)`` over the same axes, and ``epsilon`` sits in
the same place, inside the square root and added to the second moment. Measured
on a ``(4, 16)`` input, the two layers agree to ``3.576e-07`` at worst across
``epsilon`` in {1e-6, 1e-3, 1e-1} and input scales in {3, 1e-2, 1e-3}, including
the regime where ``epsilon`` dominates the variance. Centering is exact only in
exact arithmetic: the measured ``max|mean(output)|`` over the normalized axis is
``2.980e-08`` in float32.

If you want that function and nothing else, prefer
``keras.layers.LayerNormalization(center=False)``, which may reach fused kernels
this implementation cannot. Note the epsilon defaults differ: this layer uses
1e-6, ``keras.layers.LayerNormalization`` uses 1e-3.

Statistics run in ``keras.backend.result_type(input_dtype, "float32")``. That is
float32 for float16 and float32 inputs, and float64 under a float64 policy. A
hardcoded ``"float32"`` would be wrong there. Measured on the float64 input
``[[1e8+1, 1e8+2, 1e8+3, 1e8+4]]``: float32 statistics collapse the centered
tensor to exactly ``[[0, 0, 0, 0]]``, while float64 statistics return
``[[-1.342, -0.447, 0.447, 1.342]]``.

References
----------

[1] Used in the Qwen3-Next model to stop abnormal growth of layer normalization
    weights.
[2] Zhang, B., & Sennrich, R. (2019). "Root Mean Square Layer Normalization."
    https://arxiv.org/abs/1910.07467
"""


import keras
from keras import ops, initializers
from typing import Optional, Union, Tuple, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.dtype_policy import statistics_dtype
from dl_techniques.layers.norms._masking import (
    normalizes_only_the_feature_axis,
)
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.norms.zero_centered_rms_norm")
class ZeroCenteredRMSNorm(keras.layers.Layer):
    """Center over the normalization axis, then normalize by the root mean square.

    Subtracts the mean over ``axis``, divides by ``sqrt(mean(centered ** 2) +
    epsilon)``, and multiplies by an optional learnable ``scale``. Centering stops
    the activation mean drifting, which is the failure Qwen3-Next reports as
    abnormal growth of normalization weights. The output has the same shape and
    dtype as the input.

    .. note::
        This is the same function as
        ``keras.layers.LayerNormalization(center=False)``, not merely a similar
        one. Centering drives ``mean(centered)`` to zero, so
        ``mean(centered ** 2)`` is ``var(x)`` over the same axes, and ``epsilon``
        sits in the same place. Measured on a ``(4, 16)`` input, the two agree to
        ``3.576e-07`` at worst across ``epsilon`` in {1e-6, 1e-3, 1e-1} and input
        scales in {3, 1e-2, 1e-3}. Prefer the Keras layer if you want that
        function and nothing else; it may reach fused kernels this implementation
        cannot. The epsilon defaults differ: 1e-6 here, 1e-3 there.

    Statistics run in ``keras.backend.result_type(input_dtype, "float32")``:
    float32 at minimum, float64 under a float64 policy. The result is cast back
    to the input dtype before it is returned.

    ``supports_masking`` is a promise about the AXIS, not about the class. It is
    ``True`` only while every normalized axis is the trailing (feature) axis. At
    the default ``axis=-1`` each position is centered and scaled from its own
    statistics, and the measured cross-position leak on a ``(3, 5, 8)`` input is
    exactly ``0.0`` in both training regimes. Normalizing the token axis couples
    positions: the measured leak at ``axis=1`` on the same input is ``2.189``.
    The flag is ``False`` there, so Keras drops the mask and says so.
    ``__init__`` decides from the spelling alone, because only ``-1`` names the
    trailing axis at every rank. ``build()`` then makes it exact.

    **Architecture Overview:**

    .. code-block:: text

                      input: x  (batch, ..., F)
                                  │
                                  ▼
          ┌───────────────────────────────────────────────┐
          │ cast inputs to stat_dtype =                   │
          │ result_type(input dtype, "float32")           │
          └───────────────────────┬───────────────────────┘
                                  │ x_stat
                                  ▼
          ┌───────────────────────────────────────────────┐
          │ mean = mean(x_stat) over axis,                │
          │ keepdims=True                                 │
          │ centered = x_stat - mean                      │
          └───────────────────────┬───────────────────────┘
                                  │ (batch, ..., F)
                                  ▼
          ┌───────────────────────────────────────────────┐
          │ mean_square = mean(centered ** 2) over axis,  │
          │ keepdims=True                                 │
          └───────────────────────┬───────────────────────┘
                                  │ (batch, ..., 1)
                                  ▼
          ┌───────────────────────────────────────────────┐
          │ rms = sqrt(mean_square + epsilon)             │
          └───────────────────────┬───────────────────────┘
                                  │ (batch, ..., 1)
                                  ▼
          ┌───────────────────────────────────────────────┐
          │ normalized = centered / rms                   │
          └───────────────────────┬───────────────────────┘
                                  │ (batch, ..., F)
                                  ▼
          ┌───────────────────────────────────────────────┐
          │ scale = self.scale, reshaped only when        │
          │ _scale_broadcast_shape is not None            │
          │ normalized = normalized * scale               │
          │ (optional: use_scale=True)                    │
          └───────────────────────┬───────────────────────┘
                                  │
                                  ▼
          ┌───────────────────────────────────────────────┐
          │ cast back to the input dtype                  │
          └───────────────────────┬───────────────────────┘
                                  │
                                  ▼
                      output: (batch, ..., F)  same dtype as x

    :param axis: Axis or axes reduced by the mean and RMS statistics. The default
        -1 reduces the last dimension. Pass a tuple for multi-axis normalization,
        for example ``(-2, -1)``.

        Non-trailing axes work together with ``use_scale=True``. The ``scale``
        weight keeps one dimension per normalized axis, which is the shape stored
        in every checkpoint, and it is reshaped for broadcasting at call time only
        when the normalized axes are not the trailing ones. The ``axis=-1`` path
        emits no reshape op.

        One spelling is carved out. An axis tuple that is not strictly ascending,
        such as ``(-1, -2)``, keeps the legacy broadcast rather than a corrected
        one, because ``build()`` orders the scale's dimensions by the order the
        axes were WRITTEN. That spelling raises ``InvalidArgumentError`` on the
        first call. Use an ascending tuple.
    :type axis: Union[int, Tuple[int, ...]]
    :param epsilon: Constant added inside the square root for numerical
        stability. Must be positive. Typical range is [1e-8, 1e-5]. Defaults to
        1e-6.
    :type epsilon: float
    :param use_scale: Whether to create a learnable ``scale`` weight applied
        after normalization. Defaults to True.
    :type use_scale: bool
    :param scale_initializer: Initializer for the ``scale`` weight when
        ``use_scale=True``. Defaults to "ones".
    :type scale_initializer: Union[str, initializers.Initializer]
    :param kwargs: Additional keyword arguments for ``keras.layers.Layer``.
    :type kwargs: Any

    :ivar axis: The configured normalization axis or axes.
    :vartype axis: Union[int, Tuple[int, ...]]
    :ivar epsilon: The configured numerical constant.
    :vartype epsilon: float
    :ivar use_scale: Whether a ``scale`` weight is created.
    :vartype use_scale: bool
    :ivar scale_initializer: The resolved initializer for ``scale``.
    :vartype scale_initializer: initializers.Initializer
    :ivar scale: The learnable scale weight, or None until ``build()`` runs and
        None forever when ``use_scale=False``.
    :vartype scale: Optional[keras.Variable]

    :raises ValueError: If epsilon is not positive.
    :raises TypeError: If axis is not an int or a tuple of ints.
    :raises ValueError: If a normalized axis has a dynamic (None) dimension while
        ``use_scale=True``.

    Example:

    .. code-block:: python

        import keras
        from dl_techniques.layers.norms import ZeroCenteredRMSNorm

        x = keras.random.normal((4, 16, 64))
        y = ZeroCenteredRMSNorm(axis=-1)(x)
    """

    def __init__(
            self,
            axis: Union[int, Tuple[int, ...]] = -1,
            epsilon: float = 1e-6,
            use_scale: bool = True,
            scale_initializer: Union[str, initializers.Initializer] = "ones",
            **kwargs: Any
    ) -> None:
        """Initialize the layer.

        :param axis: Axis or axes reduced by the mean and RMS statistics.
        :type axis: Union[int, Tuple[int, ...]]
        :param epsilon: Constant added inside the square root. Must be positive.
        :type epsilon: float
        :param use_scale: Whether to create a learnable ``scale`` weight.
        :type use_scale: bool
        :param scale_initializer: Initializer for the ``scale`` weight.
        :type scale_initializer: Union[str, initializers.Initializer]
        :param kwargs: Additional keyword arguments for ``keras.layers.Layer``.
        :type kwargs: Any

        :raises ValueError: If epsilon is not positive.
        :raises TypeError: If axis is not an int or a tuple of ints.
        """
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
        """Reject an invalid axis or a non-positive epsilon.

        :param axis: Normalization axis or axes to validate.
        :type axis: Union[int, Tuple[int, ...]]
        :param epsilon: Epsilon value to validate.
        :type epsilon: float

        :raises ValueError: If epsilon is not positive.
        :raises TypeError: If axis is not an int or a tuple of ints.
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
        """Create the ``scale`` weight and settle the masking promise.

        Runs on the first call, when the input rank is known. Creates ``scale``
        with one dimension per normalized axis when ``use_scale=True``, and
        computes the broadcast shape ``call()`` will reshape it to.

        :param input_shape: Shape tuple of the input tensor. The batch dimension
            may be None.
        :type input_shape: Tuple[Optional[int], ...]

        :raises ValueError: If a normalized axis has a dynamic (None) dimension
            while ``use_scale=True``.
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
            # 'scale' is stored at param_shape. Measured: axis=-1 on a (2, 4, 8)
            # input stores (8,), not the full-rank (1, 1, 8). Do NOT widen it that
            # way; every saved .keras holding a ZeroCenteredRMSNorm would stop
            # loading. call() reshapes for broadcast instead. See that plan's
            # decisions.md D-004.
            rank = len(input_shape)
            if param_axes == list(range(rank - len(param_axes), rank)):
                # The normalized axes are exactly the trailing axes, in ascending
                # order. `param_shape` already broadcasts against the input, so
                # this path (axis=-1, the default) emits no reshape op at all.
                self._scale_broadcast_shape = None
            elif any(b <= a for a, b in zip(param_axes, param_axes[1:])):
                # Not strictly ascending (e.g. axis=(-1, -2)): build() orders the
                # scale's dimensions by the order the axes were WRITTEN, so a
                # broadcast shape derived from ascending order would silently
                # reinterpret the stored weight. An unsorted 'axis' tuple is an
                # unsupported spelling and keeps today's behaviour verbatim; it
                # raises InvalidArgumentError on the first call. Do NOT "fix"
                # this by sorting param_axes.
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
        """Apply zero-centered RMS normalization.

        :param inputs: Input tensor of any shape. Normalization runs along the
            axes given at construction.
        :type inputs: keras.KerasTensor
        :param training: Training-mode flag. Unused; the layer behaves the same
            in both modes and the argument is kept for API compatibility.
        :type training: Optional[bool]

        :return: Normalized tensor, same shape and dtype as ``inputs``.
        :rtype: keras.KerasTensor
        """
        # Store original dtype for casting back
        original_dtype = inputs.dtype

        # Statistics dtype: float32 at minimum for numerical stability under
        # mixed precision, and float64 when the layer really is float64. A
        # hardcoded "float32" here runs the statistics in float32 under a float64
        # policy. Measured on the float64 input [[1e8+1, 1e8+2, 1e8+3, 1e8+4]]:
        # float32 statistics collapse `centered_inputs` to exactly [[0, 0, 0, 0]]
        # while float64 ones give [[-1.342, -0.447, 0.447, 1.342]].
        stat_dtype = statistics_dtype(original_dtype)
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
                # (DECISION plan-2026-08-25T195813-d5a035ab/D-004). Measured:
                # axis=1 on a (2, 4, 8) input stores (4,) and reshapes to (1, 4, 1).
                scale = ops.reshape(scale, self._scale_broadcast_shape)
            normalized = normalized * scale

        # Cast back to original dtype
        return ops.cast(normalized, original_dtype)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Return the output shape, which equals the input shape.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]

        :return: The same shape tuple that was passed in.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration needed to rebuild this layer.

        :return: Dictionary holding every constructor argument.
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
