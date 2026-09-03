"""BandRMS: RMS normalization followed by a learnable scale inside a band.

``BandRMS`` divides an input by its RMS over the normalization axes, then
multiplies by one learnable scalar ``s`` constrained to ``[1 - alpha, 1]``, where
``alpha`` is ``max_band_width``. The result is a thick shell in RMS space rather
than a fixed unit sphere: the model can learn where inside the band to sit, but
cannot leave it.

Computation
-----------

For an input ``x`` reduced over ``axis``::

    rms = maximum(sqrt(mean(x ** 2) + epsilon), epsilon)
    s   = (1 - alpha) + alpha * sigmoid(5.0 * band_param)
    output = (x / rms) * s

``mean(x ** 2)`` is used rather than ``sum(x ** 2)``, so the normalization does
not depend on the length of the normalized axis. A 64-wide and a 512-wide layer
therefore behave the same way, and the L2 norm of the output tracks
``s * sqrt(D)``. Measured at ``alpha=0.1`` with ``band_param`` at its "zeros"
initializer: ``D=64`` gives L2 ``7.6000`` against ``sqrt(64) = 8.0000``, and
``D=512`` gives ``21.4960`` against ``sqrt(512) = 22.6274``. Both are exactly
``0.95 * sqrt(D)``.

``band_param`` is a single scalar weight, so ``s`` is one number for the whole
tensor. At the "zeros" initializer ``sigmoid(0) = 0.5``, so the output RMS starts
at exactly ``1 - alpha / 2`` with no spread across rows to six decimals
(measured spread at most ``4.83e-07`` over shapes ``(4, 32)`` to ``(64, 128)``,
which is float32 rounding): measured ``0.950000`` at ``alpha=0.1`` and
``0.750000`` at ``alpha=0.5``. Both ends of the band are
reachable during training: ``band_param = -5`` gives RMS ``0.900000`` and
``band_param = +5`` gives ``1.000000`` at ``alpha=0.1``.

Statistics run in ``statistics_dtype(input_dtype)`` (defined in
``dl_techniques.utils.dtype_policy``). That is float32 for float16 and float32
inputs, and float64 under a float64 policy. A
hardcoded ``"float32"`` would be wrong there. Measured on the float64 input
``[[1e8+1, 1e8+2, 1e8+3, 1e8+4]]`` at ``alpha=0.1``: float32 statistics return
exactly ``[[0.95, 0.95, 0.95, 0.95]]`` while float64 statistics return
``[[0.94999999, 0.95, 0.95, 0.95000001]]``, a gap of ``1.425e-08``.

References
----------

[1] Zhang, B., & Sennrich, R. (2019). "Root Mean Square Layer Normalization."
    https://arxiv.org/abs/1910.07467
"""

import keras
from keras import ops
from typing import Any, Dict, Optional, Union, Tuple

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


@register_dl_technique("dl_techniques.layers.norms.band_rms")
class BandRMS(keras.layers.Layer):
    """Normalize by the root mean square, then scale inside a learnable band.

    Divides the input by ``maximum(sqrt(mean(x ** 2) + epsilon), epsilon)`` over
    ``axis``, then multiplies by one learnable scalar ``s`` constrained to
    ``[1 - alpha, 1]``, where ``alpha`` is ``max_band_width``. ``s`` comes from a
    single scalar weight through a sigmoid, so it is one number for the whole
    tensor. The output has the same shape and dtype as the input.

    The band is a constraint on the RMS, not on the L2 norm, so it does not
    depend on the width of the normalized axis. The output L2 norm tracks
    ``s * sqrt(D)``: measured at ``alpha=0.1`` and the default ``band_param``,
    ``D=64`` gives ``7.6000`` and ``D=512`` gives ``21.4960``, both exactly
    ``0.95 * sqrt(D)``.

    At the default "zeros" initializer ``sigmoid(0) = 0.5``, so the output RMS
    starts at exactly ``1 - alpha / 2`` with no spread across rows to six
    decimals (measured spread at most ``4.83e-07`` over shapes ``(4, 32)`` to
    ``(64, 128)``, float32 rounding): measured ``0.950000`` at ``alpha=0.1`` and
    ``0.750000`` at ``alpha=0.5``. Training
    reaches both ends of the band: ``band_param = -5`` gives ``0.900000`` and
    ``band_param = +5`` gives ``1.000000`` at ``alpha=0.1``.

    Statistics run in ``statistics_dtype(input_dtype)``: float32 at minimum,
    float64 under a float64 policy. The result is cast back to the input dtype
    before it is returned.

    ``supports_masking`` is a promise about the AXIS, not about the class. It is
    ``True`` only while every normalized axis is the trailing (feature) axis. At
    the default ``axis=-1`` the band scale is one global number and the RMS is
    per-position, so the measured cross-position leak on a ``(3, 5, 8)`` input is
    exactly ``0.0`` in both training regimes. Normalizing the token axis couples
    positions: the measured leak at ``axis=1`` on the same input is ``1.960``.
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
          │ statistics_dtype(input dtype)                 │
          └───────────────────────┬───────────────────────┘
                                  │ x_stat
                                  ▼
          ┌───────────────────────────────────────────────┐
          │ mean_square = mean(x_stat ** 2) over axis,    │
          │ keepdims=True                                 │
          └───────────────────────┬───────────────────────┘
                                  │ (batch, ..., 1)
                                  ▼
          ┌───────────────────────────────────────────────┐
          │ rms = maximum(sqrt(mean_square + epsilon),    │
          │               epsilon)                        │
          └───────────────────────┬───────────────────────┘
                                  │ (batch, ..., 1)
                                  ▼
          ┌───────────────────────────────────────────────┐
          │ normalized = x_stat / rms                     │
          └───────────────────────┬───────────────────────┘
                                  │ (batch, ..., F)
                                  ▼
          ┌───────────────────────────────────────────────┐
          │ band = sigmoid(5.0 * band_param)              │
          │ s = (1 - max_band_width)                      │
          │     + max_band_width * band                   │
          └───────────────────────┬───────────────────────┘
                                  │ s: scalar in [1-alpha, 1]
                                  ▼
          ┌───────────────────────────────────────────────┐
          │ output = normalized * s                       │
          └───────────────────────┬───────────────────────┘
                                  │
                                  ▼
          ┌───────────────────────────────────────────────┐
          │ cast back to the input dtype                  │
          └───────────────────────┬───────────────────────┘
                                  │
                                  ▼
                      output: (batch, ..., F)  same dtype as x

    :param max_band_width: Thickness of the band, written ``alpha`` above. The
        learned scale is confined to ``[1 - alpha, 1]``. Must satisfy
        ``0 < alpha < 1``. Defaults to 0.1.
    :type max_band_width: float
    :param axis: Axis or axes reduced by the RMS statistic. The default -1
        reduces the last dimension. Can be an int or a tuple of ints.
    :type axis: Union[int, Tuple[int, ...]]
    :param epsilon: Constant added inside the square root, and also the floor the
        RMS is clamped to. Must be positive. Defaults to 1e-7.
    :type epsilon: float
    :param band_initializer: Initializer for the scalar ``band_param`` weight.
        Defaults to "zeros", which starts the scale at the middle of the band.
    :type band_initializer: Union[str, keras.initializers.Initializer]
    :param band_regularizer: Regularizer for ``band_param``. When None, L2(1e-5)
        is used.
    :type band_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional keyword arguments for ``keras.layers.Layer``.
    :type kwargs: Any

    :ivar max_band_width: The configured band thickness.
    :vartype max_band_width: float
    :ivar axis: The configured normalization axis or axes.
    :vartype axis: Union[int, Tuple[int, ...]]
    :ivar epsilon: The configured numerical constant.
    :vartype epsilon: float
    :ivar band_initializer: The resolved initializer for ``band_param``.
    :vartype band_initializer: keras.initializers.Initializer
    :ivar band_regularizer: The resolved regularizer for ``band_param``.
    :vartype band_regularizer: keras.regularizers.Regularizer
    :ivar band_param: The scalar learnable weight, or None until ``build()``
        runs.
    :vartype band_param: Optional[keras.Variable]

    :raises ValueError: If max_band_width is not strictly between 0 and 1.
    :raises ValueError: If epsilon is not positive.
    :raises TypeError: If axis is not an int or a tuple of ints.

    Example:

    .. code-block:: python

        import keras
        from dl_techniques.layers.norms import BandRMS

        x = keras.random.normal((4, 16, 64))
        y = BandRMS(max_band_width=0.1)(x)
    """

    def __init__(
        self,
        max_band_width: float = 0.1,
        axis: Union[int, Tuple[int, ...]] = -1,
        epsilon: float = 1e-7,
        band_initializer: Union[str, keras.initializers.Initializer] = "zeros",
        band_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs: Any
    ) -> None:
        """Initialize the layer.

        :param max_band_width: Thickness of the band. Must satisfy
            ``0 < max_band_width < 1``.
        :type max_band_width: float
        :param axis: Axis or axes reduced by the RMS statistic.
        :type axis: Union[int, Tuple[int, ...]]
        :param epsilon: Constant added inside the square root and used as the RMS
            floor. Must be positive.
        :type epsilon: float
        :param band_initializer: Initializer for the scalar ``band_param``.
        :type band_initializer: Union[str, keras.initializers.Initializer]
        :param band_regularizer: Regularizer for ``band_param``. When None,
            L2(1e-5) is used.
        :type band_regularizer: Optional[keras.regularizers.Regularizer]
        :param kwargs: Additional keyword arguments for ``keras.layers.Layer``.
        :type kwargs: Any

        :raises ValueError: If max_band_width is not strictly between 0 and 1.
        :raises ValueError: If epsilon is not positive.
        :raises TypeError: If axis is not an int or a tuple of ints.
        """
        super().__init__(**kwargs)

        # Validate inputs early
        self._validate_inputs(max_band_width, axis, epsilon)

        # Store ALL configuration parameters - required for get_config()
        self.max_band_width = max_band_width
        self.axis = axis
        self.epsilon = epsilon
        self.band_initializer = keras.initializers.get(band_initializer)

        # Default regularizer if none provided. Pass through keras.regularizers.get()
        # so a serialized regularizer dict (from from_config) is rebuilt into a
        # Regularizer object, keeping get_config/from_config round-trips correct.
        self.band_regularizer = keras.regularizers.get(band_regularizer) or keras.regularizers.L2(1e-5)

        # Initialize weight attributes - created in build()
        self.band_param = None

        # supports_masking is a promise about the AXIS, not about the class: it holds
        # only while the normalized axis is the trailing (feature) axis. Decided here
        # from the spelling alone - `-1` names the trailing axis at every rank - and
        # made exact in build(), where the input rank is finally known.
        self.supports_masking = normalizes_only_the_feature_axis(axis)

        logger.debug(f"Initialized BandRMS with max_band_width={max_band_width}, axis={axis}, epsilon={epsilon}")

    def _validate_inputs(
        self,
        max_band_width: float,
        axis: Union[int, Tuple[int, ...]],
        epsilon: float,
    ) -> None:
        """Reject an out-of-range band width, a bad axis, or a non-positive epsilon.

        :param max_band_width: Band thickness to validate.
        :type max_band_width: float
        :param axis: Normalization axis or axes to validate.
        :type axis: Union[int, Tuple[int, ...]]
        :param epsilon: Epsilon value to validate.
        :type epsilon: float

        :raises ValueError: If max_band_width is not strictly between 0 and 1.
        :raises ValueError: If epsilon is not positive.
        :raises TypeError: If axis is not an int or a sequence of ints.
        """
        if not 0 < max_band_width < 1:
            raise ValueError(
                f"max_band_width must be between 0 and 1, got {max_band_width}"
            )
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        if isinstance(axis, (list, tuple)):
            if not all(isinstance(ax, int) for ax in axis):
                raise TypeError(
                    f"All elements in axis must be integers, got {axis}"
                )
        elif not isinstance(axis, int):
            raise TypeError(
                f"axis must be int or tuple of ints, got {type(axis)}"
            )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the scalar ``band_param`` weight and settle the masking promise.

        Runs on the first call, when the input rank is known. ``band_param`` has
        shape ``()``, so its size does not depend on the input shape.

        :param input_shape: Shape tuple of the input tensor. The batch dimension
            may be None.
        :type input_shape: Tuple[Optional[int], ...]
        """
        if self.built:
            return

        # Refine the __init__ estimate now that the rank is known. Keras reads
        # supports_masking inside __call__, which runs build() first, so this is the
        # value that decides whether the mask actually survives.
        self.supports_masking = normalizes_only_the_feature_axis(
            self.axis, rank=len(input_shape)
        )

        # Create a single scalar band parameter using add_weight().
        # shape=() is a scalar: one number scales the whole tensor.
        self.band_param = self.add_weight(
            name="band_param",
            shape=(),
            initializer=self.band_initializer,
            trainable=True,
            regularizer=self.band_regularizer
        )

        logger.debug("Created scalar band parameter")

        # Always call parent build at the end
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply RMS normalization and the band scale.

        :param inputs: Input tensor of any shape. Normalization runs along the
            axes given at construction.
        :type inputs: keras.KerasTensor
        :param training: Training-mode flag. Unused; the layer behaves the same
            in both modes and the argument is kept for API compatibility.
        :type training: Optional[bool]

        :return: Normalized tensor, same shape and dtype as ``inputs``. Its RMS
            lies in ``[1 - max_band_width, 1]`` and its L2 norm therefore lies in
            ``[(1 - max_band_width) * sqrt(D), sqrt(D)]``, up to the epsilon floor.
        :rtype: keras.KerasTensor
        """
        # Store original dtype for casting back
        original_dtype = inputs.dtype

        # Statistics dtype: float32 at minimum for numerical stability under
        # mixed precision, and float64 when the layer really is float64. A
        # hardcoded "float32" here runs the statistics in float32 under a float64
        # policy. Measured on the float64 input [[1e8+1, 1e8+2, 1e8+3, 1e8+4]] at
        # max_band_width=0.1: float32 statistics return exactly [[0.95] * 4] and
        # float64 ones return [[0.94999999, 0.95, 0.95, 0.95000001]], a gap of
        # 1.425e-08.
        stat_dtype = statistics_dtype(original_dtype)
        inputs_fp32 = ops.cast(inputs, stat_dtype)

        # Step 1: RMS normalization to achieve dimension-independent scaling
        # Compute RMS: sqrt(mean(x²))
        # Using mean(x²) instead of sum(x²) ensures normalization is independent
        # of vector dimension - critical for consistent behavior across layer widths
        mean_square = ops.mean(
            ops.square(inputs_fp32),
            axis=self.axis,
            keepdims=True
        )

        # Clamp it for stability, protection for division by zero and sqrt(negative)
        rms = ops.maximum(
            ops.sqrt(mean_square + self.epsilon),
            self.epsilon
        )

        # Normalize by RMS: this sets RMS=1 and L2_norm≈sqrt(D)
        # This scaling is dimension-independent, unlike L2 normalization
        normalized = inputs_fp32 / rms

        # Step 2: Apply learnable scaling within [1-α, 1] band
        # Use sigmoid to map the band_param to [0, 1]
        # with 5x multiplier, sigmoid(-5) ~ 0, sigmoid(+5) ~ 1
        # DECISION plan_2026-05-14_3764496e/D-002: cast band_param to the statistics
        # dtype explicitly. Under mixed_float16 a variable auto-casts on read.
        # Measured inside the autocast scope Layer.__call__ opens,
        # band_param reads float16, so the following `normalized * scale` raises
        # InvalidArgumentError. Do NOT hardcode "float32"; stat_dtype keeps the
        # multiply in the policy's statistics dtype
        # (DECISION plan-2026-08-25T195813-d5a035ab/D-005). The originating plan
        # directory plan_2026-05-14_3764496e is gone, so this comment is the record.
        band_param_fp32 = ops.cast(self.band_param, stat_dtype)
        band_activation = ops.sigmoid(5.0 * band_param_fp32)

        # Scale the activation to be within [1-max_band_width, 1]
        # When band_activation = 0: scale = 1 - max_band_width
        # When band_activation = 1: scale = 1
        scale = (1.0 - self.max_band_width) + (self.max_band_width * band_activation)

        # Apply scaling to the normalized tensor
        # The single scalar scale is automatically broadcast to all elements
        output = normalized * scale

        # Cast back to original dtype
        return ops.cast(output, inputs.dtype)

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
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
            "max_band_width": self.max_band_width,
            "axis": self.axis,
            "epsilon": self.epsilon,
            "band_initializer": keras.initializers.serialize(self.band_initializer),
            "band_regularizer": keras.regularizers.serialize(self.band_regularizer),
        })
        return config

# ---------------------------------------------------------------------
