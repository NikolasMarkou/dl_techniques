"""
Zero-Centered Adaptive BandRMS Layer.

This layer combines two normalization innovations:

1. **Zero-centering** (from :class:`ZeroCenteredBandRMSNorm`) — subtracts the
   per-axis mean before normalization for enhanced training stability.
2. **Adaptive log-RMS dense-projected scaling** (from :class:`AdaptiveBandRMS`)
   — computes a per-feature adaptive scaling factor from a dense projection
   over the log-transformed aggregate RMS statistic.

The combined operation creates an input-adaptive "thick spherical shell"
constraint in zero-mean RMS space.

Mathematical Operations:

    1. mu = mean(x, axis, keepdims=True)
    2. x_centered = x - mu
    3. rms = max(sqrt(mean(x_centered^2, axis, keepdims=True) + eps), eps)
    4. x_hat = x_centered / rms
    5. log_rms = log(aggregate(rms))                      # shape (batch, 1)
    6. band_logits = Dense(num_params)(log_rms)
    7. scale = (1 - alpha) + alpha * sigmoid(5 * band_logits)  # in [1-alpha, 1]
    8. y = x_hat * reshape(scale, broadcast_shape)

References:
[1] Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)
    https://arxiv.org/abs/1910.07467
"""

import keras
from keras import ops
from typing import Any, Dict, Optional, Union, Tuple, List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class ZeroCenteredAdaptiveBandRMS(keras.layers.Layer):
    """
    Zero-Centered Adaptive Root Mean Square Normalization.

    Combines zero-centering with adaptive log-RMS dense-projected scaling. The
    input is first centered by subtracting the per-axis mean, then normalized
    by the RMS of the centered tensor; finally, a per-feature scaling factor in
    [1 - alpha, 1] is computed from a Dense projection over the log-transformed
    aggregate RMS and applied multiplicatively.

    This produces a learnable "thick spherical shell" constraint in zero-mean
    RMS space, combining the training stability of zero-centering with the
    input-adaptive flexibility of log-RMS-driven scaling.

    Supports arbitrary tensor shapes (2D, 3D, 4D, 5D+) with flexible axis
    configurations matching :class:`AdaptiveBandRMS`.

    **Mathematical Operations:**

    1. ``mu = mean(x, axis, keepdims=True)``
    2. ``x_centered = x - mu``
    3. ``rms = max(sqrt(mean(x_centered**2, axis, keepdims=True) + eps), eps)``
    4. ``x_hat = x_centered / rms``
    5. ``log_rms = log(aggregate(rms))``
    6. ``scale = (1 - alpha) + alpha * sigmoid(5 * Dense(log_rms))``
    7. ``y = x_hat * scale``

    **Architecture Overview:**

    .. code-block:: text

        Input: x (batch, ..., features)
                │
                ▼
        ┌───────────────────────────────────┐
        │  μ = mean(x, axis, keepdims=True) │
        └──────────────┬────────────────────┘
                       │
                       ▼
        ┌───────────────────────────────────┐
        │  x_centered = x - μ               │
        └──────────────┬────────────────────┘
                       │
                       ▼
        ┌───────────────────────────────────┐
        │  RMS = max(√(mean(x_c²)+ε), ε)    │
        └──────────┬────────────────────────┘
                   │
                   ├──────────────────────┐
                   ▼                      ▼
        ┌──────────────────┐   ┌──────────────────────┐
        │ x_norm =         │   │ Aggregate RMS stats  │
        │   x_centered/RMS │   │ to [batch, 1]        │
        └────────┬─────────┘   └──────────┬───────────┘
                 │                        ▼
                 │             ┌──────────────────────┐
                 │             │ log_rms = log(stats) │
                 │             └──────────┬───────────┘
                 │                        ▼
                 │             ┌──────────────────────┐
                 │             │ Dense → band_logits  │
                 │             └──────────┬───────────┘
                 │                        ▼
                 │             ┌──────────────────────┐
                 │             │ σ = sigmoid(5×logits)│
                 │             │ scale = (1-α)+α×σ    │
                 │             │ scale ∈ [1-α, 1]     │
                 │             └──────────┬───────────┘
                 │                        │
                 ▼                        ▼
        ┌───────────────────────────────────┐
        │  output = x_norm × scale          │
        └──────────────┬────────────────────┘
                       │
                       ▼
        Output: (batch, ..., features), zero-mean, scaled into adaptive band

    :param max_band_width: Maximum allowed deviation from unit normalization
        (0 < alpha < 1). Controls the thickness of the adaptive spherical shell.
    :type max_band_width: float
    :param axis: Axes along which to compute mean and RMS statistics. For 2D
        (batch, features) use ``axis=-1``; for 3D (batch, seq, features) use
        ``axis=-1`` or ``axis=1``; for 4D (batch, H, W, channels) use
        ``axis=-1`` or ``axis=(1, 2)``.
    :type axis: Union[int, Tuple[int, ...]]
    :param epsilon: Small positive constant added to denominator for numerical
        stability.
    :type epsilon: float
    :param band_initializer: Initializer for the dense layer computing scaling
        parameters. Defaults to ``"zeros"`` for stable initialization near
        unit scaling.
    :type band_initializer: Union[str, keras.initializers.Initializer]
    :param band_regularizer: Optional regularizer for the dense layer weights.
        Defaults to ``None`` (matching :class:`AdaptiveBandRMS`).
    :type band_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional arguments for the Layer base class.

    :raises ValueError: If ``max_band_width`` is not strictly between 0 and 1.
    :raises ValueError: If ``epsilon`` is not positive.
    :raises ValueError: If ``axis`` configuration includes batch dimension (0).
    :raises ValueError: If ``axis`` is out of bounds for input tensor rank.
    :raises TypeError: If ``axis`` is not int or tuple of ints.
    """

    def __init__(
        self,
        max_band_width: float = 0.1,
        axis: Union[int, Tuple[int, ...]] = -1,
        epsilon: float = 1e-7,
        band_initializer: Union[str, keras.initializers.Initializer] = "zeros",
        band_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self._validate_inputs(max_band_width, axis, epsilon)

        # Store ALL configuration parameters for get_config().
        self.max_band_width = max_band_width
        self.axis = axis
        self.epsilon = epsilon
        self.band_initializer = keras.initializers.get(band_initializer)
        self.band_regularizer = keras.regularizers.get(band_regularizer)

        # Shape computation results (set in build).
        self._param_shape: Optional[Tuple[int, ...]] = None
        self._scaling_axes: Optional[List[int]] = None

        # Sub-layer created in build() due to input-shape dependency.
        self.dense_layer: Optional[keras.layers.Dense] = None

        logger.debug(
            f"Initialized ZeroCenteredAdaptiveBandRMS: "
            f"max_band_width={max_band_width}, "
            f"axis={axis}, "
            f"epsilon={epsilon}"
        )

    def _validate_inputs(
        self,
        max_band_width: float,
        axis: Union[int, Tuple[int, ...]],
        epsilon: float,
    ) -> None:
        """Validate initialization parameters."""
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

    def _compute_param_shape_and_axes(
        self,
        input_shape: Tuple[Optional[int], ...],
    ) -> Tuple[Tuple[int, ...], List[int]]:
        """Compute parameter shape and scaling axes from input shape."""
        input_rank = len(input_shape)

        if isinstance(self.axis, int):
            axes = [self.axis]
        else:
            axes = list(self.axis)

        normalized_axes: List[int] = []
        for ax in axes:
            if ax < 0:
                ax = input_rank + ax
            if ax == 0:
                raise ValueError(
                    "axis 0 (batch dimension) cannot be normalized"
                )
            if ax < 0 or ax >= input_rank:
                raise ValueError(
                    f"axis {ax} is out of bounds for input with {input_rank} "
                    f"dimensions"
                )
            normalized_axes.append(ax)
        normalized_axes = sorted(set(normalized_axes))

        is_global = (
            input_rank > 2 and len(normalized_axes) == input_rank - 1
        )
        if is_global:
            param_shape = [1] * (input_rank - 1)
            return tuple(param_shape), []

        param_shape: List[int] = []
        scaling_axes: List[int] = []
        for i in range(1, input_rank):
            if i in normalized_axes:
                if input_shape[i] is None:
                    raise ValueError(
                        f"Normalized axis {i} has an undefined (None) size; "
                        f"ZeroCenteredAdaptiveBandRMS needs a static dimension to "
                        f"size its scaling Dense layer. Got input_shape={input_shape}."
                    )
                param_shape.append(input_shape[i])
                scaling_axes.append(i)
            else:
                param_shape.append(1)
        return tuple(param_shape), scaling_axes

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the inner dense layer with proper parameter sizing."""
        if self.built:
            return

        self._param_shape, self._scaling_axes = (
            self._compute_param_shape_and_axes(input_shape)
        )

        num_params = 1
        for dim in self._param_shape:
            if dim is not None:
                num_params *= dim

        self.dense_layer = keras.layers.Dense(
            units=num_params,
            kernel_initializer=self.band_initializer,
            kernel_regularizer=self.band_regularizer,
            use_bias=True,
            name="band_dense",
        )
        # Aggregated log-RMS is always shape (batch, 1).
        self.dense_layer.build((None, 1))

        logger.debug(
            f"Built ZeroCenteredAdaptiveBandRMS: "
            f"param_shape={self._param_shape}, num_params={num_params}"
        )

        super().build(input_shape)

    def _aggregate_rms_statistics(
        self, rms_tensor: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Aggregate RMS statistics to ``(batch, 1)`` for dense input."""
        aggregation_axes = list(range(1, len(rms_tensor.shape)))
        if aggregation_axes:
            rms_stats = ops.mean(
                rms_tensor, axis=aggregation_axes, keepdims=True
            )
        else:
            rms_stats = rms_tensor
        return ops.reshape(rms_stats, [-1, 1])

    def _reshape_scaling_factors(
        self,
        scaling_factors: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """Reshape dense outputs to broadcast against the input."""
        batch_size = ops.shape(scaling_factors)[0]
        target_shape = [batch_size] + list(self._param_shape)
        return ops.reshape(scaling_factors, target_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Apply zero-centered adaptive RMS normalization."""
        original_dtype = inputs.dtype

        # Cast to fp32 internally for numerical stability under mixed precision.
        inputs_fp32 = ops.cast(inputs, "float32")

        # Step 1: Zero-centering.
        mean = ops.mean(inputs_fp32, axis=self.axis, keepdims=True)
        centered = inputs_fp32 - mean

        # Step 2: RMS of centered inputs.
        mean_square = ops.mean(
            ops.square(centered), axis=self.axis, keepdims=True
        )
        rms = ops.maximum(
            ops.sqrt(mean_square + self.epsilon), self.epsilon
        )

        # Step 3: Normalize.
        normalized = centered / rms

        # Step 4: Aggregate RMS -> log -> Dense -> sigmoid -> band scale.
        rms_stats = self._aggregate_rms_statistics(rms)
        log_rms = ops.log(rms_stats)
        band_logits = self.dense_layer(log_rms, training=training)

        band_activation = ops.sigmoid(5.0 * band_logits)
        scale_factors = (1.0 - self.max_band_width) + (
            self.max_band_width * band_activation
        )

        # Step 5: Reshape and apply adaptive scaling.
        scale_factors = self._reshape_scaling_factors(scale_factors)
        output = normalized * ops.cast(scale_factors, "float32")

        return ops.cast(output, original_dtype)

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Output shape equals input shape."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update(
            {
                "max_band_width": self.max_band_width,
                "axis": self.axis,
                "epsilon": self.epsilon,
                "band_initializer": keras.initializers.serialize(
                    self.band_initializer
                ),
                "band_regularizer": keras.regularizers.serialize(
                    self.band_regularizer
                ),
            }
        )
        return config


# ---------------------------------------------------------------------
