"""
Adaptive BandRMS Layer: RMS Normalization with Log-Transformed RMS-Statistics-Based Scaling.

This layer implements an advanced normalization technique that extends Root Mean
Square Normalization (RMSNorm) by using logarithmically transformed RMS statistics
to dynamically compute scaling factors. This creates adaptive "thick spherical shell"
constraints based on the magnitude characteristics of the input data.

The layer operates in a four-step process:

1.  **RMS Normalization:**
    First, it applies standard RMS Normalization. For an input vector `x` of
    dimension `D`, this step computes `x_norm = x / sqrt(mean(x²) + ε)` and
    retains the RMS statistics for further processing.

2.  **Logarithmic Transformation:**
    The computed RMS statistics undergo a logarithmic transformation to stabilize
    variance and handle the long-tailed nature of magnitude distributions.

3.  **Dense Projection:**
    The log-transformed RMS statistics are passed through a dense layer to compute
    adaptive scaling parameters that depend on the input's magnitude characteristics.

4.  **Adaptive Band Scaling:**
    The scaling parameters are constrained to [1 - max_band_width, 1] using sigmoid
    activation and applied to the RMS-normalized features.

References:
[1] Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)
    https://arxiv.org/abs/1910.07467
"""

import keras
from typing import Any, Dict, Optional, Union, Tuple, List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class AdaptiveBandRMS(keras.layers.Layer):
    """
    Adaptive Root Mean Square Normalization with log-transformed RMS scaling.

    This layer implements advanced RMS normalization where scaling factors are computed
    from logarithmically transformed RMS statistics. The log transformation stabilizes
    variance and creates more symmetric distributions for adaptive scaling computation.

    The intent is to provide input-adaptive normalization that creates "thick spherical
    shells" in RMS space, where shell thickness adapts based on input magnitude
    characteristics with improved numerical stability through log-transformed statistics.

    This robust version supports arbitrary tensor shapes (2D, 3D, 4D, etc.) with
    flexible axis configurations, making it suitable for dense layers, sequence models,
    convolutions, and any tensor processing scenarios.

    **Mathematical Operations:**

    1. **RMS computation**: rms = sqrt(E[x^2] + epsilon)
    2. **Normalization**: x_hat = x / rms
    3. **Log transform**: log_rms = log(rms_aggregate)
    4. **Adaptive scaling**: y = x_hat * scale(log_rms)

    Where scale(.) is in [1-alpha, 1] computed via dense projection and sigmoid activation.

    .. note::
        **No masking support, deliberately.** ``supports_masking`` is left ``False``
        because ``_aggregate_rms_statistics`` means the RMS over every non-batch axis
        before feeding the internal ``Dense``, so one sigmoid rescales all of a
        sample's positions together: perturbing a single ``(sample, token)`` slot
        moves the other positions of that sample by up to ``1.621e-01`` (measured on
        a ``(3, 5, 8)`` input). Propagating a Keras mask would advertise
        padding-independent outputs that were in fact computed from the padding.

    **Architecture Overview:**

    .. code-block:: text

        Input: x (batch, ..., features)
                │
                ▼
        ┌───────────────────────────────────┐
        │  RMS = max(√(mean(x²)+ε), ε)      │
        └──────────┬────────────────────────┘
                   │
                   ├──────────────────────┐
                   ▼                      ▼
        ┌──────────────────┐   ┌──────────────────────┐
        │ x_norm = x / RMS │   │ Aggregate RMS stats  │
        └────────┬─────────┘   │ to [batch, 1]        │
                 │             └──────────┬───────────┘
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
        Output: (batch, ..., features)

    :param max_band_width: Maximum allowed deviation from unit normalization
        (0 < alpha < 1). Controls the thickness of the adaptive spherical shell.
        Smaller values create tighter constraints.
    :type max_band_width: float
    :param axis: Axes along which to compute RMS statistics. Similar to
        keras.layers.LayerNormalization: for 2D (batch, features) use axis=-1;
        for 3D (batch, seq, features) use axis=-1 or axis=1; for 4D
        (batch, H, W, channels) use axis=-1 or axis=(1,2).
    :type axis: Union[int, Tuple[int, ...]]
    :param epsilon: Small positive constant added to denominator for numerical
        stability.
    :type epsilon: float
    :param band_initializer: Initializer for the dense kernel computing the scaling
        parameters. Defaults to ``'zeros'``, which does NOT start the layer near
        unit scaling: a zero kernel and zero bias make the dense output zero, and
        ``sigmoid(0) = 0.5``, so the initial scale is
        ``(1 - max_band_width) + max_band_width * 0.5 = 1 - max_band_width / 2`` -
        the MIDPOINT of the band, uniformly for every element. At the default
        ``max_band_width=0.1`` that is a measured initial scale of exactly 0.95, not
        1.0. Pass a large-positive-bias initializer if you want to start near the
        band's upper edge.
    :type band_initializer: Union[str, keras.initializers.Initializer]
    :param band_regularizer: Optional regularizer for the dense layer weights.
        Can help prevent overfitting of adaptive scaling.
    :type band_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional arguments for Layer base class (name, trainable, etc.).

    :raises ValueError: If max_band_width is not between 0 and 1.
    :raises ValueError: If epsilon is not positive.
    :raises ValueError: If axis configuration includes batch dimension (axis 0).
    :raises ValueError: If axis is out of bounds for input tensor rank.
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
        super().__init__(**kwargs)

        # Validate inputs early
        self._validate_inputs(max_band_width, axis, epsilon)

        # Store ALL configuration - required for get_config()
        self.max_band_width = max_band_width
        self.axis = axis
        self.epsilon = epsilon
        self.band_initializer = keras.initializers.get(band_initializer)
        self.band_regularizer = keras.regularizers.get(band_regularizer)

        # Shape computation results (set in build)
        self._param_shape = None
        self._scaling_axes = None

        # Sub-layer created in build() due to dynamic architecture
        # Note: Cannot follow standard pattern due to input-shape dependency
        self.dense_layer = None

        logger.debug(
            f"Initialized AdaptiveBandRMS: "
            f"max_band_width={max_band_width}, "
            f"axis={axis}, "
            f"epsilon={epsilon}"
        )

    def _validate_inputs(
        self,
        max_band_width: float,
        axis: Union[int, Tuple[int, ...], List[int]],
        epsilon: float,
    ) -> None:
        """
        Validate initialization parameters.

        :param max_band_width: Maximum deviation from unit normalization.
        :type max_band_width: float
        :param axis: Axis/axes over which to normalize.
        :type axis: Union[int, Tuple[int, ...], List[int]]
        :param epsilon: Numerical stability constant.
        :type epsilon: float
        :raises ValueError: If max_band_width or epsilon is invalid.
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

    def _compute_param_shape_and_axes(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Tuple[int, ...], List[int]]:
        """
        Compute parameter shape and scaling axes based on input shape and normalization axes.

        Determines what dimensions should have independent scaling parameters,
        similar to how LayerNormalization computes parameter shapes.

        :param input_shape: Shape of input tensor including batch dimension.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Tuple of (param_shape, scaling_axes) where param_shape is the
            shape for scaling parameters (without batch dim) and scaling_axes is
            the list of axes that will be scaled independently.
        :rtype: Tuple[Tuple[int, ...], List[int]]
        :raises ValueError: If axis configuration is invalid.
        """
        input_rank = len(input_shape)

        # Normalize axis to positive integers
        if isinstance(self.axis, int):
            axes = [self.axis]
        else:
            axes = list(self.axis)

        # Convert negative indices and validate
        normalized_axes = []
        for ax in axes:
            if ax < 0:
                ax = input_rank + ax
            if ax == 0:
                raise ValueError(
                    "axis 0 (batch dimension) cannot be normalized"
                )
            if ax < 0 or ax >= input_rank:
                raise ValueError(
                    f"axis {ax} is out of bounds for input with {input_rank} dimensions"
                )
            normalized_axes.append(ax)

        # Remove duplicates and sort
        normalized_axes = sorted(set(normalized_axes))

        # Check for global normalization (all non-batch axes)
        is_global = (
            input_rank > 2 and
            len(normalized_axes) == input_rank - 1
        )

        if is_global:
            # Global normalization: single parameter broadcasts everywhere
            param_shape = [1] * (input_rank - 1)
            scaling_axes = []
            return tuple(param_shape), scaling_axes

        # Compute parameter shape for non-global cases
        param_shape = []
        scaling_axes = []

        for i in range(1, input_rank):  # Skip batch dimension
            if i in normalized_axes:
                if input_shape[i] is None:
                    raise ValueError(
                        f"Normalized axis {i} has an undefined (None) size; "
                        f"AdaptiveBandRMS needs a static dimension to size its "
                        f"scaling Dense layer. Got input_shape={input_shape}."
                    )
                param_shape.append(input_shape[i])
                scaling_axes.append(i)
            else:
                param_shape.append(1)

        return tuple(param_shape), scaling_axes

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and create sub-layers with proper parameter sizing.

        This layer uses dynamic sub-layer creation because the dense layer's
        output size depends on input shape and normalization axes. This deviates
        from the standard Keras pattern where sub-layers are created in __init__(),
        but is necessary for proper parameter shape computation.

        :param input_shape: Shape tuple indicating input tensor shape.
        :type input_shape: Tuple[Optional[int], ...]
        """
        if self.built:
            return

        # Compute parameter configuration
        self._param_shape, self._scaling_axes = self._compute_param_shape_and_axes(
            input_shape
        )

        # Calculate number of scaling parameters needed
        num_params = 1
        for dim in self._param_shape:
            if dim is not None:
                num_params *= dim

        # Create dense layer for log-RMS to scaling-parameter projection
        self.dense_layer = keras.layers.Dense(
            units=num_params,
            kernel_initializer=self.band_initializer,
            kernel_regularizer=self.band_regularizer,
            use_bias=True,
            name="band_dense"
        )

        # Build the dense layer explicitly
        # Input is always [batch, 1] for aggregated log-RMS statistics
        log_rms_input_shape = (None, 1)
        self.dense_layer.build(log_rms_input_shape)

        logger.debug(
            f"Built AdaptiveBandRMS: param_shape={self._param_shape}, "
            f"num_params={num_params}, dense_input_shape={log_rms_input_shape}"
        )

        # Always call parent build at the end
        super().build(input_shape)

    def _aggregate_rms_statistics(
        self,
        rms_tensor: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Aggregate RMS statistics to single scalar per batch element.

        For the dense layer processing, we need one representative RMS value
        per batch element. This is computed by averaging all RMS values
        across non-batch dimensions.

        :param rms_tensor: RMS tensor with keepdims=True, same rank as input.
        :type rms_tensor: keras.KerasTensor
        :return: Aggregated RMS statistics with shape [batch, 1].
        :rtype: keras.KerasTensor
        """
        # Aggregate over all non-batch dimensions
        aggregation_axes = list(range(1, len(rms_tensor.shape)))

        if aggregation_axes:
            rms_stats = keras.ops.mean(rms_tensor, axis=aggregation_axes, keepdims=True)
        else:
            # Handle edge case of 1D input (batch,)
            rms_stats = rms_tensor

        # Ensure shape is [batch, 1] for dense layer
        return keras.ops.reshape(rms_stats, [-1, 1])

    def _reshape_scaling_factors(
        self,
        scaling_factors: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Reshape scaling factors from dense layer for proper broadcasting.

        :param scaling_factors: Output from dense layer, shape [batch, num_params].
        :type scaling_factors: keras.KerasTensor
        :return: Reshaped scaling factors ready for element-wise multiplication.
        :rtype: keras.KerasTensor
        """
        batch_size = keras.ops.shape(scaling_factors)[0]
        target_shape = [batch_size] + list(self._param_shape)
        return keras.ops.reshape(scaling_factors, target_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply adaptive RMS normalization with log-transformed statistics.

        :param inputs: Input tensor of arbitrary shape.
        :type inputs: keras.KerasTensor
        :param training: Boolean indicating training mode.
        :type training: Optional[bool]
        :return: Normalized tensor with adaptive RMS-based scaling.
        :rtype: keras.KerasTensor
        """
        # Store original dtype for casting back
        original_dtype = inputs.dtype

        # Statistics dtype: float32 at minimum (numerical stability under
        # mixed precision), but float64 when the layer really is float64 -
        # a hardcoded "float32" here silently ran the statistics in float32
        # under a float64 policy (measured: the output matched a float32
        # reference exactly and missed the float64 one by 2.6e-8). This also
        # feeds the internal Dense at the policy's dtype, so a float64 policy no
        # longer promotes a float32 tensor against a float64 kernel.
        stat_dtype = keras.backend.result_type(original_dtype, "float32")
        inputs_fp32 = keras.ops.cast(inputs, stat_dtype)

        # Step 1: Compute RMS for normalization
        mean_square = keras.ops.mean(
            keras.ops.square(inputs_fp32),
            axis=self.axis,
            keepdims=True
        )

        rms = keras.ops.maximum(
            keras.ops.sqrt(mean_square + self.epsilon),
            self.epsilon
        )

        # Apply standard RMS normalization
        normalized = inputs_fp32 / rms

        # Step 2: Aggregate RMS statistics for dense layer input
        rms_stats = self._aggregate_rms_statistics(rms)

        # Step 3: Logarithmic transformation for variance stabilization
        log_rms = keras.ops.log(rms_stats)

        # Step 4: Dense projection to compute adaptive scaling parameters
        band_logits = self.dense_layer(log_rms, training=training)

        # Step 5: Convert to scaling factors in [1-α, 1] range
        band_activation = keras.ops.sigmoid(5.0 * band_logits)
        scale_factors = (1.0 - self.max_band_width) + (
            self.max_band_width * band_activation
        )

        # Step 6: Reshape for broadcasting and apply adaptive scaling
        scale_factors = self._reshape_scaling_factors(scale_factors)
        # DECISION plan-2026-08-25T195813-d5a035ab/D-005: this cast is NOT redundant
        # with the one above and must not be deleted. `self.dense_layer` returns its
        # own COMPUTE dtype, which under mixed_float16 is float16 while `normalized`
        # is float32 - measured: `fp32 * fp16` raises
        # `InvalidArgumentError: cannot compute Mul as input #1 ... is a half tensor`.
        # The destination is `stat_dtype` rather than a hardcoded "float32" so a
        # float64 policy keeps the multiply in float64.
        output = normalized * keras.ops.cast(scale_factors, stat_dtype)

        # Cast back to original dtype
        return keras.ops.cast(output, original_dtype)

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute output shape (same as input).

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple (same as input).
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration for serialization.

        :return: Dictionary containing all constructor parameters needed for
            layer reconstruction.
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
