"""
PRISM: Partitioned Representation for Iterative Sequence Modeling.

This module implements the PRISM model for multivariate time series forecasting,
combining hierarchical time-frequency decomposition with forecasting capabilities.
Supports both point forecasts and probabilistic quantile forecasts.

**Overview**:

PRISM addresses the challenge that real-world time series contain global trends,
local fine-grained structure, and features on multiple scales in between. It builds
a learnable tree-based partitioning of the signal where the root captures coarse
trends while recursive splits reveal increasingly localized views.

**Architecture**:

The model cycles through four key stages:

1. **Time Decomposition**: Recursive bisection of the time series along the time
   axis into overlapping segments, forming a binary tree hierarchy. Each level
   produces 2^i segments where i is the tree depth.

2. **Frequency Decomposition**: At each node, a time-frequency transform (Haar
   DWT by default) decomposes segments into K frequency bands, extracting
   scale-specific features.

3. **Importance Weighting**: A lightweight MLP router computes importance scores
   for each frequency band based on summary statistics (mean, std, max amplitude,
   first/second derivatives). Scores are converted to weights via temperature-
   scaled softmax.

4. **Reconstruction**: Weighted frequency bands are combined and child node
   outputs are stitched together through linear cross-fade over overlap windows,
   yielding multiscale representations for forecasting.

**Key Design Principles**:

- Joint hierarchy in both time AND frequency domains (unlike prior work that
  operates in only one domain)
- Reconstructible design with overlap-based stitching for stability
- Data-driven importance scoring enables automatic focus on predictive components
- Decoupled temporal and frequency resolutions through learned weighting

**Quantile Forecasting**:

When ``use_quantile_head=True``, the model outputs probabilistic forecasts as
quantile predictions with optional monotonicity enforcement to prevent quantile
crossing (Q_i <= Q_{i+1}).

**Performance**:

PRISM achieves state-of-the-art results across standard benchmarks (ETT, Traffic,
Electricity, Exchange, Weather) and shows particular strength on irregular,
aperiodic, incomplete, nonstationary, and drifting time series data.

References:
    Chen, Z., Andre, A., Ma, W., Knight, I., Shuvaev, S., & Dyer, E. (2025).
    PRISM: A Hierarchical Multiscale Approach for Time Series Forecasting.
    arXiv:2512.24898.

    Code: https://github.com/nerdslab/prism
"""

import keras
from keras import initializers, regularizers, layers, ops
from typing import Dict, Any, Optional, Union, List, Tuple

# ---------------------------------------------------------------------
# Local Imports
# ---------------------------------------------------------------------

from dl_techniques.layers.ffn import create_ffn_layer
from dl_techniques.layers.time_series.prism_blocks import PRISMLayer

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class PRISMModel(keras.Model):
    """
    Complete PRISM model for time series forecasting.

    Combines hierarchical time-frequency decomposition with a forecasting
    head to predict future values of a time series. Supports both point
    forecasts and probabilistic quantile forecasts.

    **Architecture**::

        Input: context window [batch, context_len, num_features]
               ↓
        (optional) Input embedding: Linear -> hidden_dim
               ↓
        num_layers × PRISMLayer (stacked)
               ↓
        Flatten: [batch, context_len * hidden_dim]
               ↓
        Forecasting Head (Point or Quantile)
               ↓
        Point Mode Output: [batch, forecast_len, num_features]
        Quantile Mode Output: [batch, forecast_len, num_features, num_quantiles]

    **Quantile Mode**:
    When ``use_quantile_head=True``, the model outputs probabilistic forecasts
    as quantile predictions. Each output feature has ``num_quantiles`` predicted
    quantile values (e.g., 10th, 50th, 90th percentiles). Optional monotonicity
    enforcement prevents quantile crossing (Q_i <= Q_{i+1}).

    :param context_len: Length of input context window.
    :type context_len: int
    :param forecast_len: Length of forecast horizon.
    :type forecast_len: int
    :param num_features: Number of input/output features (channels).
    :type num_features: int
    :param hidden_dim: Hidden dimension for processing. If None, uses
        num_features. Defaults to None.
    :type hidden_dim: Optional[int]
    :param num_layers: Number of stacked PRISM layers.
        Defaults to 2.
    :type num_layers: int
    :param tree_depth: Depth of time tree in each PRISM layer.
        Defaults to 2.
    :type tree_depth: int
    :param overlap_ratio: Overlap ratio for segment splitting.
        Defaults to 0.25.
    :type overlap_ratio: float
    :param num_wavelet_levels: Number of Haar DWT levels.
        Defaults to 3.
    :type num_wavelet_levels: int
    :param router_hidden_dim: Hidden dimension for routers.
        Defaults to 64.
    :type router_hidden_dim: int
    :param router_temperature: Temperature for router softmax.
        Defaults to 1.0.
    :type router_temperature: float
    :param dropout_rate: Dropout rate.
        Defaults to 0.1.
    :type dropout_rate: float
    :param ffn_expansion: Expansion factor for forecasting head FFN.
        Defaults to 4.
    :type ffn_expansion: int
    :param use_quantile_head: Whether to use quantile prediction head instead
        of point forecast head. Defaults to False.
    :type use_quantile_head: bool
    :param num_quantiles: Number of quantiles to predict when using quantile
        head. Defaults to 3 (typically 10th, 50th, 90th percentiles).
    :type num_quantiles: int
    :param quantile_levels: Optional list of quantile levels (e.g., [0.1, 0.5, 0.9]).
        Used for documentation and loss function configuration. Length must match
        num_quantiles if provided. Defaults to None.
    :type quantile_levels: Optional[List[float]]
    :param enforce_monotonicity: Whether to enforce non-crossing quantiles
        (Q_i <= Q_{i+1}) via cumulative softplus. Only used when
        use_quantile_head=True. Defaults to True.
    :type enforce_monotonicity: bool
    :param kernel_initializer: Initializer for kernel weights.
        Defaults to "glorot_uniform".
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for kernel weights.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional arguments for the Model base class.
    """

    # Presets for common configurations
    PRESETS: Dict[str, Dict[str, Any]] = {
        "tiny": {
            "hidden_dim": 32,
            "num_layers": 1,
            "tree_depth": 1,
            "num_wavelet_levels": 2,
            "router_hidden_dim": 32,
            "ffn_expansion": 2,
        },
        "small": {
            "hidden_dim": 64,
            "num_layers": 2,
            "tree_depth": 2,
            "num_wavelet_levels": 3,
            "router_hidden_dim": 64,
            "ffn_expansion": 4,
        },
        "base": {
            "hidden_dim": 128,
            "num_layers": 3,
            "tree_depth": 2,
            "num_wavelet_levels": 3,
            "router_hidden_dim": 128,
            "ffn_expansion": 4,
        },
        "large": {
            "hidden_dim": 256,
            "num_layers": 4,
            "tree_depth": 2,
            "num_wavelet_levels": 4,
            "router_hidden_dim": 256,
            "ffn_expansion": 4,
        },
    }

    def __init__(
        self,
        context_len: int,
        forecast_len: int,
        num_features: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        tree_depth: int = 2,
        overlap_ratio: float = 0.25,
        num_wavelet_levels: int = 3,
        router_hidden_dim: int = 64,
        router_temperature: float = 1.0,
        dropout_rate: float = 0.1,
        ffn_expansion: int = 4,
        use_quantile_head: bool = False,
        num_quantiles: int = 3,
        quantile_levels: Optional[List[float]] = None,
        enforce_monotonicity: bool = True,
        kernel_initializer: Union[str, initializers.Initializer] = "glorot_uniform",
        kernel_regularizer: Optional[regularizers.Regularizer] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if context_len <= 0:
            raise ValueError(f"context_len must be > 0, got {context_len}")
        if forecast_len <= 0:
            raise ValueError(f"forecast_len must be > 0, got {forecast_len}")
        if num_features <= 0:
            raise ValueError(f"num_features must be > 0, got {num_features}")
        if num_quantiles <= 0:
            raise ValueError(f"num_quantiles must be > 0, got {num_quantiles}")
        if quantile_levels is not None and len(quantile_levels) != num_quantiles:
            raise ValueError(
                f"quantile_levels length ({len(quantile_levels)}) must match "
                f"num_quantiles ({num_quantiles})"
            )

        self.context_len = context_len
        self.forecast_len = forecast_len
        self.num_features = num_features
        self.hidden_dim = hidden_dim if hidden_dim is not None else num_features
        self.num_layers = num_layers
        self.tree_depth = tree_depth
        self.overlap_ratio = overlap_ratio
        self.num_wavelet_levels = num_wavelet_levels
        self.router_hidden_dim = router_hidden_dim
        self.router_temperature = router_temperature
        self.dropout_rate = dropout_rate
        self.ffn_expansion = ffn_expansion
        self.use_quantile_head = use_quantile_head
        self.num_quantiles = num_quantiles
        self.quantile_levels = quantile_levels
        self.enforce_monotonicity = enforce_monotonicity
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

        # Input projection (if hidden_dim != num_features)
        self.input_projection = layers.Dense(
            self.hidden_dim,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name="input_projection"
        )

        # Stacked PRISM layers
        self.prism_layers: List[PRISMLayer] = []
        for i in range(num_layers):
            layer = PRISMLayer(
                tree_depth=tree_depth,
                overlap_ratio=overlap_ratio,
                num_wavelet_levels=num_wavelet_levels,
                router_hidden_dim=router_hidden_dim,
                router_temperature=router_temperature,
                dropout_rate=dropout_rate,
                use_residual=True,
                use_output_norm=True,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name=f"prism_layer_{i}"
            )
            self.prism_layers.append(layer)

        # Flatten layer
        self.flatten = layers.Flatten(name="flatten")

        # Head dropout (shared)
        if self.dropout_rate > 0.0:
            self.head_dropout = layers.Dropout(
                rate=dropout_rate,
                name="head_dropout"
            )
        else:
            self.head_dropout = None

        # Forecasting head - point or quantile
        head_hidden_dim = self.hidden_dim * ffn_expansion

        if use_quantile_head:
            # Quantile head: outputs [batch, forecast_len, num_features, num_quantiles]
            # Total output units = forecast_len * num_features * num_quantiles
            output_units = forecast_len * num_features * num_quantiles
            self.forecast_head = create_ffn_layer(
                "mlp",
                hidden_dim=head_hidden_dim,
                output_dim=output_units,
                activation="gelu",
                dropout_rate=0.0,  # Dropout applied separately
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name="quantile_forecast_head"
            )
        else:
            # Point forecast head: outputs [batch, forecast_len, num_features]
            self.forecast_head = create_ffn_layer(
                "mlp",
                hidden_dim=head_hidden_dim,
                output_dim=forecast_len * num_features,
                activation="gelu",
                dropout_rate=dropout_rate,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name="forecast_head"
            )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build all model components.

        :param input_shape: Input shape tuple.
        :type input_shape: Tuple[Optional[int], ...]
        """
        batch_size = input_shape[0]

        # Build input projection
        self.input_projection.build(input_shape)

        # Projected shape
        projected_shape = (batch_size, self.context_len, self.hidden_dim)

        # Build PRISM layers
        current_shape = projected_shape
        for layer in self.prism_layers:
            layer.build(current_shape)

        # Build flatten
        self.flatten.build(current_shape)

        # Build head dropout
        flat_dim = self.context_len * self.hidden_dim
        if self.head_dropout is not None:
            self.head_dropout.build((batch_size, flat_dim))

        # Build forecast head
        self.forecast_head.build((batch_size, flat_dim))

        super().build(input_shape)

    def _apply_monotonicity(
        self,
        quantiles: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Apply monotonicity constraint to quantile predictions.

        Ensures Q_i <= Q_{i+1} by predicting the first quantile directly
        and subsequent quantiles as cumulative positive deltas.

        :param quantiles: Raw quantile predictions of shape
            [batch, forecast_len, num_features, num_quantiles].
        :type quantiles: keras.KerasTensor
        :return: Monotonic quantile predictions.
        :rtype: keras.KerasTensor
        """
        if self.num_quantiles <= 1:
            return quantiles

        # Split first quantile from rest
        # q0: [batch, forecast_len, num_features, 1]
        q0 = quantiles[:, :, :, 0:1]

        # rest: [batch, forecast_len, num_features, num_quantiles - 1]
        rest = quantiles[:, :, :, 1:]

        # Force deltas to be positive using softplus
        deltas = ops.softplus(rest)

        # Accumulate deltas along quantile dimension
        accumulated_deltas = ops.cumsum(deltas, axis=-1)

        # Subsequent quantiles = base + accumulated deltas
        subsequent_quantiles = q0 + accumulated_deltas

        # Concatenate [q0, q0 + delta1, q0 + delta1 + delta2, ...]
        return ops.concatenate([q0, subsequent_quantiles], axis=-1)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Generate forecasts from context window.

        :param inputs: Input tensor of shape [batch, context_len, num_features].
        :type inputs: keras.KerasTensor
        :param training: Training mode flag.
        :type training: Optional[bool]
        :return: Point forecast of shape [batch, forecast_len, num_features] or
            quantile forecast of shape [batch, forecast_len, num_features, num_quantiles].
        :rtype: keras.KerasTensor
        """
        # Project input to hidden dimension
        x = self.input_projection(inputs)

        # Process through PRISM layers
        for layer in self.prism_layers:
            x = layer(x, training=training)

        # Flatten
        x = self.flatten(x)

        # Apply head dropout (for quantile mode, point mode has internal dropout)
        if self.use_quantile_head and self.head_dropout is not None:
            x = self.head_dropout(x, training=training)

        # Generate forecast
        x = self.forecast_head(x, training=training)

        if self.use_quantile_head:
            # Reshape to [batch, forecast_len, num_features, num_quantiles]
            x = ops.reshape(
                x,
                (-1, self.forecast_len, self.num_features, self.num_quantiles)
            )

            # Apply monotonicity constraint if enabled
            if self.enforce_monotonicity:
                x = self._apply_monotonicity(x)
        else:
            # Reshape to [batch, forecast_len, num_features]
            x = ops.reshape(x, (-1, self.forecast_len, self.num_features))

        return x

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute output shape.

        :param input_shape: Input shape tuple.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple.
        :rtype: Tuple[Optional[int], ...]
        """
        batch_size = input_shape[0]
        if self.use_quantile_head:
            return (batch_size, self.forecast_len, self.num_features, self.num_quantiles)
        return (batch_size, self.forecast_len, self.num_features)

    @classmethod
    def from_preset(
        cls,
        preset: str,
        context_len: int,
        forecast_len: int,
        num_features: int,
        **kwargs: Any
    ) -> "PRISMModel":
        """
        Create model from a predefined preset.

        :param preset: Preset name ("tiny", "small", "base", "large").
        :type preset: str
        :param context_len: Length of input context window.
        :type context_len: int
        :param forecast_len: Length of forecast horizon.
        :type forecast_len: int
        :param num_features: Number of input/output features.
        :type num_features: int
        :param kwargs: Override parameters from preset.
        :type kwargs: Any
        :return: Configured model instance.
        :rtype: PRISMModel
        """
        if preset not in cls.PRESETS:
            raise ValueError(
                f"Unknown preset '{preset}'. "
                f"Available: {list(cls.PRESETS.keys())}"
            )

        config = cls.PRESETS[preset].copy()
        config.update(kwargs)

        return cls(
            context_len=context_len,
            forecast_len=forecast_len,
            num_features=num_features,
            **config
        )

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            "context_len": self.context_len,
            "forecast_len": self.forecast_len,
            "num_features": self.num_features,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "tree_depth": self.tree_depth,
            "overlap_ratio": self.overlap_ratio,
            "num_wavelet_levels": self.num_wavelet_levels,
            "router_hidden_dim": self.router_hidden_dim,
            "router_temperature": self.router_temperature,
            "dropout_rate": self.dropout_rate,
            "ffn_expansion": self.ffn_expansion,
            "use_quantile_head": self.use_quantile_head,
            "num_quantiles": self.num_quantiles,
            "quantile_levels": self.quantile_levels,
            "enforce_monotonicity": self.enforce_monotonicity,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "PRISMModel":
        """
        Create model from configuration dictionary.

        :param config: Configuration dictionary.
        :type config: Dict[str, Any]
        :return: Model instance.
        :rtype: PRISMModel
        """
        # Deserialize initializers and regularizers
        config = config.copy()
        if "kernel_initializer" in config:
            config["kernel_initializer"] = initializers.deserialize(
                config["kernel_initializer"]
            )
        if "kernel_regularizer" in config:
            config["kernel_regularizer"] = regularizers.deserialize(
                config["kernel_regularizer"]
            )
        return cls(**config)