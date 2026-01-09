"""
PRISM: Partitioned Representation for Iterative Sequence Modeling.

This module implements the PRISM model for multivariate time series forecasting,
combining hierarchical time-frequency decomposition with forecasting capabilities.
Supports both point forecasts and probabilistic quantile forecasts using a
standardized QuantileHead.

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

- Joint hierarchy in both time AND frequency domains.
- Reconstructible design with overlap-based stitching for stability.
- Data-driven importance scoring enables automatic focus on predictive components.
- Decoupled temporal and frequency resolutions through learned weighting.
- **Efficient Decoding**: Uses a channel-independent temporal projection strategy
  (similar to DLinear) to map context steps to forecast steps, drastically reducing
  parameter count compared to flattened dense projections.

**Quantile Forecasting**:

When ``use_quantile_head=True``, the model employs a dedicated ``QuantileHead``
to output probabilistic forecasts. This ensures monotonicity (Q_i <= Q_{i+1})
and provides a standardized interface for confidence intervals.

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
import numpy as np
from keras import initializers, regularizers, layers, ops
from typing import Dict, Any, Optional, Union, List, Tuple

# ---------------------------------------------------------------------
# Local Imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.ffn import create_ffn_layer
from dl_techniques.layers.time_series.prism_blocks import PRISMLayer
from dl_techniques.layers.time_series.quantile_head_fixed_io import QuantileHead

# ---------------------------------------------------------------------

# Default quantile levels for probabilistic forecasting
DEFAULT_QUANTILES: List[float] = [0.1, 0.5, 0.9]


@keras.saving.register_keras_serializable()
class PRISMModel(keras.Model):
    """
    Complete PRISM model for time series forecasting.

    Combines hierarchical time-frequency decomposition with a forecasting
    head to predict future values of a time series. Supports both point
    forecasts and probabilistic quantile forecasts via ``QuantileHead``.

    **Architecture**:
    ```
    Input [Batch, ContextLen, Features]
           ↓
    Input Projection [Batch, ContextLen, Hidden]
           ↓
    N × PRISM Layers (Hierarchical Decomp + Stitching)
           ↓
    Latent Representation [Batch, ContextLen, Hidden]
           ↓
    ┌────────────────────────────────────────────────────────┐
    │ Efficient Temporal Decoding (DLinear Style)            │
    │ 1. Transpose to [Batch, Hidden, ContextLen]            │
    │ 2. Shared Linear Projection: ContextLen → ForecastLen  │
    │ 3. Transpose to [Batch, ForecastLen, Hidden]           │
    └────────────────────────────────────────────────────────┘
           ↓
    Reshape [Batch * ForecastLen, Hidden]
           ↓
    Forecast Head (Shared across time steps)
           ↓
    Reshape [Batch, ForecastLen, Features, (Quantiles)]
    ```

    **Quantile Mode**:
    When ``use_quantile_head=True``, the model outputs probabilistic forecasts
    as quantile predictions. Monotonicity enforcement prevents quantile
    crossing (Q_i <= Q_{i+1}).

    **Parameter Efficiency**:
    Unlike naive approaches that flatten the entire time sequence (producing
    massive Dense layers), this model uses channel-independent temporal
    projection. This reduces the head parameters from O(T_in * T_out * Hidden)
    to O(T_in * T_out + Hidden * Features * Quantiles).

    Args:
        context_len: Length of input context window.
        forecast_len: Length of forecast horizon.
        num_features: Number of input/output features (channels).
        hidden_dim: Hidden dimension for processing. If None, uses num_features.
        num_layers: Number of stacked PRISM layers. Defaults to 2.
        tree_depth: Depth of time tree in each PRISM layer. Defaults to 2.
        overlap_ratio: Overlap ratio for segment splitting. Defaults to 0.25.
        num_wavelet_levels: Number of Haar DWT levels. Defaults to 3.
        router_hidden_dim: Hidden dimension for routers. Defaults to 64.
        router_temperature: Temperature for router softmax. Defaults to 1.0.
        dropout_rate: Dropout rate. Defaults to 0.1.
        ffn_expansion: Expansion factor for forecasting head FFN. Defaults to 4.
        use_quantile_head: Whether to use quantile prediction head instead
            of point forecast head. Defaults to False.
        num_quantiles: Number of quantiles to predict when using quantile
            head. Defaults to 3 (typically 10th, 50th, 90th percentiles).
        quantile_levels: Optional list of quantile levels (e.g., [0.1, 0.5, 0.9]).
            Used for documentation and API responses. Length must match num_quantiles.
            If None, generates linear space.
        enforce_monotonicity: Whether to enforce non-crossing quantiles
            (Q_i <= Q_{i+1}). Only used when use_quantile_head=True. Defaults to True.
        kernel_initializer: Initializer for kernel weights. Defaults to "glorot_uniform".
        kernel_regularizer: Optional regularizer for kernel weights.
        **kwargs: Additional arguments for the Model base class.
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

        # ---------------------------------------------------------------------
        # 1. Validation
        # ---------------------------------------------------------------------
        if context_len <= 0:
            raise ValueError(f"context_len must be > 0, got {context_len}")
        if forecast_len <= 0:
            raise ValueError(f"forecast_len must be > 0, got {forecast_len}")
        if num_features <= 0:
            raise ValueError(f"num_features must be > 0, got {num_features}")
        if num_quantiles <= 0:
            raise ValueError(f"num_quantiles must be > 0, got {num_quantiles}")

        # Validate or generate quantile levels
        if quantile_levels is not None:
            if len(quantile_levels) != num_quantiles:
                raise ValueError(
                    f"quantile_levels length ({len(quantile_levels)}) must match "
                    f"num_quantiles ({num_quantiles})"
                )
            self.quantile_levels = quantile_levels
        else:
            if use_quantile_head:
                # Generate roughly evenly spaced quantiles if not provided
                self.quantile_levels = list(
                    np.linspace(0, 1, num_quantiles + 2)[1:-1]
                )
            else:
                self.quantile_levels = None

        # ---------------------------------------------------------------------
        # 2. Store Config
        # ---------------------------------------------------------------------
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
        self.enforce_monotonicity = enforce_monotonicity
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

        # ---------------------------------------------------------------------
        # 3. Create Layers (Unconditionally)
        # ---------------------------------------------------------------------

        # Input projection
        self.input_projection = layers.Dense(
            self.hidden_dim,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name="input_projection"
        )

        # Stacked PRISM layers
        self.prism_layers = []
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

        # Efficient Temporal Projector (Shared across hidden dim)
        # We apply this to the time axis: Input ContextLen -> Output ForecastLen
        self.temporal_projector = layers.Dense(
            forecast_len,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name="temporal_projector"
        )

        # Head Dropout
        self.head_dropout = layers.Dropout(
            rate=dropout_rate,
            name="head_dropout"
        )

        # Forecasting Head (Applied per time-step)
        head_hidden_dim = self.hidden_dim * ffn_expansion

        if use_quantile_head:
            # Quantile Head: Projects Hidden -> NumFeatures * NumQuantiles
            # We set flatten_input=False to respect the input shape structure.
            # Output length is NumFeatures because we apply it per time step.
            self.forecast_head = QuantileHead(
                num_quantiles=self.num_quantiles,
                output_length=self.num_features,
                dropout_rate=0.0,  # Handled by head_dropout
                enforce_monotonicity=self.enforce_monotonicity,
                use_bias=True,
                flatten_input=False,
                name="quantile_forecast_head"
            )
        else:
            # Point Head: Projects Hidden -> NumFeatures
            self.forecast_head = create_ffn_layer(
                "mlp",
                hidden_dim=head_hidden_dim,
                output_dim=self.num_features,
                activation="gelu",
                dropout_rate=0.0,  # Handled by head_dropout
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name="point_forecast_head"
            )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build all model components.

        Args:
            input_shape: Input shape tuple.
        """
        batch_size = input_shape[0]

        # 1. Input Projection
        self.input_projection.build(input_shape)
        # Output: (Batch, ContextLen, Hidden)
        current_shape = (batch_size, self.context_len, self.hidden_dim)

        # 2. PRISM Layers
        for layer in self.prism_layers:
            layer.build(current_shape)

        # 3. Temporal Projector
        # Logic: Transpose (B, T, H) -> (B, H, T). Dense acts on T.
        transposed_shape = (batch_size, self.hidden_dim, self.context_len)
        self.temporal_projector.build(transposed_shape)
        # Output after Dense: (Batch, Hidden, ForecastLen)

        # 4. Head Dropout & Forecast Head
        # Logic: We flatten Batch and Time dimensions to reuse the head per step
        # Input to Head: (Batch * ForecastLen, Hidden)
        # Note: We use None for the batch dimension size during build
        collapsed_shape = (None, self.hidden_dim)

        self.head_dropout.build(collapsed_shape)
        self.forecast_head.build(collapsed_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Generate forecasts from context window.

        Args:
            inputs: Input tensor of shape [batch, context_len, num_features].
            training: Training mode flag.

        Returns:
            Point forecast [batch, forecast_len, num_features] or
            Quantile forecast [batch, forecast_len, num_features, num_quantiles].
        """
        # 1. Project to Latent Space
        # Shape: [Batch, ContextLen, Hidden]
        x = self.input_projection(inputs)

        # 2. Process Hierarchical Features
        for layer in self.prism_layers:
            x = layer(x, training=training)

        # 3. Efficient Temporal Projection (Channel-Independent)
        # Transpose to [Batch, Hidden, ContextLen]
        x = ops.transpose(x, axes=(0, 2, 1))

        # Project Time Dimension: ContextLen -> ForecastLen
        # Dense acts on the last dimension (ContextLen)
        # Shape: [Batch, Hidden, ForecastLen]
        x = self.temporal_projector(x)

        # Transpose back to [Batch, ForecastLen, Hidden]
        x = ops.transpose(x, axes=(0, 2, 1))

        # 4. Collapse dimensions for Head Application
        # We merge Batch and ForecastLen to treat every time step as an independent sample
        # Shape: [Batch * ForecastLen, Hidden]
        x = ops.reshape(x, (-1, self.hidden_dim))

        # 5. Decode to Output Features/Quantiles
        x = self.head_dropout(x, training=training)

        # Shape: [Batch * ForecastLen, OutputDim]
        # For Point: OutputDim = NumFeatures
        # For Quantile: OutputDim = NumFeatures * NumQuantiles (handled by Head)
        x = self.forecast_head(x, training=training)

        # 6. Final Reshaping
        # We restore the Batch and ForecastLen dimensions
        if self.use_quantile_head:
            # QuantileHead outputs flattened features+quantiles or reshaped
            # We explicitly enforce the desired 4D shape
            x = ops.reshape(
                x,
                (-1, self.forecast_len, self.num_features, self.num_quantiles)
            )
        else:
            # Ensure shape is [Batch, ForecastLen, NumFeatures]
            x = ops.reshape(x, (-1, self.forecast_len, self.num_features))

        return x

    def predict_quantiles(
            self,
            context: Union[np.ndarray, keras.utils.PyDataset],
            quantile_levels: Optional[List[float]] = None,
            batch_size: int = 32,
            **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate specific quantile and point forecasts for time series data.

        This acts as a wrapper around `model.predict()`, mapping requested
        quantiles to output indices and extracting the median as a point forecast.

        Args:
            context: Input data array or dataset.
            quantile_levels: List of floats (e.g., [0.1, 0.5, 0.9]). If None,
                returns all trained quantiles.
            batch_size: Batch size for inference.
            **kwargs: Arguments passed to `model.predict()`.

        Returns:
            Tuple (quantile_preds, point_preds):
            - quantile_preds: [Batch, ForecastLen, Features, RequestedQuantiles]
            - point_preds: [Batch, ForecastLen, Features] (Median)
        """
        if not self.use_quantile_head:
            raise ValueError(
                "Model was not initialized with use_quantile_head=True."
            )

        # Handle Quantile Levels
        if quantile_levels is None:
            quantile_levels = self.quantile_levels

        if not self.quantile_levels:
            # Fallback safety
            self.quantile_levels = list(
                np.linspace(0, 1, self.num_quantiles + 2)[1:-1]
            )

        # Run Inference
        # Output: [Batch, ForecastLen, Features, TrainedQuantiles]
        raw_predictions = self.predict(context, batch_size=batch_size, **kwargs)

        # Map Requested Levels to Indices
        quantile_indices = []
        trained_quantiles_arr = np.array(self.quantile_levels)

        for q in quantile_levels:
            if q in self.quantile_levels:
                idx = self.quantile_levels.index(q)
            else:
                idx = int(np.argmin(np.abs(trained_quantiles_arr - q)))
                logger.warning(
                    f"Requested quantile {q} not found. Using closest: "
                    f"{self.quantile_levels[idx]}"
                )
            quantile_indices.append(idx)

        # Extract Quantiles
        quantile_preds = raw_predictions[:, :, :, quantile_indices]

        # Extract Median (Point Forecast)
        if 0.5 in self.quantile_levels:
            median_idx = self.quantile_levels.index(0.5)
        else:
            median_idx = len(self.quantile_levels) // 2

        mean_preds = raw_predictions[:, :, :, median_idx]

        return quantile_preds, mean_preds

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape based on configuration."""
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
        """Create model from a predefined preset ('tiny', 'small', 'base', 'large')."""
        if preset not in cls.PRESETS:
            raise ValueError(
                f"Unknown preset '{preset}'. Available: {list(cls.PRESETS.keys())}"
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
        """Create model from configuration."""
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