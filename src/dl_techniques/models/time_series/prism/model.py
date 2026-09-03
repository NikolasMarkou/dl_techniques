"""
PRISMModel and its factory create_prism_model, a hierarchical time-frequency
forecaster over a binary time tree with per-band routing, a
channel-independent temporal decoder, and point or monotone-quantile heads.

A real series carries structure at several time scales at once (a trend, a
weekly cycle, a short burst), and a model committed to one resolution blurs
the others. PRISM instead partitions the context window into a binary tree:
level i holds 2^i overlapping segments of increasingly local extent, and each
node runs a Haar wavelet transform to split its segment into frequency bands.
A small router reads six summary statistics per band (mean, std, min, max,
and the mean/std of the first difference) and weights bands by a
temperature-scaled softmax, so which band matters is decided per node from
data. tree_depth and num_wavelet_levels trade against each other and against
context_len through min_band_len; the constructor raises ValueError rather
than let a degenerate band reach the forward pass. The decoder transposes the
latent so a shared Dense acts on the time axis (context_len -> forecast_len)
rather than flattening the full sequence, which keeps the head parameter
count linear rather than quadratic in sequence length. In quantile mode the
output head enforces monotonicity by construction.

References:
    - Chen et al., 2025. PRISM: A Hierarchical Multiscale Approach for Time Series
      Forecasting. arXiv:2512.24898.
    - Zeng et al., 2022. Are Transformers Effective for Time Series Forecasting?
      (https://arxiv.org/abs/2205.13504)
    - Mallat, 1989. A Theory for Multiresolution Signal Decomposition: The Wavelet
      Representation. IEEE TPAMI 11(7): 674-693.
    - Koenker and Bassett, 1978. Regression Quantiles. Econometrica 46(1): 33-50.
"""

import keras
import numpy as np
from keras import initializers, regularizers, layers, ops
from typing import Dict, Any, Optional, Union, List, Tuple

from dl_techniques.utils.logger import logger
from dl_techniques.models.time_series.forecast import Forecast, ForecastMixin
from dl_techniques.layers.ffn import create_ffn_layer
from dl_techniques.layers.time_series.prism_blocks import PRISMLayer, PRISMTimeTree
from dl_techniques.layers.time_series.quantile_head_fixed_io import QuantileHead
from dl_techniques.utils.keras_registration import register_dl_technique


@register_dl_technique("dl_techniques.models.prism.model")
class PRISMModel(keras.Model, ForecastMixin):
    """Hierarchical multiscale (time + frequency) forecasting model.

    Maps a context window to point forecasts ``[B, H, F]`` or quantile
    forecasts ``[B, H, F, Q]``. Named architecture sizes are exposed through
    ``MODEL_VARIANTS`` and :meth:`from_variant`.

    Architecture:

    .. code-block:: text

        input [B, context_len, F]
             |
             v
        ┌──────────────┐
        │ input          │
        │ projection     │
        └──────────────┘
             |
             v
        ┌──────────────┐
        │ prism layer 1  │  hierarchical time-frequency decomposition
        │ ...            │
        │ prism layer N  │
        └──────────────┘
             |
             v
        latent [B, context_len, H]
             |
             v
        ┌──────────────┐
        │ transpose      │  [B, H, context_len]
        │ temporal dense │  context_len -> forecast_len, shared across H
        │ transpose      │  [B, forecast_len, H]
        └──────────────┘
             |
             v
        ┌──────────────┐
        │ head dropout   │
        │ forecast head  │  shared across every forecast step
        └──────────────┘
             |
             ├─ point forecast: [B, forecast_len, F]
             └─ quantile forecast: [B, forecast_len, F, Q]

    When ``use_quantile_head=True`` the head enforces monotonicity, so
    ``Q_i <= Q_{i+1}`` holds by construction rather than by penalty.

    :param context_len: Length of the input context window.
    :type context_len: int
    :param forecast_len: Length of the forecast horizon.
    :type forecast_len: int
    :param num_features: Number of input/output features (channels).
    :type num_features: int
    :param hidden_dim: Hidden dimension for processing. Uses ``num_features``
        if ``None``.
    :type hidden_dim: Optional[int]
    :param num_layers: Number of stacked PRISM layers.
    :type num_layers: int
    :param tree_depth: Depth of the time tree in each PRISM layer. Has no
        valid range on its own: together with ``context_len``,
        ``overlap_ratio`` and ``num_wavelet_levels`` it determines
        ``min_band_len``, and ``__init__`` raises ``ValueError`` when that
        reaches 0. Node count grows as ``2 ** tree_depth`` per layer.
    :type tree_depth: int
    :param overlap_ratio: Overlap ratio for segment splitting, in ``[0, 0.5)``.
    :type overlap_ratio: float
    :param num_wavelet_levels: Number of Haar DWT levels. Each level
        floor-halves the band length, trading against ``tree_depth`` and
        ``context_len`` through ``min_band_len``.
    :type num_wavelet_levels: int
    :param router_hidden_dim: Hidden dimension for the band routers.
    :type router_hidden_dim: int
    :param router_temperature: Temperature for the router softmax.
    :type router_temperature: float
    :param dropout_rate: Dropout rate.
    :type dropout_rate: float
    :param ffn_expansion: Expansion factor for the point forecast head's FFN.
    :type ffn_expansion: int
    :param use_quantile_head: Whether to use a quantile head instead of a
        point head.
    :type use_quantile_head: bool
    :param num_quantiles: Number of quantiles to predict when
        ``use_quantile_head`` is set.
    :type num_quantiles: int
    :param quantile_levels: Quantile levels (e.g. ``[0.1, 0.5, 0.9]``); must
        match ``num_quantiles`` in length. If ``None`` and ``num_quantiles``
        equals ``len(DEFAULT_QUANTILES)``, defaults to ``DEFAULT_QUANTILES``;
        otherwise falls back to evenly spaced interior levels.
    :type quantile_levels: Optional[List[float]]
    :param enforce_monotonicity: Whether to enforce non-crossing quantiles.
        Used only when ``use_quantile_head`` is set.
    :type enforce_monotonicity: bool
    :param kernel_initializer: Initializer for kernel weights.
    :type kernel_initializer: Union[str, initializers.Initializer]
    :param kernel_regularizer: Regularizer for kernel weights.
    :type kernel_regularizer: Optional[regularizers.Regularizer]
    :param kwargs: Additional arguments for the ``Model`` base class.
    """

    # Default quantile levels for probabilistic forecasting.
    DEFAULT_QUANTILES: List[float] = [0.1, 0.5, 0.9]

    # Named architecture variants (sizes) for common configurations.
    MODEL_VARIANTS: Dict[str, Dict[str, Any]] = {
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

        # DECISION plan-2026-08-18T073231-52a93f8c/D-011: keep these guards unconditional,
        # ahead of min_band_len; a negative overlap_ratio else hangs the search below. See decisions.md.
        if not 0 <= overlap_ratio < 0.5:
            raise ValueError(
                f"overlap_ratio must be in [0, 0.5), got {overlap_ratio}"
            )
        if tree_depth < 0:
            raise ValueError(f"tree_depth must be >= 0, got {tree_depth}")
        if num_wavelet_levels < 0:
            raise ValueError(
                f"num_wavelet_levels must be >= 0, got {num_wavelet_levels}"
            )

        # DECISION plan-2026-08-18T073231-52a93f8c/D-005: gate on min_band_len, not a
        # tree_depth range; depth alone can't separate working from broken configs (measured, 36-cell grid). See decisions.md.
        num_leaves = 2 ** tree_depth
        if num_leaves == 1:
            # A single segment: the split geometry formula does not apply.
            deepest_leaf_seg = context_len
        else:
            # Each level re-splits the full sequence, so the deepest leaf is
            # one application of the shared geometry at 2 ** tree_depth.
            _, _, deepest_leaf_seg = PRISMTimeTree._segment_len(
                context_len, overlap_ratio, num_leaves,
                dtype=self.compute_dtype
            )
        # Each Haar level floor-halves the band length.
        min_band_len = deepest_leaf_seg // (2 ** num_wavelet_levels)
        if min_band_len < 1:
            # Search for the smallest context_len that reaches min_band_len >= 1
            # at this (tree_depth, num_wavelet_levels, overlap_ratio) triple.
            _SEARCH_CAP = 1_000_000
            min_supported_context_len = None
            probe_ctx = context_len
            for _ in range(_SEARCH_CAP):
                probe_ctx += 1
                if num_leaves == 1:
                    probe_seg = probe_ctx
                else:
                    _, _, probe_seg = PRISMTimeTree._segment_len(
                        probe_ctx, overlap_ratio,
                        num_leaves, dtype=self.compute_dtype
                    )
                if probe_seg // (2 ** num_wavelet_levels) >= 1:
                    min_supported_context_len = probe_ctx
                    break
            # Only offer remedies that are real: at tree_depth 0 or
            # num_wavelet_levels 0 there is nothing left to lower.
            if min_supported_context_len is None:
                remedies = [
                    f"raise context_len (no supportable value found within "
                    f"{_SEARCH_CAP} steps of {context_len})"
                ]
            else:
                remedies = [
                    f"raise context_len (to >= {min_supported_context_len})"
                ]
            if tree_depth > 0:
                remedies.append(f"lower tree_depth (to <= {tree_depth - 1})")
            if num_wavelet_levels > 0:
                remedies.append(
                    f"lower num_wavelet_levels (to <= {num_wavelet_levels - 1})"
                )
            raise ValueError(
                f"unsupportable PRISM configuration: context_len="
                f"{context_len}, tree_depth={tree_depth}, "
                f"num_wavelet_levels={num_wavelet_levels}, overlap_ratio="
                f"{overlap_ratio} give deepest_leaf_seg={deepest_leaf_seg} "
                f"and min_band_len={min_band_len}, i.e. the deepest "
                f"frequency band carries no timesteps at all and its "
                f"statistics are undefined. Every band must have length "
                f">= 1. Remedies: " + ", or ".join(remedies) + "."
            )
        elif min_band_len == 1:
            # DECISION plan-2026-08-18T073231-52a93f8c/D-013: min_band_len == 1 is
            # allowed but degenerate (a shipped preset sits exactly there); warn once here, don't move to call() or raise. See decisions.md.
            logger.warning(
                f"PRISM configuration is at the degenerate boundary: "
                f"context_len={context_len}, tree_depth={tree_depth}, "
                f"num_wavelet_levels={num_wavelet_levels}, overlap_ratio="
                f"{overlap_ratio} give deepest_leaf_seg={deepest_leaf_seg} "
                f"and min_band_len=1, so the deepest frequency bands carry a "
                f"SINGLE timestep. Their statistics are degenerate (mean == "
                f"min == max == that one sample, and both first-difference "
                f"features are exactly 0.0 by definition rather than by "
                f"measurement), so those bands carry almost no information "
                f"into the router. This is supported, not an error. To avoid "
                f"it, raise context_len or lower tree_depth / "
                f"num_wavelet_levels."
            )

        if quantile_levels is not None:
            if len(quantile_levels) != num_quantiles:
                raise ValueError(
                    f"quantile_levels length ({len(quantile_levels)}) must match "
                    f"num_quantiles ({num_quantiles})"
                )
            self.quantile_levels = quantile_levels
        else:
            if use_quantile_head:
                # DECISION plan-2026-08-14T233721-d4f9beb2/D-075: DEFAULT_QUANTILES is the
                # default only at its own length; don't replace the linspace fallback with an unconditional DEFAULT_QUANTILES. See decisions.md.
                if num_quantiles == len(self.DEFAULT_QUANTILES):
                    self.quantile_levels = list(self.DEFAULT_QUANTILES)
                else:
                    # No canonical set at this length: fall back to evenly spaced levels.
                    self.quantile_levels = list(
                        np.linspace(0, 1, num_quantiles + 2)[1:-1]
                    )
            else:
                self.quantile_levels = None

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

        # DECISION plan-2026-08-18T073231-52a93f8c/D-012
        # This pins the TIME axis so a WRONG static length is refused at
        # ``__call__`` instead of reaching the tree with the wrong geometry --
        # ``context_len`` is a required constructor argument, so any other
        # static length is a caller error.
        # It does NOT close the dynamic-time-axis hole, and must not be cited
        # as if it did. MEASURED both before and after adding it: the fixed
        # ``tree_depth=3`` model traced as
        # ``tf.function(input_signature=[TensorSpec([None,None,7])])`` returns
        # ``nan_frac == 1.0``, while the same model returns ``0.0`` eager.
        # The reason is in Keras itself: ``assert_input_compatibility`` tests
        # ``shape[axis] not in {value, None}``
        # (``keras/src/layers/input_spec.py:223-226``), so an UNKNOWN dimension
        # is explicitly accepted by an ``axes`` constraint. An unknown time
        # axis therefore still reaches ``FrequencyBandStatistics.call``, whose
        # degenerate-band guard branches on the static length and deliberately
        # falls through when it is ``None`` (D-004) -- so under that regime the
        # original all-NaN defect is fully present.
        # Do NOT relax this to ``InputSpec(ndim=3)``; do NOT claim it closes
        # the dynamic case. See decisions.md D-012 and D-004.
        self.input_spec = keras.layers.InputSpec(
            ndim=3, axes={1: context_len}
        )

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
        # DECISION plan-2026-08-19T163559-499b6f0e/D-117: the median head is chosen by
        # VALUE (nearest level to 0.5), reusing `trained_quantiles_arr` built ten lines
        # above -- not by `0.5 in <list of floats>` with a `len(...) // 2` fallback.
        # When 0.5 IS present this is bit-identical to the old `.index(0.5)` (argmin of
        # an exact zero, first occurrence). When it is not, the old fallback took the
        # POSITIONAL middle, which is only the median for a level set that happens to be
        # symmetric; for an asymmetric caller-supplied set such as
        # [0.01, 0.05, 0.1, 0.48, 0.9] it returned the 0.1 head as the point forecast.
        # Do NOT reintroduce a float `in` test here.
        median_idx = int(np.argmin(np.abs(trained_quantiles_arr - 0.5)))

        mean_preds = raw_predictions[:, :, :, median_idx]

        return quantile_preds, mean_preds

    def _forecast(
            self,
            x: Union[np.ndarray, keras.utils.PyDataset],
            quantile_levels: Optional[List[float]] = None,
            **kwargs: Any
    ) -> Forecast:
        """Produce a unified :class:`Forecast` reusing the model's predict paths.

        This is the ``ForecastMixin`` hook. In quantile mode it delegates to
        ``predict_quantiles`` (no quantile-mapping reimplementation); in point
        mode it uses the model's normal point prediction path and emits
        ``quantiles=None`` (never fabricate intervals for a point model).

        Args:
            x: Context window, shape ``[B, context_len, F]`` (or a dataset).
            quantile_levels: Levels to extract in quantile mode; defaults to the
                model's configured ``self.quantile_levels``. Ignored in point mode.
            **kwargs: Forwarded to ``predict_quantiles``/``predict`` (e.g.
                ``batch_size``, ``verbose``).

        Returns:
            A :class:`Forecast`. Quantile mode: ``point`` ``[B, H, F]`` and
            ``quantiles`` ``[B, H, F, Q]``. Point mode: ``point`` ``[B, H, F]``
            with ``quantiles=None`` and ``quantile_levels=None``.
        """
        if self.use_quantile_head:
            levels = quantile_levels if quantile_levels is not None else self.quantile_levels
            quantile_preds, point_preds = self.predict_quantiles(x, levels, **kwargs)
            return Forecast(
                point=np.asarray(point_preds),
                quantiles=np.asarray(quantile_preds),
                quantile_levels=list(levels),
            )

        # Point mode: normal point prediction path; never fabricate intervals.
        point_preds = self.predict(x, **kwargs)
        return Forecast(
            point=np.asarray(point_preds),
            quantiles=None,
            quantile_levels=None,
        )

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
    def from_variant(
        cls,
        variant: str,
        context_len: int,
        forecast_len: int,
        num_features: int,
        **kwargs: Any
    ) -> "PRISMModel":
        """Create model from a predefined variant ('tiny', 'small', 'base', 'large')."""
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available: {list(cls.MODEL_VARIANTS.keys())}"
            )

        config = cls.MODEL_VARIANTS[variant].copy()
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


# ---------------------------------------------------------------------
# Backward-compatible module-level alias for the default quantile levels.
# Canonical source is ``PRISMModel.DEFAULT_QUANTILES``; this alias is kept so
# external code importing the module-level name keeps working.
# ---------------------------------------------------------------------

DEFAULT_QUANTILES: List[float] = PRISMModel.DEFAULT_QUANTILES


def create_prism_model(
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
    **kwargs: Any
) -> PRISMModel:
    """
    Create a PRISM model with the specified configuration.

    Mirrors the other time-series factories (e.g. ``create_tirex_model``): it
    constructs the model and runs a dummy forward pass so all sublayers are
    built and weights/shapes are initialized before the model is returned.

    Args:
        context_len: Length of input context window.
        forecast_len: Length of forecast horizon.
        num_features: Number of input/output features (channels).
        hidden_dim: Hidden dimension for processing. If None, uses num_features.
        num_layers: Number of stacked PRISM layers.
        tree_depth: Depth of time tree in each PRISM layer.
        overlap_ratio: Overlap ratio for segment splitting.
        num_wavelet_levels: Number of Haar DWT levels.
        router_hidden_dim: Hidden dimension for routers.
        router_temperature: Temperature for router softmax.
        dropout_rate: Dropout rate.
        ffn_expansion: Expansion factor for forecasting head FFN.
        use_quantile_head: Whether to use a quantile prediction head.
        num_quantiles: Number of quantiles to predict when using quantile head.
        quantile_levels: Optional list of quantile levels.
        enforce_monotonicity: Whether to enforce non-crossing quantiles.
        **kwargs: Additional arguments for :class:`PRISMModel`.

    Returns:
        A built :class:`PRISMModel` instance.
    """
    model = PRISMModel(
        context_len=context_len,
        forecast_len=forecast_len,
        num_features=num_features,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        tree_depth=tree_depth,
        overlap_ratio=overlap_ratio,
        num_wavelet_levels=num_wavelet_levels,
        router_hidden_dim=router_hidden_dim,
        router_temperature=router_temperature,
        dropout_rate=dropout_rate,
        ffn_expansion=ffn_expansion,
        use_quantile_head=use_quantile_head,
        num_quantiles=num_quantiles,
        quantile_levels=quantile_levels,
        enforce_monotonicity=enforce_monotonicity,
        **kwargs
    )

    # Build the model with a dummy input to initialize weights and shapes.
    dummy_input = np.zeros((1, context_len, num_features), dtype="float32")
    _ = model(dummy_input)

    logger.info(
        f"Created PRISM model: context_len={context_len}, "
        f"forecast_len={forecast_len}, num_features={num_features}, "
        f"hidden_dim={model.hidden_dim}, use_quantile_head={use_quantile_head}"
    )

    return model

# ---------------------------------------------------------------------
