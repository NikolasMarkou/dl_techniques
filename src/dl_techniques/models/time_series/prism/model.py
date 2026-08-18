"""
Hierarchical time-frequency forecaster over a binary time tree with per-band routing,
a channel-independent temporal decoder, and point or monotone-quantile heads.

Real series are not single-scale objects. A demand curve carries a multi-year trend,
a weekly cycle and a burst that lasts three hours, all at once, and any model that
commits to one resolution — a fixed patch length, a fixed wavelet level, a single
downsampling rate — resolves one of those and blurs the rest. Widening the receptive
field trades away the short structure; narrowing it trades away the long. The
resolution is a choice, and PRISM's answer is to stop making it globally.

The signal is partitioned along the time axis into overlapping segments, so level `i`
of the tree holds `2^i` views of increasingly local extent, with the root seeing the
whole context. Mechanically this is a LOOP, not a recursion: each level re-splits the
full, re-stitched sequence into `2^i` segments rather than bisecting the previous
level's children, so the deepest leaf's length follows from ONE application of the
split formula at `num_segments = 2^tree_depth`. At every node the segment is passed
through a Haar DWT, producing a set of frequency bands within that node's temporal
span. `tree_depth` sets how finely time is partitioned and `num_wavelet_levels` sets
how finely frequency is split inside each partition, and a node deep in the tree can
still attend to a low band while a shallow one attends to a high one. That decoupling
is the architectural claim, and it holds for WHICH band a node selects.

It does NOT hold for how far the two knobs can be turned. They multiply into a single
budget: the deepest band has length
`min_band_len = deepest_leaf_segment_len // 2 ** num_wavelet_levels`, and once that
reaches 0 the configuration is unrepresentable. `__init__` refuses those configurations
with a `ValueError`. Measured (36-cell grid, plan-2026-08-18T073231-52a93f8c): no
`tree_depth` range separates the working configurations from the broken ones —
`context_len=96, tree_depth=2, num_wavelet_levels=4` used to be all-NaN while
`context_len=256, tree_depth=4, num_wavelet_levels=3` was always fine. Read the
constraint off `min_band_len`, never off `tree_depth` alone.

Which bands matter is decided per node by data, not by hyperparameter. A small shared
MLP router reads six summary statistics of each band — mean, standard deviation, min,
max, and the mean and standard deviation of its first difference — and emits a score;
scores are turned into weights by a temperature-scaled softmax across bands. Two
consequences follow from that design. Because the router consumes statistics rather
than the band content itself, its cost is independent of segment length and its
decision is invariant to where within the segment a feature occurs. Because the
normalization is a softmax, the weighting is competitive: bands bid against each
other for a fixed budget, so the router expresses relative importance and cannot
simply amplify everything. The standard deviations are computed as `sqrt(var + eps)`
rather than through a plain `std`, since the gradient of `std` carries a `1 / (2 std)`
factor that explodes on a constant band — a real occurrence in a high-frequency band
of a smooth segment.

Reconstruction stitches children back into the parent's span with a linear cross-fade
across the overlap region. Without overlapping segments and a fade, the tree would
print its own split points into the output as discontinuities at every level, and
those artifacts would be indistinguishable from signal to everything downstream.
`overlap_ratio` is constrained below 0.5 because at half a segment length the
partition stops being a partition.

The decoder is where most of the parameter budget would ordinarily be wasted. After
the PRISM stack the latent is `[B, T_ctx, H]`; flattening it into a Dense that emits
the horizon costs `O(T_ctx * T_out * H)` weights, dominating the model. Instead the
tensor is transposed so the Dense acts on the TIME axis, and the same
context-to-horizon map is shared across every hidden channel — the DLinear
observation that a linear temporal map is both sufficient and drastically cheaper.
Batch and horizon are then collapsed into one axis so the output head is applied
identically at every forecast step. The head has no step-specific parameters at all,
which is the reason it can be a small MLP; the price is that the horizon length lives
entirely in the temporal projector, so changing it means rebuilding that layer rather
than reshaping the head.

In probabilistic mode the head is a `QuantileHead` with monotonicity enforced, so
`Q_i <= Q_{i+1}` holds by construction rather than by penalty — crossed quantiles are
not merely untidy, they do not describe a distribution, and a soft penalty leaves the
model free to violate the constraint wherever the data pressure is strong enough. The
head's own dropout is set to zero because `head_dropout` has already been applied to
its input; leaving both active would drop twice on the same path. The final reshape
to `[B, H, F, Q]` is done explicitly here rather than trusted to the head's own output
layout.

Two deliberate choices govern the prediction surface. `predict_quantiles` maps a
requested level that was not trained to the nearest trained level and warns, rather
than interpolating between quantiles — an interpolated quantile is a fabricated
number with no calibration behind it. And in point mode `_forecast` returns
`quantiles=None` instead of manufacturing an interval from a model that was never
trained to produce one.

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

# ---------------------------------------------------------------------
# Local Imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.models.time_series.forecast import Forecast, ForecastMixin
from dl_techniques.layers.ffn import create_ffn_layer
from dl_techniques.layers.time_series.prism_blocks import PRISMLayer, PRISMTimeTree
from dl_techniques.layers.time_series.quantile_head_fixed_io import QuantileHead

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class PRISMModel(keras.Model, ForecastMixin):
    """
    Complete PRISM model for time series forecasting.

    **Intent**: Provide a hierarchical multiscale (time + frequency)
    forecasting model that maps a context window to either point forecasts
    ``[B, H, F]`` or probabilistic quantile forecasts ``[B, H, F, Q]``, with a
    parameter-efficient channel-independent temporal decoder (DLinear-style)
    and a standardized ``QuantileHead`` for monotonic confidence intervals.
    Conforms to the Keras 3 custom-model contract: all sublayers are created in
    ``__init__``, built explicitly in ``build()``, and the full config
    round-trips via ``get_config``/``from_config``. Named architecture sizes are
    exposed via ``MODEL_VARIANTS`` + :meth:`from_variant`.

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
            NOT independently bounded: there is no valid range for this knob on
            its own. Together with ``context_len``, ``overlap_ratio`` and
            ``num_wavelet_levels`` it determines
            ``min_band_len = deepest_leaf_segment_len // 2 ** num_wavelet_levels``,
            and ``__init__`` raises ``ValueError`` when that reaches 0. Depth 2 at
            ``num_wavelet_levels=4`` and ``context_len=96`` is refused; depth 4 at
            ``context_len=256, num_wavelet_levels=3`` is fine. Node count grows as
            ``2 ** tree_depth`` per layer, so cost is exponential in this knob.
        overlap_ratio: Overlap ratio for segment splitting. Must be in
            ``[0, 0.5)`` -- half-open; outside that range ``__init__`` raises
            ``ValueError``. Defaults to 0.25. Feeds the segment-length formula,
            so it shifts ``min_band_len`` too -- it is not a purely cosmetic
            smoothing knob.
        num_wavelet_levels: Number of Haar DWT levels. Defaults to 3. Each level
            floor-halves the band length, so this trades directly against
            ``tree_depth`` and ``context_len`` through ``min_band_len``; raising it
            on a short context is what drives the deepest band to length 0.
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
            If None and ``num_quantiles`` equals ``len(DEFAULT_QUANTILES)``, defaults to
            ``DEFAULT_QUANTILES`` ([0.1, 0.5, 0.9]); at any other length it falls back to
            evenly spaced interior levels ``np.linspace(0, 1, num_quantiles + 2)[1:-1]``.
        enforce_monotonicity: Whether to enforce non-crossing quantiles
            (Q_i <= Q_{i+1}). Only used when use_quantile_head=True. Defaults to True.
        kernel_initializer: Initializer for kernel weights. Defaults to "glorot_uniform".
        kernel_regularizer: Optional regularizer for kernel weights.
        **kwargs: Additional arguments for the Model base class.
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

        # DECISION plan-2026-08-18T073231-52a93f8c/D-011
        # These three guards must stay AHEAD of the `min_band_len` block below,
        # and they must be unconditional. They are not decoration: MEASURED,
        # `PRISMModel(context_len=96, forecast_len=24, num_features=7,
        # overlap_ratio=-20.0)` HUNG (no return in 60s, SIGTERM exit 124)
        # because a negative `overlap_ratio` makes the segment length negative
        # at EVERY `context_len` (-18 at 96, -18750 at 100000), so the remedy
        # search below could never terminate. The same call raises here now.
        # The contract `[0, 0.5)` is `PRISMTimeTree.__init__`'s
        # (`prism_blocks.py`), duplicated here only because `PRISMTimeTree` is
        # constructed AFTER the geometry arithmetic runs. Do NOT relax it to
        # `[0, 0.5]` to match older README prose -- the code's half-open
        # interval is normative and the README was corrected to it.
        # Do NOT fold these back into a `if tree_depth >= 0 and
        # num_wavelet_levels >= 0:` guard around the block below: that shape
        # SILENTLY SKIPPED all band validation for negative values instead of
        # rejecting them. See decisions.md D-011.
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

        # DECISION plan-2026-08-18T073231-52a93f8c/D-005
        # The governing quantity is `min_band_len`, NOT a `tree_depth` range. Do
        # NOT "simplify" this into a bound on `tree_depth`: MEASURED over a
        # 36-cell grid, `context_len=96, tree_depth=2, num_wavelet_levels=4` is
        # broken while `context_len=256, tree_depth=4, num_wavelet_levels=3` is
        # perfectly finite. Depth alone cannot separate the two; only the
        # deepest band's length can, and it depends on all four knobs named in
        # the message.
        # This must be refused HERE, at construction, and not left to the
        # forward pass. Below the threshold the failure is SILENT under
        # `@tf.function` -- a length-0 band returns inf/NaN from `ops.min`/
        # `ops.max` without raising (it raises only in eager), and since the
        # degenerate-band guard landed in `FrequencyBandStatistics` it does not
        # even do that: it returns finite ALL-ZERO statistics. There is no
        # runtime signal left. See decisions.md D-004 and D-005.
        num_leaves = 2 ** tree_depth
        if num_leaves == 1:
            # `PRISMTimeTree._split_with_overlap` returns the sequence whole
            # at a single segment; the geometry formula does not apply.
            deepest_leaf_seg = context_len
        else:
            # `PRISMTimeTree.call` is NOT recursive -- every level re-splits
            # the full re-stitched sequence -- so the deepest leaf is ONE
            # application of the shared geometry at `2 ** tree_depth`.
            _, _, deepest_leaf_seg = PRISMTimeTree._segment_len(
                context_len, overlap_ratio, num_leaves,
                dtype=self.compute_dtype
            )
        # Each Haar level floor-halves the band length.
        min_band_len = deepest_leaf_seg // (2 ** num_wavelet_levels)
        if min_band_len < 1:
            # Smallest context_len that lifts this exact (tree_depth,
            # num_wavelet_levels, overlap_ratio) triple to min_band_len >= 1,
            # solved by search rather than by an approximation of the
            # segment geometry -- the remedy printed must be one that works.
            # The cap is belt-and-braces and must NOT be removed: an unbounded
            # `while True` in a constructor is a latent hang, and this one DID
            # hang before the `overlap_ratio` guard above landed (60s, no
            # return). With every knob validated the search terminates in a few
            # thousand iterations at worst, so a miss now means an unforeseen
            # geometry and the message simply omits the concrete suggestion
            # rather than spinning.
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
                # DECISION plan-2026-08-14T233721-d4f9beb2/D-075
                # `DEFAULT_QUANTILES = [0.1, 0.5, 0.9]` was dead: this branch always ran
                # `np.linspace(0, 1, 5)[1:-1] = [0.25, 0.5, 0.75]`, contradicting the
                # constructor docstring's "typically 10th, 50th, 90th percentiles". The
                # named constant is now the default at its own length. Do NOT replace the
                # linspace fallback with an unconditional `DEFAULT_QUANTILES`: that would
                # silently give a 5-quantile head 3 levels and re-break the
                # `len(quantile_levels) == num_quantiles` invariant this class validates
                # two branches above. See decisions.md D-075.
                if num_quantiles == len(self.DEFAULT_QUANTILES):
                    self.quantile_levels = list(self.DEFAULT_QUANTILES)
                else:
                    # No canonical set at this length: fall back to evenly spaced levels.
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
        if 0.5 in self.quantile_levels:
            median_idx = self.quantile_levels.index(0.5)
        else:
            median_idx = len(self.quantile_levels) // 2

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
