"""
PRISM: Partitioned Representations for Iterative Sequence Modeling.

This module implements the PRISM architecture for time-series forecasting,
which combines hierarchical time decomposition with multi-resolution
frequency analysis using Haar wavelets.

The architecture uses a "Split-Transform-Weight-Merge" philosophy applied
recursively to capture both global trends and local fine-grained structures.
"""

import keras
import numpy as np
from keras import ops, layers, initializers, regularizers
from typing import Optional, Union, Tuple, List, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.ffn import create_ffn_layer
from dl_techniques.layers.haar_wavelet_decomposition import HaarWaveletDecomposition


# ---------------------------------------------------------------------
# Frequency Band Statistics Layer
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class FrequencyBandStatistics(keras.layers.Layer):
    """
    Compute summary statistics for frequency bands.

    Extracts statistical features from each frequency band including mean,
    standard deviation, min, max, and temporal derivatives (first-difference
    mean and std). These statistics serve as input to the importance router.

    Standard deviation is computed as ``sqrt(var + epsilon)`` to prevent
    gradient explosion when processing constant sequences or zero-padded
    regions.

    **Architecture Overview:**

    .. code-block:: text

        Input: frequency band [batch, seq_len, channels]
                        │
                        ▼
               ┌────────────────────────────────┐
               │  Compute per-channel statistics│
               │  mean, std, min, max           │
               │  diff_mean, diff_std           │
               └────────────────┬───────────────┘
                                │
                                ▼
        Output: statistics [batch, channels, 6]

    :param epsilon: Small constant for numerical stability.
        Defaults to 1e-6.
    :type epsilon: float
    :param kwargs: Additional arguments for the Layer base class.
    """

    def __init__(
            self,
            epsilon: float = 1e-6,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self._num_stats = 6  # mean, std, min, max, diff_mean, diff_std

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None,
            mask: Optional[keras.KerasTensor] = None
    ) -> keras.KerasTensor:
        """
        Compute statistics for the input frequency band.

        :param inputs: Input tensor of shape [batch, seq_len, channels].
        :type inputs: keras.KerasTensor
        :param training: Training mode flag (unused).
        :type training: Optional[bool]
        :param mask: Optional mask tensor (unused in calculation, rely on
            numerical stability fixes for padded zeros).
        :type mask: Optional[keras.KerasTensor]
        :return: Statistics tensor of shape [batch, channels, num_stats].
            Degenerate bands are handled explicitly: a band of length 1 gets
            ``diff_mean``/``diff_std`` of exactly ``0.0`` (the first difference
            of a single sample is undefined), and a band of length 0 gets all
            six statistics as ``0.0``. When the time axis is not known
            statically (a traced graph over ``[batch, None, channels]``) the
            same semantics are reproduced by replacing non-finite statistics
            with ``0.0``; on a statically shaped input a genuine NaN in the data
            still propagates.
        :rtype: keras.KerasTensor
        """
        # DECISION plan-2026-08-18T073231-52a93f8c/D-004
        # Degenerate band lengths are branched on the STATIC shape, with an
        # ``is not None`` short-circuit so an unknown time axis falls through to
        # the unguarded arithmetic below.  Bands shrink as
        # ``segment_len // 2 ** num_wavelet_levels`` and reach 1, then 0, at
        # configurations the model accepts; a length-1 band makes the
        # first-difference tensor EMPTY, and ``ops.mean``/``ops.var`` over an
        # empty axis return NaN *silently*, which the router's single joint
        # softmax then spreads across every band.  A length-0 band additionally
        # makes ``ops.min``/``ops.max`` raise ``InvalidArgumentError`` in eager
        # and return +/-inf under ``tf.function`` (both measured).
        # Do NOT rewrite this as a traced-tensor ``ops.cond`` on
        # ``ops.shape(inputs)[1]``: this repo has measured exactly that rewrite
        # break under ``@tf.function`` with ``OperatorNotAllowedInGraphError``
        # (``clifford_block``'s singleton-axis check), and the static-shape form
        # with an ``is not None`` short-circuit is the form that works.
        # Do NOT "simplify" it by clamping or padding the slice to fake a
        # length: that fabricates a first difference from a sample that does not
        # exist, which is a silently wrong number rather than a defined one.
        # The DYNAMIC time axis (``static_len is None``) falls through this
        # branch by design and is closed separately, at the bottom of ``call``,
        # by a value-level non-finite repair -- see the
        # ``plan-2026-08-18T111512-29569f8b/D-001`` anchor there for why
        # ``input_spec`` cannot close it and why the repair is confined to that
        # path.
        # See decisions.md D-004, D-012 (and D-002 for the threshold ruling).
        static_len = inputs.shape[1]

        if static_len is not None and static_len == 0:
            # Nothing is defined over an empty time axis. ``ops.sum`` over a
            # zero-length axis is exactly 0.0 and, unlike ``ops.min``/``ops.max``,
            # does not raise -- it also carries the dynamic batch dimension.
            zeros = ops.sum(inputs, axis=1)  # [batch, channels], exact zeros
            return ops.stack([zeros] * self._num_stats, axis=-1)

        # Basic statistics along time axis
        mean = ops.mean(inputs, axis=1)  # [batch, channels]

        # FIX: Calculate std via sqrt(var + epsilon) to prevent gradient explosion
        # ops.std() gradients involve 1/(2*std), which explodes if std=0.
        variance = ops.var(inputs, axis=1)
        std = ops.sqrt(variance + self.epsilon)

        min_val = ops.min(inputs, axis=1)
        max_val = ops.max(inputs, axis=1)

        if static_len is not None and static_len == 1:
            # mean/std/min/max are well defined on a single sample; only the
            # first difference is not, and 0.0 is its defined stand-in.
            diff_mean = ops.zeros_like(mean)
            diff_std = ops.zeros_like(mean)
        else:
            # Temporal derivatives (first difference)
            diff = inputs[:, 1:, :] - inputs[:, :-1, :]
            diff_mean = ops.mean(diff, axis=1)

            # FIX: Apply same stability fix to diff_std
            diff_variance = ops.var(diff, axis=1)
            diff_std = ops.sqrt(diff_variance + self.epsilon)

        # Stack statistics: [batch, channels, num_stats]
        stats = ops.stack(
            [mean, std, min_val, max_val, diff_mean, diff_std],
            axis=-1
        )

        # DECISION plan-2026-08-18T111512-29569f8b/D-001
        # Dynamic-time-axis fallback.  The branch above is on the STATIC length,
        # so under a trace with an unknown time axis -- ``TensorSpec([None, None,
        # C])``, what an ONNX/SavedModel export or a ragged ``tf.data`` pipeline
        # produces -- neither degenerate case is caught and the band arithmetic
        # runs raw.  MEASURED before this fallback existed: a built
        # ``tree_depth=3`` ``PRISMModel`` traced at ``[None, None, 7]`` returned
        # ``nan_frac == 1.0`` while the SAME model returned ``0.0`` eager.
        # ``input_spec`` CANNOT close this: Keras'
        # ``assert_input_compatibility`` tests ``shape[axis] not in {value,
        # None}``, so an unknown dimension is explicitly ACCEPTED by an ``axes``
        # constraint (measured, not inferred).  ``ops.cond`` on
        # ``ops.shape(inputs)[1]`` is also out -- this repo has measured that
        # rewrite break under ``@tf.function`` with
        # ``OperatorNotAllowedInGraphError`` (``clifford_block``).
        # What IS graph-safe is a value-level repair, because in graph mode the
        # degenerate cases fail *numerically* rather than raising: a length-0
        # band gives ``ops.min``/``ops.max`` of ``+inf``/``-inf`` (in EAGER the
        # very same call raises ``InvalidArgumentError``, which is why the static
        # length-0 branch above must stay -- it cannot be repaired after the
        # fact) and ``ops.mean``/``ops.var`` of NaN; a length-1 band gives NaN
        # diff features.  Replacing every non-finite statistic with 0.0 therefore
        # reproduces EXACTLY the semantics the static branches define, with no
        # Python branch on a symbolic value.
        # The ``is None`` test below is a TRACE-TIME branch on a Python object,
        # not on a tensor value -- it is safe.
        # Do NOT hoist this sanitization out of the ``is None`` guard to "cover
        # both paths": on the static path a genuine NaN in user data must keep
        # propagating, because it is a data defect the caller has to see, not
        # something a statistics layer may silently rewrite to 0.0.  Laundering
        # it here would make a corrupt window indistinguishable from a constant
        # one.  Pinned by
        # ``test_static_path_still_propagates_genuine_nan_inputs``.
        # See decisions.md D-001 (and D-004/D-012 of
        # plan-2026-08-18T073231-52a93f8c for the static branch this completes).
        if static_len is None:
            stats = ops.where(ops.isfinite(stats), stats, ops.zeros_like(stats))

        return stats

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        :param input_shape: Input shape tuple.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple.
        :rtype: Tuple[Optional[int], ...]
        """
        batch_size = input_shape[0]
        channels = input_shape[2]
        return (batch_size, channels, self._num_stats)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            "epsilon": self.epsilon,
        })
        return config


# ---------------------------------------------------------------------
# Frequency Band Router Layer
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class FrequencyBandRouter(keras.layers.Layer):
    """
    Learnable router for computing frequency band importance weights.

    Uses a lightweight MLP to compute importance scores for different
    frequency bands based on their statistical properties. Scores are
    normalized via temperature-scaled softmax:

        weight_k = softmax(score_k / temperature)

    **Architecture Overview:**

    .. code-block:: text

        Input: [band_1, ..., band_K]
                       │
                       ▼
        ┌──────────────────────────────┐
        │  For each band:              │
        │    FrequencyBandStatistics   │
        │           │                  │
        │           ▼                  │
        │    LayerNorm(statistics)     │
        │           │                  │
        │           ▼                  │
        │    MLP(stats) ─► score       │
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │  Softmax(scores / temp)      │
        └──────────────┬───────────────┘
                       │
                       ▼
        Output: weights [batch, channels, K]

    :param hidden_dim: Hidden dimension of the router MLP.
        Defaults to 64.
    :type hidden_dim: int
    :param temperature: Temperature for softmax scaling. Lower values
        produce sharper distributions. Defaults to 1.0.
    :type temperature: float
    :param dropout_rate: Dropout rate for the router MLP.
        Defaults to 0.1.
    :type dropout_rate: float
    :param kernel_initializer: Initializer for kernel weights.
        Defaults to "glorot_uniform".
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for kernel weights.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional arguments for the Layer base class.
    """

    def __init__(
            self,
            hidden_dim: int = 64,
            temperature: float = 1.0,
            dropout_rate: float = 0.1,
            kernel_initializer: Union[str, initializers.Initializer] = "glorot_uniform",
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {hidden_dim}")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.dropout_rate = dropout_rate
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

        # Statistics extractor
        self.stats_layer = FrequencyBandStatistics(name=f"{self.name}_stats")

        # FIX: Normalize statistics before MLP to prevent softmax overflow
        # and handle varying scales of inputs (e.g. padded zeros vs real data).
        self.stats_norm = layers.LayerNormalization(
            name=f"{self.name}_stats_norm",
            axis=-1
        )

        # Router MLP (shared across bands)
        self.router_mlp = create_ffn_layer(
            "mlp",
            hidden_dim=hidden_dim,
            output_dim=1,
            activation="gelu",
            dropout_rate=dropout_rate,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name=f"{self.name}_router_mlp"
        )

    def build(self, input_shape: List[Tuple[Optional[int], ...]]) -> None:
        """
        Build the layer and initialize sub-layer weights.

        :param input_shape: List of input shapes for each frequency band.
        :type input_shape: List[Tuple[Optional[int], ...]]
        """
        if not isinstance(input_shape, list) or len(input_shape) == 0:
            raise ValueError(
                "input_shape must be a non-empty list of shapes"
            )

        # Build stats layer with first band shape
        first_band_shape = input_shape[0]
        self.stats_layer.build(first_band_shape)

        # Build norm and router MLP
        channels = first_band_shape[-1]
        stats_shape = (first_band_shape[0], channels, 6)

        self.stats_norm.build(stats_shape)
        self.router_mlp.build(stats_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: List[keras.KerasTensor],
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Compute importance weights for frequency bands.

        :param inputs: List of frequency band tensors.
        :type inputs: List[keras.KerasTensor]
        :param training: Training mode flag.
        :type training: Optional[bool]
        :return: Importance weights of shape [batch, channels, num_bands].
        :rtype: keras.KerasTensor
        """
        scores = []
        for band in inputs:
            # Compute statistics for this band
            stats = self.stats_layer(band, training=training)

            # FIX: Normalize statistics
            stats = self.stats_norm(stats)

            # Get raw score from MLP
            score = self.router_mlp(stats, training=training)
            # score shape: [batch, channels, 1]
            scores.append(score)

        # Stack scores: [batch, channels, num_bands]
        scores = ops.concatenate(scores, axis=-1)

        # Apply temperature-scaled softmax
        weights = ops.softmax(scores / self.temperature, axis=-1)
        return weights

    def compute_output_shape(
            self,
            input_shape: List[Tuple[Optional[int], ...]]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        :param input_shape: List of input shapes for each frequency band.
        :type input_shape: List[Tuple[Optional[int], ...]]
        :return: Output shape tuple.
        :rtype: Tuple[Optional[int], ...]
        """
        batch_size = input_shape[0][0]
        channels = input_shape[0][-1]
        num_bands = len(input_shape)
        return (batch_size, channels, num_bands)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "temperature": self.temperature,
            "dropout_rate": self.dropout_rate,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
        })
        return config


# ---------------------------------------------------------------------
# PRISM Node Layer
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class PRISMNode(keras.layers.Layer):
    """
    Single PRISM node combining wavelet decomposition and adaptive weighting.

    Processes a time segment by decomposing it into frequency bands via Haar
    DWT, computing importance weights for each band through a learned router,
    and reconstructing a weighted representation by interpolating all bands
    to a common length and summing them with the computed weights.

    **Architecture Overview:**

    .. code-block:: text

        Input: time segment [batch, seq_len, channels]
                        │
                        ▼
        ┌───────────────────────────────────┐
        │  HaarWaveletDecomposition         │
        │  ─► [approx, detail_1, ..., det_K]│
        └───────────────┬───────────────────┘
                        │
                ┌───────┴───────┐
                │               │
                ▼               ▼
        ┌──────────────┐  ┌──────────────────┐
        │  Interpolate │  │ FrequencyBand    │
        │  all bands   │  │ Router ─► weights│
        │  to seq_len  │  └────────┬─────────┘
        └──────┬───────┘           │
               │                   │
               └───────┬───────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │  Weighted sum of all bands   │
        └──────────────┬───────────────┘
                       │
                       ▼
        Output: processed [batch, seq_len, channels]

    :param num_wavelet_levels: Number of Haar DWT decomposition levels,
        producing ``num_wavelet_levels + 1`` bands (one detail band per level
        plus the final approximation band). Defaults to 3. Each level
        floor-halves the length, so the deepest band is
        ``seq_len // 2 ** num_wavelet_levels`` long; at 1 it is statistically
        degenerate and at 0 the configuration is unrepresentable (see
        :class:`FrequencyBandStatistics` and ``PRISMModel.__init__``).
    :type num_wavelet_levels: int
    :param router_hidden_dim: Hidden dimension for the router MLP.
        Defaults to 64.
    :type router_hidden_dim: int
    :param router_temperature: Temperature for router softmax.
        Defaults to 1.0.
    :type router_temperature: float
    :param dropout_rate: Dropout rate for the router.
        Defaults to 0.1.
    :type dropout_rate: float
    :param kernel_initializer: Initializer for kernel weights.
        Defaults to "glorot_uniform".
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for kernel weights.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional arguments for the Layer base class.
    """

    def __init__(
            self,
            num_wavelet_levels: int = 3,
            router_hidden_dim: int = 64,
            router_temperature: float = 1.0,
            dropout_rate: float = 0.1,
            kernel_initializer: Union[str, initializers.Initializer] = "glorot_uniform",
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.num_wavelet_levels = num_wavelet_levels
        self.router_hidden_dim = router_hidden_dim
        self.router_temperature = router_temperature
        self.dropout_rate = dropout_rate
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

        # Wavelet decomposition
        self.wavelet = HaarWaveletDecomposition(
            num_levels=num_wavelet_levels,
            name=f"{self.name}_wavelet"
        )

        # Importance router
        self.router = FrequencyBandRouter(
            hidden_dim=router_hidden_dim,
            temperature=router_temperature,
            dropout_rate=dropout_rate,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name=f"{self.name}_router"
        )
        self.supports_masking = True

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and initialize sub-layer weights.

        :param input_shape: Input shape tuple.
        :type input_shape: Tuple[Optional[int], ...]
        """
        self.wavelet.build(input_shape)

        # Get output shapes from wavelet
        band_shapes = self.wavelet.compute_output_shape(input_shape)
        self.router.build(band_shapes)

        super().build(input_shape)

    def _interpolate_band(
            self,
            band: keras.KerasTensor,
            target_len: int
    ) -> keras.KerasTensor:
        """
        Interpolate a frequency band to target length using linear interpolation.

        :param band: Band tensor of shape [batch, band_len, channels].
        :type band: keras.KerasTensor
        :param target_len: Target sequence length.
        :type target_len: int
        :return: Interpolated tensor of shape [batch, target_len, channels].
        :rtype: keras.KerasTensor
        """
        band_len = ops.shape(band)[1]

        # If already target length, return as-is
        def do_interpolate():
            # Linear indices for target positions
            # Map [0, target_len-1] -> [0, band_len-1]
            target_indices = ops.cast(
                ops.arange(target_len),
                band.dtype
            )
            # Safe division: max(..., 1) ensures no divide-by-zero
            scale = ops.cast(band_len - 1, band.dtype) / ops.cast(
                ops.maximum(target_len - 1, 1),
                band.dtype
            )
            source_indices = target_indices * scale

            # Floor and ceil indices
            floor_idx = ops.cast(ops.floor(source_indices), "int32")
            ceil_idx = ops.minimum(floor_idx + 1, band_len - 1)

            # Interpolation weights
            alpha = source_indices - ops.cast(floor_idx, band.dtype)
            alpha = ops.expand_dims(ops.expand_dims(alpha, 0), -1)

            # Gather and interpolate
            floor_vals = ops.take(band, floor_idx, axis=1)
            ceil_vals = ops.take(band, ceil_idx, axis=1)

            return floor_vals * (1.0 - alpha) + ceil_vals * alpha

        def no_interpolate():
            return band

        return ops.cond(
            ops.not_equal(band_len, target_len),
            do_interpolate,
            no_interpolate
        )

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None,
            mask: Optional[keras.KerasTensor] = None
    ) -> keras.KerasTensor:
        """
        Process input through wavelet decomposition and weighted reconstruction.

        :param inputs: Input tensor of shape [batch, seq_len, channels].
        :type inputs: keras.KerasTensor
        :param training: Training mode flag.
        :type training: Optional[bool]
        :param mask: Optional mask.
        :type mask: Optional[keras.KerasTensor]
        :return: Processed tensor of shape [batch, seq_len, channels].
        :rtype: keras.KerasTensor
        """
        target_len = ops.shape(inputs)[1]

        # Decompose into frequency bands
        bands = self.wavelet(inputs, training=training)

        # Compute importance weights
        # Note: Mask is not passed to wavelet bands as they are downsampled,
        # but FrequencyBandStatistics handles potential padding stability issues
        # internally via the std calculation fix.
        weights = self.router(bands, training=training)
        # weights shape: [batch, channels, num_bands]

        # Interpolate all bands to input length and weight
        weighted_sum = ops.zeros_like(inputs)
        for i, band in enumerate(bands):
            # Interpolate band to target length
            band_interp = self._interpolate_band(band, target_len)

            # Get weight for this band: [batch, channels, 1] -> [batch, 1, channels]
            band_weight = ops.expand_dims(weights[:, :, i], axis=1)

            # Weight and accumulate
            weighted_sum = weighted_sum + band_interp * band_weight

        return weighted_sum

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        :param input_shape: Input shape tuple.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple (same as input).
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            "num_wavelet_levels": self.num_wavelet_levels,
            "router_hidden_dim": self.router_hidden_dim,
            "router_temperature": self.router_temperature,
            "dropout_rate": self.dropout_rate,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
        })
        return config


# ---------------------------------------------------------------------
# PRISM Time Tree Layer
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class PRISMTimeTree(keras.layers.Layer):
    """
    Hierarchical time decomposition with PRISM nodes at each level.

    Builds a binary tree over the time domain by splitting the signal into
    overlapping segments. Each node processes its segment through wavelet
    decomposition and adaptive weighting. Segments are stitched back together
    using linear cross-fade blending in the overlap regions.

    The traversal is a LOOP over levels, not a recursion over children: level
    ``i`` re-splits the FULL, re-stitched sequence into ``2 ** i`` segments
    rather than bisecting level ``i - 1``'s outputs. The deepest leaf's length
    therefore comes from ONE application of :meth:`_segment_len` at
    ``num_segments = 2 ** tree_depth`` -- not from ``tree_depth`` successive
    halvings. Anything reasoning about the deepest segment (band lengths,
    configuration validation) must use the one-shot form.

    **Architecture Overview:**

    .. code-block:: text

        Input: [batch, T, channels]
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  Level 0: Full sequence          │
        │    └─► PRISMNode                 │
        └──────────────┬───────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  Level 1: Split ─► 2 segments    │
        │    ├─► PRISMNode (left half)     │
        │    └─► PRISMNode (right half)    │
        │    Stitch with cross-fade        │
        └──────────────┬───────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  Level 2: Split ─► 4 segments    │
        │    ├─► PRISMNode x4              │
        │    Stitch with cross-fade        │
        └──────────────┬───────────────────┘
                       │
                       ▼
        Output: [batch, T, channels]

    :param tree_depth: Depth of the binary time tree. Depth 0 means single
        node (no splitting). Defaults to 2. This knob has no valid range of its
        own: with ``overlap_ratio`` and the input length it fixes the deepest
        segment length, which ``num_wavelet_levels`` then floor-halves down to
        the deepest band. ``PRISMModel.__init__`` refuses combinations whose
        deepest band would have length 0; this layer does not validate.
    :type tree_depth: int
    :param overlap_ratio: Ratio of overlap between adjacent segments.
        Value in [0, 0.5). Defaults to 0.25.
    :type overlap_ratio: float
    :param num_wavelet_levels: Number of Haar DWT levels per node, producing
        ``num_wavelet_levels + 1`` bands. Defaults to 3. Each level floor-halves
        the band length, so the deepest band of the deepest node is
        ``segment_len // 2 ** num_wavelet_levels`` -- this trades directly
        against ``tree_depth`` and the input length.
    :type num_wavelet_levels: int
    :param router_hidden_dim: Hidden dimension for router MLPs.
        Defaults to 64.
    :type router_hidden_dim: int
    :param router_temperature: Temperature for router softmax.
        Defaults to 1.0.
    :type router_temperature: float
    :param dropout_rate: Dropout rate for routers.
        Defaults to 0.1.
    :type dropout_rate: float
    :param kernel_initializer: Initializer for kernel weights.
        Defaults to "glorot_uniform".
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for kernel weights.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional arguments for the Layer base class.
    """

    def __init__(
            self,
            tree_depth: int = 2,
            overlap_ratio: float = 0.25,
            num_wavelet_levels: int = 3,
            router_hidden_dim: int = 64,
            router_temperature: float = 1.0,
            dropout_rate: float = 0.1,
            kernel_initializer: Union[str, initializers.Initializer] = "glorot_uniform",
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if tree_depth < 0:
            raise ValueError(f"tree_depth must be >= 0, got {tree_depth}")
        if not 0 <= overlap_ratio < 0.5:
            raise ValueError(
                f"overlap_ratio must be in [0, 0.5), got {overlap_ratio}"
            )

        self.tree_depth = tree_depth
        self.overlap_ratio = overlap_ratio
        self.num_wavelet_levels = num_wavelet_levels
        self.router_hidden_dim = router_hidden_dim
        self.router_temperature = router_temperature
        self.dropout_rate = dropout_rate
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.supports_masking = True

        # Create PRISM nodes for each level of the tree
        # Use a flat list so Keras tracks all layers properly for serialization
        self.all_nodes: List[PRISMNode] = []

        for level in range(tree_depth + 1):
            num_nodes = 2 ** level
            for node_idx in range(num_nodes):
                node = PRISMNode(
                    num_wavelet_levels=num_wavelet_levels,
                    router_hidden_dim=router_hidden_dim,
                    router_temperature=router_temperature,
                    dropout_rate=dropout_rate,
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer,
                    name=f"{self.name}_level{level}_node{node_idx}"
                )
                self.all_nodes.append(node)

    # DECISION plan-2026-08-18T073231-52a93f8c/D-001
    # Three copies of this arithmetic had drifted apart. ``build()`` sized every
    # node with ``(seq_len + overlap_size * (n - 1)) // n`` (note the PLUS)
    # while both runtime sites used ``(seq_len - overlap_size * (n - 1)) // n``.
    # They disagree on ordinary configurations -- measured at
    # ``overlap_ratio=0.25``: ``context_len=96`` gives build 28 vs runtime 25 at
    # level 2 and build 14 vs runtime 12 at level 3; ``context_len=256`` gives
    # 76/68, 39/33 and 19/16 at levels 2/3/4.
    # The RUNTIME form is NORMATIVE -- it sizes the tensors the nodes actually
    # receive -- so this helper reproduces it and ``build()`` now follows it.
    # Do NOT "restore" the build-time PLUS form, and do NOT "simplify away" the
    # float round-trip in ``overlap_size``: the runtime path multiplies in the
    # tensor's compute dtype and truncates toward zero on the cast to int32, and
    # reproducing that truncation exactly is what keeps the two in step.
    # See decisions.md D-001.
    @staticmethod
    def _segment_len(
            seq_len: int,
            overlap_ratio: float,
            num_segments: int,
            dtype: str = "float32"
    ) -> Tuple[int, int, int]:
        """
        Compute the overlapping-segment geometry for one split.

        Single source of truth for the segment arithmetic shared by
        :meth:`build`, :meth:`_split_with_overlap` and
        :meth:`_stitch_with_crossfade`. Pure Python over ints; it mirrors the
        runtime expression exactly, including the truncating cast of
        ``overlap_size`` to int32.

        :param seq_len: Length of the sequence being split.
        :type seq_len: int
        :param overlap_ratio: Ratio of overlap between adjacent segments.
        :type overlap_ratio: float
        :param num_segments: Number of segments the sequence is split into.
        :type num_segments: int
        :param dtype: Float dtype the overlap is computed in, matching the
            compute dtype of the runtime tensor. Defaults to ``"float32"``.
        :type dtype: str
        :return: Tuple of ``(non_overlap_len, overlap_size, segment_len)``,
            where ``segment_len == non_overlap_len + overlap_size`` and
            segment ``i`` spans ``[i * non_overlap_len, i * non_overlap_len +
            segment_len)``.
        :rtype: Tuple[int, int, int]
        """
        float_type = np.dtype(keras.backend.standardize_dtype(dtype)).type
        overlap_size = int(
            float_type(seq_len)
            * float_type(overlap_ratio)
            / float_type(num_segments)
        )
        non_overlap_len = (
            seq_len - overlap_size * (num_segments - 1)
        ) // num_segments
        segment_len = non_overlap_len + overlap_size
        return non_overlap_len, overlap_size, segment_len

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build all PRISM nodes with appropriate segment shapes.

        :param input_shape: Input shape tuple.
        :type input_shape: Tuple[Optional[int], ...]
        """
        seq_len = input_shape[1]
        batch_size = input_shape[0]
        channels = input_shape[2]

        node_idx_counter = 0

        # Build nodes level by level
        for level in range(self.tree_depth + 1):
            num_nodes = 2 ** level

            # Determine segment shape for this level
            if seq_len is not None:
                if num_nodes > 1:
                    # Compute segment length with overlap. Same helper the
                    # forward pass uses, so a node is built for exactly the
                    # length it will be handed (see _segment_len's anchor).
                    _, _, segment_len = self._segment_len(
                        seq_len, self.overlap_ratio, num_nodes,
                        dtype=self.compute_dtype
                    )
                    segment_shape = (batch_size, segment_len, channels)
                else:
                    segment_shape = (batch_size, seq_len, channels)
            else:
                segment_shape = (batch_size, None, channels)

            # Build all nodes in this level
            for _ in range(num_nodes):
                self.all_nodes[node_idx_counter].build(segment_shape)
                node_idx_counter += 1

        super().build(input_shape)

    def _split_with_overlap(
            self,
            x: keras.KerasTensor,
            num_segments: int
    ) -> List[keras.KerasTensor]:
        """
        Split input into overlapping segments.

        :param x: Input tensor of shape [batch, seq_len, channels].
        :type x: keras.KerasTensor
        :param num_segments: Number of segments to create.
        :type num_segments: int
        :return: List of segment tensors.
        :rtype: List[keras.KerasTensor]
        """
        if num_segments == 1:
            return [x]

        static_seq_len = x.shape[1]
        if static_seq_len is not None:
            non_overlap_len, overlap_size, segment_len = self._segment_len(
                int(static_seq_len), self.overlap_ratio, num_segments,
                dtype=x.dtype
            )
        else:
            # Dynamic time axis: the same expression over traced tensors.
            seq_len = ops.shape(x)[1]
            seq_len_float = ops.cast(seq_len, x.dtype)
            overlap_size = ops.cast(
                seq_len_float * self.overlap_ratio / ops.cast(num_segments, x.dtype),
                "int32"
            )
            non_overlap_len = (
                seq_len - overlap_size * (num_segments - 1)
            ) // num_segments
            segment_len = non_overlap_len + overlap_size

        segments = []
        for i in range(num_segments):
            start_idx = i * non_overlap_len
            end_idx = start_idx + segment_len
            segment = x[:, start_idx:end_idx, :]
            segments.append(segment)

        return segments

    def _stitch_with_crossfade(
            self,
            segments: List[keras.KerasTensor],
            target_len: int
    ) -> keras.KerasTensor:
        """
        Stitch segments back together using linear cross-fade blending.

        :param segments: List of processed segment tensors.
        :type segments: List[keras.KerasTensor]
        :param target_len: Target output length.
        :type target_len: int
        :return: Stitched tensor of shape [batch, target_len, channels].
        :rtype: keras.KerasTensor
        """
        if len(segments) == 1:
            return segments[0][:, :target_len, :]

        num_segments = len(segments)

        # Calculate overlap parameters. Mirrors _split_with_overlap exactly --
        # the same helper, so the stitch offsets cannot drift from the split
        # offsets (see _segment_len's anchor).
        if isinstance(target_len, int):
            non_overlap_len, overlap_size, _ = self._segment_len(
                target_len, self.overlap_ratio, num_segments,
                dtype=segments[0].dtype
            )
        else:
            # Dynamic time axis: the same expression over traced tensors.
            seq_len_float = ops.cast(target_len, segments[0].dtype)
            overlap_size = ops.cast(
                seq_len_float * self.overlap_ratio
                / ops.cast(num_segments, segments[0].dtype),
                "int32"
            )
            non_overlap_len = (
                target_len - overlap_size * (num_segments - 1)
            ) // num_segments

        # Create output tensor
        batch_size = ops.shape(segments[0])[0]
        channels = ops.shape(segments[0])[-1]
        output = ops.zeros((batch_size, target_len, channels), dtype=segments[0].dtype)

        for i, segment in enumerate(segments):
            start_idx = i * non_overlap_len
            seg_len = ops.shape(segment)[1]

            # Create blending weights for overlap regions
            weights = ops.ones((1, seg_len, 1), dtype=segment.dtype)

            # Fade in at start (except first segment)
            if i > 0:
                # Use arange instead of linspace to avoid symbolic tensor issues with 'num'
                indices = ops.cast(ops.arange(overlap_size), segment.dtype)
                steps = ops.cast(overlap_size - 1, segment.dtype)
                # Avoid division by zero if overlap_size is 1
                steps = ops.maximum(steps, 1.0)
                fade_in = indices / steps

                fade_in = ops.reshape(fade_in, (1, overlap_size, 1))

                # Apply fade in to first overlap_size positions
                mask_after = ops.ones((1, seg_len - overlap_size, 1), dtype=segment.dtype)
                fade_mask = ops.concatenate([fade_in, mask_after], axis=1)
                weights = weights * fade_mask

            # Fade out at end (except last segment)
            if i < num_segments - 1:
                # Use arange manually
                indices = ops.cast(ops.arange(overlap_size), segment.dtype)
                steps = ops.cast(overlap_size - 1, segment.dtype)
                steps = ops.maximum(steps, 1.0)
                # fade out is 1.0 -> 0.0
                fade_out = 1.0 - (indices / steps)

                fade_out = ops.reshape(fade_out, (1, overlap_size, 1))

                mask_before = ops.ones((1, seg_len - overlap_size, 1), dtype=segment.dtype)
                fade_mask = ops.concatenate([mask_before, fade_out], axis=1)
                weights = weights * fade_mask

            # Add weighted segment to output
            weighted_segment = segment * weights

            # Construct padding
            pad_left = start_idx
            pad_right = target_len - (start_idx + seg_len)

            # Pad segment to full length
            # padding argument for pad is [[top, bottom], [left, right], ...]
            # batch dim: [0, 0], time dim: [pad_left, pad_right], channel dim: [0, 0]

            padded_segment = ops.pad(
                weighted_segment,
                [[0, 0], [pad_left, pad_right], [0, 0]]
            )

            output = output + padded_segment

        return output

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None,
            mask: Optional[keras.KerasTensor] = None
    ) -> keras.KerasTensor:
        """
        Process input through the hierarchical time tree.

        :param inputs: Input tensor of shape [batch, seq_len, channels].
        :type inputs: keras.KerasTensor
        :param training: Training mode flag.
        :type training: Optional[bool]
        :param mask: Optional mask.
        :type mask: Optional[keras.KerasTensor]
        :return: Processed tensor of shape [batch, seq_len, channels].
        :rtype: keras.KerasTensor
        """
        target_len = ops.shape(inputs)[1]
        current = inputs

        node_idx_counter = 0

        # Process through each level of the tree
        for level in range(self.tree_depth + 1):
            num_segments = 2 ** level

            # Get nodes for this level
            level_nodes = self.all_nodes[node_idx_counter: node_idx_counter + num_segments]
            node_idx_counter += num_segments

            # Split into segments
            segments = self._split_with_overlap(current, num_segments)

            # Split masks if present
            # Note: We don't propagate mask logic deeply into splitting because
            # re-stitching masks with crossfade is ambiguous.
            # We rely on PRISMNode's internal stability fixes (safe std) to handle
            # zero-padded segments that might result from splitting.

            # Process each segment with its corresponding node
            processed_segments = []
            for segment, node in zip(segments, level_nodes):
                processed = node(segment, training=training)
                processed_segments.append(processed)

            # Stitch segments back together
            current = self._stitch_with_crossfade(processed_segments, target_len)

        return current

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        :param input_shape: Input shape tuple.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple (same as input).
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            "tree_depth": self.tree_depth,
            "overlap_ratio": self.overlap_ratio,
            "num_wavelet_levels": self.num_wavelet_levels,
            "router_hidden_dim": self.router_hidden_dim,
            "router_temperature": self.router_temperature,
            "dropout_rate": self.dropout_rate,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
        })
        return config


# ---------------------------------------------------------------------
# PRISM Layer (Main Interface)
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class PRISMLayer(keras.layers.Layer):
    """
    Main PRISM layer combining hierarchical time-frequency decomposition.

    Provides the complete PRISM processing pipeline including optional
    projection layers and residual connections. The input passes through the
    PRISMTimeTree for hierarchical wavelet processing, followed by dropout,
    an optional residual connection, and optional output layer normalization.

    **Architecture Overview:**

    .. code-block:: text

        Input: [batch, context_len, channels]
                       │
                       ▼
        ┌──────────────────────────────┐
        │  PRISMTimeTree               │
        │  (hierarchical wavelet       │
        │   decomposition + routing)   │
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │  Dropout                     │
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │  Residual: output + input    │ ← (if use_residual=True)
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │  LayerNormalization          │ ← (if use_output_norm=True)
        └──────────────┬───────────────┘
                       │
                       ▼
        Output: [batch, context_len, channels]

    :param tree_depth: Depth of the binary time tree.
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
    :param use_residual: Whether to use residual connection.
        Defaults to True.
    :type use_residual: bool
    :param use_output_norm: Whether to apply output normalization.
        Defaults to True.
    :type use_output_norm: bool
    :param kernel_initializer: Initializer for kernel weights.
        Defaults to "glorot_uniform".
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for kernel weights.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional arguments for the Layer base class.
    """

    def __init__(
            self,
            tree_depth: int = 2,
            overlap_ratio: float = 0.25,
            num_wavelet_levels: int = 3,
            router_hidden_dim: int = 64,
            router_temperature: float = 1.0,
            dropout_rate: float = 0.1,
            use_residual: bool = True,
            use_output_norm: bool = True,
            kernel_initializer: Union[str, initializers.Initializer] = "glorot_uniform",
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.tree_depth = tree_depth
        self.overlap_ratio = overlap_ratio
        self.num_wavelet_levels = num_wavelet_levels
        self.router_hidden_dim = router_hidden_dim
        self.router_temperature = router_temperature
        self.dropout_rate = dropout_rate
        self.use_residual = use_residual
        self.use_output_norm = use_output_norm
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.supports_masking = True

        # Time tree processing
        self.time_tree = PRISMTimeTree(
            tree_depth=tree_depth,
            overlap_ratio=overlap_ratio,
            num_wavelet_levels=num_wavelet_levels,
            router_hidden_dim=router_hidden_dim,
            router_temperature=router_temperature,
            dropout_rate=dropout_rate,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name=f"{self.name}_time_tree"
        )

        # Output normalization (always created for weight compatibility)
        self.output_norm = layers.LayerNormalization(
            epsilon=1e-6,
            name=f"{self.name}_output_norm"
        )

        # Dropout
        self.dropout = layers.Dropout(
            rate=dropout_rate,
            name=f"{self.name}_dropout"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and initialize sub-layer weights.

        :param input_shape: Input shape tuple.
        :type input_shape: Tuple[Optional[int], ...]
        """
        self.time_tree.build(input_shape)
        self.output_norm.build(input_shape)
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None,
            mask: Optional[keras.KerasTensor] = None
    ) -> keras.KerasTensor:
        """
        Apply PRISM processing to the input sequence.

        :param inputs: Input tensor of shape [batch, seq_len, channels].
        :type inputs: keras.KerasTensor
        :param training: Training mode flag.
        :type training: Optional[bool]
        :param mask: Optional mask.
        :type mask: Optional[keras.KerasTensor]
        :return: Processed tensor of shape [batch, seq_len, channels].
        :rtype: keras.KerasTensor
        """
        # Process through time tree
        x = self.time_tree(inputs, training=training, mask=mask)

        # Apply dropout
        x = self.dropout(x, training=training)

        # Residual connection
        if self.use_residual:
            x = x + inputs

        # Output normalization
        if self.use_output_norm:
            x = self.output_norm(x)

        return x

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        :param input_shape: Input shape tuple.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple (same as input).
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            "tree_depth": self.tree_depth,
            "overlap_ratio": self.overlap_ratio,
            "num_wavelet_levels": self.num_wavelet_levels,
            "router_hidden_dim": self.router_hidden_dim,
            "router_temperature": self.router_temperature,
            "dropout_rate": self.dropout_rate,
            "use_residual": self.use_residual,
            "use_output_norm": self.use_output_norm,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
        })
        return config

# ---------------------------------------------------------------------
