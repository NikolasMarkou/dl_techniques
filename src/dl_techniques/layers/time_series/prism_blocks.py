"""
PRISM: Partitioned Representations for Iterative Sequence Modeling.

PRISM is a time-series backbone. It splits a sequence into overlapping
segments, decomposes each segment into Haar wavelet frequency bands, weights
the bands with a learned router, and stitches the segments back together.

The pipeline is split-transform-weight-merge, applied once per tree level.
Each level re-splits the FULL re-stitched sequence, so level ``i`` sees
``2 ** i`` segments of the whole signal rather than the children of level
``i - 1``.

This module exports five layers, smallest first:

- :class:`FrequencyBandStatistics` -- six summary statistics per band.
- :class:`FrequencyBandRouter` -- band importance weights from those
  statistics.
- :class:`PRISMNode` -- wavelet decomposition plus routed recombination for
  one segment.
- :class:`PRISMTimeTree` -- the level loop over split, node, stitch.
- :class:`PRISMLayer` -- the tree plus dropout, residual and output norm.
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
from dl_techniques.utils.keras_registration import register_dl_technique


# ---------------------------------------------------------------------
# Frequency Band Statistics Layer
# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.time_series.prism_blocks")
class FrequencyBandStatistics(keras.layers.Layer):
    """
    Compute six summary statistics for one frequency band.

    The statistics are mean, standard deviation, min, max, and the mean and
    standard deviation of the first difference along time. They are the input
    the importance router scores each band on. The layer owns no weights.

    Standard deviation is ``sqrt(var + epsilon)``, not ``ops.std``. The
    gradient of ``ops.std`` contains ``1 / (2 * std)`` and explodes on a
    constant sequence or a zero-padded region.

    **Architecture Overview:**

    .. code-block:: text

        Input: frequency band [batch, seq_len, channels]
                        │
                        ▼
        ┌───────────────────────────────────────┐
        │  Per-channel reduction over time      │
        │  mean, std=sqrt(var+eps), min, max    │
        │  diff = x[:, 1:] - x[:, :-1]          │
        │  diff_mean, diff_std                  │
        └───────────────────┬───────────────────┘
                            │  [batch, channels] x 6
                            ▼
        ┌───────────────────────────────────────┐
        │  Stack on a new last axis             │
        └───────────────────┬───────────────────┘
                            │
                            ▼  ('static_len is None' only)
        ┌───────────────────────────────────────┐
        │  Zero every non-finite statistic that │
        │  came from an all-finite series       │
        └───────────────────┬───────────────────┘
                            │
                            ▼
        Output: statistics [batch, channels, 6]

    The repair stage runs only when the time axis is unknown at trace time.
    On a statically shaped input it is skipped entirely.

    :param epsilon: Constant added to the variance before the square root.
        Defaults to 1e-6.
    :type epsilon: float
    :param kwargs: Additional arguments for the Layer base class.

    Input shape:
        3D tensor of shape ``[batch, seq_len, channels]``.

    Output shape:
        3D tensor of shape ``[batch, channels, 6]``.

    Example:
        .. code-block:: python

            stats = FrequencyBandStatistics(epsilon=1e-6)
            out = stats(keras.random.normal((2, 16, 3)))
            # out.shape == (2, 3, 6)

    :ivar epsilon: The variance floor passed to the constructor.
    :vartype epsilon: float
    """

    def __init__(
            self,
            epsilon: float = 1e-6,
            **kwargs: Any
    ) -> None:
        """
        Store the variance floor and the fixed statistic count.

        :param epsilon: Constant added to the variance before the square root.
            Defaults to 1e-6.
        :type epsilon: float
        :param kwargs: Additional arguments for the Layer base class.
        """
        super().__init__(**kwargs)
        self.epsilon = epsilon
        # The six statistics, in output order: mean, std, min, max,
        # diff_mean, diff_std.
        self._num_stats = 6

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
        :param mask: Optional mask tensor. Unused. Padded zeros are handled by
            the ``sqrt(var + epsilon)`` form instead.
        :type mask: Optional[keras.KerasTensor]
        :return: Statistics tensor of shape [batch, channels, num_stats].
            Two degenerate band lengths get defined values. A band of length 1
            gets ``diff_mean`` and ``diff_std`` of exactly ``0.0``, because the
            first difference of a single sample does not exist. A band of
            length 0 gets all six statistics as ``0.0``. A traced graph over
            ``[batch, None, channels]`` reaches the same values by zeroing
            non-finite statistics instead. On a statically shaped input a
            genuine NaN in the data still propagates.
        :rtype: keras.KerasTensor
        """
        # DECISION plan-2026-08-18T073231-52a93f8c/D-004
        # Branch on the STATIC length. A length-1 band gives an empty diff, so
        # mean/var go NaN silently; a length-0 band makes ops.min/max raise.
        # Do NOT rewrite as ops.cond on ops.shape(inputs)[1] -- that raises
        # OperatorNotAllowedInGraphError under tf.function. See decisions.md D-004.
        static_len = inputs.shape[1]

        if static_len is not None and static_len == 0:
            # Nothing is defined over an empty time axis. ``ops.sum`` over a
            # zero-length axis is exactly 0.0 and, unlike ``ops.min``/``ops.max``,
            # does not raise -- it also carries the dynamic batch dimension.
            # Shape [batch, channels], values exactly 0.0.
            zeros = ops.sum(inputs, axis=1)
            return ops.stack([zeros] * self._num_stats, axis=-1)

        # Reductions along the time axis, each giving [batch, channels].
        mean = ops.mean(inputs, axis=1)

        # Std is sqrt(var + epsilon), not ops.std: the ops.std gradient
        # contains 1/(2*std) and explodes when std is 0.
        variance = ops.var(inputs, axis=1)
        std = ops.sqrt(variance + self.epsilon)

        min_val = ops.min(inputs, axis=1)
        max_val = ops.max(inputs, axis=1)

        if static_len is not None and static_len == 1:
            # mean/std/min/max are well defined on a single sample. Only the
            # first difference is not, and 0.0 is its defined stand-in.
            diff_mean = ops.zeros_like(mean)
            diff_std = ops.zeros_like(mean)
        else:
            # First difference along time.
            diff = inputs[:, 1:, :] - inputs[:, :-1, :]
            diff_mean = ops.mean(diff, axis=1)

            # Same sqrt(var + epsilon) form as std above, for the same reason.
            diff_variance = ops.var(diff, axis=1)
            diff_std = ops.sqrt(diff_variance + self.epsilon)

        # Stack to [batch, channels, num_stats].
        stats = ops.stack(
            [mean, std, min_val, max_val, diff_mean, diff_std],
            axis=-1
        )

        # DECISION plan-2026-08-18T111512-29569f8b/D-001
        # Repair the degenerate cases at VALUE level when the time axis is
        # unknown. MEASURED: a tree_depth=3 PRISMModel traced at [None, None, 7]
        # gave nan_frac 1.0 where the same model gave 0.0 eager. Do NOT hoist
        # this out of the `static_len is None` guard. See decisions.md D-001.
        if static_len is None:
            # DECISION plan-2026-08-18T140459-7991552f/D-050
            # Repair only statistics from an all-finite series, per (batch,
            # channel). MEASURED: a [2,16,3] batch with one NaN and one +inf gave
            # 9 NaN statistics static but 0 under tf.function -- corruption read
            # as zeros. Do NOT use an unconditional ops.where. See decisions.md D-050.

            # Shape [batch, channels].
            series_is_finite = ops.all(ops.isfinite(inputs), axis=1)
            # Shape [batch, channels, 1], broadcasting over the statistic axis.
            repairable = ops.expand_dims(series_is_finite, axis=-1)
            stats = ops.where(
                ops.logical_and(ops.logical_not(ops.isfinite(stats)), repairable),
                ops.zeros_like(stats),
                stats,
            )

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
        """
        Return the constructor arguments needed to rebuild this layer.

        :return: Serializable configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "epsilon": self.epsilon,
        })
        return config


# ---------------------------------------------------------------------
# Frequency Band Router Layer
# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.time_series.prism_blocks")
class FrequencyBandRouter(keras.layers.Layer):
    """
    Score frequency bands and turn the scores into importance weights.

    Each band is reduced to six statistics, layer-normalized, and scored by a
    small MLP. The MLP is SHARED across bands: one instance is called once per
    band, so band count does not change the parameter count. The scores are
    concatenated and normalized jointly:

        weight_k = softmax(score_k / temperature)

    Because the softmax is joint, a NaN score in one band would spread to
    every band. That is why :class:`FrequencyBandStatistics` defines values for
    degenerate band lengths instead of letting NaN through.

    **Architecture Overview:**

    .. code-block:: text

        Input: list of K bands, band_k [B, len_k, C]
                       │
                       ▼  (loop over k, weights shared)
        ┌──────────────────────────────────────┐
        │  FrequencyBandStatistics             │
        │            │  [B, C, 6]              │
        │            ▼                         │
        │  LayerNormalization (axis=-1)        │
        │            │  [B, C, 6]              │
        │            ▼                         │
        │  MLP (hidden_dim, gelu, out 1)       │
        └──────────────┬───────────────────────┘
                       │  score_k [B, C, 1]
                       ▼
        ┌──────────────────────────────────────┐
        │  Concatenate on the last axis        │
        └──────────────┬───────────────────────┘
                       │  [B, C, K]
                       ▼
        ┌──────────────────────────────────────┐
        │  Softmax(scores / temperature)       │
        └──────────────┬───────────────────────┘
                       │
                       ▼
        Output: weights [B, C, K], summing to 1 over K

    :param hidden_dim: Hidden dimension of the router MLP.
        Defaults to 64.
    :type hidden_dim: int
    :param temperature: Divisor applied to the scores before the softmax.
        Lower values give a sharper distribution. Defaults to 1.0.
    :type temperature: float
    :param dropout_rate: Dropout rate inside the router MLP.
        Defaults to 0.1.
    :type dropout_rate: float
    :param kernel_initializer: Initializer for kernel weights.
        Defaults to "glorot_uniform".
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for kernel weights.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional arguments for the Layer base class.

    :raises ValueError: If ``hidden_dim`` is not > 0, or ``temperature`` is
        not > 0.

    Input shape:
        A non-empty list of 3D tensors ``[batch, band_len_k, channels]``. The
        band lengths may differ; the channel count may not.

    Output shape:
        3D tensor of shape ``[batch, channels, num_bands]``.

    Example:
        .. code-block:: python

            router = FrequencyBandRouter(hidden_dim=32, temperature=0.5)
            bands = [keras.random.normal((2, n, 3)) for n in (16, 8, 4)]
            w = router(bands)
            # w.shape == (2, 3, 3)

    :ivar stats_layer: The shared :class:`FrequencyBandStatistics` instance.
    :vartype stats_layer: FrequencyBandStatistics
    :ivar stats_norm: LayerNormalization over the statistic axis. It keeps the
        six statistics on a common scale, which stops the softmax saturating
        when bands carry very different magnitudes.
    :vartype stats_norm: keras.layers.LayerNormalization
    :ivar router_mlp: The scoring MLP, shared across all bands.
    :vartype router_mlp: keras.layers.Layer
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
        """
        Validate the configuration and create the three sub-layers.

        :param hidden_dim: Hidden dimension of the router MLP.
            Defaults to 64.
        :type hidden_dim: int
        :param temperature: Divisor applied to the scores before the softmax.
            Defaults to 1.0.
        :type temperature: float
        :param dropout_rate: Dropout rate inside the router MLP.
            Defaults to 0.1.
        :type dropout_rate: float
        :param kernel_initializer: Initializer for kernel weights.
            Defaults to "glorot_uniform".
        :type kernel_initializer: Union[str, keras.initializers.Initializer]
        :param kernel_regularizer: Optional regularizer for kernel weights.
        :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
        :param kwargs: Additional arguments for the Layer base class.
        :raises ValueError: If ``hidden_dim`` or ``temperature`` is not > 0.
        """
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

        # One statistics instance, reused for every band.
        self.stats_layer = FrequencyBandStatistics(name=f"{self.name}_stats")

        # Normalize the statistics before the MLP. Bands carry very different
        # magnitudes (padded zeros against real data), and without this the
        # joint softmax saturates.
        self.stats_norm = layers.LayerNormalization(
            name=f"{self.name}_stats_norm",
            axis=-1
        )

        # One MLP, called once per band, so the scoring weights are shared.
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
        Build the three sub-layers.

        Only the FIRST band shape is used. The statistics layer has no weights
        and reduces the time axis away, so the norm and the MLP see
        ``[batch, channels, 6]`` whatever the band lengths are.

        :param input_shape: List of input shapes, one per frequency band.
        :type input_shape: List[Tuple[Optional[int], ...]]
        :raises ValueError: If ``input_shape`` is not a non-empty list.
        """
        if not isinstance(input_shape, list) or len(input_shape) == 0:
            raise ValueError(
                "input_shape must be a non-empty list of shapes"
            )

        # The statistics layer is built on the first band shape.
        first_band_shape = input_shape[0]
        self.stats_layer.build(first_band_shape)

        # The norm and the MLP both operate on the statistics shape.
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
        Score every band and normalize the scores across bands.

        :param inputs: List of frequency band tensors, each
            ``[batch, band_len, channels]``.
        :type inputs: List[keras.KerasTensor]
        :param training: Training mode flag, forwarded to the MLP dropout.
        :type training: Optional[bool]
        :return: Importance weights of shape [batch, channels, num_bands],
            summing to 1 over the band axis.
        :rtype: keras.KerasTensor
        """
        scores = []
        for band in inputs:
            stats = self.stats_layer(band, training=training)

            stats = self.stats_norm(stats)

            # Shape [batch, channels, 1].
            score = self.router_mlp(stats, training=training)
            scores.append(score)

        # Shape [batch, channels, num_bands].
        scores = ops.concatenate(scores, axis=-1)

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
        """
        Return the constructor arguments needed to rebuild this layer.

        :return: Serializable configuration dictionary.
        :rtype: Dict[str, Any]
        """
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

@register_dl_technique("dl_techniques.layers.time_series.prism_blocks")
class PRISMNode(keras.layers.Layer):
    """
    Decompose one time segment into bands and recombine them by weight.

    The segment goes through a Haar DWT, giving one detail band per level plus
    a final approximation band. The router scores those bands. Every band is
    then linearly interpolated back to the input length and summed with its
    own weight. The output has the same shape as the input, so the node is a
    length-preserving transform, not a downsampler.

    The band weight is per (batch, channel) and is broadcast over time: a
    channel gets one weight per band for the whole segment.

    **Architecture Overview:**

    .. code-block:: text

        Input: time segment [B, seq_len, C]
                        │
                        ▼
        ┌────────────────────────────────────────┐
        │  HaarWaveletDecomposition (K levels)   │
        └────────────────┬───────────────────────┘
                         │ K+1 bands, band_k [B, len_k, C]
                 ┌───────┴────────┐
                 │                │
                 ▼                ▼
        ┌─────────────────┐  ┌──────────────────────┐
        │ _interpolate_   │  │ FrequencyBandRouter  │
        │ band to seq_len │  └──────────┬───────────┘
        └────────┬────────┘             │ [B, C, K+1]
                 │ [B, seq_len, C]      │
                 └───────┬──────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────┐
        │  sum_k band_k * weight[:, :, k]        │
        │  weight broadcast over the time axis   │
        └────────────────┬───────────────────────┘
                         │
                         ▼
        Output: processed [B, seq_len, C]

    The router reads the DECOMPOSED bands, not the raw input, so both arrows
    out of the decomposition carry the same K+1 band tensors.

    :param num_wavelet_levels: Number of Haar DWT levels. This gives
        ``num_wavelet_levels + 1`` bands: one detail band per level plus the
        final approximation band. Defaults to 3. Each level floor-halves the
        length, so the deepest band is ``seq_len // 2 ** num_wavelet_levels``
        long. At length 1 that band is statistically degenerate and at length 0
        the configuration is unusable. See :class:`FrequencyBandStatistics` and
        ``PRISMModel.__init__``, which rejects such combinations.
    :type num_wavelet_levels: int
    :param router_hidden_dim: Hidden dimension for the router MLP.
        Defaults to 64.
    :type router_hidden_dim: int
    :param router_temperature: Temperature for the router softmax.
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

    Input shape:
        3D tensor of shape ``[batch, seq_len, channels]``.

    Output shape:
        3D tensor of shape ``[batch, seq_len, channels]``, same as the input.

    Example:
        .. code-block:: python

            node = PRISMNode(num_wavelet_levels=2, router_hidden_dim=32)
            y = node(keras.random.normal((2, 32, 3)))
            # y.shape == (2, 32, 3)

    :ivar wavelet: The Haar decomposition sub-layer.
    :vartype wavelet: HaarWaveletDecomposition
    :ivar router: The band-scoring sub-layer.
    :vartype router: FrequencyBandRouter
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
        """
        Store the configuration and create the wavelet and router sub-layers.

        :param num_wavelet_levels: Number of Haar DWT levels.
            Defaults to 3.
        :type num_wavelet_levels: int
        :param router_hidden_dim: Hidden dimension for the router MLP.
            Defaults to 64.
        :type router_hidden_dim: int
        :param router_temperature: Temperature for the router softmax.
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
        super().__init__(**kwargs)

        self.num_wavelet_levels = num_wavelet_levels
        self.router_hidden_dim = router_hidden_dim
        self.router_temperature = router_temperature
        self.dropout_rate = dropout_rate
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

        # Haar DWT: one detail band per level plus one approximation band.
        self.wavelet = HaarWaveletDecomposition(
            num_levels=num_wavelet_levels,
            name=f"{self.name}_wavelet"
        )

        # Scores the bands the decomposition above produces.
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
        Build the wavelet layer, then the router on the band shapes it emits.

        :param input_shape: Input shape tuple ``[batch, seq_len, channels]``.
        :type input_shape: Tuple[Optional[int], ...]
        """
        self.wavelet.build(input_shape)

        # The router is built on the list of band shapes, not on the input.
        band_shapes = self.wavelet.compute_output_shape(input_shape)
        self.router.build(band_shapes)

        super().build(input_shape)

    def _interpolate_band(
            self,
            band: keras.KerasTensor,
            target_len: int
    ) -> keras.KerasTensor:
        """
        Resample a band to ``target_len`` by linear interpolation over time.

        The branch is an ``ops.cond`` on the LENGTHS, which is graph-safe,
        rather than a Python ``if`` on a traced shape.

        :param band: Band tensor of shape [batch, band_len, channels].
        :type band: keras.KerasTensor
        :param target_len: Target sequence length.
        :type target_len: int
        :return: Interpolated tensor of shape [batch, target_len, channels].
            The band is returned unchanged when it is already ``target_len``
            long.
        :rtype: keras.KerasTensor
        """
        band_len = ops.shape(band)[1]

        def do_interpolate():
            # Map target position [0, target_len-1] onto [0, band_len-1].
            target_indices = ops.cast(
                ops.arange(target_len),
                band.dtype
            )
            # maximum(..., 1) keeps target_len == 1 from dividing by zero.
            scale = ops.cast(band_len - 1, band.dtype) / ops.cast(
                ops.maximum(target_len - 1, 1),
                band.dtype
            )
            source_indices = target_indices * scale

            # The two neighbours of each source position.
            floor_idx = ops.cast(ops.floor(source_indices), "int32")
            ceil_idx = ops.minimum(floor_idx + 1, band_len - 1)

            # Fractional distance to the floor neighbour, shaped to broadcast
            # over batch and channels.
            alpha = source_indices - ops.cast(floor_idx, band.dtype)
            alpha = ops.expand_dims(ops.expand_dims(alpha, 0), -1)

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
        Decompose, score the bands, and recombine them at the input length.

        :param inputs: Input tensor of shape [batch, seq_len, channels].
        :type inputs: keras.KerasTensor
        :param training: Training mode flag, forwarded to the wavelet and the
            router.
        :type training: Optional[bool]
        :param mask: Optional mask. Not forwarded. The bands are downsampled,
            so a per-timestep mask does not line up with them; padded zeros are
            absorbed by the ``sqrt(var + epsilon)`` form in
            :class:`FrequencyBandStatistics` instead.
        :type mask: Optional[keras.KerasTensor]
        :return: Processed tensor of shape [batch, seq_len, channels].
        :rtype: keras.KerasTensor
        """
        target_len = ops.shape(inputs)[1]

        bands = self.wavelet(inputs, training=training)

        # Shape [batch, channels, num_bands].
        weights = self.router(bands, training=training)

        weighted_sum = ops.zeros_like(inputs)
        for i, band in enumerate(bands):
            band_interp = self._interpolate_band(band, target_len)

            # [batch, channels] -> [batch, 1, channels], so one weight per
            # channel broadcasts across the whole time axis.
            band_weight = ops.expand_dims(weights[:, :, i], axis=1)

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
        """
        Return the constructor arguments needed to rebuild this layer.

        :return: Serializable configuration dictionary.
        :rtype: Dict[str, Any]
        """
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

@register_dl_technique("dl_techniques.layers.time_series.prism_blocks")
class PRISMTimeTree(keras.layers.Layer):
    """
    Run a PRISM node bank over the sequence once per tree level.

    Each level splits the sequence into overlapping segments, sends each
    segment through its own :class:`PRISMNode`, and stitches the results back
    to full length with a linear cross-fade over the overlaps.

    The traversal is a LOOP over levels, not a recursion over children. Level
    ``i`` re-splits the FULL, re-stitched output of level ``i - 1`` into
    ``2 ** i`` segments. It does not bisect level ``i - 1``'s segments. So the
    deepest segment length comes from ONE call to :meth:`_segment_len` at
    ``num_segments = 2 ** tree_depth``, not from ``tree_depth`` successive
    halvings. Anything reasoning about the deepest segment, such as band
    lengths or configuration validation, must use that one-shot form.

    **Architecture Overview:**

    .. code-block:: text

        Input: [B, T, C]
                       │
                       ▼
        ┌──────────────────────────────────────┐
        │  Level 0: 1 segment (no split)       │
        │    PRISMNode x1                      │
        └──────────────┬───────────────────────┘
                       │ [B, T, C]  full length again
                       ▼
        ┌──────────────────────────────────────┐
        │  Level 1: split the WHOLE sequence   │
        │    into 2 overlapping segments       │
        │    PRISMNode x2                      │
        │    stitch with cross-fade ─► [B,T,C] │
        └──────────────┬───────────────────────┘
                       │ [B, T, C]  full length again
                       ▼
        ┌──────────────────────────────────────┐
        │  Level i: split the WHOLE sequence   │
        │    into 2**i overlapping segments    │
        │    PRISMNode x 2**i                  │
        │    stitch with cross-fade ─► [B,T,C] │
        └──────────────┬───────────────────────┘
                       │ ... up to i == tree_depth
                       ▼
        Output: [B, T, C]

    Every level re-enters at full length T. Level ``i`` never receives level
    ``i - 1``'s segments.

    **Segment geometry at one level:**

    .. code-block:: text

        seq_len = 96, overlap_ratio = 0.25, num_segments = 4
        overlap_size = 6, non_overlap_len = 19, segment_len = 25

        segment   start   end   length
              0       0    25       25
              1      19    44       25
              2      38    63       25
              3      57    96       39   (runs to seq_len)

    The last segment is longer than its siblings, so the remainder the floor
    division in ``non_overlap_len`` drops is still covered. Adjacent segments
    share ``overlap_size`` positions, and the cross-fade weights over those
    positions sum to 1.

    :param tree_depth: Depth of the binary time tree. Depth 0 means one node
        and no splitting. Defaults to 2. The knob has no valid range of its
        own. Together with ``overlap_ratio`` and the input length it fixes the
        deepest segment length, which ``num_wavelet_levels`` then floor-halves
        down to the deepest band. ``PRISMModel.__init__`` rejects combinations
        whose deepest band would have length 0. This layer does not validate.
    :type tree_depth: int
    :param overlap_ratio: Overlap between adjacent segments, in [0, 0.5).
        Defaults to 0.25.
    :type overlap_ratio: float
    :param num_wavelet_levels: Number of Haar DWT levels per node, giving
        ``num_wavelet_levels + 1`` bands. Defaults to 3. Each level
        floor-halves the band length, so the deepest band of the deepest node
        is ``segment_len // 2 ** num_wavelet_levels``. This trades directly
        against ``tree_depth`` and the input length.
    :type num_wavelet_levels: int
    :param router_hidden_dim: Hidden dimension for router MLPs.
        Defaults to 64.
    :type router_hidden_dim: int
    :param router_temperature: Temperature for the router softmax.
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

    :raises ValueError: If ``tree_depth`` is negative, or ``overlap_ratio`` is
        outside [0, 0.5).

    Input shape:
        3D tensor of shape ``[batch, seq_len, channels]``.

    Output shape:
        3D tensor of shape ``[batch, seq_len, channels]``, same as the input.

    Example:
        .. code-block:: python

            tree = PRISMTimeTree(tree_depth=2, num_wavelet_levels=2)
            y = tree(keras.random.normal((2, 96, 3)))
            # y.shape == (2, 96, 3)

    :ivar all_nodes: Every node of every level in one FLAT list, ordered level
        by level. A flat list is used so Keras tracks all sub-layers for
        serialization; a nested list would lose their weights.
    :vartype all_nodes: List[PRISMNode]
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
        """
        Validate the configuration and create every node of every level.

        :param tree_depth: Depth of the binary time tree. Defaults to 2.
        :type tree_depth: int
        :param overlap_ratio: Overlap between adjacent segments, in [0, 0.5).
            Defaults to 0.25.
        :type overlap_ratio: float
        :param num_wavelet_levels: Number of Haar DWT levels per node.
            Defaults to 3.
        :type num_wavelet_levels: int
        :param router_hidden_dim: Hidden dimension for router MLPs.
            Defaults to 64.
        :type router_hidden_dim: int
        :param router_temperature: Temperature for the router softmax.
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
        :raises ValueError: If ``tree_depth`` is negative, or
            ``overlap_ratio`` is outside [0, 0.5).
        """
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

        # One node per segment per level, in a FLAT list. Keras does not track
        # sub-layers held in a nested list, and untracked layers lose their
        # weights on save.
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
    # One helper for segment geometry, reproducing the RUNTIME form
    # (seq_len - overlap_size * (n-1)) // n. build() used PLUS and disagreed:
    # MEASURED build 28 vs runtime 25 at level 2 (overlap 0.25, len 96). Keep
    # the PLUS out and keep the float round-trip. See decisions.md D-001.
    @staticmethod
    def _segment_len(
            seq_len: int,
            overlap_ratio: float,
            num_segments: int,
            dtype: str = "float32"
    ) -> Tuple[int, int, int]:
        """
        Compute the overlapping-segment geometry for one split.

        This is the single source of the segment arithmetic. :meth:`build`,
        :meth:`_split_with_overlap` and :meth:`_stitch_with_crossfade` all call
        it, so their offsets cannot drift apart. It is pure Python over ints
        and mirrors the runtime expression exactly, including the truncating
        cast of ``overlap_size`` to int32.

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
            where ``segment_len == non_overlap_len + overlap_size``. Segment
            ``i`` spans ``[i * non_overlap_len, i * non_overlap_len +
            segment_len)``. The last segment is the exception: it runs to
            ``[i * non_overlap_len, seq_len)``. See
            :meth:`_split_with_overlap`.
        :rtype: Tuple[int, int, int]
        """
        # `getattr(d, "name", None) or str(d)`, not `keras.backend.standardize_dtype`:
        # a Keras-2 residue banned across all of `src/`. Do NOT reduce it to a bare
        # `str(d)` -- a `tf.DType` stringifies as "<dtype: 'float32'>". D-007.
        float_type = np.dtype(getattr(dtype, "name", None) or str(dtype)).type
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
        Build every node at the segment shape it will be handed.

        The last node of each level is built LONGER than its siblings, because
        its segment runs to the end of the sequence. When the time axis is
        unknown, every node is built with a ``None`` length instead.

        :param input_shape: Input shape tuple ``[batch, seq_len, channels]``.
        :type input_shape: Tuple[Optional[int], ...]
        """
        seq_len = input_shape[1]
        batch_size = input_shape[0]
        channels = input_shape[2]

        node_idx_counter = 0

        for level in range(self.tree_depth + 1):
            num_nodes = 2 ** level

            if seq_len is not None:
                if num_nodes > 1:
                    # The same helper the forward pass uses, so a node is built
                    # for exactly the length it will be handed. See the
                    # _segment_len anchor.
                    non_overlap_len, _, segment_len = self._segment_len(
                        seq_len, self.overlap_ratio, num_nodes,
                        dtype=self.compute_dtype
                    )
                    segment_shape = (batch_size, segment_len, channels)
                    # The LAST node of the level also gets the remainder the
                    # floor division in non_overlap_len discards, so it is built
                    # LONGER than its siblings. See the D-014 anchor in
                    # _split_with_overlap.
                    last_segment_shape = (
                        batch_size,
                        seq_len - (num_nodes - 1) * non_overlap_len,
                        channels
                    )
                else:
                    segment_shape = (batch_size, seq_len, channels)
                    last_segment_shape = segment_shape
            else:
                segment_shape = (batch_size, None, channels)
                last_segment_shape = segment_shape

            for node_in_level in range(num_nodes):
                node_shape = (
                    last_segment_shape
                    if node_in_level == num_nodes - 1
                    else segment_shape
                )
                self.all_nodes[node_idx_counter].build(node_shape)
                node_idx_counter += 1

        super().build(input_shape)

    def _split_with_overlap(
            self,
            x: keras.KerasTensor,
            num_segments: int
    ) -> List[keras.KerasTensor]:
        """
        Split the sequence into overlapping segments.

        :param x: Input tensor of shape [batch, seq_len, channels].
        :type x: keras.KerasTensor
        :param num_segments: Number of segments to create. A value of 1
            returns the input unchanged, in a one-element list.
        :type num_segments: int
        :return: List of ``num_segments`` segment tensors. The first
            ``num_segments - 1`` are ``segment_len`` long. The last runs to the
            end of the sequence, so it is ``>= segment_len``: it absorbs the
            remainder the floor division in ``non_overlap_len`` discards.
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

        # DECISION plan-2026-08-18T140459-7991552f/D-014
        # The LAST segment runs to the END, covering the remainder that the
        # floor division discards. MEASURED at seq_len=96, overlap 0.25:
        # num_segments=4 left positions 82..95 at exactly 0.0. Do NOT make
        # end_idx uniform -- the last segment is longer. See decisions.md D-014.
        segments = []
        for i in range(num_segments):
            start_idx = i * non_overlap_len
            if i == num_segments - 1:
                segment = x[:, start_idx:, :]
            else:
                segment = x[:, start_idx:start_idx + segment_len, :]
            segments.append(segment)

        return segments

    def _stitch_with_crossfade(
            self,
            segments: List[keras.KerasTensor],
            target_len: int
    ) -> keras.KerasTensor:
        """
        Stitch segments back to full length with a linear cross-fade.

        Each segment is weighted, zero-padded to ``target_len`` and summed. A
        segment fades in over its first ``overlap_size`` positions unless it is
        the first, and fades out over its last ``overlap_size`` positions
        unless it is the last. The two ramps are complementary, so the weights
        sum to 1 at every position.

        :param segments: List of processed segment tensors. The last one may be
            longer than the others.
        :type segments: List[keras.KerasTensor]
        :param target_len: Target output length.
        :type target_len: int
        :return: Stitched tensor of shape [batch, target_len, channels].
        :rtype: keras.KerasTensor
        """
        if len(segments) == 1:
            return segments[0][:, :target_len, :]

        num_segments = len(segments)

        # The same helper _split_with_overlap uses, so the stitch offsets
        # cannot drift from the split offsets. See the _segment_len anchor.
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

        # The accumulator every padded segment is added into.
        batch_size = ops.shape(segments[0])[0]
        channels = ops.shape(segments[0])[-1]
        output = ops.zeros((batch_size, target_len, channels), dtype=segments[0].dtype)

        for i, segment in enumerate(segments):
            start_idx = i * non_overlap_len
            seg_len = ops.shape(segment)[1]

            # Start from a flat weight of 1 and multiply the ramps in.
            weights = ops.ones((1, seg_len, 1), dtype=segment.dtype)

            # Every segment but the first fades in over its leading overlap.
            if i > 0:
                # arange, not linspace: linspace needs a static 'num' and this
                # length can be a symbolic tensor.
                indices = ops.cast(ops.arange(overlap_size), segment.dtype)
                steps = ops.cast(overlap_size - 1, segment.dtype)
                # Guards overlap_size == 1, where steps would be 0.
                steps = ops.maximum(steps, 1.0)
                fade_in = indices / steps

                fade_in = ops.reshape(fade_in, (1, overlap_size, 1))

                # Ramp over the first overlap_size positions, flat after that.
                mask_after = ops.ones((1, seg_len - overlap_size, 1), dtype=segment.dtype)
                fade_mask = ops.concatenate([fade_in, mask_after], axis=1)
                weights = weights * fade_mask

            # Every segment but the last fades out over its trailing overlap.
            if i < num_segments - 1:
                indices = ops.cast(ops.arange(overlap_size), segment.dtype)
                steps = ops.cast(overlap_size - 1, segment.dtype)
                steps = ops.maximum(steps, 1.0)
                # Runs 1.0 down to 0.0, the complement of the fade in above.
                fade_out = 1.0 - (indices / steps)

                fade_out = ops.reshape(fade_out, (1, overlap_size, 1))

                mask_before = ops.ones((1, seg_len - overlap_size, 1), dtype=segment.dtype)
                fade_mask = ops.concatenate([mask_before, fade_out], axis=1)
                weights = weights * fade_mask

            weighted_segment = segment * weights

            pad_left = start_idx
            pad_right = target_len - (start_idx + seg_len)

            # ops.pad takes one [before, after] pair per axis: batch [0, 0],
            # time [pad_left, pad_right], channels [0, 0].
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
        Run split, node bank and stitch once per level.

        The loop feeds each level the previous level's re-stitched output at
        full length. Level ``i`` therefore splits the WHOLE sequence into
        ``2 ** i`` segments; it does not subdivide level ``i - 1``'s segments.

        :param inputs: Input tensor of shape [batch, seq_len, channels].
        :type inputs: keras.KerasTensor
        :param training: Training mode flag, forwarded to every node.
        :type training: Optional[bool]
        :param mask: Optional mask. Not forwarded to the nodes. A mask cannot
            be re-stitched through a cross-fade without inventing a rule for
            the blended positions, so zero-padded segments are left to the
            ``sqrt(var + epsilon)`` form in :class:`FrequencyBandStatistics`.
        :type mask: Optional[keras.KerasTensor]
        :return: Processed tensor of shape [batch, seq_len, channels].
        :rtype: keras.KerasTensor
        """
        target_len = ops.shape(inputs)[1]
        current = inputs

        node_idx_counter = 0

        for level in range(self.tree_depth + 1):
            num_segments = 2 ** level

            # all_nodes is flat, so this level's nodes are a contiguous slice.
            level_nodes = self.all_nodes[node_idx_counter: node_idx_counter + num_segments]
            node_idx_counter += num_segments

            # Split the FULL current sequence, not the previous segments.
            segments = self._split_with_overlap(current, num_segments)

            processed_segments = []
            for segment, node in zip(segments, level_nodes):
                processed = node(segment, training=training)
                processed_segments.append(processed)

            # Back to full length, ready for the next level's split.
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
        """
        Return the constructor arguments needed to rebuild this layer.

        :return: Serializable configuration dictionary.
        :rtype: Dict[str, Any]
        """
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

@register_dl_technique("dl_techniques.layers.time_series.prism_blocks")
class PRISMLayer(keras.layers.Layer):
    """
    The PRISM block: time tree, dropout, optional residual, optional norm.

    This is the layer to use in a model. It wraps :class:`PRISMTimeTree` with
    the surrounding plumbing. There is no projection: the channel count is
    unchanged from input to output.

    **Architecture Overview:**

    .. code-block:: text

        Input: [B, context_len, C] ──────────────┐
                       │                         │
                       ▼                         │
        ┌──────────────────────────────┐         │
        │  PRISMTimeTree               │         │
        └──────────────┬───────────────┘         │
                       │                         │
                       ▼                         │
        ┌──────────────────────────────┐         │
        │  Dropout(dropout_rate)       │         │
        └──────────────┬───────────────┘         │
                       │                         │
                       ▼                         │
        ┌──────────────────────────────┐         │
        │  x + input                   │◄────────┘
        └──────────────┬───────────────┘  (use_residual only)
                       │
                       ▼
        ┌──────────────────────────────┐
        │  LayerNormalization(1e-6)    │  (use_output_norm only)
        └──────────────┬───────────────┘
                       │
                       ▼
        Output: [B, context_len, C]

    ``output_norm`` is always CREATED, even when ``use_output_norm`` is False,
    so a checkpoint stays loadable when the flag is flipped. It is only called
    when the flag is True.

    :param tree_depth: Depth of the binary time tree.
        Defaults to 2.
    :type tree_depth: int
    :param overlap_ratio: Overlap between adjacent segments, in [0, 0.5).
        Defaults to 0.25.
    :type overlap_ratio: float
    :param num_wavelet_levels: Number of Haar DWT levels per node.
        Defaults to 3.
    :type num_wavelet_levels: int
    :param router_hidden_dim: Hidden dimension for routers.
        Defaults to 64.
    :type router_hidden_dim: int
    :param router_temperature: Temperature for the router softmax.
        Defaults to 1.0.
    :type router_temperature: float
    :param dropout_rate: Dropout rate, used both by the routers and by the
        dropout after the tree. Defaults to 0.1.
    :type dropout_rate: float
    :param use_residual: Add the input back after the dropout.
        Defaults to True.
    :type use_residual: bool
    :param use_output_norm: Apply the output LayerNormalization.
        Defaults to True.
    :type use_output_norm: bool
    :param kernel_initializer: Initializer for kernel weights.
        Defaults to "glorot_uniform".
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for kernel weights.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional arguments for the Layer base class.

    Input shape:
        3D tensor of shape ``[batch, context_len, channels]``.

    Output shape:
        3D tensor of shape ``[batch, context_len, channels]``, same as the
        input.

    Example:
        .. code-block:: python

            prism = PRISMLayer(tree_depth=2, num_wavelet_levels=2)
            y = prism(keras.random.normal((2, 96, 3)))
            # y.shape == (2, 96, 3)

    :ivar time_tree: The hierarchical wavelet stage.
    :vartype time_tree: PRISMTimeTree
    :ivar output_norm: Output normalization, always created and conditionally
        applied.
    :vartype output_norm: keras.layers.LayerNormalization
    :ivar dropout: Dropout applied to the tree output.
    :vartype dropout: keras.layers.Dropout
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
        """
        Store the configuration and create the tree, norm and dropout.

        :param tree_depth: Depth of the binary time tree. Defaults to 2.
        :type tree_depth: int
        :param overlap_ratio: Overlap between adjacent segments, in [0, 0.5).
            Defaults to 0.25.
        :type overlap_ratio: float
        :param num_wavelet_levels: Number of Haar DWT levels per node.
            Defaults to 3.
        :type num_wavelet_levels: int
        :param router_hidden_dim: Hidden dimension for routers.
            Defaults to 64.
        :type router_hidden_dim: int
        :param router_temperature: Temperature for the router softmax.
            Defaults to 1.0.
        :type router_temperature: float
        :param dropout_rate: Dropout rate. Defaults to 0.1.
        :type dropout_rate: float
        :param use_residual: Add the input back after the dropout.
            Defaults to True.
        :type use_residual: bool
        :param use_output_norm: Apply the output LayerNormalization.
            Defaults to True.
        :type use_output_norm: bool
        :param kernel_initializer: Initializer for kernel weights.
            Defaults to "glorot_uniform".
        :type kernel_initializer: Union[str, keras.initializers.Initializer]
        :param kernel_regularizer: Optional regularizer for kernel weights.
        :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
        :param kwargs: Additional arguments for the Layer base class.
        """
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

        # The hierarchical wavelet stage.
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

        # Created even when use_output_norm is False, so flipping the flag
        # does not change the weight set a checkpoint expects.
        self.output_norm = layers.LayerNormalization(
            epsilon=1e-6,
            name=f"{self.name}_output_norm"
        )

        # Applied to the tree output, before the residual add.
        self.dropout = layers.Dropout(
            rate=dropout_rate,
            name=f"{self.name}_dropout"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the time tree and the output norm.

        The output norm is built whatever ``use_output_norm`` says, matching
        the fact that it is always created.

        :param input_shape: Input shape tuple ``[batch, seq_len, channels]``.
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
        Run the tree, then dropout, then the two optional stages.

        :param inputs: Input tensor of shape [batch, seq_len, channels].
        :type inputs: keras.KerasTensor
        :param training: Training mode flag, forwarded to the tree and the
            dropout.
        :type training: Optional[bool]
        :param mask: Optional mask, forwarded to the tree, which does not
            propagate it into its nodes.
        :type mask: Optional[keras.KerasTensor]
        :return: Processed tensor of shape [batch, seq_len, channels].
        :rtype: keras.KerasTensor
        """
        x = self.time_tree(inputs, training=training, mask=mask)

        x = self.dropout(x, training=training)

        if self.use_residual:
            x = x + inputs

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
        """
        Return the constructor arguments needed to rebuild this layer.

        :return: Serializable configuration dictionary.
        :rtype: Dict[str, Any]
        """
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
