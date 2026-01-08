"""
PRISM: Partitioned Representations for Iterative Sequence Modeling.

This module implements the PRISM architecture for time-series forecasting,
which combines hierarchical time decomposition with multi-resolution
frequency analysis using Haar wavelets.

The architecture uses a "Split-Transform-Weight-Merge" philosophy applied
recursively to capture both global trends and local fine-grained structures.
"""

import keras
from keras import ops
from keras import layers
from keras import initializers
from keras import regularizers
from typing import Optional, Union, Tuple, List, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.ffn import create_ffn_layer


# ---------------------------------------------------------------------
# Haar Wavelet Decomposition Layer
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class HaarWaveletDecomposition(keras.layers.Layer):
    """
    Performs Haar Discrete Wavelet Transform (DWT) decomposition.

    Decomposes an input signal into multiple frequency bands using the Haar
    wavelet basis. The Haar wavelet is the simplest wavelet and computes
    averages (approximation) and differences (detail) coefficients.

    **Architecture**::

        Input(shape=[batch, seq_len, channels])
               ↓
        For each decomposition level:
               ↓
          +----+----+
          ↓         ↓
        Low-pass  High-pass
        (approx)  (detail)
          ↓         ↓
          +----+----+
               ↓
        Output: List of [approx, detail_1, ..., detail_K]

    :param num_levels: Number of decomposition levels. Each level halves
        the temporal resolution. Defaults to 3.
    :type num_levels: int
    :param kwargs: Additional arguments for the Layer base class.
    """

    def __init__(
        self,
        num_levels: int = 3,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if num_levels < 1:
            raise ValueError(
                f"num_levels must be >= 1, got {num_levels}"
            )

        self.num_levels = num_levels

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> List[keras.KerasTensor]:
        """
        Apply Haar DWT decomposition.

        :param inputs: Input tensor of shape [batch, seq_len, channels].
        :type inputs: keras.KerasTensor
        :param training: Training mode flag (unused).
        :type training: Optional[bool]
        :return: List of frequency bands [approx, detail_1, ..., detail_K].
        :rtype: List[keras.KerasTensor]
        """
        details = []
        approximation = inputs

        for _ in range(self.num_levels):
            seq_len = ops.shape(approximation)[1]
            # Ensure even length by truncating if necessary
            even_len = (seq_len // 2) * 2

            # Slice to even length
            x = approximation[:, :even_len, :]

            # Reshape for pairwise operations: [batch, seq_len//2, 2, channels]
            x_reshaped = ops.reshape(
                x,
                (ops.shape(x)[0], even_len // 2, 2, ops.shape(x)[-1])
            )

            # Haar wavelet coefficients (normalized)
            # Low-pass (approximation): (x[2k] + x[2k+1]) / sqrt(2)
            # High-pass (detail): (x[2k] - x[2k+1]) / sqrt(2)
            sqrt2 = ops.sqrt(ops.cast(2.0, x.dtype))
            low_pass = (x_reshaped[:, :, 0, :] + x_reshaped[:, :, 1, :]) / sqrt2
            high_pass = (x_reshaped[:, :, 0, :] - x_reshaped[:, :, 1, :]) / sqrt2

            details.append(high_pass)
            approximation = low_pass

        # Return [approximation, detail_L, detail_L-1, ..., detail_1]
        # Reverse details so coarsest detail is first
        return [approximation] + details[::-1]

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> List[Tuple[Optional[int], ...]]:
        """
        Compute output shapes for all frequency bands.

        :param input_shape: Input shape tuple.
        :type input_shape: Tuple[Optional[int], ...]
        :return: List of output shapes for each band.
        :rtype: List[Tuple[Optional[int], ...]]
        """
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        channels = input_shape[2]

        shapes = []
        current_len = seq_len

        for _ in range(self.num_levels):
            if current_len is not None:
                current_len = (current_len // 2) * 2 // 2

        # Approximation shape
        shapes.append((batch_size, current_len, channels))

        # Detail shapes (from coarsest to finest)
        detail_len = current_len
        for _ in range(self.num_levels):
            shapes.append((batch_size, detail_len, channels))
            if detail_len is not None:
                detail_len = detail_len * 2

        return shapes

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            "num_levels": self.num_levels,
        })
        return config


# ---------------------------------------------------------------------
# Frequency Band Statistics Layer
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class FrequencyBandStatistics(keras.layers.Layer):
    """
    Computes summary statistics for frequency bands.

    Extracts statistical features from each frequency band including mean,
    standard deviation, min, max, and temporal derivatives. These statistics
    serve as input to the importance router.

    **Architecture**::

        Input: frequency band [batch, seq_len, channels]
               |
        Compute: mean, std, min, max, diff_mean, diff_std
               |
        Output: statistics [batch, channels, num_stats]

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
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Compute statistics for the input frequency band.

        :param inputs: Input tensor of shape [batch, seq_len, channels].
        :type inputs: keras.KerasTensor
        :param training: Training mode flag (unused).
        :type training: Optional[bool]
        :return: Statistics tensor of shape [batch, channels, num_stats].
        :rtype: keras.KerasTensor
        """
        # Basic statistics along time axis
        mean = ops.mean(inputs, axis=1)  # [batch, channels]
        std = ops.std(inputs, axis=1) + self.epsilon
        min_val = ops.min(inputs, axis=1)
        max_val = ops.max(inputs, axis=1)

        # Temporal derivatives (first difference)
        diff = inputs[:, 1:, :] - inputs[:, :-1, :]
        diff_mean = ops.mean(diff, axis=1)
        diff_std = ops.std(diff, axis=1) + self.epsilon

        # Stack statistics: [batch, channels, num_stats]
        stats = ops.stack(
            [mean, std, min_val, max_val, diff_mean, diff_std],
            axis=-1
        )
        return stats

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
    normalized via temperature-scaled softmax.

    **Architecture**::

        Input: List of frequency bands [band_1, ..., band_K]
               |
        For each band: compute statistics
               |
        For each band: MLP(statistics) -> score
               |
        Softmax(scores / temperature) -> weights
               |
        Output: weights [batch, channels, num_bands]

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
        Build the layer.

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

        # Build router MLP
        channels = first_band_shape[-1]
        stats_shape = (first_band_shape[0], channels, 6)
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
        Compute output shape.

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

    Processes a time segment by:
    1. Decomposing into frequency bands via Haar DWT
    2. Computing importance weights via the router
    3. Reconstructing a weighted representation

    **Architecture**::

        Input: time segment [batch, seq_len, channels]
               |
        HaarWaveletDecomposition -> [approx, detail_1, ..., detail_K]
               |
        FrequencyBandRouter -> weights [batch, channels, K+1]
               |
        Weighted sum of bands (interpolated to common length)
               |
        Output: processed segment [batch, seq_len, channels]

    :param num_wavelet_levels: Number of Haar DWT decomposition levels.
        Defaults to 3.
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

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer.

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
        # Use ops.cond for graph-safe conditional
        def do_interpolate():
            # Linear indices for target positions
            # Map [0, target_len-1] -> [0, band_len-1]
            target_indices = ops.cast(
                ops.arange(target_len),
                band.dtype
            )
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
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Process input through wavelet decomposition and weighted reconstruction.

        :param inputs: Input tensor of shape [batch, seq_len, channels].
        :type inputs: keras.KerasTensor
        :param training: Training mode flag.
        :type training: Optional[bool]
        :return: Processed tensor of shape [batch, seq_len, channels].
        :rtype: keras.KerasTensor
        """
        target_len = ops.shape(inputs)[1]

        # Decompose into frequency bands
        bands = self.wavelet(inputs, training=training)

        # Compute importance weights
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
        Compute output shape.

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

    Builds a binary tree over the time domain by recursively splitting
    the signal into overlapping segments. Each node processes its segment
    through wavelet decomposition and adaptive weighting.

    **Architecture**::

        Input: [batch, T, channels]
               |
        Level 0: Full sequence -> PRISMNode
               |
        Level 1: Split into 2 overlapping segments -> 2x PRISMNode
               |
        Level 2: Split into 4 overlapping segments -> 4x PRISMNode
               |
        ... (up to tree_depth levels)
               |
        Stitch segments back with cross-fade
               |
        Output: [batch, T, channels]

    :param tree_depth: Depth of the binary time tree. Depth 0 means single
        node (no splitting). Defaults to 2.
    :type tree_depth: int
    :param overlap_ratio: Ratio of overlap between adjacent segments.
        Value in [0, 0.5). Defaults to 0.25.
    :type overlap_ratio: float
    :param num_wavelet_levels: Number of Haar DWT levels per node.
        Defaults to 3.
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

        # Create PRISM nodes for each level of the tree
        # Level l has 2^l nodes
        self.prism_nodes: List[List[PRISMNode]] = []
        for level in range(tree_depth + 1):
            num_nodes = 2 ** level
            level_nodes = []
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
                level_nodes.append(node)
            self.prism_nodes.append(level_nodes)

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build all PRISM nodes.

        :param input_shape: Input shape tuple.
        :type input_shape: Tuple[Optional[int], ...]
        """
        seq_len = input_shape[1]
        batch_size = input_shape[0]
        channels = input_shape[2]

        # Build nodes for each level
        for level, level_nodes in enumerate(self.prism_nodes):
            num_segments = 2 ** level
            if seq_len is not None:
                # Compute segment length with overlap
                overlap_size = int(seq_len * self.overlap_ratio / num_segments)
                segment_len = (seq_len + overlap_size * (num_segments - 1)) // num_segments
                segment_shape = (batch_size, segment_len, channels)
            else:
                segment_shape = (batch_size, None, channels)

            for node in level_nodes:
                node.build(segment_shape)

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

        seq_len = ops.shape(x)[1]
        seq_len_float = ops.cast(seq_len, x.dtype)

        # Calculate segment parameters
        overlap_size = ops.cast(
            seq_len_float * self.overlap_ratio / ops.cast(num_segments, x.dtype),
            "int32"
        )

        # Non-overlapping part per segment
        non_overlap_len = (seq_len - overlap_size * (num_segments - 1)) // num_segments
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
        Stitch segments back together using linear cross-fade.

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
        seq_len_float = ops.cast(target_len, segments[0].dtype)

        # Calculate overlap parameters
        overlap_size = ops.cast(
            seq_len_float * self.overlap_ratio / ops.cast(num_segments, segments[0].dtype),
            "int32"
        )
        non_overlap_len = (target_len - overlap_size * (num_segments - 1)) // num_segments

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
                fade_in = ops.linspace(0.0, 1.0, overlap_size)
                fade_in = ops.reshape(fade_in, (1, overlap_size, 1))
                # Apply fade in to first overlap_size positions
                mask_before = ops.zeros((1, overlap_size, 1), dtype=segment.dtype)
                mask_after = ops.ones((1, seg_len - overlap_size, 1), dtype=segment.dtype)
                fade_mask = ops.concatenate([fade_in, mask_after], axis=1)
                weights = weights * fade_mask

            # Fade out at end (except last segment)
            if i < num_segments - 1:
                fade_out = ops.linspace(1.0, 0.0, overlap_size)
                fade_out = ops.reshape(fade_out, (1, overlap_size, 1))
                mask_before = ops.ones((1, seg_len - overlap_size, 1), dtype=segment.dtype)
                fade_mask = ops.concatenate([mask_before, fade_out], axis=1)
                weights = weights * fade_mask

            # Add weighted segment to output
            weighted_segment = segment * weights
            indices = ops.arange(start_idx, start_idx + seg_len)
            indices = ops.expand_dims(indices, 0)
            indices = ops.broadcast_to(indices, (batch_size, seg_len))

            # Use scatter_update pattern
            output = ops.slice_update(
                output,
                [0, start_idx, 0],
                output[:, start_idx:start_idx + seg_len, :] + weighted_segment
            )

        return output

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Process input through the hierarchical time tree.

        :param inputs: Input tensor of shape [batch, seq_len, channels].
        :type inputs: keras.KerasTensor
        :param training: Training mode flag.
        :type training: Optional[bool]
        :return: Processed tensor of shape [batch, seq_len, channels].
        :rtype: keras.KerasTensor
        """
        target_len = ops.shape(inputs)[1]
        current = inputs

        # Process through each level of the tree
        for level, level_nodes in enumerate(self.prism_nodes):
            num_segments = 2 ** level

            # Split into segments
            segments = self._split_with_overlap(current, num_segments)

            # Process each segment with its corresponding node
            processed_segments = []
            for seg_idx, (segment, node) in enumerate(zip(segments, level_nodes)):
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
        Compute output shape.

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
    projection layers and residual connections.

    **Architecture**::

        Input: [batch, context_len, channels]
               |
        (optional) Input projection: Linear -> channels
               |
        PRISMTimeTree: hierarchical wavelet processing
               |
        (optional) Residual connection: output + input
               |
        (optional) Output normalization
               |
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
        Build the layer.

        :param input_shape: Input shape tuple.
        :type input_shape: Tuple[Optional[int], ...]
        """
        self.time_tree.build(input_shape)
        self.output_norm.build(input_shape)
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply PRISM processing.

        :param inputs: Input tensor of shape [batch, seq_len, channels].
        :type inputs: keras.KerasTensor
        :param training: Training mode flag.
        :type training: Optional[bool]
        :return: Processed tensor of shape [batch, seq_len, channels].
        :rtype: keras.KerasTensor
        """
        # Process through time tree
        x = self.time_tree(inputs, training=training)

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
        Compute output shape.

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
# PRISM Forecasting Model
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class PRISMModel(keras.Model):
    """
    Complete PRISM model for time series forecasting.

    Combines hierarchical time-frequency decomposition with a forecasting
    head to predict future values of a time series.

    **Architecture**::

        Input: context window [batch, context_len, num_features]
               |
        (optional) Input embedding: Linear -> hidden_dim
               |
        num_layers × PRISMLayer (stacked)
               |
        Flatten: [batch, context_len * hidden_dim]
               |
        Forecasting MLP: hidden -> output
               |
        Reshape: [batch, forecast_len, num_features]
               |
        Output: forecast [batch, forecast_len, num_features]

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

        # Forecasting head
        head_hidden_dim = self.hidden_dim * ffn_expansion
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

        # Build forecast head
        flat_dim = self.context_len * self.hidden_dim
        self.forecast_head.build((batch_size, flat_dim))

        super().build(input_shape)

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
        :return: Forecast tensor of shape [batch, forecast_len, num_features].
        :rtype: keras.KerasTensor
        """
        # Project input to hidden dimension
        x = self.input_projection(inputs)

        # Process through PRISM layers
        for layer in self.prism_layers:
            x = layer(x, training=training)

        # Flatten
        x = self.flatten(x)

        # Generate forecast
        x = self.forecast_head(x, training=training)

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