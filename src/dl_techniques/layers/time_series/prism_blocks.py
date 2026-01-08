"""
PRISM: Partitioned Representations for Iterative Sequence Modeling.

This module implements the PRISM architecture for time-series forecasting,
which combines hierarchical time decomposition with multi-resolution
frequency analysis using Haar wavelets.

The architecture uses a "Split-Transform-Weight-Merge" philosophy applied
recursively to capture both global trends and local fine-grained structures.
"""

import keras
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
    Computes summary statistics for frequency bands.

    Extracts statistical features from each frequency band including mean,
    standard deviation, min, max, and temporal derivatives. These statistics
    serve as input to the importance router.

    **Architecture**::

        Input: frequency band [batch, seq_len, channels]
               ↓
        Compute: mean, std, min, max, diff_mean, diff_std
               ↓
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
               ↓
        For each band: compute statistics
               ↓
        For each band: MLP(statistics) -> score
               ↓
        Softmax(scores / temperature) -> weights
               ↓
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
               ↓
        HaarWaveletDecomposition -> [approx, detail_1, ..., detail_K]
               ↓
        FrequencyBandRouter -> weights [batch, channels, K+1]
               ↓
        Weighted sum of bands (interpolated to common length)
               ↓
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
               ↓
        Level 0: Full sequence -> PRISMNode
               ↓
        Level 1: Split into 2 overlapping segments -> 2x PRISMNode
               ↓
        Level 2: Split into 4 overlapping segments -> 4x PRISMNode
               ↓
        ... (up to tree_depth levels)
               ↓
        Stitch segments back with cross-fade
               ↓
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

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build all PRISM nodes.

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
                    # Compute segment length with overlap
                    overlap_size = int(seq_len * self.overlap_ratio / num_nodes)
                    segment_len = (seq_len + overlap_size * (num_nodes - 1)) // num_nodes
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
        # Use cast to ensure we have integer overlap_size for shape logic
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

            # Slice update with symbolic indices requires explicit tensor construction
            start_indices = ops.stack([0, start_idx, 0])

            # Note: slice_update requires the update to have same rank
            # We add input segment to existing output at specific location
            # To simulate x[start:end] += val, we slice, add, and update

            # Extract current values at destination
            # This is complex in graph mode if we want in-place addition behavior via scatter
            # Instead, we construct the update tensor and insert it

            # Since we initialized output with zeros, and segments overlap, 
            # we can't just use simple slice_update because it overwrites.
            # However, for non-overlapping parts it's overwrite.
            # For overlapping parts, we need to accumulate.
            # ops.scatter_update does overwrite. ops.scatter_add might be needed but isn't standard in keras.ops

            # Alternative: Construct a full-size tensor for this segment padded with zeros and add

            # Construct padding
            pad_left = start_idx
            pad_right = target_len - (start_idx + seg_len)

            # Pad segment to full length
            # padding argument for pad is [[top, bottom], [left, right], ...]
            # batch dim: [0, 0], time dim: [pad_left, pad_right], channel dim: [0, 0]

            # We need pad_left/pad_right to be integers or tensors
            # ops.pad expects standard list of lists. If tensors, TF handles it.

            padded_segment = ops.pad(
                weighted_segment,
                [[0, 0], [pad_left, pad_right], [0, 0]]
            )

            output = output + padded_segment

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

        node_idx_counter = 0

        # Process through each level of the tree
        for level in range(self.tree_depth + 1):
            num_segments = 2 ** level

            # Get nodes for this level
            level_nodes = self.all_nodes[node_idx_counter: node_idx_counter + num_segments]
            node_idx_counter += num_segments

            # Split into segments
            segments = self._split_with_overlap(current, num_segments)

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
               ↓
        (optional) Input projection: Linear -> channels
               ↓
        PRISMTimeTree: hierarchical wavelet processing
               ↓
        (optional) Residual connection: output + input
               ↓
        (optional) Output normalization
               ↓
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