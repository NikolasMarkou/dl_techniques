"""
A high-performance, graph-based structural regularizer for Keras 3.

This module provides a custom Keras layer, `SystemicGraphFilter`, and its
direct application within a Transformer's attention mechanism, `StructuredAttention`.
The core protocol is designed to re-architect a neural network's epistemic
process by imposing a learnable, structural prior on its internal data representations.

**Incentive & Power Dynamics:**

Neural networks are incentivized to minimize loss, but in high-dimensional,
noisy environments, they often achieve this by overfitting to spurious,
transient correlations. This leads to fragile models that fail to generalize.
This layer introduces a new, higher-order incentive: to discover and enforce
a sparse, stable, and structurally coherent dependency graph.

It fundamentally alters the network's internal power dynamics by shifting from a
"democratic" model, where every noisy signal has a vote, to a "technocratic"
one, where informational authority is granted only to a select, structurally
validated elite of high-affinity connections.

**Systemic Function:**

The layer operates as an information refinery, transmuting a high-entropy
input (like a raw correlation or attention score matrix) into a low-entropy,
structurally consistent output. This is achieved via two integrated stages:

1.  **Structural Distillation:** It uses a sparse, top-k attention mechanism
    to identify and isolate the system's core dependency backbone, effectively
    filtering out the "noise" of weak, spurious connections.

2.  **Coherent Diffusion:** It then propagates information exclusively along this
    distilled backbone, using a series of graph convolutions. This enforces a
    new systemic logic where all relationships are re-cast in a manner
    consistent with the underlying structure.

When integrated into a Transformer's self-attention (`StructuredAttention`),
this protocol transforms the attention mechanism from a reactive, short-sighted
process into a disciplined, strategic instrument that can discover more robust
and generalizable patterns in complex data.
"""

import keras
from typing import Optional, Tuple, Union, Dict, Any


class SystemicGraphFilter(keras.layers.Layer):
    """
    A principled, graph-based filter for correlation matrices.

    Denoises a correlation matrix by building a sparse, soft adjacency graph
    via top-k attention on distance-transformed affinities, then propagating
    and smoothing values along the learned graph structure using residual
    graph convolutions.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────┐
        │  Correlation Matrix (n x n)  │
        └──────────────┬───────────────┘
                       ▼
        ┌──────────────────────────────┐
        │  Distance Transform          │
        │  ─► Affinity ─► Top-k Mask   │
        │  ─► Soft Adjacency Graph     │
        └──────────────┬───────────────┘
                       ▼
        ┌──────────────────────────────┐
        │  Graph Convolution (N steps) │
        │  Residual smoothing along    │
        │  sparse graph edges          │
        └──────────────┬───────────────┘
                       ▼
        ┌──────────────────────────────┐
        │  Symmetrize + Unit Diagonal  │
        │  ─► Filtered Correlation     │
        └──────────────────────────────┘

    :param top_k_neighbors: The number of strongest neighbors to connect for each node in the graph. A value of 2-3 is recommended to create an MST-like sparse structure. Default is 2.
    :type top_k_neighbors: int
    :param n_propagation_steps: The number of graph smoothing/propagation iterations. Default is 3.
    :type n_propagation_steps: int
    :param distance_metric: Method to convert correlations to distances. Options: 'sqrt' (sqrt(2*(1-corr))), 'linear' (1-corr). Default is 'sqrt'.
    :type distance_metric: str
    :param initial_temperature: Initial temperature for the attention softmax. Higher values lead to softer attention. Default is 0.1.
    :type initial_temperature: float
    :param learnable_temperature: Whether the temperature parameter should be trainable. Default is True.
    :type learnable_temperature: bool
    **kwargs
    :param epsilon: Small constant for numerical stability. Default is 1e-8. Additional keyword arguments for the base Layer class.
    :type epsilon: float

    Attributes
    ----------
    temperature : keras.Variable
        The temperature parameter for softmax attention.

    Examples
    --------
    >>> import keras
    >>> import numpy as np
    >>> # Create a noisy correlation matrix
    >>> corr_matrix = np.random.randn(32, 10, 10)
    >>> corr_matrix = (corr_matrix + corr_matrix.transpose(0, 2, 1)) / 2
    >>> # Apply the filter
    >>> filter_layer = SystemicGraphFilter(top_k_neighbors=3)
    >>> filtered = filter_layer(corr_matrix)
    """

    def __init__(
        self,
        top_k_neighbors: int = 2,
        n_propagation_steps: int = 3,
        distance_metric: str = 'sqrt',
        initial_temperature: float = 0.1,
        learnable_temperature: bool = True,
        epsilon: float = 1e-8,
        **kwargs
    ) -> None:
        """
        Initialize the SystemicGraphFilter layer.

        :param top_k_neighbors: Number of neighbors for graph construction.
        :type top_k_neighbors: int
        :param n_propagation_steps: Number of value propagation iterations.
        :type n_propagation_steps: int
        :param distance_metric: Distance metric type ('sqrt' or 'linear').
        :type distance_metric: str
        :param initial_temperature: Initial softmax temperature.
        :type initial_temperature: float
        :param learnable_temperature: Whether temperature is trainable.
        :type learnable_temperature: bool
        **kwargs
        :param epsilon: Numerical stability constant. Additional Layer arguments.
        :type epsilon: float
        """
        super().__init__(**kwargs)

        if distance_metric not in ['sqrt', 'linear']:
            raise ValueError(
                f"distance_metric must be 'sqrt' or 'linear', got {distance_metric}"
            )

        self.top_k_neighbors = top_k_neighbors
        self.n_propagation_steps = n_propagation_steps
        self.distance_metric = distance_metric
        self.initial_temperature = initial_temperature
        self.learnable_temperature = learnable_temperature
        self.epsilon = epsilon

    def build(self, input_shape: Union[Tuple[int, ...], keras.KerasTensorShape]) -> None:
        """
        Build the layer and create trainable weights.

        :param input_shape: Shape of input tensor.
        :type input_shape: Union[Tuple[int, ...], keras.KerasTensorShape]
        """
        super().build(input_shape)

        self.temperature = self.add_weight(
            name='temperature',
            shape=(),
            initializer=keras.initializers.Constant(self.initial_temperature),
            constraint=keras.constraints.MinMaxNorm(
                min_value=1e-3,
                max_value=10.0
            ),
            trainable=self.learnable_temperature,
            dtype=self.dtype
        )

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'top_k_neighbors': self.top_k_neighbors,
            'n_propagation_steps': self.n_propagation_steps,
            'distance_metric': self.distance_metric,
            'initial_temperature': self.initial_temperature,
            'learnable_temperature': self.learnable_temperature,
            'epsilon': self.epsilon
        })
        return config

    def _compute_distance_matrix(
        self,
        correlation_matrix: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Convert correlation matrix to distance matrix.

        :param correlation_matrix: Input correlation matrix.
        :type correlation_matrix: keras.KerasTensor

        :return: Distance matrix.
        :rtype: keras.KerasTensor
        """
        if self.distance_metric == 'sqrt':
            distances = keras.ops.sqrt(
                2.0 * (1.0 - correlation_matrix) + self.epsilon
            )
        else:  # 'linear'
            distances = 1.0 - correlation_matrix
        return distances

    def _build_soft_graph(
        self,
        correlation_matrix: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Build sparse soft adjacency graph using top-k attention.

        :param correlation_matrix: Input correlation matrix.
        :type correlation_matrix: keras.KerasTensor

        :return: Soft adjacency matrix.
        :rtype: keras.KerasTensor
        """
        n = keras.ops.shape(correlation_matrix)[-1]

        # Compute distances and affinities
        distances = self._compute_distance_matrix(correlation_matrix)
        affinities = keras.ops.exp(-distances / self.temperature)

        # Create identity mask for self-connections
        eye = keras.ops.eye(n, dtype=self.dtype)
        if keras.ops.ndim(affinities) == 3:
            eye = keras.ops.expand_dims(eye, axis=0)
        affinities = affinities * (1.0 - eye)

        # Find top-k neighbors
        k = keras.ops.minimum(self.top_k_neighbors, n - 1)
        top_k_affinities, _ = keras.ops.top_k(affinities, k=k, sorted=False)

        # Create threshold mask
        min_top_k_affinity = keras.ops.min(
            top_k_affinities, axis=-1, keepdims=True
        )
        top_k_mask = keras.ops.cast(
            affinities >= min_top_k_affinity, dtype=self.dtype
        )

        # Apply masked softmax
        masked_affinities = keras.ops.where(
            top_k_mask > 0,
            affinities,
            keras.ops.full_like(affinities, -1e9)
        )
        soft_adjacency = keras.ops.softmax(masked_affinities, axis=-1)

        # Symmetrize for undirected graph
        axes = None if keras.ops.ndim(soft_adjacency) == 2 else [0, 2, 1]
        soft_adjacency = (
            soft_adjacency + keras.ops.transpose(soft_adjacency, axes=axes)
        ) / 2.0

        return soft_adjacency

    def _propagate_values(
        self,
        correlation_matrix: keras.KerasTensor,
        adjacency: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Propagate and smooth values along graph structure.

        :param correlation_matrix: Original correlation values.
        :type correlation_matrix: keras.KerasTensor
        :param adjacency: Graph adjacency matrix.
        :type adjacency: keras.KerasTensor

        :return: Smoothed correlation matrix.
        :rtype: keras.KerasTensor
        """
        filtered = correlation_matrix

        for i in range(self.n_propagation_steps):
            # Graph convolution: aggregate neighbor information
            messages = keras.ops.matmul(adjacency, filtered)

            # Residual connection with decreasing blend factor
            alpha = 1.0 / (i + 2.0)
            filtered = (1.0 - alpha) * filtered + alpha * messages

            # Preserve original correlations on strong edges
            filtered = (
                adjacency * correlation_matrix +
                (1.0 - adjacency) * filtered
            )

        return filtered

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Execute forward pass of the layer.

        :param inputs: Input correlation matrix of shape (batch_size, n, n) or (n, n).
        :type inputs: keras.KerasTensor
        :param training: Whether in training mode.
        :type training: Optional[bool]

        :return: Filtered correlation matrix with same shape as input.
        :rtype: keras.KerasTensor
        """
        # Ensure valid correlation range
        inputs_clipped = keras.ops.clip(
            inputs, -1.0 + self.epsilon, 1.0 - self.epsilon
        )

        # Ensure symmetry
        axes = None if keras.ops.ndim(inputs_clipped) == 2 else [0, 2, 1]
        inputs_symmetric = (
            inputs_clipped + keras.ops.transpose(inputs_clipped, axes=axes)
        ) / 2.0

        # Build soft sparse dependency graph
        soft_adjacency = self._build_soft_graph(inputs_symmetric)

        # Propagate values along graph
        filtered = self._propagate_values(inputs_symmetric, soft_adjacency)

        # Post-process: ensure symmetry and unit diagonal
        filtered = (
            filtered + keras.ops.transpose(filtered, axes=axes)
        ) / 2.0

        # Ensure unit diagonal
        eye = keras.ops.eye(keras.ops.shape(filtered)[-1], dtype=self.dtype)
        if keras.ops.ndim(filtered) == 3:
            eye = keras.ops.expand_dims(eye, axis=0)
        filtered = filtered * (1.0 - eye) + eye

        return filtered

    def compute_output_shape(
        self,
        input_shape: Union[Tuple[int, ...], keras.KerasTensorShape]
    ) -> Union[Tuple[int, ...], keras.KerasTensorShape]:
        """
        Compute output shape of the layer.

        :param input_shape: Shape of input tensor.
        :type input_shape: Union[Tuple[int, ...], keras.KerasTensorShape]

        :return: Shape of output tensor (same as input).
        :rtype: Union[Tuple[int, ...], keras.KerasTensorShape]
        """
        return input_shape


class StructuredAttention(keras.layers.MultiHeadAttention):
    """
    Multi-Head Attention layer regularized by a SystemicGraphFilter.

    This layer modifies standard attention by filtering raw attention scores
    through a SystemicGraphFilter, enforcing sparse and structurally consistent
    dependency graphs between tokens.

    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param key_dim: Size of each attention head for query and key.
    :type key_dim: int
    :param sgf_top_k: Top-k neighbors for SystemicGraphFilter. Default is 2.
    :type sgf_top_k: int
    **kwargs
    :param sgf_propagation_steps: Propagation steps for SystemicGraphFilter. Default is 3. Additional arguments for MultiHeadAttention.
    :type sgf_propagation_steps: int

    Attributes
    ----------
    systemic_filter : SystemicGraphFilter
        The graph filter applied to attention scores.

    Examples
    --------
    >>> import keras
    >>> import numpy as np
    >>> # Create input sequence
    >>> x = np.random.randn(32, 100, 256)  # (batch, seq_len, d_model)
    >>> # Apply structured attention
    >>> attn_layer = StructuredAttention(num_heads=8, key_dim=32)
    >>> output = attn_layer(x, x)
    """

    def __init__(
        self,
        num_heads: int,
        key_dim: int,
        sgf_top_k: int = 2,
        sgf_propagation_steps: int = 3,
        **kwargs
    ) -> None:
        """
        Initialize StructuredAttention layer.

        :param num_heads: Number of attention heads.
        :type num_heads: int
        :param key_dim: Dimension of each attention head.
        :type key_dim: int
        :param sgf_top_k: Top-k parameter for graph filter.
        :type sgf_top_k: int
        **kwargs
        :param sgf_propagation_steps: Propagation steps for graph filter. Additional MultiHeadAttention arguments.
        :type sgf_propagation_steps: int
        """
        super().__init__(num_heads=num_heads, key_dim=key_dim, **kwargs)

        self.sgf_top_k = sgf_top_k
        self.sgf_propagation_steps = sgf_propagation_steps
        self.systemic_filter = None

    def build(
        self,
        query_shape: Union[Tuple[int, ...], keras.KerasTensorShape],
        value_shape: Optional[Union[Tuple[int, ...], keras.KerasTensorShape]] = None,
        key_shape: Optional[Union[Tuple[int, ...], keras.KerasTensorShape]] = None
    ) -> None:
        """
        Build the layer and initialize sublayers.

        :param query_shape: Shape of query input.
        :type query_shape: Union[Tuple[int, ...], keras.KerasTensorShape]
        :param value_shape: Shape of value input.
        :type value_shape: Optional[Union[Tuple[int, ...], keras.KerasTensorShape]]
        :param key_shape: Shape of key input.
        :type key_shape: Optional[Union[Tuple[int, ...], keras.KerasTensorShape]]
        """
        super().build(query_shape, value_shape, key_shape)

        # Initialize the systemic filter in build
        self.systemic_filter = SystemicGraphFilter(
            top_k_neighbors=self.sgf_top_k,
            n_propagation_steps=self.sgf_propagation_steps,
            name=f"{self.name}_sgf"
        )

    def _compute_attention(
        self,
        query: keras.KerasTensor,
        key: keras.KerasTensor,
        value: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Compute attention with structural filtering.

        :param query: Query tensor.
        :type query: keras.KerasTensor
        :param key: Key tensor.
        :type key: keras.KerasTensor
        :param value: Value tensor.
        :type value: keras.KerasTensor
        :param attention_mask: Attention mask tensor.
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: Whether in training mode.
        :type training: Optional[bool]

        :return: Output tensor and attention weights.
        :rtype: Tuple[keras.KerasTensor, keras.KerasTensor]
        """
        # Compute scaled dot-product attention scores
        attention_scores = keras.ops.matmul(
            query, keras.ops.swapaxes(key, -2, -1)
        )
        dk = keras.ops.cast(self._key_dim, dtype=attention_scores.dtype)
        attention_scores = attention_scores / keras.ops.sqrt(dk)

        # Apply structural filter to attention scores
        original_shape = keras.ops.shape(attention_scores)
        batch_size = original_shape[0]
        num_heads = original_shape[1]
        from_seq_len = original_shape[2]
        to_seq_len = original_shape[3]

        # Reshape for filter: (batch * num_heads, seq, seq)
        scores_reshaped = keras.ops.reshape(
            attention_scores,
            (batch_size * num_heads, from_seq_len, to_seq_len)
        )

        # Apply the systemic graph filter
        filtered_scores_reshaped = self.systemic_filter(
            scores_reshaped, training=training
        )

        # Reshape back to original shape
        attention_scores = keras.ops.reshape(
            filtered_scores_reshaped, original_shape
        )

        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = keras.ops.add(attention_scores, attention_mask)

        # Compute attention weights
        attention_weights = keras.ops.softmax(attention_scores, axis=-1)

        # Apply dropout if specified
        if hasattr(self, '_dropout') and self._dropout > 0 and training:
            attention_weights = keras.layers.Dropout(
                self._dropout, seed=None
            )(attention_weights, training=training)

        # Compute weighted sum of values
        attention_output = keras.ops.matmul(attention_weights, value)

        return attention_output, attention_weights

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'sgf_top_k': self.sgf_top_k,
            'sgf_propagation_steps': self.sgf_propagation_steps
        })
        return config