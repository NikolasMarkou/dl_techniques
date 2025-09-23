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

    This layer denoises a correlation matrix by performing two main operations:

    1. **Builds a sparse, soft adjacency graph:** Identifies the most
       significant connections for each variable using a top-k sparse
       attention mechanism.

    2. **Propagates values:** Smooths the correlation matrix by
       diffusing values along the learned graph structure, using an
       efficient, residual-based graph convolution.

    Parameters
    ----------
    top_k_neighbors : int
        The number of strongest neighbors to connect for each node in the graph.
        A value of 2-3 is recommended to create an MST-like sparse structure.
        Default is 2.
    n_propagation_steps : int
        The number of graph smoothing/propagation iterations.
        Default is 3.
    distance_metric : str
        Method to convert correlations to distances.
        Options: 'sqrt' (sqrt(2*(1-corr))), 'linear' (1-corr).
        Default is 'sqrt'.
    initial_temperature : float
        Initial temperature for the attention softmax. Higher values lead to
        softer attention. Default is 0.1.
    learnable_temperature : bool
        Whether the temperature parameter should be trainable.
        Default is True.
    epsilon : float
        Small constant for numerical stability. Default is 1e-8.
    **kwargs
        Additional keyword arguments for the base Layer class.

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

        Parameters
        ----------
        top_k_neighbors : int
            Number of neighbors for graph construction.
        n_propagation_steps : int
            Number of value propagation iterations.
        distance_metric : str
            Distance metric type ('sqrt' or 'linear').
        initial_temperature : float
            Initial softmax temperature.
        learnable_temperature : bool
            Whether temperature is trainable.
        epsilon : float
            Numerical stability constant.
        **kwargs
            Additional Layer arguments.
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

        Parameters
        ----------
        input_shape : Union[Tuple[int, ...], keras.KerasTensorShape]
            Shape of input tensor.
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

        Returns
        -------
        Dict[str, Any]
            Configuration dictionary.
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

        Parameters
        ----------
        correlation_matrix : keras.KerasTensor
            Input correlation matrix.

        Returns
        -------
        keras.KerasTensor
            Distance matrix.
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

        Parameters
        ----------
        correlation_matrix : keras.KerasTensor
            Input correlation matrix.

        Returns
        -------
        keras.KerasTensor
            Soft adjacency matrix.
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

        Parameters
        ----------
        correlation_matrix : keras.KerasTensor
            Original correlation values.
        adjacency : keras.KerasTensor
            Graph adjacency matrix.

        Returns
        -------
        keras.KerasTensor
            Smoothed correlation matrix.
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

        Parameters
        ----------
        inputs : keras.KerasTensor
            Input correlation matrix of shape (batch_size, n, n) or (n, n).
        training : Optional[bool]
            Whether in training mode.

        Returns
        -------
        keras.KerasTensor
            Filtered correlation matrix with same shape as input.
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

        Parameters
        ----------
        input_shape : Union[Tuple[int, ...], keras.KerasTensorShape]
            Shape of input tensor.

        Returns
        -------
        Union[Tuple[int, ...], keras.KerasTensorShape]
            Shape of output tensor (same as input).
        """
        return input_shape


class StructuredAttention(keras.layers.MultiHeadAttention):
    """
    Multi-Head Attention layer regularized by a SystemicGraphFilter.

    This layer modifies standard attention by filtering raw attention scores
    through a SystemicGraphFilter, enforcing sparse and structurally consistent
    dependency graphs between tokens.

    Parameters
    ----------
    num_heads : int
        Number of attention heads.
    key_dim : int
        Size of each attention head for query and key.
    sgf_top_k : int
        Top-k neighbors for SystemicGraphFilter. Default is 2.
    sgf_propagation_steps : int
        Propagation steps for SystemicGraphFilter. Default is 3.
    **kwargs
        Additional arguments for MultiHeadAttention.

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

        Parameters
        ----------
        num_heads : int
            Number of attention heads.
        key_dim : int
            Dimension of each attention head.
        sgf_top_k : int
            Top-k parameter for graph filter.
        sgf_propagation_steps : int
            Propagation steps for graph filter.
        **kwargs
            Additional MultiHeadAttention arguments.
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

        Parameters
        ----------
        query_shape : Union[Tuple[int, ...], keras.KerasTensorShape]
            Shape of query input.
        value_shape : Optional[Union[Tuple[int, ...], keras.KerasTensorShape]]
            Shape of value input.
        key_shape : Optional[Union[Tuple[int, ...], keras.KerasTensorShape]]
            Shape of key input.
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

        Parameters
        ----------
        query : keras.KerasTensor
            Query tensor.
        key : keras.KerasTensor
            Key tensor.
        value : keras.KerasTensor
            Value tensor.
        attention_mask : Optional[keras.KerasTensor]
            Attention mask tensor.
        training : Optional[bool]
            Whether in training mode.

        Returns
        -------
        Tuple[keras.KerasTensor, keras.KerasTensor]
            Output tensor and attention weights.
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

        Returns
        -------
        Dict[str, Any]
            Configuration dictionary.
        """
        config = super().get_config()
        config.update({
            'sgf_top_k': self.sgf_top_k,
            'sgf_propagation_steps': self.sgf_propagation_steps
        })
        return config