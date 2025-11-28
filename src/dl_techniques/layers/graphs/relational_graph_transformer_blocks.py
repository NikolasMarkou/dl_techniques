"""
Relational Graph Transformer (RELGT) Building Blocks.

This module implements the core components of the Relational Graph Transformer,
a model designed for encoding heterogeneous graph node information into unified
token embeddings suitable for transformer processing.

The RELGT model addresses the challenge of representing complex, heterogeneous,
and temporal graph data by decomposing each node into five fundamental components
and learning specialized representations for each before combining them.

Components:
    - LightweightGNNLayer: Graph convolution for structural encoding
    - RELGTTokenEncoder: Multi-element tokenization strategy
    - RELGTTransformerBlock: Hybrid local-global transformer processing

References:
    - Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS.
    - Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification
      with Graph Convolutional Networks. ICLR.
"""

import keras
from typing import Optional, Union, Tuple, List, Dict, Any, Callable

# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

from ..ffn import create_ffn_layer
from ..transformers import TransformerLayer
from ..norms import create_normalization_layer

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class LightweightGNNLayer(keras.layers.Layer):
    """
    Lightweight Graph Convolutional Network layer for structural encoding.

    This layer performs a single step of message passing on a local subgraph,
    implementing symmetric normalization of the adjacency matrix for stable
    information propagation between connected nodes.

    **Intent**: Serve as the GNN(A_local, Z_random) component within the
    RELGTTokenEncoder to capture structural positional information purely
    from local graph topology without relying on node features.

    **Architecture**::

        ┌─────────────────────────────────────────────────────────────┐
        │                  LightweightGNNLayer                        │
        ├─────────────────────────────────────────────────────────────┤
        │                                                             │
        │   ┌───────────┐     ┌──────────────┐     ┌───────────────┐  │
        │   │  Node     │     │  Adjacency   │     │   Identity    │  │
        │   │  Features │     │   Matrix     │     │    Matrix     │  │
        │   │  H [B,N,D]│     │  A [B,N,N]   │     │   I [N,N]     │  │
        │   └─────┬─────┘     └──────┬───────┘     └───────┬───────┘  │
        │         │                  │                     │          │
        │         │                  └──────────┬──────────┘          │
        │         │                             ▼                     │
        │         │                    ┌────────────────┐             │
        │         │                    │  Add Self-Loops│             │
        │         │                    │   Ã = A + I    │             │
        │         │                    └────────┬───────┘             │
        │         │                             ▼                     │
        │         │                    ┌────────────────┐             │
        │         │                    │   Symmetric    │             │
        │         │                    │ Normalization  │             │
        │         │                    │ D̃⁻¹/² Ã D̃⁻¹/²  │             │
        │         │                    └────────┬───────┘             │
        │         ▼                             │                     │
        │   ┌───────────┐                       │                     │
        │   │  Linear   │                       │                     │
        │   │  H @ W    │                       │                     │
        │   └─────┬─────┘                       │                     │
        │         │                             │                     │
        │         └──────────────┬──────────────┘                     │
        │                        ▼                                    │
        │               ┌────────────────┐                            │
        │               │ Message Passing│                            │
        │               │  Ã_norm @ H'   │                            │
        │               └────────┬───────┘                            │
        │                        ▼                                    │
        │               ┌────────────────┐                            │
        │               │   Activation   │                            │
        │               │     σ(·)       │                            │
        │               └────────┬───────┘                            │
        │                        ▼                                    │
        │               ┌────────────────┐                            │
        │               │    Output      │                            │
        │               │  [B, N, units] │                            │
        │               └────────────────┘                            │
        └─────────────────────────────────────────────────────────────┘

    **Mathematical Operation**::

        H' = σ(D̃⁻¹/² Ã D̃⁻¹/² H W)

    Where:
        - Ã = A + I (adjacency with self-loops)
        - D̃ is the degree matrix of Ã
        - W is learnable weight matrix of shape (input_dim, units)
        - σ is the activation function

    Args:
        units: Integer, dimensionality of output feature space. Must be positive.
        activation: String or callable activation function. Defaults to 'relu'.
        kernel_initializer: String or initializer for kernel weights.
            Defaults to 'glorot_uniform'.
        kernel_regularizer: Optional regularizer for kernel weights.
        **kwargs: Additional arguments for Layer base class.

    Input shape:
        List of two tensors:
            - node_features: ``(batch_size, num_nodes, input_dim)``
            - adjacency_matrix: ``(batch_size, num_nodes, num_nodes)``

    Output shape:
        Tensor with shape ``(batch_size, num_nodes, units)``.

    Raises:
        ValueError: If units is not positive.
        ValueError: If last dimension of node_features is not defined.

    References:
        - Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification
          with Graph Convolutional Networks. ICLR.
        - Hamilton, W. L., et al. (2017). Inductive Representation Learning
          on Large Graphs. NeurIPS.
    """

    def __init__(
        self,
        units: int,
        activation: Optional[Union[str, Callable]] = "relu",
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if units <= 0:
            raise ValueError(f"units must be positive, got {units}")

        self.units = units
        self.activation = keras.activations.get(activation)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = kernel_regularizer

        # Weight placeholder (created in build)
        self.kernel = None

    def build(self, input_shape: List[Tuple[Optional[int], ...]]) -> None:
        """
        Create the layer's learnable weights.

        Args:
            input_shape: List of shapes [node_features_shape, adjacency_shape].
        """
        feature_shape, _ = input_shape
        input_dim = feature_shape[-1]

        if input_dim is None:
            raise ValueError("Last dimension of node_features must be defined")

        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_dim, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
        )

        super().build(input_shape)

    def call(
        self,
        inputs: List[keras.KerasTensor],
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """
        Forward pass of the GNN layer.

        Args:
            inputs: List of [node_features, adjacency_matrix].
            training: Boolean indicating training mode (unused but kept for API).

        Returns:
            Transformed node features after message passing.
        """
        node_features, adjacency_matrix = inputs
        num_nodes = keras.ops.shape(adjacency_matrix)[1]

        # Add self-loops: Ã = A + I
        eye_matrix = keras.ops.eye(num_nodes, dtype=self.compute_dtype)
        adj_with_self_loops = adjacency_matrix + eye_matrix

        # Compute degree for symmetric normalization: D̃⁻¹/²
        row_sum = keras.ops.sum(adj_with_self_loops, axis=-1)
        d_inv_sqrt = keras.ops.where(
            row_sum > 0,
            keras.ops.power(row_sum, -0.5),
            keras.ops.zeros_like(row_sum),
        )

        # Create diagonal normalization matrix
        d_inv_sqrt_expanded = keras.ops.expand_dims(d_inv_sqrt, -1)
        d_mat_inv_sqrt = d_inv_sqrt_expanded * eye_matrix

        # Symmetric normalization: D̃⁻¹/² Ã D̃⁻¹/²
        normalized_adj = keras.ops.matmul(
            keras.ops.matmul(d_mat_inv_sqrt, adj_with_self_loops),
            d_mat_inv_sqrt,
        )

        # Message passing: Ã_norm @ (H @ W)
        transformed_features = keras.ops.matmul(node_features, self.kernel)
        output = keras.ops.matmul(normalized_adj, transformed_features)

        if self.activation is not None:
            output = self.activation(output)

        return output

    def compute_output_shape(
        self,
        input_shape: List[Tuple[Optional[int], ...]],
    ) -> Tuple[Optional[int], ...]:
        """
        Compute output shape.

        Args:
            input_shape: List of shapes [node_features_shape, adjacency_shape].

        Returns:
            Output shape tuple.
        """
        feature_shape, _ = input_shape
        return (*feature_shape[:-1], self.units)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            "units": self.units,
            "activation": keras.activations.serialize(self.activation),
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class RELGTTokenEncoder(keras.layers.Layer):
    """
    Multi-element tokenization encoder for heterogeneous graph nodes.

    This layer implements the core tokenization strategy of the RELGT model,
    decomposing each node into five fundamental components and learning
    specialized representations for each before combining them into unified
    token embeddings suitable for transformer processing.

    **Intent**: Transform heterogeneous, temporal, and topological graph
    information into unified token embeddings without expensive precomputation,
    enabling efficient encoding of relational data complexity.

    **Architecture**::

        ┌─────────────────────────────────────────────────────────────────────┐
        │                      RELGTTokenEncoder                              │
        ├─────────────────────────────────────────────────────────────────────┤
        │                                                                     │
        │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │
        │  │    Node     │  │    Node     │  │    Hop      │  │  Relative  │  │
        │  │  Features   │  │   Types     │  │  Distances  │  │   Times    │  │
        │  │ [B,K,F]     │  │  [B,K]      │  │  [B,K]      │  │ [B,K,1]    │  │
        │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └─────┬──────┘  │
        │         │                │                │               │         │
        │         ▼                ▼                ▼               ▼         │
        │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌────────────┐  │
        │  │    Dense     │ │  Embedding   │ │  Embedding   │ │   Dense    │  │
        │  │  Projection  │ │    Lookup    │ │    Lookup    │ │ Projection │  │
        │  │   → [B,K,E]  │ │  → [B,K,E]   │ │  → [B,K,E]   │ │ → [B,K,E]  │  │
        │  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └─────┬──────┘  │
        │         │                │                │               │         │
        │         │     ┌──────────────────────┐    │               │         │
        │         │     │  Subgraph Adjacency  │    │               │         │
        │         │     │      [B, K, K]       │    │               │         │
        │         │     └──────────┬───────────┘    │               │         │
        │         │                │                │               │         │
        │         │                ▼                │               │         │
        │         │     ┌──────────────────────┐    │               │         │
        │         │     │   Random Features    │    │               │         │
        │         │     │  Z ~ N(0,1) [B,K,P]  │    │               │         │
        │         │     └──────────┬───────────┘    │               │         │
        │         │                │                │               │         │
        │         │                ▼                │               │         │
        │         │     ┌──────────────────────┐    │               │         │
        │         │     │   GNN Stack (L×)     │    │               │         │
        │         │     │  Structural Encoder  │    │               │         │
        │         │     └──────────┬───────────┘    │               │         │
        │         │                │                │               │         │
        │         │                ▼                │               │         │
        │         │     ┌──────────────────────┐    │               │         │
        │         │     │   Dense Projection   │    │               │         │
        │         │     │      → [B,K,E]       │    │               │         │
        │         │     └──────────┬───────────┘    │               │         │
        │         │                │                │               │         │
        │         └────────┬───────┴───────┬────────┴──────┬────────┘         │
        │                  │               │               │                  │
        │                  ▼               ▼               ▼                  │
        │         ┌────────────────────────────────────────────────┐          │
        │         │              Element-wise Addition             │          │
        │         │   E_feat + E_type + E_hop + E_time + E_pe      │          │
        │         └────────────────────────┬───────────────────────┘          │
        │                                  │                                  │
        │                                  ▼                                  │
        │                       ┌────────────────────┐                        │
        │                       │   Normalization    │                        │
        │                       └─────────┬──────────┘                        │
        │                                 │                                   │
        │                                 ▼                                   │
        │                       ┌────────────────────┐                        │
        │                       │      Dropout       │                        │
        │                       └─────────┬──────────┘                        │
        │                                 │                                   │
        │                                 ▼                                   │
        │                       ┌────────────────────┐                        │
        │                       │   Token Output     │                        │
        │                       │    [B, K, E]       │                        │
        │                       └────────────────────┘                        │
        └─────────────────────────────────────────────────────────────────────┘

        Legend: B=batch, K=k_neighbors, F=feature_dim, E=embedding_dim, P=gnn_pe_dim

    **Mathematical Operation**::

        T = Norm(Dropout(E_feat(x) + E_type(t) + E_hop(h) + E_time(τ) + E_pe(A)))

    Where:
        - E_feat: Dense projection of node features
        - E_type: Embedding lookup for node types
        - E_hop: Embedding lookup for hop distances
        - E_time: Dense projection of relative times
        - E_pe: GNN-based positional encoding from structure

    Args:
        embedding_dim: Integer, dimensionality of final token embedding.
            Must be positive.
        num_node_types: Integer, total number of unique entity types.
            Must be positive.
        max_hops: Integer, maximum hop distance to encode. Defaults to 2.
        gnn_pe_dim: Integer, output dimension for GNN layers. Defaults to 32.
        gnn_pe_layers: Integer, number of GNN layers. Defaults to 2.
        dropout_rate: Float between 0 and 1. Defaults to 0.1.
        normalization_type: String, type of normalization. Defaults to 'layer_norm'.
        kernel_initializer: String or initializer for Dense layers.
            Defaults to 'glorot_uniform'.
        **kwargs: Additional arguments for Layer base class.

    Input shape:
        Dictionary with keys:
            - ``'node_features'``: ``(batch_size, k_neighbors, feature_dim)``
            - ``'node_types'``: ``(batch_size, k_neighbors)`` integer-encoded
            - ``'hop_distances'``: ``(batch_size, k_neighbors)`` integer-encoded
            - ``'relative_times'``: ``(batch_size, k_neighbors, 1)``
            - ``'subgraph_adjacency'``: ``(batch_size, k_neighbors, k_neighbors)``

    Output shape:
        Tensor with shape ``(batch_size, k_neighbors, embedding_dim)``.

    Raises:
        ValueError: If embedding_dim is not positive.
        ValueError: If num_node_types is not positive.
        ValueError: If max_hops is negative.
        ValueError: If gnn_pe_dim is not positive.
        ValueError: If gnn_pe_layers is not positive.
        ValueError: If dropout_rate is not in [0, 1].

    References:
        - Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS.
        - Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification
          with Graph Convolutional Networks. ICLR.
        - Ying, C., et al. (2021). Do Transformers Really Perform Bad for
          Graph Representation? (Graphormer). NeurIPS.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_node_types: int,
        max_hops: int = 2,
        gnn_pe_dim: int = 32,
        gnn_pe_layers: int = 2,
        dropout_rate: float = 0.1,
        normalization_type: str = "layer_norm",
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        if num_node_types <= 0:
            raise ValueError(f"num_node_types must be positive, got {num_node_types}")
        if max_hops < 0:
            raise ValueError(f"max_hops must be non-negative, got {max_hops}")
        if gnn_pe_dim <= 0:
            raise ValueError(f"gnn_pe_dim must be positive, got {gnn_pe_dim}")
        if gnn_pe_layers <= 0:
            raise ValueError(f"gnn_pe_layers must be positive, got {gnn_pe_layers}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")

        # Store configuration
        self.embedding_dim = embedding_dim
        self.num_node_types = num_node_types
        self.max_hops = max_hops
        self.gnn_pe_dim = gnn_pe_dim
        self.gnn_pe_layers = gnn_pe_layers
        self.dropout_rate = dropout_rate
        self.normalization_type = normalization_type
        self.kernel_initializer = keras.initializers.get(kernel_initializer)

        # CREATE all sub-layers in __init__

        # 1. Node Feature Encoder
        self.feature_encoder = keras.layers.Dense(
            embedding_dim,
            kernel_initializer=self.kernel_initializer,
            name="FeatureEncoder",
        )

        # 2. Node Type Encoder
        self.type_encoder = keras.layers.Embedding(
            input_dim=num_node_types,
            output_dim=embedding_dim,
            name="TypeEncoder",
        )

        # 3. Hop Distance Encoder
        self.hop_encoder = keras.layers.Embedding(
            input_dim=max_hops + 1,  # +1 for 0-hop (self)
            output_dim=embedding_dim,
            name="HopEncoder",
        )

        # 4. Time Encoder
        self.time_encoder = keras.layers.Dense(
            embedding_dim,
            kernel_initializer=self.kernel_initializer,
            name="TimeEncoder",
        )

        # 5. Subgraph Positional Encoder (GNN stack + projection)
        self.gnn_pe_layers_list = [
            LightweightGNNLayer(gnn_pe_dim, name=f"GNNPELayer_{i}")
            for i in range(gnn_pe_layers)
        ]
        self.pe_projection = keras.layers.Dense(
            embedding_dim,
            kernel_initializer=self.kernel_initializer,
            name="PEProjection",
        )

        # Final processing layers
        self.layer_norm = create_normalization_layer(
            normalization_type, name="TokenNormalization"
        )
        self.dropout = keras.layers.Dropout(dropout_rate, name="TokenDropout")

    def build(self, input_shape: Dict[str, Tuple[Optional[int], ...]]) -> None:
        """
        Build all sub-layers.

        Args:
            input_shape: Dictionary of input shapes keyed by input name.
        """
        feature_shape = input_shape["node_features"]
        adjacency_shape = input_shape["subgraph_adjacency"]
        time_shape = input_shape["relative_times"]

        # Build feature encoder
        self.feature_encoder.build(feature_shape)

        # Build time encoder
        self.time_encoder.build(time_shape)

        # Build GNN PE layers sequentially
        gnn_input_shape = (feature_shape[0], feature_shape[1], self.gnn_pe_dim)
        for gnn_layer in self.gnn_pe_layers_list:
            gnn_layer.build([gnn_input_shape, adjacency_shape])
            gnn_input_shape = gnn_layer.compute_output_shape([gnn_input_shape, adjacency_shape])

        # Build PE projection
        self.pe_projection.build(gnn_input_shape)

        # Build normalization and dropout
        token_shape = (*feature_shape[:-1], self.embedding_dim)
        self.layer_norm.build(token_shape)
        self.dropout.build(token_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: Dict[str, keras.KerasTensor],
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """
        Forward pass for multi-element tokenization.

        Args:
            inputs: Dictionary containing node features, types, hops, times, and adjacency.
            training: Boolean indicating training mode.

        Returns:
            Token embeddings of shape (batch_size, k_neighbors, embedding_dim).
        """
        node_features = inputs["node_features"]
        node_types = inputs["node_types"]
        hop_distances = inputs["hop_distances"]
        relative_times = inputs["relative_times"]
        subgraph_adjacency = inputs["subgraph_adjacency"]

        batch_size = keras.ops.shape(node_features)[0]
        k_neighbors = keras.ops.shape(node_features)[1]

        # 1. Node Feature Encoding
        feature_embeddings = self.feature_encoder(node_features)

        # 2. Node Type Encoding
        type_embeddings = self.type_encoder(node_types)

        # 3. Hop Distance Encoding
        hop_embeddings = self.hop_encoder(hop_distances)

        # 4. Time Encoding
        time_embeddings = self.time_encoder(relative_times)

        # 5. Subgraph Positional Encoding via GNN
        random_features = keras.random.normal(
            shape=(batch_size, k_neighbors, self.gnn_pe_dim),
            dtype=self.compute_dtype,
        )

        pe_features = random_features
        for gnn_layer in self.gnn_pe_layers_list:
            pe_features = gnn_layer([pe_features, subgraph_adjacency], training=training)

        pe_embeddings = self.pe_projection(pe_features)

        # Combine via element-wise addition
        combined_embeddings = (
            feature_embeddings
            + type_embeddings
            + hop_embeddings
            + time_embeddings
            + pe_embeddings
        )

        # Apply normalization and dropout
        normalized_tokens = self.layer_norm(combined_embeddings)
        output_tokens = self.dropout(normalized_tokens, training=training)

        return output_tokens

    def compute_output_shape(
        self,
        input_shape: Dict[str, Tuple[Optional[int], ...]],
    ) -> Tuple[Optional[int], ...]:
        """
        Compute output shape.

        Args:
            input_shape: Dictionary of input shapes.

        Returns:
            Output shape tuple.
        """
        feature_shape = input_shape["node_features"]
        return (*feature_shape[:-1], self.embedding_dim)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            "embedding_dim": self.embedding_dim,
            "num_node_types": self.num_node_types,
            "max_hops": self.max_hops,
            "gnn_pe_dim": self.gnn_pe_dim,
            "gnn_pe_layers": self.gnn_pe_layers,
            "dropout_rate": self.dropout_rate,
            "normalization_type": self.normalization_type,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
        })
        return config


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class RELGTTransformerBlock(keras.layers.Layer):
    """
    Hybrid local-global Transformer block for relational graph processing.

    This block implements the core architectural innovation of RELGT: combining
    local attention over sampled subgraphs with global attention to learnable
    centroids. The local module captures fine-grained structural patterns while
    the global module enables database-wide context integration.

    **Intent**: Capture both fine-grained local structural patterns and broad
    global database patterns by integrating detailed local context with
    learnable global prototype representations.

    **Architecture**::

        ┌─────────────────────────────────────────────────────────────────────┐
        │                    RELGTTransformerBlock                            │
        ├─────────────────────────────────────────────────────────────────────┤
        │                                                                     │
        │  ┌─────────────────────┐          ┌─────────────────────────────┐   │
        │  │    Local Tokens     │          │      Seed Node Features     │   │
        │  │     [B, K, E]       │          │          [B, 1, E]          │   │
        │  └──────────┬──────────┘          └──────────────┬──────────────┘   │
        │             │                                    │                  │
        │             ▼                                    │                  │
        │  ┌─────────────────────┐                         │                  │
        │  │  LOCAL MODULE       │                         │                  │
        │  │                     │                         │                  │
        │  │  ┌───────────────┐  │                         │                  │
        │  │  │ Transformer   │  │                         │                  │
        │  │  │    Layer      │  │                         │                  │
        │  │  │ (Self-Attn +  │  │                         │                  │
        │  │  │     FFN)      │  │                         │                  │
        │  │  └───────┬───────┘  │                         │                  │
        │  │          │          │                         │                  │
        │  │          ▼          │                         │                  │
        │  │  ┌───────────────┐  │                         │                  │
        │  │  │  Mean Pooling │  │                         │                  │
        │  │  │   over K dim  │  │                         │                  │
        │  │  └───────┬───────┘  │                         │                  │
        │  └──────────┼──────────┘                         │                  │
        │             │                                    │                  │
        │             ▼                                    ▼                  │
        │      ┌────────────┐                   ┌────────────────────────┐    │
        │      │  h_local   │                   │     GLOBAL MODULE      │    │
        │      │  [B, E]    │                   │                        │    │
        │      └──────┬─────┘                   │  ┌──────────────────┐  │    │
        │             │                         │  │ Global Centroids │  │    │
        │             │                         │  │   (Learnable)    │  │    │
        │             │                         │  │    [G, E]        │  │    │
        │             │                         │  └────────┬─────────┘  │    │
        │             │                         │           │            │    │
        │             │                         │           ▼            │    │
        │             │                         │  ┌──────────────────┐  │    │
        │             │                         │  │  Cross-Attention │  │    │
        │             │                         │  │ Q=Seed, K,V=Cent │  │    │
        │             │                         │  └────────┬─────────┘  │    │
        │             │                         │           │            │    │
        │             │                         │           ▼            │    │
        │             │                         │  ┌──────────────────┐  │    │
        │             │                         │  │     Squeeze      │  │    │
        │             │                         │  │    [B,1,E]→[B,E] │  │    │
        │             │                         │  └────────┬─────────┘  │    │
        │             │                         └───────────┼────────────┘    │
        │             │                                     │                 │
        │             │                                     ▼                 │
        │             │                              ┌────────────┐           │
        │             │                              │  h_global  │           │
        │             │                              │   [B, E]   │           │
        │             │                              └──────┬─────┘           │
        │             │                                     │                 │
        │             └───────────────┬─────────────────────┘                 │
        │                             │                                       │
        │                             ▼                                       │
        │                  ┌─────────────────────┐                            │
        │                  │     Concatenate     │                            │
        │                  │   [h_local, h_glob] │                            │
        │                  │      [B, 2E]        │                            │
        │                  └──────────┬──────────┘                            │
        │                             │                                       │
        │            ┌────────────────┼───────────────┐                       │
        │            │                │               │                       │
        │            ▼                ▼               │                       │
        │  ┌──────────────────┐  ┌──────────────┐     │                       │
        │  │ Residual Dense   │  │     FFN      │     │                       │
        │  │   [2E] → [E]     │  │  [2E] → [E]  │     │                       │
        │  └────────┬─────────┘  └──────┬───────┘     │                       │
        │           │                   │             │                       │
        │           │                   ▼             │                       │
        │           │           ┌────────────┐        │                       │
        │           │           │  Dropout   │        │                       │
        │           │           └──────┬─────┘        │                       │
        │           │                  │              │                       │
        │           └────────┬─────────┘              │                       │
        │                    │                        │                       │
        │                    ▼                        │                       │
        │           ┌─────────────────┐               │                       │
        │           │  Add & Norm     │               │                       │
        │           │  (Residual)     │               │                       │
        │           └────────┬────────┘               │                       │
        │                    │                        │                       │
        │                    ▼                        │                       │
        │           ┌─────────────────┐               │                       │
        │           │     Output      │               │                       │
        │           │     [B, E]      │               │                       │
        │           └─────────────────┘               │                       │
        └─────────────────────────────────────────────────────────────────────┘

        Legend: B=batch, K=k_neighbors, E=embedding_dim, G=num_global_centroids

    **Mathematical Operation**::

        h_local  = MeanPool(TransformerLayer(local_tokens))
        h_global = Squeeze(CrossAttention(Q=seed, K=V=centroids))
        output   = Norm(FFN([h_local; h_global]) + ResidualProj([h_local; h_global]))

    Args:
        embedding_dim: Integer, dimensionality of token and output embeddings.
            Must be positive.
        num_heads: Integer, number of attention heads for both local and global
            attention. Must be positive and divide embedding_dim evenly.
        num_global_centroids: Integer, number of learnable global centroid tokens.
            Must be positive.
        ffn_dim: Integer, hidden dimension for feed-forward networks. Must be positive.
        dropout_rate: Float between 0 and 1. Defaults to 0.1.
        ffn_type: String, type of FFN to use. Defaults to 'mlp'.
        normalization_type: String, type of normalization. Defaults to 'layer_norm'.
        kernel_initializer: String or initializer for Dense layers.
            Defaults to 'glorot_uniform'.
        **kwargs: Additional arguments for Layer base class.

    Input shape:
        List of two tensors:
            - local_tokens: ``(batch_size, k_neighbors, embedding_dim)``
            - seed_node_features: ``(batch_size, 1, embedding_dim)``

    Output shape:
        Tensor with shape ``(batch_size, embedding_dim)``.

    Raises:
        ValueError: If embedding_dim is not positive.
        ValueError: If num_heads is not positive.
        ValueError: If embedding_dim is not divisible by num_heads.
        ValueError: If num_global_centroids is not positive.
        ValueError: If ffn_dim is not positive.
        ValueError: If dropout_rate is not in [0, 1].

    References:
        - Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS.
        - Ying, C., et al. (2021). Do Transformers Really Perform Bad for
          Graph Representation? (Graphormer). NeurIPS.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        num_global_centroids: int,
        ffn_dim: int,
        dropout_rate: float = 0.1,
        ffn_type: str = "mlp",
        normalization_type: str = "layer_norm",
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})"
            )
        if num_global_centroids <= 0:
            raise ValueError(f"num_global_centroids must be positive, got {num_global_centroids}")
        if ffn_dim <= 0:
            raise ValueError(f"ffn_dim must be positive, got {ffn_dim}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")

        # Store configuration
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_global_centroids = num_global_centroids
        self.ffn_dim = ffn_dim
        self.dropout_rate = dropout_rate
        self.ffn_type = ffn_type
        self.normalization_type = normalization_type
        self.kernel_initializer = keras.initializers.get(kernel_initializer)

        # CREATE all sub-layers in __init__

        # Local Module: TransformerLayer for local self-attention
        self.local_transformer = TransformerLayer(
            hidden_size=embedding_dim,
            num_heads=num_heads,
            intermediate_size=ffn_dim,
            attention_type="multi_head",
            normalization_type=normalization_type,
            ffn_type=ffn_type,
            dropout_rate=dropout_rate,
            name="LocalTransformer",
        )

        # Global Module: Cross-attention to learnable centroids
        self.global_attention = keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dim // num_heads,
            dropout=dropout_rate,
            name="GlobalAttention",
        )

        # Residual projection for combined representation
        self.residual_projection = keras.layers.Dense(
            embedding_dim,
            kernel_initializer=self.kernel_initializer,
            name="ResidualProjection",
        )

        # Combination FFN (created in build since it depends on computed input dim)
        self.combination_ffn = None

        # Combination normalization and dropout
        self.combination_norm = create_normalization_layer(
            normalization_type, name="CombinationNorm"
        )
        self.combination_dropout = keras.layers.Dropout(
            dropout_rate, name="CombinationDropout"
        )

        # Global centroids (weight created in build)
        self.global_centroids = None

    def build(self, input_shape: List[Tuple[Optional[int], ...]]) -> None:
        """
        Build sub-layers and create global centroid weights.

        Args:
            input_shape: List of shapes [local_tokens_shape, seed_features_shape].
        """
        local_tokens_shape, seed_features_shape = input_shape

        # Build local transformer
        self.local_transformer.build(local_tokens_shape)

        # Build global attention (query from seed, key/value from centroids)
        centroid_shape = (None, self.num_global_centroids, self.embedding_dim)
        self.global_attention.build(
            query_shape=seed_features_shape,
            value_shape=centroid_shape,
            key_shape=centroid_shape,
        )

        # Create global centroid weights
        self.global_centroids = self.add_weight(
            name="global_centroids",
            shape=(self.num_global_centroids, self.embedding_dim),
            initializer="glorot_uniform",
            trainable=True,
        )

        # Create combination FFN (input is concatenation: 2 * embedding_dim)
        self.combination_ffn = create_ffn_layer(
            self.ffn_type,
            hidden_dim=self.ffn_dim,
            output_dim=self.embedding_dim,
            dropout_rate=self.dropout_rate,
            name="CombinationFFN",
        )

        # Build combination layers
        combined_shape = (None, 2 * self.embedding_dim)
        self.combination_ffn.build(combined_shape)
        self.residual_projection.build(combined_shape)

        output_shape = (None, self.embedding_dim)
        self.combination_norm.build(output_shape)
        self.combination_dropout.build(output_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: List[keras.KerasTensor],
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """
        Forward pass for hybrid local-global processing.

        Args:
            inputs: List of [local_tokens, seed_node_features].
            training: Boolean indicating training mode.

        Returns:
            Combined representation of shape (batch_size, embedding_dim).
        """
        local_tokens, seed_node_features = inputs
        batch_size = keras.ops.shape(seed_node_features)[0]

        # --- Local Module ---
        # Process local tokens through transformer (self-attention + FFN)
        local_processed_tokens = self.local_transformer(local_tokens, training=training)

        # Pool to single representation per sample
        h_local = keras.ops.mean(local_processed_tokens, axis=1)  # [B, E]

        # --- Global Module ---
        # Tile global centroids for batch processing
        global_centroids_batch = keras.ops.tile(
            keras.ops.expand_dims(self.global_centroids, 0),
            [batch_size, 1, 1],
        )  # [B, G, E]

        # Cross-attention: seed queries global centroids
        h_global = self.global_attention(
            query=seed_node_features,  # [B, 1, E]
            value=global_centroids_batch,  # [B, G, E]
            key=global_centroids_batch,
            training=training,
        )
        h_global = keras.ops.squeeze(h_global, axis=1)  # [B, E]

        # --- Combination ---
        # Concatenate local and global representations
        combined_representation = keras.ops.concatenate(
            [h_local, h_global], axis=-1
        )  # [B, 2E]

        # Process through FFN
        output = self.combination_ffn(combined_representation, training=training)
        output = self.combination_dropout(output, training=training)

        # Residual connection with projection
        residual = self.residual_projection(combined_representation)
        output = self.combination_norm(output + residual)

        return output

    def compute_output_shape(
        self,
        input_shape: List[Tuple[Optional[int], ...]],
    ) -> Tuple[Optional[int], ...]:
        """
        Compute output shape.

        Args:
            input_shape: List of shapes [local_tokens_shape, seed_features_shape].

        Returns:
            Output shape tuple.
        """
        local_tokens_shape, _ = input_shape
        batch_size = local_tokens_shape[0]
        return (batch_size, self.embedding_dim)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            "embedding_dim": self.embedding_dim,
            "num_heads": self.num_heads,
            "num_global_centroids": self.num_global_centroids,
            "ffn_dim": self.ffn_dim,
            "dropout_rate": self.dropout_rate,
            "ffn_type": self.ffn_type,
            "normalization_type": self.normalization_type,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
        })
        return config


# ---------------------------------------------------------------------
