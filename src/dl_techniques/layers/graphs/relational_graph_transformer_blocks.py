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
    """Lightweight GCN layer for structural positional encoding.

    Performs a single message-passing step with symmetric normalisation of the
    adjacency matrix: H' = sigma(D_tilde^{-1/2} A_tilde D_tilde^{-1/2} H W),
    where A_tilde = A + I includes self-loops, D_tilde is its degree matrix,
    W is a learnable weight matrix, and sigma is the activation function.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────┐  ┌──────────────┐
        │ Node Features│  │  Adjacency   │
        │  [B, N, D]   │  │  [B, N, N]   │
        └──────┬───────┘  └──────┬───────┘
               │                 ▼
               │        ┌────────────────┐
               │        │ Ã = A + I      │
               │        └───────┬────────┘
               │                ▼
               │        ┌────────────────┐
               │        │ D̃⁻¹/² Ã D̃⁻¹/²  │
               │        └───────┬────────┘
               ▼                │
        ┌──────────────┐        │
        │ Linear H @ W │        │
        └──────┬───────┘        │
               └───────┬────────┘
                       ▼
               ┌────────────────┐
               │ Ã_norm @ H'    │
               └───────┬────────┘
                       ▼
               ┌────────────────┐
               │ Activation σ   │
               └───────┬────────┘
                       ▼
               ┌────────────────┐
               │ Output [B,N,U] │
               └────────────────┘

    :param units: Dimensionality of output feature space. Must be positive.
    :type units: int
    :param activation: Activation function. Defaults to ``'relu'``.
    :type activation: Optional[Union[str, Callable]]
    :param kernel_initializer: Initializer for kernel weights.
        Defaults to ``'glorot_uniform'``.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for kernel weights.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional arguments for the ``Layer`` base class.
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
        """Create the layer's learnable weights.

        :param input_shape: List of ``[node_features_shape, adjacency_shape]``.
        :type input_shape: List[Tuple[Optional[int], ...]]
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
        """Forward pass of the GNN layer.

        :param inputs: List of ``[node_features, adjacency_matrix]``.
        :type inputs: List[keras.KerasTensor]
        :param training: Whether in training mode.
        :type training: Optional[bool]
        :return: Transformed node features after message passing.
        :rtype: keras.KerasTensor
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
        """Compute output shape.

        :param input_shape: List of ``[node_features_shape, adjacency_shape]``.
        :type input_shape: List[Tuple[Optional[int], ...]]
        :return: Output shape tuple.
        :rtype: Tuple[Optional[int], ...]
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
    """Multi-element tokenisation encoder for heterogeneous graph nodes.

    Decomposes each graph node into five fundamental components (features, type,
    hop distance, relative time, structural position) and learns specialised
    representations for each before combining them via element-wise addition:
    T = Norm(Dropout(E_feat(x) + E_type(t) + E_hop(h) + E_time(tau) + E_pe(A))).
    The structural component E_pe uses a lightweight GNN stack operating on
    random features propagated through the subgraph adjacency.

    **Architecture Overview:**

    .. code-block:: text

        ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌────────────┐
        │ Node Feats  │ │ Node Types  │ │ Hop Dists   │ │ Rel Times  │
        │ [B,K,F]     │ │ [B,K]       │ │ [B,K]       │ │ [B,K,1]    │
        └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └─────┬──────┘
               ▼               ▼               ▼              ▼
        ┌────────────┐  ┌────────────┐  ┌────────────┐ ┌────────────┐
        │  Dense     │  │ Embedding  │  │ Embedding  │ │  Dense     │
        │  → [B,K,E] │  │ → [B,K,E]  │  │ → [B,K,E]  │ │ → [B,K,E]  │
        └──────┬─────┘  └──────┬─────┘  └──────┬─────┘ └─────┬──────┘
               │               │               │             │
               │  ┌────────────────────┐        │             │
               │  │ Subgraph Adj [B,K,K]│       │             │
               │  └─────────┬──────────┘        │             │
               │            ▼                   │             │
               │  ┌────────────────────┐        │             │
               │  │ Z ~ N(0,1) [B,K,P] │       │             │
               │  └─────────┬──────────┘        │             │
               │            ▼                   │             │
               │  ┌────────────────────┐        │             │
               │  │ GNN Stack (L layers)│       │             │
               │  └─────────┬──────────┘        │             │
               │            ▼                   │             │
               │  ┌────────────────────┐        │             │
               │  │ Dense → [B,K,E]    │        │             │
               │  └─────────┬──────────┘        │             │
               │            │                   │             │
               └─────┬──────┴──────┬────────────┴─────┬───────┘
                     ▼             ▼                   ▼
               ┌──────────────────────────────────────────┐
               │  Element-wise Addition                    │
               └──────────────────┬───────────────────────┘
                                  ▼
               ┌────────────────────────┐
               │  Norm → Dropout        │
               └──────────┬─────────────┘
                          ▼
               ┌────────────────────────┐
               │  Token Output [B,K,E]  │
               └────────────────────────┘

    :param embedding_dim: Dimensionality of final token embedding. Must be positive.
    :type embedding_dim: int
    :param num_node_types: Total number of unique entity types. Must be positive.
    :type num_node_types: int
    :param max_hops: Maximum hop distance to encode. Defaults to 2.
    :type max_hops: int
    :param gnn_pe_dim: Output dimension for GNN positional encoding layers.
        Defaults to 32.
    :type gnn_pe_dim: int
    :param gnn_pe_layers: Number of GNN layers for positional encoding.
        Defaults to 2.
    :type gnn_pe_layers: int
    :param dropout_rate: Dropout probability in ``[0, 1]``. Defaults to 0.1.
    :type dropout_rate: float
    :param normalization_type: Type of normalization. Defaults to ``'layer_norm'``.
    :type normalization_type: str
    :param kernel_initializer: Initializer for Dense layers.
        Defaults to ``'glorot_uniform'``.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kwargs: Additional arguments for the ``Layer`` base class.
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
        """Build all sub-layers.

        :param input_shape: Dictionary of input shapes keyed by input name.
        :type input_shape: Dict[str, Tuple[Optional[int], ...]]
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
        """Forward pass for multi-element tokenisation.

        :param inputs: Dictionary with keys ``node_features``, ``node_types``,
            ``hop_distances``, ``relative_times``, ``subgraph_adjacency``.
        :type inputs: Dict[str, keras.KerasTensor]
        :param training: Whether in training mode.
        :type training: Optional[bool]
        :return: Token embeddings ``(batch_size, k_neighbors, embedding_dim)``.
        :rtype: keras.KerasTensor
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
        """Compute output shape.

        :param input_shape: Dictionary of input shapes.
        :type input_shape: Dict[str, Tuple[Optional[int], ...]]
        :return: Output shape tuple.
        :rtype: Tuple[Optional[int], ...]
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
    """Hybrid local-global Transformer block for relational graph processing.

    Combines local self-attention over sampled subgraph tokens with global
    cross-attention to learnable centroid prototypes. The combined representation
    is computed as: output = Norm(FFN([h_local; h_global]) + ResidualProj([h_local; h_global]))
    where h_local = MeanPool(TransformerLayer(tokens)) and
    h_global = Squeeze(CrossAttention(Q=seed, K=V=centroids)).

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────┐        ┌──────────────────┐
        │ Local Tokens     │        │ Seed Node Feats  │
        │ [B, K, E]        │        │ [B, 1, E]        │
        └────────┬─────────┘        └────────┬─────────┘
                 ▼                           │
        ┌──────────────────┐                 │
        │ LOCAL MODULE     │                 │
        │ TransformerLayer │                 │
        │   → Mean Pool    │                 │
        │   → h_local [B,E]│                 │
        └────────┬─────────┘                 │
                 │                           ▼
                 │              ┌──────────────────────┐
                 │              │ GLOBAL MODULE         │
                 │              │ Centroids [G, E]      │
                 │              │ CrossAttn(Q=Seed,     │
                 │              │   K=V=Centroids)      │
                 │              │ → h_global [B, E]     │
                 │              └──────────┬───────────┘
                 └──────────┬──────────────┘
                            ▼
                 ┌─────────────────────┐
                 │ Concat [B, 2E]      │
                 └──────────┬──────────┘
                    ┌───────┴───────┐
                    ▼               ▼
             ┌────────────┐  ┌────────────┐
             │ Residual   │  │ FFN+Drop   │
             │ Dense→[E]  │  │ → [E]      │
             └──────┬─────┘  └──────┬─────┘
                    └───────┬───────┘
                            ▼
                 ┌─────────────────────┐
                 │ Add & Norm → [B, E] │
                 └─────────────────────┘

    :param embedding_dim: Dimensionality of token and output embeddings.
        Must be positive.
    :type embedding_dim: int
    :param num_heads: Number of attention heads. Must divide ``embedding_dim``.
    :type num_heads: int
    :param num_global_centroids: Number of learnable global centroid tokens.
    :type num_global_centroids: int
    :param ffn_dim: Hidden dimension for feed-forward networks. Must be positive.
    :type ffn_dim: int
    :param dropout_rate: Dropout probability in ``[0, 1]``. Defaults to 0.1.
    :type dropout_rate: float
    :param ffn_type: Type of FFN. Defaults to ``'mlp'``.
    :type ffn_type: str
    :param normalization_type: Type of normalization. Defaults to ``'layer_norm'``.
    :type normalization_type: str
    :param kernel_initializer: Initializer for Dense layers.
        Defaults to ``'glorot_uniform'``.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kwargs: Additional arguments for the ``Layer`` base class.
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
        """Build sub-layers and create global centroid weights.

        :param input_shape: List of ``[local_tokens_shape, seed_features_shape]``.
        :type input_shape: List[Tuple[Optional[int], ...]]
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
        """Forward pass for hybrid local-global processing.

        :param inputs: List of ``[local_tokens, seed_node_features]``.
        :type inputs: List[keras.KerasTensor]
        :param training: Whether in training mode.
        :type training: Optional[bool]
        :return: Combined representation ``(batch_size, embedding_dim)``.
        :rtype: keras.KerasTensor
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
        """Compute output shape.

        :param input_shape: List of ``[local_tokens_shape, seed_features_shape]``.
        :type input_shape: List[Tuple[Optional[int], ...]]
        :return: Output shape tuple.
        :rtype: Tuple[Optional[int], ...]
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
