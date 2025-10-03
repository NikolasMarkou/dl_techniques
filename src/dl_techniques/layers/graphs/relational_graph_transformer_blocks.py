"""
Encode heterogeneous graph node information into unified token embeddings.

This layer implements a multi-element tokenization strategy, a core innovation
of the Relational Graph Transformer (RELGT) model. It addresses the challenge
of representing complex, heterogeneous, and temporal graph data for processing
by a Transformer architecture. Instead of a single monolithic encoder, it
decomposes each node into five fundamental components and learns a specialized,
rich representation for each before combining them into a final token.

The primary design goal is to create a comprehensive token representation that
captures not just a node's intrinsic features but also its categorical type,
its temporal context, its distance from a query's "seed" node, and, most
importantly, its structural role within its local neighborhood.

Architectural Overview:
The encoder operates through five parallel, independent pathways, each
targeting a different aspect of the node's identity:

1.  **Feature Encoder**: A standard linear projection (`Dense`) learns a
    representation of the node's continuous feature vector.
2.  **Type Encoder**: An `Embedding` layer maps the node's categorical type
    (e.g., 'user', 'product') into the shared embedding space.
3.  **Hop Encoder**: An `Embedding` layer encodes the node's graph distance
    (hop count) from the central "seed" node of the subgraph.
4.  **Time Encoder**: A linear projection learns to represent the node's
    relative timestamp, capturing temporal dynamics.
5.  **Subgraph Positional Encoder**: This is the most critical component for
    capturing topology. It uses a lightweight Graph Neural Network (GNN) to
    generate a structural positional encoding. By feeding random features into
    the GNN, its output becomes a function purely of the local graph
    connectivity, creating a unique signature for each node's structural role
    (e.g., hub, bridge, leaf).

The outputs of these five encoders, all in the same `embedding_dim` space, are
summed element-wise. This addition fuses the different facets of information
into a single, unified token vector, which is then normalized and passed
through dropout.

Foundational Mathematics:
The final token embedding `T` for a node is a summation of the outputs of the
five specialized encoders `E_i`:

`T = Norm(Dropout(E_feat(x_feat) + E_type(x_type) + E_hop(x_hop) + E_time(x_time) + E_pe(A_local)))`

The key mathematical concept is the Subgraph Positional Encoding, `E_pe`. It is
computed by a stack of GNN layers operating on the local adjacency matrix
`A_local` with an initial random feature matrix `Z_random`:

`H_pe = GNN_n(...GNN_1(A_local, Z_random))`

The use of `Z_random` is crucial; it ensures that the resulting embedding
`H_pe` is determined solely by the graph's structure, as the GNN's message-
passing mechanism propagates and transforms these random features based on the
local topology. Each GNN layer implements the symmetrically normalized graph
convolution from Kipf & Welling:

`H' = σ(D̃⁻¹/² Ã D̃⁻¹/² H W)`

where `Ã = A + I` is the adjacency matrix with self-loops, and `D̃` is its
degree matrix. This operation effectively averages a node's features with those
of its neighbors, creating an embedding that reflects its structural context.

References:
This architecture synthesizes several foundational ideas in graph deep
learning and sequence modeling:

-   The principle of adding different embedding types (e.g., token and
    positional) is fundamental to the Transformer architecture:
    -   Vaswani, A., et al. (2017). Attention Is All You Need. NIPS.
-   The GNN-based positional encoder leverages the message-passing framework
    popularized by Graph Convolutional Networks:
    -   Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification
        with Graph Convolutional Networks. ICLR.
-   The idea of using GNNs to learn structural or positional encodings is a
    central theme in Graph Transformer models, such as Graphormer.

"""

import keras
from keras import ops, layers, initializers, regularizers, activations
from typing import Optional, Union, Tuple, List, Dict, Any, Callable

# ---------------------------------------------------------------------


from ..ffn import create_ffn_layer
from ..transformer import TransformerLayer
from ..norms import create_normalization_layer

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class LightweightGNNLayer(layers.Layer):
    """
    A lightweight Graph Convolutional Network layer for Subgraph Positional Encoding.

    This layer performs a single step of message passing on a local subgraph,
    implementing the GNN component for the Subgraph PE Encoder in RELGT.
    Uses symmetric normalization of the adjacency matrix for stable message passing.

    **Intent**: Serve as the GNN(A_local, Z_random) component within the
    RELGTTokenEncoder to capture structural information from local topology.

    **Architecture**:
    ```
    Input(A, H) → Add Self-loops → Symmetric Normalization → Message Passing → Output
    ```

    **Mathematical Operation**:
        H' = σ(D̃⁻¹/² Ã D̃⁻¹/² H W)

    Where:
    - Ã = A + I (adjacency with self-loops)
    - D̃ is the degree matrix of Ã
    - W is learnable weight matrix
    - σ is activation function

    Args:
        units: Integer, dimensionality of output feature space. Must be positive.
        activation: String or callable activation function. Defaults to 'relu'.
        kernel_initializer: String or initializer for kernel weights.
            Defaults to 'glorot_uniform'.
        kernel_regularizer: Optional regularizer for kernel weights.
        **kwargs: Additional arguments for Layer base class.

    Input shape:
        List of two tensors:
        - node_features: `(batch_size, num_nodes, input_dim)`
        - adjacency_matrix: `(batch_size, num_nodes, num_nodes)`

    Output shape:
        Tensor with shape `(batch_size, num_nodes, units)`.

    Example:
        ```python
        gnn = LightweightGNNLayer(units=64, activation='relu')
        features = keras.random.normal((2, 10, 32))
        adjacency = keras.random.randint((2, 10, 10), 0, 2)
        output = gnn([features, adjacency])
        ```
    """

    def __init__(
            self,
            units: int,
            activation: Optional[Union[str, Callable]] = "relu",
            kernel_initializer: Union[str, initializers.Initializer] = "glorot_uniform",
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if units <= 0:
            raise ValueError(f"units must be positive, got {units}")

        # Store configuration
        self.units = units
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = kernel_regularizer

        # Weight attributes (created in build)
        self.kernel = None

    def build(self, input_shape: List[Tuple[Optional[int], ...]]) -> None:
        """Create the layer's learnable weights."""
        feature_shape, adjacency_shape = input_shape
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
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass of the GNN layer."""
        node_features, adjacency_matrix = inputs
        batch_size, num_nodes = ops.shape(adjacency_matrix)[:2]

        # Add self-loops to adjacency matrix
        eye_matrix = ops.eye(num_nodes, dtype=self.compute_dtype)
        adj_with_self_loops = adjacency_matrix + eye_matrix

        # Compute symmetric normalization
        row_sum = ops.sum(adj_with_self_loops, axis=-1)  # Degree
        # Avoid division by zero
        d_inv_sqrt = ops.where(
            row_sum > 0,
            ops.power(row_sum, -0.5),
            0.0
        )

        # Create diagonal normalization matrices
        d_inv_sqrt_expanded = ops.expand_dims(d_inv_sqrt, -1)
        d_mat_inv_sqrt = d_inv_sqrt_expanded * eye_matrix

        # Symmetric normalization: D^(-1/2) * A * D^(-1/2)
        normalized_adj = ops.matmul(
            ops.matmul(d_mat_inv_sqrt, adj_with_self_loops),
            d_mat_inv_sqrt
        )

        # Message passing: Apply linear transformation then propagate
        transformed_features = ops.matmul(node_features, self.kernel)
        output = ops.matmul(normalized_adj, transformed_features)

        # Apply activation
        if self.activation is not None:
            output = self.activation(output)

        return output

    def compute_output_shape(self, input_shape: List[Tuple[Optional[int], ...]]) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        feature_shape, _ = input_shape
        output_shape = list(feature_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            "units": self.units,
            "activation": activations.serialize(self.activation),
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
        })
        return config


@keras.saving.register_keras_serializable()
class RELGTTokenEncoder(layers.Layer):
    """
    Multi-element tokenization strategy for the RELGT model.

    This layer implements the novel tokenization approach that decomposes each node
    into five components: features, type, hop distance, time, and local structure.
    Each component is encoded separately and then combined to create rich token
    embeddings suitable for transformer processing.

    **Intent**: Transform heterogeneous, temporal, and topological graph information
    into unified token embeddings without expensive precomputation, enabling efficient
    encoding of relational data complexity.

    **Architecture (5 Encoders)**:
    ```
    Node Features → Dense Projection
    Node Types → Embedding Layer
    Hop Distances → Embedding Layer
    Relative Times → Dense Projection
    Local Structure → Lightweight GNN → Dense Projection
                           ↓
    Element-wise Addition → Layer Normalization → Dropout
    ```

    Args:
        embedding_dim: Integer, dimensionality of final token embedding. Must be positive.
        num_node_types: Integer, total number of unique entity types. Must be positive.
        max_hops: Integer, maximum hop distance to encode. Defaults to 2.
        gnn_pe_dim: Integer, output dimension for GNN positional encoder. Defaults to 32.
        gnn_pe_layers: Integer, number of GNN layers for positional encoding. Defaults to 2.
        dropout_rate: Float between 0 and 1, dropout rate after final projection. Defaults to 0.1.
        normalization_type: String, type of normalization to use. Defaults to 'layer_norm'.
        kernel_initializer: String or initializer for Dense layers. Defaults to 'glorot_uniform'.
        **kwargs: Additional arguments for Layer base class.

    Input shape:
        Dictionary with keys:
        - 'node_features': `(batch_size, k_neighbors, feature_dim)`
        - 'node_types': `(batch_size, k_neighbors)` integer-encoded
        - 'hop_distances': `(batch_size, k_neighbors)` integer-encoded
        - 'relative_times': `(batch_size, k_neighbors, 1)`
        - 'subgraph_adjacency': `(batch_size, k_neighbors, k_neighbors)`

    Output shape:
        Tensor with shape `(batch_size, k_neighbors, embedding_dim)`.

    Example:
        ```python
        encoder = RELGTTokenEncoder(
            embedding_dim=128,
            num_node_types=10,
            max_hops=2,
            dropout_rate=0.1
        )

        inputs = {
            'node_features': keras.random.normal((4, 32, 64)),
            'node_types': keras.random.randint((4, 32), 0, 10),
            'hop_distances': keras.random.randint((4, 32), 0, 3),
            'relative_times': keras.random.normal((4, 32, 1)),
            'subgraph_adjacency': keras.random.randint((4, 32, 32), 0, 2)
        }
        tokens = encoder(inputs)
        ```
    """

    def __init__(
            self,
            embedding_dim: int,
            num_node_types: int,
            max_hops: int = 2,
            gnn_pe_dim: int = 32,
            gnn_pe_layers: int = 2,
            dropout_rate: float = 0.1,
            normalization_type: str = 'layer_norm',
            kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
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
        self.kernel_initializer = initializers.get(kernel_initializer)

        # CREATE all sub-layers in __init__
        # 1. Node Feature Encoder (Dense projection)
        self.feature_encoder = layers.Dense(
            embedding_dim,
            kernel_initializer=self.kernel_initializer,
            name="FeatureEncoder"
        )

        # 2. Node Type Encoder (Embedding)
        self.type_encoder = layers.Embedding(
            input_dim=num_node_types,
            output_dim=embedding_dim,
            name="TypeEncoder",
        )

        # 3. Hop Distance Encoder (Embedding)
        self.hop_encoder = layers.Embedding(
            input_dim=max_hops + 1,  # +1 for 0-hop (self)
            output_dim=embedding_dim,
            name="HopEncoder",
        )

        # 4. Time Encoder (Dense projection)
        self.time_encoder = layers.Dense(
            embedding_dim,
            kernel_initializer=self.kernel_initializer,
            name="TimeEncoder"
        )

        # 5. Subgraph Positional Encoder (Lightweight GNN stack + projection)
        self.gnn_pe_layers_list = [
            LightweightGNNLayer(gnn_pe_dim, name=f"GNNPELayer_{i}")
            for i in range(gnn_pe_layers)
        ]
        self.pe_projection = layers.Dense(
            embedding_dim,
            kernel_initializer=self.kernel_initializer,
            name="PEProjection"
        )

        # Final processing layers
        self.layer_norm = create_normalization_layer(
            normalization_type, name="TokenNormalization"
        )
        self.dropout = layers.Dropout(dropout_rate, name="TokenDropout")

    def build(self, input_shape: Dict[str, Tuple[Optional[int], ...]]) -> None:
        """Build all sub-layers."""
        # Extract shapes for building
        feature_shape = input_shape["node_features"]
        adjacency_shape = input_shape["subgraph_adjacency"]

        # Build feature encoder
        self.feature_encoder.build(feature_shape)

        # Build type and hop encoders (embeddings build automatically)

        # Build time encoder
        time_shape = input_shape["relative_times"]
        self.time_encoder.build(time_shape)

        # Build GNN PE layers sequentially
        current_shape = (None, None, self.gnn_pe_dim)  # Random features shape
        for gnn_layer in self.gnn_pe_layers_list:
            gnn_layer.build([current_shape, adjacency_shape])
            # Update shape for next layer
            current_shape = gnn_layer.compute_output_shape([current_shape, adjacency_shape])

        # Build PE projection
        self.pe_projection.build(current_shape)

        # Build normalization and dropout layers
        token_shape = (*feature_shape[:-1], self.embedding_dim)
        self.layer_norm.build(token_shape)
        self.dropout.build(token_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: Dict[str, keras.KerasTensor],
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass for multi-element tokenization."""
        # Extract input tensors
        node_features = inputs["node_features"]
        node_types = inputs["node_types"]
        hop_distances = inputs["hop_distances"]
        relative_times = inputs["relative_times"]
        subgraph_adjacency = inputs["subgraph_adjacency"]

        batch_size, k_neighbors = ops.shape(node_features)[:2]

        # 1. Node Feature Encoder
        feature_embeddings = self.feature_encoder(node_features)

        # 2. Node Type Encoder
        type_embeddings = self.type_encoder(node_types)

        # 3. Hop Distance Encoder
        hop_embeddings = self.hop_encoder(hop_distances)

        # 4. Time Encoder
        time_embeddings = self.time_encoder(relative_times)

        # 5. Subgraph Positional Encoder
        # Generate random features as described in the paper
        random_features = keras.random.normal(
            shape=(batch_size, k_neighbors, self.gnn_pe_dim),
            dtype=self.compute_dtype
        )

        # Pass through GNN layers
        pe_features = random_features
        for gnn_layer in self.gnn_pe_layers_list:
            pe_features = gnn_layer([pe_features, subgraph_adjacency], training=training)

        # Project to embedding dimension
        pe_embeddings = self.pe_projection(pe_features)

        # Combine all embeddings (element-wise addition as in transformers)
        combined_embeddings = (
                feature_embeddings +
                type_embeddings +
                hop_embeddings +
                time_embeddings +
                pe_embeddings
        )

        # Apply normalization and dropout
        normalized_tokens = self.layer_norm(combined_embeddings)
        output_tokens = self.dropout(normalized_tokens, training=training)

        return output_tokens

    def compute_output_shape(self, input_shape: Dict[str, Tuple[Optional[int], ...]]) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
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
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
        })
        return config


@keras.saving.register_keras_serializable()
class RELGTTransformerBlock(layers.Layer):
    """
    Hybrid local-global Transformer block for the RELGT model.

    This block implements the core innovation of RELGT: combining local attention
    over sampled subgraphs with global attention to learnable centroids. The local
    module processes token interactions within the subgraph, while the global module
    enables database-wide context integration.

    **Intent**: Capture both fine-grained local structural patterns and broad
    global database patterns by integrating detailed local context with learnable
    global representations.

    **Architecture**:
    ```
    Local Tokens → TransformerLayer (Local Self-Attention + FFN)
                → Mean Pooling → h_local

    Seed Node → Cross-Attention with Global Centroids → h_global

    [h_local, h_global] → Concatenate → FFN → Output Representation
    ```

    Args:
        embedding_dim: Integer, dimensionality of token and output embeddings. Must be positive.
        num_heads: Integer, number of attention heads for both local and global attention.
            Must be positive and divide embedding_dim evenly.
        num_global_centroids: Integer, number of learnable global centroid tokens.
            Must be positive.
        ffn_dim: Integer, hidden dimension for feed-forward networks. Must be positive.
        dropout_rate: Float between 0 and 1, dropout rate. Defaults to 0.1.
        ffn_type: String, type of FFN to use. Defaults to 'mlp'.
        normalization_type: String, type of normalization. Defaults to 'layer_norm'.
        **kwargs: Additional arguments for Layer base class.

    Input shape:
        List of two tensors:
        - local_tokens: `(batch_size, k_neighbors, embedding_dim)`
        - seed_node_features: `(batch_size, 1, embedding_dim)`

    Output shape:
        Tensor with shape `(batch_size, embedding_dim)`.

    Example:
        ```python
        transformer_block = RELGTTransformerBlock(
            embedding_dim=128,
            num_heads=8,
            num_global_centroids=16,
            ffn_dim=256,
            dropout_rate=0.1
        )

        local_tokens = keras.random.normal((4, 32, 128))
        seed_features = keras.random.normal((4, 1, 128))
        output = transformer_block([local_tokens, seed_features])
        ```
    """

    def __init__(
            self,
            embedding_dim: int,
            num_heads: int,
            num_global_centroids: int,
            ffn_dim: int,
            dropout_rate: float = 0.1,
            ffn_type: str = 'mlp',
            normalization_type: str = 'layer_norm',
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if embedding_dim % num_heads != 0:
            raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})")
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

        # CREATE sub-layers in __init__

        # Local Module: Use existing TransformerLayer for local processing
        self.local_transformer = TransformerLayer(
            hidden_size=embedding_dim,
            num_heads=num_heads,
            intermediate_size=ffn_dim,
            attention_type='multi_head',
            normalization_type=normalization_type,
            ffn_type=ffn_type,
            dropout_rate=dropout_rate,
            name="LocalTransformer"
        )

        # Global Module: Cross-attention to centroids
        self.global_attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dim // num_heads,
            dropout=dropout_rate,
            name="GlobalAttention"
        )

        # Final combination FFN using factory
        self.combination_ffn = None  # Created in build()
        self.combination_norm = create_normalization_layer(
            normalization_type, name="CombinationNorm"
        )
        self.combination_dropout = layers.Dropout(dropout_rate, name="CombinationDropout")

        # Global centroids (created in build)
        self.global_centroids = None

    def build(self, input_shape: List[Tuple[Optional[int], ...]]) -> None:
        """Build sub-layers and create global centroids."""
        local_tokens_shape, seed_features_shape = input_shape

        # Build local transformer
        self.local_transformer.build(local_tokens_shape)

        # Build global attention
        self.global_attention.build([seed_features_shape, (None, self.num_global_centroids, self.embedding_dim)])

        # Create global centroid weights
        self.global_centroids = self.add_weight(
            name="global_centroids",
            shape=(self.num_global_centroids, self.embedding_dim),
            initializer="glorot_uniform",
            trainable=True,
        )

        # Create combination FFN using factory
        # Input will be concatenation of h_local and h_global: 2 * embedding_dim
        self.combination_ffn = create_ffn_layer(
            self.ffn_type,
            hidden_dim=self.ffn_dim,
            output_dim=self.embedding_dim,
            dropout_rate=self.dropout_rate,
            name="CombinationFFN"
        )

        # Build combination layers
        combined_shape = (None, 2 * self.embedding_dim)
        self.combination_ffn.build(combined_shape)

        output_shape = (None, self.embedding_dim)
        self.combination_norm.build(output_shape)
        self.combination_dropout.build(output_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: List[keras.KerasTensor],
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass for hybrid local-global processing."""
        local_tokens, seed_node_features = inputs
        batch_size = ops.shape(seed_node_features)[0]

        # --- Local Module ---
        # Process local tokens through transformer
        local_processed_tokens = self.local_transformer(local_tokens, training=training)

        # Pool to get single representation per sample
        h_local = ops.mean(local_processed_tokens, axis=1)  # (batch_size, embedding_dim)

        # --- Global Module ---
        # Tile global centroids for batch processing
        global_centroids_batch = ops.tile(
            ops.expand_dims(self.global_centroids, 0),
            [batch_size, 1, 1]
        )

        # Cross-attention from seed node to global centroids
        h_global = self.global_attention(
            query=seed_node_features,  # (batch_size, 1, embedding_dim)
            value=global_centroids_batch,  # (batch_size, num_centroids, embedding_dim)
            key=global_centroids_batch,
            training=training
        )
        h_global = ops.squeeze(h_global, axis=1)  # (batch_size, embedding_dim)

        # --- Combination ---
        # Concatenate local and global representations
        combined_representation = ops.concatenate([h_local, h_global], axis=-1)

        # Process through FFN
        output = self.combination_ffn(combined_representation, training=training)
        output = self.combination_dropout(output, training=training)

        # Residual connection with projection
        # Project combined_representation to match output dimension
        residual_projection = layers.Dense(self.embedding_dim)(combined_representation)
        output = self.combination_norm(output + residual_projection)

        return output

    def compute_output_shape(self, input_shape: List[Tuple[Optional[int], ...]]) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
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
        })
        return config

# ---------------------------------------------------------------------
