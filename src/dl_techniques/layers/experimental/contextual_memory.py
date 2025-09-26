"""
Contextual Memory Bank Implementation for Keras 3.x

This module implements a sophisticated memory system that combines:
- Key-Value memory store for long-term associations
- Complete configurable Graph Neural Network for concept relationships
- Transformer encoder for temporal patterns
- Feedback loops for dynamic memory updates

Following modern Keras 3 best practices with proper sub-layer management.
"""

import keras
from keras import ops
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Any, Union, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.ffn.mlp import MLPBlock
from dl_techniques.layers.norms.rms_norm import RMSNorm
from dl_techniques.layers.transformer import TransformerLayer

# ---------------------------------------------------------------------

def normalize_adjacency_matrix(adjacency: keras.KerasTensor,
                               normalization: str = 'symmetric') -> keras.KerasTensor:
    """Normalize adjacency matrix for stable GNN training.

    Args:
        adjacency: Adjacency matrix of shape (batch_size, num_nodes, num_nodes)
        normalization: Type of normalization ('row', 'symmetric', or 'none')

    Returns:
        Normalized adjacency matrix

    Note:
        - 'row': D^(-1) * A (row normalization)
        - 'symmetric': D^(-1/2) * A * D^(-1/2) (symmetric normalization)
        - 'none': Returns adjacency matrix unchanged
    """
    if normalization == 'none':
        return adjacency

    # Add self-loops
    batch_size = ops.shape(adjacency)[0]
    num_nodes = ops.shape(adjacency)[1]
    identity = ops.eye(num_nodes)
    identity = ops.expand_dims(identity, axis=0)
    identity = ops.repeat(identity, batch_size, axis=0)
    adjacency_with_self_loops = adjacency + identity

    # Compute degree matrix
    degrees = ops.sum(adjacency_with_self_loops, axis=-1)  # (batch_size, num_nodes)

    if normalization == 'row':
        # Row normalization: D^(-1) * A
        degrees_inv = ops.where(degrees > 0, 1.0 / degrees, 0.0)
        degrees_inv = ops.expand_dims(degrees_inv, axis=-1)  # (batch_size, num_nodes, 1)
        return adjacency_with_self_loops * degrees_inv

    elif normalization == 'symmetric':
        # Symmetric normalization: D^(-1/2) * A * D^(-1/2)
        degrees_inv_sqrt = ops.where(degrees > 0, 1.0 / ops.sqrt(degrees), 0.0)
        degrees_inv_sqrt = ops.expand_dims(degrees_inv_sqrt, axis=-1)  # (batch_size, num_nodes, 1)

        # Apply D^(-1/2) from left
        normalized = adjacency_with_self_loops * degrees_inv_sqrt
        # Apply D^(-1/2) from right
        degrees_inv_sqrt_t = ops.transpose(degrees_inv_sqrt, axes=[0, 2, 1])  # (batch_size, 1, num_nodes)
        normalized = normalized * degrees_inv_sqrt_t
        return normalized

    else:
        raise ValueError(f"Unknown normalization type: {normalization}")

# ---------------------------------------------------------------------

@dataclass
class MemoryBankConfig:
    """Configuration for the Contextual Memory Bank.

    Args:
        memory_dim: Dimension of memory embeddings
        concept_dim: Dimension of concept embeddings in the graph
        temporal_dim: Dimension of temporal embeddings
        num_memory_slots: Number of memory slots in KV store
        num_graph_layers: Number of GNN layers
        num_temporal_heads: Number of attention heads in temporal encoder
        num_temporal_layers: Number of transformer layers for temporal encoding
        max_sequence_length: Maximum length of temporal sequences
        dropout_rate: Dropout rate for regularization
        use_layer_norm: Whether to use layer normalization
        memory_update_rate: Learning rate for memory updates
        graph_aggregation: Type of graph aggregation ('mean', 'max', 'attention', 'sum')
        graph_message_passing: Type of message passing ('gcn', 'graphsage', 'gat', 'gin')
        graph_normalization: Graph normalization type ('none', 'batch', 'layer', 'rms')
        graph_activation: Activation function for graph layers
        temporal_normalization: Temporal normalization type ('layer', 'rms', 'batch')
    """
    memory_dim: int = 512
    concept_dim: int = 256
    temporal_dim: int = 512
    num_memory_slots: int = 1000
    num_graph_layers: int = 3
    num_temporal_heads: int = 8
    num_temporal_layers: int = 6
    max_sequence_length: int = 128
    dropout_rate: float = 0.1
    use_layer_norm: bool = True
    memory_update_rate: float = 0.01
    graph_aggregation: str = 'attention'
    graph_message_passing: str = 'gcn'
    graph_normalization: str = 'layer'
    graph_activation: str = 'relu'
    temporal_normalization: str = 'layer'

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class KeyValueMemoryStore(keras.layers.Layer):
    """Key-Value Memory Store for long-term associations.

    This layer implements a differentiable key-value memory system
    that can store and retrieve associative memories.

    Args:
        num_slots: Number of memory slots. Must be positive.
        memory_dim: Dimension of memory embeddings. Must be positive.
        key_dim: Dimension of keys. Must be positive.
        temperature: Temperature for attention softmax. Defaults to 1.0.
        use_bias: Whether to use bias in projections. Defaults to True.
        kernel_initializer: Initializer for kernel weights.
        bias_initializer: Initializer for bias weights.
        **kwargs: Additional keyword arguments for Layer base class

    Input shape:
        Query tensor of shape (batch_size, key_dim)

    Output shape:
        Retrieved memory tensor of shape (batch_size, memory_dim)

    Example:
        ```python
        memory_store = KeyValueMemoryStore(
            num_slots=1000,
            memory_dim=512,
            key_dim=256
        )
        query = keras.Input(shape=(256,))
        retrieved = memory_store(query)
        ```
    """

    def __init__(
            self,
            num_slots: int,
            memory_dim: int,
            key_dim: int,
            temperature: float = 1.0,
            use_bias: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if num_slots <= 0:
            raise ValueError(f"num_slots must be positive, got {num_slots}")
        if memory_dim <= 0:
            raise ValueError(f"memory_dim must be positive, got {memory_dim}")
        if key_dim <= 0:
            raise ValueError(f"key_dim must be positive, got {key_dim}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        self.num_slots = num_slots
        self.memory_dim = memory_dim
        self.key_dim = key_dim
        self.temperature = temperature
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)

        # Initialize weight attributes - created in build()
        self.memory_keys = None
        self.memory_values = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the memory store weights."""
        # Create memory keys and values
        self.memory_keys = self.add_weight(
            name="memory_keys",
            shape=(self.num_slots, self.key_dim),
            initializer=self.kernel_initializer,
            trainable=True,
        )

        self.memory_values = self.add_weight(
            name="memory_values",
            shape=(self.num_slots, self.memory_dim),
            initializer=self.kernel_initializer,
            trainable=True,
        )

        super().build(input_shape)

    def call(self, query: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Retrieve memories based on query.

        Args:
            query: Query tensor of shape (batch_size, key_dim)
            training: Whether in training mode

        Returns:
            Retrieved memory tensor of shape (batch_size, memory_dim)
        """
        # Compute attention weights between query and memory keys
        # Shape: (batch_size, 1, key_dim) @ (key_dim, num_slots) -> (batch_size, 1, num_slots)
        query_expanded = ops.expand_dims(query, axis=1)
        attention_logits = ops.matmul(query_expanded, ops.transpose(self.memory_keys))

        # Apply temperature scaling
        attention_logits = attention_logits / self.temperature
        attention_weights = ops.softmax(attention_logits, axis=-1)

        # Retrieve weighted memory values
        # Shape: (batch_size, 1, num_slots) @ (num_slots, memory_dim) -> (batch_size, 1, memory_dim)
        retrieved_memory = ops.matmul(attention_weights, self.memory_values)

        # Remove the singleton dimension: (batch_size, memory_dim)
        return ops.squeeze(retrieved_memory, axis=1)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        return tuple(list(input_shape)[:-1] + [self.memory_dim])

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "num_slots": self.num_slots,
            "memory_dim": self.memory_dim,
            "key_dim": self.key_dim,
            "temperature": self.temperature,
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
        })
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class GraphNeuralNetworkLayer(keras.layers.Layer):
    """Complete configurable Graph Neural Network for concept relationship modeling.

    Implements various GNN architectures including GCN, GraphSAGE, GAT, and GIN
    with configurable message passing, aggregation, and normalization.

    Args:
        concept_dim: Dimension of concept embeddings. Must be positive.
        num_layers: Number of GNN layers. Must be positive. Defaults to 3.
        message_passing: Type of message passing ('gcn', 'graphsage', 'gat', 'gin'). Defaults to 'gcn'.
        aggregation: Type of aggregation ('mean', 'max', 'attention', 'sum'). Defaults to 'attention'.
        normalization: Type of normalization ('none', 'batch', 'layer', 'rms'). Defaults to 'layer'.
        activation: Activation function. Defaults to 'relu'.
        dropout_rate: Dropout rate. Must be between 0 and 1. Defaults to 0.1.
        use_residual: Whether to use residual connections. Defaults to True.
        use_layer_norm: Whether to use layer normalization. Defaults to True.
        num_attention_heads: Number of attention heads for GAT and attention aggregation. Defaults to 4.
        **kwargs: Additional keyword arguments for Layer base class

    Input shape:
        Tuple of (node_features, adjacency_matrix):
        - node_features: Shape (batch_size, num_nodes, concept_dim)
        - adjacency_matrix: Shape (batch_size, num_nodes, num_nodes)

    Output shape:
        Updated node embeddings of shape (batch_size, num_nodes, concept_dim)

    Example:
        ```python
        gnn = GraphNeuralNetworkLayer(
            concept_dim=256,
            num_layers=3,
            message_passing='gat',
            aggregation='attention'
        )

        node_features = keras.Input(shape=(10, 256))
        adjacency = keras.Input(shape=(10, 10))
        output = gnn((node_features, adjacency))
        ```
    """

    def __init__(
            self,
            concept_dim: int,
            num_layers: int = 3,
            message_passing: Literal['gcn', 'graphsage', 'gat', 'gin'] = 'gcn',
            aggregation: Literal['mean', 'max', 'attention', 'sum'] = 'attention',
            normalization: Literal['none', 'batch', 'layer', 'rms'] = 'layer',
            activation: str = 'relu',
            dropout_rate: float = 0.1,
            use_residual: bool = True,
            use_layer_norm: bool = True,
            num_attention_heads: int = 4,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if concept_dim <= 0:
            raise ValueError(f"concept_dim must be positive, got {concept_dim}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")
        if num_attention_heads <= 0:
            raise ValueError(f"num_attention_heads must be positive, got {num_attention_heads}")

        self.concept_dim = concept_dim
        self.num_layers = num_layers
        self.message_passing = message_passing
        self.aggregation = aggregation
        self.normalization = normalization
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        self.num_attention_heads = num_attention_heads

        # Create sub-layers in __init__
        self.gnn_layers = []
        self.dropout_layers = []
        self.norm_layers = []
        self.attention_layers = []

        for i in range(self.num_layers):
            # Message passing layers
            if self.message_passing == 'gcn':
                self.gnn_layers.append(
                    keras.layers.Dense(
                        self.concept_dim,
                        activation=None,  # Apply activation separately
                        name=f'gcn_layer_{i}'
                    )
                )
            elif self.message_passing == 'graphsage':
                # GraphSAGE uses separate transformations for self and neighbor
                self.gnn_layers.append([
                    keras.layers.Dense(self.concept_dim, activation=None, name=f'sage_self_{i}'),
                    keras.layers.Dense(self.concept_dim, activation=None, name=f'sage_neighbor_{i}')
                ])
            elif self.message_passing == 'gat':
                # Graph Attention Network
                self.gnn_layers.append(
                    keras.layers.MultiHeadAttention(
                        num_heads=self.num_attention_heads,
                        key_dim=self.concept_dim // self.num_attention_heads,
                        dropout=self.dropout_rate,
                        name=f'gat_layer_{i}'
                    )
                )
            elif self.message_passing == 'gin':
                # Graph Isomorphism Network
                self.gnn_layers.append(
                    MLPBlock(
                        hidden_dim=self.concept_dim * 2,
                        output_dim=self.concept_dim,
                        activation=self.activation,
                        dropout_rate=self.dropout_rate,
                        name=f'gin_mlp_{i}'
                    )
                )

            # Dropout layers
            self.dropout_layers.append(
                keras.layers.Dropout(self.dropout_rate, name=f'gnn_dropout_{i}')
            )

            # Normalization layers
            if self.normalization == 'layer':
                self.norm_layers.append(
                    keras.layers.LayerNormalization(name=f'gnn_layer_norm_{i}')
                )
            elif self.normalization == 'rms':
                self.norm_layers.append(
                    RMSNorm(name=f'gnn_rms_norm_{i}')
                )
            elif self.normalization == 'batch':
                self.norm_layers.append(
                    keras.layers.BatchNormalization(name=f'gnn_batch_norm_{i}')
                )
            else:
                self.norm_layers.append(None)

        # Aggregation layer
        if self.aggregation == 'attention':
            self.aggregation_attention = keras.layers.MultiHeadAttention(
                num_heads=4,
                key_dim=self.concept_dim // 4,
                name='aggregation_attention'
            )
        else:
            self.aggregation_attention = None

    def build(self, input_shape: Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]) -> None:
        """Build GNN layers."""
        node_shape, adjacency_shape = input_shape

        # Build all sub-layers
        for i in range(self.num_layers):
            if self.message_passing == 'gcn':
                self.gnn_layers[i].build(node_shape)
            elif self.message_passing == 'graphsage':
                self.gnn_layers[i][0].build(node_shape)  # self transformation
                self.gnn_layers[i][1].build(node_shape)  # neighbor transformation
            elif self.message_passing == 'gat':
                self.gnn_layers[i].build(node_shape, node_shape)
            elif self.message_passing == 'gin':
                self.gnn_layers[i].build(node_shape)

            self.dropout_layers[i].build(node_shape)

            if self.norm_layers[i] is not None:
                self.norm_layers[i].build(node_shape)

        if self.aggregation_attention is not None:
            self.aggregation_attention.build(node_shape, node_shape)

        super().build(input_shape)

    def call(self, inputs: Tuple[keras.KerasTensor, keras.KerasTensor],
             training: Optional[bool] = None) -> keras.KerasTensor:
        """Process concept graph.

        Args:
            inputs: Tuple of (node_features, adjacency_matrix)
                - node_features: Shape (batch_size, num_nodes, concept_dim)
                - adjacency_matrix: Shape (batch_size, num_nodes, num_nodes)
            training: Whether in training mode

        Returns:
            Updated node embeddings of shape (batch_size, num_nodes, concept_dim)
        """
        node_features, adjacency_matrix = inputs

        # Normalize adjacency matrix for stable training
        normalized_adj = normalize_adjacency_matrix(adjacency_matrix, normalization='symmetric')

        # Process through GNN layers
        h = node_features
        for i in range(self.num_layers):
            h_input = h

            if self.message_passing == 'gcn':
                # GCN: H' = σ(A * H * W)
                messages = ops.matmul(normalized_adj, h)
                h_new = self.gnn_layers[i](messages)

            elif self.message_passing == 'graphsage':
                # GraphSAGE: H' = σ(W_self * H + W_neighbor * AGG(A * H))
                self_transform = self.gnn_layers[i][0](h)
                neighbor_messages = ops.matmul(normalized_adj, h)
                neighbor_transform = self.gnn_layers[i][1](neighbor_messages)
                h_new = self_transform + neighbor_transform

            elif self.message_passing == 'gat':
                # GAT: Use attention to weight neighbors
                h_new = self.gnn_layers[i](h, h, training=training)

            elif self.message_passing == 'gin':
                # GIN: H' = MLP((1 + ε) * H + A * H)
                neighbor_messages = ops.matmul(normalized_adj, h)
                combined = h + neighbor_messages  # ε = 1 for simplicity
                h_new = self.gnn_layers[i](combined, training=training)

            # Apply activation
            if self.message_passing != 'gin':  # GIN MLP already has activation
                h_new = keras.activations.get(self.activation)(h_new)

            # Apply dropout
            h_new = self.dropout_layers[i](h_new, training=training)

            # Residual connection
            if self.use_residual and h_input.shape[-1] == h_new.shape[-1]:
                h = h_input + h_new
            else:
                h = h_new

            # Normalization
            if self.norm_layers[i] is not None:
                h = self.norm_layers[i](h, training=training)

        # Final aggregation
        if self.aggregation == 'attention' and self.aggregation_attention is not None:
            h = self.aggregation_attention(h, h, training=training)
        elif self.aggregation == 'mean':
            h = ops.mean(h, axis=1, keepdims=True)
        elif self.aggregation == 'max':
            h = ops.max(h, axis=1, keepdims=True)
        elif self.aggregation == 'sum':
            h = ops.sum(h, axis=1, keepdims=True)

        return h

    def compute_output_shape(self, input_shape: Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        node_shape, _ = input_shape
        if self.aggregation in ['mean', 'max', 'sum']:
            return tuple(list(node_shape)[:-2] + [1, self.concept_dim])
        return node_shape

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "concept_dim": self.concept_dim,
            "num_layers": self.num_layers,
            "message_passing": self.message_passing,
            "aggregation": self.aggregation,
            "normalization": self.normalization,
            "activation": self.activation,
            "dropout_rate": self.dropout_rate,
            "use_residual": self.use_residual,
            "use_layer_norm": self.use_layer_norm,
            "num_attention_heads": self.num_attention_heads,
        })
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class TemporalContextEncoder(keras.layers.Layer):
    """Temporal Context Encoder using modern Transformer architecture.

    Encodes temporal sequences using the framework's TransformerLayer
    to capture temporal dependencies and patterns.

    Args:
        temporal_dim: Dimension of temporal embeddings. Must be positive.
        num_heads: Number of attention heads. Must be positive. Defaults to 8.
        num_layers: Number of transformer layers. Must be positive. Defaults to 6.
        max_sequence_length: Maximum sequence length. Must be positive. Defaults to 128.
        dropout_rate: Dropout rate. Must be between 0 and 1. Defaults to 0.1.
        normalization_type: Type of normalization ('layer_norm', 'rms_norm'). Defaults to 'layer_norm'.
        ffn_type: Type of FFN ('mlp', 'swiglu'). Defaults to 'mlp'.
        **kwargs: Additional keyword arguments for Layer base class

    Input shape:
        Temporal sequence tensor of shape (batch_size, seq_length, temporal_dim)

    Output shape:
        Encoded sequence tensor of shape (batch_size, seq_length, temporal_dim)

    Example:
        ```python
        encoder = TemporalContextEncoder(
            temporal_dim=512,
            num_heads=8,
            num_layers=6
        )

        sequence = keras.Input(shape=(128, 512))
        encoded = encoder(sequence)
        ```
    """

    def __init__(
            self,
            temporal_dim: int,
            num_heads: int = 8,
            num_layers: int = 6,
            max_sequence_length: int = 128,
            dropout_rate: float = 0.1,
            normalization_type: str = 'layer_norm',
            ffn_type: str = 'mlp',
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if temporal_dim <= 0:
            raise ValueError(f"temporal_dim must be positive, got {temporal_dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if temporal_dim % num_heads != 0:
            raise ValueError(f"temporal_dim ({temporal_dim}) must be divisible by num_heads ({num_heads})")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if max_sequence_length <= 0:
            raise ValueError(f"max_sequence_length must be positive, got {max_sequence_length}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")

        self.temporal_dim = temporal_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_sequence_length = max_sequence_length
        self.dropout_rate = dropout_rate
        self.normalization_type = normalization_type
        self.ffn_type = ffn_type

        # Create sub-layers in __init__
        self.positional_embedding = keras.layers.Embedding(
            input_dim=self.max_sequence_length,
            output_dim=self.temporal_dim,
            name='positional_embedding'
        )

        # Use framework's TransformerLayer
        self.transformer_layers = []
        for i in range(self.num_layers):
            self.transformer_layers.append(
                TransformerLayer(
                    hidden_size=self.temporal_dim,
                    num_heads=self.num_heads,
                    intermediate_size=self.temporal_dim * 4,
                    attention_type='multi_head',
                    normalization_type=self.normalization_type,
                    ffn_type=self.ffn_type,
                    dropout_rate=self.dropout_rate,
                    name=f'temporal_transformer_{i}'
                )
            )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build temporal encoder layers."""
        # Build positional embedding
        pos_input_shape = (input_shape[0], input_shape[1])  # (batch_size, seq_length)
        self.positional_embedding.build(pos_input_shape)

        # Build transformer layers
        for transformer in self.transformer_layers:
            transformer.build(input_shape)

        super().build(input_shape)

    def call(self, sequence: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Encode temporal sequence.

        Args:
            sequence: Input sequence of shape (batch_size, seq_length, temporal_dim)
            training: Whether in training mode

        Returns:
            Encoded sequence of shape (batch_size, seq_length, temporal_dim)
        """
        batch_size = ops.shape(sequence)[0]
        seq_length = ops.shape(sequence)[1]

        # Add positional embeddings
        positions = ops.arange(seq_length)
        positions = ops.expand_dims(positions, axis=0)
        positions = ops.repeat(positions, batch_size, axis=0)

        pos_embeddings = self.positional_embedding(positions, training=training)
        x = sequence + pos_embeddings

        # Process through transformer layers
        for transformer in self.transformer_layers:
            x = transformer(x, training=training)

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "temporal_dim": self.temporal_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "max_sequence_length": self.max_sequence_length,
            "dropout_rate": self.dropout_rate,
            "normalization_type": self.normalization_type,
            "ffn_type": self.ffn_type,
        })
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ContextualMemoryBank(keras.layers.Layer):
    """Contextual Memory Bank integrating KV memory, GNN, and temporal encoding.

    This layer implements a sophisticated memory system that combines:
    - Key-Value memory store for long-term associations
    - Complete configurable Graph Neural Network for concept relationships
    - Transformer encoder for temporal patterns
    - Integration layer for combining all modalities

    Args:
        config: MemoryBankConfig object with system parameters. If None, uses default config.
        **kwargs: Additional keyword arguments for Layer base class

    Input shape:
        Dictionary containing:
        - 'query': Query tensor for memory retrieval (batch_size, concept_dim)
        - 'concept_graph': Tuple of (node_features, adjacency_matrix)
        - 'temporal_sequence': Temporal sequence (batch_size, seq_len, temporal_dim)

    Output shape:
        Dictionary containing:
        - 'integrated_output': Final integrated contextual output (batch_size, memory_dim)
        - 'memory_output': Retrieved memory (batch_size, memory_dim)
        - 'graph_output': Graph neural network output (batch_size, num_nodes, concept_dim)
        - 'temporal_output': Temporal encoder output (batch_size, seq_len, temporal_dim)

    Example:
        ```python
        config = MemoryBankConfig(memory_dim=512, concept_dim=256)
        memory_bank = ContextualMemoryBank(config=config)

        # Define inputs
        query = keras.Input(shape=(256,))
        nodes = keras.Input(shape=(10, 256))
        adjacency = keras.Input(shape=(10, 10))
        temporal = keras.Input(shape=(128, 512))

        outputs = memory_bank({
            'query': query,
            'concept_graph': (nodes, adjacency),
            'temporal_sequence': temporal
        })
        ```
    """

    def __init__(
            self,
            config: Optional[MemoryBankConfig] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.config = config or MemoryBankConfig()

        # Create all sub-layers in __init__
        # Key-Value Memory Store
        self.memory_store = KeyValueMemoryStore(
            num_slots=self.config.num_memory_slots,
            memory_dim=self.config.memory_dim,
            key_dim=self.config.concept_dim,
            name='kv_memory_store'
        )

        # Complete configurable Graph Neural Network
        self.graph_network = GraphNeuralNetworkLayer(
            concept_dim=self.config.concept_dim,
            num_layers=self.config.num_graph_layers,
            message_passing=self.config.graph_message_passing,
            aggregation=self.config.graph_aggregation,
            normalization=self.config.graph_normalization,
            activation=self.config.graph_activation,
            dropout_rate=self.config.dropout_rate,
            use_layer_norm=self.config.use_layer_norm,
            name='concept_graph_network'
        )

        # Temporal Context Encoder using framework's TransformerLayer
        self.temporal_encoder = TemporalContextEncoder(
            temporal_dim=self.config.temporal_dim,
            num_heads=self.config.num_temporal_heads,
            num_layers=self.config.num_temporal_layers,
            max_sequence_length=self.config.max_sequence_length,
            dropout_rate=self.config.dropout_rate,
            normalization_type=self.config.temporal_normalization,
            name='temporal_context_encoder'
        )

        # Integration layer to combine all outputs
        total_dim = self.config.memory_dim + self.config.concept_dim + self.config.temporal_dim
        self.integration_layer = keras.layers.Dense(
            total_dim,
            activation='relu',
            name='integration_layer'
        )

        # Output projection
        self.output_projection = keras.layers.Dense(
            self.config.memory_dim,
            name='output_projection'
        )

    def build(self, input_shape: Dict[str, Any]) -> None:
        """Build all components of the memory bank."""
        # Extract shapes from input dictionary
        query_shape = input_shape['query']
        node_features_shape, adjacency_shape = input_shape['concept_graph']
        temporal_shape = input_shape['temporal_sequence']

        # Build memory store
        self.memory_store.build(query_shape)

        # Build graph network
        self.graph_network.build((node_features_shape, adjacency_shape))

        # Build temporal encoder
        self.temporal_encoder.build(temporal_shape)

        # Build integration layer
        total_dim = self.config.memory_dim + self.config.concept_dim + self.config.temporal_dim
        self.integration_layer.build((query_shape[0], total_dim))

        # Build output projection
        self.output_projection.build((query_shape[0], total_dim))

        super().build(input_shape)

    def call(self, inputs: Dict[str, keras.KerasTensor],
             training: Optional[bool] = None) -> Dict[str, keras.KerasTensor]:
        """Process inputs through the contextual memory bank.

        Args:
            inputs: Dictionary containing:
                - 'query': Query tensor for memory retrieval (batch_size, concept_dim)
                - 'concept_graph': Tuple of (node_features, adjacency_matrix)
                - 'temporal_sequence': Temporal sequence (batch_size, seq_len, temporal_dim)
            training: Whether in training mode

        Returns:
            Dictionary containing:
                - 'integrated_output': Final integrated contextual output
                - 'memory_output': Retrieved memory
                - 'graph_output': Graph neural network output
                - 'temporal_output': Temporal encoder output
        """
        query = inputs['query']
        concept_graph = inputs['concept_graph']
        temporal_sequence = inputs['temporal_sequence']

        # 1. Memory Access Interface - retrieve from KV store
        memory_output = self.memory_store(query, training=training)

        # 2. Graph Neural Network - process concept relationships
        graph_output = self.graph_network(concept_graph, training=training)

        # 3. Temporal Context Encoder - encode temporal sequences
        temporal_output = self.temporal_encoder(temporal_sequence, training=training)

        # Prepare features for integration
        # Average temporal output across sequence dimension for integration
        temporal_summary = ops.mean(temporal_output, axis=1)

        # Process graph output - handle different aggregation types
        if self.config.graph_aggregation in ['mean', 'max', 'sum']:
            # Graph output is already aggregated to (batch_size, 1, concept_dim)
            graph_summary = ops.squeeze(graph_output, axis=1)
        else:
            # For attention aggregation, take mean across nodes
            graph_summary = ops.mean(graph_output, axis=1)

        # 4. Integration - combine all outputs
        concatenated = ops.concatenate([
            memory_output,
            graph_summary,
            temporal_summary
        ], axis=-1)

        integrated_features = self.integration_layer(concatenated, training=training)
        integrated_output = self.output_projection(integrated_features, training=training)

        # Return comprehensive outputs for downstream use
        return {
            'integrated_output': integrated_output,
            'memory_output': memory_output,
            'graph_output': graph_output,
            'temporal_output': temporal_output
        }

    def get_memory_state(self) -> Dict[str, keras.KerasTensor]:
        """Get current memory state for analysis or visualization.

        Returns:
            Dictionary containing memory keys and values
        """
        return {
            'memory_keys': self.memory_store.memory_keys,
            'memory_values': self.memory_store.memory_values
        }

    def compute_output_shape(self, input_shape: Dict[str, Any]) -> Dict[str, Tuple[Optional[int], ...]]:
        """Compute output shapes."""
        batch_size = input_shape.get('query', [None])[0]
        node_features_shape, _ = input_shape.get('concept_graph', ([None, None, None], [None, None, None]))
        temporal_shape = input_shape.get('temporal_sequence', [None, None, None])

        return {
            'integrated_output': (batch_size, self.config.memory_dim),
            'memory_output': (batch_size, self.config.memory_dim),
            'graph_output': (node_features_shape[0], node_features_shape[1], self.config.concept_dim),
            'temporal_output': (temporal_shape[0], temporal_shape[1], self.config.temporal_dim)
        }

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "config": {
                "memory_dim": self.config.memory_dim,
                "concept_dim": self.config.concept_dim,
                "temporal_dim": self.config.temporal_dim,
                "num_memory_slots": self.config.num_memory_slots,
                "num_graph_layers": self.config.num_graph_layers,
                "num_temporal_heads": self.config.num_temporal_heads,
                "num_temporal_layers": self.config.num_temporal_layers,
                "max_sequence_length": self.config.max_sequence_length,
                "dropout_rate": self.config.dropout_rate,
                "use_layer_norm": self.config.use_layer_norm,
                "memory_update_rate": self.config.memory_update_rate,
                "graph_aggregation": self.config.graph_aggregation,
                "graph_message_passing": self.config.graph_message_passing,
                "graph_normalization": self.config.graph_normalization,
                "graph_activation": self.config.graph_activation,
                "temporal_normalization": self.config.temporal_normalization,
            }
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ContextualMemoryBank':
        """Create layer from configuration."""
        memory_config = MemoryBankConfig(**config.pop("config", {}))
        return cls(config=memory_config, **config)

# ---------------------------------------------------------------------

def create_contextual_memory_model(
        config: Optional[MemoryBankConfig] = None,
        include_downstream_modules: bool = True
) -> keras.Model:
    """Create a complete model with Contextual Memory Bank.

    Args:
        config: Memory bank configuration. If None, uses default config.
        include_downstream_modules: Whether to include example downstream modules

    Returns:
        Keras model with contextual memory bank

    Example:
        ```python
        # Create with default configuration
        model = create_contextual_memory_model()

        # Create with custom configuration
        config = MemoryBankConfig(
            memory_dim=512,
            concept_dim=256,
            temporal_dim=512,
            num_graph_layers=4,
            graph_message_passing='gat'
        )
        model = create_contextual_memory_model(config=config)
        ```
    """
    if config is None:
        config = MemoryBankConfig()

    # Define inputs with proper shapes
    query_input = keras.Input(shape=(config.concept_dim,), name='query')
    node_features_input = keras.Input(shape=(None, config.concept_dim), name='node_features')
    adjacency_input = keras.Input(shape=(None, None), name='adjacency_matrix')
    temporal_input = keras.Input(shape=(None, config.temporal_dim), name='temporal_sequence')

    # Create the contextual memory bank
    memory_bank = ContextualMemoryBank(config=config, name='contextual_memory_bank')

    # Process inputs
    memory_outputs = memory_bank({
        'query': query_input,
        'concept_graph': (node_features_input, adjacency_input),
        'temporal_sequence': temporal_input
    })

    outputs = {'memory_bank_outputs': memory_outputs}

    if include_downstream_modules:
        # Example downstream modules using framework components

        # Decision engine with MLPBlock
        decision_features = MLPBlock(
            hidden_dim=256,
            output_dim=128,
            activation='relu',
            dropout_rate=config.dropout_rate,
            name='decision_mlp'
        )(memory_outputs['integrated_output'])

        decision_output = keras.layers.Dense(
            10, activation='softmax', name='decision_output'
        )(decision_features)
        outputs['decision'] = decision_output

        # Prediction module
        prediction_features = keras.layers.Dense(
            128, activation='relu', name='prediction_features'
        )(memory_outputs['integrated_output'])

        prediction_output = keras.layers.Dense(
            1, name='prediction_output'
        )(prediction_features)
        outputs['prediction'] = prediction_output

    model = keras.Model(
        inputs=[query_input, node_features_input, adjacency_input, temporal_input],
        outputs=outputs,
        name='contextual_memory_model'
    )

    logger.info("Created Contextual Memory Bank model with components:")
    logger.info(f"- Memory slots: {config.num_memory_slots}")
    logger.info(f"- Graph layers: {config.num_graph_layers} ({config.graph_message_passing})")
    logger.info(f"- Temporal layers: {config.num_temporal_layers}")
    logger.info(f"- Memory dimension: {config.memory_dim}")

    return model