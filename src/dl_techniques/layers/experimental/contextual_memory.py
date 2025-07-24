"""
Contextual Memory Bank Implementation for Keras 3.x

This module implements a sophisticated memory system that combines:
- Key-Value memory store for long-term associations
- Graph Neural Network for concept relationships
- Transformer encoder for temporal patterns
- Feedback loops for dynamic memory updates
"""

import keras
from keras import ops
from typing import Optional, Dict, List, Tuple, Union, Any
import numpy as np
from dataclasses import dataclass

from dl_techniques.utils.logger import logger


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
        graph_aggregation: Type of graph aggregation ('mean', 'max', 'attention')
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


@keras.saving.register_keras_serializable()
class KeyValueMemoryStore(keras.layers.Layer):
    """Key-Value Memory Store for long-term associations.

    This layer implements a differentiable key-value memory system
    that can store and retrieve associative memories.

    Args:
        num_slots: Number of memory slots
        memory_dim: Dimension of memory embeddings
        key_dim: Dimension of keys
        **kwargs: Additional keyword arguments for Layer base class
    """

    def __init__(
            self,
            num_slots: int,
            memory_dim: int,
            key_dim: int,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.num_slots = num_slots
        self.memory_dim = memory_dim
        self.key_dim = key_dim

        # Will be initialized in build()
        self.memory_keys = None
        self.memory_values = None
        self._build_input_shape = None

    def build(self, input_shape) -> None:
        """Build the memory store weights."""
        self._build_input_shape = input_shape

        # Initialize memory keys and values
        self.memory_keys = self.add_weight(
            name="memory_keys",
            shape=(self.num_slots, self.key_dim),
            initializer="glorot_uniform",
            trainable=True,
        )

        self.memory_values = self.add_weight(
            name="memory_values",
            shape=(self.num_slots, self.memory_dim),
            initializer="glorot_uniform",
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
        attention_weights = ops.softmax(attention_logits, axis=-1)

        # Retrieve weighted memory values
        # Shape: (batch_size, 1, num_slots) @ (num_slots, memory_dim) -> (batch_size, 1, memory_dim)
        retrieved_memory = ops.matmul(attention_weights, self.memory_values)

        # Remove the singleton dimension: (batch_size, memory_dim)
        return ops.squeeze(retrieved_memory, axis=1)

    def update_memory(self, key: keras.KerasTensor, value: keras.KerasTensor,
                      update_rate: float = 0.01) -> None:
        """Update memory with new key-value pairs.

        WARNING: This is a non-operational placeholder method. In practice, memory updates
        in neural networks occur through the standard training process via backpropagation.
        The memory_keys and memory_values weights are updated automatically during training
        when gradients flow through the memory retrieval operations.

        To update memories in practice:
        1. Include memory-relevant loss terms in your training objective
        2. Ensure gradients flow through memory operations during backpropagation
        3. Use standard optimizers (Adam, SGD, etc.) to update memory weights

        This method exists for API completeness but does not perform actual updates.

        Args:
            key: New key tensor
            value: New value tensor
            update_rate: Rate of memory update (unused in this placeholder)
        """
        # Find the most similar memory slot
        similarity = ops.matmul(ops.expand_dims(key, axis=0), ops.transpose(self.memory_keys))
        best_slot = ops.argmax(similarity, axis=-1)

        # This is just a placeholder - actual updates happen through training
        logger.warning(f"update_memory called - this is a placeholder. Memory updates occur through training.")
        logger.info(f"Would update memory slot {best_slot} with new association during training")

    def compute_output_shape(self, input_shape) -> Tuple[int, ...]:
        """Compute output shape."""
        return tuple(list(input_shape)[:-1] + [self.memory_dim])

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "num_slots": self.num_slots,
            "memory_dim": self.memory_dim,
            "key_dim": self.key_dim,
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get build configuration."""
        return {"input_shape": self._build_input_shape}

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build from configuration."""
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])


@keras.saving.register_keras_serializable()
class GraphNeuralNetworkLayer(keras.layers.Layer):
    """Graph Neural Network for concept relationship modeling.

    Implements a simplified GNN that processes concept graphs to learn
    relational embeddings between concepts.

    Args:
        concept_dim: Dimension of concept embeddings
        num_layers: Number of GNN layers
        aggregation: Type of aggregation ('mean', 'max', 'attention')
        dropout_rate: Dropout rate
        use_layer_norm: Whether to use layer normalization
        **kwargs: Additional keyword arguments for Layer base class
    """

    def __init__(
            self,
            concept_dim: int,
            num_layers: int = 3,
            aggregation: str = 'attention',
            dropout_rate: float = 0.1,
            use_layer_norm: bool = True,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.concept_dim = concept_dim
        self.num_layers = num_layers
        self.aggregation = aggregation
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm

        # Will be initialized in build()
        self.gnn_layers = []
        self.dropout_layers = []
        self.layer_norms = []
        self.attention_layer = None
        self._build_input_shape = None

    def build(self, input_shape) -> None:
        """Build GNN layers."""
        self._build_input_shape = input_shape

        # Build GNN layers
        for i in range(self.num_layers):
            # Message passing layer
            self.gnn_layers.append(
                keras.layers.Dense(
                    self.concept_dim,
                    activation='relu',
                    name=f'gnn_layer_{i}'
                )
            )

            # Dropout layer
            self.dropout_layers.append(
                keras.layers.Dropout(self.dropout_rate, name=f'gnn_dropout_{i}')
            )

            # Layer normalization
            if self.use_layer_norm:
                self.layer_norms.append(
                    keras.layers.LayerNormalization(name=f'gnn_norm_{i}')
                )

        # Attention layer for aggregation
        if self.aggregation == 'attention':
            self.attention_layer = keras.layers.MultiHeadAttention(
                num_heads=4,
                key_dim=self.concept_dim // 4,
                name='graph_attention'
            )

        super().build(input_shape)

    def call(self, inputs: Tuple[keras.KerasTensor, keras.KerasTensor],
             training: Optional[bool] = None) -> keras.KerasTensor:
        """Process concept graph.

        Args:
            inputs: Tuple of (node_features, adjacency_matrix)
                - node_features: Shape (batch_size, num_nodes, concept_dim)
                - adjacency_matrix: Shape (batch_size, num_nodes, num_nodes)
                  NOTE: adjacency_matrix should be normalized (e.g., using row normalization
                  or symmetric normalization) for stable GNN training. Common normalizations:
                  - Row normalization: A_norm = D^(-1) * A (where D is degree matrix)
                  - Symmetric normalization: A_norm = D^(-1/2) * A * D^(-1/2)
            training: Whether in training mode

        Returns:
            Updated node embeddings of shape (batch_size, num_nodes, concept_dim)
        """
        node_features, adjacency_matrix = inputs

        # Process through GNN layers
        h = node_features
        for i in range(self.num_layers):
            # Message passing: aggregate neighbor features
            # Shape: (batch_size, num_nodes, num_nodes) @ (batch_size, num_nodes, concept_dim)
            messages = ops.matmul(adjacency_matrix, h)

            # Apply transformation
            h_new = self.gnn_layers[i](messages)

            # Apply dropout
            h_new = self.dropout_layers[i](h_new, training=training)

            # Residual connection
            h = h + h_new

            # Layer normalization
            if self.use_layer_norm:
                h = self.layer_norms[i](h)

        # Final aggregation
        if self.aggregation == 'attention' and self.attention_layer is not None:
            h = self.attention_layer(h, h, training=training)
        elif self.aggregation == 'mean':
            h = ops.mean(h, axis=1, keepdims=True)
        elif self.aggregation == 'max':
            h = ops.max(h, axis=1, keepdims=True)

        return h

    def compute_output_shape(self, input_shape) -> Tuple[int, ...]:
        """Compute output shape."""
        node_shape, _ = input_shape
        if self.aggregation in ['mean', 'max']:
            return tuple(list(node_shape)[:-2] + [1, self.concept_dim])
        return node_shape

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "concept_dim": self.concept_dim,
            "num_layers": self.num_layers,
            "aggregation": self.aggregation,
            "dropout_rate": self.dropout_rate,
            "use_layer_norm": self.use_layer_norm,
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get build configuration."""
        return {"input_shape": self._build_input_shape}

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build from configuration."""
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])


@keras.saving.register_keras_serializable()
class TemporalContextEncoder(keras.layers.Layer):
    """Temporal Context Encoder using Transformer architecture.

    Encodes temporal sequences of events using multi-head attention
    to capture temporal dependencies and patterns.

    Args:
        temporal_dim: Dimension of temporal embeddings
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        max_sequence_length: Maximum sequence length
        dropout_rate: Dropout rate
        **kwargs: Additional keyword arguments for Layer base class
    """

    def __init__(
            self,
            temporal_dim: int,
            num_heads: int = 8,
            num_layers: int = 6,
            max_sequence_length: int = 128,
            dropout_rate: float = 0.1,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.temporal_dim = temporal_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_sequence_length = max_sequence_length
        self.dropout_rate = dropout_rate

        # Will be initialized in build()
        self.positional_embedding = None
        self.transformer_layers = []
        self.layer_norms = []
        self.ffn_layers = []
        self._build_input_shape = None

    def build(self, input_shape) -> None:
        """Build temporal encoder layers."""
        self._build_input_shape = input_shape

        # Positional embedding
        self.positional_embedding = keras.layers.Embedding(
            input_dim=self.max_sequence_length,
            output_dim=self.temporal_dim,
            name='positional_embedding'
        )

        # Build transformer layers
        for i in range(self.num_layers):
            # Multi-head attention
            self.transformer_layers.append(
                keras.layers.MultiHeadAttention(
                    num_heads=self.num_heads,
                    key_dim=self.temporal_dim // self.num_heads,
                    dropout=self.dropout_rate,
                    name=f'temporal_attention_{i}'
                )
            )

            # Layer normalization
            self.layer_norms.append([
                keras.layers.LayerNormalization(name=f'temporal_norm1_{i}'),
                keras.layers.LayerNormalization(name=f'temporal_norm2_{i}')
            ])

            # Feed-forward network
            self.ffn_layers.append([
                keras.layers.Dense(
                    self.temporal_dim * 4,
                    activation='relu',
                    name=f'temporal_ffn1_{i}'
                ),
                keras.layers.Dense(
                    self.temporal_dim,
                    name=f'temporal_ffn2_{i}'
                ),
                keras.layers.Dropout(self.dropout_rate, name=f'temporal_ffn_dropout_{i}')
            ])

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

        pos_embeddings = self.positional_embedding(positions)
        x = sequence + pos_embeddings

        # Process through transformer layers
        for i in range(self.num_layers):
            # Multi-head attention with residual connection
            attention_output = self.transformer_layers[i](x, x, training=training)
            x = self.layer_norms[i][0](x + attention_output)

            # Feed-forward network with residual connection
            ffn_output = self.ffn_layers[i][0](x)
            ffn_output = self.ffn_layers[i][1](ffn_output)
            ffn_output = self.ffn_layers[i][2](ffn_output, training=training)
            x = self.layer_norms[i][1](x + ffn_output)

        return x

    def compute_output_shape(self, input_shape) -> Tuple[int, ...]:
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
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get build configuration."""
        return {"input_shape": self._build_input_shape}

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build from configuration."""
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])


@keras.saving.register_keras_serializable()
class ContextualMemoryBank(keras.layers.Layer):
    """Contextual Memory Bank integrating KV memory, GNN, and temporal encoding.

    This layer implements a sophisticated memory system that combines:
    - Key-Value memory store for long-term associations
    - Graph Neural Network for concept relationships
    - Transformer encoder for temporal patterns
    - Feedback loops for dynamic memory updates

    Args:
        config: MemoryBankConfig object with system parameters
        **kwargs: Additional keyword arguments for Layer base class
    """

    def __init__(
            self,
            config: Optional[MemoryBankConfig] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.config = config or MemoryBankConfig()

        # Will be initialized in build()
        self.memory_store = None
        self.graph_network = None
        self.temporal_encoder = None
        self.integration_layer = None
        self.output_projection = None
        self._build_input_shape = None

    def build(self, input_shape) -> None:
        """Build all components of the memory bank."""
        self._build_input_shape = input_shape

        # Key-Value Memory Store
        self.memory_store = KeyValueMemoryStore(
            num_slots=self.config.num_memory_slots,
            memory_dim=self.config.memory_dim,
            key_dim=self.config.concept_dim,
            name='kv_memory_store'
        )

        # Graph Neural Network for concept relationships
        self.graph_network = GraphNeuralNetworkLayer(
            concept_dim=self.config.concept_dim,
            num_layers=self.config.num_graph_layers,
            aggregation=self.config.graph_aggregation,
            dropout_rate=self.config.dropout_rate,
            use_layer_norm=self.config.use_layer_norm,
            name='concept_graph_network'
        )

        # Temporal Context Encoder
        self.temporal_encoder = TemporalContextEncoder(
            temporal_dim=self.config.temporal_dim,
            num_heads=self.config.num_temporal_heads,
            num_layers=self.config.num_temporal_layers,
            max_sequence_length=self.config.max_sequence_length,
            dropout_rate=self.config.dropout_rate,
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

        # 2. Long-Term Association Module - process concept graph
        graph_output = self.graph_network(concept_graph, training=training)

        # 3. Temporal Context Encoder - encode temporal sequences
        temporal_output = self.temporal_encoder(temporal_sequence, training=training)

        # Average temporal output across sequence dimension for integration
        temporal_summary = ops.mean(temporal_output, axis=1)

        # Squeeze graph output if it has singleton dimensions
        if len(ops.shape(graph_output)) > 2:
            graph_summary = ops.squeeze(graph_output, axis=1)
        else:
            graph_summary = graph_output

        # 4. Integration - combine all outputs
        concatenated = ops.concatenate([
            memory_output,
            graph_summary,
            temporal_summary
        ], axis=-1)

        integrated_features = self.integration_layer(concatenated)
        integrated_output = self.output_projection(integrated_features)

        # Return comprehensive outputs for downstream use
        return {
            'integrated_output': integrated_output,
            'memory_output': memory_output,
            'graph_output': graph_output,
            'temporal_output': temporal_output
        }

    def update_memories(self, new_associations: List[Tuple[keras.KerasTensor, keras.KerasTensor]]) -> None:
        """Update memory bank with new associations.

        WARNING: This is a non-operational placeholder method. Memory updates in neural
        networks occur through the standard training process via backpropagation, not
        through explicit update calls.

        To properly update memories:
        1. Design your training data to include memory-relevant examples
        2. Use appropriate loss functions that encourage correct memory associations
        3. Train the entire model end-to-end using standard optimizers
        4. The memory weights will be updated automatically through gradient descent

        This method exists for API completeness but does not perform actual updates.

        Args:
            new_associations: List of (key, value) pairs to add to memory (unused in placeholder)
        """
        logger.warning(f"update_memories called - this is a placeholder. Memory updates occur through training.")
        logger.info(f"Would process {len(new_associations)} new associations during training")
        for key, value in new_associations:
            self.memory_store.update_memory(key, value, self.config.memory_update_rate)

    def get_memory_state(self) -> Dict[str, keras.KerasTensor]:
        """Get current memory state for analysis or visualization.

        Returns:
            Dictionary containing memory keys and values
        """
        return {
            'memory_keys': self.memory_store.memory_keys,
            'memory_values': self.memory_store.memory_values
        }

    def compute_output_shape(self, input_shape) -> Dict[str, Tuple[int, ...]]:
        """Compute output shapes."""
        batch_size = input_shape.get('query', [None])[0]
        return {
            'integrated_output': (batch_size, self.config.memory_dim),
            'memory_output': (batch_size, self.config.memory_dim),
            'graph_output': (batch_size, 1, self.config.concept_dim),
            'temporal_output': (batch_size, self.config.max_sequence_length, self.config.temporal_dim)
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
            }
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ContextualMemoryBank':
        """Create layer from configuration."""
        memory_config = MemoryBankConfig(**config.pop("config", {}))
        return cls(config=memory_config, **config)

    def get_build_config(self) -> Dict[str, Any]:
        """Get build configuration."""
        return {"input_shape": self._build_input_shape}

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build from configuration."""
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])


def create_contextual_memory_model(
        config: Optional[MemoryBankConfig] = None,
        include_downstream_modules: bool = True
) -> keras.Model:
    """Create a complete model with Contextual Memory Bank.

    Args:
        config: Memory bank configuration
        include_downstream_modules: Whether to include example downstream modules

    Returns:
        Keras model with contextual memory bank
    """
    if config is None:
        config = MemoryBankConfig()

    # Define inputs
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
        # Example downstream modules

        # Decision engine
        decision_features = keras.layers.Dense(
            256, activation='relu', name='decision_features'
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
    logger.info(f"- Graph layers: {config.num_graph_layers}")
    logger.info(f"- Temporal layers: {config.num_temporal_layers}")
    logger.info(f"- Memory dimension: {config.memory_dim}")

    return model


# Example usage and testing
if __name__ == "__main__":
    # Create configuration
    config = MemoryBankConfig(
        memory_dim=256,
        concept_dim=128,
        temporal_dim=256,
        num_memory_slots=500,
        num_graph_layers=2,
        num_temporal_heads=4,
        num_temporal_layers=3,
        max_sequence_length=64
    )

    # Create model
    model = create_contextual_memory_model(config, include_downstream_modules=True)

    # Print model summary
    print("Contextual Memory Bank Model:")
    model.summary()

    # Create sample data
    batch_size = 2
    num_nodes = 10
    seq_length = 20

    sample_query = np.random.randn(batch_size, config.concept_dim)
    sample_nodes = np.random.randn(batch_size, num_nodes, config.concept_dim)
    # Create adjacency matrix and normalize it for stable GNN training
    sample_adjacency = np.random.rand(batch_size, num_nodes, num_nodes)
    sample_adjacency = normalize_adjacency_matrix(sample_adjacency, normalization='symmetric')
    sample_temporal = np.random.randn(batch_size, seq_length, config.temporal_dim)

    # Test model
    outputs = model([sample_query, sample_nodes, sample_adjacency, sample_temporal])

    print("\nOutput shapes:")
    for key, output in outputs.items():
        if isinstance(output, dict):
            print(f"{key}:")
            for sub_key, sub_output in output.items():
                print(f"  {sub_key}: {sub_output.shape}")
        else:
            print(f"{key}: {output.shape}")