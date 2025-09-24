"""
Entity-Graph Refinement Component for LLMs

This module provides a sophisticated component for learning hierarchical relationships
in high-dimensional embedding spaces through iterative graph refinement. It processes
sequences of n-dimensional embeddings to dynamically extract entities, construct
directional relationship graphs, and progressively refine them using learned patterns.

## Overview

The EntityGraphRefinement layer enables dynamic relationship modeling in deep learning
systems by combining attention-based entity extraction with iterative graph learning.
This approach is particularly effective for:

- **Language Models**: Discovering semantic hierarchies and concept relationships
- **Vision Models**: Learning spatial, compositional, and object relationships
- **Multimodal Systems**: Cross-modal entity associations and alignment
- **Knowledge Graphs**: Dynamic relationship discovery and refinement

## Core Architecture

The component operates through four main stages:

1. **Entity Extraction**: Dynamic identification of salient entities from input
   embeddings using multi-head attention against a learnable entity library
2. **Graph Initialization**: Creation of dense directional relationship matrix
   with controlled initial connectivity
3. **Iterative Refinement**: Progressive learning of meaningful relationships
   through multiple refinement steps, each conditioned on entities, previous
   graph state, and global input context
4. **Sparsification**: Learned pruning mechanism to focus on most important
   connections while removing noise

## Mathematical Foundation

The layer implements the following key operations:

- **Entity Extraction**: α = softmax(Q·K^T), E = α·V where Q comes from entity library
- **Graph Initialization**: G₀ ~ U(-ρ, ρ) with density parameter ρ
- **Refinement Step**: G_{t+1} = tanh(MLP([E_i, E_j, G_t[i,j], C])) where C is
  mean-pooled global context from input embeddings
- **Sparsification**: G_final = G * σ(MLP([E_i, E_j])) with learned gating function

This mathematical framework enables learning of complex patterns including hierarchical
structures, causal relationships, temporal dependencies, and compositional semantics
directly from embedding sequences.

## Key Features

- **Modality-Agnostic**: Compatible with any n-dimensional embeddings (text, vision_heads, audio, etc.)
- **Directional Relationships**: Asymmetric adjacency matrix enables hierarchy and causality learning
- **Dynamic Entity Library**: Adaptive entity representations that evolve during training
- **Context-Aware Refinement**: Each refinement step uses global input context for better decisions
- **Configurable Architecture**: Highly flexible API supporting various architectural configurations
- **Production Ready**: Comprehensive validation, robust error handling, and full serialization support
"""

import keras
import numpy as np
from keras import ops, layers
from typing import Optional, Tuple, List, Dict, Any

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class EntityGraphRefinement(keras.layers.Layer):
    """
    Entity-Graph Refinement Component for learning hierarchical relationships in embedding space.

    This layer processes sequences of n-dimensional embeddings through a sophisticated pipeline:

    1. **Entity Extraction**: Uses multi-head attention to dynamically identify entities
       from input sequences by attending to a learnable entity library
    2. **Dense Graph Construction**: Initializes a fully-connected directional relationship
       matrix between all potential entity pairs
    3. **Iterative Refinement**: Progressively refines relationships through multiple steps,
       where each step conditions on entity representations, current graph state, and
       global input context
    4. **Learned Sparsification**: Applies trainable gating to focus on most salient
       relationships while removing noise and irrelevant connections

    **Detailed Architecture Flow**:
    ```
    Input Embeddings [B, S, D]
         │
         ├─→ Positional Encoding (optional)
         │
         ├─→ Entity Extraction via Multi-Head Attention
         │   ├─→ Query: Entity Library [M, D_e]
         │   ├─→ Key/Value: Input Embeddings [B, S, D]
         │   └─→ Output: Extracted Entities [B, M, D_e]
         │
         ├─→ Global Context: mean(Input Embeddings) [B, D]
         │
         ├─→ Dense Graph Initialization [B, M, M]
         │
         ├─→ Iterative Refinement (N steps)
         │   ├─→ Pairwise Entity Features [B, M, M, 2*D_e]
         │   ├─→ Current Edge Weights [B, M, M, 1]
         │   ├─→ Global Context Tiled [B, M, M, D]
         │   ├─→ MLP([entity_i, entity_j, edge_weight, context])
         │   └─→ Updated Graph [B, M, M]
         │
         ├─→ Learned Sparsification
         │   ├─→ Gate MLP([entity_i, entity_j]) → [0,1]
         │   └─→ Graph * Gates [B, M, M]
         │
         └─→ Output: (Entities, Sparse Graph, Entity Mask)
    ```

    Args:
        max_entities: Maximum number of entities in the learnable entity library.
            Controls the size of the relationship graph (max_entities × max_entities).
        entity_dim: Dimensionality of entity representations. Should typically match
            or be close to input embedding dimension for effective attention.
        num_refinement_steps: Number of iterative refinement passes through the graph.
            More steps allow learning of more complex relationship patterns but increase
            computational cost. Defaults to 3.
        initial_density: Controls the magnitude of initial random graph weights,
            sampled from U(-initial_density, initial_density). Higher values create
            denser initial graphs. Defaults to 0.8.
        attention_heads: Number of attention heads for entity extraction. More heads
            can capture diverse relationship patterns but increase parameters. Defaults to 8.
        dropout_rate: Dropout probability applied in refinement and sparsification MLPs
            for regularization. Defaults to 0.1.
        refinement_activation: Activation function used in refinement MLP hidden layers.
            'gelu' provides smooth gradients for better convergence. Defaults to 'gelu'.
        entity_activity_threshold: Minimum attention score required for an entity to be
            considered "active". Entities below this threshold are masked out. Defaults to 0.1.
        use_positional_encoding: Whether to add learnable positional encodings to input
            embeddings. Useful for sequence-order-dependent tasks. Defaults to True.
        max_sequence_length: Maximum sequence length supported for positional encoding.
            Should be >= typical input sequence lengths. Defaults to 1000.
        regularization_weight: Strength of L1 sparsity regularization applied to final
            graph weights. Higher values encourage sparser graphs. Defaults to 0.01.
        activity_regularization_target: Target average activity level for graph weights.
            Used to prevent dead or overly active graphs. Defaults to 0.1.
        **kwargs: Additional arguments passed to Layer base class.

    Input Shape:
        3D tensor: `(batch_size, sequence_length, embedding_dim)`

        - batch_size: Number of sequences in the batch
        - sequence_length: Length of each input sequence
        - embedding_dim: Dimensionality of input embeddings

    Output Shape:
        Tuple of three tensors:

        - **entities**: `(batch_size, max_entities, entity_dim)` - Extracted entity
          representations from attention over input embeddings
        - **graph**: `(batch_size, max_entities, max_entities)` - Sparse directional
          relationship matrix where graph[b,i,j] represents strength of relationship
          from entity i to entity j in batch b
        - **entity_mask**: `(batch_size, max_entities)` - Binary mask indicating which
          entities are active (attention score > threshold) for each batch

    Examples:
        ```python
        # Basic usage for text sequence modeling
        seq_len, embed_dim = 128, 768
        embeddings = keras.Input(shape=(seq_len, embed_dim))

        entity_layer = EntityGraphRefinement(
            max_entities=50,           # Support up to 50 entities
            entity_dim=embed_dim,      # Match input embedding dimension
            num_refinement_steps=4,    # 4 iterative refinement steps
            attention_heads=12         # 12-head attention for entity extraction
        )

        entities, graph, mask = entity_layer(embeddings)
        model = keras.Model(inputs=embeddings, outputs=[entities, graph, mask])

        # Advanced usage with custom hyperparameters
        entity_layer = EntityGraphRefinement(
            max_entities=100,
            entity_dim=512,
            num_refinement_steps=6,
            initial_density=0.5,        # Sparser initial graphs
            attention_heads=16,
            dropout_rate=0.15,          # Higher dropout for regularization
            entity_activity_threshold=0.05,  # Lower threshold for more entities
            regularization_weight=0.02       # Stronger sparsity regularization
        )

        # Integration with transformer models
        transformer_output = transformer_layer(input_embeddings)  # [B, S, D]
        entities, relationships, active_mask = entity_layer(transformer_output)

        # Use extracted relationships for downstream tasks
        relationship_features = keras.layers.GlobalAveragePooling2D()(relationships)
        classifier_output = keras.layers.Dense(num_classes)(relationship_features)
        ```

    Notes:
        - The layer automatically handles variable sequence lengths through masking
        - Entity library weights are learned end-to-end during training
        - Graph weights are directional: graph[i,j] != graph[j,i] in general
        - Sparsification is learned, not hand-tuned, adapting to data patterns
        - Regularization terms are automatically added to model losses during training
        - All sublayers are explicitly built for reliable serialization
    """

    def __init__(
            self,
            max_entities: int,
            entity_dim: int,
            num_refinement_steps: int = 3,
            initial_density: float = 0.8,
            attention_heads: int = 8,
            dropout_rate: float = 0.1,
            refinement_activation: str = 'gelu',
            entity_activity_threshold: float = 0.1,
            use_positional_encoding: bool = True,
            max_sequence_length: int = 1000,
            regularization_weight: float = 0.01,
            activity_regularization_target: float = 0.1,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate all hyperparameters before storing them
        self._validate_hyperparameters(locals())

        # Store validated configuration parameters
        self.max_entities = max_entities
        self.entity_dim = entity_dim
        self.num_refinement_steps = num_refinement_steps
        self.initial_density = initial_density
        self.attention_heads = attention_heads
        self.dropout_rate = dropout_rate
        self.refinement_activation = refinement_activation
        self.entity_activity_threshold = entity_activity_threshold
        self.use_positional_encoding = use_positional_encoding
        self.max_sequence_length = max_sequence_length
        self.regularization_weight = regularization_weight
        self.activity_regularization_target = activity_regularization_target

        # Initialize sublayer references (built in build() method)
        self.entity_library = None          # Learnable entity representations
        self.graph_refinement_mlp = None    # MLP for graph refinement steps
        self.sparsification_gate = None     # MLP for learned sparsification
        self.entity_extractor = None        # Multi-head attention for entity extraction
        self.positional_encoder = None      # Optional positional embeddings

        logger.info(f"Initialized EntityGraphRefinement with {max_entities} max entities, "
                   f"{entity_dim} entity dimensions, {num_refinement_steps} refinement steps")

    def _validate_hyperparameters(self, params: Dict[str, Any]) -> None:
        """
        Comprehensive validation of all constructor hyperparameters.

        Ensures all parameters are within valid ranges and compatible with each other.
        Raises ValueError with descriptive messages for any invalid configurations.

        Args:
            params: Dictionary of parameter names and values from constructor locals()

        Raises:
            ValueError: If any parameter is invalid or incompatible
        """
        # Define validation rules as (condition, error_message) tuples
        validation_rules = [
            (params['max_entities'] <= 0,
             f"max_entities must be positive, got {params['max_entities']}"),
            (params['entity_dim'] <= 0,
             f"entity_dim must be positive, got {params['entity_dim']}"),
            (params['num_refinement_steps'] <= 0,
             f"num_refinement_steps must be positive, got {params['num_refinement_steps']}"),
            (not (0.0 <= params['initial_density'] <= 1.0),
             f"initial_density must be between 0 and 1, got {params['initial_density']}"),
            (params['attention_heads'] <= 0,
             f"attention_heads must be positive, got {params['attention_heads']}"),
            (not (0.0 <= params['dropout_rate'] <= 1.0),
             f"dropout_rate must be between 0 and 1, got {params['dropout_rate']}"),
            (not (0.0 <= params['entity_activity_threshold'] <= 1.0),
             f"entity_activity_threshold must be between 0 and 1, got {params['entity_activity_threshold']}"),
            (params['max_sequence_length'] <= 0,
             f"max_sequence_length must be positive, got {params['max_sequence_length']}"),
            (params['regularization_weight'] < 0,
             f"regularization_weight must be non-negative, got {params['regularization_weight']}"),
            (params['activity_regularization_target'] < 0,
             f"activity_regularization_target must be non-negative, got {params['activity_regularization_target']}")
        ]

        # Check each validation rule and raise error for first violation
        for condition, error_message in validation_rules:
            if condition:
                logger.error(f"Hyperparameter validation failed: {error_message}")
                raise ValueError(error_message)

        logger.debug("All hyperparameters validated successfully")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build all sublayers and weights for the entity-graph refinement component.

        This method creates and explicitly builds all sublayers to ensure proper
        serialization and weight initialization. The build order is carefully
        designed to handle dependencies between components.

        Args:
            input_shape: Shape of input tensor (batch_size, sequence_length, embedding_dim)

        Raises:
            ValueError: If embedding dimension is not specified in input shape
        """
        # Extract and validate input dimensions
        _batch_size, sequence_length, embedding_dim = input_shape
        if embedding_dim is None:
            error_msg = "Embedding dimension of input must be specified for layer building"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"Building EntityGraphRefinement layer with input shape: {input_shape}")

        # Create learnable entity library weight matrix
        # This serves as the "query" templates for entity extraction
        self.entity_library = self.add_weight(
            name='entity_library',
            shape=(self.max_entities, self.entity_dim),
            initializer='glorot_normal',  # Xavier initialization for stable gradients
            trainable=True
        )
        logger.debug(f"Created entity library with shape: ({self.max_entities}, {self.entity_dim})")

        # Build multi-head attention for entity extraction
        # Key dimension per head should be reasonable (at least 1)
        key_dim = max(self.entity_dim // self.attention_heads, 1)
        self.entity_extractor = layers.MultiHeadAttention(
            num_heads=self.attention_heads,
            key_dim=key_dim,
            value_dim=key_dim,
            dropout=self.dropout_rate,
            name='entity_extractor'
        )

        # Explicitly build the attention layer for reliable serialization
        self.entity_extractor.build(
            query_shape=(None, self.max_entities, self.entity_dim),
            value_shape=(None, sequence_length, embedding_dim)
        )
        logger.debug(f"Built entity extractor with {self.attention_heads} heads, key_dim={key_dim}")

        # Build optional positional encoding layer
        if self.use_positional_encoding:
            self.positional_encoder = layers.Embedding(
                input_dim=self.max_sequence_length,
                output_dim=embedding_dim,
                name='positional_encoder'
            )
            # Explicitly build the embedding layer
            self.positional_encoder.build(input_shape=(None, sequence_length))
            logger.debug(f"Built positional encoder for sequences up to length {self.max_sequence_length}")

        # Build graph refinement MLP
        # Input: [entity_i, entity_j, current_edge_weight, global_context]
        refinement_input_dim = 2 * self.entity_dim + 1 + embedding_dim
        self.graph_refinement_mlp = keras.Sequential([
            layers.Dense(
                self.entity_dim,
                activation=self.refinement_activation,
                name='refinement_expand'
            ),
            layers.Dropout(self.dropout_rate, name='refinement_dropout'),
            layers.Dense(1, activation='tanh', name='refinement_contract')  # tanh for bounded outputs
        ], name='graph_refinement_mlp')

        # Explicitly build the sequential model
        self.graph_refinement_mlp.build(input_shape=(None, refinement_input_dim))
        logger.debug(f"Built graph refinement MLP with input dimension {refinement_input_dim}")

        # Build sparsification gate MLP
        # Input: [entity_i, entity_j] -> Gate probability [0,1]
        self.sparsification_gate = keras.Sequential([
            layers.Dense(
                self.entity_dim,
                activation=self.refinement_activation,
                name='sparsification_hidden'
            ),
            layers.Dropout(self.dropout_rate, name='sparsification_dropout'),
            layers.Dense(1, activation='sigmoid', name='sparsification_gate')  # sigmoid for [0,1] gates
        ], name='sparsification_gate')

        # Explicitly build the sequential model
        self.sparsification_gate.build(input_shape=(None, 2 * self.entity_dim))
        logger.debug("Built sparsification gate MLP")

        # Call parent build method to finalize layer construction
        super().build(input_shape)
        logger.info("EntityGraphRefinement layer built successfully")

    def _get_edge_features(self, entities: keras.KerasTensor) -> keras.KerasTensor:
        """
        Create pairwise feature representations for all potential edges in the graph.

        For each pair of entities (i,j), concatenates their representations to create
        edge features that will be used by the refinement and sparsification MLPs.
        This creates a [batch, max_entities, max_entities, 2*entity_dim] tensor where
        each element contains the concatenated features for one potential edge.

        Args:
            entities: Entity representations [batch_size, max_entities, entity_dim]

        Returns:
            Edge features tensor [batch_size, max_entities, max_entities, 2*entity_dim]
            where edge_features[b,i,j] = concat(entities[b,i], entities[b,j])
        """
        # Create all pairwise combinations using broadcasting
        # entities_i[b,i,j] = entities[b,i] for all j
        entities_i = ops.expand_dims(entities, axis=2)  # [B, M, 1, D]
        entities_i = ops.repeat(entities_i, self.max_entities, axis=2)  # [B, M, M, D]

        # entities_j[b,i,j] = entities[b,j] for all i
        entities_j = ops.expand_dims(entities, axis=1)  # [B, 1, M, D]
        entities_j = ops.repeat(entities_j, self.max_entities, axis=1)  # [B, M, M, D]

        # Concatenate to create edge features [B, M, M, 2*D]
        return ops.concatenate([entities_i, entities_j], axis=-1)

    def _extract_entities(
            self,
            embeddings: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Extract entity representations from input embeddings using multi-head attention.

        Uses the learnable entity library as queries to attend over input embeddings.
        Entities with high attention scores (> threshold) are considered "active" and
        will participate in graph construction and refinement.

        Args:
            embeddings: Input embedding sequences [batch_size, seq_length, embed_dim]
            training: Whether layer is in training mode (affects dropout)

        Returns:
            Tuple of:
            - extracted_entities: [batch_size, max_entities, entity_dim]
            - entity_mask: [batch_size, max_entities] binary mask for active entities
        """
        batch_size = ops.shape(embeddings)[0]

        # Prepare entity library as queries for each batch
        # Expand entity library to batch size: [1, M, D] -> [B, M, D]
        entity_queries = ops.expand_dims(self.entity_library, 0)
        entity_queries = ops.repeat(entity_queries, batch_size, axis=0)

        # Apply multi-head attention: entity queries attend to input embeddings
        extracted_entities, attention_scores = self.entity_extractor(
            query=entity_queries,          # [B, max_entities, entity_dim]
            key=embeddings,               # [B, seq_length, embed_dim]
            value=embeddings,             # [B, seq_length, embed_dim]
            training=training,
            return_attention_scores=True  # Need scores for activity detection
        )

        # Determine which entities are "active" based on attention patterns
        # An entity is active if it attends strongly to any part of the input
        # attention_scores: [batch, num_heads, max_entities, seq_length]
        # Take max over heads and sequence dimensions to get per-entity activity
        entity_activity = ops.max(attention_scores, axis=[1, 3])  # [batch, max_entities]
        entity_mask = ops.cast(
            entity_activity > self.entity_activity_threshold,
            dtype='float32'
        )

        # Log entity extraction statistics for monitoring
        if training:
            avg_active_entities = ops.mean(ops.sum(entity_mask, axis=1))
            # FIX: Removed unsupported formatting for symbolic tensor
            logger.debug(f"Entity extraction: average active entities (tensor): {avg_active_entities}")

        return extracted_entities, entity_mask

    def _initialize_dense_graph(self, batch_size: int) -> keras.KerasTensor:
        """
        Initialize dense relationship graph with random weights.

        Creates a fully-connected directional graph where each edge weight is
        randomly sampled from a uniform distribution. The initial density parameter
        controls the magnitude of these initial weights.

        Args:
            batch_size: Number of graphs to create (one per batch element)

        Returns:
            Dense graph tensor [batch_size, max_entities, max_entities] with
            random edge weights in range [-initial_density, initial_density]
        """
        graph_shape = (batch_size, self.max_entities, self.max_entities)

        # Sample initial weights from symmetric uniform distribution
        initial_graph = keras.random.uniform(
            graph_shape,
            minval=-self.initial_density,
            maxval=self.initial_density
        )

        logger.debug(f"Initialized dense graph with shape {graph_shape}, "
                    f"density range ±{self.initial_density}")

        return initial_graph

    def _refine_graph(
            self,
            graph: keras.KerasTensor,
            entities: keras.KerasTensor,
            context: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Perform one iterative refinement step on the relationship graph.

        Uses a shared MLP to update each edge weight based on:
        1. Source and target entity representations
        2. Current edge weight
        3. Global context vector from input embeddings

        This allows the model to learn complex relationship patterns that depend
        on both local entity properties and global input context.

        Args:
            graph: Current graph state [batch_size, max_entities, max_entities]
            entities: Entity representations [batch_size, max_entities, entity_dim]
            context: Global context vector [batch_size, embedding_dim]
            training: Whether layer is in training mode

        Returns:
            Refined graph [batch_size, max_entities, max_entities] with updated edge weights
        """
        batch_size = ops.shape(graph)[0]
        embedding_dim = ops.shape(context)[-1]

        # Create pairwise entity features for all potential edges
        edge_features = self._get_edge_features(entities)  # [B, M, M, 2*entity_dim]

        # Add current edge weights as features
        current_edges = ops.expand_dims(graph, axis=-1)    # [B, M, M, 1]

        # Prepare global context for broadcasting to all edges
        # Reshape context from [B, D] to [B, 1, 1, D] for broadcasting
        context_reshaped = ops.reshape(context, (batch_size, 1, 1, embedding_dim))
        # Tile to match graph dimensions: [B, M, M, D]
        context_tiled = ops.tile(context_reshaped, [1, self.max_entities, self.max_entities, 1])

        # Concatenate all features for MLP input
        # Final shape: [B, M, M, 2*entity_dim + 1 + embedding_dim]
        full_edge_features = ops.concatenate(
            [edge_features, current_edges, context_tiled], axis=-1
        )

        # Flatten for MLP processing
        flat_features = ops.reshape(
            full_edge_features,
            (batch_size * self.max_entities * self.max_entities, -1)
        )

        # Apply refinement MLP to all edges simultaneously
        refined_edges_flat = self.graph_refinement_mlp(flat_features, training=training)

        # Reshape back to graph format
        refined_graph = ops.reshape(
            refined_edges_flat,
            (batch_size, self.max_entities, self.max_entities)
        )

        logger.debug("Completed graph refinement step")
        return refined_graph

    def _apply_sparsification(
            self,
            graph: keras.KerasTensor,
            entities: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply learned sparsification to focus on most important relationships.

        Uses a separate MLP to predict gate probabilities for each edge based on
        the source and target entity representations. Edges with low gate values
        are effectively removed from the final graph.

        Args:
            graph: Dense refined graph [batch_size, max_entities, max_entities]
            entities: Entity representations [batch_size, max_entities, entity_dim]
            training: Whether layer is in training mode

        Returns:
            Sparsified graph [batch_size, max_entities, max_entities] where
            each edge is multiplied by its learned gate probability
        """
        batch_size = ops.shape(graph)[0]

        # Get pairwise entity features for gate prediction
        edge_features = self._get_edge_features(entities)  # [B, M, M, 2*entity_dim]

        # Flatten for MLP processing
        edge_features_flat = ops.reshape(
            edge_features,
            (batch_size * self.max_entities * self.max_entities, -1)
        )

        # Predict gate probabilities using sparsification MLP
        gates_flat = self.sparsification_gate(edge_features_flat, training=training)

        # Reshape gates back to graph format
        gates = ops.reshape(gates_flat, (batch_size, self.max_entities, self.max_entities))

        # Apply gates to graph (element-wise multiplication)
        sparsified_graph = graph * gates

        if training:
            # Log sparsification statistics for monitoring
            avg_gate_value = ops.mean(gates)
            sparsity_ratio = ops.mean(ops.cast(gates < 0.1, 'float32'))
            # FIX: Removed unsupported formatting for symbolic tensors
            logger.debug(f"Sparsification: avg gate value (tensor): {avg_gate_value}, "
                        f"sparsity ratio (tensor): {sparsity_ratio}")

        return sparsified_graph

    def _apply_masks(
            self,
            graph: keras.KerasTensor,
            entity_mask: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Apply entity activity mask and remove self-connections from graph.

        Ensures that:
        1. Only active entities (with high attention scores) participate in relationships
        2. Self-connections are removed (diagonal elements set to 0)
        3. Connections involving inactive entities are zeroed out

        Args:
            graph: Sparsified graph [batch_size, max_entities, max_entities]
            entity_mask: Binary activity mask [batch_size, max_entities]

        Returns:
            Final masked graph [batch_size, max_entities, max_entities]
        """
        # Create connection mask: only allow edges between active entities
        mask_i = ops.expand_dims(entity_mask, axis=-1)    # [B, M, 1]
        mask_j = ops.expand_dims(entity_mask, axis=-2)    # [B, 1, M]
        connection_mask = mask_i * mask_j                 # [B, M, M] - both endpoints must be active

        # Create diagonal mask to remove self-connections
        diagonal_mask = 1.0 - ops.eye(self.max_entities, dtype=graph.dtype)

        # Apply both masks to final graph
        final_graph = graph * connection_mask * diagonal_mask

        logger.debug("Applied entity and diagonal masks to graph")
        return final_graph

    def call(
            self,
            embeddings: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]:
        """
        Forward pass through the complete entity-graph refinement pipeline.

        Executes the full processing pipeline:
        1. Optional positional encoding of input embeddings
        2. Entity extraction via attention over learnable entity library
        3. Dense graph initialization with random weights
        4. Iterative graph refinement using entity and context information
        5. Learned sparsification to focus on important relationships
        6. Final masking to remove inactive entities and self-connections

        Args:
            embeddings: Input embedding sequences [batch_size, seq_length, embed_dim]
            training: Whether layer is in training mode (affects dropout and logging)

        Returns:
            Tuple containing:
            - entities: Extracted entity representations [batch_size, max_entities, entity_dim]
            - graph: Final sparse relationship matrix [batch_size, max_entities, max_entities]
            - entity_mask: Binary mask for active entities [batch_size, max_entities]
        """
        batch_size = ops.shape(embeddings)[0]

        logger.debug(f"Processing batch of size {batch_size} through EntityGraphRefinement")

        # Keep reference to original embeddings for context computation
        input_embeddings = embeddings

        # Apply optional positional encoding to input embeddings
        if self.use_positional_encoding and self.positional_encoder is not None:
            seq_length = ops.shape(embeddings)[1]
            # Create position indices, clipped to maximum supported length
            positions = ops.arange(0, seq_length)
            clipped_positions = ops.clip(positions, 0, self.max_sequence_length - 1)
            clipped_positions = ops.expand_dims(clipped_positions, 0)
            clipped_positions = ops.repeat(clipped_positions, batch_size, axis=0)

            # Add positional embeddings to input
            embeddings = embeddings + self.positional_encoder(clipped_positions)
            logger.debug(f"Applied positional encoding to sequence length {seq_length}")

        # Extract entities using attention mechanism
        entities, entity_mask = self._extract_entities(embeddings, training)

        # Initialize dense relationship graph
        graph = self._initialize_dense_graph(batch_size)

        # Create global context vector from original embeddings
        # Simple mean pooling provides effective global representation
        context_vector = ops.mean(input_embeddings, axis=1, keepdims=False)  # [B, D]

        logger.debug(f"Starting {self.num_refinement_steps} graph refinement steps")

        # Iteratively refine the relationship graph
        for step in range(self.num_refinement_steps):
            graph = self._refine_graph(graph, entities, context_vector, training)
            if training and step % 2 == 0:  # Log every other step to avoid spam
                logger.debug(f"Completed refinement step {step + 1}/{self.num_refinement_steps}")

        # Apply learned sparsification to focus on important relationships
        graph = self._apply_sparsification(graph, entities, training)

        # Apply final masks (entity activity and self-connection removal)
        final_graph = self._apply_masks(graph, entity_mask)

        # Add regularization losses during training
        if training and self.regularization_weight > 0:
            # L1 sparsity regularization to encourage sparse graphs
            sparsity_loss = self.regularization_weight * ops.mean(ops.abs(final_graph))
            self.add_loss(sparsity_loss)

            # Activity regularization to maintain target activity level
            avg_activity = ops.mean(ops.abs(final_graph), axis=[1, 2])
            activity_loss = self.regularization_weight * ops.mean(
                ops.square(avg_activity - self.activity_regularization_target)
            )
            self.add_loss(activity_loss)

            # FIX: Removed unsupported formatting for symbolic tensors
            logger.debug(f"Added regularization losses: sparsity (tensor)={sparsity_loss}, "
                        f"activity (tensor)={activity_loss}")

        logger.debug("EntityGraphRefinement forward pass completed")
        return entities, final_graph, entity_mask

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Tuple[Optional[int], ...], ...]:
        """
        Compute output tensor shapes for the three returned tensors.

        Args:
            input_shape: Shape of input tensor (batch_size, sequence_length, embedding_dim)

        Returns:
            Tuple of three shapes for (entities, graph, entity_mask)
        """
        batch_size = input_shape[0]
        entities_shape = (batch_size, self.max_entities, self.entity_dim)
        graph_shape = (batch_size, self.max_entities, self.max_entities)
        mask_shape = (batch_size, self.max_entities)
        return entities_shape, graph_shape, mask_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        Returns all hyperparameters needed to reconstruct the layer.

        Returns:
            Dictionary containing all layer configuration parameters
        """
        config = super().get_config()
        config.update({
            'max_entities': self.max_entities,
            'entity_dim': self.entity_dim,
            'num_refinement_steps': self.num_refinement_steps,
            'initial_density': self.initial_density,
            'attention_heads': self.attention_heads,
            'dropout_rate': self.dropout_rate,
            'refinement_activation': self.refinement_activation,
            'entity_activity_threshold': self.entity_activity_threshold,
            'use_positional_encoding': self.use_positional_encoding,
            'max_sequence_length': self.max_sequence_length,
            'regularization_weight': self.regularization_weight,
            'activity_regularization_target': self.activity_regularization_target,
        })
        return config


# ---------------------------------------------------------------------
# Utility Functions for Analysis and Debugging
# ---------------------------------------------------------------------

def get_graph_statistics(
        entities: np.ndarray,
        graph: np.ndarray,
        entity_mask: np.ndarray,
        threshold: float = 0.1
) -> Dict[str, Any]:
    """
    Compute comprehensive statistics for an extracted entity graph.

    Analyzes the graph structure to provide insights into learned relationships,
    sparsity patterns, and entity activity levels. Useful for debugging and
    understanding model behavior.

    Args:
        entities: Entity representations [max_entities, entity_dim]
        graph: Relationship matrix [max_entities, max_entities]
        entity_mask: Binary activity mask [max_entities]
        threshold: Minimum edge weight magnitude to consider as "strong" edge

    Returns:
        Dictionary containing detailed graph statistics:
        - active_entities: Number of entities above activity threshold
        - total_edges: Number of strong edges (above threshold)
        - sparsity: Fraction of possible edges that are weak/absent
        - avg_edge_weight: Average magnitude of strong edge weights
        - max_edge_weight: Maximum edge weight magnitude
        - positive_edges: Number of positive strong edges
        - negative_edges: Number of negative strong edges
    """
    # Find indices of active entities
    active_indices = np.where(entity_mask > 0.5)[0]
    active_entities = len(active_indices)

    if active_entities == 0:
        logger.warning("No active entities found in graph - check entity_activity_threshold")
        return {'active_entities': 0, 'warning': 'No active entities found'}

    # Extract subgraph for active entities only
    active_graph = graph[np.ix_(active_indices, active_indices)]

    # Identify strong edges above threshold
    strong_edges = np.abs(active_graph) > threshold
    num_edges = np.sum(strong_edges)

    # Calculate sparsity metrics
    total_possible_edges = active_entities * (active_entities - 1)  # Exclude diagonal
    sparsity = 1 - (num_edges / total_possible_edges) if total_possible_edges > 0 else 1.0

    # Compile comprehensive statistics
    stats = {
        'active_entities': active_entities,
        'active_indices': active_indices.tolist(),
        'total_edges': num_edges,
        'sparsity': sparsity,
        'avg_edge_weight': np.mean(np.abs(active_graph[strong_edges])) if num_edges > 0 else 0.0,
        'max_edge_weight': np.max(np.abs(active_graph)) if active_entities > 0 else 0.0,
        'positive_edges': np.sum(active_graph[strong_edges] > 0) if num_edges > 0 else 0,
        'negative_edges': np.sum(active_graph[strong_edges] < 0) if num_edges > 0 else 0,
    }

    logger.info(f"Graph statistics: {active_entities} active entities, "
               f"{num_edges} strong edges, {sparsity:.3f} sparsity")

    return stats


def extract_hierarchies(
        graph: np.ndarray,
        entity_mask: np.ndarray,
        threshold: float = 0.3
) -> List[Tuple[int, int, float]]:
    """
    Extract hierarchical relationships from the learned entity graph.

    Identifies directed relationships where entity A strongly influences entity B
    but not vice versa, suggesting a hierarchical or causal relationship.

    Args:
        graph: Relationship matrix [max_entities, max_entities]
        entity_mask: Binary activity mask [max_entities]
        threshold: Minimum edge weight to consider as significant relationship

    Returns:
        List of (parent_idx, child_idx, strength) tuples representing hierarchies,
        sorted by relationship strength in descending order
    """
    # Get active entity indices
    active_indices = np.where(entity_mask > 0.5)[0]

    if len(active_indices) <= 1:
        logger.info("Insufficient active entities for hierarchy extraction")
        return []

    # Extract subgraph for active entities
    active_graph = graph[np.ix_(active_indices, active_indices)]
    hierarchies = []

    # Search for asymmetric relationships indicating hierarchy
    for i_idx in range(len(active_indices)):
        for j_idx in range(len(active_indices)):
            if i_idx == j_idx:
                continue  # Skip self-connections

            forward_strength = active_graph[i_idx, j_idx]

            # Check if forward relationship is strong
            if forward_strength > threshold:
                reverse_strength = active_graph[j_idx, i_idx]

                # Hierarchy criterion: strong forward, weak reverse
                if reverse_strength < forward_strength * 0.5:
                    # Map back to original entity indices
                    original_parent_idx = active_indices[i_idx]
                    original_child_idx = active_indices[j_idx]
                    hierarchies.append((original_parent_idx, original_child_idx, forward_strength))

    # Sort by relationship strength (strongest hierarchies first)
    hierarchies.sort(key=lambda x: x[2], reverse=True)

    logger.info(f"Extracted {len(hierarchies)} hierarchical relationships")

    return hierarchies