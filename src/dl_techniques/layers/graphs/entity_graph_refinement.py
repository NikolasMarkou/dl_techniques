"""
Entity-Graph Refinement Component for LLMs

This module provides a comprehensive, production-ready component for learning hierarchical
relationships in embedding space. It processes sequences of n-dimensional embeddings to
extract entities, build directional relationship graphs, and refine them through multiple
iterative steps.

## Overview

The EntityGraphRefinement layer is designed for advanced relationship modeling in deep
learning systems. It's particularly useful for:

- **Language Models**: Learning semantic relationships between concepts
- **Vision Models**: Understanding spatial and compositional relationships
- **Multimodal Systems**: Cross-modal entity associations
- **Knowledge Graphs**: Dynamic relationship discovery

## Core Architecture

1. **Entity Extraction**: Dynamic identification of entities from input embeddings via attention.
2. **Graph Initialization**: Dense directional relationship matrix creation.
3. **Iterative Refinement**: Progressive learning of meaningful relationships conditioned on
   entities, the previous graph state, and the global input context.
4. **Sparsification**: Learned pruning to focus on important connections.

## Key Features

- **Modality-Agnostic**: Works with any n-dimensional embeddings (text, vision, audio, etc.)
- **Directional Relationships**: Asymmetric graph enables hierarchy learning
- **Dynamic Entity Library**: Adaptive entity representations
- **Configurable Architecture**: Highly flexible API for different use cases
- **Production Ready**: Comprehensive validation, testing, and serialization support

## Mathematical Foundation

- **Entity Extraction**: α = softmax(Q·K^T), entities = α·V
- **Graph Initialization**: G₀ ~ U(-ρ, ρ) where ρ controls initial density
- **Refinement**: G_{t+1} = tanh(MLP([E_i, E_j, G_t[i,j], C])) where C is a context vector
  derived from the input embeddings.
- **Sparsification**: G_final = G * σ(MLP([E_i, E_j])) with learned gating

This enables learning of hierarchical structures, causal relationships, and compositional
patterns directly from data.
"""

import keras
import numpy as np
from keras import ops, layers
from typing import Optional, Tuple, List, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

# from dl_techniques.utils.logger import logger # Assuming this is a custom logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class EntityGraphRefinement(keras.layers.Layer):
    """
    Entity-Graph Refinement Component for learning hierarchical relationships in embedding space.

    This layer processes sequences of n-dimensional embeddings to:
    1. Extract entities using an attention mechanism against a learnable entity library.
    2. Build a directional relationship graph between entities.
    3. Iteratively refine the graph, using the entities, the previous graph state, and the
       global context of the input embeddings at each step.
    4. Sparsify the graph to highlight the most salient relationships.

    **Architecture**:
    ```
    Input Embeddings → Entity Extraction → Entity Library
         │                         ↓
         └─→ Global Context  → Dense Graph ← Graph Initialization
                                   ↓
            Refinement Step 1 → Refinement Step 2 → ... → Refinement Step N
                                   ↓
                   Sparsification → Sparse Directional Graph
    ```

    Args:
        max_entities: Integer, maximum number of entities to maintain in library.
        entity_dim: Integer, dimensionality of entity representations.
        num_refinement_steps: Integer, number of iterative refinement steps. Defaults to 3.
        initial_density: Float between 0 and 1, controls magnitude of initial graph weights. Defaults to 0.8.
        attention_heads: Integer, number of attention heads for entity extraction. Defaults to 8.
        dropout_rate: Float between 0 and 1, dropout rate for regularization. Defaults to 0.1.
        refinement_activation: String, activation function for refinement MLP. Defaults to 'gelu'.
        entity_activity_threshold: Float, threshold for determining active entities from attention scores. Defaults to 0.1.
        use_positional_encoding: Boolean, whether to add positional information. Defaults to True.
        max_sequence_length: Integer, maximum sequence length for positional encoding. Defaults to 1000.
        regularization_weight: Float, strength of graph regularization. Defaults to 0.01.
        activity_regularization_target: Float, target activity level for regularization. Defaults to 0.1.
        **kwargs: Additional arguments for Layer base class.

    Input shape:
        3D tensor with shape: `(batch_size, sequence_length, embedding_dim)`.

    Output shape:
        Tuple of:
        - entities: `(batch_size, max_entities, entity_dim)` - Entity representations
        - graph: `(batch_size, max_entities, max_entities)` - Directional relationship matrix
        - entity_mask: `(batch_size, max_entities)` - Binary mask for active entities

    Example:
        ```python
        import keras
        seq_len = 128
        embedding_dim = 768

        # Basic usage with text embeddings
        embeddings = keras.Input(shape=(seq_len, embedding_dim))
        entity_graph_layer = EntityGraphRefinement(
            max_entities=50,
            entity_dim=embedding_dim,
            num_refinement_steps=4,
            attention_heads=12
        )
        entities, graph, mask = entity_graph_layer(embeddings)

        model = keras.Model(inputs=embeddings, outputs=[entities, graph, mask])
        model.summary()
        ```
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

        # --- REFINED: Simplified configuration by removing extraction_method ---
        self._validate_hyperparameters(locals())

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

        self.entity_library = None
        self.graph_refinement_mlp = None
        self.sparsification_gate = None
        self.entity_extractor = None
        self.positional_encoder = None

    def _validate_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Consolidated validation for constructor arguments."""
        # --- REFINED: Removed validation for extraction_method ---
        validation_rules = [
            (params['max_entities'] <= 0, f"max_entities must be positive, got {params['max_entities']}"),
            (params['entity_dim'] <= 0, f"entity_dim must be positive, got {params['entity_dim']}"),
            (params['num_refinement_steps'] <= 0, f"num_refinement_steps must be positive"),
            (not (0.0 <= params['initial_density'] <= 1.0), f"initial_density must be between 0 and 1"),
            (params['attention_heads'] <= 0, f"attention_heads must be positive"),
            (not (0.0 <= params['dropout_rate'] <= 1.0), f"dropout_rate must be between 0 and 1"),
            (not (0.0 <= params['entity_activity_threshold'] <= 1.0), f"entity_activity_threshold must be between 0 and 1"),
            (params['max_sequence_length'] <= 0, f"max_sequence_length must be positive"),
            (params['regularization_weight'] < 0, f"regularization_weight must be non-negative"),
            (params['activity_regularization_target'] < 0, f"activity_regularization_target must be non-negative")
        ]
        for condition, error_message in validation_rules:
            if condition:
                raise ValueError(error_message)

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer components and explicitly build all sublayers."""
        _batch_size, sequence_length, embedding_dim = input_shape
        if embedding_dim is None:
            raise ValueError("Embedding dimension of input must be specified.")

        self.entity_library = self.add_weight(
            name='entity_library',
            shape=(self.max_entities, self.entity_dim),
            initializer='glorot_normal',
            trainable=True
        )

        key_dim = max(self.entity_dim // self.attention_heads, 1)
        self.entity_extractor = layers.MultiHeadAttention(
            num_heads=self.attention_heads,
            key_dim=key_dim,
            value_dim=key_dim,
            dropout=self.dropout_rate,
            name='entity_extractor'
        )
        # --- FIX: Explicitly build the MultiHeadAttention sublayer ---
        self.entity_extractor.build(
            query_shape=(None, self.max_entities, self.entity_dim),
            value_shape=(None, sequence_length, embedding_dim)
        )

        if self.use_positional_encoding:
            self.positional_encoder = layers.Embedding(
                input_dim=self.max_sequence_length,
                output_dim=embedding_dim,
                name='positional_encoder'
            )
            # --- FIX: Explicitly build the Embedding sublayer ---
            self.positional_encoder.build(input_shape=(None, sequence_length))

        refinement_input_dim = 2 * self.entity_dim + 1 + embedding_dim
        self.graph_refinement_mlp = keras.Sequential([
            layers.Dense(self.entity_dim, activation=self.refinement_activation, name='refinement_expand'),
            layers.Dropout(self.dropout_rate, name='refinement_dropout'),
            layers.Dense(1, activation='tanh', name='refinement_contract')
        ], name='graph_refinement_mlp')
        # --- FIX: Explicitly build the Sequential sublayer ---
        self.graph_refinement_mlp.build(input_shape=(None, refinement_input_dim))

        self.sparsification_gate = keras.Sequential([
            layers.Dense(self.entity_dim, activation=self.refinement_activation, name='sparsification_hidden'),
            layers.Dropout(self.dropout_rate, name='sparsification_dropout'),
            layers.Dense(1, activation='sigmoid', name='sparsification_gate')
        ], name='sparsification_gate')
        # --- FIX: Explicitly build the Sequential sublayer ---
        self.sparsification_gate.build(input_shape=(None, 2 * self.entity_dim))

        # Finally, call the parent build method
        super().build(input_shape)

    def _get_edge_features(self, entities: keras.KerasTensor) -> keras.KerasTensor:
        """Creates features for each potential edge by combining entity pairs."""
        entities_i = ops.expand_dims(entities, axis=2)
        entities_j = ops.expand_dims(entities, axis=1)
        entities_i = ops.repeat(entities_i, self.max_entities, axis=2)
        entities_j = ops.repeat(entities_j, self.max_entities, axis=1)
        return ops.concatenate([entities_i, entities_j], axis=-1)

    def _extract_entities(
            self,
            embeddings: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Extract entities using attention mechanism."""
        batch_size = ops.shape(embeddings)[0]
        entity_queries = ops.expand_dims(self.entity_library, 0)
        entity_queries = ops.repeat(entity_queries, batch_size, axis=0)

        extracted_entities, attention_scores = self.entity_extractor(
            query=entity_queries,
            key=embeddings,
            value=embeddings,
            training=training,
            return_attention_scores=True
        )

        # An entity is "active" if it pays high attention to any part of the input sequence.
        entity_activity = ops.max(attention_scores, axis=[1, 3])  # Max over heads and sequence
        entity_mask = ops.cast(entity_activity > self.entity_activity_threshold, dtype='float32')

        return extracted_entities, entity_mask

    def _initialize_dense_graph(self, batch_size: int) -> keras.KerasTensor:
        """Initialize dense relationship graph."""
        graph_shape = (batch_size, self.max_entities, self.max_entities)
        return keras.random.uniform(
            graph_shape,
            minval=-self.initial_density,
            maxval=self.initial_density
        )

    # --- REFINED: The refinement step now accepts and uses the global input context ---
    def _refine_graph(
            self,
            graph: keras.KerasTensor,
            entities: keras.KerasTensor,
            context: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Perform one refinement step on the graph using shared MLP and global context."""
        batch_size = ops.shape(graph)[0]
        embedding_dim = ops.shape(context)[-1]

        # Get pairwise entity features
        edge_features = self._get_edge_features(entities)
        current_edges = ops.expand_dims(graph, axis=-1)

        # Prepare context vector for broadcasting
        # Reshape context from (batch, embed_dim) to (batch, 1, 1, embed_dim)
        context_reshaped = ops.reshape(context, (batch_size, 1, 1, embedding_dim))
        # Tile it to match the graph dimensions: (batch, max_entities, max_entities, embed_dim)
        context_tiled = ops.tile(context_reshaped, [1, self.max_entities, self.max_entities, 1])

        # Concatenate all features: entity pair, current edge weight, and global context
        full_edge_features = ops.concatenate(
            [edge_features, current_edges, context_tiled], axis=-1
        )

        flat_features = ops.reshape(
            full_edge_features,
            (batch_size * self.max_entities * self.max_entities, -1)
        )

        refined_edges_flat = self.graph_refinement_mlp(flat_features, training=training)

        return ops.reshape(
            refined_edges_flat,
            (batch_size, self.max_entities, self.max_entities)
        )

    def _apply_sparsification(
            self,
            graph: keras.KerasTensor,
            entities: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply learned sparsification to the graph."""
        batch_size = ops.shape(graph)[0]
        edge_features = self._get_edge_features(entities)
        edge_features_flat = ops.reshape(
            edge_features,
            (batch_size * self.max_entities * self.max_entities, -1)
        )
        gates_flat = self.sparsification_gate(edge_features_flat, training=training)
        gates = ops.reshape(gates_flat, (batch_size, self.max_entities, self.max_entities))
        return graph * gates

    def _apply_masks(
            self,
            graph: keras.KerasTensor,
            entity_mask: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Apply entity mask and remove self-connections."""
        mask_i = ops.expand_dims(entity_mask, axis=-1)
        mask_j = ops.expand_dims(entity_mask, axis=-2)
        connection_mask = mask_i * mask_j
        diagonal_mask = 1.0 - ops.eye(self.max_entities, dtype=graph.dtype)
        return graph * connection_mask * diagonal_mask

    def call(
            self,
            embeddings: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]:
        """Forward pass through the entity-graph refinement component."""
        batch_size = ops.shape(embeddings)[0]

        input_embeddings = embeddings # Keep a reference to original embeddings

        if self.use_positional_encoding and self.positional_encoder is not None:
            seq_length = ops.shape(embeddings)[1]
            positions = ops.arange(0, seq_length)
            clipped_positions = ops.clip(positions, 0, self.max_sequence_length - 1)
            clipped_positions = ops.expand_dims(clipped_positions, 0)
            clipped_positions = ops.repeat(clipped_positions, batch_size, axis=0)
            embeddings = embeddings + self.positional_encoder(clipped_positions)

        # --- REFINED: Simplified extraction call ---
        entities, entity_mask = self._extract_entities(embeddings, training)
        graph = self._initialize_dense_graph(batch_size)

        # --- REFINED: Create a global context vector from the original embeddings ---
        # This context is used in each refinement step. A simple mean pooling is effective.
        context_vector = ops.mean(input_embeddings, axis=1, keepdims=False)

        # --- REFINED: Pass the context vector into the refinement loop ---
        for _ in range(self.num_refinement_steps):
            graph = self._refine_graph(graph, entities, context_vector, training)

        graph = self._apply_sparsification(graph, entities, training)
        final_graph = self._apply_masks(graph, entity_mask)

        if training and self.regularization_weight > 0:
            sparsity_loss = self.regularization_weight * ops.mean(ops.abs(final_graph))
            self.add_loss(sparsity_loss)
            activity_loss = self.regularization_weight * ops.mean(
                ops.square(ops.mean(ops.abs(final_graph), axis=[1, 2]) - self.activity_regularization_target)
            )
            self.add_loss(activity_loss)

        return entities, final_graph, entity_mask

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Tuple[Optional[int], ...], ...]:
        batch_size = input_shape[0]
        entities_shape = (batch_size, self.max_entities, self.entity_dim)
        graph_shape = (batch_size, self.max_entities, self.max_entities)
        mask_shape = (batch_size, self.max_entities)
        return entities_shape, graph_shape, mask_shape

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        # --- REFINED: Removed extraction_method from config ---
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

# The utility functions below remain unchanged as they were already robust.

def get_graph_statistics(
        entities: np.ndarray,
        graph: np.ndarray,
        entity_mask: np.ndarray,
        threshold: float = 0.1
) -> Dict[str, Any]:
    # ... (No changes needed)
    active_indices = np.where(entity_mask > 0.5)[0]
    active_entities = len(active_indices)
    if active_entities == 0:
        return {'active_entities': 0, 'warning': 'No active entities found'}
    active_graph = graph[np.ix_(active_indices, active_indices)]
    strong_edges = np.abs(active_graph) > threshold
    num_edges = np.sum(strong_edges)
    total_possible_edges = active_entities * (active_entities - 1)
    sparsity = 1 - (num_edges / total_possible_edges) if total_possible_edges > 0 else 1.0
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
    return stats


def extract_hierarchies(
        graph: np.ndarray,
        entity_mask: np.ndarray,
        threshold: float = 0.3
) -> List[Tuple[int, int, float]]:
    # ... (No changes needed)
    active_indices = np.where(entity_mask > 0.5)[0]
    if len(active_indices) <= 1:
        return []
    active_graph = graph[np.ix_(active_indices, active_indices)]
    hierarchies = []
    for i_idx in range(len(active_indices)):
        for j_idx in range(len(active_indices)):
            if i_idx == j_idx:
                continue
            forward_strength = active_graph[i_idx, j_idx]
            if forward_strength > threshold:
                reverse_strength = active_graph[j_idx, i_idx]
                if reverse_strength < forward_strength * 0.5:
                    original_parent_idx = active_indices[i_idx]
                    original_child_idx = active_indices[j_idx]
                    hierarchies.append((original_parent_idx, original_child_idx, forward_strength))
    hierarchies.sort(key=lambda x: x[2], reverse=True)
    return hierarchies