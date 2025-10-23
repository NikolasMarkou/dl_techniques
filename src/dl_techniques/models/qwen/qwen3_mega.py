"""
Qwen3-MEGA: Memory-Enhanced Graph-Augmented Language Model
===========================================================

A Qwen3-based language model enhanced with:
1. MANN (Memory-Augmented Neural Network) for working memory/scratch pad
2. Graph Neural Network for entity and relationship modeling
3. Cross-attention mechanisms to integrate memory systems with transformer

This architecture enables:
- Explicit working memory for intermediate reasoning steps
- Structured knowledge representation via entity graphs
- Enhanced long-term dependencies and reasoning capabilities

Based on:
- Qwen3 architecture
- Neural Turing Machines (Graves et al., 2014)
- Graph Neural Networks (various architectures)
"""

import keras
import numpy as np
from typing import Optional, Union, Any, Dict, List, Tuple, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.memory.mann import MannLayer
from dl_techniques.layers.transformers import TransformerLayer
from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.layers.moe import MoEConfig, ExpertConfig, GatingConfig
from dl_techniques.layers.graphs.graph_neural_network import GraphNeuralNetworkLayer

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class MemoryIntegrationLayer(keras.layers.Layer):
    """
    Layer that integrates MANN memory and GNN entity graph with transformer hidden states.

    This layer enables the transformer to read from and write to external memory systems:
    - MANN provides episodic/working memory (scratch pad)
    - GNN provides structured entity/relationship knowledge

    **Architecture**:
    ```
    Transformer Hidden States
           │
           ├──────────────────┬──────────────────┐
           │                  │                  │
           ▼                  ▼                  ▼
    [Write to MANN]    [Query MANN]      [Query GNN]
           │                  │                  │
           └──────────────────┴──────────────────┘
                              │
                              ▼
                    Cross-Attention Fusion
                              │
                              ▼
                    Enhanced Hidden States
    ```

    Args:
        hidden_size: Integer, dimensionality of transformer hidden states.
        memory_dim: Integer, dimension of MANN memory slots.
        entity_dim: Integer, dimension of GNN entity embeddings.
        num_attention_heads: Integer, number of heads for cross-attention.
        dropout_rate: Float, dropout rate for regularization.
        use_memory_write: Boolean, whether to update MANN memory.
        use_gnn_update: Boolean, whether to update GNN entity graph.
        **kwargs: Additional arguments for Layer base class.
    """

    def __init__(
        self,
        hidden_size: int,
        memory_dim: int,
        entity_dim: int,
        num_attention_heads: int = 8,
        dropout_rate: float = 0.1,
        use_memory_write: bool = True,
        use_gnn_update: bool = True,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if memory_dim <= 0:
            raise ValueError(f"memory_dim must be positive, got {memory_dim}")
        if entity_dim <= 0:
            raise ValueError(f"entity_dim must be positive, got {entity_dim}")
        if num_attention_heads <= 0:
            raise ValueError(f"num_attention_heads must be positive, got {num_attention_heads}")

        # Store configuration
        self.hidden_size = hidden_size
        self.memory_dim = memory_dim
        self.entity_dim = entity_dim
        self.num_attention_heads = num_attention_heads
        self.dropout_rate = dropout_rate
        self.use_memory_write = use_memory_write
        self.use_gnn_update = use_gnn_update

        # CREATE sub-layers in __init__ (unbuilt)

        # Project transformer hidden states to memory query
        self.memory_query_proj = keras.layers.Dense(
            memory_dim, name="memory_query_proj"
        )

        # Project transformer hidden states to entity query
        self.entity_query_proj = keras.layers.Dense(
            entity_dim, name="entity_query_proj"
        )

        # Cross-attention to integrate memory information
        self.memory_cross_attn = keras.layers.MultiHeadAttention(
            num_heads=num_attention_heads,
            key_dim=hidden_size // num_attention_heads,
            dropout=dropout_rate,
            name="memory_cross_attention"
        )

        # Cross-attention to integrate entity information
        self.entity_cross_attn = keras.layers.MultiHeadAttention(
            num_heads=num_attention_heads,
            key_dim=hidden_size // num_attention_heads,
            dropout=dropout_rate,
            name="entity_cross_attention"
        )

        # Fusion layer to combine all information streams
        self.fusion_dense = keras.layers.Dense(
            hidden_size, activation='gelu', name="fusion_dense"
        )

        self.fusion_norm = keras.layers.LayerNormalization(
            name="fusion_norm"
        )

        self.fusion_dropout = keras.layers.Dropout(
            dropout_rate, name="fusion_dropout"
        )

    def build(self, input_shape: Tuple[Tuple[Optional[int], ...], ...]) -> None:
        """Build all sub-layers explicitly."""
        hidden_shape, memory_shape, entity_shape = input_shape

        # Build projection layers
        self.memory_query_proj.build(hidden_shape)
        self.entity_query_proj.build(hidden_shape)

        # Build cross-attention layers
        self.memory_cross_attn.build(hidden_shape, hidden_shape)
        self.entity_cross_attn.build(hidden_shape, hidden_shape)

        # Build fusion layers
        fusion_input_shape = (
            hidden_shape[0],
            hidden_shape[1],
            hidden_shape[2] * 3  # Concatenation of 3 streams
        )
        self.fusion_dense.build(fusion_input_shape)
        self.fusion_norm.build(hidden_shape)
        self.fusion_dropout.build(hidden_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor],
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Integrate memory and entity information with hidden states.

        Args:
            inputs: Tuple of (hidden_states, memory_vectors, entity_embeddings)
            training: Boolean, whether in training mode

        Returns:
            Enhanced hidden states with memory integration
        """
        hidden_states, memory_vectors, entity_embeddings = inputs

        # Query memory using hidden states
        memory_attended = self.memory_cross_attn(
            query=hidden_states,
            value=memory_vectors,
            key=memory_vectors,
            training=training
        )

        # Query entity graph using hidden states
        entity_attended = self.entity_cross_attn(
            query=hidden_states,
            value=entity_embeddings,
            key=entity_embeddings,
            training=training
        )

        # Concatenate all information streams
        fused = keras.ops.concatenate(
            [hidden_states, memory_attended, entity_attended], axis=-1
        )

        # Project back to hidden size
        fused = self.fusion_dense(fused)
        fused = self.fusion_dropout(fused, training=training)

        # Residual connection and normalization
        output = self.fusion_norm(hidden_states + fused)

        return output

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'hidden_size': self.hidden_size,
            'memory_dim': self.memory_dim,
            'entity_dim': self.entity_dim,
            'num_attention_heads': self.num_attention_heads,
            'dropout_rate': self.dropout_rate,
            'use_memory_write': self.use_memory_write,
            'use_gnn_update': self.use_gnn_update,
        })
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class Qwen3MEGA(keras.Model):
    """
    Qwen3 model enhanced with Memory-Augmented Neural Networks and Graph Neural Networks.

    This model extends Qwen3 with two external memory systems:
    1. **MANN**: Provides differentiable working memory for intermediate reasoning
    2. **GNN**: Maintains a graph of entities and their relationships

    The transformer can read from and write to these memory systems, enabling:
    - Explicit reasoning with scratch pad memory
    - Structured knowledge representation
    - Enhanced long-term dependency modeling

    **Architecture Overview**:
    ```
    Input(input_ids)
           │
           ▼
    Token Embeddings
           │
           ├──────────────────┬──────────────────┐
           │                  │                  │
           ▼                  ▼                  ▼
    Transformer Blocks    MANN Layer      GNN Layer
       (with Memory)     (Working Memory)  (Entity Graph)
           │                  │                  │
           └──────────────────┴──────────────────┘
                              │
                              ▼
                    Memory Integration
                              │
                              ▼
                        Final Norm
                              │
                              ▼
                          LM Head
    ```

    Args:
        vocab_size: Integer, size of the vocabulary.
        hidden_size: Integer, dimensionality of encoder layers.
        num_layers: Integer, number of transformer blocks.
        num_attention_heads: Integer, number of attention heads.
        num_key_value_heads: Integer, number of key-value heads for GQA.
        max_seq_len: Integer, maximum sequence length.

        # Memory configuration
        memory_locations: Integer, number of MANN memory slots.
        memory_dim: Integer, dimension of each MANN memory slot.
        controller_units: Integer, MANN controller hidden size.
        num_read_heads: Integer, number of MANN read heads.
        num_write_heads: Integer, number of MANN write heads.

        # Entity graph configuration
        num_entities: Integer, maximum number of entities in graph.
        entity_dim: Integer, dimension of entity embeddings.
        gnn_num_layers: Integer, number of GNN layers.
        gnn_message_passing: String, GNN message passing type.

        # Integration configuration
        memory_integration_layers: List of layer indices where memory integration occurs.

        # Standard transformer configuration
        moe_layers: List of integers, layer indices that use MoE.
        num_experts: Integer, total number of experts in MoE layers.
        num_experts_per_tok: Integer, number of experts activated per token.
        moe_intermediate_size: Integer, individual expert intermediate size.
        rope_theta: Float, RoPE theta parameter.
        norm_eps: Float, epsilon for normalization layers.
        dropout_rate: Float, dropout rate for regularization.
        initializer_range: Float, standard deviation for weight initialization.
        normalization_type: String, type of normalization layer.
        ffn_type: String, type of feed-forward network.
        use_stochastic_depth: Boolean, whether to enable stochastic depth.
        stochastic_depth_rate: Float, drop path rate for stochastic depth.
        **kwargs: Additional keyword arguments for keras.Model.
    """

    def __init__(
        self,
        # Core Qwen3 parameters
        vocab_size: int = 151936,
        hidden_size: int = 2048,
        num_layers: int = 12,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 4,
        max_seq_len: int = 8192,

        # MANN parameters
        memory_locations: int = 128,
        memory_dim: int = 512,
        controller_units: int = 512,
        num_read_heads: int = 2,
        num_write_heads: int = 2,

        # GNN parameters
        num_entities: int = 256,
        entity_dim: int = 512,
        gnn_num_layers: int = 3,
        gnn_message_passing: Literal['gcn', 'graphsage', 'gat', 'gin'] = 'gat',

        # Integration parameters
        memory_integration_layers: Optional[List[int]] = None,

        # Standard Qwen3 parameters
        moe_layers: Optional[List[int]] = None,
        num_experts: int = 64,
        num_experts_per_tok: int = 8,
        moe_intermediate_size: int = 1408,
        rope_theta: float = 10_000_000.0,
        norm_eps: float = 1e-6,
        dropout_rate: float = 0.0,
        initializer_range: float = 0.02,
        normalization_type: str = "rms_norm",
        ffn_type: str = "swiglu",
        use_stochastic_depth: bool = False,
        stochastic_depth_rate: float = 0.1,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Set defaults
        if moe_layers is None:
            moe_layers = []
        if memory_integration_layers is None:
            # Default: integrate memory at 1/4, 1/2, 3/4 of layers
            memory_integration_layers = [
                num_layers // 4,
                num_layers // 2,
                3 * num_layers // 4
            ]

        # Validate configuration
        self._validate_config(
            vocab_size, hidden_size, num_layers, num_attention_heads,
            num_key_value_heads, memory_locations, memory_dim,
            num_entities, entity_dim, memory_integration_layers
        )

        # Store ALL configuration
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_seq_len = max_seq_len

        self.memory_locations = memory_locations
        self.memory_dim = memory_dim
        self.controller_units = controller_units
        self.num_read_heads = num_read_heads
        self.num_write_heads = num_write_heads

        self.num_entities = num_entities
        self.entity_dim = entity_dim
        self.gnn_num_layers = gnn_num_layers
        self.gnn_message_passing = gnn_message_passing

        self.memory_integration_layers = sorted(memory_integration_layers)

        self.moe_layers = moe_layers
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.rope_theta = rope_theta
        self.norm_eps = norm_eps
        self.dropout_rate = dropout_rate
        self.initializer_range = initializer_range
        self.normalization_type = normalization_type
        self.ffn_type = ffn_type
        self.use_stochastic_depth = use_stochastic_depth
        self.stochastic_depth_rate = stochastic_depth_rate

        # Build architecture
        self._build_architecture()

        # Log model creation
        logger.info(
            f"Created Qwen3-MEGA model: {self.num_layers} layers, "
            f"hidden_size={self.hidden_size}"
        )
        logger.info(
            f"Memory: {self.memory_locations} slots × {self.memory_dim}D, "
            f"{self.num_read_heads}R/{self.num_write_heads}W heads"
        )
        logger.info(
            f"Entity Graph: {self.num_entities} entities × {self.entity_dim}D, "
            f"{self.gnn_num_layers} GNN layers ({self.gnn_message_passing})"
        )
        logger.info(
            f"Memory integration at layers: {self.memory_integration_layers}"
        )

    def _validate_config(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        memory_locations: int,
        memory_dim: int,
        num_entities: int,
        entity_dim: int,
        memory_integration_layers: List[int]
    ) -> None:
        """Validate model configuration parameters."""
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if memory_locations <= 0:
            raise ValueError(f"memory_locations must be positive, got {memory_locations}")
        if memory_dim <= 0:
            raise ValueError(f"memory_dim must be positive, got {memory_dim}")
        if num_entities <= 0:
            raise ValueError(f"num_entities must be positive, got {num_entities}")
        if entity_dim <= 0:
            raise ValueError(f"entity_dim must be positive, got {entity_dim}")
        if any(idx < 0 or idx >= num_layers for idx in memory_integration_layers):
            raise ValueError(
                f"All memory_integration_layers must be between 0 and {num_layers-1}"
            )

    def _build_architecture(self) -> None:
        """Build all model components."""

        # Token embedding
        self.embeddings = keras.layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.hidden_size,
            embeddings_initializer=keras.initializers.TruncatedNormal(
                stddev=self.initializer_range
            ),
            name="token_embedding"
        )

        # MANN layer for working memory
        # Project transformer hidden states to MANN input dimension
        self.mann_input_proj = keras.layers.Dense(
            self.controller_units, name="mann_input_projection"
        )

        self.mann = MannLayer(
            memory_locations=self.memory_locations,
            memory_dim=self.memory_dim,
            controller_units=self.controller_units,
            num_read_heads=self.num_read_heads,
            num_write_heads=self.num_write_heads,
            controller_type='lstm',
            name="mann_memory"
        )

        # Project MANN output back to memory dimension for integration
        # MANN output shape: (batch, seq, controller_units + num_read_heads * memory_dim)
        self.mann_output_dim = self.controller_units + self.num_read_heads * self.memory_dim
        self.mann_output_proj = keras.layers.Dense(
            self.memory_dim, name="mann_output_projection"
        )

        # Entity embedding and GNN
        self.entity_embeddings = self.add_weight(
            name="entity_embeddings",
            shape=(self.num_entities, self.entity_dim),
            initializer=keras.initializers.TruncatedNormal(
                stddev=self.initializer_range
            ),
            trainable=True
        )

        # GNN for entity relationships
        self.gnn = GraphNeuralNetworkLayer(
            concept_dim=self.entity_dim,
            num_layers=self.gnn_num_layers,
            message_passing=self.gnn_message_passing,
            aggregation='attention',
            normalization='layer',
            dropout_rate=self.dropout_rate,
            use_residual=True,
            name="entity_graph"
        )

        # Project GNN output for integration
        self.gnn_output_proj = keras.layers.Dense(
            self.entity_dim, name="gnn_output_projection"
        )

        # Create MoE configuration if needed
        moe_config = None
        if self.moe_layers:
            moe_config = MoEConfig(
                num_experts=self.num_experts,
                expert_config=ExpertConfig(
                    ffn_config={
                        "type": self.ffn_type,
                        "output_dim": self.hidden_size,
                        "ffn_expansion_factor": max(
                            1, self.moe_intermediate_size // self.hidden_size
                        )
                    }
                ),
                gating_config=GatingConfig(
                    top_k=self.num_experts_per_tok,
                    gating_type="linear"
                )
            )

        # Stochastic depth schedule
        if self.stochastic_depth_rate > 0:
            dpr = [
                x for x in np.linspace(
                    0.0, self.stochastic_depth_rate, self.num_layers
                )
            ]
        else:
            dpr = [0.0 for _ in range(self.num_layers)]

        # Create transformer blocks
        self.blocks = []
        for i in range(self.num_layers):
            is_moe_layer = i in self.moe_layers

            attention_args = {
                'dim': self.hidden_size,
                'num_heads': self.num_attention_heads,
                'num_kv_heads': self.num_key_value_heads,
                'max_seq_len': self.max_seq_len,
                'dropout_rate': self.dropout_rate,
                'rope_theta': self.rope_theta
            }

            ffn_args = {
                'output_dim': self.hidden_size,
                'ffn_expansion_factor': 4,
                'dropout_rate': self.dropout_rate,
                'use_bias': False
            }

            block = TransformerLayer(
                hidden_size=self.hidden_size,
                num_heads=self.num_attention_heads,
                intermediate_size=self.hidden_size * 4,
                attention_type='group_query',
                attention_args=attention_args,
                normalization_type=self.normalization_type,
                normalization_position='pre',
                moe_config=moe_config if is_moe_layer else None,
                ffn_type=self.ffn_type,
                ffn_args=ffn_args,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.dropout_rate,
                use_stochastic_depth=self.use_stochastic_depth,
                stochastic_depth_rate=dpr[i],
                use_bias=False,
                name=f"transformer_block_{i}"
            )
            self.blocks.append(block)

        # Memory integration layers
        self.memory_integration = {}
        for layer_idx in self.memory_integration_layers:
            self.memory_integration[layer_idx] = MemoryIntegrationLayer(
                hidden_size=self.hidden_size,
                memory_dim=self.memory_dim,
                entity_dim=self.entity_dim,
                num_attention_heads=8,
                dropout_rate=self.dropout_rate,
                name=f"memory_integration_{layer_idx}"
            )

        # Final normalization
        self.final_norm = create_normalization_layer(
            self.normalization_type,
            epsilon=self.norm_eps,
            name='final_norm'
        )

        # Language modeling head
        self.lm_head = keras.layers.Dense(
            units=self.vocab_size,
            use_bias=False,
            kernel_initializer=keras.initializers.TruncatedNormal(
                stddev=self.initializer_range
            ),
            name='lm_head'
        )

    def call(
        self,
        inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]],
        attention_mask: Optional[keras.KerasTensor] = None,
        entity_adjacency: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
        return_dict: bool = False
    ) -> Union[keras.KerasTensor, Dict[str, Any]]:
        """
        Forward pass of Qwen3-MEGA model.

        Args:
            inputs: Input token IDs or dictionary containing inputs.
            attention_mask: Mask to avoid attention on padding tokens.
            entity_adjacency: Adjacency matrix for entity graph.
                Shape: (batch_size, num_entities, num_entities).
                If None, uses uniform adjacency (all entities connected).
            training: Boolean, whether in training mode.
            return_dict: Boolean, whether to return outputs as dictionary.

        Returns:
            Model outputs (logits or dictionary with additional info).
        """
        # Parse inputs
        if isinstance(inputs, dict):
            input_ids = inputs.get("input_ids")
            if input_ids is None:
                raise ValueError("Dictionary input must contain 'input_ids' key")
            attention_mask = inputs.get("attention_mask", attention_mask)
            entity_adjacency = inputs.get("entity_adjacency", entity_adjacency)
        else:
            input_ids = inputs

        batch_size = keras.ops.shape(input_ids)[0]
        seq_len = keras.ops.shape(input_ids)[1]

        # Token embeddings
        hidden_states = self.embeddings(input_ids)

        # Process through MANN for working memory
        # MANN expects (batch, seq, features)
        mann_input = self.mann_input_proj(hidden_states)
        mann_output = self.mann(mann_input, training=training)
        memory_vectors = self.mann_output_proj(mann_output)

        # Process entity graph through GNN
        # Expand entity embeddings for batch
        entity_features = keras.ops.tile(
            keras.ops.expand_dims(self.entity_embeddings, 0),
            [batch_size, 1, 1]
        )

        # Create or use provided adjacency matrix
        if entity_adjacency is None:
            # Default: uniform connectivity (all entities connected)
            entity_adjacency = keras.ops.ones(
                (batch_size, self.num_entities, self.num_entities)
            ) / self.num_entities

        # Process through GNN
        entity_output = self.gnn(
            (entity_features, entity_adjacency),
            training=training
        )
        entity_embeddings = self.gnn_output_proj(entity_output)

        # Pass through transformer blocks with memory integration
        for i, block in enumerate(self.blocks):
            # Standard transformer processing
            hidden_states = block(
                hidden_states,
                attention_mask=attention_mask,
                training=training
            )

            # Integrate memory at designated layers
            if i in self.memory_integration:
                # Integrate MANN and GNN information
                hidden_states = self.memory_integration[i](
                    (hidden_states, memory_vectors, entity_embeddings),
                    training=training
                )

        # Final normalization
        hidden_states = self.final_norm(hidden_states)

        # Language modeling head
        logits = self.lm_head(hidden_states)

        # Return in requested format
        if return_dict:
            return {
                "logits": logits,
                "memory_vectors": memory_vectors,
                "entity_embeddings": entity_embeddings
            }
        else:
            return logits

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'num_attention_heads': self.num_attention_heads,
            'num_key_value_heads': self.num_key_value_heads,
            'max_seq_len': self.max_seq_len,
            'memory_locations': self.memory_locations,
            'memory_dim': self.memory_dim,
            'controller_units': self.controller_units,
            'num_read_heads': self.num_read_heads,
            'num_write_heads': self.num_write_heads,
            'num_entities': self.num_entities,
            'entity_dim': self.entity_dim,
            'gnn_num_layers': self.gnn_num_layers,
            'gnn_message_passing': self.gnn_message_passing,
            'memory_integration_layers': self.memory_integration_layers,
            'moe_layers': self.moe_layers,
            'num_experts': self.num_experts,
            'num_experts_per_tok': self.num_experts_per_tok,
            'moe_intermediate_size': self.moe_intermediate_size,
            'rope_theta': self.rope_theta,
            'norm_eps': self.norm_eps,
            'dropout_rate': self.dropout_rate,
            'initializer_range': self.initializer_range,
            'normalization_type': self.normalization_type,
            'ffn_type': self.ffn_type,
            'use_stochastic_depth': self.use_stochastic_depth,
            'stochastic_depth_rate': self.stochastic_depth_rate,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Qwen3MEGA":
        """Create model from configuration."""
        return cls(**config)


# ---------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------

def create_qwen3_mega(
    variant: Literal["tiny", "small", "medium"] = "small",
    memory_size: Literal["small", "medium", "large"] = "medium",
    entity_graph_size: Literal["small", "medium", "large"] = "medium",
    **kwargs: Any
) -> Qwen3MEGA:
    """
    Factory function to create Qwen3-MEGA models with preset configurations.

    Args:
        variant: Model size variant ('tiny', 'small', 'medium').
        memory_size: MANN memory configuration size.
        entity_graph_size: Entity graph configuration size.
        **kwargs: Additional arguments to override defaults.

    Returns:
        Configured Qwen3MEGA model.

    Example:
        ```python
        # Create small model with medium memory
        model = create_qwen3_mega("small", memory_size="medium")

        # Create tiny model with custom parameters
        model = create_qwen3_mega(
            "tiny",
            memory_size="small",
            dropout_rate=0.2
        )
        ```
    """
    # Base model configurations
    model_configs = {
        "tiny": {
            "vocab_size": 32000,
            "hidden_size": 512,
            "num_layers": 6,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "max_seq_len": 2048,
            "moe_layers": [],
        },
        "small": {
            "vocab_size": 32000,
            "hidden_size": 768,
            "num_layers": 12,
            "num_attention_heads": 12,
            "num_key_value_heads": 4,
            "max_seq_len": 4096,
            "moe_layers": [3, 6, 9],
            "num_experts": 8,
            "num_experts_per_tok": 2,
        },
        "medium": {
            "vocab_size": 100000,
            "hidden_size": 1024,
            "num_layers": 24,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "max_seq_len": 16384,
            "moe_layers": list(range(6, 24, 4)),
            "num_experts": 16,
            "num_experts_per_tok": 4,
        }
    }

    # Memory configurations
    memory_configs = {
        "small": {
            "memory_locations": 64,
            "memory_dim": 256,
            "controller_units": 256,
            "num_read_heads": 1,
            "num_write_heads": 1,
        },
        "medium": {
            "memory_locations": 128,
            "memory_dim": 512,
            "controller_units": 512,
            "num_read_heads": 2,
            "num_write_heads": 2,
        },
        "large": {
            "memory_locations": 256,
            "memory_dim": 768,
            "controller_units": 768,
            "num_read_heads": 4,
            "num_write_heads": 4,
        }
    }

    # Entity graph configurations
    entity_configs = {
        "small": {
            "num_entities": 128,
            "entity_dim": 256,
            "gnn_num_layers": 2,
        },
        "medium": {
            "num_entities": 256,
            "entity_dim": 512,
            "gnn_num_layers": 3,
        },
        "large": {
            "num_entities": 512,
            "entity_dim": 768,
            "gnn_num_layers": 4,
        }
    }

    # Validate inputs
    if variant not in model_configs:
        raise ValueError(
            f"Unknown variant '{variant}'. Choose from: {list(model_configs.keys())}"
        )
    if memory_size not in memory_configs:
        raise ValueError(
            f"Unknown memory_size '{memory_size}'. Choose from: {list(memory_configs.keys())}"
        )
    if entity_graph_size not in entity_configs:
        raise ValueError(
            f"Unknown entity_graph_size '{entity_graph_size}'. "
            f"Choose from: {list(entity_configs.keys())}"
        )

    # Combine configurations
    config = {}
    config.update(model_configs[variant])
    config.update(memory_configs[memory_size])
    config.update(entity_configs[entity_graph_size])
    config.update(kwargs)

    logger.info(f"Creating Qwen3-MEGA-{variant.upper()} with:")
    logger.info(f"  - Memory: {memory_size} ({config['memory_locations']} slots)")
    logger.info(f"  - Entity Graph: {entity_graph_size} ({config['num_entities']} entities)")

    return Qwen3MEGA(**config)

# ---------------------------------------------------------------------