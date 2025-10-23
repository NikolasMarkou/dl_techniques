"""
Qwen3-SOM: Language Model with Self-Organizing Map Memory Integration
=====================================================================

This implementation combines the Qwen3 transformer architecture with SoftSOMLayer
to create a language model with structured, topologically-organized memory mechanisms.
The SOM layers act as differentiable memory banks that organize learned representations
into spatially coherent maps, enabling:

- Structured memory consolidation between transformer blocks
- Topologically-organized feature spaces
- Differentiable clustering for improved generalization
- Memory-augmented attention mechanisms

Architecture Overview:
---------------------
```
Input Embeddings
      ↓
TransformerBlock₁
      ↓
[SOM Memory Layer] ← Optional memory consolidation
      ↓
TransformerBlock₂
      ↓
      ...
      ↓
TransformerBlockₙ
      ↓
[SOM Memory Layer] ← Optional final memory organization
      ↓
Final Normalization
      ↓
Language Modeling Head
```

Key Features:
- Configurable SOM insertion points between transformer blocks
- Multiple memory integration strategies
- Full MoE support from Qwen3 architecture
- Differentiable end-to-end training
"""

import keras
import numpy as np
from typing import Optional, Union, Any, Dict, List, Tuple, Literal

# ---------------------------------------------------------------------
# Local Imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.transformers import TransformerLayer
from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.layers.moe import MoEConfig, ExpertConfig, GatingConfig
from dl_techniques.layers.memory.som_nd_soft_layer import SoftSOMLayer

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class Qwen3SOM(keras.Model):
    """
    Qwen3 language model with Self-Organizing Map memory integration.

    This model extends the standard Qwen3 architecture by inserting SOM layers
    at strategic points to create structured, topologically-organized memory
    representations. The SOM layers learn to map hidden states onto low-dimensional
    grids, providing implicit clustering and memory organization.

    **Memory Integration Strategies:**
    - **"interleaved"**: SOM layers inserted between transformer blocks
    - **"bottleneck"**: Single SOM layer at the middle of the network
    - **"post_processing"**: SOM layer after all transformer blocks
    - **"multi_scale"**: Multiple SOM layers with different grid sizes

    **Architecture Benefits:**
    - Topologically-organized latent representations
    - Implicit regularization through soft clustering
    - Structured memory for improved generalization
    - Differentiable throughout for end-to-end training

    Args:
        vocab_size: Integer, size of the vocabulary. Defaults to 151936.
        hidden_size: Integer, dimensionality of encoder layers. Defaults to 2048.
        num_layers: Integer, number of transformer blocks. Defaults to 12.
        num_attention_heads: Integer, number of attention heads. Defaults to 16.
        num_key_value_heads: Integer, number of key-value heads for GQA. Defaults to 4.
        max_seq_len: Integer, maximum sequence length. Defaults to 8192.
        moe_layers: List of integers, layer indices that use MoE. Defaults to empty list.
        num_experts: Integer, total number of experts in MoE layers. Defaults to 64.
        num_experts_per_tok: Integer, number of experts activated per token. Defaults to 8.
        moe_intermediate_size: Integer, individual expert intermediate size. Defaults to 1408.
        rope_theta: Float, RoPE theta parameter. Defaults to 10_000_000.0.

        # SOM Configuration
        som_strategy: Literal["interleaved", "bottleneck", "post_processing", "multi_scale"],
            Memory integration strategy. Defaults to "bottleneck".
        som_layers: Optional[List[int]], specific layer indices for SOM insertion.
            If None, uses strategy-based defaults.
        som_grid_shape: Tuple[int, ...], shape of SOM grid. Defaults to (16, 16).
        som_temperature: Float, temperature for SOM softmax. Defaults to 1.0.
        som_use_per_dimension: Boolean, use per-dimension softmax. Defaults to True.
        som_reconstruction_weight: Float, SOM reconstruction loss weight. Defaults to 0.5.
        som_topological_weight: Float, SOM topological loss weight. Defaults to 0.1.
        som_sharpness_weight: Float, SOM sharpness loss weight. Defaults to 0.0.

        # Standard Qwen3 parameters
        norm_eps: Float, epsilon for normalization layers. Defaults to 1e-6.
        dropout_rate: Float, dropout rate for regularization. Defaults to 0.0.
        initializer_range: Float, standard deviation for weight initialization. Defaults to 0.02.
        normalization_type: String, type of normalization layer. Defaults to "rms_norm".
        ffn_type: String, type of feed-forward network. Defaults to "swiglu".
        use_stochastic_depth: Boolean, whether to enable stochastic depth. Defaults to False.
        stochastic_depth_rate: Float, drop path rate for stochastic depth. Defaults to 0.1.
        **kwargs: Additional keyword arguments for the `keras.Model` base class.

    Example:
        ```python
        # Create a Qwen3-SOM model with bottleneck memory
        model = Qwen3SOM(
            vocab_size=32000,
            hidden_size=768,
            num_layers=12,
            som_strategy="bottleneck",
            som_grid_shape=(16, 16),
            som_reconstruction_weight=1.0
        )

        # Create model with interleaved memory layers
        model = Qwen3SOM(
            vocab_size=32000,
            hidden_size=768,
            num_layers=12,
            som_strategy="interleaved",
            som_layers=[3, 6, 9],  # Insert SOM at these layers
            som_grid_shape=(8, 8)
        )

        # Use for text generation
        input_ids = keras.Input(shape=(None,), dtype="int32")
        attention_mask = keras.Input(shape=(None,), dtype="int32")
        logits = model({"input_ids": input_ids, "attention_mask": attention_mask})
        ```
    """

    # Model variant configurations
    MODEL_VARIANTS = {
        "small_som": {
            "vocab_size": 32000,
            "hidden_size": 768,
            "num_layers": 12,
            "num_attention_heads": 12,
            "num_key_value_heads": 4,
            "max_seq_len": 4096,
            "moe_layers": [3, 6, 9],
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "som_strategy": "bottleneck",
            "som_grid_shape": (12, 12),
            "som_reconstruction_weight": 1.0,
            "description": "Small Qwen3-SOM with bottleneck memory"
        },
        "medium_som": {
            "vocab_size": 100000,
            "hidden_size": 1024,
            "num_layers": 24,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "max_seq_len": 16384,
            "moe_layers": list(range(6, 24, 4)),
            "num_experts": 16,
            "num_experts_per_tok": 4,
            "som_strategy": "interleaved",
            "som_layers": [8, 16],
            "som_grid_shape": (16, 16),
            "som_reconstruction_weight": 0.5,
            "description": "Medium Qwen3-SOM with interleaved memory"
        },
        "tiny_som": {
            "vocab_size": 32000,
            "hidden_size": 512,
            "num_layers": 6,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "max_seq_len": 2048,
            "moe_layers": [],
            "som_strategy": "post_processing",
            "som_grid_shape": (8, 8),
            "som_reconstruction_weight": 0.8,
            "description": "Tiny Qwen3-SOM with post-processing memory"
        },
    }

    def __init__(
            self,
            # Qwen3 base parameters
            vocab_size: int = 151936,
            hidden_size: int = 2048,
            num_layers: int = 12,
            num_attention_heads: int = 16,
            num_key_value_heads: int = 4,
            max_seq_len: int = 8192,
            moe_layers: Optional[List[int]] = None,
            num_experts: int = 64,
            num_experts_per_tok: int = 8,
            moe_intermediate_size: int = 1408,
            rope_theta: float = 10_000_000.0,
            # SOM parameters
            som_strategy: Literal["interleaved", "bottleneck", "post_processing", "multi_scale"] = "bottleneck",
            som_layers: Optional[List[int]] = None,
            som_grid_shape: Tuple[int, ...] = (16, 16),
            som_temperature: float = 1.0,
            som_use_per_dimension: bool = True,
            som_reconstruction_weight: float = 0.5,
            som_topological_weight: float = 0.1,
            som_sharpness_weight: float = 0.0,
            # Standard parameters
            norm_eps: float = 1e-6,
            dropout_rate: float = 0.0,
            initializer_range: float = 0.02,
            normalization_type: str = "rms_norm",
            ffn_type: str = "swiglu",
            use_stochastic_depth: bool = False,
            stochastic_depth_rate: float = 0.1,
            **kwargs: Any
    ) -> None:
        # CRITICAL: Call super() FIRST for Keras models
        super().__init__(**kwargs)

        # Set defaults
        if moe_layers is None:
            moe_layers = []

        # Validate configuration
        self._validate_config(
            vocab_size, hidden_size, num_layers, num_attention_heads,
            num_key_value_heads, num_experts, num_experts_per_tok, moe_layers
        )

        # Store all configuration parameters for serialization
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_seq_len = max_seq_len
        self.moe_layers = moe_layers
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.rope_theta = rope_theta

        # SOM configuration
        self.som_strategy = som_strategy
        self.som_grid_shape = som_grid_shape
        self.som_temperature = som_temperature
        self.som_use_per_dimension = som_use_per_dimension
        self.som_reconstruction_weight = som_reconstruction_weight
        self.som_topological_weight = som_topological_weight
        self.som_sharpness_weight = som_sharpness_weight

        # Standard parameters
        self.norm_eps = norm_eps
        self.dropout_rate = dropout_rate
        self.initializer_range = initializer_range
        self.normalization_type = normalization_type
        self.ffn_type = ffn_type
        self.use_stochastic_depth = use_stochastic_depth
        self.stochastic_depth_rate = stochastic_depth_rate

        # Calculate head dimension
        self.head_dim = self.hidden_size // self.num_attention_heads

        # Determine SOM layer positions based on strategy
        self.som_layers = self._determine_som_layers(som_layers, som_strategy, num_layers)

        # Build the model architecture
        self._build_architecture()

        # Log model creation
        logger.info(
            f"Created Qwen3-SOM model: {self.num_layers} layers, "
            f"hidden_size={self.hidden_size}, strategy={self.som_strategy}"
        )
        logger.info(f"SOM layers at positions: {self.som_layers}")
        logger.info(f"SOM grid shape: {self.som_grid_shape}")

    def _validate_config(
            self,
            vocab_size: int,
            hidden_size: int,
            num_layers: int,
            num_attention_heads: int,
            num_key_value_heads: int,
            num_experts: int,
            num_experts_per_tok: int,
            moe_layers: List[int],
    ) -> None:
        """Validate model configuration parameters."""
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if num_attention_heads <= 0:
            raise ValueError(f"num_attention_heads must be positive, got {num_attention_heads}")
        if num_key_value_heads <= 0:
            raise ValueError(f"num_key_value_heads must be positive, got {num_key_value_heads}")
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_attention_heads ({num_attention_heads})"
            )
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({num_key_value_heads})"
            )
        if num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {num_experts}")
        if num_experts_per_tok <= 0:
            raise ValueError(f"num_experts_per_tok must be positive, got {num_experts_per_tok}")
        if num_experts_per_tok > num_experts:
            raise ValueError(
                f"num_experts_per_tok ({num_experts_per_tok}) cannot exceed "
                f"num_experts ({num_experts})"
            )
        if any(layer_idx < 0 or layer_idx >= num_layers for layer_idx in moe_layers):
            raise ValueError(f"All MoE layer indices must be between 0 and {num_layers - 1}")

    def _determine_som_layers(
            self,
            som_layers: Optional[List[int]],
            strategy: str,
            num_layers: int
    ) -> List[int]:
        """Determine SOM layer positions based on strategy."""
        if som_layers is not None:
            # Validate custom positions
            if any(idx < 0 or idx >= num_layers for idx in som_layers):
                raise ValueError(f"SOM layer indices must be between 0 and {num_layers - 1}")
            return sorted(som_layers)

        # Apply strategy-based defaults
        if strategy == "bottleneck":
            # Single SOM at middle of network
            return [num_layers // 2]
        elif strategy == "interleaved":
            # SOM every ~4 layers
            spacing = max(3, num_layers // 4)
            return list(range(spacing, num_layers, spacing))
        elif strategy == "post_processing":
            # Single SOM after all transformer blocks
            return [num_layers]
        elif strategy == "multi_scale":
            # Multiple SOMs at strategic points (1/4, 1/2, 3/4, end)
            positions = [
                num_layers // 4,
                num_layers // 2,
                (3 * num_layers) // 4,
                num_layers
            ]
            return sorted(set(positions))  # Remove duplicates
        else:
            raise ValueError(f"Unknown SOM strategy: {strategy}")

    def _build_architecture(self) -> None:
        """Build all model components following modern Keras 3 patterns."""

        # Token embedding layer
        self.embeddings = keras.layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.hidden_size,
            embeddings_initializer=keras.initializers.TruncatedNormal(
                stddev=self.initializer_range
            ),
            name="token_embedding"
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
                        "ffn_expansion_factor": max(1, self.moe_intermediate_size // self.hidden_size)
                    }
                ),
                gating_config=GatingConfig(
                    top_k=self.num_experts_per_tok,
                    gating_type="linear"
                )
            )

        # Create stochastic depth schedule
        if self.stochastic_depth_rate > 0:
            dpr = [x for x in np.linspace(0.0, self.stochastic_depth_rate, self.num_layers)]
        else:
            dpr = [0.0 for _ in range(self.num_layers)]

        # Create transformer blocks and SOM layers
        self.blocks = []
        self.som_memory_layers = {}

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

            # Add SOM layer after this transformer block if specified
            if i in self.som_layers and i < self.num_layers:
                som_layer = SoftSOMLayer(
                    grid_shape=self.som_grid_shape,
                    input_dim=self.hidden_size,
                    temperature=self.som_temperature,
                    use_per_dimension_softmax=self.som_use_per_dimension,
                    use_reconstruction_loss=True,
                    reconstruction_weight=self.som_reconstruction_weight,
                    topological_weight=self.som_topological_weight,
                    sharpness_weight=self.som_sharpness_weight,
                    name=f"som_memory_{i}"
                )
                self.som_memory_layers[i] = som_layer

        # Handle post-processing SOM (after all transformer blocks)
        if self.num_layers in self.som_layers:
            som_layer = SoftSOMLayer(
                grid_shape=self.som_grid_shape,
                input_dim=self.hidden_size,
                temperature=self.som_temperature,
                use_per_dimension_softmax=self.som_use_per_dimension,
                use_reconstruction_loss=True,
                reconstruction_weight=self.som_reconstruction_weight,
                topological_weight=self.som_topological_weight,
                sharpness_weight=self.som_sharpness_weight,
                name=f"som_memory_post"
            )
            self.som_memory_layers[self.num_layers] = som_layer

        # Final normalization layer
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
            training: Optional[bool] = None,
            return_dict: bool = False,
            return_som_assignments: bool = False
    ) -> Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]:
        """
        Forward pass of the Qwen3-SOM model.

        Args:
            inputs: Input token IDs or dictionary containing inputs.
            attention_mask: Mask to avoid attention on padding tokens.
            training: Boolean, whether the model is in training mode.
            return_dict: Boolean, whether to return outputs as a dictionary.
            return_som_assignments: Boolean, whether to return SOM assignments for analysis.

        Returns:
            Model outputs. The format depends on `return_dict` and `return_som_assignments`.
        """
        # Parse inputs
        if isinstance(inputs, dict):
            input_ids = inputs.get("input_ids")
            if input_ids is None:
                raise ValueError("Dictionary input must contain 'input_ids' key")
            attention_mask = inputs.get("attention_mask", attention_mask)
        else:
            input_ids = inputs

        # Token embeddings
        hidden_states = self.embeddings(input_ids)

        # Track SOM assignments if requested
        som_assignments = {} if return_som_assignments else None

        # Pass through transformer blocks with SOM memory layers
        for i, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                attention_mask=attention_mask,
                training=training
            )

            # Apply SOM memory layer if present at this position
            if i in self.som_memory_layers:
                # Flatten sequence dimension for SOM processing
                batch_size = keras.ops.shape(hidden_states)[0]
                seq_len = keras.ops.shape(hidden_states)[1]

                # Reshape: (batch, seq, hidden) -> (batch * seq, hidden)
                flat_states = keras.ops.reshape(
                    hidden_states,
                    (batch_size * seq_len, self.hidden_size)
                )

                # Process through SOM
                som_output = self.som_memory_layers[i](flat_states, training=training)

                # Reshape back: (batch * seq, hidden) -> (batch, seq, hidden)
                hidden_states = keras.ops.reshape(
                    som_output,
                    (batch_size, seq_len, self.hidden_size)
                )

                # Store assignments if requested
                if return_som_assignments:
                    assignments = self.som_memory_layers[i].get_soft_assignments(flat_states)
                    som_assignments[f"layer_{i}"] = keras.ops.reshape(
                        assignments,
                        (batch_size, seq_len, *self.som_grid_shape)
                    )

        # Apply post-processing SOM if present
        if self.num_layers in self.som_memory_layers:
            batch_size = keras.ops.shape(hidden_states)[0]
            seq_len = keras.ops.shape(hidden_states)[1]

            flat_states = keras.ops.reshape(
                hidden_states,
                (batch_size * seq_len, self.hidden_size)
            )

            som_output = self.som_memory_layers[self.num_layers](flat_states, training=training)

            hidden_states = keras.ops.reshape(
                som_output,
                (batch_size, seq_len, self.hidden_size)
            )

            if return_som_assignments:
                assignments = self.som_memory_layers[self.num_layers].get_soft_assignments(flat_states)
                som_assignments["post_processing"] = keras.ops.reshape(
                    assignments,
                    (batch_size, seq_len, *self.som_grid_shape)
                )

        # Final normalization
        hidden_states = self.final_norm(hidden_states)

        # Language modeling head
        logits = self.lm_head(hidden_states)

        # Return in requested format
        if return_dict or return_som_assignments:
            output = {"logits": logits}
            if return_som_assignments:
                output["som_assignments"] = som_assignments
            return output
        else:
            return logits

    def get_som_prototypes(self) -> Dict[str, keras.KerasTensor]:
        """
        Get learned SOM prototype vectors from all memory layers.

        Returns:
            Dictionary mapping layer names to their prototype weight maps.

        Example:
            ```python
            prototypes = model.get_som_prototypes()
            layer_3_prototypes = prototypes['layer_3']
            print(f"Prototypes shape: {layer_3_prototypes.shape}")
            # Shape: (grid_height, grid_width, hidden_size)
            ```
        """
        prototypes = {}
        for layer_idx, som_layer in self.som_memory_layers.items():
            key = f"post_processing" if layer_idx == self.num_layers else f"layer_{layer_idx}"
            prototypes[key] = som_layer.get_weights_map()
        return prototypes

    @classmethod
    def from_variant(
            cls,
            variant: str,
            **kwargs: Any
    ) -> "Qwen3SOM":
        """
        Create a Qwen3-SOM model from a predefined variant.

        Args:
            variant: String, one of "small_som", "medium_som", "tiny_som"
            **kwargs: Additional arguments to override defaults

        Returns:
            Qwen3SOM model instance

        Example:
            ```python
            # Create tiny variant with custom SOM config
            model = Qwen3SOM.from_variant(
                "tiny_som",
                som_reconstruction_weight=1.5
            )
            ```
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        config = cls.MODEL_VARIANTS[variant].copy()
        config.pop("description", None)

        logger.info(f"Creating Qwen3-SOM-{variant.upper()} model")
        logger.info(f"Configuration: {cls.MODEL_VARIANTS[variant]['description']}")

        return cls(**config, **kwargs)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "max_seq_len": self.max_seq_len,
            "moe_layers": self.moe_layers,
            "num_experts": self.num_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
            "moe_intermediate_size": self.moe_intermediate_size,
            "rope_theta": self.rope_theta,
            "som_strategy": self.som_strategy,
            "som_grid_shape": self.som_grid_shape,
            "som_temperature": self.som_temperature,
            "som_use_per_dimension": self.som_use_per_dimension,
            "som_reconstruction_weight": self.som_reconstruction_weight,
            "som_topological_weight": self.som_topological_weight,
            "som_sharpness_weight": self.som_sharpness_weight,
            "norm_eps": self.norm_eps,
            "dropout_rate": self.dropout_rate,
            "initializer_range": self.initializer_range,
            "normalization_type": self.normalization_type,
            "ffn_type": self.ffn_type,
            "use_stochastic_depth": self.use_stochastic_depth,
            "stochastic_depth_rate": self.stochastic_depth_rate,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Qwen3SOM":
        """Create model from configuration."""
        return cls(**config)

    def summary(self, **kwargs) -> None:
        """Print model summary with SOM-specific information."""
        super().summary(**kwargs)

        # Calculate statistics
        total_layers = self.num_layers
        moe_layers_count = len(self.moe_layers)
        dense_layers_count = total_layers - moe_layers_count
        som_layers_count = len(self.som_memory_layers)

        logger.info("Qwen3-SOM Model Configuration:")
        logger.info(f"  - Architecture: {total_layers} transformer layers")
        logger.info(f"    - {dense_layers_count} Dense layers")
        logger.info(f"    - {moe_layers_count} MoE layers")
        logger.info(f"    - {som_layers_count} SOM memory layers")
        logger.info(f"  - SOM Strategy: {self.som_strategy}")
        logger.info(f"  - SOM Grid Shape: {self.som_grid_shape}")
        logger.info(f"  - SOM Layer Positions: {sorted(self.som_memory_layers.keys())}")
        logger.info(f"  - Hidden size: {self.hidden_size}")
        logger.info(f"  - Attention heads: {self.num_attention_heads} (KV heads: {self.num_key_value_heads})")
        logger.info(f"  - Vocabulary: {self.vocab_size:,} tokens")
        logger.info(f"  - Max sequence length: {self.max_seq_len:,}")
        if self.moe_layers:
            logger.info(f"  - MoE Configuration:")
            logger.info(f"    - MoE layer indices: {self.moe_layers}")
            logger.info(f"    - Experts per layer: {self.num_experts}")
            logger.info(f"    - Active per token: {self.num_experts_per_tok}")


# ---------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------

def create_qwen3som_generation(config: Dict[str, Any]) -> keras.Model:
    """
    Create a Qwen3-SOM model optimized for text generation tasks.

    Args:
        config: Dictionary containing complete configuration for Qwen3SOM.

    Returns:
        Compiled Keras Model ready for generation tasks.

    Example:
        ```python
        config = {
            "vocab_size": 32000,
            "hidden_size": 768,
            "num_layers": 12,
            "som_strategy": "bottleneck",
            "som_grid_shape": (16, 16)
        }
        model = create_qwen3som_generation(config)
        ```
    """
    logger.info("Creating Qwen3-SOM model for text generation.")

    qwen3som_backbone = Qwen3SOM(**config, name="qwen3som_backbone")
    input_ids = keras.Input(shape=(None,), dtype="int32", name="input_ids")
    attention_mask = keras.Input(shape=(None,), dtype="int32", name="attention_mask")

    logits = qwen3som_backbone(
        inputs={"input_ids": input_ids, "attention_mask": attention_mask}
    )

    model = keras.Model(
        inputs=[input_ids, attention_mask],
        outputs=logits,
        name="qwen3som_for_generation"
    )

    param_count = model.count_params()
    logger.info(f"Created Qwen3-SOM generation model with {param_count:,} parameters.")
    return model


def create_qwen3som_classification(
        config: Dict[str, Any],
        num_labels: int,
        pooling_strategy: str = "cls",
        classifier_dropout: Optional[float] = None,
) -> keras.Model:
    """
    Create a Qwen3-SOM model for sequence classification tasks.

    The SOM memory layers provide structured representations that can improve
    classification performance through implicit regularization.

    Args:
        config: Dictionary containing complete configuration for Qwen3SOM.
        num_labels: Number of output labels for classification.
        pooling_strategy: Method to pool sequence output ("cls" or "mean").
        classifier_dropout: Dropout rate for classification head.

    Returns:
        Compiled Keras Model ready for classification.

    Example:
        ```python
        config = {"vocab_size": 32000, "hidden_size": 768, ...}
        model = create_qwen3som_classification(config, num_labels=10)
        ```
    """
    if num_labels <= 0:
        raise ValueError(f"num_labels must be positive, got {num_labels}")
    if pooling_strategy not in ["cls", "mean"]:
        raise ValueError(f"pooling_strategy must be 'cls' or 'mean', got '{pooling_strategy}'")

    logger.info(f"Creating Qwen3-SOM classification model with {num_labels} labels.")

    qwen3som_backbone = Qwen3SOM(**config, name="qwen3som_backbone")
    input_ids = keras.Input(shape=(None,), dtype="int32", name="input_ids")
    attention_mask = keras.Input(shape=(None,), dtype="int32", name="attention_mask")

    sequence_output = qwen3som_backbone(
        inputs={"input_ids": input_ids, "attention_mask": attention_mask}
    )

    # Apply pooling strategy
    if pooling_strategy == "cls":
        pooled_output = sequence_output[:, 0]
    else:  # "mean" pooling
        mask = keras.ops.expand_dims(keras.ops.cast(attention_mask, sequence_output.dtype), axis=-1)
        masked_output = sequence_output * mask
        summed_output = keras.ops.sum(masked_output, axis=1)
        num_tokens = keras.ops.maximum(
            keras.ops.sum(keras.ops.cast(attention_mask, 'float32'), axis=1, keepdims=True), 1.0
        )
        pooled_output = summed_output / num_tokens

    # Classifier head with optional dropout
    dropout_rate = classifier_dropout if classifier_dropout is not None else config.get("dropout_rate", 0.1)
    if dropout_rate > 0.0:
        pooled_output = keras.layers.Dropout(dropout_rate, name="classifier_dropout")(pooled_output)

    initializer_range = config.get("initializer_range", 0.02)
    logits = keras.layers.Dense(
        units=num_labels,
        kernel_initializer=keras.initializers.TruncatedNormal(stddev=initializer_range),
        name="classifier_head",
    )(pooled_output)

    model = keras.Model(
        inputs=[input_ids, attention_mask],
        outputs=logits,
        name="qwen3som_for_classification"
    )

    param_count = model.count_params()
    logger.info(f"Created Qwen3-SOM classification model with {param_count:,} parameters.")
    return model


def create_qwen3som(
        config_or_variant: Union[str, Dict[str, Any]],
        task_type: str = "generation",
        **kwargs: Any,
) -> keras.Model:
    """
    High-level factory to create Qwen3-SOM models for common tasks.

    This provides a unified interface for creating models with SOM-based memory
    for various tasks including generation and classification.

    Args:
        config_or_variant: Either a variant string or custom configuration dictionary.
        task_type: Type of model to create ("generation" or "classification").
        **kwargs: Additional arguments to override configuration or provide task-specific settings.

    Returns:
        Keras Model configured for the specified task.

    Example:
        ```python
        # Create generation model from variant
        gen_model = create_qwen3som("tiny_som")

        # Create classification model with custom config
        config = {"hidden_size": 512, "num_layers": 6, ...}
        clf_model = create_qwen3som(
            config,
            task_type="classification",
            num_labels=5,
            som_reconstruction_weight=1.0
        )
        ```
    """
    # Determine base configuration
    if isinstance(config_or_variant, str):
        variant = config_or_variant
        if variant not in Qwen3SOM.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available: {list(Qwen3SOM.MODEL_VARIANTS.keys())}"
            )
        config = Qwen3SOM.MODEL_VARIANTS[variant].copy()
        config.pop("description", None)
    elif isinstance(config_or_variant, dict):
        config = config_or_variant.copy()
    else:
        raise TypeError("config_or_variant must be a string or dictionary.")

    # Separate task-specific and model configuration kwargs
    task_kwargs = {}
    model_kwargs = {}
    task_specific_keys = ["num_labels", "pooling_strategy", "classifier_dropout"]

    for key, value in kwargs.items():
        if key in task_specific_keys:
            task_kwargs[key] = value
        else:
            model_kwargs[key] = value

    # Apply overrides to base config
    config.update(model_kwargs)

    # Build requested model
    if task_type == "generation":
        return create_qwen3som_generation(config)
    elif task_type == "classification":
        num_labels = task_kwargs.pop("num_labels", None)
        if num_labels is None:
            raise ValueError("num_labels required for classification task.")
        return create_qwen3som_classification(config, num_labels, **task_kwargs)
    else:
        raise ValueError(f"Unknown task_type '{task_type}'. Supported: 'generation', 'classification'.")


# ---------------------------------------------------------------------
# Utility Functions for SOM Analysis
# ---------------------------------------------------------------------

def visualize_som_assignments(
        model: Qwen3SOM,
        input_ids: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
        layer_name: Optional[str] = None
) -> Dict[str, np.ndarray]:
    """
    Visualize SOM assignments for given inputs.

    Args:
        model: Trained Qwen3SOM model.
        input_ids: Input token IDs of shape (batch, seq_len).
        attention_mask: Optional attention mask.
        layer_name: Specific SOM layer to analyze (None for all).

    Returns:
        Dictionary mapping layer names to assignment arrays.

    Example:
        ```python
        model = Qwen3SOM.from_variant("tiny_som")
        input_ids = np.random.randint(0, 32000, (4, 128))
        assignments = visualize_som_assignments(model, input_ids)

        # Visualize assignments for a specific layer
        import matplotlib.pyplot as plt
        layer_assignments = assignments['layer_3'][0, 0]  # First token, first sample
        plt.imshow(layer_assignments, cmap='viridis')
        plt.title('SOM Activation Map')
        plt.colorbar()
        plt.show()
        ```
    """
    outputs = model(
        {"input_ids": input_ids, "attention_mask": attention_mask},
        return_dict=True,
        return_som_assignments=True,
        training=False
    )

    assignments = outputs["som_assignments"]

    if layer_name is not None:
        if layer_name not in assignments:
            raise ValueError(f"Layer '{layer_name}' not found. Available: {list(assignments.keys())}")
        return {layer_name: keras.ops.convert_to_numpy(assignments[layer_name])}

    # Convert all to numpy
    return {k: keras.ops.convert_to_numpy(v) for k, v in assignments.items()}


def analyze_som_clustering(
        model: Qwen3SOM,
        input_dataset: Any,
        max_samples: int = 1000
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze clustering quality of SOM layers.

    Args:
        model: Trained Qwen3SOM model.
        input_dataset: Dataset of inputs to analyze.
        max_samples: Maximum number of samples to process.

    Returns:
        Dictionary with clustering metrics for each SOM layer.

    Example:
        ```python
        model = Qwen3SOM.from_variant("small_som")
        # Assuming you have a dataset
        metrics = analyze_som_clustering(model, train_dataset, max_samples=500)

        for layer_name, stats in metrics.items():
            print(f"{layer_name}:")
            print(f"  Average entropy: {stats['avg_entropy']:.4f}")
            print(f"  Utilization: {stats['neuron_utilization']:.2%}")
        ```
    """
    logger.info(f"Analyzing SOM clustering for up to {max_samples} samples...")

    prototypes = model.get_som_prototypes()
    results = {}

    # This is a placeholder - implement based on your specific needs
    for layer_name in prototypes.keys():
        results[layer_name] = {
            "num_neurons": int(np.prod(model.som_grid_shape)),
            "grid_shape": model.som_grid_shape,
            "message": "Implement clustering analysis based on your metrics"
        }

    return results

# ---------------------------------------------------------------------