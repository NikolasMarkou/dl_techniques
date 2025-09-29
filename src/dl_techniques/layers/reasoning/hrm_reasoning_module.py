"""
This module defines the HierarchicalReasoningModule, a composite Keras layer that
serves as a highly configurable computational unit for deep sequential models.

This layer is a "stack" or "module" that groups together multiple `TransformerLayer`
instances. Its primary purpose is to encapsulate a multi-step refinement process,
where each layer in the stack transforms the output of the previous one. While its
default configuration is optimized for Hierarchical Reasoning Models (HRM), its
core components are now fully configurable.

Key Architectural Features:

1.  **Configurable Stack of Transformer Layers:**
    -   The module is a sequential container for `num_layers` instances of the generic
        `TransformerLayer`.
    -   The internal architecture of these layers is fully configurable via the
        module's constructor, including attention type, normalization type and position,
        and FFN type.
    -   **Default Configuration (for HRM):**
        - **Normalization Position:** 'post'
        - **Normalization Type:** 'rms_norm'
        - **Feed-Forward Network:** 'swiglu'
        - **Attention Type:** 'multi_head'

2.  **Input Injection:**
    -   A defining feature of this module is its "input injection" mechanism. Before
        the main processing begins, it combines `hidden_states` and an `input_injection`
        via element-wise addition.
    -   This allows the module to incorporate fresh information at the beginning of its
        processing cycle.

The operational flow is straightforward:
1.  Add the `input_injection` to the `hidden_states`.
2.  Process the resulting sum sequentially through all the configured `TransformerLayer` instances.
3.  Return the final transformed hidden states.
"""

import keras
from typing import Optional, Union, Tuple, Any, Dict, List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..transformer import (
    TransformerLayer,
    AttentionType,
    NormalizationType,
    NormalizationPosition,
    FFNType
)

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class HierarchicalReasoningModule(keras.layers.Layer):
    """
    Configurable multi-layer reasoning module with input injection.

    This composite layer implements a stack of `TransformerLayer` instances. It is highly
    configurable, allowing for experimentation with different transformer architectures,
    while defaulting to a high-performance configuration suitable for hierarchical reasoning
    (post-norm, RMSNorm, SwiGLU). It features an input injection mechanism to integrate
    fresh information at the start of each reasoning cycle.

    **Intent**: Provide a general-purpose, stackable module for building deep transformer-based
    models, with an opinionated default configuration optimized for reasoning tasks.

    Args:
        num_layers: Integer, number of transformer layers in the stack.
        embed_dim: Integer, embedding dimension.
        num_heads: Integer, number of attention heads. Defaults to 8.
        ffn_expansion_factor: Integer, expansion factor for FFN. Defaults to 4.
        attention_type: The type of attention mechanism to use. Defaults to 'multi_head'.
        normalization_type: The type of normalization layer to use. Defaults to 'rms_norm'.
        normalization_position: Position of the norm layer ('pre' or 'post'). Defaults to 'post'.
        ffn_type: The type of feed-forward network to use. Defaults to 'swiglu'.
        dropout_rate: Float (0-1), dropout rate for attention and FFN. Defaults to 0.0.
        use_bias: Boolean, whether to use bias terms. Defaults to False.
        kernel_initializer: String or Initializer for weights. Defaults to "he_normal".
        kernel_regularizer: Optional regularizer for weights. Defaults to None.
        **kwargs: Additional Layer base class arguments.

    Input shape:
        A list of two identical 3D tensors: `[hidden_states, input_injection]`
        of shape `(batch_size, sequence_length, embed_dim)`.

    Output shape:
        3D tensor with shape `(batch_size, sequence_length, embed_dim)`.
    """

    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        num_heads: int = 8,
        ffn_expansion_factor: int = 4,
        attention_type: AttentionType = 'multi_head',
        normalization_type: NormalizationType = 'rms_norm',
        normalization_position: NormalizationPosition = 'post',
        ffn_type: FFNType = 'swiglu',
        dropout_rate: float = 0.0,
        use_bias: bool = False,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        if ffn_expansion_factor <= 0:
            raise ValueError(f"ffn_expansion_factor must be positive, got {ffn_expansion_factor}")

        # Store all configuration parameters for serialization
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ffn_expansion_factor = ffn_expansion_factor
        self.attention_type = attention_type
        self.normalization_type = normalization_type
        self.normalization_position = normalization_position
        self.ffn_type = ffn_type
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        intermediate_size = self.embed_dim * self.ffn_expansion_factor

        # Create a list of TransformerLayer instances based on the provided configuration
        self.blocks: List[TransformerLayer] = []
        for i in range(self.num_layers):
            block = TransformerLayer(
                hidden_size=self.embed_dim,
                num_heads=self.num_heads,
                intermediate_size=intermediate_size,
                # Pass configurable parameters directly to the TransformerLayer
                attention_type=self.attention_type,
                normalization_type=self.normalization_type,
                normalization_position=self.normalization_position,
                ffn_type=self.ffn_type,
                # Map other parameters
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.dropout_rate,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f"transformer_layer_{i}"
            )
            self.blocks.append(block)

    def build(self, input_shape: List[Tuple[Optional[int], ...]]) -> None:
        """Build the module and all its internal TransformerLayer sub-layers."""
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError("Input must be a list of two tensors: [hidden_states, input_injection]")

        hidden_states_shape, input_injection_shape = input_shape
        if hidden_states_shape != input_injection_shape:
            raise ValueError(f"Shapes of hidden_states {hidden_states_shape} and input_injection "
                             f"{input_injection_shape} must be identical.")
        if hidden_states_shape[-1] != self.embed_dim:
            raise ValueError(f"Input feature dimension ({hidden_states_shape[-1]}) must match "
                             f"embed_dim ({self.embed_dim})")

        for block in self.blocks:
            block.build(hidden_states_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: List[keras.KerasTensor],
        training: Optional[bool] = None,
        mask: Optional[keras.KerasTensor] = None
    ) -> keras.KerasTensor:
        """
        Forward pass with input injection and sequential refinement.

        Args:
            inputs: A list of two tensors: `[hidden_states, input_injection]`.
            training: Boolean indicating training mode.
            mask: Optional attention mask to apply to all layers.

        Returns:
            Refined hidden states tensor.
        """
        if not isinstance(inputs, list) or len(inputs) != 2:
            raise ValueError("Input must be a list of two tensors: [hidden_states, input_injection]")
        hidden_states, input_injection = inputs

        x = hidden_states + input_injection

        for block in self.blocks:
            x = block(x, training=training, attention_mask=mask)

        return x

    def compute_output_shape(self, input_shape: List[Tuple[Optional[int], ...]]) -> Tuple[Optional[int], ...]:
        """Compute output shape, which is the shape of the `hidden_states` input."""
        return input_shape[0]

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            "num_layers": self.num_layers,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ffn_expansion_factor": self.ffn_expansion_factor,
            "attention_type": self.attention_type,
            "normalization_type": self.normalization_type,
            "normalization_position": self.normalization_position,
            "ffn_type": self.ffn_type,
            "dropout_rate": self.dropout_rate,
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config

# ---------------------------------------------------------------------
