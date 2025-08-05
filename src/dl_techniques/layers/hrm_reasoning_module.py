"""
This module defines the HierarchicalReasoningModule, a composite Keras layer that
serves as a major computational unit within the Hierarchical Reasoning Model (HRM) architecture.

This layer is not a single computational block but rather a "stack" or "module" that
groups together multiple `HierarchicalReasoningBlock` layers. Its primary purpose is to
encapsulate a multi-step reasoning process, where each block in the stack refines
the output of the previous one. This creates a deep, hierarchical transformation of the
input representations.

Key Architectural Features:

1.  **Stack of Reasoning Blocks:**
    -   The module is fundamentally a sequential container for `num_layers` instances
        of `HierarchicalReasoningBlock`.
    -   This stacked design allows the model to build up increasingly abstract and
        complex representations of the data layer by layer.

2.  **Input Injection:**
    -   A defining feature of this module is its "input injection" mechanism. Before
        the main processing begins, it takes two inputs: the current `hidden_states`
        (the state being refined) and an `input_injection`.
    -   These two tensors are combined via element-wise addition (`hidden_states + input_injection`).
        This allows the module to incorporate fresh information at the beginning of its
        reasoning process.
    -   In the full HRM architecture, this is used to inject:
        -   The raw input embeddings into the low-level module.
        -   The refined low-level state into the high-level module.

3.  **Use in HRM:**
    -   The full HRM model uses two instances of this module to create its dual-level
        reasoning structure:
        -   **A Low-Level (L) Module:** Refines a detailed state representation.
        -   **A High-Level (H) Module:** Processes the output of the L-module to form a more
          abstract state.
    -   By grouping the blocks into a module, the overall architecture becomes cleaner
        and more modular.

The operational flow is straightforward:
1.  Add the `input_injection` to the `hidden_states`.
2.  Process the resulting sum sequentially through all the `HierarchicalReasoningBlock` layers.
3.  Return the final transformed hidden states.
"""

import keras
from typing import Optional, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .hrm_block import HierarchicalReasoningBlock

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class HierarchicalReasoningModule(keras.layers.Layer):
    """
    Reasoning module that contains multiple hierarchical reasoning blocks.

    Performs input injection (addition) before processing through transformer blocks.
    Used for both high-level (H) and low-level (L) reasoning modules in HRM.

    Args:
        num_layers: Number of hierarchical reasoning blocks
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        ffn_expansion_factor: Feed-forward network expansion factor
        dropout_rate: Dropout rate for attention and FFN
        use_bias: Whether to use bias in linear layers
        kernel_initializer: Initializer for kernel weights
        kernel_regularizer: Regularizer for kernel weights
        **kwargs: Additional layer arguments
    """

    def __init__(
            self,
            num_layers: int,
            embed_dim: int,
            num_heads: int = 8,
            ffn_expansion_factor: int = 4,
            dropout_rate: float = 0.0,
            use_bias: bool = False,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ffn_expansion_factor = ffn_expansion_factor
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        # Will be built in build()
        self.blocks = []

    def build(self, input_shape):
        """Build the reasoning blocks."""
        # Calculate intermediate size from expansion factor
        intermediate_size = self.embed_dim * self.ffn_expansion_factor

        # Create reasoning blocks
        self.blocks = []
        for i in range(self.num_layers):
            block = HierarchicalReasoningBlock(
                # Use correct argument names for HierarchicalReasoningBlock
                hidden_size=self.embed_dim,  # Changed from embed_dim
                num_heads=self.num_heads,
                intermediate_size=intermediate_size,  # Changed from ffn_expansion_factor
                dropout_rate=self.dropout_rate,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f"block_{i}"
            )
            self.blocks.append(block)

        super().build(input_shape)

    def call(self, hidden_states, input_injection, training=None, mask=None):
        """
        Forward pass through the reasoning module.

        Args:
            hidden_states: Current hidden states (batch_size, seq_len, embed_dim)
            input_injection: Input to inject (batch_size, seq_len, embed_dim)
            training: Whether in training mode
            mask: Optional attention mask

        Returns:
            Updated hidden states of same shape
        """
        # Input injection (addition)
        x = hidden_states + input_injection

        # Process through reasoning blocks
        for block in self.blocks:
            x = block(x, training=training, mask=mask)

        return x

    def compute_output_shape(self, input_shape):
        """Compute output shape (unchanged from hidden_states input)."""
        if isinstance(input_shape, list):
            return input_shape[0]  # Return shape of hidden_states
        return input_shape

    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "num_layers": self.num_layers,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ffn_expansion_factor": self.ffn_expansion_factor,
            "dropout_rate": self.dropout_rate,
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config

    def get_build_config(self):
        """Get build configuration."""
        return {"input_shape": getattr(self, "_build_input_shape", None)}

    def build_from_config(self, config):
        """Build from configuration."""
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

# ---------------------------------------------------------------------