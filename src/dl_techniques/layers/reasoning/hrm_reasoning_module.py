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
from typing import Optional, Union, Tuple, Any, Dict, List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .hrm_block import HierarchicalReasoningBlock

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class HierarchicalReasoningModule(keras.layers.Layer):
    """
    Multi-layer hierarchical reasoning module with input injection for deep neural processing.

    This composite layer implements a stack of hierarchical reasoning blocks that enables
    progressive refinement of hidden state representations. It features an input injection
    mechanism that allows fresh information to be integrated at the beginning of each
    reasoning cycle, making it particularly suitable for hierarchical reasoning tasks.

    **Intent**: Provide a modular, stackable reasoning component for the Hierarchical
    Reasoning Model (HRM) architecture, enabling both low-level and high-level processing
    modules with consistent interfaces and progressive information refinement.

    **Architecture**:
    ```
    hidden_states + input_injection
           ↓
    HierarchicalReasoningBlock₁
           ↓
    HierarchicalReasoningBlock₂
           ↓
          ...
           ↓
    HierarchicalReasoningBlockₙ
           ↓
    refined_hidden_states
    ```

    **Data Flow**:
    1. **Input Injection**: Element-wise addition of hidden_states and input_injection
    2. **Sequential Processing**: Pass through num_layers reasoning blocks
    3. **Progressive Refinement**: Each block refines the representation
    4. **Output**: Final refined hidden states with same shape as input

    **Usage in HRM**:
    - **Low-Level Module**: Processes detailed representations with raw input injection
    - **High-Level Module**: Processes abstract representations with low-level state injection
    - **Modularity**: Clean separation between different reasoning levels

    Args:
        num_layers: Integer, number of hierarchical reasoning blocks in the stack.
            Must be positive. Controls depth of reasoning process.
        embed_dim: Integer, embedding dimension for all transformations.
            Must be positive and divisible by num_heads.
        num_heads: Integer, number of attention heads in each reasoning block.
            Must be positive and divide evenly into embed_dim. Defaults to 8.
        ffn_expansion_factor: Integer, expansion factor for feed-forward network.
            Hidden FFN dimension = embed_dim * ffn_expansion_factor.
            Must be positive. Defaults to 4.
        dropout_rate: Float between 0 and 1, dropout rate applied in attention and FFN.
            Used for regularization during training. Defaults to 0.0.
        use_bias: Boolean, whether to use bias terms in linear transformations.
            Defaults to False for better gradient flow in deep networks.
        kernel_initializer: String or Initializer, weight initialization method.
            Defaults to 'he_normal' for ReLU-like activations.
        kernel_regularizer: Optional Regularizer, L1/L2 regularization for weights.
            Helps prevent overfitting. Defaults to None.
        **kwargs: Additional Layer base class arguments.

    Input shape:
        Two tensors of identical shape:
        - hidden_states: 3D tensor `(batch_size, sequence_length, embed_dim)`
        - input_injection: 3D tensor `(batch_size, sequence_length, embed_dim)`

    Output shape:
        3D tensor with shape: `(batch_size, sequence_length, embed_dim)`.
        Same shape as input hidden_states.

    Attributes:
        blocks: List of HierarchicalReasoningBlock instances for sequential processing.

    Example:
        ```python
        # Low-level reasoning module
        low_level_module = HierarchicalReasoningModule(
            num_layers=6,
            embed_dim=768,
            num_heads=12,
            ffn_expansion_factor=4,
            dropout_rate=0.1
        )

        # High-level reasoning module (fewer layers, same dimensions)
        high_level_module = HierarchicalReasoningModule(
            num_layers=3,
            embed_dim=768,
            num_heads=12,
            ffn_expansion_factor=4,
            dropout_rate=0.1
        )

        # Usage in HRM forward pass
        hidden_states = keras.random.normal((batch_size, seq_len, 768))
        input_embeddings = keras.random.normal((batch_size, seq_len, 768))

        # Low-level processing with input injection
        low_refined = low_level_module(hidden_states, input_embeddings)

        # High-level processing with low-level state injection
        high_refined = high_level_module(hidden_states, low_refined)
        ```

    Note:
        This implementation follows the composite layer pattern with explicit sub-layer
        building for robust serialization. The input injection mechanism is crucial
        for the HRM architecture's ability to maintain information flow between
        reasoning levels.
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
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
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
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")

        # Store configuration
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ffn_expansion_factor = ffn_expansion_factor
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        # CREATE all sub-layers in __init__ (they are unbuilt)
        # Calculate intermediate size from expansion factor
        intermediate_size = self.embed_dim * self.ffn_expansion_factor

        self.blocks: List[HierarchicalReasoningBlock] = []
        for i in range(self.num_layers):
            block = HierarchicalReasoningBlock(
                hidden_size=self.embed_dim,  # Use correct parameter name
                num_heads=self.num_heads,
                intermediate_size=intermediate_size,
                dropout_rate=self.dropout_rate,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f"block_{i}"
            )
            self.blocks.append(block)

    def build(self, input_shape: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]) -> None:
        """
        Build the module and all its sub-layers.

        CRITICAL: Explicitly build each reasoning block for robust serialization.

        Args:
            input_shape: Shape of input tensors. Can be single shape tuple or list of two shapes
                         for [hidden_states, input_injection].
        """
        # Handle both single shape and list of shapes
        if isinstance(input_shape, list):
            # Expect two inputs: [hidden_states_shape, input_injection_shape]
            if len(input_shape) != 2:
                raise ValueError(f"Expected 2 input shapes, got {len(input_shape)}")
            hidden_states_shape = input_shape[0]
            input_injection_shape = input_shape[1]

            # Validate shapes match
            if hidden_states_shape != input_injection_shape:
                raise ValueError(
                    f"hidden_states shape {hidden_states_shape} must match "
                    f"input_injection shape {input_injection_shape}"
                )
            working_shape = hidden_states_shape
        else:
            # Single shape provided
            working_shape = input_shape

        # Validate final dimension matches embed_dim
        if working_shape[-1] != self.embed_dim:
            raise ValueError(
                f"Input embedding dimension ({working_shape[-1]}) must match "
                f"embed_dim ({self.embed_dim})"
            )

        # Build sub-layers explicitly for serialization
        for block in self.blocks:
            block.build(working_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
        self,
        inputs: Union[keras.KerasTensor, List[keras.KerasTensor]],
        training: Optional[bool] = None,
        mask: Optional[keras.KerasTensor] = None
    ) -> keras.KerasTensor:
        """
        Forward pass through the reasoning module with input injection.

        Args:
            inputs: Either a list of [hidden_states, input_injection] or a single tensor
                    if called with separate arguments (backward compatibility).
            training: Boolean indicating training mode.
            mask: Optional attention mask for transformer blocks.

        Returns:
            Refined hidden states with same shape as input hidden_states.
        """
        # Handle different input formats for flexibility
        if isinstance(inputs, list):
            if len(inputs) != 2:
                raise ValueError(f"Expected 2 inputs, got {len(inputs)}")
            hidden_states, input_injection = inputs
        else:
            # Backward compatibility: assume first argument is hidden_states
            # and input_injection is passed separately
            raise ValueError(
                "HierarchicalReasoningModule requires two inputs: [hidden_states, input_injection]. "
                "Please pass as a list: layer([hidden_states, input_injection])"
            )

        # Input injection (element-wise addition)
        x = hidden_states + input_injection

        # Process through reasoning blocks sequentially
        for block in self.blocks:
            x = block(x, training=training, mask=mask)

        return x

    def compute_output_shape(self, input_shape: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]) -> Tuple[Optional[int], ...]:
        """
        Compute output shape (same as hidden_states input shape).

        Args:
            input_shape: Input shape(s).

        Returns:
            Output shape tuple, same as hidden_states input.
        """
        if isinstance(input_shape, list):
            return input_shape[0]  # Return shape of hidden_states
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration for serialization.

        Returns:
            Dictionary containing all initialization parameters.
        """
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

# ---------------------------------------------------------------------