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
        - **Attention Type:** 'group_query' (with ``num_kv_heads == num_heads``,
          i.e. arithmetically plain multi-head attention, chosen because it is
          the only plain self-attention type that also applies RoPE to Q and K)

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

from ..transformers import (
    TransformerLayer,
    AttentionType,
    NormalizationType,
    NormalizationPositionType,
    FFNType
)

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class HierarchicalReasoningModule(keras.layers.Layer):
    """
    Configurable multi-layer reasoning module with input injection.

    Implements a stack of ``TransformerLayer`` instances with an input injection
    mechanism: ``x = hidden_states + input_injection`` is processed sequentially
    through ``num_layers`` transformer blocks. Defaults to a high-performance
    configuration for hierarchical reasoning (post-norm RMSNorm, SwiGLU FFN,
    grouped-query attention with ``num_kv_heads == num_heads`` and RoPE) but is
    fully configurable.

    **Architecture Overview:**

    .. code-block:: text

        ┌────────────────────────────────────────────────┐
        │       HierarchicalReasoningModule              │
        │                                                │
        │  [hidden_states, input_injection]              │
        │         │               │                      │
        │         └──── + ────────┘                      │
        │              │                                 │
        │              ▼                                 │
        │  ┌────────────────────────┐                    │
        │  │ TransformerLayer_0     │                    │
        │  └───────────┬────────────┘                    │
        │              ▼                                 │
        │  ┌────────────────────────┐                    │
        │  │ TransformerLayer_1     │                    │
        │  └───────────┬────────────┘                    │
        │              ▼                                 │
        │            ...                                 │
        │              ▼                                 │
        │  ┌────────────────────────┐                    │
        │  │ TransformerLayer_N     │                    │
        │  └───────────┬────────────┘                    │
        │              ▼                                 │
        │  Output(batch, seq_len, embed_dim)             │
        └────────────────────────────────────────────────┘

    :param num_layers: Number of transformer layers in the stack.
    :type num_layers: int
    :param embed_dim: Embedding dimension.
    :type embed_dim: int
    :param num_heads: Number of attention heads. Defaults to 8.
    :type num_heads: int
    :param ffn_expansion_factor: FFN expansion factor. Defaults to 4.
    :type ffn_expansion_factor: int
    :param attention_type: Attention mechanism type. Defaults to
        ``'group_query'``, the only plain-self-attention registry type that also
        carries RoPE; ``'multi_head'`` carries none.
    :type attention_type: AttentionType
    :param max_seq_len: Maximum sequence length the attention RoPE tables cover.
        Only consumed when ``attention_type`` is RoPE-capable. Defaults to 2048.
    :type max_seq_len: int
    :param rope_theta: Theta parameter for the attention's rotary position
        embedding. Only consumed when ``attention_type`` is RoPE-capable.
        Defaults to 10000.0.
    :type rope_theta: float
    :param normalization_type: Normalization layer type. Defaults to ``'rms_norm'``.
    :type normalization_type: NormalizationType
    :param normalization_position: Norm position (``'pre'`` or ``'post'``). Defaults to ``'post'``.
    :type normalization_position: NormalizationPositionType
    :param ffn_type: Feed-forward network type. Defaults to ``'swiglu'``.
    :type ffn_type: FFNType
    :param dropout_rate: Dropout rate. Defaults to 0.0.
    :type dropout_rate: float
    :param use_bias: Whether to use bias terms. Defaults to False.
    :type use_bias: bool
    :param kernel_initializer: Initializer for weights. Defaults to ``"he_normal"``.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for weights.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional Layer base class arguments.
    :type kwargs: Any
    """

    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        num_heads: int = 8,
        ffn_expansion_factor: int = 4,
        # DECISION plan-2026-08-17T183311-79c63e38/D-012: the default is
        # 'group_query' with `num_kv_heads == num_heads` (arithmetically plain
        # MHA) because it is the only registry entry reachable from
        # `TransformerLayer` that gives plain self-attention AND carries RoPE.
        #
        # WHAT NOT TO DO: do NOT "simplify" this back to 'multi_head'. RoPE is a
        # per-Q/K rotation applied INSIDE attention, and `MultiHeadAttention`
        # declares no RoPE parameter at all. `HierarchicalReasoningCore` used to
        # express its `pos_encodings='rope'` setting by constructing its OWN
        # `RotaryPositionEmbedding` and building it — while every attention layer
        # under it ran rope-free, because the core never handed that layer a Q or
        # a K tensor. Nothing raised, nothing was dropped by the attention
        # factory (HRM never populated `attention_args` at all, so even the
        # strict factory of D-011 could not surface it), and the whole reasoning
        # stack was exactly permutation-equivariant. MEASURED on CPU by
        # `tests/test_models/test_hierarchical_reasoning_model/test_positional_signal.py::TestHRMIsPositionAware::test_reasoning_stack_is_not_permutation_equivariant`:
        # `max|P f(x) - f(P x)| = 2.02656e-06` (float32 noise) before this
        # change. Same defect and same fix as TRM's D-007, ModernBERT's D-007
        # and DINOv3's D-010. See decisions.md D-012.
        attention_type: AttentionType = 'group_query',
        max_seq_len: int = 2048,
        rope_theta: float = 10000.0,
        normalization_type: NormalizationType = 'rms_norm',
        normalization_position: NormalizationPositionType = 'post',
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
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        if rope_theta <= 0:
            raise ValueError(f"rope_theta must be positive, got {rope_theta}")

        # Store all configuration parameters for serialization
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ffn_expansion_factor = ffn_expansion_factor
        self.attention_type = attention_type
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.normalization_type = normalization_type
        self.normalization_position = normalization_position
        self.ffn_type = ffn_type
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        intermediate_size = self.embed_dim * self.ffn_expansion_factor

        # RoPE arguments are forwarded ONLY to a type that declares all three.
        # The attention factory raises on an undeclared key since D-011, so
        # handing `max_seq_len`/`rope_theta` to e.g. 'multi_head' (which has no
        # RoPE at all) would be a hard construction failure rather than the
        # silent drop it used to be. 'multi_head_latent' is deliberately NOT
        # included: it is RoPE-capable but has no `num_kv_heads`.
        attention_args = None
        if self.attention_type == 'group_query':
            attention_args = {
                'num_kv_heads': self.num_heads,
                'max_seq_len': self.max_seq_len,
                'rope_theta': self.rope_theta,
            }

        # Create a list of TransformerLayer instances based on the provided configuration
        self.blocks: List[TransformerLayer] = []
        for i in range(self.num_layers):
            block = TransformerLayer(
                hidden_size=self.embed_dim,
                num_heads=self.num_heads,
                intermediate_size=intermediate_size,
                # Pass configurable parameters directly to the TransformerLayer
                attention_type=self.attention_type,
                attention_args=attention_args,
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
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass with input injection and sequential refinement.

        :param inputs: List of ``[hidden_states, input_injection]``.
        :type inputs: List[keras.KerasTensor]
        :param attention_mask: Optional attention mask for all layers.
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: Whether in training mode.
        :type training: Optional[bool]
        :return: Refined hidden states tensor.
        :rtype: keras.KerasTensor
        """
        if not isinstance(inputs, list) or len(inputs) != 2:
            raise ValueError("Input must be a list of two tensors: [hidden_states, input_injection]")
        hidden_states, input_injection = inputs

        x = hidden_states + input_injection

        for block in self.blocks:
            x = block(x, attention_mask=attention_mask, training=training)

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
            "max_seq_len": self.max_seq_len,
            "rope_theta": self.rope_theta,
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
