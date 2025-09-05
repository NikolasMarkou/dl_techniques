"""
Gemma 3 270M Model Implementation for dl_techniques Framework

This module implements Google's Gemma 3 270M model using the dl_techniques framework,
following modern Keras 3 best practices and reusing framework components for maximum
compatibility and robustness.

Key architectural features:
- Dual RMSNorm: Pre and post normalization around attention/FFN blocks
- Sliding Window Attention: Mix of local (sliding window) and global (full) attention
- GELU-based GeGLU FFN: Using framework's GeGLU implementation
- Grouped Query Attention: With query-key normalization for stability
- Large vocabulary: ~262k tokens for multilingual coverage

IMPLEMENTATION NOTES:

1. **Framework Integration**:
   - Uses dl_techniques factories for attention, FFN, and normalization layers
   - Follows Modern Keras 3 Layer/Model patterns
   - Proper serialization and configuration handling

2. **Component Reuse**:
   - GroupedQueryAttention via attention factory
   - GeGLUFFN via FFN factory (GELU-based, matches Gemma 3)
   - RMSNorm via normalization factory
   - No custom implementations where framework components exist

3. **Architectural Fidelity**:
   - Maintains Gemma 3's dual normalization pattern
   - Supports mixed attention patterns (sliding window / full)
   - Proper embedding scaling and vocabulary handling
   - Configurable context length

References:
    - Gemma 3 Model Card: https://huggingface.co/google/gemma-3-270m
    - dl_techniques framework documentation
    - Modern Keras 3 Custom Layers and Models Guide
"""

import keras
from keras import ops, layers, initializers
from typing import Optional, Union, Tuple, List, Dict, Any, Literal

# Framework imports
from dl_techniques.utils.logger import logger
from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.layers.attention import create_attention_layer
from dl_techniques.layers.ffn import create_ffn_layer

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class Gemma3TransformerBlock(keras.layers.Layer):
    """
    Gemma 3 Transformer Block with dual normalization pattern using framework components.

    This block implements Gemma 3's specific dual normalization architecture:
    - Pre-normalization before attention and FFN
    - Post-normalization after attention and FFN outputs
    - Residual connections around both components
    - Support for both sliding window and full attention patterns

    Uses framework factories for all sub-components to ensure consistency,
    robustness, and maintainability while preserving Gemma 3's architecture.

    Args:
        emb_dim: Integer, embedding dimension.
        num_heads: Integer, number of attention heads.
        num_kv_groups: Integer, number of key-value groups for GQA.
        hidden_dim: Integer, hidden dimension for FFN.
        max_context_length: Integer, maximum sequence length for attention.
        attention_type: String, either 'sliding_attention' or 'full_attention'.
        use_bias: Boolean, whether to use bias in linear layers.
        dropout_rate: Float, dropout rate for FFN.
        kernel_initializer: Initializer for kernel weights.
        **kwargs: Additional Layer arguments.

    Input shape:
        3D tensor with shape: `(batch_size, sequence_length, emb_dim)`

    Output shape:
        3D tensor with shape: `(batch_size, sequence_length, emb_dim)`

    Example:
        ```python
        block = Gemma3TransformerBlock(
            emb_dim=640,
            num_heads=4,
            num_kv_groups=1,
            hidden_dim=2048,
            max_context_length=32768
        )

        inputs = keras.Input(shape=(128, 640))
        outputs = block(inputs)  # Shape: (batch, 128, 640)
        ```
    """

    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        num_kv_groups: int,
        hidden_dim: int,
        max_context_length: int = 32768,
        attention_type: Literal['sliding_attention', 'full_attention'] = 'full_attention',
        use_bias: bool = False,
        dropout_rate: float = 0.0,
        kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if emb_dim <= 0:
            raise ValueError(f"emb_dim must be positive, got {emb_dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if num_kv_groups <= 0:
            raise ValueError(f"num_kv_groups must be positive, got {num_kv_groups}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if max_context_length <= 0:
            raise ValueError(f"max_context_length must be positive, got {max_context_length}")

        # Store ALL configuration parameters
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.hidden_dim = hidden_dim
        self.max_context_length = max_context_length
        self.attention_type = attention_type
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        self.kernel_initializer = initializers.get(kernel_initializer)

        # CREATE all sub-layers using framework factories in __init__

        # Create dual normalization layers using framework factory
        self.input_layernorm = create_normalization_layer(
            'rms_norm',
            epsilon=1e-6,
            name='input_layernorm'
        )
        self.post_attention_layernorm = create_normalization_layer(
            'rms_norm',
            epsilon=1e-6,
            name='post_attention_layernorm'
        )
        self.pre_feedforward_layernorm = create_normalization_layer(
            'rms_norm',
            epsilon=1e-6,
            name='pre_feedforward_layernorm'
        )
        self.post_feedforward_layernorm = create_normalization_layer(
            'rms_norm',
            epsilon=1e-6,
            name='post_feedforward_layernorm'
        )

        # Create attention layer using framework factory
        self.attention = create_attention_layer(
            'group_query',
            d_model=emb_dim,
            n_head=num_heads,
            n_kv_head=num_kv_groups,
            max_seq_len=max_context_length,
            dropout_rate=dropout_rate,
            name='attention'
        )

        # Create FFN layer using framework factory - GeGLU for GELU-based gating
        self.ffn = create_ffn_layer(
            'geglu',
            hidden_dim=hidden_dim,
            output_dim=emb_dim,
            activation='gelu',  # Explicit GELU to match Gemma 3
            dropout_rate=dropout_rate,
            use_bias=use_bias,
            name='ffn'
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and all its sub-layers.

        CRITICAL: Explicitly build each sub-layer for robust serialization.
        """
        # Build normalization layers
        self.input_layernorm.build(input_shape)
        self.post_attention_layernorm.build(input_shape)
        self.pre_feedforward_layernorm.build(input_shape)
        self.post_feedforward_layernorm.build(input_shape)

        # Build attention layer
        self.attention.build(input_shape)

        # Build FFN layer
        self.ffn.build(input_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        shortcut = inputs

        # FIX: Remove training=training
        x = self.input_layernorm(inputs)

        # This is correct: attention layer has dropout
        if mask is not None:
            attn_output = self.attention(x, mask=mask, training=training)
        else:
            attn_output = self.attention(x, training=training)

        # FIX: Remove training=training
        attn_output = self.post_attention_layernorm(attn_output)

        x = shortcut + attn_output
        shortcut = x

        # FIX: Remove training=training
        x_ffn = self.pre_feedforward_layernorm(x)

        # This is correct: FFN layer has dropout
        ffn_output = self.ffn(x_ffn, training=training)

        # FIX: Remove training=training
        ffn_output = self.post_feedforward_layernorm(ffn_output)

        output = shortcut + ffn_output

        return output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape - same as input shape."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'emb_dim': self.emb_dim,
            'num_heads': self.num_heads,
            'num_kv_groups': self.num_kv_groups,
            'hidden_dim': self.hidden_dim,
            'max_context_length': self.max_context_length,
            'attention_type': self.attention_type,
            'use_bias': self.use_bias,
            'dropout_rate': self.dropout_rate,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
        })
        return config


@keras.saving.register_keras_serializable()
class Gemma3Model(keras.Model):
    """
    Gemma 3 270M Language Model implementation using dl_techniques framework.

    This model implements Google's Gemma 3 architecture with:
    - Token embeddings with square root scaling
    - Mixed attention pattern: sliding window and full attention layers
    - Dual RMSNorm pattern in transformer blocks
    - GELU-based GeGLU feed-forward networks
    - Grouped query attention for efficiency
    - Large vocabulary (262k tokens) for multilingual support
    - Configurable maximum context length

    Key Framework Integrations:
    - Uses attention factory for GroupedQueryAttention
    - Uses FFN factory for GeGLUFFN (GELU-based)
    - Uses normalization factory for RMSNorm
    - Follows Modern Keras 3 patterns for serialization and configuration

    Performance Characteristics:
    - ~270M parameters with default configuration
    - Memory efficient with mixed precision support
    - Configurable context length (default: 32768 tokens)
    - Compatible with all dl_techniques optimization and analysis tools

    Args:
        vocab_size: Integer, vocabulary size.
        emb_dim: Integer, embedding dimension.
        num_heads: Integer, number of attention heads.
        num_layers: Integer, number of transformer layers.
        hidden_dim: Integer, FFN hidden dimension.
        head_dim: Integer, dimension per attention head.
        num_kv_groups: Integer, number of key-value groups.
        max_context_length: Integer, maximum sequence length. Configurable.
        sliding_window: Integer, sliding window size for local attention.
        layer_types: List of strings, attention type per layer.
        query_pre_attn_scalar: Optional float, query scaling factor.
        use_bias: Boolean, whether to use bias in layers.
        dropout_rate: Float, dropout rate for regularization.
        **kwargs: Additional Model arguments.

    Input shape:
        2D tensor with shape: `(batch_size, sequence_length)`

    Output shape:
        3D tensor with shape: `(batch_size, sequence_length, vocab_size)`

    Example:
        ```python
        # Create Gemma 3 270M model with default settings
        model = create_gemma3_270m()

        # Create with custom context length
        model = create_gemma3_270m(max_context_length=8192)

        # Forward pass
        input_ids = keras.random.randint([2, 128], 0, model.vocab_size)
        logits = model(input_ids)
        print(f"Output shape: {logits.shape}")  # (2, 128, 262144)

        # For fine-tuning
        model.compile(
            optimizer='adamw',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        ```

    Note:
        This implementation balances architectural fidelity with framework best practices.
        Uses simplified RoPE handling (single configuration vs original dual RoPE)
        for better maintainability while preserving excellent performance.
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        num_heads: int,
        num_layers: int,
        hidden_dim: int,
        head_dim: int,
        num_kv_groups: int = 1,
        max_context_length: int = 32768,
        sliding_window: int = 512,
        layer_types: Optional[List[str]] = None,
        query_pre_attn_scalar: Optional[float] = None,
        use_bias: bool = False,
        dropout_rate: float = 0.0,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if emb_dim <= 0:
            raise ValueError(f"emb_dim must be positive, got {emb_dim}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if max_context_length <= 0:
            raise ValueError(f"max_context_length must be positive, got {max_context_length}")
        if layer_types is not None and len(layer_types) != num_layers:
            raise ValueError(f"layer_types length ({len(layer_types)}) must match num_layers ({num_layers})")

        # Store ALL configuration parameters
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.num_kv_groups = num_kv_groups
        self.max_context_length = max_context_length
        self.sliding_window = sliding_window
        self.layer_types = layer_types or ['full_attention'] * num_layers
        self.query_pre_attn_scalar = query_pre_attn_scalar
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate

        # CREATE all sub-layers in __init__ (Modern Keras 3 pattern)

        # Token embedding layer
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size,
            output_dim=emb_dim,
            name='token_embeddings'
        )

        # Transformer blocks - create all in __init__
        self.transformer_blocks = []
        for i, attention_type in enumerate(self.layer_types):
            block = Gemma3TransformerBlock(
                emb_dim=emb_dim,
                num_heads=num_heads,
                num_kv_groups=num_kv_groups,
                hidden_dim=hidden_dim,
                max_context_length=max_context_length,
                attention_type=attention_type,
                use_bias=use_bias,
                dropout_rate=dropout_rate,
                name=f'transformer_block_{i}'
            )
            self.transformer_blocks.append(block)

        # Final normalization and output head
        self.final_norm = create_normalization_layer(
            'rms_norm',
            epsilon=1e-6,
            name='final_norm'
        )
        self.output_head = layers.Dense(
            vocab_size,
            use_bias=use_bias,
            name='output_head'
        )

        # Embedding scaling factor (as in original Gemma 3)
        self.emb_scale = ops.sqrt(ops.cast(emb_dim, dtype='float32'))

    def _create_attention_masks(self, seq_len: int) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Create attention masks for global and local (sliding window) attention.

        Args:
            seq_len: Sequence length.

        Returns:
            Tuple of (global_mask, local_mask) tensors.
        """
        # Create indices which are safe for symbolic execution.
        # i represents row indices: [[0], [1], ..., [seq_len-1]]
        i = ops.arange(seq_len)[:, None]
        # j represents column indices: [[0, 1, ..., seq_len-1]]
        j = ops.arange(seq_len)

        # 1. Create global (causal) mask.
        # This replaces `ops.triu(ones, k=1)` with a robust equivalent.
        # A position (i, j) is masked if the column j is ahead of the row i.
        global_mask = j > i

        # 2. Create the "too-far-past" mask for the sliding window.
        # A position (i, j) is masked if the distance (i - j) is >= the window size.
        far_past_mask = (i - j) >= self.sliding_window

        # 3. Combine them for the final local attention mask.
        local_mask = ops.logical_or(global_mask, far_past_mask)

        return global_mask, local_mask

    def call(
        self,
        input_ids: keras.KerasTensor,
        training: Optional[bool] = None,
        return_logits: bool = True
    ) -> keras.KerasTensor:
        """
        Forward pass through Gemma 3 model.

        Args:
            input_ids: Integer tensor of shape (batch_size, seq_len).
            training: Training mode flag.
            return_logits: Whether to return logits (True) or hidden states (False).

        Returns:
            If return_logits=True: Logits tensor of shape (batch_size, seq_len, vocab_size).
            If return_logits=False: Hidden states of shape (batch_size, seq_len, emb_dim).
        """
        batch_size, seq_len = ops.shape(input_ids)[0], ops.shape(input_ids)[1]

        # Token embeddings with scaling (Gemma 3 pattern)
        x = self.token_embeddings(input_ids, training=training)
        x = x * self.emb_scale

        # Create attention masks for sliding window attention
        global_mask, local_mask = self._create_attention_masks(seq_len)

        # Apply transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            # Choose appropriate mask based on attention type
            attention_type = self.layer_types[i]
            mask = local_mask if attention_type == 'sliding_attention' else global_mask

            # Apply transformer block
            x = block(
                x,
                mask=mask,
                training=training
            )

        # Final normalization using framework component
        x = self.final_norm(x, training=training)

        if return_logits:
            # Apply output head to get logits
            return self.output_head(x, training=training)
        else:
            return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        batch_size, seq_len = input_shape
        return (batch_size, seq_len, self.vocab_size)

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'emb_dim': self.emb_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'hidden_dim': self.hidden_dim,
            'head_dim': self.head_dim,
            'num_kv_groups': self.num_kv_groups,
            'max_context_length': self.max_context_length,
            'sliding_window': self.sliding_window,
            'layer_types': self.layer_types,
            'query_pre_attn_scalar': self.query_pre_attn_scalar,
            'use_bias': self.use_bias,
            'dropout_rate': self.dropout_rate,
        })
        return config


def create_gemma3_270m(
    vocab_size: int = 262144,
    max_context_length: int = 32768,
    **kwargs: Any
) -> Gemma3Model:
    """
    Create Gemma 3 270M model with default configuration using framework components.

    This function provides a convenient way to create the Gemma 3 270M model with
    sensible defaults while leveraging the dl_techniques framework for robustness.

    Key Features:
    - Uses framework factories for attention, FFN, and normalization
    - GeGLU FFN with GELU activation (matches Gemma 3 architecture)
    - Grouped Query Attention for efficiency
    - RMSNorm for stability
    - Configurable context length
    - Mixed attention patterns (sliding window + full attention)

    Args:
        vocab_size: Vocabulary size. Defaults to 262144 (original Gemma 3).
        max_context_length: Maximum sequence length. Defaults to 32768.
        **kwargs: Additional configuration overrides for custom use cases.

    Returns:
        Configured Gemma3Model instance ready for training or inference.

    Example:
        ```python
        # Create model with default settings (270M parameters)
        model = create_gemma3_270m()

        # Create model with custom context length
        model = create_gemma3_270m(max_context_length=8192)

        # Create model with custom vocab size
        model = create_gemma3_270m(vocab_size=50000)

        # Create smaller model for experimentation
        model = create_gemma3_270m(
            vocab_size=10000,
            num_layers=12,
            emb_dim=512,
            max_context_length=4096
        )

        # Compile for training
        model.compile(
            optimizer='adamw',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Forward pass
        input_ids = keras.random.randint([2, 128], 0, model.vocab_size)
        logits = model(input_ids)  # Shape: (2, 128, vocab_size)
        ```

    Performance Notes:
        - ~270M parameters with default configuration
        - Memory efficient with mixed precision support
        - Optimized for sequences up to max_context_length tokens
        - Compatible with all dl_techniques framework tools
    """
    # Default Gemma 3 270M configuration
    default_config = {
        'vocab_size': vocab_size,
        'max_context_length': max_context_length,
        'emb_dim': 640,
        'num_heads': 4,
        'num_layers': 18,
        'hidden_dim': 2048,  # For GeGLU FFN
        'head_dim': 256,
        'num_kv_groups': 1,
        'sliding_window': 512,
        'query_pre_attn_scalar': 256.0,
        'use_bias': False,
        'dropout_rate': 0.0,
        'layer_types': [
            'sliding_attention', 'sliding_attention', 'sliding_attention',
            'sliding_attention', 'sliding_attention', 'full_attention',
            'sliding_attention', 'sliding_attention', 'sliding_attention',
            'sliding_attention', 'sliding_attention', 'full_attention',
            'sliding_attention', 'sliding_attention', 'sliding_attention',
            'sliding_attention', 'sliding_attention', 'full_attention'
        ]
    }

    # Override with any provided kwargs
    config = {**default_config, **kwargs}

    # CRITICAL: Adjust layer_types if num_layers was overridden
    if 'num_layers' in kwargs and kwargs['num_layers'] != default_config['num_layers']:
        new_num_layers = kwargs['num_layers']

        # If layer_types wasn't explicitly provided, generate appropriate pattern
        if 'layer_types' not in kwargs:
            config['layer_types'] = _generate_layer_types_pattern(new_num_layers)
        else:
            # If layer_types was provided, validate it matches num_layers
            if len(kwargs['layer_types']) != new_num_layers:
                raise ValueError(
                    f"Provided layer_types length ({len(kwargs['layer_types'])}) "
                    f"must match num_layers ({new_num_layers})"
                )

    # Adjust head_dim if emb_dim or num_heads were overridden
    if ('emb_dim' in kwargs or 'num_heads' in kwargs) and 'head_dim' not in kwargs:
        config['head_dim'] = config['emb_dim'] // config['num_heads']

    logger.info(f"Creating Gemma 3 270M model with framework components:")
    logger.info(f"  - Vocabulary size: {config['vocab_size']}")
    logger.info(f"  - Max context length: {config['max_context_length']}")
    logger.info(f"  - Embedding dimension: {config['emb_dim']}")
    logger.info(f"  - Number of layers: {config['num_layers']}")
    logger.info(f"  - Using framework factories for attention, FFN, and normalization")

    return Gemma3Model(**config)


def _generate_layer_types_pattern(num_layers: int) -> List[str]:
    """
    Generate appropriate layer types pattern for given number of layers.

    Uses Gemma 3's pattern: mostly sliding_attention with periodic full_attention layers.
    For smaller models, ensures at least one full_attention layer.

    Args:
        num_layers: Number of transformer layers.

    Returns:
        List of layer types ('sliding_attention' or 'full_attention').
    """
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}")

    if num_layers == 1:
        # Single layer should be full attention
        return ['full_attention']
    elif num_layers <= 3:
        # Small models: alternate or end with full attention
        if num_layers == 2:
            return ['sliding_attention', 'full_attention']
        else:  # num_layers == 3
            return ['sliding_attention', 'sliding_attention', 'full_attention']
    else:
        # Larger models: follow Gemma 3 pattern (full attention every ~6 layers)
        layer_types = []
        full_attention_interval = max(6, num_layers // 3)  # At least every 6 layers

        for i in range(num_layers):
            if (i + 1) % full_attention_interval == 0:
                layer_types.append('full_attention')
            else:
                layer_types.append('sliding_attention')

        # Ensure at least one full attention layer
        if 'full_attention' not in layer_types:
            layer_types[-1] = 'full_attention'

    return layer_types


# Model configuration constants for easy access
GEMMA3_270M_CONFIG = {
    'vocab_size': 262144,
    'max_context_length': 32768,
    'emb_dim': 640,
    'num_heads': 4,
    'num_layers': 18,
    'hidden_dim': 2048,
    'head_dim': 256,
    'num_kv_groups': 1,
    'sliding_window': 512,
    'query_pre_attn_scalar': 256.0,
    'use_bias': False,
    'dropout_rate': 0.0,
    'layer_types': [
        'sliding_attention', 'sliding_attention', 'sliding_attention',
        'sliding_attention', 'sliding_attention', 'full_attention',
        'sliding_attention', 'sliding_attention', 'sliding_attention',
        'sliding_attention', 'sliding_attention', 'full_attention',
        'sliding_attention', 'sliding_attention', 'sliding_attention',
        'sliding_attention', 'sliding_attention', 'full_attention'
    ]
}