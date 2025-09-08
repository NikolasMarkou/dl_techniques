"""
Multi-Modal Fusion Layer: A Unified Framework for Cross-Modal Information Integration

This module implements a comprehensive, factory-based multi-modal fusion layer that addresses
the fundamental challenge of combining information from heterogeneous data modalities
(e.g., vision, text, audio) in deep learning architectures. The layer provides eight distinct
fusion strategies, each with different computational properties and theoretical foundations,
enabling researchers and practitioners to experiment with various approaches to multi-modal
learning within a single, consistent interface.

## Theoretical Foundation

Multi-modal fusion addresses the core challenge of learning joint representations from
different data modalities that may have vastly different statistical properties, dimensionalities,
and semantic structures. The key insight is that different fusion strategies capture different
types of cross-modal interactions:

**Early vs. Late Fusion**: Concatenation represents early fusion (combining raw features),
while attention-based methods represent late fusion (combining processed representations).

**Additive vs. Multiplicative Interactions**: Addition captures linear relationships between
modalities, while multiplication and bilinear methods capture higher-order interactions.

**Symmetric vs. Asymmetric Fusion**: Cross-attention allows asymmetric information flow
(one modality attending to another), while element-wise operations are symmetric.

## Mathematical Formulations

Given modalities X₁ ∈ ℝᵇˣˢ¹ˣᵈ, X₂ ∈ ℝᵇˣˢ²ˣᵈ, ..., Xₙ ∈ ℝᵇˣˢⁿˣᵈ where b=batch_size,
s=sequence_length, d=embed_dim, the fusion strategies implement:

**Cross-Attention**:
    Yᵢ = Xᵢ + ∑ⱼ≠ᵢ Attention(Xᵢ, Xⱼ, Xⱼ) + FFN(LayerNorm(Yᵢ))
    Enables bidirectional information exchange with learned attention weights.

**Concatenation**:
    Y = W[X₁ ∘ X₂ ∘ ... ∘ Xₙ] + b
    Simple but effective for preserving all modal information.

**Gated Fusion**:
    Y = W[σ(G₁(X₁)) ⊙ X₁ ∘ σ(G₂(X₂)) ⊙ X₂ ∘ ... ∘ σ(Gₙ(Xₙ)) ⊙ Xₙ] + b
    Where σ=sigmoid, G=gate networks, ⊙=element-wise multiplication.

**Bilinear Pooling**:
    Y = W(X₁ ⊗ X₂) + b where ⊗ denotes outer product
    Captures all pairwise feature interactions but computationally expensive.

**Tensor Fusion**:
    Y = W_final([W₁(concat(X)) ∘ W₂(concat(X)) ∘ ... ∘ Wₖ(concat(X))]) + b
    Multi-head projection approach balancing expressiveness and efficiency.

## Key Design Principles

1. **Factory-Based Architecture**: Uses dependency injection through factories for attention,
   FFN, and normalization components, enabling easy experimentation and extension.

2. **Strategy Pattern**: Each fusion method is implemented as a separate strategy, allowing
   runtime selection and easy addition of new approaches.

3. **Modality Agnostic**: Works with any number of modalities (≥2) as long as they share
   the same embedding dimension.

4. **Serialization-First Design**: Full Keras 3 compatibility with proper save/load support
   for production deployment.

## Usage Guidelines and Strategy Selection

**Cross-Attention**: Best for scenarios where modalities should dynamically attend to each
other (e.g., image captioning, visual question answering). Computationally expensive but
highly expressive. Use when you need bidirectional information flow and have sufficient
computational resources.

**Concatenation**: Simplest and often most effective baseline. Use when modalities are
complementary rather than redundant, and when you want to preserve all information.
Memory usage scales linearly with number of modalities.

**Addition/Multiplication**: Efficient for scenarios where modalities have similar semantic
structure. Addition works well when modalities provide independent evidence for the same
concepts. Multiplication emphasizes agreement between modalities.

**Gated Fusion**: Use when you need learned modality weighting - the model can learn to
emphasize or suppress different modalities based on input. Particularly useful when
modality reliability varies across samples.

**Bilinear Pooling**: Captures rich interactions but expensive (O(d²) complexity). Use only
for small embedding dimensions or when pairwise feature interactions are crucial.

**Tensor Fusion**: Good balance between expressiveness and efficiency. Use when you need
richer representations than concatenation but want better computational properties than
full bilinear pooling.

**Attention Pooling**: Use when you need fixed-size representations from variable-length
sequences across modalities, such as in retrieval or classification tasks.

## References

Core multi-modal fusion concepts:
- Baltrusaitis, T., Ahuja, C., & Morency, L. P. (2018). "Multimodal machine learning: A survey and taxonomy"
- Ramachandram, D., & Taylor, G. W. (2017). "Deep multimodal learning: A survey on recent advances and trends"

Attention mechanisms:
- Vaswani, A., et al. (2017). "Attention is all you need" (Transformer architecture)
- Luong, M. T., et al. (2015). "Effective approaches to attention-based neural machine translation"

Specific fusion techniques:
- Gao, Y., et al. (2016). "Compact bilinear pooling for visual question answering and visual grounding"
- Liu, Z., et al. (2018). "Efficient low-rank multimodal fusion with modality-specific factors"
- Zadeh, A., et al. (2017). "Tensor fusion network for multimodal sentiment analysis"

Vision-language applications:
- Anderson, P., et al. (2018). "Bottom-up and top-down attention for image captioning and VQA"
- Lu, J., et al. (2019). "ViLBERT: Pretraining task-agnostic visiolinguistic representations"
- Li, L. H., et al. (2020). "Oscar: Object-semantics aligned pre-training for vision-language tasks"

## Examples

### Basic Cross-Modal Attention (Vision-Language)
```python
# Bidirectional fusion between vision and text features
fusion = MultiModalFusion(
    embed_dim=768,
    fusion_strategy='cross_attention',
    num_fusion_layers=6,          # Deep interaction
    attention_type='multi_head',
    num_heads=12,
    dropout_rate=0.1
)

# Forward pass with vision and text embeddings
vision_features = keras.ops.random.normal((32, 196, 768))  # 32 samples, 196 patches
text_features = keras.ops.random.normal((32, 128, 768))    # 32 samples, 128 tokens
fused_vision, fused_text = fusion([vision_features, text_features])
```

### Efficient Gated Fusion for Resource-Constrained Settings
```python
# Lightweight fusion with learned modality weighting
fusion = MultiModalFusion(
    embed_dim=384,
    fusion_strategy='gated',
    ffn_type='swiglu',           # Modern activation
    norm_type='rms_norm',        # Efficient normalization
    dropout_rate=0.1
)

# Works with any number of modalities
audio_features = keras.ops.random.normal((16, 100, 384))
text_features = keras.ops.random.normal((16, 64, 384))
vision_features = keras.ops.random.normal((16, 49, 384))
fused = fusion([audio_features, text_features, vision_features])
```

### Multi-Head Projection Fusion for Rich Representations
```python
# Advanced tensor fusion with multiple projection heads
fusion = MultiModalFusion(
    embed_dim=1024,
    fusion_strategy='tensor_fusion',
    num_tensor_projections=8,    # Multiple projection heads
    activation='gelu',
    dropout_rate=0.1
)

# Captures complex cross-modal interactions
modality1 = keras.ops.random.normal((8, 256, 1024))
modality2 = keras.ops.random.normal((8, 128, 1024))
rich_fusion = fusion([modality1, modality2])
```

### Attention Pooling for Fixed-Size Representations
```python
# Convert variable-length sequences to fixed representations
fusion = MultiModalFusion(
    embed_dim=512,
    fusion_strategy='attention_pooling',
    attention_type='differential',  # Advanced attention variant
    num_heads=8
)

# Different sequence lengths -> fixed output size
short_seq = keras.ops.random.normal((4, 32, 512))
long_seq = keras.ops.random.normal((4, 256, 512))
pooled = fusion([short_seq, long_seq])  # Shape: (4, 512)
```

### Bilinear Fusion for Rich Pairwise Interactions
```python
# Captures all pairwise feature interactions (expensive but expressive)
fusion = MultiModalFusion(
    embed_dim=256,               # Smaller dim due to O(d²) complexity
    fusion_strategy='bilinear',
    activation='relu'
)

# Only supports exactly 2 modalities
x1 = keras.ops.random.normal((16, 64, 256))
x2 = keras.ops.random.normal((16, 48, 256))
bilinear_features = fusion([x1, x2])
```

### Integration in Custom Architectures
```python
class MultiModalTransformer(keras.Model):
    def __init__(self, embed_dim=768, **kwargs):
        super().__init__(**kwargs)

        # Multiple fusion strategies in sequence
        self.early_fusion = MultiModalFusion(
            embed_dim=embed_dim,
            fusion_strategy='concatenation'
        )

        self.deep_fusion = MultiModalFusion(
            embed_dim=embed_dim,
            fusion_strategy='cross_attention',
            num_fusion_layers=4,
            attention_type='group_query',  # Efficient attention
            attention_config={'n_kv_head': 4}
        )

        self.final_fusion = MultiModalFusion(
            embed_dim=embed_dim,
            fusion_strategy='gated'
        )

    def call(self, vision, text, training=None):
        # Early fusion
        early = self.early_fusion([vision, text], training=training)

        # Deep bidirectional fusion
        v_deep, t_deep = self.deep_fusion([vision, text], training=training)

        # Final adaptive fusion
        final = self.final_fusion([v_deep, t_deep], training=training)
        return final
```

"""

import keras
from keras import layers, initializers, regularizers, activations
from typing import Optional, Union, List, Dict, Any, Tuple, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .ffn.factory import create_ffn_layer
from .norms.factory import create_normalization_layer
from .attention.factory import create_attention_layer

# ---------------------------------------------------------------------

# Type definitions for enhanced type safety
FusionStrategy = Literal[
    'cross_attention',  # Bidirectional cross-attention between modalities
    'concatenation',  # Concatenate and project
    'addition',  # Element-wise addition with optional projection
    'multiplication',  # Element-wise multiplication with optional projection
    'gated',  # Learned gating mechanism
    'attention_pooling',  # Attention-based pooling and fusion
    'bilinear',  # Bilinear pooling
    'tensor_fusion'  # Multi-dimensional tensor fusion
]

AttentionType = Literal[
    'multi_head', 'shared_weights_cross', 'perceiver', 'differential',
    'group_query', 'window', 'adaptive_multi_head'
]

FFNType = Literal[
    'mlp', 'swiglu', 'differential', 'glu', 'geglu', 'residual', 'swin_mlp'
]

NormType = Literal[
    'layer_norm', 'rms_norm', 'batch_norm', 'band_rms', 'adaptive_band_rms',
    'dynamic_tanh', 'logit_norm'
]

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class MultiModalFusion(keras.layers.Layer):
    """
    General-purpose configurable multi-modal fusion layer with multiple fusion strategies.

    This layer enables flexible fusion between multiple modalities using various strategies
    including cross-attention, concatenation, gating, and tensor fusion. It uses factory
    patterns for creating configurable attention, FFN, and normalization sublayers.

    **Intent**: Provide a unified, configurable interface for multi-modal fusion that can
    adapt to different architectural requirements and fusion strategies while maintaining
    proper serialization and following modern Keras 3 patterns.

    **Supported Fusion Strategies**:
    - `cross_attention`: Bidirectional cross-attention between modalities
    - `concatenation`: Concatenate features and apply projection
    - `addition`: Element-wise addition with optional alignment projection
    - `multiplication`: Element-wise multiplication with optional alignment
    - `gated`: Learned gating mechanism for selective fusion
    - `attention_pooling`: Attention-based pooling and fusion
    - `bilinear`: Bilinear pooling between modalities
    - `tensor_fusion`: Multi-dimensional tensor fusion using multiple projection heads.

    Args:
        embed_dim: Integer, embedding dimension for all modalities.
        fusion_strategy: FusionStrategy, the fusion method to use.
        num_fusion_layers: Integer, number of fusion layers. Only applies to iterative
            strategies like 'cross_attention'.
        attention_type: AttentionType, type of attention mechanism for attention-based strategies.
        attention_config: Optional dictionary of attention-specific parameters.
        ffn_type: FFNType, type of feed-forward network.
        ffn_config: Optional dictionary of FFN-specific parameters.
        norm_type: NormType, type of normalization layer.
        norm_config: Optional dictionary of normalization-specific parameters.
        num_heads: Integer, number of attention heads for attention-based strategies.
        num_tensor_projections: Integer, number of projection heads for the 'tensor_fusion' strategy.
        intermediate_size: Integer, intermediate size for FFN layers.
        dropout_rate: Float, dropout rate for regularization.
        use_residual: Boolean, whether to use residual connections.
        activation: String or callable, activation function.
        kernel_initializer: String or initializer, kernel weight initializer.
        bias_initializer: String or initializer, bias weight initializer.
        kernel_regularizer: Optional regularizer for kernel weights.
        bias_regularizer: Optional regularizer for bias weights.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        Tuple of tensors: (modality1, modality2, [modality3, ...])
        Each modality tensor has shape: `(batch_size, sequence_length, embed_dim)`

    Output shape:
        Tuple of tensors with same shapes as inputs (for most strategies)
        or single tensor for pooling strategies.

    Example:
        ```python
        # Cross-attention fusion
        fusion = MultiModalFusion(
            embed_dim=768,
            fusion_strategy='cross_attention',
            num_fusion_layers=3,
            attention_type='multi_head',
            num_heads=12
        )

        # Gated fusion with custom configurations
        fusion = MultiModalFusion(
            embed_dim=512,
            fusion_strategy='gated',
            ffn_type='swiglu',
            ffn_config={'ffn_expansion_factor': 4},
            norm_type='rms_norm',
            norm_config={'epsilon': 1e-6}
        )

        # Multi-head projection fusion ('tensor_fusion')
        fusion = MultiModalFusion(
            embed_dim=768,
            fusion_strategy='tensor_fusion',
            num_tensor_projections=6
        )
        ```
    """

    def __init__(
            self,
            embed_dim: int = 768,
            fusion_strategy: FusionStrategy = 'cross_attention',
            num_fusion_layers: int = 1,
            attention_type: AttentionType = 'multi_head',
            attention_config: Optional[Dict[str, Any]] = None,
            ffn_type: FFNType = 'mlp',
            ffn_config: Optional[Dict[str, Any]] = None,
            norm_type: NormType = 'layer_norm',
            norm_config: Optional[Dict[str, Any]] = None,
            num_heads: int = 12,
            num_tensor_projections: int = 8,
            intermediate_size: Optional[int] = None,
            dropout_rate: float = 0.1,
            use_residual: bool = True,
            activation: Union[str, keras.KerasTensor] = 'gelu',
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if num_fusion_layers <= 0:
            raise ValueError(f"num_fusion_layers must be positive, got {num_fusion_layers}")
        if num_heads <= 0 or embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")
        if fusion_strategy == 'tensor_fusion' and num_tensor_projections <= 0:
            raise ValueError("num_tensor_projections must be positive for tensor_fusion strategy.")

        # Store configuration
        self.embed_dim = embed_dim
        self.fusion_strategy = fusion_strategy
        self.num_fusion_layers = num_fusion_layers
        self.attention_type = attention_type
        self.attention_config = attention_config or {}
        self.ffn_type = ffn_type
        self.ffn_config = ffn_config or {}
        self.norm_type = norm_type
        self.norm_config = norm_config or {}
        self.num_heads = num_heads
        self.num_tensor_projections = num_tensor_projections
        self.intermediate_size = intermediate_size or (embed_dim * 4)
        self.dropout_rate = dropout_rate
        self.use_residual = use_residual
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        # Validate that num_fusion_layers is used correctly
        iterative_strategies = {'cross_attention'}
        if self.fusion_strategy not in iterative_strategies and self.num_fusion_layers > 1:
            raise ValueError(
                f"num_fusion_layers > 1 is only supported for iterative strategies "
                f"like {iterative_strategies}, but strategy is '{self.fusion_strategy}'."
            )

        # Initialize sublayer containers
        self.fusion_layers = []
        self.projection_layers = []
        self.norm_layers = []
        self.ffn_layers = []
        self.gate_layers = []
        self.dropout_layers = []

        # Store build input shape for serialization
        self._build_input_shape = None

    def build(self, input_shape: Union[Tuple, List]) -> None:
        """Build the multi-modal fusion layers."""
        if self.built:
            return

        self._build_input_shape = input_shape

        # Validate input shapes
        if not isinstance(input_shape, (tuple, list)):
            raise ValueError(f"Expected tuple or list of shapes, got: {type(input_shape)}")

        if len(input_shape) < 2:
            raise ValueError(f"Expected at least 2 modalities, got {len(input_shape)}")

        # Validate all modalities have same embedding dimension
        for i, shape in enumerate(input_shape):
            if len(shape) != 3:
                raise ValueError(f"Expected 3D shapes, got modality {i}: {shape}")
            if shape[-1] != self.embed_dim:
                raise ValueError(f"Modality {i} dimension {shape[-1]} != embed_dim {self.embed_dim}")

        num_modalities = len(input_shape)

        # Build strategy-specific layers
        if self.fusion_strategy == 'cross_attention':
            self._build_cross_attention_layers(input_shape, num_modalities)
        elif self.fusion_strategy == 'concatenation':
            self._build_concatenation_layers(input_shape, num_modalities)
        elif self.fusion_strategy in ['addition', 'multiplication']:
            self._build_elementwise_layers(input_shape, num_modalities)
        elif self.fusion_strategy == 'gated':
            self._build_gated_layers(input_shape, num_modalities)
        elif self.fusion_strategy == 'attention_pooling':
            self._build_attention_pooling_layers(input_shape, num_modalities)
        elif self.fusion_strategy == 'bilinear':
            self._build_bilinear_layers(input_shape, num_modalities)
        elif self.fusion_strategy == 'tensor_fusion':
            self._build_tensor_fusion_layers(input_shape, num_modalities)
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")

        super().build(input_shape)

    def _build_cross_attention_layers(
            self,
            input_shape: Union[Tuple, List],
            num_modalities: int
    ) -> None:
        """Build cross-attention fusion layers."""
        for layer_idx in range(self.num_fusion_layers):
            layer_dict = {}

            # Create attention layers between each pair of modalities
            for i in range(num_modalities):
                layer_dict[f'attention_{i}'] = []
                layer_dict[f'norm_{i}'] = []
                layer_dict[f'ffn_{i}'] = []

                for j in range(num_modalities):
                    if i != j:  # Cross-attention between different modalities
                        attn_config = self.attention_config.copy()
                        attn_config.update({
                            'embed_dim': self.embed_dim,
                            'num_heads': self.num_heads,
                            'dropout_rate': self.dropout_rate
                        })

                        attention_layer = create_attention_layer(
                            self.attention_type,
                            name=f'cross_attention_{layer_idx}_{i}_{j}',
                            **attn_config
                        )
                        layer_dict[f'attention_{i}'].append(attention_layer)

                        # Build attention layer using the unified helper
                        self._build_attention_layer(
                            attention_layer=attention_layer,
                            query_shape=input_shape[i],
                            context_shape=input_shape[j]
                        )

                # Create normalization layer
                norm_config = self.norm_config.copy()
                norm_layer = create_normalization_layer(
                    self.norm_type,
                    name=f'norm_{layer_idx}_{i}',
                    **norm_config
                )
                layer_dict[f'norm_{i}'] = norm_layer
                norm_layer.build(input_shape[i])

                # Create FFN layer
                ffn_config = self.ffn_config.copy()
                ffn_config.update({
                    'hidden_dim': self.intermediate_size,
                    'output_dim': self.embed_dim,
                    'dropout_rate': self.dropout_rate
                })

                ffn_layer = create_ffn_layer(
                    self.ffn_type,
                    name=f'ffn_{layer_idx}_{i}',
                    **ffn_config
                )
                layer_dict[f'ffn_{i}'] = ffn_layer
                ffn_layer.build(input_shape[i])

            self.fusion_layers.append(layer_dict)

    def _build_attention_layer(
            self,
            attention_layer: keras.layers.Layer,
            query_shape: Tuple[Optional[int], ...],
            context_shape: Tuple[Optional[int], ...]
    ) -> None:
        """
        Unified interface for building attention layers with different signatures.

        This method abstracts away the differences between attention layer build interfaces.
        """
        # This assumes the factory returns layers compatible with these build patterns.
        # A more advanced factory could return a wrapper to unify interfaces.
        if hasattr(attention_layer, 'build_from_signature'):  # Keras MHA
            attention_layer.build(
                query_shape=query_shape,
                value_shape=context_shape
            )
        else:  # Assumes a common (query, context) build signature
            attention_layer.build((query_shape, context_shape))

    def _build_concatenation_layers(
            self,
            input_shape: Union[Tuple, List],
            num_modalities: int
    ) -> None:
        """Build concatenation fusion layers."""
        concat_dim = self.embed_dim * num_modalities

        projection = layers.Dense(
            self.embed_dim,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='concat_projection'
        )
        concat_shape = list(input_shape[0])
        concat_shape[-1] = concat_dim
        projection.build(tuple(concat_shape))
        self.projection_layers.append(projection)

        norm_layer = create_normalization_layer(
            self.norm_type, name='concat_norm', **self.norm_config.copy()
        )
        output_shape = tuple(concat_shape[:-1] + [self.embed_dim])
        norm_layer.build(output_shape)
        self.norm_layers.append(norm_layer)

        self.dropout_layers.append(layers.Dropout(self.dropout_rate, name='concat_dropout'))

    def _build_elementwise_layers(
            self,
            input_shape: Union[Tuple, List],
            num_modalities: int
    ) -> None:
        """Build element-wise operation (addition/multiplication) layers."""
        if num_modalities > 2:
            for i in range(num_modalities):
                proj = layers.Dense(
                    self.embed_dim,
                    kernel_initializer=self.kernel_initializer,
                    bias_initializer=self.bias_initializer,
                    name=f'align_projection_{i}'
                )
                proj.build(input_shape[i])
                self.projection_layers.append(proj)

        norm_layer = create_normalization_layer(
            self.norm_type, name='elementwise_norm', **self.norm_config.copy()
        )
        norm_layer.build(input_shape[0])
        self.norm_layers.append(norm_layer)

        ffn_config = self.ffn_config.copy()
        ffn_config.update({
            'hidden_dim': self.intermediate_size,
            'output_dim': self.embed_dim,
            'dropout_rate': self.dropout_rate
        })
        ffn_layer = create_ffn_layer(
            self.ffn_type, name='elementwise_ffn', **ffn_config
        )
        ffn_layer.build(input_shape[0])
        self.ffn_layers.append(ffn_layer)

    def _build_gated_layers(
            self,
            input_shape: Union[Tuple, List],
            num_modalities: int
    ) -> None:
        """Build gated fusion layers."""
        for i in range(num_modalities):
            gate = layers.Dense(
                self.embed_dim, activation='sigmoid', name=f'gate_{i}',
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer
            )
            gate.build(input_shape[i])
            self.gate_layers.append(gate)

        fusion_input_dim = self.embed_dim * num_modalities
        projection = layers.Dense(
            self.embed_dim, activation=self.activation, name='gated_projection',
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer
        )
        fusion_shape = list(input_shape[0])
        fusion_shape[-1] = fusion_input_dim
        projection.build(tuple(fusion_shape))
        self.projection_layers.append(projection)

        norm_layer = create_normalization_layer(
            self.norm_type, name='gated_norm', **self.norm_config.copy()
        )
        output_shape = tuple(fusion_shape[:-1] + [self.embed_dim])
        norm_layer.build(output_shape)
        self.norm_layers.append(norm_layer)

    def _build_attention_pooling_layers(
            self,
            input_shape: Union[Tuple, List],
            num_modalities: int
    ) -> None:
        """Build attention pooling fusion layers."""
        for i in range(num_modalities):
            attn_config = self.attention_config.copy()
            attn_config.update({
                'embed_dim': self.embed_dim,
                'num_heads': self.num_heads,
                'dropout_rate': self.dropout_rate
            })
            attention_layer = create_attention_layer(
                self.attention_type, name=f'pool_attention_{i}', **attn_config
            )
            attention_layer.build(input_shape[i])
            self.fusion_layers.append(attention_layer)

        fusion_dim = self.embed_dim * num_modalities
        projection = layers.Dense(
            self.embed_dim, activation=self.activation, name='pool_projection',
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer
        )
        pool_shape = (input_shape[0][0], fusion_dim)
        projection.build(pool_shape)
        self.projection_layers.append(projection)

    def _build_bilinear_layers(
            self,
            input_shape: Union[Tuple, List],
            num_modalities: int
    ) -> None:
        """Build bilinear pooling layers."""
        if num_modalities != 2:
            raise ValueError("Bilinear fusion currently supports only 2 modalities")

        bilinear_dim = self.embed_dim * self.embed_dim
        projection = layers.Dense(
            self.embed_dim, activation=self.activation, name='bilinear_projection',
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer
        )
        bilinear_shape = (input_shape[0][0], input_shape[0][1], bilinear_dim)
        projection.build(bilinear_shape)
        self.projection_layers.append(projection)

        norm_layer = create_normalization_layer(
            self.norm_type, name='bilinear_norm', **self.norm_config.copy()
        )
        output_shape = (input_shape[0][0], input_shape[0][1], self.embed_dim)
        norm_layer.build(output_shape)
        self.norm_layers.append(norm_layer)

    def _build_tensor_fusion_layers(
            self,
            input_shape: Union[Tuple, List],
            num_modalities: int
    ) -> None:
        """Build tensor fusion layers."""
        total_dim = self.embed_dim * num_modalities

        for i in range(self.num_tensor_projections):
            proj = layers.Dense(
                self.embed_dim, activation=self.activation, name=f'tensor_proj_{i}',
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer
            )
            concat_shape = list(input_shape[0])
            concat_shape[-1] = total_dim
            proj.build(tuple(concat_shape))
            self.projection_layers.append(proj)

        final_proj = layers.Dense(
            self.embed_dim, name='tensor_final_proj',
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer
        )
        fusion_shape = list(input_shape[0])
        fusion_shape[-1] = self.embed_dim * self.num_tensor_projections
        final_proj.build(tuple(fusion_shape))
        self.projection_layers.append(final_proj)

    def call(
            self,
            inputs: Union[List[keras.KerasTensor], Tuple[keras.KerasTensor, ...]],
            training: Optional[bool] = None
    ) -> Union[keras.KerasTensor, Tuple[keras.KerasTensor, ...]]:
        """Forward pass of the multi-modal fusion layer."""
        if not isinstance(inputs, (list, tuple)):
            raise ValueError("Expected list or tuple of input tensors")
        if len(inputs) < 2:
            raise ValueError("Expected at least 2 input modalities")

        # Route to appropriate fusion method
        handler = getattr(self, f'_call_{self.fusion_strategy}', None)
        if handler is None:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")
        return handler(inputs, training)

    def _call_cross_attention(
            self,
            inputs: Union[List[keras.KerasTensor], Tuple[keras.KerasTensor, ...]],
            training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, ...]:
        """Cross-attention fusion implementation."""
        outputs = list(inputs)
        num_modalities = len(inputs)

        for layer_idx in range(self.num_fusion_layers):
            layer_dict = self.fusion_layers[layer_idx]
            new_outputs = []

            for i in range(num_modalities):
                attended_features = []
                attention_idx = 0
                for j in range(num_modalities):
                    if i != j:
                        attention_layer = layer_dict[f'attention_{i}'][attention_idx]
                        attended = attention_layer(query=outputs[i], value=outputs[j], training=training)
                        attended_features.append(attended)
                        attention_idx += 1

                if attended_features:
                    combined = keras.ops.mean(keras.ops.stack(attended_features, axis=0), axis=0)
                else:
                    combined = outputs[i]

                if self.use_residual:
                    combined = outputs[i] + combined

                norm_out = layer_dict[f'norm_{i}'](combined, training=training)

                ffn_out = layer_dict[f'ffn_{i}'](norm_out, training=training)
                if self.use_residual:
                    combined = norm_out + ffn_out
                else:
                    combined = ffn_out
                new_outputs.append(combined)

            outputs = new_outputs

        return tuple(outputs)

    def _call_concatenation(
            self,
            inputs: Union[List[keras.KerasTensor], Tuple[keras.KerasTensor, ...]],
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Concatenation fusion implementation."""
        concatenated = keras.ops.concatenate(inputs, axis=-1)
        output = self.projection_layers[0](concatenated)
        output = self.norm_layers[0](output, training=training)
        output = self.dropout_layers[0](output, training=training)
        return output

    def _call_elementwise(
            self,
            inputs: Union[List[keras.KerasTensor], Tuple[keras.KerasTensor, ...]],
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Element-wise operation fusion implementation."""
        aligned = [
            proj(inp) for proj, inp in zip(self.projection_layers, inputs)
        ] if self.projection_layers else list(inputs)

        if self.fusion_strategy == 'addition':
            output = keras.ops.sum(keras.ops.stack(aligned, axis=0), axis=0)
        else:  # multiplication
            output = aligned[0]
            for inp in aligned[1:]:
                output = keras.ops.multiply(output, inp)

        output = self.norm_layers[0](output, training=training)
        output = self.ffn_layers[0](output, training=training)
        return output

    def _call_gated(
            self,
            inputs: Union[List[keras.KerasTensor], Tuple[keras.KerasTensor, ...]],
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Gated fusion implementation."""
        gated_features = [
            keras.ops.multiply(inp, self.gate_layers[i](inp, training=training))
            for i, inp in enumerate(inputs)
        ]
        concatenated = keras.ops.concatenate(gated_features, axis=-1)
        output = self.projection_layers[0](concatenated)
        output = self.norm_layers[0](output, training=training)
        return output

    def _call_attention_pooling(
            self,
            inputs: Union[List[keras.KerasTensor], Tuple[keras.KerasTensor, ...]],
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Attention pooling fusion implementation."""
        pooled_features = [
            keras.ops.mean(self.fusion_layers[i](inp, training=training), axis=1)
            for i, inp in enumerate(inputs)
        ]
        concatenated = keras.ops.concatenate(pooled_features, axis=-1)
        output = self.projection_layers[0](concatenated)
        return output

    def _call_bilinear(
            self,
            inputs: Union[List[keras.KerasTensor], Tuple[keras.KerasTensor, ...]],
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Bilinear pooling fusion implementation."""
        if len(inputs) != 2:
            raise ValueError("Bilinear fusion requires exactly 2 inputs")
        x1, x2 = inputs

        x1_expanded = keras.ops.expand_dims(x1, axis=-1)
        x2_expanded = keras.ops.expand_dims(x2, axis=-2)
        bilinear = keras.ops.multiply(x1_expanded, x2_expanded)

        batch_size, seq_len = keras.ops.shape(bilinear)[:2]
        bilinear_flat = keras.ops.reshape(bilinear, [batch_size, seq_len, -1])

        output = self.projection_layers[0](bilinear_flat)
        output = self.norm_layers[0](output, training=training)
        return output

    def _call_tensor_fusion(
            self,
            inputs: Union[List[keras.KerasTensor], Tuple[keras.KerasTensor, ...]],
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Tensor fusion implementation."""
        concatenated = keras.ops.concatenate(inputs, axis=-1)

        projections = [
            self.projection_layers[i](concatenated)
            for i in range(self.num_tensor_projections)
        ]

        combined = keras.ops.concatenate(projections, axis=-1) if len(projections) > 1 else projections[0]
        output = self.projection_layers[-1](combined)
        return output

    def compute_output_shape(
            self,
            input_shape: Union[Tuple, List]
    ) -> Union[Tuple, List]:
        """Compute output shape based on fusion strategy."""
        if self.fusion_strategy == 'cross_attention':
            return input_shape
        elif self.fusion_strategy == 'attention_pooling':
            return (input_shape[0][0], self.embed_dim)

        # All other strategies return a single tensor with the same sequence shape
        return input_shape[0]

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'fusion_strategy': self.fusion_strategy,
            'num_fusion_layers': self.num_fusion_layers,
            'attention_type': self.attention_type,
            'attention_config': self.attention_config,
            'ffn_type': self.ffn_type,
            'ffn_config': self.ffn_config,
            'norm_type': self.norm_type,
            'norm_config': self.norm_config,
            'num_heads': self.num_heads,
            'num_tensor_projections': self.num_tensor_projections,
            'intermediate_size': self.intermediate_size,
            'dropout_rate': self.dropout_rate,
            'use_residual': self.use_residual,
            'activation': keras.activations.serialize(self.activation),
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

# ---------------------------------------------------------------------
