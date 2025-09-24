"""Encapsulates a configurable, general-purpose Vision Transformer encoder.

This layer implements the core architecture of a Vision Transformer (ViT),
which processes images by treating them as a sequence of flattened patches. It
provides a flexible and modular framework that can be configured to replicate
various ViT-style architectures, serving as a unified backbone for a wide
range of computer vision_heads tasks.

Architectural and Mathematical Underpinnings:

The fundamental innovation of the Vision Transformer is the application of the
highly successful Transformer architecture, originally designed for natural
language processing, to image data. This is achieved through a specific
sequence of transformations that convert a 2D grid of pixels into a 1D sequence
of vectors that the Transformer can process.

1.  **Patchification and Embedding**: An input image `I ∈ ℝ^(H×W×C)` is first
    divided into a grid of `N` non-overlapping patches, where each patch
    `pᵢ ∈ ℝ^(P×P×C)`. These 2D patches are then flattened into vectors and
    linearly projected into a `D`-dimensional embedding space via a learnable
    weight matrix `E`. This is the critical step that transforms spatial data
    into a sequence format.

        `z₀ = [x_class; E*p₁; E*p₂; ...; E*p_N] + E_pos`

    -   **`[x_class]` Token**: Inspired by BERT, a learnable `[CLS]` (class)
        token embedding is prepended to the sequence of patch embeddings. The
        final state of this token after passing through the encoder serves as
        the aggregate image representation for classification tasks.
    -   **Positional Embeddings `E_pos`**: Since the self-attention mechanism is
        permutation-invariant, explicit positional information must be added to
        the patch embeddings to retain their spatial arrangement. These are
        learnable embeddings, one for each position in the sequence.

2.  **Transformer Encoder Stack**: The resulting sequence of embeddings `z₀` is
    then processed by a stack of `L` identical Transformer layers. Each layer
    applies two main sub-layers:
    -   **Multi-Head Self-Attention (MHSA)**: This allows each patch embedding
        to be updated by attending to and integrating information from all
        other patch embeddings in the sequence. It enables the model to learn
        long-range dependencies and contextual relationships between different
        parts of the image.
    -   **Position-wise Feed-Forward Network (FFN)**: A simple MLP applied
        independently to each patch embedding, providing non-linearity and
        increasing representational capacity.

    Each sub-layer is enclosed in a residual connection and followed by layer
    normalization, ensuring stable training of deep models. The output of the
    final layer, `z_L`, is a sequence of contextually rich patch
    representations.

3.  **Factory-Based Design Philosophy**: This implementation is intentionally
    generic, utilizing a factory pattern for its core components (patch
    embedding, attention, normalization, FFN). This design choice allows the
    single `VisionEncoder` class to be configured to instantiate a wide variety
    of architectural variants from the literature (e.g., the standard ViT,
    SigLIP with its two-stage patch embedder, or efficient models using RMSNorm
    and SwiGLU). This flexibility supports rapid experimentation and architectural
    research within a unified and maintainable codebase.

References:
    - Dosovitskiy, A., et al. (2020). An Image is Worth 16x16 Words:
      Transformers for Image Recognition at Scale. *ICLR*.
    - Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS*.
    - Zhai, X., et al. (2023). Sigmoid Loss for Language Image Pre-Training.
      *ICCV*. (Introduced the SigLIP architecture and patch embedder).
"""

import keras
from keras import ops, layers, initializers, regularizers
from typing import Optional, Union, Tuple, Dict, Any, Literal, Callable

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .transformer import TransformerLayer
from .embedding import create_embedding_layer
from .norms import create_normalization_layer

# ---------------------------------------------------------------------
# Type definitions for enhanced type safety
# ---------------------------------------------------------------------

PatchEmbedType = Literal['linear', 'siglip', 'conv', 'hybrid']
AttentionType = Literal['multi_head_attention', 'window_attention', 'group_query_attention', 'differential_attention']
NormalizationType = Literal['layer_norm', 'rms_norm', 'batch_norm', 'band_rms', 'adaptive_band_rms', 'dynamic_tanh']
NormalizationPosition = Literal['pre', 'post']
FFNType = Literal['mlp', 'swiglu', 'differential', 'glu', 'geglu', 'residual', 'swin_mlp']
PoolingMode = Literal['cls', 'mean', 'max', 'none']


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class VisionEncoder(keras.layers.Layer):
    """
    General purpose configurable vision_heads encoder using factory-based components.

    This layer provides a highly flexible vision_heads encoder that can be configured to create
    various vision_heads transformer architectures. It uses factory patterns for all major
    components (patch embedding, attention, normalization, FFN) to enable easy
    experimentation and architectural exploration.

    **Intent**: Provide a single, configurable vision_heads encoder that can replace multiple
    specialized implementations by supporting different patch embedding strategies,
    attention mechanisms, normalization types, and feed-forward networks through
    factory-based component creation.

    **Architecture**:
    ```
    Input Images (batch, height, width, channels)
           ↓
    Configurable Patch Embedding:
      - 'linear': Single Conv2D layer
      - 'siglip': Two-stage Conv2D with normalization
      - 'conv': Multi-layer convolution
      - 'hybrid': CNN backbone + patch embedding
           ↓
    Reshape to Patches (batch, num_patches, embed_dim)
           ↓
    Add CLS Token → (batch, seq_len, embed_dim)
           ↓
    Positional Embedding + Dropout
           ↓
    TransformerLayer × depth (configurable components)
           ↓
    Optional Final Normalization
           ↓
    Output Features (configurable pooling)
    ```

    **Key Features**:
    - Factory-based component creation for maximum flexibility
    - Support for multiple patch embedding strategies
    - Configurable attention mechanisms (MHA, Window, GQA, etc.)
    - Multiple normalization options (LayerNorm, RMSNorm, etc.)
    - Various FFN architectures (MLP, SwiGLU, etc.)
    - Flexible output modes for different downstream tasks

    Args:
        img_size: Integer, input image size. Must be positive and divisible by patch_size.
            Defaults to 224.
        patch_size: Integer, size of image patches. Must be positive and divide img_size.
            Defaults to 16.
        embed_dim: Integer, embedding dimension. Must be positive. Defaults to 768.
        depth: Integer, number of transformer blocks. Must be positive. Defaults to 12.
        num_heads: Integer, number of attention heads. Must be positive and divide embed_dim.
            Defaults to 12.
        mlp_ratio: Float, MLP expansion ratio. Must be positive. Defaults to 4.0.
        patch_embed_type: PatchEmbedType, patch embedding strategy:
            - 'linear': Standard single convolution (ViT-style)
            - 'siglip': Two-stage convolution with normalization (SigLIP-style)
            - 'conv': Multi-layer convolution with non-linearity
            - 'hybrid': CNN backbone followed by patch embedding
            Defaults to 'linear'.
        attention_type: AttentionType, attention mechanism to use:
            - 'multi_head_attention': Standard multi-head self-attention
            - 'window_attention': Windowed attention for efficiency
            - 'group_query_attention': Grouped query attention
            - 'differential_attention': Differential attention for noise reduction
            Defaults to 'multi_head_attention'.
        normalization_type: NormalizationType, normalization layer type:
            - 'layer_norm': Standard layer normalization
            - 'rms_norm': Root mean square normalization
            - 'band_rms': Band-constrained RMS normalization
            - 'dynamic_tanh': Dynamic Tanh normalization
            Defaults to 'layer_norm'.
        normalization_position: NormalizationPosition, normalization position:
            - 'post': Post-normalization (original Transformer)
            - 'pre': Pre-normalization (often more stable)
            Defaults to 'post'.
        ffn_type: FFNType, feed-forward network architecture:
            - 'mlp': Standard MLP with intermediate expansion
            - 'swiglu': SwiGLU activation with gating mechanism
            - 'differential': Differential FFN with separate pathways
            - 'geglu': GELU-based Gated Linear Unit
            Defaults to 'mlp'.
        use_cls_token: Boolean, whether to add a CLS token for classification.
            When True, adds learnable CLS token at sequence start. Defaults to True.
        output_mode: PoolingMode, output mode for feature extraction:
            - 'cls': Return CLS token features (requires use_cls_token=True)
            - 'mean': Global average pooling over sequence
            - 'max': Global max pooling over sequence
            - 'none': Return full sequence features
            Defaults to 'cls'.
        dropout: Float, general dropout rate between 0 and 1. Defaults to 0.0.
        attention_dropout: Float, attention-specific dropout rate. Defaults to 0.0.
        pos_dropout: Float, positional embedding dropout rate. Defaults to 0.0.
        stochastic_depth_rate: Float, stochastic depth drop path rate. Defaults to 0.0.
        activation: Union[str, Callable], activation function for FFN. Defaults to 'gelu'.
        use_bias: Boolean, whether to use bias in linear layers. Defaults to True.
        kernel_initializer: Union[str, Initializer], initializer for kernel weights.
            Defaults to 'glorot_uniform'.
        bias_initializer: Union[str, Initializer], initializer for bias weights.
            Defaults to 'zeros'.
        kernel_regularizer: Optional[Regularizer], regularizer for kernel weights.
            Defaults to None.
        bias_regularizer: Optional[Regularizer], regularizer for bias weights.
            Defaults to None.
        attention_args: Optional[Dict[str, Any]], custom arguments for attention layer.
            These override default parameters for the specific attention type.
            Defaults to None.
        norm_args: Optional[Dict[str, Any]], custom arguments for normalization layers.
            Applied to all normalization layers in the encoder. Defaults to None.
        ffn_args: Optional[Dict[str, Any]], custom arguments for FFN layers.
            These override default parameters for the FFN type. Defaults to None.
        patch_embed_args: Optional[Dict[str, Any]], custom arguments for patch embedding.
            Type-specific parameters for patch embedding layer. Defaults to None.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, channels)`
        - height and width should equal img_size
        - channels is typically 3 for RGB images

    Output shape:
        Depends on output_mode:
        - 'cls': `(batch_size, embed_dim)` - CLS token features
        - 'mean': `(batch_size, embed_dim)` - Mean-pooled features
        - 'max': `(batch_size, embed_dim)` - Max-pooled features
        - 'none': `(batch_size, seq_len, embed_dim)` - Full sequence

    Attributes:
        num_patches: Integer, number of image patches.
        seq_len: Integer, sequence length including CLS token if used.
        patch_embed: Patch embedding layer created by factory.
        pos_embed: Positional embedding layer.
        transformer_layers: List of TransformerLayer instances.
        norm: Optional final normalization layer.
        cls_token: Optional learnable CLS token weight.

    Example:
        ```python
        # Standard ViT-Base configuration
        encoder = VisionEncoder(
            img_size=224,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            patch_embed_type='linear',
            attention_type='multi_head_attention'
        )

        # SigLIP-style with modern components
        siglip_encoder = VisionEncoder(
            img_size=224,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            patch_embed_type='siglip',
            attention_type='multi_head_attention',
            normalization_type='layer_norm',
            ffn_type='mlp'
        )

        # Efficient configuration with advanced components
        efficient_encoder = VisionEncoder(
            img_size=224,
            embed_dim=384,
            depth=12,
            num_heads=6,
            patch_embed_type='conv',
            attention_type='window_attention',
            normalization_type='rms_norm',
            normalization_position='pre',
            ffn_type='swiglu',
            attention_args={'window_size': 7},
            norm_args={'epsilon': 1e-6}
        )

        # Feature extraction mode
        feature_encoder = VisionEncoder(
            img_size=384,
            embed_dim=768,
            depth=6,
            output_mode='mean',
            dropout=0.1
        )
        ```

    Note:
        All components are created using factory functions to ensure consistency
        and enable easy experimentation with different architectural choices.
        The encoder follows modern Keras 3 patterns for robust serialization.

    Raises:
        ValueError: If img_size is not divisible by patch_size.
        ValueError: If embed_dim is not divisible by num_heads.
        ValueError: If any dimension parameter is not positive.
        ValueError: If use_cls_token=False but output_mode='cls'.
    """

    def __init__(
            self,
            img_size: int = 224,
            patch_size: int = 16,
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.0,
            patch_embed_type: PatchEmbedType = 'linear',
            attention_type: AttentionType = 'multi_head_attention',
            normalization_type: NormalizationType = 'layer_norm',
            normalization_position: NormalizationPosition = 'post',
            ffn_type: FFNType = 'mlp',
            use_cls_token: bool = True,
            output_mode: PoolingMode = 'cls',
            dropout: float = 0.0,
            attention_dropout: float = 0.0,
            pos_dropout: float = 0.0,
            stochastic_depth_rate: float = 0.0,
            activation: Union[str, Callable] = 'gelu',
            use_bias: bool = True,
            kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            bias_regularizer: Optional[regularizers.Regularizer] = None,
            attention_args: Optional[Dict[str, Any]] = None,
            norm_args: Optional[Dict[str, Any]] = None,
            ffn_args: Optional[Dict[str, Any]] = None,
            patch_embed_args: Optional[Dict[str, Any]] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if img_size <= 0:
            raise ValueError(f"img_size must be positive, got {img_size}")
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        if img_size % patch_size != 0:
            raise ValueError(
                f"img_size ({img_size}) must be divisible by patch_size ({patch_size})"
            )
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        if mlp_ratio <= 0.0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")
        if not (0.0 <= dropout <= 1.0):
            raise ValueError(f"dropout must be between 0 and 1, got {dropout}")
        if not (0.0 <= attention_dropout <= 1.0):
            raise ValueError(f"attention_dropout must be between 0 and 1, got {attention_dropout}")
        if not (0.0 <= pos_dropout <= 1.0):
            raise ValueError(f"pos_dropout must be between 0 and 1, got {pos_dropout}")
        if not use_cls_token and output_mode == 'cls':
            raise ValueError("output_mode='cls' requires use_cls_token=True")

        # Store ALL configuration parameters for serialization
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.patch_embed_type = patch_embed_type
        self.attention_type = attention_type
        self.normalization_type = normalization_type
        self.normalization_position = normalization_position
        self.ffn_type = ffn_type
        self.use_cls_token = use_cls_token
        self.output_mode = output_mode
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.pos_dropout = pos_dropout
        self.stochastic_depth_rate = stochastic_depth_rate
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.attention_args = attention_args or {}
        self.norm_args = norm_args or {}
        self.ffn_args = ffn_args or {}
        self.patch_embed_args = patch_embed_args or {}

        # Computed properties
        self.num_patches = (img_size // patch_size) ** 2
        self.seq_len = self.num_patches + (1 if use_cls_token else 0)
        self.intermediate_size = int(embed_dim * mlp_ratio)

        # CREATE all sub-layers in __init__ (modern Keras 3 pattern)

        # Create patch embedding using factory pattern
        self.patch_embed = self._create_patch_embedding()

        # Create positional embedding using factory
        self.pos_embed = create_embedding_layer(
            'positional_learned',
            max_seq_len=self.seq_len,
            dim=self.embed_dim,
            dropout=self.pos_dropout,
            name="pos_embed"
        )

        # Create transformer layers using factory components
        self.transformer_layers = []
        for i in range(self.depth):
            # Calculate stochastic depth rate (linearly increasing)
            layer_drop_rate = self.stochastic_depth_rate * i / max(1, self.depth - 1)

            layer = TransformerLayer(
                hidden_size=self.embed_dim,
                num_heads=self.num_heads,
                intermediate_size=self.intermediate_size,
                attention_type=self.attention_type,
                attention_args=self.attention_args,
                normalization_type=self.normalization_type,
                normalization_position=self.normalization_position,
                attention_norm_args=self.norm_args,
                ffn_norm_args=self.norm_args,
                ffn_type=self.ffn_type,
                ffn_args=self.ffn_args,
                dropout_rate=self.dropout,
                attention_dropout_rate=self.attention_dropout,
                use_stochastic_depth=self.stochastic_depth_rate > 0.0,
                stochastic_depth_rate=layer_drop_rate,
                activation=self.activation,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name=f"transformer_layer_{i}"
            )
            self.transformer_layers.append(layer)

        # Create final normalization layer (only for pre-norm)
        self.norm = None
        if self.normalization_position == 'pre':
            self.norm = create_normalization_layer(
                self.normalization_type,
                name="final_norm",
                **self.norm_args
            )

        # Create CLS token weight if needed (shape is independent of input)
        self.cls_token = None
        if self.use_cls_token:
            self.cls_token = self.add_weight(
                name="cls_token",
                shape=(1, 1, self.embed_dim),
                initializer="zeros",
                trainable=True,
            )

    def _create_patch_embedding(self) -> keras.layers.Layer:
        """
        Create patch embedding layer based on specified type.

        Returns:
            Patch embedding layer configured for the specified strategy.
        """
        base_args = {
            'kernel_initializer': self.kernel_initializer,
            'bias_initializer': self.bias_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'bias_regularizer': self.bias_regularizer,
            'use_bias': self.use_bias
        }
        base_args.update(self.patch_embed_args)

        if self.patch_embed_type == 'linear':
            # Standard ViT-style linear patch embedding
            return layers.Conv2D(
                filters=self.embed_dim,
                kernel_size=self.patch_size,
                strides=self.patch_size,
                padding='valid',
                name='patch_embed_linear',
                **base_args
            )

        elif self.patch_embed_type == 'siglip':
            # SigLIP-style two-stage patch embedding
            return keras.Sequential([
                # Stage 1: Coarse-grained patching
                layers.Conv2D(
                    filters=self.embed_dim // 2,
                    kernel_size=self.patch_size // 2,
                    strides=self.patch_size // 2,
                    padding='valid',
                    name='patch_embed_conv1',
                    **base_args
                ),
                create_normalization_layer(
                    self.normalization_type,
                    name='patch_embed_norm1',
                    **self.norm_args
                ),
                layers.Activation('gelu', name='patch_embed_activation1'),
                # Stage 2: Refinement to final embedding dimension
                layers.Conv2D(
                    filters=self.embed_dim,
                    kernel_size=2,
                    strides=2,
                    padding='valid',
                    name='patch_embed_conv2',
                    **base_args
                ),
            ], name='patch_embed_siglip')

        elif self.patch_embed_type == 'conv':
            # Multi-layer convolution patch embedding
            return keras.Sequential([
                layers.Conv2D(
                    filters=self.embed_dim // 4,
                    kernel_size=3,
                    strides=2,
                    padding='same',
                    name='patch_embed_conv1',
                    **base_args
                ),
                create_normalization_layer(
                    self.normalization_type,
                    name='patch_embed_norm1',
                    **self.norm_args
                ),
                layers.Activation(self.activation, name='patch_embed_act1'),
                layers.Conv2D(
                    filters=self.embed_dim // 2,
                    kernel_size=3,
                    strides=2,
                    padding='same',
                    name='patch_embed_conv2',
                    **base_args
                ),
                create_normalization_layer(
                    self.normalization_type,
                    name='patch_embed_norm2',
                    **self.norm_args
                ),
                layers.Activation(self.activation, name='patch_embed_act2'),
                layers.Conv2D(
                    filters=self.embed_dim,
                    kernel_size=self.patch_size // 4,
                    strides=self.patch_size // 4,
                    padding='valid',
                    name='patch_embed_conv3',
                    **base_args
                ),
            ], name='patch_embed_conv')

        else:  # hybrid
            # Hybrid CNN backbone + patch embedding (simplified)
            return keras.Sequential([
                # CNN backbone (simplified ResNet-like)
                layers.Conv2D(64, 7, strides=2, padding='same', name='hybrid_conv1', **base_args),
                create_normalization_layer(self.normalization_type, name='hybrid_norm1', **self.norm_args),
                layers.Activation(self.activation, name='hybrid_act1'),
                layers.MaxPooling2D(3, strides=2, padding='same', name='hybrid_pool1'),
                # Bottleneck
                layers.Conv2D(self.embed_dim // 2, 3, padding='same', name='hybrid_conv2', **base_args),
                create_normalization_layer(self.normalization_type, name='hybrid_norm2', **self.norm_args),
                layers.Activation(self.activation, name='hybrid_act2'),
                # Final patch embedding
                layers.Conv2D(
                    filters=self.embed_dim,
                    kernel_size=1,
                    strides=1,
                    padding='valid',
                    name='hybrid_patch_embed',
                    **base_args
                ),
            ], name='patch_embed_hybrid')

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the vision_heads encoder and all its sub-layers.

        This method validates input shape and explicitly builds sub-layers
        for robust serialization following modern Keras 3 patterns.

        Args:
            input_shape: Shape tuple of input images (batch, height, width, channels)

        Raises:
            ValueError: If input shape is invalid or incompatible with configuration.
        """
        if len(input_shape) != 4:
            raise ValueError(
                f"Expected 4D input shape (batch, height, width, channels), "
                f"got {input_shape}"
            )

        # CLS token is created in __init__ as its shape is independent of input.

        # Build patch embedding layer
        self.patch_embed.build(input_shape)

        # Build positional embedding
        pos_input_shape = (None, self.seq_len, self.embed_dim)
        self.pos_embed.build(pos_input_shape)

        # Build transformer layers
        for layer in self.transformer_layers:
            layer.build(pos_input_shape)

        # Build final normalization if present
        if self.norm is not None:
            self.norm.build(pos_input_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def _get_full_sequence_features(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Internal helper to run the full forward pass and return sequence features."""
        batch_size = ops.shape(inputs)[0]
        x = self.patch_embed(inputs, training=training)

        # Reshape to sequence format. Shape can vary by patch embedder.
        # Final shape should be (batch_size, num_patches, embed_dim)
        if len(x.shape) == 4:
            x = ops.reshape(x, [batch_size, -1, self.embed_dim])

        if self.use_cls_token:
            cls_tokens = ops.broadcast_to(self.cls_token, (batch_size, 1, self.embed_dim))
            x = ops.concatenate([cls_tokens, x], axis=1)

        x = self.pos_embed(x, training=training)

        for layer in self.transformer_layers:
            x = layer(x, training=training)

        if self.norm is not None:
            x = self.norm(x, training=training)

        return x

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass through the vision_heads encoder.

        Args:
            inputs: Input images tensor of shape [batch_size, height, width, channels]
            training: Optional boolean indicating training mode.

        Returns:
            Output tensor with shape depending on output_mode configuration.
        """
        x = self._get_full_sequence_features(inputs, training=training)

        # Apply output mode
        if self.output_mode == 'cls':
            return x[:, 0, :]
        elif self.output_mode == 'mean':
            # Exclude CLS token from mean pooling if it exists
            tokens_to_pool = x[:, 1:, :] if self.use_cls_token else x
            return ops.mean(tokens_to_pool, axis=1)
        elif self.output_mode == 'max':
            # Exclude CLS token from max pooling if it exists
            tokens_to_pool = x[:, 1:, :] if self.use_cls_token else x
            return ops.max(tokens_to_pool, axis=1)
        else:  # 'none'
            return x

    def get_cls_features(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Extract CLS token features for classification tasks.

        Args:
            inputs: Input images tensor.
            training: Optional boolean indicating training mode.

        Returns:
            CLS token features tensor of shape [batch_size, embed_dim].

        Raises:
            ValueError: If use_cls_token=False.
        """
        if not self.use_cls_token:
            raise ValueError("CLS token is not available when use_cls_token=False")

        features = self._get_full_sequence_features(inputs, training=training)
        return features[:, 0, :]

    def get_patch_features(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Extract patch token features for dense prediction tasks.

        Args:
            inputs: Input images tensor.
            training: Optional boolean indicating training mode.

        Returns:
            Patch features tensor of shape [batch_size, num_patches, embed_dim].
        """
        features = self._get_full_sequence_features(inputs, training=training)
        if self.use_cls_token:
            return features[:, 1:, :]
        else:
            return features

    def get_spatial_features(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Get spatial features reshaped for dense prediction tasks.

        Args:
            inputs: Input images tensor.
            training: Optional boolean indicating training mode.

        Returns:
            Spatial features tensor of shape [batch_size, patch_height, patch_width, embed_dim].
        """
        patch_features = self.get_patch_features(inputs, training=training)
        batch_size = ops.shape(patch_features)[0]

        patches_h = self.img_size // self.patch_size
        patches_w = self.img_size // self.patch_size

        return ops.reshape(
            patch_features,
            [batch_size, patches_h, patches_w, self.embed_dim]
        )

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute output shape given input shape.

        Args:
            input_shape: Input shape tuple (batch_size, height, width, channels)

        Returns:
            Output shape tuple based on output_mode configuration.
        """
        batch_size = input_shape[0]

        if self.output_mode in ['cls', 'mean', 'max']:
            return (batch_size, self.embed_dim)
        else:  # 'none'
            return (batch_size, self.seq_len, self.embed_dim)

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        Returns ALL parameters passed to __init__ for complete reconstruction.

        Returns:
            Dictionary containing all layer configuration parameters.
        """
        config = super().get_config()
        config.update({
            'img_size': self.img_size,
            'patch_size': self.patch_size,
            'embed_dim': self.embed_dim,
            'depth': self.depth,
            'num_heads': self.num_heads,
            'mlp_ratio': self.mlp_ratio,
            'patch_embed_type': self.patch_embed_type,
            'attention_type': self.attention_type,
            'normalization_type': self.normalization_type,
            'normalization_position': self.normalization_position,
            'ffn_type': self.ffn_type,
            'use_cls_token': self.use_cls_token,
            'output_mode': self.output_mode,
            'dropout': self.dropout,
            'attention_dropout': self.attention_dropout,
            'pos_dropout': self.pos_dropout,
            'stochastic_depth_rate': self.stochastic_depth_rate,
            'activation': self.activation,
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'attention_args': self.attention_args,
            'norm_args': self.norm_args,
            'ffn_args': self.ffn_args,
            'patch_embed_args': self.patch_embed_args,
        })
        return config


# ---------------------------------------------------------------------
# Factory Functions for Convenient Encoder Creation
# ---------------------------------------------------------------------


def create_vision_encoder(
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        patch_embed_type: PatchEmbedType = 'linear',
        attention_type: AttentionType = 'multi_head_attention',
        normalization_type: NormalizationType = 'layer_norm',
        normalization_position: NormalizationPosition = 'post',
        ffn_type: FFNType = 'mlp',
        use_cls_token: bool = True,
        output_mode: PoolingMode = 'cls',
        dropout: float = 0.0,
        **kwargs: Any
) -> VisionEncoder:
    """
    Factory function to create a VisionEncoder with validated parameters.

    This function provides parameter validation and sensible defaults for creating
    vision_heads encoders with different architectural configurations. It supports all
    major vision_heads transformer variants through configurable components.

    Args:
        img_size: Input image size. Must be divisible by patch_size.
        patch_size: Size of image patches.
        embed_dim: Embedding dimension.
        depth: Number of transformer layers.
        num_heads: Number of attention heads.
        mlp_ratio: MLP expansion ratio.
        patch_embed_type: Type of patch embedding strategy.
        attention_type: Type of attention mechanism.
        normalization_type: Type of normalization.
        normalization_position: Position of normalization layers.
        ffn_type: Type of feed-forward network.
        use_cls_token: Whether to use CLS token.
        output_mode: Output pooling mode.
        dropout: General dropout rate.
        **kwargs: Additional arguments for VisionEncoder constructor.

    Returns:
        Configured VisionEncoder instance.

    Raises:
        ValueError: If any parameter validation fails.

    Example:
        ```python
        # Standard ViT-Base
        encoder = create_vision_encoder(
            img_size=224,
            embed_dim=768,
            depth=12,
            num_heads=12
        )

        # Efficient encoder with modern components
        encoder = create_vision_encoder(
            img_size=224,
            embed_dim=384,
            depth=8,
            num_heads=6,
            patch_embed_type='siglip',
            attention_type='window_attention',
            normalization_type='rms_norm',
            ffn_type='swiglu'
        )
        ```
    """
    # Validate basic parameters
    if img_size <= 0 or patch_size <= 0:
        raise ValueError(f"img_size and patch_size must be positive, got {img_size}, {patch_size}")

    if img_size % patch_size != 0:
        raise ValueError(f"img_size ({img_size}) must be divisible by patch_size ({patch_size})")

    if embed_dim <= 0 or depth <= 0 or num_heads <= 0:
        raise ValueError("embed_dim, depth, and num_heads must be positive")

    if embed_dim % num_heads != 0:
        raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

    return VisionEncoder(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        patch_embed_type=patch_embed_type,
        attention_type=attention_type,
        normalization_type=normalization_type,
        normalization_position=normalization_position,
        ffn_type=ffn_type,
        use_cls_token=use_cls_token,
        output_mode=output_mode,
        dropout=dropout,
        **kwargs
    )

# ---------------------------------------------------------------------

def create_vit_encoder(
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        **kwargs: Any
) -> VisionEncoder:
    """Create standard ViT encoder configuration."""
    return create_vision_encoder(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        patch_embed_type='linear',
        attention_type='multi_head_attention',
        normalization_type='layer_norm',
        normalization_position='post',
        ffn_type='mlp',
        **kwargs
    )

# ---------------------------------------------------------------------

def create_siglip_encoder(
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        **kwargs: Any
) -> VisionEncoder:
    """Create SigLIP-style encoder configuration."""
    return create_vision_encoder(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        patch_embed_type='siglip',
        attention_type='multi_head_attention',
        normalization_type='layer_norm',
        normalization_position='post',
        ffn_type='mlp',
        **kwargs
    )

# ---------------------------------------------------------------------

def create_efficient_encoder(
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 384,
        depth: int = 8,
        num_heads: int = 6,
        **kwargs: Any
) -> VisionEncoder:
    """Create efficient encoder with modern components."""
    return create_vision_encoder(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        patch_embed_type='conv',
        attention_type='multi_head_attention',
        normalization_type='rms_norm',
        normalization_position='pre',
        ffn_type='swiglu',
        **kwargs
    )