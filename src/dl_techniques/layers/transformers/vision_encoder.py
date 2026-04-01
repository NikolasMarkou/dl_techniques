"""
A configurable, general-purpose Vision Transformer encoder.

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

from ..embedding import create_embedding_layer
from ..norms import create_normalization_layer
from .transformer import (
    TransformerLayer,
    NormalizationType,
    NormalizationPositionType,
    AttentionType,
    FFNType
)
from ..sequence_pooling import SequencePooling, PoolingStrategy

# ---------------------------------------------------------------------
# Type definitions for enhanced type safety
# ---------------------------------------------------------------------

PatchEmbedType = Literal['linear', 'siglip', 'conv', 'hybrid']

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class VisionEncoder(keras.layers.Layer):
    """
    General-purpose configurable vision encoder using factory-based components.

    Converts images into patch sequences, adds optional CLS token and
    positional embeddings, processes through a configurable TransformerLayer
    stack, and pools output features. Factory patterns allow replicating
    architectures from standard ViT to SigLIP, DeiT, and modern variants
    with RMSNorm + SwiGLU.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────────────┐
        │  Input Image (B, H, W, C)                │
        └──────────────────┬───────────────────────┘
                           ▼
        ┌──────────────────────────────────────────┐
        │  Patch Embedding (linear/siglip/conv)    │
        │  ─► Reshape (B, num_patches, embed_dim)  │
        └──────────────────┬───────────────────────┘
                           ▼
        ┌──────────────────────────────────────────┐
        │  [CLS Token] + Positional Embedding      │
        └──────────────────┬───────────────────────┘
                           ▼
        ┌──────────────────────────────────────────┐
        │  TransformerLayer x depth                │
        └──────────────────┬───────────────────────┘
                           ▼
        ┌──────────────────────────────────────────┐
        │  [Final Normalization] (pre-norm only)   │
        └──────────────────┬───────────────────────┘
                           ▼
        ┌──────────────────────────────────────────┐
        │  SequencePooling (cls/mean/max/none)     │
        └──────────────────┬───────────────────────┘
                           ▼
        ┌──────────────────────────────────────────┐
        │  Output Features                         │
        └──────────────────────────────────────────┘

    :param img_size: Input image spatial size. Default: 224.
    :type img_size: int
    :param patch_size: Patch side length. Default: 16.
    :type patch_size: int
    :param embed_dim: Embedding dimension. Default: 768.
    :type embed_dim: int
    :param depth: Number of transformer layers. Default: 12.
    :type depth: int
    :param num_heads: Number of attention heads. Default: 12.
    :type num_heads: int
    :param mlp_ratio: MLP expansion ratio. Default: 4.0.
    :type mlp_ratio: float
    :param patch_embed_type: Patch embedding strategy. Default: ``'linear'``.
    :type patch_embed_type: PatchEmbedType
    :param attention_type: Attention mechanism. Default: ``'multi_head'``.
    :type attention_type: AttentionType
    :param normalization_type: Normalization type. Default: ``'layer_norm'``.
    :type normalization_type: NormalizationType
    :param normalization_position: ``'pre'`` or ``'post'``. Default: ``'post'``.
    :type normalization_position: NormalizationPositionType
    :param ffn_type: FFN architecture. Default: ``'mlp'``.
    :type ffn_type: FFNType
    :param use_cls_token: Prepend a CLS token. Default: True.
    :type use_cls_token: bool
    :param output_mode: Pooling strategy. Default: ``'cls'``.
    :type output_mode: PoolingStrategy
    :param dropout_rate: General dropout. Default: 0.0.
    :type dropout_rate: float
    :param attention_dropout_rate: Attention dropout. Default: 0.0.
    :type attention_dropout_rate: float
    :param pos_dropout_rate: Positional embedding dropout. Default: 0.0.
    :type pos_dropout_rate: float
    :param stochastic_depth_rate: Drop-path rate. Default: 0.0.
    :type stochastic_depth_rate: float
    :param activation: FFN activation. Default: ``'gelu'``.
    :type activation: Union[str, Callable]
    :param use_bias: Whether layers use bias. Default: True.
    :type use_bias: bool
    :param kernel_initializer: Kernel weight initializer.
    :type kernel_initializer: Union[str, initializers.Initializer]
    :param bias_initializer: Bias weight initializer.
    :type bias_initializer: Union[str, initializers.Initializer]
    :param kernel_regularizer: Kernel weight regularizer.
    :type kernel_regularizer: Optional[regularizers.Regularizer]
    :param bias_regularizer: Bias weight regularizer.
    :type bias_regularizer: Optional[regularizers.Regularizer]
    :param attention_args: Custom attention layer arguments.
    :type attention_args: Optional[Dict[str, Any]]
    :param norm_args: Custom normalization layer arguments.
    :type norm_args: Optional[Dict[str, Any]]
    :param ffn_args: Custom FFN layer arguments.
    :type ffn_args: Optional[Dict[str, Any]]
    :param patch_embed_args: Custom patch embedding arguments.
    :type patch_embed_args: Optional[Dict[str, Any]]
    :param kwargs: Additional keyword arguments for the base Layer.
    :type kwargs: Any

    :raises ValueError: If dimension parameters are invalid.
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
            attention_type: AttentionType = 'multi_head',
            normalization_type: NormalizationType = 'layer_norm',
            normalization_position: NormalizationPositionType = 'post',
            ffn_type: FFNType = 'mlp',
            use_cls_token: bool = True,
            output_mode: PoolingStrategy = 'cls',
            dropout_rate: float = 0.0,
            attention_dropout_rate: float = 0.0,
            pos_dropout_rate: float = 0.0,
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
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout must be between 0 and 1, got {dropout_rate}")
        if not (0.0 <= attention_dropout_rate <= 1.0):
            raise ValueError(f"attention_dropout must be between 0 and 1, got {attention_dropout_rate}")
        if not (0.0 <= pos_dropout_rate <= 1.0):
            raise ValueError(f"pos_dropout must be between 0 and 1, got {pos_dropout_rate}")
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
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.pos_dropout_rate = pos_dropout_rate
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
            dropout=self.pos_dropout_rate,
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
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
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

        # Create pooling layer using SequencePooling
        # For mean and max pooling with CLS token, we exclude position 0
        exclude_positions = [0] if (use_cls_token and output_mode in ['mean', 'max']) else []

        self.pooling_layer = SequencePooling(
            strategy=output_mode,
            exclude_positions=exclude_positions,
            name='output_pooling'
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
        """Create patch embedding layer based on the specified type.

        :return: Patch embedding layer.
        :rtype: keras.layers.Layer
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
        """Build the vision encoder and all sub-layers.

        :param input_shape: Shape ``(batch, height, width, channels)``.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If shape is invalid.
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

        # Build pooling layer
        self.pooling_layer.build(pos_input_shape)

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
            attention_mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through the vision encoder.

        :param inputs: Image tensor ``(B, H, W, C)``.
        :type inputs: keras.KerasTensor
        :param attention_mask: Optional mask ``(B, seq_len)``.
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: Training mode flag.
        :type training: Optional[bool]
        :return: Output features (shape depends on ``output_mode``).
        :rtype: keras.KerasTensor
        """
        x = self._get_full_sequence_features(inputs, training=training)

        output = self.pooling_layer(x, mask=attention_mask, training=training)

        return output

    def get_cls_features(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Extract CLS token features for classification.

        :param inputs: Image tensor.
        :type inputs: keras.KerasTensor
        :param training: Training mode flag.
        :type training: Optional[bool]
        :return: CLS features ``(B, embed_dim)``.
        :rtype: keras.KerasTensor
        :raises ValueError: If ``use_cls_token=False``.
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
        """Extract patch token features for dense prediction.

        :param inputs: Image tensor.
        :type inputs: keras.KerasTensor
        :param training: Training mode flag.
        :type training: Optional[bool]
        :return: Patch features ``(B, num_patches, embed_dim)``.
        :rtype: keras.KerasTensor
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
        """Get spatial features reshaped for dense prediction.

        :param inputs: Image tensor.
        :type inputs: keras.KerasTensor
        :param training: Training mode flag.
        :type training: Optional[bool]
        :return: Spatial features ``(B, patch_H, patch_W, embed_dim)``.
        :rtype: keras.KerasTensor
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
        """Compute the output shape.

        :param input_shape: Input shape ``(B, H, W, C)``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape (depends on ``output_mode``).
        :rtype: Tuple[Optional[int], ...]
        """
        batch_size = input_shape[0]

        # Create dummy sequence shape for pooling layer
        sequence_shape = (batch_size, self.seq_len, self.embed_dim)
        return self.pooling_layer.compute_output_shape(sequence_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary for serialization.

        :return: Dictionary containing all constructor parameters.
        :rtype: Dict[str, Any]
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
            'dropout_rate': self.dropout_rate,
            'attention_dropout_rate': self.attention_dropout_rate,
            'pos_dropout_rate': self.pos_dropout_rate,
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
        attention_type: AttentionType = 'multi_head',
        normalization_type: NormalizationType = 'layer_norm',
        normalization_position: NormalizationPositionType = 'post',
        ffn_type: FFNType = 'mlp',
        use_cls_token: bool = True,
        output_mode: PoolingStrategy = 'cls',
        dropout: float = 0.0,
        **kwargs: Any
) -> VisionEncoder:
    """
    Factory function to create a VisionEncoder with validated parameters.

    This function provides parameter validation and sensible defaults for creating
    vision_heads encoders with different architectural configurations. It supports all
    major vision_heads transformer variants through configurable components.

    :param img_size: Input image size. Must be divisible by patch_size.
    :type img_size: int
    :param patch_size: Size of image patches.
    :type patch_size: int
    :param embed_dim: Embedding dimension.
    :type embed_dim: int
    :param depth: Number of transformer layers.
    :type depth: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param mlp_ratio: MLP expansion ratio.
    :type mlp_ratio: float
    :param patch_embed_type: Type of patch embedding strategy.
    :type patch_embed_type: str
    :param attention_type: Type of attention mechanism.
    :type attention_type: str
    :param normalization_type: Type of normalization.
    :type normalization_type: str
    :param normalization_position: Position of normalization layers.
    :type normalization_position: str
    :param ffn_type: Type of feed-forward network.
    :type ffn_type: str
    :param use_cls_token: Whether to use CLS token.
    :type use_cls_token: bool
    :param output_mode: Output pooling mode.
    :type output_mode: str
    :param dropout: General dropout rate.
    :type dropout: float
    :param kwargs: Additional arguments for VisionEncoder constructor.
    :return: Configured VisionEncoder instance.
    :rtype: VisionEncoder
    :raises ValueError: If any parameter validation fails.
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
        dropout_rate=dropout,
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
        attention_type='multi_head',
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
        attention_type='multi_head',
        normalization_type='layer_norm',
        normalization_position='post',
        ffn_type='mlp',
        **kwargs
    )

# ---------------------------------------------------------------------