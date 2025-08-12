import keras
from keras import ops, layers
from typing import Dict, Optional, Tuple, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..utils.logger import logger
from ..layers.norms.rms_norm import RMSNorm
from ..layers.ffn.swiglu_ffn import SwiGLUFFN
from ..layers.transformer import TransformerLayer
from ..layers.tokenizers.bpe import TokenEmbedding
from ..layers.patch_embedding import PatchEmbedding2D
from ..layers.positional_embedding import PositionalEmbedding
from ..layers.attention.multi_head_attention import MultiHeadAttention
from ..layers.attention.shared_weights_cross_attention import SharedWeightsCrossAttention

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class VisionEncoder(keras.layers.Layer):
    """
    Vision encoder using Vision Transformer architecture with TransformerLayer.

    This encoder processes images through patch embeddings and transformer layers
    to produce visual feature representations.

    Args:
        image_size: Input image size as (height, width).
        patch_size: Size of image patches.
        embed_dim: Embedding dimension.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        mlp_ratio: Ratio of MLP hidden dimension to embedding dimension.
        dropout_rate: Dropout rate.
        layer_norm_epsilon: Layer normalization epsilon.
        use_cls_token: Whether to use a classification token.
        attention_type: Type of attention mechanism to use.
        normalization_type: Type of normalization to use.
        ffn_type: Type of feed-forward network to use.
        kernel_initializer: Initializer for kernel weights.
        bias_initializer: Initializer for bias weights.
        kernel_regularizer: Optional regularizer for kernel weights.
        bias_regularizer: Optional regularizer for bias weights.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
            self,
            image_size: Tuple[int, int] = (224, 224),
            patch_size: Tuple[int, int] = (16, 16),
            embed_dim: int = 768,
            num_layers: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.0,
            dropout_rate: float = 0.1,
            layer_norm_epsilon: float = 1e-6,
            use_cls_token: bool = True,
            attention_type: str = 'multi_head_attention',
            normalization_type: str = 'layer_norm',
            ffn_type: str = 'mlp',
            kernel_initializer: str = 'glorot_uniform',
            bias_initializer: str = 'zeros',
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if num_heads <= 0 or embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")

        # Store configuration
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.use_cls_token = use_cls_token
        self.attention_type = attention_type
        self.normalization_type = normalization_type
        self.ffn_type = ffn_type
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # Calculate number of patches
        self.num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
        self.intermediate_size = int(embed_dim * mlp_ratio)

        # Components will be initialized in build() - DO NOT build here
        self.patch_embedding = None
        self.cls_token = None
        self.position_embedding = None
        self.transformer_layers = []
        self.layer_norm = None
        self.dropout = None

        # Store build input shape for serialization
        self._build_input_shape = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the vision encoder layers."""
        if self.built:
            return

        self._build_input_shape = input_shape

        # Validate input shape
        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D input shape, got {len(input_shape)}D: {input_shape}")

        # Patch embedding
        self.patch_embedding = PatchEmbedding2D(
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="patch_embedding"
        )

        # CLS token
        if self.use_cls_token:
            self.cls_token = self.add_weight(
                name="cls_token",
                shape=(1, 1, self.embed_dim),
                initializer="truncated_normal",
                trainable=True
            )

        # Positional embedding
        seq_len = self.num_patches + (1 if self.use_cls_token else 0)
        self.position_embedding = PositionalEmbedding(
            max_seq_len=seq_len,
            dim=self.embed_dim,
            dropout=self.dropout_rate,
            name="position_embedding"
        )

        # Transformer layers
        self.transformer_layers = []
        for i in range(self.num_layers):
            transformer_layer = TransformerLayer(
                hidden_size=self.embed_dim,
                num_heads=self.num_heads,
                intermediate_size=self.intermediate_size,
                attention_type=self.attention_type,
                normalization_type=self.normalization_type,
                ffn_type=self.ffn_type,
                dropout_rate=self.dropout_rate,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name=f"transformer_layer_{i}"
            )
            self.transformer_layers.append(transformer_layer)

        # Final layer norm
        if self.normalization_type == 'rms_norm':
            self.layer_norm = RMSNorm(
                epsilon=self.layer_norm_epsilon,
                name="final_layer_norm"
            )
        else:
            self.layer_norm = layers.LayerNormalization(
                epsilon=self.layer_norm_epsilon,
                name="final_layer_norm"
            )

        # Dropout
        self.dropout = layers.Dropout(self.dropout_rate, name="dropout")

        # Build all sublayers - CRITICAL: Build children before parent
        self.patch_embedding.build(input_shape)

        # Calculate patch embedding output shape
        patch_output_shape = self.patch_embedding.compute_output_shape(input_shape)

        # Add CLS token dimension if used
        if self.use_cls_token:
            seq_shape = (patch_output_shape[0], patch_output_shape[1] + 1, patch_output_shape[2])
        else:
            seq_shape = patch_output_shape

        self.position_embedding.build(seq_shape)

        # Build transformer layers
        current_shape = seq_shape
        for transformer_layer in self.transformer_layers:
            transformer_layer.build(current_shape)
            current_shape = transformer_layer.compute_output_shape(current_shape)

        self.layer_norm.build(current_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass of the vision encoder.

        Args:
            inputs: Input images of shape (batch_size, height, width, channels).
            training: Whether in training mode.

        Returns:
            Encoded vision features of shape (batch_size, seq_len, embed_dim).
        """
        batch_size = ops.shape(inputs)[0]

        # Patch embedding
        x = self.patch_embedding(inputs, training=training)  # (batch, num_patches, embed_dim)

        # Add CLS token if used
        if self.use_cls_token:
            cls_tokens = ops.tile(self.cls_token, [batch_size, 1, 1])
            x = ops.concatenate([cls_tokens, x], axis=1)

        # Add positional embedding
        x = self.position_embedding(x, training=training)

        # Apply transformer layers
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, training=training)

        # Final layer norm and dropout
        x = self.layer_norm(x, training=training)
        x = self.dropout(x, training=training)

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer."""
        seq_len = self.num_patches + (1 if self.use_cls_token else 0)
        return (input_shape[0], seq_len, self.embed_dim)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "dropout_rate": self.dropout_rate,
            "layer_norm_epsilon": self.layer_norm_epsilon,
            "use_cls_token": self.use_cls_token,
            "attention_type": self.attention_type,
            "normalization_type": self.normalization_type,
            "ffn_type": self.ffn_type,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get build configuration."""
        return {
            'input_shape': self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build from configuration."""
        if config.get('input_shape') is not None:
            self.build(config['input_shape'])


@keras.saving.register_keras_serializable()
class TextEncoder(keras.layers.Layer):
    """
    Text encoder using Transformer architecture with TransformerLayer.

    This encoder processes text tokens through embeddings and transformer layers
    to produce textual feature representations.

    Args:
        vocab_size: Size of the vocabulary.
        embed_dim: Embedding dimension.
        max_seq_len: Maximum sequence length.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        intermediate_size: Size of the intermediate layer in FFN.
        dropout_rate: Dropout rate.
        layer_norm_epsilon: Layer normalization epsilon.
        use_causal_mask: Whether to use causal masking.
        attention_type: Type of attention mechanism to use.
        normalization_type: Type of normalization to use.
        ffn_type: Type of feed-forward network to use.
        kernel_initializer: Initializer for kernel weights.
        bias_initializer: Initializer for bias weights.
        kernel_regularizer: Optional regularizer for kernel weights.
        bias_regularizer: Optional regularizer for bias weights.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
            self,
            vocab_size: int,
            embed_dim: int = 768,
            max_seq_len: int = 512,
            num_layers: int = 12,
            num_heads: int = 12,
            intermediate_size: int = 3072,
            dropout_rate: float = 0.1,
            layer_norm_epsilon: float = 1e-6,
            use_causal_mask: bool = True,
            attention_type: str = 'multi_head_attention',
            normalization_type: str = 'layer_norm',
            ffn_type: str = 'mlp',
            kernel_initializer: str = 'glorot_uniform',
            bias_initializer: str = 'zeros',
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if num_heads <= 0 or embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")

        # Store configuration
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.use_causal_mask = use_causal_mask
        self.attention_type = attention_type
        self.normalization_type = normalization_type
        self.ffn_type = ffn_type
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # Components will be initialized in build() - DO NOT build here
        self.token_embedding = None
        self.position_embedding = None
        self.transformer_layers = []
        self.layer_norm = None
        self.dropout = None

        # Store build input shape for serialization
        self._build_input_shape = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the text encoder layers."""
        if self.built:
            return

        self._build_input_shape = input_shape

        # Validate input shape
        if len(input_shape) != 2:
            raise ValueError(f"Expected 2D input shape, got {len(input_shape)}D: {input_shape}")

        # Token embedding
        self.token_embedding = TokenEmbedding(
            vocab_size=self.vocab_size,
            embedding_dim=self.embed_dim,
            mask_zero=True,
            name="token_embedding"
        )

        # Positional embedding
        self.position_embedding = PositionalEmbedding(
            max_seq_len=self.max_seq_len,
            dim=self.embed_dim,
            dropout=self.dropout_rate,
            name="position_embedding"
        )

        # Transformer layers
        self.transformer_layers = []
        for i in range(self.num_layers):
            transformer_layer = TransformerLayer(
                hidden_size=self.embed_dim,
                num_heads=self.num_heads,
                intermediate_size=self.intermediate_size,
                attention_type=self.attention_type,
                normalization_type=self.normalization_type,
                ffn_type=self.ffn_type,
                dropout_rate=self.dropout_rate,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name=f"transformer_layer_{i}"
            )
            self.transformer_layers.append(transformer_layer)

        # Final layer norm
        if self.normalization_type == 'rms_norm':
            self.layer_norm = RMSNorm(
                epsilon=self.layer_norm_epsilon,
                name="final_layer_norm"
            )
        else:
            self.layer_norm = layers.LayerNormalization(
                epsilon=self.layer_norm_epsilon,
                name="final_layer_norm"
            )

        # Dropout
        self.dropout = layers.Dropout(self.dropout_rate, name="dropout")

        # Build all sublayers - CRITICAL: Build children before parent
        self.token_embedding.build(input_shape)

        # Calculate token embedding output shape
        token_output_shape = (input_shape[0], input_shape[1], self.embed_dim)

        self.position_embedding.build(token_output_shape)

        # Build transformer layers
        current_shape = token_output_shape
        for transformer_layer in self.transformer_layers:
            transformer_layer.build(current_shape)
            current_shape = transformer_layer.compute_output_shape(current_shape)

        self.layer_norm.build(current_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass of the text encoder.

        Args:
            inputs: Input token IDs of shape (batch_size, seq_len).
            attention_mask: Attention mask of shape (batch_size, seq_len).
            training: Whether in training mode.

        Returns:
            Encoded text features of shape (batch_size, seq_len, embed_dim).
        """
        # Token embedding
        x = self.token_embedding(inputs, training=training)

        # Add positional embedding
        x = self.position_embedding(x, training=training)

        # Apply transformer layers
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, attention_mask=attention_mask, training=training)

        # Final layer norm and dropout
        x = self.layer_norm(x, training=training)
        x = self.dropout(x, training=training)

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer."""
        return (input_shape[0], input_shape[1], self.embed_dim)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "max_seq_len": self.max_seq_len,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "intermediate_size": self.intermediate_size,
            "dropout_rate": self.dropout_rate,
            "layer_norm_epsilon": self.layer_norm_epsilon,
            "use_causal_mask": self.use_causal_mask,
            "attention_type": self.attention_type,
            "normalization_type": self.normalization_type,
            "ffn_type": self.ffn_type,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get build configuration."""
        return {
            'input_shape': self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build from configuration."""
        if config.get('input_shape') is not None:
            self.build(config['input_shape'])


@keras.saving.register_keras_serializable()
class CrossModalFusion(keras.layers.Layer):
    """
    Cross-modal fusion layer using cross-attention mechanisms.

    This layer enables interaction between vision and text features through
    cross-attention and feed-forward processing.

    Args:
        embed_dim: Embedding dimension.
        num_heads: Number of attention heads.
        num_fusion_layers: Number of cross-attention layers.
        dropout_rate: Dropout rate.
        use_shared_weights: Whether to use shared weights cross-attention.
        kernel_initializer: Initializer for kernel weights.
        bias_initializer: Initializer for bias weights.
        kernel_regularizer: Optional regularizer for kernel weights.
        bias_regularizer: Optional regularizer for bias weights.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
            self,
            embed_dim: int = 768,
            num_heads: int = 12,
            num_fusion_layers: int = 6,
            dropout_rate: float = 0.1,
            use_shared_weights: bool = True,
            kernel_initializer: str = 'glorot_uniform',
            bias_initializer: str = 'zeros',
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if num_heads <= 0 or embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        if num_fusion_layers <= 0:
            raise ValueError(f"num_fusion_layers must be positive, got {num_fusion_layers}")

        # Store configuration
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_fusion_layers = num_fusion_layers
        self.dropout_rate = dropout_rate
        self.use_shared_weights = use_shared_weights
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # Components will be initialized in build() - DO NOT build here
        self.vision_to_text_layers = []
        self.text_to_vision_layers = []
        self.vision_ffn_layers = []
        self.text_ffn_layers = []
        self.vision_norm_layers = []
        self.text_norm_layers = []

        # Store build input shape for serialization
        self._build_input_shape = None

    def build(self, input_shape: Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]) -> None:
        """Build the cross-modal fusion layers."""
        if self.built:
            return

        self._build_input_shape = input_shape

        # Validate input shapes (expecting tuple of two shapes)
        if not isinstance(input_shape, (tuple, list)) or len(input_shape) != 2:
            raise ValueError(f"Expected two input shapes, got: {input_shape}")

        vision_shape, text_shape = input_shape

        if len(vision_shape) != 3 or len(text_shape) != 3:
            raise ValueError(f"Expected 3D shapes, got vision: {vision_shape}, text: {text_shape}")

        if vision_shape[-1] != self.embed_dim or text_shape[-1] != self.embed_dim:
            raise ValueError(f"Feature dimensions must match embed_dim ({self.embed_dim})")

        # Initialize layers
        self.vision_to_text_layers = []
        self.text_to_vision_layers = []
        self.vision_ffn_layers = []
        self.text_ffn_layers = []
        self.vision_norm_layers = []
        self.text_norm_layers = []

        for i in range(self.num_fusion_layers):
            # Cross-attention layers
            if self.use_shared_weights:
                v2t_attention = SharedWeightsCrossAttention(
                    dim=self.embed_dim,
                    num_heads=self.num_heads,
                    dropout_rate=self.dropout_rate,
                    name=f"vision_to_text_attention_{i}"
                )
                t2v_attention = SharedWeightsCrossAttention(
                    dim=self.embed_dim,
                    num_heads=self.num_heads,
                    dropout_rate=self.dropout_rate,
                    name=f"text_to_vision_attention_{i}"
                )
            else:
                v2t_attention = MultiHeadAttention(
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    dropout_rate=self.dropout_rate,
                    kernel_initializer=self.kernel_initializer,
                    bias_initializer=self.bias_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    bias_regularizer=self.bias_regularizer,
                    name=f"vision_to_text_attention_{i}"
                )
                t2v_attention = MultiHeadAttention(
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    dropout_rate=self.dropout_rate,
                    kernel_initializer=self.kernel_initializer,
                    bias_initializer=self.bias_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    bias_regularizer=self.bias_regularizer,
                    name=f"text_to_vision_attention_{i}"
                )

            self.vision_to_text_layers.append(v2t_attention)
            self.text_to_vision_layers.append(t2v_attention)

            # FFN layers
            vision_ffn = SwiGLUFFN(
                d_model=self.embed_dim,
                name=f"vision_ffn_{i}"
            )
            text_ffn = SwiGLUFFN(
                d_model=self.embed_dim,
                name=f"text_ffn_{i}"
            )

            self.vision_ffn_layers.append(vision_ffn)
            self.text_ffn_layers.append(text_ffn)

            # Normalization layers
            vision_norm = RMSNorm(name=f"vision_norm_{i}")
            text_norm = RMSNorm(name=f"text_norm_{i}")

            self.vision_norm_layers.append(vision_norm)
            self.text_norm_layers.append(text_norm)

        # Build all sublayers - CRITICAL: Build children before parent
        for i in range(self.num_fusion_layers):
            if self.use_shared_weights:
                # SharedWeightsCrossAttention expects (query_shape, key_value_shape)
                self.vision_to_text_layers[i].build((vision_shape, text_shape))
                self.text_to_vision_layers[i].build((text_shape, vision_shape))
            else:
                # MultiHeadAttention build
                self.vision_to_text_layers[i].build(vision_shape)
                self.text_to_vision_layers[i].build(text_shape)

            self.vision_ffn_layers[i].build(vision_shape)
            self.text_ffn_layers[i].build(text_shape)
            self.vision_norm_layers[i].build(vision_shape)
            self.text_norm_layers[i].build(text_shape)

        super().build(input_shape)

    def call(
        self,
        vision_features: keras.KerasTensor,
        text_features: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Forward pass of the cross-modal fusion.

        Args:
            vision_features: Vision features of shape (batch_size, vision_seq_len, embed_dim).
            text_features: Text features of shape (batch_size, text_seq_len, embed_dim).
            attention_mask: Text attention mask of shape (batch_size, text_seq_len).
            training: Whether in training mode.

        Returns:
            Tuple of (fused_vision_features, fused_text_features).
        """
        v_features = vision_features
        t_features = text_features

        for i in range(self.num_fusion_layers):
            # Cross-attention
            if self.use_shared_weights:
                # Vision attending to text
                v_attended = self.vision_to_text_layers[i](
                    v_features, t_features, training=training
                )
                # Text attending to vision
                t_attended = self.text_to_vision_layers[i](
                    t_features, v_features, training=training
                )
            else:
                # Vision attending to text (query=vision, key=value=text)
                v_attended = self.vision_to_text_layers[i](
                    v_features, context=t_features, training=training
                )
                # Text attending to vision (query=text, key=value=vision)
                t_attended = self.text_to_vision_layers[i](
                    t_features, context=v_features, training=training
                )

            # Residual connection and normalization
            v_features = self.vision_norm_layers[i](v_features + v_attended, training=training)
            t_features = self.text_norm_layers[i](t_features + t_attended, training=training)

            # FFN
            v_ffn_out = self.vision_ffn_layers[i](v_features, training=training)
            t_ffn_out = self.text_ffn_layers[i](t_features, training=training)

            # Residual connection
            v_features = v_features + v_ffn_out
            t_features = t_features + t_ffn_out

        return v_features, t_features

    def compute_output_shape(
        self,
        input_shape: Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]
    ) -> Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]:
        """Compute the output shape of the layer."""
        vision_shape, text_shape = input_shape
        return vision_shape, text_shape

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "num_fusion_layers": self.num_fusion_layers,
            "dropout_rate": self.dropout_rate,
            "use_shared_weights": self.use_shared_weights,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get build configuration."""
        return {
            'input_shape': self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build from configuration."""
        if config.get('input_shape') is not None:
            self.build(config['input_shape'])


@keras.saving.register_keras_serializable()
class VisionLanguageModel(keras.Model):
    """
    Vision Language Model for multimodal tasks.

    This model combines vision and text encoders with cross-modal fusion
    to handle various vision-language tasks such as image captioning,
    visual question answering, and image-text matching.

    Args:
        vision_config: Configuration dictionary for the vision encoder.
        text_config: Configuration dictionary for the text encoder.
        fusion_config: Optional configuration dictionary for cross-modal fusion.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
            self,
            vision_config: Dict[str, Any],
            text_config: Dict[str, Any],
            fusion_config: Optional[Dict[str, Any]] = None,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)

        # Validate configurations
        if not isinstance(vision_config, dict):
            raise ValueError("vision_config must be a dictionary")
        if not isinstance(text_config, dict):
            raise ValueError("text_config must be a dictionary")
        if fusion_config is not None and not isinstance(fusion_config, dict):
            raise ValueError("fusion_config must be a dictionary or None")

        # Store configurations
        self.vision_config = vision_config
        self.text_config = text_config
        self.fusion_config = fusion_config or {}

        # Initialize encoders - DO NOT build here
        self.vision_encoder = None
        self.text_encoder = None
        self.cross_modal_fusion = None
        self.vision_projection = None
        self.text_projection = None

        # Store build input shape for serialization
        self._build_input_shape = None

        logger.info("VisionLanguageModel initialized successfully")

    def build(self, input_shape: Dict[str, Tuple[Optional[int], ...]]) -> None:
        """Build the vision language model."""
        if self.built:
            return

        self._build_input_shape = input_shape

        # Validate input shape
        if not isinstance(input_shape, dict):
            raise ValueError("input_shape must be a dictionary with 'images' and 'text_tokens' keys")
        if 'images' not in input_shape or 'text_tokens' not in input_shape:
            raise ValueError("input_shape must contain 'images' and 'text_tokens' keys")

        images_shape = input_shape['images']
        text_tokens_shape = input_shape['text_tokens']

        # Initialize encoders
        self.vision_encoder = VisionEncoder(**self.vision_config)
        self.text_encoder = TextEncoder(**self.text_config)

        # Initialize fusion layer if specified
        if self.fusion_config:
            self.cross_modal_fusion = CrossModalFusion(**self.fusion_config)

        # Projection layers for similarity computation
        vision_embed_dim = self.vision_config.get("embed_dim", 768)
        text_embed_dim = self.text_config.get("embed_dim", 768)

        self.vision_projection = layers.Dense(
            vision_embed_dim,
            name="vision_projection",
            kernel_initializer="glorot_normal"
        )
        self.text_projection = layers.Dense(
            text_embed_dim,
            name="text_projection",
            kernel_initializer="glorot_normal"
        )

        # Build all sublayers - CRITICAL: Build children before parent
        self.vision_encoder.build(images_shape)
        self.text_encoder.build(text_tokens_shape)

        # Get encoder output shapes
        vision_output_shape = self.vision_encoder.compute_output_shape(images_shape)
        text_output_shape = self.text_encoder.compute_output_shape(text_tokens_shape)

        if self.cross_modal_fusion is not None:
            fusion_input_shape = (vision_output_shape, text_output_shape)
            self.cross_modal_fusion.build(fusion_input_shape)

        # Build projection layers
        vision_global_shape = (vision_output_shape[0], vision_output_shape[2])  # (batch, embed_dim)
        text_global_shape = (text_output_shape[0], text_output_shape[2])  # (batch, embed_dim)

        self.vision_projection.build(vision_global_shape)
        self.text_projection.build(text_global_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: Dict[str, keras.KerasTensor],
        training: Optional[bool] = None
    ) -> Dict[str, keras.KerasTensor]:
        """
        Forward pass of the vision language model.

        Args:
            inputs: Dictionary containing 'images' and 'text_tokens' keys.
                   Optionally contains 'attention_mask'.
            training: Whether in training mode.

        Returns:
            Dictionary containing various model outputs.
        """
        images = inputs["images"]
        text_tokens = inputs["text_tokens"]
        attention_mask = inputs.get("attention_mask", None)

        # Encode vision and text
        vision_features = self.vision_encoder(images, training=training)
        text_features = self.text_encoder(
            text_tokens, attention_mask=attention_mask, training=training
        )

        # Cross-modal fusion if enabled
        if self.cross_modal_fusion is not None:
            fused_vision, fused_text = self.cross_modal_fusion(
                vision_features, text_features,
                attention_mask=attention_mask, training=training
            )
        else:
            fused_vision = vision_features
            fused_text = text_features

        # Global pooling for similarity computation
        # For vision: use CLS token if available, otherwise mean pooling
        if self.vision_encoder.use_cls_token:
            vision_global = fused_vision[:, 0]  # CLS token
        else:
            vision_global = ops.mean(fused_vision, axis=1)  # Mean pooling

        # For text: use first token (often CLS) or mean pooling of non-masked tokens
        if attention_mask is not None:
            # Masked mean pooling
            mask_expanded = ops.expand_dims(ops.cast(attention_mask, fused_text.dtype), -1)
            text_sum = ops.sum(fused_text * mask_expanded, axis=1)
            mask_sum = ops.sum(mask_expanded, axis=1)
            text_global = text_sum / (mask_sum + 1e-8)
        else:
            text_global = ops.mean(fused_text, axis=1)  # Simple mean pooling

        # Project to common space for similarity computation
        vision_projected = self.vision_projection(vision_global, training=training)
        text_projected = self.text_projection(text_global, training=training)

        # Normalize for cosine similarity
        vision_projected = ops.l2_normalize(vision_projected, axis=-1)
        text_projected = ops.l2_normalize(text_projected, axis=-1)

        # Compute similarity matrix
        similarity_matrix = ops.matmul(vision_projected, ops.transpose(text_projected))

        return {
            "vision_features": vision_features,
            "text_features": text_features,
            "fused_vision_features": fused_vision,
            "fused_text_features": fused_text,
            "vision_global": vision_projected,
            "text_global": text_projected,
            "similarity_matrix": similarity_matrix,
        }

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        config = super().get_config()
        config.update({
            "vision_config": self.vision_config,
            "text_config": self.text_config,
            "fusion_config": self.fusion_config,
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get build configuration."""
        return {
            'input_shape': self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build from configuration."""
        if config.get('input_shape') is not None:
            self.build(config['input_shape'])

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'VisionLanguageModel':
        """Create model from configuration."""
        return cls(**config)


# ---------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------

def create_vlm_for_image_captioning(
        image_size: Tuple[int, int] = (224, 224),
        vocab_size: int = 50000,
        max_text_length: int = 128
) -> VisionLanguageModel:
    """
    Create a VLM optimized for image captioning tasks.

    Args:
        image_size: Input image size.
        vocab_size: Size of text vocabulary.
        max_text_length: Maximum text sequence length.

    Returns:
        Configured VisionLanguageModel instance.
    """
    vision_config = {
        "image_size": image_size,
        "patch_size": (16, 16),
        "embed_dim": 768,
        "num_layers": 12,
        "num_heads": 12,
        "mlp_ratio": 4.0,
        "dropout_rate": 0.1,
        "use_cls_token": True,
        "attention_type": "multi_head_attention",
        "normalization_type": "layer_norm",
        "ffn_type": "mlp",
    }

    text_config = {
        "vocab_size": vocab_size,
        "embed_dim": 768,
        "max_seq_len": max_text_length,
        "num_layers": 12,
        "num_heads": 12,
        "intermediate_size": 3072,
        "dropout_rate": 0.1,
        "use_causal_mask": True,  # For autoregressive generation
        "attention_type": "multi_head_attention",
        "normalization_type": "layer_norm",
        "ffn_type": "mlp",
    }

    fusion_config = {
        "embed_dim": 768,
        "num_heads": 12,
        "num_fusion_layers": 6,
        "dropout_rate": 0.1,
        "use_shared_weights": True,
    }

    return VisionLanguageModel(
        vision_config=vision_config,
        text_config=text_config,
        fusion_config=fusion_config,
        name="image_captioning_vlm"
    )


def create_vlm_for_vqa(
        image_size: Tuple[int, int] = (224, 224),
        vocab_size: int = 50000,
        max_text_length: int = 256
) -> VisionLanguageModel:
    """
    Create a VLM optimized for Visual Question Answering tasks.

    Args:
        image_size: Input image size.
        vocab_size: Size of text vocabulary.
        max_text_length: Maximum text sequence length.

    Returns:
        Configured VisionLanguageModel instance.
    """
    vision_config = {
        "image_size": image_size,
        "patch_size": (16, 16),
        "embed_dim": 768,
        "num_layers": 12,
        "num_heads": 12,
        "mlp_ratio": 4.0,
        "dropout_rate": 0.1,
        "use_cls_token": True,
        "attention_type": "multi_head_attention",
        "normalization_type": "layer_norm",
        "ffn_type": "mlp",
    }

    text_config = {
        "vocab_size": vocab_size,
        "embed_dim": 768,
        "max_seq_len": max_text_length,
        "num_layers": 12,
        "num_heads": 12,
        "intermediate_size": 3072,
        "dropout_rate": 0.1,
        "use_causal_mask": False,  # Bidirectional for understanding questions
        "attention_type": "multi_head_attention",
        "normalization_type": "layer_norm",
        "ffn_type": "mlp",
    }

    fusion_config = {
        "embed_dim": 768,
        "num_heads": 12,
        "num_fusion_layers": 8,  # More fusion for complex reasoning
        "dropout_rate": 0.1,
        "use_shared_weights": False,  # More flexible cross-attention
    }

    return VisionLanguageModel(
        vision_config=vision_config,
        text_config=text_config,
        fusion_config=fusion_config,
        name="vqa_vlm"
    )


def create_clip_style_vlm(
        image_size: Tuple[int, int] = (224, 224),
        vocab_size: int = 50000,
        max_text_length: int = 77
) -> VisionLanguageModel:
    """
    Create a CLIP-style VLM for image-text matching.

    Args:
        image_size: Input image size.
        vocab_size: Size of text vocabulary.
        max_text_length: Maximum text sequence length.

    Returns:
        Configured VisionLanguageModel instance.
    """
    vision_config = {
        "image_size": image_size,
        "patch_size": (32, 32),  # Larger patches for efficiency
        "embed_dim": 768,
        "num_layers": 12,
        "num_heads": 12,
        "mlp_ratio": 4.0,
        "dropout_rate": 0.0,  # No dropout for contrastive learning
        "use_cls_token": True,
        "attention_type": "multi_head_attention",
        "normalization_type": "layer_norm",
        "ffn_type": "mlp",
    }

    text_config = {
        "vocab_size": vocab_size,
        "embed_dim": 768,
        "max_seq_len": max_text_length,
        "num_layers": 12,
        "num_heads": 12,
        "intermediate_size": 3072,
        "dropout_rate": 0.0,  # No dropout for contrastive learning
        "use_causal_mask": False,  # Bidirectional for text understanding
        "attention_type": "multi_head_attention",
        "normalization_type": "layer_norm",
        "ffn_type": "mlp",
    }

    # No cross-modal fusion for CLIP-style architecture
    return VisionLanguageModel(
        vision_config=vision_config,
        text_config=text_config,
        fusion_config=None,
        name="clip_style_vlm"
    )