"""
CLIP (Contrastive Language-Image Pre-training) model implementation with SOTA attention mechanisms.

This implementation follows the 2025 SOTA Transformer architecture guidelines with:
- RMSNorm instead of LayerNorm
- Grouped Query Attention (GQA)
- RoPE positional embeddings
- SwiGLU FFN
- Stochastic Depth
- Pre-layer normalization
"""

import keras
from keras import layers, ops

from typing import Optional
from dl_techniques.utils.logger import logger


class CLIPConfig:
    """Configuration class for CLIP model with SOTA settings."""

    def __init__(
            self,
            # Vision encoder configuration
            image_size: int = 224,
            patch_size: int = 16,
            vision_layers: int = 12,
            vision_width: int = 768,

            # Text encoder configuration
            vocab_size: int = 49408,
            context_length: int = 77,
            text_layers: int = 12,
            text_width: int = 512,
            text_heads: int = 8,

            # Shared configuration
            embed_dim: int = 512,

            # SOTA attention configuration
            n_head: int = 12,
            n_kv_head: int = 4,  # GQA with 3:1 ratio
            ffn_expansion_factor: float = 8 / 3,
            ffn_multiple_of: int = 256,

            # Regularization
            dropout_prob: float = 0.0,
            attention_dropout: float = 0.0,
            stochastic_depth_prob: float = 0.1,

            # Normalization
            rms_norm_eps: float = 1e-6,

            # Positional encoding
            rope_theta: float = 10000.0,
            rope_percentage: float = 0.5,  # Percentage of dimensions to apply RoPE to

            # Training specifics
            logit_scale_init: float = 2.6592,  # ln(1/0.07)
    ):
        self.image_size = image_size
        self.patch_size = patch_size
        self.vision_layers = vision_layers
        self.vision_width = vision_width

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.text_layers = text_layers
        self.text_width = text_width
        self.text_heads = text_heads

        self.embed_dim = embed_dim

        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.ffn_expansion_factor = ffn_expansion_factor
        self.ffn_multiple_of = ffn_multiple_of

        self.dropout_prob = dropout_prob
        self.attention_dropout = attention_dropout
        self.stochastic_depth_prob = stochastic_depth_prob

        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_percentage = rope_percentage

        self.logit_scale_init = logit_scale_init

        # Derived properties
        self.vision_head_dim = self.vision_width // self.n_head
        self.text_head_dim = self.text_width // self.text_heads
        self.n_group = self.n_head // self.n_kv_head


@keras.saving.register_keras_serializable()
class RMSNorm(layers.Layer):
    """RMS Normalization layer with numerical stability."""

    def __init__(
            self,
            config: CLIPConfig,
            name: Optional[str] = None,
            **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.eps = config.rms_norm_eps
        self.config = config

    def build(self, input_shape):
        self.weight = self.add_weight(
            name='weight',
            shape=(input_shape[-1],),
            initializer='ones',
            trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        # Critical: Use float32 for stability even in mixed precision
        x_fp32 = ops.cast(x, 'float32')
        variance = ops.mean(ops.square(x_fp32), axis=-1, keepdims=True)
        x_normed = x_fp32 * ops.rsqrt(variance + self.eps)
        # Cast back to original dtype
        return ops.cast(x_normed, x.dtype) * self.weight

    def get_config(self):
        config = super().get_config()
        config.update({
            'eps': self.eps,
        })
        return config


@keras.saving.register_keras_serializable()
class StochasticDepth(layers.Layer):
    """Stochastic Depth layer for improved regularization."""

    def __init__(
            self,
            drop_rate: float,
            name: Optional[str] = None,
            **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.drop_rate = drop_rate

    def call(self, x, training=None):
        if not training or self.drop_rate == 0.0:
            return x

        # Create random tensor for dropping entire samples
        batch_size = ops.shape(x)[0]
        random_tensor = ops.random.uniform((batch_size, 1, 1), dtype=x.dtype)

        # Binary mask based on drop rate
        keep_prob = 1.0 - self.drop_rate
        binary_mask = ops.cast(random_tensor >= self.drop_rate, x.dtype)

        # Scale by keep_prob to maintain expected value
        return x * binary_mask / keep_prob

    def get_config(self):
        config = super().get_config()
        config.update({
            'drop_rate': self.drop_rate,
        })
        return config


@keras.saving.register_keras_serializable()
class RotaryPositionEmbedding(layers.Layer):
    """Rotary Position Embedding for attention layers."""

    def __init__(
            self,
            config: CLIPConfig,
            head_dim: int,
            max_seq_len: int,
            rope_percentage: float = 0.5,
            name: Optional[str] = None,
            **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.rope_theta = config.rope_theta
        self.rope_percentage = rope_percentage
        self.config = config

    def build(self, input_shape):
        self._build_rope_cache()
        super().build(input_shape)

    def _build_rope_cache(self):
        # Only apply RoPE to a portion of dimensions (typically 25-50%)
        rope_dim = self.head_dim // 2  # Apply to half the dimensions

        # Create frequency tensor
        inv_freq = 1.0 / (self.rope_theta ** (ops.arange(0, rope_dim, 2, dtype='float32') / rope_dim))

        # Position indices
        t = ops.arange(self.max_seq_len, dtype='float32')

        # Outer product to get all position-frequency combinations
        freqs = ops.outer(t, inv_freq)  # (max_seq_len, rope_dim // 2)

        # Create cos and sin tables
        cos = ops.cos(freqs)
        sin = ops.sin(freqs)

        # Store as non-trainable weights
        self.cos_cached = self.add_weight(
            name='cos_cached',
            shape=cos.shape,
            initializer='zeros',
            trainable=False
        )
        self.sin_cached = self.add_weight(
            name='sin_cached',
            shape=sin.shape,
            initializer='zeros',
            trainable=False
        )

        # Set the values
        self.cos_cached.assign(cos)
        self.sin_cached.assign(sin)

    def apply_rope(self, x, seq_len):
        """Apply rotary position embedding to input tensor."""
        # x shape: (batch, n_head, seq_len, head_dim)
        rope_dim = self.head_dim // 2

        # Split into RoPE and non-RoPE dimensions
        x_rope = x[..., :rope_dim]  # Apply RoPE here
        x_pass = x[..., rope_dim:]  # Pass through unchanged

        # Get cached values for current sequence length
        cos = self.cos_cached[:seq_len]  # (seq_len, rope_dim // 2)
        sin = self.sin_cached[:seq_len]  # (seq_len, rope_dim // 2)

        # Reshape x_rope for rotation: (batch, n_head, seq_len, rope_dim // 2, 2)
        x_rope = ops.reshape(x_rope, x_rope.shape[:-1] + (rope_dim // 2, 2))

        # Extract real and imaginary parts
        x1 = x_rope[..., 0]  # Real part
        x2 = x_rope[..., 1]  # Imaginary part

        # Apply rotation
        rotated_1 = x1 * cos - x2 * sin
        rotated_2 = x1 * sin + x2 * cos

        # Stack back together
        x_rope_rotated = ops.stack([rotated_1, rotated_2], axis=-1)
        x_rope_rotated = ops.reshape(x_rope_rotated, x_rope.shape[:-1] + (rope_dim,))

        # Concatenate with pass-through dimensions
        return ops.concatenate([x_rope_rotated, x_pass], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            'head_dim': self.head_dim,
            'max_seq_len': self.max_seq_len,
            'rope_theta': self.rope_theta,
        })
        return config


@keras.saving.register_keras_serializable()
class GroupedQueryAttention(layers.Layer):
    """Grouped Query Attention with RoPE positional embeddings."""

    def __init__(
            self,
            config: CLIPConfig,
            d_model: int,
            n_head: int,
            n_kv_head: int,
            max_seq_len: int,
            name: Optional[str] = None,
            **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.d_model = d_model
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = self.d_model // self.n_head
        self.max_seq_len = max_seq_len
        self.config = config

        # Critical: Ensure dimensions align
        assert self.d_model % self.n_head == 0
        assert self.n_head % self.n_kv_head == 0
        self.n_group = self.n_head // self.n_kv_head

        self.attention_dropout = config.attention_dropout

    def build(self, input_shape):
        # Weight matrices - note the different sizes
        self.w_q = layers.Dense(self.n_head * self.head_dim, use_bias=False, name='w_q')
        self.w_k = layers.Dense(self.n_kv_head * self.head_dim, use_bias=False, name='w_k')
        self.w_v = layers.Dense(self.n_kv_head * self.head_dim, use_bias=False, name='w_v')
        self.w_o = layers.Dense(self.d_model, use_bias=False, name='w_o')

        self.dropout = layers.Dropout(self.attention_dropout)

        # RoPE for positional encoding
        self.rope = RotaryPositionEmbedding(
            self.config,
            self.head_dim,
            self.max_seq_len,
            rope_percentage=self.config.rope_percentage,
            name='rope'
        )

        super().build(input_shape)

    def call(self, x, training=None, mask=None):
        B, T, C = ops.shape(x)

        # Project to Q, K, V
        q = self.w_q(x)  # (B, T, n_head * head_dim)
        k = self.w_k(x)  # (B, T, n_kv_head * head_dim)
        v = self.w_v(x)  # (B, T, n_kv_head * head_dim)

        # Reshape for multi-head attention
        q = ops.reshape(q, (B, T, self.n_head, self.head_dim))
        k = ops.reshape(k, (B, T, self.n_kv_head, self.head_dim))
        v = ops.reshape(v, (B, T, self.n_kv_head, self.head_dim))

        # Transpose to (B, num_heads, T, head_dim)
        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))

        # Apply RoPE to Q and K
        q = self.rope.apply_rope(q, T)
        k = self.rope.apply_rope(k, T)

        # Key insight: Repeat K,V for each group
        k = ops.repeat(k, self.n_group, axis=1)  # (B, n_head, T, head_dim)
        v = ops.repeat(v, self.n_group, axis=1)  # (B, n_head, T, head_dim)

        # Standard scaled dot-product attention
        scores = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2)))
        scores = scores / ops.sqrt(ops.cast(self.head_dim, scores.dtype))

        # Apply mask if provided
        if mask is not None:
            scores = ops.where(mask, -1e9, scores)

        weights = ops.softmax(scores, axis=-1)
        weights = self.dropout(weights, training=training)

        out = ops.matmul(weights, v)  # (B, n_head, T, head_dim)
        out = ops.transpose(out, (0, 2, 1, 3))  # (B, T, n_head, head_dim)
        out = ops.reshape(out, (B, T, C))

        return self.w_o(out)

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'n_head': self.n_head,
            'n_kv_head': self.n_kv_head,
            'head_dim': self.head_dim,
            'max_seq_len': self.max_seq_len,
            'attention_dropout': self.attention_dropout,
        })
        return config


@keras.saving.register_keras_serializable()
class SwiGLUFFN(layers.Layer):
    """SwiGLU Feed-Forward Network with gating mechanism."""

    def __init__(
            self,
            config: CLIPConfig,
            d_model: int,
            name: Optional[str] = None,
            **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.d_model = d_model
        self.config = config

    def build(self, input_shape):
        # Calculate hidden dimension with proper rounding
        hidden_dim = int(self.d_model * self.config.ffn_expansion_factor * 2 / 3)
        # Round to multiple for hardware efficiency
        hidden_dim = self.config.ffn_multiple_of * (
                    (hidden_dim + self.config.ffn_multiple_of - 1) // self.config.ffn_multiple_of)

        # Three projections for SwiGLU
        self.gate_proj = layers.Dense(hidden_dim, use_bias=False, name='gate_proj')  # Gating
        self.up_proj = layers.Dense(hidden_dim, use_bias=False, name='up_proj')  # Value
        self.down_proj = layers.Dense(self.d_model, use_bias=False, name='down_proj')  # Output

        self.dropout = layers.Dropout(self.config.dropout_prob)

        super().build(input_shape)

    def call(self, x, training=None):
        # SwiGLU formula: Swish(xW₁) ⊗ xW₂
        gate = self.gate_proj(x)
        up = self.up_proj(x)

        # Apply SiLU (Swish) activation to gate
        gate_activated = ops.silu(gate)  # x * sigmoid(x)

        # Element-wise multiplication (gating)
        hidden = gate_activated * up

        # Project back to model dimension
        output = self.down_proj(hidden)

        return self.dropout(output, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
        })
        return config


@keras.saving.register_keras_serializable()
class TransformerBlock(layers.Layer):
    """Transformer block with SOTA architecture."""

    def __init__(
            self,
            config: CLIPConfig,
            d_model: int,
            n_head: int,
            n_kv_head: int,
            max_seq_len: int,
            layer_idx: int,
            total_layers: int,
            name: Optional[str] = None,
            **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.layer_idx = layer_idx
        self.total_layers = total_layers
        self.d_model = d_model
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.max_seq_len = max_seq_len
        self.config = config

    def build(self, input_shape):
        # Linear scaling of stochastic depth rate
        depth_rate = self.config.stochastic_depth_prob * self.layer_idx / self.total_layers

        self.attention_norm = RMSNorm(self.config, name='attention_norm')
        self.attention = GroupedQueryAttention(
            self.config,
            self.d_model,
            self.n_head,
            self.n_kv_head,
            self.max_seq_len,
            name='attention'
        )

        self.ffn_norm = RMSNorm(self.config, name='ffn_norm')
        self.ffn = SwiGLUFFN(self.config, self.d_model, name='ffn')

        # Apply progressively more stochastic depth in deeper layers
        self.attn_stochastic_depth = StochasticDepth(depth_rate, name='attn_stochastic_depth')
        self.ffn_stochastic_depth = StochasticDepth(depth_rate, name='ffn_stochastic_depth')

        super().build(input_shape)

    def call(self, x, training=None, mask=None):
        # Pre-norm attention with stochastic depth
        attn_input = self.attention_norm(x)
        attn_output = self.attention(attn_input, training=training, mask=mask)
        x = x + self.attn_stochastic_depth(attn_output, training=training)

        # Pre-norm FFN with stochastic depth
        ffn_input = self.ffn_norm(x)
        ffn_output = self.ffn(ffn_input, training=training)
        x = x + self.ffn_stochastic_depth(ffn_output, training=training)

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'layer_idx': self.layer_idx,
            'total_layers': self.total_layers,
            'd_model': self.d_model,
            'n_head': self.n_head,
            'n_kv_head': self.n_kv_head,
            'max_seq_len': self.max_seq_len,
        })
        return config


@keras.saving.register_keras_serializable()
class VisionTransformer(layers.Layer):
    """Vision Transformer encoder with patch embeddings."""

    def __init__(
            self,
            config: CLIPConfig,
            name: Optional[str] = None,
            **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.config = config
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.vision_width = config.vision_width
        self.vision_layers = config.vision_layers
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head

        # Calculate number of patches
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.seq_len = self.num_patches + 1  # +1 for CLS token

    def build(self, input_shape):
        # Patch embedding projection
        self.patch_embedding = layers.Dense(
            self.vision_width,
            use_bias=False,
            name='patch_embedding'
        )

        # Class token (no positional embeddings - using RoPE instead)
        self.class_token = self.add_weight(
            name='class_token',
            shape=(1, 1, self.vision_width),
            initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True
        )

        # Transformer blocks
        self.transformer_blocks = []
        for i in range(self.vision_layers):
            block = TransformerBlock(
                self.config,
                self.vision_width,
                self.n_head,
                self.n_kv_head,
                self.seq_len,
                i,
                self.vision_layers,  # total_layers
                name=f'transformer_block_{i}'
            )
            self.transformer_blocks.append(block)

        # Final layer norm
        self.final_norm = RMSNorm(self.config, name='final_norm')

        super().build(input_shape)

    def call(self, x, training=None):
        # Input shape: (batch_size, height, width, channels)
        batch_size = ops.shape(x)[0]

        # Convert to patches
        # Reshape to (batch_size, num_patches, patch_size * patch_size * channels)
        patches = ops.image.extract_patches(
            x,
            size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            rates=(1, 1),
            padding='VALID'
        )

        # Flatten patches
        patches = ops.reshape(patches, (batch_size, self.num_patches, -1))

        # Project patches to embedding dimension
        x = self.patch_embedding(patches)

        # Add class token
        class_tokens = ops.broadcast_to(self.class_token, (batch_size, 1, self.vision_width))
        x = ops.concatenate([class_tokens, x], axis=1)

        # No absolute positional embeddings - using RoPE instead

        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, training=training)

        # Apply final normalization
        x = self.final_norm(x)

        # Return class token embedding
        return x[:, 0]  # (batch_size, vision_width)

    def get_config(self):
        config = super().get_config()
        config.update({
            'image_size': self.image_size,
            'patch_size': self.patch_size,
            'vision_width': self.vision_width,
            'vision_layers': self.vision_layers,
            'n_head': self.n_head,
            'n_kv_head': self.n_kv_head,
        })
        return config


@keras.saving.register_keras_serializable()
class TextTransformer(layers.Layer):
    """Text Transformer encoder with token embeddings."""

    def __init__(
            self,
            config: CLIPConfig,
            name: Optional[str] = None,
            **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.config = config
        self.vocab_size = config.vocab_size
        self.context_length = config.context_length
        self.text_width = config.text_width
        self.text_layers = config.text_layers
        self.text_heads = config.text_heads

        # For text, we use standard attention (not GQA) to match original CLIP
        self.n_kv_head = self.text_heads

    def build(self, input_shape):
        # Token embeddings
        self.token_embedding = layers.Embedding(
            self.vocab_size,
            self.text_width,
            embeddings_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            name='token_embedding'
        )

        # No positional embeddings - using RoPE instead

        # Transformer blocks
        self.transformer_blocks = []
        for i in range(self.text_layers):
            block = TransformerBlock(
                self.config,
                self.text_width,
                self.text_heads,
                self.n_kv_head,
                self.context_length,
                i,
                self.text_layers,  # total_layers
                name=f'transformer_block_{i}'
            )
            self.transformer_blocks.append(block)

        # Final layer norm
        self.final_norm = RMSNorm(self.config, name='final_norm')

        super().build(input_shape)

    def call(self, text_ids, training=None):
        # Input shape: (batch_size, context_length)
        seq_len = ops.shape(text_ids)[1]

        # Token embeddings
        x = self.token_embedding(text_ids)

        # No absolute positional embeddings - using RoPE instead

        # Create causal mask for autoregressive generation
        mask = ops.triu(ops.ones((seq_len, seq_len), dtype='bool'), k=1)
        mask = ops.expand_dims(ops.expand_dims(mask, 0), 0)  # (1, 1, seq_len, seq_len)

        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, training=training, mask=mask)

        # Apply final normalization
        x = self.final_norm(x)

        # FIXED: Correct text feature extraction
        # Find the index of the last non-padding token for each sequence
        # Assumes padding token ID is 0
        eos_indices = ops.sum(ops.cast(text_ids != 0, 'int32'), axis=1) - 1

        # Ensure indices are within bounds
        eos_indices = ops.clip(eos_indices, 0, seq_len - 1)

        # Use advanced indexing to gather the embeddings at EOS positions
        batch_indices = ops.arange(ops.shape(x)[0])
        gathered_embeddings = x[batch_indices, eos_indices]

        return gathered_embeddings

    def get_config(self):
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'context_length': self.context_length,
            'text_width': self.text_width,
            'text_layers': self.text_layers,
            'text_heads': self.text_heads,
        })
        return config


@keras.saving.register_keras_serializable()
class CLIPModel(keras.Model):
    """CLIP model with vision and text encoders."""

    def __init__(
            self,
            config: CLIPConfig,
            name: str = "clip_model",
            **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.config = config
        self.embed_dim = config.embed_dim

        logger.info(f"Initializing CLIP model with embed_dim={self.embed_dim}")

    def build(self, input_shape):
        # Vision encoder
        self.vision_encoder = VisionTransformer(self.config, name='vision_encoder')

        # Text encoder
        self.text_encoder = TextTransformer(self.config, name='text_encoder')

        # Projection layers to shared embedding space
        self.vision_projection = layers.Dense(
            self.embed_dim,
            use_bias=False,
            name='vision_projection'
        )

        self.text_projection = layers.Dense(
            self.embed_dim,
            use_bias=False,
            name='text_projection'
        )

        # Learnable temperature parameter for contrastive loss
        self.logit_scale = self.add_weight(
            name='logit_scale',
            shape=(),
            initializer=keras.initializers.Constant(self.config.logit_scale_init),
            trainable=True
        )

        super().build(input_shape)

    def encode_image(self, image, training=None):
        """Encode images to embedding space."""
        # Get image features from vision encoder
        image_features = self.vision_encoder(image, training=training)

        # Project to shared embedding space
        image_features = self.vision_projection(image_features)

        # L2 normalize features
        image_features = image_features / ops.norm(image_features, axis=-1, keepdims=True)

        return image_features

    def encode_text(self, text_ids, training=None):
        """Encode text to embedding space."""
        # Get text features from text encoder
        text_features = self.text_encoder(text_ids, training=training)

        # Project to shared embedding space
        text_features = self.text_projection(text_features)

        # L2 normalize features
        text_features = text_features / ops.norm(text_features, axis=-1, keepdims=True)

        return text_features

    def call(self, inputs, training=None):
        """Forward pass of CLIP model."""
        if isinstance(inputs, dict):
            images = inputs.get('image', None)
            texts = inputs.get('text', None)
        else:
            images, texts = inputs

        results = {}

        if images is not None:
            image_features = self.encode_image(images, training=training)
            results['image_features'] = image_features

        if texts is not None:
            text_features = self.encode_text(texts, training=training)
            results['text_features'] = text_features

        # If both are provided, compute similarity scores
        if images is not None and texts is not None:
            # Compute cosine similarity
            logit_scale = ops.exp(self.logit_scale)
            logits_per_image = logit_scale * ops.matmul(image_features, ops.transpose(text_features))
            logits_per_text = ops.transpose(logits_per_image)

            results.update({
                'logits_per_image': logits_per_image,
                'logits_per_text': logits_per_text,
                'logit_scale': logit_scale
            })

        return results

    def get_config(self):
        return {
            'config': {
                'image_size': self.config.image_size,
                'patch_size': self.config.patch_size,
                'vision_layers': self.config.vision_layers,
                'vision_width': self.config.vision_width,
                'vocab_size': self.config.vocab_size,
                'context_length': self.config.context_length,
                'text_layers': self.config.text_layers,
                'text_width': self.config.text_width,
                'text_heads': self.config.text_heads,
                'embed_dim': self.config.embed_dim,
                'n_head': self.config.n_head,
                'n_kv_head': self.config.n_kv_head,
                'ffn_expansion_factor': self.config.ffn_expansion_factor,
                'ffn_multiple_of': self.config.ffn_multiple_of,
                'dropout_prob': self.config.dropout_prob,
                'attention_dropout': self.config.attention_dropout,
                'stochastic_depth_prob': self.config.stochastic_depth_prob,
                'rms_norm_eps': self.config.rms_norm_eps,
                'rope_theta': self.config.rope_theta,
                'rope_percentage': self.config.rope_percentage,  # FIXED: Include rope_percentage
                'logit_scale_init': self.config.logit_scale_init,
            }
        }

    @classmethod
    def from_config(cls, config):
        clip_config = CLIPConfig(**config['config'])
        return cls(clip_config)


@keras.saving.register_keras_serializable()
class CLIPContrastiveLoss(keras.losses.Loss):
    """Contrastive loss for CLIP training with numerical stability."""

    def __init__(
            self,
            temperature: float = 0.07,
            label_smoothing: float = 0.0,
            name: str = "clip_contrastive_loss",
            **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.temperature = temperature
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        """
        Compute contrastive loss for CLIP.

        Args:
            y_true: Not used for contrastive loss, but required by Keras API
            y_pred: Dictionary containing:
                - logits_per_image: (batch_size, batch_size) similarity matrix
                - logits_per_text: (batch_size, batch_size) similarity matrix

        Returns:
            Scalar loss value
        """
        logits_per_image = y_pred['logits_per_image']
        logits_per_text = y_pred['logits_per_text']

        batch_size = ops.shape(logits_per_image)[0]

        # Create labels for contrastive learning (diagonal matrix)
        labels = ops.arange(batch_size, dtype='int32')

        # Compute cross-entropy loss for both directions
        loss_img = keras.losses.sparse_categorical_crossentropy(
            labels,
            logits_per_image,
            from_logits=True,
            label_smoothing=self.label_smoothing
        )

        loss_txt = keras.losses.sparse_categorical_crossentropy(
            labels,
            logits_per_text,
            from_logits=True,
            label_smoothing=self.label_smoothing
        )

        # Average both losses
        return (ops.mean(loss_img) + ops.mean(loss_txt)) / 2.0

    def get_config(self):
        config = super().get_config()
        config.update({
            'temperature': self.temperature,
            'label_smoothing': self.label_smoothing,
        })
        return config


@keras.saving.register_keras_serializable()
class CLIPAccuracy(keras.metrics.Metric):
    """Accuracy metric for CLIP contrastive learning (both directions)."""

    def __init__(self, name: str = "clip_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update accuracy state for both image-to-text and text-to-image."""
        logits_per_image = y_pred['logits_per_image']
        logits_per_text = y_pred['logits_per_text']
        batch_size = ops.shape(logits_per_image)[0]

        # True labels are the diagonal
        labels = ops.arange(batch_size)

        # Get predictions for both directions
        img_to_text_predictions = ops.argmax(logits_per_image, axis=-1)
        text_to_img_predictions = ops.argmax(logits_per_text, axis=-1)

        # Calculate accuracy for both directions
        img_to_text_matches = ops.cast(ops.equal(img_to_text_predictions, labels), self.dtype)
        text_to_img_matches = ops.cast(ops.equal(text_to_img_predictions, labels), self.dtype)

        # Average both directions
        total_matches = (img_to_text_matches + text_to_img_matches) / 2.0

        if sample_weight is not None:
            sample_weight = ops.cast(sample_weight, self.dtype)
            total_matches = total_matches * sample_weight

        self.total.assign_add(ops.sum(total_matches))
        self.count.assign_add(ops.cast(batch_size, self.dtype))

    def result(self):
        return self.total / self.count

    def reset_state(self):
        self.total.assign(0)
        self.count.assign(0)


def create_clip_model(
        image_size: int = 224,
        patch_size: int = 16,
        vision_layers: int = 12,
        vision_width: int = 768,
        vocab_size: int = 49408,
        context_length: int = 77,
        text_layers: int = 12,
        text_width: int = 512,
        text_heads: int = 8,
        embed_dim: int = 512,
        **kwargs
) -> CLIPModel:
    """
    Create a CLIP model with specified configuration.

    Args:
        image_size: Size of input images (default: 224)
        patch_size: Size of image patches (default: 16)
        vision_layers: Number of vision transformer layers (default: 12)
        vision_width: Width of vision transformer (default: 768)
        vocab_size: Vocabulary size for text encoder (default: 49408)
        context_length: Maximum text sequence length (default: 77)
        text_layers: Number of text transformer layers (default: 12)
        text_width: Width of text transformer (default: 512)
        text_heads: Number of attention heads in text transformer (default: 8)
        embed_dim: Dimension of shared embedding space (default: 512)
        **kwargs: Additional configuration parameters

    Returns:
        CLIPModel: Configured CLIP model
    """
    config = CLIPConfig(
        image_size=image_size,
        patch_size=patch_size,
        vision_layers=vision_layers,
        vision_width=vision_width,
        vocab_size=vocab_size,
        context_length=context_length,
        text_layers=text_layers,
        text_width=text_width,
        text_heads=text_heads,
        embed_dim=embed_dim,
        **kwargs
    )

    logger.info(f"Creating CLIP model with config: {config.__dict__}")

    return CLIPModel(config)


def create_clip_variant(variant: str = "ViT-B/32") -> CLIPModel:
    """
    Create predefined CLIP model variants.

    Args:
        variant: Model variant string. Options:
            - "ViT-B/32": Base model with 32x32 patches
            - "ViT-B/16": Base model with 16x16 patches
            - "ViT-L/14": Large model with 14x14 patches
            - "ViT-B-ResNet50-Config": ViT with ResNet-50-like config
            - "ViT-B-ResNet101-Config": ViT with ResNet-101-like config

    Returns:
        CLIPModel: Configured CLIP model
    """
    variant_configs = {
        "ViT-B/32": {
            "image_size": 224,
            "patch_size": 32,
            "vision_layers": 12,
            "vision_width": 768,
            "text_layers": 12,
            "text_width": 512,
            "text_heads": 8,
            "embed_dim": 512,
            "n_head": 12,
            "n_kv_head": 4,
        },
        "ViT-B/16": {
            "image_size": 224,
            "patch_size": 16,
            "vision_layers": 12,
            "vision_width": 768,
            "text_layers": 12,
            "text_width": 512,
            "text_heads": 8,
            "embed_dim": 512,
            "n_head": 12,
            "n_kv_head": 4,
        },
        "ViT-L/14": {
            "image_size": 224,
            "patch_size": 14,
            "vision_layers": 24,
            "vision_width": 1024,
            "text_layers": 12,
            "text_width": 768,
            "text_heads": 12,
            "embed_dim": 768,
            "n_head": 16,
            "n_kv_head": 4,
        },
        "ViT-B-ResNet50-Config": {
            # Using ViT architecture but with ResNet-50-like configuration
            "image_size": 224,
            "patch_size": 32,
            "vision_layers": 12,
            "vision_width": 1024,
            "text_layers": 12,
            "text_width": 512,
            "text_heads": 8,
            "embed_dim": 1024,
            "n_head": 16,
            "n_kv_head": 4,
        },
        "ViT-B-ResNet101-Config": {
            # Using ViT architecture but with ResNet-101-like configuration
            "image_size": 224,
            "patch_size": 32,
            "vision_layers": 16,
            "vision_width": 1024,
            "text_layers": 12,
            "text_width": 512,
            "text_heads": 8,
            "embed_dim": 1024,
            "n_head": 16,
            "n_kv_head": 4,
        },
    }

    if variant not in variant_configs:
        raise ValueError(f"Unknown variant '{variant}'. Available variants: {list(variant_configs.keys())}")

    config_dict = variant_configs[variant]
    logger.info(f"Creating CLIP {variant} variant")

    return create_clip_model(**config_dict)