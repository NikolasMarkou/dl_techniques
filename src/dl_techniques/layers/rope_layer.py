import keras
from typing import Optional
from keras import layers, ops

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class RotaryPositionEmbedding(layers.Layer):
    """Rotary Position Embedding for attention layers."""

    def __init__(
            self,
            rope_theta: float,
            head_dim: int,
            max_seq_len: int,
            rope_percentage: float = 0.5,
            name: Optional[str] = None,
            **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.rope_percentage = rope_percentage

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

# ---------------------------------------------------------------------
