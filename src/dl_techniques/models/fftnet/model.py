"""
FFTNet vision encoder: a pure feature-extracting backbone that replaces self-attention
with adaptive spectral filtering, plus the block and mixer it is built from.

Self-attention mixes tokens by forming an `N x N` score matrix, which costs `O(N^2)`
in both time and memory and is the reason long sequences and high-resolution images
are expensive. The convolution theorem offers a different route to global mixing:
a pointwise multiplication in the frequency domain is a circular convolution in the
token domain, so multiplying by a length-`N` filter couples every token to every
other token at `O(N log N)` — the cost of the transform — and with `O(N)` parameters
instead of `O(N^2)` computation.

The catch is that a fixed filter makes the mixing input-independent, which is
precisely what attention buys and what a plain Fourier mixer gives up. FFTNet
recovers the adaptivity by conditioning the filter on the input. A global context
vector `c = mean(x, axis=tokens)` is passed through a small MLP to produce a
per-feature offset, which is added to a learned base filter:

`W = W_base + MLP(mean(x))`,  `y = IFFT(modReLU(FFT(x) * W))`

so the spectral gains applied to a given image depend on that image's summary
statistics. It is weaker than attention — the modulation is one global vector, not a
per-token-pair score — and that is the trade: global receptive field and log-linear
cost, in exchange for content dependence that is global rather than pairwise.

The nonlinearity is `modReLU`, which is what keeps a stack of these layers from
collapsing. Applying a real ReLU to a complex tensor is not meaningful, and applying
none at all would make consecutive spectral filters compose into a single linear
filter. `modReLU` acts on the magnitude only — it shifts `|z|` by a learned
per-feature bias, rectifies, and rescales `z` by the ratio — so the phase, which is
where the spatial arrangement of the signal lives, passes through untouched. The
magnitude used in the denominator is floored at `1e-8` so a zero-magnitude bin does
not produce a division by zero. The bias initializes at `-0.1`, a small negative
value, so the activation starts by suppressing low-magnitude bins rather than acting
as the identity.

**Which axis the FFT runs over is the thing to get right, and it was wrong once.**
`tf.signal.fft` transforms the INNERMOST axis. The token state is `(B, N, D)`, so
calling it directly transformed `D`, the feature axis, and the layer performed no
token mixing whatsoever — the one thing the architecture exists to do. The sequence
axis is therefore transposed to the end for the transform and transposed back
afterwards. The shape of `W_base`, `(seq_len, embed_dim)`, is a gain per frequency
BIN per feature, and it is only meaningful when the bins index the token axis; that
shape is the check on this. Because `W_base` is sized by `seq_len`, the model is tied
to the token count it was built for — a fixed image resolution, unlike attention.

**ACCEPTED RAW-TF EXCEPTION (production-map §L2-5 / H10).** ``FFTMixer.call`` uses
``tf.signal.fft`` / ``tf.signal.ifft`` on a complex64 tensor. This cannot migrate to
``keras.ops``: ``keras.ops`` exposes only a real/imag-tuple ``fft`` and has NO
``ifft``, so a backend-agnostic complex forward+inverse transform is not
expressible. The raw ``tf.signal`` path is a documented exception to the
keras.ops-only rule for the forward pass.

Structurally each block is a standard pre-norm transformer block with the mixer in
the attention slot: `x + FFTMixer(norm(x))` then `x + FFN(norm(x))`. Keeping the
residual-and-norm skeleton intact is deliberate — it isolates the mixing mechanism as
the only variable, so a comparison against an attention baseline measures the mixer
rather than a differently-tuned block.

The `FFTNet` class is a pure encoder and holds no pooling or classification layer. It
embeds patches, prepends a CLS token, adds a learned positional embedding, runs the
block stack, and returns a dictionary of ``last_hidden_state``, ``cls_token`` and
``patch_features``. Returning all three unconditionally, rather than switching the
return type on a flag, gives downstream heads a stable interface: a classification
head reads the CLS token, a dense-prediction head reads the patch features, and
neither needs the encoder reconfigured. Heads are attached externally through
``create_fftnet_with_head``.

References:
    - Fein-Ashley, 2025. The FFT Strikes Back: An Efficient Alternative to
      Self-Attention. (https://arxiv.org/abs/2502.18394)
    - Lee-Thorp et al., 2021. FNet: Mixing Tokens with Fourier Transforms.
      (https://arxiv.org/abs/2105.03824)
    - Arjovsky et al., 2015. Unitary Evolution Recurrent Neural Networks.
      (https://arxiv.org/abs/1511.06464)
    - Rao et al., 2021. Global Filter Networks for Image Classification.
      (https://arxiv.org/abs/2107.00645)
    - Dosovitskiy et al., 2020. An Image is Worth 16x16 Words: Transformers for
      Image Recognition at Scale. (https://arxiv.org/abs/2010.11929)
"""

import keras
import tensorflow as tf
from typing import Optional, Dict, Any, Tuple, Literal

# ---------------------------------------------------------------------
# Local Imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.ffn import create_ffn_layer
from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.layers.embedding.patch_embedding import PatchEmbedding2D

# ---------------------------------------------------------------------
# Core FFT Mixing Layer (As Described in Paper)
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class FFTMixer(keras.layers.Layer):
    """
    Adaptive spectral filtering layer implementing the core FFTNet mechanism.

    This layer performs global token mixing in the frequency domain using the
    Fast Fourier Transform (FFT) with learned, data-dependent filtering.

    **Intent**: Replace O(N²) self-attention with O(N log N) frequency-domain
    mixing while maintaining adaptive, input-dependent behavior through learned
    spectral filters.

    **Architecture (from paper Section 3.2)**:
    ```
    Input X(B, N, D)
         ↓
    FFT → F(B, N, D) [complex]
         ↓
    Global Context: c = mean(X, axis=1)
         ↓
    MLP(c) → ΔW
         ↓
    Filter: W = W_base + ΔW
         ↓
    Apply Filter: F̃ = F ⊙ W
         ↓
    modReLU(F̃)
         ↓
    IFFT → Y(B, N, D) [real]
    ```

    Args:
        embed_dim: Embedding dimension.
        mlp_hidden_dim: Hidden dimension for the adaptive filter MLP. Default: 256.
        dropout_p: Dropout probability. Default: 0.0.
        use_bias_in_modrelu: Whether to use learnable bias in modReLU. Default: True.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        3D tensor with shape: `(batch_size, sequence_length, embed_dim)`.

    Output shape:
        3D tensor with shape: `(batch_size, sequence_length, embed_dim)`.
    """

    def __init__(
            self,
            embed_dim: int,
            mlp_hidden_dim: int = 256,
            dropout_p: float = 0.0,
            use_bias_in_modrelu: bool = True,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.embed_dim = embed_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.dropout_p = dropout_p
        self.use_bias_in_modrelu = use_bias_in_modrelu

        # Adaptive filter MLP: c -> ΔW
        self.filter_mlp = keras.Sequential([
            keras.layers.Dense(mlp_hidden_dim, activation='gelu', name='mlp_hidden'),
            keras.layers.Dense(embed_dim, name='mlp_out')
        ], name='filter_mlp')

        self.dropout = keras.layers.Dropout(dropout_p)

        # Will be created in build()
        self.modrelu_bias = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer by creating frequency-dependent parameters."""
        _, seq_len, embed_dim = input_shape

        # Base spectral filter W_base (initialized to ones)
        self.W_base = self.add_weight(
            name='W_base',
            shape=(seq_len, embed_dim),
            initializer=keras.initializers.Ones(),
            trainable=True,
            dtype="float32"
        )

        # modReLU bias (per feature, applies to magnitude)
        if self.use_bias_in_modrelu:
            self.modrelu_bias = self.add_weight(
                name='modrelu_bias',
                shape=(embed_dim,),
                initializer=keras.initializers.Constant(-0.1),
                trainable=True,
                dtype="float32"
            )

        # Build sub-layers
        self.filter_mlp.build((input_shape[0], embed_dim))

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass implementing adaptive spectral filtering."""
        # 1. Fourier Transform along the TOKEN axis.
        #
        # ``tf.signal.fft`` transforms the INNERMOST axis. ``inputs`` is
        # (B, N, D), so calling it directly transformed D — the feature axis —
        # and the layer performed no token mixing at all, which is the one thing
        # this architecture exists to do. The sequence axis is therefore moved
        # to the end for the transform and moved back afterwards.
        #
        # ``W_base`` has shape (seq_len, embed_dim), i.e. a gain per frequency
        # BIN per feature; that shape is only meaningful when the bins index the
        # token axis. The repo's other Fourier layer does the same thing
        # explicitly — see layers/attention/fnet_fourier_transform.py:368-374.
        x_complex = keras.ops.cast(inputs, dtype="complex64")
        F = keras.ops.transpose(
            tf.signal.fft(keras.ops.transpose(x_complex, (0, 2, 1))), (0, 2, 1))

        # 2. Adaptive Spectral Filtering
        c = keras.ops.mean(inputs, axis=1)
        delta_W = self.filter_mlp(c)
        delta_W_expanded = keras.ops.expand_dims(delta_W, axis=1)
        W = keras.ops.cast(self.W_base, delta_W_expanded.dtype) + delta_W_expanded
        W_complex = keras.ops.cast(W, dtype="complex64")
        F_filtered = F * W_complex

        # 3. Nonlinear Activation: modReLU
        F_activated = self._apply_modrelu(F_filtered)

        # 4. Inverse Fourier Transform (same axis handling as the forward FFT)
        Y_complex = keras.ops.transpose(
            tf.signal.ifft(keras.ops.transpose(F_activated, (0, 2, 1))), (0, 2, 1))
        # ``real()`` of a complex64 tensor is float32; hand the caller the
        # layer's own compute dtype (a no-op under the float32 policy).
        Y = keras.ops.cast(keras.ops.real(Y_complex), self.compute_dtype)

        # 5. Apply dropout
        Y = self.dropout(Y, training=training)

        return Y

    def _apply_modrelu(self, z: keras.KerasTensor) -> keras.KerasTensor:
        """Apply modReLU activation to complex tensor."""
        # DECISION plan-2026-08-19T163559-499b6f0e/D-054
        # ``magnitude`` is float32 by construction -- it is ``abs()`` of a
        # complex64 tensor and TensorFlow's complex ops have no half-precision
        # kernel. ``self.modrelu_bias`` is an ordinary autocast weight, so under
        # ``mixed_float16`` it arrived as float16 and this add raised
        # ``InvalidArgumentError: cannot compute AddV2``. The bias is lifted TO
        # the magnitude's dtype, never the magnitude cast DOWN to the bias's:
        # halving the magnitude would put the modReLU threshold and the
        # ``1e-8`` floor two lines below on different scales (float16 cannot
        # represent 1e-8 at all -- it is exactly 0.0). The sibling ``eps`` at
        # :252 already pins float32 for that reason. See decisions.md D-054.
        magnitude = keras.ops.abs(z)

        if self.modrelu_bias is not None:
            magnitude_biased = magnitude + keras.ops.cast(
                self.modrelu_bias, magnitude.dtype
            )
        else:
            magnitude_biased = magnitude

        magnitude_activated = keras.ops.relu(magnitude_biased)

        eps = keras.ops.convert_to_tensor(1e-8, dtype="float32")
        magnitude_safe = keras.ops.maximum(magnitude, eps)
        scale = magnitude_activated / magnitude_safe

        scale_complex = keras.ops.cast(scale, dtype="complex64")
        return z * scale_complex

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Output shape is identical to input shape."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'mlp_hidden_dim': self.mlp_hidden_dim,
            'dropout_p': self.dropout_p,
            'use_bias_in_modrelu': self.use_bias_in_modrelu,
        })
        return config


# ---------------------------------------------------------------------
# FFTNet Transformer Block
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class FFTNetBlock(keras.layers.Layer):
    """
    Complete Transformer-style block using FFTMixer for token mixing.

    This layer replaces standard self-attention with FFT-based adaptive
    spectral filtering, maintaining the Transformer architecture with
    pre-normalization and residual connections.

    Args:
        embed_dim: Embedding dimension.
        mlp_hidden_dim: Hidden dimension for FFTMixer's adaptive MLP. Default: 256.
        ffn_ratio: Expansion factor for FFN hidden dimension. Default: 4.
        dropout_p: Dropout probability. Default: 0.0.
        ffn_type: Type of FFN from factory. Default: 'mlp'.
        normalization_type: Type of normalization from factory. Default: 'layer_norm'.
        **kwargs: Additional keyword arguments for the Layer base class.
    """

    def __init__(
            self,
            embed_dim: int,
            mlp_hidden_dim: int = 256,
            ffn_ratio: int = 4,
            dropout_p: float = 0.0,
            ffn_type: str = 'mlp',
            normalization_type: str = 'layer_norm',
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.embed_dim = embed_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.ffn_ratio = ffn_ratio
        self.dropout_p = dropout_p
        self.ffn_type = ffn_type
        self.normalization_type = normalization_type

        # Create sub-layers using factories
        self.norm1 = create_normalization_layer(normalization_type, name='norm1')

        self.fft_mixer = FFTMixer(
            embed_dim=embed_dim,
            mlp_hidden_dim=mlp_hidden_dim,
            dropout_p=dropout_p,
            name='fft_mixer'
        )

        self.norm2 = create_normalization_layer(normalization_type, name='norm2')

        self.ffn = create_ffn_layer(
            ffn_type,
            hidden_dim=ffn_ratio * embed_dim,
            output_dim=embed_dim,
            dropout_rate=dropout_p,
            name='ffn'
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build sub-layers."""
        self.norm1.build(input_shape)
        self.fft_mixer.build(input_shape)
        self.norm2.build(input_shape)
        self.ffn.build(input_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through the FFTNet block."""
        # First residual: FFT mixing
        x = inputs + self.fft_mixer(self.norm1(inputs), training=training)

        # Second residual: FFN
        x = x + self.ffn(self.norm2(x))

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Output shape is identical to input shape."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'mlp_hidden_dim': self.mlp_hidden_dim,
            'ffn_ratio': self.ffn_ratio,
            'dropout_p': self.dropout_p,
            'ffn_type': self.ffn_type,
            'normalization_type': self.normalization_type,
        })
        return config


# ---------------------------------------------------------------------
# FFTNet Foundation Model
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class FFTNet(keras.Model):
    """
    FFTNet (Adaptive Spectral Filtering) foundation model for vision tasks.

    This is a pure encoder implementation designed to produce contextualized patch
    representations. It separates the core transformer architecture from any
    task-specific layers, making it highly flexible for pre-training, fine-tuning,
    and multi-task learning.

    **Architecture Overview:**
    ```
    Input(image)
         ↓
    Patch Embedding → (B, N, D)
         ↓
    Add CLS Token → (B, N+1, D)
         ↓
    Add Position Embedding
         ↓
    FFTNetBlock₁ (FFTMixer → FFN)
         ↓
        ...
         ↓
    FFTNetBlockₙ (FFTMixer → FFN)
         ↓
    Final Normalization
         ↓
    Output Dictionary {
        "last_hidden_state": [B, N+1, D],
        "cls_token": [B, D],
        "patch_features": [B, N, D]
    }
    ```

    Args:
        image_size: Input image size (assumes square images). Default: 224.
        patch_size: Size of each square patch. Default: 16.
        embed_dim: Embedding dimension. Default: 768.
        num_layers: Number of FFTNet blocks. Default: 12.
        mlp_hidden_dim: Hidden dimension for FFTMixer adaptive MLP. Default: 256.
        ffn_ratio: Expansion factor for FFN. Default: 4.
        dropout_p: Dropout probability. Default: 0.1.
        ffn_type: Type of FFN from factory. Default: 'mlp'.
        normalization_type: Type of normalization from factory. Default: 'layer_norm'.
        **kwargs: Additional keyword arguments for the Model base class.

    Input shape:
        4D tensor with shape: `(batch_size, image_size, image_size, 3)`.

    Output shape:
        Dictionary containing:
        - `last_hidden_state`: Full sequence (B, num_patches+1, embed_dim)
        - `cls_token`: CLS token features (B, embed_dim)
        - `patch_features`: Patch-only features (B, num_patches, embed_dim)

    Example:
        >>> # Create FFTNet-Base foundation model
        >>> model = FFTNet.from_variant("base")
        >>>
        >>> # Use as feature extractor
        >>> images = keras.random.normal((4, 224, 224, 3))
        >>> outputs = model(images)
        >>> print(outputs['cls_token'].shape)  # (4, 768)
        >>> print(outputs['last_hidden_state'].shape)  # (4, 197, 768)
    """

    # Model variant configurations matching paper Table 2
    MODEL_VARIANTS = {
        "base": {
            "embed_dim": 768,
            "num_layers": 12,
            "mlp_hidden_dim": 256,
            "ffn_ratio": 4,
            "description": "FFTNet-Base: ~76M parameters, suitable for most applications"
        },
        "large": {
            "embed_dim": 1024,
            "num_layers": 24,
            "mlp_hidden_dim": 512,
            "ffn_ratio": 4,
            "description": "FFTNet-Large: ~268M parameters, high performance"
        },
        "huge": {
            "embed_dim": 1280,
            "num_layers": 32,
            "mlp_hidden_dim": 640,
            "ffn_ratio": 4,
            "description": "FFTNet-Huge: ~540M parameters, maximum capacity"
        },
        "small": {
            "embed_dim": 512,
            "num_layers": 6,
            "mlp_hidden_dim": 128,
            "ffn_ratio": 4,
            "description": "FFTNet-Small: Lightweight for resource-constrained environments"
        },
        "tiny": {
            "embed_dim": 384,
            "num_layers": 4,
            "mlp_hidden_dim": 96,
            "ffn_ratio": 4,
            "description": "FFTNet-Tiny: Ultra-lightweight for mobile/edge deployment"
        },
    }

    # Default architecture constants
    DEFAULT_IMAGE_SIZE = 224
    DEFAULT_PATCH_SIZE = 16
    DEFAULT_DROPOUT = 0.1

    def __init__(
            self,
            image_size: int = DEFAULT_IMAGE_SIZE,
            patch_size: int = 16,
            embed_dim: int = 768,
            num_layers: int = 12,
            mlp_hidden_dim: int = 256,
            ffn_ratio: int = 4,
            dropout_p: float = DEFAULT_DROPOUT,
            ffn_type: str = 'mlp',
            normalization_type: str = 'layer_norm',
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate configuration
        self._validate_config(
            image_size, patch_size, embed_dim, num_layers, dropout_p
        )

        # Store configuration
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.mlp_hidden_dim = mlp_hidden_dim
        self.ffn_ratio = ffn_ratio
        self.dropout_p = dropout_p
        self.ffn_type = ffn_type
        self.normalization_type = normalization_type

        # Calculate number of patches
        self.num_patches = (image_size // patch_size) ** 2

        # Build architecture
        self._build_architecture()

        logger.info(
            f"Created FFTNet foundation model: {self.num_layers} layers, "
            f"embed_dim={self.embed_dim}, patches={self.num_patches}"
        )

    def _validate_config(
            self,
            image_size: int,
            patch_size: int,
            embed_dim: int,
            num_layers: int,
            dropout_p: float
    ) -> None:
        """Validate model configuration parameters."""
        if image_size <= 0:
            raise ValueError(f"image_size must be positive, got {image_size}")
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size ({image_size}) must be divisible by "
                f"patch_size ({patch_size})"
            )
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if not (0.0 <= dropout_p <= 1.0):
            raise ValueError(
                f"dropout_p must be between 0 and 1, got {dropout_p}"
            )

    def _build_architecture(self) -> None:
        """Build all model components."""
        # Patch embedding
        self.patch_embed = PatchEmbedding2D(
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            name='patch_embed'
        )

        # CLS token and positional embeddings will be created in build()
        self.cls_token = None
        self.pos_embed = None

        # Dropout after embeddings
        self.pos_drop = keras.layers.Dropout(self.dropout_p)

        # Stack of FFTNet blocks
        self.blocks = [
            FFTNetBlock(
                embed_dim=self.embed_dim,
                mlp_hidden_dim=self.mlp_hidden_dim,
                ffn_ratio=self.ffn_ratio,
                dropout_p=self.dropout_p,
                ffn_type=self.ffn_type,
                normalization_type=self.normalization_type,
                name=f'block_{i}'
            ) for i in range(self.num_layers)
        ]

        # Final normalization
        self.norm = create_normalization_layer(self.normalization_type, name='norm')

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the model by creating learnable parameters."""
        # CLS token: (1, 1, embed_dim)
        self.cls_token = self.add_weight(
            name='cls_token',
            shape=(1, 1, self.embed_dim),
            initializer=keras.initializers.RandomNormal(stddev=0.02),
            trainable=True
        )

        # Positional embeddings: (1, num_patches + 1, embed_dim)
        self.pos_embed = self.add_weight(
            name='pos_embed',
            shape=(1, self.num_patches + 1, self.embed_dim),
            initializer=keras.initializers.RandomNormal(stddev=0.02),
            trainable=True
        )

        # Explicitly build sublayers in forward order so their weights
        # materialize on .keras reload (lazy first-call build leaves the
        # patch-embed / block / norm weights unloadable on deserialization).
        self.patch_embed.build(input_shape)
        seq_shape = (input_shape[0], self.num_patches + 1, self.embed_dim)
        self.pos_drop.build(seq_shape)
        for block in self.blocks:
            block.build(seq_shape)
        self.norm.build(seq_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> Dict[str, keras.KerasTensor]:
        """
        Forward pass of the FFTNet foundation model.

        Args:
            inputs: Input images of shape (batch_size, height, width, channels).
            training: Boolean, whether the model is in training mode.

        Returns:
            A dictionary with the following keys:
            - `last_hidden_state`: The sequence of hidden states at the output
              of the final layer. Shape: (batch, num_patches+1, embed_dim).
            - `cls_token`: The CLS token features. Shape: (batch, embed_dim).
            - `patch_features`: Features for patches only (excluding CLS).
              Shape: (batch, num_patches, embed_dim).
        """
        batch_size = keras.ops.shape(inputs)[0]

        # 1. Patch embedding
        x = self.patch_embed(inputs)  # (B, N, D)

        # 2. Prepend class token
        cls_tokens = keras.ops.tile(self.cls_token, [batch_size, 1, 1])  # (B, 1, D)
        x = keras.ops.concatenate([cls_tokens, x], axis=1)  # (B, N+1, D)

        # 3. Add positional embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x, training=training)

        # 4. Apply FFTNet blocks
        for block in self.blocks:
            x = block(x, training=training)

        # 5. Final normalization
        x = self.norm(x)

        # 6. Extract features
        cls_token_output = x[:, 0]  # (B, D)
        patch_features = x[:, 1:]  # (B, N, D)

        return {
            "last_hidden_state": x,
            "cls_token": cls_token_output,
            "patch_features": patch_features
        }

    @classmethod
    def from_variant(cls, variant: str, **kwargs: Any) -> "FFTNet":
        """
        Create an FFTNet model from a predefined variant.

        Args:
            variant: String, one of "base", "large", "huge", "small", "tiny".
            **kwargs: Additional arguments to override the variant's defaults.

        Returns:
            An FFTNet model instance.

        Raises:
            ValueError: If variant is not recognized.
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        config = cls.MODEL_VARIANTS[variant].copy()
        description = config.pop("description", "")

        logger.info(f"Creating FFTNet-{variant.upper()} model")
        logger.info(f"Configuration: {description}")

        # Override defaults with kwargs
        config.update(kwargs)

        return cls(**config)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
            "num_layers": self.num_layers,
            "mlp_hidden_dim": self.mlp_hidden_dim,
            "ffn_ratio": self.ffn_ratio,
            "dropout_p": self.dropout_p,
            "ffn_type": self.ffn_type,
            "normalization_type": self.normalization_type,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FFTNet":
        """Create model from configuration."""
        return cls(**config)

    def summary(self, **kwargs) -> None:
        """Print model summary with additional FFTNet-specific information."""
        super().summary(**kwargs)
        logger.info("FFTNet Foundation Model Configuration:")
        logger.info(f"  - Architecture: {self.num_layers} layers, {self.embed_dim} hidden size")
        logger.info(f"  - Image size: {self.image_size}×{self.image_size}, patch size: {self.patch_size}")
        logger.info(f"  - Number of patches: {self.num_patches}")
        logger.info(f"  - FFT mixer MLP: {self.mlp_hidden_dim} hidden dim")
        logger.info(f"  - Feed-forward: {self.ffn_type}, ratio={self.ffn_ratio}")
        logger.info(f"  - Normalization: {self.normalization_type}")
        logger.info(f"  - Dropout: {self.dropout_p}")


# ---------------------------------------------------------------------
# Integration with Vision Task Heads
# ---------------------------------------------------------------------

def create_fftnet_with_head(
        fftnet_variant: str,
        task_type: Literal["classification", "detection", "segmentation"] = "classification",
        num_classes: Optional[int] = None,
        image_size: int = 224,
        patch_size: int = 16,
        fftnet_config_overrides: Optional[Dict[str, Any]] = None,
        head_config_overrides: Optional[Dict[str, Any]] = None,
) -> keras.Model:
    """
    Factory function to create a complete FFTNet model with a task-specific head.

    This function demonstrates the intended integration pattern:
    1. Instantiate a foundational `FFTNet` model.
    2. Create a task-specific head.
    3. Combine them into a single, end-to-end `keras.Model`.

    Args:
        fftnet_variant: String, the FFTNet variant to use (e.g., "base", "large").
        task_type: String, the vision task type: "classification", "detection", "segmentation".
        num_classes: Integer, number of classes for classification tasks. Required for classification.
        image_size: Integer, input image size. Default: 224.
        patch_size: Integer, patch size. Default: 16.
        fftnet_config_overrides: Optional dictionary to override default FFTNet
            configuration for the chosen variant.
        head_config_overrides: Optional dictionary to override default head configuration.

    Returns:
        A complete `keras.Model` ready for training or inference on a specific task.

    Example:
        >>> # Create classification model
        >>> model = create_fftnet_with_head(
        ...     fftnet_variant="base",
        ...     task_type="classification",
        ...     num_classes=1000
        ... )
        >>> model.summary()
        >>>
        >>> # Create with custom configuration
        >>> model = create_fftnet_with_head(
        ...     fftnet_variant="large",
        ...     task_type="classification",
        ...     num_classes=100,
        ...     fftnet_config_overrides={"dropout_p": 0.2, "ffn_type": "swiglu"}
        ... )
    """
    fftnet_config_overrides = fftnet_config_overrides or {}
    head_config_overrides = head_config_overrides or {}

    logger.info(f"Creating FFTNet-{fftnet_variant} with '{task_type}' head.")

    # 1. Create the foundational FFTNet model
    fftnet_encoder = FFTNet.from_variant(
        fftnet_variant,
        image_size=image_size,
        patch_size=patch_size,
        **fftnet_config_overrides
    )

    # 2. Create the task head based on task type
    if task_type == "classification":
        if num_classes is None:
            raise ValueError("num_classes must be provided for classification tasks")

        # Simple classification head
        head_dropout = head_config_overrides.get("dropout", 0.0)
        classification_head = keras.Sequential([
            keras.layers.Dropout(head_dropout) if head_dropout > 0 else keras.layers.Lambda(lambda x: x),
            keras.layers.Dense(
                num_classes,
                kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
                name="classifier"
            )
        ], name="classification_head")

        # 3. Build the end-to-end model
        inputs = keras.Input(
            shape=(image_size, image_size, 3),
            name="images"
        )

        # Get features from encoder
        encoder_outputs = fftnet_encoder(inputs)

        # Use CLS token for classification
        logits = classification_head(encoder_outputs["cls_token"])

        # Create the final model
        model = keras.Model(
            inputs=inputs,
            outputs={"logits": logits},
            name=f"fftnet_{fftnet_variant}_classifier"
        )

    elif task_type == "detection":
        raise NotImplementedError(
            "Object detection heads are not yet implemented. "
            "Use the foundation FFTNet model with your custom detection head."
        )

    elif task_type == "segmentation":
        raise NotImplementedError(
            "Segmentation heads are not yet implemented. "
            "Use the foundation FFTNet model with your custom segmentation head."
        )

    else:
        raise ValueError(
            f"Unknown task_type '{task_type}'. "
            f"Available: 'classification', 'detection', 'segmentation'"
        )

    logger.info(f"Successfully created model with {model.count_params():,} parameters.")
    return model


# ---------------------------------------------------------------------
# Convenience Functions for Backward Compatibility
# ---------------------------------------------------------------------

def create_fftnet(
        variant: Literal["base", "large", "huge", "small", "tiny"] = "base",
        image_size: int = 224,
        patch_size: int = 16,
        **kwargs: Any
) -> FFTNet:
    """
    Create FFTNet foundation model with preset configuration.

    Args:
        variant: Model variant - 'base', 'large', 'huge', 'small', or 'tiny'.
        image_size: Input image size. Default: 224.
        patch_size: Patch size. Default: 16.
        **kwargs: Additional keyword arguments to override preset configuration.

    Returns:
        Configured FFTNet foundation model.

    Example:
        >>> # Create base foundation model
        >>> model = create_fftnet('base')
        >>>
        >>> # Create large model with custom settings
        >>> model = create_fftnet(
        ...     'large',
        ...     dropout_p=0.2,
        ...     ffn_type='swiglu'
        ... )
    """
    return FFTNet.from_variant(
        variant,
        image_size=image_size,
        patch_size=patch_size,
        **kwargs
    )


def create_fftnet_classifier(
        variant: Literal["base", "large", "huge", "small", "tiny"] = "base",
        num_classes: int = 1000,
        image_size: int = 224,
        patch_size: int = 16,
        **kwargs: Any
) -> keras.Model:
    """
    Convenience function to create FFTNet classification model.

    Args:
        variant: Model variant.
        num_classes: Number of output classes.
        image_size: Input image size.
        patch_size: Patch size.
        **kwargs: Additional configuration overrides.

    Returns:
        Complete classification model.

    Example:
        >>> # Create ImageNet classifier
        >>> model = create_fftnet_classifier('base', num_classes=1000)
        >>>
        >>> # Create CIFAR-10 classifier
        >>> model = create_fftnet_classifier(
        ...     'small',
        ...     num_classes=10,
        ...     image_size=32,
        ...     dropout_p=0.3
        ... )
    """
    return create_fftnet_with_head(
        fftnet_variant=variant,
        task_type="classification",
        num_classes=num_classes,
        image_size=image_size,
        patch_size=patch_size,
        fftnet_config_overrides=kwargs
    )