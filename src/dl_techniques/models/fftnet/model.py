"""
FFTNet Foundation Model Implementation
======================================

A complete and refactored implementation of FFTNet (Adaptive Spectral Filtering)
as a pure foundation model. This version is designed with strict separation between
the core encoding logic and task-specific heads for maximum flexibility in
vision tasks.

Based on: "The FFT Strikes Back: An Efficient Alternative to Self-Attention"
(Fein-Ashley, 2025) arXiv:2502.18394v2 [cs.LG]

Refactored Architecture Philosophy:
-----------------------------------

This implementation strictly adheres to the principle of separating the foundation model
from the task-specific head. The `FFTNet` class acts as a pure feature extractor,
transforming input images into contextualized patch representations. All task-specific
logic, such as classification, detection, or segmentation, is delegated to downstream
"head" models.

**Architectural Contract:**

```
Input Processing:
    Image → Patch Embedding → Add CLS Token → Add Position Embedding
               ↓
FFTNet Transformer Stack (N blocks)
    [FFTMixer + FFN] × N layers
               ↓
Output Dictionary: {
    "last_hidden_state": [batch_size, num_patches+1, hidden_size],
    "cls_token": [batch_size, hidden_size],
    "patch_features": [batch_size, num_patches, hidden_size]
}
               ↓
Vision Task Head (e.g., ImageClassificationHead)
               ↓
Task-Specific Outputs
    (e.g., {"logits": ..., "probabilities": ...})
```

**Key Design Principles:**

1.  **Pure Encoder:** The `FFTNet` model contains no pooling or classification layers.
    Its sole responsibility is to produce high-fidelity patch representations.
2.  **Consistent Output:** The `call` method always returns a dictionary containing
    sequence features, cls token, and patch features. This provides a stable interface.
3.  **Decoupled Heads:** Task-specific heads are separate and can be attached via
    the integration function `create_fftnet_with_head`.
4.  **Simplified Interface:** Clean, composable design pattern for multi-task learning.

This refactoring enables:
- **Easy Multi-Tasking:** A single shared FFTNet encoder feeds multiple task heads.
- **Clean Fine-Tuning:** Foundation weights can be frozen separately from task heads.
- **Model Reusability:** Same pre-trained FFTNet for classification, detection, segmentation.

Usage Examples:
--------------
```python
# 1. Create the foundational FFTNet model
fftnet_encoder = FFTNet.from_variant("base", image_size=224, patch_size=16)

# 2. Use as a feature extractor
inputs = keras.random.normal((4, 224, 224, 3))
outputs = fftnet_encoder(inputs)
features = outputs['last_hidden_state']  # (4, 197, 768)
cls_token = outputs['cls_token']          # (4, 768)

# 3. Combine with a task head (see create_fftnet_with_head)
from dl_techniques.vision.heads import create_vision_head, VisionTaskConfig

task_config = VisionTaskConfig(
    name="imagenet_classification",
    task_type="image_classification",
    num_classes=1000
)
classification_model = create_fftnet_with_head(
    fftnet_variant="base",
    task_config=task_config
)
```
"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
from keras import ops, layers, initializers
from typing import Optional, Dict, Any, Tuple, Literal, Union
import math

try:
    import tensorflow as tf

    _HAVE_TF = True
except ImportError:
    _HAVE_TF = False
    raise ImportError("FFTNet requires TensorFlow backend for FFT operations.")

# Import factories from dl_techniques
from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.layers.ffn import create_ffn_layer
from dl_techniques.utils.logger import logger


# ==============================================================================
# Core FFT Mixing Layer (As Described in Paper)
# ==============================================================================

@keras.saving.register_keras_serializable()
class FFTMixer(layers.Layer):
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
            layers.Dense(mlp_hidden_dim, activation='gelu', name='mlp_hidden'),
            layers.Dense(embed_dim, name='mlp_out')
        ], name='filter_mlp')

        self.dropout = layers.Dropout(dropout_p)

        # Will be created in build()
        self.modrelu_bias = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer by creating frequency-dependent parameters."""
        _, seq_len, embed_dim = input_shape

        # Base spectral filter W_base (initialized to ones)
        self.W_base = self.add_weight(
            name='W_base',
            shape=(seq_len, embed_dim),
            initializer=initializers.Ones(),
            trainable=True,
            dtype="float32"
        )

        # modReLU bias (per feature, applies to magnitude)
        if self.use_bias_in_modrelu:
            self.modrelu_bias = self.add_weight(
                name='modrelu_bias',
                shape=(embed_dim,),
                initializer=initializers.Constant(-0.1),
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
        # 1. Fourier Transform
        F = tf.signal.fft(ops.cast(inputs, dtype="complex64"))

        # 2. Adaptive Spectral Filtering
        c = ops.mean(inputs, axis=1)
        delta_W = self.filter_mlp(c)
        delta_W_expanded = ops.expand_dims(delta_W, axis=1)
        W = self.W_base + delta_W_expanded
        W_complex = ops.cast(W, dtype="complex64")
        F_filtered = F * W_complex

        # 3. Nonlinear Activation: modReLU
        F_activated = self._apply_modrelu(F_filtered)

        # 4. Inverse Fourier Transform
        Y_complex = tf.signal.ifft(F_activated)
        Y = ops.real(Y_complex)

        # 5. Apply dropout
        Y = self.dropout(Y, training=training)

        return Y

    def _apply_modrelu(self, z: keras.KerasTensor) -> keras.KerasTensor:
        """Apply modReLU activation to complex tensor."""
        magnitude = ops.abs(z)

        if self.modrelu_bias is not None:
            magnitude_biased = magnitude + self.modrelu_bias
        else:
            magnitude_biased = magnitude

        magnitude_activated = ops.relu(magnitude_biased)

        eps = ops.convert_to_tensor(1e-8, dtype="float32")
        magnitude_safe = ops.maximum(magnitude, eps)
        scale = magnitude_activated / magnitude_safe

        scale_complex = ops.cast(scale, dtype="complex64")
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


# ==============================================================================
# FFTNet Transformer Block
# ==============================================================================

@keras.saving.register_keras_serializable()
class FFTNetBlock(layers.Layer):
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


# ==============================================================================
# Patch Embedding for Vision Transformer
# ==============================================================================

@keras.saving.register_keras_serializable()
class PatchEmbedding(layers.Layer):
    """
    Convert image into sequence of patch embeddings.

    Splits an image into non-overlapping patches and projects them to
    embedding dimension using a convolutional layer.

    Args:
        patch_size: Size of each square patch. Default: 16.
        embed_dim: Embedding dimension. Default: 768.
        **kwargs: Additional keyword arguments for the Layer base class.
    """

    def __init__(
            self,
            patch_size: int = 16,
            embed_dim: int = 768,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Patch projection using convolution
        self.projection = layers.Conv2D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding='valid',
            name='projection'
        )

    def call(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Convert image to patch embeddings."""
        x = self.projection(inputs)

        # Reshape to sequence
        batch_size = ops.shape(x)[0]
        h, w = ops.shape(x)[1], ops.shape(x)[2]
        num_patches = h * w

        x = ops.reshape(x, (batch_size, num_patches, self.embed_dim))

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        batch_size, height, width, _ = input_shape
        num_patches_h = height // self.patch_size if height else None
        num_patches_w = width // self.patch_size if width else None
        num_patches = num_patches_h * num_patches_w if (num_patches_h and num_patches_w) else None
        return (batch_size, num_patches, self.embed_dim)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'patch_size': self.patch_size,
            'embed_dim': self.embed_dim,
        })
        return config


# ==============================================================================
# FFTNet Foundation Model
# ==============================================================================

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
        self.patch_embed = PatchEmbedding(
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            name='patch_embed'
        )

        # CLS token and positional embeddings will be created in build()
        self.cls_token = None
        self.pos_embed = None

        # Dropout after embeddings
        self.pos_drop = layers.Dropout(self.dropout_p)

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
            initializer=initializers.RandomNormal(stddev=0.02),
            trainable=True
        )

        # Positional embeddings: (1, num_patches + 1, embed_dim)
        self.pos_embed = self.add_weight(
            name='pos_embed',
            shape=(1, self.num_patches + 1, self.embed_dim),
            initializer=initializers.RandomNormal(stddev=0.02),
            trainable=True
        )

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
        batch_size = ops.shape(inputs)[0]

        # 1. Patch embedding
        x = self.patch_embed(inputs)  # (B, N, D)

        # 2. Prepend class token
        cls_tokens = ops.tile(self.cls_token, [batch_size, 1, 1])  # (B, 1, D)
        x = ops.concatenate([cls_tokens, x], axis=1)  # (B, N+1, D)

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


# ==============================================================================
# Integration with Vision Task Heads
# ==============================================================================

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
            layers.Dropout(head_dropout) if head_dropout > 0 else layers.Lambda(lambda x: x),
            layers.Dense(
                num_classes,
                kernel_initializer=initializers.TruncatedNormal(stddev=0.02),
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


# ==============================================================================
# Convenience Functions for Backward Compatibility
# ==============================================================================

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