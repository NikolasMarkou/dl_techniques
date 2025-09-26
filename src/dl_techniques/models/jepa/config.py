"""
JEPA Configuration Module for dl-techniques Framework.

This module provides comprehensive configuration management for different JEPA variants
including I-JEPA (images), V-JEPA (video), A-JEPA (audio), and MC-JEPA (motion+content).
"""

from dataclasses import dataclass, field
from typing import Union, Dict, Any, Literal

from dl_techniques.utils.logger import logger


@dataclass
class JEPAConfig:
    """
    Comprehensive configuration for Joint Embedding Predictive Architecture models.

    This configuration supports all JEPA variants with reasonable defaults based on
    Meta AI's research findings and empirical results.

    Args:
        variant: JEPA variant type. Determines architecture defaults.
        embed_dim: Embedding dimension for transformer blocks.
        encoder_depth: Number of transformer blocks in encoder.
        predictor_depth: Number of transformer blocks in predictor.
        num_heads: Number of attention heads in transformer blocks.
        patch_size: Size of image patches for tokenization.
        img_size: Input image size (height, width).
        num_frames: Number of frames for video variants.
        mask_ratio: Fraction of patches to mask during training.
        num_mask_blocks: Number of semantic blocks to mask.
        block_size_range: Range of mask block sizes as fraction of image.
        context_coverage: Minimum context coverage to maintain.
        ema_decay: Exponential moving average decay for target encoder.
        learning_rate: Base learning rate for training.
        weight_decay: AdamW weight decay coefficient.
        warmup_epochs: Number of warmup epochs.
        use_mixed_precision: Enable mixed precision training.
        gradient_clip_norm: Gradient clipping norm threshold.
        dropout_rate: Dropout rate in transformer blocks.
        drop_path_rate: Stochastic depth drop path rate.
        layer_scale_init: Layer scale initialization value.
        use_layer_scale: Enable layer scale in transformer blocks.
        activation: Activation function for MLP blocks.
        norm_type: Normalization type for transformer blocks.
        kernel_initializer: Weight initialization strategy.
        bias_initializer: Bias initialization strategy.

    Example:
        ```python
        # I-JEPA configuration
        config = JEPAConfig.from_preset("i-jepa-base")

        # Custom configuration
        config = JEPAConfig(
            variant="image",
            embed_dim=1024,
            encoder_depth=24,
            predictor_depth=12,
            mask_ratio=0.75
        )

        # V-JEPA for video
        video_config = JEPAConfig.from_preset("v-jepa-large")
        ```
    """

    # Architecture
    variant: Literal["image", "video", "audio", "motion_content"] = "image"
    embed_dim: int = 768
    encoder_depth: int = 12
    predictor_depth: int = 6
    num_heads: int = 12
    patch_size: Union[int, tuple] = 16
    img_size: tuple = (224, 224)
    num_frames: int = 2  # For video variants

    # Masking strategy
    mask_ratio: float = 0.75
    num_mask_blocks: int = 4
    block_size_range: tuple = (0.15, 0.20)
    context_coverage: float = 0.85

    # Training dynamics
    ema_decay: float = 0.996
    learning_rate: float = 1.5e-4
    weight_decay: float = 0.04
    warmup_epochs: int = 10

    # Optimization
    use_mixed_precision: bool = True
    gradient_clip_norm: float = 1.0

    # Regularization
    dropout_rate: float = 0.0
    drop_path_rate: float = 0.1
    layer_scale_init: float = 1e-4
    use_layer_scale: bool = True

    # Layer configuration
    activation: str = "gelu"
    norm_type: str = "layer_norm"
    kernel_initializer: str = "truncated_normal"
    bias_initializer: str = "zeros"

    # Computed properties
    mlp_ratio: float = field(default=4.0, init=False)
    head_dim: int = field(init=False)

    def __post_init__(self):
        """Compute derived configuration values."""
        self.head_dim = self.embed_dim // self.num_heads

        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )

        # Validate configuration
        self._validate_config()

    def _validate_config(self):
        """Validate configuration parameters."""
        if self.embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {self.embed_dim}")

        if not 0 < self.mask_ratio < 1:
            raise ValueError(f"mask_ratio must be in (0, 1), got {self.mask_ratio}")

        if not 0 < self.ema_decay < 1:
            raise ValueError(f"ema_decay must be in (0, 1), got {self.ema_decay}")

        if self.variant == "video" and self.num_frames < 2:
            raise ValueError("Video variant requires num_frames >= 2")

        if self.predictor_depth >= self.encoder_depth:
            logger.warning(
                f"Predictor depth ({self.predictor_depth}) >= encoder depth "
                f"({self.encoder_depth}). Consider reducing predictor depth."
            )

    @classmethod
    def from_preset(cls, preset: str) -> "JEPAConfig":
        """
        Create configuration from preset.

        Args:
            preset: Configuration preset name.

        Returns:
            JEPAConfig instance with preset parameters.

        Available presets:
            - i-jepa-tiny: Tiny I-JEPA for testing (5M params)
            - i-jepa-small: Small I-JEPA (22M params)
            - i-jepa-base: Base I-JEPA (86M params)
            - i-jepa-large: Large I-JEPA (307M params)
            - i-jepa-huge: Huge I-JEPA (632M params)
            - v-jepa-base: Base V-JEPA for video
            - v-jepa-large: Large V-JEPA for video
            - a-jepa-base: Base A-JEPA for audio
            - mc-jepa-base: Motion-Content JEPA
        """
        presets = {
            # I-JEPA variants
            "i-jepa-tiny": cls(
                variant="image",
                embed_dim=192,
                encoder_depth=12,
                predictor_depth=6,
                num_heads=3,
                patch_size=16
            ),
            "i-jepa-small": cls(
                variant="image",
                embed_dim=384,
                encoder_depth=12,
                predictor_depth=6,
                num_heads=6,
                patch_size=16
            ),
            "i-jepa-base": cls(
                variant="image",
                embed_dim=768,
                encoder_depth=12,
                predictor_depth=6,
                num_heads=12,
                patch_size=16
            ),
            "i-jepa-large": cls(
                variant="image",
                embed_dim=1024,
                encoder_depth=24,
                predictor_depth=12,
                num_heads=16,
                patch_size=16
            ),
            "i-jepa-huge": cls(
                variant="image",
                embed_dim=1280,
                encoder_depth=32,
                predictor_depth=16,
                num_heads=16,
                patch_size=14
            ),

            # V-JEPA variants
            "v-jepa-base": cls(
                variant="video",
                embed_dim=768,
                encoder_depth=12,
                predictor_depth=8,
                num_heads=12,
                patch_size=16,
                num_frames=8
            ),
            "v-jepa-large": cls(
                variant="video",
                embed_dim=1024,
                encoder_depth=24,
                predictor_depth=12,
                num_heads=16,
                patch_size=16,
                num_frames=8
            ),

            # A-JEPA variants
            "a-jepa-base": cls(
                variant="audio",
                embed_dim=768,
                encoder_depth=12,
                predictor_depth=6,
                num_heads=12,
                patch_size=8,  # Smaller patches for spectrograms
                img_size=(128, 1024)  # Mel-spectrogram dimensions
            ),

            # MC-JEPA variant
            "mc-jepa-base": cls(
                variant="motion_content",
                embed_dim=768,
                encoder_depth=12,
                predictor_depth=6,
                num_heads=12,
                patch_size=16,
                num_frames=2
            )
        }

        if preset not in presets:
            available = ", ".join(presets.keys())
            raise ValueError(f"Unknown preset '{preset}'. Available: {available}")

        return presets[preset]

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "JEPAConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)

    def get_encoder_config(self) -> Dict[str, Any]:
        """Get configuration for encoder components."""
        return {
            "embed_dim": self.embed_dim,
            "depth": self.encoder_depth,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "patch_size": self.patch_size,
            "img_size": self.img_size,
            "dropout_rate": self.dropout_rate,
            "drop_path_rate": self.drop_path_rate,
            "layer_scale_init": self.layer_scale_init,
            "use_layer_scale": self.use_layer_scale,
            "activation": self.activation,
            "norm_type": self.norm_type,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
        }

    def get_predictor_config(self) -> Dict[str, Any]:
        """Get configuration for predictor component."""
        config = self.get_encoder_config()
        config["depth"] = self.predictor_depth
        return config

    def __str__(self) -> str:
        """String representation of configuration."""
        return (
            f"JEPAConfig(variant={self.variant}, "
            f"embed_dim={self.embed_dim}, "
            f"encoder_depth={self.encoder_depth}, "
            f"predictor_depth={self.predictor_depth}, "
            f"patch_size={self.patch_size})"
        )