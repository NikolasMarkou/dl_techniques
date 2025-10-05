"""
ResNet Model Implementation with Pretrained Support and Deep Supervision
=========================================================================

A complete implementation of the ResNet architecture with support for
loading pretrained weights and deep supervision training. This implementation
follows the original "Deep Residual Learning for Image Recognition" paper
with added deep supervision capabilities.

Deep supervision provides several benefits:
- Better gradient flow to deeper layers during training
- Multi-scale feature learning and supervision
- More stable training for very deep networks
- Improved convergence and generalization

The model outputs multiple predictions during training:
- Output 0: Final inference output (after final stage, primary output)
- Output 1-N: Intermediate supervision outputs from earlier stages

Based on: "Deep Residual Learning for Image Recognition" (He et al., 2015)
https://arxiv.org/abs/1512.03385

Model Variants:
--------------
- ResNet-18: [2, 2, 2, 2] blocks, [64, 128, 256, 512] filters, BasicBlock
- ResNet-34: [3, 4, 6, 3] blocks, [64, 128, 256, 512] filters, BasicBlock
- ResNet-50: [3, 4, 6, 3] blocks, [64, 128, 256, 512] filters, BottleneckBlock
- ResNet-101: [3, 4, 23, 3] blocks, [64, 128, 256, 512] filters, BottleneckBlock
- ResNet-152: [3, 8, 36, 3] blocks, [64, 128, 256, 512] filters, BottleneckBlock

Usage Examples:
-------------
```python
# Load pretrained ImageNet weights
model = ResNet.from_variant("resnet50", pretrained=True, num_classes=1000)

# Create model with deep supervision for training
model = ResNet.from_variant("resnet50", num_classes=1000, enable_deep_supervision=True)

# Load pretrained as feature extractor
model = create_resnet("resnet34", pretrained=True, include_top=False)

# Fine-tune on CIFAR-10 with deep supervision
model = create_resnet("resnet18", num_classes=10, input_shape=(32, 32, 3),
                      enable_deep_supervision=True)
```
"""

import os
import keras
from typing import List, Optional, Union, Tuple, Dict, Any, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.layers.activations import create_activation_layer
from dl_techniques.layers.standard_blocks import (
    BasicBlock,
    BottleneckBlock,
)

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class ResNet(keras.Model):
    """ResNet model implementation with pretrained support and deep supervision.

    A deep residual learning framework that enables training of very deep
    networks by using shortcut connections that skip one or more layers.
    This implementation supports all standard ResNet variants and can adapt
    to different input sizes.

    During training with deep supervision enabled, the model outputs multiple predictions:
    - Output 0: Final inference output (after final stage, primary output)
    - Output 1: Supervision output after stage 3
    - Output 2: Supervision output after stage 2
    - Output 3: Supervision output after stage 1

    During inference, only the final output (index 0) is typically used.

    Args:
        num_classes: Integer, number of output classes for classification.
            Only used if include_top=True.
        blocks_per_stage: List of integers, number of residual blocks in each stage.
            Default is [3, 4, 6, 3] for ResNet-50.
        filters_per_stage: List of integers, number of base filters in each stage.
            Default is [64, 128, 256, 512].
        block_type: String, type of residual block. Either "basic" or "bottleneck".
            Default is "bottleneck" for deeper networks.
        kernel_regularizer: Regularizer function applied to kernels.
        normalization_type: String, type of normalization. Default is "batch_norm".
        activation_type: String, type of activation. Default is "relu".
        include_top: Boolean, whether to include the classification head.
        enable_deep_supervision: Boolean, whether to add deep supervision outputs.
            Default is False.
        input_shape: Tuple, input shape. If None and include_top=True,
            uses (224, 224, 3) for ImageNet.
        **kwargs: Additional keyword arguments for the Model base class.

    Raises:
        ValueError: If blocks_per_stage and filters_per_stage have different lengths.
        ValueError: If block_type is not "basic" or "bottleneck".

    Example:
        >>> # Create ResNet-50 model for ImageNet
        >>> model = ResNet.from_variant("resnet50", num_classes=1000)
        >>>
        >>> # Create with deep supervision for training
        >>> model = ResNet.from_variant("resnet50", enable_deep_supervision=True)
        >>>
        >>> # Load pretrained ImageNet model
        >>> model = ResNet.from_variant("resnet50", pretrained=True)
        >>>
        >>> # Load as feature extractor
        >>> model = ResNet.from_variant("resnet34", pretrained=True, include_top=False)
    """

    # Model variant configurations
    MODEL_VARIANTS = {
        "resnet18": {
            "blocks_per_stage": [2, 2, 2, 2],
            "filters_per_stage": [64, 128, 256, 512],
            "block_type": "basic"
        },
        "resnet34": {
            "blocks_per_stage": [3, 4, 6, 3],
            "filters_per_stage": [64, 128, 256, 512],
            "block_type": "basic"
        },
        "resnet50": {
            "blocks_per_stage": [3, 4, 6, 3],
            "filters_per_stage": [64, 128, 256, 512],
            "block_type": "bottleneck"
        },
        "resnet101": {
            "blocks_per_stage": [3, 4, 23, 3],
            "filters_per_stage": [64, 128, 256, 512],
            "block_type": "bottleneck"
        },
        "resnet152": {
            "blocks_per_stage": [3, 8, 36, 3],
            "filters_per_stage": [64, 128, 256, 512],
            "block_type": "bottleneck"
        },
    }

    # Pretrained weights URLs (placeholder - update with actual URLs)
    PRETRAINED_WEIGHTS = {
        "resnet18": {
            "imagenet": "https://example.com/resnet18_imagenet.keras",
        },
        "resnet34": {
            "imagenet": "https://example.com/resnet34_imagenet.keras",
        },
        "resnet50": {
            "imagenet": "https://example.com/resnet50_imagenet.keras",
        },
        "resnet101": {
            "imagenet": "https://example.com/resnet101_imagenet.keras",
        },
        "resnet152": {
            "imagenet": "https://example.com/resnet152_imagenet.keras",
        },
    }

    def __init__(
            self,
            num_classes: int = 1000,
            blocks_per_stage: List[int] = [3, 4, 6, 3],
            filters_per_stage: List[int] = [64, 128, 256, 512],
            block_type: Literal["basic", "bottleneck"] = "bottleneck",
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            normalization_type: str = "batch_norm",
            activation_type: str = "relu",
            include_top: bool = True,
            enable_deep_supervision: bool = False,
            input_shape: Tuple[int, ...] = (224, 224, 3),
            **kwargs
    ):
        super().__init__(**kwargs)

        # Validate configuration
        if len(blocks_per_stage) != len(filters_per_stage):
            raise ValueError(
                f"Length of blocks_per_stage ({len(blocks_per_stage)}) must equal "
                f"length of filters_per_stage ({len(filters_per_stage)})"
            )

        if block_type not in ["basic", "bottleneck"]:
            raise ValueError(
                f"block_type must be 'basic' or 'bottleneck', got '{block_type}'"
            )

        # Store configuration
        self.num_classes = num_classes
        self.blocks_per_stage = blocks_per_stage
        self.filters_per_stage = filters_per_stage
        self.block_type = block_type
        self.kernel_regularizer = kernel_regularizer
        self.normalization_type = normalization_type
        self.activation_type = activation_type
        self.include_top = include_top
        self.enable_deep_supervision = enable_deep_supervision
        self.input_shape_config = input_shape

        # Validate input shape
        if input_shape is None:
            input_shape = (224, 224, 3)
        if len(input_shape) != 3:
            raise ValueError(f"input_shape must be 3D, got {input_shape}")

        self.input_height, self.input_width, self.input_channels = input_shape

        # --- Build layers ---

        # 1. Initial convolution (stem)
        self._build_stem()

        # 2. Residual stages
        self.stages = []
        self.stage_outputs = []  # Store stage outputs for deep supervision
        for stage_idx in range(len(blocks_per_stage)):
            self._build_stage(stage_idx)

        # 3. Classification head
        if self.include_top:
            self._build_head()

        # 4. Deep supervision heads (if enabled)
        self.supervision_heads = []
        if self.enable_deep_supervision and self.include_top:
            self._build_supervision_heads()

        total_blocks = sum(blocks_per_stage)
        logger.info(
            f"Created ResNet model with {total_blocks} blocks "
            f"for input {input_shape}"
        )
        logger.info(f"Deep supervision enabled: {enable_deep_supervision}")

    def _build_stem(self) -> None:
        """Build initial convolution stem."""
        self.stem_conv = keras.layers.Conv2D(
            filters=64,
            kernel_size=7,
            strides=2,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
            kernel_regularizer=self.kernel_regularizer,
            name="stem_conv"
        )
        self.stem_bn = create_normalization_layer(
            self.normalization_type,
            name="stem_bn"
        )
        self.stem_act = create_activation_layer(
            self.activation_type,
            name="stem_act"
        )
        self.stem_pool = keras.layers.MaxPooling2D(
            pool_size=3,
            strides=2,
            padding="same",
            name="stem_pool"
        )

    def _build_stage(self, stage_idx: int) -> None:
        """Build a residual stage.

        Args:
            stage_idx: Index of the stage to build.
        """
        num_blocks = self.blocks_per_stage[stage_idx]
        base_filters = self.filters_per_stage[stage_idx]

        # Select block class
        if self.block_type == "basic":
            BlockClass = BasicBlock
            expansion = 1
        else:  # bottleneck
            BlockClass = BottleneckBlock
            expansion = 4

        stage_blocks = []

        for block_idx in range(num_blocks):
            # First block in stage (except first stage) uses stride=2 for downsampling
            stride = 2 if stage_idx > 0 and block_idx == 0 else 1

            # Use projection if:
            # 1. First block of stage and stride=2 (downsampling)
            # 2. First block of first stage (channel adjustment)
            use_projection = False
            if block_idx == 0:
                if stage_idx == 0:
                    # First stage: need projection if using bottleneck
                    use_projection = (self.block_type == "bottleneck")
                else:
                    # Other stages: need projection for downsampling
                    use_projection = True

            block = BlockClass(
                filters=base_filters,
                stride=stride,
                use_projection=use_projection,
                kernel_regularizer=self.kernel_regularizer,
                normalization_type=self.normalization_type,
                activation_type=self.activation_type,
                name=f"stage{stage_idx+1}_block{block_idx+1}"
            )
            stage_blocks.append(block)

        self.stages.append(stage_blocks)

    def _build_head(self) -> None:
        """Build classification head."""
        self.gap = keras.layers.GlobalAveragePooling2D(name="global_avg_pool")

        if self.num_classes > 0:
            self.classifier = keras.layers.Dense(
                units=self.num_classes,
                kernel_initializer="he_normal",
                kernel_regularizer=self.kernel_regularizer,
                name="classifier"
            )
        else:
            self.classifier = None

    def _build_supervision_heads(self) -> None:
        """Build deep supervision classification heads.

        Creates supervision heads for intermediate stages (stages 1, 2, 3).
        Each head consists of:
        - Global Average Pooling
        - Dense layer to num_classes
        """
        # Create supervision heads for stages 1, 2, 3 (not stage 0 or final stage 4)
        num_stages = len(self.blocks_per_stage)

        for stage_idx in range(1, num_stages):
            # Each supervision head
            gap_layer = keras.layers.GlobalAveragePooling2D(
                name=f"supervision_gap_stage{stage_idx+1}"
            )

            if self.num_classes > 0:
                classifier_layer = keras.layers.Dense(
                    units=self.num_classes,
                    kernel_initializer="he_normal",
                    kernel_regularizer=self.kernel_regularizer,
                    name=f"supervision_classifier_stage{stage_idx+1}"
                )
            else:
                classifier_layer = None

            self.supervision_heads.append({
                "gap": gap_layer,
                "classifier": classifier_layer,
                "stage_idx": stage_idx
            })

            logger.info(f"Added deep supervision head for stage {stage_idx+1}")

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> Union[keras.KerasTensor, List[keras.KerasTensor]]:
        """Forward pass of the model.

        Args:
            inputs: Input tensor of shape (batch_size, height, width, channels).
            training: Boolean indicating training mode.

        Returns:
            Output tensor or list of tensors depending on configuration:
            - If deep_supervision=False: Single output tensor
              - If include_top=True: (batch_size, num_classes)
              - If include_top=False: (batch_size, H', W', channels)
            - If deep_supervision=True: List of output tensors
              [final_output, supervision_output_stage3, supervision_output_stage2, supervision_output_stage1]
        """
        # Stem
        x = self.stem_conv(inputs)
        x = self.stem_bn(x, training=training)
        x = self.stem_act(x)
        x = self.stem_pool(x)

        # Storage for stage outputs (for deep supervision)
        stage_features = []

        # Residual stages
        for stage_idx, stage_blocks in enumerate(self.stages):
            for block in stage_blocks:
                x = block(x, training=training)

            # Store stage output for deep supervision
            if self.enable_deep_supervision and self.include_top:
                stage_features.append(x)

        # Final output
        if self.include_top:
            final_features = self.gap(x)
            if self.classifier:
                final_output = self.classifier(final_features)
            else:
                final_output = final_features
        else:
            final_output = x

        # Deep supervision outputs
        if self.enable_deep_supervision and self.include_top and self.supervision_heads:
            supervision_outputs = []

            # Generate supervision outputs from stored stage features
            # Process in reverse order (stage 3, 2, 1) to match BFUNet convention
            for sup_head in reversed(self.supervision_heads):
                stage_idx = sup_head["stage_idx"]
                stage_feat = stage_features[stage_idx]

                # Apply supervision head
                sup_features = sup_head["gap"](stage_feat)
                if sup_head["classifier"]:
                    sup_output = sup_head["classifier"](sup_features)
                else:
                    sup_output = sup_features

                supervision_outputs.append(sup_output)

            # Return [final_output, stage3_output, stage2_output, stage1_output]
            all_outputs = [final_output] + supervision_outputs

            logger.info(f"Returning {len(all_outputs)} outputs (deep supervision)")
            return all_outputs

        return final_output

    def load_pretrained_weights(
            self,
            weights_path: str,
            skip_mismatch: bool = True,
            by_name: bool = True
    ) -> None:
        """Load pretrained weights into the model.

        Args:
            weights_path: String, path to the weights file (.keras format).
            skip_mismatch: Boolean, whether to skip layers with mismatched shapes.
            by_name: Boolean, whether to load weights by layer name.

        Raises:
            FileNotFoundError: If weights_path doesn't exist.
            ValueError: If weights cannot be loaded.
        """
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        try:
            # Build model if not already built
            if not self.built:
                dummy_input = keras.random.normal((1,) + tuple(self.input_shape_config))
                self(dummy_input, training=False)

            logger.info(f"Loading pretrained weights from {weights_path}")

            self.load_weights(
                weights_path,
                skip_mismatch=skip_mismatch,
                by_name=by_name
            )

            if skip_mismatch:
                logger.info(
                    "Weights loaded with skip_mismatch=True. "
                    "Layers with shape mismatches were skipped."
                )
            else:
                logger.info("All weights loaded successfully.")

        except Exception as e:
            raise ValueError(f"Failed to load weights from {weights_path}: {str(e)}")

    @staticmethod
    def _download_weights(
            variant: str,
            dataset: str = "imagenet",
            cache_dir: Optional[str] = None
    ) -> str:
        """Download pretrained weights from URL.

        Args:
            variant: String, model variant name.
            dataset: String, dataset the weights were trained on.
            cache_dir: Optional string, directory to cache downloaded weights.

        Returns:
            String, path to the downloaded weights file.

        Raises:
            ValueError: If variant or dataset is not available.
        """
        if variant not in ResNet.PRETRAINED_WEIGHTS:
            raise ValueError(
                f"No pretrained weights available for variant '{variant}'. "
                f"Available variants: {list(ResNet.PRETRAINED_WEIGHTS.keys())}"
            )

        if dataset not in ResNet.PRETRAINED_WEIGHTS[variant]:
            raise ValueError(
                f"No pretrained weights available for dataset '{dataset}'. "
                f"Available datasets for {variant}: "
                f"{list(ResNet.PRETRAINED_WEIGHTS[variant].keys())}"
            )

        url = ResNet.PRETRAINED_WEIGHTS[variant][dataset]

        logger.info(f"Downloading {variant} weights from {dataset}...")

        weights_path = keras.utils.get_file(
            fname=f"{variant}_{dataset}.keras",
            origin=url,
            cache_dir=cache_dir,
            cache_subdir="models/resnet"
        )

        logger.info(f"Weights downloaded to: {weights_path}")
        return weights_path

    @classmethod
    def from_variant(
            cls,
            variant: str,
            num_classes: int = 1000,
            input_shape: Optional[Tuple[int, ...]] = None,
            pretrained: Union[bool, str] = False,
            weights_dataset: str = "imagenet",
            weights_input_shape: Optional[Tuple[int, ...]] = None,
            cache_dir: Optional[str] = None,
            **kwargs
    ) -> "ResNet":
        """Create a ResNet model from a predefined variant.

        Args:
            variant: String, one of "resnet18", "resnet34", "resnet50",
                "resnet101", "resnet152".
            num_classes: Integer, number of output classes.
            input_shape: Tuple, input shape. If None, uses (224, 224, 3).
            pretrained: Boolean or string. If True, loads pretrained weights.
                If string, treats it as a path to local weights file.
            weights_dataset: String, dataset for pretrained weights.
            weights_input_shape: Tuple, input shape used during weight pretraining.
            cache_dir: Optional string, directory to cache downloaded weights.
            **kwargs: Additional arguments passed to the constructor.

        Returns:
            ResNet model instance.

        Raises:
            ValueError: If variant is not recognized.

        Example:
            >>> # Load pretrained ImageNet model
            >>> model = ResNet.from_variant("resnet50", pretrained=True)
            >>>
            >>> # Create with deep supervision for training
            >>> model = ResNet.from_variant("resnet50", enable_deep_supervision=True)
            >>>
            >>> # Fine-tune on custom dataset
            >>> model = ResNet.from_variant("resnet34", num_classes=10,
            ...                             input_shape=(32, 32, 3))
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        config = cls.MODEL_VARIANTS[variant]

        if input_shape is None:
            input_shape = (224, 224, 3)

        logger.info(f"Creating {variant.upper()} model")

        # Handle pretrained weights
        load_weights_path = None
        skip_mismatch = False

        if pretrained:
            if isinstance(pretrained, str):
                load_weights_path = pretrained
                logger.info(f"Will load weights from local file: {load_weights_path}")
            else:
                try:
                    load_weights_path = cls._download_weights(
                        variant=variant,
                        dataset=weights_dataset,
                        cache_dir=cache_dir
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to download pretrained weights: {str(e)}. "
                        f"Continuing with random initialization."
                    )
                    load_weights_path = None

            # Check if we need to skip mismatches
            include_top = kwargs.get("include_top", True)
            if include_top:
                pretrained_classes = 1000  # ImageNet classes
                if num_classes != pretrained_classes:
                    skip_mismatch = True
                    logger.info(
                        f"num_classes ({num_classes}) differs from pretrained "
                        f"({pretrained_classes}). Will skip classifier weights."
                    )

            if weights_input_shape and input_shape and weights_input_shape != input_shape:
                logger.info(
                    f"Loading weights pretrained on {weights_input_shape} "
                    f"for model with input shape {input_shape}."
                )
                skip_mismatch = True

        # Create model
        model = cls(
            num_classes=num_classes,
            blocks_per_stage=config["blocks_per_stage"],
            filters_per_stage=config["filters_per_stage"],
            block_type=config["block_type"],
            input_shape=input_shape,
            **kwargs
        )

        # Load pretrained weights if available
        if load_weights_path:
            try:
                model.load_pretrained_weights(
                    weights_path=load_weights_path,
                    skip_mismatch=skip_mismatch,
                    by_name=True
                )
            except Exception as e:
                logger.error(f"Failed to load pretrained weights: {str(e)}")
                raise

        return model

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        config = {
            "num_classes": self.num_classes,
            "blocks_per_stage": self.blocks_per_stage,
            "filters_per_stage": self.filters_per_stage,
            "block_type": self.block_type,
            "kernel_regularizer": keras.regularizers.serialize(
                self.kernel_regularizer) if self.kernel_regularizer else None,
            "normalization_type": self.normalization_type,
            "activation_type": self.activation_type,
            "include_top": self.include_top,
            "enable_deep_supervision": self.enable_deep_supervision,
            "input_shape": self.input_shape_config,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ResNet":
        """Create model from configuration.

        Args:
            config: Configuration dictionary.

        Returns:
            ResNet model instance.
        """
        if config.get("kernel_regularizer"):
            config["kernel_regularizer"] = keras.regularizers.deserialize(
                config["kernel_regularizer"]
            )
        return cls(**config)


# ---------------------------------------------------------------------
# Utility Functions for Deep Supervision
# ---------------------------------------------------------------------


def get_model_output_info(model: keras.Model) -> Dict[str, Any]:
    """Get information about model outputs for deep supervision models.

    Args:
        model: Keras model to analyze.

    Returns:
        Dictionary containing output information:
        - 'num_outputs': Number of outputs
        - 'has_deep_supervision': Whether model has multiple outputs
        - 'output_shapes': List of output shapes
        - 'primary_output_index': Index of the primary inference output (always 0)

    Example:
        >>> model = ResNet.from_variant('resnet50', enable_deep_supervision=True)
        >>> info = get_model_output_info(model)
        >>> print(f"Number of outputs: {info['num_outputs']}")
        >>> print(f"Primary output shape: {info['output_shapes'][info['primary_output_index']]}")
    """
    # Handle both single output and multi-output models
    if isinstance(model.output, list):
        num_outputs = len(model.output)
        output_shapes = [output.shape for output in model.output]
        has_deep_supervision = True
    else:
        num_outputs = 1
        output_shapes = [model.output.shape]
        has_deep_supervision = False

    return {
        'num_outputs': num_outputs,
        'has_deep_supervision': has_deep_supervision,
        'output_shapes': output_shapes,
        'primary_output_index': 0  # Primary output is always at index 0
    }


def create_inference_model_from_training_model(training_model: keras.Model) -> keras.Model:
    """Create a single-output inference model from a multi-output training model.

    Args:
        training_model: Multi-output training model with deep supervision.

    Returns:
        Single-output model using only the primary output (index 0).

    Example:
        >>> # Create training model with deep supervision
        >>> training_model = ResNet.from_variant('resnet50', enable_deep_supervision=True)
        >>>
        >>> # Create inference model (single output)
        >>> inference_model = create_inference_model_from_training_model(training_model)
    """
    model_info = get_model_output_info(training_model)

    if not model_info['has_deep_supervision']:
        logger.info("Model already has single output, returning as-is")
        return training_model

    # Extract only the primary output (index 0)
    primary_output = training_model.output[model_info['primary_output_index']]

    # Create new model with single output
    inference_model = keras.Model(
        inputs=training_model.input,
        outputs=primary_output,
        name=f"{training_model.name}_inference"
    )

    logger.info(f"Created inference model with single output shape: {primary_output.shape}")

    return inference_model


# ---------------------------------------------------------------------


def create_resnet(
        variant: str = "resnet50",
        num_classes: int = 1000,
        input_shape: Optional[Tuple[int, ...]] = (224, 224, 3),
        pretrained: Union[bool, str] = False,
        weights_dataset: str = "imagenet",
        weights_input_shape: Optional[Tuple[int, ...]] = None,
        cache_dir: Optional[str] = None,
        **kwargs
) -> ResNet:
    """Convenience function to create ResNet models.

    Args:
        variant: String, model variant ("resnet18", "resnet34", "resnet50",
            "resnet101", "resnet152").
        num_classes: Integer, number of output classes.
        input_shape: Tuple, input shape.
        pretrained: Boolean or string. If True, loads pretrained weights.
            If string, treats it as a path to local weights file.
        weights_dataset: String, dataset for pretrained weights.
        weights_input_shape: Tuple, input shape used during weight pretraining.
        cache_dir: Optional string, directory to cache downloaded weights.
        **kwargs: Additional arguments passed to the model constructor.

    Returns:
        ResNet model instance.

    Example:
        >>> # Create ResNet-50 with pretrained ImageNet weights
        >>> model = create_resnet("resnet50", pretrained=True)
        >>>
        >>> # Create ResNet-34 as feature extractor
        >>> model = create_resnet("resnet34", pretrained=True, include_top=False)
        >>>
        >>> # Fine-tune on CIFAR-10 with deep supervision
        >>> model = create_resnet("resnet18", num_classes=10,
        ...                       input_shape=(32, 32, 3),
        ...                       enable_deep_supervision=True)
    """
    model = ResNet.from_variant(
        variant,
        num_classes=num_classes,
        input_shape=input_shape,
        pretrained=pretrained,
        weights_dataset=weights_dataset,
        weights_input_shape=weights_input_shape,
        cache_dir=cache_dir,
        **kwargs
    )

    return model


# ---------------------------------------------------------------------