"""
ResNet Model Implementation with Deep Supervision
=================================================

The ResNet architecture with optional deep supervision. With deep supervision
enabled the model returns `[final_output, stage3, stage2, stage1]`; otherwise a
single tensor.

Based on: "Deep Residual Learning for Image Recognition" (He et al., 2015)
https://arxiv.org/abs/1512.03385

Model Variants:
--------------
- ResNet-18: [2, 2, 2, 2] blocks, [64, 128, 256, 512] filters, BasicBlock
- ResNet-34: [3, 4, 6, 3] blocks, [64, 128, 256, 512] filters, BasicBlock
- ResNet-50: [3, 4, 6, 3] blocks, [64, 128, 256, 512] filters, BottleneckBlock
- ResNet-101: [3, 4, 23, 3] blocks, [64, 128, 256, 512] filters, BottleneckBlock
- ResNet-152: [3, 8, 36, 3] blocks, [64, 128, 256, 512] filters, BottleneckBlock

No pretrained ResNet weights are distributed with `dl_techniques`; `pretrained=True`
raises `NotImplementedError`. Pass a local path instead: `pretrained="/path/to.keras"`.

Usage Examples:
-------------
```python
# Create model with deep supervision for training
model = ResNet.from_variant("resnet50", num_classes=1000, enable_deep_supervision=True)

# Feature extractor from a local checkpoint
model = create_resnet("resnet34", pretrained="/path/to.keras", include_top=False)

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
        >>> # Load as feature extractor from a local checkpoint
        >>> model = ResNet.from_variant("resnet34", pretrained="/path/to.keras",
        ...                             include_top=False)
    """

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

    def __init__(
            self,
            num_classes: int = 1000,
            blocks_per_stage: Optional[List[int]] = None,
            filters_per_stage: Optional[List[int]] = None,
            block_type: Literal["basic", "bottleneck"] = "bottleneck",
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            normalization_type: str = "batch_norm",
            normalization_kwargs: Optional[Dict[str, Any]] = None,
            activation_type: str = "relu",
            include_top: bool = True,
            enable_deep_supervision: bool = False,
            input_shape: Tuple[int, ...] = (224, 224, 3),
            **kwargs
    ):
        super().__init__(**kwargs)

        blocks_per_stage = list(blocks_per_stage) if blocks_per_stage is not None else [3, 4, 6, 3]
        filters_per_stage = list(filters_per_stage) if filters_per_stage is not None else [64, 128, 256, 512]

        if len(blocks_per_stage) != len(filters_per_stage):
            raise ValueError(
                f"Length of blocks_per_stage ({len(blocks_per_stage)}) must equal "
                f"length of filters_per_stage ({len(filters_per_stage)})"
            )

        if block_type not in ["basic", "bottleneck"]:
            raise ValueError(
                f"block_type must be 'basic' or 'bottleneck', got '{block_type}'"
            )

        if input_shape is None:
            input_shape = (224, 224, 3)
        if len(input_shape) != 3:
            raise ValueError(f"input_shape must be 3D, got {input_shape}")

        self.num_classes = num_classes
        self.blocks_per_stage = blocks_per_stage
        self.filters_per_stage = filters_per_stage
        self.block_type = block_type
        self.kernel_regularizer = kernel_regularizer
        self.normalization_type = normalization_type
        # DECISION plan_2026-05-18_6776f8ba/D-003
        # Optional `normalization_kwargs` forwarded to every
        # `create_normalization_layer` call inside the stem AND inside every
        # BasicBlock/BottleneckBlock. Default `None` -> `{}` -> all factory
        # calls byte-identical to the pre-plumbing version, preserving
        # bit-exactness for every existing ResNet checkpoint. Used by
        # `src/train/rms_variants_train/experiments/e2_resnet_cifar100.py`
        # in `--mode param_matched` to pass `use_scale=False` so the
        # gamma-removal contrast in the headline E2 result becomes a
        # pure 1-vs-d parameter-count confound rather than a norm choice.
        self.normalization_kwargs = dict(normalization_kwargs) if normalization_kwargs else {}
        self.activation_type = activation_type
        self.include_top = include_top
        self.enable_deep_supervision = enable_deep_supervision
        self.input_shape_config = input_shape
        self.input_height, self.input_width, self.input_channels = input_shape

        self._build_stem()

        self.stages = []
        for stage_idx in range(len(blocks_per_stage)):
            self._build_stage(stage_idx)

        if self.include_top:
            self._build_head()

        self.supervision_heads = []
        if self.enable_deep_supervision and self.include_top:
            self._build_supervision_heads()

        logger.info(
            f"Created ResNet with {sum(blocks_per_stage)} blocks for input "
            f"{input_shape} (deep supervision: {enable_deep_supervision})"
        )

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
            name="stem_bn",
            **self.normalization_kwargs,
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

        BlockClass = BasicBlock if self.block_type == "basic" else BottleneckBlock

        stage_blocks = []

        for block_idx in range(num_blocks):
            stride = 2 if stage_idx > 0 and block_idx == 0 else 1

            # Stage 0's first block projects only to widen channels (bottleneck);
            # every later stage's first block projects to match the stride-2 shortcut.
            use_projection = False
            if block_idx == 0:
                use_projection = (
                    self.block_type == "bottleneck" if stage_idx == 0 else True
                )

            block = BlockClass(
                filters=base_filters,
                stride=stride,
                use_projection=use_projection,
                kernel_regularizer=self.kernel_regularizer,
                normalization_type=self.normalization_type,
                normalization_kwargs=dict(self.normalization_kwargs),
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

        One GAP + Dense head per intermediate stage. Stage 0 is skipped (too
        shallow to supervise); the final stage is served by the main head.
        """
        for stage_idx in range(1, len(self.blocks_per_stage)):
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
        x = self.stem_conv(inputs)
        x = self.stem_bn(x, training=training)
        x = self.stem_act(x)
        x = self.stem_pool(x)

        stage_features = []
        for stage_blocks in self.stages:
            for block in stage_blocks:
                x = block(x, training=training)
            if self.enable_deep_supervision and self.include_top:
                stage_features.append(x)

        if self.include_top:
            final_features = self.gap(x)
            final_output = (
                self.classifier(final_features) if self.classifier else final_features
            )
        else:
            final_output = x

        if self.enable_deep_supervision and self.include_top and self.supervision_heads:
            # Reversed (stage 3, 2, 1) to match the BFUNet output convention.
            supervision_outputs = []
            for sup_head in reversed(self.supervision_heads):
                feat = sup_head["gap"](stage_features[sup_head["stage_idx"]])
                supervision_outputs.append(
                    sup_head["classifier"](feat) if sup_head["classifier"] else feat
                )
            return [final_output] + supervision_outputs

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
                logger.info("Weights loaded with skip_mismatch=True; mismatched layers skipped.")
            else:
                logger.info("All weights loaded successfully.")

        except Exception as e:
            raise ValueError(f"Failed to load weights from {weights_path}: {str(e)}")

    # `_download_weights` raises instead of falling back to random init. The
    # previous version held a `PRETRAINED_WEIGHTS` table of placeholder URLs
    # pointing at a non-existent host; `from_variant` caught the download failure,
    # logged a warning and returned a randomly-initialized model, so
    # `pretrained=True` silently produced untrained weights. Do NOT reinstate a
    # warn-and-return branch here or in `from_variant`. No public ResNet weights
    # are distributed with dl_techniques; pass a local path via
    # `pretrained="/path/to/file.keras"` or use `pretrained=False` (default).
    @staticmethod
    def _download_weights(
            variant: str,
            dataset: str = "imagenet",
            cache_dir: Optional[str] = None
    ) -> str:
        """Resolve a download path for pretrained weights of ``variant``.

        Not implemented: no public ResNet weights ship with ``dl_techniques``.
        Always raises. Kept to mirror the BERT / GPT-2 / WaveFieldLLM factory
        recipe and to give an explicit failure mode instead of a silent
        random-init fallback.

        Args:
            variant: Variant name (unused).
            dataset: Dataset name (unused).
            cache_dir: Cache directory (unused).

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            f"No pretrained ResNet weights are distributed with dl_techniques "
            f"(requested variant '{variant}', dataset '{dataset}'). Pass a local "
            f"checkpoint instead: ResNet.from_variant('{variant}', "
            f"pretrained='/path/to/weights.keras')."
        )

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
            pretrained: If a string, a path to a local weights file to load.
                If True, raises NotImplementedError — no public ResNet weights
                ship with dl_techniques. If False (default), returns a
                randomly-initialized model.
            weights_dataset: String, dataset for pretrained weights.
            weights_input_shape: Tuple, input shape used during weight pretraining.
            cache_dir: Optional string, directory to cache downloaded weights.
            **kwargs: Additional arguments passed to the constructor.

        Returns:
            ResNet model instance.

        Raises:
            ValueError: If variant is not recognized.
            NotImplementedError: If pretrained is True.

        Example:
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

        load_weights_path = None
        skip_mismatch = False

        if pretrained:
            if isinstance(pretrained, str):
                load_weights_path = pretrained
                logger.info(f"Will load weights from local file: {load_weights_path}")
            else:
                load_weights_path = cls._download_weights(
                    variant=variant,
                    dataset=weights_dataset,
                    cache_dir=cache_dir
                )

            # The ImageNet head is 1000-wide; a different num_classes or a
            # different input shape than the checkpoint was trained at means the
            # affected layers must be skipped rather than refused.
            if kwargs.get("include_top", True) and num_classes != 1000:
                skip_mismatch = True
                logger.info(
                    f"num_classes ({num_classes}) differs from the pretrained 1000; "
                    f"classifier weights will be skipped."
                )

            if weights_input_shape and input_shape and weights_input_shape != input_shape:
                logger.info(
                    f"Loading weights pretrained on {weights_input_shape} "
                    f"for model with input shape {input_shape}."
                )
                skip_mismatch = True

        model = cls(
            num_classes=num_classes,
            blocks_per_stage=config["blocks_per_stage"],
            filters_per_stage=config["filters_per_stage"],
            block_type=config["block_type"],
            input_shape=input_shape,
            **kwargs
        )

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
            "normalization_kwargs": dict(self.normalization_kwargs),
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

# Re-exported so callers of `models.resnet` get the deep-supervision helpers
# from the same module as the model itself — `src/train/resnet/train_resnet.py`
# imports `create_resnet` and `get_model_output_info` in one statement. The
# import sits here rather than at the top because it is an API re-export, not a
# dependency of the class above.
from dl_techniques.utils.deep_supervision import (  # noqa: E402
    get_model_output_info,
    create_inference_model_from_training_model,
)


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
        pretrained: If a string, a path to a local weights file. If True, raises
            NotImplementedError — no public ResNet weights ship with
            dl_techniques. If False (default), random initialization.
        weights_dataset: String, dataset for pretrained weights.
        weights_input_shape: Tuple, input shape used during weight pretraining.
        cache_dir: Optional string, directory to cache downloaded weights.
        **kwargs: Additional arguments passed to the model constructor.

    Returns:
        ResNet model instance.

    Raises:
        NotImplementedError: If pretrained is True.

    Example:
        >>> # Create ResNet-34 as a feature extractor
        >>> model = create_resnet("resnet34", include_top=False)
        >>>
        >>> # Fine-tune on CIFAR-10 with deep supervision
        >>> model = create_resnet("resnet18", num_classes=10,
        ...                       input_shape=(32, 32, 3),
        ...                       enable_deep_supervision=True)
    """
    return ResNet.from_variant(
        variant,
        num_classes=num_classes,
        input_shape=input_shape,
        pretrained=pretrained,
        weights_dataset=weights_dataset,
        weights_input_shape=weights_input_shape,
        cache_dir=cache_dir,
        **kwargs
    )


# ---------------------------------------------------------------------