"""
FractalNet Model Implementation
===========================================================

A complete implementation of the FractalNet architecture using modern Keras 3 patterns.
FractalNet is a self-similar deep neural network architecture that builds depth
through recursive fractal expansion rather than residual connections.

Based on: "FractalNet: Ultra-Deep Neural Networks without Residuals" (Larsson et al., 2016)
https://arxiv.org/abs/1605.07648

Key Features:
------------
- Modular design using FractalBlock as building blocks
- Self-similar architecture through recursive fractal expansion
- Drop-path regularization for improved generalization
- Support for multiple FractalNet variants
- Configurable depths and filter sizes per stage
- Complete serialization support with modern Keras 3 patterns
- Production-ready implementation

Architecture Concept:
-------------------
FractalNet uses a recursive expansion rule: F_{k+1}(x) = 0.5 * (DP(F_k(x)) + DP(F_k(x)))
where DP represents drop-path regularization. This creates self-similar structures
without residual connections.

Model Variants:
--------------
- FractalNet-Micro: [1, 2, 2] depths, [16, 32, 64] filters (for small datasets)
- FractalNet-Small: [2, 3, 3] depths, [32, 64, 128] filters (CIFAR-10/100)
- FractalNet-Medium: [3, 4, 4] depths, [64, 128, 256] filters
- FractalNet-Large: [4, 5, 5] depths, [96, 192, 384] filters (ImageNet)

Usage Examples:
-------------
```python
# CIFAR-10 model (32x32 input)
model = FractalNet.from_variant("small", num_classes=10, input_shape=(32, 32, 3))

# MNIST model (28x28 input)
model = FractalNet.from_variant("micro", num_classes=10, input_shape=(28, 28, 1))

# ImageNet model (224x224 input)
model = FractalNet.from_variant("large", num_classes=1000)

# Custom dataset model
model = create_fractal_net("medium", num_classes=100, input_shape=(64, 64, 3))
```
"""

import keras
from typing import List, Optional, Union, Tuple, Dict, Any, Callable

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..utils.logger import logger
from ..layers.convblock import ConvBlock
from ..layers.fractal_block import FractalBlock

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class FractalNet(keras.Model):
    """FractalNet model implementation using modern Keras 3 patterns.

    FractalNet builds depth through recursive fractal expansion rather than
    residual connections. The architecture consists of multiple stages, each
    containing a fractal block with increasing complexity.

    Args:
        num_classes: Integer, number of output classes for classification.
            Only used if include_top=True.
        depths: List of integers, number of fractal depths for each stage.
            Default is [2, 3, 3] for FractalNet-Small.
        filters: List of integers, number of filters for each stage.
            Default is [32, 64, 128] for FractalNet-Small.
        strides: List of integers, strides for each stage downsampling.
            Default is [2, 2, 2].
        drop_path_rate: Float, drop-path probability for regularization.
            Default is 0.15.
        dropout_rate: Float, dropout rate in conv blocks.
            Default is 0.1.
        use_batch_norm: Boolean, whether to use batch normalization.
            Default is True.
        kernel_initializer: String or initializer for conv layers.
            Default is "he_normal".
        kernel_regularizer: String or regularizer for conv layers.
            Default is None.
        activation: String or callable, activation function.
            Default is "relu".
        global_pool: String, global pooling type ("avg" or "max").
            Default is "avg".
        classifier_dropout: Float, dropout rate before final dense layer.
            Default is 0.2.
        include_top: Boolean, whether to include the classification head.
            Default is True.
        input_shape: Tuple, input shape. If None and include_top=True,
            uses (32, 32, 3) for CIFAR. Must be provided for other inputs.
        **kwargs: Additional keyword arguments for the Model base class.

    Raises:
        ValueError: If depths and filters have different lengths.
        ValueError: If invalid model configuration is provided.

    Example:
        >>> # Create FractalNet-Small for CIFAR-10
        >>> model = FractalNet.from_variant("small", num_classes=10, input_shape=(32, 32, 3))
        >>>
        >>> # Create FractalNet-Micro for MNIST
        >>> model = FractalNet.from_variant("micro", num_classes=10, input_shape=(28, 28, 1))
        >>>
        >>> # Create standard CIFAR model
        >>> model = FractalNet.from_variant("small", num_classes=10)
    """

    # Model variant configurations
    MODEL_VARIANTS = {
        "micro": {"depths": [1, 2, 2], "filters": [16, 32, 64]},
        "small": {"depths": [2, 3, 3], "filters": [32, 64, 128]},
        "medium": {"depths": [3, 4, 4], "filters": [64, 128, 256]},
        "large": {"depths": [4, 5, 5], "filters": [96, 192, 384]},
    }

    # Architecture constants
    DEFAULT_KERNEL_SIZE = 3
    DEFAULT_ACTIVATION = "relu"
    DEFAULT_INITIALIZER = "he_normal"

    def __init__(
        self,
        num_classes: int = 10,
        depths: List[int] = [2, 3, 3],
        filters: List[int] = [32, 64, 128],
        strides: List[int] = [2, 2, 2],
        drop_path_rate: float = 0.15,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        activation: str = "relu",
        global_pool: str = "avg",
        classifier_dropout: float = 0.2,
        include_top: bool = True,
        input_shape: Tuple[int, ...] = (32, 32, 3),
        **kwargs
    ):
        # Validate configuration
        if len(depths) != len(filters):
            raise ValueError(
                f"Length of depths ({len(depths)}) must equal length of filters ({len(filters)})"
            )

        if len(strides) != len(filters):
            raise ValueError(
                f"Length of strides ({len(strides)}) must equal length of filters ({len(filters)})"
            )

        if len(depths) < 1:
            raise ValueError("At least one stage is required")

        if input_shape is None:
            input_shape = (32, 32, 3)

        # Store configuration
        self.num_classes = num_classes
        self.depths = depths
        self.filters = filters
        self.strides = strides
        self.drop_path_rate = drop_path_rate
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.activation = activation
        self.global_pool = global_pool
        self.classifier_dropout = classifier_dropout
        self.include_top = include_top
        self._input_shape = input_shape

        # Validate input shape
        if len(input_shape) != 3:
            raise ValueError(f"input_shape must be 3D, got {input_shape}")

        height, width, channels = input_shape

        if channels not in [1, 3]:
            logger.warning(f"Unusual number of channels: {channels}. FractalNet typically uses 1 or 3 channels")

        # Store actual input shape components
        self.input_height = height
        self.input_width = width
        self.input_channels = channels

        # Build the model using functional API
        inputs = keras.Input(shape=input_shape, name="input")
        outputs = self._build_model(inputs)

        # Initialize the Model
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)

        logger.info(
            f"Created FractalNet model for input {input_shape} "
            f"with {sum(depths)} total fractal blocks"
        )

    def _build_model(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Build the complete FractalNet model architecture.

        Args:
            inputs: Input tensor

        Returns:
            Output tensor
        """
        x = inputs

        # Build fractal stages
        for stage_idx in range(len(self.depths)):
            x = self._build_fractal_stage(x, stage_idx)

        # Build classification head if requested
        if self.include_top:
            x = self._build_classification_head(x)

        return x

    def _build_fractal_stage(
        self,
        x: keras.KerasTensor,
        stage_idx: int
    ) -> keras.KerasTensor:
        """Build a fractal stage with specified depth and filters.

        Args:
            x: Input tensor
            stage_idx: Index of the current stage

        Returns:
            Processed tensor after the fractal stage
        """
        depth = self.depths[stage_idx]
        num_filters = self.filters[stage_idx]
        stride = self.strides[stage_idx]

        # Create block function for this stage
        def create_conv_block() -> ConvBlock:
            """Factory function to create conv blocks for fractal expansion."""
            return ConvBlock(
                filters=num_filters,
                strides=stride,
                use_batch_norm=self.use_batch_norm,
                dropout_rate=self.dropout_rate,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                activation=self.activation,
                name=f"stage_{stage_idx}_conv_block"
            )

        # Create and apply fractal block
        fractal_block = FractalBlock(
            block_fn=create_conv_block,
            depth=depth,
            drop_path_rate=self.drop_path_rate,
            name=f"fractal_stage_{stage_idx}"
        )

        x = fractal_block(x)

        logger.info(f"Stage {stage_idx}: depth={depth}, filters={num_filters}, stride={stride}")

        return x

    def _build_classification_head(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Build the classification head.

        Args:
            x: Input feature tensor

        Returns:
            Classification logits
        """
        # Global pooling
        if self.global_pool == "avg":
            x = keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
        elif self.global_pool == "max":
            x = keras.layers.GlobalMaxPooling2D(name="global_max_pool")(x)
        else:
            raise ValueError(f"Unsupported global_pool: {self.global_pool}")

        # Classifier dropout
        if self.classifier_dropout > 0:
            x = keras.layers.Dropout(
                self.classifier_dropout,
                name="classifier_dropout"
            )(x)

        # Final classifier
        if self.num_classes > 0:
            x = keras.layers.Dense(
                self.num_classes,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name="classifier"
            )(x)

        return x

    @classmethod
    def from_variant(
        cls,
        variant: str,
        num_classes: int = 10,
        input_shape: Optional[Tuple[int, ...]] = None,
        **kwargs
    ) -> "FractalNet":
        """Create a FractalNet model from a predefined variant.

        Args:
            variant: String, one of "micro", "small", "medium", "large"
            num_classes: Integer, number of output classes
            input_shape: Tuple, input shape. If None, uses (32, 32, 3)
            **kwargs: Additional arguments passed to the constructor

        Returns:
            FractalNet model instance

        Raises:
            ValueError: If variant is not recognized

        Example:
            >>> # CIFAR-10 model
            >>> model = FractalNet.from_variant("small", num_classes=10, input_shape=(32, 32, 3))
            >>> # MNIST model
            >>> model = FractalNet.from_variant("micro", num_classes=10, input_shape=(28, 28, 1))
            >>> # Default CIFAR model
            >>> model = FractalNet.from_variant("small", num_classes=10)
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        config = cls.MODEL_VARIANTS[variant]

        if input_shape is None:
            input_shape = (32, 32, 3)

        logger.info(f"Creating FractalNet-{variant.upper()} model")
        logger.info(f"from_variant received input_shape: {input_shape}")

        return cls(
            num_classes=num_classes,
            depths=config["depths"],
            filters=config["filters"],
            input_shape=input_shape,
            **kwargs
        )

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization.

        Returns:
            Configuration dictionary
        """
        config = {
            "num_classes": self.num_classes,
            "depths": self.depths,
            "filters": self.filters,
            "strides": self.strides,
            "drop_path_rate": self.drop_path_rate,
            "dropout_rate": self.dropout_rate,
            "use_batch_norm": self.use_batch_norm,
            "kernel_initializer": keras.initializers.serialize(
                keras.initializers.get(self.kernel_initializer)
            ),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "activation": self.activation,
            "global_pool": self.global_pool,
            "classifier_dropout": self.classifier_dropout,
            "include_top": self.include_top,
            "input_shape": self._input_shape,
        }
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FractalNet":
        """Create model from configuration.

        Args:
            config: Configuration dictionary

        Returns:
            FractalNet model instance
        """
        # Deserialize initializers and regularizers
        if config.get("kernel_initializer"):
            config["kernel_initializer"] = keras.initializers.deserialize(
                config["kernel_initializer"]
            )
        if config.get("kernel_regularizer"):
            config["kernel_regularizer"] = keras.regularizers.deserialize(
                config["kernel_regularizer"]
            )

        return cls(**config)

    def summary(self, **kwargs):
        """Print model summary with additional information."""
        super().summary(**kwargs)

        # Print additional model information
        total_blocks = sum(self.depths)
        logger.info(f"FractalNet configuration:")
        logger.info(f"  - Input shape: ({self.input_height}, {self.input_width}, {self.input_channels})")
        logger.info(f"  - Stages: {len(self.depths)}")
        logger.info(f"  - Depths: {self.depths}")
        logger.info(f"  - Filters: {self.filters}")
        logger.info(f"  - Total fractal blocks: {total_blocks}")
        logger.info(f"  - Drop path rate: {self.drop_path_rate}")
        logger.info(f"  - Include top: {self.include_top}")
        if self.include_top:
            logger.info(f"  - Number of classes: {self.num_classes}")

# ---------------------------------------------------------------------

def create_fractal_net(
    variant: str = "small",
    num_classes: int = 10,
    input_shape: Optional[Tuple[int, ...]] = None,
    optimizer: Union[str, keras.optimizers.Optimizer] = "adam",
    learning_rate: float = 0.001,
    loss: Union[str, keras.losses.Loss] = "sparse_categorical_crossentropy",
    metrics: List[Union[str, keras.metrics.Metric]] = None,
    **kwargs
) -> FractalNet:
    """Convenience function to create and compile FractalNet models.

    Args:
        variant: String, model variant ("micro", "small", "medium", "large")
        num_classes: Integer, number of output classes
        input_shape: Tuple, input shape. If None, uses (32, 32, 3)
        optimizer: String name or optimizer instance. Default is "adam"
        learning_rate: Float, learning rate for optimizer. Default is 0.001
        loss: String name or loss function. Default is "sparse_categorical_crossentropy"
        metrics: List of metrics to track. Default is ["accuracy"]
        **kwargs: Additional arguments passed to the model constructor

    Returns:
        Compiled FractalNet model ready for training

    Example:
        >>> # Create FractalNet-Small for CIFAR-10
        >>> model = create_fractal_net("small", num_classes=10, input_shape=(32, 32, 3))
        >>>
        >>> # Create FractalNet-Micro for MNIST
        >>> model = create_fractal_net("micro", num_classes=10, input_shape=(28, 28, 1))
        >>>
        >>> # Create FractalNet-Large for ImageNet
        >>> model = create_fractal_net("large", num_classes=1000, input_shape=(224, 224, 3))
    """
    if metrics is None:
        metrics = ["accuracy"]

    if input_shape is None:
        input_shape = (32, 32, 3)

    # Create the model
    model = FractalNet.from_variant(
        variant=variant,
        num_classes=num_classes,
        input_shape=input_shape,
        **kwargs
    )

    # Set up optimizer
    if isinstance(optimizer, str):
        optimizer_instance = keras.optimizers.get(optimizer)
        if hasattr(optimizer_instance, 'learning_rate'):
            optimizer_instance.learning_rate = learning_rate
    else:
        optimizer_instance = optimizer

    # Compile the model
    model.compile(
        optimizer=optimizer_instance,
        loss=loss,
        metrics=metrics
    )

    logger.info(f"Created and compiled FractalNet-{variant.upper()} with input_shape={input_shape}, "
                f"num_classes={num_classes}")

    return model

# ---------------------------------------------------------------------