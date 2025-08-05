"""
Keras FractalNet Implementation for Image Classification.

This module provides a flexible and fully Keras-compliant implementation of
FractalNet, a self-similar deep neural network architecture that builds depth
through recursive fractal expansion rather than residual connections.

FractalNet introduces a design strategy based on self-similarity where repeated
application of a simple expansion rule generates deep networks whose structural
layouts are precisely truncated fractals. Unlike ResNets which rely on skip
connections, FractalNet uses multiple interacting subpaths of different lengths
without any pass-through shortcuts.

The key components are:

1. **Base Block**: A simple convolutional block (Conv → BatchNorm → ReLU)
2. **Fractal Expansion**: Recursive rule F_{k+1}(x) = 0.5 * (DP(F_k(x)) + DP(F_k(x)))
3. **Drop-Path Regularization**: Randomly drops entire subpaths during training

Key advantages:
- Adaptive depth transitions during training (shallow to deep)
- Implicit deep supervision without auxiliary classifiers
- Strong regularization via path-level dropout
- Anytime prediction capability
- Parameter efficiency through self-similarity

Typical Usage:
    >>>
    >>> # Create FractalNet for CIFAR-10
    >>> model = create_fractal_net(
    ...     input_shape=(32, 32, 3),
    ...     num_classes=10,
    ...     depths=[2, 3, 3],
    ...     filters=[32, 64, 128],
    ...     drop_path_rate=0.15
    ... )
    >>>
    >>> model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    >>> model.fit(train_data, epochs=100)
"""

import keras
from typing import Optional, Tuple, Union, Dict, Any, List, Callable

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..utils.logger import logger
from ..layers.convblock import ConvBlock
from ..layers.fractal_block import FractalBlock

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class FractalNet(keras.Model):
    """Complete FractalNet model for image classification.

    FractalNet builds depth through recursive fractal expansion rather than
    residual connections. The architecture consists of multiple stages, each
    containing a fractal block with increasing depth.

    Args:
        num_classes: Integer, number of output classes.
        input_shape: Optional tuple specifying input image shape (H, W, C).
        depths: List of integers specifying fractal depth for each stage.
        filters: List of integers specifying number of filters for each stage.
        strides: List of integers specifying strides for each stage.
        drop_path_rate: Float, drop-path probability. Defaults to 0.15.
        dropout_rate: Float, dropout rate in conv blocks. Defaults to 0.1.
        use_batch_norm: Boolean, whether to use batch normalization. Defaults to True.
        kernel_initializer: String or initializer for conv layers. Defaults to "he_normal".
        kernel_regularizer: String or regularizer for conv layers. Defaults to None.
        activation: String or callable, activation function. Defaults to "relu".
        global_pool: String, global pooling type ("avg" or "max"). Defaults to "avg".
        classifier_dropout: Float, dropout rate before final dense layer. Defaults to 0.2.
        name: Optional string name for the model. Defaults to "fractal_net".
        **kwargs: Additional keyword arguments.
    """

    def __init__(
            self,
            num_classes: int,
            input_shape: Optional[Tuple[int, int, int]] = None,
            depths: List[int] = None,
            filters: List[int] = None,
            strides: List[int] = None,
            drop_path_rate: float = 0.15,
            dropout_rate: float = 0.1,
            use_batch_norm: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            activation: Union[str, callable] = "relu",
            global_pool: str = "avg",
            classifier_dropout: float = 0.2,
            name: Optional[str] = "fractal_net",
            **kwargs: Any
    ) -> None:
        """Initialize the FractalNet model.

        Args:
            num_classes: Number of output classes.
            input_shape: Input image shape (H, W, C).
            depths: Fractal depths for each stage.
            filters: Filter counts for each stage.
            strides: Strides for each stage.
            drop_path_rate: Drop-path probability.
            dropout_rate: Dropout rate in conv blocks.
            use_batch_norm: Whether to use batch normalization.
            kernel_initializer: Weight initializer.
            kernel_regularizer: Weight regularizer.
            activation: Activation function.
            global_pool: Global pooling type.
            classifier_dropout: Dropout rate before classifier.
            name: Model name.
            **kwargs: Additional arguments.
        """
        super().__init__(name=name, **kwargs)

        # Store configuration
        self.num_classes = num_classes
        self._input_shape_arg = input_shape
        self.depths = depths or [2, 3, 3]
        self.filters = filters or [32, 64, 128]
        self.strides = strides or [2, 2, 2]
        self.drop_path_rate = drop_path_rate
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.activation = activation
        self.global_pool = global_pool
        self.classifier_dropout = classifier_dropout

        # Validate inputs
        if len(self.depths) != len(self.filters):
            raise ValueError("depths and filters must have the same length")
        if len(self.strides) != len(self.filters):
            raise ValueError("strides and filters must have the same length")

        # Components will be built in build()
        self.stages = []
        self.global_pooling = None
        self.classifier_dropout_layer = None
        self.classifier = None

        self._build_input_shape = None

        logger.info(f"Initialized FractalNet with num_classes={num_classes}, "
                    f"depths={self.depths}, filters={self.filters}")

        # Build immediately if input_shape is provided
        if self._input_shape_arg is not None:
            self.build((None,) + tuple(self._input_shape_arg))

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Build the FractalNet architecture.

        Args:
            input_shape: Shape of input tensor including batch dimension.
        """
        if self.built:
            return

        self._build_input_shape = input_shape
        if len(input_shape) > 1:
            self._input_shape_arg = tuple(input_shape[1:])

        # Build fractal stages
        current_shape = input_shape
        for i, (depth, num_filters, stride) in enumerate(zip(self.depths, self.filters, self.strides)):
            # Create block function for this stage
            def make_block_fn(filters=num_filters, stage_stride=stride):
                def block_fn():
                    return ConvBlock(
                        filters=filters,
                        strides=stage_stride,
                        use_batch_norm=self.use_batch_norm,
                        dropout_rate=self.dropout_rate,
                        kernel_initializer=self.kernel_initializer,
                        kernel_regularizer=self.kernel_regularizer,
                        activation=self.activation,
                        name=f"stage_{i}_conv_block"
                    )

                return block_fn

            # Create fractal block
            fractal_block = FractalBlock(
                block_fn=make_block_fn(),
                depth=depth,
                drop_path_rate=self.drop_path_rate,
                name=f"fractal_stage_{i}"
            )

            # Build the block and update current shape
            fractal_block.build(current_shape)
            current_shape = fractal_block.compute_output_shape(current_shape)

            self.stages.append(fractal_block)

            logger.info(f"Stage {i}: depth={depth}, filters={num_filters}, "
                        f"stride={stride}, output_shape={current_shape}")

        # Global pooling
        if self.global_pool == "avg":
            self.global_pooling = keras.layers.GlobalAveragePooling2D(name="global_avg_pool")
        elif self.global_pool == "max":
            self.global_pooling = keras.layers.GlobalMaxPooling2D(name="global_max_pool")
        else:
            raise ValueError(f"Unsupported global_pool: {self.global_pool}")

        # Classifier dropout
        if self.classifier_dropout > 0:
            self.classifier_dropout_layer = keras.layers.Dropout(
                self.classifier_dropout,
                name="classifier_dropout"
            )

        # Final classifier
        self.classifier = keras.layers.Dense(
            self.num_classes,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="classifier"
        )

        super().build(input_shape)
        logger.info(f"FractalNet built successfully with input shape: {input_shape}")

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through FractalNet.

        Args:
            inputs: Input tensor.
            training: Whether in training mode.

        Returns:
            Output logits tensor.
        """
        x = inputs

        # Pass through fractal stages
        for stage in self.stages:
            x = stage(x, training=training)

        # Global pooling
        x = self.global_pooling(x)

        # Classifier dropout
        if self.classifier_dropout_layer is not None:
            x = self.classifier_dropout_layer(x, training=training)

        # Final classification
        outputs = self.classifier(x)

        return outputs

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration.

        Returns:
            Configuration dictionary.
        """
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "input_shape": self._input_shape_arg,
            "depths": self.depths,
            "filters": self.filters,
            "strides": self.strides,
            "drop_path_rate": self.drop_path_rate,
            "dropout_rate": self.dropout_rate,
            "use_batch_norm": self.use_batch_norm,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "activation": self.activation,
            "global_pool": self.global_pool,
            "classifier_dropout": self.classifier_dropout,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FractalNet":
        """Create FractalNet from configuration.

        Args:
            config: Configuration dictionary.

        Returns:
            FractalNet instance.
        """
        # Deserialize initializers and regularizers
        if "kernel_initializer" in config and isinstance(config["kernel_initializer"], dict):
            config["kernel_initializer"] = keras.initializers.deserialize(
                config["kernel_initializer"]
            )
        if "kernel_regularizer" in config and isinstance(config["kernel_regularizer"], dict):
            config["kernel_regularizer"] = keras.regularizers.deserialize(
                config["kernel_regularizer"]
            )
        return cls(**config)

# ---------------------------------------------------------------------

def create_fractal_net(
        input_shape: Tuple[int, int, int],
        num_classes: int,
        depths: List[int] = None,
        filters: List[int] = None,
        strides: List[int] = None,
        optimizer: Union[str, keras.optimizers.Optimizer] = "adam",
        learning_rate: float = 0.001,
        loss: Union[str, keras.losses.Loss] = "sparse_categorical_crossentropy",
        metrics: List[Union[str, keras.metrics.Metric]] = None,
        **kwargs
) -> FractalNet:
    """Create and compile a FractalNet model.

    Factory function that creates a FractalNet model with specified parameters
    and compiles it with the given optimizer and loss function.

    Args:
        input_shape: Tuple specifying input image shape (height, width, channels).
        num_classes: Integer, number of output classes.
        depths: List of fractal depths for each stage. Defaults to [2, 3, 3].
        filters: List of filter counts for each stage. Defaults to [32, 64, 128].
        strides: List of strides for each stage. Defaults to [2, 2, 2].
        optimizer: String name or optimizer instance. Defaults to "adam".
        learning_rate: Float, learning rate for optimizer. Defaults to 0.001.
        loss: String name or loss function. Defaults to "sparse_categorical_crossentropy".
        metrics: List of metrics to track. Defaults to ["accuracy"].
        **kwargs: Additional arguments passed to FractalNet constructor.

    Returns:
        Compiled FractalNet model ready for training.

    Example:
        >>> # Create FractalNet for CIFAR-10
        >>> model = create_fractal_net(
        ...     input_shape=(32, 32, 3),
        ...     num_classes=10,
        ...     depths=[2, 3, 4],
        ...     filters=[32, 64, 128],
        ...     drop_path_rate=0.15
        ... )
        >>>
        >>> # Train the model
        >>> history = model.fit(
        ...     train_dataset,
        ...     epochs=100,
        ...     validation_data=val_dataset
        ... )
    """
    # Set defaults
    if depths is None:
        depths = [2, 3, 3]
    if filters is None:
        filters = [32, 64, 128]
    if strides is None:
        strides = [2, 2, 2]
    if metrics is None:
        metrics = ["accuracy"]

    # Create the model
    model = FractalNet(
        num_classes=num_classes,
        input_shape=input_shape,
        depths=depths,
        filters=filters,
        strides=strides,
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

    logger.info(f"Created and compiled FractalNet with input_shape={input_shape}, "
                f"num_classes={num_classes}, depths={depths}, filters={filters}")

    return model

# ---------------------------------------------------------------------
