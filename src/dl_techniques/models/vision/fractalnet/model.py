"""FractalNet: a very deep classifier built by recursive expansion, with no
residual connection anywhere.

Builds :class:`FractalNet`, a Keras 3 functional model, and the
``create_fractal_net`` convenience factory.

Residual connections are not the only way to make very deep networks
trainable. FractalNet's expansion rule builds short paths from input to loss
explicitly: ``f_{C+1}(z) = [f_C(f_C(z))] join [conv(z)]``, so a level-C
fractal contains paths of length 1, 2, 4, ..., 2^(C-1) that all reach the
same output. The long path gives capacity, the short path keeps it
trainable, and the join over both substitutes for an identity shortcut.
Drop-path regularizes by dropping each join input independently per sample,
always keeping at least one survivor: a coin flip revives one path when
both are dropped.

Each fractal stage runs at constant resolution; downsampling happens
between stages via max-pooling, since a stride inside a stage would
desynchronize the short and long branches. The classification head emits
raw logits, not probabilities.

References:
    - Larsson et al., 2017. FractalNet: Ultra-Deep Neural Networks without
      Residuals. ICLR. (https://arxiv.org/abs/1605.07648)
    - Huang et al., 2016. Deep Networks with Stochastic Depth.
      (https://arxiv.org/abs/1603.09382)
    - Veit et al., 2016. Residual Networks Behave Like Ensembles of Relatively
      Shallow Networks. (https://arxiv.org/abs/1605.06431)
    - He et al., 2015. Deep Residual Learning for Image Recognition.
      (https://arxiv.org/abs/1512.03385)
"""

import keras
from typing import List, Optional, Union, Tuple, Dict, Any, Sequence

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.fractal_block import FractalBlock
from dl_techniques.layers.standard_blocks import ConvBlock
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.fractalnet.model")
class FractalNet(keras.Model):
    """FractalNet model, built with the Keras functional API.

    Builds depth through recursive fractal expansion rather than residual
    connections. The architecture is a sequence of stages, each a fractal
    block whose branches double in depth stage by stage.

    Architecture:

    .. code-block:: text

        image [B, H, W, C]
            |
        stage_0 .. stage_{N-1}   FractalBlock(depth_i) -> MaxPool(stride_i)
            |
        GlobalPooling            (avg or max)
            |
        Dropout(classifier_dropout_rate)          (optional)
            |
        Dense(num_classes)       raw logits        ('include_top' only)
            |
        [B, num_classes] or [B, filters[-1]]

    Fractal block, depth k:

    .. code-block:: text

        z ---------------------------+
            |                        |
        depth-(k-1) fractal block    |
            |                        |
        depth-(k-1) fractal block    conv (ConvBlock)
            |                        |
            +----------- join -------+
                (drop-path average, at least one survivor)

    :param num_classes: Integer, number of output classes for classification.
        Only used if include_top=True.
    :param depths: List of integers, number of fractal depths for each stage.
        Default is [2, 3, 3] for FractalNet-Small.
    :param filters: List of integers, number of filters for each stage.
        Default is [32, 64, 128] for FractalNet-Small.
    :param strides: List of integers, strides for each stage downsampling.
        Default is [2, 2, 2].
    :param drop_path_rate: Float, drop-path probability for regularization.
        Default is 0.15.
    :param dropout_rate: Float, dropout rate in conv blocks.
        Default is 0.1.
    :param normalization_type: String, type of normalization to use in conv blocks.
        Default is "batch_norm".
    :param activation_type: String, type of activation to use in conv blocks.
        Default is "relu".
    :param kernel_initializer: String or initializer for conv layers.
        Default is "he_normal".
    :param kernel_regularizer: String or regularizer for conv layers.
        Default is None.
    :param global_pool: String, global pooling type ("avg" or "max").
        Default is "avg".
    :param classifier_dropout_rate: Float, dropout rate before final dense layer.
        Default is 0.2.
    :param include_top: Boolean, whether to include the classification head.
        Default is True.
    :param input_shape: Tuple, input shape. If None and include_top=True,
        uses (32, 32, 3) for CIFAR. Must be provided for other inputs.
    :param **kwargs: Additional keyword arguments for the Model base class.

    :raises ValueError: If depths and filters have different lengths.
    :raises ValueError: If invalid model configuration is provided.

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
        depths: Sequence[int] = (2, 3, 3),
        filters: Sequence[int] = (32, 64, 128),
        strides: Sequence[int] = (2, 2, 2),
        drop_path_rate: float = 0.15,
        dropout_rate: float = 0.1,
        normalization_type: str = "batch_norm",
        activation_type: str = "relu",
        kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        global_pool: str = "avg",
        classifier_dropout_rate: float = 0.2,
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
        # DECISION plan-2026-08-19T163559-499b6f0e/D-085: store as a list even
        # though the default is a tuple, matching what `get_config` has always
        # emitted. See decisions.md.
        self.depths = list(depths)
        self.filters = list(filters)
        self.strides = list(strides)
        self.drop_path_rate = drop_path_rate
        self.dropout_rate = dropout_rate
        self.normalization_type = normalization_type
        self.activation_type = activation_type
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.global_pool = global_pool
        self.classifier_dropout_rate = classifier_dropout_rate
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

        :param inputs: Input tensor

        :return: Output tensor
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

        :param x: Input tensor
        :param stage_idx: Index of the current stage

        :return: Processed tensor after the fractal stage
        """
        depth = self.depths[stage_idx]
        num_filters = self.filters[stage_idx]
        stride = self.strides[stage_idx]

        # The fractal itself runs at constant resolution (stride 1): its deep
        # branch applies the base block 2^(depth-1) times, so a stride inside
        # it would desynchronize the deep and shallow branches. Downsampling
        # happens between stages instead, via the pooling below.
        conv_block = ConvBlock(
            filters=num_filters,
            kernel_size=self.DEFAULT_KERNEL_SIZE,
            strides=1,
            padding="same",
            normalization_type=self.normalization_type,
            activation_type=self.activation_type,
            dropout_rate=self.dropout_rate,
            use_pooling=False,  # No pooling in fractal blocks
            kernel_regularizer=self.kernel_regularizer,
            kernel_initializer=self.kernel_initializer,
        )
        block_config = conv_block.get_config()

        fractal_block = FractalBlock(
            block_config=block_config,
            depth=depth,
            drop_path_rate=self.drop_path_rate,
            name=f"fractal_stage_{stage_idx}"
        )
        x = fractal_block(x)

        if stride > 1:
            x = keras.layers.MaxPooling2D(
                pool_size=stride,
                strides=stride,
                padding="same",
                name=f"fractal_pool_{stage_idx}"
            )(x)

        logger.info(f"Stage {stage_idx}: depth={depth}, filters={num_filters}, stride={stride}")

        return x

    def _build_classification_head(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Build the classification head.

        :param x: Input feature tensor

        :return: Classification logits
        """
        # Global pooling
        if self.global_pool == "avg":
            x = keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
        elif self.global_pool == "max":
            x = keras.layers.GlobalMaxPooling2D(name="global_max_pool")(x)
        else:
            raise ValueError(f"Unsupported global_pool: {self.global_pool}")

        # Classifier dropout
        if self.classifier_dropout_rate > 0:
            x = keras.layers.Dropout(
                self.classifier_dropout_rate,
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

        :param variant: String, one of "micro", "small", "medium", "large"
        :param num_classes: Integer, number of output classes
        :param input_shape: Tuple, input shape. If None, uses (32, 32, 3)
        :param **kwargs: Additional arguments passed to the constructor

        :return: FractalNet model instance

        :raises ValueError: If variant is not recognized

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

        # DECISION plan-2026-08-19T163559-499b6f0e/D-127: copy the preset before
        # updating it with kwargs; updating the shared MODEL_VARIANTS dict in
        # place would poison it for every later caller. See decisions.md.
        config = cls.MODEL_VARIANTS[variant].copy()
        config.pop("description", None)
        config.update(kwargs)

        if input_shape is None:
            input_shape = (32, 32, 3)

        logger.info(f"Creating FractalNet-{variant.upper()} model")
        logger.info(f"from_variant received input_shape: {input_shape}")

        return cls(
            num_classes=num_classes,
            input_shape=input_shape,
            **config
        )

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization.

        :return: Configuration dictionary
        """
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "depths": self.depths,
            "filters": self.filters,
            "strides": self.strides,
            "drop_path_rate": self.drop_path_rate,
            "dropout_rate": self.dropout_rate,
            "normalization_type": self.normalization_type,
            "activation_type": self.activation_type,
            "kernel_initializer": keras.initializers.serialize(
                keras.initializers.get(self.kernel_initializer)
            ),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "global_pool": self.global_pool,
            "classifier_dropout_rate": self.classifier_dropout_rate,
            "include_top": self.include_top,
            "input_shape": self._input_shape,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FractalNet":
        """Create model from configuration.

        :param config: Configuration dictionary

        :return: FractalNet model instance
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
        logger.info(f"  - Normalization: {self.normalization_type}")
        logger.info(f"  - Activation: {self.activation_type}")
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
    loss: Optional[Union[str, keras.losses.Loss]] = None,
    metrics: List[Union[str, keras.metrics.Metric]] = None,
    **kwargs
) -> FractalNet:
    """Convenience function to create and compile FractalNet models.

    :param variant: String, model variant ("micro", "small", "medium", "large")
    :param num_classes: Integer, number of output classes
    :param input_shape: Tuple, input shape. If None, uses (32, 32, 3)
    :param optimizer: String name or optimizer instance. Default is "adam"
    :param learning_rate: Float, learning rate for optimizer. Default is 0.001
    :param loss: String name or loss object. Defaults to
        ``SparseCategoricalCrossentropy(from_logits=True)`` — the head emits
        raw logits, so the string ``"sparse_categorical_crossentropy"``
        (which is ``from_logits=False``) would silently mis-train.
    :param metrics: List of metrics to track. Default is ["accuracy"]
    :param **kwargs: Additional arguments passed to the model constructor

    :return: Compiled FractalNet model ready for training

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

    if loss is None:
        # The head emits raw logits, not probabilities. The string
        # "sparse_categorical_crossentropy" resolves to from_logits=False and
        # would mis-train silently, so use the configured object instead.
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

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