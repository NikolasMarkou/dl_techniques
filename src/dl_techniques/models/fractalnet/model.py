"""
FractalNet: a very deep classifier built by recursive expansion, with no residual
connection anywhere.

FractalNet's premise is that residual connections are not the essential
ingredient in training very deep networks — what matters is that the network
contains SHORT paths from input to loss alongside the long ones. ResNet supplies
those short paths implicitly, by making every block skippable. FractalNet
supplies them explicitly, through an expansion rule that generates branches of
differing lengths and averages them at a join. The rule composes one branch out
of two copies of the previous level while the other branch is a single
convolution:

`f_{C+1}(z) = [f_C(f_C(z))] join [conv(z)]`

so a level-`C` fractal contains paths of length `1, 2, 4, ..., 2^(C-1)` all
reaching the same output, and the shortest is a single layer regardless of how
deep the block is. The long path is what gives capacity; the short path is what
makes it trainable, and the implicit ensemble over both is what substitutes for
the identity shortcut.

The composition is the whole architecture, and it is easy to get wrong in a way
nothing catches. Until 2026-08-14 this implementation applied both sub-blocks to
the SAME input in parallel — `F_k(x) = 0.5 * (F_{k-1}(x) + F_{k-1}(x))` — which
recursion terminates with every leaf receiving the block's own input, so every
path traversed exactly ONE convolution at any `depth`. Parameter count, layer
count and output shape were all unaffected, which is why the suite stayed green.
The instrument that detects it is the RECEPTIVE FIELD: with 3x3 `same`
convolutions a path of `L` composed blocks spans `1 + 2L` pixels, so a correct
depth-`k` block spans `1 + 2 * 2^(k-1)` — 3, 5, 9, 17 — where the parallel
version measured 3 at every depth. That measurement is pinned in
`tests/test_layers/test_fractal_block.py::TestFractalExpansionRule`.

Because the deep branch applies its base block `2^(k-1)` times, the fractal must
run at CONSTANT resolution: a stride inside the block would downsample the deep
branch `2^(k-1)` times against the shallow branch's once and the join would
receive mismatched shapes. `FractalBlock` refuses a strided `block_config` for
that reason, and downsampling happens BETWEEN stages, as max-pooling, which is
where the paper puts it.

Drop-path is LOCAL and renormalized. Each input to a join is dropped by its own
per-sample Bernoulli draw, and the join averages only the SURVIVORS. Critically,
at least one path is always kept: when both draws drop, one is revived by a fair
coin. Without that rescue a join emits exactly zero for that sample — an event
with probability `drop_path_rate ** 2`, about 2.3% at the 0.15 default — and the
zero then propagates through every remaining stage. The paper's *global*
drop-path, which selects one column and runs the whole network through it, and
the alternation between the two regimes, are NOT implemented; a single
`drop_path_rate` applies at every depth with no schedule.

Structurally the model is a plain sequence: `len(depths)` stages, each a
`FractalBlock` over a `ConvBlock` (3x3 convolution, configurable normalization
and activation, dropout) followed by max-pooling where the stage's stride
exceeds 1, then a global pool, dropout and a `Dense` classifier. There is no
stem and no bottleneck. The `ConvBlock` is constructed once purely to harvest
its `get_config()`; that dict is what `FractalBlock` stores and re-instantiates
per leaf, which is what makes the recursive structure serializable and why every
leaf in a stage is configured identically while holding independent weights.

Construction happens in `__init__` through the functional API before
`super().__init__(inputs, outputs)`, so this is a Functional model wearing a
subclass's constructor rather than a subclassed model with a `call`. The head
emits RAW LOGITS — there is no softmax — so `create_fractal_net` defaults its
loss to `SparseCategoricalCrossentropy(from_logits=True)`. It previously
defaulted to the string `"sparse_categorical_crossentropy"`, which resolves to
`from_logits=False` and mis-trained silently.

References:
    - Larsson et al., 2017. FractalNet: Ultra-Deep Neural Networks without
      Residuals. ICLR. (https://arxiv.org/abs/1605.07648)
    - Huang et al., 2016. Deep Networks with Stochastic Depth.
      (https://arxiv.org/abs/1603.09382)
    - Veit et al., 2016. Residual Networks Behave Like Ensembles of Relatively
      Shallow Networks. (https://arxiv.org/abs/1605.06431)
      The path-ensemble reading that motivates both FractalNet and drop-path.
    - He et al., 2015. Deep Residual Learning for Image Recognition.
      (https://arxiv.org/abs/1512.03385)
      The residual baseline FractalNet was posed against.
"""

import keras
from typing import List, Optional, Union, Tuple, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.fractal_block import FractalBlock
from dl_techniques.layers.standard_blocks import ConvBlock

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
        normalization_type: String, type of normalization to use in conv blocks.
            Default is "batch_norm".
        activation_type: String, type of activation to use in conv blocks.
            Default is "relu".
        kernel_initializer: String or initializer for conv layers.
            Default is "he_normal".
        kernel_regularizer: String or regularizer for conv layers.
            Default is None.
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
        normalization_type: str = "batch_norm",
        activation_type: str = "relu",
        kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
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
        self.normalization_type = normalization_type
        self.activation_type = activation_type
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
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

        # The fractal itself runs at CONSTANT resolution, at stride 1. Its deep
        # branch applies the base block 2^(depth-1) times, so a stride inside
        # the block would downsample that branch 2^(depth-1) times against the
        # shallow branch's once and the join would see mismatched shapes.
        # FractalNet downsamples BETWEEN blocks, which is what the pooling below
        # does.
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
            "classifier_dropout": self.classifier_dropout,
            "include_top": self.include_top,
            "input_shape": self._input_shape,
        })
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

    Args:
        variant: String, model variant ("micro", "small", "medium", "large")
        num_classes: Integer, number of output classes
        input_shape: Tuple, input shape. If None, uses (32, 32, 3)
        optimizer: String name or optimizer instance. Default is "adam"
        learning_rate: Float, learning rate for optimizer. Default is 0.001
        loss: String name or loss object. Defaults to
            ``SparseCategoricalCrossentropy(from_logits=True)`` — the head emits
            raw logits, so the string ``"sparse_categorical_crossentropy"``
            (which is ``from_logits=False``) would silently mis-train.
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

    if loss is None:
        # The head emits RAW LOGITS (no softmax). The string
        # "sparse_categorical_crossentropy" resolves to from_logits=False, which
        # would apply a log to values that are not probabilities and mis-train
        # silently -- no error, just a worse model. Default to the configured
        # object instead of the string.
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