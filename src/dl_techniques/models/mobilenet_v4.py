"""MobileNetV4: Comprehensive Implementation and Findings

Key Findings and Architectural Details:
1. Universal Inverted Bottleneck (UIB):
   - Unifies and extends IB, ConvNext, FFN, and introduces ExtraDW variant
   - Allows flexible spatial and channel mixing
   - Provides options to extend receptive field and enhance computational efficiency
   - Four configurations: IB, ConvNext-Like, ExtraDW, and FFN

2. Mobile MQA (Multi-Query Attention):
   - Optimized for accelerators, >39% speedup over MHSA
   - Shares keys and values across heads, reducing memory bandwidth
   - Incorporates asymmetric spatial down-sampling for efficiency

3. Improved NAS (Neural Architecture Search):
   - Two-stage search: coarse-grained (filter sizes) and fine-grained (UIB config)
   - Uses offline distillation dataset to reduce sensitivity to hyper-parameters
   - Extended training to 750 epochs for deeper, higher-quality models

4. Model Variants:
   - MNv4-Conv: Pure convolutional models
   - MNv4-Hybrid: Combines UIB with Mobile MQA
"""

import keras
from keras import regularizers, layers, ops
from typing import List, Tuple, Optional, Any, Dict, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger


# ---------------------------------------------------------------------

class ModelConfig:
    """Configuration class for MobileNetV4 hyperparameters.

    Args:
        input_shape: Input shape of the model (height, width, channels)
        num_classes: Number of output classes
        width_multiplier: Multiplier for the number of filters
        use_attention: Whether to use Mobile MQA in the last stage
        weight_decay: L2 regularization factor
        dropout_rate: Dropout rate for regularization
        kernel_initializer: Initializer for the convolution kernels
    """

    def __init__(
            self,
            input_shape: Tuple[int, int, int] = (224, 224, 3),
            num_classes: int = 1000,
            width_multiplier: float = 1.0,
            use_attention: bool = False,
            weight_decay: float = 1e-5,
            dropout_rate: float = 0.2,
            kernel_initializer: str = "he_normal"
    ):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.width_multiplier = width_multiplier
        self.use_attention = use_attention
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.kernel_initializer = kernel_initializer


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class UIB(layers.Layer):
    """Universal Inverted Bottleneck (UIB) block.

    This block unifies and extends various efficient building blocks:
    Inverted Bottleneck (IB), ConvNext, Feed-Forward Network (FFN), and Extra Depthwise (ExtraDW).

    Args:
        filters: Number of output filters
        expansion_factor: Expansion factor for the block
        stride: Stride for the depthwise convolutions
        kernel_size: Kernel size for depthwise convolutions
        use_dw1: Whether to use the first depthwise convolution
        use_dw2: Whether to use the second depthwise convolution
        block_type: Type of the block ('IB', 'ConvNext', 'ExtraDW', or 'FFN')
        kernel_initializer: Initializer for the convolution kernels
        kernel_regularizer: Regularizer for the convolution kernels
    """

    def __init__(
            self,
            filters: int,
            expansion_factor: int = 4,
            stride: int = 1,
            kernel_size: int = 3,
            use_dw1: bool = False,
            use_dw2: bool = True,
            block_type: str = 'IB',
            kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs: Any
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.expansion_factor = expansion_factor
        self.stride = stride
        self.kernel_size = kernel_size
        self.use_dw1 = use_dw1
        self.use_dw2 = use_dw2
        self.block_type = block_type
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.expanded_filters = filters * expansion_factor

        # Initialize layer attributes to None - will be built in build()
        self.expand_conv = None
        self.bn1 = None
        self.activation = None
        self.dw1 = None
        self.bn_dw1 = None
        self.dw2 = None
        self.bn_dw2 = None
        self.project_conv = None
        self.bn2 = None
        self._build_input_shape = None

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Build the layer weights and sublayers.

        Args:
            input_shape: Shape of the input tensor
        """
        # Store for serialization
        self._build_input_shape = input_shape

        # Layer configurations
        conv_config = {
            "kernel_initializer": self.kernel_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "padding": "same",
            "use_bias": False
        }

        # Build sublayers following proper normalization order: Conv/Linear -> Norm -> Activation
        self.expand_conv = layers.Conv2D(self.expanded_filters, 1, **conv_config)
        self.bn1 = layers.BatchNormalization()
        self.activation = layers.ReLU()

        if self.use_dw1:
            self.dw1 = layers.DepthwiseConv2D(
                self.kernel_size,
                self.stride,
                **conv_config
            )
            self.bn_dw1 = layers.BatchNormalization()

        if self.use_dw2:
            self.dw2 = layers.DepthwiseConv2D(
                self.kernel_size,
                1,
                **conv_config
            )
            self.bn_dw2 = layers.BatchNormalization()

        self.project_conv = layers.Conv2D(self.filters, 1, **conv_config)
        self.bn2 = layers.BatchNormalization()

        # Build sublayers explicitly
        self.expand_conv.build(input_shape)
        self.bn1.build(input_shape[:-1] + (self.expanded_filters,))

        if self.use_dw1:
            dw1_input_shape = input_shape[:-1] + (self.expanded_filters,)
            self.dw1.build(dw1_input_shape)
            dw1_output_shape = self._compute_conv_output_shape(dw1_input_shape, self.stride)
            self.bn_dw1.build(dw1_output_shape)

        if self.use_dw2:
            dw2_input_shape = input_shape[:-1] + (self.expanded_filters,)
            if self.use_dw1:
                dw2_input_shape = dw1_output_shape
            self.dw2.build(dw2_input_shape)
            self.bn_dw2.build(dw2_input_shape)

        project_input_shape = input_shape[:-1] + (self.expanded_filters,)
        self.project_conv.build(project_input_shape)
        self.bn2.build(input_shape[:-1] + (self.filters,))

        super().build(input_shape)

    def _compute_conv_output_shape(self, input_shape: Tuple[int, ...], stride: int) -> Tuple[int, ...]:
        """Compute output shape after convolution with given stride.

        Args:
            input_shape: Input shape
            stride: Convolution stride

        Returns:
            Output shape
        """
        if input_shape[1] is None or input_shape[2] is None:
            return input_shape

        height = input_shape[1] // stride if input_shape[1] is not None else None
        width = input_shape[2] // stride if input_shape[2] is not None else None
        return input_shape[:1] + (height, width) + input_shape[3:]

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Forward pass of the UIB block.

        Args:
            inputs: Input tensor
            training: Whether in training mode

        Returns:
            Output tensor
        """
        x = self.expand_conv(inputs)
        x = self.bn1(x, training=training)
        x = self.activation(x)

        if self.use_dw1:
            x = self.dw1(x)
            x = self.bn_dw1(x, training=training)
            x = self.activation(x)

        if self.use_dw2:
            x = self.dw2(x)
            x = self.bn_dw2(x, training=training)
            x = self.activation(x)

        x = self.project_conv(x)
        x = self.bn2(x, training=training)

        # Residual connection
        if self.stride == 1 and ops.shape(inputs)[-1] == self.filters:
            return inputs + x
        return x

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input

        Returns:
            Output shape
        """
        input_shape_list = list(input_shape)

        # Apply stride to spatial dimensions
        if self.stride > 1:
            if input_shape_list[1] is not None:
                input_shape_list[1] = input_shape_list[1] // self.stride
            if input_shape_list[2] is not None:
                input_shape_list[2] = input_shape_list[2] // self.stride

        # Update channel dimension
        input_shape_list[-1] = self.filters

        return tuple(input_shape_list)

    def get_config(self) -> Dict[str, Any]:
        """Returns the layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration
        """
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "expansion_factor": self.expansion_factor,
            "stride": self.stride,
            "kernel_size": self.kernel_size,
            "use_dw1": self.use_dw1,
            "use_dw2": self.use_dw2,
            "block_type": self.block_type,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get the config needed to build the layer from a config.

        Returns:
            Dictionary containing the build configuration
        """
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build the layer from a config created with get_build_config.

        Args:
            config: Dictionary containing the build configuration
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])


@keras.saving.register_keras_serializable()
class MobileMQA(layers.Layer):
    """Mobile Multi-Query Attention (MQA) block.

    This block implements an efficient attention mechanism optimized for mobile accelerators.
    It uses shared keys and values across heads to reduce memory bandwidth requirements.

    Args:
        dim: Dimension of the input and output tensors
        num_heads: Number of attention heads
        use_downsampling: Whether to use spatial downsampling for keys and values
        kernel_initializer: Initializer for the convolution kernels
        kernel_regularizer: Regularizer for the convolution kernels
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            use_downsampling: bool = False,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs: Any
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_downsampling = use_downsampling
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.scale = self.head_dim ** -0.5

        # Initialize layer attributes to None - will be built in build()
        self.q_proj = None
        self.kv_proj = None
        self.o_proj = None
        self.downsample = None
        self.lambda_param = None
        self._build_input_shape = None

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Build the layer weights and sublayers.

        Args:
            input_shape: Shape of the input tensor
        """
        # Store for serialization
        self._build_input_shape = input_shape

        # Layer configurations
        dense_config = {
            "kernel_initializer": self.kernel_initializer,
            "kernel_regularizer": self.kernel_regularizer
        }

        self.q_proj = layers.Dense(self.dim, **dense_config)
        self.kv_proj = layers.Dense(2 * self.dim, **dense_config)
        self.o_proj = layers.Dense(self.dim, **dense_config)

        if self.use_downsampling:
            self.downsample = layers.DepthwiseConv2D(
                3,
                strides=2,
                padding='same',
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer
            )

        self.lambda_param = self.add_weight(
            "lambda",
            shape=(),
            initializer="ones",
            trainable=True
        )

        # Build sublayers explicitly
        self.q_proj.build(input_shape)
        self.kv_proj.build(input_shape)
        self.o_proj.build(input_shape)

        if self.use_downsampling:
            self.downsample.build(input_shape)

        super().build(input_shape)

    def call(self, x: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Forward pass of the MobileMQA block.

        Args:
            x: Input tensor
            training: Whether in training mode

        Returns:
            Output tensor
        """
        batch_size = ops.shape(x)[0]
        height = ops.shape(x)[1]
        width = ops.shape(x)[2]

        q = self.q_proj(x)
        kv = self.kv_proj(x)

        if self.use_downsampling:
            kv = self.downsample(kv)
            kv_height, kv_width = height // 2, width // 2
        else:
            kv_height, kv_width = height, width

        # Split kv into k and v - using slice operation since tf.split isn't in keras.ops
        k = kv[..., :self.dim]
        v = kv[..., self.dim:]

        q = ops.reshape(q, (batch_size, height * width, self.num_heads, self.head_dim))
        k = ops.reshape(k, (batch_size, kv_height * kv_width, 1, self.head_dim))
        v = ops.reshape(v, (batch_size, kv_height * kv_width, 1, self.head_dim))

        attn = ops.matmul(q, k, transpose_b=True) * self.scale
        attn = ops.nn.softmax(attn, axis=-1)

        out = ops.matmul(attn, v)
        out = ops.reshape(out, (batch_size, height, width, self.dim))

        return self.o_proj(out)

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input

        Returns:
            Output shape
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Returns the layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration
        """
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "use_downsampling": self.use_downsampling,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get the config needed to build the layer from a config.

        Returns:
            Dictionary containing the build configuration
        """
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build the layer from a config created with get_build_config.

        Args:
            config: Dictionary containing the build configuration
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])


# ---------------------------------------------------------------------

def create_stem(
        config: ModelConfig,
        kernel_regularizer: Optional[regularizers.Regularizer] = None
) -> keras.Model:
    """Create the stem of the network.

    Args:
        config: Model configuration
        kernel_regularizer: Regularizer for the convolution kernels

    Returns:
        Stem model
    """
    inputs = layers.Input(shape=config.input_shape)

    x = layers.Conv2D(
        32,
        3,
        strides=2,
        padding='same',
        use_bias=False,
        kernel_initializer=config.kernel_initializer,
        kernel_regularizer=kernel_regularizer
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    return keras.Model(inputs, x, name="stem")


# ---------------------------------------------------------------------

def create_body(
        num_blocks: List[int],
        filters: List[int],
        strides: List[int],
        config: ModelConfig,
        kernel_regularizer: Optional[regularizers.Regularizer] = None
) -> keras.Model:
    """Create the main body of the network.

    Args:
        num_blocks: Number of blocks in each stage
        filters: Number of filters for each stage
        strides: Stride for each stage
        config: Model configuration
        kernel_regularizer: Regularizer for the convolution kernels

    Returns:
        Body model
    """
    inputs = layers.Input(shape=(None, None, filters[0]))
    x = inputs

    for i, (blocks, f, s) in enumerate(zip(num_blocks, filters, strides)):
        logger.info(f"Creating stage {i+1} with {blocks} blocks, {f} filters, stride {s}")

        for j in range(blocks):
            stride = s if j == 0 else 1
            block_type = 'ExtraDW' if j == 0 else 'IB'

            x = UIB(
                f,
                stride=stride,
                block_type=block_type,
                kernel_initializer=config.kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name=f"uib_stage{i+1}_block{j+1}"
            )(x)

        if config.use_attention and i == len(num_blocks) - 1:
            logger.info(f"Adding Mobile MQA to final stage")
            x = MobileMQA(
                f,
                use_downsampling=True,
                kernel_initializer=config.kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name=f"mobile_mqa_stage{i+1}"
            )(x)

    return keras.Model(inputs, x, name="body")


# ---------------------------------------------------------------------

def create_head(
        config: ModelConfig,
        kernel_regularizer: Optional[regularizers.Regularizer] = None
) -> keras.Model:
    """Create the head of the network.

    Args:
        config: Model configuration
        kernel_regularizer: Regularizer for the convolution kernels

    Returns:
        Head model
    """
    inputs = layers.Input(shape=(None, None, 320))  # Updated to match the last filter size
    x = layers.GlobalAveragePooling2D()(inputs)

    x = layers.Dense(
        1280,
        kernel_initializer=config.kernel_initializer,
        kernel_regularizer=kernel_regularizer
    )(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(config.dropout_rate)(x)

    x = layers.Dense(
        config.num_classes,
        kernel_initializer=config.kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        activation='softmax'
    )(x)

    return keras.Model(inputs, x, name="head")


# ---------------------------------------------------------------------

def MobileNetV4(config: ModelConfig) -> keras.Model:
    """Create MobileNetV4 model.

    Args:
        config: Model configuration

    Returns:
        MobileNetV4 model
    """
    logger.info("Creating MobileNetV4 model")
    logger.info(f"Configuration: input_shape={config.input_shape}, num_classes={config.num_classes}")
    logger.info(f"Width multiplier: {config.width_multiplier}, use_attention: {config.use_attention}")

    kernel_regularizer = regularizers.L2(config.weight_decay)

    # Define the architecture
    num_blocks = [1, 2, 3, 4, 3, 3, 1]
    filters = [16, 24, 40, 80, 112, 192, 320]
    strides = [1, 2, 2, 2, 1, 2, 1]

    # Apply width multiplier
    filters = [int(f * config.width_multiplier) for f in filters]
    logger.info(f"Filter sizes after width multiplier: {filters}")

    # Create the model components
    stem = create_stem(config, kernel_regularizer)
    body = create_body(num_blocks, filters, strides, config, kernel_regularizer)
    head = create_head(config, kernel_regularizer)

    # Combine all parts
    inputs = layers.Input(shape=config.input_shape)
    x = stem(inputs)
    x = body(x)
    outputs = head(x)

    model = keras.Model(inputs, outputs, name="MobileNetV4")
    logger.info(f"Created MobileNetV4 model with {model.count_params()} parameters")

    return model


# ---------------------------------------------------------------------

def configure_model(model: keras.Model, config: ModelConfig) -> None:
    """Configure the model for training.

    Args:
        model: The MobileNetV4 model to configure
        config: Model configuration
    """
    logger.info("Configuring model for training")

    optimizer = keras.optimizers.AdamW(
        learning_rate=0.001,
        weight_decay=config.weight_decay
    )

    loss = keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

    metrics = [
        keras.metrics.CategoricalAccuracy(name="accuracy"),
        keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_accuracy")
    ]

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    logger.info("Model compilation completed")


# ---------------------------------------------------------------------

def create_training_callbacks(model_name: str) -> List[keras.callbacks.Callback]:
    """Create callbacks for model training.

    Args:
        model_name: Name of the model for saving checkpoints

    Returns:
        List of training callbacks
    """
    logger.info(f"Creating training callbacks for model: {model_name}")

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=f"{model_name}.keras",
            monitor="val_accuracy",
            save_best_only=True,
            save_format="keras"
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=0.2,
            patience=5,
            min_lr=1e-6
        ),
        keras.callbacks.TensorBoard(
            log_dir=f"./logs/{model_name}",
            histogram_freq=1
        )
    ]

    logger.info(f"Created {len(callbacks)} training callbacks")
    return callbacks


# ---------------------------------------------------------------------

def train_model(
        model: keras.Model,
        train_data: Any,  # Using Any to avoid tf.data.Dataset import
        val_data: Any,
        config: ModelConfig,
        epochs: int = 300,
        initial_epoch: int = 0
) -> keras.callbacks.History:
    """Train the MobileNetV4 model.

    Args:
        model: The MobileNetV4 model to train
        train_data: Training dataset
        val_data: Validation dataset
        config: Model configuration
        epochs: Number of training epochs
        initial_epoch: Initial epoch number for resuming training

    Returns:
        Training history
    """
    logger.info(f"Starting training for {epochs} epochs")

    # Configure the model
    configure_model(model, config)

    # Create callbacks
    callbacks = create_training_callbacks(model.name)

    # Train the model
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        initial_epoch=initial_epoch,
        callbacks=callbacks
    )

    logger.info("Training completed")
    return history


# ---------------------------------------------------------------------

def create_data_augmentation() -> keras.Sequential:
    """Create a data augmentation pipeline.

    Returns:
        Data augmentation model
    """
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomBrightness(0.2),
        layers.RandomContrast(0.2)
    ], name="data_augmentation")


# ---------------------------------------------------------------------

def prepare_dataset(
        dataset: Any,  # Using Any to avoid tf.data.Dataset import
        config: ModelConfig,
        is_training: bool = False
) -> Any:
    """Prepare dataset for training or validation.

    Args:
        dataset: Input dataset
        config: Model configuration
        is_training: Whether preparing for training

    Returns:
        Prepared dataset
    """
    logger.info(f"Preparing {'training' if is_training else 'validation'} dataset")

    # Set up augmentation
    if is_training:
        augmentation = create_data_augmentation()

    def prepare_sample(image: keras.KerasTensor, label: keras.KerasTensor) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        # Resize image
        import tensorflow as tf  # Only import where needed for dataset preparation
        image = tf.image.resize(image, config.input_shape[:2])

        # Apply augmentation during training
        if is_training:
            image = augmentation(tf.expand_dims(image, 0))[0]

        # Normalize image
        image = ops.cast(image, "float32") / 255.0

        # One-hot encode label
        label = ops.nn.one_hot(label, config.num_classes)

        return image, label

    # Configure dataset
    import tensorflow as tf  # Only import where needed for dataset preparation
    dataset = dataset.map(
        prepare_sample,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if is_training:
        dataset = dataset.shuffle(10000)

    dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)

    logger.info("Dataset preparation completed")
    return dataset


# ---------------------------------------------------------------------

def main() -> None:
    """Main function to demonstrate model usage."""
    logger.info("Starting MobileNetV4 demonstration")

    # Create model configuration
    config = ModelConfig(
        input_shape=(224, 224, 3),
        num_classes=1000,
        width_multiplier=1.0,
        use_attention=True,
        weight_decay=1e-5,
        dropout_rate=0.2
    )

    # Create model
    model = MobileNetV4(config)

    # Print model summary
    logger.info("Model summary:")
    model.summary()

    logger.info("MobileNetV4 demonstration completed")


if __name__ == "__main__":
    main()