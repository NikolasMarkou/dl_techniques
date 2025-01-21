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

from typing import List, Tuple, Optional, Union, Dict, Any
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers


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
            kernel_initializer: str = "he_normal",
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
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
        self.expanded_filters = filters * expansion_factor

        # Layer configurations
        conv_config = {
            "kernel_initializer": kernel_initializer,
            "kernel_regularizer": kernel_regularizer,
            "padding": "same",
            "use_bias": False
        }

        # Layers following proper normalization order: Conv/Linear -> Norm -> Activation
        self.expand_conv = layers.Conv2D(self.expanded_filters, 1, **conv_config)
        self.bn1 = layers.BatchNormalization()
        self.activation = layers.ReLU()

        if use_dw1:
            self.dw1 = layers.DepthwiseConv2D(
                kernel_size,
                stride,
                **conv_config
            )
            self.bn_dw1 = layers.BatchNormalization()

        if use_dw2:
            self.dw2 = layers.DepthwiseConv2D(
                kernel_size,
                1,
                **conv_config
            )
            self.bn_dw2 = layers.BatchNormalization()

        self.project_conv = layers.Conv2D(filters, 1, **conv_config)
        self.bn2 = layers.BatchNormalization()

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
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

        if self.stride == 1 and inputs.shape[-1] == self.filters:
            return inputs + x
        return x


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
            kernel_initializer: str = "he_normal",
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            **kwargs: Any
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_downsampling = use_downsampling

        # Layer configurations
        dense_config = {
            "kernel_initializer": kernel_initializer,
            "kernel_regularizer": kernel_regularizer
        }

        self.q_proj = layers.Dense(dim, **dense_config)
        self.kv_proj = layers.Dense(2 * dim, **dense_config)
        self.o_proj = layers.Dense(dim, **dense_config)

        if use_downsampling:
            self.downsample = layers.DepthwiseConv2D(
                3,
                strides=2,
                padding='same',
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer
            )

        self.scale = self.head_dim ** -0.5
        self.lambda_param = self.add_weight(
            "lambda",
            shape=(),
            initializer="ones",
            trainable=True
        )

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass of the MobileMQA block.

        Args:
            x: Input tensor
            training: Whether in training mode

        Returns:
            Output tensor
        """
        batch_size = tf.shape(x)[0]
        height = tf.shape(x)[1]
        width = tf.shape(x)[2]

        q = self.q_proj(x)
        kv = self.kv_proj(x)

        if self.use_downsampling:
            kv = self.downsample(kv)
            kv_height, kv_width = height // 2, width // 2
        else:
            kv_height, kv_width = height, width

        k, v = tf.split(kv, 2, axis=-1)

        q = tf.reshape(q, (batch_size, height * width, self.num_heads, self.head_dim))
        k = tf.reshape(k, (batch_size, kv_height * kv_width, 1, self.head_dim))
        v = tf.reshape(v, (batch_size, kv_height * kv_width, 1, self.head_dim))

        attn = tf.matmul(q, k, transpose_b=True) * self.scale
        attn = tf.nn.softmax(attn, axis=-1)

        out = tf.matmul(attn, v)
        out = tf.reshape(out, (batch_size, height, width, self.dim))

        return self.o_proj(out)


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
        for j in range(blocks):
            stride = s if j == 0 else 1
            block_type = 'ExtraDW' if j == 0 else 'IB'

            x = UIB(
                f,
                stride=stride,
                block_type=block_type,
                kernel_initializer=config.kernel_initializer,
                kernel_regularizer=kernel_regularizer
            )(x)

        if config.use_attention and i == len(num_blocks) - 1:
            x = MobileMQA(
                f,
                use_downsampling=True,
                kernel_initializer=config.kernel_initializer,
                kernel_regularizer=kernel_regularizer
            )(x)

    return keras.Model(inputs, x, name="body")


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
    inputs = layers.Input(shape=(None, None, 256))
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


def MobileNetV4(config: ModelConfig) -> keras.Model:
    """Create MobileNetV4 model.

    Args:
        config: Model configuration

    Returns:
        MobileNetV4 model
    """
    kernel_regularizer = regularizers.L2(config.weight_decay)

    # Define the architecture
    num_blocks = [1, 2, 3, 4, 3, 3, 1]
    filters = [16, 24, 40, 80, 112, 192, 320]
    strides = [1, 2, 2, 2, 1, 2, 1]

    # Apply width multiplier
    filters = [int(f * config.width_multiplier) for f in filters]

    # Create the model components
    stem = create_stem(config, kernel_regularizer)
    body = create_body(num_blocks, filters, strides, config, kernel_regularizer)
    head = create_head(config, kernel_regularizer)

    # Combine all parts
    inputs = layers.Input(shape=config.input_shape)
    x = stem(inputs)
    x = body(x)
    outputs = head(x)

    return keras.Model(inputs, outputs, name="MobileNetV4")


def configure_model(model: keras.Model, config: ModelConfig) -> None:
    """Configure the model for training.

    Args:
        model: The MobileNetV4 model to configure
        config: Model configuration
    """
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


def create_training_callbacks(model_name: str) -> List[keras.callbacks.Callback]:
    """Create callbacks for model training.

    Args:
        model_name: Name of the model for saving checkpoints

    Returns:
        List of training callbacks
    """
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
    return callbacks


def train_model(
        model: keras.Model,
        train_data: tf.data.Dataset,
        val_data: tf.data.Dataset,
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

    return history


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


def prepare_dataset(
        dataset: tf.data.Dataset,
        config: ModelConfig,
        is_training: bool = False
) -> tf.data.Dataset:
    """Prepare dataset for training or validation.

    Args:
        dataset: Input dataset
        config: Model configuration
        is_training: Whether preparing for training

    Returns:
        Prepared dataset
    """
    # Set up augmentation
    if is_training:
        augmentation = create_data_augmentation()

    def prepare_sample(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # Resize image
        image = tf.image.resize(image, config.input_shape[:2])

        # Apply augmentation during training
        if is_training:
            image = augmentation(tf.expand_dims(image, 0))[0]

        # Normalize image
        image = tf.cast(image, tf.float32) / 255.0

        # One-hot encode label
        label = tf.one_hot(label, config.num_classes)

        return image, label

    # Configure dataset
    dataset = dataset.map(
        prepare_sample,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if is_training:
        dataset = dataset.shuffle(10000)

    dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)

    return dataset


# Example usage:
def main() -> None:
    """Main function to demonstrate model usage."""
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
    model.summary()

    # Note: To actually train the model, you would need to:
    # 1. Prepare your datasets
    # 2. Call train_model with your datasets
    # 3. Save the trained model
    # Example:
    # train_data = prepare_dataset(raw_train_data, config, is_training=True)
    # val_data = prepare_dataset(raw_val_data, config, is_training=False)
    # history = train_model(model, train_data, val_data, config)
    # model.save("mobilenetv4_model.keras")