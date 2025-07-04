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
from keras import regularizers, layers
from typing import List, Tuple, Optional

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.mobile_mqa import MobileMQA
from dl_techniques.layers.universal_inverted_bottleneck import UIB

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

