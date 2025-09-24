"""
Latent Reasoning Vision Model Implementation

This module implements a vision_heads model using the three-layer conceptual framework:
1. Transformations: Feature extraction and processing substrate
2. Logic: Differentiable probabilistic reasoning with predicate modules
3. Arithmetic: Differentiable counting via probability aggregation

The model performs end-to-end gradient-based learning where the final quantitative
answer provides supervision to train all three layers simultaneously.
"""

import keras
import numpy as np
import tensorflow as tf
from keras import layers, ops
import matplotlib.pyplot as plt
from typing import Optional, Union, Tuple, Any, List

from dl_techniques.utils.logger import logger
from dl_techniques.layers.global_sum_pool_2d import GlobalSumPooling2D

@keras.saving.register_keras_serializable()
class PredicateModule(layers.Layer):
    """
    A predicate module that approximates logical predicates (e.g., "is_red", "is_circle").

    This layer takes high-level feature maps and outputs probabilistic "logic maps"
    where each spatial location has a probability P(predicate is TRUE at this location).

    Args:
        name_suffix: Suffix for the predicate name (e.g., "is_red", "is_circle").
        kernel_initializer: Initializer for the 1x1 convolution kernel.
        kernel_regularizer: Regularizer for the convolution kernel.
        **kwargs: Additional keyword arguments for the Layer base class.
    """

    def __init__(
            self,
            name_suffix: str = "predicate",
            kernel_initializer: str = "glorot_uniform",
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(name=f"predicate_{name_suffix}", **kwargs)

        self.name_suffix = name_suffix
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer

        # Will be created in build()
        self.conv = None
        self.activation = None
        self._build_input_shape = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the predicate module."""
        self._build_input_shape = input_shape

        # 1x1 Conv acts as pixel-wise Dense layer for predicate evaluation
        self.conv = layers.Conv2D(
            filters=1,  # Single output channel for binary predicate
            kernel_size=1,  # 1x1 convolution for pixel-wise evaluation
            padding='same',
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name=f"conv_{self.name_suffix}"
        )

        # Sigmoid activation for probabilistic output [0, 1]
        self.activation = layers.Activation('sigmoid', name=f"sigmoid_{self.name_suffix}")

        # Build sublayers
        self.conv.build(input_shape)

        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Evaluate the predicate across all spatial locations.

        Args:
            inputs: Input feature map.
            training: Whether in training mode.

        Returns:
            Logic map with probabilities P(predicate is TRUE) at each location.
        """
        # Apply 1x1 convolution for predicate evaluation
        x = self.conv(inputs)

        # Apply sigmoid for probabilistic output
        logic_map = self.activation(x)

        return logic_map

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        return input_shape[:-1] + (1,)  # Same spatial dims, single channel

    def get_config(self) -> dict:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "name_suffix": self.name_suffix,
            "kernel_initializer": self.kernel_initializer,
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config

    def get_build_config(self) -> dict:
        """Get build configuration."""
        return {"input_shape": self._build_input_shape}

    def build_from_config(self, config: dict) -> None:
        """Build from configuration."""
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])


@keras.saving.register_keras_serializable()
class LogicalCompositionLayer(layers.Layer):
    """
    A layer that combines probabilistic logic maps using differentiable logical operations.

    Implements fuzzy logic operations:
    - AND: P(A) * P(B) (using Multiply layer)
    - OR: 1 - (1 - P(A)) * (1 - P(B)) (using Lambda layer)
    - NOT: 1 - P(A) (using Lambda layer)

    Args:
        operation: Type of logical operation ('and', 'or', 'not').
        **kwargs: Additional keyword arguments for the Layer base class.
    """

    def __init__(
            self,
            operation: str = "and",
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.operation = operation.lower()
        if self.operation not in ['and', 'or', 'not']:
            raise ValueError(f"Unsupported operation: {operation}. Must be 'and', 'or', or 'not'.")

        # Will be created in build()
        self.logic_layer = None
        self._build_input_shape = None

    def build(self, input_shape: Union[Tuple, List[Tuple]]) -> None:
        """Build the logical composition layer."""
        self._build_input_shape = input_shape

        if self.operation == "and":
            # P(A ∧ B) ≈ P(A) * P(B)
            self.logic_layer = layers.Multiply(name="logical_and")
        elif self.operation == "or":
            # P(A ∨ B) ≈ 1 - (1 - P(A)) * (1 - P(B))
            self.logic_layer = layers.Lambda(
                lambda inputs: 1.0 - ops.multiply(1.0 - inputs[0], 1.0 - inputs[1]),
                name="logical_or"
            )
        elif self.operation == "not":
            # P(¬A) = 1 - P(A)
            self.logic_layer = layers.Lambda(
                lambda inputs: 1.0 - inputs,
                name="logical_not"
            )

        super().build(input_shape)

    def call(self, inputs: Union[tf.Tensor, List[tf.Tensor]], training: Optional[bool] = None) -> tf.Tensor:
        """
        Apply logical composition to input logic maps.

        Args:
            inputs: Single tensor for NOT, list of tensors for AND/OR.
            training: Whether in training mode.

        Returns:
            Composed logic map.
        """
        if self.operation == "not":
            if isinstance(inputs, list):
                raise ValueError("NOT operation expects single input, got list.")
            return self.logic_layer(inputs)
        else:
            if not isinstance(inputs, list) or len(inputs) != 2:
                raise ValueError(f"{self.operation.upper()} operation expects list of 2 inputs.")
            return self.logic_layer(inputs)

    def compute_output_shape(self, input_shape: Union[Tuple, List[Tuple]]) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        if self.operation == "not":
            return input_shape
        else:
            return input_shape[0]  # Same as first input for binary operations

    def get_config(self) -> dict:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "operation": self.operation,
        })
        return config

    def get_build_config(self) -> dict:
        """Get build configuration."""
        return {"input_shape": self._build_input_shape}

    def build_from_config(self, config: dict) -> None:
        """Build from configuration."""
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])


@keras.saving.register_keras_serializable()
class ReasoningBlock(layers.Layer):
    """
    A composite transformation layer that processes features for reasoning.

    This is a residual block that allows the network to learn complex transformations
    while maintaining stable gradients. It serves as the substrate for the logic
    operations to work on.

    Args:
        filters: Number of convolutional filters.
        kernel_size: Size of the convolutional kernels.
        use_residual: Whether to use residual connections.
        kernel_initializer: Initializer for convolution kernels.
        kernel_regularizer: Regularizer for convolution kernels.
        **kwargs: Additional keyword arguments for the Layer base class.
    """

    def __init__(
            self,
            filters: int,
            kernel_size: Union[int, Tuple[int, int]] = 3,
            use_residual: bool = True,
            kernel_initializer: str = "glorot_uniform",
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.use_residual = use_residual
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer

        # Initialize sublayers to None
        self.conv1 = None
        self.bn1 = None
        self.conv2 = None
        self.bn2 = None
        self.add = None
        self.relu = None
        self._build_input_shape = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the reasoning block."""
        self._build_input_shape = input_shape

        # First convolution
        self.conv1 = layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            padding='same',
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="conv1"
        )
        self.bn1 = layers.BatchNormalization(name="bn1")

        # Second convolution
        self.conv2 = layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            padding='same',
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="conv2"
        )
        self.bn2 = layers.BatchNormalization(name="bn2")

        # Residual connection
        if self.use_residual:
            self.add = layers.Add(name="residual_add")

        self.relu = layers.ReLU(name="relu")

        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """Forward pass through the reasoning block."""
        residual = inputs

        # First transformation
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)

        # Second transformation
        x = self.conv2(x)
        x = self.bn2(x, training=training)

        # Add residual connection
        if self.use_residual:
            x = self.add([x, residual])

        x = self.relu(x)

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        return input_shape

    def get_config(self) -> dict:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "use_residual": self.use_residual,
            "kernel_initializer": self.kernel_initializer,
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config

    def get_build_config(self) -> dict:
        """Get build configuration."""
        return {"input_shape": self._build_input_shape}

    def build_from_config(self, config: dict) -> None:
        """Build from configuration."""
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])


@keras.saving.register_keras_serializable()
class LatentReasoningVisionModel(keras.Model):
    """
    A vision_heads model implementing the three-layer reasoning framework.

    Architecture:
    1. Transformations: Feature extraction backbone + reasoning blocks
    2. Logic: Explicit predicate modules + logical composition
    3. Arithmetic: Differentiable counting via probability summation

    The model learns to solve visual reasoning tasks by decomposing them into
    learnable logical predicates and their compositions.

    Args:
        predicates: List of predicate names to learn (e.g., ["is_red", "is_circle"]).
        logical_query: String describing the logical composition (e.g., "is_red AND is_circle").
        filters: Number of filters in the transformation layers.
        num_reasoning_blocks: Number of reasoning blocks in the transformation stage.
        input_shape: Shape of input images (height, width, channels).
        kernel_initializer: Initializer for convolution kernels.
        kernel_regularizer: Regularizer for convolution kernels.
        **kwargs: Additional keyword arguments for the Model base class.
    """

    def __init__(
            self,
            predicates: List[str] = ["is_red", "is_circle"],
            logical_query: str = "is_red AND is_circle",
            filters: int = 32,
            num_reasoning_blocks: int = 3,
            input_shape: Tuple[int, int, int] = (128, 128, 3),
            kernel_initializer: str = "glorot_uniform",
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.predicates = predicates
        self.logical_query = logical_query.lower()
        self.filters = filters
        self.num_reasoning_blocks = num_reasoning_blocks
        self.input_shape_param = input_shape
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer

        # Initialize sublayers to None
        self.feature_extractor = None
        self.reasoning_blocks = None
        self.predicate_modules = None
        self.logical_composition = None
        self.arithmetic_layer = None
        self._build_input_shape = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the complete model."""
        self._build_input_shape = input_shape

        # ==================== TRANSFORMATIONS ====================
        # Feature extractor backbone (CNN)
        self.feature_extractor = keras.Sequential([
            layers.Conv2D(
                filters=self.filters,
                kernel_size=7,
                strides=2,
                padding='same',
                activation='relu',
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name="stem_conv"
            ),
            layers.MaxPooling2D((3, 3), strides=2, padding='same', name="stem_pool"),
            layers.Conv2D(
                filters=self.filters * 2,
                kernel_size=3,
                padding='same',
                activation='relu',
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name="block1_conv"
            ),
            layers.MaxPooling2D((2, 2), name="block1_pool"),
            layers.Conv2D(
                filters=self.filters * 4,
                kernel_size=3,
                padding='same',
                activation='relu',
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name="block2_conv"
            ),
        ], name="feature_extractor")

        # Reasoning blocks for feature refinement
        self.reasoning_blocks = []
        for i in range(self.num_reasoning_blocks):
            block = ReasoningBlock(
                filters=self.filters * 4,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f"reasoning_block_{i}"
            )
            self.reasoning_blocks.append(block)

        # ==================== LOGIC ====================
        # Predicate modules (one for each predicate)
        self.predicate_modules = {}
        for predicate in self.predicates:
            module = PredicateModule(
                name_suffix=predicate,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer
            )
            self.predicate_modules[predicate] = module

        # Logical composition based on query
        self.logical_composition = self._build_logical_composition()

        # ==================== ARITHMETIC ====================
        # Differentiable counting layer
        self.arithmetic_layer = keras.Sequential([
            GlobalSumPooling2D(name="differentiable_counting"),  # Sum probabilities
            layers.Dense(
                1,
                activation='linear',
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name="count_output"
            )
        ], name="arithmetic")

        # Build all sublayers
        self.feature_extractor.build(input_shape)
        feature_shape = self.feature_extractor.compute_output_shape(input_shape)

        current_shape = feature_shape
        for block in self.reasoning_blocks:
            block.build(current_shape)
            current_shape = block.compute_output_shape(current_shape)

        for module in self.predicate_modules.values():
            module.build(current_shape)

        self.arithmetic_layer.build((None, None, None, 1))

        super().build(input_shape)

    def _build_logical_composition(self) -> keras.Model:
        """Build the logical composition submodel based on the query."""
        # Simple parser for basic logical queries
        # This is a simplified version - in practice, you'd want a more robust parser

        if "and" in self.logical_query.lower():
            parts = [p.strip() for p in self.logical_query.split("and")]
            if len(parts) == 2:
                return LogicalCompositionLayer(operation="and", name="logical_and")
        elif "or" in self.logical_query.lower():
            parts = [p.strip() for p in self.logical_query.split("or")]
            if len(parts) == 2:
                return LogicalCompositionLayer(operation="or", name="logical_or")
        elif "not" in self.logical_query.lower():
            return LogicalCompositionLayer(operation="not", name="logical_not")

        # Default to AND for two predicates
        return LogicalCompositionLayer(operation="and", name="logical_and")

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Forward pass implementing the three-layer reasoning framework.

        Args:
            inputs: Input tensor of shape (batch_size, height, width, channels).
            training: Whether in training mode.

        Returns:
            Predicted count as a scalar.
        """
        # ==================== STAGE 1: TRANSFORMATIONS ====================
        # Extract features using CNN backbone
        features = self.feature_extractor(inputs, training=training)

        # Refine features using reasoning blocks
        for block in self.reasoning_blocks:
            features = block(features, training=training)

        # ==================== STAGE 2: LOGIC ====================
        # Apply predicate modules to get individual logic maps
        logic_maps = {}
        for predicate, module in self.predicate_modules.items():
            logic_maps[predicate] = module(features, training=training)

        # Compose logic maps according to the logical query
        if "and" in self.logical_query.lower():
            # Example: "is_red AND is_circle"
            predicate_list = list(self.predicates)
            if len(predicate_list) >= 2:
                final_logic_map = self.logical_composition(
                    [logic_maps[predicate_list[0]], logic_maps[predicate_list[1]]],
                    training=training
                )
            else:
                final_logic_map = logic_maps[predicate_list[0]]
        elif "or" in self.logical_query.lower():
            # Example: "is_red OR is_circle"
            predicate_list = list(self.predicates)
            if len(predicate_list) >= 2:
                final_logic_map = self.logical_composition(
                    [logic_maps[predicate_list[0]], logic_maps[predicate_list[1]]],
                    training=training
                )
            else:
                final_logic_map = logic_maps[predicate_list[0]]
        else:
            # Default: use first predicate
            final_logic_map = logic_maps[list(self.predicates)[0]]

        # ==================== STAGE 3: ARITHMETIC ====================
        # Count objects by summing probabilities in the final logic map
        count = self.arithmetic_layer(final_logic_map, training=training)

        return count

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        return (input_shape[0], 1)

    def get_config(self) -> dict:
        """Get model configuration."""
        config = super().get_config()
        config.update({
            "predicates": self.predicates,
            "logical_query": self.logical_query,
            "filters": self.filters,
            "num_reasoning_blocks": self.num_reasoning_blocks,
            "input_shape": self.input_shape_param,
            "kernel_initializer": self.kernel_initializer,
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config

    def get_build_config(self) -> dict:
        """Get build configuration."""
        return {"input_shape": self._build_input_shape}

    def build_from_config(self, config: dict) -> None:
        """Build from configuration."""
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])


def generate_synthetic_shapes(
        num_samples: int = 1000,
        image_size: int = 128,
        max_count: int = 5,
        seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic dataset with colored shapes (no OpenCV dependency).

    Creates images with circles and rectangles in different colors.
    The task is to count red circles.

    Args:
        num_samples: Number of samples to generate.
        image_size: Size of generated images (square).
        max_count: Maximum number of target objects per image.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (images, labels) where labels are red circle counts.
    """
    if seed is not None:
        np.random.seed(seed)

    images = np.zeros((num_samples, image_size, image_size, 3), dtype=np.uint8)
    labels = np.zeros(num_samples, dtype=np.float32)

    colors = {
        'red': [255, 0, 0],
        'green': [0, 255, 0],
        'blue': [0, 0, 255],
        'yellow': [255, 255, 0],
        'magenta': [255, 0, 255],
        'cyan': [0, 255, 255]
    }

    for i in range(num_samples):
        # White background
        img = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255
        red_circle_count = 0

        # Add random number of objects
        num_objects = np.random.randint(1, max_count + 3)

        for _ in range(num_objects):
            # Random position (avoid edges)
            margin = 25
            x = np.random.randint(margin, image_size - margin)
            y = np.random.randint(margin, image_size - margin)

            # Random size
            radius = np.random.randint(8, 20)

            # Random color
            color_name = np.random.choice(list(colors.keys()))
            color = colors[color_name]

            # Random shape
            is_circle = np.random.random() < 0.5

            if is_circle:
                # Draw circle using numpy operations
                yy, xx = np.ogrid[:image_size, :image_size]
                mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
                img[mask] = color

                if color_name == 'red':
                    red_circle_count += 1
            else:
                # Draw rectangle
                x1, y1 = max(0, x - radius), max(0, y - radius)
                x2, y2 = min(image_size, x + radius), min(image_size, y + radius)
                img[y1:y2, x1:x2] = color

        images[i] = img
        labels[i] = red_circle_count

    return images, labels


def create_and_train_model(
        num_samples: int = 1000,
        predicates: List[str] = ["is_red", "is_circle"],
        logical_query: str = "is_red AND is_circle",
        filters: int = 32,
        num_reasoning_blocks: int = 3,
        epochs: int = 20,
        batch_size: int = 32,
        validation_split: float = 0.2,
        seed: Optional[int] = 42
) -> Tuple[LatentReasoningVisionModel, dict]:
    """
    Create and train a latent reasoning model on the counting task.

    Args:
        num_samples: Number of training samples.
        predicates: List of predicates to learn.
        logical_query: Logical composition query.
        filters: Number of convolutional filters.
        num_reasoning_blocks: Number of reasoning blocks.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        validation_split: Fraction of data for validation.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (trained_model, training_history).
    """
    logger.info("Generating synthetic dataset...")

    # Generate dataset
    images, labels = generate_synthetic_shapes(
        num_samples=num_samples,
        seed=seed
    )

    # Normalize images
    images = images.astype(np.float32) / 255.0

    logger.info(f"Dataset generated: {images.shape[0]} samples")
    logger.info(f"Label distribution: min={labels.min():.1f}, max={labels.max():.1f}, mean={labels.mean():.2f}")

    # Create model
    model = LatentReasoningVisionModel(
        predicates=predicates,
        logical_query=logical_query,
        filters=filters,
        num_reasoning_blocks=num_reasoning_blocks,
        input_shape=(128, 128, 3)
    )

    # Build model
    model.build((None, 128, 128, 3))

    logger.info("Model architecture:")
    logger.info(f"- Predicates: {predicates}")
    logger.info(f"- Logical query: {logical_query}")
    logger.info(f"- Reasoning blocks: {num_reasoning_blocks}")
    logger.info(f"- Total parameters: {model.count_params():,}")

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    # Train model
    logger.info("Starting training...")

    history = model.fit(
        images, labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1
    )

    logger.info("Training completed!")

    return model, history.history


def visualize_logic_maps(
        model: LatentReasoningVisionModel,
        sample_images: np.ndarray,
        num_samples: int = 4
) -> None:
    """
    Visualize the learned logic maps for different predicates.

    Args:
        model: Trained model.
        sample_images: Sample images to visualize.
        num_samples: Number of samples to show.
    """
    # Get intermediate outputs
    sample_images = sample_images[:num_samples]

    # Extract features
    features = model.feature_extractor(sample_images, training=False)
    for block in model.reasoning_blocks:
        features = block(features, training=False)

    # Get logic maps for each predicate
    logic_maps = {}
    for predicate, module in model.predicate_modules.items():
        logic_maps[predicate] = module(features, training=False)

    # Create visualization
    fig, axes = plt.subplots(num_samples, len(model.predicates) + 1, figsize=(15, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        # Original image
        axes[i, 0].imshow(sample_images[i])
        axes[i, 0].set_title("Original")
        axes[i, 0].axis('off')

        # Logic maps
        for j, predicate in enumerate(model.predicates):
            logic_map = logic_maps[predicate][i, :, :, 0]
            axes[i, j + 1].imshow(logic_map, cmap='hot', vmin=0, vmax=1)
            axes[i, j + 1].set_title(f"Logic Map: {predicate}")
            axes[i, j + 1].axis('off')

    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Create and train model
    model, history = create_and_train_model(
        num_samples=20000,
        predicates=["is_red", "is_circle"],
        logical_query="is_red AND is_circle",
        filters=32,
        epochs=100,
        batch_size=32
    )

    # Test the model
    test_images, test_labels = generate_synthetic_shapes(
        num_samples=100,
        seed=123
    )
    test_images = test_images.astype(np.float32) / 255.0

    predictions = model.predict(test_images)

    # Calculate performance
    mae = np.mean(np.abs(predictions.flatten() - test_labels))
    mse = np.mean((predictions.flatten() - test_labels) ** 2)

    logger.info(f"Test Performance:")
    logger.info(f"- MAE: {mae:.3f}")
    logger.info(f"- MSE: {mse:.3f}")

    # Visualize learned logic maps
    visualize_logic_maps(model, test_images, num_samples=3)