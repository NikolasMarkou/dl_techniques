"""
Latent Reasoning Vision Model Training Script.

Implements a vision model using the three-layer conceptual framework:
1. Transformations: Feature extraction and processing substrate
2. Logic: Differentiable probabilistic reasoning with predicate modules
3. Arithmetic: Differentiable counting via probability aggregation
"""

import keras
import numpy as np
import tensorflow as tf
from keras import layers, ops
import matplotlib.pyplot as plt
from typing import Optional, Union, Tuple, Any, List

from dl_techniques.utils.logger import logger
from dl_techniques.layers.global_sum_pool_2d import GlobalSumPooling2D
from train.common import setup_gpu, create_base_argument_parser


@keras.saving.register_keras_serializable()
class PredicateModule(layers.Layer):
    """Approximates logical predicates (e.g., "is_red", "is_circle").

    Outputs probabilistic "logic maps" where each spatial location has
    P(predicate is TRUE at this location).
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
        self.conv = None
        self.activation = None
        self._build_input_shape = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        self._build_input_shape = input_shape
        self.conv = layers.Conv2D(
            filters=1, kernel_size=1, padding='same',
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name=f"conv_{self.name_suffix}"
        )
        self.activation = layers.Activation('sigmoid', name=f"sigmoid_{self.name_suffix}")
        self.conv.build(input_shape)
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        x = self.conv(inputs)
        return self.activation(x)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        return input_shape[:-1] + (1,)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "name_suffix": self.name_suffix,
            "kernel_initializer": self.kernel_initializer,
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config

    def get_build_config(self) -> dict:
        return {"input_shape": self._build_input_shape}

    def build_from_config(self, config: dict) -> None:
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])


@keras.saving.register_keras_serializable()
class LogicalCompositionLayer(layers.Layer):
    """Combines probabilistic logic maps using differentiable logical operations.

    Implements fuzzy logic: AND (multiply), OR (1-(1-A)*(1-B)), NOT (1-A).
    """

    def __init__(self, operation: str = "and", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.operation = operation.lower()
        if self.operation not in ['and', 'or', 'not']:
            raise ValueError(f"Unsupported operation: {operation}. Must be 'and', 'or', or 'not'.")
        self.logic_layer = None
        self._build_input_shape = None

    def build(self, input_shape: Union[Tuple, List[Tuple]]) -> None:
        self._build_input_shape = input_shape
        if self.operation == "and":
            self.logic_layer = layers.Multiply(name="logical_and")
        elif self.operation == "or":
            self.logic_layer = layers.Lambda(
                lambda inputs: 1.0 - ops.multiply(1.0 - inputs[0], 1.0 - inputs[1]),
                name="logical_or"
            )
        elif self.operation == "not":
            self.logic_layer = layers.Lambda(
                lambda inputs: 1.0 - inputs,
                name="logical_not"
            )
        super().build(input_shape)

    def call(self, inputs: Union[tf.Tensor, List[tf.Tensor]], training: Optional[bool] = None) -> tf.Tensor:
        if self.operation == "not":
            if isinstance(inputs, list):
                raise ValueError("NOT operation expects single input, got list.")
            return self.logic_layer(inputs)
        else:
            if not isinstance(inputs, list) or len(inputs) != 2:
                raise ValueError(f"{self.operation.upper()} operation expects list of 2 inputs.")
            return self.logic_layer(inputs)

    def compute_output_shape(self, input_shape: Union[Tuple, List[Tuple]]) -> Tuple[Optional[int], ...]:
        if self.operation == "not":
            return input_shape
        return input_shape[0]

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"operation": self.operation})
        return config

    def get_build_config(self) -> dict:
        return {"input_shape": self._build_input_shape}

    def build_from_config(self, config: dict) -> None:
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])


@keras.saving.register_keras_serializable()
class ReasoningBlock(layers.Layer):
    """Residual block that processes features for reasoning."""

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
        self.conv1 = None
        self.bn1 = None
        self.conv2 = None
        self.bn2 = None
        self.add = None
        self.relu = None
        self._build_input_shape = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        self._build_input_shape = input_shape
        self.conv1 = layers.Conv2D(
            filters=self.filters, kernel_size=self.kernel_size, padding='same',
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer, name="conv1"
        )
        self.bn1 = layers.BatchNormalization(name="bn1")
        self.conv2 = layers.Conv2D(
            filters=self.filters, kernel_size=self.kernel_size, padding='same',
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer, name="conv2"
        )
        self.bn2 = layers.BatchNormalization(name="bn2")
        if self.use_residual:
            self.add = layers.Add(name="residual_add")
        self.relu = layers.ReLU(name="relu")
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        residual = inputs
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        if self.use_residual:
            x = self.add([x, residual])
        x = self.relu(x)
        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        return input_shape

    def get_config(self) -> dict:
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
        return {"input_shape": self._build_input_shape}

    def build_from_config(self, config: dict) -> None:
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])


@keras.saving.register_keras_serializable()
class LatentReasoningVisionModel(keras.Model):
    """Vision model implementing the three-layer reasoning framework.

    Architecture:
    1. Transformations: Feature extraction backbone + reasoning blocks
    2. Logic: Explicit predicate modules + logical composition
    3. Arithmetic: Differentiable counting via probability summation
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
        self.feature_extractor = None
        self.reasoning_blocks = None
        self.predicate_modules = None
        self.logical_composition = None
        self.arithmetic_layer = None
        self._build_input_shape = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        self._build_input_shape = input_shape

        # Transformations: feature extraction backbone
        self.feature_extractor = keras.Sequential([
            layers.Conv2D(
                filters=self.filters, kernel_size=7, strides=2, padding='same',
                activation='relu', kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer, name="stem_conv"
            ),
            layers.MaxPooling2D((3, 3), strides=2, padding='same', name="stem_pool"),
            layers.Conv2D(
                filters=self.filters * 2, kernel_size=3, padding='same',
                activation='relu', kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer, name="block1_conv"
            ),
            layers.MaxPooling2D((2, 2), name="block1_pool"),
            layers.Conv2D(
                filters=self.filters * 4, kernel_size=3, padding='same',
                activation='relu', kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer, name="block2_conv"
            ),
        ], name="feature_extractor")

        self.reasoning_blocks = []
        for i in range(self.num_reasoning_blocks):
            self.reasoning_blocks.append(ReasoningBlock(
                filters=self.filters * 4,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f"reasoning_block_{i}"
            ))

        # Logic: predicate modules
        self.predicate_modules = {}
        for predicate in self.predicates:
            self.predicate_modules[predicate] = PredicateModule(
                name_suffix=predicate,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer
            )

        self.logical_composition = self._build_logical_composition()

        # Arithmetic: differentiable counting
        self.arithmetic_layer = keras.Sequential([
            GlobalSumPooling2D(name="differentiable_counting"),
            layers.Dense(
                1, activation='linear',
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
        if "and" in self.logical_query:
            parts = [p.strip() for p in self.logical_query.split("and")]
            if len(parts) == 2:
                return LogicalCompositionLayer(operation="and", name="logical_and")
        elif "or" in self.logical_query:
            parts = [p.strip() for p in self.logical_query.split("or")]
            if len(parts) == 2:
                return LogicalCompositionLayer(operation="or", name="logical_or")
        elif "not" in self.logical_query:
            return LogicalCompositionLayer(operation="not", name="logical_not")
        return LogicalCompositionLayer(operation="and", name="logical_and")

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        # Stage 1: Transformations
        features = self.feature_extractor(inputs, training=training)
        for block in self.reasoning_blocks:
            features = block(features, training=training)

        # Stage 2: Logic
        logic_maps = {}
        for predicate, module in self.predicate_modules.items():
            logic_maps[predicate] = module(features, training=training)

        predicate_list = list(self.predicates)
        if ("and" in self.logical_query or "or" in self.logical_query) and len(predicate_list) >= 2:
            final_logic_map = self.logical_composition(
                [logic_maps[predicate_list[0]], logic_maps[predicate_list[1]]],
                training=training
            )
        else:
            final_logic_map = logic_maps[predicate_list[0]]

        # Stage 3: Arithmetic
        return self.arithmetic_layer(final_logic_map, training=training)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        return (input_shape[0], 1)

    def get_config(self) -> dict:
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
        return {"input_shape": self._build_input_shape}

    def build_from_config(self, config: dict) -> None:
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])


def generate_synthetic_shapes(
        num_samples: int = 1000,
        image_size: int = 128,
        max_count: int = 5,
        seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic dataset with colored shapes for counting red circles."""
    if seed is not None:
        np.random.seed(seed)

    images = np.zeros((num_samples, image_size, image_size, 3), dtype=np.uint8)
    labels = np.zeros(num_samples, dtype=np.float32)

    colors = {
        'red': [255, 0, 0], 'green': [0, 255, 0], 'blue': [0, 0, 255],
        'yellow': [255, 255, 0], 'magenta': [255, 0, 255], 'cyan': [0, 255, 255]
    }

    for i in range(num_samples):
        img = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255
        red_circle_count = 0
        num_objects = np.random.randint(1, max_count + 3)

        for _ in range(num_objects):
            margin = 25
            x = np.random.randint(margin, image_size - margin)
            y = np.random.randint(margin, image_size - margin)
            radius = np.random.randint(8, 20)
            color_name = np.random.choice(list(colors.keys()))
            color = colors[color_name]
            is_circle = np.random.random() < 0.5

            if is_circle:
                yy, xx = np.ogrid[:image_size, :image_size]
                mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
                img[mask] = color
                if color_name == 'red':
                    red_circle_count += 1
            else:
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
    """Create and train a latent reasoning model on the counting task."""
    logger.info("Generating synthetic dataset...")
    images, labels = generate_synthetic_shapes(num_samples=num_samples, seed=seed)
    images = images.astype(np.float32) / 255.0

    logger.info(f"Dataset generated: {images.shape[0]} samples")
    logger.info(f"Label distribution: min={labels.min():.1f}, max={labels.max():.1f}, mean={labels.mean():.2f}")

    model = LatentReasoningVisionModel(
        predicates=predicates, logical_query=logical_query,
        filters=filters, num_reasoning_blocks=num_reasoning_blocks,
        input_shape=(128, 128, 3)
    )
    model.build((None, 128, 128, 3))

    logger.info(f"Model: predicates={predicates}, query={logical_query}, "
                f"blocks={num_reasoning_blocks}, params={model.count_params():,}")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse', metrics=['mae']
    )

    logger.info("Starting training...")
    history = model.fit(
        images, labels, epochs=epochs, batch_size=batch_size,
        validation_split=validation_split, verbose=1
    )
    logger.info("Training completed!")
    return model, history.history


def visualize_logic_maps(
        model: LatentReasoningVisionModel,
        sample_images: np.ndarray,
        num_samples: int = 4
) -> None:
    """Visualize the learned logic maps for different predicates."""
    sample_images = sample_images[:num_samples]

    features = model.feature_extractor(sample_images, training=False)
    for block in model.reasoning_blocks:
        features = block(features, training=False)

    logic_maps = {}
    for predicate, module in model.predicate_modules.items():
        logic_maps[predicate] = module(features, training=False)

    fig, axes = plt.subplots(num_samples, len(model.predicates) + 1, figsize=(15, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        axes[i, 0].imshow(sample_images[i])
        axes[i, 0].set_title("Original")
        axes[i, 0].axis('off')

        for j, predicate in enumerate(model.predicates):
            logic_map = logic_maps[predicate][i, :, :, 0]
            axes[i, j + 1].imshow(logic_map, cmap='hot', vmin=0, vmax=1)
            axes[i, j + 1].set_title(f"Logic Map: {predicate}")
            axes[i, j + 1].axis('off')

    plt.tight_layout()
    plt.show()


def main():
    parser = create_base_argument_parser(
        description="Train Latent Reasoning Vision Model",
        default_dataset="synthetic",
        dataset_choices=["synthetic"],
    )
    parser.add_argument('--num-samples', type=int, default=20000, help='Number of synthetic samples')
    parser.add_argument('--filters', type=int, default=32, help='Number of convolutional filters')
    parser.add_argument('--num-reasoning-blocks', type=int, default=3, help='Number of reasoning blocks')
    parser.add_argument('--predicates', nargs='+', default=["is_red", "is_circle"], help='Predicates to learn')
    parser.add_argument('--logical-query', type=str, default="is_red AND is_circle", help='Logical composition query')
    parser.add_argument('--validation-split', type=float, default=0.2, help='Validation split ratio')
    args = parser.parse_args()

    setup_gpu(gpu_id=args.gpu)

    model, history = create_and_train_model(
        num_samples=args.num_samples,
        predicates=args.predicates,
        logical_query=args.logical_query,
        filters=args.filters,
        num_reasoning_blocks=args.num_reasoning_blocks,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
    )

    test_images, test_labels = generate_synthetic_shapes(num_samples=100, seed=123)
    test_images = test_images.astype(np.float32) / 255.0

    predictions = model.predict(test_images)
    mae = np.mean(np.abs(predictions.flatten() - test_labels))
    mse = np.mean((predictions.flatten() - test_labels) ** 2)

    logger.info(f"Test Performance: MAE={mae:.3f}, MSE={mse:.3f}")

    if args.show_plots:
        visualize_logic_maps(model, test_images, num_samples=3)


if __name__ == "__main__":
    main()
