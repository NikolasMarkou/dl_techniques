import keras
import tensorflow as tf
from typing import List, Tuple, Optional, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.models.mobilenet_v4 import ModelConfig, MobileNetV4

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
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomZoom(0.1),
        keras.layers.RandomTranslation(0.1, 0.1),
        keras.layers.RandomBrightness(0.2),
        keras.layers.RandomContrast(0.2)
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

        image = tf.image.resize(image, config.input_shape[:2])

        # Apply augmentation during training
        if is_training:
            image = augmentation(tf.expand_dims(image, 0))[0]

        # Normalize image
        image = keras.ops.cast(image, "float32") / 255.0

        # One-hot encode label
        label = keras.ops.nn.one_hot(label, config.num_classes)

        return image, label

    # Configure dataset
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

# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()