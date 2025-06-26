"""
CapsNet Usage Example with Separated Architecture and Training.

This example demonstrates how to use the separated CapsNet model and trainer
for training on MNIST or similar datasets. The clean separation allows for
flexible training configurations and easy experimentation.
"""

import keras
import numpy as np
import tensorflow as tf
from typing import Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.models.capsnet import CapsNet
from .trainer import create_capsnet_trainer, CapsNetTrainer

# ---------------------------------------------------------------------
def prepare_mnist_data(
        batch_size: int = 32,
        validation_split: float = 0.1
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Prepare MNIST dataset for CapsNet training.

    Args:
        batch_size: Batch size for training.
        validation_split: Fraction of training data to use for validation.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Add channel dimension
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # Convert labels to one-hot encoding
    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # Create validation split
    val_size = int(len(x_train) * validation_split)
    x_val = x_train[:val_size]
    y_val = y_train[:val_size]
    x_train = x_train[val_size:]
    y_train = y_train[val_size:]

    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset


def create_basic_capsnet(
        input_shape: Tuple[int, int, int] = (28, 28, 1),
        num_classes: int = 10
) -> CapsNet:
    """Create a basic CapsNet model for MNIST.

    Args:
        input_shape: Shape of input images.
        num_classes: Number of output classes.

    Returns:
        CapsNet model instance.
    """
    model = CapsNet(
        num_classes=num_classes,
        routing_iterations=3,
        conv_filters=[256, 256],
        primary_capsules=32,
        primary_capsule_dim=8,
        digit_capsule_dim=16,
        reconstruction=True,
        input_shape=input_shape,
        decoder_architecture=[512, 1024],
        kernel_initializer="he_normal",
        use_batch_norm=True,
        name="mnist_capsnet"
    )

    return model


def train_capsnet_example():
    """Complete example of training CapsNet on MNIST."""
    print("CapsNet Training Example")
    print("=" * 50)

    # Prepare data
    print("Preparing MNIST dataset...")
    train_ds, val_ds, test_ds = prepare_mnist_data(batch_size=32)

    # Create model
    print("Creating CapsNet model...")
    model = create_basic_capsnet()

    # Build model by calling it with sample data
    sample_input = tf.zeros((1, 28, 28, 1))
    _ = model(sample_input, training=False)

    # Print model summary
    print("\nModel Architecture:")
    model.summary()

    # Create trainer
    print("\nCreating trainer...")
    loss_config = {
        "margin_loss_weight": 1.0,
        "reconstruction_weight": 0.0005,
        "positive_margin": 0.9,
        "negative_margin": 0.1,
        "downweight": 0.5
    }

    trainer = create_capsnet_trainer(
        model=model,
        learning_rate=0.001,
        optimizer_name="adam",
        loss_config=loss_config,
        metrics=["accuracy", "top_3_accuracy"],
        gradient_clip_norm=5.0
    )

    # Create callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        ),
        keras.callbacks.ModelCheckpoint(
            filepath='best_capsnet.keras',
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False
        )
    ]

    # Train model
    print("\nStarting training...")
    history = trainer.fit(
        train_dataset=train_ds,
        validation_dataset=val_ds,
        epochs=20,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(test_ds, verbose=1)

    # Generate predictions
    print("\nGenerating predictions...")
    predictions = trainer.predict(test_ds, return_reconstructions=True)

    print(f"Predictions shape: {predictions['lengths'].shape}")
    if 'reconstructions' in predictions:
        print(f"Reconstructions shape: {predictions['reconstructions'].shape}")

    # Save final model
    print("\nSaving final model...")
    model.save_model("final_capsnet.keras")

    # Demonstrate model loading
    print("\nTesting model loading...")
    loaded_model = CapsNet.load_model("final_capsnet.keras")
    print("Model loaded successfully!")

    return history, test_metrics, predictions


def advanced_training_example():
    """Advanced training example with custom configurations."""
    print("Advanced CapsNet Training Example")
    print("=" * 50)

    # Create model with custom configuration
    model = CapsNet(
        num_classes=10,
        routing_iterations=5,  # More routing iterations
        conv_filters=[128, 256, 512],  # More conv layers
        primary_capsules=64,  # More primary capsules
        primary_capsule_dim=16,  # Larger capsule dimension
        digit_capsule_dim=32,  # Larger digit capsules
        reconstruction=True,
        input_shape=(28, 28, 1),
        decoder_architecture=[1024, 2048, 1024],  # Deeper decoder
        kernel_regularizer="l2",  # L2 regularization
        use_batch_norm=True
    )

    # Custom loss configuration
    loss_config = {
        "margin_loss_weight": 1.0,
        "reconstruction_weight": 0.001,  # Higher reconstruction weight
        "positive_margin": 0.95,  # Tighter positive margin
        "negative_margin": 0.05,  # Tighter negative margin
        "downweight": 0.3  # Different downweight
    }

    # Create trainer with AdamW optimizer
    trainer = CapsNetTrainer(
        model=model,
        optimizer=keras.optimizers.AdamW(
            learning_rate=0.0005,
            weight_decay=0.01
        ),
        loss_config=loss_config,
        metrics=["accuracy", "top_3_accuracy"],
        gradient_clip_norm=10.0  # Higher gradient clipping
    )

    # Custom learning rate schedule
    lr_schedule = keras.callbacks.LearningRateScheduler(
        lambda epoch: 0.0005 * (0.95 ** epoch)
    )

    # Prepare data
    train_ds, val_ds, test_ds = prepare_mnist_data(batch_size=64)

    # Build model
    sample_input = tf.zeros((1, 28, 28, 1))
    _ = model(sample_input, training=False)

    print("Advanced model summary:")
    model.summary()

    # Train with custom callbacks
    callbacks = [
        lr_schedule,
        keras.callbacks.EarlyStopping(patience=10),
        keras.callbacks.ModelCheckpoint(
            'advanced_capsnet.keras',
            save_best_only=True
        )
    ]

    history = trainer.fit(
        train_dataset=train_ds,
        validation_dataset=val_ds,
        epochs=30,
        callbacks=callbacks,
        verbose=1
    )

    return history


def inference_example():
    """Example of using trained CapsNet for inference."""
    print("CapsNet Inference Example")
    print("=" * 30)

    # Load a pre-trained model (assuming it exists)
    try:
        model = CapsNet.load_model("final_capsnet.keras")
        print("Loaded pre-trained model.")
    except:
        print("No pre-trained model found. Creating new model...")
        model = create_basic_capsnet()
        # Build model
        sample_input = tf.zeros((1, 28, 28, 1))
        _ = model(sample_input, training=False)

    # Prepare a single image for inference
    (x_test, y_test), _ = keras.datasets.mnist.load_data()

    # Take first 5 images
    sample_images = x_test[:5].astype('float32') / 255.0
    sample_images = np.expand_dims(sample_images, -1)
    sample_labels = y_test[:5]

    # Run inference
    print("Running inference on sample images...")
    outputs = model(sample_images, training=False)

    # Get predictions
    predicted_lengths = outputs["length"].numpy()
    predicted_classes = np.argmax(predicted_lengths, axis=1)

    print("Predictions:")
    for i in range(5):
        print(f"Image {i}: True={sample_labels[i]}, Predicted={predicted_classes[i]}")
        print(f"  Confidence: {predicted_lengths[i].max():.4f}")

    # Show reconstruction if available
    if "reconstructed" in outputs:
        reconstructions = outputs["reconstructed"].numpy()
        print(f"Reconstructions shape: {reconstructions.shape}")


if __name__ == "__main__":
    # Run basic training example
    print("Running basic training example...")
    history, test_metrics, predictions = train_capsnet_example()

    print("\nTraining completed!")
    print("Final test metrics:", test_metrics)

    # Run inference example
    print("\n" + "=" * 50)
    inference_example()

    # Optionally run advanced example
    print("\n" + "=" * 50)
    print("Would you like to run the advanced training example? (Uncomment the line below)")
    # advanced_history = advanced_training_example()
