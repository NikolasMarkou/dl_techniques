"""
Example implementation of CCNet framework for MNIST digit generation.
Demonstrates how to plug in Keras models into the CCNet meta-framework.
"""

import keras
import tensorflow as tf
import numpy as np
from typing import Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.models.ccnets import (
    CCNetOrchestrator,
    CCNetConfig,
    CCNetTrainer,
    EarlyStoppingCallback,
    wrap_keras_model
)
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------
# Model Definitions
# ---------------------------------------------------------------------

class MNISTExplainer(keras.Model):
    """
    Explainer network for MNIST: P(E|X)
    Extracts latent style/context from digit images.
    """

    def __init__(
            self,
            explanation_dim: int = 128,
            dropout_rate: float = 0.2,
            l2_regularization: float = 1e-4,
            **kwargs
    ):
        """
        Initialize MNIST Explainer.

        Args:
            explanation_dim: Dimension of latent explanation vector E.
            dropout_rate: Dropout rate for regularization.
            l2_regularization: L2 regularization strength.
        """
        super().__init__(**kwargs)
        self.explanation_dim = explanation_dim
        self.dropout_rate = dropout_rate

        # Create regularizer
        self.regularizer = keras.regularizers.L2(l2_regularization) if l2_regularization > 0 else None

        # Layers will be built in build() method
        self.conv1 = None
        self.conv2 = None
        self.conv3 = None
        self.pool = None
        self.flatten = None
        self.dropout = None
        self.fc1 = None
        self.fc2 = None
        self.fc_latent = None
        self.batch_norm1 = None
        self.batch_norm2 = None

    def build(self, input_shape: Tuple[int, ...]):
        """
        Build the model layers.

        Args:
            input_shape: Shape of input tensor.
        """
        super().build(input_shape)

        # Convolutional encoder
        self.conv1 = keras.layers.Conv2D(
            32, 3, activation='relu', padding='same',
            kernel_regularizer=self.regularizer
        )
        self.conv2 = keras.layers.Conv2D(
            64, 3, activation='relu', padding='same',
            kernel_regularizer=self.regularizer
        )
        self.conv3 = keras.layers.Conv2D(
            128, 3, activation='relu', padding='same',
            kernel_regularizer=self.regularizer
        )

        self.pool = keras.layers.MaxPooling2D(2)
        self.flatten = keras.layers.Flatten()
        self.dropout = keras.layers.Dropout(self.dropout_rate)

        # Batch normalization
        self.batch_norm1 = keras.layers.BatchNormalization()
        self.batch_norm2 = keras.layers.BatchNormalization()

        # Fully connected layers
        self.fc1 = keras.layers.Dense(
            512, activation='relu',
            kernel_regularizer=self.regularizer
        )
        self.fc2 = keras.layers.Dense(
            256, activation='relu',
            kernel_regularizer=self.regularizer
        )

        # Latent representation (no activation for flexibility)
        self.fc_latent = keras.layers.Dense(
            self.explanation_dim,
            kernel_regularizer=self.regularizer
        )

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Forward pass through the explainer.

        Args:
            x: Input image tensor [batch, 28, 28, 1].
            training: Whether in training mode.

        Returns:
            Latent explanation vector [batch, explanation_dim].
        """
        # Convolutional encoding
        x = self.conv1(x)
        x = self.pool(x)
        x = self.batch_norm1(x, training=training)

        x = self.conv2(x)
        x = self.pool(x)
        x = self.batch_norm2(x, training=training)

        x = self.conv3(x)
        x = self.pool(x)

        # Flatten and dense layers
        x = self.flatten(x)
        x = self.dropout(x, training=training)

        x = self.fc1(x)
        x = self.dropout(x, training=training)

        x = self.fc2(x)

        # Output latent representation
        e_latent = self.fc_latent(x)

        return e_latent


class MNISTReasoner(keras.Model):
    """
    Reasoner network for MNIST: P(Y|X,E)
    Performs context-aware digit classification.
    """

    def __init__(
            self,
            num_classes: int = 10,
            explanation_dim: int = 128,
            dropout_rate: float = 0.2,
            l2_regularization: float = 1e-4,
            **kwargs
    ):
        """
        Initialize MNIST Reasoner.

        Args:
            num_classes: Number of output classes.
            explanation_dim: Dimension of latent explanation vector E.
            dropout_rate: Dropout rate for regularization.
            l2_regularization: L2 regularization strength.
        """
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.explanation_dim = explanation_dim
        self.dropout_rate = dropout_rate

        # Create regularizer
        self.regularizer = keras.regularizers.L2(l2_regularization) if l2_regularization > 0 else None

        # Layers will be built in build() method
        self.conv1 = None
        self.conv2 = None
        self.pool = None
        self.flatten = None
        self.dropout = None
        self.fc_combine = None
        self.fc_hidden = None
        self.fc_output = None
        self.batch_norm = None

    def build(self, input_shape):
        """Build the model layers."""
        super().build(input_shape)

        # Image processing branch
        self.conv1 = keras.layers.Conv2D(
            32, 3, activation='relu', padding='same',
            kernel_regularizer=self.regularizer
        )
        self.conv2 = keras.layers.Conv2D(
            64, 3, activation='relu', padding='same',
            kernel_regularizer=self.regularizer
        )
        self.pool = keras.layers.MaxPooling2D(2)
        self.flatten = keras.layers.Flatten()
        self.batch_norm = keras.layers.BatchNormalization()

        # Combination and reasoning layers
        self.fc_combine = keras.layers.Dense(
            512, activation='relu',
            kernel_regularizer=self.regularizer
        )
        self.fc_hidden = keras.layers.Dense(
            256, activation='relu',
            kernel_regularizer=self.regularizer
        )
        self.dropout = keras.layers.Dropout(self.dropout_rate)

        # Output layer (softmax for classification)
        self.fc_output = keras.layers.Dense(
            self.num_classes,
            activation='softmax',
            kernel_regularizer=self.regularizer
        )

    def call(
            self,
            x: tf.Tensor,
            e: tf.Tensor,
            training: bool = False
    ) -> tf.Tensor:
        """
        Forward pass through the reasoner.

        Args:
            x: Input image tensor [batch, 28, 28, 1].
            e: Latent explanation vector [batch, explanation_dim].
            training: Whether in training mode.

        Returns:
            Class probabilities [batch, num_classes].
        """
        # Process image
        img_features = self.conv1(x)
        img_features = self.pool(img_features)
        img_features = self.batch_norm(img_features, training=training)

        img_features = self.conv2(img_features)
        img_features = self.pool(img_features)

        img_features = self.flatten(img_features)

        # Combine image features with explanation
        combined = keras.ops.concatenate([img_features, e], axis=-1)

        # Reasoning layers
        combined = self.fc_combine(combined)
        combined = self.dropout(combined, training=training)

        combined = self.fc_hidden(combined)
        combined = self.dropout(combined, training=training)

        # Output classification
        y_probs = self.fc_output(combined)

        return y_probs


class MNISTProducer(keras.Model):
    """
    Producer network for MNIST: P(X|Y,E)
    Generates/reconstructs digit images from label and style.
    """

    def __init__(
            self,
            num_classes: int = 10,
            explanation_dim: int = 128,
            dropout_rate: float = 0.2,
            l2_regularization: float = 1e-4,
            **kwargs
    ):
        """
        Initialize MNIST Producer.

        Args:
            num_classes: Number of input classes.
            explanation_dim: Dimension of latent explanation vector E.
            dropout_rate: Dropout rate for regularization.
            l2_regularization: L2 regularization strength.
        """
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.explanation_dim = explanation_dim
        self.dropout_rate = dropout_rate

        # Create regularizer
        self.regularizer = keras.regularizers.L2(l2_regularization) if l2_regularization > 0 else None

        # Layers will be built in build() method
        self.fc_input = None
        self.fc_hidden1 = None
        self.fc_hidden2 = None
        self.fc_reshape = None
        self.dropout = None
        self.reshape = None
        self.conv_transpose1 = None
        self.conv_transpose2 = None
        self.conv_transpose3 = None
        self.conv_output = None
        self.batch_norm1 = None
        self.batch_norm2 = None

    def build(self, input_shape):
        """Build the model layers."""
        super().build(input_shape)

        # Initial processing of combined input
        self.fc_input = keras.layers.Dense(
            256, activation='relu',
            kernel_regularizer=self.regularizer
        )
        self.fc_hidden1 = keras.layers.Dense(
            512, activation='relu',
            kernel_regularizer=self.regularizer
        )
        self.fc_hidden2 = keras.layers.Dense(
            1024, activation='relu',
            kernel_regularizer=self.regularizer
        )

        # Reshape for convolutional decoder
        self.fc_reshape = keras.layers.Dense(
            7 * 7 * 128, activation='relu',
            kernel_regularizer=self.regularizer
        )
        self.reshape = keras.layers.Reshape((7, 7, 128))

        self.dropout = keras.layers.Dropout(self.dropout_rate)

        # Batch normalization
        self.batch_norm1 = keras.layers.BatchNormalization()
        self.batch_norm2 = keras.layers.BatchNormalization()

        # Convolutional decoder
        self.conv_transpose1 = keras.layers.Conv2DTranspose(
            64, 3, strides=2, padding='same', activation='relu',
            kernel_regularizer=self.regularizer
        )
        self.conv_transpose2 = keras.layers.Conv2DTranspose(
            32, 3, strides=2, padding='same', activation='relu',
            kernel_regularizer=self.regularizer
        )
        self.conv_transpose3 = keras.layers.Conv2DTranspose(
            16, 3, padding='same', activation='relu',
            kernel_regularizer=self.regularizer
        )

        # Output layer (sigmoid for pixel values)
        self.conv_output = keras.layers.Conv2D(
            1, 3, padding='same', activation='sigmoid',
            kernel_regularizer=self.regularizer
        )

    def call(
            self,
            y: tf.Tensor,
            e: tf.Tensor,
            training: bool = False
    ) -> tf.Tensor:
        """
        Forward pass through the producer.

        Args:
            y: Class labels (one-hot) [batch, num_classes].
            e: Latent explanation vector [batch, explanation_dim].
            training: Whether in training mode.

        Returns:
            Generated/reconstructed image [batch, 28, 28, 1].
        """
        # Combine label and explanation
        combined = keras.ops.concatenate([y, e], axis=-1)

        # Process through dense layers
        x = self.fc_input(combined)
        x = self.dropout(x, training=training)

        x = self.fc_hidden1(x)
        x = self.dropout(x, training=training)

        x = self.fc_hidden2(x)
        x = self.fc_reshape(x)

        # Reshape for deconvolution
        x = self.reshape(x)

        # Deconvolutional generation
        x = self.conv_transpose1(x)  # -> 14x14
        x = self.batch_norm1(x, training=training)

        x = self.conv_transpose2(x)  # -> 28x28
        x = self.batch_norm2(x, training=training)

        x = self.conv_transpose3(x)

        # Output image
        x_generated = self.conv_output(x)

        return x_generated


# ---------------------------------------------------------------------
# Training Script
# ---------------------------------------------------------------------

def create_mnist_ccnet(
        explanation_dim: int = 128,
        learning_rate: float = 1e-3,
        dropout_rate: float = 0.2,
        l2_regularization: float = 1e-4
) -> CCNetOrchestrator:
    """
    Create a complete CCNet for MNIST.

    Args:
        explanation_dim: Dimension of latent explanation.
        learning_rate: Learning rate for all modules.
        dropout_rate: Dropout rate for regularization.
        l2_regularization: L2 regularization strength.

    Returns:
        Configured CCNetOrchestrator.
    """
    # Create the three models
    explainer = MNISTExplainer(
        explanation_dim=explanation_dim,
        dropout_rate=dropout_rate,
        l2_regularization=l2_regularization
    )

    reasoner = MNISTReasoner(
        num_classes=10,
        explanation_dim=explanation_dim,
        dropout_rate=dropout_rate,
        l2_regularization=l2_regularization
    )

    producer = MNISTProducer(
        num_classes=10,
        explanation_dim=explanation_dim,
        dropout_rate=dropout_rate,
        l2_regularization=l2_regularization
    )

    # Build models with dummy input to initialize weights
    dummy_image = keras.ops.zeros((1, 28, 28, 1))
    dummy_label = keras.ops.zeros((1, 10))
    dummy_latent = keras.ops.zeros((1, explanation_dim))

    explainer(dummy_image)
    reasoner(dummy_image, dummy_latent)
    producer(dummy_label, dummy_latent)

    # Wrap models for CCNet compatibility
    explainer_wrapped = wrap_keras_model(explainer)
    reasoner_wrapped = wrap_keras_model(reasoner)
    producer_wrapped = wrap_keras_model(producer)

    # Create configuration
    config = CCNetConfig(
        explanation_dim=explanation_dim,
        loss_type='l2',
        learning_rates={
            'explainer': learning_rate,
            'reasoner': learning_rate,
            'producer': learning_rate
        },
        gradient_clip_norm=1.0,
        use_mixed_precision=False,
        sequential_data=False,
        verification_weight=1.0
    )

    # Create orchestrator
    orchestrator = CCNetOrchestrator(
        explainer=explainer_wrapped,
        reasoner=reasoner_wrapped,
        producer=producer_wrapped,
        config=config
    )

    return orchestrator


def prepare_mnist_data() -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Prepare MNIST dataset for CCNet training.

    Returns:
        Tuple of (train_dataset, val_dataset).
    """
    # Load MNIST
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Preprocess images
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Add channel dimension
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    # Convert labels to one-hot
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.batch(32).shuffle(1000).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    val_dataset = val_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset


def train_mnist_ccnet():
    """
    Main training script for MNIST CCNet.
    """
    logger.info("Creating CCNet for MNIST...")
    orchestrator = create_mnist_ccnet(
        explanation_dim=128,
        learning_rate=1e-3,
        dropout_rate=0.2,
        l2_regularization=1e-4
    )

    logger.info("Preparing data...")
    train_dataset, val_dataset = prepare_mnist_data()

    logger.info("Setting up trainer...")
    trainer = CCNetTrainer(orchestrator)

    # Create early stopping callback
    early_stopping = EarlyStoppingCallback(patience=10, threshold=1e-4)

    # Custom callback for printing counterfactual examples
    def counterfactual_callback(epoch, metrics, orch):
        if epoch % 5 == 0:
            logger.info(f"\n--- Epoch {epoch} Counterfactual Test ---")

            # Get a sample batch
            for x_batch, y_batch in val_dataset.take(1):
                # Take first image (e.g., a '3')
                x_ref = x_batch[:1]

                # Create target label (e.g., '8')
                y_target = keras.ops.zeros((1, 10))
                y_target = keras.ops.scatter_update(y_target, [[0, 8]], [1.0])

                # Generate counterfactual
                x_counter = orch.counterfactual_generation(x_ref, y_target)

                logger.info(f"Generated counterfactual shape: {x_counter.shape}")
                break

    logger.info("Starting training...")
    try:
        trainer.train(
            train_dataset=train_dataset,
            epochs=10,
            validation_dataset=val_dataset,
            callbacks=[
                early_stopping,
                counterfactual_callback
            ]
        )
    except StopIteration:
        logger.info("Training stopped early due to convergence.")

    logger.info("Training complete!")
    logger.info(f"Final losses:")
    for key in ['generation_loss', 'reconstruction_loss', 'inference_loss']:
        if trainer.history[key]:
            logger.info(f"  {key}: {trainer.history[key][-1]:.4f}")

    # Save models
    logger.info("Saving models...")
    orchestrator.save_models("mnist_ccnet")
    logger.info("Models saved successfully!")

    return orchestrator, trainer


# ---------------------------------------------------------------------
# Inference Examples
# ---------------------------------------------------------------------

def demonstrate_ccnet_capabilities(orchestrator: CCNetOrchestrator):
    """
    Demonstrate the unique capabilities of trained CCNet.

    Args:
        orchestrator: Trained CCNet orchestrator.
    """
    logger.info("=" * 60)
    logger.info("CCNet Capability Demonstrations")
    logger.info("=" * 60)

    # Load test data
    (x_test, y_test) = keras.datasets.mnist.load_data()[1]
    x_test = x_test.astype('float32') / 255.0
    x_test = np.expand_dims(x_test, axis=-1)
    y_test_onehot = keras.utils.to_categorical(y_test, 10)

    # 1. Disentanglement
    logger.info("1. Cause Disentanglement:")
    sample_idx = 0
    x_sample = x_test[sample_idx:sample_idx + 1]
    y_explicit, e_latent = orchestrator.disentangle_causes(x_sample)

    logger.info(f"   Original label: {y_test[sample_idx]}")
    logger.info(f"   Inferred label: {np.argmax(y_explicit)}")
    logger.info(f"   Latent dimension: {e_latent.shape}")

    # 2. Counterfactual Generation
    logger.info("2. Counterfactual Generation:")
    # Change a '3' to look like an '8' in the same style
    idx_3 = np.where(y_test == 3)[0][0]
    x_3 = x_test[idx_3:idx_3 + 1]

    y_8 = keras.ops.zeros((1, 10))
    y_8 = keras.ops.scatter_update(y_8, [[0, 8]], [1.0])

    x_counterfactual = orchestrator.counterfactual_generation(x_3, y_8)
    logger.info(f"   Generated '8' in style of '3': shape {x_counterfactual.shape}")

    # 3. Style Transfer
    logger.info("3. Style Transfer:")
    idx_content = np.where(y_test == 7)[0][0]
    idx_style = np.where(y_test == 4)[0][0]

    x_content = x_test[idx_content:idx_content + 1]
    x_style = x_test[idx_style:idx_style + 1]

    x_transferred = orchestrator.style_transfer(x_content, x_style)
    logger.info(f"   Transferred '7' with style of '4': shape {x_transferred.shape}")

    # 4. Consistency Verification
    logger.info("4. Internal Consistency Check:")
    consistent_samples = 0
    for i in range(100):
        if orchestrator.verify_consistency(x_test[i:i + 1], threshold=0.05):
            consistent_samples += 1

    logger.info(f"   Consistency rate: {consistent_samples}/100 samples")

    logger.info( "=" * 60)

# ==============================================================================


if __name__ == "__main__":
    # Train the CCNet
    orchestrator, trainer = train_mnist_ccnet()

    # Demonstrate capabilities
    demonstrate_ccnet_capabilities(orchestrator)

    logger.info("\nCCNet training and demonstration complete!")