"""
SoftSOM Layer MNIST Reconstruction Experiment with Convolutional Architecture - REFINED

A comprehensive experiment demonstrating the SoftSOMLayer's capabilities for
differentiable self-organizing map learning on MNIST digit reconstruction using
a convolutional encoder-decoder architecture with SoftSOM bottleneck.

REFINED VERSION: All visualizations are now saved to files following production patterns.

The experiment includes:
1. MNIST data loading and preprocessing (keeping spatial structure)
2. Convolutional encoder with strided convolutions for downsampling
3. Dense layer followed by batch normalization, activation, and SoftSOM
4. Convolutional decoder with transposed convolutions for upsampling
5. Training with proper loss combination
6. Comprehensive visualizations saved as high-quality files
7. Serialization testing following modern Keras 3 patterns

Usage:
    python softsom_mnist_experiment_refined.py
"""

import os
import tempfile
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional

import keras
from keras import ops, layers, callbacks, datasets

# Assuming these imports exist and work as intended
from dl_techniques.utils.logger import logger
from dl_techniques.layers.memory.som_nd_soft_layer import SoftSOMLayer
from dl_techniques.regularizers.soft_orthogonal import SoftOrthonormalConstraintRegularizer

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configure matplotlib for better plots
plt.style.use('default')
sns.set_palette("husl")


class SoftSOMExperiment:
    """
    Comprehensive experiment class for SoftSOM reconstruction on MNIST with convolutional architecture.

    This class encapsulates the complete experimental pipeline including data
    preparation, convolutional model creation, training, evaluation, and visualization of
    the SoftSOM layer's learned representations and reconstruction capabilities.

    REFINED: All visualizations are now saved to organized output directories.

    **Intent**: Demonstrate the SoftSOM layer's ability to learn topologically
    organized representations while performing reconstruction using a modern
    convolutional encoder-decoder architecture with spatial inductive biases.

    **Experimental Design**:
    - Convolutional encoder with strided convolutions for efficient downsampling
    - SoftSOM as differentiable bottleneck operating on learned features
    - Convolutional decoder with nearest-neighbor upsampling + Conv2D for reconstruction
    - Multiple visualization techniques for prototype and feature analysis
    - Comprehensive evaluation including reconstruction quality and topology
    - Serialization testing for production readiness
    - All outputs saved to organized directory structure

    Args:
        som_grid_shape: Tuple defining SOM grid dimensions. Defaults to (8, 8).
        som_temperature: Temperature parameter for softmax sharpness. Defaults to 0.5.
        latent_dim: Dimensionality of encoder output feeding SOM. Defaults to 64.
        batch_size: Training batch size. Defaults to 128.
        epochs: Number of training epochs. Defaults to 20.
        learning_rate: Learning rate for Adam optimizer. Defaults to 0.001.
        conv_activation: Activation function for convolutional layers. Defaults to 'relu'.
        output_dir: Base directory for saving all experiment outputs. Defaults to 'softsom_results'.

    Attributes:
        model: Complete autoencoder model with SoftSOM bottleneck
        encoder: Encoder portion of the autoencoder
        decoder: Decoder portion of the autoencoder
        som_layer: The SoftSOM layer instance for analysis
        history: Training history for visualization
        experiment_dir: Directory where all results are saved
    """

    def __init__(
            self,
            som_grid_shape: Tuple[int, int] = (8, 8),
            som_temperature: float = 0.5,
            latent_dim: int = 64,
            batch_size: int = 128,
            epochs: int = 20,
            learning_rate: float = 0.001,
            conv_activation: str = 'relu',
            output_dir: str = 'softsom_results'
    ) -> None:
        """Initialize the experiment with configuration parameters."""
        self.som_grid_shape = som_grid_shape
        self.som_temperature = som_temperature
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.conv_activation = conv_activation

        # Create timestamped experiment directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_dir = Path(output_dir) / f"softsom_mnist_{timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for organized output
        (self.experiment_dir / "visualizations").mkdir(exist_ok=True)
        (self.experiment_dir / "models").mkdir(exist_ok=True)
        (self.experiment_dir / "logs").mkdir(exist_ok=True)

        # Initialize model components (created during build)
        self.model: Optional[keras.Model] = None
        self.encoder: Optional[keras.Model] = None
        self.decoder: Optional[keras.Model] = None
        self.som_layer: Optional[SoftSOMLayer] = None
        self.history: Optional[keras.callbacks.History] = None

        # Data storage
        self.x_train: Optional[np.ndarray] = None
        self.x_test: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None

        logger.info(f"Initialized SoftSOM experiment with grid_shape={som_grid_shape}")
        logger.info(f"Results will be saved to: {self.experiment_dir}")

    def load_and_preprocess_data(self) -> None:
        """
        Load and preprocess MNIST dataset for convolutional reconstruction task.

        Applies normalization and reshaping suitable for convolutional autoencoder
        architecture while preserving spatial structure for convolutions.
        """
        logger.info("Loading and preprocessing MNIST dataset...")

        # Load MNIST data
        (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

        # Normalize pixel values to [0, 1] range
        self.x_train = x_train.astype('float32') / 255.0
        self.x_test = x_test.astype('float32') / 255.0
        self.y_train = y_train
        self.y_test = y_test

        # Add channel dimension for convolutions (28, 28) -> (28, 28, 1)
        self.x_train = np.expand_dims(self.x_train, axis=-1)
        self.x_test = np.expand_dims(self.x_test, axis=-1)

        logger.info(f"Data shapes - Train: {self.x_train.shape}, Test: {self.x_test.shape}")
        logger.info(f"Pixel value range - Min: {self.x_train.min():.3f}, Max: {self.x_train.max():.3f}")

    def create_som_autoencoder(self) -> keras.Model:
        """
        Create convolutional autoencoder model with SoftSOM bottleneck layer.

        The architecture uses strided convolutions for efficient downsampling,
        followed by a dense layer, batch normalization, activation, and SoftSOM
        as a topological bottleneck. Reconstruction is done via transposed convolutions.

        **Architecture**:
        ```
        Input(28, 28, 1)
           ↓
        Conv2D(32, 5x5, stride=2, linear) → BN → Activation → (14, 14, 32)
           ↓
        Conv2D(64, 5x5, stride=2, linear) → BN → Activation → (7, 7, 64)
           ↓
        Flatten → Dense(latent_dim, linear) → BN → ReLU
           ↓
        SoftSOM(8x8 grid, latent_dim features) ← Topological bottleneck
           ↓
        Dense(7*7*64) → Reshape(7, 7, 64)
           ↓
        Upsample(2x) → Conv2D(32, 3x3, linear) → BN → Activation → (14, 14, 32)
           ↓
        Upsample(2x) → Conv2D(1, 3x3, linear) → BN → Sigmoid → (28, 28, 1)
        ```
        """
        logger.info("Creating convolutional SoftSOM autoencoder architecture...")

        # --- ENCODER ---
        encoder_input = layers.Input(shape=(28, 28, 1), name="encoder_input")
        x = layers.Conv2D(32, 5, strides=2, padding='same', activation='linear', name='encoder_conv1')(encoder_input)
        x = layers.BatchNormalization(name='encoder_bn1')(x)
        x = layers.Activation(self.conv_activation, name='encoder_act1')(x)
        x = layers.Conv2D(64, 5, strides=2, padding='same', activation='linear', name='encoder_conv2')(x)
        x = layers.BatchNormalization(name='encoder_bn2')(x)
        x = layers.Activation(self.conv_activation, name='encoder_act2')(x)
        x = layers.Flatten(name='encoder_flatten')(x)
        encoder_output = layers.Dense(self.latent_dim, activation='linear', name='encoder_dense')(x)
        self.encoder = keras.Model(inputs=encoder_input, outputs=encoder_output, name="encoder")

        # --- SOM BOTTLENECK ---
        self.som_layer = SoftSOMLayer(
            grid_shape=self.som_grid_shape,
            input_dim=self.latent_dim,
            temperature=self.som_temperature,
            use_per_dimension_softmax=True,
            use_reconstruction_loss=False, # Rely on main autoencoder loss
            topological_weight=0.1,
            sharpness_weight=0.1,
            kernel_regularizer=SoftOrthonormalConstraintRegularizer(
                0.1, 0.0, 0.01),
            name="soft_som_bottleneck"
        )
        som_output = self.som_layer(encoder_output)

        # --- DECODER ---
        # Define the decoder layers once to ensure weight sharing
        decoder_layers_seq = [
            layers.Dense(7 * 7 * 64, name='decoder_dense'),
            layers.Reshape((7, 7, 64), name='decoder_reshape'),
            layers.UpSampling2D(size=(2, 2), interpolation='nearest', name='decoder_upsample1'),
            layers.Conv2D(32, 3, padding='same', activation='linear', name='decoder_conv1'),
            layers.BatchNormalization(name='decoder_bn1'),
            layers.Activation(self.conv_activation, name='decoder_act1'),
            layers.UpSampling2D(size=(2, 2), interpolation='nearest', name='decoder_upsample2'),
            layers.Conv2D(16, 3, padding='same', activation='linear', name='decoder_conv2'),
            layers.BatchNormalization(name='decoder_bn2'),
            layers.Activation(self.conv_activation, name='decoder_act2'),
            layers.Conv2D(1, 1, padding='same', activation='sigmoid', name='decoder_output'),
        ]

        # Build the standalone decoder model for visualization
        decoder_input = layers.Input(shape=(self.latent_dim,), name="decoder_input")
        dec_x = decoder_input
        for layer in decoder_layers_seq:
            dec_x = layer(dec_x)
        self.decoder = keras.Model(inputs=decoder_input, outputs=dec_x, name="decoder")

        # Build the full autoencoder model using the same decoder layers
        autoencoder_output = som_output
        for layer in decoder_layers_seq:
            autoencoder_output = layer(autoencoder_output)
        autoencoder = keras.Model(
            inputs=encoder_input,
            outputs=autoencoder_output,
            name="convolutional_softsom_autoencoder"
        )

        # --- COMPILE ---
        autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        return autoencoder

    def train_model(self) -> keras.callbacks.History:
        """
        Train the SoftSOM convolutional autoencoder with comprehensive monitoring.

        Uses early stopping and learning rate reduction callbacks to ensure
        optimal training while monitoring both reconstruction and SOM losses.

        Returns:
            Training history containing loss curves and metrics.
        """
        if self.model is None or self.x_train is None:
            raise ValueError("Model and data must be prepared before training")

        logger.info(f"Starting training for {self.epochs} epochs...")

        # Define callbacks for training optimization
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                filepath=str(self.experiment_dir / "models" / "best_model.keras"),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            callbacks.ProgbarLogger()
        ]

        # Train the model (reconstruction task: input == target)
        history = self.model.fit(
            self.x_train, self.x_train,  # Autoencoder: predict input from input
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(self.x_test, self.x_test),
            callbacks=callback_list,
            verbose=1
        )

        logger.info("Training completed successfully")
        return history

    def test_serialization(self) -> None:
        """
        Test complete serialization cycle following modern Keras 3 patterns.

        Verifies that the custom SoftSOM layer can be properly saved and loaded
        with identical reconstruction behavior, ensuring production readiness.

        Raises:
            AssertionError: If serialization test fails with output mismatch.
        """
        if self.model is None or self.x_test is None:
            raise ValueError("Model and test data must be available for serialization test")

        logger.info("Testing serialization cycle...")

        # Get original predictions
        test_sample = self.x_test[:10]  # Small sample for testing
        original_predictions = self.model.predict(test_sample, verbose=0)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'softsom_autoencoder.keras')

            # Save model
            self.model.save(model_path)
            logger.info(f"Model saved to {model_path}")

            # Load model
            # Ensure custom objects are registered if not done automatically
            custom_objects = {
                'SoftSOMLayer': SoftSOMLayer,
                'SoftOrthonormalConstraintRegularizer': SoftOrthonormalConstraintRegularizer
            }
            loaded_model = keras.models.load_model(model_path, custom_objects=custom_objects)
            loaded_predictions = loaded_model.predict(test_sample, verbose=0)
            logger.info("Model loaded successfully")

        # Verify identical predictions
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(original_predictions),
            keras.ops.convert_to_numpy(loaded_predictions),
            rtol=1e-6, atol=1e-6,
            err_msg="Serialization test failed: predictions don't match"
        )

        logger.info("Serialization test passed - model saves and loads correctly")

    def visualize_training_progress(self) -> None:
        """
        Visualize training progress with loss curves and metrics.

        Creates comprehensive plots showing training and validation losses,
        including the SOM's internal regularization losses.

        REFINED: Saves plots to file instead of displaying.
        """
        if self.history is None:
            logger.warning("No training history available for visualization")
            return

        try:
            output_path = self.experiment_dir / "visualizations"
            output_path.mkdir(parents=True, exist_ok=True)

            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Convolutional SoftSOM Autoencoder Training Progress', fontsize=16, fontweight='bold')

            # Training and validation loss
            axes[0, 0].plot(self.history.history['loss'], label='Training Loss', linewidth=2)
            axes[0, 0].plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
            axes[0, 0].set_title('Reconstruction Loss (MSE)')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # Mean Absolute Error
            axes[0, 1].plot(self.history.history['mae'], label='Training MAE', linewidth=2)
            axes[0, 1].plot(self.history.history['val_mae'], label='Validation MAE', linewidth=2)
            axes[0, 1].set_title('Mean Absolute Error')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('MAE')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # Learning rate progression (if available)
            if 'learning_rate' in self.history.history:
                axes[1, 0].plot(self.history.history['learning_rate'], linewidth=2, color='red')
                axes[1, 0].set_title('Learning Rate Schedule')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Learning Rate')
                axes[1, 0].set_yscale('log')
                axes[1, 0].grid(True, alpha=0.3)
            else:
                axes[1, 0].text(0.5, 0.5, 'Learning Rate\nData Not Available',
                                ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Learning Rate Schedule')

            # Training summary statistics
            final_train_loss = self.history.history['loss'][-1]
            final_val_loss = self.history.history['val_loss'][-1]
            best_val_loss = min(self.history.history['val_loss'])

            summary_text = f"""Training Summary:
                Final Training Loss: {final_train_loss:.6f}
                Final Validation Loss: {final_val_loss:.6f}
                Best Validation Loss: {best_val_loss:.6f}
                Total Epochs: {len(self.history.history['loss'])}
                
                SOM Configuration:
                Grid Shape: {self.som_grid_shape}
                Temperature: {self.som_temperature}
                Latent Dim: {self.latent_dim}
                
                Architecture: Convolutional
                Conv Pattern: Conv2D(linear) → BN → Activation
                Encoder: 2 strided Conv2D layers
                Decoder: UpSampling2D + Conv2D layers
                Activation: {self.conv_activation}
                """

            axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                            verticalalignment='top', fontfamily='monospace', fontsize=10)
            axes[1, 1].set_xlim(0, 1)
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].axis('off')

            plt.tight_layout()
            plt.savefig(output_path / 'training_progress.png', dpi=300, bbox_inches='tight')
            plt.close()

            logger.info("Training progress visualization saved")

        except Exception as e:
            logger.error(f"Failed to create training progress visualization: {e}")

    def visualize_reconstructions(self, n_samples: int = 10) -> None:
        """
        Visualize original vs reconstructed images to assess quality.

        Args:
            n_samples: Number of test samples to visualize. Defaults to 10.

        REFINED: Saves plots to file instead of displaying.
        """
        if self.model is None or self.x_test is None:
            logger.warning("Model and test data required for reconstruction visualization")
            return

        try:
            output_path = self.experiment_dir / "visualizations"
            output_path.mkdir(parents=True, exist_ok=True)

            # Get random test samples
            indices = np.random.choice(len(self.x_test), n_samples, replace=False)
            test_samples = self.x_test[indices]
            test_labels = self.y_test[indices]

            # Get reconstructions
            reconstructions = self.model.predict(test_samples, verbose=0)

            # Create visualization
            fig, axes = plt.subplots(3, n_samples, figsize=(n_samples * 2, 6))
            fig.suptitle('Original vs Reconstructed MNIST Digits (Convolutional)', fontsize=16, fontweight='bold')

            for i in range(n_samples):
                # Original images
                axes[0, i].imshow(test_samples[i].squeeze(), cmap='gray')
                axes[0, i].set_title(f'Original\n(Digit {test_labels[i]})')
                axes[0, i].axis('off')

                # Reconstructed images
                axes[1, i].imshow(reconstructions[i].squeeze(), cmap='gray')
                axes[1, i].set_title('Reconstructed')
                axes[1, i].axis('off')

                # Difference (error) images
                diff = np.abs(test_samples[i].squeeze() - reconstructions[i].squeeze())
                axes[2, i].imshow(diff, cmap='hot')
                axes[2, i].set_title(f'Error\n(MAE: {np.mean(diff):.3f})')
                axes[2, i].axis('off')

            plt.tight_layout()
            plt.savefig(output_path / 'reconstructions.png', dpi=300, bbox_inches='tight')
            plt.close()

            logger.info("Reconstruction visualization saved")

        except Exception as e:
            logger.error(f"Failed to create reconstruction visualization: {e}")

    def visualize_som_prototypes(self) -> None:
        """
        Visualize the learned SOM prototype vectors as images.

        Shows how the SoftSOM has organized different digit patterns across
        its grid, revealing the topological structure of learned representations.

        REFINED: Saves plots to file instead of displaying.
        """
        if self.som_layer is None or self.decoder is None:
            logger.warning("SOM layer and decoder not available for prototype visualization")
            return

        try:
            output_path = self.experiment_dir / "visualizations"
            output_path.mkdir(parents=True, exist_ok=True)

            # Get the learned prototype weight map
            weights_map = self.som_layer.get_weights_map()
            grid_h, grid_w = self.som_grid_shape

            # Reshape prototypes and decode them using the consistent decoder model
            prototypes_flat = ops.reshape(weights_map, (-1, self.latent_dim))
            decoded_prototypes = self.decoder.predict(prototypes_flat, verbose=0)

            # Reshape back to grid format for display
            prototype_images = decoded_prototypes.reshape(grid_h, grid_w, 28, 28)

            # Create visualization
            fig, axes = plt.subplots(grid_h, grid_w, figsize=(grid_w * 2, grid_h * 2))
            fig.suptitle('Learned SOM Prototype Vectors (Decoded as Images)',
                         fontsize=16, fontweight='bold')

            # Ensure axes is a 2D array for consistent indexing
            if grid_h == 1 and grid_w == 1:
                axes = np.array([[axes]])
            elif grid_h == 1:
                axes = axes.reshape(1, -1)
            elif grid_w == 1:
                axes = axes.reshape(-1, 1)

            for i in range(grid_h):
                for j in range(grid_w):
                    axes[i, j].imshow(prototype_images[i, j], cmap='gray')
                    axes[i, j].set_title(f'({i},{j})')
                    axes[i, j].axis('off')

            plt.tight_layout()
            plt.savefig(output_path / 'som_prototypes.png', dpi=300, bbox_inches='tight')
            plt.close()

            logger.info("SOM prototype visualization saved")

        except Exception as e:
            logger.error(f"Failed to create SOM prototype visualization: {e}")

    def visualize_som_assignments(self, n_samples: int = 5) -> None:
        """
        Visualize soft assignment patterns for sample inputs.

        Shows how different inputs activate different regions of the SOM grid,
        revealing the topological organization learned by the layer.

        Args:
            n_samples: Number of samples to visualize assignments for. Defaults to 5.

        REFINED: Saves plots to file instead of displaying.
        """
        if self.som_layer is None or self.encoder is None or self.x_test is None:
            logger.warning("Required components not available for assignment visualization")
            return

        try:
            output_path = self.experiment_dir / "visualizations"
            output_path.mkdir(parents=True, exist_ok=True)

            # Get random test samples
            indices = np.random.choice(len(self.x_test), n_samples, replace=False)
            test_samples = self.x_test[indices]
            test_labels = self.y_test[indices]

            # Get encoded representations (input to SOM)
            encoded_samples = self.encoder.predict(test_samples, verbose=0)

            # Get soft assignments from SOM
            assignments = self.som_layer.get_soft_assignments(encoded_samples)
            assignments_np = keras.ops.convert_to_numpy(assignments)

            # Create visualization
            fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 3, 6))
            fig.suptitle('Input Digits and Their SOM Soft Assignments',
                         fontsize=16, fontweight='bold')

            for i in range(n_samples):
                # Original digit
                axes[0, i].imshow(test_samples[i].squeeze(), cmap='gray')
                axes[0, i].set_title(f'Digit {test_labels[i]}')
                axes[0, i].axis('off')

                # Soft assignment heatmap
                im = axes[1, i].imshow(assignments_np[i], cmap='viridis', interpolation='nearest')
                axes[1, i].set_title('SOM Activation')
                axes[1, i].set_xlabel('Grid X')
                axes[1, i].set_ylabel('Grid Y')

                # Add colorbar
                plt.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)

            plt.tight_layout()
            plt.savefig(output_path / 'som_assignments.png', dpi=300, bbox_inches='tight')
            plt.close()

            logger.info("SOM assignment visualization saved")

        except Exception as e:
            logger.error(f"Failed to create SOM assignment visualization: {e}")

    def analyze_digit_clustering(self) -> None:
        """
        Analyze how different digits cluster on the SOM grid.

        Computes the average activation patterns for each digit class and
        visualizes the topological organization of digit representations.

        REFINED: Saves plots to file instead of displaying.
        """
        if self.som_layer is None or self.encoder is None or self.x_test is None:
            logger.warning("Required components not available for clustering analysis")
            return

        try:
            output_path = self.experiment_dir / "visualizations"
            output_path.mkdir(parents=True, exist_ok=True)

            logger.info("Analyzing digit clustering on SOM grid...")

            # Get encoded representations for all test samples
            encoded_test = self.encoder.predict(self.x_test, verbose=0, batch_size=self.batch_size)
            assignments = self.som_layer.get_soft_assignments(encoded_test)
            assignments_np = keras.ops.convert_to_numpy(assignments)

            # Compute average assignments for each digit
            digit_assignments = {}
            for digit in range(10):
                digit_mask = self.y_test == digit
                digit_avg = np.mean(assignments_np[digit_mask], axis=0)
                digit_assignments[digit] = digit_avg

            # Create visualization
            fig, axes = plt.subplots(2, 5, figsize=(15, 6))
            fig.suptitle('Average SOM Activations by Digit Class', fontsize=16, fontweight='bold')

            for digit in range(10):
                row = digit // 5
                col = digit % 5

                im = axes[row, col].imshow(digit_assignments[digit], cmap='viridis')
                axes[row, col].set_title(f'Digit {digit}')
                axes[row, col].set_xlabel('Grid X')
                axes[row, col].set_ylabel('Grid Y')

                # Add colorbar
                plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)

            plt.tight_layout()
            plt.savefig(output_path / 'digit_clustering.png', dpi=300, bbox_inches='tight')
            plt.close()

            # Compute and display clustering statistics
            logger.info("Computing clustering quality metrics...")

            # Find best matching units (BMUs) for each test sample
            assignments_flat = assignments_np.reshape(len(assignments_np), -1)
            bmus = np.argmax(assignments_flat, axis=1)

            # Compute purity for each grid position
            grid_size = np.prod(self.som_grid_shape)
            position_purity = {}

            for pos in range(grid_size):
                pos_mask = bmus == pos
                if np.sum(pos_mask) > 0:
                    pos_labels = self.y_test[pos_mask]
                    unique_labels, counts = np.unique(pos_labels, return_counts=True)
                    purity = np.max(counts) / np.sum(counts)
                    position_purity[pos] = (purity, unique_labels[np.argmax(counts)])

            if position_purity:
                avg_purity = np.mean([purity for purity, _ in position_purity.values()])
                logger.info(f"Average grid position purity: {avg_purity:.3f}")
            else:
                logger.info("No samples mapped to any grid position for purity calculation.")

            logger.info(f"Positions with samples: {len(position_purity)}/{grid_size}")

            # Save clustering statistics
            with open(self.experiment_dir / "logs" / "clustering_stats.txt", 'w') as f:
                f.write("SOM Clustering Analysis Results\n")
                f.write("=" * 50 + "\n")
                f.write(f"SOM Grid Shape: {self.som_grid_shape}\n")
                f.write(f"Total Grid Positions: {grid_size}\n")
                f.write(f"Positions with samples: {len(position_purity)}\n")
                if position_purity:
                    f.write(f"Average position purity: {avg_purity:.4f}\n")
                f.write("\nPosition Details:\n")
                for pos, (purity, dominant_class) in position_purity.items():
                    f.write(f"Position {pos}: Purity={purity:.3f}, Dominant Class={dominant_class}\n")

            logger.info("Digit clustering analysis saved")

        except Exception as e:
            logger.error(f"Failed to create digit clustering analysis: {e}")

    def save_experiment_summary(self) -> None:
        """
        Save comprehensive experiment summary and configuration to files.
        """
        try:
            # Save experiment configuration
            config_dict = {
                'som_grid_shape': self.som_grid_shape,
                'som_temperature': self.som_temperature,
                'latent_dim': self.latent_dim,
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'learning_rate': self.learning_rate,
                'conv_activation': self.conv_activation,
                'experiment_dir': str(self.experiment_dir)
            }

            import json
            with open(self.experiment_dir / "experiment_config.json", 'w') as f:
                json.dump(config_dict, f, indent=2)

            # Save training history if available
            if self.history is not None:
                with open(self.experiment_dir / "training_history.json", 'w') as f:
                    # Convert numpy arrays to lists for JSON serialization
                    history_dict = {}
                    for key, values in self.history.history.items():
                        history_dict[key] = [float(v) for v in values]
                    json.dump(history_dict, f, indent=2)

            # Save model summary
            if self.model is not None:
                with open(self.experiment_dir / "logs" / "model_summary.txt", 'w') as f:
                    self.model.summary(print_fn=lambda x: f.write(x + '\n'))

            logger.info("Experiment summary saved")

        except Exception as e:
            logger.error(f"Failed to save experiment summary: {e}")

    def run_complete_experiment(self) -> None:
        """
        Execute the complete experimental pipeline.

        Runs all phases of the experiment including data loading, model creation,
        training, evaluation, and comprehensive visualization of results.
        All outputs are saved to organized directory structure.
        """
        logger.info("=" * 60)
        logger.info("STARTING CONVOLUTIONAL SOFTSOM MNIST RECONSTRUCTION EXPERIMENT")
        logger.info("=" * 60)

        try:
            # Phase 1: Data preparation
            self.load_and_preprocess_data()

            # Phase 2: Model creation
            self.model = self.create_som_autoencoder()
            self.model.summary()
            logger.info(f"Model created with {self.model.count_params():,} parameters")

            # Phase 3: Model training
            self.history = self.train_model()

            # Phase 4: Save final model
            self.model.save(self.experiment_dir / "models" / "final_model.keras")
            logger.info("Final model saved")

            # Phase 5: Serialization testing
            self.test_serialization()

            # Phase 6: Comprehensive visualization and analysis
            logger.info("Generating and saving visualizations...")

            self.visualize_training_progress()
            self.visualize_reconstructions(n_samples=8)
            self.visualize_som_prototypes()
            self.visualize_som_assignments(n_samples=5)
            self.analyze_digit_clustering()

            # Phase 7: Save experiment summary
            self.save_experiment_summary()

            logger.info("=" * 60)
            logger.info("EXPERIMENT COMPLETED SUCCESSFULLY")
            logger.info(f"All results saved to: {self.experiment_dir}")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"Experiment failed with error: {e}", exc_info=True)
            raise


def main() -> None:
    """
    Main function to run the convolutional SoftSOM MNIST reconstruction experiment.

    Configures and executes the complete experimental pipeline with
    optimal parameters for demonstrating SoftSOM capabilities with
    convolutional architecture. All outputs are saved to files.
    """
    # Configure experiment parameters
    experiment = SoftSOMExperiment(
        som_grid_shape=(4, 4),  # 4x4 SOM grid for good visualization
        som_temperature=0.1,  # Moderate temperature for balanced assignments
        latent_dim=16,  # Sufficient dimensionality for convolutional features
        batch_size=128,  # Increased batch size for stable training
        epochs=100,  # Sufficient epochs for convolutional training
        learning_rate=0.001,  # Standard learning rate for Adam
        conv_activation='relu',  # Configurable activation for conv layers
        output_dir='results'  # Base directory for all outputs
    )

    # Run the complete experiment
    experiment.run_complete_experiment()


if __name__ == "__main__":
    main()