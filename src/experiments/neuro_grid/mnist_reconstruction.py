"""
NeuroGrid Convolutional Autoencoder Experiment on MNIST

A comprehensive convolutional autoencoder experiment demonstrating NeuroGrid's structured memory capabilities
for learning organized representations in a spatial neural network architecture with conv-deconv pattern.
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
from keras import layers, callbacks, datasets

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.neuro_grid import NeuroGrid
from dl_techniques.regularizers.soft_orthogonal import SoftOrthonormalConstraintRegularizer

# ---------------------------------------------------------------------

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configure matplotlib for better plots
plt.style.use('default')
sns.set_palette("husl")


class NeuroGridAutoencoderExperiment:
    """
    This experiment showcases how NeuroGrid can serve as an intelligent bottleneck layer
    between convolutional encoder and decoder networks, learning spatially organized
    representations while maintaining differentiability for end-to-end training. The
    conv-deconv architecture with NeuroGrid bottleneck encourages similar spatial features
    to activate nearby grid regions.

    **Intent**: Demonstrate NeuroGrid's ability to learn topologically organized
    representations in a convolutional autoencoder architecture, showcasing its structured
    memory capabilities, addressing behavior, and quality assessment features.

    **Architecture**:
    ```
    Input(28,28,1) → Conv2D → Conv2D → Flatten → Dense(latent_dim) →
    NeuroGrid(grid, latent_dim) →
    Dense → Reshape → UpSample+Conv2D → UpSample+Conv2D → Conv2D(1) → Output(28,28,1)
    ```

    Args:
        grid_shape: Tuple defining NeuroGrid dimensions. Defaults to (8, 8).
        latent_dim: Dimensionality of NeuroGrid latent vectors. Defaults to 64.
        temperature: Initial temperature for NeuroGrid addressing. Defaults to 1.0.
        conv_activation: Activation function for convolutional layers. Defaults to 'gelu'.
        batch_size: Training batch size. Defaults to 128.
        epochs: Number of training epochs. Defaults to 50.
        learning_rate: Learning rate for Adam optimizer. Defaults to 0.001.
        output_dir: Base directory for saving results. Defaults to 'neurogrid_conv_results'.
    """

    def __init__(
            self,
            grid_shape: Tuple[int, int] = (8, 8),
            latent_dim: int = 64,
            temperature: float = 1.0,
            conv_activation: str = 'relu',
            batch_size: int = 128,
            epochs: int = 50,
            learning_rate: float = 0.001,
            output_dir: str = 'results'
    ) -> None:
        """Initialize the experiment with configuration parameters."""
        self.grid_shape = grid_shape
        self.latent_dim = latent_dim
        self.temperature = temperature
        self.conv_activation = conv_activation
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

        # Create timestamped experiment directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_dir = Path(output_dir) / f"neurogrid_autoencoder_{timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.experiment_dir / "visualizations").mkdir(exist_ok=True)
        (self.experiment_dir / "models").mkdir(exist_ok=True)
        (self.experiment_dir / "logs").mkdir(exist_ok=True)

        # Initialize model components
        self.model: Optional[keras.Model] = None
        self.encoder: Optional[keras.Model] = None
        self.decoder: Optional[keras.Model] = None
        self.neurogrid_layer: Optional[NeuroGrid] = None
        self.history: Optional[keras.callbacks.History] = None

        # Data storage
        self.x_train: Optional[np.ndarray] = None
        self.x_test: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None

        logger.info(f"Initialized NeuroGrid convolutional autoencoder with grid_shape={grid_shape}")
        logger.info(f"Results will be saved to: {self.experiment_dir}")

    def load_and_preprocess_data(self) -> None:
        """
        Load and preprocess MNIST dataset for convolutional autoencoder architecture.

        Keeps images in 2D spatial format and adds channel dimension for Conv2D layers.
        """
        logger.info("Loading and preprocessing MNIST dataset...")

        # Load MNIST data
        (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

        # Keep 2D spatial format and add channel dimension for Conv2D
        self.x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255.0
        self.x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255.0
        self.y_train = y_train
        self.y_test = y_test

        logger.info(f"Data shapes - Train: {self.x_train.shape}, Test: {self.x_test.shape}")
        logger.info(f"Pixel value range - Min: {self.x_train.min():.3f}, Max: {self.x_train.max():.3f}")

    def create_neurogrid_conv_autoencoder(self) -> keras.Model:
        """
        Create convolutional autoencoder model with NeuroGrid structured memory bottleneck.

        Returns:
            Compiled autoencoder model with NeuroGrid bottleneck.
        """
        logger.info("Creating NeuroGrid convolutional autoencoder architecture...")

        # --- ENCODER ---
        encoder_input = layers.Input(shape=(28, 28, 1), name="encoder_input")
        x = layers.Conv2D(16, 4, strides=2, padding='same', activation='linear', name='encoder_conv1')(encoder_input)
        x = layers.BatchNormalization(name='encoder_bn1')(x)
        x = layers.Activation(self.conv_activation, name='encoder_act1')(x)
        x = layers.Conv2D(32, 4, strides=2, padding='same', activation='linear', name='encoder_conv2')(x)
        x = layers.BatchNormalization(name='encoder_bn2')(x)
        x = layers.Activation(self.conv_activation, name='encoder_act2')(x)
        x = layers.Conv2D(64, 4, strides=2, padding='same', activation='linear', name='encoder_conv3')(x)
        x = layers.BatchNormalization(name='encoder_bn3')(x)
        x = layers.Activation(self.conv_activation, name='encoder_act3')(x)
        x = layers.Flatten(name='encoder_flatten')(x)
        x = layers.Dense(self.latent_dim, activation='linear', name='encoder_dense')(x)
        encoder_output = layers.BatchNormalization(name='encoder_dense_bn')(x)
        self.encoder = keras.Model(inputs=encoder_input, outputs=encoder_output, name="encoder")

        # --- NEUROGRID BOTTLENECK ---
        self.neurogrid_layer = NeuroGrid(
            grid_shape=list(self.grid_shape),
            latent_dim=self.latent_dim,
            temperature=self.temperature,
            learnable_temperature=False,
            entropy_regularizer_strength=0.01,  # Encourage focused addressing
            kernel_regularizer=SoftOrthonormalConstraintRegularizer(0.1, 0.0, 0.0),
            grid_regularizer=SoftOrthonormalConstraintRegularizer(0.1, 0.0, 0.0),
            name="neurogrid_bottleneck"
        )
        neurogrid_output = self.neurogrid_layer(encoder_output)

        # --- DECODER ---
        decoder_layers_seq = [
            layers.Dense(7 * 7 * 64, name='decoder_dense'),
            layers.BatchNormalization(name='decoder_bn0'),
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

        # Build standalone decoder for analysis
        decoder_input = layers.Input(shape=(self.latent_dim,), name="decoder_input")
        dec_x = decoder_input
        for layer in decoder_layers_seq:
            dec_x = layer(dec_x)
        self.decoder = keras.Model(inputs=decoder_input, outputs=dec_x, name="decoder")

        # Build full autoencoder using same decoder layers
        autoencoder_output = neurogrid_output
        for layer in decoder_layers_seq:
            autoencoder_output = layer(autoencoder_output)

        autoencoder = keras.Model(
            inputs=encoder_input,
            outputs=autoencoder_output,
            name="neurogrid_conv_autoencoder"
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
        Train the NeuroGrid convolutional autoencoder with monitoring callbacks.

        Returns:
            Training history containing loss curves and metrics.
        """
        if self.model is None or self.x_train is None:
            raise ValueError("Model and data must be prepared before training")

        logger.info(f"Starting training for {self.epochs} epochs...")

        # Define callbacks
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-6,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                filepath=str(self.experiment_dir / "models" / "best_model.keras"),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]

        # Train autoencoder (input = target for reconstruction)
        history = self.model.fit(
            self.x_train, self.x_train,
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
        Test complete serialization cycle for NeuroGrid layer compatibility.

        Raises:
            AssertionError: If serialization test fails.
        """
        if self.model is None or self.x_test is None:
            raise ValueError("Model and test data required for serialization test")

        logger.info("Testing serialization cycle...")

        # Get original predictions
        test_sample = self.x_test[:10]
        original_predictions = self.model.predict(test_sample, verbose=0)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'neurogrid_conv_autoencoder.keras')

            # Save model (NeuroGrid should be automatically registered)
            self.model.save(model_path)
            logger.info(f"Model saved to {model_path}")

            # Load model
            loaded_model = keras.models.load_model(model_path)
            loaded_predictions = loaded_model.predict(test_sample, verbose=0)
            logger.info("Model loaded successfully")

        # Verify identical predictions
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(original_predictions),
            keras.ops.convert_to_numpy(loaded_predictions),
            rtol=1e-6, atol=1e-6,
            err_msg="Serialization test failed: predictions don't match"
        )

        logger.info("Serialization test passed - NeuroGrid saves and loads correctly")

    def visualize_training_progress(self) -> None:
        """Visualize training progress and save to file."""
        if self.history is None:
            logger.warning("No training history available for visualization")
            return

        try:
            output_path = self.experiment_dir / "visualizations"

            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('NeuroGrid Convolutional Autoencoder Training Progress', fontsize=16, fontweight='bold')

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

            # Current temperature evolution
            current_temp = self.neurogrid_layer.get_current_temperature()
            axes[1, 0].axhline(y=current_temp, color='red', linewidth=2)
            axes[1, 0].set_title(f'NeuroGrid Temperature: {current_temp:.4f}')
            axes[1, 0].set_xlabel('Training Progress')
            axes[1, 0].set_ylabel('Temperature')
            axes[1, 0].grid(True, alpha=0.3)

            # Training summary
            final_train_loss = self.history.history['loss'][-1]
            final_val_loss = self.history.history['val_loss'][-1]
            best_val_loss = min(self.history.history['val_loss'])

            summary_text = f"""Training Summary:
                Final Training Loss: {final_train_loss:.6f}
                Final Validation Loss: {final_val_loss:.6f}
                Best Validation Loss: {best_val_loss:.6f}
                Total Epochs: {len(self.history.history['loss'])}
                
                NeuroGrid Configuration:
                Grid Shape: {self.grid_shape}
                Latent Dim: {self.latent_dim}
                Temperature: {current_temp:.4f}
                Conv Activation: {self.conv_activation}
                
                Architecture: Conv2D → NeuroGrid → Deconv
                Bottleneck: {self.grid_shape[0]}×{self.grid_shape[1]} grid
                Total Parameters: {self.model.count_params():,}
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
        """Visualize original vs reconstructed images."""
        if self.model is None or self.x_test is None:
            return

        try:
            output_path = self.experiment_dir / "visualizations"

            # Get random test samples
            indices = np.random.choice(len(self.x_test), n_samples, replace=False)
            test_samples = self.x_test[indices]
            test_labels = self.y_test[indices]

            # Get reconstructions
            reconstructions = self.model.predict(test_samples, verbose=0)

            # Create visualization
            fig, axes = plt.subplots(3, n_samples, figsize=(n_samples * 2, 6))
            fig.suptitle('Original vs Reconstructed MNIST Digits (NeuroGrid Conv)', fontsize=16, fontweight='bold')

            for i in range(n_samples):
                # Original images
                axes[0, i].imshow(test_samples[i].squeeze(), cmap='gray')
                axes[0, i].set_title(f'Original\n(Digit {test_labels[i]})')
                axes[0, i].axis('off')

                # Reconstructed images
                axes[1, i].imshow(reconstructions[i].squeeze(), cmap='gray')
                axes[1, i].set_title('Reconstructed')
                axes[1, i].axis('off')

                # Difference images
                diff = np.abs(test_samples[i] - reconstructions[i]).squeeze()
                axes[2, i].imshow(diff, cmap='hot')
                axes[2, i].set_title(f'Error\n(MAE: {np.mean(diff):.3f})')
                axes[2, i].axis('off')

            plt.tight_layout()
            plt.savefig(output_path / 'reconstructions.png', dpi=300, bbox_inches='tight')
            plt.close()

            logger.info("Reconstruction visualization saved")

        except Exception as e:
            logger.error(f"Failed to create reconstruction visualization: {e}")

    def visualize_neurogrid_prototypes(self) -> None:
        """Visualize NeuroGrid learned prototypes as decoded images."""
        if self.neurogrid_layer is None or self.decoder is None:
            return

        try:
            output_path = self.experiment_dir / "visualizations"

            # Get grid weights and decode them
            grid_weights = self.neurogrid_layer.get_grid_weights()
            grid_h, grid_w = self.grid_shape

            # Reshape to (total_positions, latent_dim) and decode
            prototypes_flat = keras.ops.reshape(grid_weights, (-1, self.latent_dim))
            decoded_prototypes = self.decoder.predict(prototypes_flat, verbose=0)

            # Reshape back to grid format
            prototype_images = decoded_prototypes.reshape(grid_h, grid_w, 28, 28, 1)

            # Create visualization
            fig, axes = plt.subplots(grid_h, grid_w, figsize=(grid_w * 2, grid_h * 2))
            fig.suptitle('NeuroGrid Learned Prototypes (Decoded as Images)',
                         fontsize=16, fontweight='bold')

            # Handle different grid shapes
            if grid_h == 1 and grid_w == 1:
                axes = np.array([[axes]])
            elif grid_h == 1:
                axes = axes.reshape(1, -1)
            elif grid_w == 1:
                axes = axes.reshape(-1, 1)

            for i in range(grid_h):
                for j in range(grid_w):
                    axes[i, j].imshow(prototype_images[i, j].squeeze(), cmap='gray')
                    axes[i, j].set_title(f'({i},{j})')
                    axes[i, j].axis('off')

            plt.tight_layout()
            plt.savefig(output_path / 'neurogrid_prototypes.png', dpi=300, bbox_inches='tight')
            plt.close()

            logger.info("NeuroGrid prototype visualization saved")

        except Exception as e:
            logger.error(f"Failed to create NeuroGrid prototype visualization: {e}")

    def visualize_addressing_patterns(self, n_samples: int = 5) -> None:
        """
        Visualize NeuroGrid addressing patterns for sample inputs with consistent colormap scaling.

        Shows how different inputs activate different regions of the NeuroGrid,
        revealing the probabilistic addressing behavior with consistent [0, 1] scaling.

        Args:
            n_samples: Number of samples to visualize addressing for. Defaults to 5.
        """
        if self.neurogrid_layer is None or self.encoder is None or self.x_test is None:
            logger.warning("Required components not available for addressing pattern visualization")
            return

        try:
            output_path = self.experiment_dir / "visualizations"
            logger.info("Creating NeuroGrid addressing pattern visualization...")

            # Get random test samples
            indices = np.random.choice(len(self.x_test), n_samples, replace=False)
            test_samples = self.x_test[indices]
            test_labels = self.y_test[indices]

            # Get encoded representations (input to NeuroGrid)
            encoded_samples = self.encoder.predict(test_samples, verbose=0)

            # Get addressing probabilities
            prob_info = self.neurogrid_layer.get_addressing_probabilities(encoded_samples)
            joint_prob = prob_info['joint']
            individual_probs = prob_info['individual']

            # Convert to numpy for visualization
            joint_prob_np = keras.ops.convert_to_numpy(joint_prob)

            # Create visualization
            fig, axes = plt.subplots(3, n_samples, figsize=(n_samples * 3, 9))
            fig.suptitle('Input Digits and Their NeuroGrid Addressing Patterns',
                         fontsize=16, fontweight='bold')

            for i in range(n_samples):
                # Original digit
                axes[0, i].imshow(test_samples[i].squeeze(), cmap='gray')
                axes[0, i].set_title(f'Digit {test_labels[i]}')
                axes[0, i].axis('off')

                # Joint addressing probability heatmap with fixed scale [0, 1]
                im1 = axes[1, i].imshow(joint_prob_np[i], cmap='viridis', interpolation='nearest',
                                       vmin=0, vmax=1)
                axes[1, i].set_title('Joint Probability')
                axes[1, i].set_xlabel('Grid X')
                axes[1, i].set_ylabel('Grid Y')
                plt.colorbar(im1, ax=axes[1, i], fraction=0.046, pad=0.04)

                # Individual dimension probabilities (show first dimension) with fixed scale [0, 1]
                if len(individual_probs) > 0:
                    dim0_prob = keras.ops.convert_to_numpy(individual_probs[0][i])
                    dim1_prob = keras.ops.convert_to_numpy(individual_probs[1][i]) if len(individual_probs) > 1 else dim0_prob

                    # Create 2D visualization from 1D probabilities
                    combined_prob = np.outer(dim0_prob, dim1_prob)
                    im2 = axes[2, i].imshow(combined_prob, cmap='plasma', interpolation='nearest',
                                           vmin=0, vmax=1)
                    axes[2, i].set_title('Dim Probabilities')
                    axes[2, i].set_xlabel('Dimension 1')
                    axes[2, i].set_ylabel('Dimension 0')
                    plt.colorbar(im2, ax=axes[2, i], fraction=0.046, pad=0.04)

            plt.tight_layout()
            plt.savefig(output_path / 'addressing_patterns.png', dpi=300, bbox_inches='tight')
            plt.close()

            logger.info("NeuroGrid addressing pattern visualization saved")

        except Exception as e:
            logger.error(f"Failed to create addressing pattern visualization: {e}")

    def analyze_digit_clustering(self) -> None:
        """
        Analyze how different digits cluster in the NeuroGrid addressing space.

        Computes average addressing patterns for each digit class and visualizes
        the topological organization of digit representations in the grid.
        """
        if self.neurogrid_layer is None or self.encoder is None or self.x_test is None:
            logger.warning("Required components not available for digit clustering analysis")
            return

        try:
            output_path = self.experiment_dir / "visualizations"
            logger.info("Analyzing digit clustering in NeuroGrid addressing space...")

            # Get sample of test data for analysis (limit for memory efficiency)
            sample_size = min(2000, len(self.x_test))
            sample_indices = np.random.choice(len(self.x_test), sample_size, replace=False)
            test_sample = self.x_test[sample_indices]
            test_labels = self.y_test[sample_indices]

            # Get encoded representations
            encoded_sample = self.encoder.predict(test_sample, verbose=0, batch_size=self.batch_size)

            # Get addressing probabilities
            prob_info = self.neurogrid_layer.get_addressing_probabilities(encoded_sample)
            joint_prob = prob_info['joint']
            joint_prob_np = keras.ops.convert_to_numpy(joint_prob)

            # Compute average addressing patterns for each digit
            digit_addressing = {}
            for digit in range(10):
                digit_mask = test_labels == digit
                if np.sum(digit_mask) > 0:
                    digit_avg = np.mean(joint_prob_np[digit_mask], axis=0)
                    digit_addressing[digit] = digit_avg
                else:
                    digit_addressing[digit] = np.zeros(self.grid_shape)

            # Create visualization with consistent colormap scaling
            fig, axes = plt.subplots(2, 5, figsize=(15, 6))
            fig.suptitle('Average NeuroGrid Addressing Patterns by Digit Class', fontsize=16, fontweight='bold')

            for digit in range(10):
                row = digit // 5
                col = digit % 5

                im = axes[row, col].imshow(digit_addressing[digit], cmap='viridis', interpolation='nearest',
                                          vmin=0, vmax=1)
                axes[row, col].set_title(f'Digit {digit}')
                axes[row, col].set_xlabel('Grid X')
                axes[row, col].set_ylabel('Grid Y')
                plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)

            plt.tight_layout()
            plt.savefig(output_path / 'digit_clustering.png', dpi=300, bbox_inches='tight')
            plt.close()

            # Analyze clustering quality using BMUs
            bmu_info = self.neurogrid_layer.find_best_matching_units(encoded_sample)
            bmu_coords = keras.ops.convert_to_numpy(bmu_info['bmu_coordinates'])

            # Compute clustering statistics
            position_purity = {}
            total_grid_positions = np.prod(self.grid_shape)

            for pos in range(total_grid_positions):
                pos_mask = bmu_coords == pos
                if np.sum(pos_mask) > 0:
                    pos_labels = test_labels[pos_mask]
                    unique_labels, counts = np.unique(pos_labels, return_counts=True)
                    purity = np.max(counts) / np.sum(counts)
                    dominant_class = unique_labels[np.argmax(counts)]
                    position_purity[pos] = (purity, dominant_class, np.sum(counts))

            # Save clustering statistics
            with open(self.experiment_dir / "logs" / "digit_clustering_stats.txt", 'w') as f:
                f.write("NeuroGrid Digit Clustering Analysis\n")
                f.write("=" * 50 + "\n")
                f.write(f"Grid Shape: {self.grid_shape}\n")
                f.write(f"Total Grid Positions: {total_grid_positions}\n")
                f.write(f"Positions with samples: {len(position_purity)}\n")
                f.write(f"Sample size: {sample_size}\n\n")

                if position_purity:
                    avg_purity = np.mean([purity for purity, _, _ in position_purity.values()])
                    f.write(f"Average position purity: {avg_purity:.4f}\n\n")

                    f.write("Position Details (position: purity, dominant_class, sample_count):\n")
                    for pos, (purity, dominant_class, count) in position_purity.items():
                        f.write(f"Position {pos}: {purity:.3f}, Class {dominant_class}, {count} samples\n")

                    logger.info(f"Average clustering purity: {avg_purity:.3f}")
                else:
                    f.write("No samples found for any grid position.\n")
                    logger.warning("No samples mapped to any grid positions")

            logger.info("Digit clustering analysis saved")

        except Exception as e:
            logger.error(f"Failed to create digit clustering analysis: {e}")

    def analyze_grid_utilization(self) -> None:
        """
        Analyze NeuroGrid utilization patterns and create heatmaps.

        Shows which parts of the grid are most/least utilized and provides
        insights into the learned representational structure.
        """
        if self.neurogrid_layer is None or self.encoder is None or self.x_test is None:
            logger.warning("Required components not available for grid utilization analysis")
            return

        try:
            output_path = self.experiment_dir / "visualizations"
            logger.info("Analyzing NeuroGrid utilization patterns...")

            # Get sample of test data
            sample_size = min(2000, len(self.x_test))
            sample_indices = np.random.choice(len(self.x_test), sample_size, replace=False)
            test_sample = self.x_test[sample_indices]

            # Get encoded representations
            encoded_sample = self.encoder.predict(test_sample, verbose=0, batch_size=self.batch_size)

            # Get grid utilization statistics
            utilization_stats = self.neurogrid_layer.get_grid_utilization(encoded_sample)

            # Convert to numpy arrays
            activation_counts = keras.ops.convert_to_numpy(utilization_stats['activation_counts'])
            total_activation = keras.ops.convert_to_numpy(utilization_stats['total_activation'])
            utilization_rate = keras.ops.convert_to_numpy(utilization_stats['utilization_rate'])

            # Reshape to grid format for visualization
            grid_h, grid_w = self.grid_shape
            activation_counts_grid = activation_counts.reshape(grid_h, grid_w)
            total_activation_grid = total_activation.reshape(grid_h, grid_w)
            utilization_rate_grid = utilization_rate.reshape(grid_h, grid_w)

            # Create visualization
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('NeuroGrid Utilization Analysis', fontsize=16, fontweight='bold')

            # Activation counts heatmap
            im1 = axes[0, 0].imshow(activation_counts_grid, cmap='YlOrRd', interpolation='nearest')
            axes[0, 0].set_title('Activation Counts (BMU Hits)')
            axes[0, 0].set_xlabel('Grid X')
            axes[0, 0].set_ylabel('Grid Y')
            plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)

            # Total activation heatmap
            im2 = axes[0, 1].imshow(total_activation_grid, cmap='Blues', interpolation='nearest')
            axes[0, 1].set_title('Total Activation (Sum of Probabilities)')
            axes[0, 1].set_xlabel('Grid X')
            axes[0, 1].set_ylabel('Grid Y')
            plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)

            # Utilization rate heatmap
            im3 = axes[1, 0].imshow(utilization_rate_grid, cmap='Greens', interpolation='nearest')
            axes[1, 0].set_title('Utilization Rate (Normalized)')
            axes[1, 0].set_xlabel('Grid X')
            axes[1, 0].set_ylabel('Grid Y')
            plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

            # Utilization statistics
            total_positions = np.prod(self.grid_shape)
            used_positions = np.sum(activation_counts > 0)
            max_activations = np.max(activation_counts)
            mean_utilization = np.mean(utilization_rate)

            stats_text = f"""Utilization Statistics:
                Sample Size: {sample_size:,}
                Grid Positions: {total_positions}
                Used Positions: {used_positions} ({used_positions/total_positions*100:.1f}%)
                Unused Positions: {total_positions - used_positions}
                
                Max Activations: {max_activations:.0f}
                Mean Utilization Rate: {mean_utilization:.4f}
                
                Most Active Position: {np.unravel_index(np.argmax(activation_counts_grid), self.grid_shape)}
                Least Active Position: {np.unravel_index(np.argmin(activation_counts_grid), self.grid_shape)}
                """

            axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes,
                            verticalalignment='top', fontfamily='monospace', fontsize=10)
            axes[1, 1].set_xlim(0, 1)
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].axis('off')

            plt.tight_layout()
            plt.savefig(output_path / 'grid_utilization.png', dpi=300, bbox_inches='tight')
            plt.close()

            # Save detailed utilization statistics
            with open(self.experiment_dir / "logs" / "grid_utilization_stats.txt", 'w') as f:
                f.write("NeuroGrid Utilization Analysis\n")
                f.write("=" * 50 + "\n")
                f.write(f"Grid Shape: {self.grid_shape}\n")
                f.write(f"Sample Size: {sample_size}\n")
                f.write(f"Total Grid Positions: {total_positions}\n")
                f.write(f"Used Positions: {used_positions} ({used_positions/total_positions*100:.1f}%)\n")
                f.write(f"Mean Utilization Rate: {mean_utilization:.6f}\n")
                f.write(f"Max Activations: {max_activations}\n\n")

                f.write("Per-position activation counts:\n")
                for i in range(grid_h):
                    for j in range(grid_w):
                        pos_idx = i * grid_w + j
                        f.write(f"({i},{j}): {activation_counts[pos_idx]:.0f} activations\n")

            logger.info("Grid utilization analysis saved")

        except Exception as e:
            logger.error(f"Failed to create grid utilization analysis: {e}")

    def analyze_addressing_quality(self) -> None:
        """Analyze NeuroGrid addressing patterns and quality measures."""
        if self.neurogrid_layer is None or self.encoder is None or self.x_test is None:
            return

        try:
            output_path = self.experiment_dir / "visualizations"

            logger.info("Analyzing NeuroGrid addressing quality...")

            # Get sample of test data for analysis
            sample_size = min(1000, len(self.x_test))
            sample_indices = np.random.choice(len(self.x_test), sample_size, replace=False)
            test_sample = self.x_test[sample_indices]
            test_labels = self.y_test[sample_indices]

            # Get encoded representations (input to NeuroGrid)
            encoded_sample = self.encoder.predict(test_sample, verbose=0)

            # Compute quality measures
            quality_measures = self.neurogrid_layer.compute_input_quality(encoded_sample)
            quality_stats = self.neurogrid_layer.get_quality_statistics(encoded_sample)

            # Create quality analysis visualization
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('NeuroGrid Addressing Quality Analysis', fontsize=16, fontweight='bold')

            # Quality measures histograms
            quality_names = ['overall_quality', 'addressing_confidence', 'addressing_entropy']
            for i, measure in enumerate(quality_names):
                values = keras.ops.convert_to_numpy(quality_measures[measure])
                axes[0, i].hist(values, bins=30, alpha=0.7, edgecolor='black')
                axes[0, i].set_title(f'{measure.replace("_", " ").title()}')
                axes[0, i].set_xlabel('Value')
                axes[0, i].set_ylabel('Frequency')
                axes[0, i].grid(True, alpha=0.3)
                axes[0, i].set_xlim(0, 1)

                # Add mean line
                mean_val = np.mean(values)
                axes[0, i].axvline(mean_val, color='red', linestyle='--',
                                   label=f'Mean: {mean_val:.3f}')
                axes[0, i].legend()

            # Quality by digit class
            overall_quality = keras.ops.convert_to_numpy(quality_measures['overall_quality'])
            confidence = keras.ops.convert_to_numpy(quality_measures['addressing_confidence'])

            digit_quality_means = []
            digit_confidence_means = []
            for digit in range(10):
                digit_mask = test_labels == digit
                if np.sum(digit_mask) > 0:
                    digit_quality_means.append(np.mean(overall_quality[digit_mask]))
                    digit_confidence_means.append(np.mean(confidence[digit_mask]))
                else:
                    digit_quality_means.append(0)
                    digit_confidence_means.append(0)

            axes[1, 0].bar(range(10), digit_quality_means)
            axes[1, 0].set_title('Average Quality by Digit Class')
            axes[1, 0].set_xlabel('Digit')
            axes[1, 0].set_ylabel('Quality Score')
            axes[1, 0].set_xticks(range(10))
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_ylim(0, 1)

            axes[1, 1].bar(range(10), digit_confidence_means)
            axes[1, 1].set_title('Average Confidence by Digit Class')
            axes[1, 1].set_xlabel('Digit')
            axes[1, 1].set_ylabel('Confidence Score')
            axes[1, 1].set_xticks(range(10))
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_ylim(0, 1)

            # Quality statistics summary
            summary_text = "Quality Statistics Summary:\n" + "\n".join([
                f"{key}: {value:.4f}" for key, value in quality_stats.items()
                if 'overall_quality' in key or 'addressing_confidence' in key
            ])

            axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
                            verticalalignment='top', fontfamily='monospace', fontsize=10)
            axes[1, 2].set_xlim(0, 1)
            axes[1, 2].set_ylim(0, 1)
            axes[1, 2].axis('off')

            plt.tight_layout()
            plt.savefig(output_path / 'addressing_quality.png', dpi=300, bbox_inches='tight')
            plt.close()

            # Save detailed quality statistics
            with open(self.experiment_dir / "logs" / "quality_stats.txt", 'w') as f:
                f.write("NeuroGrid Addressing Quality Analysis\n")
                f.write("=" * 50 + "\n")
                f.write(f"Sample Size: {sample_size}\n")
                f.write(f"Grid Shape: {self.grid_shape}\n")
                f.write(f"Temperature: {self.neurogrid_layer.get_current_temperature():.4f}\n\n")
                f.write("Detailed Statistics:\n")
                for key, value in quality_stats.items():
                    f.write(f"{key}: {value:.6f}\n")

            logger.info("NeuroGrid addressing quality analysis saved")

        except Exception as e:
            logger.error(f"Failed to create addressing quality analysis: {e}")

    def demonstrate_quality_filtering(self) -> None:
        """
        Demonstrate NeuroGrid's quality-based filtering capabilities.

        Shows how to use quality measures to filter inputs and process
        high-quality vs low-quality samples differently.
        """
        if self.neurogrid_layer is None or self.encoder is None or self.x_test is None:
            logger.warning("Required components not available for quality filtering demonstration")
            return

        try:
            output_path = self.experiment_dir / "visualizations"
            logger.info("Demonstrating NeuroGrid quality-based filtering...")

            # Get sample of test data
            sample_size = min(500, len(self.x_test))
            sample_indices = np.random.choice(len(self.x_test), sample_size, replace=False)
            test_sample = self.x_test[sample_indices]
            test_labels = self.y_test[sample_indices]

            # Get encoded representations
            encoded_sample = self.encoder.predict(test_sample, verbose=0)

            # Apply quality filtering with different thresholds
            thresholds = [0.3, 0.5, 0.7]
            filtering_results = {}

            for threshold in thresholds:
                filtered = self.neurogrid_layer.filter_by_quality_threshold(
                    encoded_sample,
                    quality_threshold=threshold,
                    quality_measure='overall_quality'
                )
                # Convert tensor shapes to Python integers
                high_quality_shape = keras.ops.convert_to_numpy(keras.ops.shape(filtered['high_quality_inputs']))
                low_quality_shape = keras.ops.convert_to_numpy(keras.ops.shape(filtered['low_quality_inputs']))

                filtering_results[threshold] = {
                    'high_quality_count': int(high_quality_shape[0]),
                    'low_quality_count': int(low_quality_shape[0]),
                    'quality_scores': keras.ops.convert_to_numpy(filtered['quality_scores'])
                }

            # Create visualization
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('NeuroGrid Quality-Based Filtering Demonstration', fontsize=16, fontweight='bold')

            # Quality score distribution with thresholds
            all_quality_scores = filtering_results[0.5]['quality_scores']
            axes[0, 0].hist(all_quality_scores, bins=30, alpha=0.7, edgecolor='black', label='All Samples')

            colors = ['red', 'orange', 'green']
            for i, threshold in enumerate(thresholds):
                axes[0, 0].axvline(threshold, color=colors[i], linestyle='--', linewidth=2,
                                   label=f'Threshold {threshold}')

            axes[0, 0].set_title('Quality Score Distribution')
            axes[0, 0].set_xlabel('Overall Quality Score')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # Filtering results bar chart
            high_quality_counts = [filtering_results[t]['high_quality_count'] for t in thresholds]
            low_quality_counts = [filtering_results[t]['low_quality_count'] for t in thresholds]

            x_pos = np.arange(len(thresholds))
            width = 0.35

            axes[0, 1].bar(x_pos - width/2, high_quality_counts, width, label='High Quality', color='lightgreen')
            axes[0, 1].bar(x_pos + width/2, low_quality_counts, width, label='Low Quality', color='lightcoral')
            axes[0, 1].set_title('Filtering Results by Threshold')
            axes[0, 1].set_xlabel('Quality Threshold')
            axes[0, 1].set_ylabel('Sample Count')
            axes[0, 1].set_xticks(x_pos)
            axes[0, 1].set_xticklabels([str(t) for t in thresholds])
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # Quality by digit class (showing potential biases)
            digit_qualities = []
            for digit in range(10):
                digit_mask = test_labels == digit
                if np.sum(digit_mask) > 0:
                    digit_quality = np.mean(all_quality_scores[digit_mask])
                    digit_qualities.append(digit_quality)
                else:
                    digit_qualities.append(0)

            axes[1, 0].bar(range(10), digit_qualities, color='skyblue')
            axes[1, 0].set_title('Average Quality by Digit Class')
            axes[1, 0].set_xlabel('Digit')
            axes[1, 0].set_ylabel('Average Quality Score')
            axes[1, 0].set_xticks(range(10))
            axes[1, 0].grid(True, alpha=0.3)

            # Filtering summary statistics
            mean_quality = np.mean(all_quality_scores)
            std_quality = np.std(all_quality_scores)

            summary_text = f"""Quality Filtering Summary:
                Total Samples: {sample_size:,}
                Mean Quality: {mean_quality:.3f} ± {std_quality:.3f}
                
                Threshold 0.3: {high_quality_counts[0]:,} high ({high_quality_counts[0]/sample_size*100:.1f}%)
                Threshold 0.5: {high_quality_counts[1]:,} high ({high_quality_counts[1]/sample_size*100:.1f}%)
                Threshold 0.7: {high_quality_counts[2]:,} high ({high_quality_counts[2]/sample_size*100:.1f}%)
                
                Applications:
                • Data preprocessing pipelines
                • Confidence-aware inference
                • Anomaly detection
                • Active learning sample selection
                """

            axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                            verticalalignment='top', fontfamily='monospace', fontsize=10)
            axes[1, 1].set_xlim(0, 1)
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].axis('off')

            plt.tight_layout()
            plt.savefig(output_path / 'quality_filtering.png', dpi=300, bbox_inches='tight')
            plt.close()

            # Save filtering demonstration results
            with open(self.experiment_dir / "logs" / "quality_filtering_demo.txt", 'w') as f:
                f.write("NeuroGrid Quality-Based Filtering Demonstration\n")
                f.write("=" * 50 + "\n")
                f.write(f"Sample Size: {sample_size}\n")
                f.write(f"Mean Quality: {mean_quality:.6f} ± {std_quality:.6f}\n\n")

                f.write("Filtering Results:\n")
                for threshold in thresholds:
                    high_count = filtering_results[threshold]['high_quality_count']
                    low_count = filtering_results[threshold]['low_quality_count']
                    f.write(f"Threshold {threshold}: {high_count} high, {low_count} low "
                           f"({high_count/sample_size*100:.1f}% retained)\n")

                f.write("\nDigit-specific quality scores:\n")
                for digit in range(10):
                    f.write(f"Digit {digit}: {digit_qualities[digit]:.4f}\n")

            logger.info("Quality filtering demonstration saved")

        except Exception as e:
            logger.error(f"Failed to create quality filtering demonstration: {e}")

    def save_experiment_summary(self) -> None:
        """Save comprehensive experiment summary."""
        try:
            # Save configuration
            config_dict = {
                'grid_shape': self.grid_shape,
                'latent_dim': self.latent_dim,
                'temperature': self.temperature,
                'conv_activation': self.conv_activation,
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'learning_rate': self.learning_rate,
                'final_temperature': self.neurogrid_layer.get_current_temperature() if self.neurogrid_layer else None
            }

            import json
            with open(self.experiment_dir / "experiment_config.json", 'w') as f:
                json.dump(config_dict, f, indent=2)

            # Save training history
            if self.history is not None:
                history_dict = {key: [float(v) for v in values]
                                for key, values in self.history.history.items()}
                with open(self.experiment_dir / "training_history.json", 'w') as f:
                    json.dump(history_dict, f, indent=2)

            # Save model summary
            if self.model is not None:
                with open(self.experiment_dir / "logs" / "model_summary.txt", 'w') as f:
                    self.model.summary(print_fn=lambda x: f.write(x + '\n'))

            logger.info("Experiment summary saved")

        except Exception as e:
            logger.error(f"Failed to save experiment summary: {e}")

    def run_complete_experiment(self) -> None:
        """Execute the complete experimental pipeline."""
        logger.info("=" * 60)
        logger.info("STARTING ENHANCED NEUROGRID CONV AUTOENCODER EXPERIMENT")
        logger.info("=" * 60)

        try:
            # Phase 1: Data preparation
            self.load_and_preprocess_data()

            # Phase 2: Model creation
            self.model = self.create_neurogrid_conv_autoencoder()
            self.model.summary()
            logger.info(f"Model created with {self.model.count_params():,} parameters")

            # Phase 3: Training
            self.history = self.train_model()

            # Phase 4: Save model
            self.model.save(self.experiment_dir / "models" / "final_model.keras")

            # Phase 5: Serialization test
            self.test_serialization()

            # Phase 6: COMPREHENSIVE visualizations and analysis
            logger.info("Generating comprehensive visualizations and analysis...")
            self.visualize_training_progress()
            self.visualize_reconstructions(n_samples=8)
            self.visualize_neurogrid_prototypes()

            # Enhanced visualizations with consistent scaling
            self.visualize_addressing_patterns(n_samples=5)
            self.analyze_digit_clustering()
            self.analyze_grid_utilization()
            self.analyze_addressing_quality()
            self.demonstrate_quality_filtering()

            # Phase 7: Save summary
            self.save_experiment_summary()

            logger.info("=" * 60)
            logger.info("ENHANCED CONV EXPERIMENT COMPLETED SUCCESSFULLY")
            logger.info(f"All results saved to: {self.experiment_dir}")
            logger.info("Generated visualizations:")
            logger.info("  - training_progress.png")
            logger.info("  - reconstructions.png")
            logger.info("  - neurogrid_prototypes.png")
            logger.info("  - addressing_patterns.png (with consistent [0,1] colormap)")
            logger.info("  - digit_clustering.png")
            logger.info("  - grid_utilization.png")
            logger.info("  - addressing_quality.png")
            logger.info("  - quality_filtering.png")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"Experiment failed: {e}", exc_info=True)
            raise


def main() -> None:
    """Run the enhanced NeuroGrid convolutional autoencoder experiment."""
    experiment = NeuroGridAutoencoderExperiment(
        grid_shape=(4, 4),
        latent_dim=32,
        temperature=0.8,
        conv_activation='silu',
        batch_size=128,
        epochs=200,
        learning_rate=0.001,
        output_dir='results'
    )

    experiment.run_complete_experiment()


if __name__ == "__main__":
    main()