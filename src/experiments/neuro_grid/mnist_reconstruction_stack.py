"""
Multi-NeuroGrid Stacked Convolutional Autoencoder Experiment on MNIST

A comprehensive convolutional autoencoder experiment demonstrating multiple stacked NeuroGrid layers
with configurable residual connections for learning hierarchical organized representations.
"""

import os
import tempfile
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional, List, Dict, Any

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


class MultiNeuroGridAutoencoderExperiment:
    """
    This experiment showcases multiple stacked NeuroGrid layers with optional residual connections
    between encoder and decoder networks, learning hierarchical spatially organized representations
    while maintaining differentiability for end-to-end training.

    **Intent**: Demonstrate how multiple NeuroGrid layers can work together to learn increasingly
    abstract and organized representations, with each layer specializing in different aspects
    of the input distribution.

    **Architecture**:
    ```
    Input(28,28,1) → Conv2D → Conv2D → Flatten → Dense(latent_dim) →
    NeuroGrid₁(grid₁, latent_dim) → [+residual] →
    NeuroGrid₂(grid₂, latent_dim) → [+residual] →
    ...
    NeuroGridₙ(gridₙ, latent_dim) →
    Dense → Reshape → UpSample+Conv2D → UpSample+Conv2D → Conv2D(1) → Output(28,28,1)
    ```

    Args:
        num_neurogrid_layers: Number of NeuroGrid layers to stack. Defaults to 2.
        grid_shapes: List of grid shapes for each NeuroGrid layer. If single tuple, replicates for all layers.
        latent_dim: Dimensionality of NeuroGrid latent vectors. Defaults to 64.
        use_residual_connections: Whether to add residual connections between NeuroGrid layers. Defaults to True.
        temperature: Initial temperature for NeuroGrid addressing. Defaults to 1.0.
        conv_activation: Activation function for convolutional layers. Defaults to 'gelu'.
        batch_size: Training batch size. Defaults to 128.
        epochs: Number of training epochs. Defaults to 50.
        learning_rate: Learning rate for Adam optimizer. Defaults to 0.001.
        output_dir: Base directory for saving results. Defaults to 'multi_neurogrid_results'.
    """

    def __init__(
            self,
            num_neurogrid_layers: int = 2,
            grid_shapes: List[Tuple[int, int]] = [(8, 8), (6, 6)],
            latent_dim: int = 64,
            use_residual_connections: bool = True,
            temperature: float = 1.0,
            conv_activation: str = 'relu',
            batch_size: int = 128,
            epochs: int = 50,
            learning_rate: float = 0.001,
            output_dir: str = 'results'
    ) -> None:
        """Initialize the experiment with configuration parameters."""

        # Validate inputs
        if num_neurogrid_layers <= 0:
            raise ValueError(f"num_neurogrid_layers must be positive, got {num_neurogrid_layers}")

        self.num_neurogrid_layers = num_neurogrid_layers

        # Handle grid_shapes configuration
        if isinstance(grid_shapes, tuple):
            # Single tuple provided, replicate for all layers
            self.grid_shapes = [grid_shapes] * num_neurogrid_layers
        elif len(grid_shapes) == 1:
            # Single grid shape in list, replicate for all layers
            self.grid_shapes = grid_shapes * num_neurogrid_layers
        elif len(grid_shapes) == num_neurogrid_layers:
            # Perfect match
            self.grid_shapes = list(grid_shapes)
        else:
            # Mismatch - extend or truncate
            logger.warning(f"Grid shapes count ({len(grid_shapes)}) doesn't match layer count ({num_neurogrid_layers})")
            if len(grid_shapes) < num_neurogrid_layers:
                # Extend by replicating last grid shape
                self.grid_shapes = list(grid_shapes) + [grid_shapes[-1]] * (num_neurogrid_layers - len(grid_shapes))
            else:
                # Truncate
                self.grid_shapes = list(grid_shapes[:num_neurogrid_layers])

        self.latent_dim = latent_dim
        self.use_residual_connections = use_residual_connections
        self.temperature = temperature
        self.conv_activation = conv_activation
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

        # Create timestamped experiment directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_dir = Path(output_dir) / f"multi_neurogrid_autoencoder_{timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.experiment_dir / "visualizations").mkdir(exist_ok=True)
        (self.experiment_dir / "models").mkdir(exist_ok=True)
        (self.experiment_dir / "logs").mkdir(exist_ok=True)

        # Create layer-specific directories
        for i in range(self.num_neurogrid_layers):
            (self.experiment_dir / "visualizations" / f"layer_{i}").mkdir(exist_ok=True)

        # Initialize model components
        self.model: Optional[keras.Model] = None
        self.encoder: Optional[keras.Model] = None
        self.decoder: Optional[keras.Model] = None
        self.neurogrid_layers: List[NeuroGrid] = []
        self.history: Optional[keras.callbacks.History] = None

        # Data storage
        self.x_train: Optional[np.ndarray] = None
        self.x_test: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None

        logger.info(f"Initialized Multi-NeuroGrid autoencoder with {num_neurogrid_layers} layers")
        logger.info(f"Grid shapes: {self.grid_shapes}")
        logger.info(f"Residual connections: {use_residual_connections}")
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

    def create_multi_neurogrid_conv_autoencoder(self) -> keras.Model:
        """
        Create convolutional autoencoder model with multiple stacked NeuroGrid layers.

        Returns:
            Compiled autoencoder model with multiple NeuroGrid bottlenecks.
        """
        logger.info("Creating Multi-NeuroGrid convolutional autoencoder architecture...")

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

        # --- STACKED NEUROGRID LAYERS ---
        self.neurogrid_layers = []
        neurogrid_output = encoder_output

        for i in range(self.num_neurogrid_layers):
            # Create NeuroGrid layer
            neurogrid_layer = NeuroGrid(
                grid_shape=list(self.grid_shapes[i]),
                latent_dim=self.latent_dim,
                temperature=self.temperature,
                learnable_temperature=False,
                entropy_regularizer_strength=0.01,  # Encourage focused addressing
                kernel_regularizer=None,
                grid_regularizer=None,
                name=f"neurogrid_layer_{i}"
            )
            self.neurogrid_layers.append(neurogrid_layer)

            # Apply NeuroGrid transformation
            neurogrid_transformed = neurogrid_layer(neurogrid_output)

            # Add residual connection if enabled and not the first layer
            if self.use_residual_connections and i > 0:
                neurogrid_output = layers.Add(name=f"residual_connection_{i}")(
                    [neurogrid_output, neurogrid_transformed])
            else:
                neurogrid_output = neurogrid_transformed

            neurogrid_output = layers.BatchNormalization()(neurogrid_output)

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
            name="multi_neurogrid_conv_autoencoder"
        )

        # --- COMPILE ---
        autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )

        return autoencoder

    def create_intermediate_models(self) -> Dict[str, keras.Model]:
        """
        Create models that output intermediate representations from each NeuroGrid layer.

        Returns:
            Dictionary of models for accessing intermediate representations.
        """
        if self.encoder is None or len(self.neurogrid_layers) == 0:
            raise ValueError("Encoder and NeuroGrid layers must be created first")

        intermediate_models = {}

        # Create model for encoder output
        intermediate_models['encoder'] = self.encoder

        # Create models for each NeuroGrid layer output
        encoder_input = self.encoder.input
        current_output = self.encoder.output

        for i, neurogrid_layer in enumerate(self.neurogrid_layers):
            # Apply current layer transformation
            neurogrid_transformed = neurogrid_layer(current_output)

            # Add residual connection if applicable
            if self.use_residual_connections and i > 0:
                current_output = layers.Add(name=f"intermediate_residual_{i}")([current_output, neurogrid_transformed])
            else:
                current_output = neurogrid_transformed

            # Create model up to this layer
            model_name = f"up_to_layer_{i}"
            intermediate_models[model_name] = keras.Model(
                inputs=encoder_input,
                outputs=current_output,
                name=model_name
            )

        return intermediate_models

    def train_model(self) -> keras.callbacks.History:
        """
        Train the Multi-NeuroGrid convolutional autoencoder with monitoring callbacks.

        Returns:
            Training history containing loss curves and metrics.
        """
        if self.model is None or self.x_train is None:
            raise ValueError("Model and data must be prepared before training")

        logger.info(f"Starting training for {self.epochs} epochs with {self.num_neurogrid_layers} NeuroGrid layers...")

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
        Test complete serialization cycle for multi-NeuroGrid layer compatibility.

        Raises:
            AssertionError: If serialization test fails.
        """
        if self.model is None or self.x_test is None:
            raise ValueError("Model and test data required for serialization test")

        logger.info("Testing serialization cycle for multi-NeuroGrid model...")

        # Get original predictions
        test_sample = self.x_test[:10]
        original_predictions = self.model.predict(test_sample, verbose=0)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'multi_neurogrid_conv_autoencoder.keras')

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

        logger.info("Serialization test passed - Multi-NeuroGrid saves and loads correctly")

    def visualize_training_progress(self) -> None:
        """Visualize training progress and save to file."""
        if self.history is None:
            logger.warning("No training history available for visualization")
            return

        try:
            output_path = self.experiment_dir / "visualizations"

            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Multi-NeuroGrid ({self.num_neurogrid_layers} layers) Training Progress', fontsize=16,
                         fontweight='bold')

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

            # Temperature evolution for all layers
            axes[1, 0].set_title('NeuroGrid Temperatures by Layer')
            for i, neurogrid_layer in enumerate(self.neurogrid_layers):
                current_temp = neurogrid_layer.get_current_temperature()
                axes[1, 0].axhline(y=current_temp, label=f'Layer {i}: {current_temp:.4f}', linewidth=2)
            axes[1, 0].set_xlabel('Training Progress')
            axes[1, 0].set_ylabel('Temperature')
            axes[1, 0].legend()
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

Multi-NeuroGrid Configuration:
Number of Layers: {self.num_neurogrid_layers}
Grid Shapes: {self.grid_shapes}
Latent Dim: {self.latent_dim}
Residual Connections: {self.use_residual_connections}
Conv Activation: {self.conv_activation}

Architecture: Conv2D → Multi-NeuroGrid → Deconv
Total Parameters: {self.model.count_params():,}
"""

            axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                            verticalalignment='top', fontfamily='monospace', fontsize=9)
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
            fig.suptitle(f'Original vs Reconstructed MNIST Digits (Multi-NeuroGrid x{self.num_neurogrid_layers})',
                         fontsize=16, fontweight='bold')

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

    def visualize_all_neurogrid_prototypes(self) -> None:
        """Visualize learned prototypes for all NeuroGrid layers."""
        if len(self.neurogrid_layers) == 0 or self.decoder is None:
            return

        try:
            output_path = self.experiment_dir / "visualizations"

            # Create comprehensive view for all layers
            total_cols = sum(shape[1] for shape in self.grid_shapes)
            max_rows = max(shape[0] for shape in self.grid_shapes)

            fig, axes = plt.subplots(max_rows, total_cols, figsize=(total_cols * 2, max_rows * 2))
            fig.suptitle('All NeuroGrid Learned Prototypes (Decoded as Images)', fontsize=16, fontweight='bold')

            # Handle single row case
            if max_rows == 1:
                axes = axes.reshape(1, -1)

            col_offset = 0
            for layer_idx, (neurogrid_layer, grid_shape) in enumerate(zip(self.neurogrid_layers, self.grid_shapes)):
                # Get grid weights and decode them
                grid_weights = neurogrid_layer.get_grid_weights()
                grid_h, grid_w = grid_shape

                # Reshape to (total_positions, latent_dim) and decode
                prototypes_flat = keras.ops.reshape(grid_weights, (-1, self.latent_dim))
                decoded_prototypes = self.decoder.predict(prototypes_flat, verbose=0)

                # Reshape back to grid format
                prototype_images = decoded_prototypes.reshape(grid_h, grid_w, 28, 28, 1)

                # Plot this layer's prototypes
                for i in range(grid_h):
                    for j in range(grid_w):
                        col_idx = col_offset + j
                        if i < max_rows and col_idx < total_cols:
                            axes[i, col_idx].imshow(prototype_images[i, j].squeeze(), cmap='gray')
                            axes[i, col_idx].set_title(f'L{layer_idx}({i},{j})')
                            axes[i, col_idx].axis('off')

                # Fill empty spaces for shorter grids
                for i in range(grid_h, max_rows):
                    for j in range(grid_w):
                        col_idx = col_offset + j
                        if col_idx < total_cols:
                            axes[i, col_idx].axis('off')

                col_offset += grid_w

            plt.tight_layout()
            plt.savefig(output_path / 'all_neurogrid_prototypes.png', dpi=300, bbox_inches='tight')
            plt.close()

            # Create individual layer visualizations
            for layer_idx, (neurogrid_layer, grid_shape) in enumerate(zip(self.neurogrid_layers, self.grid_shapes)):
                self.visualize_single_layer_prototypes(layer_idx, neurogrid_layer, grid_shape)

            logger.info("NeuroGrid prototype visualizations saved")

        except Exception as e:
            logger.error(f"Failed to create NeuroGrid prototype visualization: {e}")

    def visualize_single_layer_prototypes(self, layer_idx: int, neurogrid_layer: NeuroGrid,
                                          grid_shape: Tuple[int, int]) -> None:
        """Visualize prototypes for a single NeuroGrid layer."""
        try:
            output_path = self.experiment_dir / "visualizations" / f"layer_{layer_idx}"

            # Get grid weights and decode them
            grid_weights = neurogrid_layer.get_grid_weights()
            grid_h, grid_w = grid_shape

            # Reshape to (total_positions, latent_dim) and decode
            prototypes_flat = keras.ops.reshape(grid_weights, (-1, self.latent_dim))
            decoded_prototypes = self.decoder.predict(prototypes_flat, verbose=0)

            # Reshape back to grid format
            prototype_images = decoded_prototypes.reshape(grid_h, grid_w, 28, 28, 1)

            # Create visualization
            fig, axes = plt.subplots(grid_h, grid_w, figsize=(grid_w * 2, grid_h * 2))
            fig.suptitle(f'NeuroGrid Layer {layer_idx} Prototypes ({grid_h}×{grid_w})',
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
            plt.savefig(output_path / f'layer_{layer_idx}_prototypes.png', dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            logger.error(f"Failed to create layer {layer_idx} prototype visualization: {e}")

    def visualize_addressing_patterns_all_layers(self, n_samples: int = 5) -> None:
        """
        Visualize NeuroGrid addressing patterns for all layers with sample inputs.
        """
        if len(self.neurogrid_layers) == 0 or self.encoder is None or self.x_test is None:
            logger.warning("Required components not available for addressing pattern visualization")
            return

        try:
            output_path = self.experiment_dir / "visualizations"
            logger.info("Creating addressing pattern visualization for all layers...")

            # Get random test samples
            indices = np.random.choice(len(self.x_test), n_samples, replace=False)
            test_samples = self.x_test[indices]
            test_labels = self.y_test[indices]

            # Create intermediate models to get outputs at each layer
            intermediate_models = self.create_intermediate_models()

            # Create comprehensive visualization
            num_rows = 1 + self.num_neurogrid_layers  # 1 for input images + 1 per layer
            fig, axes = plt.subplots(num_rows, n_samples, figsize=(n_samples * 3, num_rows * 3))
            fig.suptitle('Input Digits and Multi-Layer NeuroGrid Addressing Patterns',
                         fontsize=16, fontweight='bold')

            # Handle single column case
            if n_samples == 1:
                axes = axes.reshape(-1, 1)

            for i in range(n_samples):
                # Show original digit in first row
                axes[0, i].imshow(test_samples[i].squeeze(), cmap='gray')
                axes[0, i].set_title(f'Digit {test_labels[i]}')
                axes[0, i].axis('off')

                # Get representations for each layer
                current_input = test_samples[i:i + 1]  # Keep batch dimension
                encoded_input = self.encoder.predict(current_input, verbose=0)

                layer_input = encoded_input
                for layer_idx, neurogrid_layer in enumerate(self.neurogrid_layers):
                    # Get addressing probabilities for current layer
                    prob_info = neurogrid_layer.get_addressing_probabilities(layer_input)
                    joint_prob = prob_info['joint']
                    joint_prob_np = keras.ops.convert_to_numpy(joint_prob)

                    # Plot addressing pattern
                    row_idx = layer_idx + 1
                    im = axes[row_idx, i].imshow(joint_prob_np[0], cmap='viridis',
                                                 interpolation='nearest', vmin=0, vmax=1)
                    axes[row_idx, i].set_title(f'Layer {layer_idx}\n{self.grid_shapes[layer_idx]}')
                    axes[row_idx, i].set_xlabel('Grid X')
                    axes[row_idx, i].set_ylabel('Grid Y')

                    # Apply transformation for next layer
                    layer_transformed = neurogrid_layer(layer_input)
                    if self.use_residual_connections and layer_idx > 0:
                        layer_input = layers.Add()([layer_input, layer_transformed])
                    else:
                        layer_input = layer_transformed

            # Add colorbars
            for row_idx in range(1, num_rows):
                im = axes[row_idx, -1].get_images()[0] if axes[row_idx, -1].get_images() else None
                if im is not None:
                    plt.colorbar(im, ax=axes[row_idx, :].tolist(), fraction=0.046, pad=0.04)

            plt.tight_layout()
            plt.savefig(output_path / 'addressing_patterns_all_layers.png', dpi=300, bbox_inches='tight')
            plt.close()

            # Create individual layer visualizations
            for layer_idx in range(self.num_neurogrid_layers):
                self.visualize_single_layer_addressing_patterns(layer_idx, n_samples)

            logger.info("Addressing pattern visualizations saved")

        except Exception as e:
            logger.error(f"Failed to create addressing pattern visualization: {e}")

    def visualize_single_layer_addressing_patterns(self, layer_idx: int, n_samples: int = 5) -> None:
        """Visualize addressing patterns for a single NeuroGrid layer."""
        try:
            output_path = self.experiment_dir / "visualizations" / f"layer_{layer_idx}"

            # Get random test samples
            indices = np.random.choice(len(self.x_test), n_samples, replace=False)
            test_samples = self.x_test[indices]
            test_labels = self.y_test[indices]

            # Get input to the specific layer
            layer_input = self.encoder.predict(test_samples, verbose=0)

            # Apply transformations up to the target layer
            for i in range(layer_idx):
                neurogrid_layer = self.neurogrid_layers[i]
                layer_transformed = neurogrid_layer.predict(layer_input, verbose=0)
                if self.use_residual_connections and i > 0:
                    layer_input = layer_input + layer_transformed
                else:
                    layer_input = layer_transformed

            # Get addressing probabilities for target layer
            prob_info = self.neurogrid_layers[layer_idx].get_addressing_probabilities(layer_input)
            joint_prob = prob_info['joint']
            joint_prob_np = keras.ops.convert_to_numpy(joint_prob)

            # Create visualization
            fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 3, 6))
            fig.suptitle(f'Layer {layer_idx} Addressing Patterns ({self.grid_shapes[layer_idx]})',
                         fontsize=16, fontweight='bold')

            for i in range(n_samples):
                # Original digit
                axes[0, i].imshow(test_samples[i].squeeze(), cmap='gray')
                axes[0, i].set_title(f'Digit {test_labels[i]}')
                axes[0, i].axis('off')

                # Addressing pattern
                im = axes[1, i].imshow(joint_prob_np[i], cmap='viridis',
                                       interpolation='nearest', vmin=0, vmax=1)
                axes[1, i].set_title('Addressing Pattern')
                axes[1, i].set_xlabel('Grid X')
                axes[1, i].set_ylabel('Grid Y')
                plt.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)

            plt.tight_layout()
            plt.savefig(output_path / f'layer_{layer_idx}_addressing_patterns.png', dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            logger.error(f"Failed to create layer {layer_idx} addressing patterns: {e}")

    def analyze_digit_clustering_all_layers(self) -> None:
        """
        Analyze how different digits cluster in each NeuroGrid layer's addressing space.
        """
        if len(self.neurogrid_layers) == 0 or self.encoder is None or self.x_test is None:
            logger.warning("Required components not available for digit clustering analysis")
            return

        try:
            output_path = self.experiment_dir / "visualizations"
            logger.info("Analyzing digit clustering for all NeuroGrid layers...")

            # Get sample of test data for analysis
            sample_size = min(1000, len(self.x_test))
            sample_indices = np.random.choice(len(self.x_test), sample_size, replace=False)
            test_sample = self.x_test[sample_indices]
            test_labels = self.y_test[sample_indices]

            # Get encoded representations
            layer_input = self.encoder.predict(test_sample, verbose=0, batch_size=self.batch_size)

            # Analyze each layer
            all_layer_addressing = {}

            for layer_idx, neurogrid_layer in enumerate(self.neurogrid_layers):
                # Get addressing probabilities for current layer
                prob_info = neurogrid_layer.get_addressing_probabilities(layer_input)
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
                        digit_addressing[digit] = np.zeros(self.grid_shapes[layer_idx])

                all_layer_addressing[layer_idx] = digit_addressing

                # Create individual layer visualization
                self.visualize_single_layer_digit_clustering(layer_idx, digit_addressing)

                # Apply transformation for next layer
                layer_transformed = neurogrid_layer.predict(layer_input, verbose=0, batch_size=self.batch_size)
                if self.use_residual_connections and layer_idx > 0:
                    layer_input = layer_input + layer_transformed
                else:
                    layer_input = layer_transformed

            # Create comprehensive comparison visualization
            self.create_digit_clustering_comparison(all_layer_addressing)

            logger.info("Digit clustering analysis saved for all layers")

        except Exception as e:
            logger.error(f"Failed to create digit clustering analysis: {e}")

    def visualize_single_layer_digit_clustering(self, layer_idx: int, digit_addressing: Dict[int, np.ndarray]) -> None:
        """Visualize digit clustering for a single layer."""
        try:
            output_path = self.experiment_dir / "visualizations" / f"layer_{layer_idx}"

            # Create visualization
            fig, axes = plt.subplots(2, 5, figsize=(15, 6))
            fig.suptitle(
                f'Layer {layer_idx} Average Addressing Patterns by Digit Class ({self.grid_shapes[layer_idx]})',
                fontsize=14, fontweight='bold')

            for digit in range(10):
                row = digit // 5
                col = digit % 5

                im = axes[row, col].imshow(digit_addressing[digit], cmap='viridis',
                                           interpolation='nearest', vmin=0, vmax=1)
                axes[row, col].set_title(f'Digit {digit}')
                axes[row, col].set_xlabel('Grid X')
                axes[row, col].set_ylabel('Grid Y')
                plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)

            plt.tight_layout()
            plt.savefig(output_path / f'layer_{layer_idx}_digit_clustering.png', dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            logger.error(f"Failed to create layer {layer_idx} digit clustering: {e}")

    def create_digit_clustering_comparison(self, all_layer_addressing: Dict[int, Dict[int, np.ndarray]]) -> None:
        """Create comparison visualization of digit clustering across all layers."""
        try:
            output_path = self.experiment_dir / "visualizations"

            # Create comprehensive comparison for a few representative digits
            representative_digits = [0, 1, 4, 7]  # Digits with distinct shapes

            fig, axes = plt.subplots(len(representative_digits), self.num_neurogrid_layers,
                                     figsize=(self.num_neurogrid_layers * 3, len(representative_digits) * 3))
            fig.suptitle('Digit Clustering Comparison Across All NeuroGrid Layers',
                         fontsize=16, fontweight='bold')

            # Handle single layer case
            if self.num_neurogrid_layers == 1:
                axes = axes.reshape(-1, 1)

            for digit_idx, digit in enumerate(representative_digits):
                for layer_idx in range(self.num_neurogrid_layers):
                    digit_pattern = all_layer_addressing[layer_idx][digit]

                    im = axes[digit_idx, layer_idx].imshow(digit_pattern, cmap='viridis',
                                                           interpolation='nearest', vmin=0, vmax=1)
                    axes[digit_idx, layer_idx].set_title(
                        f'Digit {digit}\nLayer {layer_idx} ({self.grid_shapes[layer_idx]})')
                    axes[digit_idx, layer_idx].set_xlabel('Grid X')
                    axes[digit_idx, layer_idx].set_ylabel('Grid Y')

            plt.tight_layout()
            plt.savefig(output_path / 'digit_clustering_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            logger.error(f"Failed to create digit clustering comparison: {e}")

    def analyze_grid_utilization_all_layers(self) -> None:
        """
        Analyze NeuroGrid utilization patterns for all layers.
        """
        if len(self.neurogrid_layers) == 0 or self.encoder is None or self.x_test is None:
            logger.warning("Required components not available for grid utilization analysis")
            return

        try:
            logger.info("Analyzing grid utilization for all NeuroGrid layers...")

            # Get sample of test data
            sample_size = min(1000, len(self.x_test))
            sample_indices = np.random.choice(len(self.x_test), sample_size, replace=False)
            test_sample = self.x_test[sample_indices]

            # Get encoded representations
            layer_input = self.encoder.predict(test_sample, verbose=0, batch_size=self.batch_size)

            # Analyze each layer
            for layer_idx, neurogrid_layer in enumerate(self.neurogrid_layers):
                # Get grid utilization statistics
                utilization_stats = neurogrid_layer.get_grid_utilization(layer_input)

                # Create individual layer visualization
                self.visualize_single_layer_utilization(layer_idx, utilization_stats, sample_size)

                # Apply transformation for next layer
                layer_transformed = neurogrid_layer.predict(layer_input, verbose=0, batch_size=self.batch_size)
                if self.use_residual_connections and layer_idx > 0:
                    layer_input = layer_input + layer_transformed
                else:
                    layer_input = layer_transformed

            # Create utilization comparison across layers
            self.create_utilization_comparison()

            logger.info("Grid utilization analysis saved for all layers")

        except Exception as e:
            logger.error(f"Failed to create grid utilization analysis: {e}")

    def visualize_single_layer_utilization(self, layer_idx: int, utilization_stats: Dict, sample_size: int) -> None:
        """Visualize utilization for a single layer."""
        try:
            output_path = self.experiment_dir / "visualizations" / f"layer_{layer_idx}"

            # Convert to numpy arrays
            activation_counts = keras.ops.convert_to_numpy(utilization_stats['activation_counts'])
            total_activation = keras.ops.convert_to_numpy(utilization_stats['total_activation'])
            utilization_rate = keras.ops.convert_to_numpy(utilization_stats['utilization_rate'])

            # Reshape to grid format for visualization
            grid_h, grid_w = self.grid_shapes[layer_idx]
            activation_counts_grid = activation_counts.reshape(grid_h, grid_w)
            total_activation_grid = total_activation.reshape(grid_h, grid_w)
            utilization_rate_grid = utilization_rate.reshape(grid_h, grid_w)

            # Create visualization
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'Layer {layer_idx} NeuroGrid Utilization Analysis ({self.grid_shapes[layer_idx]})',
                         fontsize=14, fontweight='bold')

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
            total_positions = np.prod(self.grid_shapes[layer_idx])
            used_positions = np.sum(activation_counts > 0)
            max_activations = np.max(activation_counts)
            mean_utilization = np.mean(utilization_rate)

            stats_text = f"""Layer {layer_idx} Utilization Statistics:
Sample Size: {sample_size:,}
Grid Shape: {self.grid_shapes[layer_idx]}
Grid Positions: {total_positions}
Used Positions: {used_positions} ({used_positions / total_positions * 100:.1f}%)

Max Activations: {max_activations:.0f}
Mean Utilization Rate: {mean_utilization:.4f}

Most Active Position: {np.unravel_index(np.argmax(activation_counts_grid), self.grid_shapes[layer_idx])}
"""

            axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes,
                            verticalalignment='top', fontfamily='monospace', fontsize=10)
            axes[1, 1].set_xlim(0, 1)
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].axis('off')

            plt.tight_layout()
            plt.savefig(output_path / f'layer_{layer_idx}_utilization.png', dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            logger.error(f"Failed to create layer {layer_idx} utilization visualization: {e}")

    def create_utilization_comparison(self) -> None:
        """Create comparison visualization of utilization across all layers."""
        try:
            output_path = self.experiment_dir / "visualizations"

            # Get utilization statistics for all layers
            sample_size = min(1000, len(self.x_test))
            sample_indices = np.random.choice(len(self.x_test), sample_size, replace=False)
            test_sample = self.x_test[sample_indices]
            layer_input = self.encoder.predict(test_sample, verbose=0, batch_size=self.batch_size)

            utilization_stats = []
            for layer_idx, neurogrid_layer in enumerate(self.neurogrid_layers):
                stats = neurogrid_layer.get_grid_utilization(layer_input)
                utilization_rate = keras.ops.convert_to_numpy(stats['utilization_rate'])
                mean_utilization = np.mean(utilization_rate)
                total_positions = np.prod(self.grid_shapes[layer_idx])
                used_positions = np.sum(keras.ops.convert_to_numpy(stats['activation_counts']) > 0)

                utilization_stats.append({
                    'layer': layer_idx,
                    'mean_utilization': mean_utilization,
                    'usage_percentage': used_positions / total_positions * 100,
                    'grid_size': total_positions
                })

                # Update input for next layer
                layer_transformed = neurogrid_layer.predict(layer_input, verbose=0, batch_size=self.batch_size)
                if self.use_residual_connections and layer_idx > 0:
                    layer_input = layer_input + layer_transformed
                else:
                    layer_input = layer_transformed

            # Create comparison plots
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle('NeuroGrid Utilization Comparison Across All Layers', fontsize=16, fontweight='bold')

            # Mean utilization comparison
            layers = [s['layer'] for s in utilization_stats]
            mean_utils = [s['mean_utilization'] for s in utilization_stats]
            usage_percs = [s['usage_percentage'] for s in utilization_stats]
            grid_sizes = [s['grid_size'] for s in utilization_stats]

            axes[0].bar(layers, mean_utils, color='skyblue')
            axes[0].set_title('Mean Utilization Rate by Layer')
            axes[0].set_xlabel('Layer Index')
            axes[0].set_ylabel('Mean Utilization Rate')
            axes[0].set_xticks(layers)
            axes[0].grid(True, alpha=0.3)

            axes[1].bar(layers, usage_percs, color='lightgreen')
            axes[1].set_title('Grid Position Usage Percentage by Layer')
            axes[1].set_xlabel('Layer Index')
            axes[1].set_ylabel('Usage Percentage (%)')
            axes[1].set_xticks(layers)
            axes[1].grid(True, alpha=0.3)

            # Grid size comparison
            axes[2].bar(layers, grid_sizes, color='coral')
            axes[2].set_title('Grid Size by Layer')
            axes[2].set_xlabel('Layer Index')
            axes[2].set_ylabel('Total Grid Positions')
            axes[2].set_xticks(layers)
            axes[2].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_path / 'utilization_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            logger.error(f"Failed to create utilization comparison: {e}")

    def analyze_addressing_quality_all_layers(self) -> None:
        """Analyze addressing quality for all NeuroGrid layers."""
        if len(self.neurogrid_layers) == 0 or self.encoder is None or self.x_test is None:
            return

        try:
            logger.info("Analyzing addressing quality for all NeuroGrid layers...")

            # Get sample of test data for analysis
            sample_size = min(500, len(self.x_test))
            sample_indices = np.random.choice(len(self.x_test), sample_size, replace=False)
            test_sample = self.x_test[sample_indices]
            test_labels = self.y_test[sample_indices]

            # Get encoded representations (input to first NeuroGrid)
            layer_input = self.encoder.predict(test_sample, verbose=0)

            # Analyze each layer
            for layer_idx, neurogrid_layer in enumerate(self.neurogrid_layers):
                # Compute quality measures
                quality_measures = neurogrid_layer.compute_input_quality(layer_input)
                quality_stats = neurogrid_layer.get_quality_statistics(layer_input)

                # Create individual layer visualization
                self.visualize_single_layer_quality(layer_idx, quality_measures, quality_stats, test_labels)

                # Apply transformation for next layer
                layer_transformed = neurogrid_layer.predict(layer_input, verbose=0)
                if self.use_residual_connections and layer_idx > 0:
                    layer_input = layer_input + layer_transformed
                else:
                    layer_input = layer_transformed

            # Create quality comparison across layers
            self.create_quality_comparison()

            logger.info("Addressing quality analysis saved for all layers")

        except Exception as e:
            logger.error(f"Failed to create addressing quality analysis: {e}")

    def visualize_single_layer_quality(self, layer_idx: int, quality_measures: Dict,
                                       quality_stats: Dict, test_labels: np.ndarray) -> None:
        """Visualize quality analysis for a single layer."""
        try:
            output_path = self.experiment_dir / "visualizations" / f"layer_{layer_idx}"

            # Create quality analysis visualization
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f'Layer {layer_idx} NeuroGrid Addressing Quality Analysis', fontsize=14, fontweight='bold')

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
            summary_text = f"Layer {layer_idx} Quality Statistics:\n" + "\n".join([
                f"{key}: {value:.4f}" for key, value in quality_stats.items()
                if 'overall_quality' in key or 'addressing_confidence' in key
            ])

            axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
                            verticalalignment='top', fontfamily='monospace', fontsize=10)
            axes[1, 2].set_xlim(0, 1)
            axes[1, 2].set_ylim(0, 1)
            axes[1, 2].axis('off')

            plt.tight_layout()
            plt.savefig(output_path / f'layer_{layer_idx}_quality.png', dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            logger.error(f"Failed to create layer {layer_idx} quality visualization: {e}")

    def create_quality_comparison(self) -> None:
        """Create comparison visualization of quality across all layers."""
        try:
            output_path = self.experiment_dir / "visualizations"

            # Get quality statistics for all layers
            sample_size = min(500, len(self.x_test))
            sample_indices = np.random.choice(len(self.x_test), sample_size, replace=False)
            test_sample = self.x_test[sample_indices]
            layer_input = self.encoder.predict(test_sample, verbose=0)

            quality_comparison = []
            for layer_idx, neurogrid_layer in enumerate(self.neurogrid_layers):
                quality_measures = neurogrid_layer.compute_input_quality(layer_input)

                overall_quality = keras.ops.convert_to_numpy(quality_measures['overall_quality'])
                confidence = keras.ops.convert_to_numpy(quality_measures['addressing_confidence'])
                entropy = keras.ops.convert_to_numpy(quality_measures['addressing_entropy'])

                quality_comparison.append({
                    'layer': layer_idx,
                    'mean_quality': np.mean(overall_quality),
                    'mean_confidence': np.mean(confidence),
                    'mean_entropy': np.mean(entropy),
                    'std_quality': np.std(overall_quality)
                })

                # Update input for next layer
                layer_transformed = neurogrid_layer.predict(layer_input, verbose=0)
                if self.use_residual_connections and layer_idx > 0:
                    layer_input = layer_input + layer_transformed
                else:
                    layer_input = layer_transformed

            # Create comparison plots
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle('NeuroGrid Quality Comparison Across All Layers', fontsize=16, fontweight='bold')

            layers = [q['layer'] for q in quality_comparison]
            mean_qualities = [q['mean_quality'] for q in quality_comparison]
            mean_confidences = [q['mean_confidence'] for q in quality_comparison]
            mean_entropies = [q['mean_entropy'] for q in quality_comparison]

            axes[0].bar(layers, mean_qualities, color='lightblue')
            axes[0].set_title('Mean Overall Quality by Layer')
            axes[0].set_xlabel('Layer Index')
            axes[0].set_ylabel('Mean Quality Score')
            axes[0].set_xticks(layers)
            axes[0].grid(True, alpha=0.3)
            axes[0].set_ylim(0, 1)

            axes[1].bar(layers, mean_confidences, color='lightgreen')
            axes[1].set_title('Mean Addressing Confidence by Layer')
            axes[1].set_xlabel('Layer Index')
            axes[1].set_ylabel('Mean Confidence Score')
            axes[1].set_xticks(layers)
            axes[1].grid(True, alpha=0.3)
            axes[1].set_ylim(0, 1)

            axes[2].bar(layers, mean_entropies, color='lightcoral')
            axes[2].set_title('Mean Addressing Entropy by Layer')
            axes[2].set_xlabel('Layer Index')
            axes[2].set_ylabel('Mean Entropy Score')
            axes[2].set_xticks(layers)
            axes[2].grid(True, alpha=0.3)
            axes[2].set_ylim(0, 1)

            plt.tight_layout()
            plt.savefig(output_path / 'quality_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            logger.error(f"Failed to create quality comparison: {e}")

    def demonstrate_quality_filtering_all_layers(self) -> None:
        """
        Demonstrate NeuroGrid's quality-based filtering capabilities for all layers.
        """
        if len(self.neurogrid_layers) == 0 or self.encoder is None or self.x_test is None:
            logger.warning("Required components not available for quality filtering demonstration")
            return

        try:
            logger.info("Demonstrating quality-based filtering for all NeuroGrid layers...")

            # Analyze each layer individually
            for layer_idx in range(self.num_neurogrid_layers):
                self.demonstrate_single_layer_quality_filtering(layer_idx)

            # Create comparison across layers
            self.create_quality_filtering_comparison()

            logger.info("Quality filtering demonstration saved for all layers")

        except Exception as e:
            logger.error(f"Failed to create quality filtering demonstration: {e}")

    def demonstrate_single_layer_quality_filtering(self, layer_idx: int) -> None:
        """Demonstrate quality filtering for a single layer."""
        try:
            output_path = self.experiment_dir / "visualizations" / f"layer_{layer_idx}"

            # Get sample of test data
            sample_size = min(500, len(self.x_test))
            sample_indices = np.random.choice(len(self.x_test), sample_size, replace=False)
            test_sample = self.x_test[sample_indices]
            test_labels = self.y_test[sample_indices]

            # Get input to the specific layer
            layer_input = self.encoder.predict(test_sample, verbose=0)
            for i in range(layer_idx):
                neurogrid_layer = self.neurogrid_layers[i]
                layer_transformed = neurogrid_layer.predict(layer_input, verbose=0)
                if self.use_residual_connections and i > 0:
                    layer_input = layer_input + layer_transformed
                else:
                    layer_input = layer_transformed

            # Apply quality filtering with different thresholds
            thresholds = [0.3, 0.5, 0.7]
            filtering_results = {}

            for threshold in thresholds:
                filtered = self.neurogrid_layers[layer_idx].filter_by_quality_threshold(
                    layer_input,
                    quality_threshold=threshold,
                    quality_measure='overall_quality'
                )
                high_quality_shape = keras.ops.convert_to_numpy(keras.ops.shape(filtered['high_quality_inputs']))
                low_quality_shape = keras.ops.convert_to_numpy(keras.ops.shape(filtered['low_quality_inputs']))

                filtering_results[threshold] = {
                    'high_quality_count': int(high_quality_shape[0]),
                    'low_quality_count': int(low_quality_shape[0]),
                    'quality_scores': keras.ops.convert_to_numpy(filtered['quality_scores'])
                }

            # Create visualization
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'Layer {layer_idx} Quality-Based Filtering Demonstration', fontsize=14, fontweight='bold')

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

            axes[0, 1].bar(x_pos - width / 2, high_quality_counts, width, label='High Quality', color='lightgreen')
            axes[0, 1].bar(x_pos + width / 2, low_quality_counts, width, label='Low Quality', color='lightcoral')
            axes[0, 1].set_title('Filtering Results by Threshold')
            axes[0, 1].set_xlabel('Quality Threshold')
            axes[0, 1].set_ylabel('Sample Count')
            axes[0, 1].set_xticks(x_pos)
            axes[0, 1].set_xticklabels([str(t) for t in thresholds])
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # Quality by digit class
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

            # Summary statistics
            mean_quality = np.mean(all_quality_scores)
            std_quality = np.std(all_quality_scores)

            summary_text = f"""Layer {layer_idx} Quality Filtering:
Total Samples: {sample_size:,}
Mean Quality: {mean_quality:.3f} ± {std_quality:.3f}

Threshold 0.3: {high_quality_counts[0]:,} high ({high_quality_counts[0] / sample_size * 100:.1f}%)
Threshold 0.5: {high_quality_counts[1]:,} high ({high_quality_counts[1] / sample_size * 100:.1f}%)
Threshold 0.7: {high_quality_counts[2]:,} high ({high_quality_counts[2] / sample_size * 100:.1f}%)
"""

            axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                            verticalalignment='top', fontfamily='monospace', fontsize=10)
            axes[1, 1].set_xlim(0, 1)
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].axis('off')

            plt.tight_layout()
            plt.savefig(output_path / f'layer_{layer_idx}_quality_filtering.png', dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            logger.error(f"Failed to create layer {layer_idx} quality filtering: {e}")

    def create_quality_filtering_comparison(self) -> None:
        """Create comparison of quality filtering across all layers."""
        try:
            output_path = self.experiment_dir / "visualizations"

            # Get filtering results for all layers at a standard threshold
            threshold = 0.5
            sample_size = min(500, len(self.x_test))
            sample_indices = np.random.choice(len(self.x_test), sample_size, replace=False)
            test_sample = self.x_test[sample_indices]
            layer_input = self.encoder.predict(test_sample, verbose=0)

            filtering_comparison = []
            for layer_idx, neurogrid_layer in enumerate(self.neurogrid_layers):
                filtered = neurogrid_layer.filter_by_quality_threshold(
                    layer_input,
                    quality_threshold=threshold,
                    quality_measure='overall_quality'
                )

                high_quality_shape = keras.ops.convert_to_numpy(keras.ops.shape(filtered['high_quality_inputs']))
                quality_scores = keras.ops.convert_to_numpy(filtered['quality_scores'])

                filtering_comparison.append({
                    'layer': layer_idx,
                    'high_quality_count': int(high_quality_shape[0]),
                    'retention_rate': int(high_quality_shape[0]) / sample_size,
                    'mean_quality': np.mean(quality_scores)
                })

                # Update input for next layer
                layer_transformed = neurogrid_layer.predict(layer_input, verbose=0)
                if self.use_residual_connections and layer_idx > 0:
                    layer_input = layer_input + layer_transformed
                else:
                    layer_input = layer_transformed

            # Create comparison visualization
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle(f'Quality Filtering Comparison Across All Layers (Threshold {threshold})',
                         fontsize=16, fontweight='bold')

            layers = [f['layer'] for f in filtering_comparison]
            retention_rates = [f['retention_rate'] * 100 for f in filtering_comparison]
            mean_qualities = [f['mean_quality'] for f in filtering_comparison]

            axes[0].bar(layers, retention_rates, color='lightgreen')
            axes[0].set_title('Sample Retention Rate by Layer')
            axes[0].set_xlabel('Layer Index')
            axes[0].set_ylabel('Retention Rate (%)')
            axes[0].set_xticks(layers)
            axes[0].grid(True, alpha=0.3)
            axes[0].set_ylim(0, 100)

            axes[1].bar(layers, mean_qualities, color='lightblue')
            axes[1].set_title('Mean Quality Score by Layer')
            axes[1].set_xlabel('Layer Index')
            axes[1].set_ylabel('Mean Quality Score')
            axes[1].set_xticks(layers)
            axes[1].grid(True, alpha=0.3)
            axes[1].set_ylim(0, 1)

            plt.tight_layout()
            plt.savefig(output_path / 'quality_filtering_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            logger.error(f"Failed to create quality filtering comparison: {e}")

    def create_layer_interaction_analysis(self) -> None:
        """Analyze how layers interact through residual connections."""
        if not self.use_residual_connections or len(self.neurogrid_layers) < 2:
            logger.info("Skipping layer interaction analysis - requires residual connections and multiple layers")
            return

        try:
            output_path = self.experiment_dir / "visualizations"
            logger.info("Analyzing layer interactions through residual connections...")

            # Get sample data
            sample_size = min(500, len(self.x_test))
            sample_indices = np.random.choice(len(self.x_test), sample_size, replace=False)
            test_sample = self.x_test[sample_indices]

            # Track representations through the network
            representations = []
            layer_input = self.encoder.predict(test_sample, verbose=0)
            representations.append(('encoder_output', layer_input))

            for layer_idx, neurogrid_layer in enumerate(self.neurogrid_layers):
                # Store input to current layer
                representations.append((f'layer_{layer_idx}_input', layer_input))

                # Apply transformation
                layer_transformed = neurogrid_layer.predict(layer_input, verbose=0)
                representations.append((f'layer_{layer_idx}_output', layer_transformed))

                # Apply residual connection
                if self.use_residual_connections and layer_idx > 0:
                    layer_input = layer_input + layer_transformed
                    representations.append((f'layer_{layer_idx}_residual', layer_input))
                else:
                    layer_input = layer_transformed

            # Analyze representation similarities
            similarities = self.compute_representation_similarities(representations)

            # Create visualization
            self.visualize_layer_interactions(similarities)

            logger.info("Layer interaction analysis completed")

        except Exception as e:
            logger.error(f"Failed to create layer interaction analysis: {e}")

    def compute_representation_similarities(self, representations: List[Tuple[str, np.ndarray]]) -> np.ndarray:
        """Compute cosine similarities between different representations."""
        try:
            # Compute pairwise cosine similarities
            n_representations = len(representations)
            similarity_matrix = np.zeros((n_representations, n_representations))

            for i in range(n_representations):
                for j in range(n_representations):
                    if i == j:
                        similarity_matrix[i, j] = 1.0
                    else:
                        # Flatten representations and compute cosine similarity
                        repr1 = representations[i][1].flatten()
                        repr2 = representations[j][1].flatten()

                        # Compute cosine similarity
                        dot_product = np.dot(repr1, repr2)
                        norm1 = np.linalg.norm(repr1)
                        norm2 = np.linalg.norm(repr2)

                        if norm1 > 0 and norm2 > 0:
                            similarity = dot_product / (norm1 * norm2)
                        else:
                            similarity = 0.0

                        similarity_matrix[i, j] = similarity

            return similarity_matrix

        except Exception as e:
            logger.error(f"Failed to compute representation similarities: {e}")
            return np.eye(len(representations))

    def visualize_layer_interactions(self, similarity_matrix: np.ndarray) -> None:
        """Visualize layer interactions through similarity matrix."""
        try:
            output_path = self.experiment_dir / "visualizations"

            fig, ax = plt.subplots(1, 1, figsize=(10, 8))

            # Create labels for the similarity matrix
            labels = []
            labels.append('Encoder Output')
            for layer_idx in range(self.num_neurogrid_layers):
                labels.append(f'L{layer_idx} Input')
                labels.append(f'L{layer_idx} Output')
                if self.use_residual_connections and layer_idx > 0:
                    labels.append(f'L{layer_idx} Residual')

            # Plot similarity matrix
            im = ax.imshow(similarity_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            ax.set_title('Layer Representation Similarity Matrix\n(Cosine Similarity)',
                         fontsize=14, fontweight='bold')

            # Set ticks and labels
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_yticklabels(labels)

            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Add text annotations for key values
            for i in range(len(labels)):
                for j in range(len(labels)):
                    if abs(similarity_matrix[i, j]) > 0.5 and i != j:  # Only show strong similarities
                        text = f'{similarity_matrix[i, j]:.2f}'
                        ax.text(j, i, text, ha='center', va='center',
                                color='white' if abs(similarity_matrix[i, j]) > 0.7 else 'black')

            plt.tight_layout()
            plt.savefig(output_path / 'layer_interactions.png', dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            logger.error(f"Failed to visualize layer interactions: {e}")

    def save_experiment_summary(self) -> None:
        """Save comprehensive experiment summary."""
        try:
            # Save configuration
            config_dict = {
                'num_neurogrid_layers': self.num_neurogrid_layers,
                'grid_shapes': self.grid_shapes,
                'latent_dim': self.latent_dim,
                'use_residual_connections': self.use_residual_connections,
                'temperature': self.temperature,
                'conv_activation': self.conv_activation,
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'learning_rate': self.learning_rate,
                'final_temperatures': [layer.get_current_temperature() for layer in self.neurogrid_layers]
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

            # Save layer-specific information
            with open(self.experiment_dir / "logs" / "neurogrid_layers_info.txt", 'w') as f:
                f.write("Multi-NeuroGrid Layers Information\n")
                f.write("=" * 50 + "\n")
                f.write(f"Number of layers: {self.num_neurogrid_layers}\n")
                f.write(f"Residual connections: {self.use_residual_connections}\n\n")

                for i, (layer, grid_shape) in enumerate(zip(self.neurogrid_layers, self.grid_shapes)):
                    f.write(f"Layer {i}:\n")
                    f.write(f"  Grid shape: {grid_shape}\n")
                    f.write(f"  Grid size: {np.prod(grid_shape)}\n")
                    f.write(f"  Temperature: {layer.get_current_temperature():.6f}\n")
                    f.write(f"  Parameters: {layer.count_params()}\n\n")

            logger.info("Experiment summary saved")

        except Exception as e:
            logger.error(f"Failed to save experiment summary: {e}")

    def run_complete_experiment(self) -> None:
        """Execute the complete experimental pipeline for multi-NeuroGrid architecture."""
        logger.info("=" * 80)
        logger.info(f"STARTING MULTI-NEUROGRID CONV AUTOENCODER EXPERIMENT ({self.num_neurogrid_layers} LAYERS)")
        logger.info("=" * 80)

        try:
            # Phase 1: Data preparation
            self.load_and_preprocess_data()

            # Phase 2: Model creation
            self.model = self.create_multi_neurogrid_conv_autoencoder()
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

            # Multi-layer specific visualizations
            self.visualize_all_neurogrid_prototypes()
            self.visualize_addressing_patterns_all_layers(n_samples=5)
            self.analyze_digit_clustering_all_layers()
            self.analyze_grid_utilization_all_layers()
            self.analyze_addressing_quality_all_layers()
            self.demonstrate_quality_filtering_all_layers()

            # Advanced analysis
            self.create_layer_interaction_analysis()

            # Phase 7: Save summary
            self.save_experiment_summary()

            logger.info("=" * 80)
            logger.info("MULTI-NEUROGRID EXPERIMENT COMPLETED SUCCESSFULLY")
            logger.info(f"All results saved to: {self.experiment_dir}")
            logger.info("Generated visualizations:")
            logger.info("  Global:")
            logger.info("    - training_progress.png")
            logger.info("    - reconstructions.png")
            logger.info("    - all_neurogrid_prototypes.png")
            logger.info("    - addressing_patterns_all_layers.png")
            logger.info("    - digit_clustering_comparison.png")
            logger.info("    - utilization_comparison.png")
            logger.info("    - quality_comparison.png")
            logger.info("    - quality_filtering_comparison.png")
            if self.use_residual_connections:
                logger.info("    - layer_interactions.png")
            logger.info("  Per-layer (in layer_X subdirectories):")
            logger.info("    - layer_X_prototypes.png")
            logger.info("    - layer_X_addressing_patterns.png")
            logger.info("    - layer_X_digit_clustering.png")
            logger.info("    - layer_X_utilization.png")
            logger.info("    - layer_X_quality.png")
            logger.info("    - layer_X_quality_filtering.png")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Experiment failed: {e}", exc_info=True)
            raise


def main() -> None:
    """Run the Multi-NeuroGrid autoencoder experiment."""
    experiment = MultiNeuroGridAutoencoderExperiment(
        num_neurogrid_layers=3,
        grid_shapes=[(6, 6), (4, 4), (8, 8)],  # Different grid sizes per layer
        latent_dim=64,
        use_residual_connections=True,
        temperature=0.8,
        conv_activation='silu',
        batch_size=128,
        epochs=100,
        learning_rate=0.001,
        output_dir='results'
    )

    experiment.run_complete_experiment()


if __name__ == "__main__":
    main()