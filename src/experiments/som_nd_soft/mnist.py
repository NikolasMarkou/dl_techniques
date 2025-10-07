"""
SoftSOM Layer MNIST Reconstruction Experiment
==============================================

A comprehensive experiment demonstrating the SoftSOMLayer's capabilities for
differentiable self-organizing map learning on MNIST digit reconstruction using
a convolutional encoder-decoder architecture with SoftSOM bottleneck.

Hypothesis:
-----------
The hypothesis is that a differentiable Self-Organizing Map (SoftSOM) can learn
topologically-organized representations in a deep learning architecture while
maintaining reconstruction quality. The soft assignment mechanism should enable
end-to-end training while creating interpretable, spatially-organized latent
representations.

Experimental Design:
--------------------
- **Dataset**: MNIST (28×28 grayscale images, 10 classes), using standardized
  dataset builder with proper preprocessing.

- **Architecture**: Convolutional encoder-decoder with SoftSOM bottleneck:
    - Encoder: Strided Conv2D layers with BatchNorm and activation
    - Bottleneck: SoftSOM layer for topological organization
    - Decoder: UpSampling + Conv2D layers for reconstruction

- **SoftSOM Properties**:
    - Differentiable soft assignments via softmax
    - Topological neighborhood preservation
    - Grid-based organization of latent space
    - Per-dimension softmax for better gradient flow

- **Analysis**: Comprehensive visualization of learned prototypes, soft
  assignments, digit clustering, and reconstruction quality.

This version integrates the visualization framework and uses standardized dataset loaders.
"""

# ==============================================================================
# IMPORTS AND DEPENDENCIES
# ==============================================================================

import json
import keras
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

from dl_techniques.utils.logger import logger
from dl_techniques.layers.memory.som_nd_soft_layer import SoftSOMLayer
from dl_techniques.regularizers.soft_orthogonal import SoftOrthonormalConstraintRegularizer

from dl_techniques.datasets.vision.common import (
    create_dataset_builder,
)

from dl_techniques.visualization import (
    VisualizationManager,
    VisualizationPlugin,
    TrainingHistory,
    PlotConfig,
    PlotStyle,
    ColorScheme,
    TrainingCurvesVisualization,
)

# ==============================================================================
# CUSTOM DATA STRUCTURES
# ==============================================================================

@dataclass
class SOMPrototypeData:
    """
    Data structure for SOM prototype visualization.

    Attributes:
        prototypes: Grid of prototype images
        grid_shape: Shape of the SOM grid (height, width)
    """
    prototypes: np.ndarray
    grid_shape: Tuple[int, int]


@dataclass
class SOMAssignmentData:
    """
    Data structure for SOM soft assignment visualization.

    Attributes:
        original_images: Original input images
        assignments: Soft assignment maps
        labels: Labels for the images
    """
    original_images: np.ndarray
    assignments: np.ndarray
    labels: np.ndarray


@dataclass
class SOMClusteringData:
    """
    Data structure for SOM digit clustering analysis.

    Attributes:
        digit_assignments: Average activation patterns per digit class
        purity_stats: Grid position purity statistics
        grid_shape: Shape of the SOM grid
    """
    digit_assignments: Dict[int, np.ndarray]
    purity_stats: Dict[str, Any]
    grid_shape: Tuple[int, int]


@dataclass
class ReconstructionData:
    """
    Data structure for reconstruction visualization.

    Attributes:
        originals: Original input images
        reconstructions: Reconstructed images
        labels: Labels for the images
    """
    originals: np.ndarray
    reconstructions: np.ndarray
    labels: np.ndarray


# ==============================================================================
# CUSTOM VISUALIZATION PLUGINS
# ==============================================================================

class SOMPrototypeVisualization(VisualizationPlugin):
    """Visualization plugin for SOM learned prototypes."""

    @property
    def name(self) -> str:
        return "som_prototypes"

    @property
    def description(self) -> str:
        return "Visualizes learned SOM prototype vectors as decoded images"

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, SOMPrototypeData)

    def create_visualization(
        self,
        data: SOMPrototypeData,
        ax: Optional[Any] = None,
        **kwargs
    ) -> Any:
        """Create SOM prototype grid visualization."""
        import matplotlib.pyplot as plt

        grid_h, grid_w = data.grid_shape
        fig, axes = plt.subplots(
            grid_h, grid_w,
            figsize=(grid_w * 2, grid_h * 2),
            dpi=self.config.dpi
        )

        # Ensure axes is a 2D array
        if grid_h == 1 and grid_w == 1:
            axes = np.array([[axes]])
        elif grid_h == 1:
            axes = axes.reshape(1, -1)
        elif grid_w == 1:
            axes = axes.reshape(-1, 1)

        for i in range(grid_h):
            for j in range(grid_w):
                axes[i, j].imshow(data.prototypes[i, j], cmap='gray')
                axes[i, j].set_title(f'({i},{j})', fontsize=10)
                axes[i, j].axis('off')

        plt.suptitle(
            'Learned SOM Prototype Vectors (Decoded as Images)',
            fontsize=self.config.title_fontsize,
            fontweight='bold'
        )
        plt.tight_layout()

        return fig


class SOMAssignmentVisualization(VisualizationPlugin):
    """Visualization plugin for SOM soft assignments."""

    @property
    def name(self) -> str:
        return "som_assignments"

    @property
    def description(self) -> str:
        return "Visualizes soft assignment patterns for sample inputs"

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, SOMAssignmentData)

    def create_visualization(
        self,
        data: SOMAssignmentData,
        ax: Optional[Any] = None,
        **kwargs
    ) -> Any:
        """Create SOM assignment visualization."""
        import matplotlib.pyplot as plt

        n_samples = len(data.original_images)
        fig, axes = plt.subplots(
            2, n_samples,
            figsize=(n_samples * 3, 6),
            dpi=self.config.dpi
        )

        for i in range(n_samples):
            # Original digit
            axes[0, i].imshow(data.original_images[i].squeeze(), cmap='gray')
            axes[0, i].set_title(f'Digit {data.labels[i]}', fontsize=12)
            axes[0, i].axis('off')

            # Soft assignment heatmap
            im = axes[1, i].imshow(
                data.assignments[i],
                cmap='viridis',
                interpolation='nearest'
            )
            axes[1, i].set_title('SOM Activation', fontsize=12)
            axes[1, i].set_xlabel('Grid X', fontsize=10)
            axes[1, i].set_ylabel('Grid Y', fontsize=10)

            # Add colorbar
            plt.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)

        plt.suptitle(
            'Input Digits and Their SOM Soft Assignments',
            fontsize=self.config.title_fontsize,
            fontweight='bold'
        )
        plt.tight_layout()

        return fig


class SOMClusteringVisualization(VisualizationPlugin):
    """Visualization plugin for SOM digit clustering analysis."""

    @property
    def name(self) -> str:
        return "som_clustering"

    @property
    def description(self) -> str:
        return "Analyzes and visualizes digit clustering on SOM grid"

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, SOMClusteringData)

    def create_visualization(
        self,
        data: SOMClusteringData,
        ax: Optional[Any] = None,
        **kwargs
    ) -> Any:
        """Create digit clustering visualization."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 5, figsize=(15, 6), dpi=self.config.dpi)

        for digit in range(10):
            row = digit // 5
            col = digit % 5

            im = axes[row, col].imshow(
                data.digit_assignments[digit],
                cmap='viridis'
            )
            axes[row, col].set_title(f'Digit {digit}', fontsize=12)
            axes[row, col].set_xlabel('Grid X', fontsize=10)
            axes[row, col].set_ylabel('Grid Y', fontsize=10)

            # Add colorbar
            plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)

        plt.suptitle(
            f"Average SOM Activations by Digit Class\nGrid Purity: {data.purity_stats['avg_purity']:.3f}",
            fontsize=self.config.title_fontsize,
            fontweight='bold'
        )
        plt.tight_layout()

        return fig


class ReconstructionVisualization(VisualizationPlugin):
    """Visualization plugin for reconstruction quality."""

    @property
    def name(self) -> str:
        return "reconstructions"

    @property
    def description(self) -> str:
        return "Visualizes original vs reconstructed images with error maps"

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, ReconstructionData)

    def create_visualization(
        self,
        data: ReconstructionData,
        ax: Optional[Any] = None,
        **kwargs
    ) -> Any:
        """Create reconstruction comparison visualization."""
        import matplotlib.pyplot as plt

        n_samples = len(data.originals)
        fig, axes = plt.subplots(
            3, n_samples,
            figsize=(n_samples * 2, 6),
            dpi=self.config.dpi
        )

        for i in range(n_samples):
            # Original images
            axes[0, i].imshow(data.originals[i].squeeze(), cmap='gray')
            axes[0, i].set_title(f'Original\n(Digit {data.labels[i]})', fontsize=10)
            axes[0, i].axis('off')

            # Reconstructed images
            axes[1, i].imshow(data.reconstructions[i].squeeze(), cmap='gray')
            axes[1, i].set_title('Reconstructed', fontsize=10)
            axes[1, i].axis('off')

            # Difference (error) images
            diff = np.abs(data.originals[i].squeeze() - data.reconstructions[i].squeeze())
            axes[2, i].imshow(diff, cmap='hot')
            axes[2, i].set_title(f'Error\n(MAE: {np.mean(diff):.3f})', fontsize=10)
            axes[2, i].axis('off')

        plt.suptitle(
            'Original vs Reconstructed MNIST Digits (Convolutional)',
            fontsize=self.config.title_fontsize,
            fontweight='bold'
        )
        plt.tight_layout()

        return fig


# ==============================================================================
# EXPERIMENT CONFIGURATION
# ==============================================================================

@dataclass
class TrainingConfig:
    """
    Minimal training configuration for dataset builder compatibility.

    Attributes:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of output classes
        batch_size: Training batch size
        epochs: Number of training epochs
    """
    input_shape: Tuple[int, int, int] = (28, 28, 1)
    num_classes: int = 10
    batch_size: int = 128
    epochs: int = 100


@dataclass
class SoftSOMExperimentConfig:
    """
    Configuration for the SoftSOM MNIST reconstruction experiment.

    This class encapsulates all configurable parameters for the experiment,
    including SoftSOM parameters, architecture settings, training parameters,
    and analysis configuration.
    """

    # --- Dataset Configuration ---
    dataset_name: str = "mnist"
    num_classes: int = 10
    input_shape: Tuple[int, ...] = (28, 28, 1)
    use_rgb: bool = False  # Keep MNIST in grayscale

    # --- SoftSOM Parameters ---
    som_grid_shape: Tuple[int, int] = (4, 4)
    som_temperature: float = 0.1
    use_per_dimension_softmax: bool = True
    topological_weight: float = 0.1
    sharpness_weight: float = 0.1

    # --- Architecture Parameters ---
    latent_dim: int = 16
    conv_activation: str = 'relu'

    # --- Training Parameters ---
    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 0.001
    early_stopping_patience: int = 10
    lr_patience: int = 5

    # --- Experiment Configuration ---
    output_dir: Path = Path("results")
    experiment_name: str = "softsom_mnist_reconstruction"
    random_seed: int = 42


# ==============================================================================
# MODEL ARCHITECTURE
# ==============================================================================

def create_som_autoencoder(config: SoftSOMExperimentConfig) -> Tuple[keras.Model, keras.Model, keras.Model, SoftSOMLayer]:
    """
    Create convolutional autoencoder model with SoftSOM bottleneck.

    Architecture:
        Encoder: Conv2D(stride=2) → BN → Activation → ... → Dense → latent
        SoftSOM: Differentiable topological bottleneck
        Decoder: Dense → Reshape → UpSample → Conv2D → ... → output

    Args:
        config: Experiment configuration

    Returns:
        Tuple of (full_model, encoder, decoder, som_layer)
    """
    logger.info("Creating convolutional SoftSOM autoencoder architecture...")

    # --- ENCODER ---
    encoder_input = keras.layers.Input(shape=(28, 28, 1), name="encoder_input")
    x = keras.layers.Conv2D(32, 5, strides=2, padding='same', activation='linear', name='encoder_conv1')(encoder_input)
    x = keras.layers.BatchNormalization(name='encoder_bn1')(x)
    x = keras.layers.Activation(config.conv_activation, name='encoder_act1')(x)
    x = keras.layers.Conv2D(64, 5, strides=2, padding='same', activation='linear', name='encoder_conv2')(x)
    x = keras.layers.BatchNormalization(name='encoder_bn2')(x)
    x = keras.layers.Activation(config.conv_activation, name='encoder_act2')(x)
    x = keras.layers.Flatten(name='encoder_flatten')(x)
    encoder_output = keras.layers.Dense(config.latent_dim, activation='linear', name='encoder_dense')(x)
    encoder = keras.Model(inputs=encoder_input, outputs=encoder_output, name="encoder")

    # --- SOM BOTTLENECK ---
    som_layer = SoftSOMLayer(
        grid_shape=config.som_grid_shape,
        input_dim=config.latent_dim,
        temperature=config.som_temperature,
        use_per_dimension_softmax=config.use_per_dimension_softmax,
        use_reconstruction_loss=False,
        topological_weight=config.topological_weight,
        sharpness_weight=config.sharpness_weight,
        kernel_regularizer=SoftOrthonormalConstraintRegularizer(0.1, 0.0, 0.01),
        name="soft_som_bottleneck"
    )
    som_output = som_layer(encoder_output)

    # --- DECODER ---
    decoder_layers = [
        keras.layers.Dense(7 * 7 * 64, name='decoder_dense'),
        keras.layers.Reshape((7, 7, 64), name='decoder_reshape'),
        keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest', name='decoder_upsample1'),
        keras.layers.Conv2D(32, 3, padding='same', activation='linear', name='decoder_conv1'),
        keras.layers.BatchNormalization(name='decoder_bn1'),
        keras.layers.Activation(config.conv_activation, name='decoder_act1'),
        keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest', name='decoder_upsample2'),
        keras.layers.Conv2D(16, 3, padding='same', activation='linear', name='decoder_conv2'),
        keras.layers.BatchNormalization(name='decoder_bn2'),
        keras.layers.Activation(config.conv_activation, name='decoder_act2'),
        keras.layers.Conv2D(1, 1, padding='same', activation='sigmoid', name='decoder_output'),
    ]

    # Build standalone decoder
    decoder_input = keras.layers.Input(shape=(config.latent_dim,), name="decoder_input")
    dec_x = decoder_input
    for layer in decoder_layers:
        dec_x = layer(dec_x)
    decoder = keras.Model(inputs=decoder_input, outputs=dec_x, name="decoder")

    # Build full autoencoder
    autoencoder_output = som_output
    for layer in decoder_layers:
        autoencoder_output = layer(autoencoder_output)
    autoencoder = keras.Model(
        inputs=encoder_input,
        outputs=autoencoder_output,
        name="convolutional_softsom_autoencoder"
    )

    # Compile
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss='mse',
        metrics=['mae']
    )

    return autoencoder, encoder, decoder, som_layer


# ==============================================================================
# TRAINING UTILITIES
# ==============================================================================

def train_model(
    model: keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    config: SoftSOMExperimentConfig,
    steps_per_epoch: int,
    val_steps: int,
    output_dir: Path
) -> Dict[str, List[float]]:
    """
    Train the SoftSOM autoencoder model.

    Args:
        model: Keras model to train
        train_ds: Training dataset
        val_ds: Validation dataset
        config: Experiment configuration
        steps_per_epoch: Number of steps per epoch
        val_steps: Number of validation steps
        output_dir: Directory to save checkpoints

    Returns:
        Training history dictionary
    """
    # Create callbacks
    callbacks_list = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=config.lr_patience,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / 'best_model.keras'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

    # Train the model (autoencoder: input == target)
    history = model.fit(
        train_ds.map(lambda x, y: (x, x)),  # Use images as both input and target
        epochs=config.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds.map(lambda x, y: (x, x)),
        validation_steps=val_steps,
        callbacks=callbacks_list,
        verbose=1
    )

    return history.history


# ==============================================================================
# MAIN EXPERIMENT RUNNER
# ==============================================================================

def run_experiment(config: SoftSOMExperimentConfig) -> Dict[str, Any]:
    """
    Run the complete SoftSOM MNIST reconstruction experiment.

    This function orchestrates the entire experimental pipeline, including:
    1. Dataset loading and preprocessing using standardized builder
    2. Model creation with SoftSOM bottleneck
    3. Model training
    4. Comprehensive visualization and analysis
    5. Results compilation and reporting

    Args:
        config: Experiment configuration specifying all parameters

    Returns:
        Dictionary containing all experimental results and analysis
    """
    # Set random seed for reproducibility
    keras.utils.set_random_seed(config.random_seed)
    np.random.seed(config.random_seed)
    tf.random.set_seed(config.random_seed)

    # Create timestamped output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = config.output_dir / f"{config.experiment_name}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (experiment_dir / "visualizations").mkdir(exist_ok=True)
    (experiment_dir / "models").mkdir(exist_ok=True)
    (experiment_dir / "logs").mkdir(exist_ok=True)

    # Initialize visualization manager
    viz_config = PlotConfig(
        style=PlotStyle.SCIENTIFIC,
        color_scheme=ColorScheme(
            primary='#2E86AB',
            secondary='#A23B72',
            accent='#F18F01'
        ),
        title_fontsize=14,
        save_format='png'
    )

    viz_manager = VisualizationManager(
        experiment_name=config.experiment_name,
        output_dir=experiment_dir / "visualizations",
        config=viz_config
    )

    # Register visualization plugins
    viz_manager.register_template("som_prototypes", SOMPrototypeVisualization)
    viz_manager.register_template("som_assignments", SOMAssignmentVisualization)
    viz_manager.register_template("som_clustering", SOMClusteringVisualization)
    viz_manager.register_template("reconstructions", ReconstructionVisualization)
    viz_manager.register_template("training_curves", TrainingCurvesVisualization)

    # Log experiment start
    logger.info("Starting SoftSOM MNIST Reconstruction Experiment")
    logger.info(f"Results will be saved to: {experiment_dir}")
    logger.info("=" * 80)

    # ===== DATASET LOADING =====
    logger.info("Loading MNIST dataset using standardized builder...")

    # Create training config for dataset builder
    train_config = TrainingConfig(
        input_shape=config.input_shape,
        num_classes=config.num_classes,
        batch_size=config.batch_size,
        epochs=config.epochs
    )

    # Create dataset builder with grayscale format
    dataset_builder = create_dataset_builder('mnist', train_config, use_rgb=config.use_rgb)

    # Build datasets
    train_ds, val_ds, steps_per_epoch, val_steps = dataset_builder.build()

    # Get test data for evaluation
    test_data = dataset_builder.get_test_data()

    logger.info("Dataset loaded successfully")
    logger.info(f"Training shape: {test_data.x_data.shape}")
    logger.info(f"Steps per epoch: {steps_per_epoch}, Validation steps: {val_steps}")

    # ===== MODEL CREATION =====
    logger.info("Creating convolutional SoftSOM autoencoder...")

    model, encoder, decoder, som_layer = create_som_autoencoder(config)

    model.summary(print_fn=logger.info)
    logger.info(f"Model created with {model.count_params():,} parameters")

    # ===== MODEL TRAINING =====
    logger.info(f"Starting training for {config.epochs} epochs...")

    import time
    start_time = time.time()
    history = train_model(
        model, train_ds, val_ds, config,
        steps_per_epoch, val_steps,
        experiment_dir / "models"
    )
    training_time = time.time() - start_time

    logger.info(f"Training completed in {training_time:.2f} seconds")

    # ===== SAVE FINAL MODEL =====
    model.save(experiment_dir / "models" / "final_model.keras")
    logger.info("Final model saved")

    # ===== VISUALIZATION GENERATION =====
    logger.info("Generating comprehensive visualizations...")

    # 1. Training progress
    training_history = TrainingHistory(
        epochs=list(range(len(history['loss']))),
        train_loss=history['loss'],
        val_loss=history['val_loss'],
        train_metrics={'mae': history['mae']},
        val_metrics={'mae': history['val_mae']}
    )

    viz_manager.visualize(
        data=training_history,
        plugin_name="training_curves",
        show=False
    )

    # 2. Reconstructions
    test_sample_idx = np.random.choice(len(test_data.x_data), 8, replace=False)
    test_samples = test_data.x_data[test_sample_idx]
    test_labels = test_data.y_data[test_sample_idx]

    reconstructions = model.predict(test_samples, verbose=0)

    recon_data = ReconstructionData(
        originals=test_samples,
        reconstructions=reconstructions,
        labels=test_labels
    )

    viz_manager.visualize(
        data=recon_data,
        plugin_name="reconstructions",
        show=False
    )

    # 3. SOM prototypes
    weights_map = som_layer.get_weights_map()
    grid_h, grid_w = config.som_grid_shape

    prototypes_flat = keras.ops.reshape(weights_map, (-1, config.latent_dim))
    decoded_prototypes = decoder.predict(prototypes_flat, verbose=0)
    prototype_images = decoded_prototypes.reshape(grid_h, grid_w, 28, 28)

    prototype_data = SOMPrototypeData(
        prototypes=prototype_images,
        grid_shape=config.som_grid_shape
    )

    viz_manager.visualize(
        data=prototype_data,
        plugin_name="som_prototypes",
        show=False
    )

    # 4. SOM assignments
    assign_sample_idx = np.random.choice(len(test_data.x_data), 5, replace=False)
    assign_samples = test_data.x_data[assign_sample_idx]
    assign_labels = test_data.y_data[assign_sample_idx]

    encoded_samples = encoder.predict(assign_samples, verbose=0)
    assignments = som_layer.get_soft_assignments(encoded_samples)
    assignments_np = keras.ops.convert_to_numpy(assignments)

    assignment_data = SOMAssignmentData(
        original_images=assign_samples,
        assignments=assignments_np,
        labels=assign_labels
    )

    viz_manager.visualize(
        data=assignment_data,
        plugin_name="som_assignments",
        show=False
    )

    # 5. Digit clustering analysis
    logger.info("Analyzing digit clustering on SOM grid...")

    encoded_test = encoder.predict(test_data.x_data, verbose=0, batch_size=config.batch_size)
    assignments_all = som_layer.get_soft_assignments(encoded_test)
    assignments_all_np = keras.ops.convert_to_numpy(assignments_all)

    # Compute average assignments per digit
    digit_assignments = {}
    for digit in range(10):
        digit_mask = test_data.y_data == digit
        digit_avg = np.mean(assignments_all_np[digit_mask], axis=0)
        digit_assignments[digit] = digit_avg

    # Compute clustering statistics
    assignments_flat = assignments_all_np.reshape(len(assignments_all_np), -1)
    bmus = np.argmax(assignments_flat, axis=1)

    grid_size = np.prod(config.som_grid_shape)
    position_purity = {}

    for pos in range(grid_size):
        pos_mask = bmus == pos
        if np.sum(pos_mask) > 0:
            pos_labels = test_data.y_data[pos_mask]
            unique_labels, counts = np.unique(pos_labels, return_counts=True)
            purity = float(np.max(counts) / np.sum(counts))
            position_purity[pos] = (purity, int(unique_labels[np.argmax(counts)]))

    avg_purity = float(np.mean([purity for purity, _ in position_purity.values()])) if position_purity else 0.0

    purity_stats = {
        'avg_purity': avg_purity,
        'grid_size': grid_size,
        'positions_with_samples': len(position_purity),
        'position_details': position_purity
    }

    clustering_data = SOMClusteringData(
        digit_assignments=digit_assignments,
        purity_stats=purity_stats,
        grid_shape=config.som_grid_shape
    )

    viz_manager.visualize(
        data=clustering_data,
        plugin_name="som_clustering",
        show=False
    )

    # ===== RESULTS COMPILATION =====
    final_train_loss = history['loss'][-1]
    final_val_loss = history['val_loss'][-1]
    best_val_loss = min(history['val_loss'])

    results_payload = {
        'training_time': training_time,
        'final_train_loss': final_train_loss,
        'final_val_loss': final_val_loss,
        'best_val_loss': best_val_loss,
        'history': history,
        'clustering_stats': purity_stats,
        'model': model,
        'encoder': encoder,
        'decoder': decoder,
        'som_layer': som_layer
    }

    # Save and print results
    save_experiment_results(results_payload, experiment_dir, config)
    print_experiment_summary(results_payload, config)

    return results_payload


# ==============================================================================
# RESULTS SAVING AND REPORTING
# ==============================================================================

def save_experiment_results(
    results: Dict[str, Any],
    experiment_dir: Path,
    config: SoftSOMExperimentConfig
) -> None:
    """
    Save experiment results in multiple formats.

    Args:
        results: Experiment results dictionary
        experiment_dir: Directory to save results
        config: Experiment configuration
    """
    try:
        # Save configuration
        config_dict = {
            'experiment_name': config.experiment_name,
            'som_grid_shape': config.som_grid_shape,
            'som_temperature': config.som_temperature,
            'latent_dim': config.latent_dim,
            'batch_size': config.batch_size,
            'epochs': config.epochs,
            'learning_rate': config.learning_rate,
            'conv_activation': config.conv_activation,
            'random_seed': config.random_seed
        }
        with open(experiment_dir / "experiment_config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)

        # Save training history
        history_dict = {key: [float(v) for v in values] for key, values in results['history'].items()}
        with open(experiment_dir / "training_history.json", 'w') as f:
            json.dump(history_dict, f, indent=2)

        # Save clustering statistics
        clustering_stats = {
            'avg_purity': results['clustering_stats']['avg_purity'],
            'grid_size': results['clustering_stats']['grid_size'],
            'positions_with_samples': results['clustering_stats']['positions_with_samples']
        }
        with open(experiment_dir / "clustering_stats.json", 'w') as f:
            json.dump(clustering_stats, f, indent=2)

        # Save model summary
        with open(experiment_dir / "logs" / "model_summary.txt", 'w') as f:
            results['model'].summary(print_fn=lambda x: f.write(x + '\n'))

        logger.info("Experiment results saved successfully")

    except Exception as e:
        logger.error(f"Failed to save experiment results: {e}", exc_info=True)


def print_experiment_summary(
    results: Dict[str, Any],
    config: SoftSOMExperimentConfig
) -> None:
    """
    Print comprehensive experiment summary.

    Args:
        results: Experiment results dictionary
        config: Experiment configuration
    """
    logger.info("=" * 80)
    logger.info("SOFTSOM MNIST RECONSTRUCTION EXPERIMENT SUMMARY")
    logger.info("=" * 80)

    logger.info("\nModel Configuration:")
    logger.info(f"  SOM Grid Shape: {config.som_grid_shape[0]}×{config.som_grid_shape[1]} = {config.som_grid_shape[0] * config.som_grid_shape[1]} neurons")
    logger.info(f"  SOM Temperature: {config.som_temperature}")
    logger.info(f"  Latent Dimension: {config.latent_dim}")
    logger.info(f"  Convolutional Activation: {config.conv_activation}")

    logger.info("\nTraining Results:")
    logger.info(f"  Training Time: {results['training_time']:.2f} seconds")
    logger.info(f"  Final Training Loss: {results['final_train_loss']:.6f}")
    logger.info(f"  Final Validation Loss: {results['final_val_loss']:.6f}")
    logger.info(f"  Best Validation Loss: {results['best_val_loss']:.6f}")
    logger.info(f"  Total Epochs: {len(results['history']['loss'])}")

    logger.info("\nSOM Clustering Analysis:")
    clustering_stats = results['clustering_stats']
    logger.info(f"  Average Grid Position Purity: {clustering_stats['avg_purity']:.4f}")
    logger.info(f"  Total Grid Positions: {clustering_stats['grid_size']}")
    logger.info(f"  Positions with Samples: {clustering_stats['positions_with_samples']}")
    logger.info(f"  Grid Utilization: {clustering_stats['positions_with_samples']/clustering_stats['grid_size']:.2%}")

    logger.info("=" * 80)


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main() -> None:
    """
    Main execution function for running the SoftSOM MNIST reconstruction experiment.
    """
    logger.info("SoftSOM MNIST Reconstruction Experiment")
    logger.info("=" * 80)

    # Configure GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            logger.info(e)

    # Initialize experiment configuration
    config = SoftSOMExperimentConfig()

    # Log key configuration parameters
    logger.info("EXPERIMENT CONFIGURATION:")
    logger.info(f"  SOM Grid Shape: {config.som_grid_shape[0]}×{config.som_grid_shape[1]}")
    logger.info(f"  SOM Temperature: {config.som_temperature}")
    logger.info(f"  Latent Dimension: {config.latent_dim}")
    logger.info(f"  Epochs: {config.epochs}, Batch Size: {config.batch_size}")
    logger.info(f"  Architecture: Convolutional Encoder-Decoder with SoftSOM Bottleneck")
    logger.info("")

    try:
        # Run the complete experiment
        results = run_experiment(config)
        logger.info("Experiment completed successfully")

    except Exception as e:
        logger.error(f"Experiment failed with error: {e}", exc_info=True)
        raise


# ==============================================================================
# SCRIPT ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()