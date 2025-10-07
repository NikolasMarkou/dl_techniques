"""
SOM Memory Experiment on MNIST
===============================

This experiment demonstrates how Self-Organizing Maps (SOMs) function as memory
structures by organizing MNIST digits in a topological map and performing
memory-based retrieval. The experiment explores the SOM's ability to create
associative memories and generalize to corrupted or noisy inputs.

Hypothesis:
-----------
The hypothesis is that SOMs create topologically-organized memory structures
where similar patterns are stored in nearby locations. This organization
enables effective pattern recall even from partial or corrupted inputs,
demonstrating content-addressable memory properties.

Experimental Design:
--------------------
- **Dataset**: MNIST (28×28 grayscale images, 10 classes), using standardized
  dataset builder with proper preprocessing.

- **SOM Architecture**: A 2D grid of neurons (default 12×12) that self-organize
  to represent the input space:
    - Each neuron stores a prototype vector matching input dimensionality (784)
    - Neighborhood functions for topological learning
    - Configurable learning rates and sigma values

- **Memory Operations**:
    1. **Organization**: Training creates topological memory map
    2. **Recall**: Retrieval of stored patterns from queries
    3. **Generalization**: Recall from noisy or occluded inputs
    4. **Neighborhood Activation**: Examining nearby memory activations

- **Analysis**: Comprehensive visualization of memory structure, recall
  performance, and generalization capabilities.

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
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional

from dl_techniques.utils.logger import logger
from dl_techniques.models.som.model import SOMModel

from dl_techniques.datasets.vision.common import (
    create_dataset_builder,
)

# Visualization framework
from dl_techniques.visualization import (
    VisualizationManager,
    VisualizationPlugin,
    TrainingHistory,
    ClassificationResults,
    PlotConfig,
    PlotStyle,
    ColorScheme,
    ConfusionMatrixVisualization,
)

# Model analyzer
from dl_techniques.analyzer import (
    ModelAnalyzer,
    AnalysisConfig,
    DataInput
)


# ==============================================================================
# CUSTOM DATA STRUCTURES
# ==============================================================================

@dataclass
class MemoryGeneralizationData:
    """
    Data structure for memory generalization results.

    Attributes:
        test_types: Names of test types (original, noisy, occluded)
        accuracies: Accuracy for each test type
        noise_level: Level of noise applied
        occlusion_size: Size of occlusion square
    """
    test_types: List[str]
    accuracies: List[float]
    noise_level: float
    occlusion_size: int


# ==============================================================================
# CUSTOM VISUALIZATION PLUGINS
# ==============================================================================

class MemoryGeneralizationVisualization(VisualizationPlugin):
    """Visualization plugin for memory generalization comparison."""

    @property
    def name(self) -> str:
        return "memory_generalization"

    @property
    def description(self) -> str:
        return "Visualizes memory recall performance with corrupted inputs"

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, MemoryGeneralizationData)

    def create_visualization(
            self,
            data: MemoryGeneralizationData,
            ax: Optional[Any] = None,
            **kwargs
    ) -> Any:
        """Create memory generalization comparison bar chart."""
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=self.config.fig_size, dpi=self.config.dpi)
        else:
            fig = ax.get_figure()

        # Define colors for different test types
        colors = [
            self.config.color_scheme.primary,  # Original
            self.config.color_scheme.secondary,  # Noisy
            self.config.color_scheme.accent  # Occluded
        ]

        bars = ax.bar(data.test_types, data.accuracies, color=colors, alpha=0.7, edgecolor='black')

        ax.set_title(
            'Memory Recall Performance with Corrupted Inputs',
            fontsize=self.config.title_fontsize
        )
        ax.set_xlabel('Input Type', fontsize=self.config.label_fontsize)
        ax.set_ylabel('Accuracy', fontsize=self.config.label_fontsize)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, acc in zip(bars, data.accuracies):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height + 0.02,
                f'{acc:.4f}',
                ha='center', va='bottom',
                fontsize=10
            )

        # Add corruption parameters as text
        ax.text(
            0.02, 0.98,
            f'Noise level: {data.noise_level}\nOcclusion: {data.occlusion_size}×{data.occlusion_size}px',
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
            fontsize=9
        )

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
    batch_size: int = 64
    epochs: int = 3


@dataclass
class SOMExperimentConfig:
    """
    Configuration for the SOM memory experiment.

    This class encapsulates all configurable parameters for the experiment,
    including SOM parameters, training settings, memory generalization tests,
    and analysis configuration.
    """

    # --- Dataset Configuration ---
    dataset_name: str = "mnist"
    num_classes: int = 10
    input_shape: Tuple[int, ...] = (28, 28, 1)
    input_dim: int = 784  # Flattened input dimension
    use_rgb: bool = False  # Keep MNIST in grayscale

    # --- SOM Model Parameters ---
    map_size: Tuple[int, int] = (12, 12)
    initial_learning_rate: float = 0.5
    sigma: float = 2.0
    neighborhood_function: str = 'gaussian'

    # --- Training Parameters ---
    epochs: int = 3
    batch_size: int = 64
    subset_size: int = 2000
    test_subset_size: int = 100

    # --- Regularization ---
    use_regularization: bool = True
    regularization_factor: float = 1e-5

    # --- Memory Generalization Parameters ---
    noise_level: float = 0.3
    occlusion_size: int = 10
    generalization_train_samples: int = 5000

    # --- Experiment Configuration ---
    output_dir: Path = Path("results")
    experiment_name: str = "som_memory_experiment"
    random_seed: int = 42

    # --- Analysis Configuration ---
    analyzer_config: AnalysisConfig = field(default_factory=lambda: AnalysisConfig(
        analyze_weights=True,
        analyze_calibration=False,  # Not applicable for SOM
        analyze_information_flow=False,  # Not applicable for SOM
        save_plots=True,
        plot_style='publication',
    ))


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def add_noise(image: np.ndarray, noise_level: float = 0.3) -> np.ndarray:
    """
    Add random Gaussian noise to an image.

    Args:
        image: Input image
        noise_level: Standard deviation of Gaussian noise

    Returns:
        Noisy image clipped to [0, 1]
    """
    noise = np.random.normal(0, noise_level, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 1)


def add_occlusion(image: np.ndarray, occlusion_size: int = 10) -> np.ndarray:
    """
    Add a random square occlusion to an image.

    Args:
        image: Input image (flattened)
        occlusion_size: Size of the occlusion square

    Returns:
        Occluded image
    """
    occluded_image = image.copy()

    if len(image.shape) == 1:
        # Reshape to 2D
        img_size = int(np.sqrt(image.shape[0]))
        occluded_image = occluded_image.reshape(img_size, img_size)

        # Add random occlusion
        x_start = np.random.randint(0, img_size - occlusion_size)
        y_start = np.random.randint(0, img_size - occlusion_size)
        occluded_image[y_start:y_start + occlusion_size, x_start:x_start + occlusion_size] = 0

        # Reshape back to flat
        occluded_image = occluded_image.reshape(-1)

    return occluded_image


# ==============================================================================
# MAIN EXPERIMENT RUNNER
# ==============================================================================

def run_experiment(config: SOMExperimentConfig) -> Dict[str, Any]:
    """
    Run the complete SOM memory experiment.

    This function orchestrates the entire experimental pipeline, including:
    1. Dataset loading and preprocessing using standardized builder
    2. SOM training (memory organization)
    3. Memory recall evaluation
    4. Visualization generation using framework
    5. Memory generalization tests
    6. Results compilation and reporting

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
    viz_manager.register_template("confusion_matrix", ConfusionMatrixVisualization)
    viz_manager.register_template("memory_generalization", MemoryGeneralizationVisualization)

    # Log experiment start
    logger.info("Starting SOM Memory Experiment")
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
    class_names = dataset_builder.get_class_names()

    logger.info("Dataset loaded successfully")
    logger.info(f"Test data shape: {test_data.x_data.shape}")

    # Flatten images for SOM
    x_train_flat = test_data.x_data[:config.subset_size].reshape(-1, config.input_dim)
    y_train_subset = test_data.y_data[:config.subset_size].astype(int)

    x_test_flat = test_data.x_data[config.subset_size:config.subset_size + config.test_subset_size].reshape(-1,
                                                                                                            config.input_dim)
    y_test_subset = test_data.y_data[config.subset_size:config.subset_size + config.test_subset_size].astype(int)

    logger.info(f"Using {config.subset_size} training samples and {config.test_subset_size} test samples")

    # ===== SOM MODEL CREATION =====
    logger.info("Creating SOM memory model...")

    # Configure regularization
    regularizer = None
    if config.use_regularization:
        regularizer = keras.regularizers.L2(config.regularization_factor)

    som_model = SOMModel(
        map_size=config.map_size,
        input_dim=config.input_dim,
        initial_learning_rate=config.initial_learning_rate,
        sigma=config.sigma,
        neighborhood_function=config.neighborhood_function,
        weights_initializer=keras.initializers.RandomUniform(
            minval=0.0, maxval=1.0, seed=config.random_seed
        ),
        regularizer=regularizer
    )

    # Compile the model
    som_model.compile(optimizer='adam')

    # ===== SOM TRAINING (MEMORY ORGANIZATION) =====
    logger.info("Training SOM as a memory structure...")
    import time
    start_time = time.time()

    history = som_model.train_som(
        x_train_flat,
        epochs=config.epochs,
        batch_size=config.batch_size,
        verbose=1
    )

    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")

    # ===== MEMORY PROTOTYPE FITTING =====
    logger.info("Creating memory prototypes for each digit class...")
    som_model.fit_class_prototypes(x_train_flat, y_train_subset)

    # ===== MEMORY RECALL EVALUATION =====
    logger.info("Evaluating memory recall performance...")
    y_pred = som_model.predict_classes(x_test_flat)

    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test_subset)
    logger.info(f"Memory recall accuracy: {accuracy:.4f}")

    # ===== VISUALIZATION GENERATION =====
    logger.info("Generating memory structure visualizations...")

    # 1. Confusion matrix
    classification_results = ClassificationResults(
        y_true=y_test_subset,
        y_pred=y_pred,
        class_names=class_names,
        model_name="SOM Memory"
    )

    viz_manager.visualize(
        data=classification_results,
        plugin_name="confusion_matrix",
        normalize='true',
        show=False
    )

    # 2. Training history (quantization error)
    training_history = TrainingHistory(
        epochs=list(range(len(history['quantization_error']))),
        train_loss=history['quantization_error'],
        val_loss=[],
        train_metrics={},
        val_metrics={}
    )

    # Plot quantization error manually (not using TrainingCurvesVisualization as it expects val_loss)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6), dpi=viz_config.dpi)
    ax.plot(training_history.epochs, training_history.train_loss,
            color=viz_config.color_scheme.primary, linewidth=2)
    ax.set_title('Memory Organization Quality During Training', fontsize=viz_config.title_fontsize)
    ax.set_xlabel('Epoch', fontsize=viz_config.label_fontsize)
    ax.set_ylabel('Average Quantization Error', fontsize=viz_config.label_fontsize)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(experiment_dir / "visualizations" / "memory_organization_error.png",
                dpi=viz_config.dpi, bbox_inches='tight')
    plt.close()

    # 3. SOM-specific visualizations
    som_viz_dir = experiment_dir / "som_visualizations"
    som_viz_dir.mkdir(exist_ok=True)

    # SOM grid
    som_model.visualize_som_grid(
        figsize=(12, 12),
        save_path=str(som_viz_dir / 'memory_prototypes_grid.png')
    )

    # Class distribution
    som_model.visualize_class_distribution(
        x_train_flat, y_train_subset,
        figsize=(12, 12),
        save_path=str(som_viz_dir / 'memory_class_distribution.png')
    )

    # U-Matrix
    som_model.visualize_u_matrix(
        figsize=(10, 10),
        save_path=str(som_viz_dir / 'memory_boundaries.png')
    )

    # Hit histogram
    som_model.visualize_hit_histogram(
        x_train_flat,
        figsize=(10, 10),
        log_scale=True,
        save_path=str(som_viz_dir / 'memory_activation_frequency.png')
    )

    # 4. Memory recall visualizations for each digit
    logger.info("Generating memory recall visualizations for sample digits...")
    for digit in range(10):
        digit_indices = np.where(y_test_subset == digit)[0]
        if len(digit_indices) > 0:
            test_sample = x_test_flat[digit_indices[0]]
            som_model.visualize_memory_recall(
                test_sample,
                n_similar=5,
                x_train=x_train_flat,
                y_train=y_train_subset,
                figsize=(15, 3),
                save_path=str(som_viz_dir / f'memory_recall_digit_{digit}.png')
            )

    # 5. Neighborhood activation visualization
    logger.info("Demonstrating neighborhood memory activation...")
    visualize_neighborhood_activation(
        som_model, x_test_flat, y_test_subset,
        x_train_flat, y_train_subset, som_viz_dir
    )

    # ===== MEMORY GENERALIZATION EXPERIMENT =====
    logger.info("Testing memory generalization with corrupted inputs...")

    generalization_results = run_generalization_tests(
        som_model, x_test_flat, y_test_subset,
        x_train_flat, y_train_subset,
        config, som_viz_dir
    )

    # Visualize generalization results
    gen_data = MemoryGeneralizationData(
        test_types=['Original', 'Noisy', 'Occluded'],
        accuracies=[
            generalization_results['original'],
            generalization_results['noisy'],
            generalization_results['occluded']
        ],
        noise_level=config.noise_level,
        occlusion_size=config.occlusion_size
    )

    viz_manager.visualize(
        data=gen_data,
        plugin_name="memory_generalization",
        show=False
    )

    # ===== SAVE MODEL =====
    model_path = experiment_dir / "models" / "som_memory_model.keras"
    model_path.parent.mkdir(exist_ok=True)
    som_model.save(str(model_path))
    logger.info(f"Memory model saved to {model_path}")

    # ===== RESULTS COMPILATION =====
    results_payload = {
        'memory_recall_accuracy': accuracy,
        'generalization_results': generalization_results,
        'training_time': training_time,
        'history': history,
        'som_model': som_model,
        'test_predictions': y_pred,
        'test_labels': y_test_subset
    }

    # Save and print results
    save_experiment_results(results_payload, experiment_dir, config)
    print_experiment_summary(results_payload, config)

    return results_payload


# ==============================================================================
# GENERALIZATION TESTS
# ==============================================================================

def run_generalization_tests(
        som_model: SOMModel,
        x_test: np.ndarray,
        y_test: np.ndarray,
        x_train: np.ndarray,
        y_train: np.ndarray,
        config: SOMExperimentConfig,
        output_dir: Path
) -> Dict[str, float]:
    """
    Run memory generalization tests with corrupted inputs.

    Args:
        som_model: Trained SOM model
        x_test: Test data
        y_test: Test labels
        x_train: Training data for memory recall visualization
        y_train: Training labels
        config: Experiment configuration
        output_dir: Directory to save visualizations

    Returns:
        Dictionary with accuracy results for different corruption types
    """
    import matplotlib.pyplot as plt

    # Prepare example digits for each class
    example_digits = []
    for digit in range(10):
        digit_indices = np.where(y_test == digit)[0]
        if len(digit_indices) > 0:
            example_digits.append(x_test[digit_indices[0]])

    # Create corrupted versions
    noisy_digits = [add_noise(img, config.noise_level) for img in example_digits]
    occluded_digits = [add_occlusion(img, config.occlusion_size) for img in example_digits]

    # Visualize corruption examples
    for i, (original, noisy, occluded) in enumerate(zip(example_digits, noisy_digits, occluded_digits)):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(original.reshape(28, 28), cmap='gray')
        axes[0].set_title(f'Original Digit {i}')
        axes[0].axis('off')

        axes[1].imshow(noisy.reshape(28, 28), cmap='gray')
        axes[1].set_title(f'Noisy Digit {i}')
        axes[1].axis('off')

        axes[2].imshow(occluded.reshape(28, 28), cmap='gray')
        axes[2].set_title(f'Occluded Digit {i}')
        axes[2].axis('off')

        plt.suptitle('Memory Generalization Test Samples', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / f'memory_generalization_samples_{i}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        # Show memory recall for corrupted inputs
        som_model.visualize_memory_recall(
            noisy, n_similar=5, x_train=x_train, y_train=y_train,
            figsize=(15, 3), save_path=str(output_dir / f'memory_recall_noisy_{i}.png')
        )

        som_model.visualize_memory_recall(
            occluded, n_similar=5, x_train=x_train, y_train=y_train,
            figsize=(15, 3), save_path=str(output_dir / f'memory_recall_occluded_{i}.png')
        )

    # Test classification accuracy on corrupted test set
    noisy_test = np.array([add_noise(img, config.noise_level) for img in x_test])
    occluded_test = np.array([add_occlusion(img, config.occlusion_size) for img in x_test])

    # Classification accuracy
    original_pred = som_model.predict_classes(x_test)
    original_acc = float(np.mean(original_pred == y_test))

    noisy_pred = som_model.predict_classes(noisy_test)
    noisy_acc = float(np.mean(noisy_pred == y_test))

    occluded_pred = som_model.predict_classes(occluded_test)
    occluded_acc = float(np.mean(occluded_pred == y_test))

    logger.info("Memory Generalization Results:")
    logger.info(f"  Original data accuracy: {original_acc:.4f}")
    logger.info(f"  Noisy data accuracy: {noisy_acc:.4f}")
    logger.info(f"  Occluded data accuracy: {occluded_acc:.4f}")

    return {
        'original': original_acc,
        'noisy': noisy_acc,
        'occluded': occluded_acc
    }


def visualize_neighborhood_activation(
        som_model: SOMModel,
        x_test: np.ndarray,
        y_test: np.ndarray,
        x_train: np.ndarray,
        y_train: np.ndarray,
        output_dir: Path
) -> None:
    """
    Visualize neighborhood activation for a random test sample.

    Args:
        som_model: Trained SOM model
        x_test: Test data
        y_test: Test labels
        x_train: Training data
        y_train: Training labels
        output_dir: Directory to save visualization
    """
    import matplotlib.pyplot as plt

    # Select a random test sample
    test_idx = np.random.randint(0, len(x_test))
    test_sample = x_test[test_idx]
    test_label = y_test[test_idx]

    # Find its BMU
    bmu_indices, _ = som_model.som_layer(test_sample.reshape(1, -1), training=False)
    bmu_index = bmu_indices[0].numpy()

    # Find training samples mapping to nearby BMUs
    train_bmu_indices, _ = som_model.som_layer(x_train, training=False)
    train_bmu_indices = train_bmu_indices.numpy()

    # Calculate distances from training BMUs to test sample's BMU
    distances = np.sum((train_bmu_indices - bmu_index) ** 2, axis=1)

    # Get 16 nearest neighbors
    nearest_indices = np.argsort(distances)[:16]
    nearest_samples = x_train[nearest_indices]
    nearest_labels = y_train[nearest_indices]

    # Visualize test sample and neighbors
    fig = plt.figure(figsize=(10, 10))

    # Test sample in center
    plt.subplot(4, 4, 6)
    plt.imshow(test_sample.reshape(28, 28), cmap='gray')
    plt.title(f'Test: {test_label}', color='red')
    plt.axis('off')

    # Surrounding neighbors
    positions = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    for i, (sample, label, pos) in enumerate(zip(nearest_samples[:15], nearest_labels[:15], positions)):
        plt.subplot(4, 4, pos)
        plt.imshow(sample.reshape(28, 28), cmap='gray')
        plt.title(f'Mem: {label}')
        plt.axis('off')

    plt.suptitle('Neighborhood Memory Activation', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / 'neighborhood_memory_recall.png',
                dpi=300, bbox_inches='tight')
    plt.close()


# ==============================================================================
# RESULTS SAVING AND REPORTING
# ==============================================================================

def save_experiment_results(
        results: Dict[str, Any],
        experiment_dir: Path,
        config: SOMExperimentConfig
) -> None:
    """
    Save experiment results in multiple formats.

    Args:
        results: Experiment results dictionary
        experiment_dir: Directory to save results
        config: Experiment configuration
    """
    try:
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [convert_numpy(i) for i in obj]
            return obj

        # Save configuration
        config_dict = {
            'experiment_name': config.experiment_name,
            'map_size': config.map_size,
            'input_dim': config.input_dim,
            'epochs': config.epochs,
            'batch_size': config.batch_size,
            'subset_size': config.subset_size,
            'initial_learning_rate': config.initial_learning_rate,
            'sigma': config.sigma,
            'neighborhood_function': config.neighborhood_function,
            'use_regularization': config.use_regularization,
            'regularization_factor': config.regularization_factor,
            'random_seed': config.random_seed
        }
        with open(experiment_dir / "experiment_config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)

        # Save performance results
        perf_dict = {
            'memory_recall_accuracy': convert_numpy(results['memory_recall_accuracy']),
            'training_time': convert_numpy(results['training_time']),
            'generalization_results': convert_numpy(results['generalization_results'])
        }
        with open(experiment_dir / "performance_results.json", 'w') as f:
            json.dump(perf_dict, f, indent=2)

        # Save detailed report
        with open(experiment_dir / "memory_recall_report.txt", 'w') as f:
            f.write("SOM Memory Recall Performance\n")
            f.write("============================\n\n")
            f.write("Model Parameters:\n")
            f.write(f"  Map size: {config.map_size}\n")
            f.write(f"  Training samples: {config.subset_size}\n")
            f.write(f"  Epochs: {config.epochs}\n")
            f.write(f"  Batch size: {config.batch_size}\n")
            f.write(f"  Initial learning rate: {config.initial_learning_rate}\n")
            f.write(f"  Sigma: {config.sigma}\n")
            f.write(f"  Neighborhood function: {config.neighborhood_function}\n")
            if config.use_regularization:
                f.write(f"  Regularization: L2 with factor {config.regularization_factor}\n")
            f.write(f"\nTraining time: {results['training_time']:.2f} seconds\n")
            f.write(f"\nMemory Recall Accuracy: {results['memory_recall_accuracy']:.4f}\n")
            f.write("\nGeneralization Results:\n")
            f.write(f"  Original: {results['generalization_results']['original']:.4f}\n")
            f.write(f"  Noisy: {results['generalization_results']['noisy']:.4f}\n")
            f.write(f"  Occluded: {results['generalization_results']['occluded']:.4f}\n")

        logger.info("Experiment results saved successfully")

    except Exception as e:
        logger.error(f"Failed to save experiment results: {e}", exc_info=True)


def print_experiment_summary(
        results: Dict[str, Any],
        config: SOMExperimentConfig
) -> None:
    """
    Print comprehensive experiment summary.

    Args:
        results: Experiment results dictionary
        config: Experiment configuration
    """
    logger.info("=" * 80)
    logger.info("SOM MEMORY EXPERIMENT SUMMARY")
    logger.info("=" * 80)

    logger.info("\nModel Configuration:")
    logger.info(
        f"  SOM Map Size: {config.map_size[0]}×{config.map_size[1]} = {config.map_size[0] * config.map_size[1]} neurons")
    logger.info(f"  Training Samples: {config.subset_size}")
    logger.info(f"  Epochs: {config.epochs}")
    logger.info(f"  Training Time: {results['training_time']:.2f} seconds")

    logger.info("\nMemory Recall Performance:")
    logger.info(f"  Test Accuracy: {results['memory_recall_accuracy']:.4f}")

    logger.info("\nMemory Generalization Performance:")
    gen_results = results['generalization_results']
    logger.info(f"  Original Images: {gen_results['original']:.4f}")
    logger.info(f"  Noisy Images (noise={config.noise_level}): {gen_results['noisy']:.4f}")
    logger.info(
        f"  Occluded Images (size={config.occlusion_size}×{config.occlusion_size}): {gen_results['occluded']:.4f}")

    # Calculate robustness metrics
    noise_robustness = gen_results['noisy'] / gen_results['original']
    occlusion_robustness = gen_results['occluded'] / gen_results['original']

    logger.info("\nMemory Robustness Analysis:")
    logger.info(f"  Noise Robustness: {noise_robustness:.2%} of original performance retained")
    logger.info(f"  Occlusion Robustness: {occlusion_robustness:.2%} of original performance retained")

    logger.info("=" * 80)


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main() -> None:
    """
    Main execution function for running the SOM memory experiment.
    """
    logger.info("SOM Memory Experiment")
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
    config = SOMExperimentConfig()

    # Log key configuration parameters
    logger.info("EXPERIMENT CONFIGURATION:")
    logger.info(f"  SOM Map Size: {config.map_size[0]}×{config.map_size[1]}")
    logger.info(f"  Training Subset: {config.subset_size} samples")
    logger.info(f"  Epochs: {config.epochs}, Batch Size: {config.batch_size}")
    logger.info(
        f"  Generalization Tests: Noise={config.noise_level}, Occlusion={config.occlusion_size}×{config.occlusion_size}")
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