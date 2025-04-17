import os
import time
import keras
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Optional, Any
from sklearn.metrics import confusion_matrix, classification_report

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.models.som_memory import SOMModel
from dl_techniques.utils.datasets import load_and_preprocess_mnist

# ---------------------------------------------------------------------


@dataclass
class SOMExperimentConfig:
    """Configuration parameters for SOM memory experiments.

    This dataclass centralizes all configuration constants used in the SOM
    memory experiments, making it easier to modify parameters and ensuring
    type safety through type hints.
    """
    # SOM model parameters
    map_size: Tuple[int, int] = (12, 12)
    input_dim: int = 784  # 28x28 pixels
    initial_learning_rate: float = 0.5
    sigma: float = 2.0
    neighborhood_function: str = 'gaussian'

    # Training parameters
    subset_size: int = 2000
    test_subset_size: int = 100
    epochs: int = 3
    batch_size: int = 64

    # Regularization and initialization
    use_regularization: bool = True
    regularization_factor: float = 1e-5
    random_seed: int = 42

    # Visualization parameters
    fig_size_standard: Tuple[int, int] = (10, 10)
    fig_size_large: Tuple[int, int] = (12, 12)
    fig_size_wide: Tuple[int, int] = (15, 5)
    fig_size_recall: Tuple[int, int] = (15, 3)
    cmap: str = 'gray'
    heatmap_cmap: str = 'Blues'

    # Memory generalization parameters
    noise_level: float = 0.3
    occlusion_size: int = 10
    generalization_train_samples: int = 5000

    # Output parameters
    output_dir: str = 'som_memory_visualizations'
    dpi: int = 300

    # Color scheme for bar charts
    bar_colors: List[str] = field(default_factory=lambda: ['green', 'orange', 'red'])


# ---------------------------------------------------------------------


def run_experiment(config: SOMExperimentConfig = SOMExperimentConfig()) -> None:
    """
    Run a complete SOM memory experiment on the MNIST dataset.

    This experiment demonstrates how Self-Organizing Maps function as memory structures
    by organizing MNIST digits in a topological map and performing memory-based retrieval.

    Parameters
    ----------
    config : SOMExperimentConfig, optional
        Configuration parameters for the experiment.
        Defaults to the default values defined in SOMExperimentConfig.
    """
    # Create a directory for saving visualizations
    os.makedirs(config.output_dir, exist_ok=True)

    logger.info("Loading and preprocessing MNIST dataset...")
    data = load_and_preprocess_mnist()
    x_train, y_train, x_test, y_test = data.x_train, data.y_train, data.x_test, data.y_test

    # Check input dimensions and reshape if needed
    if len(x_train.shape) > 2 and x_train.shape[1] * x_train.shape[2] == config.input_dim:
        logger.info(f"Reshaping images from {x_train.shape[1:]} to {config.input_dim} dimensions")
        x_train = x_train.reshape(x_train.shape[0], -1)  # Flatten to (n_samples, 784)
        x_test = x_test.reshape(x_test.shape[0], -1)

    # Use a subset for faster training
    x_train_subset = x_train[:config.subset_size]
    y_train_subset = y_train[:config.subset_size]

    x_test_subset = x_test[:config.test_subset_size]
    y_test_subset = y_test[:config.test_subset_size]

    logger.info(f"Using {config.subset_size} training samples and {config.test_subset_size} test samples")

    # Create SOM model with customized initialization and regularization
    logger.info("Creating SOM memory model...")

    # Configure regularization
    regularizer = None
    if config.use_regularization:
        regularizer = keras.regularizers.l2(config.regularization_factor)

    som_model = SOMModel(
        map_size=config.map_size,
        input_dim=config.input_dim,
        initial_learning_rate=config.initial_learning_rate,
        sigma=config.sigma,
        neighborhood_function=config.neighborhood_function,
        weights_initializer=keras.initializers.RandomUniform(minval=0.0, maxval=1.0, seed=config.random_seed),
        regularizer=regularizer
    )

    # Compile the model
    som_model.compile(optimizer='adam')

    # Train the SOM (organize the memory space)
    logger.info("Training SOM as a memory structure...")
    start_time = time.time()
    history = som_model.train_som(
        x_train_subset,
        epochs=config.epochs,
        batch_size=config.batch_size,
        verbose=1
    )
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")

    # Fit class prototypes for classification and memory retrieval
    logger.info("Creating memory prototypes for each digit class...")
    som_model.fit_class_prototypes(x_train_subset, y_train_subset)

    # Evaluate classification performance
    logger.info("Evaluating memory recall performance...")
    y_pred = som_model.predict_classes(x_test_subset)

    # Convert one-hot encoded labels back to class indices for classification report
    if len(y_test_subset.shape) > 1 and y_test_subset.shape[1] > 1:
        logger.info("Converting one-hot encoded labels to class indices")
        y_test_indices = np.argmax(y_test_subset, axis=1)
    else:
        y_test_indices = y_test_subset

    # Print classification report (memory recall accuracy)
    logger.info("Memory Recall Performance (Classification Report):")
    report = classification_report(y_test_indices, y_pred)
    logger.info(f"\n{report}")

    # Save report to file
    report_path = os.path.join(config.output_dir, 'memory_recall_report.txt')
    save_report_to_file(
        report_path, report, config, training_time
    )

    # Visualize the confusion matrix
    if len(y_test_subset.shape) > 1 and y_test_subset.shape[1] > 1:
        y_test_indices = np.argmax(y_test_subset, axis=1)
    else:
        y_test_indices = y_test_subset

    cm = confusion_matrix(y_test_indices, y_pred)
    plt.figure(figsize=config.fig_size_standard)
    sns.heatmap(cm, annot=True, fmt='d', cmap=config.heatmap_cmap)
    plt.title('Memory Recall Confusion Matrix')
    plt.xlabel('Recalled Digit (Predicted)')
    plt.ylabel('True Digit')
    plt.savefig(
        os.path.join(config.output_dir, 'memory_confusion_matrix.png'),
        bbox_inches='tight',
        dpi=config.dpi
    )
    plt.close()

    # Visualize training error (memory organization quality)
    plt.figure(figsize=(10, 6))
    plt.plot(history['quantization_error'])
    plt.title('Memory Organization Quality During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Average Quantization Error')
    plt.grid(True)
    plt.savefig(
        os.path.join(config.output_dir, 'memory_organization_error.png'),
        bbox_inches='tight',
        dpi=config.dpi
    )
    plt.close()

    # Generate memory structure visualizations
    logger.info("Generating memory structure visualizations...")

    # 1. SOM grid - how digits are represented in the memory space
    som_model.visualize_som_grid(
        figsize=config.fig_size_large,
        save_path=os.path.join(config.output_dir, 'memory_prototypes_grid.png')
    )

    # 2. Class distribution - how digit classes are organized in memory
    som_model.visualize_class_distribution(
        x_train_subset, y_train_subset,
        figsize=config.fig_size_large,
        save_path=os.path.join(config.output_dir, 'memory_class_distribution.png')
    )

    # 3. U-Matrix - memory cluster boundaries
    som_model.visualize_u_matrix(
        figsize=config.fig_size_standard,
        save_path=os.path.join(config.output_dir, 'memory_boundaries.png')
    )

    # 4. Hit histogram - memory activation frequency
    som_model.visualize_hit_histogram(
        x_train_subset,
        figsize=config.fig_size_standard,
        log_scale=True,
        save_path=os.path.join(config.output_dir, 'memory_activation_frequency.png')
    )

    # 5. Memory recall visualization for a few test samples
    logger.info("Demonstrating memory recall for sample digits...")
    visualize_memory_recall_for_digits(
        som_model, x_test_subset, y_test_subset, x_train_subset, y_train_subset, config
    )

    # 6. Demonstrate neighborhood recall - how a test sample activates nearby memories
    logger.info("Demonstrating neighborhood memory activation...")
    visualize_neighborhood_activation(
        som_model, x_test_subset, y_test_subset, x_train_subset, y_train_subset, config
    )

    # Save the model
    model_path = os.path.join(config.output_dir, 'som_memory_model.keras')
    som_model.save(model_path)
    logger.info(f"Memory model saved to {model_path}")

    logger.info(f"Experiment complete! All memory visualizations have been saved to: {config.output_dir}")

# ---------------------------------------------------------------------


def memory_generalization_experiment(
        config: SOMExperimentConfig
) -> None:
    """
    Experiment demonstrating memory generalization properties of SOMs.

    This experiment shows how the SOM's memory generalizes to noisy
    or corrupted versions of digit images.

    Parameters
    ----------
    config : SOMExperimentConfig, optional
        Configuration parameters for the experiment.
        Defaults to the default values defined in SOMExperimentConfig.
    """
    # Ensure output directory exists
    os.makedirs(config.output_dir, exist_ok=True)

    logger.info("Loading trained SOM memory model...")

    # Try to load the model
    try:
        model_path = os.path.join(config.output_dir, 'som_memory_model.keras')
        som_model = tf.keras.models.load_model(model_path)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Could not load model. Make sure to run run_experiment first:\n{e}")
        return

    # Load MNIST data
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_mnist()

    # Check input dimensions and reshape if needed
    if len(x_train.shape) > 2 and x_train.shape[1] * x_train.shape[2] == config.input_dim:
        x_train = x_train.reshape(x_train.shape[0], -1)  # Flatten to (n_samples, 784)
        x_test = x_test.reshape(x_test.shape[0], -1)

    # Use a subset
    x_test_subset = x_test[:config.test_subset_size]
    y_test_subset = y_test[:config.test_subset_size]

    # Use a subset of training data for memory recall visualizations
    x_train_subset = x_train[:config.generalization_train_samples]
    y_train_subset = y_train[:config.generalization_train_samples]

    logger.info("Testing memory generalization with corrupted digits...")

    # Create corrupted versions and visualize results
    generalization_results = run_generalization_tests(
        som_model, x_test_subset, y_test_subset, x_train_subset, y_train_subset, config
    )

    # Create comparative visualization
    create_generalization_comparison_chart(generalization_results, config)

    logger.info(f"Memory generalization experiment complete. Results saved to {config.output_dir}")

# ---------------------------------------------------------------------


def save_report_to_file(
        report_path: str,
        report: str,
        config: SOMExperimentConfig,
        training_time: float) -> None:
    """
    Save the classification report to a file.

    Parameters
    ----------
    report_path : str
        Path to save the report.
    report : str
        Classification report text.
    config : SOMExperimentConfig
        Experiment configuration.
    training_time : float
        Time taken for training in seconds.
    """
    with open(report_path, 'w') as f:
        f.write("SOM Memory Recall Performance\n")
        f.write("===========================\n\n")
        f.write(f"Model parameters:\n")
        f.write(f"- Map size: {config.map_size}\n")
        f.write(f"- Training samples: {config.subset_size}\n")
        f.write(f"- Epochs: {config.epochs}\n")
        f.write(f"- Batch size: {config.batch_size}\n")
        f.write(f"- Initial learning rate: {config.initial_learning_rate}\n")
        f.write(f"- Sigma: {config.sigma}\n")
        f.write(f"- Neighborhood function: {config.neighborhood_function}\n")
        if config.use_regularization:
            f.write(f"- Regularization: L2 with factor {config.regularization_factor}\n")
        f.write(f"\nTraining time: {training_time:.2f} seconds\n\n")
        f.write("Classification Report:\n")
        f.write(report)

# ---------------------------------------------------------------------


def visualize_memory_recall_for_digits(
        som_model: SOMModel,
        x_test: np.ndarray,
        y_test: np.ndarray,
        x_train: np.ndarray,
        y_train: np.ndarray,
        config: SOMExperimentConfig
) -> None:
    """
    Visualize memory recall for each digit.

    Parameters
    ----------
    som_model : SOMModel
        Trained SOM model.
    x_test : np.ndarray
        Test data.
    y_test : np.ndarray
        Test labels.
    x_train : np.ndarray
        Training data for finding similar samples.
    y_train : np.ndarray
        Training labels.
    config : SOMExperimentConfig
        Experiment configuration.
    """
    # Convert one-hot encoded labels to class indices if needed
    if len(y_test.shape) > 1 and y_test.shape[1] > 1:
        y_test_indices = np.argmax(y_test, axis=1)
    else:
        y_test_indices = y_test

    # Convert training labels if needed for memory recall visualization
    if len(y_train.shape) > 1 and y_train.shape[1] > 1:
        y_train_indices = np.argmax(y_train, axis=1)
    else:
        y_train_indices = y_train

    for digit in range(10):
        # Find samples of this digit in the test set
        digit_indices = np.where(y_test_indices == digit)[0]
        digit_samples = x_test[digit_indices]

        if len(digit_samples) > 0:
            # Take the first sample
            test_sample = digit_samples[0]
            som_model.visualize_memory_recall(
                test_sample,
                n_similar=5,
                x_train=x_train,
                y_train=y_train_indices,  # Use indices instead of one-hot
                figsize=config.fig_size_recall,
                save_path=os.path.join(config.output_dir, f'memory_recall_digit_{digit}.png')
            )

# ---------------------------------------------------------------------


def visualize_neighborhood_activation(
        som_model: SOMModel,
        x_test: np.ndarray,
        y_test: np.ndarray,
        x_train: np.ndarray,
        y_train: np.ndarray,
        config: SOMExperimentConfig
) -> None:
    """
    Visualize neighborhood activation for a random test sample.

    Parameters
    ----------
    som_model : SOMModel
        Trained SOM model.
    x_test : np.ndarray
        Test data.
    y_test : np.ndarray
        Test labels.
    x_train : np.ndarray
        Training data for finding similar samples.
    y_train : np.ndarray
        Training labels.
    config : SOMExperimentConfig
        Experiment configuration.
    """
    # Convert labels to indices if they're one-hot encoded
    if len(y_test.shape) > 1 and y_test.shape[1] > 1:
        y_test_indices = np.argmax(y_test, axis=1)
    else:
        y_test_indices = y_test

    if len(y_train.shape) > 1 and y_train.shape[1] > 1:
        y_train_indices = np.argmax(y_train, axis=1)
    else:
        y_train_indices = y_train

    # Select a random test sample
    test_idx = np.random.randint(0, len(x_test))
    test_sample = x_test[test_idx]
    test_label = y_test_indices[test_idx]  # Use the index version

    # Find its BMU
    bmu_indices, _ = som_model.som_layer(test_sample.reshape(1, -1), training=False)
    bmu_index = bmu_indices[0].numpy()

    # Find training samples that map to nearby BMUs
    train_bmu_indices, _ = som_model.som_layer(x_train, training=False)
    train_bmu_indices = train_bmu_indices.numpy()

    # Calculate distances from all training BMUs to the test sample's BMU
    distances = np.sum((train_bmu_indices - bmu_index) ** 2, axis=1)

    # Get the 16 nearest neighbors
    nearest_indices = np.argsort(distances)[:16]
    nearest_samples = x_train[nearest_indices]
    nearest_labels = y_train_indices[nearest_indices]  # Use the index version

    # Visualize the test sample and its neighbors in the memory space
    plt.figure(figsize=config.fig_size_standard)

    # Test sample in the center
    plt.subplot(4, 4, 6)
    plt.imshow(test_sample.reshape(28, 28), cmap=config.cmap)
    plt.title(f"Test: {test_label}", color='red')
    plt.axis('off')

    # Surrounding neighbors
    positions = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    for i, (sample, label, pos) in enumerate(zip(nearest_samples[:15], nearest_labels[:15], positions)):
        plt.subplot(4, 4, pos)
        plt.imshow(sample.reshape(28, 28), cmap=config.cmap)
        plt.title(f"Mem: {label}")
        plt.axis('off')

    plt.suptitle("Neighborhood Memory Activation", fontsize=16)
    plt.tight_layout()
    plt.savefig(
        os.path.join(config.output_dir, 'neighborhood_memory_recall.png'),
        bbox_inches='tight',
        dpi=config.dpi
    )
    plt.close()

# ---------------------------------------------------------------------


def add_noise(image: np.ndarray, noise_level: float = 0.3) -> np.ndarray:
    """
    Add random noise to an image.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    noise_level : float, optional
        Level of noise to add. Defaults to 0.3.

    Returns
    -------
    np.ndarray
        Noisy image.
    """
    noise = np.random.normal(0, noise_level, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 1)

# ---------------------------------------------------------------------


def add_occlusion(image: np.ndarray, occlusion_size: int = 10) -> np.ndarray:
    """
    Add a square occlusion to an image.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    occlusion_size : int, optional
        Size of the occlusion square. Defaults to 10.

    Returns
    -------
    np.ndarray
        Occluded image.
    """
    occluded_image = image.copy()
    if len(image.shape) == 1:
        # Reshape to 2D
        img_size = int(np.sqrt(image.shape[0]))
        occluded_image = occluded_image.reshape(img_size, img_size)

        # Add occlusion
        x_start = np.random.randint(0, img_size - occlusion_size)
        y_start = np.random.randint(0, img_size - occlusion_size)
        occluded_image[y_start:y_start + occlusion_size, x_start:x_start + occlusion_size] = 0

        # Reshape back
        occluded_image = occluded_image.reshape(-1)
    return occluded_image

# ---------------------------------------------------------------------


def run_generalization_tests(
        som_model: SOMModel,
        x_test: np.ndarray,
        y_test: np.ndarray,
        x_train: np.ndarray,
        y_train: np.ndarray,
        config: SOMExperimentConfig
) -> Dict[str, float]:
    """
    Run generalization tests with original, noisy, and occluded images.

    Parameters
    ----------
    som_model : SOMModel
        Trained SOM model.
    x_test : np.ndarray
        Test data.
    y_test : np.ndarray
        Test labels.
    x_train : np.ndarray
        Training data for finding similar samples.
    y_train : np.ndarray
        Training labels.
    config : SOMExperimentConfig
        Experiment configuration.

    Returns
    -------
    Dict[str, float]
        Dictionary with accuracy results for different test types.
    """
    # Prepare examples for each corruption type
    example_digits = []
    for digit in range(10):
        # Find a sample of this digit in the test set
        digit_samples = x_test[y_test == digit]
        if len(digit_samples) > 0:
            example_digits.append(digit_samples[0])

    # Create corrupted versions
    noisy_digits = [add_noise(img, config.noise_level) for img in example_digits]
    occluded_digits = [add_occlusion(img, config.occlusion_size) for img in example_digits]

    # Memory recall visualization for original and corrupted digits
    for i, (original, noisy, occluded) in enumerate(zip(example_digits, noisy_digits, occluded_digits)):
        # Original digit
        plt.figure(figsize=config.fig_size_wide)

        plt.subplot(1, 3, 1)
        plt.imshow(original.reshape(28, 28), cmap=config.cmap)
        plt.title(f"Original Digit {i}")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(noisy.reshape(28, 28), cmap=config.cmap)
        plt.title(f"Noisy Digit {i}")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(occluded.reshape(28, 28), cmap=config.cmap)
        plt.title(f"Occluded Digit {i}")
        plt.axis('off')

        plt.suptitle(f"Memory Generalization Test Samples", fontsize=14)
        plt.tight_layout()
        plt.savefig(
            os.path.join(config.output_dir, f'memory_generalization_samples_{i}.png'),
            bbox_inches='tight',
            dpi=config.dpi
        )
        plt.close()

        # Show memory recall for noisy digit
        som_model.visualize_memory_recall(
            noisy,
            n_similar=5,
            x_train=x_train,
            y_train=y_train,
            figsize=config.fig_size_recall,
            save_path=os.path.join(config.output_dir, f'memory_recall_noisy_{i}.png')
        )

        # Show memory recall for occluded digit
        som_model.visualize_memory_recall(
            occluded,
            n_similar=5,
            x_train=x_train,
            y_train=y_train,
            figsize=config.fig_size_recall,
            save_path=os.path.join(config.output_dir, f'memory_recall_occluded_{i}.png')
        )

    # Test classification accuracy on corrupted digits
    # Prepare larger test sets with corruption
    n_test = len(x_test)
    noisy_test = np.array([add_noise(img, config.noise_level) for img in x_test])
    occluded_test = np.array([add_occlusion(img, config.occlusion_size) for img in x_test])

    # Classification on original test data
    original_pred = som_model.predict_classes(x_test)
    if len(y_test.shape) > 1 and y_test.shape[1] > 1:
        y_test_indices = np.argmax(y_test, axis=1)
    else:
        y_test_indices = y_test

    original_acc = np.mean(original_pred == y_test_indices)

    # Classification on noisy test data
    noisy_pred = som_model.predict_classes(noisy_test)
    noisy_acc = np.mean(noisy_pred == y_test)

    # Classification on occluded test data
    occluded_pred = som_model.predict_classes(occluded_test)
    occluded_acc = np.mean(occluded_pred == y_test)

    # Report results
    logger.info("\nMemory Generalization Results:")
    logger.info(f"Original data accuracy: {original_acc:.4f}")
    logger.info(f"Noisy data accuracy: {noisy_acc:.4f}")
    logger.info(f"Occluded data accuracy: {occluded_acc:.4f}")

    # Save results to file
    with open(os.path.join(config.output_dir, 'memory_generalization_results.txt'), 'w') as f:
        f.write("SOM Memory Generalization Results\n")
        f.write("================================\n\n")
        f.write(f"Original data accuracy: {original_acc:.4f}\n")
        f.write(f"Noisy data accuracy: {noisy_acc:.4f}\n")
        f.write(f"Occluded data accuracy: {occluded_acc:.4f}\n\n")
        f.write(f"Noise level: {config.noise_level}\n")
        f.write(f"Occlusion size: {config.occlusion_size}x{config.occlusion_size} pixels\n")

    return {
        'original': original_acc,
        'noisy': noisy_acc,
        'occluded': occluded_acc
    }

# ---------------------------------------------------------------------

def create_generalization_comparison_chart(
        results: Dict[str, float],
        config: SOMExperimentConfig
) -> None:
    """
    Create a bar chart comparing generalization results.

    Parameters
    ----------
    results : Dict[str, float]
        Dictionary with accuracy results for different test types.
    config : SOMExperimentConfig
        Experiment configuration.
    """
    plt.figure(figsize=(10, 6))
    acc_data = [results['original'], results['noisy'], results['occluded']]
    labels = ['Original', 'Noisy', 'Occluded']

    plt.bar(labels, acc_data, color=config.bar_colors)
    plt.title('Memory Recall Performance with Corrupted Inputs')
    plt.xlabel('Input Type')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)

    for i, v in enumerate(acc_data):
        plt.text(i, v + 0.02, f'{v:.4f}', ha='center')

    plt.savefig(
        os.path.join(config.output_dir, 'memory_generalization_comparison.png'),
        bbox_inches='tight',
        dpi=config.dpi
    )
    plt.close()

# ---------------------------------------------------------------------


if __name__ == "__main__":
    # Create custom configuration
    config = SOMExperimentConfig()

    # Set random seeds for reproducibility
    np.random.seed(config.random_seed)
    tf.random.set_seed(config.random_seed)

    # Run the main SOM memory experiment
    run_experiment(config)

    # Run the memory generalization experiment
    memory_generalization_experiment(config)

# ---------------------------------------------------------------------
