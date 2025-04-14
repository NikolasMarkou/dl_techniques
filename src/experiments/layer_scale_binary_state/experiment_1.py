"""
LayerScale and LearnableMultiplier Feature Selection Experiment
==============================================================

This experiment demonstrates how the LearnableMultiplier layer with BinaryPreferenceRegularizer
can act as an effective feature selector on a synthetic dataset with many irrelevant features.

The experiment:
1. Creates a synthetic dataset with 2 important features and many noise features
2. Builds a model with LearnableMultiplier layer
3. Trains the model and visualizes which features were selected
4. Compares with a model using LayerScale

NOTE: This code assumes the custom layers and regularizers are already imported.
"""

import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict, Any, Optional

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

from dl_techniques.layers.layer_scale import LearnableMultiplier, LayerScale
from dl_techniques.regularizers.binary_preference import BinaryPreferenceRegularizer


def generate_synthetic_data(
        n_samples: int = 1000,
        n_noise_features: int = 20,
        noise_level: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic classification dataset with important and noise features.

    Args:
        n_samples: Number of samples to generate
        n_noise_features: Number of irrelevant noise features to add
        noise_level: Standard deviation of noise

    Returns:
        Tuple of (X, y) with features and labels
    """
    # Generate two informative features
    X_informative = np.random.randn(n_samples, 2)

    # Generate 4 clusters (classes) in these two features
    centers = np.array([
        [-2, -2],  # Class 0
        [-2, 2],  # Class 1
        [2, -2],  # Class 2
        [2, 2]  # Class 3
    ])

    # Assign each point to nearest center
    y = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        dists = np.sum((X_informative[i] - centers) ** 2, axis=1)
        y[i] = np.argmin(dists)

    # Move points towards their assigned centers
    for i in range(n_samples):
        X_informative[i] = 0.8 * centers[y[i]] + 0.2 * X_informative[i]

    # Add Gaussian noise
    X_informative += noise_level * np.random.randn(n_samples, 2)

    # Generate noise features with no predictive power
    X_noise = np.random.randn(n_samples, n_noise_features)

    # Combine informative and noise features
    X = np.hstack([X_informative, X_noise])

    return X, y


def create_model_with_learnable_multiplier(
        input_dim: int,
        num_classes: int
) -> keras.Model:
    """
    Create a model with LearnableMultiplier layer to identify important features.

    Args:
        input_dim: Number of input features
        num_classes: Number of output classes

    Returns:
        Compiled keras model
    """
    inputs = keras.layers.Input(shape=(input_dim,))

    # Add the LearnableMultiplier layer for feature selection
    x = LearnableMultiplier(
        multiplier_type="CHANNEL",
        initializer=keras.initializers.Constant(0.5),
        regularizer=BinaryPreferenceRegularizer(),
    )(inputs)

    # Add dense layers
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=x)

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def create_model_with_layer_scale(
        input_dim: int,
        num_classes: int
) -> keras.Model:
    """
    Create a model with LayerScale layer for comparison.

    Args:
        input_dim: Number of input features
        num_classes: Number of output classes

    Returns:
        Compiled keras model
    """
    inputs = keras.layers.Input(shape=(input_dim,))

    # Add the LayerScale layer
    x = LayerScale(
        init_values=1.0,
        projection_dim=input_dim
    )(inputs)

    # Add dense layers (same as the other model)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=x)

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def run_experiment() -> Dict[str, Any]:
    """
    Run the full experiment and return results.

    Returns:
        Dictionary with experiment results
    """
    # Generate dataset with 2 important features and 20 noise features
    print("Generating synthetic dataset...")
    X, y = generate_synthetic_data(n_samples=2000, n_noise_features=20)
    n_features = X.shape[1]
    n_classes = len(np.unique(y))

    print(f"Dataset: {X.shape[0]} samples, {n_features} features ({n_features - 20} important, 20 noise)")
    print(f"Classes: {n_classes}")

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Create and train model with LearnableMultiplier
    print("\nTraining model with LearnableMultiplier...")
    model_lm = create_model_with_learnable_multiplier(n_features, n_classes)

    history_lm = model_lm.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )

    # Create and train model with LayerScale
    print("\nTraining model with LayerScale...")
    model_ls = create_model_with_layer_scale(n_features, n_classes)

    history_ls = model_ls.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )

    # Evaluate both models
    print("\nEvaluating models...")
    lm_eval = model_lm.evaluate(X_test, y_test, verbose=0)
    ls_eval = model_ls.evaluate(X_test, y_test, verbose=0)

    print(f"LearnableMultiplier Test Accuracy: {lm_eval[1]:.4f}")
    print(f"LayerScale Test Accuracy: {ls_eval[1]:.4f}")

    # Extract feature weights from LearnableMultiplier
    lm_layer = model_lm.layers[1]
    lm_weights = lm_layer.get_weights()[0].flatten()

    # Extract feature weights from LayerScale
    ls_layer = model_ls.layers[1]
    ls_weights = ls_layer.get_weights()[0]

    # Visualize the weights and feature importance
    visualize_results(lm_weights, ls_weights)

    return {
        "lm_weights": lm_weights,
        "ls_weights": ls_weights,
        "lm_accuracy": lm_eval[1],
        "ls_accuracy": ls_eval[1],
        "lm_history": history_lm.history,
        "ls_history": history_ls.history,
    }


def visualize_results(
        lm_weights: np.ndarray,
        ls_weights: np.ndarray
) -> None:
    """
    Visualize the learned feature weights.

    Args:
        lm_weights: Weights from LearnableMultiplier
        ls_weights: Weights from LayerScale
    """
    feature_names = [f'Important {i + 1}' if i < 2 else f'Noise {i - 1}' for i in range(len(lm_weights))]

    # Plotting
    plt.figure(figsize=(15, 10))

    # Plot LearnableMultiplier weights
    plt.subplot(2, 1, 1)
    bars = plt.bar(feature_names, lm_weights, color=['green' if i < 2 else 'red' for i in range(len(lm_weights))])
    plt.title('LearnableMultiplier Feature Weights')
    plt.ylabel('Weight Value')
    plt.xticks(rotation=90)
    plt.axhline(y=0.5, linestyle='--', color='black')
    plt.ylim(-0.1, 1.1)

    # Add value annotations
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.05,
                 f'{height:.2f}', ha='center', va='bottom', rotation=0)

    # Plot LayerScale weights
    plt.subplot(2, 1, 2)
    bars = plt.bar(feature_names, ls_weights, color=['green' if i < 2 else 'red' for i in range(len(ls_weights))])
    plt.title('LayerScale Feature Weights')
    plt.ylabel('Weight Value')
    plt.xticks(rotation=90)

    # Add value annotations
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.05,
                 f'{height:.2f}', ha='center', va='bottom', rotation=0)

    plt.tight_layout()
    plt.savefig('feature_weights_comparison.png')
    plt.show()

    # Plot weight distribution
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(lm_weights, bins=50, range=(-1.5, +1.5))
    plt.title('LearnableMultiplier Weight Distribution')
    plt.xlabel('Weight Value')
    plt.ylabel('Count')
    plt.axvline(x=0.5, linestyle='--', color='red')

    plt.subplot(1, 2, 2)
    plt.hist(ls_weights, bins=50, range=(-1.5, +1.5))
    plt.title('LayerScale Weight Distribution')
    plt.xlabel('Weight Value')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.savefig('weight_distributions.png')
    plt.show()

    # Highlight the important features
    important_features = list(range(2))  # First 2 features are important
    lm_important_avg = np.mean(lm_weights[important_features])
    lm_noise_avg = np.mean(lm_weights[2:])
    ls_important_avg = np.mean(ls_weights[important_features])
    ls_noise_avg = np.mean(ls_weights[2:])

    print("\nFeature Importance Analysis:")
    print(f"LearnableMultiplier - Important features avg weight: {lm_important_avg:.4f}")
    print(f"LearnableMultiplier - Noise features avg weight: {lm_noise_avg:.4f}")
    print(f"LearnableMultiplier - Important/Noise ratio: {lm_important_avg / lm_noise_avg:.4f}")
    print(f"LayerScale - Important features avg weight: {ls_important_avg:.4f}")
    print(f"LayerScale - Noise features avg weight: {ls_noise_avg:.4f}")
    print(f"LayerScale - Important/Noise ratio: {ls_important_avg / ls_noise_avg:.4f}")


def visualize_2d_data(X: np.ndarray, y: np.ndarray, lm_weights: Optional[np.ndarray] = None) -> None:
    """
    Visualize the 2D data and feature selection.

    Args:
        X: Feature matrix
        y: Class labels
        lm_weights: Optional weights from LearnableMultiplier
    """
    plt.figure(figsize=(15, 6))

    # Original data with true informative features
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7, s=30, edgecolors='k')
    plt.title('Original Data (True Important Features)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Class')

    if lm_weights is not None:
        # Select two features with highest weights
        top_features = np.argsort(lm_weights)[-2:]

        plt.subplot(1, 2, 2)
        plt.scatter(X[:, top_features[0]], X[:, top_features[1]], c=y, cmap='viridis', alpha=0.7, s=30, edgecolors='k')
        plt.title(f'LearnableMultiplier Selected Features\n(Features {top_features[0] + 1} and {top_features[1] + 1})')
        plt.xlabel(f'Feature {top_features[0] + 1} (weight={lm_weights[top_features[0]]:.2f})')
        plt.ylabel(f'Feature {top_features[1] + 1} (weight={lm_weights[top_features[1]]:.2f})')
        plt.colorbar(label='Class')

    plt.tight_layout()
    plt.savefig('data_visualization.png')
    plt.show()


if __name__ == "__main__":
    # Run the experiment
    print("Starting LayerScale and LearnableMultiplier experiment...")
    results = run_experiment()

    # Generate dataset just for visualization
    X_viz, y_viz = generate_synthetic_data(n_samples=500, n_noise_features=20)

    # Visualize 2D data and selected features
    visualize_2d_data(X_viz, y_viz, results["lm_weights"])

    print("Experiment completed. Results saved as PNG files.")