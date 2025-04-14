"""
Complete LayerScale and LearnableMultiplier Feature Selection Experiment
=======================================================================

This experiment demonstrates how the LearnableMultiplier layer with BinaryPreferenceRegularizer
and the LayerScale layer can act as effective feature selectors on a synthetic dataset
with many irrelevant features. The experiment includes visualization of decision boundaries.

The experiment:
1. Creates a synthetic dataset with 2 important features and many noise features
2. Builds models with LearnableMultiplier and LayerScale layers
3. Trains the models and visualizes which features were selected
4. Visualizes decision boundaries in different feature spaces

Note: This code assumes you have already imported the following custom layers:
- LayerScale
- LearnableMultiplier
- BinaryPreferenceRegularizer
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

# Uncomment these lines in your actual implementation:
from dl_techniques.layers.layer_scale import LearnableMultiplier, LayerScale
from dl_techniques.regularizers.binary_preference import BinaryPreferenceRegularizer


def generate_synthetic_data(
        n_samples: int = 1000,
        n_noise_features: int = 20,
        noise_level: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a synthetic classification dataset with important and noise features.

    Args:
        n_samples: Number of samples to generate
        n_noise_features: Number of irrelevant noise features to add
        noise_level: Standard deviation of noise

    Returns:
        Tuple of (X, y, centers) with features, labels, and class centers
    """
    # Generate two informative features
    X_informative = np.random.randn(n_samples, 2)

    # Generate 4 clusters (classes) in these two features
    centers = np.array([
        [-2, -2],  # Class 0
        [-2, 2],   # Class 1
        [2, -2],   # Class 2
        [2, 2]     # Class 3
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

    return X, y, centers


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
    x = keras.layers.Dense(64, activation='relu')(x)
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
        init_values=0.1,  # Start with small values to learn important features
        projection_dim=input_dim
    )(inputs)

    # Add dense layers (same as the other model)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=x)

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def visualize_training_history(
        history_lm: Dict,
        history_ls: Dict
) -> None:
    """
    Visualize the training history of both models.

    Args:
        history_lm: Training history for LearnableMultiplier model
        history_ls: Training history for LayerScale model
    """
    plt.figure(figsize=(15, 5))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history_lm['accuracy'], label='LM Train')
    plt.plot(history_lm['val_accuracy'], label='LM Validation')
    plt.plot(history_ls['accuracy'], label='LS Train', linestyle='--')
    plt.plot(history_ls['val_accuracy'], label='LS Validation', linestyle='--')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(alpha=0.3)

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history_lm['loss'], label='LM Train')
    plt.plot(history_lm['val_loss'], label='LM Validation')
    plt.plot(history_ls['loss'], label='LS Train', linestyle='--')
    plt.plot(history_ls['val_loss'], label='LS Validation', linestyle='--')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300)
    plt.show()


def visualize_feature_weights(
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
    plt.title('LearnableMultiplier Feature Weights', fontsize=14)
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
    plt.title('LayerScale Feature Weights', fontsize=14)
    plt.ylabel('Weight Value')
    plt.xticks(rotation=90)

    # Add value annotations
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.05,
                 f'{height:.2f}', ha='center', va='bottom', rotation=0)

    plt.tight_layout()
    plt.savefig('feature_weights_comparison.png', dpi=300)
    plt.show()

    # Plot weight distribution
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(lm_weights, bins=20, range=(0, 1))
    plt.title('LearnableMultiplier Weight Distribution')
    plt.xlabel('Weight Value')
    plt.ylabel('Count')
    plt.axvline(x=0.5, linestyle='--', color='red')
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.hist(ls_weights, bins=20)
    plt.title('LayerScale Weight Distribution')
    plt.xlabel('Weight Value')
    plt.ylabel('Count')
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('weight_distributions.png', dpi=300)
    plt.show()

    # Print feature importance statistics
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


def visualize_single_boundary(
        X: np.ndarray,
        y: np.ndarray,
        feature_indices: Tuple[int, int],
        title: str,
        ax: plt.Axes,
        model: Optional[keras.Model] = None,
        centers: Optional[np.ndarray] = None
) -> None:
    """
    Visualize decision boundary for a single pair of features.

    Args:
        X: Feature matrix
        y: Class labels
        feature_indices: Tuple of (feature1_idx, feature2_idx)
        title: Plot title
        ax: Matplotlib axes to plot on
        model: Optional model to generate decision boundaries
        centers: Optional class centers to plot
    """
    # Extract the two features we want to visualize
    X_reduced = X[:, feature_indices]

    # Create a meshgrid for the feature space
    h = 0.1  # Step size
    x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
    y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Plot decision boundaries if model is provided
    if model is not None:
        # Create a grid of points
        grid = np.c_[xx.ravel(), yy.ravel()]

        # Pad with zeros for other features
        if X.shape[1] > 2:
            grid_full = np.zeros((grid.shape[0], X.shape[1]))
            grid_full[:, feature_indices[0]] = grid[:, 0]
            grid_full[:, feature_indices[1]] = grid[:, 1]
        else:
            grid_full = grid

        # Get predictions
        Z = model.predict(grid_full, verbose=0)
        Z = np.argmax(Z, axis=1)

        # Reshape to match grid
        Z = Z.reshape(xx.shape)

        # Plot decision boundaries
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')

    # Plot class centers if provided
    if centers is not None and feature_indices[0] < 2 and feature_indices[1] < 2:
        ax.scatter(centers[:, 0], centers[:, 1],
                   marker='X', s=200, c='red',
                   edgecolors='white', linewidths=2,
                   label='Class Centers')

    # Plot the data points
    scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1],
                         c=y, cmap='viridis', s=30, alpha=0.8,
                         edgecolors='k')

    # Add legend and labels
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(f'Feature {feature_indices[0]+1}')
    ax.set_ylabel(f'Feature {feature_indices[1]+1}')
    ax.legend(*scatter.legend_elements(), title='Classes')

    # Add grid for readability
    ax.grid(alpha=0.3)


def visualize_decision_boundaries(
        X: np.ndarray,
        y: np.ndarray,
        model_lm: keras.Model,
        model_ls: keras.Model,
        lm_weights: np.ndarray,
        ls_weights: np.ndarray,
        centers: np.ndarray
) -> None:
    """
    Visualize decision boundaries in the feature space.

    Args:
        X: Feature matrix
        y: Class labels
        model_lm: Trained model with LearnableMultiplier
        model_ls: Trained model with LayerScale
        lm_weights: Weights from LearnableMultiplier
        ls_weights: Weights from LayerScale
        centers: Class centers in the original 2D space
    """
    # Create figure with 3 subplots
    plt.figure(figsize=(18, 6))

    # 1. True feature space (first 2 features)
    ax1 = plt.subplot(1, 3, 1)
    visualize_single_boundary(X, y, (0, 1), "True Feature Space (Features 1 & 2)", ax1, centers=centers)

    # 2. LearnableMultiplier top features
    lm_top_features = np.argsort(lm_weights)[-2:]
    ax2 = plt.subplot(1, 3, 2)
    visualize_single_boundary(
        X, y, lm_top_features,
        f"LearnableMultiplier Selected Features\n({lm_top_features[0]+1} & {lm_top_features[1]+1})",
        ax2, model=model_lm
    )

    # 3. LayerScale top features
    ls_top_features = np.argsort(ls_weights)[-2:]
    ax3 = plt.subplot(1, 3, 3)
    visualize_single_boundary(
        X, y, ls_top_features,
        f"LayerScale Selected Features\n({ls_top_features[0]+1} & {ls_top_features[1]+1})",
        ax3, model=model_ls
    )

    plt.tight_layout()
    plt.savefig('decision_boundaries.png', dpi=300)
    plt.show()


def run_experiment() -> Dict[str, Any]:
    """
    Run the full experiment and return results.

    Returns:
        Dictionary with experiment results
    """
    # Generate dataset with 2 important features and 20 noise features
    print("Generating synthetic dataset...")
    X, y, centers = generate_synthetic_data(n_samples=10000, n_noise_features=20)
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

    # Visualize training history
    visualize_training_history(history_lm.history, history_ls.history)

    # Visualize the feature weights
    visualize_feature_weights(lm_weights, ls_weights)

    # Generate smaller dataset for visualization
    X_viz, y_viz, centers_viz = generate_synthetic_data(n_samples=500, n_noise_features=20)

    # Visualize decision boundaries
    visualize_decision_boundaries(
        X_viz, y_viz,
        model_lm, model_ls,
        lm_weights, ls_weights,
        centers_viz
    )

    return {
        "models": {
            "lm": model_lm,
            "ls": model_ls
        },
        "weights": {
            "lm": lm_weights,
            "ls": ls_weights
        },
        "accuracy": {
            "lm": lm_eval[1],
            "ls": ls_eval[1]
        },
        "history": {
            "lm": history_lm.history,
            "ls": history_ls.history
        },
    }


def main():
    """Execute the experiment."""
    print("Starting LayerScale and LearnableMultiplier Feature Selection Experiment...")
    print("=" * 80)

    # Run the experiment
    results = run_experiment()

    print("=" * 80)
    print("Experiment completed. Results saved as PNG files.")

    # Print final accuracy comparison
    print("\nFinal Performance Comparison:")
    print(f"LearnableMultiplier Accuracy: {results['accuracy']['lm']:.4f}")
    print(f"LayerScale Accuracy: {results['accuracy']['ls']:.4f}")

    # Print feature selection success analysis
    lm_weights = results['weights']['lm']
    ls_weights = results['weights']['ls']

    # Find if the top 2 features match the true informative features (0 and 1)
    lm_top_features = np.argsort(lm_weights)[-2:]
    ls_top_features = np.argsort(ls_weights)[-2:]

    lm_success = set(lm_top_features) == {0, 1}
    ls_success = set(ls_top_features) == {0, 1}

    print("\nFeature Selection Success:")
    print(f"LearnableMultiplier: {'✓' if lm_success else '✗'} (Selected features: {lm_top_features})")
    print(f"LayerScale: {'✓' if ls_success else '✗'} (Selected features: {ls_top_features})")


if __name__ == "__main__":
    main()