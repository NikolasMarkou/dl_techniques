"""
DifferentialFFN Experimental Analysis
====================================

This experiment evaluates the performance of the DifferentialFFN layer with customizable
branch activations (defaulting to "gelu") on a synthetic dataset. The experiment includes
comparison with baseline models and visualization of learned representations.

The experiment:
1. Creates a synthetic dataset requiring nuanced feature interaction understanding
2. Builds models with DifferentialFFN (gelu), DifferentialFFN (relu), and baseline architectures
3. Trains and evaluates all models
4. Visualizes performance and analyzes learned representations
5. Saves all results and a comprehensive summary in a timestamped directory
"""

import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from typing import Tuple, List, Dict, Any
import os
import json
import datetime
import pandas as pd

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Import the custom DifferentialFFN layer (adjust the import path as needed)
from dl_techniques.layers.ffn.diff_ffn import DifferentialFFN


def create_experiment_directory() -> str:
    """
    Create a timestamped directory to store all experiment results.

    Returns:
        Path to the created directory
    """
    # Create a timestamped directory name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"differential_ffn_experiment_{timestamp}"

    # Create the directory
    os.makedirs(dir_name, exist_ok=True)

    # Create subdirectories for visualization, models, and metrics
    os.makedirs(os.path.join(dir_name, "visualizations"), exist_ok=True)
    os.makedirs(os.path.join(dir_name, "model_summaries"), exist_ok=True)
    os.makedirs(os.path.join(dir_name, "metrics"), exist_ok=True)

    print(f"Created experiment directory: {dir_name}")
    return dir_name


def generate_synthetic_data(
        n_samples: int = 5000,
        n_features: int = 20,
        n_informative: int = 8,
        n_classes: int = 4,
        class_sep: float = 1.0,
        noise_level: float = 0.3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic classification dataset that benefits from differential processing.

    The dataset contains features with both positive and negative correlations to the target,
    as well as interaction effects that require nuanced understanding of feature relationships.

    Args:
        n_samples: Number of samples to generate
        n_features: Total number of features
        n_informative: Number of informative features
        n_classes: Number of classes
        class_sep: Factor controlling class separation
        noise_level: Intensity of noise added to the data

    Returns:
        Tuple of (X, y) with features and class labels
    """
    # Create a base matrix of random features
    X = np.random.randn(n_samples, n_features)

    # Initialize class centers with some separation
    centers = np.random.randn(n_classes, n_informative) * class_sep

    # Generate class memberships
    y = np.random.randint(0, n_classes, size=n_samples)

    # Make the first half of informative features have positive correlation
    # and the second half have negative correlation with the target
    pos_features = n_informative // 2
    neg_features = n_informative - pos_features

    # For each sample, adjust the informative features based on class
    for i in range(n_samples):
        class_idx = y[i]

        # Add positive correlation for first set of features
        X[i, :pos_features] += centers[class_idx, :pos_features]

        # Add negative correlation for second set of features
        X[i, pos_features:n_informative] -= centers[class_idx, pos_features:]

        # Add interaction effects between positive and negative features
        for p in range(pos_features):
            for n in range(neg_features):
                X[i, p] += 0.5 * X[i, pos_features + n] * np.sign(centers[class_idx, p])
                X[i, pos_features + n] -= 0.5 * X[i, p] * np.sign(centers[class_idx, pos_features + n])

    # Add noise to all features
    X += noise_level * np.random.randn(n_samples, n_features)

    return X, y


def create_model_with_differential_ffn(
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        activation: str = "gelu",
        dropout_rate: float = 0.5
) -> keras.Model:
    """
    Create a model using DifferentialFFN with specified activation.

    Args:
        input_dim: Number of input features
        num_classes: Number of output classes
        hidden_dim: Dimension of hidden layers
        activation: Activation to use in branch pathways ("gelu" or "elu")
        dropout_rate: Dropout rate

    Returns:
        Compiled keras model
    """
    inputs = keras.layers.Input(shape=(input_dim,))

    x = DifferentialFFN(
        hidden_dim=hidden_dim,
        output_dim=num_classes,
        branch_activation=activation,
        dropout_rate=dropout_rate
    )(inputs)

    # Output layer
    outputs = keras.layers.Softmax()(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def create_baseline_dense_model(
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        activation: str = "gelu",
        dropout_rate: float = 0.5
) -> keras.Model:
    """
    Create a baseline model using standard Dense layers.

    Args:
        input_dim: Number of input features
        num_classes: Number of output classes
        hidden_dim: Dimension of hidden layers
        activation: Activation function to use
        dropout_rate: Dropout rate

    Returns:
        Compiled keras model
    """
    inputs = keras.layers.Input(shape=(input_dim,))

    # First Dense layer with equivalent parameters to DifferentialFFN
    x = keras.layers.Dense(hidden_dim * 2, activation=activation)(inputs)
    x = keras.layers.LayerNormalization()(x)
    x = keras.layers.Dropout(dropout_rate)(x)

    # Second Dense layer
    x = keras.layers.Dense(hidden_dim, activation=activation)(x)
    x = keras.layers.LayerNormalization()(x)
    x = keras.layers.Dropout(dropout_rate)(x)

    # Third Dense layer to match parameter count
    x = keras.layers.Dense(hidden_dim // 2, activation=activation)(x)

    # Output layer
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def create_feedforward_model(
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        activation: str = "gelu",
        dropout_rate: float = 0.5
) -> keras.Model:
    """
    Create a simpler feedforward model as another baseline.

    Args:
        input_dim: Number of input features
        num_classes: Number of output classes
        hidden_dim: Dimension of hidden layers
        activation: Activation function to use
        dropout_rate: Dropout rate

    Returns:
        Compiled keras model
    """
    inputs = keras.layers.Input(shape=(input_dim,))

    # Two-pathway processing manually
    # Positive path
    pos = keras.layers.Dense(hidden_dim, activation=activation)(inputs)
    pos = keras.layers.Dense(hidden_dim, activation="sigmoid")(pos)

    # Negative path
    neg = keras.layers.Dense(hidden_dim, activation=activation)(inputs)
    neg = keras.layers.Dense(hidden_dim, activation="sigmoid")(neg)

    # Combine with subtraction
    x = keras.layers.Subtract()([pos, neg])
    x = keras.layers.LayerNormalization()(x)
    x = keras.layers.Dropout(dropout_rate)(x)

    # Second layer
    x = keras.layers.Dense(hidden_dim // 2, activation=activation)(x)

    # Output layer
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def save_model_summaries(models: Dict[str, keras.Model], output_dir: str) -> None:
    """
    Save model architecture summaries to text files.

    Args:
        models: Dictionary mapping model names to model instances
        output_dir: Directory to save the summaries
    """
    for name, model in models.items():
        # Create a file for the model summary
        summary_path = os.path.join(output_dir, "model_summaries", f"{name.replace(' ', '_').lower()}_summary.txt")

        # Save model summary using a custom function since model.summary() doesn't return a string
        with open(summary_path, 'w') as f:
            # Redirect summary to file
            def print_summary_to_file(line):
                f.write(line + '\n')

            model.summary(print_fn=print_summary_to_file)


def visualize_training_history(histories: Dict[str, Dict[str, List[float]]], output_dir: str) -> None:
    """
    Visualize the training history of multiple models.

    Args:
        histories: Dictionary mapping model names to their training histories
        output_dir: Directory to save visualizations
    """
    plt.figure(figsize=(15, 5))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    for name, history in histories.items():
        plt.plot(history["accuracy"], label=f"{name} Train")
        plt.plot(history["val_accuracy"], label=f"{name} Validation", linestyle="--")

    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(alpha=0.3)

    # Plot loss
    plt.subplot(1, 2, 2)
    for name, history in histories.items():
        plt.plot(history["loss"], label=f"{name} Train")
        plt.plot(history["val_loss"], label=f"{name} Validation", linestyle="--")

    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    viz_path = os.path.join(output_dir, "visualizations", "training_history_comparison.png")
    plt.savefig(viz_path, dpi=300)
    plt.show()


def visualize_feature_representations(
        models: Dict[str, keras.Model],
        X: np.ndarray,
        y: np.ndarray,
        output_dir: str
) -> None:
    """
    Visualize the learned feature representations using PCA.

    Args:
        models: Dictionary mapping model names to model instances
        X: Input features
        y: Class labels
        output_dir: Directory to save visualizations
    """
    plt.figure(figsize=(20, 5))

    for i, (name, model) in enumerate(models.items()):
        # Create a model that outputs the representation before the final layer
        feature_model = keras.Model(
            inputs=model.inputs,
            outputs=model.layers[-2].output
        )

        # Extract features
        features = feature_model.predict(X)

        # Apply PCA for visualization
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)

        # Plot
        plt.subplot(1, len(models), i + 1)
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1],
                             c=y, cmap="viridis", s=30, alpha=0.8)
        plt.title(f"{name} Feature Space")
        plt.colorbar(scatter)
        plt.grid(alpha=0.3)

    plt.tight_layout()
    viz_path = os.path.join(output_dir, "visualizations", "feature_representations.png")
    plt.savefig(viz_path, dpi=300)
    plt.show()


def visualize_decision_boundaries(
        models: Dict[str, keras.Model],
        X: np.ndarray,
        y: np.ndarray,
        output_dir: str
) -> None:
    """
    Visualize decision boundaries of models using PCA projection.

    Args:
        models: Dictionary mapping model names to model instances
        X: Input features
        y: Class labels
        output_dir: Directory to save visualizations
    """
    # Apply PCA to reduce to 2D for visualization
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    # Create a meshgrid for the 2D feature space
    h = 0.1  # Step size
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Create figure
    plt.figure(figsize=(20, 5))

    for i, (name, model) in enumerate(models.items()):
        ax = plt.subplot(1, len(models), i + 1)

        # Create a grid of points in PCA space
        grid_2d = np.c_[xx.ravel(), yy.ravel()]

        # Project back to original feature space
        grid = pca.inverse_transform(grid_2d)

        # Get predictions
        Z = model.predict(grid, verbose=0)
        Z = np.argmax(Z, axis=1)

        # Reshape to match grid
        Z = Z.reshape(xx.shape)

        # Plot decision boundaries
        ax.contourf(xx, yy, Z, alpha=0.3, cmap="viridis")

        # Plot the data points
        scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1],
                           c=y, cmap="viridis", s=30, alpha=0.8,
                           edgecolors="k")

        # Add title and labels
        ax.set_title(f"{name} Decision Boundary")
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.legend(*scatter.legend_elements(), title="Classes")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    viz_path = os.path.join(output_dir, "visualizations", "decision_boundaries.png")
    plt.savefig(viz_path, dpi=300)
    plt.show()


def save_experiment_metrics(
        evaluations: Dict[str, List[float]],
        output_dir: str
) -> None:
    """
    Save experiment metrics to CSV file.

    Args:
        evaluations: Dictionary mapping model names to evaluation metrics
        output_dir: Directory to save metrics
    """
    # Convert evaluations to DataFrame
    model_names = []
    accuracies = []
    losses = []

    for name, [loss, accuracy] in evaluations.items():
        model_names.append(name)
        accuracies.append(accuracy)
        losses.append(loss)

    df = pd.DataFrame({
        'Model': model_names,
        'Accuracy': accuracies,
        'Loss': losses
    })

    # Save to CSV
    df.to_csv(os.path.join(output_dir, "metrics", "model_comparison.csv"), index=False)


def save_experiment_history(
        histories: Dict[str, Dict[str, List[float]]],
        output_dir: str
) -> None:
    """
    Save training history data.

    Args:
        histories: Dictionary mapping model names to their training histories
        output_dir: Directory to save history data
    """
    # Save as JSON file
    with open(os.path.join(output_dir, "metrics", "training_history.json"), 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_histories = {}
        for name, history in histories.items():
            serializable_histories[name] = {
                key: [float(val) for val in values]
                for key, values in history.items()
            }
        json.dump(serializable_histories, f, indent=2)


def create_experiment_summary(
        evaluations: Dict[str, List[float]],
        output_dir: str,
        experiment_config: Dict[str, Any]
) -> None:
    """
    Create a comprehensive experiment summary markdown file.

    Args:
        evaluations: Dictionary mapping model names to evaluation metrics
        output_dir: Directory to save the summary
        experiment_config: Dictionary with experiment configuration details
    """
    # Identify best model
    best_model = max(evaluations.items(), key=lambda x: x[1][1])  # Sort by accuracy

    summary = f"""# DifferentialFFN Experimental Analysis

## Experiment Summary

**Date:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

### Dataset Configuration
- Samples: {experiment_config['n_samples']}
- Features: {experiment_config['n_features']}
- Informative Features: {experiment_config['n_informative']}
- Classes: {experiment_config['n_classes']}
- Class Separation: {experiment_config['class_sep']}
- Noise Level: {experiment_config['noise_level']}

### Model Architecture
- Hidden Dimension: {experiment_config['hidden_dim']}
- Dropout Rate: {experiment_config['dropout_rate']}
- Training Epochs: {experiment_config['epochs']}
- Batch Size: {experiment_config['batch_size']}

### Performance Results

#### Model Comparison
| Model | Accuracy | Loss |
|-------|----------|------|
"""

    # Add model performance rows
    for name, [loss, accuracy] in evaluations.items():
        summary += f"| {name} | {accuracy:.4f} | {loss:.4f} |\n"

    summary += f"""
**Best Model:** {best_model[0]} with accuracy {best_model[1][1]:.4f}

### Key Findings

1. The DifferentialFFN with GELU activation {'' if best_model[0] == 'DifferentialFFN (GELU)' else 'did not '} 
   perform best among all tested models.
   
2. The dual-pathway architecture in DifferentialFFN 
   {'' if evaluations["DifferentialFFN (GELU)"][1] > evaluations["Manual Dual-Pathway"][1] else 'did not '} 
   outperform a manual implementation of a similar concept.

3. The DifferentialFFN 
   {'' if evaluations["DifferentialFFN (GELU)"][1] > evaluations["Dense Baseline"][1] else 'did not '} 
   outperform a standard dense network with similar parameter count.

### Visualizations

1. Training History: [training_history_comparison.png](visualizations/training_history_comparison.png)
2. Feature Representations: [feature_representations.png](visualizations/feature_representations.png)
3. Decision Boundaries: [decision_boundaries.png](visualizations/decision_boundaries.png)

### Conclusion

The experiment demonstrates that the DifferentialFFN layer with its dual-pathway architecture 
{'provides advantages' if best_model[0].startswith('DifferentialFFN') else 'may not provide significant advantages'} 
for modeling complex feature interactions in this synthetic dataset.
"""

    # Save summary
    with open(os.path.join(output_dir, "experiment_summary.md"), 'w') as f:
        f.write(summary)

    # Also save a plain text version for quick reference
    with open(os.path.join(output_dir, "experiment_summary.txt"), 'w') as f:
        f.write(summary.replace('# ', '').replace('## ', '').replace('### ', '').replace('#### ', ''))


def run_experiment() -> Dict[str, Any]:
    """
    Run the full experiment and return results.

    Returns:
        Dictionary with experiment results
    """
    # Create directory for experiment results
    output_dir = create_experiment_directory()

    # Store experiment configuration
    experiment_config = {
        "n_samples": 10000,
        "n_features": 20,
        "n_informative": 8,
        "n_classes": 4,
        "class_sep": 1.0,
        "noise_level": 0.3,
        "hidden_dim": 128,
        "dropout_rate": 0.5,
        "epochs": 30,
        "batch_size": 32
    }

    # Save configuration
    with open(os.path.join(output_dir, "experiment_config.json"), 'w') as f:
        json.dump(experiment_config, f, indent=2)

    # Generate dataset
    print("Generating synthetic dataset...")
    X, y = generate_synthetic_data(
        n_samples=experiment_config["n_samples"],
        n_features=experiment_config["n_features"],
        n_informative=experiment_config["n_informative"],
        n_classes=experiment_config["n_classes"],
        class_sep=experiment_config["class_sep"],
        noise_level=experiment_config["noise_level"]
    )
    n_features = X.shape[1]
    n_classes = len(np.unique(y))

    print(f"Dataset: {X.shape[0]} samples, {n_features} features")
    print(f"Classes: {n_classes}")

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Create models
    models = {
        "DifferentialFFN (GELU)": create_model_with_differential_ffn(
            input_dim=n_features,
            num_classes=n_classes,
            activation="gelu",
            hidden_dim=experiment_config["hidden_dim"],
            dropout_rate=experiment_config["dropout_rate"]
        ),
        "DifferentialFFN (ELU)": create_model_with_differential_ffn(
            input_dim=n_features,
            num_classes=n_classes,
            activation="elu",
            hidden_dim=experiment_config["hidden_dim"],
            dropout_rate=experiment_config["dropout_rate"]
        ),
        "Dense Baseline": create_baseline_dense_model(
            input_dim=n_features,
            num_classes=n_classes,
            hidden_dim=experiment_config["hidden_dim"],
            dropout_rate=experiment_config["dropout_rate"]
        ),
        "Manual Dual-Pathway": create_feedforward_model(
            input_dim=n_features,
            num_classes=n_classes,
            hidden_dim=experiment_config["hidden_dim"],
            dropout_rate=experiment_config["dropout_rate"]
        )
    }

    # Save model summaries
    save_model_summaries(models, output_dir)

    # Train all models
    histories = {}
    evaluations = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")

        history = model.fit(
            X_train, y_train,
            epochs=experiment_config["epochs"],
            batch_size=experiment_config["batch_size"],
            validation_data=(X_test, y_test),
            verbose=1
        )

        histories[name] = history.history

        # Evaluate model
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        evaluations[name] = [loss, accuracy]

        print(f"{name} - Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

    # Save metrics and history
    save_experiment_metrics(evaluations, output_dir)
    save_experiment_history(histories, output_dir)

    # Visualize training histories
    visualize_training_history(histories, output_dir)

    # Generate smaller dataset for visualization
    X_viz, y_viz = generate_synthetic_data(
        n_samples=500,
        n_features=experiment_config["n_features"],
        n_informative=experiment_config["n_informative"],
        n_classes=experiment_config["n_classes"],
        class_sep=experiment_config["class_sep"],
        noise_level=experiment_config["noise_level"]
    )

    # Visualize feature representations
    visualize_feature_representations(models, X_viz, y_viz, output_dir)

    # Visualize decision boundaries
    visualize_decision_boundaries(models, X_viz, y_viz, output_dir)

    # Create experiment summary
    create_experiment_summary(
        evaluations,
        output_dir,
        experiment_config
    )

    return {
        "models": models,
        "evaluations": evaluations,
        "histories": histories,
        "output_dir": output_dir
    }


def main():
    """Execute the experiment."""
    print("Starting DifferentialFFN Experimental Analysis...")
    print("=" * 80)

    # Run the experiment
    results = run_experiment()

    print("=" * 80)
    output_dir = results["output_dir"]
    print(f"Experiment completed. Results saved in: {output_dir}")

    # Print final performance comparison
    print("\nFinal Performance Comparison:")
    for name, [loss, accuracy] in results["evaluations"].items():
        print(f"{name} - Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")

    # Identify best model
    best_model = max(results["evaluations"].items(),
                    key=lambda x: x[1][1])  # Sort by accuracy

    print(f"\nBest performing model: {best_model[0]} with accuracy {best_model[1][1]:.4f}")
    print(f"\nDetailed experiment summary available at: {os.path.join(output_dir, 'experiment_summary.md')}")


if __name__ == "__main__":
    main()