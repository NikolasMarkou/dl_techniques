import sys

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.differentiable_kmeans import DifferentiableKMeansLayer


# ---------------------------------------------------------------------

# Generate synthetic clustered data
def generate_clustered_data(
        n_samples: int = 1000,
        n_features: int = 2,
        n_true_clusters: int = 5,
        noise: float = 0.1,
        random_state: int = 42
) -> tf.Tensor:
    """
    Generate synthetic data with clear cluster structure.

    Args:
        n_samples: Number of samples to generate
        n_features: Number of features per sample
        n_true_clusters: Number of true clusters in the data
        noise: Standard deviation of gaussian noise
        random_state: Random seed for reproducibility

    Returns:
        Tensor of shape (n_samples, n_features) with clustered data
    """
    X, _ = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_true_clusters,
        cluster_std=noise,
        random_state=random_state
    )
    return tf.cast(X, tf.float32)


# ---------------------------------------------------------------------

# Create and train a model
def train_kmeans_model(
        data: tf.Tensor,
        n_clusters: int,
        temperature: float = 0.1,
        epochs: int = 50,
        batch_size: int = 32
) -> tf.keras.Model:
    """
    Create and train a model with the DifferentiableKMeansLayer.

    Args:
        data: Input data tensor
        n_clusters: Number of clusters to learn
        temperature: Temperature parameter for soft assignments
        epochs: Number of training epochs
        batch_size: Batch size for training

    Returns:
        Trained model
    """
    # Create model
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(data.shape[1],)),
        DifferentiableKMeansLayer(n_clusters=n_clusters, temperature=temperature)
    ])

    # Custom loss to encourage well-separated clusters
    def cluster_separation_loss(y_true, y_pred):
        # Encourage high confidence assignments (closer to one-hot)
        entropy_loss = tf.reduce_mean(
            tf.reduce_sum(y_pred * tf.math.log(y_pred + 1e-7), axis=1)
        )
        return -entropy_loss  # Negative because we want to maximize confidence

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=cluster_separation_loss
    )

    # Train model
    history = model.fit(
        data, data,  # We don't need labels, just use data twice
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    return model, history


# ---------------------------------------------------------------------

# Visualize results
def plot_clustering_results(
        data: tf.Tensor,
        model: tf.keras.Model,
        title: str = "Clustering Results"
) -> None:
    """
    Plot clustering results for 2D data.

    Args:
        data: Input data tensor
        model: Trained model with DifferentiableKMeansLayer
        title: Plot title
    """
    # Get cluster assignments
    assignments = model.predict(data)
    cluster_indices = tf.argmax(assignments, axis=1)

    # Get centroids from the layer
    centroids = model.layers[0].centroids.numpy()

    # Create plot
    plt.figure(figsize=(10, 8))

    # Plot data points colored by cluster
    scatter = plt.scatter(
        data[:, 0], data[:, 1],
        c=cluster_indices,
        cmap='viridis',
        alpha=0.6
    )

    # Plot centroids
    plt.scatter(
        centroids[:, 0], centroids[:, 1],
        c='red',
        marker='x',
        s=200,
        linewidths=3,
        label='Centroids'
    )

    plt.title(title)
    plt.legend()
    plt.colorbar(scatter, label='Cluster Assignment')
    plt.show()


# ---------------------------------------------------------------------

def main():
    # Set random seed for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    # Generate synthetic data
    data = generate_clustered_data(
        n_samples=1000,
        n_features=2,  # 2D for easy visualization
        n_true_clusters=5,
        noise=0.1
    )

    # Train model
    model, history = train_kmeans_model(
        data=data,
        n_clusters=5,
        temperature=0.1,
        epochs=50
    )

    # Plot results
    plot_clustering_results(data, model, "DifferentiableKMeans Clustering Results")

    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'])
    plt.title('Model Loss During Training')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()

    # Show sample soft assignments
    sample_assignments = model.predict(data[:100])
    logger.info(f"Sample soft assignments for first 5 points:\n{sample_assignments}")
    logger.info(f"Sum of assignments (should be 1.0):\n{sample_assignments.sum(axis=1)}")


# ---------------------------------------------------------------------


# Main execution
if __name__ == "__main__":
    sys.exit(main())
