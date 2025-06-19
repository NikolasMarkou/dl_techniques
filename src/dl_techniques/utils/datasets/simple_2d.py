"""
2D Classification Dataset Generator
==================================

This module provides a DatasetGenerator class for creating various 2D classification datasets
that are useful for testing and visualizing the behavior of classification algorithms,
particularly for demonstrating the limitations of linear decision boundaries.

Each dataset generation method returns feature matrix X and labels y suitable for
binary classification tasks.

Example usage:
    from dataset_generator import DatasetGenerator, DatasetType

    # Generate an XOR dataset
    X, y = DatasetGenerator.generate_dataset(DatasetType.XOR, n_samples=2000, noise_level=0.1)

    # Generate a simple clusters dataset with centers
    X, y, centers = DatasetGenerator.generate_dataset(
        DatasetType.CLUSTERS,
        n_samples=1000,
        return_centers=True
    )
"""

import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Union
from sklearn.datasets import make_blobs, make_moons, make_circles

#------------------------------------------------------------------------------
# local imports
#------------------------------------------------------------------------------

from ..logger import logger

#------------------------------------------------------------------------------

class DatasetType(Enum):
    """Enumeration of available dataset types."""
    CLUSTERS = "Gaussian Clusters"
    MOONS = "Two Moons"
    CIRCLES = "Concentric Circles"
    XOR = "XOR Pattern"
    SPIRAL = "Spiral Pattern"
    GAUSSIAN_QUANTILES = "Gaussian Quantiles"
    MIXTURE = "Gaussian Mixture"
    CHECKER = "Checkerboard"

#------------------------------------------------------------------------------

# Default dataset parameters
DEFAULT_SAMPLES = 2000
DEFAULT_NOISE = 0.1
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42

# Dataset-specific parameters
CIRCLES_FACTOR = 0.5  # Scale factor between circles for the circles dataset
CLUSTER_SEPARATION = 4.0  # Default separation between clusters
CLUSTER_STD = 1.0  # Default standard deviation of clusters

# Visualization parameters
FIGURE_SIZE = (12, 10)
SCATTER_POINT_SIZE = 40
DPI = 300

# Set random seed for reproducibility
np.random.seed(DEFAULT_RANDOM_STATE)

#------------------------------------------------------------------------------


class DatasetGenerator:
    """Generator for various synthetic 2D datasets that challenge classification algorithms."""

    @staticmethod
    def generate_dataset(
            dataset_type: DatasetType,
            n_samples: int = DEFAULT_SAMPLES,
            noise_level: float = DEFAULT_NOISE,
            random_state: int = DEFAULT_RANDOM_STATE,
            return_centers: bool = False,
            **kwargs
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Generate a dataset of the specified type.

        Args:
            dataset_type: Type of dataset to generate from DatasetType enum
            n_samples: Number of samples to generate
            noise_level: Standard deviation of noise (where applicable)
            random_state: Random seed for reproducibility
            return_centers: Whether to return the cluster centers (only applicable for some datasets)
            **kwargs: Additional dataset-specific parameters
                - For CLUSTERS: centers, cluster_std, center_box
                - For CIRCLES: factor
                - For MIXTURE: centers, n_classes, cluster_std
                - For GAUSSIAN_QUANTILES: n_classes
                - For CHECKER: n_classes

        Returns:
            If return_centers is False or not applicable: Tuple of (X, y)
            If return_centers is True and applicable: Tuple of (X, y, centers)
        """
        # Pass common parameters to all methods
        common_params = {
            'n_samples': n_samples,
            'random_state': random_state
        }

        # Add dataset-specific parameters
        if dataset_type == DatasetType.CLUSTERS:
            centers = kwargs.get('centers', 2)
            cluster_std = kwargs.get('cluster_std', CLUSTER_STD)
            center_box = kwargs.get('center_box', (-CLUSTER_SEPARATION, CLUSTER_SEPARATION))

            return DatasetGenerator.generate_clusters(
                centers=centers,
                cluster_std=cluster_std,
                center_box=center_box,
                return_centers=return_centers,
                **common_params
            )

        elif dataset_type == DatasetType.MOONS:
            return DatasetGenerator.generate_moons(
                noise_level=noise_level,
                **common_params
            )

        elif dataset_type == DatasetType.CIRCLES:
            factor = kwargs.get('factor', CIRCLES_FACTOR)

            return DatasetGenerator.generate_circles(
                noise_level=noise_level,
                factor=factor,
                **common_params
            )

        elif dataset_type == DatasetType.XOR:
            return DatasetGenerator.generate_xor(
                noise_level=noise_level,
                **common_params
            )

        elif dataset_type == DatasetType.SPIRAL:
            return DatasetGenerator.generate_spiral(
                noise_level=noise_level,
                **common_params
            )

        elif dataset_type == DatasetType.GAUSSIAN_QUANTILES:
            n_classes = kwargs.get('n_classes', 2)

            return DatasetGenerator.generate_gaussian_quantiles(
                n_classes=n_classes,
                **common_params
            )

        elif dataset_type == DatasetType.MIXTURE:
            centers = kwargs.get('centers', 3)
            n_classes = kwargs.get('n_classes', 2)
            cluster_std = kwargs.get('cluster_std', CLUSTER_STD)

            return DatasetGenerator.generate_mixture(
                centers=centers,
                n_classes=n_classes,
                cluster_std=cluster_std,
                **common_params
            )

        elif dataset_type == DatasetType.CHECKER:
            n_classes = kwargs.get('n_classes', 2)

            return DatasetGenerator.generate_checker(
                n_classes=n_classes,
                noise_level=noise_level,
                **common_params
            )

        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

    @staticmethod
    def generate_moons(
            n_samples: int = DEFAULT_SAMPLES,
            noise_level: float = DEFAULT_NOISE,
            random_state: int = DEFAULT_RANDOM_STATE
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a moons dataset (two interleaving half-circles).

        Args:
            n_samples: Number of samples to generate
            noise_level: Standard deviation of Gaussian noise added to the data
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (X, y) with features and labels
        """
        return make_moons(
            n_samples=n_samples,
            noise=noise_level,
            random_state=random_state
        )

    @staticmethod
    def generate_circles(
            n_samples: int = DEFAULT_SAMPLES,
            noise_level: float = DEFAULT_NOISE,
            factor: float = CIRCLES_FACTOR,
            random_state: int = DEFAULT_RANDOM_STATE
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a concentric circles dataset.

        Args:
            n_samples: Number of samples to generate
            noise_level: Standard deviation of Gaussian noise added to the data
            factor: Scale factor between the circles (inner circle radius)
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (X, y) with features and labels
        """
        return make_circles(
            n_samples=n_samples,
            noise=noise_level,
            factor=factor,
            random_state=random_state
        )

    @staticmethod
    def generate_clusters(
            n_samples: int = DEFAULT_SAMPLES,
            centers: Union[int, List[List[float]]] = 2,
            n_features: int = 2,
            cluster_std: Union[float, List[float]] = CLUSTER_STD,
            center_box: Tuple[float, float] = (-CLUSTER_SEPARATION, CLUSTER_SEPARATION),
            random_state: int = DEFAULT_RANDOM_STATE,
            return_centers: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Generate Gaussian clusters dataset.

        This creates linearly separable clusters of points, with each cluster
        having a Gaussian distribution.

        Args:
            n_samples: Number of samples to generate
            centers: Number of centers (clusters) or list of center coordinates
            n_features: Number of features (default is 2 for 2D visualization)
            cluster_std: Standard deviation of clusters (can be float or array-like)
            center_box: Bounding box for cluster centers when centers is an int
            random_state: Random seed for reproducibility
            return_centers: Whether to return the cluster centers

        Returns:
            If return_centers is False: Tuple of (X, y) with features and labels
            If return_centers is True: Tuple of (X, y, centers) with features, labels, and centers
        """
        if isinstance(centers, int) and centers == 2:
            # For binary classification with 2 centers, place them along the x-axis
            # for a clear linear separation
            centers = [[-CLUSTER_SEPARATION/2, 0], [CLUSTER_SEPARATION/2, 0]]

        X, y, centers_out = make_blobs(
            n_samples=n_samples,
            n_features=n_features,
            centers=centers,
            cluster_std=cluster_std,
            center_box=center_box,
            return_centers=True,
            random_state=random_state
        )

        if return_centers:
            return X, y, centers_out
        else:
            return X, y

    @staticmethod
    def generate_xor(
            n_samples: int = DEFAULT_SAMPLES,
            noise_level: float = DEFAULT_NOISE,
            random_state: int = DEFAULT_RANDOM_STATE
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate an XOR-like dataset (points in opposite quadrants belong to the same class).

        Args:
            n_samples: Number of samples to generate
            noise_level: Standard deviation of noise
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (X, y) with features and labels
        """
        # Set random seed
        np.random.seed(random_state)

        n_samples_per_quadrant = n_samples // 4

        # Generate points in 4 quadrants
        X1 = np.random.rand(n_samples_per_quadrant, 2) - 0.5  # Quadrant 3 (-, -)
        X2 = np.random.rand(n_samples_per_quadrant, 2) - np.array([0.5, -0.5])  # Quadrant 2 (-, +)
        X3 = np.random.rand(n_samples_per_quadrant, 2) + 0.5  # Quadrant 1 (+, +)
        X4 = np.random.rand(n_samples_per_quadrant, 2) - np.array([-0.5, 0.5])  # Quadrant 4 (+, -)

        # Scale points to spread them out
        X1 *= 2
        X2 *= 2
        X3 *= 2
        X4 *= 2

        # Combine points
        X = np.vstack([X1, X2, X3, X4])

        # Add noise
        X += noise_level * np.random.randn(n_samples, 2)

        # Create labels (XOR pattern: class 0 for quadrants 1 and 3, class 1 for quadrants 2 and 4)
        y = np.hstack([
            np.zeros(n_samples_per_quadrant),  # Quadrant 3
            np.ones(n_samples_per_quadrant),   # Quadrant 2
            np.zeros(n_samples_per_quadrant),  # Quadrant 1
            np.ones(n_samples_per_quadrant)    # Quadrant 4
        ])

        return X, y

    @staticmethod
    def generate_spiral(
            n_samples: int = DEFAULT_SAMPLES,
            noise_level: float = DEFAULT_NOISE,
            random_state: int = DEFAULT_RANDOM_STATE
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate intertwined spirals dataset.

        Args:
            n_samples: Number of samples to generate
            noise_level: Standard deviation of noise
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (X, y) with features and labels
        """
        # Set random seed
        np.random.seed(random_state)

        n_samples_per_class = n_samples // 2

        # Generate spiral parameters
        theta = np.sqrt(np.random.rand(n_samples_per_class)) * 4 * np.pi

        # Generate first spiral
        r1 = theta + np.pi
        x1 = np.cos(r1) * r1
        y1 = np.sin(r1) * r1

        # Generate second spiral
        r2 = theta
        x2 = np.cos(r2) * r2
        y2 = np.sin(r2) * r2

        # Combine spirals
        X = np.vstack([
            np.column_stack([x1, y1]),
            np.column_stack([x2, y2])
        ])

        # Add noise
        X += noise_level * np.random.randn(n_samples, 2)

        # Normalize to reasonable range
        X = X / np.max(np.abs(X)) * 3

        # Create labels
        y = np.hstack([np.zeros(n_samples_per_class), np.ones(n_samples_per_class)])

        return X, y

    @staticmethod
    def generate_gaussian_quantiles(
            n_samples: int = DEFAULT_SAMPLES,
            n_classes: int = 2,
            random_state: int = DEFAULT_RANDOM_STATE
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a dataset with concentric normal distributions separated by quantiles.

        Args:
            n_samples: Number of samples to generate
            n_classes: Number of classes (rings)
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (X, y) with features and labels
        """
        from sklearn.datasets import make_gaussian_quantiles

        return make_gaussian_quantiles(
            n_samples=n_samples,
            n_features=2,
            n_classes=n_classes,
            random_state=random_state
        )

    @staticmethod
    def generate_mixture(
            n_samples: int = DEFAULT_SAMPLES,
            centers: int = 3,
            n_classes: int = 2,
            cluster_std: float = CLUSTER_STD,
            random_state: int = DEFAULT_RANDOM_STATE
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a dataset with overlapping clusters that belongs to different classes.

        Args:
            n_samples: Number of samples to generate
            centers: Number of cluster centers (should be greater than n_classes)
            n_classes: Number of classes
            cluster_std: Standard deviation of clusters
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (X, y) with features and labels
        """
        # Ensure centers > n_classes
        centers = max(centers, n_classes + 1)

        # Generate cluster centers
        np.random.seed(random_state)
        center_box_size = 5.0  # Controls the spread of centers
        center_positions = np.random.uniform(-center_box_size, center_box_size, (centers, 2))

        # Assign each center to a class
        center_classes = np.random.randint(0, n_classes, centers)

        # Generate samples for each center
        samples_per_center = n_samples // centers
        X = np.zeros((samples_per_center * centers, 2))
        y = np.zeros(samples_per_center * centers, dtype=int)

        for i in range(centers):
            center_pos = center_positions[i]
            center_class = center_classes[i]

            # Generate points around this center
            X_center = np.random.normal(
                loc=center_pos,
                scale=cluster_std,
                size=(samples_per_center, 2)
            )

            # Store points and labels
            start_idx = i * samples_per_center
            end_idx = (i + 1) * samples_per_center
            X[start_idx:end_idx] = X_center
            y[start_idx:end_idx] = center_class

        return X, y

    @staticmethod
    def generate_checker(
            n_samples: int = DEFAULT_SAMPLES,
            n_classes: int = 2,
            noise_level: float = DEFAULT_NOISE,
            random_state: int = DEFAULT_RANDOM_STATE
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a checkerboard dataset.

        Args:
            n_samples: Number of samples to generate
            n_classes: Number of classes (should be 2 for binary classification)
            noise_level: Standard deviation of noise
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (X, y) with features and labels
        """
        # Set random seed
        np.random.seed(random_state)

        # Number of checkerboard tiles per dimension
        n_tiles = 4  # Higher values = more complex decision boundary

        # Generate random points in unit square
        X = np.random.rand(n_samples, 2) * n_tiles

        # Determine class based on checkerboard pattern
        y = np.zeros(n_samples)
        for i in range(n_samples):
            x_tile = int(X[i, 0])
            y_tile = int(X[i, 1])
            # If sum of tile coordinates is even, assign to class 0, else class 1
            y[i] = (x_tile + y_tile) % 2

        # Add noise to make the problem more challenging
        X += noise_level * np.random.randn(n_samples, 2)

        # Scale to cover a reasonable range
        X = X / n_tiles * 4.0

        return X, y

    @staticmethod
    def visualize_dataset(
            X: np.ndarray,
            y: np.ndarray,
            title: str = "Dataset",
            filename: Optional[str] = None,
            show: bool = True,
            centers: Optional[np.ndarray] = None
    ) -> None:
        """
        Visualize a 2D classification dataset.

        Args:
            X: Feature matrix of shape (n_samples, 2)
            y: Labels of shape (n_samples,)
            title: Plot title
            filename: If provided, save the plot to this file
            show: Whether to display the plot
            centers: If provided, plot the cluster centers
        """
        plt.figure(figsize=FIGURE_SIZE)

        # Create a scatter plot of the data points
        plt.scatter(
            X[:, 0], X[:, 1],
            c=y,
            cmap='viridis',
            s=SCATTER_POINT_SIZE,
            edgecolors='k',
            alpha=0.8
        )

        # If centers are provided, plot them
        if centers is not None:
            plt.scatter(
                centers[:, 0], centers[:, 1],
                c='red',
                s=200,
                marker='X',
                edgecolors='k',
                label='Cluster Centers'
            )
            plt.legend()

        # Add labels and title
        plt.title(title, fontsize=16)
        plt.xlabel('Feature 1', fontsize=14)
        plt.ylabel('Feature 2', fontsize=14)

        # Add colorbar to indicate classes
        cbar = plt.colorbar()
        cbar.set_label('Class', fontsize=14)

        # Add grid for better readability
        plt.grid(alpha=0.3)

        # Adjust layout and save if filename is provided
        plt.tight_layout()

        if filename:
            plt.savefig(filename, dpi=DPI)

        if show:
            plt.show()
        else:
            plt.close()


def main():
    """
    Demonstrate usage by generating and visualizing all dataset types.
    """
    logger.info("Generating and visualizing datasets...")

    # Generate and visualize each dataset type
    for dataset_type in DatasetType:
        title = f"{dataset_type.value} ({dataset_type.name.lower()})"
        logger.info(f"Processing {title}...")

        # Generate data
        if dataset_type == DatasetType.CLUSTERS:
            # Special case for clusters to also return centers
            X, y, centers = DatasetGenerator.generate_dataset(
                dataset_type,
                return_centers=True
            )
            DatasetGenerator.visualize_dataset(
                X, y,
                title=f"{dataset_type.value} (linear boundary)",
                filename=f"{dataset_type.name.lower()}.png",
                centers=centers
            )
        else:
            X, y = DatasetGenerator.generate_dataset(dataset_type)
            boundary_type = "non-linear boundary"
            if dataset_type == DatasetType.GAUSSIAN_QUANTILES:
                boundary_type = "radial boundary"
            elif dataset_type in [DatasetType.MIXTURE, DatasetType.CHECKER]:
                boundary_type = "complex boundary"

            DatasetGenerator.visualize_dataset(
                X, y,
                title=f"{dataset_type.value} ({boundary_type})",
                filename=f"{dataset_type.name.lower()}.png"
            )

        # Print dataset statistics
        logger.info(f"  - Generated {X.shape[0]} samples with {X.shape[1]} features")
        logger.info(f"  - Class distribution: {np.bincount(y.astype(int))}")

    logger.info("\nAll datasets generated and visualized.")


if __name__ == "__main__":
    main()