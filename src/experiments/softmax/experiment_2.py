"""
Complete Softmax Decision Boundary Experiment with Enhanced Visualizations
=========================================================================

This module implements a comprehensive experiment for exploring softmax decision boundaries
with enhanced visualization capabilities. It integrates the dataset generator, experiment code,
and visualization techniques into a single, easy-to-use package.

The experiment:
1. Creates synthetic 2D datasets with 2 classes where softmax boundaries would be suboptimal
2. Builds a baseline model using standard softmax classification
3. Allows for registering and testing alternative models
4. Visualizes and compares decision boundaries and model performance with advanced techniques

Example usage:
    python softmax_boundaries_complete.py
"""
import copy

import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict, Any, Optional, Callable, Union
from enum import Enum
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

import dl_techniques.layers.rms_norm_spherical_bound
from dl_techniques.layers.layer_scale import LearnableMultiplier
from dl_techniques.layers.mish import SaturatedMish
from dl_techniques.layers.rms_norm_spherical_bound import SphericalBoundRMS
from dl_techniques.regularizers.binary_preference import BinaryPreferenceRegularizer
from dl_techniques.regularizers.soft_orthogonal import SoftOrthogonalConstraintRegularizer, \
    SoftOrthonormalConstraintRegularizer

#------------------------------------------------------------------------------
# GLOBAL CONSTANTS
#------------------------------------------------------------------------------

# Random seed configuration
RANDOM_SEED = 42

# Dataset generation parameters
DEFAULT_SAMPLES = 2000
DEFAULT_NOISE = 0.1
DEFAULT_TEST_SIZE = 0.2
CIRCLES_FACTOR = 0.5  # Scale factor between circles for the circles dataset

# Model parameters
DEFAULT_HIDDEN_LAYERS = [32, 16]
DEFAULT_ACTIVATION = 'relu'
DEFAULT_DROPOUT_RATE = 0.2
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_KERNEL_INITIALIZER = 'glorot_uniform'
NUM_CLASSES = 2  # Explicitly using 2 classes with softmax

# Training parameters
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 32
DEFAULT_VERBOSE = 1

# Visualization parameters
DECISION_BOUNDARY_STEP = 0.05
DATASET_FIGURE_SIZE = (10, 8)
HISTORY_FIGURE_SIZE = (15, 6)
SCATTER_POINT_SIZE = 30
BOUNDARY_FIGURE_SIZE_PER_MODEL = 6  # Width per model
BOUNDARY_FIGURE_HEIGHT = 5
CUSTOM_COLORS = ListedColormap(['#FF0000', '#0000FF'])
CONFIDENCE_CMAP = plt.cm.RdYlGn  # Red-Yellow-Green colormap for confidence
BOUNDARY_CMAP = plt.cm.RdBu      # Red-Blue colormap for decision boundaries
FIGURE_SIZE_LARGE = (16, 12)
FIGURE_SIZE_MEDIUM = (12, 10)
FIGURE_SIZE_SMALL = (10, 8)
DPI = 300
GRID_ALPHA = 0.2
FONT_SIZE_TITLE = 16
FONT_SIZE_LABEL = 12

# Set random seeds for reproducibility
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


#------------------------------------------------------------------------------
# DATASET GENERATOR
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


class DatasetGenerator:
    """Generator for various synthetic 2D datasets that challenge classification algorithms."""

    @staticmethod
    def generate_dataset(
            dataset_type: DatasetType,
            n_samples: int = DEFAULT_SAMPLES,
            noise_level: float = DEFAULT_NOISE,
            random_state: int = RANDOM_SEED,
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
            from sklearn.datasets import make_blobs
            centers = kwargs.get('centers', 2)
            cluster_std = kwargs.get('cluster_std', 1.0)

            if isinstance(centers, int) and centers == 2:
                # For binary classification with 2 centers, place them along the x-axis
                centers = [[-2, 0], [2, 0]]

            X, y, centers_out = make_blobs(
                n_samples=n_samples,
                n_features=2,
                centers=centers,
                cluster_std=cluster_std,
                return_centers=True,
                random_state=random_state
            )

            if return_centers:
                return X, y, centers_out
            else:
                return X, y

        elif dataset_type == DatasetType.MOONS:
            from sklearn.datasets import make_moons
            return make_moons(
                n_samples=n_samples,
                noise=noise_level,
                random_state=random_state
            )

        elif dataset_type == DatasetType.CIRCLES:
            from sklearn.datasets import make_circles
            factor = kwargs.get('factor', CIRCLES_FACTOR)

            return make_circles(
                n_samples=n_samples,
                noise=noise_level,
                factor=factor,
                random_state=random_state
            )

        elif dataset_type == DatasetType.XOR:
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

        elif dataset_type == DatasetType.SPIRAL:
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

        elif dataset_type == DatasetType.GAUSSIAN_QUANTILES:
            from sklearn.datasets import make_gaussian_quantiles
            n_classes = kwargs.get('n_classes', 2)

            return make_gaussian_quantiles(
                n_samples=n_samples,
                n_features=2,
                n_classes=n_classes,
                random_state=random_state
            )

        elif dataset_type == DatasetType.MIXTURE:
            # Set random seed
            np.random.seed(random_state)

            centers = kwargs.get('centers', 3)
            n_classes = kwargs.get('n_classes', 2)
            cluster_std = kwargs.get('cluster_std', 1.0)

            # Ensure centers > n_classes
            centers = max(centers, n_classes + 1)

            # Generate cluster centers
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

        elif dataset_type == DatasetType.CHECKER:
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

        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

    @staticmethod
    def visualize_dataset(
            X: np.ndarray,
            y: np.ndarray,
            title: str = "Dataset",
            filename: Optional[str] = None,
            show: bool = True,
            centers: Optional[np.ndarray] = None,
            fig_size: Tuple[int, int] = DATASET_FIGURE_SIZE
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
            fig_size: Figure size as (width, height)
        """
        plt.figure(figsize=fig_size)

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
        plt.title(title, fontsize=FONT_SIZE_TITLE)
        plt.xlabel('Feature 1', fontsize=FONT_SIZE_LABEL)
        plt.ylabel('Feature 2', fontsize=FONT_SIZE_LABEL)

        # Add colorbar to indicate classes
        cbar = plt.colorbar()
        cbar.set_label('Class', fontsize=FONT_SIZE_LABEL)

        # Add grid for better readability
        plt.grid(alpha=GRID_ALPHA)

        # Adjust layout and save if filename is provided
        plt.tight_layout()

        if filename:
            plt.savefig(filename, dpi=DPI)

        if show:
            plt.show()
        else:
            plt.close()


#------------------------------------------------------------------------------
# MODEL REGISTRY
#------------------------------------------------------------------------------

class ModelRegistry:
    """Registry for models to be evaluated in the experiment."""

    def __init__(self):
        """Initialize the model registry."""
        self.models: Dict[str, Callable[[], keras.Model]] = {}

    def register(
            self,
            name: str,
            model_fn: Callable[[], keras.Model]
    ) -> None:
        """
        Register a model for the experiment.

        Args:
            name: Unique name for the model
            model_fn: Function that returns a compiled keras model
        """
        if name in self.models:
            print(f"Warning: Overwriting existing model '{name}'")

        self.models[name] = model_fn

    def get_models(self) -> Dict[str, Callable[[], keras.Model]]:
        """
        Get all registered model functions.

        Returns:
            Dictionary of model creation functions
        """
        return self.models


#------------------------------------------------------------------------------
# MODEL DEFINITIONS
#------------------------------------------------------------------------------

class BaseModel:
    """Base class for all models in the experiment."""

    @staticmethod
    def create_softmax_baseline(
            input_dim: int = 2,
            hidden_layers: List[int] = DEFAULT_HIDDEN_LAYERS,
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            kernel_initializer: Union[str, keras.initializers.Initializer] = DEFAULT_KERNEL_INITIALIZER,
            activation: str = DEFAULT_ACTIVATION,
            dropout_rate: float = DEFAULT_DROPOUT_RATE
    ) -> keras.Model:
        """
        Create a baseline model with softmax output to demonstrate boundary issues.

        Args:
            input_dim: Number of input features
            hidden_layers: List of hidden layer sizes
            kernel_regularizer: Regularization for kernel weights
            kernel_initializer: Initializer for kernel weights
            activation: Activation function for hidden layers
            dropout_rate: Dropout rate (0 to disable)

        Returns:
            Compiled keras model
        """
        inputs = keras.layers.Input(shape=(input_dim,))
        x = inputs

        # Hidden layers
        for units in hidden_layers:
            x = keras.layers.Dense(
                units=units,
                activation=activation,
                kernel_regularizer=kernel_regularizer,
                kernel_initializer=kernel_initializer
            )(x)

            if dropout_rate > 0:
                x = keras.layers.Dropout(dropout_rate)(x)

        # Output layer with softmax activation for 2 classes
        outputs = keras.layers.Dense(
            units=NUM_CLASSES,
            activation='softmax',
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=kernel_initializer
        )(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=DEFAULT_LEARNING_RATE),
            loss='sparse_categorical_crossentropy',  # Use sparse since our labels are integers
            metrics=['accuracy']
        )

        return model


class SoftmaxBoundaryExperiment:
    """Custom layer for demonstrating softmax boundary alternatives"""

    class CustomSoftmaxLayer(keras.layers.Layer):
        """
        A custom layer that allows experimenting with alternatives to standard softmax.

        This layer can be extended to implement different approaches to classification
        boundaries that may improve over standard softmax.
        """

        def __init__(
                self,
                units: int = NUM_CLASSES,
                kernel_initializer: Union[str, keras.initializers.Initializer] = DEFAULT_KERNEL_INITIALIZER,
                kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
                **kwargs
        ):
            """
            Initialize the custom softmax layer.

            Args:
                units: Number of output units (classes)
                kernel_initializer: Initializer for the kernel weights
                kernel_regularizer: Regularizer for the kernel weights
                **kwargs: Additional arguments to pass to the Layer constructor
            """
            super().__init__(**kwargs)
            self.units = units
            self.kernel_initializer = keras.initializers.get(kernel_initializer)
            self.kernel_regularizer = kernel_regularizer

        def build(self, input_shape):
            """
            Build the layer with weights.

            Args:
                input_shape: Shape of the input tensor
            """
            # Create the weights matrix
            self.kernel = self.add_weight(
                name='kernel',
                shape=(input_shape[-1], self.units),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                trainable=True
            )

            # Optional bias
            self.bias = self.add_weight(
                name='bias',
                shape=(self.units,),
                initializer='zeros',
                trainable=True
            )

            super().build(input_shape)

        def call(self, inputs):
            """
            Forward pass through the layer.

            Args:
                inputs: Input tensor

            Returns:
                Output tensor with class probabilities
            """
            # Standard linear transform
            outputs = tf.matmul(inputs, self.kernel) + self.bias

            # Apply softmax activation
            return tf.nn.softmax(outputs)

        def get_config(self):
            """Get layer configuration for serialization."""
            config = super().get_config()
            config.update({
                'units': self.units,
                'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
                'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer)
            })
            return config

    class MarginSoftmaxLayer(CustomSoftmaxLayer):
        """
        A custom layer that implements margin-based softmax.

        This approach adds a margin to the logits before applying softmax,
        which can help create clearer decision boundaries.
        """

        def __init__(
                self,
                margin: float = 1.0,
                **kwargs
        ):
            """
            Initialize the margin softmax layer.

            Args:
                margin: Margin value to apply to the logits
                **kwargs: Additional arguments to pass to the parent constructor
            """
            super().__init__(**kwargs)
            self.margin = margin

        def call(self, inputs, training=None):
            """
            Forward pass with margin-based softmax.

            Args:
                inputs: Input tensor
                training: Whether in training mode

            Returns:
                Output tensor with class probabilities
            """
            # Standard linear transform
            logits = tf.matmul(inputs, self.kernel) + self.bias

            if training:
                # Apply margin during training only
                # This is a simplified approach - more sophisticated margin-based
                # softmax would use the true labels to apply the margin selectively
                scaled_logits = logits * self.margin
            else:
                scaled_logits = logits

            # Apply softmax activation
            return tf.nn.softmax(scaled_logits)

        def get_config(self):
            """Get layer configuration for serialization."""
            config = super().get_config()
            config.update({
                'margin': self.margin
            })
            return config

    @staticmethod
    def create_model_with_custom_softmax(
            layer_class,
            input_dim: int = 2,
            hidden_layers: List[int] = DEFAULT_HIDDEN_LAYERS,
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            kernel_initializer: Union[str, keras.initializers.Initializer] = DEFAULT_KERNEL_INITIALIZER,
            activation: str = DEFAULT_ACTIVATION,
            dropout_rate: float = DEFAULT_DROPOUT_RATE,
            **layer_kwargs
    ) -> keras.Model:
        """
        Create a model with a custom softmax layer.

        Args:
            layer_class: Custom layer class to use for classification
            input_dim: Number of input features
            hidden_layers: List of hidden layer sizes
            kernel_regularizer: Regularization for kernel weights
            kernel_initializer: Initializer for kernel weights
            activation: Activation function for hidden layers
            dropout_rate: Dropout rate (0 to disable)
            **layer_kwargs: Additional arguments to pass to the custom layer

        Returns:
            Compiled keras model
        """
        inputs = keras.layers.Input(shape=(input_dim,))
        x = inputs

        # Hidden layers
        for units in hidden_layers:
            x = keras.layers.Dense(
                units=units,
                activation=activation,
                kernel_regularizer=kernel_regularizer,
                kernel_initializer=kernel_initializer
            )(x)

            if dropout_rate > 0:
                x = keras.layers.Dropout(dropout_rate)(x)

        # Custom output layer
        outputs = layer_class(
            units=NUM_CLASSES,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            **layer_kwargs
        )(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=DEFAULT_LEARNING_RATE),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model


def create_mlp_with_custom_activation(
        activation: str = 'tanh',
        hidden_layers: List[int] = [64, 32],
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        kernel_initializer: Union[str, keras.initializers.Initializer] = DEFAULT_KERNEL_INITIALIZER
) -> keras.Model:
    """
    Create an MLP model with a custom activation function.

    Args:
        activation: Activation function to use
        hidden_layers: List of hidden layer sizes
        kernel_regularizer: Regularization for kernel weights
        kernel_initializer: Initializer for kernel weights

    Returns:
        Compiled keras model
    """
    inputs = keras.layers.Input(shape=(2,))
    x = inputs

    # Hidden layers
    for units in hidden_layers:
        x = keras.layers.Dense(
            units=units,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=kernel_initializer
        )(x)

    # Output layer with softmax activation for 2 classes
    outputs = keras.layers.Dense(
        units=NUM_CLASSES,
        activation='softmax',
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=DEFAULT_LEARNING_RATE),
        loss='sparse_categorical_crossentropy',  # Use sparse since our labels are integers
        metrics=['accuracy']
    )

    return model

def create_mlp_with_orthogonal_activation(
        activation: str = 'tanh',
        hidden_layers: List[int] = [64, 32],
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = SoftOrthonormalConstraintRegularizer(),
        kernel_initializer: Union[str, keras.initializers.Initializer] = DEFAULT_KERNEL_INITIALIZER
) -> keras.Model:
    """
    Create an MLP model with a custom activation function.

    Args:
        activation: Activation function to use
        hidden_layers: List of hidden layer sizes
        kernel_regularizer: Regularization for kernel weights
        kernel_initializer: Initializer for kernel weights

    Returns:
        Compiled keras model
    """
    inputs = keras.layers.Input(shape=(2,))
    x = inputs

    # Hidden layers
    for units in hidden_layers:
        x = keras.layers.Dense(
            units=units,
            activation="linear",
            kernel_regularizer=copy.deepcopy(kernel_regularizer),
            kernel_initializer=kernel_initializer
        )(x)
        x = SaturatedMish()(x)

    # Output layer with softmax activation for 2 classes
    x = keras.layers.Dense(
        units=NUM_CLASSES,
        activation='linear',
        kernel_regularizer=copy.deepcopy(kernel_regularizer),
        kernel_initializer=kernel_initializer
    )(x)
    outputs = keras.layers.Softmax()(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=DEFAULT_LEARNING_RATE),
        loss='sparse_categorical_crossentropy',  # Use sparse since our labels are integers
        metrics=['accuracy']
    )

    return model

#------------------------------------------------------------------------------
# ENHANCED VISUALIZATIONS
#------------------------------------------------------------------------------

class EnhancedVisualizations:
    """Enhanced visualization methods for classification models."""

    """
    Fixed visualize_confidence function to handle edge cases with uniform confidence values.
    """

    @staticmethod
    def visualize_confidence(
            model: keras.Model,
            X: np.ndarray,
            y: np.ndarray,
            title: str = "Model Confidence",
            plot_points: bool = True,
            grid_resolution: float = 0.05,
            fig_size: Tuple[int, int] = FIGURE_SIZE_MEDIUM,
            save_path: Optional[str] = None
    ) -> None:
        """
        Visualize model prediction confidence across the feature space.

        Args:
            model: Trained Keras model
            X: Feature matrix of shape (n_samples, 2)
            y: True labels of shape (n_samples,)
            title: Plot title
            plot_points: Whether to plot data points
            grid_resolution: Resolution of the grid for confidence visualization
            fig_size: Figure size as (width, height)
            save_path: Path to save the figure (None to not save)
        """
        # Create meshgrid for visualization
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, grid_resolution),
            np.arange(y_min, y_max, grid_resolution)
        )

        # Get predictions for the grid points
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        predictions = model.predict(grid_points, verbose=0)

        # Get confidence scores
        if predictions.shape[1] > 1:  # Multi-class with softmax
            # Get the highest probability for each point
            confidence = np.max(predictions, axis=1)
            # Get the predicted class
            pred_class = np.argmax(predictions, axis=1)
            # Map confidence to [0, 1] based on the predicted class
            # If class 0, invert confidence so red=class 0, green=class 1
            confidence_mapped = np.where(pred_class == 0, 1 - confidence, confidence)
        else:  # Binary with sigmoid
            # Sigmoid output is already a probability for class 1
            confidence = predictions.flatten()
            # Map to [0, 1] range where 0.5 is the decision boundary
            confidence_mapped = confidence

        # Reshape for plotting
        confidence_mapped = confidence_mapped.reshape(xx.shape)

        # Create plot
        plt.figure(figsize=fig_size)

        # Check if we have sufficient variation in confidence values
        unique_values = np.unique(confidence_mapped)
        if len(unique_values) <= 1:
            print(f"Warning: Cannot create confidence visualization for {title} - model predictions are uniform")
            # Create a simple decision boundary plot instead
            if predictions.shape[1] > 1:  # Multi-class with softmax
                Z = pred_class.reshape(xx.shape)
            else:  # Binary with sigmoid
                Z = (confidence > 0.5).astype(int).reshape(xx.shape)

            plt.contourf(xx, yy, Z, cmap=BOUNDARY_CMAP, alpha=0.8)
        else:
            # Use more robust level generation to avoid colorbar issues
            levels = np.linspace(np.min(confidence_mapped), np.max(confidence_mapped), 50)

            # Plot the confidence heatmap with explicit levels
            contour_plot = plt.contourf(xx, yy, confidence_mapped, cmap=CONFIDENCE_CMAP,
                                        alpha=0.8, levels=levels)

            # Add colorbar with the specific contour plot
            cbar = plt.colorbar(contour_plot)
            cbar.set_label('Confidence (class 1)', fontsize=FONT_SIZE_LABEL)

        # Plot the decision boundary as a contour line
        if predictions.shape[1] > 1:  # Multi-class with softmax
            Z = pred_class.reshape(xx.shape)
        else:  # Binary with sigmoid
            Z = (confidence > 0.5).astype(int).reshape(xx.shape)

        plt.contour(xx, yy, Z, colors=['k'], linewidths=2, levels=[0.5])

        # Plot the original data points if requested
        if plot_points:
            plt.scatter(
                X[:, 0], X[:, 1],
                c=y, cmap=BOUNDARY_CMAP,
                s=SCATTER_POINT_SIZE, edgecolors='k',
                alpha=0.6
            )

        # Add title and labels
        plt.title(title, fontsize=FONT_SIZE_TITLE)
        plt.xlabel('Feature 1', fontsize=FONT_SIZE_LABEL)
        plt.ylabel('Feature 2', fontsize=FONT_SIZE_LABEL)
        plt.grid(alpha=GRID_ALPHA)

        # Save figure if a path is provided
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, dpi=DPI)

        plt.show()

    @staticmethod
    def visualize_confusion_matrix(
            model: keras.Model,
            X: np.ndarray,
            y: np.ndarray,
            class_names: List[str] = ['Class 0', 'Class 1'],
            title: str = "Confusion Matrix",
            normalize: bool = True,
            fig_size: Tuple[int, int] = FIGURE_SIZE_SMALL,
            save_path: Optional[str] = None
    ) -> None:
        """
        Visualize the confusion matrix for model predictions.

        Args:
            model: Trained Keras model
            X: Feature matrix
            y: True labels
            class_names: Names of the classes
            title: Plot title
            normalize: Whether to normalize the confusion matrix
            fig_size: Figure size as (width, height)
            save_path: Path to save the figure (None to not save)
        """
        # Get predictions
        y_pred_proba = model.predict(X, verbose=0)

        # Convert to class predictions
        if y_pred_proba.shape[1] > 1:  # Multi-class with softmax
            y_pred = np.argmax(y_pred_proba, axis=1)
        else:  # Binary with sigmoid
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()

        # Calculate confusion matrix
        cm = confusion_matrix(y, y_pred)

        # Normalize if requested
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'

        # Create plot
        plt.figure(figsize=fig_size)

        # Plot confusion matrix
        sns.heatmap(
            cm, annot=True, fmt=fmt, cmap='Blues',
            xticklabels=class_names, yticklabels=class_names
        )

        # Add title and labels
        plt.title(title, fontsize=FONT_SIZE_TITLE)
        plt.ylabel('True Label', fontsize=FONT_SIZE_LABEL)
        plt.xlabel('Predicted Label', fontsize=FONT_SIZE_LABEL)

        # Save figure if a path is provided
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, dpi=DPI)

        plt.show()

    @staticmethod
    def visualize_model_comparison(
            models: Dict[str, keras.Model],
            X: np.ndarray,
            y: np.ndarray,
            title: str = "Model Comparison",
            grid_resolution: float = 0.05,
            fig_size: Optional[Tuple[int, int]] = None,
            save_path: Optional[str] = None
    ) -> None:
        """
        Create a side-by-side comparison of multiple models' decision boundaries.

        Args:
            models: Dictionary of model names and their Keras model instances
            X: Feature matrix of shape (n_samples, 2)
            y: True labels of shape (n_samples,)
            title: Main plot title
            grid_resolution: Resolution of the grid for visualization
            fig_size: Figure size as (width, height), if None calculated based on number of models
            save_path: Path to save the figure (None to not save)
        """
        n_models = len(models)

        # Calculate figure size if not provided
        if fig_size is None:
            width = min(16, 5 * n_models)  # Limit width to 16 inches
            fig_size = (width, 5)

        # Create meshgrid for visualization
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, grid_resolution),
            np.arange(y_min, y_max, grid_resolution)
        )

        # Create the figure and subplots
        fig, axes = plt.subplots(1, n_models, figsize=fig_size)

        # Handle the case of a single model
        if n_models == 1:
            axes = [axes]

        # Plot each model
        for i, (name, model) in enumerate(models.items()):
            # Get predictions for the grid points
            grid_points = np.c_[xx.ravel(), yy.ravel()]
            predictions = model.predict(grid_points, verbose=0)

            # Get class predictions
            if predictions.shape[1] > 1:  # Multi-class with softmax
                Z = np.argmax(predictions, axis=1)
            else:  # Binary with sigmoid
                Z = (predictions > 0.5).astype(int).flatten()

            # Reshape for plotting
            Z = Z.reshape(xx.shape)

            # Plot decision boundary
            axes[i].contourf(xx, yy, Z, cmap=BOUNDARY_CMAP, alpha=0.8)

            # Plot the original data points
            axes[i].scatter(
                X[:, 0], X[:, 1],
                c=y, cmap=BOUNDARY_CMAP,
                s=SCATTER_POINT_SIZE, edgecolors='k',
                alpha=0.6
            )

            # Set title and labels
            axes[i].set_title(name, fontsize=FONT_SIZE_LABEL)
            axes[i].set_xlabel('Feature 1', fontsize=FONT_SIZE_LABEL)

            # Only show y-axis label for the first subplot
            if i == 0:
                axes[i].set_ylabel('Feature 2', fontsize=FONT_SIZE_LABEL)

            # Add grid
            axes[i].grid(alpha=GRID_ALPHA)

        # Set overall title
        fig.suptitle(title, fontsize=FONT_SIZE_TITLE)

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the suptitle

        # Save figure if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=DPI)

        plt.show()

    @staticmethod
    def visualize_roc_curve(
            models: Dict[str, keras.Model],
            X: np.ndarray,
            y: np.ndarray,
            title: str = "ROC Curve Comparison",
            fig_size: Tuple[int, int] = FIGURE_SIZE_SMALL,
            save_path: Optional[str] = None
    ) -> None:
        """
        Visualize ROC curves for multiple models.

        Args:
            models: Dictionary of model names and their Keras model instances
            X: Feature matrix
            y: True labels
            title: Plot title
            fig_size: Figure size as (width, height)
            save_path: Path to save the figure (None to not save)
        """
        plt.figure(figsize=fig_size)

        # Plot ROC curve for each model
        for name, model in models.items():
            # Get predictions
            y_pred_proba = model.predict(X, verbose=0)

            # Convert to class 1 probability
            if y_pred_proba.shape[1] > 1:  # Multi-class with softmax
                y_pred_proba_1 = y_pred_proba[:, 1]  # Probability of class 1
            else:  # Binary with sigmoid
                y_pred_proba_1 = y_pred_proba.flatten()

            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y, y_pred_proba_1)
            roc_auc = auc(fpr, tpr)

            # Plot ROC curve
            plt.plot(
                fpr, tpr,
                label=f'{name} (AUC = {roc_auc:.3f})',
                linewidth=2
            )

        # Plot random classifier line
        plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)')

        # Add title and labels
        plt.title(title, fontsize=FONT_SIZE_TITLE)
        plt.xlabel('False Positive Rate', fontsize=FONT_SIZE_LABEL)
        plt.ylabel('True Positive Rate', fontsize=FONT_SIZE_LABEL)
        plt.legend(loc='lower right')
        plt.grid(alpha=GRID_ALPHA)

        # Save figure if a path is provided
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, dpi=DPI)

        plt.show()

    @staticmethod
    def visualize_precision_recall(
            models: Dict[str, keras.Model],
            X: np.ndarray,
            y: np.ndarray,
            title: str = "Precision-Recall Curve Comparison",
            fig_size: Tuple[int, int] = FIGURE_SIZE_SMALL,
            save_path: Optional[str] = None
    ) -> None:
        """
        Visualize precision-recall curves for multiple models.

        Args:
            models: Dictionary of model names and their Keras model instances
            X: Feature matrix
            y: True labels
            title: Plot title
            fig_size: Figure size as (width, height)
            save_path: Path to save the figure (None to not save)
        """
        plt.figure(figsize=fig_size)

        # Plot precision-recall curve for each model
        for name, model in models.items():
            # Get predictions
            y_pred_proba = model.predict(X, verbose=0)

            # Convert to class 1 probability
            if y_pred_proba.shape[1] > 1:  # Multi-class with softmax
                y_pred_proba_1 = y_pred_proba[:, 1]  # Probability of class 1
            else:  # Binary with sigmoid
                y_pred_proba_1 = y_pred_proba.flatten()

            # Calculate precision-recall curve
            precision, recall, _ = precision_recall_curve(y, y_pred_proba_1)
            pr_auc = auc(recall, precision)

            # Plot precision-recall curve
            plt.plot(
                recall, precision,
                label=f'{name} (AUC = {pr_auc:.3f})',
                linewidth=2
            )

        # Add title and labels
        plt.title(title, fontsize=FONT_SIZE_TITLE)
        plt.xlabel('Recall', fontsize=FONT_SIZE_LABEL)
        plt.ylabel('Precision', fontsize=FONT_SIZE_LABEL)
        plt.legend(loc='best')
        plt.grid(alpha=GRID_ALPHA)

        # Save figure if a path is provided
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, dpi=DPI)

        plt.show()

    @staticmethod
    def visualize_feature_space_transformation(
            model: keras.Model,
            X: np.ndarray,
            y: np.ndarray,
            layer_indices: Optional[List[int]] = None,
            title: str = "Feature Space Transformation",
            fig_size: Optional[Tuple[int, int]] = None,
            save_path: Optional[str] = None
    ) -> None:
        """
        Visualize how the feature space is transformed through the layers of the model.

        Args:
            model: Trained Keras model
            X: Feature matrix of shape (n_samples, 2)
            y: True labels of shape (n_samples,)
            layer_indices: Indices of layers to visualize (None for all layers)
            title: Main plot title
            fig_size: Figure size as (width, height)
            save_path: Path to save the figure (None to not save)
        """
        # Identify which layers to visualize (only layers with 2D outputs can be visualized)
        if layer_indices is None:
            # Get all intermediate layers that might be interesting to visualize
            layer_indices = []
            for i, layer in enumerate(model.layers):
                # Skip input layer and layers without weights
                if i == 0 or len(layer.weights) == 0:
                    continue

                # Skip non-dense layers like dropout
                if not isinstance(layer, keras.layers.Dense):
                    continue

                layer_indices.append(i)

            # Add the final layer
            layer_indices.append(len(model.layers) - 1)

        # Count how many layers we'll visualize
        n_layers = len(layer_indices) + 1  # +1 for the input

        # Set figure size if not provided
        if fig_size is None:
            # Calculate width based on number of layers to visualize
            width = min(16, 4 * n_layers)  # Limit width to 16 inches
            height = 4
            fig_size = (width, height)

        # Create figure and subplots
        fig, axes = plt.subplots(1, n_layers, figsize=fig_size)

        # Handle case of a single subplot
        if n_layers == 1:
            axes = [axes]

        # Plot the input features
        axes[0].scatter(
            X[:, 0], X[:, 1],
            c=y, cmap=BOUNDARY_CMAP,
            s=SCATTER_POINT_SIZE, edgecolors='k',
            alpha=0.8
        )
        axes[0].set_title('Input Features', fontsize=FONT_SIZE_LABEL)
        axes[0].set_xlabel('Feature 1', fontsize=FONT_SIZE_LABEL)
        axes[0].set_ylabel('Feature 2', fontsize=FONT_SIZE_LABEL)
        axes[0].grid(alpha=GRID_ALPHA)

        # Create intermediate models to get layer outputs
        for i, layer_idx in enumerate(layer_indices):
            # Create a model that outputs the layer activations
            layer_model = keras.Model(
                inputs=model.input,
                outputs=model.layers[layer_idx].output
            )

            # Get layer output for the input data
            layer_output = layer_model.predict(X, verbose=0)

            # Check if we can visualize this layer (need 2D output)
            if layer_output.shape[1] < 2:
                # Skip if not enough dimensions to visualize
                axes[i+1].text(
                    0.5, 0.5,
                    f"Layer {layer_idx}\nOutput dim: {layer_output.shape[1]}",
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=axes[i+1].transAxes
                )
                continue

            if layer_output.shape[1] > 2:
                # If more than 2D, use PCA to reduce to 2D for visualization
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                layer_output_2d = pca.fit_transform(layer_output)
                dim_note = f"\nPCA from {layer_output.shape[1]}D"
            else:
                # Already 2D
                layer_output_2d = layer_output
                dim_note = ""

            # Plot the transformed features
            axes[i+1].scatter(
                layer_output_2d[:, 0], layer_output_2d[:, 1],
                c=y, cmap=BOUNDARY_CMAP,
                s=SCATTER_POINT_SIZE, edgecolors='k',
                alpha=0.8
            )
            axes[i+1].set_title(f'Layer {layer_idx}{dim_note}', fontsize=FONT_SIZE_LABEL)
            axes[i+1].set_xlabel('Component 1', fontsize=FONT_SIZE_LABEL)
            axes[i+1].set_ylabel('Component 2', fontsize=FONT_SIZE_LABEL)
            axes[i+1].grid(alpha=GRID_ALPHA)

        # Set overall title
        fig.suptitle(title, fontsize=FONT_SIZE_TITLE)

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the suptitle

        # Save figure if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=DPI)

        plt.show()


#------------------------------------------------------------------------------
# MAIN EXPERIMENT
#------------------------------------------------------------------------------

class Experiment:
    """Main experiment class for evaluating softmax decision boundaries."""

    def __init__(self):
        """Initialize the experiment."""
        self.model_registry = ModelRegistry()
        self.results: Dict[str, Any] = {
            'models': {},
            'history': {},
            'accuracy': {},
            'dataset': None
        }

        # Register the baseline model
        self.model_registry.register(
            'Softmax Baseline',
            lambda: BaseModel.create_softmax_baseline()
        )

    def register_model(
            self,
            name: str,
            model_fn: Callable[[], keras.Model]
    ) -> None:
        """
        Register a model for the experiment.

        Args:
            name: Unique name for the model
            model_fn: Function that returns a compiled keras model
        """
        self.model_registry.register(name, model_fn)

    def run(
            self,
            dataset_type: DatasetType = DatasetType.SPIRAL,
            n_samples: int = DEFAULT_SAMPLES,
            noise_level: float = DEFAULT_NOISE,
            test_size: float = DEFAULT_TEST_SIZE,
            epochs: int = DEFAULT_EPOCHS,
            batch_size: int = DEFAULT_BATCH_SIZE,
            verbose: int = DEFAULT_VERBOSE
    ) -> Dict[str, Any]:
        """
        Run the experiment.

        Args:
            dataset_type: Type of dataset to generate from DatasetType enum
            n_samples: Number of samples to generate
            noise_level: Standard deviation of noise
            test_size: Fraction of data to use for testing
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Verbosity level for training

        Returns:
            Dictionary with experiment results
        """
        # Generate dataset
        print(f"Generating {dataset_type.value} dataset...")

        X, y = DatasetGenerator.generate_dataset(
            dataset_type=dataset_type,
            n_samples=n_samples,
            noise_level=noise_level
        )

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y
        )

        # Store dataset
        self.results['dataset'] = {
            'X': X,
            'y': y,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'type': dataset_type
        }

        # Train and evaluate each model
        model_fns = self.model_registry.get_models()

        for name, model_fn in model_fns.items():
            print(f"\nTraining model: {name}")

            # Create model
            model = model_fn()

            # Train model
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test, y_test),
                verbose=verbose
            )

            # Evaluate model
            evaluation = model.evaluate(X_test, y_test, verbose=0)
            accuracy = evaluation[1]  # Accuracy is usually the second metric

            # Store results
            self.results['models'][name] = model
            self.results['history'][name] = history.history
            self.results['accuracy'][name] = accuracy

            print(f"{name} Test Accuracy: {accuracy:.4f}")

        return self.results

    def visualize_dataset(self) -> None:
        """Visualize the dataset used in the experiment."""
        if self.results['dataset'] is None:
            print("No dataset available. Run the experiment first.")
            return

        X = self.results['dataset']['X']
        y = self.results['dataset']['y']
        dataset_type = self.results['dataset']['type']

        # Use the value property of the enum to get the display name
        dataset_name = dataset_type.value if hasattr(dataset_type, 'value') else str(dataset_type)

        DatasetGenerator.visualize_dataset(
            X, y,
            title=f"{dataset_name} Dataset",
            filename=f"{dataset_type.name.lower() if hasattr(dataset_type, 'name') else str(dataset_type).lower()}_dataset.png"
        )

    def visualize_training_history(self) -> None:
        """Visualize the training history of all models."""
        if not self.results['history']:
            print("No training history available. Run the experiment first.")
            return

        plt.figure(figsize=HISTORY_FIGURE_SIZE)

        # Plot accuracy
        plt.subplot(1, 2, 1)
        for name, history in self.results['history'].items():
            plt.plot(history['accuracy'], label=f'{name} (Train)')
            plt.plot(history['val_accuracy'], label=f'{name} (Val)', linestyle='--')

        plt.title('Model Accuracy', fontsize=FONT_SIZE_TITLE)
        plt.ylabel('Accuracy', fontsize=FONT_SIZE_LABEL)
        plt.xlabel('Epoch', fontsize=FONT_SIZE_LABEL)
        plt.legend()
        plt.grid(alpha=GRID_ALPHA)

        # Plot loss
        plt.subplot(1, 2, 2)
        for name, history in self.results['history'].items():
            plt.plot(history['loss'], label=f'{name} (Train)')
            plt.plot(history['val_loss'], label=f'{name} (Val)', linestyle='--')

        plt.title('Model Loss', fontsize=FONT_SIZE_TITLE)
        plt.ylabel('Loss', fontsize=FONT_SIZE_LABEL)
        plt.xlabel('Epoch', fontsize=FONT_SIZE_LABEL)
        plt.legend()
        plt.grid(alpha=GRID_ALPHA)

        plt.tight_layout()
        plt.savefig('training_history.png', dpi=DPI)
        plt.show()

    def visualize_decision_boundaries(self) -> None:
        """Visualize decision boundaries for all models."""
        if not self.results['models'] or self.results['dataset'] is None:
            print("No models or dataset available. Run the experiment first.")
            return

        X = self.results['dataset']['X']
        y = self.results['dataset']['y']

        # Create a meshgrid for the feature space
        h = DECISION_BOUNDARY_STEP  # Step size
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Create a figure
        n_models = len(self.results['models'])
        fig, axes = plt.subplots(1, n_models, figsize=(BOUNDARY_FIGURE_SIZE_PER_MODEL*n_models, BOUNDARY_FIGURE_HEIGHT))

        # Handle the case of a single model
        if n_models == 1:
            axes = [axes]

        # Define custom colormap
        cm = plt.cm.RdBu
        cm_bright = CUSTOM_COLORS

        # Plot decision boundaries and data points for each model
        for i, (name, model) in enumerate(self.results['models'].items()):
            # Plot decision boundary
            grid_points = np.c_[xx.ravel(), yy.ravel()]
            Z = model.predict(grid_points, verbose=0)

            # For softmax, get the predicted class (highest probability)
            if Z.shape[1] > 1:  # If using softmax with multiple output units
                Z = np.argmax(Z, axis=1)
            else:  # If using sigmoid
                Z = (Z > 0.5).astype(int).flatten()

            # Reshape result to match xx shape
            Z = Z.reshape(xx.shape)

            # Plot the result
            axes[i].contourf(xx, yy, Z, cmap=cm, alpha=0.8)

            # Plot the data points
            axes[i].scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright, s=20, edgecolors='k')

            # Configure the plot
            axes[i].set_title(f'{name}\nAccuracy: {self.results["accuracy"][name]:.4f}', fontsize=FONT_SIZE_LABEL)
            axes[i].set_xlabel('Feature 1', fontsize=FONT_SIZE_LABEL)
            axes[i].set_ylabel('Feature 2', fontsize=FONT_SIZE_LABEL)
            axes[i].set_xlim(xx.min(), xx.max())
            axes[i].set_ylim(yy.min(), yy.max())
            axes[i].grid(alpha=GRID_ALPHA)

        plt.tight_layout()
        plt.savefig('decision_boundaries.png', dpi=DPI)
        plt.show()

    def visualize_all(self) -> None:
        """Visualize dataset, training history, and decision boundaries."""
        self.visualize_dataset()
        self.visualize_training_history()
        self.visualize_decision_boundaries()

    def visualize_model_confidence(self, model_name: Optional[str] = None) -> None:
        """
        Visualize confidence of a model or all models.

        Args:
            model_name: Name of the model to visualize (None for all models)
        """
        if self.results['dataset'] is None:
            print("No dataset available. Run the experiment first.")
            return

        if not self.results['models']:
            print("No models available. Run the experiment first.")
            return

        X = self.results['dataset']['X']
        y = self.results['dataset']['y']

        if model_name is not None:
            # Visualize a specific model
            if model_name not in self.results['models']:
                print(f"Model '{model_name}' not found.")
                return

            model = self.results['models'][model_name]
            EnhancedVisualizations.visualize_confidence(
                model, X, y,
                title=f"{model_name} Prediction Confidence",
                save_path=f"{model_name.lower().replace(' ', '_')}_confidence.png"
            )
        else:
            # Visualize all models
            for name, model in self.results['models'].items():
                EnhancedVisualizations.visualize_confidence(
                    model, X, y,
                    title=f"{name} Prediction Confidence",
                    save_path=f"{name.lower().replace(' ', '_')}_confidence.png"
                )

    def visualize_confusion_matrices(self) -> None:
        """Visualize confusion matrices for all models."""
        if self.results['dataset'] is None or not self.results['models']:
            print("No dataset or models available. Run the experiment first.")
            return

        X_test = self.results['dataset']['X_test']
        y_test = self.results['dataset']['y_test']

        for name, model in self.results['models'].items():
            EnhancedVisualizations.visualize_confusion_matrix(
                model, X_test, y_test,
                title=f"{name} Confusion Matrix",
                save_path=f"{name.lower().replace(' ', '_')}_confusion_matrix.png"
            )

    def visualize_roc_curves(self) -> None:
        """Visualize ROC curves for all models."""
        if self.results['dataset'] is None or not self.results['models']:
            print("No dataset or models available. Run the experiment first.")
            return

        X_test = self.results['dataset']['X_test']
        y_test = self.results['dataset']['y_test']

        EnhancedVisualizations.visualize_roc_curve(
            self.results['models'], X_test, y_test,
            save_path="roc_curves.png"
        )

    def visualize_precision_recall_curves(self) -> None:
        """Visualize precision-recall curves for all models."""
        if self.results['dataset'] is None or not self.results['models']:
            print("No dataset or models available. Run the experiment first.")
            return

        X_test = self.results['dataset']['X_test']
        y_test = self.results['dataset']['y_test']

        EnhancedVisualizations.visualize_precision_recall(
            self.results['models'], X_test, y_test,
            save_path="precision_recall_curves.png"
        )

    def visualize_models_comparison(self) -> None:
        """Visualize a direct comparison of all model decision boundaries."""
        if self.results['dataset'] is None or not self.results['models']:
            print("No dataset or models available. Run the experiment first.")
            return

        X = self.results['dataset']['X']
        y = self.results['dataset']['y']

        EnhancedVisualizations.visualize_model_comparison(
            self.results['models'], X, y,
            title="Model Decision Boundaries Comparison",
            save_path="model_comparison.png"
        )

    def visualize_feature_transformation(self, model_name: str) -> None:
        """
        Visualize how the feature space is transformed through a model's layers.

        Args:
            model_name: Name of the model to visualize
        """
        if self.results['dataset'] is None:
            print("No dataset available. Run the experiment first.")
            return

        if model_name not in self.results['models']:
            print(f"Model '{model_name}' not found.")
            return

        X = self.results['dataset']['X']
        y = self.results['dataset']['y']
        model = self.results['models'][model_name]

        EnhancedVisualizations.visualize_feature_space_transformation(
            model, X, y,
            title=f"{model_name} Feature Space Transformation",
            save_path=f"{model_name.lower().replace(' ', '_')}_feature_transformation.png"
        )

    def visualize_advanced(self) -> None:
        """Run all advanced visualizations."""
        print("Generating model confidence visualizations...")
        self.visualize_model_confidence()

        print("Generating confusion matrices...")
        self.visualize_confusion_matrices()

        print("Generating ROC curves...")
        self.visualize_roc_curves()

        print("Generating precision-recall curves...")
        self.visualize_precision_recall_curves()

        print("Generating model comparison visualization...")
        self.visualize_models_comparison()

        print("Generating feature transformation visualizations...")
        for name in self.results['models'].keys():
            self.visualize_feature_transformation(name)

        print("All visualizations completed.")

    def visualize_all_enhanced(self) -> None:
        """Run all basic and advanced visualizations."""
        print("Running basic visualizations...")
        self.visualize_all()

        print("Running advanced visualizations...")
        self.visualize_advanced()


def run_experiment(
        dataset_type: DatasetType = DatasetType.SPIRAL,
        include_example_models: bool = True
) -> Dict[str, Any]:
    """
    Run the full experiment with enhanced visualizations.

    Args:
        dataset_type: Type of dataset to generate from DatasetType enum
        include_example_models: Whether to include example custom models

    Returns:
        Dictionary with experiment results
    """
    # Create experiment
    experiment = Experiment()

    # Register example models if requested
    if include_example_models:
        experiment.register_model(
            'MLP with Tanh',
            lambda: create_mlp_with_custom_activation(activation='tanh')
        )

        experiment.register_model(
            'MLP with Orthogonal',
            lambda: create_mlp_with_orthogonal_activation(activation='tanh')
        )

        experiment.register_model(
            'Margin Softmax',
            lambda: SoftmaxBoundaryExperiment.create_model_with_custom_softmax(
                SoftmaxBoundaryExperiment.MarginSoftmaxLayer,
                margin=1.5
            )
        )

    # Run experiment
    results = experiment.run(
        dataset_type=dataset_type,
        epochs=DEFAULT_EPOCHS,
        verbose=DEFAULT_VERBOSE
    )

    # Generate all visualizations (basic and advanced)
    experiment.visualize_all_enhanced()

    return results


def main():
    """Execute the enhanced experiment."""
    print("Starting Enhanced Softmax Decision Boundary Experiment...")
    print("=" * 80)

    # Set the dataset type to use for the experiment
    dataset_type = DatasetType.SPIRAL  # Change this to explore different datasets

    # Run the experiment with all visualizations
    results = run_experiment(dataset_type=dataset_type)

    print("=" * 80)
    print("Experiment completed with enhanced visualizations.")
    print("All visualization files saved to current directory.")

    # Print final accuracy comparison
    print("\nFinal Performance Comparison:")
    for name, accuracy in results['accuracy'].items():
        print(f"{name} Accuracy: {accuracy:.4f}")

    print("\nNote on Softmax Boundaries:")
    print("Softmax produces linear decision boundaries that may not be optimal")
    print("for complex datasets. The enhanced visualizations provide deeper")
    print("insights into model behavior and decision boundary characteristics.")


if __name__ == "__main__":
    main()