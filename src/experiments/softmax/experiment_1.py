"""
Softmax Decision Boundary Experiment
===================================

This experiment demonstrates the limitations of softmax decision boundaries in 2D space
and provides a framework for testing alternative models that can mitigate these limitations.

The experiment:
1. Creates synthetic 2D datasets with 2 classes where softmax boundaries would be suboptimal
2. Builds a baseline model using standard softmax classification
3. Allows for registering and testing alternative models
4. Visualizes and compares decision boundaries and model performance

Example usage:
    python softmax_boundaries_experiment.py
"""

import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict, Any, Optional, Callable, Union

from dl_techniques.utils.datasets.simple_2d import (
    DatasetGenerator,
    DatasetType
)

#------------------------------------------------------------------------------
# GLOBAL CONSTANTS
#------------------------------------------------------------------------------

# Random seed configuration
RANDOM_SEED = 42

# Dataset generation parameters
DEFAULT_SAMPLES = 2000
DEFAULT_NOISE = 0.1
DEFAULT_TEST_SIZE = 0.2
DEFAULT_DATASET_TYPE = DatasetType.SPIRAL
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

# Set random seeds for reproducibility
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)




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
            dataset_type: DatasetType = DEFAULT_DATASET_TYPE,
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
            dataset_type: Type of dataset to generate ('moons', 'circles', 'xor', 'spiral')
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
        print(f"Generating {dataset_type} dataset...")

        X, y = DatasetGenerator.generate_dataset(
            dataset_type=dataset_type,
            n_samples=n_samples,
            noise_level=noise_level
        )

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
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

        plt.figure(figsize=DATASET_FIGURE_SIZE)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=SCATTER_POINT_SIZE, edgecolors='k')
        plt.title(f'{dataset_type.capitalize()} Dataset', fontsize=16)
        plt.xlabel('Feature 1', fontsize=12)
        plt.ylabel('Feature 2', fontsize=12)
        plt.colorbar(label='Class')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{dataset_type}_dataset.png', dpi=300)
        plt.show()

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

        plt.title('Model Accuracy', fontsize=14)
        plt.ylabel('Accuracy', fontsize=12)
        plt.xlabel('Epoch', fontsize=12)
        plt.legend()
        plt.grid(alpha=0.3)

        # Plot loss
        plt.subplot(1, 2, 2)
        for name, history in self.results['history'].items():
            plt.plot(history['loss'], label=f'{name} (Train)')
            plt.plot(history['val_loss'], label=f'{name} (Val)', linestyle='--')

        plt.title('Model Loss', fontsize=14)
        plt.ylabel('Loss', fontsize=12)
        plt.xlabel('Epoch', fontsize=12)
        plt.legend()
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300)
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
            axes[i].set_title(f'{name}\nAccuracy: {self.results["accuracy"][name]:.4f}', fontsize=14)
            axes[i].set_xlabel('Feature 1', fontsize=12)
            axes[i].set_ylabel('Feature 2', fontsize=12)
            axes[i].set_xlim(xx.min(), xx.max())
            axes[i].set_ylim(yy.min(), yy.max())
            axes[i].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig('decision_boundaries.png', dpi=300)
        plt.show()

    def visualize_all(self) -> None:
        """Visualize dataset, training history, and decision boundaries."""
        self.visualize_dataset()
        self.visualize_training_history()
        self.visualize_decision_boundaries()


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


def run_experiment(
        dataset_type: str = DEFAULT_DATASET_TYPE,
        include_example_model: bool = True
) -> Dict[str, Any]:
    """
    Run the full experiment.

    Args:
        dataset_type: Type of dataset to generate ('moons', 'circles', 'xor', 'spiral')
        include_example_model: Whether to include an example custom model

    Returns:
        Dictionary with experiment results
    """
    # Create experiment
    experiment = Experiment()

    # Register example custom model if requested
    if include_example_model:
        experiment.register_model(
            'MLP with Tanh',
            lambda: create_mlp_with_custom_activation(activation='tanh')
        )

    # Run experiment
    results = experiment.run(
        dataset_type=dataset_type,
        epochs=DEFAULT_EPOCHS,
        verbose=DEFAULT_VERBOSE
    )

    # Visualize results
    experiment.visualize_all()

    return results


def main():
    """Execute the experiment."""
    print("Starting Softmax Decision Boundary Experiment...")
    print("=" * 80)

    # Create experiment
    experiment = Experiment()

    # Register an example custom model with tanh activation
    experiment.register_model(
        'MLP with Tanh',
        lambda: create_mlp_with_custom_activation(activation='tanh')
    )

    # Register a margin-based softmax model
    experiment.register_model(
        'Margin Softmax',
        lambda: SoftmaxBoundaryExperiment.create_model_with_custom_softmax(
            SoftmaxBoundaryExperiment.MarginSoftmaxLayer,
            margin=1.5
        )
    )

    # Run the experiment
    results = experiment.run(dataset_type=DEFAULT_DATASET_TYPE)

    print("=" * 80)
    print("Experiment completed. Results saved as PNG files.")

    # Print final accuracy comparison
    print("\nFinal Performance Comparison:")
    for name, accuracy in results['accuracy'].items():
        print(f"{name} Accuracy: {accuracy:.4f}")

    print("\nNote on Softmax Boundaries:")
    print("Softmax produces linear decision boundaries that may not be optimal")
    print("for complex datasets. The experiment demonstrates how alternative")
    print("approaches can create more flexible decision boundaries.")


if __name__ == "__main__":
    main()