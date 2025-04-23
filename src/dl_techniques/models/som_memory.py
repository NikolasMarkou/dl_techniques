import time
import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union, List, Dict, Any, Callable

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.som_2d_layer import SOM2dLayer


# ---------------------------------------------------------------------


@keras.utils.register_keras_serializable()
class SOMModel(keras.Model):
    """
    A Keras model implementing a Self-Organizing Map as a memory structure.

    This model wraps the SOM layer and provides additional methods for training
    and visualization specific to Self-Organizing Maps, demonstrating how
    SOMs can function as associative memory systems.

    Parameters
    ----------
    map_size : Tuple[int, int]
        Size of the SOM grid (height, width).
    input_dim : int
        Dimensionality of the input data.
    initial_learning_rate : float, optional
        Initial learning rate for weight updates. Defaults to 0.1.
    sigma : float, optional
        Initial neighborhood radius. Defaults to 1.0.
    neighborhood_function : str, optional
        Type of neighborhood function to use ('gaussian' or 'bubble').
        Defaults to 'gaussian'.
    weights_initializer : Union[str, initializers.Initializer], optional
        Initialization method for weights. Defaults to 'random'.
    regularizer : Optional[regularizers.Regularizer], optional
        Regularizer function applied to the weights. Defaults to None.
    name : str, optional
        Name of the model. Defaults to None.
    """

    def __init__(
            self,
            map_size: Tuple[int, int],
            input_dim: int,
            initial_learning_rate: float = 0.1,
            sigma: float = 1.0,
            neighborhood_function: str = 'gaussian',
            weights_initializer: Union[str, keras.initializers.Initializer] = 'random',
            regularizer: Optional[keras.regularizers.Regularizer] = None,
            name: Optional[str] = None,
            **kwargs
    ):
        """Initialize the SOM model."""
        super(SOMModel, self).__init__(name=name, **kwargs)

        # Input layer
        self.input_layer = keras.Input(shape=(input_dim,))

        # Create the SOM layer
        self.som_layer = SOM2dLayer(
            map_size=map_size,
            input_dim=input_dim,
            initial_learning_rate=initial_learning_rate,
            sigma=sigma,
            neighborhood_function=neighborhood_function,
            weights_initializer=weights_initializer,
            regularizer=regularizer
        )

        # Class prototypes for classification and memory retrieval
        self.class_prototypes = None

    def call(self, inputs: tf.Tensor, training: bool = None) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Forward pass for the SOM model.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor of shape (batch_size, input_dim).
        training : bool, optional
            Boolean indicating whether the model should behave in
            training mode or inference mode.

        Returns
        -------
        Tuple[tf.Tensor, tf.Tensor]
            A tuple containing:
            - BMU coordinates of shape (batch_size, 2)
            - Quantization error of shape (batch_size,)
        """
        return self.som_layer(inputs, training=training)

    def train_som(
            self,
            x_train: np.ndarray,
            epochs: int = 10,
            batch_size: int = 32,
            shuffle: bool = True,
            verbose: int = 1
    ) -> Dict[str, List[float]]:
        """
        Train the SOM on the given data, organizing it into a topological memory structure.

        Parameters
        ----------
        x_train : np.ndarray
            Training data of shape (n_samples, input_dim).
        epochs : int, optional
            Number of training epochs. Defaults to 10.
        batch_size : int, optional
            Number of samples per batch. Defaults to 32.
        shuffle : bool, optional
            Whether to shuffle the data before each epoch. Defaults to True.
        verbose : int, optional
            Verbosity level (0, 1, or 2). Defaults to 1.

        Returns
        -------
        Dict[str, List[float]]
            Training history containing 'quantization_error' per epoch.
        """
        # Set the max iterations
        total_iterations = epochs * (len(x_train) // batch_size)
        self.som_layer.max_iterations.assign(float(total_iterations))

        # Training history
        history = {'quantization_error': []}

        for epoch in range(epochs):
            start_time = time.time()
            epoch_quant_errors = []

            # Shuffle data if needed
            if shuffle:
                indices = np.arange(len(x_train))
                np.random.shuffle(indices)
                x_train_shuffled = x_train[indices]
            else:
                x_train_shuffled = x_train

            # Train in batches
            for i in range(0, len(x_train_shuffled), batch_size):
                x_batch = x_train_shuffled[i:i + batch_size]

                # Then flatten the data right before passing to the model
                x_batch = x_batch.reshape(x_batch.shape[0], -1)

                # Forward pass in training mode (weights are updated inside)
                _, quant_errors = self.som_layer(x_batch, training=True)
                epoch_quant_errors.append(tf.reduce_mean(quant_errors).numpy())

            # Compute average error for the epoch
            avg_error = np.mean(epoch_quant_errors)
            history['quantization_error'].append(avg_error)

            if verbose > 0 and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                end_time = time.time()
                logger.info(f"Epoch {epoch + 1}/{epochs} - Quantization Error: {avg_error:.6f} - "
                      f"Time: {end_time - start_time:.2f}s")

        return history

    def fit_class_prototypes(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Fit class prototypes by finding the most common BMU for each class.

        This method maps each class to its most representative location in the SOM grid,
        which can be used for classification and demonstrates how SOMs store and
        retrieve class-specific "memories".

        Parameters
        ----------
        x_train : np.ndarray
            Training data of shape (n_samples, input_dim).
        y_train : np.ndarray
            Training labels of shape (n_samples,).
        """
        # Find BMU for each sample
        bmu_indices, _ = self.som_layer(x_train, training=False)
        bmu_indices = bmu_indices.numpy()

        # Unique classes
        unique_classes = np.unique(y_train)

        # Create class to BMU mapping
        class_to_bmu = {}
        from collections import Counter

        for c in unique_classes:
            # FIXED: Use np.where to get the indices where y_train == c
            class_indices = np.where(y_train == c)[0]

            # Get BMUs for this class using the indices
            class_samples = bmu_indices[class_indices]

            # Convert to tuples for counting
            bmu_tuples = [tuple(bmu) for bmu in class_samples]

            # Find the most common BMU for this class
            bmu_counts = Counter(bmu_tuples)
            most_common_bmu = bmu_counts.most_common(1)[0][0]

            # Store the prototype
            class_to_bmu[c] = most_common_bmu

        self.class_prototypes = class_to_bmu

    def predict_classes(self, x_test: np.ndarray) -> np.ndarray:
        """
        Predict classes for test data using the fitted class prototypes.

        This demonstrates the associative memory retrieval function of SOMs,
        where the model recalls the class based on similarity to stored prototypes.

        Parameters
        ----------
        x_test : np.ndarray
            Test data of shape (n_samples, input_dim).

        Returns
        -------
        np.ndarray
            Predicted class labels of shape (n_samples,).

        Raises
        ------
        ValueError
            If class prototypes have not been fitted yet.
        """
        if self.class_prototypes is None:
            raise ValueError("Class prototypes have not been fitted. "
                             "Call fit_class_prototypes() first.")

        # Find BMU for each test sample
        bmu_indices, _ = self.som_layer(x_test, training=False)
        bmu_indices = bmu_indices.numpy()

        # Convert BMUs to tuples
        bmu_tuples = [tuple(bmu) for bmu in bmu_indices]

        # Prepare a mapping from BMU to class
        bmu_to_class = {bmu: c for c, bmu in self.class_prototypes.items()}

        # Predict classes - ensure we return integers
        predictions = []
        for bmu in bmu_tuples:
            # Find the closest prototype if exact BMU was not seen in training
            if bmu not in bmu_to_class:
                # Calculate distances to all prototypes
                distances = {c: np.sum((np.array(bmu) - np.array(prototype)) ** 2)
                             for c, prototype in self.class_prototypes.items()}
                # Find the closest
                closest_class = min(distances, key=distances.get)
                predictions.append(closest_class)  # Will be automatically cast to appropriate type
            else:
                predictions.append(bmu_to_class[bmu])

        # Convert to numpy array with same dtype as original labels to ensure compatibility
        return np.array(predictions)

    def visualize_som_grid(
            self,
            figsize: Tuple[int, int] = (10, 10),
            cmap: str = 'viridis',
            save_path: Optional[str] = None
    ) -> None:
        """
        Visualize the SOM grid by showing each neuron's weight vector.

        This visualization shows how the SOM has organized the input space
        into a 2D memory structure where each cell represents a prototype memory.

        Parameters
        ----------
        figsize : Tuple[int, int], optional
            Figure size (width, height) in inches. Defaults to (10, 10).
        cmap : str, optional
            Colormap to use for visualization. Defaults to 'viridis'.
        save_path : str, optional
            If provided, the visualization will be saved to this path.
        """
        weights = self.som_layer.get_weights_as_grid().numpy()
        grid_height, grid_width, input_dim = weights.shape

        plt.figure(figsize=figsize)

        # For MNIST or similar image data
        if input_dim in [28 * 28, 32 * 32, 64 * 64]:
            # Reshape each weight vector to an image
            side_length = int(np.sqrt(input_dim))

            # Create a grid to display all neurons
            full_grid = np.zeros((grid_height * side_length, grid_width * side_length))

            # Fill the grid with neuron weight vectors as images
            for i in range(grid_height):
                for j in range(grid_width):
                    neuron_weights = weights[i, j].reshape(side_length, side_length)
                    full_grid[i * side_length:(i + 1) * side_length,
                    j * side_length:(j + 1) * side_length] = neuron_weights

            plt.imshow(full_grid, cmap='gray')
            plt.title('SOM Memory Grid - Prototype Digit Memories')
            plt.axis('off')

        else:
            # For other types of data, show a simplified visualization
            # Calculate the norm of each weight vector
            weight_norms = np.linalg.norm(weights, axis=2)

            # Create a heatmap
            plt.imshow(weight_norms, cmap=cmap, interpolation='nearest')
            plt.colorbar(label='Weight Vector Norm')
            plt.title('SOM Grid - Weight Vector Norms')
            plt.xlabel('Grid Width')
            plt.ylabel('Grid Height')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)

        plt.show()

    def visualize_class_distribution(
            self,
            x_data: np.ndarray,
            y_data: np.ndarray,
            figsize: Tuple[int, int] = (10, 10),
            cmap: str = 'tab10',
            alpha: float = 0.5,
            marker_size: int = 100,
            save_path: Optional[str] = None
    ) -> None:
        """
        Visualize how different classes are distributed across the SOM grid.

        This shows how the SOM organizes memories by class, demonstrating
        its topological preservation properties.

        Parameters
        ----------
        x_data : np.ndarray
            Data samples of shape (n_samples, input_dim).
        y_data : np.ndarray
            Label for each sample of shape (n_samples,).
        figsize : Tuple[int, int], optional
            Figure size (width, height) in inches. Defaults to (10, 10).
        cmap : str, optional
            Colormap to use for different classes. Defaults to 'tab10'.
        alpha : float, optional
            Transparency level for markers. Defaults to 0.5.
        marker_size : int, optional
            Size of markers. Defaults to 100.
        save_path : str, optional
            If provided, the visualization will be saved to this path.
        """
        # Find BMU for each sample
        bmu_indices, _ = self.som_layer(x_data, training=False)
        bmu_indices = bmu_indices.numpy()

        plt.figure(figsize=figsize)

        # Create a scatter plot of BMUs colored by class
        # Convert one-hot encoded labels to class indices if needed
        if len(y_data.shape) > 1 and y_data.shape[1] > 1:
            y_data_indices = np.argmax(y_data, axis=1)
        else:
            y_data_indices = y_data

        unique_classes = np.unique(y_data_indices)
        colors = plt.cm.get_cmap(cmap, len(unique_classes))

        # Plot each class
        for i, c in enumerate(unique_classes):
            # FIXED: Use np.where to get the indices where y_data_indices == c
            class_indices = np.where(y_data_indices == c)[0]

            # Get BMUs for this class using the indices
            class_bmus = bmu_indices[class_indices]

            plt.scatter(class_bmus[:, 1], class_bmus[:, 0],
                        color=colors(i), label=f'Class {c}',
                        alpha=alpha, s=marker_size)

        # Add class prototypes if available
        if self.class_prototypes is not None:
            for c, bmu in self.class_prototypes.items():
                plt.scatter(bmu[1], bmu[0], color='black', marker='*',
                            s=marker_size * 2, label=f'Prototype {c}' if c == unique_classes[0] else "")

        plt.title('Class Distribution in SOM Memory Space')
        plt.xlabel('Grid Width')
        plt.ylabel('Grid Height')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)

        plt.show()

    def visualize_u_matrix(
            self,
            figsize: Tuple[int, int] = (10, 10),
            cmap: str = 'viridis_r',
            save_path: Optional[str] = None
    ) -> None:
        """
        Visualize the U-Matrix (Unified Distance Matrix) of the SOM.

        The U-Matrix visualizes the distances between neighboring neurons,
        which helps identify cluster boundaries in the SOM's memory space.

        Parameters
        ----------
        figsize : Tuple[int, int], optional
            Figure size (width, height) in inches. Defaults to (10, 10).
        cmap : str, optional
            Colormap to use for visualization. Defaults to 'viridis_r'.
        save_path : str, optional
            If provided, the visualization will be saved to this path.
        """
        weights = self.som_layer.get_weights_as_grid().numpy()
        grid_height, grid_width, _ = weights.shape

        # Create the U-Matrix
        u_matrix = np.zeros((grid_height, grid_width))

        # For each neuron, calculate average distance to its neighbors
        for i in range(grid_height):
            for j in range(grid_width):
                # Get the neuron's weight vector
                weight = weights[i, j]

                # Get the indices of the neighboring neurons
                neighbors = []
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue  # Skip the neuron itself
                        ni, nj = i + di, j + dj
                        if 0 <= ni < grid_height and 0 <= nj < grid_width:
                            neighbors.append((ni, nj))

                # Calculate the average distance to neighbors
                if neighbors:
                    neighbor_weights = np.array([weights[ni, nj] for ni, nj in neighbors])
                    distances = np.linalg.norm(weight - neighbor_weights, axis=1)
                    avg_distance = np.mean(distances)
                    u_matrix[i, j] = avg_distance

        plt.figure(figsize=figsize)
        plt.imshow(u_matrix, cmap=cmap, interpolation='nearest')
        plt.colorbar(label='Average Distance to Neighbors')
        plt.title('U-Matrix: Memory Cluster Boundaries')
        plt.xlabel('Grid Width')
        plt.ylabel('Grid Height')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)

        plt.show()

    def visualize_hit_histogram(
            self,
            x_data: np.ndarray,
            figsize: Tuple[int, int] = (10, 10),
            cmap: str = 'viridis',
            log_scale: bool = False,
            save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Visualize a hit histogram showing how many samples map to each neuron.

        This visualization shows which areas of the memory space are most
        frequently activated by the input data.

        Parameters
        ----------
        x_data : np.ndarray
            Data samples of shape (n_samples, input_dim).
        figsize : Tuple[int, int], optional
            Figure size (width, height) in inches. Defaults to (10, 10).
        cmap : str, optional
            Colormap to use for visualization. Defaults to 'viridis'.
        log_scale : bool, optional
            Whether to use log scale for the color mapping. Defaults to False.
        save_path : str, optional
            If provided, the visualization will be saved to this path.

        Returns
        -------
        np.ndarray
            The hit histogram array.
        """
        # Find BMU for each sample
        bmu_indices, _ = self.som_layer(x_data, training=False)
        bmu_indices = bmu_indices.numpy()

        # Create a histogram
        hit_histogram = np.zeros((self.som_layer.grid_height, self.som_layer.grid_width))

        for bmu in bmu_indices:
            hit_histogram[bmu[0], bmu[1]] += 1

        plt.figure(figsize=figsize)

        if log_scale and np.max(hit_histogram) > 0:
            # Add a small constant to avoid log(0)
            hit_histogram_log = np.log1p(hit_histogram)
            plt.imshow(hit_histogram_log, cmap=cmap, interpolation='nearest')
            plt.colorbar(label='Log(Hits + 1)')
            plt.title('Memory Activation Frequency (Log Scale)')
        else:
            plt.imshow(hit_histogram, cmap=cmap, interpolation='nearest')
            plt.colorbar(label='Number of Hits')
            plt.title('Memory Activation Frequency')

        plt.xlabel('Grid Width')
        plt.ylabel('Grid Height')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)

        plt.show()

        return hit_histogram

    def visualize_memory_recall(
            self,
            test_sample: np.ndarray,
            n_similar: int = 5,
            x_train: np.ndarray = None,
            y_train: np.ndarray = None,
            figsize: Tuple[int, int] = (15, 3),
            cmap: str = 'gray',
            save_path: Optional[str] = None
    ) -> None:
        """
        Visualize how the SOM recalls similar memories for a given test sample.

        This demonstrates the associative memory property of SOMs by showing
        the input sample and the memories it activates.

        Parameters
        ----------
        test_sample : np.ndarray
            A single test sample of shape (input_dim,).
        n_similar : int, optional
            Number of similar samples to retrieve. Defaults to 5.
        x_train : np.ndarray, optional
            Training data to find similar samples. Required if finding similar samples.
        y_train : np.ndarray, optional
            Labels for the training data.
        figsize : Tuple[int, int], optional
            Figure size (width, height) in inches. Defaults to (15, 3).
        cmap : str, optional
            Colormap to use for visualization. Defaults to 'gray'.
        save_path : str, optional
            If provided, the visualization will be saved to this path.
        """
        # Reshape test sample if needed
        if len(test_sample.shape) == 1:
            test_sample = test_sample.reshape(1, -1)

        # Find the BMU for the test sample
        bmu_indices, _ = self.som_layer(test_sample, training=False)
        bmu_index = bmu_indices[0].numpy()

        # Get the weights of the BMU (the memory prototype)
        bmu_weights = self.som_layer.weights_map[bmu_index[0], bmu_index[1]].numpy()

        # Check if we can find similar training samples
        similar_samples = []
        similar_labels = []

        if x_train is not None:
            # Find BMUs for all training samples
            train_bmu_indices, _ = self.som_layer(x_train, training=False)
            train_bmu_indices = train_bmu_indices.numpy()

            # Find samples that map to the same or neighboring BMUs
            distances = np.sum((train_bmu_indices - bmu_index) ** 2, axis=1)
            similar_indices = np.argsort(distances)[:n_similar]

            similar_samples = [x_train[i] for i in similar_indices]
            if y_train is not None:
                similar_labels = [y_train[i] for i in similar_indices]

        # Visualize
        plt.figure(figsize=figsize)

        # Determine if the data represents images
        if test_sample.shape[1] in [28 * 28, 32 * 32, 64 * 64]:
            is_image = True
            side_length = int(np.sqrt(test_sample.shape[1]))
        else:
            is_image = False

        # Plot the test sample
        plt.subplot(1, n_similar + 2, 1)
        if is_image:
            plt.imshow(test_sample.reshape(side_length, side_length), cmap=cmap)
            plt.title("Test Sample")
        else:
            plt.bar(range(len(test_sample[0])), test_sample[0])
            plt.title("Test Sample")
        plt.axis('off' if is_image else 'on')

        # Plot the BMU weights (memory prototype)
        plt.subplot(1, n_similar + 2, 2)
        if is_image:
            plt.imshow(bmu_weights.reshape(side_length, side_length), cmap=cmap)
            plt.title("Memory Prototype")
        else:
            plt.bar(range(len(bmu_weights)), bmu_weights)
            plt.title("Memory Prototype")
        plt.axis('off' if is_image else 'on')

        # Plot similar samples if available
        for i, sim_sample in enumerate(similar_samples):
            plt.subplot(1, n_similar + 2, i + 3)
            if is_image:
                plt.imshow(sim_sample.reshape(side_length, side_length), cmap=cmap)
                if y_train is not None:
                    plt.title(f"Similar {similar_labels[i]}")
                else:
                    plt.title(f"Similar {i + 1}")
            else:
                plt.bar(range(len(sim_sample)), sim_sample)
                plt.title(f"Similar {i + 1}")
            plt.axis('off' if is_image else 'on')

        plt.suptitle("SOM Memory Recall: Test Sample → Memory Prototype → Similar Samples")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)

        plt.show()

    def get_config(self) -> Dict[str, Any]:
        """
        Get configuration for the model.

        Returns
        -------
        Dict[str, Any]
            Configuration dictionary for the model.
        """
        config = super(SOMModel, self).get_config()
        config.update({
            'map_size': self.som_layer.map_size,
            'input_dim': self.som_layer.input_dim,
            'initial_learning_rate': self.som_layer.initial_learning_rate,
            'sigma': self.som_layer.sigma,
            'neighborhood_function': self.som_layer.neighborhood_function,
            'weights_initializer': self.som_layer.weights_initializer
        })
        return config