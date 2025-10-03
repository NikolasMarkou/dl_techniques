"""
A Keras model implementing a Self-Organizing Map (SOM) as an associative memory system.

This model demonstrates how Self-Organizing Maps can function as topological memory structures
that learn to organize and recall patterns in an unsupervised manner. The SOM creates a
low-dimensional (2D grid) representation of high-dimensional input data while preserving
topological relationships, making it an effective tool for data visualization, clustering,
and associative memory tasks.

**Core Functionality:**

The SOMModel wraps a SOM2dLayer and provides a complete framework for:

1. **Unsupervised Learning**: Trains on input data to create a topological map where
   similar inputs activate nearby neurons in the grid, forming clusters of related memories.

2. **Associative Memory**: Once trained, the model can recall stored "memories" by finding
   the Best Matching Unit (BMU) for new inputs and retrieving similar patterns from the
   learned representation.

3. **Classification**: Can be extended for supervised learning by fitting class prototypes
   to specific grid locations, enabling classification based on topological similarity.

4. **Memory Visualization**: Provides extensive visualization tools to understand how
   memories are organized, including grid visualization, class distribution maps,
   U-matrices for cluster boundaries, and memory recall demonstrations.

**Key Concepts:**

- **Best Matching Unit (BMU)**: The neuron in the grid that most closely matches an input
- **Topological Preservation**: Similar inputs map to nearby locations in the grid
- **Neighborhood Learning**: Updates not only the BMU but also neighboring neurons
- **Competitive Learning**: Neurons compete to represent different input patterns

**Training Process:**

1. For each input sample, find the BMU (neuron with weights closest to the input)
2. Update the BMU and its neighbors to become more similar to the input
3. Gradually reduce learning rate and neighborhood size over time
4. Result: A organized map where similar patterns cluster together

**Memory Retrieval:**

1. Present a query pattern to the trained SOM
2. Find the BMU that best matches the query
3. Retrieve the stored prototype (neuron weights) and similar training samples
4. Demonstrate associative recall by showing related memories

**Applications:**

- **Data Visualization**: Project high-dimensional data to 2D while preserving structure
- **Clustering**: Discover natural groupings in data without supervision
- **Dimensionality Reduction**: Create meaningful low-dimensional representations
- **Anomaly Detection**: Identify patterns that don't fit the learned topology
- **Classification**: Use topology for nearest-neighbor-like classification
- **Memory Systems**: Model associative memory and pattern completion

**Visualization Capabilities:**

- **Grid Visualization**: Shows learned prototypes as a 2D grid (especially useful for image data)
- **Class Distribution**: Maps how different classes are distributed across the topology
- **U-Matrix**: Reveals cluster boundaries and data structure
- **Hit Histogram**: Shows which areas of the memory space are most active
- **Memory Recall**: Demonstrates how the SOM retrieves similar memories for a query

**Example Use Cases:**

1. **MNIST Digit Organization**: Train on handwritten digits to see how the SOM organizes
   digit prototypes topologically, with similar digits (e.g., 6 and 8) located near each other.

2. **Customer Segmentation**: Organize customer data to find natural market segments
   while preserving relationships between similar customer types.

3. **Color Organization**: Learn color relationships and create smooth transitions
   across the color space in the 2D grid.

4. **Gene Expression Analysis**: Organize genes or samples based on expression patterns
   while maintaining biological relationships.

**Memory System Analogy:**

Think of the SOM as a library where books (data samples) are organized on shelves (grid neurons)
such that similar books are placed near each other. When you want to find a book (query),
you go to the most relevant shelf (BMU) and can also browse nearby shelves for related content.
The librarian (training process) learns this organization by repeatedly placing books in
locations that make sense based on their content similarity.

This implementation extends the basic SOM concept by adding classification capabilities
through class prototypes and comprehensive visualization tools for understanding the
learned memory structure, making it particularly useful for educational purposes and
research into associative memory systems.
"""

import time
import keras
import numpy as np
from keras import ops
from collections import Counter
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union, List, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.memory.som_2d_layer import SOM2dLayer

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
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
    weights_initializer : Union[str, keras.initializers.Initializer], optional
        Initialization method for weights. Defaults to 'random'.
    regularizer : Optional[keras.regularizers.Regularizer], optional
        Regularizer function applied to the weights. Defaults to None.
    name : str, optional
        Name of the model. Defaults to None.
    **kwargs : Any
        Additional keyword arguments for the base Model class.
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
            **kwargs: Any
    ) -> None:
        """Initialize the SOM model."""
        super().__init__(name=name, **kwargs)

        # Store configuration for serialization
        self.map_size = map_size
        self.input_dim = input_dim
        self.initial_learning_rate = initial_learning_rate
        self.sigma = sigma
        self.neighborhood_function = neighborhood_function
        self.weights_initializer = weights_initializer
        self.regularizer = regularizer

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
        self._is_built = False

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """
        Build the model layers.

        Parameters
        ----------
        input_shape : Tuple[int, ...]
            Shape of the input tensor.
        """
        if not self._is_built:
            # Build the SOM layer
            self.som_layer.build(input_shape)
            self._is_built = True
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Forward pass for the SOM model.

        Parameters
        ----------
        inputs : keras.KerasTensor
            Input tensor of shape (batch_size, input_dim).
        training : bool, optional
            Boolean indicating whether the model should behave in
            training mode or inference mode.

        Returns
        -------
        Tuple[keras.KerasTensor, keras.KerasTensor]
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
        # Ensure the model is built
        if not self._is_built:
            sample_batch = x_train[:1].reshape(1, -1)
            self.build(sample_batch.shape)

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

                # Flatten the data right before passing to the model
                x_batch = x_batch.reshape(x_batch.shape[0], -1)
                x_batch_tensor = ops.convert_to_tensor(x_batch)

                # Forward pass in training mode (weights are updated inside)
                _, quant_errors = self.som_layer(x_batch_tensor, training=True)

                # Use keras.ops for mean calculation
                avg_error = ops.mean(quant_errors)
                epoch_quant_errors.append(ops.convert_to_numpy(avg_error))

            # Compute average error for the epoch
            avg_error = np.mean(epoch_quant_errors)
            history['quantization_error'].append(avg_error)

            if verbose > 0 and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                end_time = time.time()
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} - Quantization Error: {avg_error:.6f} - "
                    f"Time: {end_time - start_time:.2f}s"
                )

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
        # Ensure the model is built
        if not self._is_built:
            sample_batch = x_train[:1].reshape(1, -1)
            self.build(sample_batch.shape)

        # Convert to tensor and find BMU for each sample
        x_train_tensor = ops.convert_to_tensor(x_train.reshape(x_train.shape[0], -1))
        bmu_indices, _ = self.som_layer(x_train_tensor, training=False)
        bmu_indices = ops.convert_to_numpy(bmu_indices)

        # Unique classes
        unique_classes = np.unique(y_train)

        # Create class to BMU mapping
        class_to_bmu = {}

        for c in unique_classes:
            # Get indices where y_train == c
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
            raise ValueError(
                "Class prototypes have not been fitted. "
                "Call fit_class_prototypes() first."
            )

        # Convert to tensor and find BMU for each test sample
        x_test_tensor = ops.convert_to_tensor(x_test.reshape(x_test.shape[0], -1))
        bmu_indices, _ = self.som_layer(x_test_tensor, training=False)
        bmu_indices = ops.convert_to_numpy(bmu_indices)

        # Convert BMUs to tuples
        bmu_tuples = [tuple(bmu) for bmu in bmu_indices]

        # Prepare a mapping from BMU to class
        bmu_to_class = {bmu: c for c, bmu in self.class_prototypes.items()}

        # Predict classes
        predictions = []
        for bmu in bmu_tuples:
            # Find the closest prototype if exact BMU was not seen in training
            if bmu not in bmu_to_class:
                # Calculate distances to all prototypes
                distances = {
                    c: np.sum((np.array(bmu) - np.array(prototype)) ** 2)
                    for c, prototype in self.class_prototypes.items()
                }
                # Find the closest
                closest_class = min(distances, key=distances.get)
                predictions.append(closest_class)
            else:
                predictions.append(bmu_to_class[bmu])

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
        weights = ops.convert_to_numpy(self.som_layer.get_weights_as_grid())
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
                    full_grid[
                    i * side_length:(i + 1) * side_length,
                    j * side_length:(j + 1) * side_length
                    ] = neuron_weights

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
        # Convert to tensor and find BMU for each sample
        x_data_tensor = ops.convert_to_tensor(x_data.reshape(x_data.shape[0], -1))
        bmu_indices, _ = self.som_layer(x_data_tensor, training=False)
        bmu_indices = ops.convert_to_numpy(bmu_indices)

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
            # Get indices where y_data_indices == c
            class_indices = np.where(y_data_indices == c)[0]

            # Get BMUs for this class using the indices
            class_bmus = bmu_indices[class_indices]

            plt.scatter(
                class_bmus[:, 1], class_bmus[:, 0],
                color=colors(i), label=f'Class {c}',
                alpha=alpha, s=marker_size
            )

        # Add class prototypes if available
        if self.class_prototypes is not None:
            for c, bmu in self.class_prototypes.items():
                plt.scatter(
                    bmu[1], bmu[0], color='black', marker='*',
                    s=marker_size * 2,
                    label=f'Prototype {c}' if c == unique_classes[0] else ""
                )

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
        weights = ops.convert_to_numpy(self.som_layer.get_weights_as_grid())
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
        # Convert to tensor and find BMU for each sample
        x_data_tensor = ops.convert_to_tensor(x_data.reshape(x_data.shape[0], -1))
        bmu_indices, _ = self.som_layer(x_data_tensor, training=False)
        bmu_indices = ops.convert_to_numpy(bmu_indices)

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
            x_train: Optional[np.ndarray] = None,
            y_train: Optional[np.ndarray] = None,
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

        # Convert to tensor and find the BMU for the test sample
        test_sample_tensor = ops.convert_to_tensor(test_sample)
        bmu_indices, _ = self.som_layer(test_sample_tensor, training=False)
        bmu_index = ops.convert_to_numpy(bmu_indices[0])

        # Get the weights of the BMU (the memory prototype)
        bmu_weights = ops.convert_to_numpy(
            self.som_layer.weights_map[bmu_index[0], bmu_index[1]]
        )

        # Check if we can find similar training samples
        similar_samples = []
        similar_labels = []

        if x_train is not None:
            # Convert to tensor and find BMUs for all training samples
            x_train_tensor = ops.convert_to_tensor(x_train.reshape(x_train.shape[0], -1))
            train_bmu_indices, _ = self.som_layer(x_train_tensor, training=False)
            train_bmu_indices = ops.convert_to_numpy(train_bmu_indices)

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
        config = super().get_config()
        config.update({
            'map_size': self.map_size,
            'input_dim': self.input_dim,
            'initial_learning_rate': self.initial_learning_rate,
            'sigma': self.sigma,
            'neighborhood_function': self.neighborhood_function,
            'weights_initializer': keras.initializers.serialize(
                keras.initializers.get(self.weights_initializer)
            ),
            'regularizer': keras.regularizers.serialize(self.regularizer) if self.regularizer else None,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'SOMModel':
        """
        Create a model from its configuration.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary.

        Returns
        -------
        SOMModel
            New instance of the model.
        """
        # Deserialize complex objects
        if config.get('weights_initializer'):
            config['weights_initializer'] = keras.initializers.deserialize(
                config['weights_initializer']
            )
        if config.get('regularizer'):
            config['regularizer'] = keras.regularizers.deserialize(
                config['regularizer']
            )

        return cls(**config)

# ---------------------------------------------------------------------