"""
Self-Organizing Map for topological memory and pattern organization.

This model provides a complete framework for a Self-Organizing Map (SOM),
an unsupervised neural network that learns to produce a low-dimensional,
discretized representation of high-dimensional input data. The SOM forms
a topological map where the spatial arrangement of neurons on a 2D grid
reflects the intrinsic relationships within the input data, making it a
powerful tool for clustering, visualization, and associative memory.

Architectural Overview:
    The core of the model is a 2D grid of neurons, where each neuron `i`
    maintains a prototype or weight vector `w_i` of the same dimension as
    the input space. The learning process is competitive and cooperative,
    unfolding in three key steps for each training input vector `x`:
    1.  **Competition**: All neurons on the grid compete to be the "winner"
        by calculating their distance to the input vector. The neuron
        whose weight vector is most similar to the input is designated
        the Best Matching Unit (BMU).
    2.  **Cooperation**: The BMU determines the spatial center of a
        neighborhood of excited neurons on the grid. The BMU and its
        topological neighbors are activated for learning.
    3.  **Adaptation**: The weight vectors of the activated neurons are
        updated to become more similar to the input vector. The magnitude
        of the update is dependent on the neuron's distance from the BMU
        within the neighborhood.

    This iterative process causes neighboring neurons in the grid to learn
    to represent similar input patterns, effectively organizing the map to
    preserve the topological structure of the input data space.

Foundational Mathematics and Intuition:
    The SOM algorithm is defined by two primary mathematical operations:
    finding the BMU and updating the weights.

    -   **BMU Selection (Competition)**: The BMU, denoted by index `c`, is
        found by minimizing the Euclidean distance between the input vector
        `x` and the weight vector `w_i` of each neuron `i`:
        `c = argmin_i || x(t) - w_i(t) ||`
        This step identifies the neuron that currently serves as the best
        prototype for the given input.

    -   **Weight Update (Adaptation)**: The weights of all neurons are then
        updated according to the rule:
        `w_i(t+1) = w_i(t) + η(t) * h_ci(t) * (x(t) - w_i(t))`
        -   `η(t)` is the learning rate, a monotonically decreasing
            function of time `t`. It controls the magnitude of weight
            changes, starting larger for coarse organization and becoming
            smaller for fine-tuning.
        -   `h_ci(t)` is the neighborhood function, which is the cornerstone
            of topological preservation. It depends on the grid distance
            between neuron `i` and the BMU `c`. A common choice is the
            Gaussian function:
            `h_ci(t) = exp(-||r_i - r_c||^2 / (2 * σ(t)^2))`
            where `r_i` and `r_c` are the grid coordinates and `σ(t)` is
            the neighborhood radius, which also decreases over time.

    The intuition behind the decaying neighborhood radius `σ(t)` is crucial.
    Initially, a large `σ` allows distant neurons to be influenced by the
    BMU, establishing a coarse, global order on the map. As `σ` shrinks,
    the updates become localized, allowing the map to fine-tune its
    representation of the local data topology.

References:
    -   Kohonen, T. (1990). The self-organizing map. *Proceedings of the
        IEEE*, 78(9), 1464-1480.
"""

import time
import keras
import numpy as np
from keras import ops
from collections import Counter
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union, List, Dict, Any

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.memory.som_2d_layer import SOM2dLayer

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class SOMModel(keras.Model):
    """
    Self-Organizing Map model implementing associative memory and topological learning.

    This model wraps a SOM2dLayer to provide a complete framework for unsupervised
    learning, pattern organization, and memory-based classification. It demonstrates
    how competitive learning creates topological maps where similar inputs activate
    nearby neurons, forming a structured memory system suitable for clustering,
    visualization, and associative recall tasks.

    **Intent**: Provide a complete, production-ready implementation of Self-Organizing
    Maps that can function as an associative memory system for data visualization,
    clustering, dimensionality reduction, and classification tasks while maintaining
    topological relationships in the learned representation.

    **Architecture**:
    ```
    Input(shape=[batch_size, input_dim])
            ↓
    SOM2dLayer: Competitive Learning
    - Find Best Matching Unit (BMU)
    - Update BMU and neighbors
    - Preserve topology
            ↓
    Output: (BMU coordinates, quantization error)
    - BMU coords: shape=[batch_size, 2]
    - Quant error: shape=[batch_size]
    ```

    **Memory Organization Process**:
    1. **Competition**: For each input, neurons compete to be the Best Matching Unit
    2. **Cooperation**: BMU and neighboring neurons update toward the input
    3. **Adaptation**: Learning rate and neighborhood decay over training
    4. **Result**: Topological map where similar patterns cluster together

    **Key Concepts**:
    - **Best Matching Unit (BMU)**: Neuron with weights closest to input
    - **Topological Preservation**: Similar inputs map to nearby grid locations
    - **Neighborhood Learning**: Updates extend beyond BMU to neighbors
    - **Competitive Learning**: Neurons compete to represent input patterns

    **Training Process**:
    The training follows these steps per sample:
    1. Find BMU (neuron most similar to input)
    2. Update BMU weights toward input
    3. Update neighbor weights based on distance from BMU
    4. Gradually reduce learning rate and neighborhood size

    **Memory Retrieval**:
    For associative recall:
    1. Present query pattern to trained SOM
    2. Find BMU that best matches query
    3. Retrieve stored prototype (BMU weights)
    4. Optionally find similar training samples

    **Applications**:
    - **Data Visualization**: Project high-dimensional data to 2D grid
    - **Clustering**: Discover natural groupings without supervision
    - **Dimensionality Reduction**: Create meaningful low-dimensional maps
    - **Anomaly Detection**: Identify patterns outside learned topology
    - **Classification**: Use topology for nearest-neighbor classification
    - **Memory Systems**: Model associative memory and pattern completion

    Args:
        map_size: Tuple of (height, width) defining the 2D grid dimensions.
            Larger maps provide finer-grained organization but require more
            training. Typical sizes range from (10, 10) to (50, 50).
        input_dim: Integer dimensionality of input vectors. Must match the
            feature dimension of training data.
        initial_learning_rate: Float learning rate at start of training.
            Controls how much neurons update toward inputs. Typical values
            range from 0.1 to 0.5. Defaults to 0.1.
        sigma: Float initial neighborhood radius. Determines how many
            neighboring neurons are updated along with the BMU. Larger values
            preserve more topology early in training. Defaults to 1.0.
        neighborhood_function: String specifying neighborhood kernel type.
            Options are 'gaussian' (smooth falloff) or 'bubble' (hard cutoff).
            Defaults to 'gaussian'.
        weights_initializer: Weight initialization strategy. Can be string
            name ('random_uniform', 'glorot_uniform') or Initializer instance.
            Defaults to 'random_uniform'.
        regularizer: Optional weight regularizer to prevent overfitting.
            Can be L1, L2, or custom Regularizer instance. Defaults to None.
        class_prototypes: Optional dictionary mapping class labels to BMU
            coordinates for classification. Typically computed via
            fit_class_prototypes() rather than provided directly. Defaults to None.
        name: Optional name for the model. Defaults to None.
        **kwargs: Additional keyword arguments for Model base class.

    Input shape:
        2D tensor with shape: `(batch_size, input_dim)`.
        Input data should be normalized for best results.

    Output shape:
        Tuple of two tensors:
        - BMU coordinates: `(batch_size, 2)` with integer grid positions
        - Quantization errors: `(batch_size,)` with distance to BMU

    Attributes:
        som_layer: The underlying SOM2dLayer performing competitive learning.
        class_prototypes: Dict mapping class labels to representative BMU positions.
        map_size: Stored grid dimensions (height, width).
        input_dim: Stored input dimensionality.

    Methods:
        train(): Train the SOM on data to organize the memory structure.
        fit_class_prototypes(): Map classes to grid locations for classification.
        predict_class(): Classify inputs using learned class prototypes.
        visualize_grid(): Display learned neuron prototypes as a 2D grid.
        visualize_class_distribution(): Show how classes map across the grid.
        visualize_u_matrix(): Display cluster boundaries in the learned map.
        visualize_hit_histogram(): Show activation frequency across neurons.
        visualize_memory_recall(): Demonstrate associative memory retrieval.

    Example:
        ```python
        # Create SOM for MNIST digits (28x28 = 784 dimensions)
        som = SOMModel(
            map_size=(20, 20),
            input_dim=784,
            initial_learning_rate=0.1,
            sigma=2.0
        )

        # Train on data to organize memory
        history = som.train(
            x_train,
            epochs=10,
            batch_size=32
        )

        # Fit class prototypes for classification
        som.fit_class_prototypes(x_train, y_train)

        # Classify new data
        predictions = som.predict_class(x_test)

        # Visualize the learned memory structure
        som.visualize_grid()
        som.visualize_class_distribution(x_train, y_train)

        # Demonstrate memory recall
        som.visualize_memory_recall(
            test_sample=x_test[0],
            x_train=x_train,
            y_train=y_train
        )
        ```

    Note:
        For Models, Keras automatically handles sub-layer building during the
        first forward pass. The build() method explicitly builds the SOM layer
        to ensure proper weight initialization and serialization support.

        Input data should typically be normalized (e.g., to [0, 1] or [-1, 1])
        for optimal convergence. The SOM is particularly effective for high-
        dimensional data where traditional visualization methods fail.

        Class prototypes must be fitted before classification can be performed.
        The model supports both supervised (with labels) and unsupervised (without
        labels) modes of operation.
    """

    def __init__(
            self,
            map_size: Tuple[int, int],
            input_dim: int,
            initial_learning_rate: float = 0.1,
            sigma: float = 1.0,
            neighborhood_function: str = 'gaussian',
            weights_initializer: Union[str, keras.initializers.Initializer] = 'random_uniform',
            regularizer: Optional[keras.regularizers.Regularizer] = None,
            class_prototypes: Optional[Dict[int, Tuple[int, int]]] = None,
            name: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        """Initialize the SOM model with configuration parameters."""
        super().__init__(name=name, **kwargs)

        # Validate inputs
        if len(map_size) != 2 or any(dim <= 0 for dim in map_size):
            raise ValueError(f"map_size must be tuple of two positive integers, got {map_size}")
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if initial_learning_rate <= 0:
            raise ValueError(f"initial_learning_rate must be positive, got {initial_learning_rate}")
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")

        # Store configuration for serialization and introspection
        self.map_size = map_size
        self.input_dim = input_dim
        self.initial_learning_rate = initial_learning_rate
        self.sigma = sigma
        self.neighborhood_function = neighborhood_function
        self.weights_initializer = weights_initializer
        self.regularizer = regularizer
        self.class_prototypes = class_prototypes

        # Track build state
        self._is_built = False

        # Create the SOM layer - instantiated in __init__, built in build()
        self.som_layer = SOM2dLayer(
            map_size=map_size,
            input_dim=input_dim,
            initial_learning_rate=initial_learning_rate,
            sigma=sigma,
            neighborhood_function=neighborhood_function,
            weights_initializer=weights_initializer,
            regularizer=regularizer
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the model by initializing the SOM layer.

        This method explicitly builds the SOM layer to ensure proper weight
        initialization and serialization support. Called automatically on
        first forward pass or can be called explicitly.

        Args:
            input_shape: Shape tuple of input data, typically (batch_size, input_dim).
        """
        if not self._is_built:
            # Explicitly build the SOM layer for proper serialization
            self.som_layer.build(input_shape)
            self._is_built = True

        # Always call parent build at the end
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Forward pass computing Best Matching Units and quantization errors.

        In training mode, this also updates neuron weights via competitive learning.
        In inference mode, only BMU coordinates and errors are computed.

        Args:
            inputs: Input tensor of shape (batch_size, input_dim). Should be
                normalized for best results.
            training: Boolean or None indicating training mode. When True,
                performs weight updates. When False or None, only performs
                inference. Defaults to None.

        Returns:
            Tuple containing:
            - bmu_coords: Integer tensor of shape (batch_size, 2) with grid
                coordinates of the Best Matching Unit for each input.
            - quant_errors: Float tensor of shape (batch_size,) with Euclidean
                distances between inputs and their BMUs (quantization error).
        """
        return self.som_layer(inputs, training=training)

    def train(
            self,
            x_train: np.ndarray,
            epochs: int = 10,
            batch_size: int = 32,
            shuffle: bool = True,
            verbose: int = 1
    ) -> Dict[str, List[float]]:
        """
        Train the SOM to organize input data into a topological memory structure.

        This method performs unsupervised learning via competitive learning,
        where neurons compete to represent input patterns and organize themselves
        to preserve topological relationships. The learning rate and neighborhood
        size decay over training for stable convergence.

        Args:
            x_train: Training data array of shape (n_samples, input_dim) or
                (n_samples, height, width) for images. Automatically flattened
                if needed. Should be normalized to [0, 1] or similar range.
            epochs: Number of complete passes through the training data.
                More epochs allow finer organization but risk overfitting.
                Typical values: 10-100. Defaults to 10.
            batch_size: Number of samples per gradient update. Larger batches
                provide more stable updates but slower convergence. Typical
                values: 16-128. Defaults to 32.
            shuffle: Whether to shuffle training data before each epoch.
                Recommended for better convergence. Defaults to True.
            verbose: Verbosity level controlling logging frequency.
                0: silent, 1: progress updates every 10%, 2: every epoch.
                Defaults to 1.

        Returns:
            Dictionary containing training history with keys:
            - 'mean_quantization_error': List of average quantization errors
                per epoch. Lower values indicate better organization.

        Example:
            ```python
            # Train with default settings
            history = som.train(x_train, epochs=10)

            # Train with custom settings
            history = som.train(
                x_train,
                epochs=50,
                batch_size=64,
                shuffle=True,
                verbose=2
            )

            # Plot training curve
            plt.plot(history['mean_quantization_error'])
            plt.xlabel('Epoch')
            plt.ylabel('Quantization Error')
            plt.show()
            ```

        Note:
            The quantization error represents how well inputs match their BMUs.
            Decreasing error indicates successful organization. Very low errors
            may indicate overfitting if the map size is too large.
        """
        # Ensure model is built before training
        if not self._is_built:
            sample_batch = x_train[:1].reshape(1, -1)
            self.build(sample_batch.shape)

        # Configure total training iterations for decay schedules
        total_iterations = epochs * (len(x_train) // batch_size)
        if total_iterations == 0 and len(x_train) > 0:
            total_iterations = epochs  # Handle case where batch_size > dataset size
        self.som_layer.max_iterations.assign(float(total_iterations))

        # Initialize training history
        history = {'mean_quantization_error': []}

        # Training loop over epochs
        for epoch in range(epochs):
            start_time = time.time()
            epoch_quant_errors = []

            # Shuffle data if requested for better convergence
            if shuffle:
                indices = np.arange(len(x_train))
                np.random.shuffle(indices)
                x_train_shuffled = x_train[indices]
            else:
                x_train_shuffled = x_train

            # Process data in batches
            for i in range(0, len(x_train_shuffled), batch_size):
                x_batch = x_train_shuffled[i:i + batch_size]
                if x_batch.shape[0] == 0:
                    continue

                # Flatten spatial dimensions if needed (e.g., images)
                x_batch = x_batch.reshape(x_batch.shape[0], -1)
                x_batch_tensor = ops.convert_to_tensor(x_batch)

                # Forward pass with training=True triggers weight updates
                _, quant_errors = self.som_layer(x_batch_tensor, training=True)

                # Track quantization error for monitoring
                avg_error = ops.mean(quant_errors)
                epoch_quant_errors.append(ops.convert_to_numpy(avg_error))

            # Compute and store epoch statistics
            avg_error = np.mean(epoch_quant_errors) if epoch_quant_errors else 0.0
            history['mean_quantization_error'].append(avg_error)

            # Log progress based on verbosity level
            if verbose > 0 and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                end_time = time.time()
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} - Mean Quantization Error: {avg_error:.6f} - "
                    f"Time: {end_time - start_time:.2f}s"
                )

        return history

    def fit_class_prototypes(
            self,
            x_train: np.ndarray,
            y_train: np.ndarray
    ) -> None:
        """
        Learn class-to-grid mappings by finding representative BMU for each class.

        This method analyzes how training samples of each class map to the SOM
        grid and identifies the most representative neuron (BMU) for each class.
        These prototypes enable classification of new samples based on topological
        similarity and demonstrate how SOMs store class-specific "memories".

        The process finds where each class naturally clusters in the trained map,
        providing interpretable class organization that respects the learned topology.

        Args:
            x_train: Training data array of shape (n_samples, input_dim) or
                (n_samples, height, width) for images. Should be the same data
                used to train the SOM for accurate prototype fitting.
            y_train: Class labels array of shape (n_samples,) with integer
                class indices. Each unique value represents a distinct class.

        Raises:
            ValueError: If y_train contains no valid samples or if model is not built.

        Example:
            ```python
            # Train SOM first
            som.train(x_train, epochs=10)

            # Fit class prototypes for classification
            som.fit_class_prototypes(x_train, y_train)

            # View learned mappings
            for class_id, bmu in som.class_prototypes.items():
                print(f"Class {class_id} → Grid position {bmu}")

            # Now classification is enabled
            predictions = som.predict_class(x_test)
            ```

        Note:
            This method should be called after training the SOM. The prototypes
            represent the most common grid location for each class, which may
            not capture all class variation if the class has multiple clusters.

            For multi-modal classes (classes with multiple clusters), consider
            using distance-based classification or allowing multiple prototypes
            per class in extended implementations.
        """
        # Ensure model is built
        if not self._is_built:
            sample_batch = x_train[:1].reshape(1, -1)
            self.build(sample_batch.shape)

        # Find BMU for each training sample
        x_train_tensor = ops.convert_to_tensor(x_train.reshape(x_train.shape[0], -1))
        bmu_indices, _ = self.som_layer(x_train_tensor, training=False)
        bmu_indices = ops.convert_to_numpy(bmu_indices)

        # Get unique class labels
        unique_classes = np.unique(y_train)

        # Map each class to its most representative BMU
        class_to_bmu = {}

        for c in unique_classes:
            # Find all samples belonging to this class
            class_mask = (y_train == c)
            class_bmus = bmu_indices[class_mask]

            if len(class_bmus) == 0:
                continue

            # Convert to tuples for counting
            bmu_tuples = [tuple(bmu) for bmu in class_bmus]

            # Find the most frequently activated BMU for this class
            bmu_counts = Counter(bmu_tuples)
            most_common_bmu = bmu_counts.most_common(1)[0][0]

            # Store as class prototype
            class_to_bmu[c] = most_common_bmu

        self.class_prototypes = class_to_bmu
        logger.info(f"Fitted {len(class_to_bmu)} class prototypes")

    def predict_class(
            self,
            x_test: np.ndarray
    ) -> np.ndarray:
        """
        Classify samples using fitted class prototypes and topological similarity.

        This method demonstrates associative memory retrieval where the SOM
        recalls class labels based on similarity to stored prototypes. Each test
        sample is mapped to its BMU, which is then matched to the nearest class
        prototype in the grid topology.

        The classification leverages the topological organization learned during
        training, making predictions based on location in the memory structure
        rather than direct feature matching.

        Args:
            x_test: Test data array of shape (n_samples, input_dim) or
                (n_samples, height, width) for images. Should use the same
                normalization as training data.

        Returns:
            Array of predicted class labels with shape (n_samples,). Each value
            is an integer corresponding to the nearest class prototype.

        Raises:
            ValueError: If class prototypes have not been fitted. Call
                fit_class_prototypes() before prediction.

        Example:
            ```python
            # After training and fitting prototypes
            predictions = som.predict_class(x_test)

            # Evaluate accuracy
            accuracy = np.mean(predictions == y_test)
            print(f"Classification accuracy: {accuracy:.2%}")

            # For samples without exact BMU match, uses nearest prototype
            # This provides robustness to novel patterns
            ```

        Note:
            If a test sample's BMU doesn't match any trained prototype exactly,
            the method finds the closest prototype by Euclidean distance in the
            2D grid. This provides graceful handling of out-of-distribution samples.

            Classification accuracy depends on how well the SOM's topology
            separates classes. Visualize class distribution to diagnose issues.
        """
        if self.class_prototypes is None:
            raise ValueError(
                "Class prototypes have not been fitted. "
                "Call fit_class_prototypes() first."
            )

        # Find BMU for each test sample
        x_test_tensor = ops.convert_to_tensor(x_test.reshape(x_test.shape[0], -1))
        bmu_indices, _ = self.som_layer(x_test_tensor, training=False)
        bmu_indices = ops.convert_to_numpy(bmu_indices)

        # Convert BMUs to tuples for lookup
        bmu_tuples = [tuple(bmu) for bmu in bmu_indices]

        # Create reverse mapping from BMU to class
        bmu_to_class = {bmu: c for c, bmu in self.class_prototypes.items()}

        # Predict class for each sample
        predictions = []
        for bmu in bmu_tuples:
            # Check for exact prototype match
            if bmu in bmu_to_class:
                predictions.append(bmu_to_class[bmu])
            else:
                # Find nearest prototype in grid space for novel BMUs
                distances = {
                    c: np.sum((np.array(bmu) - np.array(prototype)) ** 2)
                    for c, prototype in self.class_prototypes.items()
                }
                closest_class = min(distances, key=distances.get)
                predictions.append(closest_class)

        return np.array(predictions)

    def visualize_grid(
            self,
            figsize: Tuple[int, int] = (10, 10),
            cmap: str = 'viridis',
            save_path: Optional[str] = None
    ) -> None:
        """
        Visualize the learned SOM grid showing neuron prototype memories.

        This visualization displays the weight vectors of each neuron in the 2D
        grid, providing insight into how the SOM has organized the input space.
        For image data (square input dimensions), neurons are displayed as
        reconstructed images. For other data, weight vector norms are shown as
        a heatmap.

        The visualization reveals the topological organization where similar
        prototypes cluster together, demonstrating the memory structure.

        Args:
            figsize: Tuple of (width, height) in inches for the figure size.
                Larger values provide more detail for large grids.
                Defaults to (10, 10).
            cmap: Matplotlib colormap name for visualization. Used for image
                grids ('gray' for grayscale images) or heatmaps ('viridis').
                Defaults to 'viridis'.
            save_path: Optional file path to save the visualization. If None,
                displays interactively. Supports formats like .png, .pdf, .svg.
                Defaults to None.

        Example:
            ```python
            # Basic visualization
            som.visualize_grid()

            # For MNIST (28x28 images), shows digit prototypes
            som.visualize_grid(cmap='gray')

            # Save high-resolution version
            som.visualize_grid(
                figsize=(15, 15),
                cmap='gray',
                save_path='som_prototypes.png'
            )
            ```

        Note:
            For image data, the visualization is most informative when input_dim
            is a perfect square (e.g., 784 = 28×28). Non-square dimensions fall
            back to displaying weight vector norms as a heatmap.

            The grid shows smooth transitions between nearby neurons, confirming
            proper topological preservation. Abrupt changes suggest discontinuities
            in the learned representation.
        """
        # Get neuron weights as grid: (height, width, input_dim)
        weights = ops.convert_to_numpy(self.som_layer.get_weights_as_grid())
        grid_height, grid_width, input_dim = weights.shape

        plt.figure(figsize=figsize)

        # Check if input dimension is a perfect square (likely images)
        side_length_f = np.sqrt(input_dim)
        if side_length_f == int(side_length_f):
            # Visualize as image grid
            side_length = int(side_length_f)
            full_grid = np.zeros((grid_height * side_length, grid_width * side_length))

            # Tile neuron weights as images
            for i in range(grid_height):
                for j in range(grid_width):
                    neuron_weights = weights[i, j].reshape(side_length, side_length)
                    full_grid[
                        i * side_length:(i + 1) * side_length,
                        j * side_length:(j + 1) * side_length
                    ] = neuron_weights

            plt.imshow(full_grid, cmap='gray')
            plt.title('SOM Memory Grid - Prototype Memories')
            plt.axis('off')

        else:
            # For non-image data, show weight vector norms as heatmap
            weight_norms = np.linalg.norm(weights, axis=2)

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
        Visualize how different classes distribute across the SOM grid topology.

        This visualization maps training samples to their BMUs and colors them
        by class, revealing how the SOM organizes different classes topologically.
        Well-separated classes should occupy distinct grid regions, while similar
        classes may have overlapping territories.

        Class prototypes (if fitted) are overlaid as starred markers, showing the
        representative locations for classification.

        Args:
            x_data: Data samples of shape (n_samples, input_dim) to visualize.
                Typically training data used to train the SOM.
            y_data: Class labels of shape (n_samples,). Can be integer labels
                or one-hot encoded. Automatically converted to class indices.
            figsize: Tuple of (width, height) in inches for the figure.
                Defaults to (10, 10).
            cmap: Matplotlib colormap for class colors. 'tab10' provides
                distinct colors for up to 10 classes. Use 'tab20' for more.
                Defaults to 'tab10'.
            alpha: Transparency of data points (0=transparent, 1=opaque).
                Lower values help visualize overlapping regions.
                Defaults to 0.5.
            marker_size: Size of scatter plot markers in points squared.
                Defaults to 100.
            save_path: Optional file path to save visualization.
                Defaults to None.

        Example:
            ```python
            # Basic class distribution
            som.visualize_class_distribution(x_train, y_train)

            # With prototypes overlaid (requires fit_class_prototypes)
            som.fit_class_prototypes(x_train, y_train)
            som.visualize_class_distribution(x_train, y_train)

            # Customize appearance
            som.visualize_class_distribution(
                x_train, y_train,
                cmap='Set3',
                alpha=0.3,
                marker_size=50,
                save_path='class_distribution.png'
            )
            ```

        Note:
            This visualization is crucial for diagnosing classification issues.
            Overlapping classes in the grid indicate that the SOM cannot separate
            them, suggesting either insufficient training, inadequate map size,
            or inherent class similarity.

            The legend is placed outside the plot area to avoid obscuring data.
        """
        # Find BMU for each sample
        x_data_tensor = ops.convert_to_tensor(x_data.reshape(x_data.shape[0], -1))
        bmu_indices, _ = self.som_layer(x_data_tensor, training=False)
        bmu_indices = ops.convert_to_numpy(bmu_indices)

        plt.figure(figsize=figsize)

        # Convert one-hot encoded labels to class indices if needed
        if len(y_data.shape) > 1 and y_data.shape[1] > 1:
            y_data_indices = np.argmax(y_data, axis=1)
        else:
            y_data_indices = y_data

        # Get unique classes and color mapping
        unique_classes = np.unique(y_data_indices)
        colors = plt.cm.get_cmap(cmap, len(unique_classes))

        # Plot each class separately for legend
        for i, c in enumerate(unique_classes):
            # Get samples belonging to this class
            class_mask = (y_data_indices == c)
            class_bmus = bmu_indices[class_mask]

            plt.scatter(
                class_bmus[:, 1],  # x-coordinate (width)
                class_bmus[:, 0],  # y-coordinate (height)
                color=colors(i),
                label=f'Class {c}',
                alpha=alpha,
                s=marker_size
            )

        # Overlay class prototypes if available
        if self.class_prototypes is not None:
            for c, bmu in self.class_prototypes.items():
                plt.scatter(
                    bmu[1], bmu[0],  # (width, height)
                    color='black',
                    marker='*',
                    s=marker_size * 2,
                    edgecolors='white',
                    linewidths=1,
                    label='Prototype' if c == unique_classes[0] else "",
                    zorder=10  # Ensure prototypes are on top
                )

        plt.title('Class Distribution in SOM Memory Space')
        plt.xlabel('Grid Width')
        plt.ylabel('Grid Height')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)

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
        Visualize the Unified Distance Matrix (U-Matrix) revealing cluster boundaries.

        The U-Matrix displays the average distance between each neuron and its
        neighbors in weight space. High values (bright regions) indicate cluster
        boundaries where dissimilar patterns meet, while low values (dark regions)
        indicate coherent clusters of similar patterns.

        This visualization is essential for understanding the cluster structure
        learned by the SOM, revealing natural groupings in the data.

        Args:
            figsize: Tuple of (width, height) in inches for the figure.
                Defaults to (10, 10).
            cmap: Matplotlib colormap where bright indicates boundaries.
                'viridis_r' (reversed) makes boundaries bright. Can also use
                'hot', 'plasma_r', etc. Defaults to 'viridis_r'.
            save_path: Optional file path to save visualization.
                Defaults to None.

        Example:
            ```python
            # Basic U-Matrix visualization
            som.visualize_u_matrix()

            # Bright regions show cluster boundaries
            som.visualize_u_matrix(cmap='hot')

            # Save for publication
            som.visualize_u_matrix(
                figsize=(12, 12),
                cmap='plasma_r',
                save_path='umatrix.png'
            )
            ```

        Note:
            The U-Matrix complements class distribution visualizations by showing
            data structure without requiring labels. Sharp boundaries in the
            U-Matrix suggest clear cluster separation, while gradual transitions
            indicate continuous variation in the data space.

            For interpretation: dark valleys = clusters, bright ridges = boundaries.
        """
        # Get neuron weights as grid
        weights = ops.convert_to_numpy(self.som_layer.get_weights_as_grid())
        grid_height, grid_width, _ = weights.shape

        # Compute U-Matrix values
        u_matrix = np.zeros((grid_height, grid_width))

        for i in range(grid_height):
            for j in range(grid_width):
                # Current neuron's weight vector
                weight = weights[i, j]

                # Collect neighboring neurons (8-connectivity)
                neighbors = []
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue  # Skip the neuron itself
                        ni, nj = i + di, j + dj
                        if 0 <= ni < grid_height and 0 <= nj < grid_width:
                            neighbors.append((ni, nj))

                # Calculate average distance to neighbors
                if neighbors:
                    neighbor_weights = np.array([weights[ni, nj] for ni, nj in neighbors])
                    distances = np.linalg.norm(weight - neighbor_weights, axis=1)
                    avg_distance = np.mean(distances)
                    u_matrix[i, j] = avg_distance

        # Visualize
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
        Visualize activation frequency across the SOM grid (hit histogram).

        This visualization shows how many training samples map to each neuron,
        revealing which areas of the memory space are most active and which are
        underutilized. Uniform utilization indicates good map organization, while
        many "dead" neurons suggest overparameterization or poor initialization.

        Args:
            x_data: Data samples of shape (n_samples, input_dim) to analyze.
                Typically the training data.
            figsize: Tuple of (width, height) in inches. Defaults to (10, 10).
            cmap: Matplotlib colormap for the heatmap. Defaults to 'viridis'.
            log_scale: Whether to use logarithmic color scaling. Useful when
                activation frequencies vary by orders of magnitude. Defaults to False.
            save_path: Optional file path to save visualization. Defaults to None.

        Returns:
            Array of shape (grid_height, grid_width) containing hit counts for
            each neuron. Useful for quantitative analysis of map utilization.

        Example:
            ```python
            # Basic hit histogram
            hits = som.visualize_hit_histogram(x_train)

            # With log scale for large variance
            som.visualize_hit_histogram(x_train, log_scale=True)

            # Analyze utilization
            total_neurons = np.prod(hits.shape)
            active_neurons = np.sum(hits > 0)
            print(f"Active neurons: {active_neurons}/{total_neurons}")
            ```

        Note:
            "Dead" neurons (zero hits) may indicate the map is too large for the
            dataset or that training didn't converge. A well-trained SOM should
            have most neurons active, though some imbalance is normal due to
            uneven data distribution.

            The hit histogram helps diagnose training issues and choose appropriate
            map sizes for the dataset.
        """
        # Find BMU for each sample
        x_data_tensor = ops.convert_to_tensor(x_data.reshape(x_data.shape[0], -1))
        bmu_indices, _ = self.som_layer(x_data_tensor, training=False)
        bmu_indices = ops.convert_to_numpy(bmu_indices)

        # Create histogram
        hit_histogram = np.zeros((self.som_layer.map_size[0], self.som_layer.map_size[1]))

        for bmu in bmu_indices:
            hit_histogram[bmu[0], bmu[1]] += 1

        # Visualize
        plt.figure(figsize=figsize)

        if log_scale and np.max(hit_histogram) > 0:
            # Use log scale for better visualization of varying frequencies
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
        Demonstrate associative memory recall for a query sample.

        This visualization shows how the SOM retrieves similar memories for a
        given input, demonstrating the associative memory property. It displays:
        1. The query sample (test input)
        2. The memory prototype (BMU weights) that best matches the query
        3. Similar training samples that map to nearby grid locations

        This illustrates how SOMs function as content-addressable memory where
        partial or noisy inputs can retrieve complete, similar memories.

        Args:
            test_sample: Single test sample of shape (input_dim,) or (1, input_dim).
                This is the query to the associative memory.
            n_similar: Number of similar training samples to retrieve and display.
                Defaults to 5.
            x_train: Optional training data of shape (n_samples, input_dim) for
                finding similar samples. If None, only shows query and prototype.
                Defaults to None.
            y_train: Optional training labels for annotating similar samples.
                Defaults to None.
            figsize: Tuple of (width, height) in inches. Should be wide enough
                for all panels. Defaults to (15, 3).
            cmap: Matplotlib colormap for visualization. Use 'gray' for grayscale
                images. Defaults to 'gray'.
            save_path: Optional file path to save visualization. Defaults to None.

        Example:
            ```python
            # Basic memory recall (query + prototype only)
            som.visualize_memory_recall(x_test[0])

            # With similar samples from training set
            som.visualize_memory_recall(
                test_sample=x_test[0],
                x_train=x_train,
                y_train=y_train,
                n_similar=10
            )

            # Save for presentation
            som.visualize_memory_recall(
                x_test[42],
                x_train=x_train,
                y_train=y_train,
                figsize=(20, 4),
                save_path='memory_recall.png'
            )
            ```

        Note:
            This visualization is particularly powerful for image data where you
            can see how partial or noisy query images retrieve complete prototypes
            and similar examples, demonstrating the SOM's pattern completion
            capability.

            Similar samples are found by proximity in grid space (nearby BMUs),
            which respects the learned topology. This differs from direct feature
            similarity and may retrieve semantically related patterns.
        """
        # Reshape test sample if needed
        if len(test_sample.shape) == 1:
            test_sample = test_sample.reshape(1, -1)

        # Find BMU for the test sample
        test_sample_tensor = ops.convert_to_tensor(test_sample)
        bmu_indices, _ = self.som_layer(test_sample_tensor, training=False)
        bmu_index = ops.convert_to_numpy(bmu_indices[0])

        # Get the BMU's weight vector (memory prototype)
        bmu_weights = ops.convert_to_numpy(
            self.som_layer.weights_map[bmu_index[0], bmu_index[1]]
        )

        # Find similar training samples if provided
        similar_samples = []
        similar_labels = []

        if x_train is not None:
            # Find BMUs for all training samples
            x_train_tensor = ops.convert_to_tensor(x_train.reshape(x_train.shape[0], -1))
            train_bmu_indices, _ = self.som_layer(x_train_tensor, training=False)
            train_bmu_indices = ops.convert_to_numpy(train_bmu_indices)

            # Find samples with BMUs close to the query's BMU
            distances = np.sum((train_bmu_indices - bmu_index) ** 2, axis=1)
            similar_indices = np.argsort(distances)[:n_similar]

            similar_samples = [x_train[i] for i in similar_indices]
            if y_train is not None:
                similar_labels = [y_train[i] for i in similar_indices]

        # Determine if data represents images
        side_length_f = np.sqrt(test_sample.shape[1])
        is_image = (side_length_f == int(side_length_f))
        if is_image:
            side_length = int(side_length_f)

        # Create visualization
        plt.figure(figsize=figsize)

        # Plot test sample (query)
        plt.subplot(1, n_similar + 2, 1)
        if is_image:
            plt.imshow(test_sample.reshape(side_length, side_length), cmap=cmap)
            plt.title("Test Sample")
            plt.axis('off')
        else:
            plt.bar(range(len(test_sample[0])), test_sample[0])
            plt.title("Test Sample")

        # Plot BMU weights (memory prototype)
        plt.subplot(1, n_similar + 2, 2)
        if is_image:
            plt.imshow(bmu_weights.reshape(side_length, side_length), cmap=cmap)
            plt.title("Memory Prototype")
            plt.axis('off')
        else:
            plt.bar(range(len(bmu_weights)), bmu_weights)
            plt.title("Memory Prototype")

        # Plot similar samples if available
        for i, sim_sample in enumerate(similar_samples):
            plt.subplot(1, n_similar + 2, i + 3)
            if is_image:
                plt.imshow(sim_sample.reshape(side_length, side_length), cmap=cmap)
                if y_train is not None:
                    plt.title(f"Similar {similar_labels[i]}")
                else:
                    plt.title(f"Similar {i + 1}")
                plt.axis('off')
            else:
                plt.bar(range(len(sim_sample)), sim_sample)
                plt.title(f"Similar {i + 1}")

        plt.suptitle("SOM Memory Recall: Test Sample → Memory Prototype → Similar Samples")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)

        plt.show()

    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration dictionary for model serialization.

        This method is called by Keras during model saving to get all necessary
        parameters for reconstructing the model. All constructor parameters must
        be included for proper serialization.

        Returns:
            Dictionary containing all configuration parameters passed to __init__.
            Includes serialized initializers and regularizers.
        """
        config = super().get_config()

        # Ensure class prototypes use standard Python types for JSON serialization
        prototypes_for_config = None
        if self.class_prototypes is not None:
            prototypes_for_config = {
                int(k): v for k, v in self.class_prototypes.items()
            }

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
            'class_prototypes': prototypes_for_config,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'SOMModel':
        """
        Create model instance from configuration dictionary.

        This method is called by Keras during model loading to reconstruct the
        model from saved configuration. It deserializes complex objects and
        converts JSON-compatible types back to their original forms.

        Args:
            config: Configuration dictionary from get_config().

        Returns:
            New instance of SOMModel with the saved configuration.
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

        # JSON serialization converts tuples to lists and may stringify keys
        # Convert back to expected types
        prototypes_config = config.get("class_prototypes")
        if prototypes_config is not None:
            config["class_prototypes"] = {
                int(k): tuple(v) for k, v in prototypes_config.items()
            }

        return cls(**config)

# ------------------------------------------------------------------------