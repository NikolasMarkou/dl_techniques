"""
Self-Organizing Map for topological memory and pattern organization.
``SOMModel`` wraps :class:`SOM2dLayer` in a full training and
visualization framework.

A SOM is a 2D grid of neurons, each holding a prototype vector in the
input space. For each input, neurons compete on Euclidean distance; the
closest one, the Best Matching Unit, wins. The BMU's weights, and those
of neurons in a shrinking neighborhood around it, move toward the input:
``w_i(t+1) = w_i(t) + eta(t) * h_ci(t) * (x(t) - w_i(t))``, where
``eta(t)`` is a decaying learning rate and ``h_ci(t)`` is a Gaussian or
bubble function of grid distance from the BMU. Unlike a plain vector
quantizer, this neighborhood update makes nearby grid cells learn similar
prototypes, so the trained grid preserves the topology of the input space.

Input should be normalized. Class prototypes for classification are
learned separately, via :meth:`SOMModel.fit_class_prototypes`, not at
construction time.

References:
    - Kohonen, T. (1990). The self-organizing map. Proceedings of the
      IEEE, 78(9), 1464-1480.
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
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.som.model")
class SOMModel(keras.Model):
    """Self-organizing map wrapping :class:`SOM2dLayer` with training and
    visualization methods.

    Architecture:

    .. code-block:: text

        input [B, input_dim]
          │
          ▼
        SOM2dLayer (competitive learning)
          - find Best Matching Unit (BMU)
          - update BMU and neighbors ('training' only)
          │
          ▼
        (bmu_coords [B, 2], quant_errors [B])

    :meth:`train` trains the map. :meth:`fit_class_prototypes` then assigns
    each class a representative BMU location, and :meth:`predict_class`
    classifies new inputs by nearest-prototype BMU. The ``visualize_*``
    methods plot the learned grid, class distribution, U-matrix, hit
    histogram, and a memory-recall example.

    :param map_size: Grid dimensions ``(height, width)``.
    :type map_size: Tuple[int, int]
    :param input_dim: Dimensionality of input vectors.
    :type input_dim: int
    :param initial_learning_rate: Learning rate at the start of training.
    :type initial_learning_rate: float
    :param sigma: Initial neighborhood radius.
    :type sigma: float
    :param neighborhood_function: ``'gaussian'`` or ``'bubble'``.
    :type neighborhood_function: str
    :param weights_initializer: Initializer name or instance for the SOM weights.
    :type weights_initializer: Union[str, keras.initializers.Initializer]
    :param regularizer: Optional weight regularizer.
    :type regularizer: Optional[keras.regularizers.Regularizer]
    :param class_prototypes: Optional pre-computed class-to-BMU mapping;
        normally set by :meth:`fit_class_prototypes` instead.
    :type class_prototypes: Optional[Dict[int, Tuple[int, int]]]
    :param name: Optional model name.
    :type name: Optional[str]
    :param kwargs: Forwarded to ``keras.Model``.

    Input shape:
        ``(batch_size, input_dim)``, normalized.

    Output shape:
        ``(bmu_coords, quant_errors)`` — integer ``(batch_size, 2)`` and
        float ``(batch_size,)``.

    :ivar som_layer: The underlying :class:`SOM2dLayer`.
    :ivar class_prototypes: Dict mapping class labels to BMU positions.
    :ivar map_size: Grid dimensions.
    :ivar input_dim: Input dimensionality.

    Example:
        >>> som = SOMModel(map_size=(20, 20), input_dim=784, sigma=2.0)
        >>> history = som.train(x_train, epochs=10, batch_size=32)
        >>> som.fit_class_prototypes(x_train, y_train)
        >>> predictions = som.predict_class(x_test)
        >>> som.visualize_grid()
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

        if len(map_size) != 2 or any(dim <= 0 for dim in map_size):
            raise ValueError(f"map_size must be tuple of two positive integers, got {map_size}")
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if initial_learning_rate <= 0:
            raise ValueError(f"initial_learning_rate must be positive, got {initial_learning_rate}")
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        if neighborhood_function not in ("gaussian", "bubble"):
            raise ValueError(
                f"neighborhood_function must be 'gaussian' or 'bubble', "
                f"got {neighborhood_function!r}"
            )

        self.map_size = map_size
        self.input_dim = input_dim
        self.initial_learning_rate = initial_learning_rate
        self.sigma = sigma
        self.neighborhood_function = neighborhood_function
        self.weights_initializer = weights_initializer
        self.regularizer = regularizer
        self.class_prototypes = class_prototypes

        self._is_built = False

        # DECISION plan-2026-08-19T163559-499b6f0e/D-081: name is explicit, not
        # auto-generated, so two instances in one process don't disagree on weight paths. See decisions.md.
        self.som_layer = SOM2dLayer(
            map_size=map_size,
            input_dim=input_dim,
            initial_learning_rate=initial_learning_rate,
            sigma=sigma,
            neighborhood_function=neighborhood_function,
            weights_initializer=weights_initializer,
            regularizer=regularizer,
            name="som2d_layer"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the SOM layer explicitly, for correct weight init and serialization.

        :param input_shape: Shape of the input, typically ``(batch_size, input_dim)``.
        """
        if not self._is_built:
            self.som_layer.build(input_shape)
            self._is_built = True

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Compute Best Matching Units and quantization errors.

        Also updates neuron weights via competitive learning when ``training``
        is ``True``; otherwise only inference is performed.

        :param inputs: Input tensor, shape ``(batch_size, input_dim)``, normalized.
        :param training: Whether to update weights.
        :return: ``(bmu_coords, quant_errors)`` — integer ``(batch_size, 2)`` grid
            coordinates of each input's BMU, and float ``(batch_size,)`` Euclidean
            distances to those BMUs.
        """
        return self.som_layer(inputs, training=training)

    # DECISION plan-2026-08-14T233721-d4f9beb2/D-062: every inference-time BMU lookup
    # goes through here, chunked; an unbatched call on 60000 MNIST samples is ~75 GB. See decisions.md.
    def _bmu_indices_in_batches(
            self,
            x: np.ndarray,
            batch_size: int = 1024
    ) -> np.ndarray:
        """Return the BMU grid coordinates for ``x``, one chunk at a time.


        :param x: Samples of shape ``(n_samples, ...)``. Flattened to ``(n_samples, input_dim)`` before lookup, matching what :meth:`train` does per batch.
        :param batch_size: Rows per forward pass. Must be positive. The default 1024 keeps the ``(batch, num_neurons, input_dim)`` intermediate bounded regardless of dataset size.

        :return: Integer array of shape ``(n_samples, 2)`` with the grid coordinates of each sample's Best Matching Unit -- exactly what a single unbatched ``self.som_layer(x, training=False)`` would return.
        Failure mode: raises ``ValueError`` for a non-positive ``batch_size``.
        An empty ``x`` returns an empty ``(0, 2)`` array rather than raising.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        if x.shape[0] == 0:
            # Before the reshape: `np.reshape(0, -1)` cannot infer a dimension
            # from an empty array and raises.
            return np.zeros((0, 2), dtype="int32")
        flat = x.reshape(x.shape[0], -1)

        chunks = []
        for start in range(0, flat.shape[0], batch_size):
            chunk = ops.convert_to_tensor(flat[start:start + batch_size])
            bmu_indices, _ = self.som_layer(chunk, training=False)
            chunks.append(ops.convert_to_numpy(bmu_indices))
        return np.concatenate(chunks, axis=0)

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


        :param x_train: Training data array of shape (n_samples, input_dim) or (n_samples, height, width) for images. Automatically flattened if needed. Should be normalized to [0, 1] or similar range.
        :param epochs: Number of complete passes through the training data. More epochs allow finer organization but risk overfitting. Typical values: 10-100. Defaults to 10.
        :param batch_size: Number of samples per gradient update. Larger batches provide more stable updates but slower convergence. Typical values: 16-128. Defaults to 32.
        :param shuffle: Whether to shuffle training data before each epoch. Recommended for better convergence. Defaults to True.
        :param verbose: Verbosity level controlling logging frequency. 0: silent, 1: progress updates every 10%, 2: every epoch. Defaults to 1.

        :return: Dictionary containing training history with keys: - 'mean_quantization_error': List of average quantization errors per epoch. Lower values indicate better organization.
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

        # Iteration budget must count samples, not batches: SOMLayer.call does
        # iterations.assign_add(shape(inputs)[0]) each step.
        total_iterations = epochs * len(x_train)
        if total_iterations == 0:
            total_iterations = epochs  # Degenerate case: empty training set.
        self.som_layer.max_iterations.assign(float(total_iterations))

        # DECISION plan-2026-08-14T233721-d4f9beb2/D-061: reset the counter, not just the
        # budget, or a second train() call starts already at max_iterations and adapts nothing. See decisions.md.
        self.som_layer.iterations.assign(0.0)

        history = {'mean_quantization_error': []}

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


        :param x_train: Training data array of shape (n_samples, input_dim) or (n_samples, height, width) for images. Should be the same data used to train the SOM for accurate prototype fitting.
        :param y_train: Class labels array of shape (n_samples,) with integer class indices. Each unique value represents a distinct class.

        :raises ValueError: If y_train contains no valid samples or if model is not built.
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

        bmu_indices = self._bmu_indices_in_batches(x_train)

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


        :param x_test: Test data array of shape (n_samples, input_dim) or (n_samples, height, width) for images. Should use the same normalization as training data.

        :return: Array of predicted class labels with shape (n_samples,). Each value is an integer corresponding to the nearest class prototype.

        :raises ValueError: If class prototypes have not been fitted. Call fit_class_prototypes() before prediction.
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

        bmu_indices = self._bmu_indices_in_batches(x_test)

        # Convert BMUs to tuples for lookup
        bmu_tuples = [tuple(bmu) for bmu in bmu_indices]

        # Create reverse mapping from BMU to class
        bmu_to_class = {bmu: c for c, bmu in self.class_prototypes.items()}

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
            save_path: Optional[str] = None,
            show: bool = False
    ) -> "plt.Figure":
        """
        Visualize the learned SOM grid showing neuron prototype memories.

        This visualization displays the weight vectors of each neuron in the 2D
        grid, providing insight into how the SOM has organized the input space.
        For image data (square input dimensions), neurons are displayed as
        reconstructed images. For other data, weight vector norms are shown as
        a heatmap.

        The visualization reveals the topological organization where similar
        prototypes cluster together, demonstrating the memory structure.


        :param figsize: Tuple of (width, height) in inches for the figure size. Larger values provide more detail for large grids. Defaults to (10, 10).
        :param cmap: Matplotlib colormap name for visualization. Used for image grids ('gray' for grayscale images) or heatmaps ('viridis'). Defaults to 'viridis'.
        :param save_path: Optional file path to save the visualization. Supports formats like .png, .pdf, .svg. Defaults to None.
        :param show: If ``True``, call ``plt.show()`` before returning. Defaults to ``False``: this is library code, so it never blocks on a GUI unless the caller asks. When ``False`` the figure is closed, so repeated calls do not leak figures.

        :return: The ``matplotlib`` ``Figure``. It is already closed when ``show=False``; use ``show=True`` or ``save_path`` to render it.
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
        weights = ops.convert_to_numpy(self.som_layer.get_weights_as_grid())
        grid_height, grid_width, input_dim = weights.shape

        fig = plt.figure(figsize=figsize)

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

        # DECISION plan-2026-08-22T035419-a11304c8/D-051: never call plt.show()
        # unconditionally; on a headless host it leaks the figure instead of blocking. See decisions.md.
        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def visualize_class_distribution(
            self,
            x_data: np.ndarray,
            y_data: np.ndarray,
            figsize: Tuple[int, int] = (10, 10),
            cmap: str = 'tab10',
            alpha: float = 0.5,
            marker_size: int = 100,
            save_path: Optional[str] = None,
            show: bool = False
    ) -> "plt.Figure":
        """
        Visualize how different classes distribute across the SOM grid topology.

        This visualization maps training samples to their BMUs and colors them
        by class, revealing how the SOM organizes different classes topologically.
        Well-separated classes should occupy distinct grid regions, while similar
        classes may have overlapping territories.

        Class prototypes (if fitted) are overlaid as starred markers, showing the
        representative locations for classification.


        :param x_data: Data samples of shape (n_samples, input_dim) to visualize. Typically training data used to train the SOM.
        :param y_data: Class labels of shape (n_samples,). Can be integer labels or one-hot encoded. Automatically converted to class indices.
        :param figsize: Tuple of (width, height) in inches for the figure. Defaults to (10, 10).
        :param cmap: Matplotlib colormap for class colors. 'tab10' provides distinct colors for up to 10 classes. Use 'tab20' for more. Defaults to 'tab10'.
        :param alpha: Transparency of data points (0=transparent, 1=opaque). Lower values help visualize overlapping regions. Defaults to 0.5.
        :param marker_size: Size of scatter plot markers in points squared. Defaults to 100.
        :param save_path: Optional file path to save visualization. Defaults to None.
        :param show: If ``True``, call ``plt.show()`` before returning. Defaults to ``False``: this is library code, so it never blocks on a GUI unless the caller asks. When ``False`` the figure is closed, so repeated calls do not leak figures.

        :return: The ``matplotlib`` ``Figure``. It is already closed when ``show=False``; use ``show=True`` or ``save_path`` to render it.
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
        bmu_indices = self._bmu_indices_in_batches(x_data)

        fig = plt.figure(figsize=figsize)

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

        # DECISION plan-2026-08-22T035419-a11304c8/D-051: never call plt.show()
        # unconditionally; on a headless host it leaks the figure instead of blocking. See decisions.md.
        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def visualize_u_matrix(
            self,
            figsize: Tuple[int, int] = (10, 10),
            cmap: str = 'viridis_r',
            save_path: Optional[str] = None,
            show: bool = False
    ) -> "plt.Figure":
        """
        Visualize the Unified Distance Matrix (U-Matrix) revealing cluster boundaries.

        The U-Matrix displays the average distance between each neuron and its
        neighbors in weight space. High values (bright regions) indicate cluster
        boundaries where dissimilar patterns meet, while low values (dark regions)
        indicate coherent clusters of similar patterns.

        This visualization is essential for understanding the cluster structure
        learned by the SOM, revealing natural groupings in the data.


        :param figsize: Tuple of (width, height) in inches for the figure. Defaults to (10, 10).
        :param cmap: Matplotlib colormap where bright indicates boundaries. 'viridis_r' (reversed) makes boundaries bright. Can also use 'hot', 'plasma_r', etc. Defaults to 'viridis_r'.
        :param save_path: Optional file path to save visualization. Defaults to None.
        :param show: If ``True``, call ``plt.show()`` before returning. Defaults to ``False``: this is library code, so it never blocks on a GUI unless the caller asks. When ``False`` the figure is closed, so repeated calls do not leak figures.

        :return: The ``matplotlib`` ``Figure``. It is already closed when ``show=False``; use ``show=True`` or ``save_path`` to render it.
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
        fig = plt.figure(figsize=figsize)
        plt.imshow(u_matrix, cmap=cmap, interpolation='nearest')
        plt.colorbar(label='Average Distance to Neighbors')
        plt.title('U-Matrix: Memory Cluster Boundaries')
        plt.xlabel('Grid Width')
        plt.ylabel('Grid Height')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)

        # DECISION plan-2026-08-22T035419-a11304c8/D-051: never call plt.show()
        # unconditionally; on a headless host it leaks the figure instead of blocking. See decisions.md.
        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def visualize_hit_histogram(
            self,
            x_data: np.ndarray,
            figsize: Tuple[int, int] = (10, 10),
            cmap: str = 'viridis',
            log_scale: bool = False,
            save_path: Optional[str] = None,
            show: bool = False
    ) -> Tuple[np.ndarray, "plt.Figure"]:
        """
        Visualize activation frequency across the SOM grid (hit histogram).

        This visualization shows how many training samples map to each neuron,
        revealing which areas of the memory space are most active and which are
        underutilized. Uniform utilization indicates good map organization, while
        many "dead" neurons suggest overparameterization or poor initialization.


        :param x_data: Data samples of shape (n_samples, input_dim) to analyze. Typically the training data.
        :param figsize: Tuple of (width, height) in inches. Defaults to (10, 10).
        :param cmap: Matplotlib colormap for the heatmap. Defaults to 'viridis'.
        :param log_scale: Whether to use logarithmic color scaling. Useful when activation frequencies vary by orders of magnitude. Defaults to False.
        :param save_path: Optional file path to save visualization. Defaults to None.
        :param show: If ``True``, call ``plt.show()`` before returning. Defaults to ``False``: this is library code, so it never blocks on a GUI unless the caller asks. When ``False`` the figure is closed, so repeated calls do not leak figures.

        :return: A ``(hit_histogram, figure)`` tuple. ``hit_histogram`` is an array of shape (grid_height, grid_width) containing hit counts for each neuron, useful for quantitative analysis of map utilization; the ``matplotlib`` ``Figure`` is already closed when ``show=False``.
        Example:
            ```python
            # Basic hit histogram
            hits, fig = som.visualize_hit_histogram(x_train)

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
        bmu_indices = self._bmu_indices_in_batches(x_data)

        # Create histogram
        hit_histogram = np.zeros((self.som_layer.map_size[0], self.som_layer.map_size[1]))

        for bmu in bmu_indices:
            hit_histogram[bmu[0], bmu[1]] += 1

        # Visualize
        fig = plt.figure(figsize=figsize)

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

        # DECISION plan-2026-08-22T035419-a11304c8/D-051: never call plt.show()
        # unconditionally; on a headless host it leaks the figure instead of blocking. See decisions.md.
        if show:
            plt.show()
        else:
            plt.close(fig)

        return hit_histogram, fig

    def visualize_memory_recall(
            self,
            test_sample: np.ndarray,
            n_similar: int = 5,
            x_train: Optional[np.ndarray] = None,
            y_train: Optional[np.ndarray] = None,
            figsize: Tuple[int, int] = (15, 3),
            cmap: str = 'gray',
            save_path: Optional[str] = None,
            show: bool = False
    ) -> "plt.Figure":
        """
        Demonstrate associative memory recall for a query sample.

        This visualization shows how the SOM retrieves similar memories for a
        given input, demonstrating the associative memory property. It displays:
        1. The query sample (test input)
        2. The memory prototype (BMU weights) that best matches the query
        3. Similar training samples that map to nearby grid locations

        This illustrates how SOMs function as content-addressable memory where
        partial or noisy inputs can retrieve complete, similar memories.


        :param test_sample: Single test sample of shape (input_dim,) or (1, input_dim). This is the query to the associative memory.
        :param n_similar: Number of similar training samples to retrieve and display. Defaults to 5.
        :param x_train: Optional training data of shape (n_samples, input_dim) for finding similar samples. If None, only shows query and prototype. Defaults to None.
        :param y_train: Optional training labels for annotating similar samples. Defaults to None.
        :param figsize: Tuple of (width, height) in inches. Should be wide enough for all panels. Defaults to (15, 3).
        :param cmap: Matplotlib colormap for visualization. Use 'gray' for grayscale images. Defaults to 'gray'.
        :param save_path: Optional file path to save visualization. Defaults to None.
        :param show: If ``True``, call ``plt.show()`` before returning. Defaults to ``False``: this is library code, so it never blocks on a GUI unless the caller asks. When ``False`` the figure is closed, so repeated calls do not leak figures.

        :return: The ``matplotlib`` ``Figure``. It is already closed when ``show=False``; use ``show=True`` or ``save_path`` to render it.
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
            train_bmu_indices = self._bmu_indices_in_batches(x_train)

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
        fig = plt.figure(figsize=figsize)

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

        # DECISION plan-2026-08-22T035419-a11304c8/D-051: never call plt.show()
        # unconditionally; on a headless host it leaks the figure instead of blocking. See decisions.md.
        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration dictionary for model serialization.

        This method is called by Keras during model saving to get all necessary
        parameters for reconstructing the model. All constructor parameters must
        be included for proper serialization.

        :return: Dictionary containing all configuration parameters passed to __init__. Includes serialized initializers and regularizers.
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


        :param config: Configuration dictionary from get_config().
        :return: New instance of SOMModel with the saved configuration.
        """
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


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------


def create_som(
        map_size: Tuple[int, int] = (10, 10),
        input_dim: int = 784,
        initial_learning_rate: float = 0.1,
        sigma: float = 1.0,
        neighborhood_function: str = 'gaussian',
        weights_initializer: Union[str, keras.initializers.Initializer] = 'random_uniform',
        regularizer: Optional[keras.regularizers.Regularizer] = None,
        class_prototypes: Optional[Dict[int, Tuple[int, int]]] = None,
        **kwargs: Any
) -> SOMModel:
    """
    Create a Self-Organizing Map model.

    There is no ``MODEL_VARIANTS`` table and none was invented: a SOM is
    specified entirely by its grid extent and input dimension, both of which
    are continuous problem-specific quantities. Kohonen defines no named scale
    family, so this factory constructs the class directly.


    :param map_size: (height, width) of the neuron grid. Both entries must be positive.
    :param input_dim: Dimensionality of the input vectors. Must be positive.
    :param initial_learning_rate: Starting learning rate for the adaptation step. Must be positive; it decays over the configured iteration budget.
    :param sigma: Initial neighborhood radius. Must be positive.
    :param neighborhood_function: Either ``'gaussian'`` or ``'bubble'``.
    :param weights_initializer: Initializer for the neuron prototype vectors.
    :param regularizer: Optional regularizer applied to the prototype weights.
    :param class_prototypes: Optional pre-fitted class -> BMU coordinate mapping.
    :param kwargs: Additional arguments forwarded to the model constructor.

    :return: A configured SOMModel instance. Calling it returns ``(bmu_coordinates, quantization_errors)``.

    :raises ValueError: If any argument is outside its valid range.
    Example:
        >>> model = create_som(map_size=(4, 4), input_dim=8)
        >>> bmu, err = model(keras.random.normal((6, 8)), training=False)
        >>> tuple(bmu.shape)
        (6, 2)
    """
    return SOMModel(
        map_size=map_size,
        input_dim=input_dim,
        initial_learning_rate=initial_learning_rate,
        sigma=sigma,
        neighborhood_function=neighborhood_function,
        weights_initializer=weights_initializer,
        regularizer=regularizer,
        class_prototypes=class_prototypes,
        **kwargs
    )

# ------------------------------------------------------------------------