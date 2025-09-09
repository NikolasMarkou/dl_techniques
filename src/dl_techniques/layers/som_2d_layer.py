"""
Self-Organizing Map (SOM) 2D layer implementation for Keras.

This file should be saved as: dl_techniques/layers/som_2d_layer.py

A Self-Organizing Map is an unsupervised neural network that performs dimensionality
reduction through competitive learning and topological organization. The layer maps
high-dimensional input data onto a lower-dimensional grid of neurons, preserving
topological relationships between input patterns.

SOMs function as associative memory structures where:
  - Each neuron represents a prototype vector in the input space
  - Competitive learning determines which neuron "wins" for each input
  - Neighborhood functions create topological organization
  - Similar inputs activate nearby regions in the grid
  - The grid preserves distance relationships from the input space

Learning Process:
----------------
1. For each input vector:
   a. Find the Best Matching Unit (BMU) - the neuron with weights closest to the input
   b. Update the BMU and its neighbors to move them closer to the input
   c. The neighborhood size and learning rate decrease over time

Grid Structure:
--------------
The SOM organizes neurons in a 2D grid:

    y
    ^
    |  (0,0)-----(0,1)-----(0,2)---...
    |    |         |         |
    |    |         |         |
    |  (1,0)-----(1,1)-----(1,2)---...
    |    |         |         |
    |    |         |         |
    |  (2,0)-----(2,1)-----(2,2)---...
    |    :         :         :
    |
    +------------------------------------> x

Each neuron (i,j) has a weight vector w_ij with the same dimensionality as the input.

Competitive Learning:
-------------------
For an input x, the best matching unit (BMU) is the neuron with weight vector w_bmu that
minimizes the Euclidean distance:

    BMU = argmin_ij ||x - w_ij||^2

Neighborhood Functions:
---------------------
1. Gaussian neighborhood:
   h_ij = exp(-d^2/(2*sigma^2))
   where d = distance from neuron (i,j) to the BMU

2. Bubble neighborhood:
   h_ij = 1 if d <= sigma, 0 otherwise
   where d = distance from neuron (i,j) to the BMU

Weight Update:
------------
For each neuron (i,j), the weight update is:

    w_ij(t+1) = w_ij(t) + α(t) * h_ij(t) * (x - w_ij(t))

Where:
    - α(t) is the learning rate at time t
    - h_ij(t) is the neighborhood function value for neuron (i,j) at time t
    - Both α(t) and h_ij(t) decrease over time, leading to fine-tuning

Applications:
-----------
- Dimensionality reduction and visualization
- Clustering and pattern recognition
- Feature extraction and organization
- Associative memory systems
- Anomaly detection
- Topological data analysis

Memory Properties:
----------------
SOMs function as content-addressable memory where:
- Each neuron stores a prototype pattern
- Similar inputs activate the same or nearby memory locations
- The topology preserves relationships between stored memories
- Retrieval works by finding the best matching memory unit
- The memory has generalization capabilities for novel inputs

Notes
-----
- Weight vectors are updated manually during training, not through backpropagation
- Both learning rate and neighborhood radius decrease over training iterations
- The SOM preserves topological relationships, not exact distances
- The output provides BMU coordinates (grid position) and quantization error

Examples
--------
>>> # Create a 10x10 SOM for 784-dimensional MNIST digits
>>> som_layer = SOM2dLayer(map_size=(10, 10), input_dim=784,
...                     initial_learning_rate=0.5, sigma=2.0)
>>>
>>> # Forward pass (finding BMUs)
>>> bmu_indices, quant_errors = som_layer(input_data, training=True)
>>>
>>> # The grid can be visualized to see the organization of the memory space
>>> weights_grid = som_layer.get_weights_as_grid()

References
----------
[1] Kohonen, T. (1982). Self-organized formation of topologically correct feature
       maps. Biological Cybernetics, 43(1), 59-69.

[2] Kohonen, T. (1990). The self-organizing map. Proceedings of the IEEE, 78(9),
       1464-1480.

[3] Kohonen, T. (2001). Self-Organizing Maps. Springer Series in Information
       Sciences, Vol. 30, Springer, Berlin.

[4] Ultsch, A., & Siemon, H. P. (1990). Kohonen's Self Organizing Feature Maps for
       Exploratory Data Analysis. In Proceedings of International Neural Networks
       Conference (INNC).

[5] Vesanto, J., & Alhoniemi, E. (2000). Clustering of the self-organizing map.
       IEEE Transactions on Neural Networks, 11(3), 586-600.
"""

import keras
from typing import Tuple, Optional, Union, Dict, Any, Callable

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .som_nd_layer import SOMLayer

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class SOM2dLayer(SOMLayer):
    """
    2D Self-Organizing Map (SOM) layer for competitive learning and topological data organization.

    This layer is a specialized version of the general N-Dimensional SOMLayer,
    specifically optimized for creating 2D memory grids. It maps high-dimensional
    input data onto a 2D discretized grid, preserving topological properties of
    the input space through competitive learning and neighborhood-based weight updates.

    **Intent**: Provide a convenient 2D interface for self-organizing maps while
    leveraging all the robust functionality of the general SOMLayer base class,
    including vectorized operations, multiple neighborhood functions, and proper
    Keras 3 serialization support.

    **Architecture**:
    ```
    Input(shape=[batch, input_dim])
             ↓
    Find BMU: argmin_ij ||x - w_ij||²
             ↓
    Compute: h_ij = neighborhood_function(d_ij, σ)
             ↓
    Update: w_ij ← w_ij + α·h_ij·(x - w_ij)
             ↓
    Output: BMU_indices(shape=[batch, 2]), quantization_errors(shape=[batch,])
    ```

    **Mathematical Operations**:
    1. **Distance Computation**: d²_ij = ||x - w_ij||²
    2. **BMU Selection**: BMU = argmin_ij d²_ij
    3. **Neighborhood**: h_ij = exp(-d²_grid/(2σ²)) for Gaussian
    4. **Weight Update**: Δw_ij = α·h_ij·(x - w_ij)

    The 2D grid enables intuitive visualization and interpretation of the learned
    topological organization, making it ideal for data exploration and clustering tasks.

    Args:
        map_size: Tuple[int, int], the shape of the 2D SOM grid (height, width).
            Must contain exactly 2 positive integers. Controls memory capacity
            and topological resolution. Larger grids provide finer organization
            but require more computation and memory.
        input_dim: int, dimensionality of the input data. Must be positive.
            Each neuron will have a weight vector of this dimensionality.
        initial_learning_rate: float, optional, initial learning rate for weight
            updates. Controls adaptation speed during early training. Should be
            in range (0, 1]. Defaults to 0.1.
        decay_function: Callable, optional, learning rate decay function that takes
            current iteration and returns decay multiplier. If None, uses constant
            learning rate. Defaults to None.
        sigma: float, optional, initial neighborhood radius in grid coordinates.
            Controls the size of the influence region around the BMU. Should be
            positive, typically 1-5 for small grids. Defaults to 1.0.
        neighborhood_function: str, optional, type of neighborhood function to use.
            Either 'gaussian' for smooth decay or 'bubble' for hard cutoff.
            Defaults to 'gaussian'.
        weights_initializer: Union[str, keras.initializers.Initializer], optional,
            initialization method for neuron weights. Can be string name or
            initializer instance. Defaults to 'random_uniform'.
        regularizer: keras.regularizers.Regularizer, optional, regularizer function
            applied to the neuron weights during training. Helps prevent overfitting.
            Defaults to None.
        name: str, optional, name of the layer for identification in model summaries
            and debugging. Defaults to None.
        **kwargs: Any, additional keyword arguments for the base Layer class.

    Input shape:
        2D tensor with shape: `(batch_size, input_dim)`.
        Each sample represents a high-dimensional data point to be mapped.

    Output shape:
        Tuple of two tensors:
        - BMU indices: 2D tensor with shape `(batch_size, 2)` containing grid
          coordinates (row, col) of the best matching unit for each input.
        - Quantization errors: 1D tensor with shape `(batch_size,)` containing
          the Euclidean distance between each input and its BMU.

    Attributes:
        map_size: Tuple[int, int], the 2D grid dimensions (height, width).
        grid_shape: Tuple[int, int], alias for map_size (inherited from SOMLayer).
        input_dim: int, dimensionality of input vectors.
        initial_learning_rate: float, base learning rate for training.
        decay_function: Callable or None, learning rate decay function.
        sigma: float, neighborhood radius parameter.
        neighborhood_function: str, type of neighborhood function ('gaussian'/'bubble').
        weights: keras.Variable, neuron weight matrix of shape (height*width, input_dim).

    Example:
        ```python
        # Create a 10x10 SOM for MNIST digit clustering (784-dimensional)
        som = SOM2dLayer(
            map_size=(10, 10),
            input_dim=784,
            initial_learning_rate=0.5,
            sigma=2.0,
            neighborhood_function='gaussian'
        )

        # Build model for training
        inputs = keras.Input(shape=(784,))
        bmu_indices, quant_errors = som(inputs)
        model = keras.Model(inputs, [bmu_indices, quant_errors])

        # Training requires custom loop since SOM uses competitive learning
        for epoch in range(100):
            for batch in dataset:
                with tf.GradientTape() as tape:
                    bmu_coords, errors = model(batch, training=True)
                    # Custom SOM training logic here

        # Visualize the organized feature map
        weights_grid = som.get_weights_as_grid()  # Shape: (10, 10, 784)

        # For smaller input dimensions, can visualize weight patterns
        if input_dim == 2:
            import matplotlib.pyplot as plt
            plt.scatter(weights_grid[:,:,0], weights_grid[:,:,1])
            plt.title("SOM Weight Distribution")

        # Extract BMU positions for clustering analysis
        bmu_positions, _ = som(test_data)
        cluster_assignments = bmu_positions[:, 0] * 10 + bmu_positions[:, 1]
        ```

    Note:
        This layer inherits all core functionality from SOMLayer including vectorized
        operations, proper weight management, and Keras serialization support. The
        2D specialization provides the convenience of `map_size` parameter and
        `get_weights_as_grid()` method for visualization and backward compatibility.

        Training requires custom loops since SOMs use competitive learning rather
        than gradient descent. The layer provides the building blocks for SOM
        training but doesn't implement the training loop itself.
    """

    def __init__(
            self,
            map_size: Tuple[int, int],
            input_dim: int,
            initial_learning_rate: float = 0.1,
            decay_function: Optional[Callable] = None,
            sigma: float = 1.0,
            neighborhood_function: str = 'gaussian',
            weights_initializer: Union[str, keras.initializers.Initializer] = 'random_uniform',
            regularizer: Optional[keras.regularizers.Regularizer] = None,
            name: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        """Initialize the 2D SOM layer with validation and configuration setup."""
        # Validate the 2D-specific input
        if not (isinstance(map_size, (tuple, list)) and len(map_size) == 2):
            raise ValueError(f"map_size must be a tuple of exactly 2 integers, got {map_size}")

        if not all(isinstance(dim, int) and dim > 0 for dim in map_size):
            raise ValueError(f"map_size must contain positive integers, got {map_size}")

        # Initialize the parent SOMLayer with grid_shape from map_size
        super().__init__(
            grid_shape=map_size,
            input_dim=input_dim,
            initial_learning_rate=initial_learning_rate,
            decay_function=decay_function,
            sigma=sigma,
            neighborhood_function=neighborhood_function,
            weights_initializer=weights_initializer,
            regularizer=regularizer,
            name=name,
            **kwargs
        )

        # Store map_size for backward compatibility and clear interface
        self.map_size = tuple(map_size)  # Ensure it's a tuple

    def get_weights_as_grid(self) -> keras.KerasTensor:
        """
        Get the current neuron weights organized as a 2D grid for visualization.

        This method provides a convenient way to access the weight matrix in its
        natural 2D grid organization, making it easy to visualize the learned
        feature organization and topological structure.

        Returns
        -------
        keras.KerasTensor
            The SOM weights reshaped as a grid with shape (height, width, input_dim).
            Each position (i, j) contains the weight vector for neuron at grid
            coordinate (i, j). This format is ideal for visualization and analysis
            of the topological organization learned by the SOM.

        Example
        -------
        ```python
        som = SOM2dLayer(map_size=(8, 8), input_dim=3)
        # ... after training ...

        weights_grid = som.get_weights_as_grid()  # Shape: (8, 8, 3)

        # Visualize RGB color organization (for 3D input)
        import matplotlib.pyplot as plt
        plt.imshow(weights_grid)  # Shows color organization
        plt.title("SOM Color Organization")

        # Access specific neuron weights
        center_neuron_weights = weights_grid[4, 4, :]  # Center of 8x8 grid
        ```

        Note
        ----
        This is an alias for `get_weights_map()` provided for backward compatibility
        and intuitive naming for 2D SOMs. Both methods return identical results.
        """
        return self.get_weights_map()

    def get_config(self) -> Dict[str, Any]:
        """
        Return layer configuration for serialization and model saving.

        This method ensures proper serialization of the 2D SOM layer by returning
        all necessary parameters in a format that can be used to reconstruct the
        layer exactly. It uses `map_size` instead of `grid_shape` for backward
        compatibility with existing 2D SOM implementations.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing all layer configuration parameters needed for
            reconstruction. Includes serialized initializers and regularizers
            for complete state preservation.

        Note
        ----
        The configuration uses `map_size` rather than `grid_shape` to maintain
        the 2D-specific interface while ensuring compatibility with the base
        SOMLayer serialization system.
        """
        # Get the base configuration from parent SOMLayer
        config = super().get_config()

        # Replace 'grid_shape' with 'map_size' for 2D layer compatibility
        if 'grid_shape' in config:
            config['map_size'] = self.map_size
            del config['grid_shape']

        # Ensure proper serialization of initializer and regularizer
        # The parent class already handles this robustly, but we maintain
        # explicit serialization for clarity and debugging
        if hasattr(self, '_weights_initializer_config'):
            config['weights_initializer'] = keras.initializers.serialize(
                keras.initializers.get(self._weights_initializer_config)
            )

        if hasattr(self, '_regularizer_config') and self._regularizer_config is not None:
            config['regularizer'] = keras.regularizers.serialize(
                keras.regularizers.get(self._regularizer_config)
            )

        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'SOM2dLayer':
        """
        Create layer instance from configuration dictionary.

        This class method enables proper deserialization of saved 2D SOM layers,
        handling the conversion between serialized configurations and layer parameters.

        Args:
            config: Dictionary containing layer configuration parameters.

        Returns:
            SOM2dLayer instance reconstructed from the configuration.
        """
        # Handle initializer deserialization
        if 'weights_initializer' in config:
            config['weights_initializer'] = keras.initializers.deserialize(
                config['weights_initializer']
            )

        # Handle regularizer deserialization
        if 'regularizer' in config and config['regularizer'] is not None:
            config['regularizer'] = keras.regularizers.deserialize(
                config['regularizer']
            )

        return cls(**config)

# ---------------------------------------------------------------------