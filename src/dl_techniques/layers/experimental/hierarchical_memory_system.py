"""
Hierarchical Memory System using Multiple Self-Organizing Map Layers.

A hierarchical memory system that organizes information at multiple levels of abstraction
using a cascade of Self-Organizing Maps (SOMs). The system processes input data through
multiple levels, from abstract (small grids capturing broad patterns) to fine-grained
(large grids capturing detailed patterns).

The hierarchy works as follows:
- Level 1 (Abstract): Small grid size, captures broad categories and patterns
- Level 2 (Intermediate): Medium grid size, captures more detailed patterns
- Level N (Fine-grained): Large grid size, captures fine-detailed patterns

Each level in the hierarchy processes the same input data but with different grid
resolutions, allowing the system to capture multi-scale patterns and organize
memory at different levels of granularity.

Architecture:
------------
Input → Level 1 SOM (abstract, small grid)
     → Level 2 SOM (intermediate, medium grid)
     → Level N SOM (fine-grained, large grid)

Memory Organization:
------------------
- Abstract levels capture general categories and broad patterns
- Fine-grained levels capture specific details and nuanced patterns
- The hierarchy preserves both global structure and local details
- Each level maintains its own topological organization

Applications:
-----------
- Multi-scale pattern recognition
- Hierarchical clustering and classification
- Content-addressable memory with multiple levels of detail
- Associative memory systems with abstraction hierarchies
- Feature extraction at multiple scales

References
----------
[1] Kohonen, T. (2001). Self-Organizing Maps. Springer.
[2] Miikkulainen, R. (1990). Hierarchical feature maps. In Parallel Models of
    Associative Memory (pp. 143-186).
[3] Ultsch, A. (2003). Emergent self-organizing feature maps used for prediction
    and classification of multivariate data. In Emergent Neural Computational
    Architectures Based on Neuroscience (pp. 301-317).
"""

import keras
from typing import Tuple, Optional, Union, Dict, Any, Callable, List

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.memory.som_nd_layer import SOMLayer

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class HierarchicalMemorySystem(keras.layers.Layer):
    """
    Hierarchical memory system using multiple Self-Organizing Map layers.

    Creates a hierarchy of SOM layers that process input data at multiple levels
    of abstraction, from abstract (small grids) to fine-grained (large grids).
    Each level captures different scales of patterns in the input data, providing
    a multi-resolution memory organization.

    The system organizes memory hierarchically where:
    - Abstract levels (smaller grids) capture broad, general patterns
    - Fine-grained levels (larger grids) capture detailed, specific patterns
    - Each level maintains topological organization of its input space
    - All levels process the same input but at different resolutions

    Args:
        input_dim: Integer, the dimensionality of the input data vectors.
            Must be positive.
        levels: Integer, the number of hierarchy levels in the memory system.
            Must be at least 1. Defaults to 3.
        grid_dimensions: Integer, the number of dimensions for each SOM grid
            (e.g., 2 for 2D grids). Must be at least 1. Defaults to 2.
        base_grid_size: Integer, the grid size for the most abstract level (Level 1).
            Subsequent levels will have larger grids based on the expansion factor.
            Must be at least 1. Defaults to 5.
        grid_expansion_factor: Float, factor by which the grid size increases at each level.
            For example, with base_grid_size=5 and expansion_factor=2.0:
            Level 1: 5x5, Level 2: 10x10, Level 3: 20x20. Must be positive. Defaults to 2.0.
        initial_learning_rate: Float, the starting learning rate for all SOM layers.
            Must be positive. Defaults to 0.1.
        decay_function: Optional callable for learning rate decay. Applied to all SOM layers.
            Defaults to None (uses linear decay).
        sigma: Float, the initial neighborhood radius for all SOM layers.
            Must be positive. Defaults to 1.0.
        neighborhood_function: String, the neighborhood function type ('gaussian' or 'bubble')
            for all SOM layers. Defaults to 'gaussian'.
        weights_initializer: String or keras.initializers.Initializer, weight initialization
            method for all SOM layers. Defaults to 'random'.
        regularizer: Optional keras.regularizers.Regularizer applied to all SOM layers.
            Defaults to None.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        2D tensor with shape: `(batch_size, input_dim)`.

    Output shape:
        A tuple containing results from each hierarchy level:
        - bmu_indices_list: List of (batch_size, grid_dimensions) tensors
        - quantization_errors_list: List of (batch_size,) tensors

    Returns:
        Tuple of (List[keras.KerasTensor], List[keras.KerasTensor]) containing:
        - List of BMU coordinate tensors, one per level
        - List of quantization error tensors, one per level

    Raises:
        ValueError: If input_dim is not a positive integer.
        ValueError: If levels is less than 1.
        ValueError: If grid_dimensions is less than 1.
        ValueError: If base_grid_size is less than 1.
        ValueError: If grid_expansion_factor is not positive.
        ValueError: If initial_learning_rate is not positive.
        ValueError: If sigma is not positive.

    Example:
        ```python
        # Create a 3-level hierarchical memory system
        hierarchical_memory = HierarchicalMemorySystem(
            input_dim=784,  # For MNIST
            levels=3,
            base_grid_size=5,
            grid_expansion_factor=2.0
        )

        # Process input data
        inputs = keras.Input(shape=(784,))
        bmu_coords_list, q_errors_list = hierarchical_memory(inputs)
        model = keras.Model(inputs=inputs, outputs=[bmu_coords_list, q_errors_list])

        # Apply to data
        input_data = keras.random.uniform((128, 784))
        bmu_coords_list, q_errors_list = hierarchical_memory(input_data, training=True)

        # Level 1 (abstract): 5x5 grid
        print(bmu_coords_list[0].shape)  # (128, 2)
        # Level 2 (intermediate): 10x10 grid
        print(bmu_coords_list[1].shape)  # (128, 2)
        # Level 3 (fine-grained): 20x20 grid
        print(bmu_coords_list[2].shape)  # (128, 2)

        # Advanced configuration
        hierarchical_memory = HierarchicalMemorySystem(
            input_dim=512,
            levels=4,
            grid_dimensions=3,  # 3D grids
            base_grid_size=3,
            grid_expansion_factor=1.5,
            initial_learning_rate=0.05,
            sigma=2.0,
            neighborhood_function='bubble',
            weights_initializer='glorot_uniform',
            regularizer=keras.regularizers.L2(1e-4)
        )
        ```

    Notes:
        - Each level processes the same input data but organizes it at different resolutions
        - Abstract levels (smaller grids) capture broad patterns and categories
        - Fine-grained levels (larger grids) capture detailed and specific patterns
        - The hierarchy maintains topological organization at each level
        - Use `get_level_weights(level)` to access weight maps for specific levels
        - Use `get_all_weights()` to access weight maps for all levels
    """

    def __init__(
        self,
        input_dim: int,
        levels: int = 3,
        grid_dimensions: int = 2,
        base_grid_size: int = 5,
        grid_expansion_factor: float = 2.0,
        initial_learning_rate: float = 0.1,
        decay_function: Optional[Callable] = None,
        sigma: float = 1.0,
        neighborhood_function: str = 'gaussian',
        weights_initializer: Union[str, keras.initializers.Initializer] = 'he_uniform',
        regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs: Any
    ) -> None:
        """Initialize the hierarchical memory system."""
        super().__init__(**kwargs)

        # Validation
        if not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError(f"input_dim must be a positive integer, got {input_dim}")
        if not isinstance(levels, int) or levels < 1:
            raise ValueError(f"levels must be at least 1, got {levels}")
        if not isinstance(grid_dimensions, int) or grid_dimensions < 1:
            raise ValueError(f"grid_dimensions must be at least 1, got {grid_dimensions}")
        if not isinstance(base_grid_size, int) or base_grid_size < 1:
            raise ValueError(f"base_grid_size must be at least 1, got {base_grid_size}")
        if not isinstance(grid_expansion_factor, (int, float)) or grid_expansion_factor <= 0:
            raise ValueError(f"grid_expansion_factor must be positive, got {grid_expansion_factor}")
        if not isinstance(initial_learning_rate, (int, float)) or initial_learning_rate <= 0:
            raise ValueError(f"initial_learning_rate must be positive, got {initial_learning_rate}")
        if not isinstance(sigma, (int, float)) or sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        if neighborhood_function not in ['gaussian', 'bubble']:
            raise ValueError(f"neighborhood_function must be 'gaussian' or 'bubble', got {neighborhood_function}")

        # Store configuration parameters
        self.input_dim = input_dim
        self.levels = levels
        self.grid_dimensions = grid_dimensions
        self.base_grid_size = base_grid_size
        self.grid_expansion_factor = grid_expansion_factor
        self.initial_learning_rate = initial_learning_rate
        self.decay_function = decay_function
        self.sigma = sigma
        self.neighborhood_function = neighborhood_function
        self.weights_initializer = keras.initializers.get(weights_initializer)
        self.regularizer = keras.regularizers.get(regularizer)

        # CREATE all SOM layers in __init__ (following modern Keras 3 pattern)
        self.som_layers = []
        for level in range(self.levels):
            # Calculate grid size for this level
            current_grid_size = int(self.base_grid_size * (self.grid_expansion_factor ** level))
            grid_shape = tuple([current_grid_size] * self.grid_dimensions)

            # Create SOM layer for this level
            som_layer = SOMLayer(
                grid_shape=grid_shape,
                input_dim=self.input_dim,
                initial_learning_rate=self.initial_learning_rate,
                decay_function=self.decay_function,
                sigma=self.sigma,
                neighborhood_function=self.neighborhood_function,
                weights_initializer=self.weights_initializer,
                regularizer=self.regularizer,
                name=f"som_level_{level + 1}"
            )
            self.som_layers.append(som_layer)

        logger.debug(f"Initialized HierarchicalMemorySystem with {self.levels} levels, "
                    f"grid sizes: {[int(self.base_grid_size * (self.grid_expansion_factor ** i)) for i in range(self.levels)]}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the hierarchical memory system by building all SOM layers.

        This method explicitly builds each SOM layer to ensure proper
        serialization and weight management following modern Keras 3 patterns.

        Args:
            input_shape: Shape tuple (tuple of integers), indicating the
                input shape of the layer.
        """
        logger.debug(f"Building HierarchicalMemorySystem with input_shape: {input_shape}")

        # BUILD all SOM layers in computational order
        for level, som_layer in enumerate(self.som_layers):
            som_layer.build(input_shape)
            logger.debug(f"Built SOM layer {level + 1} with grid shape: {som_layer.grid_shape}")

        super().build(input_shape)

        logger.info(f"Built HierarchicalMemorySystem with {self.levels} levels, "
                    f"grid shapes: {[som.grid_shape for som in self.som_layers]}")

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> Tuple[List[keras.KerasTensor], List[keras.KerasTensor]]:
        """
        Forward pass through the hierarchical memory system.

        Processes the input through all hierarchy levels, from abstract to
        fine-grained. Each level finds its Best Matching Units (BMUs) and
        updates its weights if in training mode.

        Args:
            inputs: Input tensor of shape (batch_size, input_dim).
            training: Boolean indicating whether to perform weight updates.

        Returns:
            Tuple containing:
            - List of BMU coordinate tensors from each level
            - List of quantization error tensors from each level
        """
        bmu_indices_list = []
        quantization_errors_list = []

        # Process input through each hierarchy level
        for level_idx, som_layer in enumerate(self.som_layers):
            bmu_indices, quantization_errors = som_layer(inputs, training=training)

            bmu_indices_list.append(bmu_indices)
            quantization_errors_list.append(quantization_errors)

        return bmu_indices_list, quantization_errors_list

    def get_level_weights(self, level: int) -> keras.KerasTensor:
        """
        Get the weight map for a specific hierarchy level.

        Args:
            level: The hierarchy level (0-indexed) to get weights for.

        Returns:
            The weight map for the specified level.

        Raises:
            ValueError: If level is out of range.
        """
        if not isinstance(level, int) or level < 0 or level >= self.levels:
            raise ValueError(f"Level must be between 0 and {self.levels - 1}, got {level}")

        return self.som_layers[level].get_weights_map()

    def get_all_weights(self) -> List[keras.KerasTensor]:
        """
        Get weight maps for all hierarchy levels.

        Returns:
            List of weight maps, one for each hierarchy level.
        """
        return [som_layer.get_weights_map() for som_layer in self.som_layers]

    def get_grid_shapes(self) -> List[Tuple[int, ...]]:
        """
        Get the grid shapes for all hierarchy levels.

        Returns:
            List of grid shapes, one for each hierarchy level.
        """
        return [som_layer.grid_shape for som_layer in self.som_layers]

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[List[Tuple[Optional[int], ...]], List[Tuple[Optional[int], ...]]]:
        """
        Compute the output shape of the hierarchical memory system.

        Args:
            input_shape: Shape tuple of the input tensor.

        Returns:
            Tuple containing:
            - List of BMU coordinate shapes for each level
            - List of quantization error shapes for each level
        """
        # Convert to list for consistent manipulation
        input_shape_list = list(input_shape)
        batch_size = input_shape_list[0]

        bmu_shapes = []
        error_shapes = []

        for level in range(self.levels):
            bmu_shape = tuple([batch_size, self.grid_dimensions])
            error_shape = tuple([batch_size])

            bmu_shapes.append(bmu_shape)
            error_shapes.append(error_shape)

        return bmu_shapes, error_shapes

    def get_config(self) -> Dict[str, Any]:
        """
        Get configuration for the hierarchical memory system.

        Returns:
            Dictionary containing ALL constructor parameters needed for
            layer reconstruction.
        """
        config = super().get_config()
        config.update({
            'input_dim': self.input_dim,
            'levels': self.levels,
            'grid_dimensions': self.grid_dimensions,
            'base_grid_size': self.base_grid_size,
            'grid_expansion_factor': self.grid_expansion_factor,
            'initial_learning_rate': self.initial_learning_rate,
            'decay_function': self.decay_function,  # Note: functions may not serialize properly
            'sigma': self.sigma,
            'neighborhood_function': self.neighborhood_function,
            'weights_initializer': keras.initializers.serialize(self.weights_initializer),
            'regularizer': keras.regularizers.serialize(self.regularizer),
        })
        return config

# ---------------------------------------------------------------------