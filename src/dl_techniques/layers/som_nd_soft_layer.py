"""
Differentiable Soft Self-Organizing Map (Soft SOM) layer.

This is a differentiable variant of the classical SOM that can be trained using
backpropagation. Instead of hard competitive learning, it uses soft assignments
computed via per-dimension softmax operations, making all operations differentiable.

Key Differences from Classical SOM:
- Uses soft competitive learning instead of winner-takes-all
- Per-dimension softmax enables independent spatial reasoning
- Fully trainable with standard optimizers (Adam, SGD, etc.)
- Continuous probability distributions instead of discrete BMU selection
- Differentiable reconstruction and regularization losses

Learning Process:
----------------
1. For each input vector:
   a. Compute distances to all neurons in the grid
   b. Apply per-dimension softmax to create soft spatial assignments
   c. Use soft assignments for differentiable reconstruction
   d. Minimize reconstruction error via backpropagation

Per-Dimension Softmax:
---------------------
For an N-dimensional grid, softmax is applied separately along each spatial dimension:
- 2D grid (H, W): softmax over height + softmax over width
- 3D grid (D, H, W): softmax over depth + height + width
- Results are combined multiplicatively for final soft assignments

This allows the model to learn spatial relationships independently in each dimension.

Applications:
-----------
- Differentiable clustering and representation learning
- Soft topological mapping with gradient-based optimization
- End-to-end training in larger neural architectures
- Regularized dimensionality reduction
- Continuous associative memory systems

Examples
--------
>>> # Create a differentiable 10x10 Soft SOM
>>> soft_som = SoftSOMLayer(grid_shape=(10, 10), input_dim=784, temperature=0.5)
>>>
>>> # Standard Keras training
>>> model = keras.Sequential([
...     keras.layers.Dense(512, activation='relu'),
...     soft_som,
...     keras.layers.Dense(10, activation='softmax')
... ])
>>>
>>> model.compile(optimizer='adam', loss='categorical_crossentropy')
>>> model.fit(x_train, y_train, epochs=10)

References
----------
[1] Kohonen, T. (1982). Self-organized formation of topologically correct feature
       maps. Biological Cybernetics, 43(1), 59-69.
[2] Ritter, H., & Schulten, K. (1988). Convergence properties of Kohonen's topology
       conserving maps: fluctuations, stability, and dimension selection.
       Biological Cybernetics, 60(1), 59-71.
[3] Soft competitive learning approaches in neural networks literature.
"""

import keras
from keras import ops
from typing import Tuple, Optional, Union, Dict, Any

from dl_techniques.utils.logger import logger


@keras.saving.register_keras_serializable()
class SoftSOMLayer(keras.layers.Layer):
    """
    Differentiable Soft Self-Organizing Map layer for end-to-end training.

    This layer implements a soft, differentiable variant of the Self-Organizing Map
    that can be trained using backpropagation. Instead of hard competitive learning,
    it uses per-dimension softmax operations to create continuous probability
    distributions over the neuron grid.

    The key innovation is applying softmax separately along each spatial dimension
    of the grid, allowing independent spatial reasoning in each dimension while
    maintaining differentiability for gradient-based optimization.

    Parameters
    ----------
    grid_shape : Tuple[int, ...]
        The shape of the SOM neuron grid (e.g., (10, 10) for 2D, (5, 5, 5) for 3D).
    input_dim : int
        The dimensionality of the input data vectors.
    temperature : float, optional
        Temperature parameter for softmax operations. Lower values create sharper
        distributions (more like hard assignments), higher values create smoother
        distributions. Defaults to 1.0.
    use_per_dimension_softmax : bool, optional
        Whether to use per-dimension softmax (True) or global softmax (False).
        Per-dimension allows independent spatial reasoning. Defaults to True.
    use_reconstruction_loss : bool, optional
        Whether to add reconstruction loss as a regularization term. Defaults to True.
    reconstruction_weight : float, optional
        Weight for the reconstruction loss term. Defaults to 1.0.
    topological_weight : float, optional
        Weight for topological preservation regularization. Defaults to 0.1.
    kernel_initializer : Union[str, keras.initializers.Initializer], optional
        Initialization method for the SOM weights. Defaults to 'glorot_uniform'.
    kernel_regularizer : Optional[keras.regularizers.Regularizer], optional
        Optional regularizer for the weight parameters. Defaults to None.
    name : str, optional
        The name of the layer. Defaults to None.
    **kwargs : Any
        Additional keyword arguments for the base layer.

    Input shape:
        A 2D tensor with shape: `(batch_size, input_dim)`.

    Output shape:
        A 2D tensor with shape: `(batch_size, input_dim)` representing the
        soft reconstruction of the input through the SOM.

    Returns
    -------
    keras.KerasTensor
        Soft reconstruction of the input with shape (batch_size, input_dim).

    Raises
    ------
    ValueError
        If grid_shape contains non-positive integers.
    ValueError
        If input_dim is not positive.
    ValueError
        If temperature is not positive.

    Example
    -------
    >>> # Create and use in a model
    >>> soft_som = SoftSOMLayer(
    ...     grid_shape=(8, 8),
    ...     input_dim=128,
    ...     temperature=0.5,
    ...     use_per_dimension_softmax=True
    ... )
    >>>
    >>> # Build a simple autoencoder with Soft SOM
    >>> encoder = keras.Sequential([
    ...     keras.layers.Dense(256, activation='relu'),
    ...     keras.layers.Dense(128, activation='relu'),
    ...     soft_som
    ... ])
    >>>
    >>> decoder = keras.Sequential([
    ...     keras.layers.Dense(256, activation='relu'),
    ...     keras.layers.Dense(784, activation='sigmoid')
    ... ])
    >>>
    >>> # End-to-end differentiable training
    >>> autoencoder = keras.Sequential([encoder, decoder])
    >>> autoencoder.compile(optimizer='adam', loss='mse')
    """

    def __init__(
        self,
        grid_shape: Tuple[int, ...],
        input_dim: int,
        temperature: float = 1.0,
        use_per_dimension_softmax: bool = True,
        use_reconstruction_loss: bool = True,
        reconstruction_weight: float = 1.0,
        topological_weight: float = 0.1,
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        name: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """Initialize the Soft SOM layer."""
        super().__init__(name=name, **kwargs)

        # Validation
        if not all(isinstance(d, int) and d > 0 for d in grid_shape):
            raise ValueError("grid_shape must contain positive integers.")
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")

        self.grid_shape = grid_shape
        self.grid_dim = len(grid_shape)
        self.input_dim = input_dim
        self.temperature = temperature
        self.use_per_dimension_softmax = use_per_dimension_softmax
        self.use_reconstruction_loss = use_reconstruction_loss
        self.reconstruction_weight = reconstruction_weight
        self.topological_weight = topological_weight

        # Store serialization configs
        self._kernel_initializer_config = kernel_initializer
        self._kernel_regularizer_config = kernel_regularizer

        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        # Layers will be initialized in build()
        self.weights_map = None
        self.grid_positions = None

        # Store build input shape for serialization
        self._build_input_shape = None

    def build(self, input_shape: Tuple) -> None:
        """
        Build the Soft SOM layer by creating trainable weight parameters.

        Parameters
        ----------
        input_shape : Tuple
            Shape of the input tensor.
        """
        # Store input shape for serialization
        self._build_input_shape = input_shape

        # Convert to list for manipulation
        input_shape_list = list(input_shape)

        # Verify input shape
        if len(input_shape_list) != 2 or input_shape_list[-1] != self.input_dim:
            raise ValueError(
                f"Expected input shape (batch_size, {self.input_dim}), "
                f"got {input_shape}"
            )

        # Create trainable weight map (this is the key difference from classical SOM)
        self.weights_map = self.add_weight(
            name="som_weights",
            shape=(*self.grid_shape, self.input_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True  # Enable backpropagation!
        )

        # Create grid positions for topological regularization
        self.grid_positions = self._create_grid_positions()

        super().build(input_shape)

        logger.info(
            f"Built SoftSOMLayer with grid_shape={self.grid_shape}, "
            f"input_dim={self.input_dim}, trainable weights: {self.weights_map.shape}"
        )

    def _create_grid_positions(self) -> keras.KerasTensor:
        """
        Create N-dimensional grid position coordinates.

        Returns
        -------
        keras.KerasTensor
            Grid positions with shape (*grid_shape, grid_dim).
        """
        coord_ranges = [ops.cast(ops.arange(d), "float32") for d in self.grid_shape]
        mesh_coords = ops.meshgrid(*coord_ranges, indexing='ij')
        position_grid = ops.stack(mesh_coords, axis=-1)
        return position_grid

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass with soft competitive learning.

        Parameters
        ----------
        inputs : keras.KerasTensor
            Input tensor of shape (batch_size, input_dim).
        training : bool, optional
            Whether in training mode (affects regularization).

        Returns
        -------
        keras.KerasTensor
            Soft reconstruction of inputs with shape (batch_size, input_dim).
        """
        # Compute soft assignments using per-dimension or global softmax
        soft_assignments = self._compute_soft_assignments(inputs)

        # Perform soft reconstruction
        reconstruction = self._soft_reconstruction(soft_assignments)

        # Add losses for training
        if training and self.use_reconstruction_loss:
            recon_loss = self._reconstruction_loss(inputs, reconstruction)
            self.add_loss(self.reconstruction_weight * recon_loss)

        if training and self.topological_weight > 0:
            topo_loss = self._topological_loss(soft_assignments)
            self.add_loss(self.topological_weight * topo_loss)

        return reconstruction

    def _compute_soft_assignments(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """
        Compute soft assignments using per-dimension softmax.

        Parameters
        ----------
        inputs : keras.KerasTensor
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        keras.KerasTensor
            Soft assignment weights of shape (batch_size, *grid_shape).
        """
        # Compute squared distances from inputs to all neurons
        # inputs: (batch_size, input_dim)
        # weights_map: (*grid_shape, input_dim)

        batch_size = ops.shape(inputs)[0]

        # Expand inputs: (batch_size, 1, 1, ..., input_dim)
        expanded_inputs = inputs
        for _ in range(self.grid_dim):
            expanded_inputs = ops.expand_dims(expanded_inputs, axis=1)

        # Expand weights: (1, *grid_shape, input_dim)
        expanded_weights = ops.expand_dims(self.weights_map, axis=0)

        # Compute squared distances: (batch_size, *grid_shape)
        squared_distances = ops.sum(
            ops.square(expanded_inputs - expanded_weights),
            axis=-1
        )

        if self.use_per_dimension_softmax:
            return self._per_dimension_softmax(squared_distances)
        else:
            return self._global_softmax(squared_distances)

    def _per_dimension_softmax(self, distances: keras.KerasTensor) -> keras.KerasTensor:
        """
        Apply softmax separately along each spatial dimension.

        Parameters
        ----------
        distances : keras.KerasTensor
            Distance tensor of shape (batch_size, *grid_shape).

        Returns
        -------
        keras.KerasTensor
            Combined soft assignments of shape (batch_size, *grid_shape).
        """
        # Apply softmax along each grid dimension independently
        dim_softmaxes = []

        for dim_idx in range(self.grid_dim):
            # Apply softmax along the current dimension (dim_idx + 1 because of batch dimension)
            spatial_axis = dim_idx + 1

            # Apply softmax with negative distances (closer = higher probability)
            dim_softmax = ops.softmax(-distances / self.temperature, axis=spatial_axis)
            dim_softmaxes.append(dim_softmax)

        # Combine dimension-wise softmaxes multiplicatively
        combined = dim_softmaxes[0]
        for i in range(1, len(dim_softmaxes)):
            combined = combined * dim_softmaxes[i]

        # Normalize to ensure probabilities sum to 1
        spatial_axes = list(range(1, self.grid_dim + 1))
        total = ops.sum(combined, axis=spatial_axes, keepdims=True)
        combined = combined / (total + 1e-8)

        return combined

    def _global_softmax(self, distances: keras.KerasTensor) -> keras.KerasTensor:
        """
        Apply global softmax over all neurons.

        Parameters
        ----------
        distances : keras.KerasTensor
            Distance tensor of shape (batch_size, *grid_shape).

        Returns
        -------
        keras.KerasTensor
            Global soft assignments of shape (batch_size, *grid_shape).
        """
        # Flatten spatial dimensions
        batch_size = ops.shape(distances)[0]
        flat_distances = ops.reshape(distances, (batch_size, -1))

        # Apply global softmax
        flat_softmax = ops.softmax(-flat_distances / self.temperature, axis=1)

        # Reshape back to grid shape
        return ops.reshape(flat_softmax, (batch_size,) + self.grid_shape)

    def _soft_reconstruction(self, soft_assignments: keras.KerasTensor) -> keras.KerasTensor:
        """
        Reconstruct inputs using soft assignments.

        Parameters
        ----------
        soft_assignments : keras.KerasTensor
            Soft assignment weights of shape (batch_size, *grid_shape).

        Returns
        -------
        keras.KerasTensor
            Reconstructed inputs of shape (batch_size, input_dim).
        """
        # soft_assignments: (batch_size, *grid_shape)
        # weights_map: (*grid_shape, input_dim)

        # Expand assignments: (batch_size, *grid_shape, 1)
        expanded_assignments = ops.expand_dims(soft_assignments, axis=-1)

        # Expand weights: (1, *grid_shape, input_dim)
        expanded_weights = ops.expand_dims(self.weights_map, axis=0)

        # Weighted sum: (batch_size, *grid_shape, input_dim)
        weighted_neurons = expanded_assignments * expanded_weights

        # Sum over spatial dimensions: (batch_size, input_dim)
        reconstruction = ops.sum(
            weighted_neurons,
            axis=list(range(1, self.grid_dim + 1))
        )

        return reconstruction

    def _reconstruction_loss(
        self,
        inputs: keras.KerasTensor,
        reconstruction: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Compute reconstruction loss (MSE between input and reconstruction).

        Parameters
        ----------
        inputs : keras.KerasTensor
            Original inputs.
        reconstruction : keras.KerasTensor
            Reconstructed inputs.

        Returns
        -------
        keras.KerasTensor
            Scalar reconstruction loss.
        """
        mse = ops.mean(ops.square(inputs - reconstruction))
        return mse

    def _topological_loss(self, soft_assignments: keras.KerasTensor) -> keras.KerasTensor:
        """
        Compute topological preservation loss to maintain spatial organization.

        Parameters
        ----------
        soft_assignments : keras.KerasTensor
            Soft assignment weights of shape (batch_size, *grid_shape).

        Returns
        -------
        keras.KerasTensor
            Scalar topological loss.
        """
        # Encourage neighboring neurons to have similar activation patterns
        # This preserves the topological organization of the SOM

        batch_size = ops.shape(soft_assignments)[0]

        # Flatten grid positions and assignments for easier computation
        total_neurons = ops.prod(ops.convert_to_tensor(self.grid_shape))
        flat_positions = ops.reshape(self.grid_positions, (total_neurons, self.grid_dim))
        flat_assignments = ops.reshape(soft_assignments, (batch_size, total_neurons))

        # Compute pairwise spatial distances between grid positions
        # Shape: (total_neurons, total_neurons)
        position_diff = ops.expand_dims(flat_positions, axis=1) - ops.expand_dims(flat_positions, axis=0)
        position_distances = ops.sqrt(ops.sum(ops.square(position_diff), axis=-1))

        # Create neighborhood weights (closer positions should have similar activations)
        # Use negative exponential to make nearby neurons have higher correlation weights
        neighborhood_weights = ops.exp(-position_distances)

        # Compute activation similarities between neurons (across the batch)
        # Shape: (total_neurons, total_neurons)
        assignment_similarities = ops.matmul(
            ops.transpose(flat_assignments), flat_assignments
        ) / ops.cast(batch_size, "float32")

        # Topological loss: encourage high similarity for nearby neurons
        # Multiply element-wise and take mean
        topo_loss = -ops.mean(neighborhood_weights * assignment_similarities)

        return topo_loss

    def get_weights_map(self) -> keras.KerasTensor:
        """
        Get the learned weight map.

        Returns
        -------
        keras.KerasTensor
            Weight map of shape (*grid_shape, input_dim).
        """
        return self.weights_map

    def get_soft_assignments(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """
        Get soft assignments for given inputs.

        Parameters
        ----------
        inputs : keras.KerasTensor
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        keras.KerasTensor
            Soft assignments of shape (batch_size, *grid_shape).
        """
        return self._compute_soft_assignments(inputs)

    def compute_output_shape(self, input_shape: Tuple) -> Tuple:
        """
        Compute output shape (same as input for reconstruction).

        Parameters
        ----------
        input_shape : Tuple
            Input tensor shape.

        Returns
        -------
        Tuple
            Output tensor shape (same as input).
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'grid_shape': self.grid_shape,
            'input_dim': self.input_dim,
            'temperature': self.temperature,
            'use_per_dimension_softmax': self.use_per_dimension_softmax,
            'use_reconstruction_loss': self.use_reconstruction_loss,
            'reconstruction_weight': self.reconstruction_weight,
            'topological_weight': self.topological_weight,
            'kernel_initializer': self._kernel_initializer_config,
            'kernel_regularizer': self._kernel_regularizer_config,
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get build configuration."""
        return {"input_shape": self._build_input_shape}

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build from configuration."""
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SoftSOMLayer":
        """Create layer from configuration."""
        # Handle complex object deserialization
        if 'kernel_initializer' in config:
            if isinstance(config['kernel_initializer'], dict):
                config['kernel_initializer'] = keras.initializers.deserialize(
                    config['kernel_initializer']
                )
        if 'kernel_regularizer' in config:
            if isinstance(config['kernel_regularizer'], dict):
                config['kernel_regularizer'] = keras.regularizers.deserialize(
                    config['kernel_regularizer']
                )

        return cls(**config)