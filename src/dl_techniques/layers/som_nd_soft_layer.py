"""
Differentiable Soft Self-Organizing Map (Soft SOM) Layer.

This Keras layer implements a fully differentiable variant of the classical
Self-Organizing Map (SOM). It is designed for end-to-end training within larger
neural networks using standard backpropagation and gradient-based optimizers
(e.g., Adam, SGD). The layer learns to map high-dimensional input data onto a
low-dimensional, topologically ordered grid of neurons, making it useful for
differentiable clustering, feature extraction, and dimensionality reduction.

---
What it Does
---
At its core, a Soft SOM learns a "codebook" or "prototype" vector for each
neuron on a predefined grid (e.g., 10x10). For any given input vector, the layer
determines how closely it matches each of these prototype vectors. Instead of
selecting a single "winner" neuron (Best Matching Unit or BMU) like a classical
SOM, this layer calculates a soft, probabilistic assignment across all neurons.
The layer's primary output is a "soft reconstruction" of the input, which is a
weighted average of all neuron prototypes, where the weights are determined by
the soft assignments. This process preserves the topological structure of the
input data on its grid, meaning similar inputs will activate nearby neurons.

---
How it Does It
---
The differentiability of the layer is achieved by replacing the non-differentiable
"winner-takes-all" mechanism of classical SOMs with continuous, differentiable
operations. The process for each input vector is as follows:

1.  **Distance Calculation**: The squared Euclidean distance is computed between the
    input vector and the prototype vector of every neuron in the grid. This
    results in a distance tensor with the same spatial dimensions as the grid.

2.  **Soft Assignment Generation**: These distances are converted into a probability
    distribution over the grid, representing the "soft assignments." This is
    the key innovation.
    -   **Per-Dimension Softmax (Default)**: Instead of a single softmax over all
        neurons, the layer applies a separate softmax operation along each spatial
        dimension of the grid (e.g., one for height and one for width in a 2D grid).
        The results are combined multiplicatively and re-normalized. This allows
        the model to reason about spatial location independently in each dimension,
        which can lead to more structured and disentangled representations.
    -   **Global Softmax**: Alternatively, a standard global softmax can be
        applied across all flattened neurons.
    -   **Temperature (`temperature`)**: A temperature parameter controls the
        "sharpness" of the softmax output. Low temperatures create a peaky
        distribution (closer to a one-hot winner), while high temperatures create
        a smoother, more distributed assignment.

3.  **Soft Reconstruction (Layer Output)**: The layer's output is a weighted average
    of all neuron prototype vectors. The weights used in this average are the
    soft assignments computed in the previous step. This operation is fully
    differentiable.

4.  **Learning via Backpropagation**: The neuron prototypes (`weights_map`) are
    trainable variables. Gradients from a downstream loss function (e.g.,
    classification cross-entropy) or from the layer's own internal losses can
    flow back through the reconstruction process to update these prototypes.

5.  **Internal Regularization Losses**: The layer can add two types of losses
    during training to guide the learning process:
    -   **Reconstruction Loss**: An optional Mean Squared Error (MSE) loss between
        the original input and its soft reconstruction. This encourages the neuron
        prototypes to form a good codebook for representing the data manifold.
    -   **Topological Loss**: This loss encourages nearby neurons on the grid to
        have similar prototype vectors. It works by measuring the similarity of
        activation patterns (soft assignments) across a batch and rewarding cases
        where spatially close neurons have similar activation patterns. This is
        what enforces the classic map-like, topological structure of the SOM.

---
How the Prototype Vectors are Defined
---
The prototype vectors (also called the "codebook" or `weights_map`) are the
central learnable parameters of the Soft SOM layer.

-   **Structure**: They are stored in a single Keras weight tensor with a shape of
    `(*grid_shape, input_dim)`. For example, a Soft SOM with a `grid_shape` of
    `(10, 10)` for input data of dimension `784` will have a `weights_map` tensor
    of shape `(10, 10, 784)`. Each element `weights_map[i, j, :]` is a vector
    of length `784` that represents the prototype for the neuron at grid
    position `(i, j)`.

-   **Initialization**: When the layer is built, this tensor is created and
    initialized using a standard Keras initializer (e.g., 'glorot_uniform').
    Initially, the prototype vectors are just random points in the high-dimensional
    input space.

-   **Learning**: Because the `weights_map` is marked as `trainable=True`, its values
    are updated during the training process. The optimizer (e.g., Adam) adjusts
    these vectors based on the gradients flowing from the total loss. Over time,
    this "self-organizing" process moves the vectors to positions in the input
    space that effectively model the underlying data distribution and its topology.

---
What it Can Be Used For
---
This layer is a versatile tool that can be integrated into deep learning models
for various tasks:

-   **Differentiable Clustering and Visualization**: After training, the soft
    assignments for input data can be used to visualize how data clusters onto
    the 2D or 3D grid, revealing the underlying structure of the dataset.

-   **Topological Feature Extraction**: Use it as an intermediate layer in a deep
    network to extract features that are not only descriptive but also have a
    meaningful spatial organization. This can be beneficial for downstream
    tasks that rely on structured representations.

-   **Regularized Autoencoders**: The Soft SOM can serve as a bottleneck layer
    in an autoencoder. The topological constraint acts as a powerful regularizer,
    forcing the latent space to be structured and continuous, which can improve
    generalization and interpretability.

-   **Generative Modeling**: The learned grid of prototypes represents the data
    manifold. One can potentially sample from this manifold by interpolating
    between neuron prototypes on the grid.

-   **Continuous Associative Memory**: The soft reconstruction mechanism allows
    the layer to function as a content-addressable memory system where noisy or
    incomplete inputs can be mapped to their closest learned prototype.

Examples
--------
>>> # Create a differentiable 10x10 Soft SOM
>>> soft_som = SoftSOMLayer(grid_shape=(10, 10), input_dim=784, temperature=0.5)
>>>
>>> # Use it as a feature extractor in a standard Keras classification model
>>> model = keras.Sequential([
...     keras.layers.Dense(512, activation='relu'),
...     soft_som,
...     keras.layers.Dense(10, activation='softmax')
... ])
>>>
>>> model.compile(optimizer='adam', loss='categorical_crossentropy')
>>> # The Soft SOM's internal losses (reconstruction, topological) will be
>>> # automatically added to the main loss during training.
>>> model.fit(x_train, y_train, epochs=10)

References
----------
[1] Kohonen, T. (1982). Self-organized formation of topologically correct feature
       maps. Biological Cybernetics, 43(1), 59-69.
[2] Ritter, H., & Schulten, K. (1988). Convergence properties of Kohonen's topology
       conserving maps: fluctuations, stability, and dimension selection.
       Biological Cybernetics, 60(1), 59-71.
[3] The concept of using soft assignments for differentiable clustering and
       representation learning is explored in various forms across deep learning
       literature, forming the basis for this implementation.
"""

import keras
from keras import ops
from typing import Tuple, Optional, Union, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class SoftSOMLayer(keras.layers.Layer):
    """
    Differentiable Soft Self-Organizing Map layer for end-to-end training.

    This layer implements a soft, differentiable variant of the Self-Organizing Map
    that can be trained using backpropagation. Instead of hard competitive learning,
    it uses per-dimension or global softmax operations to create continuous probability
    distributions over the neuron grid, enabling gradient-based optimization while
    preserving topological structure.

    **Intent**: Provide a differentiable clustering and representation learning layer
    that maintains the topological ordering properties of classical SOMs while being
    fully compatible with modern deep learning training pipelines using backpropagation.

    **Architecture**:
    ```
    Input(shape=[batch, input_dim])
           ↓
    Distance Computation: ||x - w_ij||² for all neurons (i,j)
           ↓
    Soft Assignment: softmax(-distances / temperature)
           ↓
    Soft Reconstruction: Σ(assignment_ij * w_ij)
           ↓
    Output(shape=[batch, input_dim]) + Optional Regularization Losses
    ```

    **Mathematical Operations**:
    1. **Distance**: d_ij = ||x - w_ij||² (squared Euclidean distance)
    2. **Soft Assignment**: a_ij = softmax(-d_ij / τ) where τ is temperature
    3. **Reconstruction**: y = Σ_ij a_ij * w_ij (weighted prototype average)
    4. **Regularization**: L_recon + λ_topo * L_topo (optional)

    The key innovation is using differentiable soft assignments instead of hard
    winner-take-all competition, allowing gradient flow for end-to-end training.

    Args:
        grid_shape: Tuple of integers defining the shape of the SOM neuron grid.
            Example: (10, 10) for 2D grid, (5, 5, 5) for 3D grid. All values
            must be positive integers.
        input_dim: Integer, the dimensionality of input data vectors. Must be positive.
            This determines the size of each prototype vector in the weight map.
        temperature: Float, temperature parameter for softmax operations. Lower values
            create sharper distributions (approaching winner-take-all), higher values
            create smoother distributions. Must be positive. Defaults to 1.0.
        use_per_dimension_softmax: Boolean, whether to use per-dimension softmax (True)
            or global softmax (False). Per-dimension softmax applies separate softmax
            along each spatial dimension then combines results, enabling independent
            spatial reasoning. Defaults to True.
        use_reconstruction_loss: Boolean, whether to add MSE reconstruction loss
            as regularization. Encourages prototypes to form good data representation.
            Defaults to True.
        reconstruction_weight: Float, weight for reconstruction loss term. Higher values
            emphasize reconstruction quality. Must be non-negative. Defaults to 1.0.
        topological_weight: Float, weight for topological preservation regularization.
            Encourages neighboring neurons to have similar activation patterns.
            Must be non-negative. Defaults to 0.1.
        sharpness_weight: Float, weight for entropy-based sharpness regularization.
            Encourages sharper (more decisive) softmax distributions by penalizing
            high-entropy assignments. Higher values promote more winner-take-all
            behavior. Must be non-negative. Defaults to 0.0 (disabled).
        kernel_initializer: String or keras.initializers.Initializer, initialization
            method for the SOM weight map. Defaults to 'glorot_uniform'.
        kernel_regularizer: Optional keras.regularizers.Regularizer for weight
            parameters. Defaults to None.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        2D tensor with shape: `(batch_size, input_dim)`.

    Output shape:
        2D tensor with shape: `(batch_size, input_dim)` representing the
        soft reconstruction of the input through the learned prototype map.

    Attributes:
        weights_map: Trainable weight tensor of shape (*grid_shape, input_dim)
            containing the prototype vectors for each neuron in the grid.
        grid_positions: Non-trainable coordinate tensor used for topological
            regularization, shape (*grid_shape, grid_dim).

    Example:
        ```python
        # Create 2D Soft SOM for MNIST-like data
        soft_som = SoftSOMLayer(
            grid_shape=(8, 8),
            input_dim=784,
            temperature=0.5,
            use_per_dimension_softmax=True,
            reconstruction_weight=1.0,
            topological_weight=0.1,
            sharpness_weight=0.05  # Encourage sharper assignments
        )

        # Use as feature extractor in classification model
        model = keras.Sequential([
            keras.layers.Dense(256, activation='relu'),
            soft_som,  # Learns topologically organized features
            keras.layers.Dense(10, activation='softmax')
        ])

        # In an autoencoder bottleneck
        encoder = keras.Sequential([
            keras.layers.Dense(128, activation='relu'),
            soft_som  # Topological bottleneck
        ])

        # For visualization and clustering
        inputs = keras.Input(shape=(784,))
        features = soft_som(inputs)
        assignments = soft_som.get_soft_assignments(inputs)  # For visualization
        ```

    References:
        - Kohonen, T. (1982). Self-organized formation of topologically correct feature maps.
        - Modern differentiable clustering literature for soft assignment techniques.

    Raises:
        ValueError: If grid_shape contains non-positive integers.
        ValueError: If input_dim is not positive.
        ValueError: If temperature is not positive.
        ValueError: If reconstruction_weight, topological_weight, or sharpness_weight is negative.

    Note:
        This layer adds regularization losses during training that are automatically
        included in the total model loss. The soft assignment mechanism makes it
        fully differentiable and suitable for end-to-end gradient-based training.
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
        sharpness_weight: float = 0.0,
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs: Any
    ) -> None:
        """Initialize the Soft SOM layer."""
        super().__init__(**kwargs)

        # Validate inputs
        if not all(isinstance(d, int) and d > 0 for d in grid_shape):
            raise ValueError("grid_shape must contain positive integers.")
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        if reconstruction_weight < 0:
            raise ValueError("reconstruction_weight must be non-negative.")
        if topological_weight < 0:
            raise ValueError("topological_weight must be non-negative.")
        if sharpness_weight < 0:
            raise ValueError("sharpness_weight must be non-negative.")

        # Store ALL configuration parameters for serialization
        self.grid_shape = grid_shape
        self.grid_dim = len(grid_shape)
        self.input_dim = input_dim
        self.temperature = temperature
        self.use_per_dimension_softmax = use_per_dimension_softmax
        self.use_reconstruction_loss = use_reconstruction_loss
        self.reconstruction_weight = reconstruction_weight
        self.topological_weight = topological_weight
        self.sharpness_weight = sharpness_weight

        # Store initializers and regularizers for serialization
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        # Initialize weight attributes - actual weights created in build()
        self.weights_map = None
        self.grid_positions = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the Soft SOM layer by creating trainable weight parameters.

        Creates the SOM weight map containing prototype vectors and grid position
        coordinates for topological regularization.

        Args:
            input_shape: Shape tuple of the input tensor. Expected format is
                (batch_size, input_dim).

        Raises:
            ValueError: If input shape doesn't match expected dimensions.
        """
        # Verify input shape compatibility
        if len(input_shape) != 2:
            raise ValueError(f"Expected 2D input shape (batch_size, input_dim), got {input_shape}")

        if input_shape[-1] is None:
            raise ValueError("Input dimension must be defined, got None")

        if input_shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, got input shape with "
                f"last dimension={input_shape[-1]}"
            )

        # Create trainable weight map - core learnable parameters
        self.weights_map = self.add_weight(
            name="som_weights",
            shape=(*self.grid_shape, self.input_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True  # Enable backpropagation
        )

        # Create grid positions for topological regularization
        self.grid_positions = self._create_grid_positions()

        # Always call parent build at the end
        super().build(input_shape)

        logger.info(
            f"Built SoftSOMLayer with grid_shape={self.grid_shape}, "
            f"input_dim={self.input_dim}, trainable weights: {self.weights_map.shape}"
        )

    def _create_grid_positions(self) -> keras.KerasTensor:
        """
        Create N-dimensional grid position coordinates for topological regularization.

        Generates coordinate arrays for each dimension of the grid and combines them
        into a single tensor containing the spatial position of each neuron.

        Returns:
            Grid position tensor with shape (*grid_shape, grid_dim) where each
            position contains the coordinates of that neuron in the grid.
        """
        # Create coordinate ranges for each dimension
        coord_ranges = [ops.cast(ops.arange(d), "float32") for d in self.grid_shape]

        # Create meshgrid and stack into position tensor
        mesh_coords = ops.meshgrid(*coord_ranges, indexing='ij')
        position_grid = ops.stack(mesh_coords, axis=-1)

        return position_grid

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass implementing soft competitive learning.

        Computes soft assignments between inputs and prototype vectors, then
        reconstructs inputs using weighted prototype combinations. During training,
        adds optional regularization losses for reconstruction quality and
        topological structure preservation.

        Args:
            inputs: Input tensor of shape (batch_size, input_dim).
            training: Boolean indicating whether in training mode. Affects whether
                regularization losses are added to the model.

        Returns:
            Soft reconstruction tensor of shape (batch_size, input_dim) representing
            the weighted combination of prototype vectors based on soft assignments.
        """
        # Compute soft assignments and get intermediate softmax results for regularization
        soft_assignments, dim_softmaxes = self._compute_soft_assignments(inputs)

        # Perform soft reconstruction
        reconstruction = self._soft_reconstruction(soft_assignments)

        # Add regularization losses during training
        if training:
            if self.use_reconstruction_loss and self.reconstruction_weight > 0:
                recon_loss = self._reconstruction_loss(inputs, reconstruction)
                self.add_loss(self.reconstruction_weight * recon_loss)

            if self.topological_weight > 0:
                topo_loss = self._topological_loss(soft_assignments)
                self.add_loss(self.topological_weight * topo_loss)

            # Add sharpness loss for encouraging decisive assignments
            if (self.sharpness_weight > 0 and
                self.use_per_dimension_softmax and
                dim_softmaxes is not None):
                sharp_loss = self._sharpness_loss(dim_softmaxes)
                self.add_loss(self.sharpness_weight * sharp_loss)

        return reconstruction

    def _compute_soft_assignments(self, inputs: keras.KerasTensor) -> Tuple[keras.KerasTensor, Optional[list]]:
        """
        Compute soft assignments between inputs and prototype vectors.

        Calculates squared Euclidean distances from each input to all prototype
        vectors, then converts to probability distributions using either per-dimension
        or global softmax operations.

        Args:
            inputs: Input tensor of shape (batch_size, input_dim).

        Returns:
            Tuple containing:
            - Soft assignment weights of shape (batch_size, *grid_shape).
            - List of per-dimension softmax tensors for sharpness loss, or None
              if global softmax is used.
        """
        # Compute squared distances from inputs to all neurons
        # inputs: (batch_size, input_dim)
        # weights_map: (*grid_shape, input_dim)

        # Expand inputs to broadcast with weights_map
        expanded_inputs = inputs
        for _ in range(self.grid_dim):
            expanded_inputs = ops.expand_dims(expanded_inputs, axis=1)

        # Expand weights for broadcasting
        expanded_weights = ops.expand_dims(self.weights_map, axis=0)

        # Compute squared distances: (batch_size, *grid_shape)
        squared_distances = ops.sum(
            ops.square(expanded_inputs - expanded_weights),
            axis=-1
        )

        # Convert distances to soft assignments
        if self.use_per_dimension_softmax:
            return self._per_dimension_softmax(squared_distances)
        else:
            global_assignments = self._global_softmax(squared_distances)
            return global_assignments, None

    def _per_dimension_softmax(self, distances: keras.KerasTensor) -> Tuple[keras.KerasTensor, list]:
        """
        Apply softmax separately along each spatial dimension of the grid.

        This approach allows independent spatial reasoning in each dimension,
        potentially leading to more structured representations compared to global softmax.

        Args:
            distances: Distance tensor of shape (batch_size, *grid_shape).

        Returns:
            Tuple containing:
            - Combined soft assignments of shape (batch_size, *grid_shape).
            - List of per-dimension softmax tensors for entropy calculation.
        """
        # Apply softmax along each grid dimension independently
        dim_softmaxes = []

        for dim_idx in range(self.grid_dim):
            # Apply softmax along current dimension (dim_idx + 1 due to batch dimension)
            spatial_axis = dim_idx + 1

            # Apply softmax with negative distances (closer = higher probability)
            dim_softmax = ops.softmax(-distances / self.temperature, axis=spatial_axis)
            dim_softmaxes.append(dim_softmax)

        # Combine dimension-wise softmaxes multiplicatively
        combined = dim_softmaxes[0]
        for i in range(1, len(dim_softmaxes)):
            combined = combined * dim_softmaxes[i]

        # Normalize to ensure probabilities sum to 1 across spatial dimensions
        spatial_axes = list(range(1, self.grid_dim + 1))
        total = ops.sum(combined, axis=spatial_axes, keepdims=True)
        combined = combined / (total + 1e-8)

        return combined, dim_softmaxes

    def _global_softmax(self, distances: keras.KerasTensor) -> keras.KerasTensor:
        """
        Apply global softmax over all neurons in the grid.

        Standard approach that treats all neurons equally without spatial structure
        considerations during the softmax computation.

        Args:
            distances: Distance tensor of shape (batch_size, *grid_shape).

        Returns:
            Global soft assignments of shape (batch_size, *grid_shape) where
            probabilities sum to 1 across all spatial positions.
        """
        # Flatten spatial dimensions for global softmax
        batch_size = ops.shape(distances)[0]
        flat_distances = ops.reshape(distances, (batch_size, -1))

        # Apply global softmax over all neurons
        flat_softmax = ops.softmax(-flat_distances / self.temperature, axis=1)

        # Reshape back to original grid shape
        return ops.reshape(flat_softmax, (batch_size,) + self.grid_shape)

    def _soft_reconstruction(self, soft_assignments: keras.KerasTensor) -> keras.KerasTensor:
        """
        Reconstruct inputs using soft-weighted prototype vectors.

        Computes a weighted average of all prototype vectors where weights are
        the soft assignment probabilities. This creates a differentiable
        reconstruction that approximates the input.

        Args:
            soft_assignments: Soft assignment weights of shape (batch_size, *grid_shape).

        Returns:
            Reconstructed inputs of shape (batch_size, input_dim) as weighted
            combinations of prototype vectors.
        """
        # soft_assignments: (batch_size, *grid_shape)
        # weights_map: (*grid_shape, input_dim)

        # Expand assignments for element-wise multiplication
        expanded_assignments = ops.expand_dims(soft_assignments, axis=-1)

        # Expand weights for broadcasting
        expanded_weights = ops.expand_dims(self.weights_map, axis=0)

        # Compute weighted prototype vectors
        weighted_neurons = expanded_assignments * expanded_weights

        # Sum over all spatial dimensions to get reconstruction
        spatial_axes = list(range(1, self.grid_dim + 1))
        reconstruction = ops.sum(weighted_neurons, axis=spatial_axes)

        return reconstruction

    def _reconstruction_loss(
        self,
        inputs: keras.KerasTensor,
        reconstruction: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Compute Mean Squared Error reconstruction loss.

        Measures how well the soft reconstruction approximates the original input,
        encouraging the prototype vectors to form a good representation basis.

        Args:
            inputs: Original input tensor of shape (batch_size, input_dim).
            reconstruction: Reconstructed tensor of shape (batch_size, input_dim).

        Returns:
            Scalar MSE loss between inputs and reconstruction.
        """
        mse_loss = ops.mean(ops.square(inputs - reconstruction))
        return mse_loss

    def _topological_loss(self, soft_assignments: keras.KerasTensor) -> keras.KerasTensor:
        """
        Compute topological preservation loss to maintain spatial organization.

        Encourages neighboring neurons in the grid to have similar activation patterns,
        preserving the topological structure characteristic of self-organizing maps.
        Uses spatial distance weighting to emphasize locality preservation.

        Args:
            soft_assignments: Soft assignment weights of shape (batch_size, *grid_shape).

        Returns:
            Scalar topological loss promoting spatial coherence in activations.
        """
        batch_size = ops.shape(soft_assignments)[0]

        # Flatten grid positions and assignments for pairwise computations
        total_neurons = ops.prod(ops.convert_to_tensor(self.grid_shape))
        flat_positions = ops.reshape(self.grid_positions, (total_neurons, self.grid_dim))
        flat_assignments = ops.reshape(soft_assignments, (batch_size, total_neurons))

        # Compute pairwise spatial distances between all grid positions
        position_diff = ops.expand_dims(flat_positions, axis=1) - ops.expand_dims(flat_positions, axis=0)
        position_distances = ops.sqrt(ops.sum(ops.square(position_diff), axis=-1) + 1e-8)

        # Create neighborhood weights (exponential decay with distance)
        neighborhood_weights = ops.exp(-position_distances)

        # Compute activation correlations between neurons across batch
        normalized_assignments = flat_assignments - ops.mean(flat_assignments, axis=0, keepdims=True)
        assignment_correlations = ops.matmul(
            ops.transpose(normalized_assignments),
            normalized_assignments
        ) / ops.cast(ops.maximum(batch_size - 1, 1), "float32")

        # Topological loss: promote correlation between spatially close neurons
        topo_loss = -ops.mean(neighborhood_weights * assignment_correlations)

        return topo_loss

    def _sharpness_loss(self, dim_softmaxes: list) -> keras.KerasTensor:
        """
        Compute entropy-based sharpness loss to encourage peaky distributions.

        This loss penalizes high-entropy (flat) softmax distributions along each
        spatial dimension, pushing the model towards more confident, one-hot-like
        assignments. Lower entropy means sharper, more decisive distributions.

        The entropy formula used is: H(p) = -Σ(p * log(p))

        Args:
            dim_softmaxes: List of per-dimension softmax tensors, each of shape
                (batch_size, *grid_shape) with softmax applied along one spatial axis.

        Returns:
            Scalar sharpness loss representing the average entropy across all
            spatial dimensions and batch samples.
        """
        if not dim_softmaxes:
            return ops.convert_to_tensor(0.0, dtype="float32")

        total_entropy = ops.convert_to_tensor(0.0, dtype="float32")

        for dim_idx, softmax_tensor in enumerate(dim_softmaxes):
            # The spatial axis along which softmax was computed
            spatial_axis = dim_idx + 1

            # Compute entropy: H(p) = -Σ(p * log(p))
            # Add small epsilon for numerical stability
            log_probs = ops.log(softmax_tensor + 1e-9)
            entropy = -ops.sum(softmax_tensor * log_probs, axis=spatial_axis)

            # Average entropy across batch and remaining spatial dimensions
            total_entropy += ops.mean(entropy)

        # Return average entropy across all spatial dimensions
        return total_entropy / len(dim_softmaxes)

    def get_weights_map(self) -> keras.KerasTensor:
        """
        Get the learned prototype weight map.

        Returns:
            Weight map tensor of shape (*grid_shape, input_dim) containing the
            learned prototype vectors for each neuron in the grid.

        Raises:
            RuntimeError: If called before the layer is built.
        """
        if self.weights_map is None:
            raise RuntimeError("Layer must be built before accessing weights_map")
        return self.weights_map

    def get_soft_assignments(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """
        Get soft assignment probabilities for given inputs.

        Useful for visualization and analysis of how inputs map to the grid structure.

        Args:
            inputs: Input tensor of shape (batch_size, input_dim).

        Returns:
            Soft assignments of shape (batch_size, *grid_shape) representing
            the probability distribution over grid neurons for each input.
        """
        soft_assignments, _ = self._compute_soft_assignments(inputs)
        return soft_assignments

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute output tensor shape.

        For soft reconstruction, output shape is identical to input shape.

        Args:
            input_shape: Input tensor shape tuple.

        Returns:
            Output tensor shape tuple (same as input for reconstruction).
        """
        return tuple(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        Returns all parameters needed to reconstruct the layer, ensuring
        proper serialization and deserialization of the model.

        Returns:
            Dictionary containing complete layer configuration including all
            initialization parameters.
        """
        config = super().get_config()
        config.update({
            'grid_shape': self.grid_shape,
            'input_dim': self.input_dim,
            'temperature': self.temperature,
            'use_per_dimension_softmax': self.use_per_dimension_softmax,
            'use_reconstruction_loss': self.use_reconstruction_loss,
            'reconstruction_weight': self.reconstruction_weight,
            'topological_weight': self.topological_weight,
            'sharpness_weight': self.sharpness_weight,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config

# ---------------------------------------------------------------------