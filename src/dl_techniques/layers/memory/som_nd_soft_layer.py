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
>>> import keras
>>> import numpy as np

>>> # --- Example 1: Basic Usage as a Standalone Layer ---
>>> # Create a Soft SOM for 64-dimensional input data on an 8x8 grid.
>>> som_layer = SoftSOMLayer(
...     grid_shape=(8, 8),
...     input_dim=64,
...     temperature=0.5,
...     name="soft_som"
... )
>>>
>>> # Create some dummy input data (batch of 4 vectors)
>>> dummy_input = np.random.rand(4, 64).astype("float32")
>>>
>>> # The layer's output is the "soft reconstruction" of the input.
>>> # It has the same shape as the input.
>>> reconstruction = som_layer(dummy_input)
>>> print(f"Input shape: {dummy_input.shape}")
>>> print(f"Output (reconstruction) shape: {reconstruction.shape}")
Input shape: (4, 64)
Output (reconstruction) shape: (4, 64)

>>> # --- Example 2: Use as a Feature Extractor in a Classification Model ---
>>> # The Soft SOM learns to map similar inputs to nearby locations on its
>>> # grid, creating a topologically ordered feature representation.
>>> inputs = keras.Input(shape=(784,), name="input_digits")
>>> x = keras.layers.Dense(256, activation='relu')(inputs)
>>>
>>> # The Soft SOM layer processes the dense features. Its output (a soft
>>> # reconstruction) serves as a structured feature vector for the next layer.
>>> som_features = SoftSOMLayer(
...     grid_shape=(10, 10),
...     input_dim=256,
...     name="som_feature_extractor"
... )(x)
>>>
>>> outputs = keras.layers.Dense(10, activation='softmax')(som_features)
>>> classification_model = keras.Model(inputs, outputs)
>>>
>>> # The model's total loss will automatically include the SOM's internal
>>> # reconstruction and topological losses, regularizing the training.
>>> classification_model.compile(
...     optimizer='adam',
...     loss='categorical_crossentropy'
... )
>>> # To train with real data:
>>> # (x_train, y_train), _ = keras.datasets.mnist.load_data()
>>> # x_train = np.reshape(x_train, (-1, 784)).astype("float32") / 255
>>> # y_train = keras.utils.to_categorical(y_train)
>>> # classification_model.fit(x_train, y_train, epochs=5, batch_size=64)

>>> # --- Example 3: Use as a Bottleneck in an Autoencoder ---
>>> # The SOM acts as a powerful regularizer, forcing the latent space to
>>> # be structured and continuous.
>>> encoder_input = keras.Input(shape=(784,), name="encoder_input")
>>> x = keras.layers.Dense(128, activation='relu')(encoder_input)
>>>
>>> # The Soft SOM forms the topological bottleneck.
>>> bottleneck = SoftSOMLayer(
...     grid_shape=(16, 16),
...     input_dim=128,
...     reconstruction_weight=1.0, # Weight for SOM's internal recon. loss
...     topological_weight=0.5,  # Weight for topological regularization
...     name="som_bottleneck"
... )(x)
>>>
>>> # The decoder reconstructs the original input from the SOM's output.
>>> x = keras.layers.Dense(128, activation='relu')(bottleneck)
>>> decoder_output = keras.layers.Dense(784, activation='sigmoid')(x)
>>>
>>> autoencoder = keras.Model(
...     encoder_input, decoder_output, name="som_autoencoder"
... )
>>>
>>> # The autoencoder is trained on its own reconstruction loss (e.g., MSE).
>>> # The SOM's internal losses are added automatically, providing powerful
>>> # regularization on the latent space.
>>> autoencoder.compile(optimizer='adam', loss='mse')

>>> # --- Example 4: Visualization and Clustering Analysis ---
>>> # After training, you can analyze how the SOM has organized the data.
>>> # Let's use the 'autoencoder' model from Example 3.
>>> som_layer = autoencoder.get_layer("som_bottleneck")
>>>
>>> # First, create a sub-model to get the SOM's input (the encoded data).
>>> encoder_model = keras.Model(autoencoder.input, som_layer.input)
>>>
>>> # Create some dummy test images.
>>> test_images = np.random.rand(5, 784).astype("float32")
>>>
>>> # Pass the images through the encoder to get the latent vectors.
>>> encoded_vectors = encoder_model.predict(test_images, verbose=0)
>>>
>>> # Now, use the SOM layer's 'get_soft_assignments' method.
>>> assignments = som_layer.get_soft_assignments(encoded_vectors)
>>> print(f"Shape of soft assignments: {assignments.shape}")
Shape of soft assignments: (5, 16, 16)
>>>
>>> # To find the Best Matching Unit (BMU) for each image, find the neuron
>>> # with the highest activation probability.
>>> assignments_flat = np.reshape(assignments, (assignments.shape[0], -1))
>>> bmu_indices = np.argmax(assignments_flat, axis=1)
>>>
>>> # Convert the flat index back to grid coordinates.
>>> bmu_coords = np.unravel_index(bmu_indices, som_layer.grid_shape)
>>> bmu_coords_stacked = np.stack(bmu_coords, axis=1)
>>>
>>> print(f"BMU flat indices: {bmu_indices}")
>>> print(f"BMU coordinates for each input:\\n{bmu_coords_stacked}")

>>> # --- Example 5: Controlling Behavior with Parameters ---
>>> # Sharper, more "winner-take-all" assignments with a very low temperature.
>>> sharp_som = SoftSOMLayer(grid_shape=(5, 5), input_dim=32, temperature=0.01)
>>>
>>> # Smoother, more distributed assignments with a high temperature.
>>> smooth_som = SoftSOMLayer(grid_shape=(5, 5), input_dim=32, temperature=10.0)
>>>
>>> # Disable the internal reconstruction loss and rely only on the topological
>>> # loss and the main model's loss.
>>> topo_only_som = SoftSOMLayer(
...     grid_shape=(5, 5),
...     input_dim=32,
...     use_reconstruction_loss=False,
...     topological_weight=0.5
... )
>>>
>>> # Use a global softmax instead of the default per-dimension softmax, which
>>> # can be simpler but may lose some dimensional independence.
>>> global_softmax_som = SoftSOMLayer(
...     grid_shape=(5, 5),
...     input_dim=32,
...     use_per_dimension_softmax=False
... )
>>>
>>> # Encourage sharper assignments by penalizing high-entropy distributions.
>>> sharp_assignments_som = SoftSOMLayer(
...     grid_shape=(5, 5),
...     input_dim=32,
...     sharpness_weight=0.1
... )

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

    Implements a soft, differentiable SOM variant trainable via backpropagation.
    Instead of hard competitive learning, soft assignments are computed as
    ``a_{i,j} = softmax(-||x - w_{i,j}||^2 / tau)`` using either per-dimension
    or global softmax. The output is a soft reconstruction
    ``y = sum_{i,j} a_{i,j} * w_{i,j}`` which serves as a differentiable
    approximation through the learned prototype codebook. Optional regularization
    losses (reconstruction MSE + topological preservation + sharpness entropy)
    guide the training process.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────────────────┐
        │              SoftSOMLayer                    │
        │                                              │
        │  Input(batch, input_dim)                     │
        │         │                                    │
        │         ▼                                    │
        │  Distance: ||x - w_{i,j}||^2                │
        │         │                                    │
        │         ▼                                    │
        │  Soft Assignment:                            │
        │    softmax(-dist / temperature)              │
        │    (per-dim or global)                       │
        │         │                                    │
        │         ▼                                    │
        │  Reconstruction: sum(a_{i,j} * w_{i,j})     │
        │         │                                    │
        │         ├──► L_recon (MSE, optional)         │
        │         ├──► L_topo  (topological, optional) │
        │         ├──► L_sharp (entropy, optional)     │
        │         ▼                                    │
        │  Output(batch, input_dim)                    │
        └──────────────────────────────────────────────┘

    :param grid_shape: Shape of the SOM neuron grid, e.g. ``(10, 10)`` for 2D.
    :type grid_shape: Tuple[int, ...]
    :param input_dim: Dimensionality of input data vectors.
    :type input_dim: int
    :param temperature: Temperature for softmax operations. Lower = sharper.
        Defaults to 1.0.
    :type temperature: float
    :param use_per_dimension_softmax: Whether to use per-dimension softmax.
        Defaults to True.
    :type use_per_dimension_softmax: bool
    :param use_reconstruction_loss: Whether to add MSE reconstruction loss.
        Defaults to True.
    :type use_reconstruction_loss: bool
    :param reconstruction_weight: Weight for reconstruction loss. Defaults to 1.0.
    :type reconstruction_weight: float
    :param topological_weight: Weight for topological preservation loss.
        Defaults to 0.1.
    :type topological_weight: float
    :param sharpness_weight: Weight for entropy-based sharpness loss.
        Defaults to 0.0 (disabled).
    :type sharpness_weight: float
    :param kernel_initializer: Initialization method for SOM weight map.
        Defaults to ``'glorot_uniform'``.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for weight parameters.
        Defaults to ``L2(1e-5)``.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional keyword arguments for the Layer base class.
    :type kwargs: Any
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
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = keras.regularizers.L2(1e-5),
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

        :param input_shape: Shape tuple ``(batch_size, input_dim)``.
        :type input_shape: Tuple[Optional[int], ...]
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

        :return: Grid position tensor of shape ``(*grid_shape, grid_dim)``.
        :rtype: keras.KerasTensor
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

        :param inputs: Input tensor of shape ``(batch_size, input_dim)``.
        :type inputs: keras.KerasTensor
        :param training: Whether in training mode (controls regularization losses).
        :type training: Optional[bool]
        :return: Soft reconstruction of shape ``(batch_size, input_dim)``.
        :rtype: keras.KerasTensor
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

        :param inputs: Input tensor of shape ``(batch_size, input_dim)``.
        :type inputs: keras.KerasTensor
        :return: Tuple of (soft assignments, per-dim softmax list or None).
        :rtype: Tuple[keras.KerasTensor, Optional[list]]
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

        :param distances: Distance tensor of shape ``(batch_size, *grid_shape)``.
        :type distances: keras.KerasTensor
        :return: Tuple of (combined soft assignments, per-dim softmax list).
        :rtype: Tuple[keras.KerasTensor, list]
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

        :param distances: Distance tensor of shape ``(batch_size, *grid_shape)``.
        :type distances: keras.KerasTensor
        :return: Global soft assignments of shape ``(batch_size, *grid_shape)``.
        :rtype: keras.KerasTensor
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

        :param soft_assignments: Soft assignment weights of shape ``(batch_size, *grid_shape)``.
        :type soft_assignments: keras.KerasTensor
        :return: Reconstructed inputs of shape ``(batch_size, input_dim)``.
        :rtype: keras.KerasTensor
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

        :param inputs: Original input tensor of shape ``(batch_size, input_dim)``.
        :type inputs: keras.KerasTensor
        :param reconstruction: Reconstructed tensor of shape ``(batch_size, input_dim)``.
        :type reconstruction: keras.KerasTensor
        :return: Scalar MSE loss.
        :rtype: keras.KerasTensor
        """
        mse_loss = ops.mean(ops.square(inputs - reconstruction))
        return mse_loss

    def _topological_loss(self, soft_assignments: keras.KerasTensor) -> keras.KerasTensor:
        """
        Compute topological preservation loss to maintain spatial organization.

        :param soft_assignments: Soft assignment weights of shape ``(batch_size, *grid_shape)``.
        :type soft_assignments: keras.KerasTensor
        :return: Scalar topological loss.
        :rtype: keras.KerasTensor
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
        Compute entropy-based sharpness loss ``H(p) = -sum(p * log(p))``.

        :param dim_softmaxes: List of per-dimension softmax tensors.
        :type dim_softmaxes: list
        :return: Scalar average entropy loss across spatial dimensions.
        :rtype: keras.KerasTensor
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

        :return: Weight map tensor of shape ``(*grid_shape, input_dim)``.
        :rtype: keras.KerasTensor
        :raises RuntimeError: If called before the layer is built.
        """
        if self.weights_map is None:
            raise RuntimeError("Layer must be built before accessing weights_map")
        return self.weights_map

    def get_soft_assignments(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """
        Get soft assignment probabilities for given inputs.

        :param inputs: Input tensor of shape ``(batch_size, input_dim)``.
        :type inputs: keras.KerasTensor
        :return: Soft assignments of shape ``(batch_size, *grid_shape)``.
        :rtype: keras.KerasTensor
        """
        soft_assignments, _ = self._compute_soft_assignments(inputs)
        return soft_assignments

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute output tensor shape (same as input for reconstruction).

        :param input_shape: Input tensor shape tuple.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output tensor shape tuple.
        :rtype: Tuple[Optional[int], ...]
        """
        return tuple(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        :return: Dictionary containing complete layer configuration.
        :rtype: Dict[str, Any]
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