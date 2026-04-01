"""
Implements a differentiable, n-dimensional memory with probabilistic addressing.

This layer provides a structured, spatially organized memory system that is fully
differentiable, enabling end-to-end training within modern deep learning
frameworks. It bridges the gap between the topology-preserving properties of
classical Self-Organizing Maps (SOMs) and the requirements of gradient-based
optimization, offering a powerful mechanism for learning structured
representations.

Architectural and Mathematical Underpinnings:

The core architecture consists of two main components: a probabilistic
addressing mechanism and an n-dimensional grid of learnable latent vectors
(the memory lattice). The goal is to perform a "soft lookup" where an input
vector is mapped not to a single memory location, but to a weighted average
of all memory locations, with weights determined by learned similarity.

1.  **Factorized, Probabilistic Addressing**: For an n-dimensional grid of
    shape (d₁, d₂, ..., dₙ), the addressing is factorized across dimensions.
    An input vector `x` is fed into `n` independent projection networks,
    `f₁, f₂, ..., fₙ`. Each network `fᵢ` outputs a logit vector of size `dᵢ`.
    These logits are then transformed into a probability distribution `Pᵢ`
    over the `i`-th dimension using a temperature-controlled softmax:

        Pᵢ = softmax(fᵢ(x) / T)

    The temperature parameter `T` controls the sharpness of the distribution.
    A low temperature results in a sparse, focused distribution (akin to a
    hard lookup), while a high temperature yields a smooth, diffuse
    distribution. This continuous, differentiable transformation is the key
    departure from the discrete Best Matching Unit (BMU) search in a
    traditional SOM.

2.  **Joint Probability via Outer Product**: The individual dimensional
    probabilities `P₁, P₂, ..., Pₙ` are combined to form a joint probability
    distribution over the entire n-dimensional grid. This is achieved through
    an outer product (tensor product):

        P_joint = P₁ ⊗ P₂ ⊗ ... ⊗ Pₙ

    The resulting tensor `P_joint` has the shape (d₁, d₂, ..., dₙ), where each
    element `P_joint[i₁, i₂, ..., iₙ]` represents the probability of addressing
    the memory cell at that specific coordinate. This compositional approach
    is inspired by Product Key Memories, allowing for a vast addressable
    memory space with a modest number of parameters.

3.  **Differentiable Soft Lookup**: The final output is the expectation of the
    latent vectors in the memory grid `G`, weighted by the joint probability
    distribution. This is a weighted sum over all grid positions:

        Output = Σ_{i₁, i₂, ..., iₙ} P_joint[i₁, i₂, ..., iₙ] * G[i₁, i₂, ..., iₙ]

    This operation is fully differentiable with respect to both the grid
    vectors `G` and the parameters of the projection networks `fᵢ`. During
    training, gradients flow through this weighted sum, simultaneously
    updating the memory content (`G`) and the addressing logic (`fᵢ`) to
    minimize the task-specific loss. This process encourages semantically
    similar inputs to activate neighboring regions of the grid, leading to an
    emergent, self-organized topological representation of the input space.

This layer serves as a modern reinterpretation of classic unsupervised
learning algorithms, reformulated for deep learning. By replacing discrete,
non-differentiable operations with their probabilistic, soft counterparts, it
integrates the powerful concept of structured, topologically-aware memory
directly into end-to-end trainable models.

References:
    - Kohonen, T. (1990). The Self-Organizing Map. *Proceedings of the IEEE*.
      (Conceptual foundation for topological data representation).
    - Graves, A., et al. (2014). Neural Turing Machines. *arXiv preprint*.
      (Pioneered differentiable addressing for external memory).
    - Lample, G., et al. (2019). Large Memory Layers with Product Keys. *NeurIPS*.
      (Inspiration for compositional, factorized addressing).
    - Hinton, G., et al. (2015). Distilling the Knowledge in a Neural Network.
      *arXiv preprint*. (Popularized the use of temperature in softmax).
"""

import keras
import numpy as np
from typing import List, Tuple, Optional, Union, Any, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..regularizers.soft_orthogonal import SoftOrthonormalConstraintRegularizer
from ..initializers.hypersphere_orthogonal_initializer import OrthogonalHypersphereInitializer


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class NeuroGrid(keras.layers.Layer):
    """Differentiable N-dimensional memory lattice with probabilistic addressing.

    This layer implements a differentiable soft lookup table over an
    n-dimensional grid of learnable latent vectors. For each grid
    dimension ``d_i`` a dedicated dense projection followed by
    temperature-controlled softmax produces a probability distribution
    ``P_i = softmax(Dense_i(x) / T)``. The joint addressing probability
    is formed by their outer product
    ``P_joint = P_1 (x) P_2 (x) ... (x) P_N`` and the output is the
    weighted expectation over the grid:
    ``output = sum P_joint * grid_weights``.
    Both 2-D ``(batch, input_dim)`` and 3-D ``(batch, seq_len, dim)``
    inputs are supported; for 3-D inputs each token is processed
    independently through the shared grid, preserving sequence structure.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────────┐
        │  Input [B, (seq,) input_dim]         │
        └───────────────┬──────────────────────┘
                        │  (reshape to 2-D if 3-D)
                        ▼
        ┌──────────────────────────────────────┐
        │  Dense_1(d1) → softmax(T) → P_1     │
        │  Dense_2(d2) → softmax(T) → P_2     │
        │       ...                            │
        │  Dense_N(dn) → softmax(T) → P_N     │
        └───────────────┬──────────────────────┘
                        │
                        ▼
        ┌──────────────────────────────────────┐
        │  Outer product: P_joint              │
        │  [B, d1, d2, ..., dn]               │
        └───────────────┬──────────────────────┘
                        │
                        ▼
        ┌──────────────────────────────────────┐
        │  Soft lookup via einsum              │
        │  sum(P_joint * grid_weights)         │
        │  grid_weights [d1,..,dn, latent_dim] │
        └───────────────┬──────────────────────┘
                        │  (reshape back if 3-D)
                        ▼
        ┌──────────────────────────────────────┐
        │  Output [B, (seq,) latent_dim]       │
        └──────────────────────────────────────┘

    :param grid_shape: List of integers defining grid dimensions, e.g.
        ``[10, 8, 6]`` for a 10x8x6 grid. All values must be positive.
    :type grid_shape: Union[List[int], Tuple[int, ...]]
    :param latent_dim: Dimensionality of each grid latent vector
        (output feature size). Must be positive.
    :type latent_dim: int
    :param use_bias: Whether dense projection layers use bias.
    :type use_bias: bool
    :param temperature: Initial softmax temperature; lower values
        yield sharper addressing. Must be positive.
    :type temperature: float
    :param learnable_temperature: Whether temperature is trainable.
    :type learnable_temperature: bool
    :param entropy_regularizer_strength: Strength of entropy
        regularisation encouraging sharper distributions. Non-negative.
    :type entropy_regularizer_strength: float
    :param kernel_initializer: Initializer for projection Dense kernels.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for projection Dense biases.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param grid_initializer: Initializer for the grid latent vectors.
    :type grid_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for Dense kernels.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param bias_regularizer: Optional regularizer for Dense biases.
    :type bias_regularizer: Optional[keras.regularizers.Regularizer]
    :param grid_regularizer: Optional regularizer for grid weights.
    :type grid_regularizer: Optional[keras.regularizers.Regularizer]
    :param epsilon: Small constant for numerical stability.
    :type epsilon: float
    :param kwargs: Additional keyword arguments for the Layer base class.
    :type kwargs: Any"""

    def __init__(
            self,
            grid_shape: Union[List[int], Tuple[int, ...]],
            latent_dim: int,
            use_bias: bool = False,
            temperature: float = 1.0,
            learnable_temperature: bool = False,
            entropy_regularizer_strength: float = 0.0,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            grid_initializer: Union[str, keras.initializers.Initializer] = OrthogonalHypersphereInitializer(),
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
            grid_regularizer: Optional[keras.regularizers.Regularizer] = SoftOrthonormalConstraintRegularizer(0.1, 0.0, 0.001),
            epsilon: float = 1e-7,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if not grid_shape or len(grid_shape) == 0:
            raise ValueError("grid_shape cannot be empty")
        if any(dim <= 0 for dim in grid_shape):
            raise ValueError(f"All grid dimensions must be positive, got {grid_shape}")
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if entropy_regularizer_strength < 0:
            raise ValueError(f"entropy_regularizer_strength must be non-negative, got {entropy_regularizer_strength}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        # Store configuration
        self.grid_shape = tuple(grid_shape)
        self.latent_dim = latent_dim
        self.initial_temperature = temperature
        self.learnable_temperature = learnable_temperature
        self.entropy_regularizer_strength = entropy_regularizer_strength
        self.epsilon = epsilon
        self.n_dims = len(self.grid_shape)
        self.use_bias = use_bias
        self.total_grid_size = int(np.prod(self.grid_shape))

        # Store initializers and regularizers
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.grid_initializer = keras.initializers.get(grid_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.grid_regularizer = keras.regularizers.get(grid_regularizer)

        # Create projection layers in __init__
        self.projection_layers = []
        for i, dim_size in enumerate(self.grid_shape):
            # Create Dense layer without activation (we'll apply softmax manually with temperature)
            layer = keras.layers.Dense(
                units=dim_size,
                use_bias=self.use_bias,
                activation=None,  # No activation, we'll apply temperature-controlled softmax
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name=f'projection_{i}'
            )
            self.projection_layers.append(layer)

        # Grid weights and temperature created in build()
        self.grid_weights = None
        self.temperature = None
        self.input_is_3d = None  # Set during build based on input shape

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create grid weights, temperature parameter, and build projections.

        :param input_shape: Shape tuple of the input tensor (2-D or 3-D).
        :type input_shape: Tuple[Optional[int], ...]"""
        if len(input_shape) < 2 or len(input_shape) > 3:
            raise ValueError(f"Expected 2D or 3D input, got shape {input_shape}")

        # For 3D inputs (transformer mode), we use the last dimension (embed_dim)
        # For 2D inputs (traditional mode), we use the last dimension (input_dim)
        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError("Last dimension (input/embedding dimension) must be defined")

        # Store input shape info for call method
        self.input_is_3d = len(input_shape) == 3

        # Build projection layers - they work on the last dimension regardless of 2D/3D
        projection_input_shape = (None, input_dim)  # Generic shape for Dense layers
        for layer in self.projection_layers:
            layer.build(projection_input_shape)

        # Create learnable temperature parameter
        if self.learnable_temperature:
            self.temperature = self.add_weight(
                name='temperature',
                shape=(),
                initializer=keras.initializers.Constant(self.initial_temperature),
                constraint=keras.constraints.NonNeg(),  # Ensure temperature stays positive
                trainable=True
            )
        else:
            # Fixed temperature as non-trainable weight
            self.temperature = self.add_weight(
                name='temperature',
                shape=(),
                initializer=keras.initializers.Constant(self.initial_temperature),
                trainable=False
            )

        # Create grid weights: (d1, d2, ..., dn, latent_dim)
        grid_weight_shape = self.grid_shape + (self.latent_dim,)
        self.grid_weights = self.add_weight(
            name='grid_weights',
            shape=grid_weight_shape,
            initializer=self.grid_initializer,
            regularizer=self.grid_regularizer,
            trainable=True
        )

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass: project, compute joint probability, and soft lookup.

        :param inputs: Input tensor (2-D or 3-D).
        :type inputs: keras.KerasTensor
        :param training: Whether in training mode.
        :type training: Optional[bool]
        :return: Output tensor with same rank as input.
        :rtype: keras.KerasTensor"""
        original_shape = keras.ops.shape(inputs)
        input_rank = len(inputs.shape)

        # Handle 3D inputs by reshaping to 2D for processing
        if input_rank == 3:
            batch_size, seq_len, embed_dim = original_shape[0], original_shape[1], original_shape[2]
            # Reshape to (batch_size * seq_len, embed_dim) for processing
            inputs_2d = keras.ops.reshape(inputs, (batch_size * seq_len, embed_dim))
        else:
            inputs_2d = inputs

        # Get temperature-controlled probability distributions for each dimension
        probabilities = []
        total_entropy_loss = 0.0

        for layer in self.projection_layers:
            logits = layer(inputs_2d, training=training)  # (batch_size [* seq_len], dim_i)
            # Apply learnable temperature-controlled softmax for sharper/smoother addressing
            scaled_logits = logits / (self.temperature + self.epsilon)
            prob = keras.ops.softmax(scaled_logits, axis=-1)
            probabilities.append(prob)

            # Add entropy regularization to encourage sharper probabilities
            if self.entropy_regularizer_strength > 0.0 and training:
                # Compute entropy: -sum(p * log(p))
                entropy = -keras.ops.sum(prob * keras.ops.log(prob + self.epsilon), axis=-1)
                # Average entropy across batch and add as regularization loss
                avg_entropy = keras.ops.mean(entropy)
                entropy_loss = self.entropy_regularizer_strength * avg_entropy
                total_entropy_loss += entropy_loss

        # Add entropy regularization loss if enabled
        if self.entropy_regularizer_strength > 0.0 and training:
            self.add_loss(total_entropy_loss)

        # Compute joint probability using efficient outer product
        joint_prob = self._compute_joint_probability(probabilities)

        # Perform soft lookup with numerical stability
        output_2d = self._soft_lookup(joint_prob)

        # Restore original shape for 3D inputs (transformer mode)
        if input_rank == 3:
            output = keras.ops.reshape(output_2d, (batch_size, seq_len, self.latent_dim))
        else:
            output = output_2d

        return output

    def _compute_joint_probability(self, probabilities: List[keras.KerasTensor]) -> keras.KerasTensor:
        """Compute joint probability via outer product of per-dimension distributions.

        :param probabilities: List of probability tensors ``(batch, d_i)``.
        :type probabilities: List[keras.KerasTensor]
        :return: Joint probability tensor ``(batch, d1, ..., dn)``.
        :rtype: keras.KerasTensor"""
        # Start with first probability: (batch, d1)
        joint_prob = probabilities[0]

        # Sequentially compute outer products with numerical stability
        for i, prob in enumerate(probabilities[1:], 1):
            # Add new axis for broadcasting: joint_prob becomes (batch, d1, ..., di-1, 1)
            joint_prob = keras.ops.expand_dims(joint_prob, axis=-1)

            # Add axes to prob for proper broadcasting: (batch, 1, ..., 1, di)
            for _ in range(i):
                if len(prob.shape) == 2:  # First iteration
                    prob = keras.ops.expand_dims(prob, axis=1)
                else:
                    prob = keras.ops.expand_dims(prob, axis=-2)

            # Element-wise multiplication gives outer product
            joint_prob = joint_prob * prob

        # Add small epsilon for numerical stability
        joint_prob = joint_prob + self.epsilon

        # Renormalize to ensure probabilities sum to 1
        prob_sum = keras.ops.sum(
            joint_prob,
            axis=tuple(range(1, len(joint_prob.shape))),
            keepdims=True
        )
        joint_prob = joint_prob / (prob_sum + self.epsilon)

        return joint_prob

    def _soft_lookup(self, joint_prob: keras.KerasTensor) -> keras.KerasTensor:
        """Perform weighted lookup over the grid using joint probabilities.

        :param joint_prob: Joint probability tensor ``(batch, d1, ..., dn)``.
        :type joint_prob: keras.KerasTensor
        :return: Weighted sum tensor ``(batch, latent_dim)``.
        :rtype: keras.KerasTensor"""
        # Create einsum equation for the weighted sum
        # joint_prob: (batch, d1, d2, ..., dn)
        # grid_weights: (d1, d2, ..., dn, latent_dim)
        # output: (batch, latent_dim)

        # Build einsum equation dynamically based on grid dimensions
        batch_idx = 'b'
        # Fix: Use alphabet letters that don't conflict and handle higher dimensions
        grid_indices = ''.join([chr(ord('i') + j) for j in range(min(self.n_dims, 23))])  # Limit to available letters
        latent_idx = 'z'  # Use 'z' to avoid conflicts

        # joint_prob indices: batch + grid dimensions
        joint_indices = batch_idx + grid_indices

        # grid_weights indices: grid dimensions + latent
        grid_indices_with_latent = grid_indices + latent_idx

        # output indices: batch + latent
        output_indices = batch_idx + latent_idx

        # Einsum equation: e.g., 'bijk,ijkz->bz' for 3D grid
        equation = f"{joint_indices},{grid_indices_with_latent}->{output_indices}"

        # For very high dimensional grids, fall back to manual computation
        if self.n_dims > 6:  # Einsum becomes inefficient/problematic for very high dims
            # Reshape for manual computation
            batch_size = keras.ops.shape(joint_prob)[0]
            joint_flat = keras.ops.reshape(joint_prob, (batch_size, self.total_grid_size))
            grid_flat = keras.ops.reshape(self.grid_weights, (self.total_grid_size, self.latent_dim))
            output = keras.ops.matmul(joint_flat, grid_flat)
        else:
            # Use einsum for lower dimensional grids
            output = keras.ops.einsum(equation, joint_prob, self.grid_weights)

        return output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape for 2-D or 3-D inputs.

        :param input_shape: Input shape tuple.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple.
        :rtype: Tuple[Optional[int], ...]"""
        if len(input_shape) == 2:
            # 2D input: (batch_size, input_dim) → (batch_size, latent_dim)
            return (input_shape[0], self.latent_dim)
        elif len(input_shape) == 3:
            # 3D input: (batch_size, seq_len, embed_dim) → (batch_size, seq_len, latent_dim)
            return (input_shape[0], input_shape[1], self.latent_dim)
        else:
            raise ValueError(f"Unsupported input shape: {input_shape}")

    def get_config(self) -> Dict[str, Any]:
        """Return layer configuration for serialization.

        :return: Dictionary containing all constructor parameters.
        :rtype: Dict[str, Any]"""
        config = super().get_config()
        config.update({
            'use_bias': self.use_bias,
            'grid_shape': list(self.grid_shape),
            'latent_dim': self.latent_dim,
            'temperature': self.initial_temperature,
            'learnable_temperature': self.learnable_temperature,
            'entropy_regularizer_strength': self.entropy_regularizer_strength,
            'epsilon': self.epsilon,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'grid_initializer': keras.initializers.serialize(self.grid_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'grid_regularizer': keras.regularizers.serialize(self.grid_regularizer),
        })
        # Note: input_is_3d is not serialized as it's determined during build() from input shape
        return config

    def get_grid_weights(self) -> keras.KerasTensor:
        """Get the current grid weights for analysis or visualisation.

        :return: Grid weights ``(d1, d2, ..., dn, latent_dim)``.
        :rtype: keras.KerasTensor"""
        if self.grid_weights is None:
            raise ValueError("Layer must be built before accessing grid weights")
        return self.grid_weights

    def get_addressing_probabilities(
            self,
            inputs: keras.KerasTensor
    ) -> Dict[str, Union[List[keras.KerasTensor], keras.KerasTensor]]:
        """Get addressing probabilities for analysis and interpretation.

        :param inputs: Input tensor (2-D or 3-D).
        :type inputs: keras.KerasTensor
        :return: Dictionary with keys ``'individual'``, ``'joint'``, and
            ``'entropy'``.
        :rtype: Dict[str, Union[List[keras.KerasTensor], keras.KerasTensor]]"""
        if not self.built:
            raise ValueError("Layer must be built before getting probabilities")

        # Handle input reshaping for 3D inputs
        input_rank = len(inputs.shape)
        if input_rank == 3:
            original_shape = keras.ops.shape(inputs)
            batch_size, seq_len, embed_dim = original_shape[0], original_shape[1], original_shape[2]
            inputs_2d = keras.ops.reshape(inputs, (batch_size * seq_len, embed_dim))
        else:
            inputs_2d = inputs

        # Get individual probabilities (recompute to match forward pass)
        probabilities = []
        entropies = []

        for layer in self.projection_layers:
            logits = layer(inputs_2d)
            scaled_logits = logits / (self.temperature + self.epsilon)
            prob = keras.ops.softmax(scaled_logits, axis=-1)
            probabilities.append(prob)

            # Compute entropy as uncertainty measure
            entropy = -keras.ops.sum(prob * keras.ops.log(prob + self.epsilon), axis=-1)
            entropies.append(entropy)

        # Compute joint probability
        joint_prob = self._compute_joint_probability(probabilities)

        return {
            'individual': probabilities,
            'joint': joint_prob,
            'entropy': entropies
        }

    def get_grid_utilization(self, inputs: keras.KerasTensor) -> Dict[str, keras.KerasTensor]:
        """Compute grid utilisation statistics.

        :param inputs: Input tensor.
        :type inputs: keras.KerasTensor
        :return: Dictionary with ``'activation_counts'``,
            ``'total_activation'``, and ``'utilization_rate'``.
        :rtype: Dict[str, keras.KerasTensor]"""
        if not self.built:
            raise ValueError("Layer must be built before computing utilization")

        prob_info = self.get_addressing_probabilities(inputs)
        joint_prob = prob_info['joint']

        # Find most activated position per input
        joint_prob_flat = keras.ops.reshape(joint_prob, (keras.ops.shape(joint_prob)[0], -1))
        max_positions = keras.ops.argmax(joint_prob_flat, axis=-1)

        # Fix: Replace scatter_update with proper counting using one-hot
        activation_counts = keras.ops.zeros((self.total_grid_size,))

        # Convert positions to one-hot and sum
        one_hot_positions = keras.ops.one_hot(max_positions, self.total_grid_size)
        activation_counts = keras.ops.sum(one_hot_positions, axis=0)

        # Total activations per position (sum of all probabilities)
        total_activation = keras.ops.sum(joint_prob_flat, axis=0)

        # Utilization rate (normalized)
        total_inputs = keras.ops.cast(keras.ops.shape(inputs)[0], 'float32')
        utilization_rate = activation_counts / (total_inputs + self.epsilon)

        return {
            'activation_counts': activation_counts,
            'total_activation': total_activation,
            'utilization_rate': utilization_rate
        }

    def find_best_matching_units(self, inputs: keras.KerasTensor) -> Dict[str, keras.KerasTensor]:
        """Find Best Matching Units (BMUs) for given inputs.

        :param inputs: Input tensor.
        :type inputs: keras.KerasTensor
        :return: Dictionary with ``'bmu_indices'``, ``'bmu_probabilities'``,
            and ``'bmu_coordinates'``.
        :rtype: Dict[str, keras.KerasTensor]"""
        if not self.built:
            raise ValueError("Layer must be built before finding BMUs")

        prob_info = self.get_addressing_probabilities(inputs)
        joint_prob = prob_info['joint']

        # Find maximum probability positions
        joint_prob_flat = keras.ops.reshape(joint_prob, (keras.ops.shape(joint_prob)[0], -1))
        bmu_coordinates = keras.ops.argmax(joint_prob_flat, axis=-1)
        bmu_probabilities = keras.ops.max(joint_prob_flat, axis=-1)

        # Convert flat indices back to n-dimensional indices
        bmu_indices = []
        remaining = bmu_coordinates

        for dim_size in reversed(self.grid_shape):
            dim_indices = remaining % dim_size
            bmu_indices.append(dim_indices)
            remaining = remaining // dim_size

        # Reverse to get correct order and stack
        bmu_indices = keras.ops.stack(list(reversed(bmu_indices)), axis=-1)

        return {
            'bmu_indices': bmu_indices,
            'bmu_probabilities': bmu_probabilities,
            'bmu_coordinates': bmu_coordinates
        }

    def set_temperature(self, new_temperature: float) -> None:
        """Update the temperature parameter.

        :param new_temperature: New temperature value (must be positive).
        :type new_temperature: float"""
        if new_temperature <= 0:
            raise ValueError(f"temperature must be positive, got {new_temperature}")
        if self.temperature is None:
            raise ValueError("Layer must be built before setting temperature")

        # Update the temperature weight value
        self.temperature.assign(new_temperature)

    def get_current_temperature(self) -> float:
        """Get the current temperature value.

        :return: Current temperature as a float.
        :rtype: float"""
        if self.temperature is None:
            raise ValueError("Layer must be built before getting temperature")

        return float(keras.ops.convert_to_numpy(self.temperature))

    def compute_input_quality(self, inputs: keras.KerasTensor) -> Dict[str, keras.KerasTensor]:
        """Compute quality measures for inputs based on addressing behaviour.

        Returns six complementary metrics: ``addressing_confidence``
        (peak joint probability), ``addressing_entropy``, ``dimension_consistency``
        (mean per-dimension peak probability), ``grid_coherence``
        (inverse variance of the joint distribution), ``uncertainty``
        (mean per-dimension entropy), and a weighted ``overall_quality``
        composite in ``[0, 1]``. For 3-D inputs each metric is computed
        per token, yielding shape ``(batch, seq_len)``.

        :param inputs: Input tensor (2-D or 3-D).
        :type inputs: keras.KerasTensor
        :return: Dictionary of quality measure tensors.
        :rtype: Dict[str, keras.KerasTensor]"""
        if not self.built:
            raise ValueError("Layer must be built before computing quality")

        original_shape = keras.ops.shape(inputs)
        input_rank = len(inputs.shape)

        # Handle 3D inputs by reshaping to 2D for processing
        if input_rank == 3:
            batch_size, seq_len, embed_dim = original_shape[0], original_shape[1], original_shape[2]
            inputs_2d = keras.ops.reshape(inputs, (batch_size * seq_len, embed_dim))
            effective_batch_size = batch_size * seq_len
        else:
            inputs_2d = inputs
            effective_batch_size = keras.ops.shape(inputs)[0]

        prob_info = self.get_addressing_probabilities(inputs_2d)
        individual_probs = prob_info['individual']
        joint_prob = prob_info['joint']
        entropies = prob_info['entropy']

        # 1. Addressing Confidence: Maximum probability in joint distribution
        joint_prob_flat = keras.ops.reshape(joint_prob, (effective_batch_size, -1))
        addressing_confidence = keras.ops.max(joint_prob_flat, axis=-1)

        # 2. Addressing Entropy: Entropy of joint probability distribution (lower = better quality)
        joint_entropy = -keras.ops.sum(
            joint_prob_flat * keras.ops.log(joint_prob_flat + self.epsilon),
            axis=-1
        )

        # 3. Dimension Consistency: How sharp/consistent are individual dimensions
        dimension_sharpness = []
        for prob in individual_probs:
            # Higher max probability = more consistent/sharp addressing
            max_prob = keras.ops.max(prob, axis=-1)
            dimension_sharpness.append(max_prob)

        # Average sharpness across all dimensions
        dimension_consistency = keras.ops.mean(keras.ops.stack(dimension_sharpness, axis=-1), axis=-1)

        # 4. Grid Coherence: Measure based on probability distribution spread
        # Lower spread (higher concentration) indicates better mapping to grid structure
        prob_variance = keras.ops.var(joint_prob_flat, axis=-1)
        # Invert so higher values mean better coherence
        grid_coherence = 1.0 / (1.0 + prob_variance)

        # 5. Combined Uncertainty: Average of individual dimension entropies
        avg_dimension_entropy = keras.ops.mean(keras.ops.stack(entropies, axis=-1), axis=-1)
        uncertainty = avg_dimension_entropy

        # 6. Overall Quality Score: Composite measure (0-1 scale)
        # Normalize and combine multiple factors
        confidence_norm = addressing_confidence  # Already 0-1
        consistency_norm = dimension_consistency  # Already 0-1
        coherence_norm = grid_coherence  # Already 0-1

        # Entropy-based terms (invert and normalize to 0-1)
        max_joint_entropy = keras.ops.log(keras.ops.cast(self.total_grid_size, 'float32'))
        entropy_quality = 1.0 - (joint_entropy / (max_joint_entropy + self.epsilon))
        entropy_quality = keras.ops.clip(entropy_quality, 0.0, 1.0)

        max_dim_entropy = keras.ops.log(keras.ops.cast(keras.ops.max(keras.ops.array(self.grid_shape)), 'float32'))
        uncertainty_quality = 1.0 - (uncertainty / (max_dim_entropy + self.epsilon))
        uncertainty_quality = keras.ops.clip(uncertainty_quality, 0.0, 1.0)

        # Weighted combination of quality factors
        overall_quality = (
                0.25 * confidence_norm +  # Addressing confidence
                0.25 * entropy_quality +  # Joint entropy quality
                0.20 * consistency_norm +  # Dimension consistency
                0.15 * coherence_norm +  # Grid coherence
                0.15 * uncertainty_quality  # Individual dimension quality
        )

        # Reshape results back to match input format for 3D inputs
        if input_rank == 3:
            # Reshape from (batch * seq_len,) to (batch, seq_len)
            addressing_confidence = keras.ops.reshape(addressing_confidence, (batch_size, seq_len))
            joint_entropy = keras.ops.reshape(joint_entropy, (batch_size, seq_len))
            dimension_consistency = keras.ops.reshape(dimension_consistency, (batch_size, seq_len))
            grid_coherence = keras.ops.reshape(grid_coherence, (batch_size, seq_len))
            uncertainty = keras.ops.reshape(uncertainty, (batch_size, seq_len))
            overall_quality = keras.ops.reshape(overall_quality, (batch_size, seq_len))

        return {
            'addressing_confidence': addressing_confidence,
            'addressing_entropy': joint_entropy,
            'dimension_consistency': dimension_consistency,
            'grid_coherence': grid_coherence,
            'uncertainty': uncertainty,
            'overall_quality': overall_quality
        }

    def get_quality_statistics(self, inputs: keras.KerasTensor) -> Dict[str, float]:
        """Compute batch-level statistical summaries for all quality measures.

        This method provides detailed statistical analysis of quality measure distributions
        across a batch of inputs, enabling comprehensive assessment of data quality patterns,
        detection of distribution shifts, and identification of statistical anomalies.

        **Statistical Measures Computed**:

        For each quality measure (addressing_confidence, addressing_entropy, dimension_consistency,
        grid_coherence, uncertainty, overall_quality), the following statistics are calculated:

        - **Mean**: Central tendency of the quality distribution
        - **Standard Deviation**: Variability and spread in quality scores
        - **Minimum**: Worst-case quality in the batch
        - **Maximum**: Best-case quality in the batch
        - **Median**: Robust central tendency measure (50th percentile)

        **Interpretation Guidelines**:

        **Mean Quality Patterns**:
        - High mean (>0.7): Batch contains predominantly high-quality samples
        - Medium mean (0.3-0.7): Mixed quality batch with diverse sample types
        - Low mean (<0.3): Batch dominated by problematic or out-of-distribution samples

        **Standard Deviation Patterns**:
        - Low std (<0.1): Homogeneous batch with consistent quality
        - Medium std (0.1-0.3): Natural variation in sample quality
        - High std (>0.3): Heterogeneous batch with wide quality range

        **Min-Max Range Analysis**:
        - Narrow range: Consistent data quality across batch
        - Wide range: Batch contains both excellent and poor samples
        - Low minimum: Presence of outliers or problematic samples

        **Mean vs Median Comparison**:
        - Similar values: Symmetric quality distribution
        - Mean < Median: Distribution skewed toward lower quality (outliers present)
        - Mean > Median: Distribution skewed toward higher quality

        **Applications**:

        1. **Data Quality Monitoring**: Track quality distributions over time
        2. **Batch Assessment**: Evaluate incoming data quality before processing
        3. **Distribution Analysis**: Understand quality patterns in datasets
        4. **Outlier Detection**: Identify batches with unusual quality characteristics
        5. **Model Performance Correlation**: Relate quality statistics to model performance
        6. **Data Filtering Thresholds**: Set appropriate quality filtering thresholds
        7. **Quality Control**: Monitor data pipeline quality in production systems

        :param inputs: Input tensor.
        :type inputs: keras.KerasTensor
        :return: Dictionary with keys ``'{measure}_{stat}'`` where stat is
            one of mean, std, min, max, median. Values are Python floats.
        :rtype: Dict[str, float]"""
        quality_measures = self.compute_input_quality(inputs)

        statistics = {}
        for measure_name, measure_values in quality_measures.items():
            measure_np = keras.ops.convert_to_numpy(measure_values)
            statistics[f"{measure_name}_mean"] = float(np.mean(measure_np))
            statistics[f"{measure_name}_std"] = float(np.std(measure_np))
            statistics[f"{measure_name}_min"] = float(np.min(measure_np))
            statistics[f"{measure_name}_max"] = float(np.max(measure_np))
            statistics[f"{measure_name}_median"] = float(np.median(measure_np))

        return statistics

    def filter_by_quality_threshold(
            self,
            inputs: keras.KerasTensor,
            quality_threshold: float = 0.5,
            quality_measure: str = 'overall_quality'
    ) -> Dict[str, keras.KerasTensor]:
        """Partition inputs into high- and low-quality subsets by threshold.

        :param inputs: Input tensor.
        :type inputs: keras.KerasTensor
        :param quality_threshold: Threshold for partitioning.
        :type quality_threshold: float
        :param quality_measure: Quality metric name to threshold on.
        :type quality_measure: str
        :return: Dictionary with ``'high_quality_inputs'``,
            ``'low_quality_inputs'``, ``'high_quality_mask'``, and
            ``'quality_scores'``.
        :rtype: Dict[str, keras.KerasTensor]"""
        quality_measures = self.compute_input_quality(inputs)

        if quality_measure not in quality_measures:
            raise ValueError(f"Unknown quality measure: {quality_measure}")

        quality_scores = quality_measures[quality_measure]
        high_quality_mask = quality_scores >= quality_threshold
        low_quality_mask = quality_scores < quality_threshold

        # Fix: Replace keras.ops.boolean_mask with proper tensor indexing
        high_quality_indices = keras.ops.where(high_quality_mask)[:, 0]
        low_quality_indices = keras.ops.where(low_quality_mask)[:, 0]

        high_quality_inputs = keras.ops.take(inputs, high_quality_indices, axis=0)
        low_quality_inputs = keras.ops.take(inputs, low_quality_indices, axis=0)

        return {
            'high_quality_inputs': high_quality_inputs,
            'low_quality_inputs': low_quality_inputs,
            'high_quality_mask': high_quality_mask,
            'quality_scores': quality_scores
        }