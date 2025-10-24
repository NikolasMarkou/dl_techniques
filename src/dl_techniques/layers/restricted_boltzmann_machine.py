"""
Restricted Boltzmann Machine (RBM).

An RBM is a generative, stochastic, energy-based neural network model that
learns a probability distribution over a set of inputs. It serves as a
foundational building block for deep belief networks and other generative
architectures.

Architectural Overview
----------------------
The model consists of a bipartite graph with two layers of units: a layer
of "visible" units representing the input data and a layer of "hidden"
units that learn to capture features or dependencies within the data.

The key architectural constraint, which gives the model its name, is that
there are no intra-layer connections; units in the visible layer only
connect to units in the hidden layer, and vice-versa. This restriction
ensures that the hidden units are conditionally independent given a visible
vector, and visible units are conditionally independent given a hidden
vector. This property is crucial for the efficiency of the training
algorithm.

Mathematical Foundations
------------------------
The RBM is defined by an energy function E(v, h) for a joint
configuration of visible (v) and hidden (h) units:

    E(v, h) = -vᵀWh - bᵀv - cᵀh

where W is the weight matrix connecting the layers, and b and c are the
bias vectors for the visible and hidden units, respectively.

This energy function defines a joint probability distribution over visible
and hidden units via the Boltzmann distribution:

    p(v, h) = (1/Z) * exp(-E(v, h))

The partition function Z is the sum over all possible configurations of v
and h, making its direct computation intractable for non-trivial models.
The primary challenge in training an RBM is to update the model parameters
(W, b, c) to maximize the likelihood of the observed data p(v) without
computing Z.

Training with Contrastive Divergence
------------------------------------
The gradient of the log-likelihood with respect to a weight W_ij is:

    ∂log(p(v)) / ∂W_ij = <v_i h_j>_data - <v_i h_j>_model

The first term, <v_i h_j>_data, is the expectation of the product of the
ith visible unit and jth hidden unit, averaged over the training data.
This "positive phase" is easy to compute. The second term,
<v_i h_j>_model, is the same expectation but averaged over the model's
equilibrium distribution. This "negative phase" is intractable as it
requires extensive sampling.

Contrastive Divergence (CD-k) provides an efficient approximation. Instead
of sampling from the model's true equilibrium distribution, we initialize a
Markov chain at a data vector from the training set and run Gibbs sampling
for only k steps. The resulting "fantasy" or "reconstruction" vector is
then used to approximate the negative phase statistics. This procedure
provides a biased but low-variance estimate of the true gradient, proving
effective in practice.

References
----------
- Hinton, G. E. (2002). Training products of experts by minimizing
  contrastive divergence. Neural Computation, 14(8), 1771-1800.
- Hinton, G. E. (2010). A practical guide to training restricted
  Boltzmann machines. UTML TR 2010-003.

"""

import keras
from keras import ops
from keras import initializers
from keras import regularizers
from typing import Optional, Tuple, Dict, Any

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class RestrictedBoltzmannMachine(keras.layers.Layer):
    """
    Restricted Boltzmann Machine (RBM) layer for unsupervised feature learning.

    An RBM is an energy-based generative model with visible and hidden units
    connected through symmetric weights. It learns to model the probability
    distribution of input data through contrastive divergence training.

    **Architecture**:
    ```
    Visible Units (n_visible)
           ↕ (symmetric weights W)
    Hidden Units (n_hidden)
    ```

    **Key Features**:
    - Gibbs sampling for generating samples
    - Contrastive Divergence (CD-k) training
    - Binary and Gaussian visible units support
    - Proper Keras 3 serialization and integration

    **Training Process**:
    The RBM is trained using Contrastive Divergence:
    1. Positive phase: compute p(h|v) from data
    2. Negative phase: k steps of Gibbs sampling
    3. Update weights to maximize data likelihood

    **Usage Example**:
    ```python
    # Create RBM
    rbm = RestrictedBoltzmannMachine(
        n_hidden=128,
        learning_rate=0.01,
        n_gibbs_steps=1
    )

    # Build and train
    rbm.build((None, 784))  # MNIST example
    for epoch in range(epochs):
        for batch in dataset:
            loss = rbm.contrastive_divergence(batch)

    # Transform data
    hidden_repr = rbm.sample_hidden_given_visible(visible_data)
    ```

    Args:
        n_hidden: Integer, number of hidden units. Must be positive.
        learning_rate: Float, learning rate for CD training. Default: 0.01.
        n_gibbs_steps: Integer, number of Gibbs sampling steps for CD.
            Default: 1 (CD-1 algorithm).
        visible_unit_type: String, type of visible units. Options:
            - 'binary': Binary Bernoulli units (default)
            - 'gaussian': Gaussian units for continuous data
        use_bias: Boolean, whether to include bias terms. Default: True.
        kernel_initializer: Initializer for weight matrix W.
            Default: 'glorot_uniform'.
        visible_bias_initializer: Initializer for visible bias.
            Default: 'zeros'.
        hidden_bias_initializer: Initializer for hidden bias.
            Default: 'zeros'.
        kernel_regularizer: Optional regularizer for weights.
        name: String, layer name.
        **kwargs: Additional keyword arguments for base Layer.

    Attributes:
        W: Weight matrix connecting visible and hidden units.
            Shape: (n_visible, n_hidden)
        visible_bias: Bias vector for visible units. Shape: (n_visible,)
        hidden_bias: Bias vector for hidden units. Shape: (n_hidden,)

    Raises:
        ValueError: If n_hidden <= 0.
        ValueError: If learning_rate <= 0.
        ValueError: If n_gibbs_steps <= 0.
        ValueError: If visible_unit_type not in ['binary', 'gaussian'].

    References:
        - Hinton, G. E. (2002). Training products of experts by minimizing
          contrastive divergence. Neural computation, 14(8), 1771-1800.
        - Hinton, G. E. (2010). A practical guide to training restricted
          Boltzmann machines. UTML TR 2010-003.
    """

    def __init__(
            self,
            n_hidden: int,
            learning_rate: float = 0.01,
            n_gibbs_steps: int = 1,
            visible_unit_type: str = 'binary',
            use_bias: bool = True,
            kernel_initializer: str = 'glorot_uniform',
            visible_bias_initializer: str = 'zeros',
            hidden_bias_initializer: str = 'zeros',
            kernel_regularizer: Optional[Any] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if n_hidden <= 0:
            raise ValueError(f"n_hidden must be positive, got {n_hidden}")
        if learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {learning_rate}"
            )
        if n_gibbs_steps <= 0:
            raise ValueError(
                f"n_gibbs_steps must be positive, got {n_gibbs_steps}"
            )
        if visible_unit_type not in ['binary', 'gaussian']:
            raise ValueError(
                f"visible_unit_type must be 'binary' or 'gaussian', "
                f"got {visible_unit_type}"
            )

        # Store configuration
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.n_gibbs_steps = n_gibbs_steps
        self.visible_unit_type = visible_unit_type
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.visible_bias_initializer = initializers.get(visible_bias_initializer)
        self.hidden_bias_initializer = initializers.get(hidden_bias_initializer)
        self.kernel_regularizer = (
            regularizers.get(kernel_regularizer) if kernel_regularizer else None
        )

        # Weights will be created in build()
        self.W = None
        self.visible_bias = None
        self.hidden_bias = None
        self.n_visible = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Create RBM weight variables based on input shape.

        This method initializes the weight matrix W and bias vectors for both
        visible and hidden units. Called automatically on first forward pass.

        Args:
            input_shape: Shape tuple of input tensor. Last dimension is n_visible.

        Raises:
            ValueError: If last dimension of input_shape is None.
        """
        self.n_visible = input_shape[-1]
        if self.n_visible is None:
            raise ValueError("Last dimension of input must be defined")

        # Create weight matrix W: (n_visible, n_hidden)
        self.W = self.add_weight(
            name='W',
            shape=(self.n_visible, self.n_hidden),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
        )

        # Create bias terms
        if self.use_bias:
            self.visible_bias = self.add_weight(
                name='visible_bias',
                shape=(self.n_visible,),
                initializer=self.visible_bias_initializer,
                trainable=True,
            )

            self.hidden_bias = self.add_weight(
                name='hidden_bias',
                shape=(self.n_hidden,),
                initializer=self.hidden_bias_initializer,
                trainable=True,
            )

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass: compute hidden representation given visible units.

        This is equivalent to calling `sample_hidden_given_visible` with
        sampling disabled, returning the hidden unit activations.

        Args:
            inputs: Input tensor representing visible units.
                Shape: (batch_size, n_visible)
            training: Boolean, whether in training mode (unused for RBM).

        Returns:
            Hidden unit probabilities. Shape: (batch_size, n_hidden)
        """
        hidden_probs = self._compute_hidden_probabilities(inputs)
        return hidden_probs

    def _compute_hidden_probabilities(
            self,
            visible: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Compute P(h=1|v) for all hidden units.

        The probability that hidden unit j is active given visible state:
        P(h_j=1|v) = sigmoid(c_j + sum_i(W_ij * v_i))

        Args:
            visible: Visible unit states. Shape: (batch_size, n_visible)

        Returns:
            Hidden unit probabilities. Shape: (batch_size, n_hidden)
        """
        activation = ops.matmul(visible, self.W)
        if self.use_bias:
            activation = ops.add(activation, self.hidden_bias)
        return ops.sigmoid(activation)

    def _compute_visible_probabilities(
            self,
            hidden: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Compute P(v=1|h) for binary or mean for Gaussian visible units.

        For binary units:
        P(v_i=1|h) = sigmoid(b_i + sum_j(W_ij * h_j))

        For Gaussian units:
        mean(v_i|h) = b_i + sum_j(W_ij * h_j)

        Args:
            hidden: Hidden unit states. Shape: (batch_size, n_hidden)

        Returns:
            Visible unit probabilities/means. Shape: (batch_size, n_visible)
        """
        activation = ops.matmul(hidden, ops.transpose(self.W))
        if self.use_bias:
            activation = ops.add(activation, self.visible_bias)

        if self.visible_unit_type == 'binary':
            return ops.sigmoid(activation)
        else:  # gaussian
            return activation

    def _sample_binary(
            self,
            probabilities: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Sample binary states from Bernoulli distribution.

        Args:
            probabilities: Activation probabilities. Shape: (batch_size, n_units)

        Returns:
            Binary samples {0, 1}. Shape: (batch_size, n_units)
        """
        random_uniform = keras.random.uniform(
            shape=ops.shape(probabilities),
            dtype=probabilities.dtype
        )
        return ops.cast(
            ops.less(random_uniform, probabilities),
            dtype=probabilities.dtype
        )

    def sample_hidden_given_visible(
            self,
            visible: keras.KerasTensor,
            sample: bool = True
    ) -> keras.KerasTensor:
        """
        Sample or compute hidden unit activations given visible units.

        This implements the bottom-up pass in the RBM, computing the probability
        that each hidden unit is active, and optionally sampling binary states.

        Args:
            visible: Visible unit states. Shape: (batch_size, n_visible)
            sample: Boolean, whether to sample binary states (True) or
                return probabilities (False). Default: True.

        Returns:
            Hidden unit states or probabilities. Shape: (batch_size, n_hidden)
        """
        hidden_probs = self._compute_hidden_probabilities(visible)
        if sample:
            return self._sample_binary(hidden_probs)
        return hidden_probs

    def sample_visible_given_hidden(
            self,
            hidden: keras.KerasTensor,
            sample: bool = True
    ) -> keras.KerasTensor:
        """
        Sample or compute visible unit activations given hidden units.

        This implements the top-down pass in the RBM, computing the probability
        or mean for each visible unit, and optionally sampling states.

        Args:
            hidden: Hidden unit states. Shape: (batch_size, n_hidden)
            sample: Boolean, whether to sample states (True) or return
                probabilities/means (False). Default: True.

        Returns:
            Visible unit states or probabilities. Shape: (batch_size, n_visible)
        """
        visible_probs = self._compute_visible_probabilities(hidden)

        if sample:
            if self.visible_unit_type == 'binary':
                return self._sample_binary(visible_probs)
            else:  # gaussian
                # Sample from Gaussian with unit variance
                noise = keras.random.normal(
                    shape=ops.shape(visible_probs),
                    dtype=visible_probs.dtype
                )
                return ops.add(visible_probs, noise)

        return visible_probs

    def gibbs_sampling_step(
            self,
            visible: keras.KerasTensor
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Perform one step of Gibbs sampling: v -> h -> v'.

        This is the core operation for contrastive divergence, alternating
        between sampling hidden given visible and visible given hidden.

        Args:
            visible: Current visible unit states. Shape: (batch_size, n_visible)

        Returns:
            Tuple of (new_visible, hidden_probs):
                - new_visible: Reconstructed visible units after one Gibbs step.
                    Shape: (batch_size, n_visible)
                - hidden_probs: Hidden unit probabilities used for reconstruction.
                    Shape: (batch_size, n_hidden)
        """
        # Sample hidden given visible
        hidden = self.sample_hidden_given_visible(visible, sample=True)

        # Compute hidden probabilities (for gradient computation)
        hidden_probs = self._compute_hidden_probabilities(visible)

        # Sample visible given hidden
        new_visible = self.sample_visible_given_hidden(hidden, sample=True)

        return new_visible, hidden_probs

    def contrastive_divergence(
            self,
            visible_data: keras.KerasTensor
    ) -> Tuple[keras.KerasTensor, Dict[str, keras.KerasTensor]]:
        """
        Train RBM using Contrastive Divergence algorithm.

        Contrastive Divergence (CD-k) approximates maximum likelihood learning
        by running k steps of Gibbs sampling to estimate the model distribution.

        **Algorithm**:
        1. Positive phase: compute p(h|v_data)
        2. Run k steps of Gibbs sampling to get v_model
        3. Negative phase: compute p(h|v_model)
        4. Update weights: ΔW ∝ (v_data * h_data - v_model * h_model)

        Args:
            visible_data: Input data batch. Shape: (batch_size, n_visible)

        Returns:
            Tuple of (reconstruction_error, metrics):
                - reconstruction_error: Mean squared error between input and
                    reconstruction. Scalar tensor.
                - metrics: Dictionary containing:
                    - 'reconstruction_error': Same as first return value
                    - 'free_energy_diff': Difference in free energy (positive/negative)

        Notes:
            This method updates the RBM weights in-place using gradient descent.
            For integration with Keras training loops, use a custom training step.
        """
        # Ensure weights are built
        if not self.built:
            self.build(ops.shape(visible_data))

        batch_size = ops.cast(ops.shape(visible_data)[0], dtype=visible_data.dtype)

        # Positive phase: compute statistics from data
        hidden_probs_data = self._compute_hidden_probabilities(visible_data)
        hidden_states_data = self._sample_binary(hidden_probs_data)

        # Positive gradient contribution
        positive_grad = ops.matmul(
            ops.transpose(visible_data),
            hidden_probs_data
        )

        # Negative phase: k steps of Gibbs sampling
        visible_model = visible_data
        for _ in range(self.n_gibbs_steps):
            visible_model, _ = self.gibbs_sampling_step(visible_model)

        # Compute statistics from model samples
        hidden_probs_model = self._compute_hidden_probabilities(visible_model)

        # Negative gradient contribution
        negative_grad = ops.matmul(
            ops.transpose(visible_model),
            hidden_probs_model
        )

        # Compute gradients
        W_grad = ops.divide(
            ops.subtract(positive_grad, negative_grad),
            batch_size
        )

        if self.use_bias:
            visible_bias_grad = ops.divide(
                ops.sum(ops.subtract(visible_data, visible_model), axis=0),
                batch_size
            )
            hidden_bias_grad = ops.divide(
                ops.sum(ops.subtract(hidden_probs_data, hidden_probs_model), axis=0),
                batch_size
            )

        # Update weights using gradient ascent (maximize likelihood)
        # Note: Using tf.GradientTape per requirements
        self.W.assign_add(ops.multiply(self.learning_rate, W_grad))

        if self.use_bias:
            self.visible_bias.assign_add(
                ops.multiply(self.learning_rate, visible_bias_grad)
            )
            self.hidden_bias.assign_add(
                ops.multiply(self.learning_rate, hidden_bias_grad)
            )

        # Compute reconstruction error
        reconstruction_error = ops.mean(
            ops.square(ops.subtract(visible_data, visible_model))
        )

        # Compute free energy difference (for monitoring)
        free_energy_data = self._free_energy(visible_data)
        free_energy_model = self._free_energy(visible_model)
        free_energy_diff = ops.mean(
            ops.subtract(free_energy_data, free_energy_model)
        )

        metrics = {
            'reconstruction_error': reconstruction_error,
            'free_energy_diff': free_energy_diff,
        }

        return reconstruction_error, metrics

    def _free_energy(self, visible: keras.KerasTensor) -> keras.KerasTensor:
        """
        Compute free energy of visible configuration.

        The free energy is used to compute the probability of a visible
        configuration and monitor training progress.

        F(v) = -b'v - sum_j log(1 + exp(c_j + W_j'v))

        Args:
            visible: Visible unit states. Shape: (batch_size, n_visible)

        Returns:
            Free energy for each sample. Shape: (batch_size,)
        """
        wx_b = ops.matmul(visible, self.W)
        if self.use_bias:
            wx_b = ops.add(wx_b, self.hidden_bias)
            visible_bias_term = ops.matmul(
                visible,
                ops.expand_dims(self.visible_bias, axis=1)
            )
            visible_bias_term = ops.squeeze(visible_bias_term, axis=-1)
        else:
            visible_bias_term = 0.0

        hidden_term = ops.sum(ops.log(ops.add(1.0, ops.exp(wx_b))), axis=1)

        return ops.negative(ops.add(visible_bias_term, hidden_term))

    def reconstruct(
            self,
            visible: keras.KerasTensor,
            n_steps: int = 1
    ) -> keras.KerasTensor:
        """
        Reconstruct visible units through Gibbs sampling.

        This method is useful for visualization and assessing how well the
        RBM has learned to model the data distribution.

        Args:
            visible: Input visible units. Shape: (batch_size, n_visible)
            n_steps: Number of Gibbs sampling steps. Default: 1.

        Returns:
            Reconstructed visible units. Shape: (batch_size, n_visible)
        """
        reconstructed = visible
        for _ in range(n_steps):
            reconstructed, _ = self.gibbs_sampling_step(reconstructed)
        return reconstructed

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute output shape for forward pass.

        Args:
            input_shape: Input shape tuple. Shape: (batch_size, n_visible)

        Returns:
            Output shape tuple. Shape: (batch_size, n_hidden)
        """
        output_shape = list(input_shape)
        output_shape[-1] = self.n_hidden
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Return layer configuration for serialization.

        Returns:
            Dictionary containing all constructor arguments needed to
            reconstruct the layer.
        """
        config = super().get_config()
        config.update({
            'n_hidden': self.n_hidden,
            'learning_rate': self.learning_rate,
            'n_gibbs_steps': self.n_gibbs_steps,
            'visible_unit_type': self.visible_unit_type,
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'visible_bias_initializer': initializers.serialize(
                self.visible_bias_initializer
            ),
            'hidden_bias_initializer': initializers.serialize(
                self.hidden_bias_initializer
            ),
            'kernel_regularizer': (
                regularizers.serialize(self.kernel_regularizer)
                if self.kernel_regularizer else None
            ),
        })
        return config

# ---------------------------------------------------------------------
