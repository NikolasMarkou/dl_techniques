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
    """Restricted Boltzmann Machine layer for unsupervised feature learning.

    An RBM is an energy-based generative model defined on a bipartite
    graph of visible and hidden units connected by symmetric weights
    ``W``. The energy function
    ``E(v, h) = -v^T W h - b^T v - c^T h``
    induces a joint probability ``p(v,h) ~ exp(-E(v,h))``. Training
    uses the Contrastive Divergence (CD-k) algorithm: a positive
    phase computes ``<v_i h_j>_data``, then ``k`` Gibbs sampling steps
    approximate the model expectation for the negative phase. Supports
    both binary (Bernoulli) and Gaussian visible units.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────┐
        │  Visible Units [n_visible]       │
        │  (input data)                    │
        └──────────────┬───────────────────┘
                       │
                       │  W [n_visible, n_hidden]
                       │  (symmetric weights)
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  Hidden Units [n_hidden]         │
        │  P(h_j=1|v) = sigmoid(c_j+W'v)   │
        └──────────────┬───────────────────┘
                       │
                       │  W^T (top-down)
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  Reconstruction [n_visible]      │
        │  P(v_i|h) via Gibbs sampling     │
        └──────────────────────────────────┘

    :param n_hidden: Number of hidden units. Must be positive.
    :type n_hidden: int
    :param learning_rate: Learning rate for CD training.
    :type learning_rate: float
    :param n_gibbs_steps: Number of Gibbs sampling steps (CD-k).
    :type n_gibbs_steps: int
    :param visible_unit_type: Type of visible units, ``'binary'`` or
        ``'gaussian'``.
    :type visible_unit_type: str
    :param use_bias: Whether to include bias terms.
    :type use_bias: bool
    :param kernel_initializer: Initializer for weight matrix ``W``.
    :type kernel_initializer: str
    :param visible_bias_initializer: Initializer for visible bias.
    :type visible_bias_initializer: str
    :param hidden_bias_initializer: Initializer for hidden bias.
    :type hidden_bias_initializer: str
    :param kernel_regularizer: Optional regularizer for weights.
    :type kernel_regularizer: Optional[Any]
    :param kwargs: Additional keyword arguments for the Layer base class.
    :type kwargs: Any"""

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
        """Create weight matrix and bias vectors.

        :param input_shape: Shape tuple; last dimension is ``n_visible``.
        :type input_shape: Tuple[Optional[int], ...]"""
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
        """Compute hidden unit probabilities given visible units.

        :param inputs: Visible unit tensor ``(batch, n_visible)``.
        :type inputs: keras.KerasTensor
        :param training: Training flag (unused).
        :type training: Optional[bool]
        :return: Hidden probabilities ``(batch, n_hidden)``.
        :rtype: keras.KerasTensor"""
        hidden_probs = self._compute_hidden_probabilities(inputs)
        return hidden_probs

    def _compute_hidden_probabilities(
            self,
            visible: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Compute ``P(h=1|v)`` for all hidden units.

        :param visible: Visible unit states ``(batch, n_visible)``.
        :type visible: keras.KerasTensor
        :return: Hidden unit probabilities ``(batch, n_hidden)``.
        :rtype: keras.KerasTensor"""
        activation = ops.matmul(visible, self.W)
        if self.use_bias:
            activation = ops.add(activation, self.hidden_bias)
        return ops.sigmoid(activation)

    def _compute_visible_probabilities(
            self,
            hidden: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Compute ``P(v|h)`` (binary) or mean (Gaussian) for visible units.

        :param hidden: Hidden unit states ``(batch, n_hidden)``.
        :type hidden: keras.KerasTensor
        :return: Visible probabilities or means ``(batch, n_visible)``.
        :rtype: keras.KerasTensor"""
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
        """Sample binary states from a Bernoulli distribution.

        :param probabilities: Activation probabilities ``(batch, n_units)``.
        :type probabilities: keras.KerasTensor
        :return: Binary samples ``{0, 1}`` of the same shape.
        :rtype: keras.KerasTensor"""
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
        """Compute or sample hidden activations given visible units.

        :param visible: Visible states ``(batch, n_visible)``.
        :type visible: keras.KerasTensor
        :param sample: If ``True`` return binary samples, else probabilities.
        :type sample: bool
        :return: Hidden states or probabilities ``(batch, n_hidden)``.
        :rtype: keras.KerasTensor"""
        hidden_probs = self._compute_hidden_probabilities(visible)
        if sample:
            return self._sample_binary(hidden_probs)
        return hidden_probs

    def sample_visible_given_hidden(
            self,
            hidden: keras.KerasTensor,
            sample: bool = True
    ) -> keras.KerasTensor:
        """Compute or sample visible activations given hidden units.

        :param hidden: Hidden states ``(batch, n_hidden)``.
        :type hidden: keras.KerasTensor
        :param sample: If ``True`` return sampled states, else
            probabilities/means.
        :type sample: bool
        :return: Visible states or probabilities ``(batch, n_visible)``.
        :rtype: keras.KerasTensor"""
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
        """Perform one Gibbs sampling step: ``v -> h -> v'``.

        :param visible: Current visible states ``(batch, n_visible)``.
        :type visible: keras.KerasTensor
        :return: Tuple of (reconstructed visible, hidden probabilities).
        :rtype: Tuple[keras.KerasTensor, keras.KerasTensor]"""
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
        """Train the RBM using CD-k and update weights in-place.

        :param visible_data: Input data batch ``(batch, n_visible)``.
        :type visible_data: keras.KerasTensor
        :return: Tuple of (reconstruction_error, metrics dict).
        :rtype: Tuple[keras.KerasTensor, Dict[str, keras.KerasTensor]]"""
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
        """Compute free energy ``F(v) = -b'v - sum log(1+exp(c_j+W_j'v))``.

        :param visible: Visible states ``(batch, n_visible)``.
        :type visible: keras.KerasTensor
        :return: Free energy per sample ``(batch,)``.
        :rtype: keras.KerasTensor"""
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
        """Reconstruct visible units via Gibbs sampling.

        :param visible: Input visible states ``(batch, n_visible)``.
        :type visible: keras.KerasTensor
        :param n_steps: Number of Gibbs steps.
        :type n_steps: int
        :return: Reconstructed visible ``(batch, n_visible)``.
        :rtype: keras.KerasTensor"""
        reconstructed = visible
        for _ in range(n_steps):
            reconstructed, _ = self.gibbs_sampling_step(reconstructed)
        return reconstructed

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the forward pass.

        :param input_shape: Input shape tuple.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape ``(batch, n_hidden)``.
        :rtype: Tuple[Optional[int], ...]"""
        output_shape = list(input_shape)
        output_shape[-1] = self.n_hidden
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return layer configuration for serialization.

        :return: Dictionary containing all constructor parameters.
        :rtype: Dict[str, Any]"""
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
