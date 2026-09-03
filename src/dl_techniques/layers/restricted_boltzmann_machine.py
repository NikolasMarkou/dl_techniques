"""Restricted Boltzmann Machine, built by :class:`RestrictedBoltzmannMachine`.

An RBM is an energy-based generative model on a bipartite graph of visible
and hidden units. Unlike a general Boltzmann machine, it forbids
intra-layer connections, so each layer is conditionally independent given
the other. That independence is what lets the model train by alternating
Gibbs sampling instead of full joint inference. Training uses Contrastive
Divergence (CD-k): a cheap positive phase computed directly from data, and
a negative phase approximated by running only ``k`` Gibbs steps from a
data-initialized chain rather than sampling the true equilibrium
distribution.

The layer supports binary (Bernoulli) and Gaussian visible units, chosen
via ``visible_unit_type``. Weight updates happen inside
:meth:`RestrictedBoltzmannMachine.contrastive_divergence`, not through
Keras's optimizer, so the layer manages its own gradient ascent step.

References:
    - Hinton, G. E., 2002. Training Products of Experts by Minimizing
      Contrastive Divergence. Neural Computation, 14(8), 1771-1800.
    - Hinton, G. E., 2010. A Practical Guide to Training Restricted
      Boltzmann Machines. UTML TR 2010-003.
"""

import keras
from keras import ops
from keras import initializers
from keras import regularizers
from typing import Optional, Tuple, Dict, Any
from dl_techniques.utils.keras_registration import register_dl_technique


@register_dl_technique("dl_techniques.layers.restricted_boltzmann_machine")
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

    Architecture:

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

        self.W = self.add_weight(
            name='W',
            shape=(self.n_visible, self.n_hidden),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
        )

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
        hidden = self.sample_hidden_given_visible(visible, sample=True)

        hidden_probs = self._compute_hidden_probabilities(visible)

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
        if not self.built:
            self.build(ops.shape(visible_data))

        batch_size = ops.cast(ops.shape(visible_data)[0], dtype=visible_data.dtype)

        hidden_probs_data = self._compute_hidden_probabilities(visible_data)
        hidden_states_data = self._sample_binary(hidden_probs_data)

        positive_grad = ops.matmul(
            ops.transpose(visible_data),
            hidden_probs_data
        )

        visible_model = visible_data
        for _ in range(self.n_gibbs_steps):
            visible_model, _ = self.gibbs_sampling_step(visible_model)

        hidden_probs_model = self._compute_hidden_probabilities(visible_model)

        negative_grad = ops.matmul(
            ops.transpose(visible_model),
            hidden_probs_model
        )

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

        # Gradient ascent step, applied directly via assign_add (no optimizer).
        self.W.assign_add(ops.multiply(self.learning_rate, W_grad))

        if self.use_bias:
            self.visible_bias.assign_add(
                ops.multiply(self.learning_rate, visible_bias_grad)
            )
            self.hidden_bias.assign_add(
                ops.multiply(self.learning_rate, hidden_bias_grad)
            )

        reconstruction_error = ops.mean(
            ops.square(ops.subtract(visible_data, visible_model))
        )

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
