"""
Mixture Exponential Network (MEN) Layer

A Mixture Exponential Network combines a neural network with a mixture density
model to predict probability distributions for non-negative target values. This
implementation provides a custom Keras layer that outputs parameters for a
mixture of Exponential distributions.

Key Features:
    - Models complex, multi-modal distributions for non-negative data
    - Outputs full probability distributions instead of point estimates
    - Provides uncertainty quantification through distribution parameters
    - Suitable for modeling waiting times, durations, or failure rates

Theory:
    MENs extend traditional neural networks by replacing the single output value
    with a mixture of Exponential distributions. For each input and each output
    dimension, the network outputs:

    1. The rate parameters (λ) for each mixture component
    2. The mixture weights (π) that determine component importance

    The resulting predicted distribution for a single output `y` is:
        p(y|x) = Σ π_i(x) * Exponential(y | λ_i(x))

    where the PDF for the Exponential distribution is `λ * exp(-λ*y)`.

Applications:
    - Survival analysis
    - Predictive maintenance (time-to-failure)
    - Customer churn prediction (time-to-churn)
    - Queueing theory (modeling wait times)
"""

import keras
import numpy as np
from keras import ops
from typing import Dict, Optional, Tuple, Union, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from .activations.explanded_activations import elu_plus_one_plus_epsilon

# ---------------------------------------------------------------------

EPSILON_CONSTANT = 1e-6

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class MixtureExponentialLayer(keras.layers.Layer):
    """Mixture Density Network Layer for Exponential Distributions.

    This layer outputs parameters for a mixture of Exponential distributions,
    making it suitable for modeling non-negative continuous variables. It assumes
    that for a multi-dimensional output, each dimension is modeled by an
    independent mixture model.

    The layer creates two separate dense sublayers internally:
    1. men_lambdas: Outputs rate parameters (λ) with a positivity constraint.
    2. men_pi: Outputs mixture weights (π) as unnormalized logits.

    Parameters
    ----------
    output_dimension : int
        Dimensionality of the output space. Each dimension is modeled by an
        independent exponential mixture. Must be positive.
    num_mixtures : int
        Number of Exponential mixtures per output dimension. Must be positive.
    **kwargs
        Additional layer arguments (initializers, regularizers, etc.)

    Raises
    ------
    ValueError
        If output_dimension or num_mixtures are not positive integers
    """

    def __init__(
            self,
            output_dimension: int,
            num_mixtures: int,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
            bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        """Initialize the MixtureExponentialLayer."""
        super().__init__(**kwargs)

        if output_dimension <= 0:
            raise ValueError(f"output_dimension must be positive, got {output_dimension}")
        if num_mixtures <= 0:
            raise ValueError(f"num_mixtures must be positive, got {num_mixtures}")

        self.output_dim = output_dimension
        self.num_mix = num_mixtures
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        self.men_lambdas = None
        self.men_pi = None
        self._build_input_shape = None

        logger.info(f"Initialized MixtureExponentialLayer with {num_mixtures} mixtures and {output_dimension}D output")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer weights and sublayers based on input shape."""
        self._build_input_shape = input_shape

        # LAMBDA OUTPUTS: λ rate parameters
        # Activation ensures λ > 0, which is a requirement for Exponential dist.
        # Shape: [batch_size, output_dim * num_mix]
        self.men_lambdas = keras.layers.Dense(
            self.output_dim * self.num_mix,
            activation=elu_plus_one_plus_epsilon,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='men_lambdas'
        )

        # MIXTURE WEIGHTS: π parameters (mixing coefficients)
        # Outputs unnormalized logits for each component in each dimension.
        # Shape: [batch_size, output_dim * num_mix]
        self.men_pi = keras.layers.Dense(
            self.output_dim * self.num_mix,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='men_pi'
        )

        self.men_lambdas.build(input_shape)
        self.men_pi.build(input_shape)

        super().build(input_shape)
        logger.debug(f"MixtureExponentialLayer built with input shape: {input_shape}")

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass of the layer."""
        lambda_output = self.men_lambdas(inputs, training=training)
        pi_output = self.men_pi(inputs, training=training)

        # Concatenate all parameters into single output tensor
        # Structure: [λ_params, π_params]
        return keras.layers.concatenate(
            [lambda_output, pi_output],
            name='men_outputs'
        )

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer."""
        # Total size = (output_dim * num_mix for lambdas) + (output_dim * num_mix for pis)
        output_size = 2 * self.output_dim * self.num_mix
        return tuple(list(input_shape[:-1]) + [output_size])

    def split_mixture_params(
            self,
            y_pred: keras.KerasTensor
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Split the mixture parameters into components (lambdas and pis).

        Separates and reshapes the concatenated output for easier math.

        Parameters
        ----------
        y_pred : keras.KerasTensor
            Predicted parameters tensor from the MEN layer.

        Returns
        -------
        Tuple[keras.KerasTensor, keras.KerasTensor]
            Tuple of (lambdas, pi_logits) tensors with shapes:
            - lambdas: [batch_size, output_dim, num_mix]
            - pi_logits: [batch_size, output_dim, num_mix]
        """
        params_per_dim = self.output_dim * self.num_mix
        out_lambda = y_pred[..., :params_per_dim]
        out_pi = y_pred[..., params_per_dim:]

        batch_size = ops.shape(y_pred)[0]

        # Reshape for per-dimension, per-mixture calculations
        out_lambda = ops.reshape(out_lambda, [batch_size, self.output_dim, self.num_mix])
        out_pi = ops.reshape(out_pi, [batch_size, self.output_dim, self.num_mix])

        return out_lambda, out_pi

    def loss_func(
            self,
            y_true: keras.KerasTensor,
            y_pred: keras.KerasTensor
    ) -> keras.KerasTensor:
        """MEN loss function using negative log likelihood.

        This function computes the negative log likelihood of the target values
        under the predicted mixture of Exponentials. The log PDF of an
        Exponential distribution is `log(λ) - λ*y`.

        The total loss is the sum of losses across all output dimensions.

        Parameters
        ----------
        y_true : keras.KerasTensor
            Target values tensor of shape [batch_size, output_dim].
            IMPORTANT: All values in y_true must be non-negative (>= 0).
        y_pred : keras.KerasTensor
            Predicted parameters tensor from the MEN layer.

        Returns
        -------
        keras.KerasTensor
            Negative log likelihood loss value.
        """
        y_true = ops.reshape(y_true, [-1, self.output_dim])
        out_lambda, out_pi = self.split_mixture_params(y_pred)

        # Convert mixture weight logits to probabilities via softmax
        # Applied independently for each output dimension
        # Shape: [batch_size, output_dim, num_mix]
        mix_weights = keras.activations.softmax(out_pi, axis=-1)

        # Expand y_true for broadcasting with mixture components
        # [batch, output_dim] -> [batch, output_dim, 1]
        y_true_expanded = ops.expand_dims(y_true, -1)

        # Calculate log probability density for each component
        # log(p(y|λ)) = log(λ) - λ*y
        # Shape: [batch, output_dim, num_mix]
        component_log_probs = ops.log(out_lambda) - out_lambda * y_true_expanded

        # To calculate log(Σᵢ πᵢ * pᵢ), we need to use the LogSumExp trick
        # log(Σᵢ exp(log(πᵢ) + log(pᵢ)))
        log_mix_weights = ops.log(mix_weights + EPSILON_CONSTANT)
        log_weighted_probs = component_log_probs + log_mix_weights

        # LogSumExp over the mixture components
        # Shape: [batch_size, output_dim]
        log_prob_per_dim = keras.ops.logsumexp(log_weighted_probs, axis=-1)

        # Total log probability is the sum across independent dimensions
        # Shape: [batch_size]
        total_log_prob = ops.sum(log_prob_per_dim, axis=-1)

        # Calculate negative log likelihood (NLL) loss
        # Average across batch dimension for final loss value
        loss = -ops.mean(total_log_prob)
        return ops.maximum(loss, 0.0)

    def sample(self, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        """Sample from the mixture of Exponentials distribution.

        Performs ancestral sampling:
        1. Sample a mixture component using a categorical distribution over π.
        2. Sample from the selected Exponential(λᵢ) distribution.

        This uses the inverse transform sampling method for the exponential
        distribution: `y = -log(U) / λ`, where `U ~ Uniform(0,1)`.

        Parameters
        ----------
        y_pred : keras.KerasTensor
            Predicted parameters from the MEN layer.

        Returns
        -------
        keras.KerasTensor
            Samples from the predicted distribution with shape [batch_size, output_dim].
        """
        out_lambda, out_pi = self.split_mixture_params(y_pred)
        mix_weights = keras.activations.softmax(out_pi, axis=-1)

        # Select mixture component for each output dimension using Gumbel-Max trick
        gumbel_noise = -ops.log(-ops.log(keras.random.uniform(ops.shape(out_pi))))
        selected_indices = ops.argmax(ops.log(mix_weights + EPSILON_CONSTANT) + gumbel_noise, axis=-1)

        # Gather the lambda values for the chosen components
        # Shape: [batch_size, output_dim]
        selected_lambda = ops.take_along_axis(out_lambda, ops.expand_dims(selected_indices, -1), axis=-1)
        selected_lambda = ops.squeeze(selected_lambda, axis=-1)

        # Generate uniform random numbers for inverse transform sampling
        uniform_samples = keras.random.uniform(ops.shape(selected_lambda))

        # Sample from Exponential(selected_lambda)
        # y = -log(U) / λ
        samples = -ops.log(uniform_samples + EPSILON_CONSTANT) / selected_lambda
        return samples

    def get_config(self) -> Dict[str, Any]:
        """Returns the layer configuration for serialization."""
        config = super().get_config()
        config.update({
            "output_dimension": self.output_dim,
            "num_mixtures": self.num_mix,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MixtureExponentialLayer":
        """Creates a layer from its config."""
        return cls(**config)


# ---------------------------------------------------------------------

def get_point_estimate_exp(
        y_pred: np.ndarray,
        men_layer: MixtureExponentialLayer
) -> np.ndarray:
    """Calculate point estimates (expected value) from MEN outputs.

    This computes the weighted average of the component means. For an
    Exponential distribution with rate λ, the mean is 1/λ.

    E[y|x] = Σᵢ πᵢ(x) * (1/λᵢ(x))

    Parameters
    ----------
    y_pred : np.ndarray
        Predicted parameters from the MEN layer.
    men_layer : MixtureExponentialLayer
        The MEN layer instance used in the model.

    Returns
    -------
    np.ndarray
        Point estimates with shape [batch_size, output_dim].
    """
    out_lambda, out_pi = men_layer.split_mixture_params(y_pred)

    mix_weights = keras.activations.softmax(out_pi, axis=-1)

    # Expected value of each component is 1/lambda
    component_means = 1.0 / out_lambda

    # Weighted average of component means
    # Shape: [batch_size, output_dim, num_mix] * [batch_size, output_dim, num_mix]
    weighted_means = mix_weights * component_means

    # Sum over the mixture components
    # Shape: [batch_size, output_dim]
    point_estimates = ops.sum(weighted_means, axis=-1)

    return ops.convert_to_numpy(point_estimates)

# ---------------------------------------------------------------------
