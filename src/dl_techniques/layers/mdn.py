"""
Mixture Density Network (MDN) Layer Implementation for Keras.

This module provides an implementation of Mixture Density Networks as a custom Keras layer
using pure TensorFlow operations without external probability libraries.

Example:
    >>> mdn_layer = MDN(output_dimension=2, num_mixtures=5)
    >>> model = keras.Sequential([
    ...     keras.layers.Dense(128),
    ...     mdn_layer
    ... ])

References:
    - Bishop, C. M. (1994). Mixture Density Networks.
    - Martin, C. (2018). Keras MDN Layer Implementation.
"""


import keras
import numpy as np
import tensorflow as tf
from keras import layers
from typing import Dict, List, Optional, Tuple, Union, Callable


def elu_plus_one_plus_epsilon(x: tf.Tensor) -> tf.Tensor:
    """Enhanced ELU activation to prevent NaN in loss computation.

    Args:
        x: Input tensor

    Returns:
        Tensor with ELU activation plus small offset
    """
    return keras.activations.elu(x) + 1.0 + keras.backend.epsilon()


def softmax(x: tf.Tensor, axis: int = -1) -> tf.Tensor:
    """Compute softmax values with improved numerical stability.

    Args:
        x: Input tensor
        axis: Axis along which softmax is computed

    Returns:
        Softmax probabilities
    """
    x_max = tf.reduce_max(x, axis=axis, keepdims=True)
    exp_x = tf.exp(x - x_max)
    return exp_x / tf.reduce_sum(exp_x, axis=axis, keepdims=True)


def gaussian_probability(y: tf.Tensor, mu: tf.Tensor, sigma: tf.Tensor) -> tf.Tensor:
    """Compute Gaussian probability density.

    Args:
        y: Target values
        mu: Mean values
        sigma: Standard deviations

    Returns:
        Probability densities
    """
    EPSILON = 1e-6
    sigma = sigma + EPSILON  # Numerical stability

    # Compute normalized squared difference
    norm = tf.sqrt(2 * np.pi) * sigma
    exp = -0.5 * tf.square((y - mu) / sigma)

    return tf.exp(exp) / norm


class MDN(layers.Layer):
    """Mixture Density Network Layer.

    This layer outputs parameters for a mixture of Gaussian distributions.
    It includes safeguards against numerical instability:
    - Uses ELU + 1 + ε activation for variances
    - Trains mixture weights as logits

    Args:
        output_dimension: Dimensionality of the output space
        num_mixtures: Number of Gaussian mixtures
        kernel_initializer: Initializer for the kernel weights matrix
        kernel_regularizer: Regularizer function applied to the kernel weights matrix
        **kwargs: Additional layer arguments
    """

    def __init__(
            self,
            output_dimension: int,
            num_mixtures: int,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.output_dim = output_dimension
        self.num_mix = num_mixtures

        with tf.name_scope('MDN'):
            self.mdn_mus = layers.Dense(
                self.num_mix * self.output_dim,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name='mdn_mus'
            )

            self.mdn_sigmas = layers.Dense(
                self.num_mix * self.output_dim,
                activation=elu_plus_one_plus_epsilon,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name='mdn_sigmas'
            )

            self.mdn_pi = layers.Dense(
                self.num_mix,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name='mdn_pi'
            )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Builds the layer with the given input shape."""
        with tf.name_scope('mus'):
            self.mdn_mus.build(input_shape)
        with tf.name_scope('sigmas'):
            self.mdn_sigmas.build(input_shape)
        with tf.name_scope('pis'):
            self.mdn_pi.build(input_shape)
        super().build(input_shape)

    @property
    def trainable_weights(self) -> List[tf.Variable]:
        """Get trainable weights of all components."""
        return (self.mdn_mus.trainable_weights +
                self.mdn_sigmas.trainable_weights +
                self.mdn_pi.trainable_weights)

    @property
    def non_trainable_weights(self) -> List[tf.Variable]:
        """Get non-trainable weights of all components."""
        return (self.mdn_mus.non_trainable_weights +
                self.mdn_sigmas.non_trainable_weights +
                self.mdn_pi.non_trainable_weights)

    def call(self, x: tf.Tensor, mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        """Forward pass of the layer."""
        with tf.name_scope('MDN'):
            return layers.concatenate(
                [self.mdn_mus(x), self.mdn_sigmas(x), self.mdn_pi(x)],
                name='mdn_outputs'
            )

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Computes the output shape of the layer."""
        return (input_shape[0], (2 * self.output_dim * self.num_mix) + self.num_mix)

    def get_config(self) -> Dict:
        """Gets layer configuration."""
        config = {
            "output_dimension": self.output_dim,
            "num_mixtures": self.num_mix
        }
        base_config = super().get_config()
        return {**base_config, **config}


def get_mixture_loss_func(
        output_dim: int,
        num_mixes: int
) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """Creates a loss function for the MDN layer.

    Args:
        output_dim: Dimensionality of the output space
        num_mixes: Number of Gaussian mixtures

    Returns:
        Loss function that takes true values and predictions
    """

    def mdn_loss_func(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """MDN loss function implementation using pure TensorFlow ops."""
        # Reshape inputs
        y_pred = tf.reshape(
            y_pred,
            [-1, (2 * num_mixes * output_dim) + num_mixes],
            name='reshape_ypreds'
        )
        y_true = tf.reshape(y_true, [-1, output_dim], name='reshape_ytrue')

        # Split parameters
        out_mu, out_sigma, out_pi = tf.split(
            y_pred,
            num_or_size_splits=[
                num_mixes * output_dim,
                num_mixes * output_dim,
                num_mixes
            ],
            axis=-1,
            name='mdn_coef_split'
        )

        # Reshape mus and sigmas
        out_mu = tf.reshape(out_mu, [-1, num_mixes, output_dim])
        out_sigma = tf.reshape(out_sigma, [-1, num_mixes, output_dim])

        # Compute mixture weights
        mix_weights = softmax(out_pi, axis=-1)

        # Compute component probabilities
        y_true_expanded = tf.expand_dims(y_true, 1)  # [batch, 1, output_dim]

        # Compute gaussian probabilities for each component
        component_probs = gaussian_probability(
            y_true_expanded,
            out_mu,
            out_sigma
        )  # [batch, num_mixes, output_dim]

        # Multiply probabilities across output dimensions
        component_probs = tf.reduce_prod(component_probs, axis=-1)  # [batch, num_mixes]

        # Weight the probabilities by mixture weights
        weighted_probs = mix_weights * component_probs  # [batch, num_mixes]

        # Sum across mixtures and take log
        total_prob = tf.reduce_sum(weighted_probs, axis=-1)  # [batch]
        log_prob = tf.math.log(total_prob + keras.backend.epsilon())

        return -tf.reduce_mean(log_prob)

    with tf.name_scope('MDN'):
        return mdn_loss_func


def get_mixture_sampling_func(
        output_dim: int,
        num_mixes: int
) -> Callable[[tf.Tensor], tf.Tensor]:
    """Creates a sampling function for the MDN layer.

    Args:
        output_dim: Dimensionality of the output space
        num_mixes: Number of Gaussian mixtures

    Returns:
        Function that samples from the mixture distribution
    """

    def sampling_func(y_pred: tf.Tensor) -> tf.Tensor:
        """Sampling function implementation using pure TensorFlow ops."""
        # Reshape predictions
        y_pred = tf.reshape(
            y_pred,
            [-1, (2 * num_mixes * output_dim) + num_mixes],
            name='reshape_ypreds'
        )

        # Split parameters
        out_mu, out_sigma, out_pi = tf.split(
            y_pred,
            num_or_size_splits=[
                num_mixes * output_dim,
                num_mixes * output_dim,
                num_mixes
            ],
            axis=1,
            name='mdn_coef_split'
        )

        # Reshape parameters
        out_mu = tf.reshape(out_mu, [-1, num_mixes, output_dim])
        out_sigma = tf.reshape(out_sigma, [-1, num_mixes, output_dim])

        # Select mixture component
        mix_weights = softmax(out_pi, axis=-1)
        selected_components = tf.random.categorical(
            tf.math.log(mix_weights + keras.backend.epsilon()),
            1
        )  # [batch, 1]

        batch_size = tf.shape(y_pred)[0]
        batch_idx = tf.range(batch_size, dtype=tf.int32)
        gather_idx = tf.concat([
            tf.expand_dims(batch_idx, 1),
            tf.cast(selected_components, tf.int32)
        ], axis=1)

        # Select corresponding mus and sigmas
        selected_mu = tf.gather_nd(out_mu, gather_idx)  # [batch, output_dim]
        selected_sigma = tf.gather_nd(out_sigma, gather_idx)  # [batch, output_dim]

        # Sample from selected Gaussian
        epsilon = tf.random.normal(tf.shape(selected_mu))
        samples = selected_mu + selected_sigma * epsilon

        return samples

    with tf.name_scope('MDNLayer'):
        return sampling_func
