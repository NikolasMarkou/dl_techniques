"""
Mixture Density Network (MDN) Layer Implementation for Keras.

A Mixture Density Network combines a neural network with a mixture density model
to predict probability distributions rather than single point estimates. This
implementation provides a custom Keras layer that outputs parameters for a
mixture of Gaussian distributions.

Key Features:
- Models complex, multi-modal target distributions
- Outputs full probability distributions instead of point estimates
- Provides uncertainty quantification through distribution parameters
- Handles ambiguous or one-to-many mapping problems
- Enables sampling from the predicted distributions

Theory:
MDNs extend traditional neural networks by replacing the single output value
with a mixture of probability distributions (typically Gaussians). For each
input, the network outputs:
1. The means (μ) for each mixture component
2. The standard deviations (σ) for each mixture component
3. The mixture weights (π) that determine component importance

The resulting predicted distribution is:
    p(y|x) = Σ π_i(x) * N(y | μ_i(x), σ_i(x))

This allows the model to:
- Express uncertainty in its predictions
- Capture multi-modal relationships (where a single input could reasonably
  produce multiple different outputs)
- Model heteroscedastic noise (different variance at different input points)

Applications:
- Time series forecasting with uncertainty
- Control systems with multiple possible outcomes
- Robotics and reinforcement learning
- Modeling inverse problems (where multiple inputs map to the same output)
- Generative models for complex distributions
- Financial modeling with risk assessment

Usage:
    # Create an MDN layer with 2-dimensional output and 5 mixture components
    mdn_layer = MDNLayer(output_dimension=2, num_mixtures=5)

    # Build a model with the MDN layer
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        keras.layers.Dense(32, activation='relu'),
        mdn_layer
    ])

    # Compile with the MDN loss function
    model.compile(optimizer='adam', loss=mdn_layer.loss_func)

    # Train the model
    model.fit(x_train, y_train, epochs=100)

    # Generate samples from the predicted distribution
    predictions = model.predict(x_test)
    samples = mdn_layer.sample(predictions)

References:
    - Bishop, C. M. (1994). Mixture Density Networks.
    - Graves, A. (2013). Generating Sequences With Recurrent Neural Networks.
    - Ha, D., & Schmidhuber, J. (2018). World Models.
"""

import keras
import numpy as np
import tensorflow as tf
from keras import layers, backend
from typing import Dict, List, Optional, Tuple, Union, Callable, Any, cast

# ---------------------------------------------------------------------

def elu_plus_one_plus_epsilon(x: tf.Tensor) -> tf.Tensor:
    """Enhanced ELU activation to ensure positive values for standard deviations.

    This activation ensures that the output is always positive and greater than
    a small epsilon value, which is important for numerical stability when these
    values are used as standard deviations.

    Args:
        x: Input tensor

    Returns:
        tf.Tensor: Tensor with ELU activation plus one plus a small epsilon
    """
    return keras.activations.elu(x) + 1.0 + backend.epsilon()

# ---------------------------------------------------------------------


def get_point_estimate(
        model: keras.Model,
        x_data: np.ndarray,
        mdn_layer: Any
) -> np.ndarray:
    """
    Calculate point estimates from MDN outputs as the weighted average of mixture components.

    This function computes the expected value of the mixture distribution by taking
    the weighted average of the component means. The weights come from the mixture
    probabilities (pi values).

    Parameters
    ----------
    model : keras.Model
        Trained model with MDN layer
    x_data : np.ndarray
        Input data for prediction, shape [batch_size, input_dim]
    mdn_layer : MDNLayer
        The MDN layer instance used in the model

    Returns
    -------
    np.ndarray
        Point estimates with shape [batch_size, output_dim]

    Notes
    -----
    The point estimate represents the expected value of the predicted distribution,
    which is the sum of each component mean weighted by its mixture probability.

    Examples
    --------
    >>> point_estimates = get_point_estimate(model, x_test, mdn_layer)
    >>> print(f"Point prediction shape: {point_estimates.shape}")
    """
    # Get model predictions
    y_pred = model.predict(x_data)

    # Split the mixture parameters
    mu, sigma, pi_logits = mdn_layer.split_mixture_params(y_pred)

    # Convert pi_logits to probabilities
    # Shape: [batch_size, num_mixtures]
    pi = tf.nn.softmax(pi_logits, axis=-1).numpy()

    # Expand pi dimensions for broadcasting
    # Shape: [batch_size, num_mixtures, 1]
    pi_expanded = np.expand_dims(pi, axis=-1)

    # Compute weighted average of means (expected value of the mixture)
    # Shape: [batch_size, num_mixtures, output_dim]
    weighted_mu = mu.numpy() * pi_expanded

    # Sum over mixture components to get point estimates
    # Shape: [batch_size, output_dim]
    point_estimates = np.sum(weighted_mu, axis=1)

    return point_estimates

# ---------------------------------------------------------------------


def get_uncertainty(
        model: keras.Model,
        x_data: np.ndarray,
        mdn_layer: Any,
        point_estimates: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate uncertainty estimates from MDN parameters.

    This function decomposes predictive uncertainty into aleatoric uncertainty
    (data noise) and epistemic uncertainty (model uncertainty). The total
    predictive variance is the sum of these two components.

    Parameters
    ----------
    model : keras.Model
        Trained model with MDN layer
    x_data : np.ndarray
        Input data for prediction, shape [batch_size, input_dim]
    mdn_layer : MDNLayer
        The MDN layer instance used in the model
    point_estimates : np.ndarray
        Point estimates calculated from the model outputs, shape [batch_size, output_dim]

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:
            - total_variance: Total predictive variance, shape [batch_size, output_dim]
            - aleatoric_variance: Aleatoric uncertainty component, shape [batch_size, output_dim]

    Notes
    -----
    The uncertainty is decomposed as:
    1. Aleatoric uncertainty: Captured by the variance of each Gaussian component
    2. Epistemic uncertainty: Captured by the variance of the means across components

    The total predictive variance is the sum of these two components.

    Examples
    --------
    >>> total_var, aleatoric_var = get_uncertainty(model, x_test, mdn_layer, point_estimates)
    >>> epistemic_var = total_var - aleatoric_var
    >>> print(f"Average uncertainty: {np.mean(total_var):.4f}")
    """
    # Get model predictions
    # Shape: [batch_size, (2*output_dim*num_mixtures + num_mixtures)]
    y_pred = model.predict(x_data)

    # Split the mixture parameters
    # Shapes: mu, sigma: [batch_size, num_mixtures, output_dim], pi_logits: [batch_size, num_mixtures]
    mu, sigma, pi_logits = mdn_layer.split_mixture_params(y_pred)
    mu = mu.numpy()
    sigma = sigma.numpy()

    # Convert pi_logits to probabilities
    # Shape: [batch_size, num_mixtures]
    pi = tf.nn.softmax(pi_logits, axis=-1).numpy()

    # Expand pi dimensions for broadcasting
    # Shape: [batch_size, num_mixtures, 1]
    pi_expanded = np.expand_dims(pi, axis=-1)

    # Expand point estimates for broadcasting
    # Shape: [batch_size, 1, output_dim]
    point_expanded = np.expand_dims(point_estimates, axis=1)

    # 1. Aleatoric uncertainty (inherent data noise)
    # Weighted average of variances from each component
    # Shape: [batch_size, output_dim]
    aleatoric_variance = np.sum(pi_expanded * sigma ** 2, axis=1)

    # 2. Epistemic uncertainty (model uncertainty)
    # Weighted variance of means around expected value
    # Shape: [batch_size, num_mixtures, output_dim]
    squared_diff = (mu - point_expanded) ** 2

    # Shape: [batch_size, output_dim]
    epistemic_variance = np.sum(pi_expanded * squared_diff, axis=1)

    # Total predictive variance (sum of aleatoric and epistemic)
    # Shape: [batch_size, output_dim]
    total_variance = aleatoric_variance + epistemic_variance

    return total_variance, aleatoric_variance

# ---------------------------------------------------------------------


def get_prediction_intervals(
        point_estimates: np.ndarray,
        total_variance: np.ndarray,
        confidence_level: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate prediction intervals for MDN outputs.

    This function computes lower and upper bounds for prediction intervals
    based on the point estimates and their associated uncertainty.

    Parameters
    ----------
    point_estimates : np.ndarray
        Point estimates from the model, shape [batch_size, output_dim]
    total_variance : np.ndarray
        Total variance of predictions, shape [batch_size, output_dim]
    confidence_level : float, optional
        Desired confidence level, default is 0.95 (95% confidence interval)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:
            - lower_bound: Lower bounds of prediction intervals, shape [batch_size, output_dim]
            - upper_bound: Upper bounds of prediction intervals, shape [batch_size, output_dim]

    Notes
    -----
    The function assumes a Gaussian distribution for the prediction intervals.
    The bounds are calculated as point_estimate ± z_score * sqrt(variance),
    where z_score is derived from the inverse error function based on the
    specified confidence level.

    Examples
    --------
    >>> lower, upper = get_prediction_intervals(point_estimates, total_variance, 0.95)
    >>> print(f"Average interval width: {np.mean(upper - lower):.4f}")
    """
    # Calculate z-score for the given confidence level using the inverse error function
    # For 95% confidence, z_score ≈ 1.96
    z_score = tf.cast(
        tf.abs(tf.math.erfinv(confidence_level) * tf.sqrt(2.0)),
        tf.float32
    )

    # Calculate standard deviation from variance
    # Shape: [batch_size, output_dim]
    std_dev = np.sqrt(total_variance)

    # Calculate lower bounds of prediction intervals
    # Shape: [batch_size, output_dim]
    lower_bound = point_estimates - z_score * std_dev

    # Calculate upper bounds of prediction intervals
    # Shape: [batch_size, output_dim]
    upper_bound = point_estimates + z_score * std_dev

    return lower_bound, upper_bound

# ---------------------------------------------------------------------

@tf.function
def gaussian_probability(y: tf.Tensor, mu: tf.Tensor, sigma: tf.Tensor) -> tf.Tensor:
    """Compute Gaussian probability density.

    Args:
        y: Target values tensor of shape [batch_size, 1, output_dim] or [batch_size, output_dim]
        mu: Mean values tensor of shape [batch_size, num_mixtures, output_dim]
        sigma: Standard deviation tensor of shape [batch_size, num_mixtures, output_dim]

    Returns:
        tf.Tensor: Probability densities tensor of shape [batch_size, num_mixtures, output_dim]
    """
    # Ensure numerical stability with a minimum standard deviation
    sigma = tf.math.maximum(1e-6, sigma)
    sigma = tf.cast(sigma, tf.float32)

    # Compute normalized squared difference
    norm = tf.sqrt(2.0 * np.pi) * sigma
    y = tf.cast(y, tf.float32)
    mu = tf.cast(mu, tf.float32)
    norm = tf.cast(norm, tf.float32)
    exp = -0.5 * tf.square((y - mu) / sigma)

    return tf.exp(exp) / norm

# ---------------------------------------------------------------------

@keras.utils.register_keras_serializable()
class MDNLayer(layers.Layer):
    """Mixture Density Network Layer.

    This layer outputs parameters for a mixture of Gaussian distributions.
    It includes safeguards against numerical instability:
    - Uses ELU + 1 + ε activation for standard deviations
    - Uses stable softmax computation for mixture weights
    - Applies parameter validation

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
        """Initialize the MDN layer.

        Args:
            output_dimension: Dimensionality of the output space
            num_mixtures: Number of Gaussian mixtures
            kernel_initializer: Initializer for the kernel weights matrix
            kernel_regularizer: Regularizer function applied to the kernel weights matrix
            **kwargs: Additional layer arguments
        """
        super().__init__(**kwargs)

        # Parameter validation
        if output_dimension <= 0:
            raise ValueError(f"output_dimension must be positive, got {output_dimension}")
        if num_mixtures <= 0:
            raise ValueError(f"num_mixtures must be positive, got {num_mixtures}")

        self.output_dim = output_dimension
        self.num_mix = num_mixtures
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = kernel_regularizer

        # Define the sub-layers for mixture parameters
        with tf.name_scope('MDN'):
            self.mdn_mus = layers.Dense(
                self.num_mix * self.output_dim,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name='mdn_mus'
            )

            self.mdn_sigmas = layers.Dense(
                self.num_mix * self.output_dim,
                activation=elu_plus_one_plus_epsilon,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name='mdn_sigmas'
            )

            self.mdn_pi = layers.Dense(
                self.num_mix,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name='mdn_pi'
            )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer with the given input shape.

        Args:
            input_shape: Shape tuple of the input
        """
        with tf.name_scope('mus'):
            self.mdn_mus.build(input_shape)
        with tf.name_scope('sigmas'):
            self.mdn_sigmas.build(input_shape)
        with tf.name_scope('pis'):
            self.mdn_pi.build(input_shape)

        super().build(input_shape)

    @property
    def trainable_weights(self) -> List[tf.Variable]:
        """Get trainable weights of all components.

        Returns:
            List[tf.Variable]: List of trainable weights
        """
        return (self.mdn_mus.trainable_weights +
                self.mdn_sigmas.trainable_weights +
                self.mdn_pi.trainable_weights)

    @property
    def non_trainable_weights(self) -> List[tf.Variable]:
        """Get non-trainable weights of all components.

        Returns:
            List[tf.Variable]: List of non-trainable weights
        """
        return (self.mdn_mus.non_trainable_weights +
                self.mdn_sigmas.non_trainable_weights +
                self.mdn_pi.non_trainable_weights)

    def call(self, x: tf.Tensor, mask: Optional[tf.Tensor] = None, training: Optional[bool] = None) -> tf.Tensor:
        """Forward pass of the layer.

        Args:
            x: Input tensor
            mask: Optional mask tensor
            training: Boolean indicating whether the layer should behave in training mode

        Returns:
            tf.Tensor: Output tensor containing mixture parameters
        """
        with tf.name_scope('MDN'):
            mu_output = self.mdn_mus(x, training=training)
            sigma_output = self.mdn_sigmas(x, training=training)
            pi_output = self.mdn_pi(x, training=training)

            # Add TensorBoard metrics during training if summaries are enabled
            if training and tf.summary.experimental.get_step() is not None:
                tf.summary.histogram('mu_values', mu_output, step=tf.summary.experimental.get_step())
                tf.summary.histogram('sigma_values', sigma_output, step=tf.summary.experimental.get_step())
                tf.summary.histogram('pi_logits', pi_output, step=tf.summary.experimental.get_step())

            return layers.concatenate(
                [mu_output, sigma_output, pi_output],
                name='mdn_outputs'
            )

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape tuple of the input

        Returns:
            Tuple[Optional[int], ...]: Shape tuple of the output
        """
        return (input_shape[0], (2 * self.output_dim * self.num_mix) + self.num_mix)

    @tf.function
    def split_mixture_params(self, y_pred: tf.Tensor, axis: int = -1) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Split the mixture parameters into components.

        Args:
            y_pred: Predicted parameters tensor from the MDN layer
            axis: Axis along which to split the tensor

        Returns:
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor]: Tuple of (mu, sigma, pi) tensors
        """
        # Ensure correct shape
        if y_pred.shape[-1] != (2 * self.output_dim * self.num_mix) + self.num_mix:
            y_pred = tf.reshape(
                y_pred,
                [-1, (2 * self.output_dim * self.num_mix) + self.num_mix],
                name='reshape_ypreds'
            )

        # Split parameters
        out_mu, out_sigma, out_pi = tf.split(
            y_pred,
            num_or_size_splits=[
                self.num_mix * self.output_dim,
                self.num_mix * self.output_dim,
                self.num_mix
            ],
            axis=axis,
            name='mdn_coef_split'
        )

        # Reshape mus and sigmas to [batch, num_mixtures, output_dim]
        batch_size = tf.shape(y_pred)[0]
        out_mu = tf.reshape(out_mu, [batch_size, self.num_mix, self.output_dim])
        out_sigma = tf.reshape(out_sigma, [batch_size, self.num_mix, self.output_dim])

        return out_mu, out_sigma, out_pi

    def loss_func(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """MDN loss function implementation using negative log likelihood.

        Args:
            y_true: Target values tensor of shape [batch_size, output_dim]
            y_pred: Predicted parameters tensor from the MDN layer

        Returns:
            tf.Tensor: Negative log likelihood loss value
        """
        with tf.name_scope('MDN_Loss'):
            # Reshape target if needed
            y_true = tf.reshape(y_true, [-1, self.output_dim], name='reshape_ytrue')

            # Split the parameters
            out_mu, out_sigma, out_pi = self.split_mixture_params(y_pred)

            # Compute mixture weights using Keras softmax
            mix_weights = keras.activations.softmax(out_pi, axis=-1)

            # Add TensorBoard summary if enabled
            if tf.summary.experimental.get_step() is not None:
                tf.summary.histogram('mixture_weights', mix_weights, step=tf.summary.experimental.get_step())

            # Expand y_true for broadcasting with mixture components
            y_true_expanded = tf.expand_dims(y_true, 1)  # [batch, 1, output_dim]

            # Compute Gaussian probabilities for each component
            component_probs = gaussian_probability(
                y_true_expanded,
                out_mu,
                out_sigma
            )  # [batch, num_mixes, output_dim]

            # Multiply probabilities across output dimensions (assuming independence)
            component_probs = tf.reduce_prod(component_probs, axis=-1)  # [batch, num_mixes]

            # Weight the probabilities by mixture weights
            weighted_probs = mix_weights * component_probs  # [batch, num_mixes]

            # Sum across mixtures (total probability from all components)
            total_prob = tf.reduce_sum(weighted_probs, axis=-1)  # [batch]

            # Prevent log(0) with improved numerical stability
            total_prob = tf.math.maximum(total_prob, 1e-10)
            log_prob = tf.math.log(total_prob)

            # Calculate negative log likelihood
            loss = -tf.reduce_mean(log_prob)

            # Add TensorBoard loss tracking if enabled
            if tf.summary.experimental.get_step() is not None:
                tf.summary.scalar('mdn_loss', loss, step=tf.summary.experimental.get_step())

            return loss

    @tf.function
    def sample(self, y_pred: tf.Tensor, temp: float = 1.0, seed: Optional[int] = None) -> tf.Tensor:
        """Sample from the mixture distribution.

        Args:
            y_pred: Predicted parameters from the MDN layer
            temp: Temperature parameter for sampling (higher = more random)
            seed: Optional seed for reproducible sampling

        Returns:
            tf.Tensor: Samples from the predicted distribution with shape [batch_size, output_dim]
        """
        with tf.name_scope('MDN_Sampling'):
            # Split the parameters
            out_mu, out_sigma, out_pi = self.split_mixture_params(y_pred, axis=1)

            # Ensure numerical stability for sigma values
            out_sigma = tf.math.maximum(out_sigma, 1e-6)

            # Control sampling randomness with temperature
            if temp != 1.0:
                out_pi = out_pi / temp

            # Compute mixture weights
            mix_weights = keras.activations.softmax(out_pi, axis=-1)

            # For reproducibility, we need to set seeds for both random operations
            if seed is not None:
                # Create two seeds from the original one to avoid correlation
                categorical_seed = seed
                normal_seed = seed + 1

                # Seed the TensorFlow random generators
                tf.random.set_seed(seed)
            else:
                categorical_seed = None
                normal_seed = None

            # Draw samples from categorical distribution
            log_mix_weights = tf.math.log(mix_weights + backend.epsilon())
            selected_components = tf.random.stateless_categorical(
                log_mix_weights,
                1,
                seed=[categorical_seed or 42, categorical_seed or 0]
            )  # [batch, 1]

            # Create indices for gather_nd
            batch_size = tf.shape(y_pred)[0]
            batch_idx = tf.range(batch_size, dtype=tf.int32)
            batch_idx = tf.expand_dims(batch_idx, 1)
            gather_idx = tf.concat([batch_idx, tf.cast(selected_components, tf.int32)], axis=1)

            # Select corresponding mus and sigmas for the chosen components
            selected_mu = tf.gather_nd(out_mu, gather_idx)  # [batch, output_dim]
            selected_sigma = tf.gather_nd(out_sigma, gather_idx)  # [batch, output_dim]

            # Sample from selected Gaussian with stateless random normal for reproducibility
            normal_shape = tf.shape(selected_mu)
            rng_state = tf.random.get_global_generator().state

            if seed is not None:
                epsilon = tf.random.stateless_normal(
                    normal_shape,
                    seed=[normal_seed or 42, normal_seed or 0]
                )
            else:
                epsilon = tf.random.normal(normal_shape)

            samples = selected_mu + selected_sigma * epsilon

            return samples

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        Returns:
            Dict[str, Any]: Layer configuration dictionary
        """
        config = {
            "output_dimension": self.output_dim,
            "num_mixtures": self.num_mix,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer)
        }
        base_config = super().get_config()
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MDNLayer":
        """Create a layer from its config.

        Args:
            config: Dictionary with the layer configuration

        Returns:
            MDNLayer: A new MDN layer instance
        """
        config_copy = config.copy()
        config_copy["kernel_initializer"] = keras.initializers.deserialize(config["kernel_initializer"])
        if config["kernel_regularizer"] is not None:
            config_copy["kernel_regularizer"] = keras.regularizers.deserialize(config["kernel_regularizer"])
        return cls(**config_copy)


# ---------------------------------------------------------------------
