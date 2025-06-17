"""
Mixture Density Network (MDN) Layer Implementation for Keras 3.x.

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

Applications:
    - Time series forecasting with uncertainty
    - Control systems with multiple possible outcomes
    - Robotics and reinforcement learning
    - Modeling inverse problems
    - Financial modeling with risk assessment

References:
    - Bishop, C. M. (1994). Mixture Density Networks.
    - Graves, A. (2013). Generating Sequences With Recurrent Neural Networks.
    - Ha, D., & Schmidhuber, J. (2018). World Models.
"""

import keras
from keras import ops
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dl_techniques.utils.logger import logger


def elu_plus_one_plus_epsilon(x: keras.KerasTensor) -> keras.KerasTensor:
    """Enhanced ELU activation to ensure positive values for standard deviations.

    This activation ensures that the output is always positive and greater than
    a small epsilon value, which is important for numerical stability when these
    values are used as standard deviations.

    Parameters
    ----------
    x : keras.KerasTensor
        Input tensor

    Returns
    -------
    keras.KerasTensor
        Tensor with ELU activation plus one plus a small epsilon
    """
    return keras.activations.elu(x) + 1.0 + keras.backend.epsilon()


def gaussian_probability(y: keras.KerasTensor, mu: keras.KerasTensor, sigma: keras.KerasTensor) -> keras.KerasTensor:
    """Compute Gaussian probability density using Keras operations.

    Parameters
    ----------
    y : keras.KerasTensor
        Target values tensor of shape [batch_size, 1, output_dim] or [batch_size, output_dim]
    mu : keras.KerasTensor
        Mean values tensor of shape [batch_size, num_mixtures, output_dim]
    sigma : keras.KerasTensor
        Standard deviation tensor of shape [batch_size, num_mixtures, output_dim]

    Returns
    -------
    keras.KerasTensor
        Probability densities tensor of shape [batch_size, num_mixtures, output_dim]
    """
    # Ensure numerical stability with a minimum standard deviation
    sigma = ops.maximum(1e-6, sigma)
    sigma = ops.cast(sigma, "float32")

    # Compute normalized squared difference
    norm = ops.sqrt(2.0 * np.pi) * sigma
    y = ops.cast(y, "float32")
    mu = ops.cast(mu, "float32")
    norm = ops.cast(norm, "float32")
    exp_term = -0.5 * ops.square((y - mu) / sigma)

    return ops.exp(exp_term) / norm


@keras.saving.register_keras_serializable()
class MDNLayer(keras.layers.Layer):
    """Mixture Density Network Layer for Keras 3.x.

    This layer outputs parameters for a mixture of Gaussian distributions.
    It includes safeguards against numerical instability and follows Keras 3.x
    best practices for serialization and backend compatibility.

    Parameters
    ----------
    output_dimension : int
        Dimensionality of the output space. Must be positive.
    num_mixtures : int
        Number of Gaussian mixtures. Must be positive.
    kernel_initializer : Union[str, keras.initializers.Initializer], optional
        Initializer for the kernel weights matrix, by default "glorot_uniform"
    bias_initializer : Union[str, keras.initializers.Initializer], optional
        Initializer for the bias vectors, by default "zeros"
    kernel_regularizer : Optional[keras.regularizers.Regularizer], optional
        Regularizer function applied to the kernel weights matrix, by default None
    bias_regularizer : Optional[keras.regularizers.Regularizer], optional
        Regularizer function applied to the bias vectors, by default None
    **kwargs
        Additional layer arguments

    Raises
    ------
    ValueError
        If output_dimension or num_mixtures are not positive integers

    Examples
    --------
    >>> # Create an MDN layer with 2D output and 5 mixture components
    >>> mdn_layer = MDNLayer(output_dimension=2, num_mixtures=5)
    >>>
    >>> # Build a model
    >>> model = keras.Sequential([
    ...     keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
    ...     keras.layers.Dense(32, activation='relu'),
    ...     mdn_layer
    ... ])
    >>>
    >>> # Compile with the MDN loss function
    >>> model.compile(optimizer='adam', loss=mdn_layer.loss_func)
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
        """Initialize the MDN layer.

        Parameters
        ----------
        output_dimension : int
            Dimensionality of the output space
        num_mixtures : int
            Number of Gaussian mixtures
        kernel_initializer : Union[str, keras.initializers.Initializer], optional
            Initializer for the kernel weights matrix
        bias_initializer : Union[str, keras.initializers.Initializer], optional
            Initializer for the bias vectors
        kernel_regularizer : Optional[keras.regularizers.Regularizer], optional
            Regularizer function applied to the kernel weights matrix
        bias_regularizer : Optional[keras.regularizers.Regularizer], optional
            Regularizer function applied to the bias vectors
        **kwargs : Any
            Additional layer arguments
        """
        super().__init__(**kwargs)

        # Parameter validation
        if output_dimension <= 0:
            raise ValueError(f"output_dimension must be positive, got {output_dimension}")
        if num_mixtures <= 0:
            raise ValueError(f"num_mixtures must be positive, got {num_mixtures}")

        # Store configuration parameters
        self.output_dim = output_dimension
        self.num_mix = num_mixtures
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # Initialize sublayers to None - will be created in build()
        self.mdn_mus = None
        self.mdn_sigmas = None
        self.mdn_pi = None

        # Store build shape for serialization
        self._build_input_shape = None

        logger.info(f"Initialized MDN layer with {num_mixtures} mixtures and {output_dimension}D output")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer weights and sublayers based on input shape.

        Parameters
        ----------
        input_shape : Tuple[Optional[int], ...]
            Shape tuple of the input tensor
        """
        # Store input shape for serialization
        self._build_input_shape = input_shape

        # Create sublayers for mixture parameters
        self.mdn_mus = keras.layers.Dense(
            self.num_mix * self.output_dim,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='mdn_mus'
        )

        self.mdn_sigmas = keras.layers.Dense(
            self.num_mix * self.output_dim,
            activation=elu_plus_one_plus_epsilon,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='mdn_sigmas'
        )

        self.mdn_pi = keras.layers.Dense(
            self.num_mix,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='mdn_pi'
        )

        # Build sublayers explicitly
        self.mdn_mus.build(input_shape)
        self.mdn_sigmas.build(input_shape)
        self.mdn_pi.build(input_shape)

        super().build(input_shape)
        logger.debug(f"MDN layer built with input shape: {input_shape}")

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Forward pass of the layer.

        Parameters
        ----------
        inputs : keras.KerasTensor
            Input tensor
        training : Optional[bool], optional
            Boolean indicating whether the layer should behave in training mode, by default None

        Returns
        -------
        keras.KerasTensor
            Output tensor containing concatenated mixture parameters
        """
        # Compute mixture parameters using sublayers
        mu_output = self.mdn_mus(inputs, training=training)
        sigma_output = self.mdn_sigmas(inputs, training=training)
        pi_output = self.mdn_pi(inputs, training=training)

        # Concatenate all parameters
        return keras.layers.concatenate(
            [mu_output, sigma_output, pi_output],
            name='mdn_outputs'
        )

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        Parameters
        ----------
        input_shape : Tuple[Optional[int], ...]
            Shape tuple of the input

        Returns
        -------
        Tuple[Optional[int], ...]
            Shape tuple of the output
        """
        # Convert to list for manipulation, then back to tuple
        input_shape_list = list(input_shape)
        output_size = (2 * self.output_dim * self.num_mix) + self.num_mix
        return tuple(input_shape_list[:-1] + [output_size])

    def split_mixture_params(self, y_pred: keras.KerasTensor) -> Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]:
        """Split the mixture parameters into components.

        Parameters
        ----------
        y_pred : keras.KerasTensor
            Predicted parameters tensor from the MDN layer

        Returns
        -------
        Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]
            Tuple of (mu, sigma, pi) tensors
        """
        # Split parameters along the last axis
        mu_end = self.num_mix * self.output_dim
        sigma_end = mu_end + (self.num_mix * self.output_dim)

        out_mu = y_pred[..., :mu_end]
        out_sigma = y_pred[..., mu_end:sigma_end]
        out_pi = y_pred[..., sigma_end:]

        # Reshape mus and sigmas to [batch, num_mixtures, output_dim]
        batch_size = ops.shape(y_pred)[0]
        out_mu = ops.reshape(out_mu, [batch_size, self.num_mix, self.output_dim])
        out_sigma = ops.reshape(out_sigma, [batch_size, self.num_mix, self.output_dim])

        return out_mu, out_sigma, out_pi

    def loss_func(self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        """MDN loss function implementation using negative log likelihood.

        This function computes the negative log likelihood of the target values
        under the predicted mixture of Gaussians.

        Parameters
        ----------
        y_true : keras.KerasTensor
            Target values tensor of shape [batch_size, output_dim]
        y_pred : keras.KerasTensor
            Predicted parameters tensor from the MDN layer

        Returns
        -------
        keras.KerasTensor
            Negative log likelihood loss value
        """
        # Reshape target if needed
        y_true = ops.reshape(y_true, [-1, self.output_dim])

        # Split the parameters
        out_mu, out_sigma, out_pi = self.split_mixture_params(y_pred)

        # Compute mixture weights using softmax
        mix_weights = keras.activations.softmax(out_pi, axis=-1)

        # Expand y_true for broadcasting with mixture components
        y_true_expanded = ops.expand_dims(y_true, 1)  # [batch, 1, output_dim]

        # Compute Gaussian probabilities for each component
        component_probs = gaussian_probability(
            y_true_expanded,
            out_mu,
            out_sigma
        )  # [batch, num_mixes, output_dim]

        # Multiply probabilities across output dimensions (assuming independence)
        component_probs = ops.prod(component_probs, axis=-1)  # [batch, num_mixes]

        # Weight the probabilities by mixture weights
        weighted_probs = mix_weights * component_probs  # [batch, num_mixes]

        # Sum across mixtures (total probability from all components)
        total_prob = ops.sum(weighted_probs, axis=-1)  # [batch]

        # Prevent log(0) with improved numerical stability
        total_prob = ops.maximum(total_prob, 1e-10)
        log_prob = ops.log(total_prob)

        # Calculate negative log likelihood
        loss = -ops.mean(log_prob)

        return loss

    def sample(self, y_pred: keras.KerasTensor, temperature: float = 1.0) -> keras.KerasTensor:
        """Sample from the mixture distribution.

        Parameters
        ----------
        y_pred : keras.KerasTensor
            Predicted parameters from the MDN layer
        temperature : float, optional
            Temperature parameter for sampling (higher = more random), by default 1.0

        Returns
        -------
        keras.KerasTensor
            Samples from the predicted distribution with shape [batch_size, output_dim]
        """
        # Split the parameters
        out_mu, out_sigma, out_pi = self.split_mixture_params(y_pred)

        # Ensure numerical stability for sigma values
        out_sigma = ops.maximum(out_sigma, 1e-6)

        # Control sampling randomness with temperature
        if temperature != 1.0:
            out_pi = out_pi / temperature

        # Compute mixture weights
        mix_weights = keras.activations.softmax(out_pi, axis=-1)

        # Draw samples from categorical distribution to select mixture component
        # Using gumbel-max trick for differentiable sampling
        gumbel_noise = -ops.log(-ops.log(keras.random.uniform(ops.shape(out_pi))))
        log_mix_weights = ops.log(mix_weights + keras.backend.epsilon())
        selected_logits = log_mix_weights + gumbel_noise
        selected_components = ops.argmax(selected_logits, axis=-1)  # [batch_size]

        # Convert selected components to one-hot encoding for selection
        one_hot = ops.one_hot(selected_components, num_classes=self.num_mix)  # [batch_size, num_mixtures]
        one_hot_expanded = ops.expand_dims(one_hot, -1)  # [batch_size, num_mixtures, 1]

        # Select corresponding mus and sigmas using one-hot encoding
        selected_mu = ops.sum(out_mu * one_hot_expanded, axis=1)  # [batch_size, output_dim]
        selected_sigma = ops.sum(out_sigma * one_hot_expanded, axis=1)  # [batch_size, output_dim]

        # Sample from selected Gaussian
        epsilon = keras.random.normal(ops.shape(selected_mu))
        samples = selected_mu + selected_sigma * epsilon

        return samples

    def get_config(self) -> Dict[str, Any]:
        """Returns the layer configuration for serialization.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing the layer configuration
        """
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

    def get_build_config(self) -> Dict[str, Any]:
        """Get the config needed to build the layer from a config.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing the build configuration
        """
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build the layer from a config created with get_build_config.

        Parameters
        ----------
        config : Dict[str, Any]
            Dictionary containing the build configuration
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MDNLayer":
        """Create a layer from its config.

        Parameters
        ----------
        config : Dict[str, Any]
            Dictionary with the layer configuration

        Returns
        -------
        MDNLayer
            A new MDN layer instance
        """
        config_copy = config.copy()
        config_copy["kernel_initializer"] = keras.initializers.deserialize(config["kernel_initializer"])
        config_copy["bias_initializer"] = keras.initializers.deserialize(config["bias_initializer"])

        if config["kernel_regularizer"] is not None:
            config_copy["kernel_regularizer"] = keras.regularizers.deserialize(config["kernel_regularizer"])
        if config["bias_regularizer"] is not None:
            config_copy["bias_regularizer"] = keras.regularizers.deserialize(config["bias_regularizer"])

        return cls(**config_copy)


def get_point_estimate(
    model: keras.Model,
    x_data: np.ndarray,
    mdn_layer: MDNLayer
) -> np.ndarray:
    """Calculate point estimates from MDN outputs as the weighted average of mixture components.

    This function computes the expected value of the mixture distribution by taking
    the weighted average of the component means.

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
    pi = keras.activations.softmax(pi_logits, axis=-1)

    # Convert to numpy for computation
    mu_np = ops.convert_to_numpy(mu)
    pi_np = ops.convert_to_numpy(pi)

    # Expand pi dimensions for broadcasting
    pi_expanded = np.expand_dims(pi_np, axis=-1)

    # Compute weighted average of means (expected value of the mixture)
    weighted_mu = mu_np * pi_expanded

    # Sum over mixture components to get point estimates
    point_estimates = np.sum(weighted_mu, axis=1)

    return point_estimates


def get_uncertainty(
    model: keras.Model,
    x_data: np.ndarray,
    mdn_layer: MDNLayer,
    point_estimates: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate uncertainty estimates from MDN parameters.

    This function decomposes predictive uncertainty into aleatoric uncertainty
    (data noise) and epistemic uncertainty (model uncertainty).

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

    Examples
    --------
    >>> total_var, aleatoric_var = get_uncertainty(model, x_test, mdn_layer, point_estimates)
    >>> epistemic_var = total_var - aleatoric_var
    >>> logger.info(f"Average uncertainty: {np.mean(total_var):.4f}")
    """
    # Get model predictions
    y_pred = model.predict(x_data)

    # Split the mixture parameters
    mu, sigma, pi_logits = mdn_layer.split_mixture_params(y_pred)

    # Convert to numpy
    mu_np = ops.convert_to_numpy(mu)
    sigma_np = ops.convert_to_numpy(sigma)
    pi_np = ops.convert_to_numpy(keras.activations.softmax(pi_logits, axis=-1))

    # Expand pi dimensions for broadcasting
    pi_expanded = np.expand_dims(pi_np, axis=-1)

    # Expand point estimates for broadcasting
    point_expanded = np.expand_dims(point_estimates, axis=1)

    # 1. Aleatoric uncertainty (inherent data noise)
    aleatoric_variance = np.sum(pi_expanded * sigma_np ** 2, axis=1)

    # 2. Epistemic uncertainty (model uncertainty)
    squared_diff = (mu_np - point_expanded) ** 2
    epistemic_variance = np.sum(pi_expanded * squared_diff, axis=1)

    # Total predictive variance
    total_variance = aleatoric_variance + epistemic_variance

    return total_variance, aleatoric_variance


def get_prediction_intervals(
    point_estimates: np.ndarray,
    total_variance: np.ndarray,
    confidence_level: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate prediction intervals for MDN outputs.

    Parameters
    ----------
    point_estimates : np.ndarray
        Point estimates from the model, shape [batch_size, output_dim]
    total_variance : np.ndarray
        Total variance of predictions, shape [batch_size, output_dim]
    confidence_level : float, optional
        Desired confidence level, by default 0.95

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:
            - lower_bound: Lower bounds of prediction intervals
            - upper_bound: Upper bounds of prediction intervals

    Examples
    --------
    >>> lower, upper = get_prediction_intervals(point_estimates, total_variance, 0.95)
    >>> logger.info(f"Average interval width: {np.mean(upper - lower):.4f}")
    """
    from scipy import stats

    # Calculate z-score for the given confidence level
    alpha = 1.0 - confidence_level
    z_score = stats.norm.ppf(1 - alpha / 2)

    # Calculate standard deviation from variance
    std_dev = np.sqrt(total_variance)

    # Calculate bounds
    lower_bound = point_estimates - z_score * std_dev
    upper_bound = point_estimates + z_score * std_dev

    return lower_bound, upper_bound