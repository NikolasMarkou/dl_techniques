"""
Mixture Density Network (MDN) Layer

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
import numpy as np
from keras import ops
from typing import Dict, Optional, Tuple, Union, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.tensors import gaussian_probability

# ---------------------------------------------------------------------

EPSILON_SIGMA = 1e-6

# ---------------------------------------------------------------------


def elu_plus_one_plus_epsilon(x: keras.KerasTensor) -> keras.KerasTensor:
    """Enhanced ELU activation to ensure positive values for standard deviations.

    This activation ensures that the output is always positive and greater than
    a small epsilon value, which is important for numerical stability when these
    values are used as standard deviations.

    Mathematical form: ELU(x) + 1 + ε
    - ELU(x) can be negative for x < 0, but approaches -1 asymptotically
    - Adding 1 ensures the result is always ≥ ε > 0
    - This prevents division by zero or log(0) in probability calculations

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

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class MDNLayer(keras.layers.Layer):
    """Mixture Density Network Layer

    This layer outputs parameters for a mixture of Gaussian distributions.
    It includes safeguards against numerical instability and follows Keras 3.x
    best practices for serialization and backend compatibility.

    The layer creates three separate dense layers internally:
    1. mdn_mus: Outputs means (μ) for each mixture component and output dimension
    2. mdn_sigmas: Outputs standard deviations (σ) with positivity constraint
    3. mdn_pi: Outputs mixture weights (π) as unnormalized logits

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

        Creates three Dense sublayers:
        1. mdn_mus: num_mix * output_dim outputs (means for each component and dimension)
        2. mdn_sigmas: num_mix * output_dim outputs (std devs, forced positive)
        3. mdn_pi: num_mix outputs (mixture weights as logits)

        Parameters
        ----------
        input_shape : Tuple[Optional[int], ...]
            Shape tuple of the input tensor
        """
        # Store input shape for serialization
        self._build_input_shape = input_shape

        # MEAN OUTPUTS: μ parameters
        # Creates outputs for means of each Gaussian component
        # Shape: [batch_size, num_mix * output_dim]
        # Will be reshaped to [batch_size, num_mix, output_dim] later
        self.mdn_mus = keras.layers.Dense(
            self.num_mix * self.output_dim,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='mdn_mus'
        )

        # SIGMA OUTPUTS: σ parameters (standard deviations)
        # Notice the activation that forces the output to be always positive
        # This is crucial because σ must be > 0 for valid Gaussian distributions
        # Shape: [batch_size, num_mix * output_dim]
        self.mdn_sigmas = keras.layers.Dense(
            self.num_mix * self.output_dim,
            activation=elu_plus_one_plus_epsilon,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='mdn_sigmas'
        )

        # MIXTURE WEIGHTS: π parameters (mixing coefficients)
        # Outputs unnormalized logits that will be converted to probabilities via softmax
        # These determine the relative importance/probability of each mixture component
        # Shape: [batch_size, num_mix]
        self.mdn_pi = keras.layers.Dense(
            self.num_mix,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='mdn_pi'
        )

        # Build sublayers explicitly to ensure proper weight initialization
        self.mdn_mus.build(input_shape)
        self.mdn_sigmas.build(input_shape)
        self.mdn_pi.build(input_shape)

        super().build(input_shape)
        logger.debug(f"MDN layer built with input shape: {input_shape}")

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass of the layer.

        Computes all mixture parameters and concatenates them into a single output tensor.
        The output structure is: [mu_params, sigma_params, pi_params]

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
            Shape: [batch_size, (2 * num_mix * output_dim) + num_mix]
        """
        # Compute mixture parameters using sublayers
        # μ parameters: means for each mixture component and output dimension
        mu_output = self.mdn_mus(inputs, training=training)

        # σ parameters: standard deviations (automatically made positive by activation)
        sigma_output = self.mdn_sigmas(inputs, training=training)
        # Additional safety clamp to prevent numerical issues
        # Even with the positive activation, ensure σ ≥ ε for stability
        sigma_output = ops.maximum(sigma_output, EPSILON_SIGMA)

        # π parameters: mixture weights as unnormalized logits
        # Will be converted to probabilities later via softmax
        pi_output = self.mdn_pi(inputs, training=training)

        # Concatenate all parameters into single output tensor
        # Structure: [μ₁, μ₂, ..., μₙ, σ₁, σ₂, ..., σₙ, π₁, π₂, ..., πₘ]
        # where n = num_mix * output_dim, m = num_mix
        return keras.layers.concatenate(
            [mu_output, sigma_output, pi_output],
            name='mdn_outputs'
        )

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        The output contains:
        - num_mix * output_dim values for μ parameters
        - num_mix * output_dim values for σ parameters
        - num_mix values for π parameters
        Total: (2 * num_mix * output_dim) + num_mix

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
        # Calculate total output size: means + sigmas + mixing weights
        output_size = (2 * self.output_dim * self.num_mix) + self.num_mix
        return tuple(input_shape_list[:-1] + [output_size])

    def split_mixture_params(
            self,
            y_pred: keras.KerasTensor
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]:
        """Split the mixture parameters into components.

        Separates the concatenated output into individual parameter tensors
        and reshapes them for easier mathematical operations.

        Parameters
        ----------
        y_pred : keras.KerasTensor
            Predicted parameters tensor from the MDN layer
            Shape: [batch_size, (2 * num_mix * output_dim) + num_mix]

        Returns
        -------
        Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]
            Tuple of (mu, sigma, pi) tensors with shapes:
            - mu: [batch_size, num_mix, output_dim]
            - sigma: [batch_size, num_mix, output_dim]
            - pi: [batch_size, num_mix]
        """
        # Calculate split points in the concatenated tensor
        # First: μ parameters (means)
        mu_end = self.num_mix * self.output_dim
        # Second: σ parameters (standard deviations)
        sigma_end = mu_end + (self.num_mix * self.output_dim)
        # Third: π parameters (mixture weights) - everything remaining

        # Split the concatenated tensor using calculated indices
        out_mu = y_pred[..., :mu_end]                    # μ parameters
        out_sigma = y_pred[..., mu_end:sigma_end]        # σ parameters
        out_pi = y_pred[..., sigma_end:]                 # π parameters

        # Reshape μ and σ from flat vectors to [batch, mixtures, dimensions]
        # This makes it easier to perform per-component calculations
        batch_size = ops.shape(y_pred)[0]

        # Reshape means: [batch, num_mix * output_dim] → [batch, num_mix, output_dim]
        out_mu = ops.reshape(out_mu, [batch_size, self.num_mix, self.output_dim])

        # Reshape standard deviations: same transformation as means
        out_sigma = ops.reshape(out_sigma, [batch_size, self.num_mix, self.output_dim])

        # π parameters are already in correct shape [batch, num_mix]
        return out_mu, out_sigma, out_pi

    def loss_func(
            self,
            y_true: keras.KerasTensor,
            y_pred: keras.KerasTensor
    ) -> keras.KerasTensor:
        """MDN loss function implementation using negative log likelihood.

        This function computes the negative log likelihood of the target values
        under the predicted mixture of Gaussians. The mathematical formulation:

        L = -log(Σᵢ πᵢ * N(y_true | μᵢ, σᵢ))

        where:
        - πᵢ are the mixture weights (after softmax normalization)
        - N(y_true | μᵢ, σᵢ) is the Gaussian probability density
        - The sum is over all mixture components

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
        # Ensure y_true has correct shape for mixture calculations
        # Reshape to [batch_size, output_dim] if needed
        y_true = ops.reshape(y_true, [-1, self.output_dim])

        # Extract mixture parameters from network output
        out_mu, out_sigma, out_pi = self.split_mixture_params(y_pred)

        # Convert mixture weight logits to probabilities using softmax
        # This ensures Σᵢ πᵢ = 1 (valid probability distribution)
        # Shape: [batch_size, num_mix]
        mix_weights = keras.activations.softmax(out_pi, axis=-1)

        # Expand y_true for broadcasting with mixture components
        # [batch_size, output_dim] → [batch_size, 1, output_dim]
        # This allows element-wise operations with all mixture components simultaneously
        y_true_expanded = ops.expand_dims(y_true, 1)

        # Compute Gaussian probability density for each mixture component
        # gaussian_probability computes: (2π)^(-d/2) * |Σ|^(-1/2) * exp(-½(x-μ)ᵀΣ⁻¹(x-μ))
        # For diagonal covariance: |Σ| = Πⱼσⱼ², Σ⁻¹ = diag(1/σⱼ²)
        # Shape: [batch_size, num_mix, output_dim]
        component_probs = gaussian_probability(
            y_true_expanded,  # [batch, 1, output_dim]
            out_mu,          # [batch, num_mix, output_dim]
            out_sigma        # [batch, num_mix, output_dim]
        )

        # Multiply probabilities across output dimensions (assuming independence)
        # This gives the joint probability for all output dimensions
        # Πⱼ p(yⱼ | μᵢⱼ, σᵢⱼ) for each mixture component i
        # Shape: [batch_size, num_mix]
        component_probs = ops.prod(component_probs, axis=-1)

        # Weight each component probability by its mixture weight
        # πᵢ * N(y | μᵢ, σᵢ) for each component
        # Shape: [batch_size, num_mix]
        weighted_probs = mix_weights * component_probs

        # Sum across all mixture components to get total probability
        # Σᵢ πᵢ * N(y | μᵢ, σᵢ) - this is the mixture distribution probability
        # Shape: [batch_size]
        total_prob = ops.sum(weighted_probs, axis=-1)

        # Prevent log(0) with improved numerical stability
        # Clamp to minimum value to avoid -∞ in log calculation
        total_prob = ops.maximum(total_prob, EPSILON_SIGMA)

        # Compute log probability for each sample
        log_prob = ops.log(total_prob)

        # Calculate negative log likelihood (NLL) loss
        # Take negative because we want to maximize likelihood (minimize NLL)
        # Average across batch dimension for final loss value
        loss = -ops.mean(log_prob)

        # Additional safety: ensure loss is non-negative
        return ops.maximum(loss, 0.0)

    def sample(self, y_pred: keras.KerasTensor, temperature: float = 1.0) -> keras.KerasTensor:
        """Sample from the mixture distribution.

        Performs ancestral sampling from the mixture:
        1. Sample mixture component using categorical distribution over π weights
        2. Sample from the selected Gaussian component N(μᵢ, σᵢ)

        Parameters
        ----------
        y_pred : keras.KerasTensor
            Predicted parameters from the MDN layer
        temperature : float, optional
            Temperature parameter for sampling (higher = more random), by default 1.0
            - temperature > 1: more uniform/random sampling
            - temperature < 1: more peaked/deterministic sampling
            - temperature = 1: unmodified probabilities

        Returns
        -------
        keras.KerasTensor
            Samples from the predicted distribution with shape [batch_size, output_dim]
        """
        # Extract mixture parameters
        out_mu, out_sigma, out_pi = self.split_mixture_params(y_pred)

        # Ensure numerical stability for standard deviations
        # Even with built-in safeguards, add extra protection
        out_sigma = ops.maximum(out_sigma, 1e-6)

        # Apply temperature scaling to mixture weights
        # Higher temperature → more uniform sampling
        # Lower temperature → more peaked sampling around dominant components
        if temperature != 1.0:
            out_pi = out_pi / temperature

        # Convert logits to probabilities for mixture component selection
        # Shape: [batch_size, num_mix]
        mix_weights = keras.activations.softmax(out_pi, axis=-1)

        # Sample mixture components using Gumbel-Max trick for differentiable sampling
        # This avoids the need for explicit categorical sampling which isn't differentiable

        # Generate Gumbel noise: -log(-log(U)) where U ~ Uniform(0,1)
        # Shape: [batch_size, num_mix]
        gumbel_noise = -ops.log(-ops.log(keras.random.uniform(ops.shape(out_pi))))

        # Add Gumbel noise to log probabilities
        log_mix_weights = ops.log(mix_weights + keras.backend.epsilon())
        selected_logits = log_mix_weights + gumbel_noise

        # Select component with highest noisy log probability
        # Shape: [batch_size] (indices of selected components)
        selected_components = ops.argmax(selected_logits, axis=-1)

        # Convert selected component indices to one-hot encoding
        # This allows us to select the corresponding μ and σ values
        # Shape: [batch_size, num_mix]
        one_hot = ops.one_hot(selected_components, num_classes=self.num_mix)

        # Expand dimensions for broadcasting with parameter tensors
        # Shape: [batch_size, num_mix, 1]
        one_hot_expanded = ops.expand_dims(one_hot, -1)

        # Select μ and σ parameters for chosen mixture components
        # Broadcasting: [batch, num_mix, output_dim] * [batch, num_mix, 1]
        # → Sum over mixture dimension to get selected parameters
        # Shape: [batch_size, output_dim]
        selected_mu = ops.sum(out_mu * one_hot_expanded, axis=1)
        selected_sigma = ops.sum(out_sigma * one_hot_expanded, axis=1)

        # Sample from the selected Gaussian distributions
        # Generate standard normal noise: ε ~ N(0, I)
        epsilon = keras.random.normal(ops.shape(selected_mu))

        # Transform to desired Gaussian: y = μ + σ * ε
        # This gives samples from N(μ, σ²) distributions
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

# ---------------------------------------------------------------------

def get_point_estimate(
    model: keras.Model,
    x_data: np.ndarray,
    mdn_layer: MDNLayer
) -> np.ndarray:
    """Calculate point estimates from MDN outputs as the weighted average of mixture components.

    This function computes the expected value of the mixture distribution by taking
    the weighted average of the component means. Mathematically:

    E[y|x] = Σᵢ πᵢ(x) * μᵢ(x)

    where πᵢ are the mixture weights and μᵢ are the component means.

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
    # Get model predictions (concatenated mixture parameters)
    y_pred = model.predict(x_data)

    # Split the mixture parameters into components
    mu, sigma, pi_logits = mdn_layer.split_mixture_params(y_pred)

    # Convert mixture weight logits to normalized probabilities
    # Ensures Σᵢ πᵢ = 1 for proper weighted average
    pi = keras.activations.softmax(pi_logits, axis=-1)

    # Convert tensors to numpy arrays for computation
    mu_np = ops.convert_to_numpy(mu)      # [batch, num_mix, output_dim]
    pi_np = ops.convert_to_numpy(pi)      # [batch, num_mix]

    # Expand π dimensions for broadcasting with μ
    # [batch, num_mix] → [batch, num_mix, 1]
    # This allows element-wise multiplication with μ across all output dimensions
    pi_expanded = np.expand_dims(pi_np, axis=-1)

    # Compute weighted means: πᵢ * μᵢ for each component
    # Shape: [batch, num_mix, output_dim]
    weighted_mu = mu_np * pi_expanded

    # Sum over mixture components to get expected value
    # E[y|x] = Σᵢ πᵢ * μᵢ
    # Shape: [batch, output_dim]
    point_estimates = np.sum(weighted_mu, axis=1)

    return point_estimates

# ---------------------------------------------------------------------

def get_uncertainty(
    model: keras.Model,
    x_data: np.ndarray,
    mdn_layer: MDNLayer,
    point_estimates: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate uncertainty estimates from MDN parameters.

    This function decomposes predictive uncertainty into aleatoric uncertainty
    (data noise) and epistemic uncertainty (model uncertainty) using the
    law of total variance:

    Var[y|x] = E[Var[y|x,θ]] + Var[E[y|x,θ]]

    where:
    - Aleatoric = E[Var[y|x,θ]] = Σᵢ πᵢ * σᵢ² (expected within-component variance)
    - Epistemic = Var[E[y|x,θ]] = Σᵢ πᵢ * (μᵢ - E[y])² (variance of component means)

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
    # Get model predictions (concatenated mixture parameters)
    y_pred = model.predict(x_data)

    # Extract mixture parameters
    mu, sigma, pi_logits = mdn_layer.split_mixture_params(y_pred)

    # Convert to numpy arrays for computation
    mu_np = ops.convert_to_numpy(mu)              # [batch, num_mix, output_dim]
    sigma_np = ops.convert_to_numpy(sigma)        # [batch, num_mix, output_dim]
    pi_np = ops.convert_to_numpy(keras.activations.softmax(pi_logits, axis=-1))  # [batch, num_mix]

    # Expand π dimensions for broadcasting
    # [batch, num_mix] → [batch, num_mix, 1]
    pi_expanded = np.expand_dims(pi_np, axis=-1)

    # Expand point estimates for broadcasting with mixture components
    # [batch, output_dim] → [batch, 1, output_dim]
    point_expanded = np.expand_dims(point_estimates, axis=1)

    # 1. ALEATORIC UNCERTAINTY (inherent data noise)
    # This represents the irreducible uncertainty due to noise in the data
    # Computed as weighted average of component variances: Σᵢ πᵢ * σᵢ²
    # Shape: [batch, output_dim]
    aleatoric_variance = np.sum(pi_expanded * sigma_np ** 2, axis=1)

    # 2. EPISTEMIC UNCERTAINTY (model uncertainty)
    # This represents uncertainty about which component should be active
    # Computed as weighted variance of component means: Σᵢ πᵢ * (μᵢ - E[y])²
    # Shape: [batch, num_mix, output_dim]
    squared_diff = (mu_np - point_expanded) ** 2
    # Shape: [batch, output_dim]
    epistemic_variance = np.sum(pi_expanded * squared_diff, axis=1)

    # 3. TOTAL PREDICTIVE VARIANCE
    # By law of total variance: Var[y] = E[Var[y|component]] + Var[E[y|component]]
    # Total variance = aleatoric + epistemic
    total_variance = aleatoric_variance + epistemic_variance

    return total_variance, aleatoric_variance

# ---------------------------------------------------------------------

def get_prediction_intervals(
    point_estimates: np.ndarray,
    total_variance: np.ndarray,
    confidence_level: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate prediction intervals for MDN outputs.

    Computes confidence intervals assuming the total predictive distribution
    is approximately Gaussian (via Central Limit Theorem for mixture components).

    Interval bounds: μ ± z_(α/2) * σ
    where z_(α/2) is the (1-α/2) quantile of the standard normal distribution.

    Parameters
    ----------
    point_estimates : np.ndarray
        Point estimates from the model, shape [batch_size, output_dim]
    total_variance : np.ndarray
        Total variance of predictions, shape [batch_size, output_dim]
    confidence_level : float, optional
        Desired confidence level, by default 0.95
        (e.g., 0.95 gives 95% confidence intervals)

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
    # For 95% confidence: α = 0.05, z = 1.96
    alpha = 1.0 - confidence_level
    z_score = stats.norm.ppf(1 - alpha / 2)

    # Calculate standard deviation from variance
    # σ = √(Var[y])
    std_dev = np.sqrt(total_variance)

    # Calculate confidence interval bounds
    # Lower bound: μ - z_(α/2) * σ
    lower_bound = point_estimates - z_score * std_dev
    # Upper bound: μ + z_(α/2) * σ
    upper_bound = point_estimates + z_score * std_dev

    return lower_bound, upper_bound

# ---------------------------------------------------------------------