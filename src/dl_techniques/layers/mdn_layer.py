"""
Enhanced Mixture Density Network (MDN) Layer with Intermediate Processing

This implementation extends the traditional MDN layer with practical improvements
for better training stability and performance:

1. Intermediate processing layers before each head (Dense -> BN -> Activation)
2. Diversity regularization to prevent component collapse
3. Sigma constraint at 0.01 minimum to prevent overconfident predictions
4. Configurable bias usage (default False for cleaner learning)
5. Improved numerical stability

Key Features:
    - Models complex, multi-modal target distributions
    - Outputs full probability distributions instead of point estimates
    - Provides uncertainty quantification through distribution parameters
    - Handles ambiguous or one-to-many mapping problems
    - Enables sampling from the predicted distributions
    - Prevents common training issues (component collapse, overconfidence)
    - Enhanced architecture with intermediate processing layers

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
from .activations.explanded_activations import elu_plus_one_plus_epsilon

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

EPSILON_SIGMA = 1e-6  # For internal numerical stability
MIN_SIGMA = 1e-3      # Minimum sigma value to prevent overconfidence

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class MDNLayer(keras.layers.Layer):
    """Enhanced Mixture Density Network Layer with Intermediate Processing

    This layer outputs parameters for a mixture of Gaussian distributions with
    practical improvements for training stability and enhanced architecture.
    Each head now includes intermediate processing layers for better representation
    learning.

    Architecture:
    Input -> [Dense(intermediate_units) -> BatchNorm -> Activation] -> Final Dense -> Output

    The layer creates six intermediate processing layers and three final output layers:

    Intermediate layers:
    1. intermediate_mu_dense: Dense layer before mean outputs
    2. intermediate_mu_bn: Optional batch normalization for mean path
    3. intermediate_sigma_dense: Dense layer before sigma outputs
    4. intermediate_sigma_bn: Optional batch normalization for sigma path
    5. intermediate_pi_dense: Dense layer before mixture weight outputs
    6. intermediate_pi_bn: Optional batch normalization for mixture weight path

    Final output layers:
    1. mdn_mus: Outputs means (μ) for each mixture component and output dimension
    2. mdn_sigmas: Outputs standard deviations (σ) with positivity constraint
    3. mdn_pi: Outputs mixture weights (π) as unnormalized logits

    Key Improvements:
    - Intermediate processing layers for better representation learning
    - Optional batch normalization for training stability
    - Configurable intermediate layer size
    - Diversity regularization to prevent component collapse
    - Sigma constraint at 0.001 minimum to prevent overconfidence
    - Configurable bias usage (default False)
    - Enhanced numerical stability

    Parameters
    ----------
    output_dimension : int
        Dimensionality of the output space. Must be positive.
    num_mixtures : int
        Number of Gaussian mixtures. Must be positive.
    use_bias : bool, optional
        Whether to use bias vectors in the Dense layers, by default False
    diversity_regularizer_strength : float, optional
        Strength of diversity regularization to prevent component collapse, by default 0.0
    intermediate_units : int, optional
        Number of units in intermediate dense layers, by default 32
    use_batch_norm : bool, optional
        Whether to use batch normalization in intermediate layers, by default True
    intermediate_activation : str, optional
        Activation function for intermediate layers, by default "relu"
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
        If diversity_regularizer_strength is negative
        If intermediate_units is not positive

    Examples
    --------
    >>> # Create an enhanced MDN layer with intermediate processing
    >>> mdn_layer = MDNLayer(
    ...     output_dimension=2,
    ...     num_mixtures=5,
    ...     use_bias=False,
    ...     diversity_regularizer_strength=0.01,
    ...     intermediate_units=64,
    ...     use_batch_norm=True,
    ...     intermediate_activation="swish"
    ... )
    >>>
    >>> # Build a model
    >>> model = keras.Sequential([
    ...     keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
    ...     keras.layers.Dense(64, activation='relu'),
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
        use_bias: bool = False,
        diversity_regularizer_strength: float = 0.0,
        intermediate_units: int = 32,
        use_batch_norm: bool = True,
        intermediate_activation: str = "gelu",
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_normal",
        bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = keras.regularizers.L2(1e-5),
        bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
        min_sigma: float = MIN_SIGMA,
        **kwargs: Any
    ) -> None:
        """Initialize the enhanced MDN layer with intermediate processing.

        Parameters
        ----------
        output_dimension : int
            Dimensionality of the output space
        num_mixtures : int
            Number of Gaussian mixtures
        use_bias : bool, optional
            Whether to use bias vectors in Dense layers, by default False
        diversity_regularizer_strength : float, optional
            Strength of diversity regularization, by default 0.0
        intermediate_units : int, optional
            Number of units in intermediate dense layers, by default 32
        use_batch_norm : bool, optional
            Whether to use batch normalization in intermediate layers, by default True
        intermediate_activation : str, optional
            Activation function for intermediate layers, by default "relu"
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
        if diversity_regularizer_strength < 0:
            raise ValueError(f"diversity_regularizer_strength must be non-negative, got {diversity_regularizer_strength}")
        if intermediate_units <= 0:
            raise ValueError(f"intermediate_units must be positive, got {intermediate_units}")

        # Store configuration parameters
        self.output_dim = output_dimension
        self.num_mix = num_mixtures
        self.use_bias = use_bias
        self.diversity_regularizer_strength = diversity_regularizer_strength
        self.intermediate_units = intermediate_units
        self.use_batch_norm = use_batch_norm
        self.intermediate_activation = intermediate_activation
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.min_sigma = min_sigma

        # Initialize sublayers to None - will be created in build()
        # Intermediate processing layers
        self.intermediate_mu_dense = None
        self.intermediate_mu_bn = None
        self.intermediate_sigma_dense = None
        self.intermediate_sigma_bn = None
        self.intermediate_pi_dense = None
        self.intermediate_pi_bn = None

        # Final output layers
        self.mdn_mus = None
        self.mdn_sigmas = None
        self.mdn_pi = None

        # Store build shape for serialization
        self._build_input_shape = None

        logger.info(f"Initialized enhanced MDN layer with {num_mixtures} mixtures and {output_dimension}D output")
        logger.info(f"  use_bias: {use_bias}")
        logger.info(f"  diversity_regularizer_strength: {diversity_regularizer_strength}")
        logger.info(f"  intermediate_units: {intermediate_units}")
        logger.info(f"  use_batch_norm: {use_batch_norm}")
        logger.info(f"  intermediate_activation: {intermediate_activation}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer weights and sublayers based on input shape.

        Creates intermediate processing layers and final output layers:

        For each head (mu, sigma, pi):
        1. Intermediate Dense layer with configurable units
        2. Optional Batch Normalization
        3. Activation function
        4. Final Dense layer with head-specific outputs

        Parameters
        ----------
        input_shape : Tuple[Optional[int], ...]
            Shape tuple of the input tensor
        """
        # Store input shape for serialization
        self._build_input_shape = input_shape

        # =================================================================
        # INTERMEDIATE PROCESSING LAYERS
        # =================================================================

        # Intermediate layer for MU (means) path
        self.intermediate_mu_dense = keras.layers.Dense(
            self.intermediate_units,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='intermediate_mu_dense'
        )

        # Optional batch normalization for MU path
        if self.use_batch_norm:
            self.intermediate_mu_bn = keras.layers.BatchNormalization(
                name='intermediate_mu_bn',
                center=self.use_bias,
            )

        # Intermediate layer for SIGMA (standard deviations) path
        self.intermediate_sigma_dense = keras.layers.Dense(
            self.intermediate_units,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='intermediate_sigma_dense'
        )

        # Optional batch normalization for SIGMA path
        if self.use_batch_norm:
            self.intermediate_sigma_bn = keras.layers.BatchNormalization(
                name='intermediate_sigma_bn',
                center=self.use_bias,
            )

        # Intermediate layer for PI (mixture weights) path
        self.intermediate_pi_dense = keras.layers.Dense(
            self.intermediate_units,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='intermediate_pi_dense'
        )

        # Optional batch normalization for PI path
        if self.use_batch_norm:
            self.intermediate_pi_bn = keras.layers.BatchNormalization(
                name='intermediate_pi_bn',
                center=self.use_bias,
            )

        # =================================================================
        # FINAL OUTPUT LAYERS
        # =================================================================

        # MEAN OUTPUTS: μ parameters
        # Creates outputs for means of each Gaussian component
        # Shape: [batch_size, num_mix * output_dim]
        # Will be reshaped to [batch_size, num_mix, output_dim] later
        self.mdn_mus = keras.layers.Dense(
            self.num_mix * self.output_dim,
            use_bias=self.use_bias,
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
            use_bias=self.use_bias,
            activation=lambda x: keras.activations.softplus(x) + MIN_SIGMA,
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
            use_bias=self.use_bias,
            activation=lambda x: keras.activations.softplus(x),
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='mdn_pi'
        )

        # =================================================================
        # BUILD ALL SUBLAYERS
        # =================================================================

        # Build intermediate processing layers
        self.intermediate_mu_dense.build(input_shape)
        if self.use_batch_norm:
            self.intermediate_mu_bn.build((input_shape[0], self.intermediate_units))

        self.intermediate_sigma_dense.build(input_shape)
        if self.use_batch_norm:
            self.intermediate_sigma_bn.build((input_shape[0], self.intermediate_units))

        self.intermediate_pi_dense.build(input_shape)
        if self.use_batch_norm:
            self.intermediate_pi_bn.build((input_shape[0], self.intermediate_units))

        # Build final output layers
        intermediate_shape = (input_shape[0], self.intermediate_units)
        self.mdn_mus.build(intermediate_shape)
        self.mdn_sigmas.build(intermediate_shape)
        self.mdn_pi.build(intermediate_shape)

        super().build(input_shape)
        logger.debug(f"Enhanced MDN layer built with input shape: {input_shape}")

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass of the layer with intermediate processing.

        Processes inputs through intermediate layers before computing mixture parameters:
        1. Each path: Dense -> [Optional BN] -> Activation
        2. Final output layers compute mixture parameters
        3. Concatenate all parameters into single output tensor

        The output structure is: [mu_params, sigma_params, pi_params]

        If diversity regularization is enabled, adds diversity loss during training.

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
        # =================================================================
        # PROCESS MU (MEANS) PATH
        # =================================================================

        # Intermediate processing: Dense -> [BN] -> Activation
        mu_intermediate = self.intermediate_mu_dense(inputs, training=training)
        if self.use_batch_norm:
            mu_intermediate = self.intermediate_mu_bn(mu_intermediate, training=training)
        mu_intermediate = keras.activations.get(self.intermediate_activation)(mu_intermediate)

        # Final mu output
        mu_output = self.mdn_mus(mu_intermediate, training=training)

        # =================================================================
        # PROCESS SIGMA (STANDARD DEVIATIONS) PATH
        # =================================================================

        # Intermediate processing: Dense -> [BN] -> Activation
        sigma_intermediate = self.intermediate_sigma_dense(inputs, training=training)
        if self.use_batch_norm:
            sigma_intermediate = self.intermediate_sigma_bn(sigma_intermediate, training=training)
        sigma_intermediate = keras.activations.get(self.intermediate_activation)(sigma_intermediate)

        # Final sigma output (automatically made positive and clamped)
        sigma_output = self.mdn_sigmas(sigma_intermediate, training=training)

        # =================================================================
        # PROCESS PI (MIXTURE WEIGHTS) PATH
        # =================================================================

        # Intermediate processing: Dense -> [BN] -> Activation
        pi_intermediate = self.intermediate_pi_dense(inputs, training=training)
        if self.use_batch_norm:
            pi_intermediate = self.intermediate_pi_bn(pi_intermediate, training=training)
        pi_intermediate = keras.activations.get(self.intermediate_activation)(pi_intermediate)

        # Final pi output (unnormalized logits)
        pi_output = self.mdn_pi(pi_intermediate, training=training)

        # =================================================================
        # DIVERSITY REGULARIZATION
        # =================================================================

        # Add diversity regularization loss if enabled and training
        if self.diversity_regularizer_strength > 0.0 and training:
            diversity_loss = self._compute_diversity_loss(mu_output, sigma_output, pi_output)
            self.add_loss(diversity_loss)
            logger.debug(f"Added diversity loss: {diversity_loss}")

        # =================================================================
        # CONCATENATE OUTPUT
        # =================================================================

        # Concatenate all parameters into single output tensor
        # Structure: [μ₁, μ₂, ..., μₙ, σ₁, σ₂, ..., σₙ, π₁, π₂, ..., πₘ]
        # where n = num_mix * output_dim, m = num_mix
        return keras.layers.concatenate(
            [mu_output, sigma_output, pi_output],
            name='mdn_outputs'
        )

    def _compute_diversity_loss(
        self,
        mu_output: keras.KerasTensor,
        sigma_output: keras.KerasTensor,
        pi_output: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Compute diversity loss to prevent mixture components from collapsing.

        Penalizes when mixture components have similar means, encouraging
        the components to spread out and capture different modes of the data.

        Parameters
        ----------
        mu_output : keras.KerasTensor
            Flattened means output, shape [batch_size, num_mix * output_dim]
        sigma_output : keras.KerasTensor
            Flattened sigmas output, shape [batch_size, num_mix * output_dim]
        pi_output : keras.KerasTensor
            Logits output, shape [batch_size, num_mix]

        Returns
        -------
        keras.KerasTensor
            Diversity loss value (scalar)
        """
        if self.num_mix <= 1:
            return ops.cast(0.0, dtype=mu_output.dtype)

        # Reshape means to [batch_size, num_mix, output_dim]
        batch_size = ops.shape(mu_output)[0]
        mus = ops.reshape(mu_output, [batch_size, self.num_mix, self.output_dim])

        # Calculate pairwise distances between mixture component means
        # Shape: [batch_size, num_mix, 1, output_dim]
        mus_expanded_1 = ops.expand_dims(mus, axis=2)
        # Shape: [batch_size, 1, num_mix, output_dim]
        mus_expanded_2 = ops.expand_dims(mus, axis=1)

        # Pairwise squared distances: [batch_size, num_mix, num_mix, output_dim]
        pairwise_distances = ops.square(mus_expanded_1 - mus_expanded_2)

        # Sum over output dimensions: [batch_size, num_mix, num_mix]
        pairwise_distances = ops.sum(pairwise_distances, axis=-1)

        # Create mask to ignore diagonal (distance from component to itself)
        mask = 1.0 - ops.eye(self.num_mix, dtype=pairwise_distances.dtype)

        # Apply mask and compute diversity loss
        # Penalize small distances (high similarity)
        diversity_loss = ops.exp(-pairwise_distances) * mask

        # Average over all pairs and batch
        diversity_loss = ops.mean(diversity_loss)

        return self.diversity_regularizer_strength * diversity_loss

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
        """Enhanced MDN loss function implementation using negative log likelihood.

        This function computes the negative log likelihood of the target values
        under the predicted mixture of Gaussians. The mathematical formulation:

        L = -log(Σᵢ πᵢ * N(y_true | μᵢ, σᵢ))

        where:
        - πᵢ are the mixture weights (after softmax normalization)
        - N(y_true | μᵢ, σᵢ) is the Gaussian probability density
        - The sum is over all mixture components

        Additional diversity regularization is handled automatically via add_loss().

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

        # Clamp with a tiny epsilon for numerical stability
        total_prob = ops.maximum(total_prob, keras.backend.epsilon())

        # Calculate negative log likelihood (NLL) loss
        log_prob = ops.log(total_prob)
        loss = -ops.mean(log_prob)

        return loss

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
        # The sigma constraint is already applied, but add extra protection
        out_sigma = ops.maximum(out_sigma, MIN_SIGMA)

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
        log_mix_weights = ops.log(mix_weights + MIN_SIGMA)
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
            "min_sigma": self.min_sigma,
            "output_dimension": self.output_dim,
            "num_mixtures": self.num_mix,
            "use_bias": self.use_bias,
            "diversity_regularizer_strength": self.diversity_regularizer_strength,
            "intermediate_units": self.intermediate_units,
            "use_batch_norm": self.use_batch_norm,
            "intermediate_activation": self.intermediate_activation,
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
# Utility Functions (Enhanced)
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


def check_component_diversity(
    model: keras.Model,
    x_data: np.ndarray,
    mdn_layer: MDNLayer
) -> Dict[str, Any]:
    """Check diversity of mixture components to monitor training quality.

    Parameters
    ----------
    model : keras.Model
        Trained model with MDN layer
    x_data : np.ndarray
        Input data for analysis
    mdn_layer : MDNLayer
        The MDN layer instance

    Returns
    -------
    Dict[str, Any]
        Dictionary containing diversity metrics
    """
    # Get predictions
    y_pred = model.predict(x_data)
    mu, sigma, pi_logits = mdn_layer.split_mixture_params(y_pred)

    # Convert to numpy
    mu_np = ops.convert_to_numpy(mu)
    sigma_np = ops.convert_to_numpy(sigma)
    pi_np = ops.convert_to_numpy(keras.activations.softmax(pi_logits, axis=-1))

    # Calculate pairwise distances between components
    batch_size, num_mix, output_dim = mu_np.shape
    component_distances = []

    for i in range(num_mix):
        for j in range(i + 1, num_mix):
            # Calculate L2 distance between components i and j
            distances = np.linalg.norm(mu_np[:, i, :] - mu_np[:, j, :], axis=-1)
            component_distances.append(distances)

    component_distances = np.array(component_distances)

    return {
        "mean_component_separation": np.mean(component_distances),
        "std_component_separation": np.std(component_distances),
        "min_component_separation": np.min(component_distances),
        "max_component_separation": np.max(component_distances),
        "mean_sigma_values": np.mean(sigma_np),
        "min_sigma_values": np.min(sigma_np),
        "max_sigma_values": np.max(sigma_np),
        "mean_mixture_weights": np.mean(pi_np, axis=0),
        "std_mixture_weights": np.std(pi_np, axis=0)
    }