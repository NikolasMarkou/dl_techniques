"""
Mixture Density Network (MDN) Layer with Intermediate Processing

This implementation extends the traditional MDN layer with practical improvements
for better training stability and performance:

1. Intermediate processing layers before each head (Dense -> BN -> Activation)
2. Diversity regularization to prevent component collapse
3. Sigma constraint at a configurable minimum to prevent overconfident predictions
4. Configurable bias usage
5. Improved numerical stability and modern Keras 3 compliance

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
# Constants
# ---------------------------------------------------------------------

MIN_SIGMA_DEFAULT = 1e-3      # Default minimum sigma value

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class MDNLayer(keras.layers.Layer):
    """Mixture Density Network Layer with separated processing paths.

    This layer outputs parameters for a mixture of Gaussian distributions,
    implementing several practical improvements for training stability. Each
    parameter head (means, sigmas, weights) has its own intermediate
    processing path for better representation learning. This architecture
    follows modern Keras 3 best practices for robust serialization.

    Key features:
    - Independent processing paths for μ, σ, and π for specialized learning.
    - Optional batch normalization in each path for training stability.
    - Diversity regularization to prevent component collapse.
    - Configurable minimum sigma to prevent overconfident (zero-variance) predictions.
    - Fully serializable and compliant with the Keras 3 API.

    Args:
        output_dimension: Integer, the dimensionality of the target output space.
            Must be positive.
        num_mixtures: Integer, the number of Gaussian mixture components to use.
            Must be positive.
        use_bias: Boolean, whether to use bias vectors in the Dense layers.
            Defaults to True.
        diversity_regularizer_strength: Float, strength of diversity regularization
            to prevent component collapse. A value > 0 adds a penalty for
            component means being too close. Defaults to 0.0.
        intermediate_units: Integer, number of units in the intermediate dense
            layers for each path. Must be positive. Defaults to 32.
        use_batch_norm: Boolean, whether to include BatchNormalization layers
            in each intermediate path. Defaults to True.
        intermediate_activation: String or callable, the activation function for
            the intermediate layers. Defaults to "relu".
        kernel_initializer: Initializer for kernel weights. Defaults to 'glorot_normal'.
        bias_initializer: Initializer for bias vectors. Defaults to 'zeros'.
        kernel_regularizer: Regularizer for kernel weights. Defaults to L2(1e-5).
        bias_regularizer: Regularizer for bias vectors. Defaults to L2(1e-6).
        min_sigma: Float, the minimum value for the standard deviation (σ) outputs.
            Helps prevent numerical instability and overconfidence. Defaults to 1e-3.
        **kwargs: Additional Layer base class arguments.

    Input shape:
        A 2D tensor with shape `(batch_size, input_dim)`.

    Output shape:
        A 2D tensor with shape `(batch_size, total_params)`, where
        `total_params = (2 * num_mixtures * output_dimension) + num_mixtures`.
        The output is a concatenation of the flattened means, sigmas, and mixture logits.

    Attributes:
        intermediate_mu_dense: Dense layer for the mean (μ) path.
        intermediate_sigma_dense: Dense layer for the sigma (σ) path.
        intermediate_pi_dense: Dense layer for the mixture weight (π) path.
        mdn_mus: Final Dense layer for producing mean (μ) parameters.
        mdn_sigmas: Final Dense layer for producing sigma (σ) parameters.
        mdn_pi: Final Dense layer for producing mixture weight (π) logits.

    Example:
        ```python
        # Create an MDN layer
        mdn_layer = MDNLayer(
            output_dimension=2,
            num_mixtures=5,
            intermediate_units=64,
            diversity_regularizer_strength=0.01
        )

        # Build a model
        inputs = keras.Input(shape=(128,))
        x = keras.layers.Dense(256, activation='relu')(inputs)
        mdn_params = mdn_layer(x)
        model = keras.Model(inputs, mdn_params)

        # Compile with the MDN loss function
        model.compile(optimizer='adam', loss=mdn_layer.loss_func)
        ```

    Raises:
        ValueError: If `output_dimension`, `num_mixtures`, or `intermediate_units`
            are not positive, or if `diversity_regularizer_strength` is negative.
    """

    def __init__(
        self,
        output_dimension: int,
        num_mixtures: int,
        use_bias: bool = True,
        diversity_regularizer_strength: float = 0.0,
        intermediate_units: int = 32,
        use_batch_norm: bool = True,
        intermediate_activation: str = "relu",
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_normal",
        bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = keras.regularizers.L2(1e-5),
        bias_regularizer: Optional[keras.regularizers.Regularizer] = keras.regularizers.L2(1e-6),
        min_sigma: float = MIN_SIGMA_DEFAULT,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # === Parameter Validation ===
        if output_dimension <= 0:
            raise ValueError(f"output_dimension must be positive, got {output_dimension}")
        if num_mixtures <= 0:
            raise ValueError(f"num_mixtures must be positive, got {num_mixtures}")
        if diversity_regularizer_strength < 0:
            raise ValueError(f"diversity_regularizer_strength must be non-negative, got {diversity_regularizer_strength}")
        if intermediate_units <= 0:
            raise ValueError(f"intermediate_units must be positive, got {intermediate_units}")

        # === Store Configuration ===
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

        # === CREATE all sub-layers (unbuilt) ===
        # This follows the "Create vs. Build" golden rule.

        # --- Intermediate processing layers ---
        self.intermediate_mu_dense = keras.layers.Dense(
            self.intermediate_units, use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer,
            name='intermediate_mu_dense'
        )
        self.intermediate_sigma_dense = keras.layers.Dense(
            self.intermediate_units, use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer,
            name='intermediate_sigma_dense'
        )
        self.intermediate_pi_dense = keras.layers.Dense(
            self.intermediate_units, use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer,
            name='intermediate_pi_dense'
        )

        if self.use_batch_norm:
            self.intermediate_mu_bn = keras.layers.BatchNormalization(name='intermediate_mu_bn', center=self.use_bias)
            self.intermediate_sigma_bn = keras.layers.BatchNormalization(name='intermediate_sigma_bn', center=self.use_bias)
            self.intermediate_pi_bn = keras.layers.BatchNormalization(name='intermediate_pi_bn', center=self.use_bias)
        else:
            self.intermediate_mu_bn = None
            self.intermediate_sigma_bn = None
            self.intermediate_pi_bn = None

        # --- Final output layers ---
        self.mdn_mus = keras.layers.Dense(
            self.num_mix * self.output_dim, use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer,
            name='mdn_mus'
        )
        self.mdn_sigmas = keras.layers.Dense(
            self.num_mix * self.output_dim, use_bias=self.use_bias,
            activation=lambda x: keras.activations.softplus(x) + self.min_sigma,
            kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer,
            name='mdn_sigmas'
        )
        self.mdn_pi = keras.layers.Dense(
            self.num_mix, use_bias=self.use_bias,
            activation=lambda x: keras.activations.softplus(x),
            kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer,
            name='mdn_pi'
        )

        logger.info(f"Initialized MDN layer with {num_mixtures} mixtures and {output_dimension}D output")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer's weights and sub-layers.

        This method explicitly builds each sub-layer created in `__init__`,
        which is critical for robust serialization and weight restoration.
        """
        # Build intermediate dense layers, which take the primary input
        self.intermediate_mu_dense.build(input_shape)
        self.intermediate_sigma_dense.build(input_shape)
        self.intermediate_pi_dense.build(input_shape)

        # The subsequent layers operate on the intermediate dimension
        intermediate_shape = (input_shape[0], self.intermediate_units)

        # Build batch norm layers if they exist
        if self.use_batch_norm:
            self.intermediate_mu_bn.build(intermediate_shape)
            self.intermediate_sigma_bn.build(intermediate_shape)
            self.intermediate_pi_bn.build(intermediate_shape)

        # Build the final output layers
        self.mdn_mus.build(intermediate_shape)
        self.mdn_sigmas.build(intermediate_shape)
        self.mdn_pi.build(intermediate_shape)

        # Always call the parent's build() method at the end
        super().build(input_shape)
        logger.debug(f"MDN layer built with input shape: {input_shape}")

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass of the layer with separate processing paths.

        The input is processed through three parallel paths to compute the
        mixture parameters (μ, σ, π), which are then concatenated into
        a single output tensor.
        """
        # === Process MU (Means) Path ===
        mu_intermediate = self.intermediate_mu_dense(inputs, training=training)
        if self.use_batch_norm:
            mu_intermediate = self.intermediate_mu_bn(mu_intermediate, training=training)
        mu_intermediate = keras.activations.get(self.intermediate_activation)(mu_intermediate)
        mu_output = self.mdn_mus(mu_intermediate, training=training)

        # === Process SIGMA (Standard Deviations) Path ===
        sigma_intermediate = self.intermediate_sigma_dense(inputs, training=training)
        if self.use_batch_norm:
            sigma_intermediate = self.intermediate_sigma_bn(sigma_intermediate, training=training)
        sigma_intermediate = keras.activations.get(self.intermediate_activation)(sigma_intermediate)
        sigma_output = self.mdn_sigmas(sigma_intermediate, training=training)

        # === Process PI (Mixture Weights) Path ===
        pi_intermediate = self.intermediate_pi_dense(inputs, training=training)
        if self.use_batch_norm:
            pi_intermediate = self.intermediate_pi_bn(pi_intermediate, training=training)
        pi_intermediate = keras.activations.get(self.intermediate_activation)(pi_intermediate)
        pi_output = self.mdn_pi(pi_intermediate, training=training)

        # === Diversity Regularization ===
        if self.diversity_regularizer_strength > 0.0 and training:
            diversity_loss = self._compute_diversity_loss(mu_output)
            self.add_loss(diversity_loss)

        # === Concatenate Output ===
        return keras.layers.concatenate(
            [mu_output, sigma_output, pi_output],
            name='mdn_outputs'
        )

    def _compute_diversity_loss(
        self,
        mu_output: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """Computes diversity loss to prevent component collapse.

        This loss penalizes mixture components for having similar means,
        encouraging them to capture different modes of the data distribution.
        """
        if self.num_mix <= 1:
            return ops.cast(0.0, dtype=mu_output.dtype)

        batch_size = ops.shape(mu_output)[0]
        mus = ops.reshape(mu_output, [batch_size, self.num_mix, self.output_dim])
        mus_expanded_1 = ops.expand_dims(mus, axis=2)
        mus_expanded_2 = ops.expand_dims(mus, axis=1)
        pairwise_distances = ops.sum(ops.square(mus_expanded_1 - mus_expanded_2), axis=-1)
        mask = 1.0 - ops.eye(self.num_mix, dtype=pairwise_distances.dtype)
        diversity_loss = ops.mean(ops.exp(-pairwise_distances) * mask)

        return self.diversity_regularizer_strength * diversity_loss

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Computes the output shape of the layer."""
        output_size = (2 * self.output_dim * self.num_mix) + self.num_mix
        return tuple(list(input_shape)[:-1] + [output_size])

    def split_mixture_params(
            self,
            y_pred: keras.KerasTensor
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]:
        """Splits the concatenated network output into parameter tensors."""
        mu_end = self.num_mix * self.output_dim
        sigma_end = mu_end + (self.num_mix * self.output_dim)

        out_mu = y_pred[..., :mu_end]
        out_sigma = y_pred[..., mu_end:sigma_end]
        out_pi = y_pred[..., sigma_end:]

        batch_size = ops.shape(y_pred)[0]
        out_mu = ops.reshape(out_mu, [batch_size, self.num_mix, self.output_dim])
        out_sigma = ops.reshape(out_sigma, [batch_size, self.num_mix, self.output_dim])

        return out_mu, out_sigma, out_pi

    def loss_func(
            self,
            y_true: keras.KerasTensor,
            y_pred: keras.KerasTensor
    ) -> keras.KerasTensor:
        """MDN loss function using negative log-likelihood.

        Computes the loss L = -log(Σᵢ πᵢ * N(y_true | μᵢ, σᵢ)), which represents
        the negative log-likelihood of the true data under the predicted
        mixture distribution.
        """
        y_true = ops.reshape(y_true, [-1, self.output_dim])
        out_mu, out_sigma, out_pi = self.split_mixture_params(y_pred)

        mix_weights = keras.activations.softmax(out_pi, axis=-1)
        y_true_expanded = ops.expand_dims(y_true, 1)

        component_probs = gaussian_probability(
            y_true_expanded, out_mu, out_sigma
        )
        component_probs = ops.prod(component_probs, axis=-1)
        weighted_probs = mix_weights * component_probs
        total_prob = ops.sum(weighted_probs, axis=-1)
        total_prob = ops.maximum(total_prob, keras.backend.epsilon())
        loss = -ops.mean(ops.log(total_prob))

        return loss

    def sample(self, y_pred: keras.KerasTensor, temperature: float = 1.0) -> keras.KerasTensor:
        """Samples from the predicted mixture distribution.

        Performs ancestral sampling: first, a mixture component is chosen based
        on the π weights, and then a sample is drawn from the selected Gaussian.
        """
        out_mu, out_sigma, out_pi = self.split_mixture_params(y_pred)
        out_sigma = ops.maximum(out_sigma, self.min_sigma)

        if temperature != 1.0:
            out_pi = out_pi / temperature

        mix_weights = keras.activations.softmax(out_pi, axis=-1)
        gumbel_noise = -ops.log(-ops.log(keras.random.uniform(ops.shape(out_pi))))
        selected_logits = ops.log(mix_weights + keras.backend.epsilon()) + gumbel_noise
        selected_components = ops.argmax(selected_logits, axis=-1)

        one_hot = ops.one_hot(selected_components, num_classes=self.num_mix)
        one_hot_expanded = ops.expand_dims(one_hot, -1)

        selected_mu = ops.sum(out_mu * one_hot_expanded, axis=1)
        selected_sigma = ops.sum(out_sigma * one_hot_expanded, axis=1)

        epsilon = keras.random.normal(ops.shape(selected_mu))
        samples = selected_mu + selected_sigma * epsilon

        return samples

    def get_config(self) -> Dict[str, Any]:
        """Returns the layer configuration for serialization."""
        config = super().get_config()
        config.update({
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
            "min_sigma": self.min_sigma,
        })
        return config


# ---------------------------------------------------------------------
# Utility Functions for MDN Analysis
# ---------------------------------------------------------------------


def get_point_estimate(
    model: keras.Model,
    x_data: np.ndarray,
    mdn_layer: MDNLayer
) -> np.ndarray:
    """Calculate point estimates from MDN outputs as the weighted average of means.

    Computes the expected value E[y|x] = Σᵢ πᵢ(x) * μᵢ(x).

    Parameters
    ----------
    model : keras.Model
        Trained model with an MDNLayer.
    x_data : np.ndarray
        Input data for which to generate predictions.
    mdn_layer : MDNLayer
        The MDNLayer instance from the model.

    Returns
    -------
    np.ndarray
        Point estimates with shape [batch_size, output_dim].
    """
    y_pred = model.predict(x_data)
    mu, _, pi_logits = mdn_layer.split_mixture_params(y_pred)
    pi = keras.activations.softmax(pi_logits, axis=-1)

    mu_np = ops.convert_to_numpy(mu)
    pi_np = ops.convert_to_numpy(pi)

    pi_expanded = np.expand_dims(pi_np, axis=-1)
    weighted_mu = mu_np * pi_expanded
    point_estimates = np.sum(weighted_mu, axis=1)

    return point_estimates


def get_uncertainty(
    model: keras.Model,
    x_data: np.ndarray,
    mdn_layer: MDNLayer,
    point_estimates: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate and decompose predictive uncertainty from an MDN.

    Uses the law of total variance to split uncertainty into:
    - Aleatoric (data noise): E[Var[y|x,θ]] = Σᵢ πᵢ * σᵢ²
    - Epistemic (model uncertainty): Var[E[y|x,θ]] = Σᵢ πᵢ * (μᵢ - E[y])²

    Parameters
    ----------
    model : keras.Model
        Trained model with an MDNLayer.
    x_data : np.ndarray
        Input data for prediction.
    mdn_layer : MDNLayer
        The MDNLayer instance from the model.
    point_estimates : np.ndarray
        Point estimates (weighted average of means).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing (total_variance, aleatoric_variance), each with
        shape [batch_size, output_dim]. Epistemic variance can be computed as
        `total_variance - aleatoric_variance`.
    """
    y_pred = model.predict(x_data)
    mu, sigma, pi_logits = mdn_layer.split_mixture_params(y_pred)

    mu_np = ops.convert_to_numpy(mu)
    sigma_np = ops.convert_to_numpy(sigma)
    pi_np = ops.convert_to_numpy(keras.activations.softmax(pi_logits, axis=-1))

    pi_expanded = np.expand_dims(pi_np, axis=-1)
    point_expanded = np.expand_dims(point_estimates, axis=1)

    # Aleatoric uncertainty (weighted average of component variances)
    aleatoric_variance = np.sum(pi_expanded * sigma_np ** 2, axis=1)

    # Epistemic uncertainty (weighted variance of component means)
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
    """Calculate prediction intervals from MDN outputs.

    Assumes the total predictive distribution is approximately Gaussian. The
    interval is computed as `μ ± z * σ`.

    Note:
        This function requires `scipy` to be installed.

    Parameters
    ----------
    point_estimates : np.ndarray
        Point estimates from the model.
    total_variance : np.ndarray
        Total predictive variance from the model.
    confidence_level : float, optional
        The desired confidence level (e.g., 0.95 for 95%). Defaults to 0.95.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing (lower_bound, upper_bound) of the intervals.
    """
    try:
        from scipy import stats
    except ImportError:
        raise ImportError("`scipy` is required to calculate prediction intervals. Please install it with `pip install scipy`.")

    alpha = 1.0 - confidence_level
    z_score = stats.norm.ppf(1 - alpha / 2)
    std_dev = np.sqrt(total_variance)

    lower_bound = point_estimates - z_score * std_dev
    upper_bound = point_estimates + z_score * std_dev

    return lower_bound, upper_bound


def check_component_diversity(
    model: keras.Model,
    x_data: np.ndarray,
    mdn_layer: MDNLayer
) -> Dict[str, Any]:
    """Analyzes the diversity of mixture components for trained MDN.

    This utility helps diagnose issues like component collapse by reporting
    metrics on component separation, variance, and weight distribution.

    Parameters
    ----------
    model : keras.Model
        Trained model with an MDNLayer.
    x_data : np.ndarray
        A sample of input data for analysis.
    mdn_layer : MDNLayer
        The MDNLayer instance from the model.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing various diversity metrics.
    """
    y_pred = model.predict(x_data)
    mu, sigma, pi_logits = mdn_layer.split_mixture_params(y_pred)

    mu_np = ops.convert_to_numpy(mu)
    sigma_np = ops.convert_to_numpy(sigma)
    pi_np = ops.convert_to_numpy(keras.activations.softmax(pi_logits, axis=-1))

    num_mix = mdn_layer.num_mix
    component_distances = []
    if num_mix > 1:
        for i in range(num_mix):
            for j in range(i + 1, num_mix):
                distances = np.linalg.norm(mu_np[:, i, :] - mu_np[:, j, :], axis=-1)
                component_distances.append(distances)
        component_distances = np.array(component_distances)
    else:
        component_distances = np.array([0.0])


    return {
        "mean_component_separation": np.mean(component_distances),
        "std_component_separation": np.std(component_distances),
        "mean_sigma_values": np.mean(sigma_np),
        "mean_mixture_weights": np.mean(pi_np, axis=0),
        "std_mixture_weights": np.std(pi_np, axis=0)
    }