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
    """Mixture Density Network layer with separated processing paths.

    Outputs parameters for a mixture of Gaussian distributions with the predicted
    density ``p(y|x) = sum_i pi_i(x) N(y | mu_i(x), sigma_i(x))``. Each parameter
    head (means mu, standard deviations sigma, mixture weights pi) has its own
    intermediate processing path consisting of Dense, optional BatchNormalization,
    and activation layers before the final projection, enabling specialized
    representation learning per parameter type. Diversity regularization penalizes
    component collapse by adding ``exp(-||mu_i - mu_j||^2)`` terms.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────┐
        │  Input (batch, input_dim)│
        └────┬─────┬─────┬────────┘
             │     │     │
             ▼     ▼     ▼
        ┌────┐ ┌────┐ ┌────┐
        │ mu │ │ sig│ │ pi │   Intermediate Dense + BN + Act
        │path│ │path│ │path│
        └──┬─┘ └──┬─┘ └──┬─┘
           ▼      ▼      ▼
        ┌────┐ ┌────┐ ┌────┐
        │ mu │ │sig │ │ pi │   Final Dense projections
        │out │ │out │ │out │
        └──┬─┘ └──┬─┘ └──┬─┘
           └──┬───┘──┬───┘
              ▼
        ┌──────────────────────────┐
        │  Concatenate [mu,sig,pi] │
        │  (batch, total_params)   │
        └──────────────────────────┘

    :param output_dimension: Dimensionality of the target output space. Must be positive.
    :type output_dimension: int
    :param num_mixtures: Number of Gaussian mixture components. Must be positive.
    :type num_mixtures: int
    :param use_bias: Whether to use bias vectors in Dense layers. Defaults to ``True``.
    :type use_bias: bool
    :param diversity_regularizer_strength: Strength of diversity regularization.
        Defaults to 0.0.
    :type diversity_regularizer_strength: float
    :param intermediate_units: Units in intermediate dense layers. Defaults to 32.
    :type intermediate_units: int
    :param use_batch_norm: Whether to include BatchNormalization. Defaults to ``True``.
    :type use_batch_norm: bool
    :param intermediate_activation: Activation for intermediate layers. Defaults to ``"relu"``.
    :type intermediate_activation: str
    :param kernel_initializer: Initializer for kernel weights. Defaults to ``'glorot_normal'``.
    :type kernel_initializer: str | keras.initializers.Initializer
    :param bias_initializer: Initializer for bias vectors. Defaults to ``'zeros'``.
    :type bias_initializer: str | keras.initializers.Initializer
    :param kernel_regularizer: Regularizer for kernel weights. Defaults to ``L2(1e-5)``.
    :type kernel_regularizer: keras.regularizers.Regularizer | None
    :param bias_regularizer: Regularizer for bias vectors. Defaults to ``L2(1e-6)``.
    :type bias_regularizer: keras.regularizers.Regularizer | None
    :param min_sigma: Minimum standard deviation value. Defaults to 1e-3.
    :type min_sigma: float
    :param kwargs: Additional Layer base class arguments.
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

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: tuple[int | None, ...]
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
        """Forward pass through three parallel processing paths for mu, sigma, and pi.

        :param inputs: Input tensor of shape ``(batch_size, input_dim)``.
        :type inputs: keras.KerasTensor
        :param training: Boolean flag for training mode.
        :type training: bool | None
        :return: Concatenated mixture parameters ``[mu, sigma, pi]``.
        :rtype: keras.KerasTensor
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
        """Compute diversity loss to prevent component collapse.

        :param mu_output: Mean outputs of shape ``(batch_size, num_mix * output_dim)``.
        :type mu_output: keras.KerasTensor
        :return: Scalar diversity loss.
        :rtype: keras.KerasTensor
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
        """Compute the output shape of the layer.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: tuple[int | None, ...]
        :return: Output shape tuple.
        :rtype: tuple[int | None, ...]
        """
        output_size = (2 * self.output_dim * self.num_mix) + self.num_mix
        return tuple(list(input_shape)[:-1] + [output_size])

    def split_mixture_params(
            self,
            y_pred: keras.KerasTensor
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]:
        """Split the concatenated network output into parameter tensors.

        :param y_pred: Concatenated prediction tensor.
        :type y_pred: keras.KerasTensor
        :return: Tuple of ``(mu, sigma, pi)`` tensors.
        :rtype: tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]
        """
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
        """Compute MDN negative log-likelihood loss ``L = -log(sum_i pi_i N(y | mu_i, sigma_i))``.

        :param y_true: Ground truth targets.
        :type y_true: keras.KerasTensor
        :param y_pred: Concatenated prediction parameters from ``call()``.
        :type y_pred: keras.KerasTensor
        :return: Scalar loss value.
        :rtype: keras.KerasTensor
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
        """Sample from the predicted mixture distribution via ancestral sampling.

        :param y_pred: Concatenated prediction parameters.
        :type y_pred: keras.KerasTensor
        :param temperature: Sampling temperature controlling diversity. Defaults to 1.0.
        :type temperature: float
        :return: Sampled values of shape ``(batch_size, output_dim)``.
        :rtype: keras.KerasTensor
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
        """Return the layer configuration for serialization.

        :return: Configuration dictionary.
        :rtype: dict[str, Any]
        """
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
    """Calculate point estimates as ``E[y|x] = sum_i pi_i(x) mu_i(x)``.

    :param model: Trained model with an MDNLayer.
    :type model: keras.Model
    :param x_data: Input data for which to generate predictions.
    :type x_data: np.ndarray
    :param mdn_layer: The MDNLayer instance from the model.
    :type mdn_layer: MDNLayer
    :return: Point estimates with shape ``[batch_size, output_dim]``.
    :rtype: np.ndarray
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
    """Decompose predictive uncertainty via the law of total variance.

    :param model: Trained model with an MDNLayer.
    :type model: keras.Model
    :param x_data: Input data for prediction.
    :type x_data: np.ndarray
    :param mdn_layer: The MDNLayer instance from the model.
    :type mdn_layer: MDNLayer
    :param point_estimates: Point estimates (weighted average of means).
    :type point_estimates: np.ndarray
    :return: Tuple of ``(total_variance, aleatoric_variance)``.
    :rtype: tuple[np.ndarray, np.ndarray]
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
    """Calculate Gaussian prediction intervals ``mu +/- z * sigma``.

    :param point_estimates: Point estimates from the model.
    :type point_estimates: np.ndarray
    :param total_variance: Total predictive variance from the model.
    :type total_variance: np.ndarray
    :param confidence_level: Desired confidence level. Defaults to 0.95.
    :type confidence_level: float
    :return: Tuple of ``(lower_bound, upper_bound)`` arrays.
    :rtype: tuple[np.ndarray, np.ndarray]
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
    """Analyze mixture component diversity to diagnose collapse.

    :param model: Trained model with an MDNLayer.
    :type model: keras.Model
    :param x_data: Sample input data for analysis.
    :type x_data: np.ndarray
    :param mdn_layer: The MDNLayer instance from the model.
    :type mdn_layer: MDNLayer
    :return: Dictionary containing diversity metrics.
    :rtype: dict[str, Any]
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