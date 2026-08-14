"""
A Dense feature stack whose output head emits the parameters of a conditional
Gaussian mixture over the target, with ancestral sampling and a variance
decomposition built on top of it.

The failure this architecture repairs is specific. Regression under squared error is
maximum likelihood under a fixed-variance unimodal Gaussian, so the network can only
ever return `E[y|x]`. When the conditional distribution is multi-modal — inverse
problems, where several distinct outputs explain the same input — the conditional
mean falls between the modes and is a value the process never generates. Averaging
two correct answers produces a wrong one. Widening the network does not help, because
the target being fit is the mean itself.

A mixture density network replaces the point head with a density:

`p(y|x) = sum_i pi_i(x) * N(y; mu_i(x), diag sigma_i(x)^2)`

and trains under `L = -log sum_i pi_i(x) N(y; mu_i(x), sigma_i(x))`. The gradient now
rewards placing probability mass where the data is, not splitting the difference, so
components separate onto modes and `sigma_i` absorbs whatever local noise remains.
Heteroscedasticity comes free: `sigma_i` is a function of `x`, so the model can be
confident in one region and diffuse in another.

The parameterization is where the numerics live. `sigma` is produced by a softplus
plus a `min_sigma` floor (1e-3), which keeps it strictly positive and, more
importantly, bounds `log(sigma)` and `1/sigma` — an unfloored MDN drives `sigma`
toward zero on any component that lands exactly on a data point and the likelihood
diverges. `pi` is emitted as RAW LOGITS with no activation at all; every consumer
(loss, sampling, point estimate, uncertainty) applies exactly one softmax or
log_softmax to that slice, and adding an activation at the head would compress the
logits through a double application. The likelihood is evaluated entirely in log
space: `log_softmax` over the mixture axis, per-dimension Gaussian log-densities
summed over the last axis, then `logsumexp` over the mixture axis. The ordering is
load-bearing — the sum over output dimensions is the log of a product (components are
diagonal, so dimensions are independent given a component) and must happen before the
mixture is reduced, and doing either step in probability space underflows as
`output_dimension` grows.

Structurally the model is a stack of hidden blocks feeding one `MDNLayer`. Each
configured hidden size expands to `Dense -> [BatchNorm] -> Activation -> [Dropout]`,
with normalization before the activation and dropout last; the Dense carries no
activation of its own precisely so the normalization can sit between them. All
sublayers are created in `__init__` and explicitly built in `build()` rather than
lazily on first call: a sublayer that does not exist when a `.keras` file restores
weights is re-initialized afterwards, and the saved values are discarded without any
error. The head returns a single concatenated tensor of width
`2 * output_dim * num_mixtures + num_mixtures` laid out `[mu | sigma | pi]`, which is
why sampling and uncertainty are methods on this model rather than caller-side
post-processing — the split is the layer's contract, not a convention.

`compile()` deliberately does not accept a `loss` argument. It hard-wires
`mdn_layer.loss_func`, because the output tensor is a parameter vector and any
ordinary regression loss applied to it is meaningless arithmetic over concatenated
`mu`, `sigma` and logits. Making the loss unpassable removes the failure mode rather
than documenting it.

`sample()` draws ancestrally: a Gumbel-max categorical pick over `pi / temperature`
selects a component, then a Gaussian draw is taken from that component's parameters.
Temperature acts on the logits before the softmax, so below 1 it concentrates on
dominant components and above 1 it flattens toward uniform selection — it changes
which mode is visited, not the width of the component once chosen. When a `seed` is
given, sample `i` uses `seed + i` so the draws are reproducible yet uncorrelated, and
inside the layer the Gaussian draw is offset again so it does not alias the
categorical stream. Samples stack on axis 1, giving `[batch, num_samples, output_dim]`.

`predict_with_uncertainty` applies the law of total variance to the mixture:
`E[y|x] = sum_i pi_i mu_i`, with `sum_i pi_i sigma_i^2` as the within-component term
and `sum_i pi_i (mu_i - E[y|x])^2` as the between-component term. The keys name these
`aleatoric_variance` and `epistemic_variance`, and that second name should be read as
a label for the between-component spread, not as a claim about parameter uncertainty:
a single deterministically-trained MDN has one set of weights and cannot express
uncertainty about them. Genuine epistemic uncertainty requires an ensemble or a
posterior over weights. The returned intervals are likewise `point +/- z * sqrt(total
variance)`, a Gaussian approximation applied to a distribution chosen for being
non-Gaussian; they are calibrated in the unimodal case and merely indicative when the
mixture is genuinely multi-modal, where the honest interval is a set of disjoint
regions that a single lower/upper pair cannot represent.

References:
    - Bishop, 1994. Mixture Density Networks. Aston University Technical Report
      NCRG/94/004.
    - Graves, 2013. Generating Sequences With Recurrent Neural Networks.
      (https://arxiv.org/abs/1308.0850)
    - Ha and Schmidhuber, 2018. World Models. (https://arxiv.org/abs/1803.10122)
    - Kendall and Gal, 2017. What Uncertainties Do We Need in Bayesian Deep Learning
      for Computer Vision? (https://arxiv.org/abs/1703.04977)
"""

import keras
import numpy as np
from keras import ops
from keras import layers
from typing import List, Union, Optional, Dict, Any, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.statistics.mdn_layer import (
    MDNLayer,
    get_uncertainty,
    get_point_estimate,
    get_prediction_intervals
)

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class MDNModel(keras.Model):
    """A complete Mixture Density Network model.

    **Intent**: Wrap a configurable Dense feature-extraction stack and an
    ``MDNLayer`` output head into a single serializable ``keras.Model`` that
    predicts the parameters of a Gaussian-mixture distribution P(y|x) instead of
    a point estimate, enabling uncertainty quantification, multi-modal regression,
    and probabilistic sampling. All sublayers are created in ``__init__`` (every
    architectural parameter is construction-time known); ``build()`` only threads
    shapes through them so saved weights restore losslessly.

    This model combines a feature extraction network with an MDN layer and handles
    the appropriate loss function and sampling functionality. It enables the prediction
    of probability distributions instead of single point estimates, which is valuable
    for regression problems with multi-modal outputs or heteroscedastic noise.

    Architecture Overview:
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │   Input     │ -> │  Feature    │ -> │  Feature    │ -> │    MDN      │
    │ [batch, D]  │    │ Extraction  │    │ Extraction  │    │   Layer     │
    │             │    │  Layer 1    │    │  Layer N    │    │ [μ,σ,π]     │
    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                            │                    │                   │
                      ┌─────────────┐    ┌─────────────┐             │
                      │ BatchNorm   │    │ BatchNorm   │             │
                      │ (optional)  │    │ (optional)  │             │
                      └─────────────┘    └─────────────┘             │
                            │                    │                   │
                      ┌─────────────┐    ┌─────────────┐             │
                      │ Activation  │    │ Activation  │             │
                      │   (ReLU)    │    │   (ReLU)    │             │
                      └─────────────┘    └─────────────┘             │
                            │                    │                   │
                      ┌─────────────┐    ┌─────────────┐             │
                      │  Dropout    │    │  Dropout    │             │
                      │ (optional)  │    │ (optional)  │             │
                      └─────────────┘    └─────────────┘             │
                                                                     │
                                                                     v
                                                           ┌─────────────┐
                                                           │   Output    │
                                                           │Distribution │
                                                           │ Parameters  │
                                                           └─────────────┘

    Args:
        hidden_layers: List of hidden layer sizes for feature extraction.
            Each integer represents the number of units in that layer.
        output_dimension: Dimensionality of the output space.
            This is the number of target variables being predicted.
        num_mixtures: Number of Gaussian mixtures in the MDN layer.
            More mixtures allow modeling more complex distributions but increase parameters.
        hidden_activation: Activation function for hidden layers.
            Defaults to 'relu'.
        kernel_initializer: Initializer for the kernel weights matrix.
            Defaults to 'glorot_uniform'.
        kernel_regularizer: Regularizer function applied to the kernel weights matrix.
            Helps prevent overfitting. Defaults to None.
        use_batch_norm: Whether to use batch normalization between hidden layers.
            Can help with training stability and convergence. Defaults to False.
        dropout_rate: Dropout rate for regularization. Set to None for no dropout.
            Randomly sets input units to 0 during training to prevent overfitting.
            Defaults to None.
        **kwargs: Additional model arguments passed to the parent Model class.

    Example:
        >>> # Create a model for 2D output with 5 mixture components
        >>> model = MDNModel(
        ...     hidden_layers=[64, 32],          # Two hidden layers
        ...     output_dimension=2,              # 2D target space (e.g., x,y coordinates)
        ...     num_mixtures=5,                  # 5 Gaussian components
        ...     kernel_initializer='he_normal',  # Good for ReLU activations
        ...     kernel_regularizer=keras.regularizers.L2(1e-5)  # L2 regularization
        ... )
        >>> model.compile(optimizer='adam')      # Uses MDN loss automatically
        >>> model.fit(x_train, y_train, epochs=100)
        >>> samples = model.sample(x_test, num_samples=10)  # Generate 10 samples per input

    Note:
        The model automatically uses the MDN loss function when compiled.
        The sampling functionality allows for uncertainty quantification and
        probabilistic predictions.
    """

    # Class-level defaults (discoverability).
    DEFAULT_HIDDEN_ACTIVATION: str = "relu"
    DEFAULT_KERNEL_INITIALIZER: str = "glorot_uniform"
    DEFAULT_USE_BATCH_NORM: bool = False

    def __init__(
            self,
            hidden_layers: List[int],
            output_dimension: int,
            num_mixtures: int,
            hidden_activation: str = DEFAULT_HIDDEN_ACTIVATION,
            kernel_initializer: Union[str, keras.initializers.Initializer] = DEFAULT_KERNEL_INITIALIZER,
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            use_batch_norm: bool = DEFAULT_USE_BATCH_NORM,
            dropout_rate: Optional[float] = None,
            **kwargs: Any
    ) -> None:
        """Initialize the MDN model.

        Validates all input parameters and CREATES all sublayers (every
        architectural parameter is construction-time known). ``build()`` then only
        threads shapes through them. Creating layers here (not in ``build()``) is
        required so that on ``.keras`` weight-restore the sublayers already exist
        and saved weights land losslessly instead of being silently re-initialized.

        Raises:
            ValueError: If hidden_layers is empty or contains non-positive values.
            ValueError: If output_dimension or num_mixtures are not positive integers.
            ValueError: If dropout_rate is not in the range [0, 1).
        """
        super().__init__(**kwargs)

        # Validate architecture parameters
        if not hidden_layers or any(units <= 0 for units in hidden_layers):
            raise ValueError("hidden_layers must be a non-empty list of positive integers")

        if output_dimension <= 0:
            raise ValueError("output_dimension must be a positive integer")

        if num_mixtures <= 0:
            raise ValueError("num_mixtures must be a positive integer")

        # Validate regularization parameters
        if dropout_rate is not None and (dropout_rate < 0 or dropout_rate >= 1):
            raise ValueError("dropout_rate must be in the range [0, 1) or None")

        # Store configuration parameters for use in build()
        self.hidden_layers_sizes = hidden_layers
        self.output_dim = output_dimension
        self.num_mix = num_mixtures
        self.hidden_activation = hidden_activation
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate

        # Convert string initializers/regularizers to objects
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = kernel_regularizer

        self._build_input_shape = None  # For serialization

        # CREATE ALL SUBLAYERS (Golden Rule).
        # Every architectural parameter is construction-time known, so all
        # sublayers are instantiated here. build() only threads shapes through
        # them via explicit .build() calls.
        #
        # Each "hidden layer" expands to up to 4 sublayers:
        #   Dense -> [BatchNorm] -> Activation -> [Dropout]
        # (BatchNorm before activation; Dropout last — standard ordering.)
        self.feature_layers = []  # [Dense, BatchNorm?, Activation, Dropout?]*N
        for i, units in enumerate(self.hidden_layers_sizes):
            self.feature_layers.append(layers.Dense(
                units,
                activation=None,  # Activation applied separately after BatchNorm
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f"dense_{i}",
            ))
            if self.use_batch_norm:
                self.feature_layers.append(
                    layers.BatchNormalization(name=f"batch_norm_{i}"))
            self.feature_layers.append(layers.Activation(
                self.hidden_activation, name=f"activation_{i}"))
            if self.dropout_rate is not None:
                self.feature_layers.append(
                    layers.Dropout(self.dropout_rate, name=f"dropout_{i}"))

        # The final MDN output layer (μ, σ, π parameters).
        self.mdn_layer = MDNLayer(
            output_dimension=self.output_dim,
            num_mixtures=self.num_mix,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="mdn_layer",
        )

        logger.info(f"Initialized MDNModel with {len(hidden_layers)} hidden layers, "
                   f"{output_dimension}D output, {num_mixtures} mixtures")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the model with the given input shape.

        Sublayers are CREATED in ``__init__``; this method only threads shapes
        through them by calling each sublayer's ``.build()`` explicitly (each layer
        needs the output shape of the previous one). The explicit per-sublayer
        ``.build()`` chain is REQUIRED: a ``build()`` that defers sublayer building
        to first ``call`` leaves them unbuilt at ``.keras`` weight-restore time,
        which silently re-initializes the restored weights.

        Args:
            input_shape: Shape tuple of the input tensor.
                Format: (batch_size, feature_dim) where batch_size can be None
        """
        # Store input shape for serialization support
        self._build_input_shape = input_shape

        logger.info(f"Building MDNModel with input shape: {input_shape}")

        # BUILD ALL SUBLAYERS SEQUENTIALLY
        # Each layer needs to know the output shape of the previous layer; we
        # propagate shapes forward (Dropout/Activation pass shape through).
        current_shape = input_shape
        for layer in self.feature_layers:
            layer.build(current_shape)
            if hasattr(layer, 'compute_output_shape'):
                current_shape = layer.compute_output_shape(current_shape)

        # Build the final MDN layer with the shape after all feature layers.
        self.mdn_layer.build(current_shape)

        logger.info("MDNModel built successfully")

        # Mark the model as built (must be the LAST statement of build()).
        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Forward pass of the model.

        Implements the complete forward computation:
        1. Feature extraction through hidden layers
        2. Mixture parameter prediction via MDN layer

        The data flow is:
        input -> feature_layer_1 -> ... -> feature_layer_N -> mdn_layer -> output

        Each feature layer may include batch normalization and dropout, which behave
        differently during training vs inference:
        - BatchNorm: Uses batch statistics during training, moving averages during inference
        - Dropout: Active during training (randomly zeros units), disabled during inference

        Args:
            inputs: Input tensor with shape [batch_size, input_dim]
            training: Boolean indicating whether the model should behave in training mode.
                - True: Enables dropout, uses batch statistics for BatchNorm
                - False/None: Disables dropout, uses moving averages for BatchNorm
                Defaults to None (inference mode).

        Returns:
            Output tensor containing mixture parameters with shape:
            [batch_size, (2 * output_dim * num_mixtures) + num_mixtures]

            The output structure is: [μ₁, μ₂, ..., μₙ, σ₁, σ₂, ..., σₙ, π₁, π₂, ..., πₘ]
            where:
            - n = num_mixtures * output_dim (means and std devs for each component/dimension)
            - m = num_mixtures (mixture weights)
        """
        x = inputs

        # FEATURE EXTRACTION PHASE
        # Pass input through all feature extraction layers sequentially
        # Each layer transforms the representation to be more suitable for the task
        for layer in self.feature_layers:
            # Propagate training flag to each layer
            # This is crucial for layers like BatchNorm and Dropout
            x = layer(x, training=training)

        # MDN PARAMETER PREDICTION PHASE
        # Transform the learned features into mixture distribution parameters
        # Returns concatenated [μ, σ, π] parameters for all mixture components
        return self.mdn_layer(x, training=training)

    def sample(
            self,
            inputs: keras.KerasTensor,
            num_samples: int = 1,
            temperature: float = 1.0,
            seed: Optional[int] = None
    ) -> keras.KerasTensor:
        """Generate samples from the predicted distribution.

        Performs Monte Carlo sampling from the mixture distribution predicted by the model.
        This is useful for:
        - Uncertainty quantification: Multiple samples show prediction spread
        - Multi-modal exploration: Samples can come from different mixture components
        - Probabilistic decision making: Use sample statistics for robust decisions

        The sampling process:
        1. Forward pass to get mixture parameters [μ, σ, π]
        2. For each sample:
           a. Select mixture component using categorical distribution over π
           b. Sample from selected Gaussian N(μᵢ, σᵢ²)
        3. Stack all samples for return

        Args:
            inputs: Input tensor with shape [batch_size, input_dim]
            num_samples: Number of samples to generate for each input.
                More samples give better uncertainty estimates but increase computation.
                Defaults to 1.
            temperature: Temperature parameter for sampling (higher = more random).
                - temperature > 1: More uniform sampling across mixture components
                - temperature = 1: Uses predicted mixture weights exactly
                - temperature < 1: More concentrated sampling around dominant components
                Defaults to 1.0.
            seed: Optional seed for reproducible sampling. If provided, each sample
                uses seed + sample_index for deterministic results. Defaults to None.

        Returns:
            Samples from the predicted distribution with shape:
            [batch_size, num_samples, output_dim]

            Each sample[i, j, :] represents the j-th sample for the i-th input.

        Raises:
            ValueError: If num_samples is not positive or temperature is not positive.

        Example:
            >>> # Generate 100 samples for uncertainty quantification
            >>> samples = model.sample(x_test, num_samples=100)
            >>> # Compute sample statistics
            >>> sample_mean = ops.mean(samples, axis=1)      # [batch, output_dim]
            >>> sample_std = ops.std(samples, axis=1)        # [batch, output_dim]
            >>> # Use samples for robust decision making
            >>> confidence_intervals = ops.percentile(samples, [5, 95], axis=1)
        """
        # Input validation
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        # Get mixture parameters from forward pass
        # Use inference mode (training=False) for consistent predictions
        predictions = self(inputs, training=False)

        # Generate multiple independent samples
        # Each sample involves stochastic choices, so we need multiple draws
        samples = []
        for i in range(num_samples):
            # Use different seeds for each sample if a seed is provided
            # This ensures reproducible but uncorrelated samples
            sample_seed = None if seed is None else seed + i

            # Generate one sample from the mixture distribution
            # This involves: component selection + Gaussian sampling
            # DECISION plan_2026-06-09_be55db55/D-004: forward the per-sample
            # `sample_seed` (seed + i) into mdn_layer.sample. It was previously
            # computed and DISCARDED (latent bug), making `seed=` a no-op. Do NOT
            # drop this arg; MDNLayer.sample was extended to accept `seed`. See
            # decisions.md D-004.
            sample = self.mdn_layer.sample(
                predictions, temperature=temperature, seed=sample_seed)
            samples.append(sample)

        # Stack samples along a new dimension: [batch, samples, output_dim]
        # This makes it easy to compute statistics across samples
        return ops.stack(samples, axis=1)

    def predict_with_uncertainty(
            self,
            inputs: keras.KerasTensor,
            confidence_level: float = 0.95
    ) -> Dict[str, keras.KerasTensor]:
        """Generate predictions with comprehensive uncertainty estimates.

        This method provides a complete uncertainty analysis of the model's predictions,
        decomposing uncertainty into its fundamental components and providing
        interpretable confidence intervals.

        Uncertainty Decomposition:
        The total predictive uncertainty is decomposed using the law of total variance:

        Var[y|x] = E[Var[y|x,θ]] + Var[E[y|x,θ]]
                 = Aleatoric    + Epistemic

        Where:
        - Aleatoric uncertainty: Irreducible noise in the data (heteroscedastic noise)
        - Epistemic uncertainty: Model uncertainty due to limited training data

        Mathematical Details:
        - Point estimate: E[y|x] = Σᵢ πᵢ(x) * μᵢ(x)
        - Aleatoric variance: E[Var[y|x,θ]] = Σᵢ πᵢ(x) * σᵢ²(x)
        - Epistemic variance: Var[E[y|x,θ]] = Σᵢ πᵢ(x) * (μᵢ(x) - E[y|x])²

        Args:
            inputs: Input tensor with shape [batch_size, input_dim]
            confidence_level: Confidence level for prediction intervals (0-1).
                Common values: 0.95 (95%), 0.99 (99%), 0.68 (68% ≈ 1σ)
                Defaults to 0.95.

        Returns:
            Dictionary containing comprehensive uncertainty estimates:

            * **point_estimates**: Mean predictions [batch_size, output_dim]
                The expected value of the mixture distribution

            * **total_variance**: Total predictive variance [batch_size, output_dim]
                Combined aleatoric + epistemic uncertainty

            * **aleatoric_variance**: Data uncertainty component [batch_size, output_dim]
                Irreducible uncertainty due to noise in the data
                High values indicate inherently noisy/ambiguous regions

            * **epistemic_variance**: Model uncertainty component [batch_size, output_dim]
                Uncertainty due to limited training data
                High values indicate regions where more data would help

            * **lower_bound**: Lower prediction interval bounds [batch_size, output_dim]
                Lower bound of confidence interval assuming Gaussian approximation

            * **upper_bound**: Upper prediction interval bounds [batch_size, output_dim]
                Upper bound of confidence interval assuming Gaussian approximation

        Raises:
            ValueError: If confidence_level is not in the range (0, 1).

        Example:
            >>> # Get comprehensive uncertainty analysis
            >>> uncertainty = model.predict_with_uncertainty(x_test, confidence_level=0.95)
            >>>
            >>> # Extract components
            >>> predictions = uncertainty['point_estimates']
            >>> total_unc = uncertainty['total_variance']
            >>> data_noise = uncertainty['aleatoric_variance']
            >>> model_unc = uncertainty['epistemic_variance']
            >>>
            >>> # Identify high-uncertainty regions
            >>> high_epistemic = ops.where(model_unc > ops.percentile(model_unc, 90))
            >>> print(f"Regions needing more training data: {high_epistemic}")
            >>>
            >>> # Use confidence intervals for decision making
            >>> pred_width = uncertainty['upper_bound'] - uncertainty['lower_bound']
            >>> confident_predictions = predictions[pred_width < threshold]
        """
        # Input validation
        if not (0 < confidence_level < 1):
            raise ValueError("confidence_level must be in the range (0, 1)")

        # Get model predictions (mixture parameters)
        # Use the model's predict method for batch processing
        predictions = self.predict(inputs)

        # COMPUTE POINT ESTIMATES
        # Calculate the expected value of the mixture distribution
        # E[y|x] = Σᵢ πᵢ(x) * μᵢ(x)
        point_estimates = get_point_estimate(
            model=self,
            x_data=inputs,
            mdn_layer=self.mdn_layer
        )

        # DECOMPOSE UNCERTAINTY
        # Separate total uncertainty into aleatoric (data) and epistemic (model) components
        # This decomposition is crucial for understanding prediction reliability
        total_variance, aleatoric_variance = get_uncertainty(
            model=self,
            x_data=inputs,
            mdn_layer=self.mdn_layer,
            point_estimates=point_estimates
        )

        # Calculate epistemic variance (model uncertainty)
        # By law of total variance: Total = Aleatoric + Epistemic
        epistemic_variance = total_variance - aleatoric_variance

        # COMPUTE CONFIDENCE INTERVALS
        # Assume the mixture distribution is approximately Gaussian (CLT)
        # Use z-scores from normal distribution for interval bounds
        lower_bound, upper_bound = get_prediction_intervals(
            point_estimates=point_estimates,
            total_variance=total_variance,
            confidence_level=confidence_level
        )

        # Convert all numpy arrays back to Keras tensors for consistency
        # This ensures compatibility with the rest of the Keras ecosystem
        return {
            'point_estimates': ops.convert_to_tensor(point_estimates),
            'total_variance': ops.convert_to_tensor(total_variance),
            'aleatoric_variance': ops.convert_to_tensor(aleatoric_variance),
            'epistemic_variance': ops.convert_to_tensor(epistemic_variance),
            'lower_bound': ops.convert_to_tensor(lower_bound),
            'upper_bound': ops.convert_to_tensor(upper_bound)
        }

    def compile(
            self,
            optimizer: Union[str, keras.optimizers.Optimizer],
            metrics: Optional[List[Union[str, keras.metrics.Metric]]] = None,
            **kwargs: Any
    ) -> None:
        """Configure the model for training.

        Automatically sets up the MDN-specific loss function and configures the
        optimizer and metrics for training. The MDN loss function is the negative
        log-likelihood of the mixture distribution.

        Mathematical Background:
        The loss function maximizes the likelihood of the observed data under the
        predicted mixture distribution:

        L = -log(Σᵢ πᵢ(x) * N(y_true | μᵢ(x), σᵢ(x)))

        This loss automatically:
        - Encourages accurate mean predictions (μᵢ close to y_true)
        - Learns appropriate uncertainty levels (σᵢ matching data noise)
        - Balances mixture weights (πᵢ) based on local data density

        Args:
            optimizer: Optimizer instance or string name.
                Common choices:
                - 'adam': Adaptive learning rates, good default
                - 'rmsprop': Good for recurrent architectures
                - 'sgd': Simple but may need learning rate tuning
            metrics: List of metrics to track during training.
                Note: Standard regression metrics may not be directly applicable
                since the model outputs distribution parameters, not predictions.
                Consider custom metrics that evaluate the quality of the distributions.
                Defaults to None.
            **kwargs: Additional compile arguments (e.g., loss_weights, run_eagerly).

        Example:
            >>> # Basic compilation
            >>> model.compile(optimizer='adam')
            >>>
            >>> # Advanced compilation with custom optimizer
            >>> model.compile(
            ...     optimizer=keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
            ...     metrics=['mae']  # Track mean absolute error of point estimates
            ... )
        """
        # Use the MDN layer's loss function automatically
        # This loss function is specifically designed for mixture distributions
        super().compile(
            optimizer=optimizer,
            loss=self.mdn_layer.loss_func,  # Negative log-likelihood loss
            metrics=metrics,
            **kwargs
        )
        logger.info(f"MDNModel compiled with optimizer: {optimizer}")

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization.

        Serializes all constructor parameters needed to recreate the model.
        This enables saving and loading the model architecture.

        Returns:
            Dictionary containing the model configuration with all parameters
            needed to reconstruct the model via from_config().
        """
        config = {
            "hidden_layers": self.hidden_layers_sizes,
            "output_dimension": self.output_dim,
            "num_mixtures": self.num_mix,
            "hidden_activation": self.hidden_activation,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "use_batch_norm": self.use_batch_norm,
            "dropout_rate": self.dropout_rate
        }
        # Merge with base model configuration
        base_config = super().get_config()
        return {**base_config, **config}

    def get_build_config(self) -> Dict[str, Any]:
        """Get the build configuration for serialization.

        Stores information needed to rebuild the model layers after loading.
        This is separate from get_config() which stores constructor parameters.

        Returns:
            Dictionary containing the build configuration, specifically the
            input shape needed to reconstruct the layer architecture.
        """
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build the model from a build configuration.

        Reconstructs the model layers using the stored build configuration.
        This is called automatically when loading a saved model.

        Args:
            config: Dictionary containing the build configuration from get_build_config().
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MDNModel":
        """Create a model from its configuration.

        Reconstructs the model from a configuration dictionary created by get_config().
        This enables loading saved models with their exact architecture.

        Args:
            config: Dictionary with the model configuration from get_config().

        Returns:
            A new MDN model instance with the same architecture as the original.
        """
        config_copy = config.copy()

        # Deserialize complex objects that were serialized in get_config()
        config_copy["kernel_initializer"] = keras.initializers.deserialize(
            config["kernel_initializer"]
        )

        # Handle optional regularizer (may be None)
        if config["kernel_regularizer"] is not None:
            config_copy["kernel_regularizer"] = keras.regularizers.deserialize(
                config["kernel_regularizer"]
            )

        return cls(**config_copy)

    def save(self, filepath: str, **kwargs: Any) -> None:
        """Save the model to a file.

        Saves the complete model including architecture, weights, and training configuration
        in Keras format. The saved model can be loaded with keras.models.load_model().

        Args:
            filepath: Path where to save the model. If the path doesn't end with
                '.keras', the extension will be added automatically for consistency.
            **kwargs: Additional save arguments passed to the parent save method.
                Common options:
                - save_format: 'h5' or 'tf' (default is 'tf' for .keras files)
                - save_traces: Whether to save function traces (default True)

        Example:
            >>> # Save model
            >>> model.save('my_mdn_model')  # Automatically becomes 'my_mdn_model.keras'
            >>>
            >>> # Load model later
            >>> loaded_model = keras.models.load_model(
            ...     'my_mdn_model.keras',
            ...     custom_objects={'MDNModel': MDNModel, 'MDNLayer': MDNLayer}
            ... )
        """
        # Ensure consistent file extension for clarity
        if not filepath.endswith('.keras'):
            filepath += '.keras'

        logger.info(f"Saving MDNModel to: {filepath}")
        super().save(filepath, **kwargs)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the model.

        Calculates the shape of the output tensor based on the input shape.
        The output contains all mixture parameters concatenated together.

        Output Structure:
        The model outputs a concatenated tensor with:
        - μ parameters: num_mixtures * output_dim values (means)
        - σ parameters: num_mixtures * output_dim values (std deviations)
        - π parameters: num_mixtures values (mixture weights)

        Total size: (2 * num_mixtures * output_dim) + num_mixtures

        Args:
            input_shape: Shape of the input tensor.
                Format: (batch_size, input_features) where batch_size can be None

        Returns:
            Output shape tuple: (batch_size, total_mixture_params)
            where total_mixture_params = (2 * output_dim * num_mixtures) + num_mixtures

        Example:
            >>> # Model with output_dim=2, num_mixtures=3
            >>> input_shape = (None, 10)  # Batch size unknown, 10 input features
            >>> output_shape = model.compute_output_shape(input_shape)
            >>> print(output_shape)  # (None, 21)
            >>> # Breakdown: (2*2*3) + 3 = 12 + 9 = 21 total parameters
            >>> # 12 = μ and σ parameters for 3 mixtures × 2 dimensions
            >>> # 3 = π parameters for 3 mixture weights
        """
        # Convert input_shape to list for manipulation
        input_shape_list = list(input_shape)

        # Calculate total number of mixture parameters
        # Each mixture component needs: output_dim μ values + output_dim σ values + 1 π value
        # Total across all mixtures: num_mix * (output_dim + output_dim + 1/num_mix)
        # Simplified: (2 * output_dim * num_mix) + num_mix
        output_features = (2 * self.output_dim * self.num_mix) + self.num_mix

        # Return shape preserving batch dimension
        return tuple(input_shape_list[:-1] + [output_features])

# ---------------------------------------------------------------------

def create_mdn_model(
        hidden_layers: List[int],
        output_dimension: int,
        num_mixtures: int,
        input_dimension: int,
        hidden_activation: str = MDNModel.DEFAULT_HIDDEN_ACTIVATION,
        kernel_initializer: Union[str, keras.initializers.Initializer] = MDNModel.DEFAULT_KERNEL_INITIALIZER,
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        use_batch_norm: bool = MDNModel.DEFAULT_USE_BATCH_NORM,
        dropout_rate: Optional[float] = None,
        **kwargs: Any
) -> MDNModel:
    """Construct and build an :class:`MDNModel`.

    Convenience factory mirroring ``create_nbeats_model``/``create_deepar``: it
    instantiates the model and runs a tiny inference-mode dummy forward pass so
    the returned model is already BUILT (weights materialized), ready for
    ``.summary()``, ``.save()``, or weight transfer without a separate warmup.

    Args:
        hidden_layers: List of hidden layer sizes for feature extraction.
        output_dimension: Dimensionality of the target/output space.
        num_mixtures: Number of Gaussian mixture components.
        input_dimension: Feature dimension of the input, used only to build the
            model via the dummy forward pass.
        hidden_activation: Activation for hidden layers. Defaults to ``"relu"``.
        kernel_initializer: Kernel weight initializer. Defaults to
            ``"glorot_uniform"``.
        kernel_regularizer: Optional kernel regularizer. Defaults to ``None``.
        use_batch_norm: Whether to insert BatchNormalization. Defaults to ``False``.
        dropout_rate: Dropout rate in ``[0, 1)`` or ``None``. Defaults to ``None``.
        **kwargs: Forwarded to :class:`MDNModel` (e.g. ``name``).

    Returns:
        A built :class:`MDNModel` instance.

    Example:
        >>> model = create_mdn_model(
        ...     hidden_layers=[64, 32],
        ...     output_dimension=2,
        ...     num_mixtures=5,
        ...     input_dimension=10,
        ... )
        >>> model.compile(optimizer="adam")
    """
    model = MDNModel(
        hidden_layers=hidden_layers,
        output_dimension=output_dimension,
        num_mixtures=num_mixtures,
        hidden_activation=hidden_activation,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        use_batch_norm=use_batch_norm,
        dropout_rate=dropout_rate,
        **kwargs
    )

    # Build via a tiny inference-mode dummy forward pass.
    dummy = np.zeros((1, input_dimension), dtype="float32")
    model(dummy, training=False)
    return model

# ---------------------------------------------------------------------