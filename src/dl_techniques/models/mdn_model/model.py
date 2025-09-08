"""
Mixture Density Network (MDN) Model.

A Mixture Density Network is a neural network architecture that predicts probability
distributions rather than single point estimates, making it particularly valuable for
regression problems with multi-modal outputs, heteroscedastic noise, or when uncertainty
quantification is crucial.

## What It Does

This MDNModel combines traditional neural network feature extraction with a specialized
output layer that models the target variable as a mixture of Gaussian distributions.
Instead of predicting a single value ŷ, it predicts the parameters of a probability
distribution P(y|x), allowing you to:

- Generate multiple plausible predictions for each input
- Quantify prediction uncertainty (both aleatoric and epistemic)
- Handle multi-modal target distributions where multiple correct answers exist
- Capture heteroscedastic noise patterns in your data

## Core Functionality

**Probabilistic Predictions**: Rather than outputting deterministic values, the model
outputs mixture parameters (mixing coefficients π, means μ, and variances σ²) that
define a Gaussian mixture model for each input.

**Uncertainty Quantification**: Separates uncertainty into:
- Aleatoric uncertainty: Inherent data noise and ambiguity
- Epistemic uncertainty: Model uncertainty due to limited training data
- Total predictive uncertainty: Combined uncertainty estimate

**Sampling Capability**: Generate multiple samples from the predicted distribution
to explore the full range of possible outputs and their relative probabilities.

**Confidence Intervals**: Compute prediction intervals at any confidence level
to provide interpretable uncertainty bounds.

## When to Use

- **Multi-modal regression**: When inputs can map to multiple valid outputs
- **Inverse problems**: Where one input corresponds to multiple possible solutions
- **Noisy data**: When you need to model and account for heteroscedastic noise
- **Risk assessment**: When prediction confidence is as important as the prediction itself
- **Active learning**: To identify regions where the model is most uncertain
- **Robust decision-making**: When downstream decisions need uncertainty estimates

## Architecture

The model consists of:
1. **Feature Extraction Network**: Dense layers with configurable architecture,
   batch normalization, dropout, and activation functions
2. **MDN Output Layer**: Specialized layer that outputs mixture parameters
3. **Sampling Mechanism**: Methods to draw samples from the predicted distributions
4. **Uncertainty Analysis**: Tools to decompose and interpret prediction uncertainty

## Key Advantages

- **Captures output uncertainty**: Unlike standard regression, provides confidence estimates
- **Handles complex distributions**: Can model multi-modal and skewed target distributions
- **Flexible architecture**: Configurable hidden layers, regularization, and normalization
- **Production ready**: Full serialization support, logging, and robust error handling
- **Interpretable**: Separates different types of uncertainty for better decision-making

## Mathematical Foundation

For each input x, the model predicts:
P(y|x) = Σᵢ πᵢ(x) * N(y; μᵢ(x), σᵢ²(x))

Where:
- πᵢ(x): Mixing coefficient for component i (must sum to 1)
- μᵢ(x): Mean of Gaussian component i
- σᵢ²(x): Variance of Gaussian component i
- N(y; μ, σ²): Normal distribution with mean μ and variance σ²

This allows the model to represent arbitrarily complex probability distributions
as weighted combinations of simpler Gaussian components.
"""

import keras
from keras import ops
from keras import layers
from typing import List, Union, Optional, Dict, Any, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.statistics.mdn_layer import (
    MDNLayer,
    get_uncertainty,
    get_point_estimate,
    get_prediction_intervals
)
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class MDNModel(keras.Model):
    """A complete Mixture Density Network model.

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

    def __init__(
            self,
            hidden_layers: List[int],
            output_dimension: int,
            num_mixtures: int,
            hidden_activation: str = "relu",
            kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            use_batch_norm: bool = False,
            dropout_rate: Optional[float] = None,
            **kwargs: Any
    ) -> None:
        """Initialize the MDN model.

        Validates all input parameters and stores configuration for later use in build().
        Does not create the actual layers - this happens in build() when input shape is known.

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

        # Initialize layer containers - actual layers created in build()
        self.feature_layers = []  # Will contain: [Dense, BatchNorm?, Activation, Dropout?]*N
        self.mdn_layer = None     # The final MDN output layer
        self._build_input_shape = None  # For serialization

        logger.info(f"Initialized MDNModel with {len(hidden_layers)} hidden layers, "
                   f"{output_dimension}D output, {num_mixtures} mixtures")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the model with the given input shape.

        This method creates the complete architecture:
        1. Feature extraction layers (Dense + optional BatchNorm + Activation + optional Dropout)
        2. Final MDN layer that outputs mixture parameters

        The feature extraction network transforms raw inputs into a representation suitable
        for mixture parameter prediction. Each layer can optionally include:
        - Batch normalization: Normalizes activations for better training stability
        - Dropout: Randomly zeros activations for regularization
        - Configurable activation: Non-linearity (default ReLU)

        Args:
            input_shape: Shape tuple of the input tensor.
                Format: (batch_size, feature_dim) where batch_size can be None
        """
        # Store input shape for serialization support
        self._build_input_shape = input_shape

        logger.info(f"Building MDNModel with input shape: {input_shape}")

        # BUILD FEATURE EXTRACTION NETWORK
        # Each "hidden layer" actually consists of up to 4 sublayers arranged as:
        # Dense -> [BatchNorm] -> Activation -> [Dropout]
        # This ordering follows best practices for deep networks

        for i, units in enumerate(self.hidden_layers_sizes):
            # 1. DENSE LAYER: Linear transformation W*x + b
            # No activation here - applied after optional batch normalization
            dense_layer = layers.Dense(
                units,
                activation=None,  # Activation applied separately after BatchNorm
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f"dense_{i}"
            )
            self.feature_layers.append(dense_layer)

            # 2. BATCH NORMALIZATION (optional)
            # Normalizes layer inputs to have zero mean and unit variance
            # Helps with training stability and allows higher learning rates
            # Applied before activation for best performance
            if self.use_batch_norm:
                batch_norm_layer = layers.BatchNormalization(name=f"batch_norm_{i}")
                self.feature_layers.append(batch_norm_layer)

            # 3. ACTIVATION FUNCTION
            # Introduces non-linearity after the linear transformation
            # ReLU is default: f(x) = max(0, x)
            activation_layer = layers.Activation(
                self.hidden_activation,
                name=f"activation_{i}"
            )
            self.feature_layers.append(activation_layer)

            # 4. DROPOUT (optional)
            # Randomly sets fraction of inputs to 0 during training
            # Prevents overfitting by reducing co-adaptation between neurons
            # Only active during training, disabled during inference
            if self.dropout_rate is not None:
                dropout_layer = layers.Dropout(
                    self.dropout_rate,
                    name=f"dropout_{i}"
                )
                self.feature_layers.append(dropout_layer)

        # BUILD MDN OUTPUT LAYER
        # This layer takes the learned features and outputs mixture parameters:
        # - μ parameters: means for each mixture component and output dimension
        # - σ parameters: standard deviations (forced positive)
        # - π parameters: mixture weights (converted to probabilities via softmax)
        self.mdn_layer = MDNLayer(
            output_dimension=self.output_dim,
            num_mixtures=self.num_mix,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="mdn_layer"
        )

        # BUILD ALL LAYERS SEQUENTIALLY
        # Each layer needs to know the output shape of the previous layer
        # We simulate the forward pass to determine shapes
        current_shape = input_shape

        for layer in self.feature_layers:
            # Build each layer with the current shape
            layer.build(current_shape)

            # Update shape for next layer
            # Some layers (like Dropout) don't change shape, others do
            if hasattr(layer, 'compute_output_shape'):
                current_shape = layer.compute_output_shape(current_shape)
            # If layer doesn't have compute_output_shape, shape remains unchanged

        # Build the final MDN layer with the shape after all feature layers
        self.mdn_layer.build(current_shape)

        # Mark the model as built
        super().build(input_shape)
        logger.info("MDNModel built successfully")

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
            sample = self.mdn_layer.sample(predictions, temperature=temperature)
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