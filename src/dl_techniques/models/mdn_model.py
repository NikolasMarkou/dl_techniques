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
- **Interpretable**: Separates different types of uncertainty for better decision making

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

from dl_techniques.layers.mdn_layer import (
    MDNLayer,
    get_point_estimate,
    get_uncertainty,
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

    Args:
        hidden_layers: List of hidden layer sizes for feature extraction.
        output_dimension: Dimensionality of the output space.
        num_mixtures: Number of Gaussian mixtures in the MDN layer.
        hidden_activation: Activation function for hidden layers.
            Defaults to 'relu'.
        kernel_initializer: Initializer for the kernel weights matrix.
            Defaults to 'glorot_uniform'.
        kernel_regularizer: Regularizer function applied to the kernel weights matrix.
            Defaults to None.
        use_batch_norm: Whether to use batch normalization between hidden layers.
            Defaults to False.
        dropout_rate: Dropout rate for regularization. Set to None for no dropout.
            Defaults to None.
        **kwargs: Additional model arguments passed to the parent Model class.

    Example:
        >>> model = MDNModel(
        ...     hidden_layers=[64, 32],
        ...     output_dimension=2,
        ...     num_mixtures=5,
        ...     kernel_initializer='he_normal',
        ...     kernel_regularizer=keras.regularizers.L2(1e-5)
        ... )
        >>> model.compile(optimizer='adam')
        >>> model.fit(x_train, y_train, epochs=100)
        >>> samples = model.sample(x_test, num_samples=10)

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

        Raises:
            ValueError: If hidden_layers is empty or contains non-positive values.
            ValueError: If output_dimension or num_mixtures are not positive integers.
            ValueError: If dropout_rate is not in the range [0, 1).
        """
        super().__init__(**kwargs)

        # Validate inputs
        if not hidden_layers or any(units <= 0 for units in hidden_layers):
            raise ValueError("hidden_layers must be a non-empty list of positive integers")

        if output_dimension <= 0:
            raise ValueError("output_dimension must be a positive integer")

        if num_mixtures <= 0:
            raise ValueError("num_mixtures must be a positive integer")

        if dropout_rate is not None and (dropout_rate < 0 or dropout_rate >= 1):
            raise ValueError("dropout_rate must be in the range [0, 1) or None")

        # Store parameters
        self.hidden_layers_sizes = hidden_layers
        self.output_dim = output_dimension
        self.num_mix = num_mixtures
        self.hidden_activation = hidden_activation
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate

        # Get initializer and regularizer
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = kernel_regularizer

        # Initialize feature layers and MDN layer as None, will build in build()
        self.feature_layers = []
        self.mdn_layer = None
        self._build_input_shape = None

        logger.info(f"Initialized MDNModel with {len(hidden_layers)} hidden layers, "
                   f"{output_dimension}D output, {num_mixtures} mixtures")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the model with the given input shape.

        This method creates the feature extraction layers and the MDN layer
        when the model is first called.

        Args:
            input_shape: Shape tuple of the input tensor.
        """
        # Store for serialization
        self._build_input_shape = input_shape

        logger.info(f"Building MDNModel with input shape: {input_shape}")

        # Build feature extraction layers
        for i, units in enumerate(self.hidden_layers_sizes):
            # Add Dense layer
            dense_layer = layers.Dense(
                units,
                activation=None,  # Activation applied after batch norm if used
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f"dense_{i}"
            )
            self.feature_layers.append(dense_layer)

            # Add batch normalization if requested
            if self.use_batch_norm:
                batch_norm_layer = layers.BatchNormalization(name=f"batch_norm_{i}")
                self.feature_layers.append(batch_norm_layer)

            # Add activation
            activation_layer = layers.Activation(
                self.hidden_activation,
                name=f"activation_{i}"
            )
            self.feature_layers.append(activation_layer)

            # Add dropout if specified
            if self.dropout_rate is not None:
                dropout_layer = layers.Dropout(
                    self.dropout_rate,
                    name=f"dropout_{i}"
                )
                self.feature_layers.append(dropout_layer)

        # Create MDN layer
        self.mdn_layer = MDNLayer(
            output_dimension=self.output_dim,
            num_mixtures=self.num_mix,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="mdn_layer"
        )

        # Build feature layers sequentially
        current_shape = input_shape
        for layer in self.feature_layers:
            layer.build(current_shape)
            # Update shape for next layer
            if hasattr(layer, 'compute_output_shape'):
                current_shape = layer.compute_output_shape(current_shape)

        # Build MDN layer
        self.mdn_layer.build(current_shape)

        super().build(input_shape)
        logger.info("MDNModel built successfully")

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Forward pass of the model.

        Args:
            inputs: Input tensor.
            training: Boolean indicating whether the model should behave in training mode.
                Defaults to None.

        Returns:
            Output tensor containing mixture parameters with shape
            (batch_size, output_dim * num_mixtures * 3) where the last dimension
            contains [pi, mu, sigma] parameters for each mixture component.
        """
        x = inputs

        # Pass through feature extraction layers
        for layer in self.feature_layers:
            x = layer(x, training=training)

        # Pass through MDN layer
        return self.mdn_layer(x, training=training)

    def sample(
            self,
            inputs: keras.KerasTensor,
            num_samples: int = 1,
            temp: float = 1.0,
            seed: Optional[int] = None
    ) -> keras.KerasTensor:
        """Generate samples from the predicted distribution.

        Args:
            inputs: Input tensor.
            num_samples: Number of samples to generate for each input.
                Defaults to 1.
            temp: Temperature parameter for sampling (higher = more random).
                Defaults to 1.0.
            seed: Optional seed for reproducible sampling. Defaults to None.

        Returns:
            Samples from the predicted distribution with shape
            (batch_size, num_samples, output_dim).

        Raises:
            ValueError: If num_samples is not positive or temp is not positive.
        """
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if temp <= 0:
            raise ValueError("temp must be positive")

        predictions = self(inputs, training=False)

        samples = []
        for i in range(num_samples):
            # Use different seeds for each sample if a seed is provided
            sample_seed = None if seed is None else seed + i
            sample = self.mdn_layer.sample(predictions, temp=temp, seed=sample_seed)
            samples.append(sample)

        # Stack samples along a new dimension using keras.ops
        return ops.stack(samples, axis=1)

    def predict_with_uncertainty(
            self,
            inputs: keras.KerasTensor,
            confidence_level: float = 0.95
    ) -> Dict[str, keras.KerasTensor]:
        """Generate predictions with uncertainty estimates.

        Args:
            inputs: Input tensor.
            confidence_level: Confidence level for prediction intervals (0-1).
                Defaults to 0.95.

        Returns:
            Dictionary containing uncertainty estimates:

            * **point_estimates**: Mean predictions (batch_size, output_dim)
            * **total_variance**: Total predictive variance (batch_size, output_dim)
            * **aleatoric_variance**: Data uncertainty component (batch_size, output_dim)
            * **epistemic_variance**: Model uncertainty component (batch_size, output_dim)
            * **lower_bound**: Lower prediction interval bounds (batch_size, output_dim)
            * **upper_bound**: Upper prediction interval bounds (batch_size, output_dim)

        Raises:
            ValueError: If confidence_level is not in the range (0, 1).
        """
        if not (0 < confidence_level < 1):
            raise ValueError("confidence_level must be in the range (0, 1)")

        # Get model predictions
        predictions = self.predict(inputs)

        # Get point estimates (mean of mixture distribution)
        point_estimates = get_point_estimate(
            model=self,
            x_data=inputs,
            mdn_layer=self.mdn_layer
        )

        # Get uncertainty estimates
        total_variance, aleatoric_variance = get_uncertainty(
            model=self,
            x_data=inputs,
            mdn_layer=self.mdn_layer,
            point_estimates=point_estimates
        )

        # Calculate epistemic variance (model uncertainty)
        epistemic_variance = total_variance - aleatoric_variance

        # Get prediction intervals
        lower_bound, upper_bound = get_prediction_intervals(
            point_estimates=point_estimates,
            total_variance=total_variance,
            confidence_level=confidence_level
        )

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

        Args:
            optimizer: Optimizer instance or string name.
            metrics: List of metrics to track during training. Defaults to None.
            **kwargs: Additional compile arguments.
        """
        super().compile(
            optimizer=optimizer,
            loss=self.mdn_layer.loss_func,
            metrics=metrics,
            **kwargs
        )
        logger.info(f"MDNModel compiled with optimizer: {optimizer}")

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization.

        Returns:
            Dictionary containing the model configuration.
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
        base_config = super().get_config()
        return {**base_config, **config}

    def get_build_config(self) -> Dict[str, Any]:
        """Get the build configuration for serialization.

        Returns:
            Dictionary containing the build configuration.
        """
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build the model from a build configuration.

        Args:
            config: Dictionary containing the build configuration.
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MDNModel":
        """Create a model from its configuration.

        Args:
            config: Dictionary with the model configuration.

        Returns:
            A new MDN model instance.
        """
        config_copy = config.copy()

        # Deserialize initializer and regularizer
        config_copy["kernel_initializer"] = keras.initializers.deserialize(
            config["kernel_initializer"]
        )

        if config["kernel_regularizer"] is not None:
            config_copy["kernel_regularizer"] = keras.regularizers.deserialize(
                config["kernel_regularizer"]
            )

        return cls(**config_copy)

    def save(self, filepath: str, **kwargs: Any) -> None:
        """Save the model to a file.

        Args:
            filepath: Path where to save the model. If the path doesn't end with
                '.keras', the extension will be added automatically.
            **kwargs: Additional save arguments.
        """
        # Ensure the file extension is .keras
        if not filepath.endswith('.keras'):
            filepath += '.keras'

        logger.info(f"Saving MDNModel to: {filepath}")
        super().save(filepath, **kwargs)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the model.

        Args:
            input_shape: Shape of the input tensor.

        Returns:
            Output shape tuple.
        """
        # Convert input_shape to list for manipulation
        input_shape_list = list(input_shape)

        # The output shape is [batch_size, output_dim * num_mixtures * 3]
        # where 3 accounts for pi, mu, sigma parameters
        output_features = self.output_dim * self.num_mix * 3

        # Return as tuple
        return tuple(input_shape_list[:-1] + [output_features])

# ---------------------------------------------------------------------
