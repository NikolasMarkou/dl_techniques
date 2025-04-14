"""
Mixture Density Network (MDN) Model Implementation for Keras.

This module provides a Keras model implementation for Mixture Density Networks.
It combines feature extraction layers with an MDN layer to predict probability
distributions rather than single point estimates.
"""

import keras
import tensorflow as tf
from keras.api import layers
from typing import List, Union, Optional, Dict, Any, Tuple
from dl_techniques.layers.mdn_layer import (
    MDNLayer,
    get_point_estimate,
    get_uncertainty,
    get_prediction_intervals
)


class MDNModel(keras.Model):
    """A complete Mixture Density Network model.

    This model combines a feature extraction network with an MDN layer and handles
    the appropriate loss function and sampling functionality. It enables the prediction
    of probability distributions instead of single point estimates, which is valuable
    for regression problems with multi-modal outputs or heteroscedastic noise.

    Args:
        hidden_layers: List of hidden layer sizes
        output_dimension: Dimensionality of the output space
        num_mixtures: Number of Gaussian mixtures
        hidden_activation: Activation function for hidden layers
        kernel_initializer: Initializer for the kernel weights matrix
        kernel_regularizer: Regularizer function applied to the kernel weights matrix
        use_batch_norm: Whether to use batch normalization between hidden layers
        dropout_rate: Dropout rate for regularization (None for no dropout)
        **kwargs: Additional model arguments

    Examples:
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
            **kwargs
    ) -> None:
        """Initialize the MDN model."""
        super().__init__(**kwargs)

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

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the model with the given input shape.

        This method creates the feature extraction layers and the MDN layer
        when the model is first called.

        Args:
            input_shape: Shape tuple of the input
        """
        # Build feature extraction layers
        for units in self.hidden_layers_sizes:
            # Add Dense layer
            dense_layer = layers.Dense(
                units,
                activation=None,  # Activation applied after batch norm if used
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer
            )
            self.feature_layers.append(dense_layer)

            # Add batch normalization if requested
            if self.use_batch_norm:
                self.feature_layers.append(layers.BatchNormalization())

            # Add activation
            self.feature_layers.append(layers.Activation(self.hidden_activation))

            # Add dropout if specified
            if self.dropout_rate is not None:
                self.feature_layers.append(layers.Dropout(self.dropout_rate))

        # Create MDN layer
        self.mdn_layer = MDNLayer(
            output_dimension=self.output_dim,
            num_mixtures=self.num_mix,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer
        )

        # Build feature layers
        x = keras.Input(shape=input_shape[1:])
        for layer in self.feature_layers:
            x = layer(x)

        # Build MDN layer
        _ = self.mdn_layer(x)

        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """Forward pass of the model.

        Args:
            inputs: Input tensor
            training: Boolean indicating whether the model should behave in training mode

        Returns:
            tf.Tensor: Output tensor containing mixture parameters
        """
        x = inputs
        # Pass through feature extraction layers
        for layer in self.feature_layers:
            x = layer(x, training=training)

        # Pass through MDN layer
        return self.mdn_layer(x, training=training)

    def sample(self, inputs: tf.Tensor, num_samples: int = 1, temp: float = 1.0,
               seed: Optional[int] = None) -> tf.Tensor:
        """Generate samples from the predicted distribution.

        Args:
            inputs: Input tensor
            num_samples: Number of samples to generate for each input
            temp: Temperature parameter for sampling (higher = more random)
            seed: Optional seed for reproducible sampling

        Returns:
            tf.Tensor: Samples from the predicted distribution,
                       shape [batch_size, num_samples, output_dim]
        """
        predictions = self(inputs, training=False)

        samples = []
        for i in range(num_samples):
            # Use different seeds for each sample if a seed is provided
            sample_seed = None if seed is None else seed + i
            sample = self.mdn_layer.sample(predictions, temp=temp, seed=sample_seed)
            samples.append(sample)

        # Stack samples along a new dimension
        return tf.stack(samples, axis=1)

    def predict_with_uncertainty(
            self,
            inputs: tf.Tensor,
            confidence_level: float = 0.95
    ) -> Dict[str, tf.Tensor]:
        """Generate predictions with uncertainty estimates.

        Args:
            inputs: Input tensor
            confidence_level: Confidence level for prediction intervals (0-1)

        Returns:
            Dict[str, tf.Tensor]: Dictionary containing:
                - point_estimates: Mean predictions [batch_size, output_dim]
                - total_variance: Total predictive variance [batch_size, output_dim]
                - aleatoric_variance: Data uncertainty component [batch_size, output_dim]
                - epistemic_variance: Model uncertainty component [batch_size, output_dim]
                - lower_bound: Lower prediction interval bounds [batch_size, output_dim]
                - upper_bound: Upper prediction interval bounds [batch_size, output_dim]
        """
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
            'point_estimates': tf.convert_to_tensor(point_estimates),
            'total_variance': tf.convert_to_tensor(total_variance),
            'aleatoric_variance': tf.convert_to_tensor(aleatoric_variance),
            'epistemic_variance': tf.convert_to_tensor(epistemic_variance),
            'lower_bound': tf.convert_to_tensor(lower_bound),
            'upper_bound': tf.convert_to_tensor(upper_bound)
        }

    def compile(
            self,
            optimizer: Union[str, keras.optimizers.Optimizer],
            metrics: Optional[List[Union[str, keras.metrics.Metric]]] = None,
            **kwargs
    ) -> None:
        """Configures the model for training.

        Args:
            optimizer: Optimizer instance or string name
            metrics: List of metrics to track
            **kwargs: Additional compile arguments
        """
        super().compile(
            optimizer=optimizer,
            loss=self.mdn_layer.loss_func,
            metrics=metrics,
            **kwargs
        )

    def get_config(self) -> Dict[str, Any]:
        """Gets model configuration for serialization.

        Returns:
            Dict[str, Any]: Model configuration dictionary
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

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MDNModel":
        """Create a model from its config.

        Args:
            config: Dictionary with the model configuration

        Returns:
            MDNModel: A new MDN model instance
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

    def save(self, filepath: str, **kwargs) -> None:
        """Save the model to a file.

        Args:
            filepath: Path where to save the model
            **kwargs: Additional save arguments
        """
        # Ensure the file extension is .keras
        if not filepath.endswith('.keras'):
            filepath += '.keras'

        super().save(filepath, **kwargs)
