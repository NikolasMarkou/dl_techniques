import keras
import tensorflow as tf
from keras.api import layers
from typing import List, Union, Optional, Dict, Any
from dl_techniques.layers.mdn import (
    MDNLayer,
    get_mixture_loss_func,
    get_mixture_sampling_func
)

from dl_techniques.utils.logger import logger

class MDNModel(keras.Model):
    """A complete Mixture Density Network model.

    This combines a feature extraction network with an MDN layer and handles
    the appropriate loss function and sampling functionality.

    Args:
        hidden_layers: List of hidden layer sizes
        output_dimension: Dimensionality of the output space
        num_mixtures: Number of Gaussian mixtures
        hidden_activation: Activation function for hidden layers
        kernel_initializer: Initializer for the kernel weights matrix
        kernel_regularizer: Regularizer function applied to the kernel weights matrix
    """

    def __init__(
            self,
            hidden_layers: List[int],
            output_dimension: int,
            num_mixtures: int,
            hidden_activation: str = "relu",
            kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs
    ) -> None:
        """Initialize the MDN model.

        Args:
            hidden_layers: List of hidden layer sizes
            output_dimension: Dimensionality of the output space
            num_mixtures: Number of Gaussian mixtures
            hidden_activation: Activation function for hidden layers
            kernel_initializer: Initializer for the kernel weights matrix
            kernel_regularizer: Regularizer function applied to the kernel weights matrix
            **kwargs: Additional model arguments
        """
        super().__init__(**kwargs)

        self.hidden_layers_sizes = hidden_layers
        self.output_dim = output_dimension
        self.num_mix = num_mixtures
        self.hidden_activation = hidden_activation
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = kernel_regularizer

        # Build feature extraction layers
        self.feature_layers = []
        for units in hidden_layers:
            self.feature_layers.append(
                layers.Dense(
                    units,
                    activation=hidden_activation,
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer
                )
            )

        # MDN layer
        self.mdn_layer = MDNLayer(
            output_dimension=output_dimension,
            num_mixtures=num_mixtures,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer
        )

        # Create loss function
        self.loss_func = get_mixture_loss_func(output_dimension, num_mixtures)

        # Create sampling function
        self.sampling_func = get_mixture_sampling_func(output_dimension, num_mixtures)

        logger.info(f"Initialized MDNModel with {len(hidden_layers)} hidden layers and {num_mixtures} mixtures")

    @tf.function
    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """Forward pass of the model.

        Args:
            inputs: Input tensor
            training: Boolean indicating whether the layer should behave in training mode

        Returns:
            tf.Tensor: Output tensor containing mixture parameters
        """
        x = inputs
        # Pass through feature extraction layers
        for layer in self.feature_layers:
            x = layer(x, training=training)

        # Pass through MDN layer
        return self.mdn_layer(x, training=training)

    def sample(self, inputs: tf.Tensor, num_samples: int = 1) -> tf.Tensor:
        """Generate samples from the predicted distribution.

        Args:
            inputs: Input tensor
            num_samples: Number of samples to generate for each input

        Returns:
            tf.Tensor: Samples from the predicted distribution
        """
        predictions = self(inputs, training=False)

        samples = []
        for _ in range(num_samples):
            sample = self.sampling_func(predictions)
            samples.append(sample)

        return tf.stack(samples, axis=1)

    def compile(self, optimizer: Union[str, keras.optimizers.Optimizer], **kwargs) -> None:
        """Configures the model for training.

        Args:
            optimizer: Optimizer instance
            **kwargs: Additional compile arguments
        """
        super().compile(optimizer=optimizer, loss=self.loss_func, **kwargs)

    def get_config(self) -> Dict[str, Any]:
        """Gets model configuration.

        Returns:
            Dict[str, Any]: Model configuration dictionary
        """
        config = {
            "hidden_layers": self.hidden_layers_sizes,
            "output_dimension": self.output_dim,
            "num_mixtures": self.num_mix,
            "hidden_activation": self.hidden_activation,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer)
        }
        base_config = super().get_config()
        return {**base_config, **config}