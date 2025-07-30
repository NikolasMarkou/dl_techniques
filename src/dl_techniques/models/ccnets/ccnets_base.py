"""
Causal Cooperative Networks (CCNets) - Base Network Classes

This module implements the three fundamental networks of CCNets:
- Explainer Network: Creates compressed explanations of input data
- Reasoner Network: Makes predictions based on input and explanations
- Producer Network: Generates and reconstructs data based on labels and explanations
"""

from typing import Optional, Union, Tuple, Dict, Any
import keras
from keras import ops
from dl_techniques.utils.logger import logger


class ExplainerNetwork(keras.Model):
    """
    Explainer Network for CCNets.

    Creates compressed explanation vectors from input observations.
    The explanation vector serves as a bridge between all three networks,
    providing interpretable latent representations.

    Args:
        input_dim: Dimensionality of input data
        explanation_dim: Dimensionality of explanation vector
        hidden_dims: List of hidden layer dimensions
        activation: Activation function for hidden layers
        dropout_rate: Dropout rate for regularization
        kernel_initializer: Weight initialization method
        **kwargs: Additional arguments for keras.Model
    """

    def __init__(
            self,
            input_dim: int,
            explanation_dim: int,
            hidden_dims: Optional[list] = None,
            activation: str = 'relu',
            dropout_rate: float = 0.3,
            kernel_initializer: str = 'glorot_uniform',
            **kwargs
    ):
        super().__init__(**kwargs)

        self.input_dim = input_dim
        self.explanation_dim = explanation_dim
        self.hidden_dims = hidden_dims or [512, 256]
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.kernel_initializer = kernel_initializer

        # Build encoder layers
        self.encoder_layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(self.hidden_dims):
            self.encoder_layers.extend([
                keras.layers.Dense(
                    hidden_dim,
                    activation=self.activation,
                    kernel_initializer=self.kernel_initializer,
                    name=f'explainer_hidden_{i}'
                ),
                keras.layers.Dropout(self.dropout_rate, name=f'explainer_dropout_{i}')
            ])
            prev_dim = hidden_dim

        # Final explanation layer with tanh activation for bounded outputs
        self.explanation_layer = keras.layers.Dense(
            self.explanation_dim,
            activation='tanh',
            kernel_initializer=self.kernel_initializer,
            name='explanation_output'
        )

        logger.info(f"Initialized ExplainerNetwork: input_dim={input_dim}, explanation_dim={explanation_dim}")

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """
        Forward pass through explainer network.

        Args:
            inputs: Input observations [batch_size, input_dim]
            training: Whether in training mode

        Returns:
            Explanation vectors [batch_size, explanation_dim]
        """
        x = inputs

        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = layer(x, training=training)

        # Generate explanation vector
        explanation = self.explanation_layer(x, training=training)

        return explanation

    def get_config(self) -> Dict[str, Any]:
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            'input_dim': self.input_dim,
            'explanation_dim': self.explanation_dim,
            'hidden_dims': self.hidden_dims,
            'activation': self.activation,
            'dropout_rate': self.dropout_rate,
            'kernel_initializer': self.kernel_initializer,
        })
        return config


class ReasonerNetwork(keras.Model):
    """
    Reasoner Network for CCNets.

    Makes predictions based on both input observations and explanation vectors.
    This network performs reasoning using both raw data and explanations,
    enabling cross-verification with the Producer network.

    Args:
        input_dim: Dimensionality of input data
        explanation_dim: Dimensionality of explanation vector
        output_dim: Dimensionality of output predictions
        hidden_dims: List of hidden layer dimensions
        activation: Activation function for hidden layers
        output_activation: Activation function for output layer
        dropout_rate: Dropout rate for regularization
        kernel_initializer: Weight initialization method
        **kwargs: Additional arguments for keras.Model
    """

    def __init__(
            self,
            input_dim: int,
            explanation_dim: int,
            output_dim: int,
            hidden_dims: Optional[list] = None,
            activation: str = 'relu',
            output_activation: str = 'softmax',
            dropout_rate: float = 0.3,
            kernel_initializer: str = 'glorot_uniform',
            **kwargs
    ):
        super().__init__(**kwargs)

        self.input_dim = input_dim
        self.explanation_dim = explanation_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims or [512, 256]
        self.activation = activation
        self.output_activation = output_activation
        self.dropout_rate = dropout_rate
        self.kernel_initializer = kernel_initializer

        # Fusion layer to combine input and explanation
        fusion_dim = 512
        self.fusion_layer = keras.layers.Dense(
            fusion_dim,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            name='reasoner_fusion'
        )

        # Build classifier layers
        self.classifier_layers = []
        prev_dim = fusion_dim

        for i, hidden_dim in enumerate(self.hidden_dims):
            self.classifier_layers.extend([
                keras.layers.Dense(
                    hidden_dim,
                    activation=self.activation,
                    kernel_initializer=self.kernel_initializer,
                    name=f'reasoner_hidden_{i}'
                ),
                keras.layers.Dropout(self.dropout_rate, name=f'reasoner_dropout_{i}')
            ])
            prev_dim = hidden_dim

        # Output layer
        self.output_layer = keras.layers.Dense(
            self.output_dim,
            activation=self.output_activation,
            kernel_initializer=self.kernel_initializer,
            name='reasoner_output'
        )

        logger.info(
            f"Initialized ReasonerNetwork: input_dim={input_dim}, explanation_dim={explanation_dim}, output_dim={output_dim}")

    def call(
            self,
            inputs: Tuple[keras.KerasTensor, keras.KerasTensor],
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass through reasoner network.

        Args:
            inputs: Tuple of (observations, explanations)
                observations: [batch_size, input_dim]
                explanations: [batch_size, explanation_dim]
            training: Whether in training mode

        Returns:
            Predictions [batch_size, output_dim]
        """
        observations, explanations = inputs

        # Fuse input and explanation
        combined = ops.concatenate([observations, explanations], axis=-1)
        fused = self.fusion_layer(combined, training=training)

        # Pass through classifier layers
        x = fused
        for layer in self.classifier_layers:
            x = layer(x, training=training)

        # Generate predictions
        predictions = self.output_layer(x, training=training)

        return predictions

    def get_config(self) -> Dict[str, Any]:
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            'input_dim': self.input_dim,
            'explanation_dim': self.explanation_dim,
            'output_dim': self.output_dim,
            'hidden_dims': self.hidden_dims,
            'activation': self.activation,
            'output_activation': self.output_activation,
            'dropout_rate': self.dropout_rate,
            'kernel_initializer': self.kernel_initializer,
        })
        return config


class ProducerNetwork(keras.Model):
    """
    Producer Network for CCNets.

    Generates and reconstructs data based on labels and explanation vectors.
    This network serves dual purposes:
    1. Generation: Creates data from ground truth labels and explanations
    2. Reconstruction: Creates data from predicted labels and explanations

    Args:
        label_dim: Dimensionality of label input
        explanation_dim: Dimensionality of explanation vector
        output_dim: Dimensionality of generated/reconstructed output
        hidden_dims: List of hidden layer dimensions
        activation: Activation function for hidden layers
        output_activation: Activation function for output layer
        dropout_rate: Dropout rate for regularization
        kernel_initializer: Weight initialization method
        **kwargs: Additional arguments for keras.Model
    """

    def __init__(
            self,
            label_dim: int,
            explanation_dim: int,
            output_dim: int,
            hidden_dims: Optional[list] = None,
            activation: str = 'relu',
            output_activation: str = 'sigmoid',
            dropout_rate: float = 0.3,
            kernel_initializer: str = 'glorot_uniform',
            **kwargs
    ):
        super().__init__(**kwargs)

        self.label_dim = label_dim
        self.explanation_dim = explanation_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims or [512, 256]
        self.activation = activation
        self.output_activation = output_activation
        self.dropout_rate = dropout_rate
        self.kernel_initializer = kernel_initializer

        # Build generator layers
        self.generator_layers = []
        input_dim = label_dim + explanation_dim
        prev_dim = input_dim

        for i, hidden_dim in enumerate(self.hidden_dims):
            self.generator_layers.extend([
                keras.layers.Dense(
                    hidden_dim,
                    activation=self.activation,
                    kernel_initializer=self.kernel_initializer,
                    name=f'producer_hidden_{i}'
                ),
                keras.layers.Dropout(self.dropout_rate, name=f'producer_dropout_{i}')
            ])
            prev_dim = hidden_dim

        # Output layer
        self.output_layer = keras.layers.Dense(
            self.output_dim,
            activation=self.output_activation,
            kernel_initializer=self.kernel_initializer,
            name='producer_output'
        )

        logger.info(
            f"Initialized ProducerNetwork: label_dim={label_dim}, explanation_dim={explanation_dim}, output_dim={output_dim}")

    def call(
            self,
            inputs: Tuple[keras.KerasTensor, keras.KerasTensor],
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass through producer network.

        Args:
            inputs: Tuple of (labels, explanations)
                labels: [batch_size, label_dim]
                explanations: [batch_size, explanation_dim]
            training: Whether in training mode

        Returns:
            Generated/reconstructed observations [batch_size, output_dim]
        """
        labels, explanations = inputs

        # Combine labels and explanations
        combined = ops.concatenate([labels, explanations], axis=-1)

        # Pass through generator layers
        x = combined
        for layer in self.generator_layers:
            x = layer(x, training=training)

        # Generate output
        output = self.output_layer(x, training=training)

        return output

    def get_config(self) -> Dict[str, Any]:
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            'label_dim': self.label_dim,
            'explanation_dim': self.explanation_dim,
            'output_dim': self.output_dim,
            'hidden_dims': self.hidden_dims,
            'activation': self.activation,
            'output_activation': self.output_activation,
            'dropout_rate': self.dropout_rate,
            'kernel_initializer': self.kernel_initializer,
        })
        return config


def create_explainer_network(
        input_dim: int,
        explanation_dim: int,
        **kwargs
) -> ExplainerNetwork:
    """
    Factory function to create an ExplainerNetwork.

    Args:
        input_dim: Dimensionality of input data
        explanation_dim: Dimensionality of explanation vector
        **kwargs: Additional arguments for ExplainerNetwork

    Returns:
        Configured ExplainerNetwork instance
    """
    return ExplainerNetwork(
        input_dim=input_dim,
        explanation_dim=explanation_dim,
        **kwargs
    )


def create_reasoner_network(
        input_dim: int,
        explanation_dim: int,
        output_dim: int,
        **kwargs
) -> ReasonerNetwork:
    """
    Factory function to create a ReasonerNetwork.

    Args:
        input_dim: Dimensionality of input data
        explanation_dim: Dimensionality of explanation vector
        output_dim: Dimensionality of output predictions
        **kwargs: Additional arguments for ReasonerNetwork

    Returns:
        Configured ReasonerNetwork instance
    """
    return ReasonerNetwork(
        input_dim=input_dim,
        explanation_dim=explanation_dim,
        output_dim=output_dim,
        **kwargs
    )


def create_producer_network(
        label_dim: int,
        explanation_dim: int,
        output_dim: int,
        **kwargs
) -> ProducerNetwork:
    """
    Factory function to create a ProducerNetwork.

    Args:
        label_dim: Dimensionality of label input
        explanation_dim: Dimensionality of explanation vector
        output_dim: Dimensionality of generated/reconstructed output
        **kwargs: Additional arguments for ProducerNetwork

    Returns:
        Configured ProducerNetwork instance
    """
    return ProducerNetwork(
        label_dim=label_dim,
        explanation_dim=explanation_dim,
        output_dim=output_dim,
        **kwargs
    )