"""
Causal Cooperative Networks (CCNets) - Base Network Classes

This module implements the three fundamental networks of CCNets:
- Explainer Network: Creates compressed explanations of input data
- Reasoner Network: Makes predictions based on input and explanations
- Producer Network: Generates and reconstructs data based on labels and explanations

The architecture implements cooperative learning inspired by the Three Kingdoms philosophy,
where three networks collaborate rather than compete to achieve explainability,
reasoning, and faithful data reproduction.
"""

import keras
from keras import ops
from typing import Optional, Tuple, Dict, Any, List, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class ExplainerNetwork(keras.Model):
    """
    Explainer Network for CCNets.

    Creates compressed explanation vectors from input observations that serve as
    interpretable latent representations bridging all three networks in the
    cooperative architecture.

    **Purpose**: Transforms raw input data into meaningful explanation vectors
    that capture the essential information needed for both prediction and
    data generation/reconstruction tasks.

    **Architecture Flow**:
    ```
    Input(shape=[batch, input_dim])
           ↓
    Hidden₁(hidden_dims[0], activation) → Dropout
           ↓
    Hidden₂(hidden_dims[1], activation) → Dropout
           ↓
          ...
           ↓
    Explanation(explanation_dim, activation='tanh')
           ↓
    Output(shape=[batch, explanation_dim])  # Bounded explanation vector
    ```

    Args:
        input_dim: Dimensionality of input data
        explanation_dim: Dimensionality of explanation vector (compressed representation)
        hidden_dims: List of hidden layer dimensions. Defaults to [512, 256].
        activation: Activation function for hidden layers. Defaults to 'relu'.
        dropout_rate: Dropout rate for regularization. Defaults to 0.3.
        kernel_initializer: Weight initialization method. Defaults to 'glorot_uniform'.
        kernel_regularizer: Optional regularizer for weights.
        use_batch_norm: Whether to use batch normalization. Defaults to False.
        **kwargs: Additional arguments for keras.Model

    Input shape:
        2D tensor with shape: `(batch_size, input_dim)`.

    Output shape:
        2D tensor with shape: `(batch_size, explanation_dim)`.
        Values are bounded to [-1, 1] via tanh activation.

    Example:
        ```python
        explainer = ExplainerNetwork(
            input_dim=784,
            explanation_dim=64,
            hidden_dims=[512, 256, 128],
            dropout_rate=0.2
        )

        # Forward pass
        x = keras.random.normal((32, 784))
        explanations = explainer(x)  # Shape: (32, 64)
        ```
    """

    def __init__(
        self,
        input_dim: int,
        explanation_dim: int,
        hidden_dims: Optional[List[int]] = None,
        activation: Union[str, callable] = 'relu',
        dropout_rate: float = 0.3,
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        use_batch_norm: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)

        # Validate inputs
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if explanation_dim <= 0:
            raise ValueError(f"explanation_dim must be positive, got {explanation_dim}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")

        # Store configuration
        self.input_dim = input_dim
        self.explanation_dim = explanation_dim
        self.hidden_dims = hidden_dims or [512, 256]
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.use_batch_norm = use_batch_norm

        # Build encoder layers
        self.encoder_layers = []
        for i, hidden_dim in enumerate(self.hidden_dims):
            # Dense layer
            self.encoder_layers.append(
                keras.layers.Dense(
                    hidden_dim,
                    activation=None,  # Apply activation separately for flexibility
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name=f'explainer_dense_{i}'
                )
            )

            # Batch normalization (optional)
            if self.use_batch_norm:
                self.encoder_layers.append(
                    keras.layers.BatchNormalization(name=f'explainer_bn_{i}')
                )

            # Activation
            self.encoder_layers.append(
                keras.layers.Activation(self.activation, name=f'explainer_act_{i}')
            )

            # Dropout
            if self.dropout_rate > 0:
                self.encoder_layers.append(
                    keras.layers.Dropout(self.dropout_rate, name=f'explainer_dropout_{i}')
                )

        # Final explanation layer with tanh activation for bounded outputs [-1, 1]
        self.explanation_layer = keras.layers.Dense(
            self.explanation_dim,
            activation='tanh',
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='explanation_output'
        )

        logger.info(
            f"Initialized ExplainerNetwork: input_dim={input_dim}, "
            f"explanation_dim={explanation_dim}, hidden_dims={self.hidden_dims}"
        )

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass through explainer network.

        Args:
            inputs: Input observations [batch_size, input_dim]
            training: Whether in training mode

        Returns:
            Explanation vectors [batch_size, explanation_dim] with values in [-1, 1]
        """
        x = inputs

        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = layer(x, training=training)

        # Generate bounded explanation vector
        explanation = self.explanation_layer(x, training=training)

        return explanation

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        batch_size = input_shape[0]
        return (batch_size, self.explanation_dim)

    def get_config(self) -> Dict[str, Any]:
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            'input_dim': self.input_dim,
            'explanation_dim': self.explanation_dim,
            'hidden_dims': self.hidden_dims,
            'activation': self.activation,
            'dropout_rate': self.dropout_rate,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'use_batch_norm': self.use_batch_norm,
        })
        return config


@keras.saving.register_keras_serializable()
class ReasonerNetwork(keras.Model):
    """
    Reasoner Network for CCNets.

    Makes predictions based *solely* on explanation vectors, forcing the
    Explainer to encode all necessary information for the reasoning task.

    **Purpose**: Performs reasoning on compressed explanations, providing the
    prediction capability in the cooperative framework.

    **Architecture Flow**:
    ```
    Explanations([batch, explanation_dim])
                    ↓
    Hidden₁(hidden_dims[0], activation) → Dropout
                    ↓
    Hidden₂(hidden_dims[1], activation) → Dropout
                    ↓
                  ...
                    ↓
    Output(output_dim, output_activation)  # Predictions
                    ↓
    Predictions([batch, output_dim])
    ```

    Args:
        explanation_dim: Dimensionality of explanation vector
        output_dim: Dimensionality of output predictions
        hidden_dims: List of hidden layer dimensions. Defaults to [512, 256].
        activation: Activation function for hidden layers. Defaults to 'relu'.
        output_activation: Activation function for output layer. Defaults to 'softmax'.
        dropout_rate: Dropout rate for regularization. Defaults to 0.3.
        kernel_initializer: Weight initialization method. Defaults to 'glorot_uniform'.
        kernel_regularizer: Optional regularizer for weights.
        use_batch_norm: Whether to use batch normalization. Defaults to False.
        **kwargs: Additional arguments for keras.Model

    Input shape:
        2D tensor with shape: `(batch_size, explanation_dim)`

    Output shape:
        2D tensor with shape: `(batch_size, output_dim)`.

    Example:
        ```python
        reasoner = ReasonerNetwork(
            explanation_dim=64,
            output_dim=10,
            hidden_dims=[512, 256]
        )

        # Forward pass
        explanations = keras.random.normal((32, 64))
        predictions = reasoner(explanations)  # Shape: (32, 10)
        ```
    """

    def __init__(
        self,
        explanation_dim: int,
        output_dim: int,
        hidden_dims: Optional[List[int]] = None,
        activation: Union[str, callable] = 'relu',
        output_activation: Union[str, callable] = 'softmax',
        dropout_rate: float = 0.3,
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        use_batch_norm: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)

        # Validate inputs
        if explanation_dim <= 0:
            raise ValueError(f"explanation_dim must be positive, got {explanation_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")

        # Store configuration
        self.explanation_dim = explanation_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims or [512, 256]
        self.activation = activation
        self.output_activation = output_activation
        self.dropout_rate = dropout_rate
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.use_batch_norm = use_batch_norm

        # Build classifier layers
        self.classifier_layers = []
        for i, hidden_dim in enumerate(self.hidden_dims):
            # Dense layer
            self.classifier_layers.append(
                keras.layers.Dense(
                    hidden_dim,
                    activation=None,
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name=f'reasoner_dense_{i}'
                )
            )

            # Batch normalization (optional)
            if self.use_batch_norm:
                self.classifier_layers.append(
                    keras.layers.BatchNormalization(name=f'reasoner_bn_{i}')
                )

            # Activation
            self.classifier_layers.append(
                keras.layers.Activation(self.activation, name=f'reasoner_act_{i}')
            )

            # Dropout
            if self.dropout_rate > 0:
                self.classifier_layers.append(
                    keras.layers.Dropout(self.dropout_rate, name=f'reasoner_dropout_{i}')
                )

        # Output layer
        self.output_layer = keras.layers.Dense(
            self.output_dim,
            activation=self.output_activation,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='reasoner_output'
        )

        logger.info(
            f"Initialized ReasonerNetwork: "
            f"explanation_dim={explanation_dim}, output_dim={output_dim}"
        )

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass through reasoner network.

        Args:
            inputs: Explanation vectors [batch_size, explanation_dim]
            training: Whether in training mode

        Returns:
            Predictions [batch_size, output_dim]
        """
        explanations = inputs
        x = explanations

        # Pass through classifier layers
        for layer in self.classifier_layers:
            x = layer(x, training=training)

        # Generate predictions
        predictions = self.output_layer(x, training=training)

        return predictions

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        batch_size = input_shape[0]
        return (batch_size, self.output_dim)

    def get_config(self) -> Dict[str, Any]:
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            'explanation_dim': self.explanation_dim,
            'output_dim': self.output_dim,
            'hidden_dims': self.hidden_dims,
            'activation': self.activation,
            'output_activation': self.output_activation,
            'dropout_rate': self.dropout_rate,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'use_batch_norm': self.use_batch_norm,
        })
        return config


@keras.saving.register_keras_serializable()
class ProducerNetwork(keras.Model):
    """
    Producer Network for CCNets.

    Generates and reconstructs data based on labels and explanation vectors,
    serving dual purposes in the cooperative architecture.

    **Dual Purpose**:
    1. **Generation**: Creates data from ground truth labels and explanations
    2. **Reconstruction**: Creates data from predicted labels and explanations

    **Architecture Flow**:
    ```
    Labels([batch, label_dim]) + Explanations([batch, explanation_dim])
                    ↓
    Concatenate → Hidden₁(hidden_dims[0], activation) → Dropout
                    ↓
    Hidden₂(hidden_dims[1], activation) → Dropout
                    ↓
                  ...
                    ↓
    Output(output_dim, output_activation)  # Generated/Reconstructed data
                    ↓
    Outputs([batch, output_dim])
    ```

    Args:
        label_dim: Dimensionality of label input
        explanation_dim: Dimensionality of explanation vector
        output_dim: Dimensionality of generated/reconstructed output
        hidden_dims: List of hidden layer dimensions. Defaults to [256, 512].
        activation: Activation function for hidden layers. Defaults to 'relu'.
        output_activation: Activation function for output layer. Defaults to 'sigmoid'.
        dropout_rate: Dropout rate for regularization. Defaults to 0.3.
        kernel_initializer: Weight initialization method. Defaults to 'glorot_uniform'.
        kernel_regularizer: Optional regularizer for weights.
        use_batch_norm: Whether to use batch normalization. Defaults to False.
        **kwargs: Additional arguments for keras.Model

    Input shape:
        Tuple of two 2D tensors:
        - labels: `(batch_size, label_dim)`
        - explanations: `(batch_size, explanation_dim)`

    Output shape:
        2D tensor with shape: `(batch_size, output_dim)`.

    Example:
        ```python
        producer = ProducerNetwork(
            label_dim=10,
            explanation_dim=64,
            output_dim=784,
            hidden_dims=[256, 512, 512]
        )

        # Forward pass
        labels = keras.random.normal((32, 10))
        explanations = keras.random.normal((32, 64))
        generated_data = producer([labels, explanations])  # Shape: (32, 784)
        ```
    """

    def __init__(
        self,
        label_dim: int,
        explanation_dim: int,
        output_dim: int,
        hidden_dims: Optional[List[int]] = None,
        activation: Union[str, callable] = 'relu',
        output_activation: Union[str, callable] = 'sigmoid',
        dropout_rate: float = 0.3,
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        use_batch_norm: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)

        # Validate inputs
        if label_dim <= 0:
            raise ValueError(f"label_dim must be positive, got {label_dim}")
        if explanation_dim <= 0:
            raise ValueError(f"explanation_dim must be positive, got {explanation_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")

        # Store configuration
        self.label_dim = label_dim
        self.explanation_dim = explanation_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims or [256, 512]  # Expanding for generation
        self.activation = activation
        self.output_activation = output_activation
        self.dropout_rate = dropout_rate
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.use_batch_norm = use_batch_norm

        # Build generator layers
        self.generator_layers = []
        for i, hidden_dim in enumerate(self.hidden_dims):
            # Dense layer
            self.generator_layers.append(
                keras.layers.Dense(
                    hidden_dim,
                    activation=None,
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name=f'producer_dense_{i}'
                )
            )

            # Batch normalization (optional)
            if self.use_batch_norm:
                self.generator_layers.append(
                    keras.layers.BatchNormalization(name=f'producer_bn_{i}')
                )

            # Activation
            self.generator_layers.append(
                keras.layers.Activation(self.activation, name=f'producer_act_{i}')
            )

            # Dropout
            if self.dropout_rate > 0:
                self.generator_layers.append(
                    keras.layers.Dropout(self.dropout_rate, name=f'producer_dropout_{i}')
                )

        # Output layer
        self.output_layer = keras.layers.Dense(
            self.output_dim,
            activation=self.output_activation,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='producer_output'
        )

        logger.info(
            f"Initialized ProducerNetwork: label_dim={label_dim}, "
            f"explanation_dim={explanation_dim}, output_dim={output_dim}"
        )

    def call(
        self,
        inputs: Tuple[keras.KerasTensor, keras.KerasTensor],
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass through producer network.

        Args:
            inputs: Tuple of (labels, explanations)
                labels: [batch_size, label_dim] - Either ground truth or predicted
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

    def compute_output_shape(self, input_shape: Tuple[Tuple[Optional[int], ...], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        batch_size = input_shape[0][0]  # From first input (labels)
        return (batch_size, self.output_dim)

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
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'use_batch_norm': self.use_batch_norm,
        })
        return config

# ---------------------------------------------------------------------

def create_explainer_network(
    input_dim: int,
    explanation_dim: int,
    **kwargs
) -> ExplainerNetwork:
    """
    Factory function to create an ExplainerNetwork.

    Args:
        input_dim: Dimensionality of input data
        explanation_dim: Dimensionality of explanation vectors
        **kwargs: Additional arguments for ExplainerNetwork

    Returns:
        Configured ExplainerNetwork instance
    """
    return ExplainerNetwork(
        input_dim=input_dim,
        explanation_dim=explanation_dim,
        **kwargs
    )

# ---------------------------------------------------------------------

def create_reasoner_network(
    explanation_dim: int,
    output_dim: int,
    **kwargs
) -> ReasonerNetwork:
    """
    Factory function to create a ReasonerNetwork.

    Args:
        explanation_dim: Dimensionality of explanation vectors
        output_dim: Dimensionality of output predictions
        **kwargs: Additional arguments for ReasonerNetwork

    Returns:
        Configured ReasonerNetwork instance
    """
    return ReasonerNetwork(
        explanation_dim=explanation_dim,
        output_dim=output_dim,
        **kwargs
    )

# ---------------------------------------------------------------------

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
        explanation_dim: Dimensionality of explanation vectors
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

# ---------------------------------------------------------------------