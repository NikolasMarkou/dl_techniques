"""
Holographic Encoder-Decoder with Entropy-Based Architecture

This implementation creates a neural network inspired by concepts from quantum information
theory and holographic principles, providing an alternative approach to traditional neural networks
that doesn't rely on attention mechanisms or common gradient optimization tricks.

## Key Concepts Explained
### Entanglement Entropy
Originally from quantum mechanics, entanglement entropy measures how strongly different
parts of a system are correlated with each other. In our neural network context:
- It quantifies the information shared between different parts of the network
- High entanglement suggests strong correlations between features
- Low entanglement suggests more independent features
We use this concept to guide how information flows through the network.

### Holographic Principle
The holographic principle, borrowed from theoretical physics, suggests that information
about a system can be encoded on its boundary. In our network:
- Information is distributed across different "scales" or "layers" of the architecture
- Each scale captures different aspects of the input data
- The full information is preserved across these scales, not just in a single layer

### Entropic Geometry
This refers to using information theory principles (particularly entropy) to structure
the architecture of neural networks:
- We use entropy regularization to shape how information is distributed
- Different parts of the network are encouraged to have specific entropy profiles
- This creates a structured way for information to flow through the network

## Calculation Details



### Multi-Scale Holographic Decoding
The decoder uses multiple branches, each with a different entropy target:
    y_i = MPS_decoder_i(z) for i=1...N
The outputs are then combined:
    y_combined = concatenate(y_1, y_2, ..., y_N)
This creates a holographic-like representation where information is distributed
across different "scales" with low-entropy branches capturing global features and
high-entropy branches encoding local details.

## Practical Applications

This approach may offer benefits for:
- Problems requiring long-range correlations between input features
- Tasks where traditional attention mechanisms are computationally expensive
- Scenarios where model interpretability through information flow is important
- Applications where capturing hierarchical structure in data is beneficial

## Implementation Notes

The model consists of:
1. An encoder using MPS-inspired tensor contractions
2. A multi-branch decoder with different entropy targets
3. Custom regularizers that shape the information distribution

Rather than allowing network structure to emerge solely through gradient descent,
this architecture explicitly incorporates information-theoretic principles into
its design.
"""

from typing import Tuple, Optional, List, Union, Dict, Any, Callable
import numpy as np
import tensorflow as tf
import keras
import math

from dl_techniques.layers.mps_layer import MPSLayer
from dl_techniques.regularizers.entropy_regularizer import EntropyRegularizer




class HolographicEncoderDecoder(keras.Model):
    """
    Holographic Encoder-Decoder inspired by quantum information theory.

    This model attempts to capture the idea of holographic encoding where information
    about the whole system is encoded in a way that respects entropy scaling principles.

    Args:
        input_shape: Shape of the input data.
        latent_dim: Dimension of the latent space.
        output_shape: Shape of the output data. If None, uses input_shape.
        bond_dim: Bond dimension for MPS layers.
        num_branches: Number of decoder branches.
        regularization_strength: Strength of the entropy regularization.
        use_bias: Whether to use bias in layers.
        kernel_initializer: Initializer for kernels.
        kernel_regularizer: Additional regularizer for kernels.
    """

    def __init__(
            self,
            input_shape: Tuple[int, ...],
            latent_dim: int = 64,
            output_shape: Optional[Tuple[int, ...]] = None,
            bond_dim: int = 16,
            num_branches: int = 3,
            regularization_strength: float = 0.01,
            use_bias: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs
    ) -> None:
        """Initialize the model.

        Args:
            input_shape: Shape of the input data.
            latent_dim: Dimension of the latent space.
            output_shape: Shape of the output data. If None, uses input_shape.
            bond_dim: Bond dimension for MPS layers.
            num_branches: Number of decoder branches.
            regularization_strength: Strength of the entropy regularization.
            use_bias: Whether to use bias in layers.
            kernel_initializer: Initializer for kernels.
            kernel_regularizer: Additional regularizer for kernels.
        """
        super().__init__(**kwargs)

        # Store configuration
        self.input_shape = input_shape
        self.input_dim = np.prod(input_shape)
        self.latent_dim = latent_dim
        self.output_shape = output_shape if output_shape is not None else input_shape
        self.output_dim = np.prod(self.output_shape)
        self.bond_dim = bond_dim
        self.num_branches = num_branches
        self.regularization_strength = regularization_strength
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer

        # Initialize sublayer attributes
        self.encoder_mps = None
        self.decoder_branches = []
        self.output_projection = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the model components.

        Args:
            input_shape: Shape of the input tensor including batch dimension.
        """
        # Create entropy regularizer
        entropy_reg = EntropyRegularizer(strength=self.regularization_strength)

        # Combine regularizers
        if self.kernel_regularizer is not None:
            combined_regularizer = keras.regularizers.L1L2.from_config({
                **keras.regularizers.get(self.kernel_regularizer).get_config(),
                'l1': 0.0,  # Default in case the provided regularizer doesn't have l1
                'l2': 0.0,  # Default in case the provided regularizer doesn't have l2
            })
        else:
            combined_regularizer = keras.regularizers.L2(1e-4)

        # Encoder layers
        self.encoder_mps = MPSLayer(
            output_dim=self.latent_dim,
            bond_dim=self.bond_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=combined_regularizer,
            name="encoder_mps"
        )

        # Create multiple decoder branches with different entropy profiles
        # This creates a holographic-like property where information is distributed
        # across different entropy scales
        self.decoder_branches = []
        for i in range(self.num_branches):
            # Each branch has a different entropy target
            target_entropy = 0.3 + (0.6 * i / (self.num_branches - 1))
            branch_reg = EntropyRegularizer(
                strength=self.regularization_strength,
                target_entropy=target_entropy
            )

            branch = MPSLayer(
                output_dim=self.output_dim // self.num_branches,
                bond_dim=self.bond_dim,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=branch_reg,
                name=f"decoder_branch_{i}"
            )
            self.decoder_branches.append(branch)

        # Final output projection
        self.output_projection = keras.layers.Dense(
            self.output_dim,
            activation=None,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=combined_regularizer,
            name="output_projection"
        )

        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """Forward pass through the holographic encoder-decoder model.

        This method implements the holographic encoding-decoding process:
        1. Flatten input data to 2D tensor (batch_size, input_dim)
        2. Encode data through MPS layer to latent representation
        3. Decode through multiple branches with different entropy targets
        4. Combine multi-scale features and project to output space

        The multi-branch decoding is key to the holographic principle implementation:
        - Each branch has a different entropy target (from low to high)
        - Low-entropy branches capture global, coarse-grained features
        - High-entropy branches encode local, fine-grained features
        - Together they form a holographic representation where information is
          distributed across different "scales" or "layers"

        Mathematically, the process is:
        1. z = MPS_encoder(flatten(x))  # Latent encoding
        2. y_i = MPS_decoder_i(z) for i=1...N  # Multi-branch decoding
        3. y_combined = concatenate(y_1, y_2, ..., y_N)  # Combine scales
        4. y = projection(y_combined)  # Final output projection

        This multi-scale approach resembles holographic principles where information
        about a system is encoded at different "scales" or "regions" of the representation,
        with the complete information recoverable from the entire set.

        Args:
            inputs: Input tensor of original shape.
            training: Whether the model is in training mode.

        Returns:
            Output tensor of shape [batch_size, *output_shape].
        """
        # Flatten input to 2D tensor
        batch_size = tf.shape(inputs)[0]
        x = tf.reshape(inputs, [batch_size, -1])  # Shape: [batch_size, input_dim]

        # Encode using MPS layer to latent representation
        latent = self.encoder_mps(x)  # Shape: [batch_size, latent_dim]

        # Decode through multiple branches with different entropy targets
        # Each branch has a different entropy regularization target
        # This creates a holographic-like encoding across different "scales"
        branch_outputs = [branch(latent) for branch in self.decoder_branches]

        # Concatenate branch outputs to combine information from different scales
        multi_scale_features = tf.concat(branch_outputs, axis=-1)

        # Final projection to output dimension
        output = self.output_projection(multi_scale_features)

        # Reshape to original output shape
        output = tf.reshape(output, [batch_size] + list(self.output_shape))

        return output

    def get_config(self) -> Dict[str, Any]:
        """Get the model configuration.

        Returns:
            Dictionary containing the model configuration.
        """
        config = {
            'input_shape': self.input_shape,
            'latent_dim': self.latent_dim,
            'output_shape': self.output_shape,
            'bond_dim': self.bond_dim,
            'num_branches': self.num_branches,
            'regularization_strength': self.regularization_strength,
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer) if isinstance(
                self.kernel_initializer, keras.initializers.Initializer) else self.kernel_initializer,
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
        }
        return config


class ModelConfig:
    """Configuration class for model hyperparameters.

    Args:
        input_shape: Shape of input data.
        num_classes: Number of output classes.
        learning_rate: Initial learning rate.
        weight_decay: L2 regularization factor.
        latent_dim: Dimension of the latent space.
        bond_dim: Bond dimension for MPS layers.
        num_branches: Number of decoder branches.
    """

    def __init__(
            self,
            input_shape: Tuple[int, ...],
            num_classes: int,
            learning_rate: float = 0.001,
            weight_decay: float = 0.0001,
            latent_dim: int = 64,
            bond_dim: int = 16,
            num_branches: int = 3
    ) -> None:
        """Initialize the configuration.

        Args:
            input_shape: Shape of input data.
            num_classes: Number of output classes.
            learning_rate: Initial learning rate.
            weight_decay: L2 regularization factor.
            latent_dim: Dimension of the latent space.
            bond_dim: Bond dimension for MPS layers.
            num_branches: Number of decoder branches.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.latent_dim = latent_dim
        self.bond_dim = bond_dim
        self.num_branches = num_branches


def create_holographic_model(config: ModelConfig) -> keras.Model:
    """Creates a holographic encoder-decoder model.

    Args:
        config: ModelConfig instance containing model parameters.

    Returns:
        Configured holographic model.
    """
    # Create model
    model = HolographicEncoderDecoder(
        input_shape=config.input_shape,
        latent_dim=config.latent_dim,
        bond_dim=config.bond_dim,
        num_branches=config.num_branches,
        regularization_strength=config.weight_decay,
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.L2(config.weight_decay)
    )

    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=config.learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    )

    return model


def save_model(model: keras.Model, filepath: str) -> None:
    """Saves model in .keras format.

    Args:
        model: Keras model to save.
        filepath: Path to save the model.
    """
    model.save(filepath, save_format="keras")


def load_model(filepath: str) -> keras.Model:
    """Loads model from .keras format.

    Args:
        filepath: Path to saved model.

    Returns:
        Loaded Keras model.
    """
    # Register custom objects to ensure loading works
    custom_objects = {
        'EntropyRegularizer': EntropyRegularizer,
        'MPSLayer': MPSLayer,
        'HolographicEncoderDecoder': HolographicEncoderDecoder
    }
    return keras.models.load_model(filepath, custom_objects=custom_objects)

