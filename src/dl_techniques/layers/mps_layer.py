"""
# Matrix Product States (MPS)
Matrix Product States are a representation method from quantum physics that efficiently
represents quantum systems with many particles. In machine learning terms, it's a way to
decompose high-dimensional data into a series of interconnected lower-dimensional tensors.
This can be seen as a specialized form of tensor network that:
- Reduces the parameter space needed to represent complex data
- Preserves important correlations between features
- Scales efficiently with input size

## MPS Tensor Contraction
The MPS layer implements a sequential tensor contraction process:
    output = B * (A¹[x₁] * A²[x₂] * ... * Aⁿ[xₙ]) * P
where:
- B is the boundary vector (initialized to ones)
- Aⁱ are the core tensors activated by input features x_i
- P is the projection matrix to output space
- * represents tensor contraction operations
"""
import math
import keras
import tensorflow as tf
from typing import Tuple, Optional, List, Union, Dict, Any, Callable


class MPSLayer(keras.layers.Layer):
    """
    Matrix Product State inspired layer for tensor decomposition.

    This layer implements a simplified version of MPS tensor contraction,
    which can capture long-range correlations while maintaining computational efficiency.

    Args:
        output_dim: Dimension of the output tensor.
        bond_dim: Internal bond dimension for the MPS representation.
        use_bias: Whether to include bias terms.
        kernel_initializer: Initializer for the kernel weights.
        kernel_regularizer: Regularizer for the kernel weights.
        bias_initializer: Initializer for the bias terms.
        bias_regularizer: Regularizer for the bias terms.
    """

    def __init__(
            self,
            output_dim: int,
            bond_dim: int = 16,
            use_bias: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
            bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs
    ) -> None:
        """Initialize the layer.

        Args:
            output_dim: Dimension of the output tensor.
            bond_dim: Internal bond dimension for the MPS representation.
            use_bias: Whether to include bias terms.
            kernel_initializer: Initializer for the kernel weights.
            kernel_regularizer: Regularizer for the kernel weights.
            bias_initializer: Initializer for the bias terms.
            bias_regularizer: Regularizer for the bias terms.
        """
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.bond_dim = bond_dim
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # Initialize weight attributes
        self.cores = None
        self.projection = None
        self.bias_weight = None

    def build(self, input_shape: Union[List[int], Tuple[int, ...]]) -> None:
        """Build the layer weights based on input shape.

        Args:
            input_shape: The shape of the input tensor.
        """
        input_dim = int(input_shape[-1])

        # Create MPS-inspired tensor cores
        self.cores = self.add_weight(
            name="mps_cores",
            shape=(input_dim, self.bond_dim, self.bond_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True
        )

        # Create projection weights
        self.projection = self.add_weight(
            name="projection",
            shape=(self.bond_dim, self.bond_dim, self.output_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True
        )

        if self.use_bias:
            self.bias_weight = self.add_weight(
                name="bias",
                shape=(self.output_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                trainable=True
            )

        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """Forward pass through the layer implementing MPS tensor contraction.

        This method performs a Matrix Product State (MPS) calculation with these steps:
        1. Initialize a boundary vector (analogous to an MPS boundary condition)
        2. Sequentially contract each input feature with its corresponding MPS core tensor
        3. This creates a chain of matrix multiplications that efficiently parameterizes
           correlations between input features
        4. Project the final contracted state to the output dimension

        The MPS approach has these computational advantages:
        - Reduces parameter count from O(n²) to O(n) compared to dense layers
        - Captures long-range correlations between input features
        - Maintains an efficient parameterization of the quantum-inspired "entanglement"

        Mathematically, for input x with features x_i, the operation is approximately:
        output = B * (A¹[x₁] * A²[x₂] * ... * Aⁿ[xₙ]) * P
        where:
        - B is the boundary vector
        - Aⁱ are the core tensors activated by input x_i
        - P is the projection matrix
        - * represents tensor contraction

        Args:
            inputs: Input tensor of shape [batch_size, input_dim].
            training: Whether the model is in training mode.

        Returns:
            Output tensor of shape [batch_size, output_dim].
        """
        # Create initial boundary vector (analogous to left boundary condition in MPS)
        batch_size = tf.shape(inputs)[0]
        boundary = tf.ones((batch_size, 1, self.bond_dim))  # Initialize with ones

        # Sequential contraction of MPS tensor cores with input features
        for i in range(inputs.shape[-1]):
            # Get input feature for position i
            x_i = tf.expand_dims(tf.expand_dims(inputs[:, i], -1), -1)  # Shape: [batch_size, 1, 1]

            # Select core for position i and broadcast to batch dimension
            core_i = self.cores[i:i + 1, :, :]  # Shape: [1, bond_dim, bond_dim]
            core_i = tf.tile(core_i, [batch_size, 1, 1])  # Shape: [batch_size, bond_dim, bond_dim]

            # Weight core by input feature (element-wise multiplication)
            weighted_core = x_i * core_i  # Shape: [batch_size, bond_dim, bond_dim]

            # Contract with boundary state (matrix multiplication)
            # This creates the sequential MPS chain: B * A¹ * A² * ... * Aⁿ
            boundary = tf.matmul(boundary, weighted_core)  # Shape: [batch_size, 1, bond_dim]

        # Project to output dimension
        # First reshape boundary to [batch_size, bond_dim, 1]
        boundary = tf.transpose(boundary, [0, 2, 1])

        # Contract with projection tensor via reshaping and matrix multiplication
        # projection shape: [bond_dim, bond_dim, output_dim]
        projection_reshaped = tf.reshape(self.projection, [self.bond_dim * self.bond_dim, self.output_dim])
        boundary_reshaped = tf.reshape(boundary, [batch_size, self.bond_dim * self.bond_dim])

        # Final matrix multiplication to get output
        output = tf.matmul(boundary_reshaped, projection_reshaped)

        # Add bias if configured
        if self.use_bias:
            output = output + self.bias_weight

        return output

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input tensor.

        Returns:
            Shape of the output tensor.
        """
        return (input_shape[0], self.output_dim)

    def get_config(self) -> Dict[str, Any]:
        """Get the layer configuration.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = {
            'output_dim': self.output_dim,
            'bond_dim': self.bond_dim,
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
        }
        base_config = super().get_config()
        return {**base_config, **config}
