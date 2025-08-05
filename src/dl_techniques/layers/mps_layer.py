"""
Matrix Product States (MPS) Layer Implementation
===============================================

Theory and Background:
---------------------
Matrix Product States are a representation method from quantum physics that efficiently
represents quantum systems with many particles. In machine learning terms, it's a way to
decompose high-dimensional data into a series of interconnected lower-dimensional tensors.
This can be seen as a specialized form of tensor network that:

- Reduces the parameter space needed to represent complex data
- Preserves important correlations between features
- Scales efficiently with input size
- Captures long-range dependencies through sequential tensor contractions

## MPS Tensor Contraction Process:
The MPS layer implements a sequential tensor contraction process:
    output = B * (A¹[x₁] * A²[x₂] * ... * Aⁿ[xₙ]) * P

where:
- B is the boundary vector (initialized to ones)
- Aⁱ are the core tensors activated by input features x_i
- P is the projection matrix to output space
- * represents tensor contraction operations

## Computational Advantages:
- Parameter count: O(n * bond_dim²) vs O(n²) for dense layers
- Captures long-range correlations between input features
- Efficient quantum-inspired "entanglement" parameterization
- Scalable to high-dimensional inputs

## References:
[1] Stoudenmire, E., & Schwab, D. J. (2016). "Supervised Learning with Tensor Networks"
[2] Novikov, A., et al. (2015). "Tensorizing Neural Networks"
[3] Cohen, N., & Shashua, A. (2016). "Inductive Bias of Deep Convolutional Networks"
"""

import keras
from keras import ops
from typing import Tuple, Optional, List, Union, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class MPSLayer(keras.layers.Layer):
    """
    Matrix Product State inspired layer for tensor decomposition.

    This layer implements a simplified version of MPS tensor contraction,
    which can capture long-range correlations while maintaining computational efficiency.
    The layer performs sequential tensor contractions that efficiently parameterize
    correlations between input features using quantum-inspired tensor networks.

    Args:
        output_dim: Dimension of the output tensor. Must be positive.
        bond_dim: Internal bond dimension for the MPS representation. Controls the
                 expressiveness of the tensor network. Higher values capture more
                 complex correlations but increase computational cost.
        use_bias: Whether to include bias terms in the final projection.
        kernel_initializer: Initializer for the MPS core tensors and projection weights.
        kernel_regularizer: Regularizer for the core tensors and projection weights.
        bias_initializer: Initializer for the bias terms.
        bias_regularizer: Regularizer for the bias terms.
        activity_regularizer: Regularizer for the layer output.
        **kwargs: Additional keyword arguments for the Layer parent class.

    Call arguments:
        inputs: Input tensor of shape `(batch_size, input_dim)`.
        training: Boolean indicating whether the layer should behave in training mode.

    Returns:
        output: Tensor of shape `(batch_size, output_dim)` containing the MPS-processed
               features with captured long-range correlations.

    Raises:
        ValueError: If output_dim <= 0 or bond_dim <= 0.

    Example:
        >>> # Basic usage
        >>> x = keras.random.normal((32, 128))
        >>> mps_layer = MPSLayer(output_dim=64, bond_dim=16)
        >>> output = mps_layer(x)
        >>> print(output.shape)
        (32, 64)

        >>> # With regularization
        >>> mps_layer = MPSLayer(
        ...     output_dim=32,
        ...     bond_dim=8,
        ...     kernel_regularizer="l2",
        ...     activity_regularizer="l1"
        ... )
        >>> output = mps_layer(x)
        >>> print(output.shape)
        (32, 32)
    """

    def __init__(
        self,
        output_dim: int,
        bond_dim: int = 16,
        use_bias: bool = True,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
        bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        activity_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate parameters
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if bond_dim <= 0:
            raise ValueError(f"bond_dim must be positive, got {bond_dim}")

        # Store configuration parameters
        self.output_dim = output_dim
        self.bond_dim = bond_dim
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = keras.regularizers.get(activity_regularizer)

        # Initialize weight attributes to None - will be created in build()
        self.cores = None
        self.projection = None
        self.bias_weight = None

        # Store build input shape for serialization
        self._build_input_shape = None

        logger.info(f"Initialized MPSLayer with output_dim={output_dim}, "
                   f"bond_dim={bond_dim}, use_bias={use_bias}")

    def build(self, input_shape: Union[List[int], Tuple[int, ...]]) -> None:
        """
        Build the layer weights based on input shape.

        Creates the MPS core tensors and projection weights. The core tensors
        represent the quantum-inspired tensor network that captures correlations
        between input features.

        Args:
            input_shape: The shape of the input tensor.
        """
        # Store input shape for serialization
        self._build_input_shape = input_shape

        input_dim = int(input_shape[-1])
        logger.debug(f"Building MPSLayer with input_dim={input_dim}")

        # Create MPS-inspired tensor cores
        # Shape: (input_dim, bond_dim, bond_dim)
        # Each input feature gets its own core tensor for contraction
        self.cores = self.add_weight(
            name="mps_cores",
            shape=(input_dim, self.bond_dim, self.bond_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True
        )

        # Create projection weights to map final MPS state to output
        # Shape: (bond_dim, output_dim)
        # This projects the final contracted MPS state to the desired output dimension
        self.projection = self.add_weight(
            name="projection",
            shape=(self.bond_dim, self.output_dim),
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
        logger.debug("MPSLayer build completed")

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass implementing MPS tensor contraction.

        This method performs a Matrix Product State (MPS) calculation with these steps:
        1. Initialize a boundary vector (analogous to an MPS boundary condition)
        2. Sequentially contract each input feature with its corresponding MPS core tensor
        3. This creates a chain of matrix multiplications that efficiently parameterizes
           correlations between input features
        4. Project the final contracted state to the output dimension

        The MPS approach has these computational advantages:
        - Reduces parameter count from O(n²) to O(n * bond_dim²) compared to dense layers
        - Captures long-range correlations between input features
        - Maintains an efficient parameterization of quantum-inspired "entanglement"

        Mathematically, for input x with features x_i, the operation is:
        output = B * (A¹[x₁] * A²[x₂] * ... * Aⁿ[xₙ]) * P
        where:
        - B is the boundary vector (initialized to ones)
        - Aⁱ are the core tensors activated by input x_i
        - P is the projection matrix
        - * represents tensor contraction

        Args:
            inputs: Input tensor of shape [batch_size, input_dim].
            training: Whether the model is in training mode.

        Returns:
            Output tensor of shape [batch_size, output_dim].
        """
        # Get input dimensions using Keras ops
        batch_size = ops.shape(inputs)[0]
        input_dim = ops.shape(inputs)[1]

        # Create initial boundary vector (analogous to left boundary condition in MPS)
        # Initialize with ones - this serves as the "left boundary" of the MPS chain
        boundary = ops.ones((batch_size, self.bond_dim))

        # Sequential contraction of MPS tensor cores with input features
        # This implements the core MPS computation: B * A¹[x₁] * A²[x₂] * ... * Aⁿ[xₙ]
        for i in range(input_dim):
            # Get input feature for position i
            # Shape: [batch_size, 1]
            x_i = ops.expand_dims(inputs[:, i], axis=-1)

            # Select core tensor for position i
            # Shape: [bond_dim, bond_dim]
            core_i = self.cores[i, :, :]

            # Weight the core tensor by the input feature value
            # This implements the "activation" of the core by the input
            # Broadcasting: [batch_size, 1] * [bond_dim, bond_dim] -> [batch_size, bond_dim, bond_dim]
            weighted_core = ops.expand_dims(x_i, axis=-1) * ops.expand_dims(core_i, axis=0)

            # Contract with the current boundary state
            # This performs: boundary * weighted_core
            # [batch_size, bond_dim] @ [batch_size, bond_dim, bond_dim] -> [batch_size, bond_dim]
            boundary = ops.sum(
                ops.expand_dims(boundary, axis=-1) * weighted_core,
                axis=1
            )

        # Project the final MPS state to output dimension
        # [batch_size, bond_dim] @ [bond_dim, output_dim] -> [batch_size, output_dim]
        output = ops.matmul(boundary, self.projection)

        # Add bias if configured
        if self.use_bias:
            output = output + self.bias_weight

        return output

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input tensor.

        Returns:
            Shape of the output tensor.
        """
        # Convert to list for manipulation
        input_shape_list = list(input_shape)

        # Output shape: same batch dimension, output_dim for features
        output_shape_list = input_shape_list[:-1] + [self.output_dim]

        return tuple(output_shape_list)

    def get_config(self) -> Dict[str, Any]:
        """
        Get the layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            'output_dim': self.output_dim,
            'bond_dim': self.bond_dim,
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': keras.regularizers.serialize(self.activity_regularizer),
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """
        Get the build configuration for serialization.

        Returns:
            Dictionary containing the build configuration.
        """
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """
        Build the layer from a configuration dictionary.

        Args:
            config: Dictionary containing the build configuration.
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

# ---------------------------------------------------------------------
