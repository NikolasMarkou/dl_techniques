"""
Matrix Product States (MPS) Layer Implementation

This module implements a Matrix Product State-inspired tensor decomposition layer
that efficiently captures long-range correlations in input data using quantum-inspired
tensor networks.
"""

import keras
from typing import Tuple, Optional, Union, Dict, Any

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class MPSLayer(keras.layers.Layer):
    """
    Matrix Product State inspired layer for tensor decomposition.

    This layer implements a quantum-inspired tensor network that efficiently
    parameterizes correlations between input features using Matrix Product States (MPS).
    The approach decomposes high-dimensional data into a series of interconnected
    lower-dimensional tensors, capturing long-range dependencies through sequential
    tensor contractions.

    Theory and Background:
        Matrix Product States are a representation method from quantum physics that
        efficiently represents quantum systems with many particles. In machine learning,
        this translates to a method for decomposing high-dimensional data that:
        - Reduces parameter space from O(n²) to O(n × bond_dim²)
        - Preserves important correlations between features
        - Scales efficiently with input size
        - Captures long-range dependencies through sequential contractions

    Mathematical Formulation:
        The MPS layer performs sequential tensor contraction:
        ```
        output = B × (A¹[x₁] × A²[x₂] × ... × Aⁿ[xₙ]) × P
        ```
        Where:
        - B is the boundary vector (initialized to ones)
        - Aⁱ are the core tensors activated by input features x_i
        - P is the projection matrix to output space
        - × represents tensor contraction operations

    Computational Advantages:
        - Parameter efficiency: O(n × bond_dim²) vs O(n²) for dense layers
        - Long-range correlation capture between distant input features
        - Quantum-inspired "entanglement" parameterization
        - Scalable to high-dimensional inputs

    Args:
        output_dim: Integer, dimension of the output tensor. Must be positive.
        bond_dim: Integer, internal bond dimension for the MPS representation.
            Controls the expressiveness of the tensor network. Higher values capture
            more complex correlations but increase computational cost. Defaults to 16.
        use_bias: Boolean, whether to include bias terms in the final projection.
            Defaults to True.
        kernel_initializer: String or initializer instance, initializer for the MPS
            core tensors and projection weights. Defaults to 'glorot_uniform'.
        kernel_regularizer: String or regularizer instance, regularizer for the
            core tensors and projection weights. Defaults to None.
        bias_initializer: String or initializer instance, initializer for bias terms.
            Defaults to 'zeros'.
        bias_regularizer: String or regularizer instance, regularizer for bias terms.
            Defaults to None.
        activity_regularizer: String or regularizer instance, regularizer for the
            layer output. Defaults to None.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        2D tensor with shape: `(batch_size, input_dim)`

    Output shape:
        2D tensor with shape: `(batch_size, output_dim)`

    Attributes:
        cores: Core tensor weights of shape (input_dim, bond_dim, bond_dim).
        projection: Projection matrix of shape (bond_dim, output_dim).
        bias_weight: Bias vector of shape (output_dim,) if use_bias=True.

    Example:
        ```python
        # Basic usage
        inputs = keras.Input(shape=(128,))
        mps_layer = MPSLayer(output_dim=64, bond_dim=16)
        outputs = mps_layer(inputs)

        # Advanced configuration with regularization
        mps_layer = MPSLayer(
            output_dim=32,
            bond_dim=8,
            kernel_regularizer=keras.regularizers.L2(1e-4),
            activity_regularizer=keras.regularizers.L1(1e-5),
            kernel_initializer='he_normal'
        )

        # In a complete model
        model = keras.Sequential([
            keras.layers.Dense(256, activation='relu'),
            MPSLayer(output_dim=64, bond_dim=12),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(10, activation='softmax')
        ])

        # Comparing parameter efficiency
        dense_layer = keras.layers.Dense(64)  # Parameters: input_dim * 64
        mps_layer = MPSLayer(64, bond_dim=8)  # Parameters: input_dim * 8² + 8 * 64
        ```

    References:
        - Stoudenmire, E., & Schwab, D. J. (2016). "Supervised Learning with Tensor Networks"
        - Novikov, A., et al. (2015). "Tensorizing Neural Networks"
        - Cohen, N., & Shashua, A. (2016). "Inductive Bias of Deep Convolutional Networks"

    Raises:
        ValueError: If output_dim or bond_dim are not positive integers.

    Note:
        This layer is most effective when the input features have some inherent
        sequential or spatial structure, as the MPS representation assumes a
        chain-like correlation structure between features.
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
        """
        Initialize the MPSLayer.

        Args:
            output_dim: Output dimension, must be positive.
            bond_dim: Bond dimension for MPS representation, must be positive.
            use_bias: Whether to use bias in final projection.
            kernel_initializer: Initializer for core tensors and projection.
            kernel_regularizer: Regularizer for core tensors and projection.
            bias_initializer: Initializer for bias terms.
            bias_regularizer: Regularizer for bias terms.
            activity_regularizer: Regularizer for layer output.
            **kwargs: Additional Layer arguments.

        Raises:
            ValueError: If output_dim or bond_dim are not positive.
        """
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

        # Initialize weight attributes - will be created in build()
        self.cores = None
        self.projection = None
        self.bias_weight = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer weights based on input shape.

        Creates the MPS core tensors and projection weights. The core tensors
        represent the quantum-inspired tensor network that captures correlations
        between input features.

        Args:
            input_shape: Shape tuple of the input tensor.

        Raises:
            ValueError: If input_shape is invalid or input_dim cannot be determined.
        """
        if len(input_shape) < 2:
            raise ValueError(
                f"MPSLayer requires input with at least 2 dimensions, "
                f"got shape {input_shape}"
            )

        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError(
                "Last dimension of input must be defined (not None). "
                f"Got input_shape: {input_shape}"
            )

        input_dim = int(input_dim)

        # Create MPS core tensors
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
        self.projection = self.add_weight(
            name="projection",
            shape=(self.bond_dim, self.output_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True
        )

        # Create bias weights if requested
        if self.use_bias:
            self.bias_weight = self.add_weight(
                name="bias",
                shape=(self.output_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                trainable=True
            )

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass implementing MPS tensor contraction.

        Performs Matrix Product State calculation through sequential tensor
        contractions that efficiently parameterize correlations between input features.

        The computation follows these steps:
        1. Initialize boundary vector (analogous to MPS boundary condition)
        2. Sequentially contract each input feature with its corresponding core tensor
        3. Create chain of matrix multiplications capturing feature correlations
        4. Project final contracted state to output dimension

        Mathematical operation:
        ```
        output = B × (A¹[x₁] × A²[x₂] × ... × Aⁿ[xₙ]) × P
        ```

        Args:
            inputs: Input tensor of shape (batch_size, input_dim).
            training: Boolean indicating training mode (unused but kept for consistency).

        Returns:
            keras.KerasTensor: Output tensor of shape (batch_size, output_dim)
                              containing MPS-processed features with captured
                              long-range correlations.
        """
        # Get input dimensions
        batch_size = keras.ops.shape(inputs)[0]
        input_dim = keras.ops.shape(inputs)[1]

        # Initialize boundary vector (left boundary condition of MPS chain)
        # Shape: (batch_size, bond_dim)
        boundary = keras.ops.ones((batch_size, self.bond_dim))

        # Sequential contraction of MPS tensor cores with input features
        # This implements: B × A¹[x₁] × A²[x₂] × ... × Aⁿ[xₙ]
        for i in range(input_dim):
            # Extract input feature for position i
            # Shape: (batch_size,) -> (batch_size, 1)
            x_i = keras.ops.expand_dims(inputs[:, i], axis=-1)

            # Select core tensor for position i
            # Shape: (bond_dim, bond_dim)
            core_i = self.cores[i, :, :]

            # Weight core tensor by input feature value
            # Broadcasting: (batch_size, 1) × (bond_dim, bond_dim)
            # -> (batch_size, bond_dim, bond_dim)
            weighted_core = keras.ops.expand_dims(x_i, axis=-1) * keras.ops.expand_dims(core_i, axis=0)

            # Contract with current boundary state
            # (batch_size, bond_dim) × (batch_size, bond_dim, bond_dim)
            # -> (batch_size, bond_dim)
            boundary = keras.ops.sum(
                keras.ops.expand_dims(boundary, axis=-1) * weighted_core,
                axis=1
            )

        # Project final MPS state to output dimension
        # (batch_size, bond_dim) @ (bond_dim, output_dim) -> (batch_size, output_dim)
        output = keras.ops.matmul(boundary, self.projection)

        # Add bias if configured
        if self.use_bias:
            output = keras.ops.add(output, self.bias_weight)

        return output

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Shape tuple of the input tensor.

        Returns:
            Tuple representing the output shape.
        """
        # Convert to list for manipulation, then back to tuple
        output_shape = list(input_shape)
        output_shape[-1] = self.output_dim
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        Returns:
            Dictionary containing all configuration parameters needed
            to reconstruct the layer.
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

# ---------------------------------------------------------------------

