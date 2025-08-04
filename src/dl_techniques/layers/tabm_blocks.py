"""
This module provides a suite of custom Keras layers designed for building and
training deep ensembles in a highly efficient and memory-conscious manner.

Deep Ensembling is a powerful technique for improving model robustness, accuracy, and
uncertainty estimation. The standard approach involves training multiple independent
models and averaging their predictions. However, this can be computationally expensive
and slow. This module implements several building blocks for creating "implicit" or
"batched" ensembles, where all `k` ensemble members are represented and trained
simultaneously within a single model, using an additional dimension.

The key to this module's efficiency is its clever use of weight sharing and
specialized tensor operations (`einsum`) to perform parallel computations across all
ensemble members with minimal overhead.

The module contains the following key components:

1.  **`ScaleEnsemble` (Learnable Member Scaling):**
    -   A simple but powerful layer that applies a learnable, per-feature scaling
        factor to each ensemble member.
    -   This allows the model to learn the relative importance of different features
        for each individual member of the ensemble, providing a simple mechanism for
        each member to specialize.

2.  **`LinearEfficientEnsemble` (The Core Ensemble Layer):**
    -   This is the workhorse of the module. It implements an efficient linear
        transformation for `k` ensemble members.
    -   **Weight Sharing:** It uses a single, shared `kernel` (weight matrix) for the
        main linear projection across all ensemble members. This is a massive saving
        in parameters compared to having `k` separate weight matrices.
    -   **Rank-1 Perturbations:** To allow each ensemble member to learn a unique
        function, it applies learnable, rank-1 scaling factors (`r` for input, `s` for
        output) to the shared kernel. This is equivalent to applying a unique diagonal
        matrix transformation to the input and output for each member, providing
        diversity without the full cost of independent weights.

3.  **`NLinear` (Fully Independent Parallel Layers):**
    -   This layer provides an alternative to the efficient, weight-sharing approach.
        It implements `n` truly independent linear layers that are processed in parallel.
    -   It uses a single weight tensor of shape `(n, input_dim, output_dim)` and
        `einsum` to perform `n` independent matrix multiplications in one operation.
    -   This is useful for the final output heads of an ensemble, where each member needs
        its own independent classifier.

4.  **`MLPBlock` and `TabMBackbone` (High-Level Abstractions):**
    -   These are convenience layers that assemble the lower-level components into
        standard MLP blocks and a full MLP backbone.
    -   They can operate in either a "plain" mode (a single model) or an "ensemble"
        mode (with `k` members) by simply setting the `k` parameter, making it easy
        to switch between standard and ensemble architectures.
"""

import keras
from keras import ops
from typing import Dict, List, Literal, Optional, Tuple, Union, Any

# ---------------------------------------------------------------------

class ScaleEnsemble(keras.layers.Layer):
    """Enhanced ensemble adapter with learnable scaling weights.

    This layer implements efficient learnable scaling for ensemble members,
    inspired by the project's scaling layer patterns with improved initialization
    and regularization support.

    Args:
        k: Number of ensemble members.
        input_dim: Input feature dimension.
        init_distribution: Initialization distribution ('normal' or 'random-signs').
        kernel_initializer: Initializer for the scaling weights.
        kernel_regularizer: Optional regularizer for scaling weights.
        **kwargs: Additional layer arguments.
    """

    def __init__(
            self,
            k: int,
            input_dim: int,
            init_distribution: Literal['normal', 'random-signs'] = 'normal',
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'ones',
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.k = k
        self.input_dim = input_dim
        self.init_distribution = init_distribution
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        # Adjust initializer based on distribution type
        if init_distribution == 'random-signs':
            self.kernel_initializer = keras.initializers.RandomUniform(
                minval=-1.0, maxval=1.0, seed=None
            )
        elif init_distribution == 'normal':
            self.kernel_initializer = keras.initializers.RandomNormal(
                mean=0.0, stddev=0.1, seed=None
            )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the scaling weights with proper initialization."""
        self.weight = self.add_weight(
            shape=(self.k, self.input_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
            name='ensemble_weight'
        )
        super().build(input_shape)

    def call(self, inputs: Any) -> Any:
        """Apply ensemble scaling to inputs.

        Args:
            inputs: Input tensor of shape (batch_size, k, input_dim).

        Returns:
            Scaled tensor of shape (batch_size, k, input_dim).
        """
        # Efficient broadcasting: inputs (B, K, D) * weight (K, D) -> (B, K, D)
        return ops.multiply(inputs, ops.expand_dims(self.weight, axis=0))

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            "k": self.k,
            "input_dim": self.input_dim,
            "init_distribution": self.init_distribution,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config

# ---------------------------------------------------------------------

class LinearEfficientEnsemble(keras.layers.Layer):
    """Efficient ensemble linear layer with separate input/output scaling.

    This layer implements efficient ensemble linear transformations with optional
    input and output scaling, following the project's patterns for robust layer design.

    Args:
        units: Output dimension.
        k: Number of ensemble members.
        use_bias: Whether to use bias.
        ensemble_scaling_in: Whether to use input scaling.
        ensemble_scaling_out: Whether to use output scaling.
        kernel_initializer: Initializer for the main weights.
        bias_initializer: Initializer for bias.
        kernel_regularizer: Optional regularizer for kernel weights.
        bias_regularizer: Optional regularizer for bias weights.
        **kwargs: Additional layer arguments.
    """

    def __init__(
            self,
            units: int,
            k: int,
            use_bias: bool = True,
            ensemble_scaling_in: bool = True,
            ensemble_scaling_out: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.units = units
        self.k = k
        self.use_bias = use_bias
        self.ensemble_scaling_in = ensemble_scaling_in
        self.ensemble_scaling_out = ensemble_scaling_out
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the ensemble linear layer weights."""
        input_dim = input_shape[-1]

        # Main weight matrix shared across ensemble members
        self.kernel = self.add_weight(
            shape=(input_dim, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
            name='kernel'
        )

        # Input scaling weights
        if self.ensemble_scaling_in:
            self.r = self.add_weight(
                shape=(self.k, input_dim),
                initializer='ones',
                trainable=True,
                name='input_scaling'
            )

        # Output scaling weights
        if self.ensemble_scaling_out:
            self.s = self.add_weight(
                shape=(self.k, self.units),
                initializer='ones',
                trainable=True,
                name='output_scaling'
            )

        # Bias weights
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.k, self.units),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                trainable=True,
                name='bias'
            )

        super().build(input_shape)

    def call(self, inputs: Any) -> Any:
        """Forward pass through efficient ensemble layer.

        Args:
            inputs: Input tensor of shape (batch_size, k, input_dim).

        Returns:
            Output tensor of shape (batch_size, k, units).
        """
        x = inputs

        # Apply input scaling if enabled
        if self.ensemble_scaling_in:
            x = ops.multiply(x, ops.expand_dims(self.r, axis=0))

        # Apply main linear transformation efficiently
        # Use einsum for better performance and clarity
        x = ops.einsum('bki,iu->bku', x, self.kernel)

        # Apply output scaling if enabled
        if self.ensemble_scaling_out:
            x = ops.multiply(x, ops.expand_dims(self.s, axis=0))

        # Add bias if enabled
        if self.use_bias:
            x = ops.add(x, ops.expand_dims(self.bias, axis=0))

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], int, int]:
        """Compute the output shape of the layer."""
        return (input_shape[0], input_shape[1], self.units)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            "units": self.units,
            "k": self.k,
            "use_bias": self.use_bias,
            "ensemble_scaling_in": self.ensemble_scaling_in,
            "ensemble_scaling_out": self.ensemble_scaling_out,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

# ---------------------------------------------------------------------

class NLinear(keras.layers.Layer):
    """N parallel linear layers for ensemble output with enhanced efficiency.

    This layer implements efficient parallel linear transformations for ensemble
    outputs using optimized tensor operations.

    Args:
        n: Number of parallel linear layers.
        input_dim: Input dimension.
        output_dim: Output dimension per linear layer.
        use_bias: Whether to use bias.
        kernel_initializer: Initializer for weights.
        bias_initializer: Initializer for bias.
        kernel_regularizer: Optional regularizer for kernel weights.
        bias_regularizer: Optional regularizer for bias weights.
        **kwargs: Additional layer arguments.
    """

    def __init__(
            self,
            n: int,
            input_dim: int,
            output_dim: int,
            use_bias: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.n = n
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the parallel linear layer weights."""
        self.kernels = self.add_weight(
            shape=(self.n, self.input_dim, self.output_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
            name='kernels'
        )

        if self.use_bias:
            self.biases = self.add_weight(
                shape=(self.n, self.output_dim),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                trainable=True,
                name='biases'
            )

        super().build(input_shape)

    def call(self, inputs: Any) -> Any:
        """Forward pass through N parallel linear layers.

        Args:
            inputs: Input tensor of shape (batch_size, n, input_dim).

        Returns:
            Output tensor of shape (batch_size, n, output_dim).
        """
        # Efficient parallel matrix multiplication using einsum
        outputs = ops.einsum('bni,nio->bno', inputs, self.kernels)

        if self.use_bias:
            outputs = ops.add(outputs, ops.expand_dims(self.biases, axis=0))

        return outputs

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], int, int]:
        """Compute the output shape of the layer."""
        return (input_shape[0], self.n, self.output_dim)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            "n": self.n,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

# ---------------------------------------------------------------------

class MLPBlock(keras.layers.Layer):
    """MLP block with efficient ensemble support and enhanced configurability.

    This layer implements a single MLP block that can work in both plain and ensemble
    modes with proper regularization and initialization support.

    Args:
        units: Number of units in the hidden layer.
        k: Number of ensemble members (None for plain MLP).
        activation: Activation function.
        dropout_rate: Dropout rate.
        use_bias: Whether to use bias.
        kernel_initializer: Initializer for weights.
        bias_initializer: Initializer for bias.
        kernel_regularizer: Optional regularizer for kernel weights.
        bias_regularizer: Optional regularizer for bias weights.
        **kwargs: Additional layer arguments.
    """

    def __init__(
            self,
            units: int,
            k: Optional[int] = None,
            activation: str = 'relu',
            dropout_rate: float = 0.0,
            use_bias: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.units = units
        self.k = k
        self.activation = keras.activations.get(activation)
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # Layers will be built in build()
        self.linear = None
        self.dropout = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the MLP block layers."""
        if self.k is None:
            # Plain linear layer
            self.linear = keras.layers.Dense(
                self.units,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name='linear'
            )
        else:
            # Efficient ensemble layer
            self.linear = LinearEfficientEnsemble(
                self.units,
                self.k,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name='linear'
            )

        if self.dropout_rate > 0:
            self.dropout = keras.layers.Dropout(self.dropout_rate)

        super().build(input_shape)

    def call(self, inputs: Any, training: Optional[bool] = None) -> Any:
        """Forward pass through MLP block.

        Args:
            inputs: Input tensor.
            training: Training mode flag.

        Returns:
            Output tensor after linear transformation, activation, and dropout.
        """
        x = self.linear(inputs)
        x = self.activation(x)

        if self.dropout_rate > 0 and self.dropout is not None:
            x = self.dropout(x, training=training)

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer."""
        if self.k is None:
            return (input_shape[0], self.units)
        else:
            return (input_shape[0], self.k, self.units)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            "units": self.units,
            "k": self.k,
            "activation": keras.activations.serialize(self.activation),
            "dropout_rate": self.dropout_rate,
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

# ---------------------------------------------------------------------

class TabMBackbone(keras.layers.Layer):
    """TabM backbone MLP with ensemble support and proper layer management.

    This layer implements the core TabM backbone using a stack of MLP blocks
    with proper weight sharing and ensemble support.

    Args:
        hidden_dims: List of hidden layer dimensions.
        k: Number of ensemble members (None for plain MLP).
        activation: Activation function.
        dropout_rate: Dropout rate.
        use_bias: Whether to use bias.
        kernel_initializer: Initializer for weights.
        bias_initializer: Initializer for bias.
        kernel_regularizer: Optional regularizer for kernel weights.
        bias_regularizer: Optional regularizer for bias weights.
        **kwargs: Additional layer arguments.
    """

    def __init__(
            self,
            hidden_dims: List[int],
            k: Optional[int] = None,
            activation: str = 'relu',
            dropout_rate: float = 0.0,
            use_bias: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.hidden_dims = hidden_dims
        self.k = k
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # Blocks will be built in build()
        self.blocks = []

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the backbone MLP blocks."""
        self.blocks = []
        for i, units in enumerate(self.hidden_dims):
            block = MLPBlock(
                units=units,
                k=self.k,
                activation=self.activation,
                dropout_rate=self.dropout_rate,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name=f'block_{i}'
            )
            self.blocks.append(block)

        super().build(input_shape)

    def call(self, inputs: Any, training: Optional[bool] = None) -> Any:
        """Forward pass through backbone MLP.

        Args:
            inputs: Input tensor.
            training: Training mode flag.

        Returns:
            Output tensor after passing through all MLP blocks.
        """
        x = inputs
        for block in self.blocks:
            x = block(x, training=training)
        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer."""
        shape = input_shape
        for block in self.blocks:
            shape = block.compute_output_shape(shape)
        return shape

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            "hidden_dims": self.hidden_dims,
            "k": self.k,
            "activation": self.activation,
            "dropout_rate": self.dropout_rate,
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

# ---------------------------------------------------------------------

