"""
TabM (Tabular Model) Implementation for Keras 3.x

This module implements the TabM architecture for tabular deep learning,
including various ensemble mechanisms and feature processing strategies.
"""

import keras
import numpy as np
from keras import ops
from typing import Dict, List, Literal, Optional, Tuple, Union, Any

from dl_techniques.utils.logger import logger


class OneHotEncoding(keras.layers.Layer):
    """One-hot encoding layer for categorical features with enhanced efficiency.

    This layer efficiently converts categorical features to one-hot encoded representations
    using vectorized operations and proper memory management.

    Args:
        cardinalities: List of cardinalities for each categorical feature.
        **kwargs: Additional layer arguments.
    """

    def __init__(
            self,
            cardinalities: List[int],
            **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.cardinalities = cardinalities
        self.total_dim = sum(cardinalities)
        self.cumulative_cardinalities = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer by computing cumulative cardinalities for efficient indexing."""
        if self.cardinalities:
            # Precompute cumulative cardinalities for efficient slicing
            self.cumulative_cardinalities = [0]
            for card in self.cardinalities:
                self.cumulative_cardinalities.append(self.cumulative_cardinalities[-1] + card)
        super().build(input_shape)

    def call(self, inputs: Any) -> Any:
        """Apply one-hot encoding to categorical inputs.

        Args:
            inputs: Categorical input tensor of shape (batch_size, n_cat_features).

        Returns:
            One-hot encoded tensor of shape (batch_size, total_categorical_dim).
        """
        if len(self.cardinalities) == 0:
            batch_size = ops.shape(inputs)[0]
            return ops.zeros((batch_size, 0), dtype=self.compute_dtype)

        # Convert to int32 for one_hot operation
        inputs_int = ops.cast(inputs, "int32")

        outputs = []
        for i, cardinality in enumerate(self.cardinalities):
            # Extract the i-th categorical feature efficiently
            cat_feature = inputs_int[:, i]

            # One-hot encode with proper dtype
            one_hot = ops.one_hot(
                cat_feature,
                cardinality,
                dtype=self.compute_dtype
            )
            outputs.append(one_hot)

        # Concatenate all one-hot encodings efficiently
        if outputs:
            return ops.concatenate(outputs, axis=-1)
        else:
            batch_size = ops.shape(inputs)[0]
            return ops.zeros((batch_size, 0), dtype=self.compute_dtype)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], int]:
        """Compute the output shape of the layer."""
        return (input_shape[0], self.total_dim)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            "cardinalities": self.cardinalities,
        })
        return config


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


class TabMModel(keras.Model):
    """TabM (Tabular Model) implementation with various ensemble architectures.

    This model supports multiple architectures including:
    - Plain MLP
    - TabM with efficient ensemble
    - TabM-mini with minimal ensemble adapter
    - TabM-packed with packed ensemble

    Args:
        n_num_features: Number of numerical features.
        cat_cardinalities: List of cardinalities for categorical features.
        n_classes: Number of output classes (None for regression).
        hidden_dims: List of hidden layer dimensions.
        arch_type: Architecture type ('plain', 'tabm', 'tabm-mini', 'tabm-packed', etc.).
        k: Number of ensemble members (required for ensemble variants).
        activation: Activation function.
        dropout_rate: Dropout rate.
        use_bias: Whether to use bias.
        share_training_batches: Whether to share training batches across ensemble members.
        kernel_initializer: Initializer for weights.
        bias_initializer: Initializer for bias.
        kernel_regularizer: Optional regularizer for kernel weights.
        bias_regularizer: Optional regularizer for bias weights.
        **kwargs: Additional model arguments.
    """

    def __init__(
            self,
            n_num_features: int,
            cat_cardinalities: List[int],
            n_classes: Optional[int],
            hidden_dims: List[int],
            arch_type: Literal[
                'plain', 'tabm', 'tabm-mini', 'tabm-packed',
                'tabm-normal', 'tabm-mini-normal'
            ] = 'plain',
            k: Optional[int] = None,
            activation: str = 'relu',
            dropout_rate: float = 0.0,
            use_bias: bool = True,
            share_training_batches: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)

        # Validate arguments
        assert n_num_features >= 0
        assert n_num_features or cat_cardinalities

        if arch_type == 'plain':
            assert k is None
            assert share_training_batches, 'Plain architecture must use share_training_batches=True'
        else:
            assert k is not None
            assert k > 0

        self.n_num_features = n_num_features
        self.cat_cardinalities = cat_cardinalities
        self.n_classes = n_classes
        self.hidden_dims = hidden_dims
        self.arch_type = arch_type
        self.k = k
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.share_training_batches = share_training_batches
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # Calculate input dimensions
        self.d_num = n_num_features
        self.d_cat = sum(cat_cardinalities)
        self.d_flat = self.d_num + self.d_cat

        # Build layers
        self._build_layers()

    def _build_layers(self) -> None:
        """Build all model layers with proper initialization."""

        # Categorical encoding
        if self.cat_cardinalities:
            self.cat_encoder = OneHotEncoding(self.cat_cardinalities)
        else:
            self.cat_encoder = None

        # Minimal ensemble adapter for tabm-mini variants
        if self.arch_type in ('tabm-mini', 'tabm-mini-normal'):
            init_distribution = 'normal' if self.arch_type == 'tabm-mini-normal' else 'random-signs'
            self.minimal_ensemble_adapter = ScaleEnsemble(
                k=self.k,
                input_dim=self.d_flat,
                init_distribution=init_distribution,
                kernel_regularizer=self.kernel_regularizer
            )
        else:
            self.minimal_ensemble_adapter = None

        # Backbone MLP
        backbone_k = None if self.arch_type == 'plain' else self.k
        self.backbone = TabMBackbone(
            hidden_dims=self.hidden_dims,
            k=backbone_k,
            activation=self.activation,
            dropout_rate=self.dropout_rate,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer
        )

        # Output layer
        d_out = 1 if self.n_classes is None else self.n_classes

        if self.arch_type == 'plain':
            self.output_layer = keras.layers.Dense(
                d_out,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name='output'
            )
        else:
            self.output_layer = NLinear(
                n=self.k,
                input_dim=self.hidden_dims[-1],
                output_dim=d_out,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer
            )

    def call(
            self,
            inputs: Union[Tuple[Any, Any], Dict[str, Any]],
            training: Optional[bool] = None
    ) -> Any:
        """Forward pass through the TabM model.

        Args:
            inputs: Input data, either as tuple (x_num, x_cat) or dict with keys 'x_num', 'x_cat'.
            training: Training mode flag.

        Returns:
            Model predictions with shape:
            - Plain: (batch_size, 1, n_classes_or_1)
            - Ensemble: (batch_size, k, n_classes_or_1)
        """
        # Handle different input formats
        if isinstance(inputs, dict):
            x_num = inputs.get('x_num')
            x_cat = inputs.get('x_cat')
        elif isinstance(inputs, (tuple, list)) and len(inputs) == 2:
            x_num, x_cat = inputs
        else:
            # Single tensor input - assume all numerical
            x_num = inputs
            x_cat = None

        # Process features
        features = []

        # Numerical features
        if x_num is not None and self.n_num_features > 0:
            features.append(x_num)

        # Categorical features
        if x_cat is not None and self.cat_cardinalities:
            cat_encoded = self.cat_encoder(x_cat)
            features.append(cat_encoded)

        # Combine features efficiently
        if len(features) == 0:
            raise ValueError("No valid features provided")
        elif len(features) == 1:
            x = features[0]
        else:
            x = ops.concatenate(features, axis=-1)

        # Handle ensemble dimensions
        if self.k is not None:
            batch_size = ops.shape(x)[0]

            if self.share_training_batches or not training:
                # (B, D) -> (B, K, D) using efficient operations
                x = ops.expand_dims(x, axis=1)  # (B, 1, D)
                x = ops.tile(x, [1, self.k, 1])  # (B, K, D)
            else:
                # (B * K, D) -> (B, K, D)
                # Note: In practice, this requires careful batch preparation
                x = ops.reshape(x, (batch_size // self.k, self.k, -1))

            # Apply minimal ensemble adapter if present
            if self.minimal_ensemble_adapter is not None:
                x = self.minimal_ensemble_adapter(x)

        # Backbone forward pass
        x = self.backbone(x, training=training)

        # Output layer
        x = self.output_layer(x)

        # Adjust output shape for compatibility
        if self.k is None:
            # (B, D) -> (B, 1, D) for consistency with ensemble outputs
            x = ops.expand_dims(x, axis=1)

        return x

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        config = super().get_config()
        config.update({
            "n_num_features": self.n_num_features,
            "cat_cardinalities": self.cat_cardinalities,
            "n_classes": self.n_classes,
            "hidden_dims": self.hidden_dims,
            "arch_type": self.arch_type,
            "k": self.k,
            "activation": self.activation,
            "dropout_rate": self.dropout_rate,
            "use_bias": self.use_bias,
            "share_training_batches": self.share_training_batches,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'TabMModel':
        """Create model from configuration."""
        return cls(**config)


# Factory functions remain unchanged but with enhanced type hints
def create_tabm_model(
        n_num_features: int,
        cat_cardinalities: List[int],
        n_classes: Optional[int],
        hidden_dims: List[int] = [256, 256],
        arch_type: Literal[
            'plain', 'tabm', 'tabm-mini', 'tabm-packed',
            'tabm-normal', 'tabm-mini-normal'
        ] = 'tabm',
        k: Optional[int] = 8,
        activation: str = 'relu',
        dropout_rate: float = 0.0,
        use_bias: bool = True,
        share_training_batches: bool = True,
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
        **kwargs
) -> TabMModel:
    """Create a TabM model with specified configuration.

    Args:
        n_num_features: Number of numerical features.
        cat_cardinalities: List of cardinalities for categorical features.
        n_classes: Number of output classes (None for regression).
        hidden_dims: List of hidden layer dimensions.
        arch_type: Architecture type.
        k: Number of ensemble members.
        activation: Activation function.
        dropout_rate: Dropout rate.
        use_bias: Whether to use bias.
        share_training_batches: Whether to share training batches across ensemble members.
        kernel_initializer: Initializer for weights.
        bias_initializer: Initializer for bias.
        **kwargs: Additional model arguments.

    Returns:
        Configured TabM model.

    Example:
        >>> # Binary classification with numerical and categorical features
        >>> model = create_tabm_model(
        ...     n_num_features=10,
        ...     cat_cardinalities=[5, 3, 8],
        ...     n_classes=2,
        ...     arch_type='tabm',
        ...     k=8
        ... )

        >>> # Regression with only numerical features
        >>> model = create_tabm_model(
        ...     n_num_features=15,
        ...     cat_cardinalities=[],
        ...     n_classes=None,
        ...     arch_type='tabm-mini',
        ...     k=4
        ... )
    """
    return TabMModel(
        n_num_features=n_num_features,
        cat_cardinalities=cat_cardinalities,
        n_classes=n_classes,
        hidden_dims=hidden_dims,
        arch_type=arch_type,
        k=k,
        activation=activation,
        dropout_rate=dropout_rate,
        use_bias=use_bias,
        share_training_batches=share_training_batches,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        **kwargs
    )


def create_tabm_plain(
        n_num_features: int,
        cat_cardinalities: List[int],
        n_classes: Optional[int],
        hidden_dims: List[int] = [256, 256],
        **kwargs
) -> TabMModel:
    """Create a plain MLP without ensembling.

    Args:
        n_num_features: Number of numerical features.
        cat_cardinalities: List of cardinalities for categorical features.
        n_classes: Number of output classes (None for regression).
        hidden_dims: List of hidden layer dimensions.
        **kwargs: Additional model arguments.

    Returns:
        Plain MLP model.
    """
    return create_tabm_model(
        n_num_features=n_num_features,
        cat_cardinalities=cat_cardinalities,
        n_classes=n_classes,
        hidden_dims=hidden_dims,
        arch_type='plain',
        k=None,
        **kwargs
    )


def create_tabm_ensemble(
        n_num_features: int,
        cat_cardinalities: List[int],
        n_classes: Optional[int],
        k: int = 8,
        hidden_dims: List[int] = [256, 256],
        **kwargs
) -> TabMModel:
    """Create a TabM model with efficient ensemble.

    Args:
        n_num_features: Number of numerical features.
        cat_cardinalities: List of cardinalities for categorical features.
        n_classes: Number of output classes (None for regression).
        k: Number of ensemble members.
        hidden_dims: List of hidden layer dimensions.
        **kwargs: Additional model arguments.

    Returns:
        TabM ensemble model.
    """
    return create_tabm_model(
        n_num_features=n_num_features,
        cat_cardinalities=cat_cardinalities,
        n_classes=n_classes,
        hidden_dims=hidden_dims,
        arch_type='tabm',
        k=k,
        **kwargs
    )


def create_tabm_mini(
        n_num_features: int,
        cat_cardinalities: List[int],
        n_classes: Optional[int],
        k: int = 8,
        hidden_dims: List[int] = [256, 256],
        **kwargs
) -> TabMModel:
    """Create a TabM-mini model with minimal ensemble adapter.

    Args:
        n_num_features: Number of numerical features.
        cat_cardinalities: List of cardinalities for categorical features.
        n_classes: Number of output classes (None for regression).
        k: Number of ensemble members.
        hidden_dims: List of hidden layer dimensions.
        **kwargs: Additional model arguments.

    Returns:
        TabM-mini model.
    """
    return create_tabm_model(
        n_num_features=n_num_features,
        cat_cardinalities=cat_cardinalities,
        n_classes=n_classes,
        hidden_dims=hidden_dims,
        arch_type='tabm-mini',
        k=k,
        **kwargs
    )


class TabMLoss(keras.losses.Loss):
    """Custom loss for TabM ensemble training with enhanced efficiency.

    Args:
        base_loss: Base loss function to use.
        share_training_batches: Whether batches are shared across ensemble members.
        name: Loss name.
    """

    def __init__(
            self,
            base_loss: Union[str, keras.losses.Loss] = 'mse',
            share_training_batches: bool = True,
            name: str = 'tabm_loss',
            **kwargs
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.base_loss = keras.losses.get(base_loss)
        self.share_training_batches = share_training_batches

    def call(self, y_true: Any, y_pred: Any) -> Any:
        """Compute loss for TabM ensemble predictions.

        Args:
            y_true: True labels with shape (batch_size,) or (batch_size, n_classes).
            y_pred: Ensemble predictions with shape (batch_size, k, n_outputs).

        Returns:
            Computed loss value.
        """
        # Get shapes for efficient operations
        batch_size = ops.shape(y_pred)[0]
        k = ops.shape(y_pred)[1]
        n_outputs = ops.shape(y_pred)[-1]

        # Flatten ensemble predictions: (batch_size, k, n_outputs) -> (batch_size * k, n_outputs)
        y_pred_flat = ops.reshape(y_pred, (batch_size * k, n_outputs))

        if self.share_training_batches:
            # Repeat true labels for each ensemble member efficiently
            if len(ops.shape(y_true)) == 1:
                # For 1D labels: (batch_size,) -> (batch_size * k,)
                y_true_expanded = ops.repeat(y_true, k, axis=0)
            else:
                # For multi-dimensional labels: (batch_size, n_classes) -> (batch_size * k, n_classes)
                y_true_expanded = ops.repeat(y_true, k, axis=0)
        else:
            # Labels are already arranged for each ensemble member
            y_true_expanded = y_true

        return self.base_loss(y_true_expanded, y_pred_flat)

    def get_config(self) -> Dict[str, Any]:
        """Get loss configuration for serialization."""
        config = super().get_config()
        config.update({
            'base_loss': keras.losses.serialize(self.base_loss),
            'share_training_batches': self.share_training_batches,
        })
        return config


def ensemble_predict(
        model: TabMModel,
        x_data: Union[Tuple, Dict, Any],
        method: Literal['mean', 'best', 'greedy'] = 'mean'
) -> np.ndarray:
    """Make predictions using ensemble model with different aggregation methods.

    Args:
        model: Trained TabM model.
        x_data: Input data.
        method: Aggregation method ('mean', 'best', 'greedy').

    Returns:
        Aggregated predictions.
    """
    # Get ensemble predictions
    predictions = model.predict(x_data)  # (batch_size, k, n_outputs)

    if method == 'mean':
        # Simple ensemble averaging using ops for consistency
        return np.mean(predictions, axis=1)

    elif method == 'best':
        # Return predictions from the best single ensemble member
        # Note: This would require validation data to determine the best member
        logger.warning("Best member selection requires validation data. Using mean instead.")
        return np.mean(predictions, axis=1)

    elif method == 'greedy':
        # Greedy ensemble selection (simplified version)
        # Note: Full implementation would require validation data and iterative selection
        logger.warning("Greedy selection requires validation data. Using mean instead.")
        return np.mean(predictions, axis=1)

    else:
        raise ValueError(f"Unknown method: {method}")


def create_tabm_for_dataset(
        X_train: np.ndarray,
        y_train: np.ndarray,
        categorical_indices: Optional[List[int]] = None,
        categorical_cardinalities: Optional[List[int]] = None,
        arch_type: str = 'tabm',
        k: int = 8,
        hidden_dims: List[int] = [256, 256],
        **kwargs
) -> TabMModel:
    """Create a TabM model configured for a specific dataset.

    Args:
        X_train: Training features.
        y_train: Training labels.
        categorical_indices: Indices of categorical features in X_train.
        categorical_cardinalities: Cardinalities of categorical features.
        arch_type: Architecture type.
        k: Number of ensemble members.
        hidden_dims: Hidden layer dimensions.
        **kwargs: Additional model arguments.

    Returns:
        Configured TabM model.

    Example:
        >>> import numpy as np
        >>> from dl_techniques.models.tabm import create_tabm_for_dataset

        >>> # Generate sample data
        >>> X_train = np.random.randn(1000, 15)  # 15 features
        >>> y_train = np.random.randint(0, 3, 1000)  # 3-class classification

        >>> # Assume first 3 features are categorical with cardinalities [5, 3, 8]
        >>> categorical_indices = [0, 1, 2]
        >>> categorical_cardinalities = [5, 3, 8]

        >>> model = create_tabm_for_dataset(
        ...     X_train, y_train,
        ...     categorical_indices=categorical_indices,
        ...     categorical_cardinalities=categorical_cardinalities,
        ...     arch_type='tabm',
        ...     k=8
        ... )
    """
    # Determine problem type
    if len(y_train.shape) == 1:
        unique_labels = np.unique(y_train)
        if len(unique_labels) == 2 and set(unique_labels) == {0, 1}:
            # Binary classification
            n_classes = 2
        elif len(unique_labels) > 2 and np.all(unique_labels == np.arange(len(unique_labels))):
            # Multiclass classification
            n_classes = len(unique_labels)
        else:
            # Regression
            n_classes = None
    else:
        # Multi-output classification
        n_classes = y_train.shape[1]

    # Determine feature splits
    if categorical_indices is None:
        categorical_indices = []
    if categorical_cardinalities is None:
        categorical_cardinalities = []

    n_total_features = X_train.shape[1]
    n_categorical = len(categorical_indices)
    n_numerical = n_total_features - n_categorical

    logger.info(f"Dataset configuration:")
    logger.info(f"  Total features: {n_total_features}")
    logger.info(f"  Numerical features: {n_numerical}")
    logger.info(f"  Categorical features: {n_categorical}")
    logger.info(f"  Problem type: {'Regression' if n_classes is None else f'{n_classes}-class classification'}")

    return create_tabm_model(
        n_num_features=n_numerical,
        cat_cardinalities=categorical_cardinalities,
        n_classes=n_classes,
        arch_type=arch_type,
        k=k,
        hidden_dims=hidden_dims,
        **kwargs
    )