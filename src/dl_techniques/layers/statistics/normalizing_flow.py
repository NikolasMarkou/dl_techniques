"""
Normalizing Flow Layer for Conditional Density Estimation using Keras 3.

This module provides a Normalizing Flow layer implemented from scratch using
only Keras 3 and NumPy. It uses a series of conditional Affine Coupling Layers
to construct a highly flexible, expressive probability distribution conditioned
on input features.

This layer is designed to be a direct, more powerful replacement for MDNs,
making almost no assumptions about the final shape of the data distribution.

Key Features:
    - Learns arbitrary, complex probability distributions.
    - Avoids strict parametric assumptions (e.g., Gaussian, Exponential).
    - Correctly models data with sharp boundaries, multiple modes, and high skew.
    - Built on the highly successful and efficient coupling layer architecture.

Theory:
    A Normalizing Flow transforms a simple base distribution (e.g., a standard
    Gaussian) into a complex target distribution through a series of invertible
    and differentiable transformations (bijectors).

    The core of this implementation is the Affine Coupling Layer:
    1. The input vector `z` is split into two halves, `z_a` and `z_b`.
    2. A neural network computes scale `s` and shift `t` parameters from `z_a`.
    3. The transformation is applied: `y_a = z_a` and `y_b = z_b * s + t`.
    4. The log-determinant of the Jacobian (LDJ) is simply `sum(log(s))`, making
       it extremely efficient.

    By stacking these layers and permuting the inputs between them, the model can
    learn exceptionally complex, high-dimensional distributions.
"""

import keras
import numpy as np
from keras import ops
from typing import Dict, Optional, Tuple, Any, List, Union

# ---------------------------------------------------------------------

EPSILON_CONSTANT = 1e-6

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class AffineCouplingLayer(keras.layers.Layer):
    """A single step of an Affine Normalizing Flow.

    This layer implements the transformation:
    y_a = z_a
    y_b = z_b * exp(log_s) + t
    where `log_s` and `t` are produced by a neural network that takes `z_a`
    and an optional external `context` as input.

    Using `exp(log_s)` ensures the scale factor is always positive.

    Args:
        input_dim: Integer, dimensionality of the input data.
        context_dim: Integer, dimensionality of the conditioning context.
        hidden_units: Integer, number of hidden units in transformation network.
        reverse: Boolean, if True, transforms the first half based on the second.
        activation: String or callable, activation function for hidden layers.
        use_tanh_stabilization: Boolean, whether to use tanh stabilization for scaling.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        - data: A tensor with shape `(batch_size, input_dim)`
        - context: A tensor with shape `(batch_size, context_dim)`

    Output shape:
        - forward: A tensor with shape `(batch_size, input_dim)`
        - inverse: A tuple of (transformed_data, log_det_jacobian) where
          transformed_data has shape `(batch_size, input_dim)` and
          log_det_jacobian has shape `(batch_size,)`

    Example:
        >>> coupling_layer = AffineCouplingLayer(
        ...     input_dim=4,
        ...     context_dim=8,
        ...     hidden_units=64
        ... )
        >>> # Build the layer
        >>> coupling_layer.build([(None, 4), (None, 8)])
        >>> # Forward pass
        >>> z = keras.random.normal((32, 4))
        >>> context = keras.random.normal((32, 8))
        >>> y = coupling_layer.forward(z, context)
        >>> # Inverse pass
        >>> z_inv, ldj = coupling_layer.inverse(y, context)
    """

    def __init__(
            self,
            input_dim: int,
            context_dim: int,
            hidden_units: int = 64,
            reverse: bool = False,
            activation: Union[str, callable] = "relu",
            use_tanh_stabilization: bool = True,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if input_dim < 2:
            raise ValueError("input_dim must be >= 2 to allow splitting")
        if context_dim < 1:
            raise ValueError("context_dim must be >= 1")
        if hidden_units < 1:
            raise ValueError("hidden_units must be >= 1")

        self.input_dim = input_dim
        self.context_dim = context_dim
        self.hidden_units = hidden_units
        self.reverse = reverse
        self.activation = activation
        self.use_tanh_stabilization = use_tanh_stabilization

        # Splitting the dimension for the two halves
        self.split_dim = input_dim // 2

        # Will be initialized in build()
        self.transformation_net = None
        self._build_input_shapes = None

    def build(self, input_shapes: List[Tuple[Optional[int], ...]]) -> None:
        """Build the layer's transformation network.

        Args:
            input_shapes: List of two shape tuples for [data, context].
        """
        if not isinstance(input_shapes, list) or len(input_shapes) != 2:
            raise ValueError("input_shapes must be a list of two shape tuples")

        self._build_input_shapes = input_shapes

        # Input size for the transformation network
        net_input_size = self.input_dim - self.split_dim + self.context_dim

        # Build the transformation network
        self.transformation_net = keras.Sequential([
            keras.layers.Dense(
                self.hidden_units,
                activation=self.activation,
                name="dense_1"
            ),
            keras.layers.Dense(
                self.hidden_units,
                activation=self.activation,
                name="dense_2"
            ),
            # Output two values (log_scale, shift) for each dimension in the dynamic part
            keras.layers.Dense(
                self.split_dim * 2,
                activation=None,
                name="output_dense"
            )
        ], name="transformation_net")

        # Build the transformation network
        self.transformation_net.build((None, net_input_size))

        super().build(input_shapes)

    def _apply_split_and_reverse(self, tensor: keras.KerasTensor) -> keras.KerasTensor:
        """Apply reverse permutation if needed."""
        if self.reverse:
            return ops.concatenate([
                tensor[..., self.split_dim:],
                tensor[..., :self.split_dim]
            ], axis=-1)
        return tensor

    def _compute_scale_and_shift(
            self,
            static_part: keras.KerasTensor,
            context: keras.KerasTensor
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Compute scale and shift parameters."""
        net_input = ops.concatenate([static_part, context], axis=-1)
        params = self.transformation_net(net_input)

        log_s = params[..., :self.split_dim]
        t = params[..., self.split_dim:]

        # Stabilize the scaling factor
        if self.use_tanh_stabilization:
            s = ops.exp(ops.tanh(log_s))
        else:
            s = ops.exp(ops.clip(log_s, -10.0, 10.0))  # Prevent overflow

        return s, t

    def forward(
            self,
            z: keras.KerasTensor,
            context: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Forward pass z -> y used for sampling.

        Args:
            z: Input tensor with shape `(batch_size, input_dim)`.
            context: Context tensor with shape `(batch_size, context_dim)`.

        Returns:
            Transformed tensor with shape `(batch_size, input_dim)`.
        """
        # Apply reverse permutation if needed
        z = self._apply_split_and_reverse(z)

        z_a = z[..., :self.split_dim]
        z_b = z[..., self.split_dim:]

        # Compute transformation parameters
        s, t = self._compute_scale_and_shift(z_a, context)

        # Apply transformation
        y_b = z_b * s + t
        y = ops.concatenate([z_a, y_b], axis=-1)

        # Apply reverse permutation back if needed
        y = self._apply_split_and_reverse(y)

        return y

    def inverse(
            self,
            y: keras.KerasTensor,
            context: keras.KerasTensor
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Inverse pass y -> z with log-det-jacobian, used for loss calculation.

        Args:
            y: Input tensor with shape `(batch_size, input_dim)`.
            context: Context tensor with shape `(batch_size, context_dim)`.

        Returns:
            Tuple of (z, log_det_jacobian) where:
            - z: Transformed tensor with shape `(batch_size, input_dim)`.
            - log_det_jacobian: Log-determinant of Jacobian with shape `(batch_size,)`.
        """
        # Apply reverse permutation if needed
        y = self._apply_split_and_reverse(y)

        y_a = y[..., :self.split_dim]
        y_b = y[..., self.split_dim:]

        # Compute transformation parameters
        s, t = self._compute_scale_and_shift(y_a, context)

        # Apply inverse transformation
        z_b = (y_b - t) / (s + EPSILON_CONSTANT)
        z = ops.concatenate([y_a, z_b], axis=-1)

        # Apply reverse permutation back if needed
        z = self._apply_split_and_reverse(z)

        # Log-determinant of the Jacobian is sum(log(scale))
        log_det_jacobian = ops.sum(ops.log(s + EPSILON_CONSTANT), axis=-1)

        return z, log_det_jacobian

    def compute_output_shape(self, input_shapes: List[Tuple[Optional[int], ...]]) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        data_shape = input_shapes[0]
        return data_shape  # Output has same shape as input data

    def get_config(self) -> Dict[str, Any]:
        """Returns the layer configuration for serialization."""
        config = super().get_config()
        config.update({
            "input_dim": self.input_dim,
            "context_dim": self.context_dim,
            "hidden_units": self.hidden_units,
            "reverse": self.reverse,
            "activation": self.activation,
            "use_tanh_stabilization": self.use_tanh_stabilization,
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get build configuration."""
        return {"input_shapes": self._build_input_shapes}

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build from configuration."""
        if config.get("input_shapes") is not None:
            self.build(config["input_shapes"])

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class NormalizingFlowLayer(keras.layers.Layer):
    """A conditional Normalizing Flow model using a stack of Affine Coupling Layers.

    This layer learns a complex, conditional probability distribution `p(y|x)`.

    The interaction is different from a standard Keras layer:
    1. The `call` method is used during training. It takes `y_true` and the
       `context` (from upstream layers) and performs an *inverse* pass to
       compute the latent vector `z` and the log-det-jacobian (LDJ).
    2. The `loss_func` takes `y_true` and the tuple `(z, ldj)` returned by
       `call`, and computes the final negative log-likelihood.
    3. The `sample` method performs a *forward* pass to generate new data.

    Args:
        output_dimension: Integer, dimensionality of the data to be modeled.
        num_flow_steps: Integer, number of coupling layers to stack.
        context_dim: Integer, dimensionality of the conditioning context vector.
        hidden_units_coupling: Integer, number of hidden units in each coupling layer.
        activation: String or callable, activation function for coupling layers.
        use_tanh_stabilization: Boolean, whether to use tanh stabilization.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        - data: A tensor with shape `(batch_size, output_dimension)`
        - context: A tensor with shape `(batch_size, context_dim)`

    Output shape:
        A tuple of (z, log_det_jacobian) where:
        - z: Latent tensor with shape `(batch_size, output_dimension)`
        - log_det_jacobian: Log-determinant with shape `(batch_size,)`

    Raises:
        ValueError: If output_dimension < 2 or num_flow_steps < 1.

    Example:
        >>> flow = NormalizingFlowLayer(
        ...     output_dimension=4,
        ...     num_flow_steps=4,
        ...     context_dim=8
        ... )
        >>> # Build the layer
        >>> flow.build([(None, 4), (None, 8)])
        >>> # Training usage
        >>> y_true = keras.random.normal((32, 4))
        >>> context = keras.random.normal((32, 8))
        >>> z, ldj = flow([y_true, context])
        >>> loss = flow.loss_func(y_true, (z, ldj))
        >>> # Sampling usage
        >>> samples = flow.sample(num_samples=10, context=context)
    """

    def __init__(
            self,
            output_dimension: int,
            num_flow_steps: int,
            context_dim: int,
            hidden_units_coupling: int = 64,
            activation: Union[str, callable] = "relu",
            use_tanh_stabilization: bool = True,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if output_dimension < 2:
            raise ValueError("output_dimension must be >= 2 to allow splitting")
        if num_flow_steps < 1:
            raise ValueError("num_flow_steps must be >= 1")
        if context_dim < 1:
            raise ValueError("context_dim must be >= 1")
        if hidden_units_coupling < 1:
            raise ValueError("hidden_units_coupling must be >= 1")

        self.output_dim = output_dimension
        self.num_flow_steps = num_flow_steps
        self.context_dim = context_dim
        self.hidden_units_coupling = hidden_units_coupling
        self.activation = activation
        self.use_tanh_stabilization = use_tanh_stabilization

        # Will be initialized in build()
        self.coupling_layers = []
        self._build_input_shapes = None

    def build(self, input_shapes: List[Tuple[Optional[int], ...]]) -> None:
        """Build the coupling layers.

        Args:
            input_shapes: List of two shape tuples for [data, context].
        """
        if not isinstance(input_shapes, list) or len(input_shapes) != 2:
            raise ValueError("input_shapes must be a list of two shape tuples")

        self._build_input_shapes = input_shapes

        # Create coupling layers
        self.coupling_layers = []
        for i in range(self.num_flow_steps):
            layer = AffineCouplingLayer(
                input_dim=self.output_dim,
                context_dim=self.context_dim,
                hidden_units=self.hidden_units_coupling,
                reverse=(i % 2 == 1),  # Alternate which half is transformed
                activation=self.activation,
                use_tanh_stabilization=self.use_tanh_stabilization,
                name=f"affine_coupling_{i}"
            )
            layer.build(input_shapes)
            self.coupling_layers.append(layer)

        super().build(input_shapes)

    def call(
            self,
            inputs: List[keras.KerasTensor],
            training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Performs the inverse pass y -> z for loss calculation.

        Args:
            inputs: List of [y, context] tensors where:
                - y: The true data tensor with shape `(batch_size, output_dim)`.
                - context: The conditioning vector with shape `(batch_size, context_dim)`.
            training: Boolean indicating training mode (unused but included for compatibility).

        Returns:
            Tuple containing:
            - z: The data transformed into latent space with shape `(batch_size, output_dim)`.
            - total_log_det_jacobian: Sum of LDJs with shape `(batch_size,)`.
        """
        if len(inputs) != 2:
            raise ValueError("Expected exactly 2 inputs: [y, context]")

        y, context = inputs

        # Initialize log-det-jacobian accumulator
        batch_size = ops.shape(y)[0]
        total_log_det_jacobian = ops.zeros(batch_size)

        # Apply the inverse transformations in reverse order
        z = y
        for layer in reversed(self.coupling_layers):
            z, ldj = layer.inverse(z, context)
            total_log_det_jacobian += ldj

        return z, total_log_det_jacobian

    def loss_func(
            self,
            y_true: keras.KerasTensor,
            y_pred: Tuple[keras.KerasTensor, keras.KerasTensor]
    ) -> keras.KerasTensor:
        """Computes the negative log-likelihood of the model.

        Args:
            y_true: The true data (used for matching shapes, not values).
            y_pred: Tuple of (z, total_log_det_jacobian) returned by call method.

        Returns:
            The negative log-likelihood loss as a scalar tensor.
        """
        z, total_log_det_jacobian = y_pred

        # Log-probability of z under the base distribution (Standard Normal)
        # log p(z) = -0.5 * [d*log(2π) + sum(z²)]
        log_prob_z = -0.5 * (
                self.output_dim * ops.log(2 * np.pi) +
                ops.sum(z ** 2, axis=-1)
        )

        # Change of variables formula: log p(y) = log p(z) + log|det(J)|
        log_prob_y = log_prob_z + total_log_det_jacobian

        return -ops.mean(log_prob_y)

    def sample(
            self,
            num_samples: int,
            context: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Performs the forward pass z -> y to generate new samples.

        Args:
            num_samples: Number of samples to generate for each context vector.
            context: Conditioning vector with shape `(batch_size, context_dim)`.

        Returns:
            Generated samples with shape `(batch_size, num_samples, output_dim)`.
        """
        if num_samples < 1:
            raise ValueError("num_samples must be >= 1")

        batch_size = ops.shape(context)[0]

        # Sample from the base distribution (Standard Normal)
        z = keras.random.normal(shape=(batch_size, num_samples, self.output_dim))

        # Reshape context for broadcasting with samples
        context_expanded = ops.expand_dims(context, 1)

        # Apply the forward transformations
        y = z
        for layer in self.coupling_layers:
            # Reshape for the layer's call
            y_shape = ops.shape(y)
            y_flat = ops.reshape(y, [-1, self.output_dim])
            context_flat = ops.reshape(
                ops.broadcast_to(context_expanded, y_shape),
                [-1, self.context_dim]
            )

            y_transformed_flat = layer.forward(y_flat, context_flat)
            y = ops.reshape(y_transformed_flat, y_shape)

        return y

    def compute_output_shape(self, input_shapes: List[Tuple[Optional[int], ...]]) -> Tuple[
        Tuple[Optional[int], ...], Tuple[Optional[int], ...]]:
        """Compute output shapes for the tuple (z, log_det_jacobian)."""
        data_shape = input_shapes[0]
        batch_size = data_shape[0] if data_shape else None
        ldj_shape = (batch_size,)
        return data_shape, ldj_shape

    def get_config(self) -> Dict[str, Any]:
        """Returns the layer configuration for serialization."""
        config = super().get_config()
        config.update({
            "output_dimension": self.output_dim,
            "num_flow_steps": self.num_flow_steps,
            "context_dim": self.context_dim,
            "hidden_units_coupling": self.hidden_units_coupling,
            "activation": self.activation,
            "use_tanh_stabilization": self.use_tanh_stabilization,
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get build configuration."""
        return {"input_shapes": self._build_input_shapes}

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build from configuration."""
        if config.get("input_shapes") is not None:
            self.build(config["input_shapes"])

# ---------------------------------------------------------------------
