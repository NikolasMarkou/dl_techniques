"""
Normalizing Flow Layer for Conditional Density Estimation using Keras 3.

This module implements a sophisticated normalizing flow architecture using a series
of conditional Affine Coupling Layers to construct highly flexible, expressive
probability distributions conditioned on input features.

Normalizing flows represent a paradigm shift from traditional parametric density
models (like Mixture Density Networks) by learning transformations between simple
and complex probability distributions through invertible neural networks. This
approach makes minimal assumptions about the target distribution shape, enabling
accurate modeling of multimodal, skewed, and bounded distributions.

Key Advantages over Traditional Approaches:

1. **Distribution Flexibility**: Can model arbitrary complex distributions without
   parametric assumptions (Gaussian, Exponential, etc.)

2. **Exact Likelihood**: Provides exact likelihood computation through the change
   of variables formula, enabling principled probabilistic inference

3. **Bidirectional Mapping**: Supports both density estimation (inverse pass) and
   sampling (forward pass) through the same model

4. **Conditional Modeling**: Full support for learning p(y|x) with complex
   conditioning relationships

Theoretical Foundation:
A normalizing flow transforms a simple base distribution π(z) (typically standard
Gaussian) into a complex target distribution p(y) through a sequence of invertible
transformations f₁, f₂, ..., fₖ:

    y = fₖ ∘ fₖ₋₁ ∘ ... ∘ f₁(z)

The likelihood is computed using the change of variables formula:
    log p(y) = log π(z) + Σᵢ log|det(∂fᵢ/∂zᵢ₋₁)|

The Affine Coupling Layer provides an efficient, stable implementation where:
- Input z is split into two parts: z_a and z_b
- Transformation: y_a = z_a, y_b = z_b * s(z_a, context) + t(z_a, context)
- Jacobian determinant: |det(J)| = ∏s(z_a, context) (computationally efficient)

This architecture scales to high-dimensional problems while maintaining exact
likelihood computation and stable training dynamics.
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
    """
    Affine coupling transformation layer for normalizing flows with conditional context.

    Implements a single invertible transformation step using the Real NVP coupling
    architecture. The input is split at dimension ``input_dim // 2``; one half remains
    unchanged while the other is transformed via ``y_b = z_b * s(z_a, ctx) + t(z_a, ctx)``
    where ``s`` and ``t`` are scale and shift functions computed by a neural network
    conditioned on the static half and external context. The log-determinant of the
    Jacobian is ``sum(log(s))``, computed in ``O(d)`` time. Alternating which half
    is transformed across stacked layers ensures all dimensions are eventually
    transformed.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────────┐
        │  Input z (batch, input_dim)          │
        │  + Context (batch, context_dim)      │
        └────────────┬─────────────────────────┘
                     ▼
        ┌────────────────────────┐
        │  Split: z_a | z_b     │
        └───┬────────────┬──────┘
            │            │
            ▼            │
        ┌────────────┐   │
        │ [z_a; ctx] │   │
        │  ─► Net    │   │
        │  ─► s, t   │   │
        └───┬────────┘   │
            │            ▼
            │   ┌────────────────┐
            └──►│ y_b = z_b*s+t │
                └───────┬────────┘
                        ▼
        ┌────────────────────────┐
        │  Concat: z_a | y_b    │
        │  ─► Output y          │
        └────────────────────────┘

    :param input_dim: Dimensionality of the input data. Must be >= 2.
    :type input_dim: int
    :param context_dim: Dimensionality of the conditioning context. Must be >= 1.
    :type context_dim: int
    :param hidden_units: Hidden units in the transformation network. Defaults to 64.
    :type hidden_units: int
    :param reverse: Whether to reverse the split ordering. Defaults to ``False``.
    :type reverse: bool
    :param activation: Activation function for hidden layers. Defaults to ``'relu'``.
    :type activation: str | callable
    :param use_tanh_stabilization: Whether to apply tanh to log-scale parameters.
        Defaults to ``True``.
    :type use_tanh_stabilization: bool
    :param kwargs: Additional keyword arguments for the Layer base class.
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
        """Initialize the AffineCouplingLayer."""
        super().__init__(**kwargs)

        # Validate input parameters
        if input_dim < 2:
            raise ValueError("input_dim must be >= 2 to allow splitting")
        if context_dim < 1:
            raise ValueError("context_dim must be >= 1")
        if hidden_units < 1:
            raise ValueError("hidden_units must be >= 1")

        # Store configuration parameters
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.hidden_units = hidden_units
        self.reverse = reverse
        self.activation = activation
        self.use_tanh_stabilization = use_tanh_stabilization

        # Compute splitting dimension
        self.split_dim = input_dim // 2
        dim_to_transform = self.input_dim - self.split_dim

        # CREATE transformation network in __init__ (modern Keras 3 pattern)
        # Input size: unchanged part + context
        net_input_size = self.split_dim + self.context_dim

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
            # Output: scale and shift parameters for each transformed dimension
            keras.layers.Dense(
                dim_to_transform * 2,  # 2x for scale and shift
                activation=None,
                name="output_dense"
            )
        ], name="transformation_net")

    def build(self, input_shapes: List[Tuple[Optional[int], ...]]) -> None:
        """Build the layer and its transformation network.

        :param input_shapes: List of two shape tuples for ``[data, context]``.
        :type input_shapes: list[tuple[int | None, ...]]
        """
        if not isinstance(input_shapes, list) or len(input_shapes) != 2:
            raise ValueError("input_shapes must be a list of two shape tuples")

        # Input size for the transformation network
        net_input_size = self.split_dim + self.context_dim

        # BUILD the transformation network (critical for serialization)
        self.transformation_net.build((None, net_input_size))

        # Always call parent build at the end
        super().build(input_shapes)

    def _apply_split_and_reverse(self, tensor: keras.KerasTensor) -> keras.KerasTensor:
        """Apply reverse permutation if needed to alternate transformed dimensions."""
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
        """Compute scale and shift parameters from static input part and context.

        :param static_part: The unchanged part of the input.
        :type static_part: keras.KerasTensor
        :param context: The conditioning context vector.
        :type context: keras.KerasTensor
        :return: Tuple of ``(scale, shift)`` parameters.
        :rtype: tuple[keras.KerasTensor, keras.KerasTensor]
        """
        # Concatenate static part and context for transformation network input
        net_input = ops.concatenate([static_part, context], axis=-1)

        # Get transformation parameters from network
        params = self.transformation_net(net_input)

        # Split into scale (log) and shift parameters
        dim_to_transform = self.input_dim - self.split_dim
        log_s = params[..., :dim_to_transform]
        t = params[..., dim_to_transform:]

        # Compute scale with numerical stabilization
        if self.use_tanh_stabilization:
            # Tanh keeps values in reasonable range but limits scale factor range
            s = ops.exp(ops.tanh(log_s))
        else:
            # Clipping prevents overflow but allows larger scale factors
            s = ops.exp(ops.clip(log_s, -10.0, 10.0))

        return s, t

    def forward(
        self,
        z: keras.KerasTensor,
        context: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Forward transformation z to y for sampling.

        :param z: Input from base distribution, shape ``(batch_size, input_dim)``.
        :type z: keras.KerasTensor
        :param context: Conditioning context, shape ``(batch_size, context_dim)``.
        :type context: keras.KerasTensor
        :return: Transformed tensor y.
        :rtype: keras.KerasTensor
        """
        # Apply permutation if this layer reverses the split
        z = self._apply_split_and_reverse(z)

        # Split input into static and dynamic parts
        z_a = z[..., :self.split_dim]       # Static part (unchanged)
        z_b = z[..., self.split_dim:]       # Dynamic part (to be transformed)

        # Compute transformation parameters from static part and context
        s, t = self._compute_scale_and_shift(z_a, context)

        # Apply affine transformation to dynamic part
        y_b = z_b * s + t
        y = ops.concatenate([z_a, y_b], axis=-1)

        # Apply reverse permutation to restore original ordering
        y = self._apply_split_and_reverse(y)

        return y

    def inverse(
        self,
        y: keras.KerasTensor,
        context: keras.KerasTensor
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Inverse transformation y to z with log-determinant for likelihood.

        :param y: Transformed data tensor, shape ``(batch_size, input_dim)``.
        :type y: keras.KerasTensor
        :param context: Conditioning context, shape ``(batch_size, context_dim)``.
        :type context: keras.KerasTensor
        :return: Tuple of ``(z, log_det_jacobian)``.
        :rtype: tuple[keras.KerasTensor, keras.KerasTensor]
        """
        # Apply permutation if this layer reverses the split
        y = self._apply_split_and_reverse(y)

        # Split transformed data
        y_a = y[..., :self.split_dim]       # Static part (unchanged)
        y_b = y[..., self.split_dim:]       # Transformed part (to be inverted)

        # Compute transformation parameters from static part and context
        s, t = self._compute_scale_and_shift(y_a, context)

        # Apply inverse transformation to dynamic part
        z_b = (y_b - t) / (s + EPSILON_CONSTANT)
        z = ops.concatenate([y_a, z_b], axis=-1)

        # Apply reverse permutation to restore original ordering
        z = self._apply_split_and_reverse(z)

        # Compute log-determinant of Jacobian (sum of log scale factors)
        log_det_jacobian = ops.sum(ops.log(s + EPSILON_CONSTANT), axis=-1)

        return z, log_det_jacobian

    def compute_output_shape(
        self,
        input_shapes: List[Tuple[Optional[int], ...]]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape (same as data input shape).

        :param input_shapes: List of input shapes ``[data_shape, context_shape]``.
        :type input_shapes: list[tuple[int | None, ...]]
        :return: Output shape tuple.
        :rtype: tuple[int | None, ...]
        """
        data_shape = input_shapes[0]
        return data_shape

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        :return: Configuration dictionary.
        :rtype: dict[str, Any]
        """
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


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class NormalizingFlowLayer(keras.layers.Layer):
    """
    Conditional normalizing flow layer using stacked affine coupling transformations.

    Implements a complete normalizing flow model that learns ``p(y|x)`` by composing
    ``K`` invertible affine coupling transformations: ``y = f_K . f_{K-1} . ... . f_1(z)``
    where ``z ~ N(0, I)``. The exact log-likelihood is computed via the change of
    variables formula: ``log p(y|ctx) = log p(z) + sum_i log|det(df_i / dz_{i-1})|``.
    During training, the inverse pass maps observed data to latent space for
    likelihood evaluation; during sampling, the forward pass generates new data
    from the base distribution.

    **Architecture Overview:**

    .. code-block:: text

        Training (Inverse: y ─► z):
        ┌──────────────────────────────┐
        │ Data y + Context             │
        └──────────────┬───────────────┘
                       ▼
        ┌──────────────────────────────┐
        │ Coupling_K^{-1} ─► ldj_K    │
        ├──────────────────────────────┤
        │ Coupling_{K-1}^{-1} ─► ...  │
        ├──────────────────────────────┤
        │ Coupling_1^{-1} ─► ldj_1    │
        └──────────────┬───────────────┘
                       ▼
        ┌──────────────────────────────┐
        │ z, total_log_det_jacobian    │
        └──────────────────────────────┘

        Sampling (Forward: z ─► y):
        ┌──────────────────────────────┐
        │ z ~ N(0, I) + Context        │
        └──────────────┬───────────────┘
                       ▼
        ┌──────────────────────────────┐
        │ Coupling_1 ─► Coupling_2     │
        │ ─► ... ─► Coupling_K         │
        └──────────────┬───────────────┘
                       ▼
        ┌──────────────────────────────┐
        │ Samples y                    │
        └──────────────────────────────┘

    :param output_dimension: Dimensionality of the target distribution. Must be >= 2.
    :type output_dimension: int
    :param num_flow_steps: Number of coupling layers. Must be >= 1.
    :type num_flow_steps: int
    :param context_dim: Dimensionality of the conditioning context. Must be >= 1.
    :type context_dim: int
    :param hidden_units_coupling: Hidden units per coupling layer. Defaults to 64.
    :type hidden_units_coupling: int
    :param activation: Activation for coupling networks. Defaults to ``'relu'``.
    :type activation: str | callable
    :param use_tanh_stabilization: Whether to apply tanh stabilization.
        Defaults to ``True``.
    :type use_tanh_stabilization: bool
    :param kwargs: Additional keyword arguments for the Layer base class.
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
        """Initialize the NormalizingFlowLayer."""
        super().__init__(**kwargs)

        # Validate input parameters
        if output_dimension < 2:
            raise ValueError("output_dimension must be >= 2 to allow splitting")
        if num_flow_steps < 1:
            raise ValueError("num_flow_steps must be >= 1")
        if context_dim < 1:
            raise ValueError("context_dim must be >= 1")
        if hidden_units_coupling < 1:
            raise ValueError("hidden_units_coupling must be >= 1")

        # Store configuration parameters
        self.output_dim = output_dimension
        self.num_flow_steps = num_flow_steps
        self.context_dim = context_dim
        self.hidden_units_coupling = hidden_units_coupling
        self.activation = activation
        self.use_tanh_stabilization = use_tanh_stabilization

        # CREATE coupling layers in __init__ (modern Keras 3 pattern)
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
            self.coupling_layers.append(layer)

    def build(self, input_shapes: List[Tuple[Optional[int], ...]]) -> None:
        """Build the layer and all coupling layers.

        :param input_shapes: List of two shape tuples for ``[data, context]``.
        :type input_shapes: list[tuple[int | None, ...]]
        """
        if not isinstance(input_shapes, list) or len(input_shapes) != 2:
            raise ValueError("input_shapes must be a list of two shape tuples")

        # BUILD all coupling layers (critical for serialization)
        for layer in self.coupling_layers:
            layer.build(input_shapes)

        # Always call parent build at the end
        super().build(input_shapes)

    def call(
        self,
        inputs: List[keras.KerasTensor],
        training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Inverse transformation y to z for likelihood computation.

        :param inputs: List of ``[data, context]`` tensors.
        :type inputs: list[keras.KerasTensor]
        :param training: Boolean for training mode.
        :type training: bool | None
        :return: Tuple of ``(z, total_log_det_jacobian)``.
        :rtype: tuple[keras.KerasTensor, keras.KerasTensor]
        """
        if len(inputs) != 2:
            raise ValueError("Expected exactly 2 inputs: [data, context]")

        y, context = inputs

        # Initialize log-determinant accumulator
        batch_size = ops.shape(y)[0]
        total_log_det_jacobian = ops.zeros(batch_size, dtype=y.dtype)

        # Apply inverse transformations in reverse order (y → z)
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
        """Compute exact negative log-likelihood loss.

        :param y_true: True data (for shape compatibility).
        :type y_true: keras.KerasTensor
        :param y_pred: Tuple of ``(z, total_log_det_jacobian)`` from call.
        :type y_pred: tuple[keras.KerasTensor, keras.KerasTensor]
        :return: Scalar negative log-likelihood loss.
        :rtype: keras.KerasTensor
        """
        z, total_log_det_jacobian = y_pred

        # Log-probability under base distribution (standard multivariate normal)
        # log π(z) = -0.5 * [d*log(2π) + ||z||²]
        log_prob_z = -0.5 * (
            self.output_dim * ops.log(2 * np.pi) +
            ops.sum(z ** 2, axis=-1)
        )

        # Change of variables: log p(y) = log π(z) + log|det(J)|
        log_prob_y = log_prob_z + total_log_det_jacobian

        # Return negative log-likelihood for minimization
        return -ops.mean(log_prob_y)

    def sample(
        self,
        num_samples: int,
        context: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Generate samples from the learned conditional distribution.

        :param num_samples: Number of samples per context. Must be >= 1.
        :type num_samples: int
        :param context: Conditioning context, shape ``(batch_size, context_dim)``.
        :type context: keras.KerasTensor
        :return: Samples of shape ``(batch_size, num_samples, output_dim)``.
        :rtype: keras.KerasTensor
        """
        if num_samples < 1:
            raise ValueError("num_samples must be >= 1")

        batch_size = ops.shape(context)[0]

        # Sample from base distribution (standard multivariate normal)
        z = keras.random.normal(
            shape=(batch_size, num_samples, self.output_dim),
            dtype=context.dtype
        )

        # Reshape z for efficient batch processing through layers
        # (batch, num_samples, output_dim) -> (batch * num_samples, output_dim)
        y_flat = ops.reshape(z, (-1, self.output_dim))

        # Prepare context to match the flattened z
        # context: (batch, context_dim) -> (batch, 1, context_dim)
        context_expanded = ops.expand_dims(context, 1)
        # -> (batch, num_samples, context_dim)
        context_tiled = ops.repeat(context_expanded, num_samples, axis=1)
        # -> (batch * num_samples, context_dim)
        context_flat = ops.reshape(context_tiled, (-1, self.context_dim))

        # Apply forward transformations (z → y)
        for layer in self.coupling_layers:
            y_flat = layer.forward(y_flat, context_flat)

        # Reshape back to the original sample structure: (batch, num_samples, output_dim)
        y = ops.reshape(y_flat, (batch_size, num_samples, self.output_dim))

        return y

    def compute_output_shape(
        self,
        input_shapes: List[Tuple[Optional[int], ...]]
    ) -> Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]:
        """Compute output shapes for ``(z, log_det_jacobian)``.

        :param input_shapes: List of input shapes.
        :type input_shapes: list[tuple[int | None, ...]]
        :return: Tuple of output shapes.
        :rtype: tuple[tuple[int | None, ...], tuple[int | None, ...]]
        """
        data_shape = input_shapes[0]
        batch_size = data_shape[0] if data_shape else None
        ldj_shape = (batch_size,)
        return data_shape, ldj_shape

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        :return: Configuration dictionary.
        :rtype: dict[str, Any]
        """
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


# ---------------------------------------------------------------------c