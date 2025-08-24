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

    This layer implements a single invertible transformation step in a normalizing flow,
    using the affine coupling architecture. It splits the input into two parts, keeps
    one part unchanged, and transforms the other using learned scale and shift parameters
    that depend on both the unchanged part and external context.

    The transformation is designed to be:
    - **Invertible**: Enables exact likelihood computation and sampling
    - **Efficient**: Jacobian determinant computation is O(d) instead of O(d³)
    - **Stable**: Uses careful numerical stabilization to prevent overflow/underflow
    - **Flexible**: Conditioning on external context enables complex dependencies

    Architecture details:
    - Input splitting avoids the need for matrix determinant computation
    - Neural network parameterizes transformation based on unchanged input and context
    - Alternating which half is transformed across layers increases expressiveness
    - Tanh stabilization prevents numerical instabilities in scale parameters

    Mathematical formulation:
        Forward (z → y):
        - y_a = z_a (unchanged)
        - y_b = z_b * s(z_a, context) + t(z_a, context) (transformed)

        Inverse (y → z):
        - z_a = y_a (unchanged)
        - z_b = (y_b - t(y_a, context)) / s(y_a, context) (inverse transform)

        Log-determinant: log|det(J)| = Σ log(s(z_a, context))

    Args:
        input_dim: int, dimensionality of the input data. Must be >= 2 to allow
            splitting into two parts. Determines the size of vectors being transformed.
        context_dim: int, dimensionality of the conditioning context vector. Must be >= 1.
            Larger context dimensions allow more complex conditioning relationships.
        hidden_units: int, number of hidden units in the transformation neural network.
            Larger values increase expressiveness but also computational cost.
            Defaults to 64.
        reverse: bool, if True, transforms the second half based on the first half
            instead of the first half based on the second. Used to alternate
            transformations across coupling layers. Defaults to False.
        activation: Union[str, callable], activation function for hidden layers in
            the transformation network. Common choices: 'relu', 'tanh', 'gelu'.
            Defaults to 'relu'.
        use_tanh_stabilization: bool, whether to apply tanh stabilization to log-scale
            parameters before exponentiation. Prevents numerical overflow but may
            limit the range of scale factors. Defaults to True.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        List of two tensors:
        - data: ``(batch_size, input_dim)`` - The data to be transformed
        - context: ``(batch_size, context_dim)`` - Conditioning context vector

    Output shape:
        Forward pass: ``(batch_size, input_dim)`` - Transformed data
        Inverse pass: Tuple of:
        - ``(batch_size, input_dim)`` - Inverse transformed data
        - ``(batch_size,)`` - Log-determinant of Jacobian

    Attributes:
        transformation_net: keras.Sequential, neural network that computes scale and
            shift parameters from the unchanged input part and context.
        split_dim: int, dimension at which to split the input (input_dim // 2).

    Example:
        ```python
        # Basic coupling layer setup
        coupling_layer = AffineCouplingLayer(
            input_dim=4,
            context_dim=8,
            hidden_units=128,
            activation='gelu'
        )

        # Build and use
        data = keras.random.normal((32, 4))
        context = keras.random.normal((32, 8))

        # Forward transformation (z → y) for sampling
        y = coupling_layer.forward(data, context)

        # Inverse transformation (y → z) for likelihood computation
        z, log_det_jac = coupling_layer.inverse(y, context)

        # High-dimensional example with stabilization
        high_dim_layer = AffineCouplingLayer(
            input_dim=256,
            context_dim=64,
            hidden_units=512,
            use_tanh_stabilization=True,  # Important for high dimensions
            activation='swish'
        )

        # Multi-step flow alternating transformations
        layers = [
            AffineCouplingLayer(input_dim=10, context_dim=5, reverse=False),
            AffineCouplingLayer(input_dim=10, context_dim=5, reverse=True),
            AffineCouplingLayer(input_dim=10, context_dim=5, reverse=False),
        ]
        ```

    Raises:
        ValueError: If input_dim < 2 (cannot split for coupling).
        ValueError: If context_dim < 1 (no conditioning information).
        ValueError: If hidden_units < 1 (invalid network architecture).

    Note:
        This layer implements the Real NVP coupling architecture, which has proven
        highly effective for density modeling tasks. The alternating reverse pattern
        across multiple layers is crucial for ensuring that all input dimensions
        are eventually transformed.

        For very high-dimensional inputs, consider using larger hidden_units and
        enabling tanh_stabilization to maintain numerical stability during training.
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
        """
        Build the layer and its transformation network.

        Args:
            input_shapes: List of two shape tuples for [data, context].

        Raises:
            ValueError: If input_shapes is not a list of exactly 2 shapes.
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
        """
        Compute scale and shift parameters from static input part and context.

        Args:
            static_part: The unchanged part of the input.
            context: The conditioning context vector.

        Returns:
            Tuple of (scale, shift) parameters.
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
        """
        Forward transformation z → y for sampling.

        Args:
            z: Input tensor from base distribution, shape ``(batch_size, input_dim)``.
            context: Conditioning context, shape ``(batch_size, context_dim)``.

        Returns:
            Transformed tensor y, shape ``(batch_size, input_dim)``.
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
        """
        Inverse transformation y → z with log-determinant for likelihood computation.

        Args:
            y: Transformed data tensor, shape ``(batch_size, input_dim)``.
            context: Conditioning context, shape ``(batch_size, context_dim)``.

        Returns:
            Tuple containing:
            - z: Original data tensor, shape ``(batch_size, input_dim)``
            - log_det_jacobian: Log-determinant of Jacobian, shape ``(batch_size,)``
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
        """
        Compute output shape (same as data input shape).

        Args:
            input_shapes: List of input shapes [data_shape, context_shape].

        Returns:
            Output shape tuple (same as data input shape).
        """
        data_shape = input_shapes[0]
        return data_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        Returns:
            Dictionary containing all layer configuration parameters.
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

    This layer implements a complete normalizing flow model that learns complex
    conditional probability distributions p(y|x) by composing multiple invertible
    affine coupling transformations. It provides a powerful alternative to
    traditional parametric density models with exact likelihood computation
    and efficient sampling capabilities.

    The layer operates through two primary modes:
    - **Training mode**: Inverse pass (y → z) to compute likelihood for observed data
    - **Sampling mode**: Forward pass (z → y) to generate new samples from learned distribution

    Architecture design principles:
    - **Alternating coupling patterns**: Each layer transforms different input dimensions
    - **Context conditioning**: All transformations depend on external context
    - **Numerical stability**: Careful handling of scale parameters and determinants
    - **Scalable depth**: Configurable number of coupling layers for complexity control

    The model learns transformations f₁, f₂, ..., fₖ such that:
        y = fₖ ∘ fₖ₋₁ ∘ ... ∘ f₁(z)

    Where z ~ N(0, I) and the likelihood is computed as:
        log p(y|context) = log p(z) + Σᵢ log|det(∂fᵢ/∂zᵢ₋₁)|

    Key advantages over mixture density networks:
    - **Exact likelihood**: No approximation errors in probability computation
    - **Complex distributions**: Can model multimodal, skewed, bounded distributions
    - **Efficient sampling**: Direct sampling without rejection or MCMC
    - **Principled training**: Maximum likelihood objective with exact gradients

    Args:
        output_dimension: int, dimensionality of the target distribution. Must be >= 2
            to enable coupling layer splits. Determines the size of generated samples.
        num_flow_steps: int, number of coupling layers to stack. More layers enable
            more complex distributions but increase computational cost. Must be >= 1.
            Typical values: 4-16 depending on data complexity.
        context_dim: int, dimensionality of the conditioning context vector. Must be >= 1.
            Should match the output dimension of upstream network layers.
        hidden_units_coupling: int, number of hidden units in each coupling layer's
            transformation network. Larger values increase expressiveness but also
            computational cost. Defaults to 64.
        activation: Union[str, callable], activation function for coupling layer networks.
            Common effective choices: 'relu', 'gelu', 'swish'. Defaults to 'relu'.
        use_tanh_stabilization: bool, whether to apply tanh stabilization to scale
            parameters. Recommended for high-dimensional or numerically sensitive
            applications. Defaults to True.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        List of two tensors:
        - data: ``(batch_size, output_dimension)`` - Target data for likelihood computation
        - context: ``(batch_size, context_dim)`` - Conditioning context from upstream layers

    Output shape:
        Training (inverse pass): Tuple of:
        - z: ``(batch_size, output_dimension)`` - Latent variables
        - log_det_jacobian: ``(batch_size,)`` - Log-determinant for likelihood

        Sampling: ``(batch_size, num_samples, output_dimension)`` - Generated samples

    Attributes:
        coupling_layers: List[AffineCouplingLayer], the sequence of coupling
            transformations that define the normalizing flow.

    Example:
        ```python
        # Basic regression with uncertainty quantification
        inputs = keras.Input(shape=(10,))
        features = keras.layers.Dense(64, activation='relu')(inputs)
        context = keras.layers.Dense(32)(features)  # Context for flow

        # Create normalizing flow layer
        flow = NormalizingFlowLayer(
            output_dimension=2,  # Predict 2D target to allow splits
            num_flow_steps=6,
            context_dim=32,
            hidden_units_coupling=128
        )

        # Training setup
        targets = keras.Input(shape=(2,))
        z, log_det_jac = flow([targets, context])

        # Custom training model with likelihood loss
        model = keras.Model([inputs, targets], [z, log_det_jac])

        # Custom loss function
        def flow_loss(y_true, y_pred):
            z, ldj = y_pred
            return flow.loss_func(y_true, (z, ldj))

        model.compile(optimizer='adam', loss=flow_loss)

        # Multi-dimensional time series forecasting
        time_series_flow = NormalizingFlowLayer(
            output_dimension=5,    # 5-dimensional forecasts
            num_flow_steps=8,      # More complex for multi-dim
            context_dim=128,       # Rich context from LSTM
            hidden_units_coupling=256
        )

        # Financial modeling with complex distributions
        financial_flow = NormalizingFlowLayer(
            output_dimension=3,    # Price, volume, volatility
            num_flow_steps=12,     # Very complex for finance
            context_dim=64,
            activation='gelu',     # Better for finance
            use_tanh_stabilization=True  # Important for stability
        )

        # Sampling usage after training
        context_test = keras.random.normal((100, 32))
        samples = flow.sample(num_samples=10, context=context_test)
        # Shape: (100, 10, 2) - 10 samples for each of 100 contexts

        # Complete end-to-end example
        def create_flow_model(input_dim, output_dim, context_dim=64):
            inputs = keras.Input(shape=(input_dim,))

            # Feature extraction
            x = keras.layers.Dense(128, activation='relu')(inputs)
            x = keras.layers.Dense(128, activation='relu')(x)
            context = keras.layers.Dense(context_dim)(x)

            # Normalizing flow
            flow_layer = NormalizingFlowLayer(
                output_dimension=output_dim,
                num_flow_steps=8,
                context_dim=context_dim
            )

            # Training inputs
            targets = keras.Input(shape=(output_dim,))
            z, ldj = flow_layer([targets, context])

            # Create models
            training_model = keras.Model([inputs, targets], [z, ldj])
            prediction_model = keras.Model(inputs, context)

            return training_model, prediction_model, flow_layer
        ```

    Raises:
        ValueError: If output_dimension < 2 (cannot split for coupling layers).
        ValueError: If num_flow_steps < 1 (need at least one transformation).
        ValueError: If context_dim < 1 (need conditioning information).
        ValueError: If hidden_units_coupling < 1 (invalid network size).

    Note:
        This implementation uses the Real NVP architecture with alternating coupling
        patterns. For optimal performance:

        1. **Layer depth**: Start with 4-8 layers, increase for complex distributions
        2. **Hidden units**: Scale with problem complexity, typical range 64-512
        3. **Numerical stability**: Always use tanh_stabilization for training stability
        4. **Context design**: Ensure context captures all relevant conditioning information

        The loss_func method implements exact negative log-likelihood, providing
        principled probabilistic training. The sample method enables efficient
        generation of new data points from the learned conditional distribution.
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
        """
        Build the layer and all coupling layers.

        Args:
            input_shapes: List of two shape tuples for [data, context].

        Raises:
            ValueError: If input_shapes is not a list of exactly 2 shapes.
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
        """
        Inverse transformation y → z for likelihood computation during training.

        This method performs the inverse pass through the normalizing flow,
        transforming observed data into the latent space for likelihood evaluation.

        Args:
            inputs: List of [data, context] tensors where:
                - data: Target data tensor, shape ``(batch_size, output_dim)``
                - context: Conditioning context, shape ``(batch_size, context_dim)``
            training: Boolean indicating training mode (unused but included for API consistency).

        Returns:
            Tuple containing:
            - z: Latent variables in base distribution space, shape ``(batch_size, output_dim)``
            - total_log_det_jacobian: Accumulated log-determinant, shape ``(batch_size,)``

        Raises:
            ValueError: If inputs doesn't contain exactly 2 tensors.
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
        """
        Compute negative log-likelihood loss for maximum likelihood training.

        This function implements the exact negative log-likelihood using the
        change of variables formula for normalizing flows.

        Args:
            y_true: True data (used for shape compatibility, values not used directly).
            y_pred: Tuple of (z, total_log_det_jacobian) from the call method.

        Returns:
            Negative log-likelihood loss as scalar tensor for minimization.
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
        """
        Generate samples from the learned conditional distribution.

        This method performs the forward pass through the normalizing flow,
        transforming samples from the base distribution into the target space.

        Args:
            num_samples: Number of samples to generate per context. Must be >= 1.
            context: Conditioning context, shape ``(batch_size, context_dim)``.

        Returns:
            Generated samples, shape ``(batch_size, num_samples, output_dim)``.

        Raises:
            ValueError: If num_samples < 1.
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
        """
        Compute output shapes for the tuple (z, log_det_jacobian).

        Args:
            input_shapes: List of input shapes [data_shape, context_shape].

        Returns:
            Tuple of output shapes: (z_shape, log_det_jacobian_shape).
        """
        data_shape = input_shapes[0]
        batch_size = data_shape[0] if data_shape else None
        ldj_shape = (batch_size,)
        return data_shape, ldj_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        Returns:
            Dictionary containing all layer configuration parameters.
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