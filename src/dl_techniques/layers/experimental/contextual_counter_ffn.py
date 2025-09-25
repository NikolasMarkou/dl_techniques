import keras
from typing import Literal, Tuple, Optional, Union, Any, Dict

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ContextualCounterFFN(keras.layers.Layer):
    """
    Feed-forward network that modulates sequences through contextual counting.

    This layer implements a sophisticated modulation protocol that identifies learned
    countable features, aggregates their frequency across configurable scopes,
    transforms the count information, and integrates it back into token representations
    through gated mechanisms.

    **Intent**: Enable sequence modeling that captures frequency-based patterns and
    positional information through learnable counting operations, useful for tasks
    requiring awareness of token occurrence patterns or sequence structure.

    **Architecture**:
    ```
    Input(batch, seq, input_dim)
           ↓
    [Sense] Key Projection → Countable Events (sigmoid)
           ↓
    [Aggregate] Count by Scope:
      - global: DC offset (all positions same)
      - causal: Low-pass filter (cumsum forward)
      - bidirectional: Band-pass (forward + backward cumsum)
           ↓
    [Transform] Count → Output Dim (with activation)
           ↓
    [Modulate] Gate Control:
      - blend: gate*counts + (1-gate)*input (if dims match)
      - project: gate*counts + projection(input)
      - gate_only: gate*counts
           ↓
    Output(batch, seq, output_dim)
    ```

    **Mathematical Operations**:
    1. Events = σ(W_key @ X + b_key)
    2. Counts = Aggregate(Events, scope)
    3. Transformed = f(W_count @ Counts + b_count)
    4. Gate = σ(W_gate @ X + b_gate)
    5. Output = Modulate(Transformed, Input, Gate, mode)

    Where:
    - σ denotes sigmoid activation
    - f denotes the specified activation function
    - @ denotes matrix multiplication
    - Aggregate varies by counting_scope
    - Modulate varies by residual_mode

    Args:
        output_dim: Integer, final output dimension. Must be positive.
            Determines the feature space after modulation.
        count_dim: Integer, intermediate counting dimension. Must be positive.
            Controls complexity of countable features.
        counting_scope: Literal['global', 'causal', 'bidirectional'], scope of counting.
            - 'global': Zero-frequency, all tokens get same global signal
            - 'causal': Integrator sensitive to sequence history
            - 'bidirectional': Encodes position relative to entire sequence
            Defaults to 'bidirectional'.
        residual_mode: Literal['blend', 'project', 'gate_only'], integration mode.
            - 'blend': If dims match: gate*counts + (1-gate)*input
            - 'project': Always: gate*counts + linear(input)
            - 'gate_only': Pure gated counts: gate*counts
            Defaults to 'blend'.
        activation: Activation for count transformation. String name or callable.
            Common choices: 'gelu', 'relu', 'tanh'. Defaults to 'gelu'.
        use_bias: Boolean, whether to use bias in dense layers. Defaults to True.
        kernel_initializer: Initializer for kernel weights. Defaults to 'glorot_uniform'.
        bias_initializer: Initializer for bias weights. Defaults to 'zeros'.
        kernel_regularizer: Optional regularizer for kernel weights.
        bias_regularizer: Optional regularizer for bias weights.
        **kwargs: Additional arguments for Layer base class.

    Input shape:
        3D tensor with shape: `(batch_size, sequence_length, input_dim)`.

    Output shape:
        3D tensor with shape: `(batch_size, sequence_length, output_dim)`.

    Attributes:
        key_projection: Dense layer identifying countable events.
        count_transform: Dense layer transforming aggregated counts.
        gate: Dense layer controlling modulation strength.
        residual_projection: Optional Dense layer for input projection.

    Example:
        ```python
        # Standard bidirectional counting with residual
        layer = ContextualCounterFFN(
            output_dim=768,
            count_dim=128,
            counting_scope='bidirectional',
            residual_mode='blend'
        )

        # Causal counting for autoregressive models
        layer = ContextualCounterFFN(
            output_dim=512,
            count_dim=64,
            counting_scope='causal',
            residual_mode='project',
            activation='tanh'
        )

        # Pure counting features without residual
        layer = ContextualCounterFFN(
            output_dim=256,
            count_dim=32,
            counting_scope='global',
            residual_mode='gate_only'
        )
        ```

    Raises:
        ValueError: If configuration parameters are invalid.

    Note:
        The bidirectional counting scope concatenates forward and backward counts,
        doubling the feature dimension before transformation. This provides richer
        positional information but increases computation.
    """

    def __init__(
            self,
            output_dim: int,
            count_dim: int,
            counting_scope: Literal["global", "causal", "bidirectional"] = "bidirectional",
            residual_mode: Literal["blend", "project", "gate_only"] = "blend",
            activation: Union[str, callable] = "gelu",
            use_bias: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
            bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs: Any,
    ) -> None:
        """Initialize the ContextualCounterFFN layer with configuration."""
        super().__init__(**kwargs)

        # Validate configuration
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if count_dim <= 0:
            raise ValueError(f"count_dim must be positive, got {count_dim}")
        if counting_scope not in ["global", "causal", "bidirectional"]:
            raise ValueError(
                f"counting_scope must be one of 'global', 'causal', 'bidirectional', "
                f"got {counting_scope}"
            )
        if residual_mode not in ["blend", "project", "gate_only"]:
            raise ValueError(
                f"residual_mode must be one of 'blend', 'project', 'gate_only', "
                f"got {residual_mode}"
            )

        # Store configuration
        self.output_dim = output_dim
        self.count_dim = count_dim
        self.counting_scope = counting_scope
        self.residual_mode = residual_mode
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # Determine count transform input dimension
        count_transform_input_dim = (
            self.count_dim * 2 if self.counting_scope == "bidirectional"
            else self.count_dim
        )

        # CREATE all sub-layers in __init__ (following Modern Keras 3 patterns)
        self.key_projection = keras.layers.Dense(
            units=self.count_dim,
            activation="sigmoid",
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="key_projection"
        )

        self.count_transform = keras.layers.Dense(
            units=self.output_dim,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="count_transform"
        )

        self.gate = keras.layers.Dense(
            units=self.output_dim,
            activation="sigmoid",
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="gate"
        )

        # Create residual projection only if needed
        if self.residual_mode == "project":
            self.residual_projection = keras.layers.Dense(
                units=self.output_dim,
                activation=None,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name="residual_projection"
            )
        else:
            self.residual_projection = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and all sub-layers with proper shape inference.

        Following Modern Keras 3 best practices: explicitly build each sub-layer
        for robust serialization support.

        Args:
            input_shape: Shape tuple of the input tensor.

        Raises:
            ValueError: If input shape is invalid.
        """
        if self.built:
            return

        # Validate input shape
        if len(input_shape) < 2:
            raise ValueError(
                f"Input must be at least 2D, got {len(input_shape)}D: {input_shape}"
            )

        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError("Input feature dimension must be specified")

        logger.info(
            f"Building ContextualCounterFFN: "
            f"input_dim={input_dim}, output_dim={self.output_dim}, "
            f"count_dim={self.count_dim}, counting_scope='{self.counting_scope}', "
            f"residual_mode='{self.residual_mode}'"
        )

        # BUILD sub-layers explicitly (critical for serialization)
        # Build in computational order with proper shape tracking
        self.key_projection.build(input_shape)

        # Compute count transform input shape based on aggregation scope
        batch_shape = input_shape[:-1]
        if self.counting_scope == "bidirectional":
            count_transform_shape = batch_shape + (self.count_dim * 2,)
        else:
            count_transform_shape = batch_shape + (self.count_dim,)

        self.count_transform.build(count_transform_shape)
        self.gate.build(input_shape)

        if self.residual_projection is not None:
            self.residual_projection.build(input_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass implementing sense-aggregate-transform-modulate protocol.

        Args:
            inputs: Input tensor of shape (batch, sequence, features).
            training: Boolean flag for training mode, affects regularization.

        Returns:
            Output tensor of shape (batch, sequence, output_dim).
        """
        # 1. SENSE: Identify countable events through key projection
        countable_events = self.key_projection(inputs, training=training)

        # 2. AGGREGATE: Sum events according to counting scope
        if self.counting_scope == "global":
            # DC component: collapse all temporal information
            global_sum = keras.ops.sum(countable_events, axis=1, keepdims=True)
            # Broadcast to all sequence positions
            aggregated_counts = keras.ops.broadcast_to(
                global_sum,
                keras.ops.shape(countable_events)
            )

        elif self.counting_scope == "causal":
            # Low-pass filter: cumulative sum forward only
            aggregated_counts = keras.ops.cumsum(countable_events, axis=1)

        else:  # bidirectional
            # Band-pass filter: forward and backward cumulative sums
            forward_counts = keras.ops.cumsum(countable_events, axis=1)
            # Reverse cumsum for backward counts
            reversed_events = keras.ops.flip(countable_events, axis=1)
            backward_counts = keras.ops.flip(
                keras.ops.cumsum(reversed_events, axis=1),
                axis=1
            )
            # Concatenate for richer positional encoding
            aggregated_counts = keras.ops.concatenate(
                [forward_counts, backward_counts],
                axis=-1
            )

        # 3. TRANSFORM: Project aggregated counts to output dimension
        transformed_counts = self.count_transform(aggregated_counts, training=training)

        # 4. MODULATE: Apply gating and integrate with input
        gate_values = self.gate(inputs, training=training)
        gated_counts = gate_values * transformed_counts

        # Apply residual mode
        if self.residual_mode == "project":
            # Always add projected residual
            projected_input = self.residual_projection(inputs, training=training)
            output = gated_counts + projected_input

        elif self.residual_mode == "blend":
            # Check dimension compatibility
            input_dim = keras.ops.shape(inputs)[-1]
            if self.output_dim == input_dim:
                # Blend: interpolate between counts and input
                output = gated_counts + ((1 - gate_values) * inputs)
            else:
                # Fallback to gate_only when dimensions don't match
                logger.warning(
                    f"Dimension mismatch for 'blend' mode: "
                    f"output_dim={self.output_dim} != input_dim={input_dim}. "
                    f"Falling back to 'gate_only' mode."
                )
                output = gated_counts

        else:  # gate_only
            output = gated_counts

        return output

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Shape tuple of the input.

        Returns:
            Shape tuple of the output with last dimension as output_dim.
        """
        return tuple(input_shape[:-1]) + (self.output_dim,)

    def get_config(self) -> Dict[str, Any]:
        """
        Return layer configuration for serialization.

        Returns:
            Dictionary containing all configuration parameters needed
            to reconstruct this layer.
        """
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "count_dim": self.count_dim,
            "counting_scope": self.counting_scope,
            "residual_mode": self.residual_mode,
            "activation": keras.activations.serialize(self.activation),
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

# ---------------------------------------------------------------------
