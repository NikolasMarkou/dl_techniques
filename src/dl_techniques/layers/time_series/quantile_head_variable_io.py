"""
A sequence-to-sequence quantile prediction head for probabilistic forecasting.

This layer serves as the output stage for deep sequence models, transforming
encoded feature representations into quantile predictions at each time step.
Unlike horizon-based forecasting heads that predict a fixed future window,
this layer operates in a sequence-to-sequence manner, producing quantile
estimates independently for each position in the input sequence.

Architecture and Design Philosophy:
The architecture follows a pointwise transformation approach: each time step
in the input sequence is processed independently through the same learned
projection. This design consists of:

1. Optional dropout regularization applied per time step
2. A Dense layer that projects features to quantile space at each position
3. Optional monotonicity enforcement to prevent quantile crossing

This pointwise processing ensures no cross-temporal mixing occurs in the head
itself - all temporal dependencies are expected to be captured by the upstream
encoder network. The head's sole responsibility is mapping the rich encoded
representation at each time step to a distribution over quantiles.

Key Characteristics:
- **Sequence Preservation**: Input length equals output length; no temporal
  aggregation or expansion occurs
- **Independent Processing**: Each time step is transformed independently,
  equivalent to applying TimeDistributed(Dense(...)) but more efficient
- **Distribution Output**: Instead of point predictions, outputs multiple
  quantiles per time step for uncertainty quantification

Monotonicity Enforcement:
When enabled, the layer guarantees non-crossing quantiles (Q_i ≤ Q_{i+1}) at
each time step by predicting the first quantile directly and modeling
subsequent quantiles as cumulative positive deltas:

    Q_0 = raw_output[:, :, 0]
    Q_i = Q_0 + Σ(Softplus(raw_output[:, :, 1:i]))  for i > 0

This approach is mathematically sound and maintains differentiability while
ensuring valid quantile orderings throughout training and inference.

Training Objective:
This layer is designed to be trained with the quantile loss (pinball loss):

    L_τ(y, ŷ) = max((y - ŷ) * τ, (y - ŷ) * (τ - 1))

where τ is the quantile level (e.g., 0.1, 0.5, 0.9) and ŷ are the predicted
quantiles. The loss is asymmetric, penalizing over-predictions and
under-predictions differently based on the target quantile level.

Use Cases:
- Sequence labeling with uncertainty (e.g., predicting quantiles of sensor
  readings at each time step)
- Probabilistic sequence-to-sequence forecasting where each output corresponds
  to an input position
- Multi-horizon forecasting when combined with appropriate encoders that
  produce future-aligned representations

References:
    - Koenker, R., & Bassett Jr, G. (1978). Regression Quantiles.
      Econometrica, 46(1), 33-50.
      https://www.jstor.org/stable/1913643
    - Rodrigues, F., & Pereira, F. C. (2020). Beyond expectation:
      Deep joint mean and quantile regression for spatiotemporal problems.
      IEEE Transactions on Neural Networks and Learning Systems.
"""

import keras
from typing import Optional, Union, Tuple, Any

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class QuantileSequenceHead(keras.layers.Layer):
    """
    Sequence-wise quantile prediction head for probabilistic time series forecasting.

    This layer implements a neural network head for predicting multiple quantiles
    at each time step in a sequence. It takes a sequence of encoded features and
    projects each time step independently to quantile predictions, enabling
    probabilistic sequence-to-sequence modeling.

    **Intent**: Enable probabilistic sequence-to-sequence forecasting by predicting
    multiple quantiles (e.g., 10th, 50th, 90th percentiles) at each sequence position.

    **Architecture**:
    ```
    Input(shape=[batch, seq, feature_dim])
           ↓
    Dropout(rate=dropout_rate)  ← optional, applied per time step
           ↓
    Dense(num_quantiles)  ← applied pointwise to each time step
           ↓
    Monotonicity Constraint ← optional, enforces Q_i ≤ Q_{i+1}
           ↓
    Output(shape=[batch, seq, num_quantiles])
    ```

    **Pointwise Processing**:
    The Dense layer is applied to each time step independently, ensuring no
    cross-temporal mixing. This is mathematically equivalent to:
    ```python
    for t in range(seq_len):
        output[:, t, :] = Dense(input[:, t, :])
    ```
    but implemented efficiently in a single vectorized operation.

    **Monotonicity Logic**:
    When `enforce_monotonicity=True`, the network outputs raw values [r_0, r_1, r_2, ...]
    at each time step. The final quantiles are calculated as:

    - Q_0 = r_0
    - Q_i = Q_0 + Σ_{j=1}^{i} Softplus(r_j)  for i > 0

    This guarantees Q_0 ≤ Q_1 ≤ Q_2 ≤ ... at every sequence position, preventing
    "crossing quantiles" which would violate the mathematical definition of quantiles.

    Args:
        num_quantiles: Integer, number of quantiles to predict simultaneously
            at each sequence position. Must be positive.
        dropout_rate: Float in [0, 1]. Dropout probability applied before projection.
            Helps prevent overfitting. Defaults to 0.1.
        use_bias: Boolean, whether to include learnable bias terms in the
            projection layer. Defaults to True.
        enforce_monotonicity: Boolean. If True, enforces that quantile predictions
            are strictly non-decreasing (Q_i ≤ Q_{i+1}) at each time step through
            a cumulative softplus transformation. Defaults to False.
        kernel_initializer: Initializer for projection weights.
            Defaults to 'glorot_uniform'.
        bias_initializer: Initializer for projection biases.
            Defaults to 'zeros'.
        kernel_regularizer: Regularizer applied to projection weights.
            Defaults to None.
        bias_regularizer: Regularizer applied to projection biases.
            Defaults to None.
        activity_regularizer: Regularizer applied to the output activations.
            Defaults to None.
        kernel_constraint: Constraint applied to projection weights.
            Defaults to None.
        bias_constraint: Constraint applied to projection biases.
            Defaults to None.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        3D tensor with shape: `(batch_size, sequence_length, features)`.

    Output shape:
        3D tensor with shape: `(batch_size, sequence_length, num_quantiles)`.

    Example:
        ```python
        # Create a sequence quantile head for probabilistic predictions
        quantile_head = QuantileSequenceHead(
            num_quantiles=3,  # Predict 10th, 50th, 90th percentiles
            dropout_rate=0.1,
            enforce_monotonicity=True
        )

        # Input: encoded sequence features from upstream encoder
        inputs = keras.Input(shape=(100, 512))  # 100 time steps, 512 features
        outputs = quantile_head(inputs)  # Shape: (batch, 100, 3)

        # Each output[:, t, :] contains [q_0.1, q_0.5, q_0.9] for time step t
        # These represent the 10th, 50th, and 90th percentile predictions

        # Build a complete model
        model = keras.Model(inputs, outputs)

        # Train with quantile loss for each quantile level
        # (implementation of quantile loss not shown)
        ```

    Note:
        This layer does NOT implement the quantile loss function. The loss must be
        implemented separately and provided during model compilation. A typical
        quantile loss implementation uses asymmetric penalties based on the
        quantile levels being predicted.
    """

    def __init__(
            self,
            num_quantiles: int,
            dropout_rate: float = 0.1,
            use_bias: bool = True,
            enforce_monotonicity: bool = False,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
            bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
            activity_regularizer: Optional[keras.regularizers.Regularizer] = None,
            kernel_constraint: Optional[keras.constraints.Constraint] = None,
            bias_constraint: Optional[keras.constraints.Constraint] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if num_quantiles <= 0:
            raise ValueError(f"num_quantiles must be positive, got {num_quantiles}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be in [0, 1], got {dropout_rate}")

        # Store configuration
        self.num_quantiles = num_quantiles
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.enforce_monotonicity = enforce_monotonicity
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = keras.regularizers.get(activity_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)

        # Initialize sub-layers following Keras 3 best practices
        # The Dense layer will be applied pointwise to each time step
        self.projection = keras.layers.Dense(
            units=self.num_quantiles,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            name="quantile_projection"
        )

        # Optional dropout for regularization
        if self.dropout_rate > 0.0:
            self.dropout = keras.layers.Dropout(
                rate=self.dropout_rate,
                name="quantile_dropout"
            )
        else:
            self.dropout = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and initialize all sub-layer weights.

        This method validates the input shape and builds the projection and
        dropout sub-layers based on the input dimensions.

        Args:
            input_shape: Tuple representing the input shape. Expected format
                is (batch_size, sequence_length, features) where batch_size
                may be None but sequence_length and features should be defined.

        Raises:
            ValueError: If input is not 3D or if feature dimension is undefined.
        """
        # Validate input dimensionality
        if len(input_shape) != 3:
            raise ValueError(
                f"QuantileSequenceHead expects a 3D input tensor "
                f"(batch, sequence, features), but received shape {input_shape}."
            )

        batch_size, seq_len, features = input_shape

        # Ensure feature dimension is defined for weight initialization
        if features is None:
            raise ValueError(
                "The feature dimension must be defined (not None) to build "
                "the projection layer weights. "
                f"Received shape: {input_shape}"
            )

        # Build sub-layers with validated input shape
        # Dropout layer preserves input shape
        if self.dropout is not None:
            self.dropout.build(input_shape)

        # Dense layer in Keras 3 handles 3D inputs naturally, applying
        # transformation to the last axis while preserving batch and sequence dims
        self.projection.build(input_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Predict quantiles for each time step in the input sequence.

        The processing is performed pointwise - each time step is transformed
        independently through the projection layer without any cross-temporal
        mixing. This ensures the output preserves the sequence structure.

        Args:
            inputs: Input tensor of shape (batch_size, sequence_length, features).
                Typically the output of an encoder network that has captured
                temporal dependencies.
            training: Boolean or None. Indicates whether the layer is in training
                mode (affects dropout behavior). If None, defaults to the global
                Keras learning phase.

        Returns:
            Quantile predictions tensor of shape (batch_size, sequence_length, num_quantiles).
            Each position [batch, time_step, :] contains num_quantiles predicted
            quantile values for that time step.
        """
        x = inputs

        # Step 1: Apply dropout regularization (per time step, per feature)
        if self.dropout is not None:
            x = self.dropout(x, training=training)

        # Step 2: Project to quantile space pointwise
        # Keras Dense layer naturally handles 3D inputs by applying the
        # transformation to the last axis: (batch, seq, features) -> (batch, seq, num_quantiles)
        # This is equivalent to TimeDistributed(Dense) but more efficient
        quantiles = self.projection(x, training=training)
        # Shape: (batch, seq_len, num_quantiles)

        # Step 3: Enforce monotonicity constraint if enabled
        # Ensures Q(tau_i) ≤ Q(tau_{i+1}) at each sequence position independently
        if self.enforce_monotonicity and self.num_quantiles > 1:
            # Extract first quantile (predicted directly)
            # Shape: (batch, seq_len, 1)
            q0 = quantiles[:, :, 0:1]

            # Extract remaining raw values (interpreted as deltas)
            # Shape: (batch, seq_len, num_quantiles - 1)
            rest = quantiles[:, :, 1:]

            # Force deltas to be non-negative using softplus activation
            # Softplus is smooth and differentiable: softplus(x) = log(1 + exp(x))
            # This ensures that all subsequent quantiles are >= the first quantile
            deltas = keras.ops.softplus(rest)

            # Accumulate deltas along the quantile dimension
            # cumsum ensures Q_i = Q_0 + delta_1 + delta_2 + ... + delta_i
            # Each time step's quantiles are accumulated independently
            accumulated_deltas = keras.ops.cumsum(deltas, axis=-1)

            # Compute final monotonic quantiles: Q_i = Q_0 + Σ(deltas)
            subsequent_quantiles = q0 + accumulated_deltas

            # Concatenate first quantile with subsequent monotonic quantiles
            quantiles = keras.ops.concatenate([q0, subsequent_quantiles], axis=-1)

        return quantiles

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer given an input shape.

        This is used by Keras for shape inference during model construction,
        enabling automatic shape validation and model summary generation.

        Args:
            input_shape: Tuple representing the input shape. Expected format
                is (batch_size, sequence_length, features).

        Returns:
            Output shape tuple: (batch_size, sequence_length, num_quantiles).
            The batch and sequence dimensions are preserved; only the feature
            dimension is transformed to the number of quantiles.

        Raises:
            ValueError: If input shape is not 3-dimensional.
        """
        if len(input_shape) != 3:
            raise ValueError(
                f"Expected 3D input shape, got {len(input_shape)}D: {input_shape}"
            )

        batch_size = input_shape[0]
        seq_len = input_shape[1]
        # Feature dimension is replaced by num_quantiles
        return (batch_size, seq_len, self.num_quantiles)

    def get_config(self) -> dict[str, Any]:
        """
        Return configuration dictionary for serialization.

        This method enables the layer to be saved and loaded using Keras
        model serialization (save/load), preserving all hyperparameters
        and configuration.

        Returns:
            Configuration dictionary containing all layer parameters including
            initializers, regularizers, and constraints in serialized form.
        """
        config = super().get_config()
        config.update({
            "num_quantiles": self.num_quantiles,
            "dropout_rate": self.dropout_rate,
            "use_bias": self.use_bias,
            "enforce_monotonicity": self.enforce_monotonicity,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": keras.regularizers.serialize(self.activity_regularizer),
            "kernel_constraint": keras.constraints.serialize(self.kernel_constraint),
            "bias_constraint": keras.constraints.serialize(self.bias_constraint),
        })
        return config


# ---------------------------------------------------------------------
