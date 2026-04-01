"""
Sequence-to-sequence quantile prediction head for probabilistic forecasting.

This layer serves as the output stage for deep sequence models, transforming
encoded feature representations into quantile predictions at each time step.
Unlike horizon-based forecasting heads that predict a fixed future window,
this layer operates in a sequence-to-sequence manner, producing quantile
estimates independently for each position in the input sequence.

The architecture follows a pointwise transformation approach: each time step
in the input sequence is processed independently through the same learned
projection. This ensures no cross-temporal mixing occurs in the head
itself -- all temporal dependencies are expected to be captured by the
upstream encoder network.

When monotonicity enforcement is enabled, the layer guarantees non-crossing
quantiles (Q_i <= Q_{i+1}) at each time step by predicting the first
quantile directly and modeling subsequent quantiles as cumulative positive
deltas:

    Q_0 = raw_output[:, :, 0]
    Q_i = Q_0 + sum(Softplus(raw_output[:, :, 1:i]))  for i > 0

The layer is designed to be trained with the quantile loss (pinball loss):

    L_tau(y, y_hat) = max((y - y_hat) * tau, (y - y_hat) * (tau - 1))

where tau is the quantile level (e.g., 0.1, 0.5, 0.9).

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
    Sequence-wise quantile prediction head for probabilistic forecasting.

    Takes a sequence of encoded features and projects each time step
    independently to quantile predictions via a pointwise Dense layer,
    enabling probabilistic sequence-to-sequence modeling. The Dense layer
    is applied to each time step independently (equivalent to
    ``TimeDistributed(Dense(...))`` but more efficient), ensuring no
    cross-temporal mixing.

    When ``enforce_monotonicity=True``, the network outputs raw values
    [r_0, r_1, r_2, ...] at each time step. The final quantiles are:

        Q_0 = r_0
        Q_i = Q_0 + sum_{j=1}^{i} Softplus(r_j)  for i > 0

    This guarantees Q_0 <= Q_1 <= Q_2 <= ... at every sequence position,
    preventing "crossing quantiles".

    **Architecture Overview:**

    .. code-block:: text

        Input: [batch, seq_len, feature_dim]
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  Dropout(rate=dropout_rate)      │
        └──────────────┬───────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  Dense(num_quantiles)            │
        │  (applied pointwise per step)    │
        └──────────────┬───────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  Monotonicity Constraint         │ ← (if enforce_monotonicity=True)
        │  Q_0 = r_0                       │
        │  Q_i = Q_0 + cumsum(softplus(r)) │
        └──────────────┬───────────────────┘
                       │
                       ▼
        Output: [batch, seq_len, num_quantiles]

    :param num_quantiles: Number of quantiles to predict at each sequence
        position. Must be positive.
    :type num_quantiles: int
    :param dropout_rate: Dropout probability applied before projection.
        Defaults to 0.1.
    :type dropout_rate: float
    :param use_bias: Whether to include learnable bias terms in the
        projection layer. Defaults to True.
    :type use_bias: bool
    :param enforce_monotonicity: If True, enforces non-decreasing quantile
        predictions (Q_i <= Q_{i+1}) at each time step through a cumulative
        softplus transformation. Defaults to False.
    :type enforce_monotonicity: bool
    :param kernel_initializer: Initializer for projection weights.
        Defaults to 'glorot_uniform'.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for projection biases.
        Defaults to 'zeros'.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Regularizer applied to projection weights.
        Defaults to None.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param bias_regularizer: Regularizer applied to projection biases.
        Defaults to None.
    :type bias_regularizer: Optional[keras.regularizers.Regularizer]
    :param activity_regularizer: Regularizer applied to the output activations.
        Defaults to None.
    :type activity_regularizer: Optional[keras.regularizers.Regularizer]
    :param kernel_constraint: Constraint applied to projection weights.
        Defaults to None.
    :type kernel_constraint: Optional[keras.constraints.Constraint]
    :param bias_constraint: Constraint applied to projection biases.
        Defaults to None.
    :type bias_constraint: Optional[keras.constraints.Constraint]
    :param kwargs: Additional keyword arguments for the Layer base class.
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

        Validates the input shape and builds the projection and dropout
        sub-layers based on the input dimensions.

        :param input_shape: Shape tuple of the input tensor. Expected format
            is (batch_size, sequence_length, features).
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If input is not 3D or if the feature dimension
            is undefined.
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

        Each time step is transformed independently through the projection
        layer without cross-temporal mixing.

        :param inputs: Input tensor of shape (batch, sequence_length, features).
        :type inputs: keras.KerasTensor
        :param training: Boolean or None indicating training mode
            (affects dropout behavior).
        :type training: Optional[bool]
        :return: Quantile predictions of shape
            (batch, sequence_length, num_quantiles).
        :rtype: keras.KerasTensor
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
        # Ensures Q(tau_i) <= Q(tau_{i+1}) at each sequence position independently
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

            # Compute final monotonic quantiles: Q_i = Q_0 + sum(deltas)
            subsequent_quantiles = q0 + accumulated_deltas

            # Concatenate first quantile with subsequent monotonic quantiles
            quantiles = keras.ops.concatenate([q0, subsequent_quantiles], axis=-1)

        return quantiles

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        :param input_shape: Input shape tuple of format
            (batch_size, sequence_length, features).
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple
            (batch_size, sequence_length, num_quantiles).
        :rtype: Tuple[Optional[int], ...]
        :raises ValueError: If input shape is not 3-dimensional.
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
        """Return configuration for serialization."""
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
