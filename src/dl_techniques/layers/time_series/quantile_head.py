"""
A quantile prediction head for probabilistic forecasting.

This layer serves as the final output stage for a deep forecasting model,
transforming a latent feature representation into a set of quantile
predictions for a future time horizon. Its purpose is to enable
probabilistic forecasting, which moves beyond single-point predictions to
provide a richer, uncertainty-aware view of the future.

Architecture and Design Philosophy:
The architecture is intentionally simple: a linear projection from the encoder's
feature space to the target space defined by the quantiles and the forecast
horizon. It consists of a single `Dense` layer that maps the input features
to a flat vector of size `output_length * num_quantiles`, followed by a
reshape operation to structure the output.

This design assumes that the upstream encoder network is responsible for
extracting all necessary complex, non-linear patterns from the input time
series. This head then acts as a simple, learnable mapping from that rich
representation to the parameters of the forecast distribution.

Enhanced Configurations:
1. Flatten Input: Optionally flattens the input sequence (Batch, Seq, Dim) into
   (Batch, Seq*Dim) before projection. This allows the dense layer to utilize
   the specific temporal order of features rather than pooling them.
2. Enforce Monotonicity: Optionally enforces non-crossing quantiles (Q1 <= Q2 <= Q3)
   by predicting the first quantile and positive deltas for subsequent levels.

Foundational Mathematics:
This layer is designed to be trained with a quantile loss function ("pinball loss").
L_τ(y, ŷ) = max((y - ŷ) * τ, (y - ŷ) * (τ - 1))

References:
    - [Koenker, R., & Bassett Jr, G. (1978). Regression Quantiles.
      Econometrica.](https://www.jstor.org/stable/1913643)
"""

import keras
from keras import ops
from typing import Optional, Union, Tuple, Any

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class QuantileHead(keras.layers.Layer):
    """
    Quantile prediction head for probabilistic time series forecasting.

    This layer implements a neural network head for predicting multiple quantiles
    of a time series distribution. It takes encoded features and projects them 
    to quantile predictions across a specified forecast horizon.

    **Intent**: Enable probabilistic time series forecasting by predicting multiple
    quantiles (e.g., 10th, 50th, 90th percentiles) simultaneously.

    **Architecture**:
    ```
    Input(shape=[batch, seq, feature_dim])
           ↓
    Reshape(shape=[batch, -1]) ← (if flatten_input=True)
           ↓
    Dropout(rate=dropout_rate)
           ↓
    Dense(output_length × num_quantiles)
           ↓
    Reshape(shape=[batch, output_length, num_quantiles])
           ↓
    Monotonicity Constraint ← (if enforce_monotonicity=True)
    ```

    **Monotonicity Logic**:
    If enabled, the network outputs raw values [r_0, r_1, r_2...].
    The final quantiles are calculated as:
    Q_0 = r_0
    Q_i = Q_{i-1} + Softplus(r_i)  (for i > 0)
    This guarantees Q_0 <= Q_1 <= Q_2, preventing "crossing quantiles".

    Args:
        num_quantiles: Integer, number of quantiles to predict simultaneously.
        output_length: Integer, length of the forecast horizon.
        dropout_rate: Float between 0 and 1. Defaults to 0.1.
        use_bias: Boolean, whether to include learnable bias terms. Defaults to True.
        flatten_input: Boolean. If True, the input tensor is flattened (preserving batch)
            before the dense projection. This allows the head to learn from the full
            sequence history rather than a pooled representation. Defaults to False.
            Note: Requires fixed sequence length.
        enforce_monotonicity: Boolean. If True, enforces that quantile predictions
            are strictly non-decreasing (Q_i <= Q_{i+1}). Requires input quantile_levels
            to be sorted. Defaults to False.
        kernel_initializer: Initializer for projection weights. Defaults to 'glorot_uniform'.
        bias_initializer: Initializer for projection biases. Defaults to 'zeros'.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        If flatten_input=True: 3D tensor `(batch_size, seq_len, features)`.
        If flatten_input=False: 2D tensor `(batch_size, features)`.

    Output shape:
        3D tensor with shape: `(batch_size, output_length, num_quantiles)`.
    """

    def __init__(
        self,
        num_quantiles: int,
        output_length: int,
        dropout_rate: float = 0.1,
        use_bias: bool = True,
        flatten_input: bool = False,
        enforce_monotonicity: bool = False,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if num_quantiles <= 0:
            raise ValueError(f"num_quantiles must be positive, got {num_quantiles}")
        if output_length <= 0:
            raise ValueError(f"output_length must be positive, got {output_length}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")

        # Store configuration
        self.num_quantiles = num_quantiles
        self.output_length = output_length
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.flatten_input = flatten_input
        self.enforce_monotonicity = enforce_monotonicity
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)

        # CREATE all sub-layers in __init__ (following modern Keras 3 pattern)
        self.projection = keras.layers.Dense(
            units=self.output_length * self.num_quantiles,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            name="quantile_projection"
        )

        if self.dropout_rate > 0.0:
            self.dropout = keras.layers.Dropout(
                rate=self.dropout_rate,
                name="quantile_dropout"
            )
        else:
            self.dropout = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and all its sub-layers.
        """
        # Handle logical reshaping for the build process
        # If flattening is enabled, the Dense layer needs to see the flattened dimension
        dense_input_shape = input_shape
        
        if self.flatten_input:
            # Expecting (Batch, Seq, Feat)
            if len(input_shape) != 3:
                raise ValueError(
                    f"flatten_input=True expects a 3D input tensor (Batch, Seq, Feat), "
                    f"but received shape {input_shape}."
                )
            
            seq_len = input_shape[-2]
            features = input_shape[-1]
            
            # Dense layer weights depend on a fixed input dimension.
            if features is None or seq_len is None:
                raise ValueError(
                    "flatten_input=True requires both sequence length and feature dimension "
                    "to be defined (not None) to build the projection layer weights. "
                    f"Received shape: {input_shape}"
                )

            flat_dim = features * seq_len
            dense_input_shape = (input_shape[0], flat_dim)

        # Build sub-layers
        if self.dropout is not None:
            self.dropout.build(dense_input_shape)

        self.projection.build(dense_input_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Predict quantiles from the input feature vector.

        Args:
            inputs: Input tensor.
            training: Boolean indicating training mode for dropout.

        Returns:
            Quantile predictions tensor of shape (batch_size, output_length, num_quantiles).
        """
        x = inputs

        # 1. FLATTEN (Configuration Option)
        if self.flatten_input:
            input_shape = ops.shape(x)
            batch_size = input_shape[0]
            # Reshape to (Batch, Seq*Dim) using -1 to infer dimension
            x = ops.reshape(x, (batch_size, -1))

        # 2. DROPOUT
        if self.dropout is not None:
            x = self.dropout(x, training=training)

        # 3. PROJECTION
        # Project features to flattened quantile predictions
        quantile_preds = self.projection(x, training=training)

        # 4. RESHAPE OUTPUT
        # Reshape to [batch_size, output_length, num_quantiles]
        # Using -1 for batch dimension handles dynamic batch sizes and symbolic tensors
        quantiles = ops.reshape(
            quantile_preds,
            (-1, self.output_length, self.num_quantiles)
        )

        # 5. MONOTONICITY (Configuration Option)
        # Ensures Q(tau_i) <= Q(tau_{i+1})
        if self.enforce_monotonicity and self.num_quantiles > 1:
            # Split the first quantile from the rest
            # q0: (Batch, Len, 1)
            q0 = quantiles[:, :, 0:1]
            
            # The rest are interpreted as deltas
            # rest: (Batch, Len, num_quantiles - 1)
            rest = quantiles[:, :, 1:]
            
            # Force deltas to be positive using softplus
            deltas = ops.softplus(rest)
            
            # Accumulate deltas
            accumulated_deltas = ops.cumsum(deltas, axis=-1)
            
            # Add base to accumulation
            subsequent_quantiles = q0 + accumulated_deltas
            
            # Recombine
            quantiles = ops.concatenate([q0, subsequent_quantiles], axis=-1)

        return quantiles

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer."""
        batch_size = input_shape[0]
        return (batch_size, self.output_length, self.num_quantiles)

    def get_config(self) -> dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            "num_quantiles": self.num_quantiles,
            "output_length": self.output_length,
            "dropout_rate": self.dropout_rate,
            "use_bias": self.use_bias,
            "flatten_input": self.flatten_input,
            "enforce_monotonicity": self.enforce_monotonicity,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
        })
        return config


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class QuantileSequenceHead(keras.layers.Layer):
    """
    Sequence-wise quantile prediction head for probabilistic time series forecasting.

    This layer implements a neural network head for predicting multiple quantiles
    at each time step in a sequence. It takes a sequence of encoded features and
    projects each time step independently to quantile predictions.

    **Intent**: Enable probabilistic sequence-to-sequence forecasting by predicting
    multiple quantiles (e.g., 10th, 50th, 90th percentiles) at each sequence position.

    **Architecture**:
    ```
    Input(shape=[batch, seq, feature_dim])
           ↓
    Dropout(rate=dropout_rate)  ← (applied per time step)
           ↓
    Dense(num_quantiles)  ← (applied independently to each time step)
           ↓
    Output(shape=[batch, seq, num_quantiles])
           ↓
    Monotonicity Constraint ← (if enforce_monotonicity=True)
    ```

    **Pointwise Processing**:
    The Dense layer is applied to each time step independently, ensuring no
    cross-temporal mixing. This is equivalent to:
    ```python
    for t in range(seq_len):
        output[:, t, :] = Dense(input[:, t, :])
    ```

    **Monotonicity Logic**:
    If enabled, at each time step the network outputs raw values [r_0, r_1, r_2...].
    The final quantiles are calculated as:
    Q_0 = r_0
    Q_i = Q_{i-1} + Softplus(r_i)  (for i > 0)
    This guarantees Q_0 <= Q_1 <= Q_2, preventing "crossing quantiles" at each position.

    Args:
        num_quantiles: Integer, number of quantiles to predict simultaneously
            at each sequence position.
        dropout_rate: Float between 0 and 1. Dropout applied before projection.
            Defaults to 0.1.
        use_bias: Boolean, whether to include learnable bias terms in the
            projection layer. Defaults to True.
        enforce_monotonicity: Boolean. If True, enforces that quantile predictions
            are strictly non-decreasing (Q_i <= Q_{i+1}) at each time step.
            Requires input quantile_levels to be sorted. Defaults to False.
        kernel_initializer: Initializer for projection weights.
            Defaults to 'glorot_uniform'.
        bias_initializer: Initializer for projection biases.
            Defaults to 'zeros'.
        kernel_regularizer: Regularizer for projection weights.
            Defaults to None.
        bias_regularizer: Regularizer for projection biases.
            Defaults to None.
        activity_regularizer: Regularizer for the output.
            Defaults to None.
        kernel_constraint: Constraint for projection weights.
            Defaults to None.
        bias_constraint: Constraint for projection biases.
            Defaults to None.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        3D tensor with shape: `(batch_size, sequence_length, features)`.

    Output shape:
        3D tensor with shape: `(batch_size, sequence_length, num_quantiles)`.

    Example:
        ```python
        # Create a sequence quantile head
        quantile_head = QuantileSequenceHead(
            num_quantiles=3,  # Predict 10th, 50th, 90th percentiles
            dropout_rate=0.1,
            enforce_monotonicity=True
        )

        # Input: encoded sequence features
        inputs = keras.Input(shape=(100, 512))  # 100 time steps, 512 features
        outputs = quantile_head(inputs)  # Shape: (batch, 100, 3)

        # Each output[:, t, :] contains 3 quantiles for time step t
        ```
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
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")

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

        # CREATE all sub-layers in __init__ (following modern Keras 3 pattern)
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

        if self.dropout_rate > 0.0:
            self.dropout = keras.layers.Dropout(
                rate=self.dropout_rate,
                name="quantile_dropout"
            )
        else:
            self.dropout = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and all its sub-layers.

        Args:
            input_shape: Expected to be (batch_size, sequence_length, features).
        """
        # Validate input shape
        if len(input_shape) != 3:
            raise ValueError(
                f"QuantileSequenceHead expects a 3D input tensor "
                f"(batch, sequence, features), but received shape {input_shape}."
            )

        batch_size, seq_len, features = input_shape

        # Ensure features dimension is defined for building Dense layer
        if features is None:
            raise ValueError(
                "The feature dimension must be defined (not None) to build "
                "the projection layer weights. "
                f"Received shape: {input_shape}"
            )

        # Build sub-layers
        # Dropout preserves shape, so same input shape
        if self.dropout is not None:
            self.dropout.build(input_shape)

        # Dense layer will be applied to the last dimension
        # In Keras 3, Dense handles 3D inputs naturally by applying to the last axis
        self.projection.build(input_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Predict quantiles for each time step in the input sequence.

        The processing is done pointwise - each time step is transformed
        independently without any cross-temporal mixing.

        Args:
            inputs: Input tensor of shape (batch_size, sequence_length, features).
            training: Boolean indicating training mode for dropout.

        Returns:
            Quantile predictions tensor of shape (batch_size, sequence_length, num_quantiles).
        """
        x = inputs

        # 1. DROPOUT (applied per time step)
        if self.dropout is not None:
            x = self.dropout(x, training=training)

        # 2. POINTWISE PROJECTION
        # Dense layer in Keras 3 naturally handles 3D inputs by applying
        # the transformation to the last axis while preserving other dimensions
        # This is equivalent to TimeDistributed(Dense(...)) but more efficient
        quantiles = self.projection(x, training=training)
        # Shape: (batch, seq_len, num_quantiles)

        # 3. MONOTONICITY CONSTRAINT (per time step)
        # Ensures Q(tau_i) <= Q(tau_{i+1}) at each sequence position
        if self.enforce_monotonicity and self.num_quantiles > 1:
            # Split the first quantile from the rest
            # q0: (batch, seq_len, 1)
            q0 = quantiles[:, :, 0:1]

            # The rest are interpreted as deltas
            # rest: (batch, seq_len, num_quantiles - 1)
            rest = quantiles[:, :, 1:]

            # Force deltas to be positive using softplus
            # This ensures monotonicity at each time step independently
            deltas = keras.ops.softplus(rest)

            # Accumulate deltas along the quantile dimension
            # Each time step's quantiles are accumulated independently
            accumulated_deltas = keras.ops.cumsum(deltas, axis=-1)

            # Add base quantile to accumulated deltas
            subsequent_quantiles = q0 + accumulated_deltas

            # Recombine first quantile with subsequent ones
            quantiles = keras.ops.concatenate([q0, subsequent_quantiles], axis=-1)

        return quantiles

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Expected to be (batch_size, sequence_length, features).

        Returns:
            Output shape tuple: (batch_size, sequence_length, num_quantiles).
        """
        if len(input_shape) != 3:
            raise ValueError(
                f"Expected 3D input shape, got {len(input_shape)}D: {input_shape}"
            )

        batch_size = input_shape[0]
        seq_len = input_shape[1]
        return (batch_size, seq_len, self.num_quantiles)

    def get_config(self) -> dict[str, Any]:
        """
        Return configuration for serialization.

        Returns:
            Configuration dictionary containing all layer parameters.
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

