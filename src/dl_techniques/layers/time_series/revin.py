"""
A Reversible Instance Normalization for time series forecasting.

This layer addresses the challenge of distribution shift in time series
forecasting, a common problem where the statistical properties (e.g., mean,
variance) of the time series change over time. Models trained on past data
may perform poorly when faced with future data from a different distribution.

Architecture and Design Philosophy:
RevIN tackles this problem by decoupling the core forecasting task from the
learning of instance-specific statistics. It operates as a stateful,
two-part module: a normalization forward pass and a denormalization reverse
pass.

1.  **Normalization (Forward Pass)**: Before the main forecasting model
    processes the input time series, RevIN normalizes each instance in the
    batch independently. It calculates the mean and standard deviation along
    the time dimension for each individual series and standardizes it to have
    a zero mean and unit variance. The key insight is that this makes the
    task for the downstream model stationary; it always sees input sequences
    with a consistent distribution, regardless of the original scale or
    level of the series. Crucially, the layer stores the computed mean and
    standard deviation for each instance.

2.  **Denormalization (Reverse Pass)**: After the forecasting model produces
    its predictions in the normalized space, the `denormalize` method is
    called. This method uses the stored statistics from the corresponding
    input instance to reverse the normalization process, projecting the
    forecast back into the original scale and level of the input series.

This reversible design ensures that the model can focus on learning temporal
patterns without being confounded by shifts in magnitude or level, while the
final forecast remains correctly scaled and interpretable.

Foundational Mathematics:
The core of RevIN is instance normalization applied to time series. For each
time series instance `xᵢ` in a batch, with `i` indexing the sample:
-   **Statistics Computation**: The mean `μᵢ` and standard deviation `σᵢ` are
    calculated across the time-step dimension of `xᵢ`.
-   **Normalization**: The input to the forecasting model is `x̂ᵢ = (xᵢ - μᵢ) / (σᵢ + ε)`.
-   **Denormalization**: If the model produces a forecast `ŷᵢ`, the final output
    is restored via `yᵢ = ŷᵢ * (σᵢ + ε) + μᵢ`.

An optional learnable affine transformation (`yᵢ = γ * x̂ᵢ + β`) can be
applied after normalization, allowing the model to learn an optimal data
scale for its internal processing. This transformation is also reversed during
denormalization. By making the normalization instance-specific and perfectly
reversible, RevIN preserves the unique statistical identity of each time
series while standardizing the learning problem for the model.

References:
    - [Kim, T., Kim, J., Tae, Y., Park, C., Choi, J. H., & Choo, J. (2022).
      Reversible Instance Normalization for Accurate Time-Series
      Forecasting against Distribution Shift. In ICLR.](
      https://arxiv.org/abs/2107.03445)
"""

import keras
from typing import Optional, Union, Tuple, Any, Dict

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class RevIN(keras.layers.Layer):
    """
    Reversible Instance Normalization for Time-Series Forecasting.

    RevIN is a normalization method that addresses distribution shift problems
    in time-series forecasting by removing and restoring statistical information
    of time-series instances. It computes statistics per instance and provides
    both normalization and denormalization capabilities for maintaining scale
    consistency between input and output time series.

    **Intent**: Enable robust time-series forecasting by normalizing input sequences
    to have zero mean and unit variance per instance, while preserving the ability
    to denormalize predictions back to the original scale. This is particularly
    valuable for handling distribution shifts between training and inference.

    **Architecture & Process**:
    ```
    Input(shape=[batch, seq_len, features])
           ↓
    Compute Statistics: μ = mean(x), σ = std(x) per instance
           ↓
    Normalize: x_norm = (x - μ) / σ
           ↓
    Affine Transform: output = γ * x_norm + β (if affine=True)
           ↓
    [Model Processing...]
           ↓
    Denormalize: x_orig = (pred - β) / γ * σ + μ
           ↓
    Output(shape=[batch, seq_len, features])
    ```

    **Mathematical Operations**:
    1. **Statistics Computation**: μᵢ = mean(xᵢ), σᵢ = std(xᵢ) for each instance i
    2. **Normalization**: x̂ᵢ = (xᵢ - μᵢ) / σᵢ
    3. **Affine Transform**: yᵢ = γ ⊙ x̂ᵢ + β (element-wise, if affine=True)
    4. **Denormalization**: x'ᵢ = (y'ᵢ - β) / γ * σᵢ + μᵢ

    Where:
    - i indexes individual instances in the batch
    - μᵢ, σᵢ are per-instance statistics computed over sequence length
    - γ (scale) and β (shift) are learnable parameters (if affine=True)
    - ⊙ denotes element-wise multiplication

    Args:
        num_features: Integer, number of features/channels in the input time series.
            Must be positive and should match the last dimension of input tensors.
        eps: Float, small value added to standard deviation for numerical stability.
            Prevents division by zero when variance is very small. Must be positive.
            Defaults to 1e-5.
        affine: Boolean, whether to apply learnable affine transformation after
            normalization. When True, adds learnable scale (γ) and shift (β)
            parameters. Defaults to True.
        affine_weight_initializer: Initializer for the affine weight parameter γ.
            Only used when affine=True. Defaults to "ones".
        affine_bias_initializer: Initializer for the affine bias parameter β.
            Only used when affine=True. Defaults to "zeros".
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        3D tensor with shape: `(batch_size, sequence_length, num_features)`.

    Output shape:
        3D tensor with shape: `(batch_size, sequence_length, num_features)`.
        Shape is preserved through the normalization process.

    Attributes:
        affine_weight: Scale parameter γ of shape (num_features,) if affine=True, else None.
        affine_bias: Bias parameter β of shape (num_features,) if affine=True, else None.

    Methods:
        denormalize: Reverse the normalization using stored statistics from last forward pass.

    Example:
        ```python
        import keras
        import numpy as np

        # Create RevIN layer for 10 features
        revin = RevIN(num_features=10)

        # Sample input: (batch=32, seq_len=100, features=10)
        x_input = keras.random.normal((32, 100, 10))

        # Normalize input (stores statistics internally)
        x_norm = revin(x_input)

        # Pass through forecasting model
        model = keras.Sequential([
            keras.layers.LSTM(64, return_sequences=True),
            keras.layers.Dense(10)
        ])
        predictions = model(x_norm)

        # Denormalize predictions to restore original scale
        predictions_denorm = revin.denormalize(predictions)

        # Without affine transformation
        revin_simple = RevIN(num_features=5, affine=False)

        # Custom initialization for specific use cases
        revin_custom = RevIN(
            num_features=20,
            eps=1e-6,
            affine_weight_initializer='he_normal',
            affine_bias_initializer='normal'
        )
        ```

    References:
        - Kim et al., "Reversible Instance Normalization for Accurate Time-Series
          Forecasting against Distribution Shift", ICLR 2022.

    Raises:
        ValueError: If num_features is not positive.
        ValueError: If eps is not positive.
        ValueError: If denormalize is called before statistics are computed.

    Note:
        The layer stores normalization statistics (mean and std) computed from
        the input during the forward pass. These statistics persist until the
        next forward pass, allowing for consistent denormalization of predictions.
        This makes RevIN particularly suitable for time series forecasting where
        scale preservation is critical.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = True,
        affine_weight_initializer: Union[str, keras.initializers.Initializer] = "ones",
        affine_bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if num_features <= 0:
            raise ValueError(f"num_features must be positive, got {num_features}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")

        # Store configuration
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.affine_weight_initializer = keras.initializers.get(affine_weight_initializer)
        self.affine_bias_initializer = keras.initializers.get(affine_bias_initializer)

        # Weight attributes (created in build)
        self.affine_weight = None
        self.affine_bias = None

        # Statistics storage (set during forward pass)
        self._mean = None
        self._stdev = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Create the layer's learnable parameters.

        Args:
            input_shape: Shape tuple of the input tensor.

        Raises:
            ValueError: If input shape is invalid.
        """
        # Validate input shape
        if len(input_shape) != 3:
            raise ValueError(
                f"RevIN expects 3D input (batch, seq_len, features), got shape {input_shape}"
            )

        if input_shape[-1] is None:
            raise ValueError("Last dimension (num_features) of input must be defined")

        if input_shape[-1] != self.num_features:
            raise ValueError(
                f"Input features ({input_shape[-1]}) must match num_features ({self.num_features})"
            )

        # Create affine parameters if enabled
        if self.affine:
            self.affine_weight = self.add_weight(
                name="affine_weight",
                shape=(self.num_features,),
                initializer=self.affine_weight_initializer,
                trainable=True,
            )

            self.affine_bias = self.add_weight(
                name="affine_bias",
                shape=(self.num_features,),
                initializer=self.affine_bias_initializer,
                trainable=True,
            )

        super().build(input_shape)

    def _get_statistics(self, x: keras.KerasTensor) -> None:
        """
        Compute mean and standard deviation statistics for the input.

        This method computes instance-wise statistics by reducing over the
        sequence length dimension while preserving batch and feature dimensions.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, num_features).
        """
        # For 3D input (batch, seq_len, features), reduce over seq_len (axis=1)
        # This computes per-instance statistics across the sequence dimension
        self._mean = keras.ops.mean(x, axis=1, keepdims=True)
        variance = keras.ops.var(x, axis=1, keepdims=True)
        self._stdev = keras.ops.sqrt(variance + self.eps)

    def _normalize(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """
        Apply normalization with optional affine transformation.

        Args:
            x: Input tensor to normalize.

        Returns:
            Normalized tensor.
        """
        # Subtract mean and divide by standard deviation
        x_norm = (x - self._mean) / self._stdev

        # Apply learnable affine transformation if enabled
        if self.affine:
            x_norm = x_norm * self.affine_weight + self.affine_bias

        return x_norm

    def _denormalize(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """
        Reverse the normalization process.

        Args:
            x: Input tensor to denormalize.

        Returns:
            Denormalized tensor with original scale and offset restored.

        Raises:
            ValueError: If statistics have not been computed yet.
        """
        if self._mean is None or self._stdev is None:
            raise ValueError(
                "Cannot denormalize: statistics not computed. "
                "Call the layer with input data first to compute statistics."
            )

        # Reverse affine transformation if enabled
        if self.affine:
            # Reverse: x = (x_norm - bias) / weight
            x = (x - self.affine_bias) / self.affine_weight

        # Reverse normalization: multiply by std and add mean
        x = x * self._stdev + self._mean

        return x

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply RevIN normalization.

        This method computes statistics from the input and applies normalization.
        The computed statistics are stored internally for later denormalization.

        Args:
            inputs: Input tensor of shape (batch_size, seq_len, num_features).
            training: Boolean indicating whether in training mode. Not used but
                     kept for API consistency.

        Returns:
            Normalized tensor with same shape as input.
        """
        # Compute and store statistics for current input
        self._get_statistics(inputs)

        # Apply normalization
        return self._normalize(inputs)

    def denormalize(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """
        Apply RevIN denormalization.

        This method reverses the normalization applied by the forward pass,
        using the statistics computed during the last call to the layer.

        Args:
            inputs: Input tensor to denormalize, typically model predictions
                   of shape (batch_size, seq_len, num_features).

        Returns:
            Denormalized tensor with original scale and mean restored.

        Raises:
            ValueError: If no statistics have been computed yet.

        Example:
            ```python
            # Forward pass stores statistics
            x_norm = revin_layer(x_input)

            # Model processes normalized input
            predictions = model(x_norm)

            # Denormalize to restore original scale
            predictions_original_scale = revin_layer.denormalize(predictions)
            ```
        """
        return self._denormalize(inputs)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input tensor.

        Returns:
            Output shape (same as input shape for RevIN).
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return the layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "num_features": self.num_features,
            "eps": self.eps,
            "affine": self.affine,
            "affine_weight_initializer": keras.initializers.serialize(self.affine_weight_initializer),
            "affine_bias_initializer": keras.initializers.serialize(self.affine_bias_initializer),
        })
        return config

# ---------------------------------------------------------------------