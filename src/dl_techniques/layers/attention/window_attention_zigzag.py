"""
ZigzagWindowAttention Layer

A Keras layer implementing a highly configurable windowed multi-head self-attention.
This version arranges tokens within the window in a zigzag pattern, inspired by
the coefficient scanning order in JPEG compression.

It extends the standard attention mechanism with two advanced, optional normalization
strategies as alternatives to the traditional softmax function:

1.  **Adaptive Temperature Softmax**: Dynamically adjusts the sharpness of the
    attention distribution based on its entropy, helping to prevent over-confidence
    and improve model calibration.
2.  **Hierarchical Routing**: A parameter-free, deterministic alternative that
    computes attention probabilities by routing mass through a fixed binary tree.

## Conceptual Overview

The layer's operational flow is as follows:
1.  **Padding**: Handles partial windows by padding inputs to the full window size.
2.  **QKV Projection**: Projects inputs into Query, Key, and Value tensors.
3.  **Score Calculation**: Computes scaled dot-product attention scores and adds the
    specialized zigzag relative position bias.
4.  **Configurable Normalization**: Converts scores to probabilities using one of
    three mutually exclusive methods:
    - `AdaptiveTemperatureSoftmax` (if `use_adaptive_softmax=True`)
    - `HierarchicalRouting` (if `use_hierarchical_routing=True`)
    - Standard `softmax` (default)
5.  **Output Computation**: Computes the final output by attending to Value vectors,
    projecting the result, and un-padding to the original sequence length.

### Key Features & Benefits:

1.  **Frequency-Domain Locality**: The zigzag ordering of relative positions
    prioritizes relationships between tokens that are close in the frequency
    domain, which can be beneficial for image or signal processing tasks.
2.  **Advanced Normalization**: Offers state-of-the-art alternatives to softmax
    that can improve performance, calibration, and offer different inductive biases.
3.  **Robust and Flexible**: Handles partial windows automatically, is fully
    serializable with Keras, and provides a clear, configurable interface.

### Usage Example:
```python
# The user is responsible for arranging the input tokens in zigzag order.
# For example, if you have a 7x7 grid of tokens, flatten it using a
# zigzag mapping before feeding it to the layer.
x_full = keras.random.normal((4, 49, 96)) # Full 7x7 window

# Standard usage with softmax
zigzag_attn = WindowZigZagAttention(dim=96, window_size=7, num_heads=3)
output_softmax = zigzag_attn(x_full)

# With hierarchical routing instead of softmax
routing_attn = WindowZigZagAttention(
    dim=96, window_size=7, num_heads=3, use_hierarchical_routing=True
)
output_routing = routing_attn(x_full)

# With adaptive temperature softmax for better calibration
adaptive_attn = WindowZigZagAttention(
    dim=96, window_size=7, num_heads=3,
    use_adaptive_softmax=True,
    adaptive_softmax_config={"min_temp": 0.1, "max_temp": 2.0}
)
output_adaptive = adaptive_attn(x_full)
```
"""
import keras
from keras import ops
from typing import Tuple, Optional, Dict, Any, Union, List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..activations.routing_probabilities import RoutingProbabilitiesLayer
from ..activations.adaptive_softmax import AdaptiveTemperatureSoftmax

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class WindowZigZagAttention(keras.layers.Layer):
    """
    Zigzag Window Multi-head Self-Attention with advanced normalization options.

    This layer implements windowed multi-head self-attention with a zigzag-ordered
    relative position bias. It can be configured to use standard softmax,
    adaptive temperature softmax, or hierarchical routing for attention
    normalization.

    Args:
        dim: Integer, dimensionality of the input feature space.
        window_size: Integer, the height and width of the attention window.
        num_heads: Integer, number of attention heads.
        qkv_bias: Boolean, whether to use a bias term in the QKV projection.
        qk_scale: Optional float, override for query scaling factor.
        attn_dropout_rate: Float, dropout rate for attention probabilities.
        proj_dropout_rate: Float, dropout rate for the final output projection.
        proj_bias: Boolean, whether to use a bias term in the output projection.
        use_hierarchical_routing: Boolean, if True, uses hierarchical routing
            instead of softmax. Cannot be True if `use_adaptive_softmax` is True.
        use_adaptive_softmax: Boolean, if True, uses adaptive temperature softmax
            instead of standard softmax. Cannot be True if
            `use_hierarchical_routing` is True.
        adaptive_softmax_config: Optional dictionary of arguments for
            AdaptiveTemperatureSoftmax. Used only when `use_adaptive_softmax=True`.
        kernel_initializer: Initializer for kernel weights.
        bias_initializer: Initializer for bias vectors.
        kernel_regularizer: Optional regularizer for kernel weights.
        bias_regularizer: Optional regularizer for bias weights.
        **kwargs: Additional keyword arguments for the base Layer class.
    """

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_dropout_rate: float = 0.0,
        proj_dropout_rate: float = 0.0,
        proj_bias: bool = True,
        use_hierarchical_routing: bool = False,
        use_adaptive_softmax: bool = False,
        adaptive_softmax_config: Optional[Dict[str, Any]] = None,
        kernel_initializer: Union[
            str, keras.initializers.Initializer
        ] = "glorot_uniform",
        bias_initializer: Union[
            str, keras.initializers.Initializer
        ] = "zeros",
        kernel_regularizer: Optional[
            Union[str, keras.regularizers.Regularizer]
        ] = None,
        bias_regularizer: Optional[
            Union[str, keras.regularizers.Regularizer]
        ] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # --- Configuration Validation ---
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})"
            )
        if not (0.0 <= attn_dropout_rate <= 1.0):
            raise ValueError(
                f"attn_dropout_rate must be in [0, 1], got {attn_dropout_rate}"
            )
        if not (0.0 <= proj_dropout_rate <= 1.0):
            raise ValueError(
                f"proj_dropout_rate must be in [0, 1], got {proj_dropout_rate}"
            )
        # Enforce mutual exclusivity of custom normalization methods.
        if use_adaptive_softmax and use_hierarchical_routing:
            raise ValueError(
                "Only one of `use_adaptive_softmax` or "
                "`use_hierarchical_routing` can be True."
            )

        # --- Store ALL configuration parameters for serialization ---
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_scale = qk_scale
        self.scale = qk_scale if qk_scale is not None else self.head_dim**-0.5
        self.qkv_bias = qkv_bias
        self.proj_bias = proj_bias
        self.attn_dropout_rate = attn_dropout_rate
        self.proj_dropout_rate = proj_dropout_rate
        self.use_hierarchical_routing = use_hierarchical_routing
        self.use_adaptive_softmax = use_adaptive_softmax
        self.adaptive_softmax_config = adaptive_softmax_config
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # --- Adaptive Softmax Configuration and Validation ---
        if self.use_adaptive_softmax:
            if self.adaptive_softmax_config is None:
                self.adaptive_softmax_config = {}
            # Set defaults and validate, storing them back for serialization.
            min_temp = self.adaptive_softmax_config.setdefault("min_temp", 0.1)
            max_temp = self.adaptive_softmax_config.setdefault("max_temp", 1.0)
            self.adaptive_softmax_config.setdefault("entropy_threshold", 0.5)
            if min_temp <= 0:
                raise ValueError(f"min_temp must be positive, got {min_temp}")
            if max_temp <= min_temp:
                raise ValueError(
                    f"max_temp ({max_temp}) must be > min_temp ({min_temp})"
                )
        else:
            self.adaptive_softmax_config = None

        # --- CREATE all sub-layers in __init__ ---
        dense_kwargs = {
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "bias_regularizer": self.bias_regularizer,
        }
        self.qkv = keras.layers.Dense(
            self.dim * 3, use_bias=self.qkv_bias, name="qkv", **dense_kwargs
        )
        self.proj = keras.layers.Dense(
            self.dim, use_bias=self.proj_bias, name="proj", **dense_kwargs
        )
        self.attn_dropout = (
            keras.layers.Dropout(self.attn_dropout_rate, name="attn_dropout")
            if self.attn_dropout_rate > 0.0
            else None
        )
        self.proj_dropout = (
            keras.layers.Dropout(self.proj_dropout_rate, name="proj_dropout")
            if self.proj_dropout_rate > 0.0
            else None
        )

        # Conditionally create normalization layers
        if self.use_hierarchical_routing:
            self.hierarchical_routing = RoutingProbabilitiesLayer(
                axis=-1, name="routing_probs"
            )
        else:
            self.hierarchical_routing = None

        if self.use_adaptive_softmax:
            self.adaptive_softmax = AdaptiveTemperatureSoftmax(
                name="adaptive_softmax", **self.adaptive_softmax_config
            )
        else:
            self.adaptive_softmax = None

        # --- Pre-compute ZIGZAG relative position indices ---
        # This is a non-trainable buffer that defines the layer's core logic.
        # 1. Generate 2D coordinates in zigzag order.
        zigzag_coords = self._generate_zigzag_coords(self.window_size)
        coords = ops.convert_to_tensor(zigzag_coords, dtype="int32")
        # Shape: (N, 2), where N = window_size*window_size

        # 2. Calculate pairwise relative coordinates using broadcasting.
        # `coords[:, None, :]` -> (N, 1, 2)
        # `coords[None, :, :]` -> (1, N, 2)
        # `relative_coords` -> (N, N, 2)
        # where `relative_coords[i, j, :] = coords[i, :] - coords[j, :]`
        relative_coords = coords[:, None, :] - coords[None, :, :]

        # 3. Shift coordinates to be non-negative for table indexing.
        # The range of relative differences is [-(W-1), W-1].
        # Adding (W-1) shifts this range to [0, 2W-2].
        relative_coords += self.window_size - 1

        # 4. Flatten 2D relative coordinates into a 1D index.
        # This is equivalent to `row * num_cols + col`, creating a unique
        # index for each of the (2W-1) * (2W-1) possible relative positions.
        num_cols = 2 * self.window_size - 1
        self.relative_position_index = (
            relative_coords[:, :, 0] * num_cols + relative_coords[:, :, 1]
        )

    @staticmethod
    def _generate_zigzag_coords(size: int) -> List[Tuple[int, int]]:
        """Generates (row, col) coordinates for a zigzag scan."""
        coords = []
        r, c = 0, 0
        for _ in range(size * size):
            coords.append((r, c))
            if (r + c) % 2 == 0:  # Moving up-right
                if c == size - 1:
                    r += 1  # Hit right wall, move down
                elif r == 0:
                    c += 1  # Hit top wall, move right
                else:
                    r -= 1
                    c += 1
            else:  # Moving down-left
                if r == size - 1:
                    c += 1  # Hit bottom wall, move right
                elif c == 0:
                    r += 1  # Hit left wall, move down
                else:
                    r += 1
                    c -= 1
        return coords

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Creates the layer's weights and builds its sub-layers.

        This method is critical for robust serialization. By explicitly building
        all sub-layers, we ensure their weights are created before Keras
        attempts to load a saved model's state.
        """
        num_bias_entries = (2 * self.window_size - 1) ** 2
        self.relative_position_bias_table = self.add_weight(
            name="relative_position_bias_table",
            shape=(num_bias_entries, self.num_heads),
            initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
            dtype=self.dtype,
        )

        # Define the shape for a fully padded window.
        padded_shape = list(input_shape)
        padded_shape[1] = self.window_size * self.window_size
        padded_shape = tuple(padded_shape)

        # Build all sub-layers with the padded shape.
        self.qkv.build(padded_shape)
        self.proj.build(padded_shape)
        if self.attn_dropout is not None:
            self.attn_dropout.build(None)  # Shape-agnostic
        if self.proj_dropout is not None:
            self.proj_dropout.build(padded_shape)

        # Explicitly build normalization layers with the attention scores' shape.
        # Shape: (batch, num_heads, num_tokens, num_tokens)
        attn_scores_shape = (
            input_shape[0],
            self.num_heads,
            padded_shape[1],
            padded_shape[1],
        )
        if self.hierarchical_routing is not None:
            self.hierarchical_routing.build(attn_scores_shape)
        if self.adaptive_softmax is not None:
            self.adaptive_softmax.build(attn_scores_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Defines the forward pass of the ZigzagWindowAttention layer."""
        B_actual, N_actual, C_actual = ops.shape(inputs)
        N_target = self.window_size * self.window_size
        if N_actual > N_target:
            raise ValueError(
                f"Input sequence length ({N_actual}) > window area ({N_target})."
            )

        # --- PADDING ---
        # Handles partial windows by padding inputs to the full window size.
        # This ensures all internal matrix operations have static shapes.
        padded_inputs = inputs
        if N_actual < N_target:
            padding_amount = N_target - N_actual
            padding_tensor = ops.zeros(
                (B_actual, padding_amount, C_actual), dtype=inputs.dtype
            )
            padded_inputs = ops.concatenate([inputs, padding_tensor], axis=1)

            # Create a mask to ignore the padded tokens during attention.
            padding_mask = ops.concatenate(
                [
                    ops.ones((B_actual, N_actual), dtype="int32"),
                    ops.zeros((B_actual, padding_amount), dtype="int32"),
                ],
                axis=1,
            )
            # Combine internal padding mask with any user-provided mask.
            if attention_mask is None:
                attention_mask = padding_mask
            else:
                # Multiplication acts as a logical AND.
                attention_mask = attention_mask * padding_mask

        B, N, C = ops.shape(padded_inputs)

        # --- SCORE COMPUTATION ---
        # 1. Project to Q, K, V and reshape for multi-head attention.
        qkv = self.qkv(padded_inputs, training=training)
        # Reshape to (B, N, 3, H, D_h)
        qkv = ops.reshape(qkv, (B, N, 3, self.num_heads, self.head_dim))
        # Transpose to (3, B, H, N, D_h) for easy unpacking
        qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each shape: (B, H, N, D_h)

        # 2. Calculate scaled attention scores.
        q = q * self.scale
        # (B, H, N, D_h) @ (B, H, D_h, N) -> (B, H, N, N)
        attn_scores = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2)))

        # 3. Add zigzag relative position bias.
        # Use the pre-computed index to look up biases from the trainable table.
        bias = ops.take(
            self.relative_position_bias_table,
            self.relative_position_index,
            axis=0,
        )  # Shape: (N, N, H)
        # Transpose to match attention scores: (H, N, N)
        bias = ops.transpose(bias, (2, 0, 1))
        # Add bias (broadcasts across batch dimension B).
        attn_scores = attn_scores + bias[None, :, :, :]

        # 4. Apply attention mask if it exists.
        if attention_mask is not None:
            # Broadcast mask from (B, N) to (B, 1, 1, N) to match attn_scores.
            broadcast_mask = attention_mask[:, None, None, :]
            # Convert the mask (0s and 1s) to an additive mask (-inf and 0s).
            inf_value = ops.convert_to_tensor(-1e9, dtype=attn_scores.dtype)
            additive_mask = (
                1.0 - ops.cast(broadcast_mask, dtype=attn_scores.dtype)
            ) * inf_value
            attn_scores = attn_scores + additive_mask

        # --- NORMALIZATION STEP ---
        # Apply the selected normalization strategy.
        if self.use_adaptive_softmax and self.adaptive_softmax is not None:
            attn_weights = self.adaptive_softmax(attn_scores, training=training)
        elif (
            self.use_hierarchical_routing
            and self.hierarchical_routing is not None
        ):
            attn_weights = self.hierarchical_routing(
                attn_scores, training=training
            )
        else:
            attn_weights = ops.softmax(attn_scores, axis=-1)

        if self.attn_dropout is not None:
            attn_weights = self.attn_dropout(attn_weights, training=training)

        # --- OUTPUT COMPUTATION ---
        # (B, H, N, N) @ (B, H, N, D_h) -> (B, H, N, D_h)
        x = ops.matmul(attn_weights, v)
        # Transpose to (B, N, H, D_h) to prepare for head concatenation.
        x = ops.transpose(x, (0, 2, 1, 3))
        # Reshape to (B, N, C) to merge heads.
        x = ops.reshape(x, (B, N, C))
        x = self.proj(x, training=training)
        if self.proj_dropout is not None:
            x = self.proj_dropout(x, training=training)

        # --- UN-PADDING ---
        # Remove padded tokens to return a tensor of the original length.
        if N_actual < N_target:
            x = x[:, :N_actual, :]

        return x

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """The output shape is identical to the input shape."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Returns the layer's configuration for serialization."""
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "window_size": self.window_size,
                "num_heads": self.num_heads,
                "qkv_bias": self.qkv_bias,
                "qk_scale": self.qk_scale,
                "attn_dropout_rate": self.attn_dropout_rate,
                "proj_dropout_rate": self.proj_dropout_rate,
                "proj_bias": self.proj_bias,
                "use_hierarchical_routing": self.use_hierarchical_routing,
                "use_adaptive_softmax": self.use_adaptive_softmax,
                "adaptive_softmax_config": self.adaptive_softmax_config,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": keras.initializers.serialize(
                    self.bias_initializer
                ),
                "kernel_regularizer": keras.regularizers.serialize(
                    self.kernel_regularizer
                ),
                "bias_regularizer": keras.regularizers.serialize(
                    self.bias_regularizer
                ),
            }
        )
        return config

# ---------------------------------------------------------------------
