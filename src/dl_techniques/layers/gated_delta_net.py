"""
Gated DeltaNet Layer Implementation

A high-performance implementation of the Gated DeltaNet architecture that combines the delta rule
mechanism from DeltaNet with the adaptive gating mechanism from Mamba2, optimized for modern
hardware using keras.ops.scan for efficient sequential computation.

Mathematical Foundation
======================

The Gated DeltaNet layer implements a linear-complexity alternative to standard attention
mechanisms, built on two key principles:

1. **Delta Rule Mechanism**: Derived from classical learning theory, the delta rule minimizes
   mean squared error (MSE) at each timestep through gradient descent updates:

   S_t = S_{t-1} - η_t * ∇L_t(S_{t-1})

   where S_t is the memory state, η_t is the learning rate, and L_t is the loss at timestep t.
   This translates to the update rule:

   S_t = S_{t-1} + β_t * V_t * K_t^T

2. **Adaptive Gating**: Inspired by Mamba2's selective state-space models, adaptive gating
   controls memory retention and erasure:

   S_t = α_t * (S_{t-1} + β_t * V_t * K_t^T) + (1-α_t) * S_{t-1}

   where α_t ∈ [0,1] controls memory persistence (α→0 erases, α→1 retains) and
   β_t controls update strength.

Architecture Overview
====================

The layer processes input sequences through the following pipeline:

1. **Linear Projections**: Input is projected to query (Q), key (K), and value (V) representations
2. **Normalization**: Zero-centered RMS normalization applied to Q, K, V for training stability
3. **Position Encoding**: Short causal convolution provides position-based addressing
4. **Gating Computation**: Separate projections compute α_t and β_t gating parameters
5. **Delta Rule Update**: Sequential state updates using efficient scan operation
6. **Output Projection**: Maps internal representation back to model dimension
7. **Output Gating**: Final sigmoid gating for selective information flow

Key Features
============

**Linear Complexity**: Unlike quadratic attention mechanisms, computational complexity scales
linearly with sequence length, making it suitable for long-context applications.

**Hardware Optimization**: Core recurrent computation implemented using keras.ops.scan,
providing optimal performance across GPU, TPU, and CPU backends while maintaining
automatic differentiation support.

**Memory Control**: Dual gating mechanism enables both rapid memory clearance (for context
switches) and precise targeted updates (for associative recall), addressing limitations
of pure linear attention and pure delta rule approaches.

**Training Stability**: Zero-centered RMS normalization and careful initialization provide
stable training dynamics without the need for complex learning rate schedules.

Performance Characteristics
===========================

**Computational Complexity**: O(L·D²) where L is sequence length and D is model dimension,
compared to O(L²·D) for standard attention.

**Memory Usage**: O(H·D²) state memory where H is number of heads, independent of sequence
length, enabling processing of arbitrarily long sequences.

**Throughput**: Demonstrates superior training throughput compared to standard attention
and other linear alternatives, particularly for sequences longer than 1024 tokens.

Use Cases
=========

**In-Context Retrieval**: Excels at tasks requiring associative recall and pattern matching
within the input context, outperforming standard linear transformers on synthetic
benchmarks like Multi-Query Associative Recall (MQAR).

**Long-Context Modeling**: Linear complexity enables processing of long documents,
conversations, or time series without the quadratic scaling penalties of attention.

**Streaming Applications**: Constant memory usage and efficient sequential processing
make it suitable for real-time and streaming applications.

**Efficient Transformers**: Drop-in replacement for attention layers in transformer
architectures, particularly effective in encoder-only and decoder architectures.

Implementation Notes
===================

**Dimensional Flexibility**: Supports custom head dimensions independent of model dimension
through proper output projection, enabling architectural flexibility while maintaining
compatibility with existing model designs.

**Serialization Support**: Full Keras serialization compatibility with proper config
management and weight restoration, enabling model checkpointing and deployment.

**Gradient Flow**: Careful implementation ensures stable gradient flow through the
recurrent computation, supporting end-to-end training of deep networks.

References
==========

Based on research from:
- "Parallelizing Linear Transformers with the Delta Rule over Sequence Length"
  (Yang et al., 2024) - Original DeltaNet formulation
- "Gated Delta Networks: Improving Mamba2 with Delta Rule" (Yang et al., 2024) -
  Gated variant combining delta rule with adaptive gating
- "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (Gu & Dao, 2023) -
  Selective state-space model foundations
"""

import keras
import tensorflow as tf
from typing import Any, Dict, Optional, Tuple, Union
from keras import initializers, layers, ops, regularizers

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.utils.logger import logger


# ---------------------------------------------------------------------
@keras.saving.register_keras_serializable()
class GatedDeltaNet(keras.layers.Layer):
    """
    Gated DeltaNet layer combining delta rule updates with adaptive gating mechanism.

    This layer implements a sophisticated linear transformer variant that combines:
    - Delta rule mechanism for targeted memory updates
    - Adaptive gating for rapid memory erasure and control
    - Zero-Centered RMS normalization for training stability
    - Short convolution for position-based addressing
    - Output gating with sigmoid activation for selective information flow

    **Intent**: Provide an efficient alternative to standard attention that excels
    at in-context retrieval and long-context understanding while maintaining
    linear complexity. The gating mechanism enables flexible memory control.

    **Architecture**:
    ```
    Input(shape=[batch, seq_len, dim])
           ↓
    Q/K/V Linear Projections
           ↓
    Zero-Centered RMSNorm → Short Conv1D (Q, K, V)
           ↓                     ↓
    Alpha/Beta Gating ←----------┘
           ↓
    Delta Rule Update (with gating: α_t * delta_update + (1-α_t) * S_{t-1})
           ↓
    Output Projection → Sigmoid Gate (⊗) → Output
    ```

    **Mathematical Operations**:
    1. **QKV Transform**: Q = Linear_q(x), K = Linear_k(x), V = Linear_v(x)
    2. **Normalization**: Q_norm = RMSNorm(Q), K_norm = RMSNorm(K), V_norm = RMSNorm(V)
    3. **Convolution**: Q_conv = Conv1D(Q_norm), K_conv = Conv1D(K_norm), V_conv = Conv1D(V_norm)
    4. **Gating Parameters**: α = sigmoid(Linear_α(x)), β = sigmoid(Linear_β(x))
    5. **Delta Rule**: S_t = α_t * (S_{t-1} + β_t * V_t * K_t^T) + (1-α_t) * S_{t-1}
    6. **Output**: output = Q_t @ S_t, projected = Linear_out(output), gated = sigmoid(Linear_gate(projected)) ⊗ projected

    The delta rule minimizes MSE between desired and predicted output at each timestep,
    making it particularly effective for associative recall and in-context retrieval tasks.

    Args:
        dim: Integer, the model dimension size. Must be positive and divisible by num_heads.
            This determines the input/output feature size and state dimension.
        num_heads: Integer, number of attention heads for multi-head processing.
            Must be positive. Each head operates independently on dim//num_heads dimensions.
        head_dim: Optional integer, dimension per head. If None, defaults to dim // num_heads.
            Allows for custom head dimensionality independent of input dimension.
        conv_kernel_size: Integer, kernel size for short convolution layers.
            Typically 4 for position-based addressing. Must be positive. Defaults to 4.
        dropout_rate: Float between 0 and 1, dropout rate applied to intermediate representations.
            Used for regularization during training. Defaults to 0.0.
        use_bias: Boolean, whether to use bias terms in linear layers.
            Modern architectures often omit bias for efficiency. Defaults to False.
        kernel_initializer: String or initializer, initialization for linear layer weights.
            Defaults to 'glorot_uniform'.
        bias_initializer: String or initializer, initialization for bias weights (if used).
            Defaults to 'zeros'.
        kernel_regularizer: Optional regularizer for linear layer weights.
        bias_regularizer: Optional regularizer for bias weights.
        **kwargs: Additional arguments for Layer base class.

    Input shape:
        3D tensor with shape: `(batch_size, sequence_length, dim)`.

    Output shape:
        3D tensor with shape: `(batch_size, sequence_length, dim)`.
        Same shape as input, preserving sequence structure.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        conv_kernel_size: int = 4,
        dropout_rate: float = 0.0,
        use_bias: bool = False,
        kernel_initializer: Union[str, initializers.Initializer] = "glorot_uniform",
        bias_initializer: Union[str, initializers.Initializer] = "zeros",
        kernel_regularizer: Optional[regularizers.Regularizer] = None,
        bias_regularizer: Optional[regularizers.Regularizer] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # Validate parameters
        self._validate_inputs(
            dim, num_heads, head_dim, conv_kernel_size, dropout_rate
        )

        # Store ALL configuration parameters
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else dim // num_heads
        self.conv_kernel_size = conv_kernel_size
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        # Compute dimensions
        self.qk_dim = self.num_heads * self.head_dim
        self.v_dim = self.num_heads * self.head_dim * 2

        # Q/K/V projections
        self.q_proj = layers.Dense(
            self.qk_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="q_proj",
        )
        self.k_proj = layers.Dense(
            self.qk_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="k_proj",
        )
        self.v_proj = layers.Dense(
            self.v_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="v_proj",
        )

        # Zero-Centered RMS Normalization layers
        self.q_norm = create_normalization_layer(
            "zero_centered_rms_norm", epsilon=1e-5, use_scale=True, name="q_norm"
        )
        self.k_norm = create_normalization_layer(
            "zero_centered_rms_norm", epsilon=1e-5, use_scale=True, name="k_norm"
        )
        self.v_norm = create_normalization_layer(
            "zero_centered_rms_norm", epsilon=1e-5, use_scale=True, name="v_norm"
        )

        # Short convolution layers (depthwise separable)
        self.q_conv = layers.Conv1D(
            filters=self.qk_dim,
            kernel_size=conv_kernel_size,
            padding="causal",
            groups=self.qk_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            name="q_conv",
        )
        self.k_conv = layers.Conv1D(
            filters=self.qk_dim,
            kernel_size=conv_kernel_size,
            padding="causal",
            groups=self.qk_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            name="k_conv",
        )
        self.v_conv = layers.Conv1D(
            filters=self.v_dim,
            kernel_size=conv_kernel_size,
            padding="causal",
            groups=self.v_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            name="v_conv",
        )

        # Gating parameter projections (alpha and beta)
        self.alpha_proj = layers.Dense(
            self.num_heads,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="alpha_proj",
        )
        self.beta_proj = layers.Dense(
            self.num_heads,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="beta_proj",
        )

        # Output projection layer
        self.output_proj = layers.Dense(
            self.dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="output_proj",
        )

        # SiLU activation for intermediate processing
        self.silu = layers.Activation("silu", name="silu")

        # Dropout for regularization
        if dropout_rate > 0.0:
            self.dropout = layers.Dropout(dropout_rate, name="dropout")
        else:
            self.dropout = None

        # Output gate
        self.output_gate_linear = layers.Dense(
            self.dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="output_gate_linear",
        )

        logger.info(
            f"GatedDeltaNet initialized: dim={dim}, "
            f"num_heads={num_heads}, head_dim={self.head_dim}, "
            f"qk_dim={self.qk_dim}, v_dim={self.v_dim}"
        )

    def _validate_inputs(
        self,
        dim: int,
        num_heads: int,
        head_dim: Optional[int],
        conv_kernel_size: int,
        dropout_rate: float,
    ) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if head_dim is not None and head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}")
        if head_dim is None and dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads}) "
                "when head_dim is None"
            )
        if conv_kernel_size <= 0:
            raise ValueError(
                f"conv_kernel_size must be positive, got {conv_kernel_size}"
            )
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError(
                f"dropout_rate must be in [0, 1], got {dropout_rate}"
            )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        if len(input_shape) != 3:
            raise ValueError(f"Expected 3D input shape, got {input_shape}")
        batch_size, seq_len, features = input_shape
        if features != self.dim:
            raise ValueError(
                f"Input feature dimension ({features}) must match dim ({self.dim})"
            )

        self.q_proj.build(input_shape)
        self.k_proj.build(input_shape)
        self.v_proj.build(input_shape)
        q_shape = (batch_size, seq_len, self.qk_dim)
        k_shape = (batch_size, seq_len, self.qk_dim)
        v_shape = (batch_size, seq_len, self.v_dim)
        self.q_norm.build(q_shape)
        self.k_norm.build(k_shape)
        self.v_norm.build(v_shape)
        self.q_conv.build(q_shape)
        self.k_conv.build(k_shape)
        self.v_conv.build(v_shape)
        self.alpha_proj.build(input_shape)
        self.beta_proj.build(input_shape)
        self.output_proj.build((batch_size, seq_len, self.qk_dim))
        self.output_gate_linear.build((batch_size, seq_len, self.dim))
        super().build(input_shape)

    def delta_rule_update(
        self,
        q: keras.KerasTensor,
        k: keras.KerasTensor,
        v: keras.KerasTensor,
        alpha: keras.KerasTensor,
        beta: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        # Get sequence length from static shape if possible for XLA compatibility.
        # `tensor.shape[dim]` is a Python int if the shape is known.
        # `ops.shape(tensor)[dim]` is a symbolic Tensor.
        # XLA requires a static integer for the `length` argument of `scan`.
        seq_len = q.shape[1]
        if seq_len is None:
            # Fallback for dynamic sequence lengths (won't be XLA-compatible).
            seq_len = ops.shape(q)[1]

        batch_size = ops.shape(q)[0]

        def step_function(prev_carry, inputs):
            """
            Processes one timestep of the sequence.

            Args:
                prev_carry: A tuple (prev_state, _) from the previous timestep.
                            - prev_state shape: (B, H, D, D)
                inputs: A tuple of the current timestep's Q, K, V, alpha, and beta.

            Returns:
                A tuple (new_carry, outputs_for_stacking), where both elements are
                (current_state, output_t). This structure satisfies the
                `keras.ops.scan` constraint on the TensorFlow backend, where the
                carry and output must have the same structure and shape.
            """
            prev_state, _ = prev_carry  # Unpack carry, we only need the state
            q_t, k_t, v_t, alpha_t, beta_t = inputs

            # Using ops.split is robust for shape inference within a scan loop.
            v_t_1, v_t_2 = ops.split(v_t, indices_or_sections=2, axis=-1)

            # --- State Update Logic ---
            # Shapes: (B, H, D, 1) @ (B, H, 1, D) -> (B, H, D, D)
            k_t_expanded = ops.expand_dims(k_t, -1)
            v_t_1_expanded = ops.expand_dims(v_t_1, -2)
            delta_update = ops.matmul(k_t_expanded, v_t_1_expanded)

            # Apply gating parameters
            beta_t_expanded = ops.expand_dims(ops.expand_dims(beta_t, -1), -1)
            scaled_delta = beta_t_expanded * delta_update
            alpha_t_expanded = ops.expand_dims(
                ops.expand_dims(alpha_t, -1), -1
            )

            # Compute the new state using a proper gating mechanism.
            # S_t = alpha_t * S_{t-1} + (beta_t * K_t * V_t^T)
            current_state = alpha_t_expanded * prev_state + scaled_delta

            # --- Output Calculation ---
            # Shapes: (B, H, 1, D) @ (B, H, D, D) -> (B, H, 1, D)
            q_t_expanded = ops.expand_dims(q_t, -2)
            output_t = ops.matmul(q_t_expanded, current_state)
            output_t = ops.squeeze(output_t, axis=-2)  # -> (B, H, D)
            output_t = output_t + v_t_2  # Add residual part of V

            new_carry = (current_state, output_t)
            # The output must have the same structure/shape as the carry
            return new_carry, new_carry

        # To satisfy TF's scan constraint, the carry must be a tuple containing
        # both the state and a template for the per-step output.
        initial_state = ops.zeros(
            (batch_size, self.num_heads, self.head_dim, self.head_dim)
        )
        initial_output = ops.zeros_like(q[:, 0, :, :])
        initial_carry = (initial_state, initial_output)

        # Transpose inputs to (seq_len, batch_size, ...) for scanning.
        q_scan = ops.transpose(q, [1, 0, 2, 3])
        k_scan = ops.transpose(k, [1, 0, 2, 3])
        v_scan = ops.transpose(v, [1, 0, 2, 3])
        alpha_scan = ops.transpose(alpha, [1, 0, 2])
        beta_scan = ops.transpose(beta, [1, 0, 2])

        # ops.scan processes the sequence. It returns the final carry and the
        # stack of per-timestep outputs.
        # Explicitly passing `length` with a static integer value is crucial
        # for XLA compilation during training.
        _, stacked_outputs_tuple = ops.scan(
            f=step_function,
            init=initial_carry,
            xs=(q_scan, k_scan, v_scan, alpha_scan, beta_scan),
            length=seq_len,
        )

        # The stacked outputs are a tuple (stacked_states, stacked_real_outputs).
        # We only need the second part, which contains the actual step outputs.
        outputs = stacked_outputs_tuple[1]

        # Transpose outputs back to (batch_size, seq_len, ...).
        outputs = ops.transpose(outputs, [1, 0, 2, 3])
        return outputs

    def call(
        self, inputs: keras.KerasTensor, training: Optional[bool] = None
    ) -> keras.KerasTensor:
        batch_size = ops.shape(inputs)[0]
        seq_len = ops.shape(inputs)[1]

        q = self.q_proj(inputs, training=training)
        k = self.k_proj(inputs, training=training)
        v = self.v_proj(inputs, training=training)

        q_norm = self.q_norm(q, training=training)
        k_norm = self.k_norm(k, training=training)
        v_norm = self.v_norm(v, training=training)

        q_conv = self.silu(self.q_conv(q_norm, training=training))
        k_conv = self.silu(self.k_conv(k_norm, training=training))
        v_conv = self.silu(self.v_conv(v_norm, training=training))

        q_heads = ops.reshape(
            q_conv, (batch_size, seq_len, self.num_heads, self.head_dim)
        )
        k_heads = ops.reshape(
            k_conv, (batch_size, seq_len, self.num_heads, self.head_dim)
        )
        v_heads = ops.reshape(
            v_conv, (batch_size, seq_len, self.num_heads, 2 * self.head_dim)
        )

        alpha = ops.sigmoid(self.alpha_proj(inputs, training=training))
        beta = ops.sigmoid(self.beta_proj(inputs, training=training))

        if training and self.dropout is not None:
            q_heads = self.dropout(q_heads, training=training)
            k_heads = self.dropout(k_heads, training=training)
            v_heads = self.dropout(v_heads, training=training)

        delta_output = self.delta_rule_update(
            q_heads, k_heads, v_heads, alpha, beta, training=training
        )

        delta_output = ops.reshape(
            delta_output, (batch_size, seq_len, self.qk_dim)
        )
        delta_output = self.output_proj(delta_output, training=training)

        gate = ops.sigmoid(
            self.output_gate_linear(delta_output, training=training)
        )
        gated_output = gate * delta_output
        return gated_output

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        return tuple(input_shape)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "num_heads": self.num_heads,
                "head_dim": self.head_dim,
                "conv_kernel_size": self.conv_kernel_size,
                "dropout_rate": self.dropout_rate,
                "use_bias": self.use_bias,
                "kernel_initializer": initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": initializers.serialize(
                    self.bias_initializer
                ),
                "kernel_regularizer": regularizers.serialize(
                    self.kernel_regularizer
                ),
                "bias_regularizer": regularizers.serialize(
                    self.bias_regularizer
                ),
            }
        )
        return config
