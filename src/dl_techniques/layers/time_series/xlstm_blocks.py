"""
xLSTM (Extended Long Short-Term Memory) Implementation.

This module provides a production-ready implementation of the xLSTM architecture
from the paper "xLSTM: Extended Long Short-Term Memory" (arXiv:2405.04517v2).

The implementation follows the dl_techniques framework standards:
- Uses normalization factory for all normalization layers
- Uses FFN factory for feed-forward networks
- Follows Keras 3 custom layer/model guidelines
- Implements proper serialization and build patterns

References:
    Beck, M., et al. (2024). xLSTM: Extended Long Short-Term Memory.
    arXiv:2405.04517v2
"""

import keras
import numpy as np
from keras import ops, layers, initializers, activations
from typing import Optional, Union, Tuple, List, Any, Dict, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..ffn import create_ffn_layer
from ..norms import create_normalization_layer

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class sLSTMCell(keras.layers.Layer):
    """
    Scalar LSTM (sLSTM) Cell with exponential gating and normalizer state.

    This cell implements the sLSTM from Section 2.2 of the xLSTM paper, featuring:
    - Exponential gating for improved memory dynamics
    - Normalizer state (n_t) to stabilize memory updates
    - Stabilization technique to prevent numerical overflow

    **Intent**: Provide a recurrent cell with enhanced memory revision capabilities
    through exponential gating, suitable for sequence modeling tasks.

    **Architecture**:
    The sLSTM maintains three internal states per timestep:
    1. `cell_state (c_t)`: Primary memory content
    2. `normalizer_state (n_t)`: Stabilization state
    3. `hidden_state (h_t)`: Output state, derived as c_t / n_t

    **Mathematical Operations** (per timestep t):
    Gates:
        - i_t = exp(W_i @ x_t + R_i @ h_{t-1} + b_i)  (Input gate)
        - f_t = activation(W_f @ x_t + R_f @ h_{t-1} + b_f)  (Forget gate)
        - o_t = sigmoid(W_o @ x_t + R_o @ h_{t-1} + b_o)  (Output gate)
        - z_t = tanh(W_z @ x_t + R_z @ h_{t-1} + b_z)  (Cell input)

    State Updates:
        - c_t = f_t * c_{t-1} + i_t * z_t
        - n_t = f_t * n_{t-1} + i_t

    Output:
        - h_t = o_t * (c_t / n_t)

    With stabilization (Equations 15-17 in paper):
        - m_t = max(m_{t-1} + log(f_t), log(i_t))
        - i_t_tilde = exp(log(i_t) - m_t)
        - f_t_tilde = exp(log(f_t) + m_{t-1} - m_t)

    Args:
        units: Integer, dimensionality of the output space. Must be positive.
        forget_gate_activation: Literal['sigmoid', 'exp'], activation for forget gate.
            Defaults to 'sigmoid'. Use 'exp' for exponential gating as in the paper.
        kernel_initializer: Initializer for input weight matrices (W).
            Defaults to 'glorot_uniform'.
        recurrent_initializer: Initializer for recurrent weight matrices (R).
            Defaults to 'orthogonal'.
        bias_initializer: Initializer for bias vectors. Defaults to 'zeros'.
        kernel_regularizer: Optional regularizer for kernel weights.
        recurrent_regularizer: Optional regularizer for recurrent weights.
        bias_regularizer: Optional regularizer for bias weights.
        **kwargs: Additional arguments for the Layer base class.

    Input shape:
        2D tensor with shape: `(batch_size, input_dim)` for single timestep.

    Output shape:
        Tuple of three 2D tensors:
        - h_t: `(batch_size, units)` - Hidden state
        - c_t: `(batch_size, units)` - Cell state
        - n_t: `(batch_size, units)` - Normalizer state
        - m_t: `(batch_size, units)` - Stabilizer state

    State shape:
        Tuple of four tensors, each with shape `(batch_size, units)`.

    Example:
        ```python
        # Create sLSTM cell
        cell = sLSTMCell(units=128, forget_gate_activation='exp')

        # Single timestep forward pass
        batch_size = 32
        input_dim = 64
        x_t = keras.random.normal((batch_size, input_dim))

        # Initialize states
        state = cell.get_initial_state(batch_size=batch_size)

        # Forward pass
        output, new_state = cell(x_t, state)
        h_t, c_t, n_t, m_t = new_state

        # Use with RNN layer for full sequences
        rnn = keras.layers.RNN(cell, return_sequences=True)
        inputs = keras.random.normal((batch_size, 10, input_dim))
        outputs = rnn(inputs)
        ```
    """

    def __init__(
        self,
        units: int,
        forget_gate_activation: Literal['sigmoid', 'exp'] = 'sigmoid',
        kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
        recurrent_initializer: Union[str, initializers.Initializer] = 'orthogonal',
        bias_initializer: Union[str, initializers.Initializer] = 'zeros',
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        recurrent_regularizer: Optional[keras.regularizers.Regularizer] = None,
        bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if units <= 0:
            raise ValueError(f"`units` must be positive, but got {units}")
        self.units = units

        if forget_gate_activation not in ['sigmoid', 'exp']:
            raise ValueError(
                f"`forget_gate_activation` must be 'sigmoid' or 'exp', "
                f"but got {forget_gate_activation}"
            )
        self.forget_gate_activation = forget_gate_activation

        # Store initializers and regularizers
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = kernel_regularizer
        self.recurrent_regularizer = recurrent_regularizer
        self.bias_regularizer = bias_regularizer

        # Activation functions
        self.f_activation = (
            activations.get(forget_gate_activation)
            if forget_gate_activation != "exp"
            else None
        )
        self.o_activation = activations.get('sigmoid')
        self.z_activation = activations.get('tanh')

        # RNN cell properties
        self.state_size = [self.units, self.units, self.units, self.units]  # [h, c, n, m]
        self.output_size = self.units

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the cell's weight matrices."""
        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError("Last dimension of input_shape cannot be None.")

        # Input weight matrix (W) for all gates: [i, f, o, z]
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_dim, self.units * 4),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
        )

        # Recurrent weight matrix (R) for all gates
        self.recurrent_kernel = self.add_weight(
            name='recurrent_kernel',
            shape=(self.units, self.units * 4),
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            trainable=True,
        )

        # Bias vectors for all gates
        self.bias = self.add_weight(
            name='bias',
            shape=(self.units * 4,),
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            trainable=True,
        )

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        states: List[keras.KerasTensor],
        training: Optional[bool] = None,
    ) -> Tuple[keras.KerasTensor, List[keras.KerasTensor]]:
        """
        Forward pass for a single timestep.

        Args:
            inputs: Input tensor of shape (batch_size, input_dim).
            states: List of state tensors [h_tm1, c_tm1, n_tm1, m_tm1].
            training: Boolean, whether in training mode.

        Returns:
            Tuple of (output, new_states):
                - output: h_t with shape (batch_size, units)
                - new_states: List [h_t, c_t, n_t, m_t]
        """
        h_tm1, c_tm1, n_tm1, m_tm1 = states

        # Compute gate pre-activations
        x_proj = ops.matmul(inputs, self.kernel) + self.bias
        h_proj = ops.matmul(h_tm1, self.recurrent_kernel)
        projections = x_proj + h_proj

        # Split into gates: [input, forget, output, cell_input]
        i_proj, f_proj, o_proj, z_proj = ops.split(projections, 4, axis=-1)

        # Stabilizer state update (Equation 15)
        if self.forget_gate_activation == 'exp':
            log_f_t = f_proj
        else:
            log_f_t = ops.log(self.f_activation(f_proj) + 1e-8)

        m_t = ops.maximum(m_tm1 + log_f_t, i_proj)

        # Stabilized exponential gating (Equations 16, 17)
        i_t = ops.exp(i_proj - m_t)
        f_t = ops.exp(m_tm1 + log_f_t - m_t)

        # Other gates
        o_t = self.o_activation(o_proj)
        z_t = self.z_activation(z_proj)

        # State updates (Equations 8, 9)
        c_t = f_t * c_tm1 + i_t * z_t
        n_t = f_t * n_tm1 + i_t

        # Output (Equation 10)
        h_t = o_t * (c_t / (n_t + 1e-8))

        return h_t, [h_t, c_t, n_t, m_t]

    def get_initial_state(
        self,
        batch_size: Optional[int] = None,
    ) -> List[keras.KerasTensor]:
        """
        Get initial states for the cell.

        Args:
            batch_size: Integer, batch size.

        Returns:
            List of initial state tensors [h_0, c_0, n_0, m_0].
        """
        return [
            ops.zeros((batch_size, self.units), dtype=self.compute_dtype),
            ops.zeros((batch_size, self.units), dtype=self.compute_dtype),
            ops.zeros((batch_size, self.units), dtype=self.compute_dtype),
            ops.zeros((batch_size, self.units), dtype=self.compute_dtype),
        ]

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration of the cell."""
        config = super().get_config()
        config.update({
            'units': self.units,
            'forget_gate_activation': self.forget_gate_activation,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer': keras.regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class sLSTMLayer(keras.layers.Layer):
    """
    Scalar LSTM (sLSTM) layer for processing sequences.

    This layer wraps the sLSTMCell in a keras.layers.RNN to process full sequences.
    It provides a high-level interface for using sLSTM in models.

    **Intent**: Provide a drop-in replacement for standard LSTM layers with
    enhanced memory dynamics through exponential gating.

    Args:
        units: Integer, dimensionality of the output space.
        forget_gate_activation: Literal['sigmoid', 'exp'], forget gate activation.
        return_sequences: Boolean, whether to return the full sequence or just
            the last output. Defaults to True.
        return_state: Boolean, whether to return the last state in addition to
            the output. Defaults to False.
        go_backwards: Boolean, whether to process the sequence backwards.
            Defaults to False.
        stateful: Boolean, whether to maintain states between batches.
            Defaults to False.
        unroll: Boolean, whether to unroll the RNN. Defaults to False.
        kernel_initializer: Initializer for kernel weights.
        recurrent_initializer: Initializer for recurrent weights.
        bias_initializer: Initializer for bias weights.
        kernel_regularizer: Optional regularizer for kernel weights.
        recurrent_regularizer: Optional regularizer for recurrent weights.
        bias_regularizer: Optional regularizer for bias weights.
        **kwargs: Additional arguments for the Layer base class.

    Input shape:
        3D tensor with shape: `(batch_size, sequence_length, input_dim)`.

    Output shape:
        - If return_sequences=True: `(batch_size, sequence_length, units)`
        - If return_sequences=False: `(batch_size, units)`
        - If return_state=True: tuple of (output, h_state, c_state, n_state, m_state)

    Example:
        ```python
        # Create sLSTM layer
        layer = sLSTMLayer(units=128, return_sequences=True)

        # Process sequence
        inputs = keras.random.normal((32, 10, 64))
        outputs = layer(inputs)
        print(outputs.shape)  # (32, 10, 128)

        # Use in a model
        model = keras.Sequential([
            keras.layers.Input(shape=(None, 64)),
            sLSTMLayer(128, return_sequences=True),
            sLSTMLayer(64, return_sequences=False),
            keras.layers.Dense(10, activation='softmax')
        ])
        ```
    """

    def __init__(
        self,
        units: int,
        forget_gate_activation: Literal['sigmoid', 'exp'] = 'sigmoid',
        return_sequences: bool = True,
        return_state: bool = False,
        go_backwards: bool = False,
        stateful: bool = False,
        unroll: bool = False,
        kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
        recurrent_initializer: Union[str, initializers.Initializer] = 'orthogonal',
        bias_initializer: Union[str, initializers.Initializer] = 'zeros',
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        recurrent_regularizer: Optional[keras.regularizers.Regularizer] = None,
        bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.units = units
        self.forget_gate_activation = forget_gate_activation
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.go_backwards = go_backwards
        self.stateful = stateful
        self.unroll = unroll
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.recurrent_regularizer = recurrent_regularizer
        self.bias_regularizer = bias_regularizer

        # Create the cell
        self.cell = sLSTMCell(
            units=units,
            forget_gate_activation=forget_gate_activation,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
        )

        # Create the RNN wrapper
        self.rnn = keras.layers.RNN(
            self.cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the RNN layer."""
        self.rnn.build(input_shape)
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
        initial_state: Optional[List[keras.KerasTensor]] = None,
    ) -> Union[keras.KerasTensor, Tuple[keras.KerasTensor, ...]]:
        """
        Forward pass through the RNN layer.

        Args:
            inputs: Input tensor of shape (batch_size, seq_len, input_dim).
            mask: Optional mask tensor.
            training: Boolean, whether in training mode.
            initial_state: Optional initial state.

        Returns:
            Output tensor(s) depending on return_sequences and return_state.
        """
        return self.rnn(
            inputs,
            mask=mask,
            training=training,
            initial_state=initial_state,
        )

    def compute_output_shape(self, input_shape):
        return self.rnn.compute_output_shape(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration of the layer."""
        config = super().get_config()
        config.update({
            'units': self.units,
            'forget_gate_activation': self.forget_gate_activation,
            'return_sequences': self.return_sequences,
            'return_state': self.return_state,
            'go_backwards': self.go_backwards,
            'stateful': self.stateful,
            'unroll': self.unroll,
            'kernel_initializer': keras.initializers.serialize(
                initializers.get(self.kernel_initializer)
            ),
            'recurrent_initializer': keras.initializers.serialize(
                initializers.get(self.recurrent_initializer)
            ),
            'bias_initializer': keras.initializers.serialize(
                initializers.get(self.bias_initializer)
            ),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer': keras.regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
        })
        return config


@keras.saving.register_keras_serializable()
class mLSTMCell(keras.layers.Layer):
    """
    Matrix LSTM (mLSTM) Cell with matrix memory and covariance update rule.

    This cell implements the fully parallelizable mLSTM from Section 2.3 of the
    xLSTM paper. It uses a matrix memory C_t and a covariance-style update rule
    for enhanced storage capacity.

    **Intent**: Provide a parallelizable memory mechanism with matrix-valued
    memory for improved storage capacity compared to traditional LSTMs.

    **Architecture**:
    The mLSTM uses matrix memory C_t of shape (d_key, d_value) that stores
    associations between keys and values. The cell computes:

    1. Query (q_t), Key (k_t), Value (v_t) projections
    2. Input gate (i_t), Forget gate (f_t), Output gate (o_t)
    3. Matrix memory update: C_t = f_t ⊙ C_{t-1} + i_t ⊙ (v_t ⊗ k_t^T)
    4. Normalizer update: n_t = f_t ⊙ n_{t-1} + i_t ⊙ k_t
    5. Hidden state: h_t = o_t ⊙ (C_t @ q_t / (n_t^T @ q_t))

    Where ⊙ is element-wise multiplication and ⊗ is outer product.

    Args:
        units: Integer, dimensionality of the output space (d_model).
        num_heads: Integer, number of attention heads. Defaults to 1.
        key_dim: Optional integer, dimensionality of keys per head.
            If None, defaults to units // num_heads.
        value_dim: Optional integer, dimensionality of values per head.
            If None, defaults to units // num_heads.
        kernel_initializer: Initializer for input weight matrices.
        recurrent_initializer: Initializer for recurrent weight matrices.
        bias_initializer: Initializer for bias vectors.
        kernel_regularizer: Optional regularizer for kernel weights.
        recurrent_regularizer: Optional regularizer for recurrent weights.
        bias_regularizer: Optional regularizer for bias weights.
        **kwargs: Additional arguments for the Layer base class.

    Input shape:
        2D tensor with shape: `(batch_size, input_dim)`.

    Output shape:
        Tuple of (h_t, new_states):
        - h_t: `(batch_size, units)` - Hidden state
        - new_states: List of state tensors

    Example:
        ```python
        # Create mLSTM cell
        cell = mLSTMCell(units=256, num_heads=4)

        # Use with RNN for sequences
        rnn = keras.layers.RNN(cell, return_sequences=True)
        inputs = keras.random.normal((32, 10, 128))
        outputs = rnn(inputs)
        ```
    """

    def __init__(
        self,
        units: int,
        num_heads: int = 1,
        key_dim: Optional[int] = None,
        value_dim: Optional[int] = None,
        kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
        recurrent_initializer: Union[str, initializers.Initializer] = 'orthogonal',
        bias_initializer: Union[str, initializers.Initializer] = 'zeros',
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        recurrent_regularizer: Optional[keras.regularizers.Regularizer] = None,
        bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if units <= 0:
            raise ValueError(f"`units` must be positive, but got {units}")
        if num_heads <= 0:
            raise ValueError(f"`num_heads` must be positive, but got {num_heads}")
        if units % num_heads != 0:
            raise ValueError(
                f"units ({units}) must be divisible by `num_heads` ({num_heads})"
            )

        self.units = units
        self.num_heads = num_heads
        self.head_dim = units // num_heads
        self.key_dim = key_dim if key_dim is not None else self.head_dim
        self.value_dim = value_dim if value_dim is not None else self.head_dim

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = kernel_regularizer
        self.recurrent_regularizer = recurrent_regularizer
        self.bias_regularizer = bias_regularizer

        # State size: [h, C (flattened), n]
        # C is (num_heads, key_dim, value_dim) flattened
        # n is (num_heads, key_dim)
        self.matrix_memory_size = self.num_heads * self.key_dim * self.value_dim
        self.normalizer_size = self.num_heads * self.key_dim

        self.state_size = [
            self.units,  # h_t
            self.matrix_memory_size,  # C_t (flattened)
            self.normalizer_size,  # n_t
        ]
        self.output_size = self.units

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the cell's weight matrices."""
        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError("Last dimension of input_shape cannot be None.")

        # Total projection size: q, k, v, i, f, o
        total_proj_size = (
            self.num_heads * self.key_dim +  # q
            self.num_heads * self.key_dim +  # k
            self.num_heads * self.value_dim +  # v
            self.num_heads +  # i (scalar per head)
            self.num_heads +  # f (scalar per head)
            self.units  # o (full dimension)
        )

        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_dim, total_proj_size),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
        )

        self.recurrent_kernel = self.add_weight(
            name='recurrent_kernel',
            shape=(self.units, total_proj_size),
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            trainable=True,
        )

        self.bias = self.add_weight(
            name='bias',
            shape=(total_proj_size,),
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            trainable=True,
        )

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        states: List[keras.KerasTensor],
        training: Optional[bool] = None,
    ) -> Tuple[keras.KerasTensor, List[keras.KerasTensor]]:
        """
        Forward pass for a single timestep.

        Args:
            inputs: Input tensor of shape (batch_size, input_dim).
            states: List of state tensors [h_tm1, C_tm1_flat, n_tm1_flat].
            training: Boolean, whether in training mode.

        Returns:
            Tuple of (output, new_states).
        """
        h_tm1, C_tm1_flat, n_tm1_flat = states
        batch_size = ops.shape(inputs)[0]

        # Reshape states
        C_tm1 = ops.reshape(
            C_tm1_flat,
            (batch_size, self.num_heads, self.key_dim, self.value_dim)
        )
        n_tm1 = ops.reshape(
            n_tm1_flat,
            (batch_size, self.num_heads, self.key_dim)
        )

        # Compute projections
        x_proj = ops.matmul(inputs, self.kernel) + self.bias
        h_proj = ops.matmul(h_tm1, self.recurrent_kernel)
        projections = x_proj + h_proj

        # Split projections
        q_size = self.num_heads * self.key_dim
        k_size = self.num_heads * self.key_dim
        v_size = self.num_heads * self.value_dim
        i_size = self.num_heads
        f_size = self.num_heads

        sections = [q_size, k_size, v_size, i_size, f_size]
        indices = np.cumsum(sections)
        projections_list = ops.split(projections, indices.tolist(), axis=-1)
        q_proj, k_proj, v_proj, i_proj, f_proj, o_proj = projections_list

        # Reshape to multi-head format
        q_t = ops.reshape(q_proj, (batch_size, self.num_heads, self.key_dim))
        k_t = ops.reshape(k_proj, (batch_size, self.num_heads, self.key_dim))
        v_t = ops.reshape(v_proj, (batch_size, self.num_heads, self.value_dim))

        # Gates (with exponential for i, sigmoid for f and o)
        i_t = ops.exp(i_proj)  # (batch_size, num_heads)
        f_t = ops.sigmoid(f_proj)  # (batch_size, num_heads)
        o_t = ops.sigmoid(o_proj)  # (batch_size, units)

        # Reshape gates for broadcasting
        i_t = ops.reshape(i_t, (batch_size, self.num_heads, 1, 1))
        f_t = ops.reshape(f_t, (batch_size, self.num_heads, 1, 1))

        # Matrix memory update: C_t = f_t * C_{t-1} + i_t * (v_t ⊗ k_t^T)
        # Outer product: v_t (batch, heads, value_dim, 1) @ k_t (batch, heads, 1, key_dim)
        v_t_expanded = ops.expand_dims(v_t, axis=-1)  # (batch, heads, value_dim, 1)
        k_t_expanded = ops.expand_dims(k_t, axis=-2)  # (batch, heads, 1, key_dim)
        outer_product = ops.matmul(v_t_expanded, k_t_expanded)  # (batch, heads, value_dim, key_dim)

        # Transpose to match C format (batch, heads, key_dim, value_dim)
        outer_product = ops.transpose(outer_product, [0, 1, 3, 2])

        C_t = f_t * C_tm1 + i_t * outer_product

        # Normalizer update: n_t = f_t * n_{t-1} + i_t * k_t
        f_t_norm = ops.reshape(f_t, (batch_size, self.num_heads, 1))
        i_t_norm = ops.reshape(i_t, (batch_size, self.num_heads, 1))
        n_t = f_t_norm * n_tm1 + i_t_norm * k_t

        # Compute output: h_t = o_t * (C_t @ q_t / (n_t^T @ q_t))
        # C_t @ q_t: (batch, heads, key_dim, value_dim) @ (batch, heads, key_dim, 1)
        q_t_expanded = ops.expand_dims(q_t, axis=-1)  # (batch, heads, key_dim, 1)
        memory_retrieval = ops.matmul(
            ops.transpose(C_t, [0, 1, 3, 2]),  # (batch, heads, value_dim, key_dim)
            q_t_expanded
        )  # (batch, heads, value_dim, 1)
        memory_retrieval = ops.squeeze(memory_retrieval, axis=-1)  # (batch, heads, value_dim)

        # Normalization: n_t^T @ q_t
        normalization = ops.sum(n_t * q_t, axis=-1, keepdims=True) + 1e-8  # (batch, heads, 1)

        # Normalized retrieval
        normalized_retrieval = memory_retrieval / normalization  # (batch, heads, value_dim)

        # Reshape to (batch, units)
        normalized_retrieval = ops.reshape(
            normalized_retrieval,
            (batch_size, self.num_heads * self.value_dim)
        )

        # Apply output gate
        h_t = o_t * normalized_retrieval

        # Flatten states for storage
        C_t_flat = ops.reshape(C_t, (batch_size, self.matrix_memory_size))
        n_t_flat = ops.reshape(n_t, (batch_size, self.normalizer_size))

        return h_t, [h_t, C_t_flat, n_t_flat]

    def get_initial_state(
        self,
        batch_size: Optional[int] = None,
    ) -> List[keras.KerasTensor]:
        """Get initial states for the cell."""
        return [
            ops.zeros((batch_size, self.units), dtype=self.compute_dtype),
            ops.zeros((batch_size, self.matrix_memory_size), dtype=self.compute_dtype),
            ops.zeros((batch_size, self.normalizer_size), dtype=self.compute_dtype),
        ]

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration of the cell."""
        config = super().get_config()
        config.update({
            'units': self.units,
            'num_heads': self.num_heads,
            'key_dim': self.key_dim,
            'value_dim': self.value_dim,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer': keras.regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
        })
        return config


@keras.saving.register_keras_serializable()
class mLSTMLayer(keras.layers.Layer):
    """
    Matrix LSTM (mLSTM) layer for processing sequences.

    This layer wraps the mLSTMCell in a keras.layers.RNN to process full sequences.

    Args:
        units: Integer, dimensionality of the output space.
        num_heads: Integer, number of attention heads.
        key_dim: Optional integer, dimensionality of keys per head.
        value_dim: Optional integer, dimensionality of values per head.
        return_sequences: Boolean, whether to return the full sequence.
        return_state: Boolean, whether to return the last state.
        go_backwards: Boolean, whether to process backwards.
        stateful: Boolean, whether to maintain states between batches.
        unroll: Boolean, whether to unroll the RNN.
        kernel_initializer: Initializer for kernel weights.
        recurrent_initializer: Initializer for recurrent weights.
        bias_initializer: Initializer for bias weights.
        kernel_regularizer: Optional regularizer for kernel weights.
        recurrent_regularizer: Optional regularizer for recurrent weights.
        bias_regularizer: Optional regularizer for bias weights.
        **kwargs: Additional arguments for the Layer base class.

    Example:
        ```python
        layer = mLSTMLayer(units=256, num_heads=4, return_sequences=True)
        inputs = keras.random.normal((32, 10, 128))
        outputs = layer(inputs)
        ```
    """

    def __init__(
        self,
        units: int,
        num_heads: int = 1,
        key_dim: Optional[int] = None,
        value_dim: Optional[int] = None,
        return_sequences: bool = True,
        return_state: bool = False,
        go_backwards: bool = False,
        stateful: bool = False,
        unroll: bool = False,
        kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
        recurrent_initializer: Union[str, initializers.Initializer] = 'orthogonal',
        bias_initializer: Union[str, initializers.Initializer] = 'zeros',
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        recurrent_regularizer: Optional[keras.regularizers.Regularizer] = None,
        bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.units = units
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.go_backwards = go_backwards
        self.stateful = stateful
        self.unroll = unroll
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.recurrent_regularizer = recurrent_regularizer
        self.bias_regularizer = bias_regularizer

        # Create the cell
        self.cell = mLSTMCell(
            units=units,
            num_heads=num_heads,
            key_dim=key_dim,
            value_dim=value_dim,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
        )

        # Create the RNN wrapper
        self.rnn = keras.layers.RNN(
            self.cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the RNN layer."""
        self.rnn.build(input_shape)
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
        initial_state: Optional[List[keras.KerasTensor]] = None,
    ) -> Union[keras.KerasTensor, Tuple[keras.KerasTensor, ...]]:
        """Forward pass through the RNN layer."""
        return self.rnn(
            inputs,
            mask=mask,
            training=training,
            initial_state=initial_state,
        )

    def compute_output_shape(self, input_shape):
        return self.rnn.compute_output_shape(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration of the layer."""
        config = super().get_config()
        config.update({
            'units': self.units,
            'num_heads': self.num_heads,
            'key_dim': self.key_dim,
            'value_dim': self.value_dim,
            'return_sequences': self.return_sequences,
            'return_state': self.return_state,
            'go_backwards': self.go_backwards,
            'stateful': self.stateful,
            'unroll': self.unroll,
            'kernel_initializer': keras.initializers.serialize(
                initializers.get(self.kernel_initializer)
            ),
            'recurrent_initializer': keras.initializers.serialize(
                initializers.get(self.recurrent_initializer)
            ),
            'bias_initializer': keras.initializers.serialize(
                initializers.get(self.bias_initializer)
            ),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer': keras.regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class sLSTMBlock(keras.layers.Layer):
    """
    sLSTM residual block with post-normalization architecture.

    This block implements the architecture from Figure 10 of the xLSTM paper:
    Input → sLSTM → Normalization → FFN → Residual Add

    **Intent**: Provide a complete residual block for sLSTM with configurable
    normalization and feed-forward network, suitable for stacking in deep models.

    **Architecture Flow**:
    ```
    Input (residual)
       ↓
    sLSTMLayer
       ↓
    Normalization
       ↓
    Feed-Forward Network (configurable via factory)
       ↓
    Add(residual) → Output
    ```

    Args:
        units: Integer, dimensionality of the layer.
        ffn_type: String, type of FFN to use. Options: 'mlp', 'swiglu', 'geglu',
            'glu', 'differential', 'residual', 'swin_mlp'. Defaults to 'swiglu'.
        ffn_expansion_factor: Integer, expansion factor for FFN intermediate size.
            Defaults to 2.
        normalization_type: String, type of normalization. Options: 'layer_norm',
            'rms_norm', 'batch_norm', 'band_rms', etc. Defaults to 'layer_norm'.
        normalization_kwargs: Optional dictionary of kwargs for normalization layer.
        forget_gate_activation: Literal['sigmoid', 'exp'], sLSTM forget gate activation.
        dropout_rate: Float, dropout rate for FFN. Defaults to 0.0.
        kernel_initializer: Initializer for kernel weights.
        recurrent_initializer: Initializer for recurrent weights.
        bias_initializer: Initializer for bias weights.
        kernel_regularizer: Optional regularizer for kernel weights.
        recurrent_regularizer: Optional regularizer for recurrent weights.
        bias_regularizer: Optional regularizer for bias weights.
        **kwargs: Additional arguments for the Layer base class.

    Input shape:
        3D tensor with shape: `(batch_size, sequence_length, units)`.

    Output shape:
        3D tensor with shape: `(batch_size, sequence_length, units)`.

    Example:
        ```python
        # Standard sLSTM block
        block = sLSTMBlock(units=256, ffn_type='swiglu')

        # With custom normalization
        block = sLSTMBlock(
            units=256,
            normalization_type='rms_norm',
            normalization_kwargs={'epsilon': 1e-6}
        )

        # Stack multiple blocks
        inputs = keras.Input(shape=(None, 256))
        x = inputs
        for i in range(6):
            x = sLSTMBlock(units=256, name=f'slstm_block_{i}')(x)
        model = keras.Model(inputs, x)
        ```
    """

    def __init__(
        self,
        units: int,
        ffn_type: str = 'swiglu',
        ffn_expansion_factor: int = 2,
        normalization_type: str = 'layer_norm',
        normalization_kwargs: Optional[Dict[str, Any]] = None,
        forget_gate_activation: Literal['sigmoid', 'exp'] = 'sigmoid',
        dropout_rate: float = 0.0,
        kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
        recurrent_initializer: Union[str, initializers.Initializer] = 'orthogonal',
        bias_initializer: Union[str, initializers.Initializer] = 'zeros',
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        recurrent_regularizer: Optional[keras.regularizers.Regularizer] = None,
        bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.units = units
        self.ffn_type = ffn_type
        self.ffn_expansion_factor = ffn_expansion_factor
        self.normalization_type = normalization_type
        self.normalization_kwargs = normalization_kwargs or {}
        self.forget_gate_activation = forget_gate_activation
        self.dropout_rate = dropout_rate
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.recurrent_regularizer = recurrent_regularizer
        self.bias_regularizer = bias_regularizer

        # Create sub-layers (Create in __init__, Build in build())
        self.slstm = sLSTMLayer(
            units=units,
            forget_gate_activation=forget_gate_activation,
            return_sequences=True,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            name='slstm',
        )

        self.norm = create_normalization_layer(
            normalization_type=normalization_type,
            name='norm',
            **self.normalization_kwargs
        )

        # Create FFN using factory
        self.ffn = create_ffn_layer(
            ffn_type=ffn_type,
            output_dim=units,
            ffn_expansion_factor=ffn_expansion_factor,
            dropout_rate=dropout_rate,
            name='ffn',
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build all sub-layers."""
        # Build sLSTM
        self.slstm.build(input_shape)

        # Build normalization
        slstm_output_shape = self.slstm.compute_output_shape(input_shape)
        self.norm.build(slstm_output_shape)

        # Build FFN
        norm_output_shape = slstm_output_shape  # Norm doesn't change shape
        self.ffn.build(norm_output_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
        mask: Optional[keras.KerasTensor] = None,
    ) -> keras.KerasTensor:
        """Forward pass through the block."""
        residual = inputs

        # sLSTM
        x = self.slstm(inputs, training=training, mask=mask)

        # Normalization
        x = self.norm(x, training=training)

        # FFN
        x = self.ffn(x, training=training)

        # Residual connection
        return x + residual

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration of the layer."""
        config = super().get_config()
        config.update({
            'units': self.units,
            'ffn_type': self.ffn_type,
            'ffn_expansion_factor': self.ffn_expansion_factor,
            'normalization_type': self.normalization_type,
            'normalization_kwargs': self.normalization_kwargs,
            'forget_gate_activation': self.forget_gate_activation,
            'dropout_rate': self.dropout_rate,
            'kernel_initializer': keras.initializers.serialize(
                initializers.get(self.kernel_initializer)
            ),
            'recurrent_initializer': keras.initializers.serialize(
                initializers.get(self.recurrent_initializer)
            ),
            'bias_initializer': keras.initializers.serialize(
                initializers.get(self.bias_initializer)
            ),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer': keras.regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class mLSTMBlock(keras.layers.Layer):
    """
    mLSTM residual block with pre-up-projection architecture.

    This block implements the architecture from Figure 11 of the xLSTM paper:
    Input → Up-Project → Conv1D → mLSTM → Norm → Down-Project → Residual

    **Intent**: Provide a complete residual block for mLSTM with SSM-style
    pre-expansion, suitable for parallelizable sequence processing.

    **Architecture Flow**:
    ```
    Input (residual)
       ↓
    Up-Projection (Dense: units → units * expansion_factor)
       ↓
    Depthwise Conv1D (causal, kernel_size=4)
       ↓
    Activation (swish)
       ↓
    mLSTMLayer
       ↓
    Normalization
       ↓
    Down-Projection (Dense: units * expansion_factor → units)
       ↓
    Add(residual) → Output
    ```

    Args:
        units: Integer, dimensionality of the layer.
        expansion_factor: Integer, expansion factor for internal dimension.
            Defaults to 2.
        num_heads: Integer, number of mLSTM heads. Defaults to 1.
        conv_kernel_size: Integer, kernel size for depthwise conv. Defaults to 4.
        normalization_type: String, type of normalization. Defaults to 'layer_norm'.
        normalization_kwargs: Optional dictionary of kwargs for normalization layer.
        kernel_initializer: Initializer for kernel weights.
        recurrent_initializer: Initializer for recurrent weights.
        bias_initializer: Initializer for bias weights.
        kernel_regularizer: Optional regularizer for kernel weights.
        recurrent_regularizer: Optional regularizer for recurrent weights.
        bias_regularizer: Optional regularizer for bias weights.
        **kwargs: Additional arguments for the Layer base class.

    Input shape:
        3D tensor with shape: `(batch_size, sequence_length, units)`.

    Output shape:
        3D tensor with shape: `(batch_size, sequence_length, units)`.

    Example:
        ```python
        # Standard mLSTM block
        block = mLSTMBlock(units=256, expansion_factor=2, num_heads=4)

        # With custom configuration
        block = mLSTMBlock(
            units=256,
            expansion_factor=3,
            num_heads=8,
            conv_kernel_size=7,
            normalization_type='rms_norm'
        )
        ```
    """

    def __init__(
        self,
        units: int,
        expansion_factor: int = 2,
        num_heads: int = 1,
        conv_kernel_size: int = 4,
        normalization_type: str = 'layer_norm',
        normalization_kwargs: Optional[Dict[str, Any]] = None,
        kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
        recurrent_initializer: Union[str, initializers.Initializer] = 'orthogonal',
        bias_initializer: Union[str, initializers.Initializer] = 'zeros',
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        recurrent_regularizer: Optional[keras.regularizers.Regularizer] = None,
        bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.units = units
        self.expansion_factor = expansion_factor
        self.num_heads = num_heads
        self.conv_kernel_size = conv_kernel_size
        self.normalization_type = normalization_type
        self.normalization_kwargs = normalization_kwargs or {}
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.recurrent_regularizer = recurrent_regularizer
        self.bias_regularizer = bias_regularizer

        self.inner_dim = units * expansion_factor

        # Create sub-layers (Create in __init__, Build in build())
        self.up_proj = layers.Dense(
            self.inner_dim,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name='up_proj',
        )

        # Depthwise (grouped) conv for mixing
        self.conv = layers.Conv1D(
            filters=self.inner_dim,
            kernel_size=conv_kernel_size,
            padding='causal',
            groups=self.inner_dim,  # Depthwise
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name='conv1d',
        )

        self.mlstm = mLSTMLayer(
            units=self.inner_dim,
            num_heads=num_heads,
            return_sequences=True,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            name='mlstm',
        )

        self.norm = create_normalization_layer(
            normalization_type=normalization_type,
            name='norm',
            **self.normalization_kwargs
        )

        self.down_proj = layers.Dense(
            units,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name='down_proj',
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build all sub-layers."""
        # Build up projection
        self.up_proj.build(input_shape)

        # Build conv
        up_shape = self.up_proj.compute_output_shape(input_shape)
        self.conv.build(up_shape)

        # Build mLSTM
        conv_shape = self.conv.compute_output_shape(up_shape)
        self.mlstm.build(conv_shape)

        # Build normalization
        mlstm_shape = self.mlstm.compute_output_shape(conv_shape)
        self.norm.build(mlstm_shape)

        # Build down projection
        self.down_proj.build(mlstm_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
        mask: Optional[keras.KerasTensor] = None,
    ) -> keras.KerasTensor:
        """Forward pass through the block."""
        residual = inputs

        # Up projection
        x = self.up_proj(inputs, training=training)

        # Depthwise conv
        x = self.conv(x, training=training)

        # Activation
        x = activations.swish(x)

        # mLSTM
        x = self.mlstm(x, training=training, mask=mask)

        # Normalization
        x = self.norm(x, training=training)

        # Down projection
        x = self.down_proj(x, training=training)

        # Residual connection
        return x + residual

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration of the layer."""
        config = super().get_config()
        config.update({
            'units': self.units,
            'expansion_factor': self.expansion_factor,
            'num_heads': self.num_heads,
            'conv_kernel_size': self.conv_kernel_size,
            'normalization_type': self.normalization_type,
            'normalization_kwargs': self.normalization_kwargs,
            'kernel_initializer': keras.initializers.serialize(
                initializers.get(self.kernel_initializer)
            ),
            'recurrent_initializer': keras.initializers.serialize(
                initializers.get(self.recurrent_initializer)
            ),
            'bias_initializer': keras.initializers.serialize(
                initializers.get(self.bias_initializer)
            ),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer': keras.regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

# ---------------------------------------------------------------------
