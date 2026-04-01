"""
xLSTM (Extended Long Short-Term Memory) implementation.

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
    Scalar LSTM (sLSTM) cell with exponential gating and normalizer state.

    This cell implements the sLSTM from Section 2.2 of the xLSTM paper, featuring
    exponential gating for improved memory dynamics, a normalizer state (n_t) to
    stabilize memory updates, and a stabilization technique to prevent numerical
    overflow.

    The sLSTM maintains three internal states per timestep:
    ``cell_state (c_t)``, ``normalizer_state (n_t)``, and ``hidden_state (h_t)``
    derived as ``c_t / n_t``.

    **Gate equations** (per timestep t):

        i_t = exp(W_i @ x_t + R_i @ h_{t-1} + b_i)
        f_t = activation(W_f @ x_t + R_f @ h_{t-1} + b_f)
        o_t = sigmoid(W_o @ x_t + R_o @ h_{t-1} + b_o)
        z_t = tanh(W_z @ x_t + R_z @ h_{t-1} + b_z)

    **State updates:**

        c_t = f_t * c_{t-1} + i_t * z_t
        n_t = f_t * n_{t-1} + i_t
        h_t = o_t * (c_t / n_t)

    **Stabilization** (Equations 15-17 in paper):

        m_t = max(m_{t-1} + log(f_t), log(i_t))
        i_t_tilde = exp(log(i_t) - m_t)
        f_t_tilde = exp(log(f_t) + m_{t-1} - m_t)

    **Architecture Overview:**

    .. code-block:: text

        x_t: (batch, input_dim)    h_{t-1}: (batch, units)
              │                          │
              ▼                          ▼
        ┌──────────┐             ┌──────────────┐
        │ W @ x_t  │             │ R @ h_{t-1}  │
        └────┬─────┘             └──────┬───────┘
             └──────────┬───────────────┘
                        ▼
              ┌───────────────────┐
              │ Split into 4 gates│
              │  i, f, o, z       │
              └────────┬──────────┘
                       ▼
              ┌───────────────────┐
              │  Stabilize (m_t)  │
              │  Exp gating       │
              └────────┬──────────┘
                       ▼
              ┌───────────────────┐
              │  State Updates    │
              │  c_t, n_t         │
              └────────┬──────────┘
                       ▼
              ┌───────────────────┐
              │  h_t = o * c/n    │
              └────────┬──────────┘
                       ▼
              Output: h_t (batch, units)

    :param units: Dimensionality of the output space. Must be positive.
    :type units: int
    :param forget_gate_activation: Activation for the forget gate, either
        ``'sigmoid'`` or ``'exp'`` for exponential gating as in the paper.
    :type forget_gate_activation: str
    :param kernel_initializer: Initializer for input weight matrices (W).
    :type kernel_initializer: str or keras.initializers.Initializer
    :param recurrent_initializer: Initializer for recurrent weight matrices (R).
    :type recurrent_initializer: str or keras.initializers.Initializer
    :param bias_initializer: Initializer for bias vectors.
    :type bias_initializer: str or keras.initializers.Initializer
    :param kernel_regularizer: Optional regularizer for kernel weights.
    :type kernel_regularizer: keras.regularizers.Regularizer, optional
    :param recurrent_regularizer: Optional regularizer for recurrent weights.
    :type recurrent_regularizer: keras.regularizers.Regularizer, optional
    :param bias_regularizer: Optional regularizer for bias weights.
    :type bias_regularizer: keras.regularizers.Regularizer, optional
    :param kwargs: Additional arguments for the Layer base class.
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
        """
        Build the cell's weight matrices.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: tuple

        :raises ValueError: If the last dimension of ``input_shape`` is None.
        """
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

        :param inputs: Input tensor of shape ``(batch_size, input_dim)``.
        :type inputs: keras.KerasTensor
        :param states: List of state tensors ``[h_tm1, c_tm1, n_tm1, m_tm1]``.
        :type states: list of keras.KerasTensor
        :param training: Whether the layer is in training mode.
        :type training: bool, optional
        :return: Tuple of ``(h_t, [h_t, c_t, n_t, m_t])``.
        :rtype: tuple
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

        :param batch_size: Batch size for the initial state tensors.
        :type batch_size: int, optional
        :return: List of initial state tensors ``[h_0, c_0, n_0, m_0]``.
        :rtype: list of keras.KerasTensor
        """
        return [
            ops.zeros((batch_size, self.units), dtype=self.compute_dtype),
            ops.zeros((batch_size, self.units), dtype=self.compute_dtype),
            ops.zeros((batch_size, self.units), dtype=self.compute_dtype),
            ops.zeros((batch_size, self.units), dtype=self.compute_dtype),
        ]

    def get_config(self) -> Dict[str, Any]:
        """
        Return the configuration of the cell for serialization.

        :return: Configuration dictionary.
        :rtype: dict
        """
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

    Wraps the ``sLSTMCell`` in a ``keras.layers.RNN`` to process full sequences.
    Provides a drop-in replacement for standard LSTM layers with enhanced memory
    dynamics through exponential gating.

    **Architecture Overview:**

    .. code-block:: text

        Input: (batch, seq_len, input_dim)
                    │
                    ▼
        ┌───────────────────────────┐
        │   keras.layers.RNN        │
        │   ┌───────────────────┐   │
        │   │   sLSTMCell       │   │
        │   │  (per timestep)   │   │
        │   └───────────────────┘   │
        └───────────┬───────────────┘
                    ▼
        Output: (batch, seq_len, units)
             or (batch, units) if
             return_sequences=False

    :param units: Dimensionality of the output space.
    :type units: int
    :param forget_gate_activation: Forget gate activation, ``'sigmoid'`` or ``'exp'``.
    :type forget_gate_activation: str
    :param return_sequences: Whether to return the full sequence or just the last output.
    :type return_sequences: bool
    :param return_state: Whether to return the last state in addition to the output.
    :type return_state: bool
    :param go_backwards: Whether to process the sequence backwards.
    :type go_backwards: bool
    :param stateful: Whether to maintain states between batches.
    :type stateful: bool
    :param unroll: Whether to unroll the RNN loop.
    :type unroll: bool
    :param kernel_initializer: Initializer for kernel weights.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param recurrent_initializer: Initializer for recurrent weights.
    :type recurrent_initializer: str or keras.initializers.Initializer
    :param bias_initializer: Initializer for bias weights.
    :type bias_initializer: str or keras.initializers.Initializer
    :param kernel_regularizer: Optional regularizer for kernel weights.
    :type kernel_regularizer: keras.regularizers.Regularizer, optional
    :param recurrent_regularizer: Optional regularizer for recurrent weights.
    :type recurrent_regularizer: keras.regularizers.Regularizer, optional
    :param bias_regularizer: Optional regularizer for bias weights.
    :type bias_regularizer: keras.regularizers.Regularizer, optional
    :param kwargs: Additional arguments for the Layer base class.
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
        """
        Build the RNN layer.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: tuple
        """
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
        Forward pass through the sLSTM RNN layer.

        :param inputs: Input tensor of shape ``(batch_size, seq_len, input_dim)``.
        :type inputs: keras.KerasTensor
        :param mask: Optional mask tensor.
        :type mask: keras.KerasTensor, optional
        :param training: Whether the layer is in training mode.
        :type training: bool, optional
        :param initial_state: Optional initial state tensors.
        :type initial_state: list of keras.KerasTensor, optional
        :return: Output tensor(s) depending on ``return_sequences`` and ``return_state``.
        :rtype: keras.KerasTensor or tuple of keras.KerasTensor
        """
        return self.rnn(
            inputs,
            mask=mask,
            training=training,
            initial_state=initial_state,
        )

    def compute_output_shape(self, input_shape):
        """
        Compute the output shape of the layer.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: tuple
        :return: Output shape tuple.
        :rtype: tuple
        """
        return self.rnn.compute_output_shape(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Return the configuration of the layer for serialization.

        :return: Configuration dictionary.
        :rtype: dict
        """
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
    Matrix LSTM (mLSTM) cell with matrix memory and covariance update rule.

    This cell implements the fully parallelizable mLSTM from Section 2.3 of the
    xLSTM paper. It uses a matrix memory ``C_t`` of shape ``(d_key, d_value)``
    and a covariance-style update rule for enhanced storage capacity compared
    to traditional LSTMs.

    The cell computes query, key, and value projections alongside input, forget,
    and output gates:

        C_t = f_t * C_{t-1} + i_t * (v_t (outer) k_t^T)
        n_t = f_t * n_{t-1} + i_t * k_t
        h_t = o_t * (C_t @ q_t / (n_t^T @ q_t))

    **Architecture Overview:**

    .. code-block:: text

        x_t: (batch, input_dim)    h_{t-1}: (batch, units)
              │                          │
              ▼                          ▼
        ┌──────────┐             ┌──────────────┐
        │ W @ x_t  │             │ R @ h_{t-1}  │
        └────┬─────┘             └──────┬───────┘
             └──────────┬───────────────┘
                        ▼
              ┌───────────────────────┐
              │ Split into projections│
              │  q, k, v, i, f, o     │
              └────────┬──────────────┘
                       ▼
              ┌───────────────────────┐
              │ Matrix Memory Update  │
              │ C_t = f*C + i*(v⊗k^T) │
              └────────┬──────────────┘
                       ▼
              ┌───────────────────────┐
              │ Normalizer Update     │
              │ n_t = f*n + i*k       │
              └────────┬──────────────┘
                       ▼
              ┌───────────────────────┐
              │ Memory Retrieval      │
              │ o * (C@q / (n^T@q))   │
              └────────┬──────────────┘
                       ▼
              Output: h_t (batch, units)

    :param units: Dimensionality of the output space (d_model). Must be positive.
    :type units: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param key_dim: Dimensionality of keys per head. If None, defaults to
        ``units // num_heads``.
    :type key_dim: int, optional
    :param value_dim: Dimensionality of values per head. If None, defaults to
        ``units // num_heads``.
    :type value_dim: int, optional
    :param kernel_initializer: Initializer for input weight matrices.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param recurrent_initializer: Initializer for recurrent weight matrices.
    :type recurrent_initializer: str or keras.initializers.Initializer
    :param bias_initializer: Initializer for bias vectors.
    :type bias_initializer: str or keras.initializers.Initializer
    :param kernel_regularizer: Optional regularizer for kernel weights.
    :type kernel_regularizer: keras.regularizers.Regularizer, optional
    :param recurrent_regularizer: Optional regularizer for recurrent weights.
    :type recurrent_regularizer: keras.regularizers.Regularizer, optional
    :param bias_regularizer: Optional regularizer for bias weights.
    :type bias_regularizer: keras.regularizers.Regularizer, optional
    :param kwargs: Additional arguments for the Layer base class.

    :raises ValueError: If ``units`` is not positive.
    :raises ValueError: If ``num_heads`` is not positive.
    :raises ValueError: If ``units`` is not divisible by ``num_heads``.
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
        """
        Build the cell's weight matrices.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: tuple

        :raises ValueError: If the last dimension of ``input_shape`` is None.
        """
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

        :param inputs: Input tensor of shape ``(batch_size, input_dim)``.
        :type inputs: keras.KerasTensor
        :param states: List of state tensors ``[h_tm1, C_tm1_flat, n_tm1_flat]``.
        :type states: list of keras.KerasTensor
        :param training: Whether the layer is in training mode.
        :type training: bool, optional
        :return: Tuple of ``(h_t, [h_t, C_t_flat, n_t_flat])``.
        :rtype: tuple
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
        """
        Get initial states for the cell.

        :param batch_size: Batch size for the initial state tensors.
        :type batch_size: int, optional
        :return: List of initial state tensors ``[h_0, C_0_flat, n_0_flat]``.
        :rtype: list of keras.KerasTensor
        """
        return [
            ops.zeros((batch_size, self.units), dtype=self.compute_dtype),
            ops.zeros((batch_size, self.matrix_memory_size), dtype=self.compute_dtype),
            ops.zeros((batch_size, self.normalizer_size), dtype=self.compute_dtype),
        ]

    def get_config(self) -> Dict[str, Any]:
        """
        Return the configuration of the cell for serialization.

        :return: Configuration dictionary.
        :rtype: dict
        """
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

    Wraps the ``mLSTMCell`` in a ``keras.layers.RNN`` to process full sequences,
    providing a high-level interface for using matrix-valued memory in models.

    **Architecture Overview:**

    .. code-block:: text

        Input: (batch, seq_len, input_dim)
                    │
                    ▼
        ┌───────────────────────────┐
        │   keras.layers.RNN        │
        │   ┌───────────────────┐   │
        │   │   mLSTMCell       │   │
        │   │  (per timestep)   │   │
        │   └───────────────────┘   │
        └───────────┬───────────────┘
                    ▼
        Output: (batch, seq_len, units)
             or (batch, units) if
             return_sequences=False

    :param units: Dimensionality of the output space.
    :type units: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param key_dim: Optional dimensionality of keys per head.
    :type key_dim: int, optional
    :param value_dim: Optional dimensionality of values per head.
    :type value_dim: int, optional
    :param return_sequences: Whether to return the full sequence.
    :type return_sequences: bool
    :param return_state: Whether to return the last state.
    :type return_state: bool
    :param go_backwards: Whether to process the sequence backwards.
    :type go_backwards: bool
    :param stateful: Whether to maintain states between batches.
    :type stateful: bool
    :param unroll: Whether to unroll the RNN loop.
    :type unroll: bool
    :param kernel_initializer: Initializer for kernel weights.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param recurrent_initializer: Initializer for recurrent weights.
    :type recurrent_initializer: str or keras.initializers.Initializer
    :param bias_initializer: Initializer for bias weights.
    :type bias_initializer: str or keras.initializers.Initializer
    :param kernel_regularizer: Optional regularizer for kernel weights.
    :type kernel_regularizer: keras.regularizers.Regularizer, optional
    :param recurrent_regularizer: Optional regularizer for recurrent weights.
    :type recurrent_regularizer: keras.regularizers.Regularizer, optional
    :param bias_regularizer: Optional regularizer for bias weights.
    :type bias_regularizer: keras.regularizers.Regularizer, optional
    :param kwargs: Additional arguments for the Layer base class.
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
        """
        Build the RNN layer.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: tuple
        """
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
        Forward pass through the mLSTM RNN layer.

        :param inputs: Input tensor of shape ``(batch_size, seq_len, input_dim)``.
        :type inputs: keras.KerasTensor
        :param mask: Optional mask tensor.
        :type mask: keras.KerasTensor, optional
        :param training: Whether the layer is in training mode.
        :type training: bool, optional
        :param initial_state: Optional initial state tensors.
        :type initial_state: list of keras.KerasTensor, optional
        :return: Output tensor(s) depending on ``return_sequences`` and ``return_state``.
        :rtype: keras.KerasTensor or tuple of keras.KerasTensor
        """
        return self.rnn(
            inputs,
            mask=mask,
            training=training,
            initial_state=initial_state,
        )

    def compute_output_shape(self, input_shape):
        """
        Compute the output shape of the layer.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: tuple
        :return: Output shape tuple.
        :rtype: tuple
        """
        return self.rnn.compute_output_shape(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Return the configuration of the layer for serialization.

        :return: Configuration dictionary.
        :rtype: dict
        """
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

    Implements the architecture from Figure 10 of the xLSTM paper: a residual
    block that applies sLSTM followed by normalization and a configurable
    feed-forward network, with a skip connection around the entire block.

    **Architecture Overview:**

    .. code-block:: text

        Input: (batch, seq_len, units)
                │
                ├───────────────────────────┐
                ▼                           │ (residual)
        ┌───────────────────────┐           │
        │      sLSTMLayer       │           │
        └───────────┬───────────┘           │
                    ▼                       │
        ┌───────────────────────┐           │
        │    Normalization      │           │
        └───────────┬───────────┘           │
                    ▼                       │
        ┌───────────────────────┐           │
        │   Feed-Forward Net    │           │
        │   (configurable)      │           │
        └───────────┬───────────┘           │
                    ▼                       │
                  ( + ) ◄───────────────────┘
                    │
                    ▼
        Output: (batch, seq_len, units)

    :param units: Dimensionality of the layer.
    :type units: int
    :param ffn_type: Type of FFN to use (e.g., ``'mlp'``, ``'swiglu'``, ``'geglu'``,
        ``'glu'``, ``'differential'``, ``'residual'``, ``'swin_mlp'``).
    :type ffn_type: str
    :param ffn_expansion_factor: Expansion factor for FFN intermediate size.
    :type ffn_expansion_factor: int
    :param normalization_type: Type of normalization (e.g., ``'layer_norm'``,
        ``'rms_norm'``, ``'batch_norm'``, ``'band_rms'``).
    :type normalization_type: str
    :param normalization_kwargs: Optional dictionary of kwargs for the normalization layer.
    :type normalization_kwargs: dict, optional
    :param forget_gate_activation: sLSTM forget gate activation, ``'sigmoid'`` or ``'exp'``.
    :type forget_gate_activation: str
    :param dropout_rate: Dropout rate for the FFN.
    :type dropout_rate: float
    :param kernel_initializer: Initializer for kernel weights.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param recurrent_initializer: Initializer for recurrent weights.
    :type recurrent_initializer: str or keras.initializers.Initializer
    :param bias_initializer: Initializer for bias weights.
    :type bias_initializer: str or keras.initializers.Initializer
    :param kernel_regularizer: Optional regularizer for kernel weights.
    :type kernel_regularizer: keras.regularizers.Regularizer, optional
    :param recurrent_regularizer: Optional regularizer for recurrent weights.
    :type recurrent_regularizer: keras.regularizers.Regularizer, optional
    :param bias_regularizer: Optional regularizer for bias weights.
    :type bias_regularizer: keras.regularizers.Regularizer, optional
    :param kwargs: Additional arguments for the Layer base class.
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
        """
        Build all sub-layers.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: tuple
        """
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
        """
        Forward pass through the sLSTM residual block.

        :param inputs: Input tensor of shape ``(batch_size, seq_len, units)``.
        :type inputs: keras.KerasTensor
        :param training: Whether the layer is in training mode.
        :type training: bool, optional
        :param mask: Optional mask tensor.
        :type mask: keras.KerasTensor, optional
        :return: Output tensor of shape ``(batch_size, seq_len, units)``.
        :rtype: keras.KerasTensor
        """
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
        """
        Return the configuration of the layer for serialization.

        :return: Configuration dictionary.
        :rtype: dict
        """
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

    Implements the architecture from Figure 11 of the xLSTM paper: a residual
    block that up-projects the input, applies depthwise causal convolution with
    swish activation, processes through mLSTM, normalizes, and down-projects
    back to the original dimension with a skip connection.

    **Architecture Overview:**

    .. code-block:: text

        Input: (batch, seq_len, units)
                │
                ├──────────────────────────────────┐
                ▼                                  │ (residual)
        ┌───────────────────────────┐              │
        │  Up-Projection Dense      │              │
        │  (units → units * exp)    │              │
        └───────────┬───────────────┘              │
                    ▼                              │
        ┌───────────────────────────┐              │
        │  Depthwise Causal Conv1D  │              │
        └───────────┬───────────────┘              │
                    ▼                              │
        ┌───────────────────────────┐              │
        │       Swish Activation    │              │
        └───────────┬───────────────┘              │
                    ▼                              │
        ┌───────────────────────────┐              │
        │       mLSTMLayer          │              │
        └───────────┬───────────────┘              │
                    ▼                              │
        ┌───────────────────────────┐              │
        │     Normalization         │              │
        └───────────┬───────────────┘              │
                    ▼                              │
        ┌───────────────────────────┐              │
        │  Down-Projection Dense    │              │
        │  (units * exp → units)    │              │
        └───────────┬───────────────┘              │
                    ▼                              │
                  ( + ) ◄──────────────────────────┘
                    │
                    ▼
        Output: (batch, seq_len, units)

    :param units: Dimensionality of the layer.
    :type units: int
    :param expansion_factor: Expansion factor for the internal dimension.
    :type expansion_factor: int
    :param num_heads: Number of mLSTM attention heads.
    :type num_heads: int
    :param conv_kernel_size: Kernel size for the depthwise causal convolution.
    :type conv_kernel_size: int
    :param normalization_type: Type of normalization to apply.
    :type normalization_type: str
    :param normalization_kwargs: Optional dictionary of kwargs for the normalization layer.
    :type normalization_kwargs: dict, optional
    :param kernel_initializer: Initializer for kernel weights.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param recurrent_initializer: Initializer for recurrent weights.
    :type recurrent_initializer: str or keras.initializers.Initializer
    :param bias_initializer: Initializer for bias weights.
    :type bias_initializer: str or keras.initializers.Initializer
    :param kernel_regularizer: Optional regularizer for kernel weights.
    :type kernel_regularizer: keras.regularizers.Regularizer, optional
    :param recurrent_regularizer: Optional regularizer for recurrent weights.
    :type recurrent_regularizer: keras.regularizers.Regularizer, optional
    :param bias_regularizer: Optional regularizer for bias weights.
    :type bias_regularizer: keras.regularizers.Regularizer, optional
    :param kwargs: Additional arguments for the Layer base class.
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
        """
        Build all sub-layers.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: tuple
        """
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
        """
        Forward pass through the mLSTM residual block.

        :param inputs: Input tensor of shape ``(batch_size, seq_len, units)``.
        :type inputs: keras.KerasTensor
        :param training: Whether the layer is in training mode.
        :type training: bool, optional
        :param mask: Optional mask tensor.
        :type mask: keras.KerasTensor, optional
        :return: Output tensor of shape ``(batch_size, seq_len, units)``.
        :rtype: keras.KerasTensor
        """
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
        """
        Return the configuration of the layer for serialization.

        :return: Configuration dictionary.
        :rtype: dict
        """
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
