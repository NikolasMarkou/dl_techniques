"""
xLSTM (Extended Long Short-Term Memory) layers.

Implements the xLSTM architecture from "xLSTM: Extended Long Short-Term Memory"
(Beck et al. 2024, arXiv:2405.04517v2).

The module exports six classes, in two families:

- ``sLSTMCell`` / ``sLSTMLayer`` / ``sLSTMBlock`` -- the scalar variant of
  Section 2.2. Memory is a vector, gating is exponential.
- ``mLSTMCell`` / ``mLSTMLayer`` / ``mLSTMBlock`` -- the matrix variant of
  Section 2.3. Memory is a matrix updated by a covariance rule.

In each family the ``Cell`` runs one timestep, the ``Layer`` wraps that cell in a
``keras.layers.RNN`` to run a sequence, and the ``Block`` adds the residual
wrapper the paper draws in Figure 10 (sLSTM) and Figure 11 (mLSTM).

Normalization layers come from the norms factory and feed-forward networks from
the FFN factory, so both are selected by string.

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
from dl_techniques.utils.activation_serialization import (
    serialize_activation,
    deserialize_activation,
)
from dl_techniques.utils.dtype_policy import stability_floor
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.time_series.xlstm_blocks")
class sLSTMCell(keras.layers.Layer):
    """
    Scalar LSTM (sLSTM) cell with exponential gating and a normalizer state.

    This is the sLSTM of Section 2.2 of the xLSTM paper. It differs from a stock
    LSTM in three ways. The input gate is exponential rather than sigmoid, which
    lets one timestep dominate the memory. A normalizer state ``n_t`` accumulates
    the same gates and divides them back out, so the memory stays on scale. A
    log-domain running maximum ``m_t`` keeps the exponentials from overflowing.

    The cell carries four states per timestep: ``h_t``, ``c_t``, ``n_t`` and
    ``m_t``, each of shape ``(batch, units)``. The hidden state is
    ``h_t = o_t * (c_t / (n_t + eps))``. The ``eps`` is what keeps a near-zero
    normalizer from turning the output into a NaN; it is
    ``utils.dtype_policy.stability_floor(compute_dtype, 1e-8)``, not the bare
    literal, because ``float16(1e-8)`` is exactly ``0.0``.

    **Gate equations** (per timestep t):

        i_t = exp(W_i @ x_t + R_i @ h_{t-1} + b_i)
        f_t = activation(W_f @ x_t + R_f @ h_{t-1} + b_f)
        o_t = sigmoid(W_o @ x_t + R_o @ h_{t-1} + b_o)
        z_t = tanh(W_z @ x_t + R_z @ h_{t-1} + b_z)

    **State updates:**

        c_t = f_t * c_{t-1} + i_t * z_t
        n_t = f_t * n_{t-1} + i_t
        h_t = o_t * (c_t / (n_t + eps))

    **Stabilization** (Equations 15-17 in paper):

        m_t = max(m_{t-1} + log(f_t), log(i_t))
        i_t_tilde = exp(log(i_t) - m_t)
        f_t_tilde = exp(log(f_t) + m_{t-1} - m_t)

    ``log(i_t)`` is the raw input-gate pre-activation, used as is. With
    ``forget_gate_activation='exp'`` the same is true of ``log(f_t)``. With
    ``'sigmoid'`` the code takes ``log(sigmoid(f_proj) + eps)``; the ``eps``
    stops a saturated forget gate from giving ``log(0)``. It is
    ``stability_floor(compute_dtype, 1e-8)`` rather than the literal, so the
    floor survives a ``float16`` compute dtype.

    **Architecture Overview:**

    .. code-block:: text

        x_t: (batch, input_dim)   h_{t-1}: (batch, units)
              │                         │
              ▼                         ▼
        ┌─────────────┐         ┌──────────────┐
        │ W @ x_t + b │         │ R @ h_{t-1}  │
        └─────┬───────┘         └──────┬───────┘
              └──────────┬─────────────┘
                         │ (batch, 4 * units)
                         ▼
             ┌──────────────────────────┐
             │ split into i, f, o, z    │
             └────────────┬─────────────┘
                          │ each (batch, units)
                          ▼
             ┌──────────────────────────┐
             │ m_t = max(m_{t-1}+log f, │
             │            log i)        │
             │ i, f rescaled by exp     │
             └────────────┬─────────────┘
                          ▼
             ┌───────────────────────────┐
             │ c_t = f * c_{t-1} + i * z │
             │ n_t = f * n_{t-1} + i     │
             └────────────┬──────────────┘
                          ▼
             ┌──────────────────────────┐
             │ h_t = o*(c_t/(n_t+1e-8)) │
             └────────────┬─────────────┘
                          ▼
        h_t: (batch, units), plus states [h_t, c_t, n_t, m_t]

    Every state is (batch, units); m_t is the log-domain stabilizer.

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

    :raises ValueError: If ``units`` is not positive.
    :raises ValueError: If ``forget_gate_activation`` is not ``'sigmoid'`` or
        ``'exp'``.

    Input shape:
        2D tensor ``(batch_size, input_dim)``, one timestep.

    Output shape:
        2D tensor ``(batch_size, units)``, plus the four state tensors.

    Example:
        .. code-block:: python

            cell = sLSTMCell(units=32)
            rnn = keras.layers.RNN(cell, return_sequences=True)
            y = rnn(keras.random.normal((2, 10, 8)))
            # y.shape == (2, 10, 32)

    :ivar state_size: ``[units, units, units, units]`` for ``[h, c, n, m]``.
    :vartype state_size: list of int
    :ivar output_size: Equal to ``units``.
    :vartype output_size: int
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
        """
        Validate the configuration and record the state layout.

        See the class docstring for the full parameter list.

        :raises ValueError: If ``units`` is not positive, or
            ``forget_gate_activation`` is not ``'sigmoid'`` or ``'exp'``.
        """
        super().__init__(**kwargs)

        if units <= 0:
            raise ValueError(f"`units` must be positive, but got {units}")
        self.units = units

        if forget_gate_activation not in ['sigmoid', 'exp']:
            raise ValueError(
                f"`forget_gate_activation` must be 'sigmoid' or 'exp', "
                f"but got {forget_gate_activation}"
            )
        self.forget_gate_activation = deserialize_activation(forget_gate_activation)

        # Store initializers and regularizers
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = keras.regularizers.get(recurrent_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # Activation functions
        self.f_activation = (
            activations.get(forget_gate_activation)
            if forget_gate_activation != "exp"
            else None
        )
        self.o_activation = activations.get('sigmoid')
        self.z_activation = activations.get('tanh')

        # RNN cell state layout, in the order call() unpacks it:
        # [h, c, n, m], each of shape (batch, units).
        self.state_size = [self.units, self.units, self.units, self.units]
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

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the per-time-step output shape of the cell.

        :param input_shape: Per-step input shape ``(batch_size, input_dim)``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Per-step output shape ``(batch_size, units)``.
        :rtype: Tuple[Optional[int], ...]
        """
        return (input_shape[0], self.units)

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
            # DECISION plan-2026-08-31T134711-6271592d/D-008
            # Do NOT write a bare `+ 1e-8`: float16(1e-8) is exactly 0.0, so
            # the floor vanishes, a saturated sigmoid gives log(0) = -inf and
            # the gradient is inf/NaN at every weight. `stability_floor`
            # lifts it to the smallest NORMAL magnitude of the compute dtype
            # (6.10e-05 in float16) and is a no-op in float32/float64.
            log_f_t = ops.log(
                self.f_activation(f_proj)
                + stability_floor(self.compute_dtype, 1e-8)
            )

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
        # DECISION plan-2026-08-31T134711-6271592d/D-008: same floor, same
        # reason -- a bare `+ 1e-8` is 0.0 under float16 and `c_t / n_t` is
        # 0/0 = NaN as soon as the input gate underflows.
        h_t = o_t * (c_t / (n_t + stability_floor(self.compute_dtype, 1e-8)))

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
            'forget_gate_activation': serialize_activation(self.forget_gate_activation),
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer': keras.regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.time_series.xlstm_blocks")
class sLSTMLayer(keras.layers.Layer):
    """
    Scalar LSTM (sLSTM) layer for processing sequences.

    Wraps ``sLSTMCell`` in a ``keras.layers.RNN`` so it runs over a whole
    sequence. It is a drop-in replacement for ``keras.layers.LSTM``, with
    exponential gating instead of sigmoid gating on the input gate.

    **Architecture Overview:**

    .. code-block:: text

        Input: (batch, seq_len, input_dim)
                    │
                    ▼
        ┌───────────────────────────┐
        │   keras.layers.RNN        │
        │   ┌───────────────────┐   │
        │   │   sLSTMCell       │   │
        │   │  (one timestep)   │   │
        │   └───────────────────┘   │
        └───────────┬───────────────┘
                    │
          ┌─────────┴──────────┐
          ▼                    ▼
    return_sequences=True  return_sequences=False
    (batch, seq_len, units)  (batch, units)

    With return_state=True the four final states [h, c, n, m],
    each (batch, units), are appended to whichever leaf applies.

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

    Input shape:
        3D tensor ``(batch_size, seq_len, input_dim)``.

    Output shape:
        3D tensor ``(batch_size, seq_len, units)`` when
        ``return_sequences=True``, otherwise 2D ``(batch_size, units)``.

    Example:
        .. code-block:: python

            layer = sLSTMLayer(units=32, forget_gate_activation='exp')
            y = layer(keras.random.normal((2, 10, 8)))
            # y.shape == (2, 10, 32)

    :ivar cell: The wrapped :class:`sLSTMCell`.
    :vartype cell: sLSTMCell
    :ivar rnn: The ``keras.layers.RNN`` that drives the cell.
    :vartype rnn: keras.layers.RNN
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
        """
        Store the configuration and create the cell and its RNN wrapper.

        See the class docstring for the full parameter list. Validation of
        ``units`` and ``forget_gate_activation`` happens inside ``sLSTMCell``.
        """
        super().__init__(**kwargs)

        self.units = units
        self.forget_gate_activation = deserialize_activation(forget_gate_activation)
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.go_backwards = go_backwards
        self.stateful = stateful
        self.unroll = unroll
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = keras.regularizers.get(recurrent_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

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
            'forget_gate_activation': serialize_activation(self.forget_gate_activation),
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


@register_dl_technique("dl_techniques.layers.time_series.xlstm_blocks")
class mLSTMCell(keras.layers.Layer):
    """
    Matrix LSTM (mLSTM) cell with matrix memory and covariance update rule.

    This is the mLSTM of Section 2.3 of the xLSTM paper. Memory is a matrix
    ``C_t`` of shape ``(key_dim, value_dim)`` per head, not a vector. Each
    timestep adds an outer product to it, which is why it stores more than a
    scalar LSTM of the same width.

    One combined kernel produces all six projections: query, key, value, and the
    input, forget and output gates. Queries and keys are ``key_dim`` wide per
    head, values are ``value_dim`` wide, and the input and forget gates are one
    scalar per head.

    The exponential input gate is stabilized in log space by a running maximum
    ``m_t``, one scalar per head. This mirrors ``sLSTMCell`` and follows Eq.
    15-17 of the paper. Without it the matrix memory overflows fp32 at around 64
    timesteps. The stabilized form below is mathematically equivalent while the
    values stay finite:

        log_f = log(sigmoid(f_proj) + 1e-8)
        m_t   = max(m_{t-1} + log_f, i_proj)            # log-domain stabilizer
        i_t   = exp(i_proj - m_t)                       # bounded in (0, 1]
        f_t   = exp(m_{t-1} + log_f - m_t)              # bounded
        C_t   = f_t * C_{t-1} + i_t * (v_t (outer) k_t^T)
        n_t   = f_t * n_{t-1} + i_t * k_t
        h_t   = o_t * (C_t^T @ q_t / (max(|n_t^T @ q_t|, exp(-m_t)) + 1e-8))

    ``i_proj`` is the raw input-gate pre-activation and the code uses it
    directly as ``log(i_t)``. The two ``1e-8`` terms are in the code, each
    passed through ``stability_floor(compute_dtype, 1e-8)`` so it cannot round
    to ``0.0`` under ``float16``: one stops ``log(0)`` on a saturated forget
    gate, the other stops a divide by zero.
    ``C_t`` is stored as ``(key_dim, value_dim)``, so reading it back with the
    query needs the transpose. Don't drop the ``^T``.

    **Architecture Overview:**

    .. code-block:: text

        x_t: (batch, input_dim)   h_{t-1}: (batch, units)
              │                         │
              ▼                         ▼
        ┌─────────────┐         ┌──────────────┐
        │ W @ x_t + b │         │ R @ h_{t-1}  │
        └─────┬───────┘         └──────┬───────┘
              └──────────┬─────────────┘
                         ▼
             ┌──────────────────────────┐
             │ split into q, k, v,      │
             │           i, f, o        │
             └────────────┬─────────────┘
                          ▼
             ┌──────────────────────────┐
             │ m_t = max(m_{t-1}+log_f, │
             │            i_proj)       │
             │ i, f rescaled by exp     │
             └────────────┬─────────────┘
                          ▼
             ┌───────────────────────────┐
             │ C_t = f * C_{t-1}         │
             │       + i * (v_t ⊗ k_t^T) │
             │ n_t = f * n_{t-1} + i*k_t │
             └────────────┬──────────────┘
                          ▼
             ┌──────────────────────────┐
             │ h_t = o * (C_t^T @ q_t)  │
             │  / (max(|n_t . q_t|,     │
             │       exp(-m_t)) + 1e-8) │
             └────────────┬─────────────┘
                          ▼
        h_t: (batch, units), states [h_t, C_t, n_t, m_t]

    q and k are (batch, heads, key_dim), v is (batch, heads, value_dim).
    i, f and m carry one scalar per head. C_t and n_t are stored flat.

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

    Input shape:
        2D tensor ``(batch_size, input_dim)``, one timestep.

    Output shape:
        2D tensor ``(batch_size, units)``, plus the four state tensors.

    Example:
        .. code-block:: python

            cell = mLSTMCell(units=32, num_heads=4)
            rnn = keras.layers.RNN(cell, return_sequences=True)
            y = rnn(keras.random.normal((2, 10, 8)))
            # y.shape == (2, 10, 32)

    :ivar matrix_memory_size: ``num_heads * key_dim * value_dim``, the flat size
        of ``C_t``.
    :vartype matrix_memory_size: int
    :ivar normalizer_size: ``num_heads * key_dim``, the flat size of ``n_t``.
    :vartype normalizer_size: int
    :ivar state_size: ``[units, matrix_memory_size, normalizer_size, num_heads]``.
    :vartype state_size: list of int
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
        """
        Validate the configuration and record the state layout.

        See the class docstring for the full parameter list.

        :raises ValueError: If ``units`` or ``num_heads`` is not positive, or if
            ``units`` is not divisible by ``num_heads``.
        """
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
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = keras.regularizers.get(recurrent_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # State layout, in the order call() unpacks it: [h, C, n, m].
        # h is (units,). C is (num_heads, key_dim, value_dim), stored flat.
        # n is (num_heads, key_dim), stored flat. m is (num_heads,) -- the
        # log-domain max-stabilizer, one scalar per head, mirroring sLSTMCell.
        # DECISION plan_2026-06-11_50891da1/D-001: the 4th state m_t is the
        # paper-correct (Beck et al. 2024) log-domain max-stabilizer. Do NOT
        # remove it or revert to the bare exp() input gate -- that form overflows
        # fp32 at seq>=64 (repro_isolate.py). Do NOT add a forecaster-only wrapper
        # cell or clip the gate instead: both were rejected (see decisions.md D-001).
        self.matrix_memory_size = self.num_heads * self.key_dim * self.value_dim
        self.normalizer_size = self.num_heads * self.key_dim

        self.state_size = [
            self.units,
            self.matrix_memory_size,
            self.normalizer_size,
            self.num_heads,
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

        # One combined kernel produces all six projections. The terms below are
        # in the order call() splits them: q, k (num_heads * key_dim each),
        # v (num_heads * value_dim), i and f (one scalar per head), then
        # o (the full units dimension).
        total_proj_size = (
            self.num_heads * self.key_dim +
            self.num_heads * self.key_dim +
            self.num_heads * self.value_dim +
            self.num_heads +
            self.num_heads +
            self.units
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

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the per-time-step output shape of the cell.

        :param input_shape: Per-step input shape ``(batch_size, input_dim)``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Per-step output shape ``(batch_size, units)``.
        :rtype: Tuple[Optional[int], ...]
        """
        return (input_shape[0], self.units)

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
        :param states: List of state tensors ``[h_tm1, C_tm1_flat, n_tm1_flat, m_tm1]``.
        :type states: list of keras.KerasTensor
        :param training: Whether the layer is in training mode.
        :type training: bool, optional
        :return: Tuple of ``(h_t, [h_t, C_t_flat, n_t_flat, m_t])``.
        :rtype: tuple
        """
        h_tm1, C_tm1_flat, n_tm1_flat, m_tm1 = states
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

        # Stabilized gates, mirroring sLSTMCell. The log-domain running maximum
        # m_t keeps the exponential input gate bounded, so the matrix-memory
        # recurrence cannot overflow fp32 over a long sequence. log_f is the
        # same sigmoid forget gate as the unstabilized form, taken in log space.
        # log_f, m_t, i_t and f_t are (batch_size, num_heads); o_t is
        # (batch_size, units).
        # DECISION plan_2026-06-11_50891da1/D-001: do NOT revert to a bare
        # `i_t = ops.exp(i_proj)`; that overflows fp32 at seq>=64. See decisions.md D-001.
        # DECISION plan-2026-08-31T134711-6271592d/D-008: the floor must be
        # derived from the compute dtype; a literal 1e-8 is exactly 0.0 in
        # float16 and log(sigmoid(-30)) is then -inf with an inf gradient.
        log_f = ops.log(
            ops.sigmoid(f_proj) + stability_floor(self.compute_dtype, 1e-8)
        )
        m_t = ops.maximum(m_tm1 + log_f, i_proj)
        i_t = ops.exp(i_proj - m_t)
        f_t = ops.exp(m_tm1 + log_f - m_t)
        o_t = ops.sigmoid(o_proj)

        # Reshape gates for broadcasting
        i_t = ops.reshape(i_t, (batch_size, self.num_heads, 1, 1))
        f_t = ops.reshape(f_t, (batch_size, self.num_heads, 1, 1))

        # Matrix memory update: C_t = f_t * C_{t-1} + i_t * (v_t ⊗ k_t^T).
        # The outer product is a matmul of v_t as a column, (batch, heads,
        # value_dim, 1), with k_t as a row, (batch, heads, 1, key_dim). That
        # gives (batch, heads, value_dim, key_dim).
        v_t_expanded = ops.expand_dims(v_t, axis=-1)
        k_t_expanded = ops.expand_dims(k_t, axis=-2)
        outer_product = ops.matmul(v_t_expanded, k_t_expanded)

        # Transpose to match C format (batch, heads, key_dim, value_dim)
        outer_product = ops.transpose(outer_product, [0, 1, 3, 2])

        C_t = f_t * C_tm1 + i_t * outer_product

        # Normalizer update: n_t = f_t * n_{t-1} + i_t * k_t
        f_t_norm = ops.reshape(f_t, (batch_size, self.num_heads, 1))
        i_t_norm = ops.reshape(i_t, (batch_size, self.num_heads, 1))
        n_t = f_t_norm * n_tm1 + i_t_norm * k_t

        # Read the memory back out with the query: C_t^T @ q_t.
        # C_t is (batch, heads, key_dim, value_dim), so the transpose is
        # (batch, heads, value_dim, key_dim) and q_t as a column is
        # (batch, heads, key_dim, 1). The matmul gives
        # (batch, heads, value_dim, 1), squeezed to (batch, heads, value_dim).
        q_t_expanded = ops.expand_dims(q_t, axis=-1)
        memory_retrieval = ops.matmul(
            ops.transpose(C_t, [0, 1, 3, 2]),
            q_t_expanded
        )
        memory_retrieval = ops.squeeze(memory_retrieval, axis=-1)

        # The stabilized mLSTM denominator, max(|n_t^T @ q_t|, exp(-m_t)) +
        # 1e-8 (Beck et al. 2024). exp(-m_t) is a floor on the divisor, so a
        # near-zero n_t^T q_t cannot blow up the retrieval. The 1e-8 covers
        # the case where exp(-m_t) itself underflows to zero -- which it does
        # in float16 for m_t >~ 20, hence the dtype-derived floor below.
        # nq, m_t3 and normalization are all (batch, heads, 1).
        # DECISION plan_2026-06-11_50891da1/D-001: keep the exp(-m_t) floor; the
        # bare `+ 1e-8` form was insufficient. See decisions.md D-001.
        nq = ops.sum(n_t * q_t, axis=-1, keepdims=True)
        m_t3 = ops.reshape(m_t, (batch_size, self.num_heads, 1))
        # DECISION plan-2026-08-31T134711-6271592d/D-008: `exp(-m_t)` itself
        # underflows to 0.0 in float16 for m_t >~ 20, and a literal 1e-8 is
        # ALSO 0.0 there, so the divisor is exactly zero. Keep the floor
        # dtype-derived.
        normalization = (
            ops.maximum(ops.abs(nq), ops.exp(-m_t3))
            + stability_floor(self.compute_dtype, 1e-8)
        )

        # Divide, giving (batch, heads, value_dim).
        normalized_retrieval = memory_retrieval / normalization

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

        return h_t, [h_t, C_t_flat, n_t_flat, m_t]

    def get_initial_state(
        self,
        batch_size: Optional[int] = None,
    ) -> List[keras.KerasTensor]:
        """
        Get initial states for the cell.

        :param batch_size: Batch size for the initial state tensors.
        :type batch_size: int, optional
        :return: List of initial state tensors ``[h_0, C_0_flat, n_0_flat, m_0]``.
        :rtype: list of keras.KerasTensor
        """
        return [
            ops.zeros((batch_size, self.units), dtype=self.compute_dtype),
            ops.zeros((batch_size, self.matrix_memory_size), dtype=self.compute_dtype),
            ops.zeros((batch_size, self.normalizer_size), dtype=self.compute_dtype),
            ops.zeros((batch_size, self.num_heads), dtype=self.compute_dtype),
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


@register_dl_technique("dl_techniques.layers.time_series.xlstm_blocks")
class mLSTMLayer(keras.layers.Layer):
    """
    Matrix LSTM (mLSTM) layer for processing sequences.

    Wraps ``mLSTMCell`` in a ``keras.layers.RNN`` so it runs over a whole
    sequence. This is the layer to use when you want matrix memory in a model
    without driving the cell yourself.

    **Architecture Overview:**

    .. code-block:: text

        Input: (batch, seq_len, input_dim)
                    │
                    ▼
        ┌───────────────────────────┐
        │   keras.layers.RNN        │
        │   ┌───────────────────┐   │
        │   │   mLSTMCell       │   │
        │   │  (one timestep)   │   │
        │   └───────────────────┘   │
        └───────────┬───────────────┘
                    │
          ┌─────────┴──────────┐
          ▼                    ▼
    return_sequences=True  return_sequences=False
    (batch, seq_len, units)  (batch, units)

    With return_state=True the four final states are appended to
    whichever leaf applies: h (batch, units), C flattened to
    (batch, heads*key_dim*value_dim), n flattened to
    (batch, heads*key_dim), and m (batch, heads).

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

    Input shape:
        3D tensor ``(batch_size, seq_len, input_dim)``.

    Output shape:
        3D tensor ``(batch_size, seq_len, units)`` when
        ``return_sequences=True``, otherwise 2D ``(batch_size, units)``.

    Example:
        .. code-block:: python

            layer = mLSTMLayer(units=32, num_heads=4)
            y = layer(keras.random.normal((2, 10, 8)))
            # y.shape == (2, 10, 32)

    :ivar cell: The wrapped :class:`mLSTMCell`.
    :vartype cell: mLSTMCell
    :ivar rnn: The ``keras.layers.RNN`` that drives the cell.
    :vartype rnn: keras.layers.RNN
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
        """
        Store the configuration and create the cell and its RNN wrapper.

        See the class docstring for the full parameter list. Validation of
        ``units`` and ``num_heads`` happens inside ``mLSTMCell``.
        """
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
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = keras.regularizers.get(recurrent_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

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

@register_dl_technique("dl_techniques.layers.time_series.xlstm_blocks")
class sLSTMBlock(keras.layers.Layer):
    """
    sLSTM residual block with post-normalization.

    This is Figure 10 of the xLSTM paper. The input goes through ``sLSTMLayer``,
    then normalization, then a feed-forward network, and the original input is
    added back at the end. Normalization sits after the recurrence, not before
    it, which is what makes this post-norm.

    Both the normalization type and the FFN type are strings resolved by the
    repo factories, so the block can be reshaped without subclassing.

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

    Input shape:
        3D tensor ``(batch_size, seq_len, units)``.

    Output shape:
        3D tensor ``(batch_size, seq_len, units)``. The residual fixes the
        output width to ``units``.

    Example:
        .. code-block:: python

            block = sLSTMBlock(units=32, ffn_type='swiglu')
            y = block(keras.random.normal((2, 10, 32)))
            # y.shape == (2, 10, 32)

    :ivar slstm: The recurrent path, an :class:`sLSTMLayer` with
        ``return_sequences=True``.
    :vartype slstm: sLSTMLayer
    :ivar norm: Normalization layer built by the norms factory.
    :vartype norm: keras.layers.Layer
    :ivar ffn: Feed-forward network built by the FFN factory.
    :vartype ffn: keras.layers.Layer
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
        """
        Store the configuration and create the three sub-layers.

        See the class docstring for the full parameter list. The sub-layers are
        created here and built in ``build()``.
        """
        super().__init__(**kwargs)

        self.units = units
        self.ffn_type = ffn_type
        self.ffn_expansion_factor = ffn_expansion_factor
        self.normalization_type = normalization_type
        self.normalization_kwargs = normalization_kwargs or {}
        self.forget_gate_activation = deserialize_activation(forget_gate_activation)
        self.dropout_rate = dropout_rate
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = keras.regularizers.get(recurrent_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

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

        # Build the FFN. Normalization does not change the shape, so the FFN
        # sees the sLSTM output shape.
        norm_output_shape = slstm_output_shape
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

    def compute_output_shape(self, input_shape):
        """
        Compute the output shape. The residual makes it equal the input shape.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: tuple
        :return: The same shape tuple.
        :rtype: tuple
        """
        return input_shape

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
            'forget_gate_activation': serialize_activation(self.forget_gate_activation),
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

@register_dl_technique("dl_techniques.layers.time_series.xlstm_blocks")
class mLSTMBlock(keras.layers.Layer):
    """
    mLSTM residual block with an up-projection around the recurrence.

    This is Figure 11 of the xLSTM paper. The input is projected up to
    ``units * expansion_factor``, mixed by a depthwise causal Conv1D, passed
    through swish, run through ``mLSTMLayer``, normalized, and projected back
    down to ``units``. The original input is added at the end.

    The convolution uses ``padding='causal'`` and ``groups=inner_dim``, so it
    mixes across time only, never across channels, and never looks ahead.

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

    Input shape:
        3D tensor ``(batch_size, seq_len, units)``.

    Output shape:
        3D tensor ``(batch_size, seq_len, units)``. The residual fixes the
        output width to ``units``.

    Example:
        .. code-block:: python

            block = mLSTMBlock(units=32, expansion_factor=2, num_heads=4)
            y = block(keras.random.normal((2, 10, 32)))
            # y.shape == (2, 10, 32)

    :ivar inner_dim: ``units * expansion_factor``, the width of the inner path.
    :vartype inner_dim: int
    :ivar up_proj: Dense layer widening ``units`` to ``inner_dim``.
    :vartype up_proj: keras.layers.Dense
    :ivar conv: Depthwise causal Conv1D over the inner path.
    :vartype conv: keras.layers.Conv1D
    :ivar mlstm: The recurrent path, an :class:`mLSTMLayer` at ``inner_dim``.
    :vartype mlstm: mLSTMLayer
    :ivar norm: Normalization layer built by the norms factory.
    :vartype norm: keras.layers.Layer
    :ivar down_proj: Dense layer narrowing ``inner_dim`` back to ``units``.
    :vartype down_proj: keras.layers.Dense
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
        """
        Store the configuration and create the five sub-layers.

        See the class docstring for the full parameter list. The sub-layers are
        created here and built in ``build()``.
        """
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
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = keras.regularizers.get(recurrent_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

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

        # Depthwise conv for mixing across time. groups == filters makes it
        # depthwise, so no channel is mixed with any other.
        self.conv = layers.Conv1D(
            filters=self.inner_dim,
            kernel_size=conv_kernel_size,
            padding='causal',
            groups=self.inner_dim,
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

    def compute_output_shape(self, input_shape):
        """
        Compute the output shape. The residual makes it equal the input shape.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: tuple
        :return: The same shape tuple.
        :rtype: tuple
        """
        return input_shape

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
