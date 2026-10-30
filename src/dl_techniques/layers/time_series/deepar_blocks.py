"""
Building blocks for the DeepAR probabilistic forecaster.

Two layers live here. ``ScaleLayer`` divides inputs by a per-item scale and
multiplies outputs back up. ``DeepARCell`` wraps a Keras ``LSTMCell`` so it
can be driven step by step. The likelihood heads that turn LSTM hidden states
into distribution parameters live in
``dl_techniques.layers.heads.time_series.likelihood_heads``.

Reference:
    DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks
    Salinas et al., 2019
    https://arxiv.org/abs/1704.04110
"""

import keras
from keras import ops, layers
from typing import Optional, Tuple, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.time_series.deepar_blocks")
class ScaleLayer(keras.layers.Layer):
    """
    Divides inputs by an item-dependent scale, or multiplies outputs by it.

    DeepAR trains on series whose magnitudes differ by orders of magnitude.
    This layer is how one model copes with all of them: autoregressive inputs
    are divided by a per-item scale, and the predicted likelihood parameters
    are multiplied back up.

    The scale is

        ``nu = mean(conditioning_range) + epsilon``

    Forward scaling divides by ``nu``. Inverse scaling multiplies by whatever
    ``scale`` the caller passes. The layer applies exactly one division or one
    multiplication and holds no per-parameter policy, so the call site picks
    the scale:

    - Gaussian ``mu`` **and** ``sigma``: pass ``nu``. The forward path divides
      ``z`` by ``nu`` exactly once, so ``mu = nu * mu~`` and
      ``sigma = nu * sigma~``; ``sigma`` is a first-moment-scale quantity
      here, not a variance.
    - Negative-binomial shape ``alpha``: pass ``1 / sqrt(nu)``.

    **Architecture Overview:**

    .. code-block:: text

        Input: x [B, T, F]
        Optional input: scale [B, 1, F], passed by the caller
                    │
                    ▼
        ┌──────────────────────────────────┐
        │ if scale is None and             │
        │    scale_per_sample:             │
        │      scale = mean(x, axis=1)+eps │
        └───────────────────┬──────────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
        ┌───────────┐ ┌───────────┐ ┌───────────┐
        │  forward  │ │  inverse  │ │  no scale │
        │ x / scale │ │ x * scale │ │  x as-is  │
        └─────┬─────┘ └─────┬─────┘ └─────┬─────┘
              ▼             ▼             ▼
           Scaled       Descaled     Passthrough

    The third leaf is reached when ``scale`` is None and ``scale_per_sample``
    is False. The layer then returns its input untouched.

    Note:
        Do not restore ``sqrt(nu)`` for the Gaussian ``sigma``. This
        docstring said until 2026-08-15 that inverse scaling for the
        Gaussian ``sigma`` "scales by sqrt(nu)". That is wrong: no call site
        ever did it, and the diagram above (``x * scale``) always
        contradicted it.

    :param scale_per_sample: If True, compute the scale from the input when the
        caller passes none. If False, use only the provided scale. Defaults to
        True.
    :type scale_per_sample: bool
    :param epsilon: Constant added to the computed mean. Keeps the scale away
        from zero for a near-empty series. Defaults to 1.0.
    :type epsilon: float
    :param kwargs: Additional arguments for the Layer base class.
    """

    def __init__(
            self,
            scale_per_sample: bool = True,
            epsilon: float = 1.0,
            **kwargs: Any
    ) -> None:
        """
        Initialize the ScaleLayer.

        :param scale_per_sample: If True, compute the scale per sample when the
            caller passes none.
        :type scale_per_sample: bool
        :param epsilon: Constant added to the computed mean.
        :type epsilon: float
        :param kwargs: Additional arguments for the Layer base class.
        """
        super().__init__(**kwargs)
        self.scale_per_sample = scale_per_sample
        self.epsilon = epsilon

    def call(
            self,
            inputs: keras.KerasTensor,
            scale: Optional[keras.KerasTensor] = None,
            inverse: bool = False
    ) -> keras.KerasTensor:
        """
        Apply scaling or inverse scaling.

        With no usable scale the input is returned unchanged. That happens when
        ``scale`` is None and ``scale_per_sample`` is False.

        :param inputs: Tensor to scale.
        :type inputs: keras.KerasTensor
        :param scale: Pre-computed scale. If None and ``scale_per_sample`` is
            True, it is computed from ``inputs``.
        :type scale: keras.KerasTensor or None
        :param inverse: If True, multiply by the scale. If False, divide by it.
        :type inverse: bool
        :return: The scaled, inverse-scaled or unchanged tensor.
        :rtype: keras.KerasTensor
        """
        if scale is None and self.scale_per_sample:
            # Mean over the time axis, plus epsilon.
            scale = ops.mean(inputs, axis=1, keepdims=True) + self.epsilon

        if scale is None:
            return inputs

        if inverse:
            return inputs * scale
        else:
            return inputs / scale

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Return the output shape, which equals the input shape.

        Scaling is elementwise, so no dimension changes.

        :param input_shape: Shape of the input tensor.
        :type input_shape: tuple[int or None, ...]
        :return: The same shape.
        :rtype: tuple[int or None, ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return the layer configuration for serialization.

        :return: Dictionary containing the layer configuration.
        :rtype: dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'scale_per_sample': self.scale_per_sample,
            'epsilon': self.epsilon,
        })
        return config

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.time_series.deepar_blocks")
class DeepARCell(keras.layers.Layer):
    """
    One autoregressive DeepAR step, as a thin wrapper over a Keras LSTMCell.

    ``call`` forwards its input and states straight to the LSTM cell and
    returns what the cell returns. The recurrence DeepAR describes is

        ``h_t = h(h_{t-1}, z_{t-1}, x_t, Theta)``

    but this cell does not build ``[z_{t-1}, x_t]`` itself. The caller
    concatenates the previous observation with the covariates and passes the
    result as one tensor. Drive the cell with ``keras.layers.RNN``, or step it
    by hand for autoregressive sampling.

    **Architecture Overview:**

    .. code-block:: text

        Input: x_t [B, input_dim]   States: (h_{t-1}, c_{t-1})
                    │                        │
                    └───────────┬────────────┘
                                ▼
                ┌───────────────────────────────┐
                │ keras LSTMCell(units)         │
                │ dropout, recurrent_dropout    │
                └───────────────┬───────────────┘
                                ▼
                Output: h_t [B, units]
                States: (h_t, c_t), each [B, units]

    Nothing else happens in ``call``. There is no concatenation, no projection
    and no scaling stage inside this cell.

    :param units: Number of LSTM units. This is also the hidden width.
    :type units: int
    :param dropout_rate: Dropout on the LSTM inputs. Defaults to 0.0.
    :type dropout_rate: float
    :param recurrent_dropout_rate: Dropout on the recurrent connections.
        Defaults to 0.0.
    :type recurrent_dropout_rate: float
    :param kwargs: Additional arguments for the Layer base class.

    :ivar state_size: Width of one state, so ``units``. The cell itself carries
        two states of that width, and ``get_initial_state`` returns both.
    :vartype state_size: int
    """

    def __init__(
            self,
            units: int,
            dropout_rate: float = 0.0,
            recurrent_dropout_rate: float = 0.0,
            **kwargs: Any
    ) -> None:
        """
        Initialize the DeepARCell.

        :param units: Number of LSTM units.
        :type units: int
        :param dropout_rate: Dropout on the LSTM inputs.
        :type dropout_rate: float
        :param recurrent_dropout_rate: Dropout on the recurrent connections.
        :type recurrent_dropout_rate: float
        :param kwargs: Additional arguments for the Layer base class.
        """
        super().__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate
        self.recurrent_dropout_rate = recurrent_dropout_rate
        self.state_size = units

        # The whole computation of this cell.
        self.lstm_cell = layers.LSTMCell(
            units,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout_rate,
            name='lstm_cell'
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the LSTM sub-layer explicitly, then the cell itself.

        :param input_shape: Shape of the input tensor.
        :type input_shape: tuple[int or None, ...]
        """
        self.lstm_cell.build(input_shape)
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            states: Tuple[keras.KerasTensor, ...],
            training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, Tuple[keras.KerasTensor, ...]]:
        """
        Process one time step through the LSTM cell.

        :param inputs: Input at time t, shape ``(batch_size, input_dim)``. The
            caller has already concatenated the previous observation with the
            covariates.
        :type inputs: keras.KerasTensor
        :param states: State tensors from the previous step, ``(h, c)``.
        :type states: tuple[keras.KerasTensor, ...]
        :param training: Whether in training mode. Both dropout rates are
            applied only when this is True.
        :type training: bool or None
        :return: Tuple of (output, new_states), exactly as the LSTM cell
            returns them.
        :rtype: tuple[keras.KerasTensor, tuple[keras.KerasTensor, ...]]
        """
        output, new_states = self.lstm_cell(inputs, states, training=training)
        return output, new_states

    def get_initial_state(
            self,
            batch_size: Optional[int] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Get the initial state, delegated to the LSTM cell.

        Two tensors come back, the hidden state and the carry state, each of
        shape ``(batch_size, units)``.

        :param batch_size: Batch size for the initial state tensors.
        :type batch_size: int or None
        :return: Tuple of initial hidden and cell state tensors.
        :rtype: tuple[keras.KerasTensor, keras.KerasTensor]
        """
        return self.lstm_cell.get_initial_state(batch_size=batch_size)

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute the per-time-step output shape of the cell.

        :param input_shape: Shape of the per-step input ``(batch_size, input_dim)``.
        :type input_shape: tuple[int or None, ...]
        :return: Per-step output shape ``(batch_size, units)``.
        :rtype: tuple[int or None, ...]
        """
        return (input_shape[0], self.units)

    def get_config(self) -> Dict[str, Any]:
        """
        Return the layer configuration for serialization.

        :return: Dictionary containing the layer configuration.
        :rtype: dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'units': self.units,
            'dropout_rate': self.dropout_rate,
            'recurrent_dropout_rate': self.recurrent_dropout_rate,
        })
        return config

# ---------------------------------------------------------------------
