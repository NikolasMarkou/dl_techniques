"""
N-BEATSx exogenous block.

Extends N-BEATS (see :mod:`nbeats_blocks`) so a block can use exogenous
covariates, following Olivares et al. (2023,
https://arxiv.org/abs/2104.05522).

A standard N-BEATS block expands theta over a fixed basis: polynomials for
trend, a Fourier series for seasonality, a learned matrix for generic. This
block builds the basis from data instead. The exogenous history and future
are concatenated along time and run through a TCN encoder, and the result
is the basis.

The target residual still goes through the inherited 4-layer dense stack,
which produces one theta vector per sample. Backcast and forecast are the
basis weighted by theta:

    backcast = einsum('btc,bc->bt', basis_backcast, theta_backcast)
    forecast = einsum('btc,bc->bt', basis_forecast, theta_forecast)

With ``use_tcn=True`` (NBEATSx-G) the basis is ``C = TCN(X)``. With
``use_tcn=False`` (NBEATSx-I) the raw exogenous tensor is the basis.
"""

import keras
from keras import ops, layers

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .nbeats_blocks import NBeatsBlock
from .temporal_convolutional_network import TemporalConvNet
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.time_series.nbeatsx_blocks")
class ExogenousBlock(NBeatsBlock):
    """
    N-BEATS block whose basis comes from exogenous covariates.

    A standard N-BEATS block has a fixed or learned basis. This one builds
    the basis from the exogenous inputs at run time, then weights it with
    theta coefficients read off the target residual. Backcast and forecast
    are both single-channel.

    The dense spine is inherited from :class:`NBeatsBlock` and is drawn
    there. This block re-runs those four layers itself, because it needs
    the exogenous basis before it can project theta. The theta heads are
    replaced: they are bias-free Dense layers whose width is the number of
    basis channels, not ``thetas_dim``.

    Two configurations follow the paper. NBEATSx-G (``use_tcn=True``)
    encodes the covariates with a TCN. NBEATSx-I (``use_tcn=False``) uses
    the raw covariates, which keeps each basis channel readable as one
    named input variable.

    Here B is the batch size, E is ``exogenous_dim``, B_len is
    ``backcast_length``, F_len is ``forecast_length``, T = B_len + F_len
    and C is the number of basis channels.

    **Architecture Overview:**

    .. code-block:: text

        Residual y                Exogenous covariates
        [B, B_len*in_dim]         x_back [B, B_len, E]
                │                 x_fore [B, F_len, E]
                ▼                          │
        ┌───────────────┐                  ▼
        │ Dense1..Dense4│         ┌──────────────────┐
        │ + RMSNorm opt │         │ concat on time   │
        │ (NBeatsBlock  │         │ [B, T, E]        │
        │  spine)       │         │ T = B_len+F_len  │
        └───────┬───────┘         └────────┬─────────┘
                │ [B, units]               ▼
                │                 ┌──────────────────┐
                │                 │ TCN encoder      │
                │                 │ (use_tcn only)   │
                │                 └────────┬─────────┘
                │                          │ [B, T, C]
        ┌───────┴───────┐         ┌────────┴─────────┐
        ▼               ▼         ▼                  ▼
     theta_b         theta_f   basis_b            basis_f
     [B, C]          [B, C]  [B,B_len,C]        [B,F_len,C]
        └───────┬───────┘         └────────┬─────────┘
                └────────────┬─────────────┘
                             ▼
            einsum('btc,bc->bt', basis, theta)
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
     backcast [B, B_len]           forecast [B, F_len]

    C is ``tcn_filters`` when ``use_tcn`` is True and ``exogenous_dim``
    when it is False. The einsum sums over C, so each output step is one
    number. See :class:`NBeatsBlock` for the dense stack and theta heads
    this diagram points at.

    :param exogenous_dim: Number of exogenous features per time step.
    :type exogenous_dim: int
    :param tcn_filters: Number of filters in the TCN encoder. This is also
        the basis channel count when ``use_tcn`` is True. Defaults to 16.
    :type tcn_filters: int
    :param tcn_kernel_size: Kernel size for the TCN convolutions.
        Defaults to 2.
    :type tcn_kernel_size: int
    :param tcn_dropout_rate: Dropout rate inside the TCN encoder.
        Defaults to 0.0.
    :type tcn_dropout_rate: float
    :param use_tcn: If True, encode the covariates with a TCN (NBEATSx-G).
        If False, use them directly as the basis (NBEATSx-I).
        Defaults to True.
    :type use_tcn: bool
    :param kwargs: Arguments passed to :class:`NBeatsBlock`, including
        ``units``, ``thetas_dim``, ``backcast_length`` and
        ``forecast_length``.

    Input shape:
        Residual tensor (batch, backcast_length * input_dim), plus an
        ``exogenous_inputs`` tuple passed to ``call``:
        (batch, backcast_length, exogenous_dim) and
        (batch, forecast_length, exogenous_dim).

    Output shape:
        Tuple of (batch, backcast_length) and (batch, forecast_length).

    Example:
        .. code-block:: python

            block = ExogenousBlock(
                exogenous_dim=6, tcn_filters=16, use_tcn=True,
                units=32, thetas_dim=8,
                backcast_length=10, forecast_length=4,
            )
            back, fore = block(
                residual, exogenous_inputs=(x_back, x_fore)
            )
            # back.shape == (batch, 10), fore.shape == (batch, 4)

    Note:
        ``exogenous_inputs`` is a keyword argument of ``call``, so this
        block cannot be swapped into a plain N-BEATS stack that calls
        ``block(x)``. Calling it without covariates raises ValueError.

    Note:
        The dense stack is copied from the parent rather than delegated to,
        because the exogenous basis must be built before theta is projected.
        It applies the same RMSNorm AND the same four Dropout layers as the
        parent, in the same order, so ``dropout_rate`` behaves identically
        here. One thing still differs, measured and not assumed: the final
        reshape multiplies the length by a literal 1, so the outputs stay
        single-channel even when ``input_dim`` or ``output_dim`` is larger
        than 1.
    """

    def __init__(
            self,
            exogenous_dim: int,
            tcn_filters: int = 16,
            tcn_kernel_size: int = 2,
            tcn_dropout_rate: float = 0.0,
            use_tcn: bool = True,
            **kwargs
    ):
        # In NBEATSx, theta dimension corresponds to the channels of the basis/exog
        # Ensure kwargs['thetas_dim'] matches or is set correctly if not provided
        super().__init__(**kwargs)

        self.exogenous_dim = exogenous_dim
        self.use_tcn = use_tcn
        self.tcn_filters = tcn_filters
        self.tcn_kernel_size = tcn_kernel_size
        self.tcn_dropout_rate = tcn_dropout_rate

        # The encoder for exogenous variables
        if self.use_tcn:
            self.encoder = TemporalConvNet(
                filters=tcn_filters,
                kernel_size=tcn_kernel_size,
                dropout_rate=tcn_dropout_rate
            )
        else:
            # For Interpretable (NBEATSx-I), the basis is just X itself.
            # No complex encoder needed, but we might need projection if dims don't match
            self.encoder = None

        # Redefine Theta layers to match TCN/Exog dimensionality
        # Theta must project to the dimension of the Basis channels
        basis_channels = tcn_filters if use_tcn else exogenous_dim

        self.theta_backcast = layers.Dense(
            basis_channels,
            use_bias=False,
            name='theta_backcast_exog'
        )
        self.theta_forecast = layers.Dense(
            basis_channels,
            use_bias=False,
            name='theta_forecast_exog'
        )

    def build(self, input_shape):
        """
        Build the TCN encoder, then the inherited dense spine.

        The encoder sees the concatenated covariates, so it is built with
        length ``backcast_length + forecast_length``. The parent build runs
        last, because it marks the layer as built.

        :param input_shape: Shape of the residual input y, given as
            (batch, backcast_length * input_dim).
        :type input_shape: tuple
        """
        # Build encoder
        # Input to encoder is (Batch, Time, Exog_Dim)
        total_len = self.backcast_length + self.forecast_length
        if self.use_tcn:
            self.encoder.build((None, total_len, self.exogenous_dim))

        # Build the Dense stack from the parent (input_shape refers to the
        # residual input y). MUST be last so the layer is marked built only
        # after all sub-layers exist.
        super().build(input_shape)

    def call(self, inputs, training=None, exogenous_inputs=None):
        """
        Run the block over one residual and its covariates.

        The residual goes through the dense stack and becomes two theta
        vectors. The covariates are concatenated along time, encoded, and
        split back into a backcast basis and a forecast basis. Each output
        is its basis weighted by its theta.

        :param inputs: Residual target signal of shape
            (batch, backcast_len * input_dim).
        :type inputs: keras.KerasTensor
        :param training: Whether in training mode.
        :type training: bool or None
        :param exogenous_inputs: Tuple of (x_backcast, x_forecast) where
            x_backcast has shape (batch, backcast_len, exog_dim) and
            x_forecast has shape (batch, forecast_len, exog_dim).
        :type exogenous_inputs: tuple
        :return: Tuple of (backcast, forecast) tensors.
        :rtype: tuple[keras.KerasTensor, keras.KerasTensor]

        :raises ValueError: If exogenous_inputs is None.
        """
        if exogenous_inputs is None:
            raise ValueError("ExogenousBlock requires 'exogenous_inputs' passed to call.")

        x_back, x_fore = exogenous_inputs

        # 1. Process Target Residual through Dense Stack (Parent Logic)
        # -----------------------------------------------------------
        # Note: We duplicate the FC stack logic here because the parent .call()
        # calls _generate_backcast immediately, but we need the exogenous basis first.

        x = self.dense1(inputs, training=training)
        if self.use_normalization: x = self.norm1(x)
        if self.dropout_rate > 0:
            x = self.dropout1(x, training=training)
        x = self.dense2(x, training=training)
        if self.use_normalization: x = self.norm2(x)
        if self.dropout_rate > 0:
            x = self.dropout2(x, training=training)
        x = self.dense3(x, training=training)
        if self.use_normalization: x = self.norm3(x)
        if self.dropout_rate > 0:
            x = self.dropout3(x, training=training)
        x = self.dense4(x, training=training)
        if self.use_normalization: x = self.norm4(x)
        if self.dropout_rate > 0:
            x = self.dropout4(x, training=training)

        # 2. Generate Theta Coefficients (Weights for the Basis)
        # -----------------------------------------------------------
        # Shape: (Batch, Basis_Channels)
        # Note: Unlike standard N-BEATS, these are global weights per sample,
        # not per time-step. They scale the TCN basis vectors.
        theta_b = self.theta_backcast(x, training=training)
        theta_f = self.theta_forecast(x, training=training)

        # 3. Generate Basis from Exogenous Variables
        # -----------------------------------------------------------
        # Concatenate history and future exogenous: (Batch, Total_Len, Exog_Dim)
        x_full = ops.concatenate([x_back, x_fore], axis=1)

        if self.use_tcn:
            # Basis C = TCN(X)
            # Shape: (Batch, Total_Len, TCN_Filters)
            basis_full = self.encoder(x_full, training=training)
        else:
            # Basis = X (Interpretable configuration)
            basis_full = x_full

        # Split basis back into backcast and forecast periods
        basis_b = basis_full[:, :self.backcast_length, :]
        basis_f = basis_full[:, self.backcast_length:, :]

        # 4. Projection (Eq 9)
        # -----------------------------------------------------------
        # Backcast = Basis_b * theta_b
        # basis: (B, T, C), theta: (B, C) -> result: (B, T) (summed over C)
        backcast = ops.einsum('btc,bc->bt', basis_b, theta_b)
        forecast = ops.einsum('btc,bc->bt', basis_f, theta_f)

        # Reshape to flatten (Batch, Time) -> (Batch, Time * 1) if input_dim=1
        backcast = ops.reshape(backcast, (-1, self.backcast_length * 1))
        forecast = ops.reshape(forecast, (-1, self.forecast_length * 1))

        return backcast, forecast

    def _generate_backcast(self, theta):
        """
        Not used. The basis projection happens inline in ``call``.

        The parent declares this abstract, so it is overridden here to
        satisfy that contract. It returns None.

        :param theta: Unused theta coefficients.
        :type theta: keras.KerasTensor
        :return: None.
        :rtype: None
        """
        pass

    def _generate_forecast(self, theta):
        """
        Not used. The basis projection happens inline in ``call``.

        The parent declares this abstract, so it is overridden here to
        satisfy that contract. It returns None.

        :param theta: Unused theta coefficients.
        :type theta: keras.KerasTensor
        :return: None.
        :rtype: None
        """
        pass

    def get_config(self):
        """
        Return the constructor arguments needed to rebuild this block.

        The parent's configuration is included, so ``units``,
        ``thetas_dim`` and the two lengths round-trip as well.

        :return: Serializable configuration dictionary.
        :rtype: dict
        """
        config = super().get_config()
        config.update({
            'exogenous_dim': self.exogenous_dim,
            'tcn_filters': self.tcn_filters,
            'tcn_kernel_size': self.tcn_kernel_size,
            'tcn_dropout_rate': self.tcn_dropout_rate,
            'use_tcn': self.use_tcn
        })
        return config

# ---------------------------------------------------------------------
