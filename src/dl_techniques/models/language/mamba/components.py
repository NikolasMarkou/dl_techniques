"""
Mamba selective state space layer and its pre-norm residual wrapper.

This file defines ``MambaLayer``, a selective state space layer, and
``MambaResidualBlock``, which wraps that layer in pre-norm residual form and
is the unit stacked to build a full Mamba model. The layer runs a causal 1D
convolution and then a state space scan whose discretization parameters
delta, B and C are projected from the input at every step, so the layer can
choose per token what to keep or forget; a standard state space model uses
fixed parameters and cannot do this. Because the parameters differ at every
step, the scan runs sequentially over the sequence length with
``keras.ops.while_loop`` rather than as a convolution, and it runs in the
layer's variable dtype instead of its compute dtype to avoid precision drift
over long accumulations. ``MambaResidualBlock`` returns the layer output and
the running residual as two separate tensors, so a stack has to thread the
second value into the next block.
"""

import math
import keras
import numpy as np
from typing import Optional, Union, Any, Dict, Tuple

# ---------------------------------------------------------------------

from dl_techniques.layers.norms.factory import create_normalization_layer
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.models.mamba.components")
class MambaLayer(keras.layers.Layer):
    """
    Run a causal convolution followed by an input-dependent state space scan.

    The discretization parameters delta, B and C come from a projection of
    the input at each step, so the state update below is data-dependent
    rather than fixed:

        h_t = A_bar * h_(t-1) + B_bar * x_t
        y_t = C * h_t

    where A_bar = exp(delta * A) and B_bar = delta * B.

    Architecture:

    .. code-block:: text

                hidden_states [B, L, d_model]
                              │
                              ▼
                      ┌───────────────┐
                      │    in_proj    │
                      └───────────────┘
                              │
                              ▼
                       split channels
                              ├─────────────────┐
                              ▼                 │
                              x                 z
                              │                 │
                        ┌───────────┐           │
                        │  conv1d   │           │
                        └───────────┘           │
                              │                 │
                              ▼                 │
                            silu                │
                              │                 │
               ┌──────────────┤                 │
               ▼              │                 │
        ┌─────────────┐       │                 │
        │ ssm params  │       │                 │
        └─────────────┘       │                 │
               │              │                 │
               ▼              │                 │
          delta, B, C         u                 │
               │              │                 │
               ▼              ▼                 ▼
        ┌─────────────────────────────────────────────┐
        │  selective scan and gate                    │
        └─────────────────────────────────────────────┘
                               │
                               ▼
                       y [B, d_inner, L]
                               │
                               ▼
                           transpose
                               │
                               ▼
                       ┌───────────────┐
                       │   out_proj    │
                       └───────────────┘
                               │
                               ▼
                    output [B, L, d_model]

    The gate ``z`` is applied inside the scan, not after it.

    SSM parameters:

    .. code-block:: text

                 x_conv [B, L, d_inner]
                            │
                            ▼
                     flatten tokens
                            │
                            ▼
                    ┌───────────────┐
                    │    x_proj     │
                    └───────────────┘
                            │
                            ▼
                   split dt_raw, B, C
                ┌───────────┴───────────┐
                ▼                       │
         ┌─────────────┐                │
         │   dt_proj   │                │
         └─────────────┘                │
                │                       │
                ▼                       ▼
            softplus                transpose
                │                       │
                ▼                       ▼
              delta                   B, C
         [B, d_inner, L]            [B, N, L]

    ``flatten tokens`` folds the sequence axis into the batch axis.

    Scan internals:

    .. code-block:: text

                 u, delta, B, C, D              z
                         │                      │
                         ▼                      │
              cast to variable dtype            │
                         │                      │
                         ▼                      │
        ┌─────────────────────────────────┐     │
        │  while_loop over L              │     │
        │  h = exp(delta A) h + delta B u │     │
        │  y_t = <C_t, h>                 │     │
        └─────────────────────────────────┘     │
                         │                      │
                         ▼                      │
                      + D * u                   │
                         │                      │
                         ▼                      │
               cast to compute dtype            │
                         │                      │
                         └──────────┬───────────┘
                                    ▼
                               y * silu(z)

    The variable dtype covers the scan only; ``z`` stays at compute dtype.

    :param d_model: Dimensionality of input and output embeddings.
    :type d_model: int
    :param d_state: Dimensionality of the SSM latent state (N).
        Controls state space capacity. Defaults to 16.
    :type d_state: int
    :param d_conv: Kernel size for causal 1D convolution. Larger values
        increase the local context window. Defaults to 4.
    :type d_conv: int
    :param expand: Expansion factor for internal dimension (d_inner = expand * d_model).
        Defaults to 2.
    :type expand: int
    :param dt_rank: Rank for the step size delta projection. 'auto' sets it to
        ceil(d_model/16). Controls expressiveness of temporal discretization.
        Defaults to "auto".
    :type dt_rank: Union[str, int]
    :param dt_min: Lower end of the step size range at initialization.
        Defaults to 0.001.
    :type dt_min: float
    :param dt_max: Upper end of the step size range at initialization.
        Defaults to 0.1.
    :type dt_max: float
    :param dt_init: Initialization strategy for the delta projection kernel
        ("random" or "constant"). Defaults to "random".
    :type dt_init: str
    :param dt_scale: Scaling factor for delta initialization. Defaults to 1.0.
    :type dt_scale: float
    :param dt_init_floor: Lower clip on the sampled step size before the
        inverse softplus. Defaults to 1e-4.
    :type dt_init_floor: float
    :param conv_bias: Whether to use bias in the convolution layer. Defaults to True.
    :type conv_bias: bool
    :param use_bias: Whether to use bias in linear projections. Defaults to False.
    :type use_bias: bool
    :param layer_idx: Optional layer index for caching in inference. Defaults to None.
    :type layer_idx: Optional[int]
    :param kwargs: Additional keyword arguments for Layer base class.
    :raises ValueError: If ``d_model``, ``d_state``, ``d_conv`` or ``expand``
        is not positive.

    Input shape:
        3D tensor with shape: `(batch_size, sequence_length, d_model)`.

    Output shape:
        3D tensor with shape: `(batch_size, sequence_length, d_model)`.

    :ivar A_log: Learnable log of state transition matrix A, shape (d_inner, d_state).
    :vartype A_log: keras.Variable
    :ivar D: Learnable skip connection parameter, shape (d_inner,).
    :vartype D: keras.Variable
    :ivar in_proj: Input projection layer mapping to 2*d_inner dimensions.
    :vartype in_proj: keras.layers.Dense
    :ivar conv1d: Causal 1D convolution with depthwise groups.
    :vartype conv1d: keras.layers.Conv1D
    :ivar x_proj: Projects activations to dt_rank + 2*d_state dimensions.
    :vartype x_proj: keras.layers.Dense
    :ivar dt_proj: Projects dt_rank to d_inner for step size computation.
    :vartype dt_proj: keras.layers.Dense
    :ivar out_proj: Final output projection back to d_model dimensions.
    :vartype out_proj: keras.layers.Dense

    Example:
        .. code-block:: python

            mamba = MambaLayer(d_model=768, d_state=16, d_conv=4, expand=2)
            x = keras.random.normal((2, 512, 768))
            y = mamba(x)

            mamba = MambaLayer(
                d_model=1024,
                d_state=32,
                d_conv=8,
                expand=3,
                dt_rank=64
            )

    Note:
        The scan uses `keras.ops.while_loop` because the SSM parameters are
        data-dependent, which rules out a convolutional implementation. The
        layer is therefore sequential over the sequence length.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Union[str, int] = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        use_bias: bool = False,
        layer_idx: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if d_model <= 0:
            raise ValueError(f"d_model must be positive, but got {d_model}")
        if d_state <= 0:
            raise ValueError(f"d_state must be positive, but got {d_state}")
        if d_conv <= 0:
            raise ValueError(f"d_conv must be positive, but got {d_conv}")
        if expand <= 0:
            raise ValueError(f"expand must be a positive integer, but got {expand}")

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init = dt_init
        self.dt_scale = dt_scale
        self.dt_init_floor = dt_init_floor
        self.conv_bias = conv_bias
        self.use_bias = use_bias
        self.layer_idx = layer_idx

        self.in_proj = keras.layers.Dense(
            self.d_inner * 2,
            use_bias=use_bias,
            name="in_proj"
        )

        self.conv1d = keras.layers.Conv1D(
            filters=self.d_inner,
            kernel_size=d_conv,
            # One filter per channel, so no mixing across channels.
            groups=self.d_inner,
            padding="causal",
            use_bias=conv_bias,
            name="conv1d",
        )

        self.activation = keras.layers.Activation("silu", name="silu")

        self.x_proj = keras.layers.Dense(
            self.dt_rank + self.d_state * 2,
            use_bias=False,
            name="x_proj"
        )

        # DECISION plan-2026-08-14T233721-d4f9beb2/D-084: dt_proj's init is passed
        # as initializers; an `.assign()` in build() is discarded. See decisions.md.
        self.dt_proj = keras.layers.Dense(
            self.d_inner,
            use_bias=True,
            kernel_initializer=self._dt_kernel_initializer(),
            bias_initializer=self._dt_bias_initializer(),
            name="dt_proj"
        )

        self.out_proj = keras.layers.Dense(
            self.d_model,
            use_bias=use_bias,
            name="out_proj"
        )

        # Created in `build`.
        self.A_log = None
        self.D = None

    def _dt_kernel_initializer(self):
        """Build the initializer for ``dt_proj.kernel``.

        The scale follows the paper's ``dt_init_std``. The draw itself happens
        inside the returned callable: a tensor computed here and closed over
        belongs to a scratch ``FuncGraph`` during the symbolic build pass and
        then fails on the eager pass with "cannot be accessed from here ... out
        of scope" (the D-021 trap).

        :return: A callable ``(shape, dtype) -> tensor``.
        """
        dt_init_std = self.dt_rank ** -0.5 * self.dt_scale
        if self.dt_init == "constant":
            return keras.initializers.Constant(dt_init_std)
        # A Keras initializer, not a bare `keras.random.uniform`: in the symbolic
        # build pass the global seed generator has no state variable to assign.
        return keras.initializers.RandomUniform(
            minval=-dt_init_std, maxval=dt_init_std
        )

    def _dt_bias_initializer(self):
        """Build the initializer for ``dt_proj.bias``.

        The bias is the inverse softplus of a log-uniform draw, so
        ``softplus(bias)``, which is the SSM timestep the scan uses, lands
        log-uniformly in ``[dt_min, dt_max]``, as in the Mamba paper.

        :return: A callable ``(shape, dtype) -> tensor``.
        """
        log_min = math.log(self.dt_min)
        log_max = math.log(self.dt_max)
        floor = self.dt_init_floor
        # A Keras initializer, not a bare `keras.random.uniform`: in the symbolic
        # build pass the global seed generator has no state variable to assign.
        uniform = keras.initializers.RandomUniform(minval=0.0, maxval=1.0)

        def initializer(shape, dtype=None):
            u = uniform(shape, dtype=dtype or "float32")
            dt = keras.ops.exp(u * (log_max - log_min) + log_min)
            dt = keras.ops.clip(dt, floor, float("inf"))
            # `expm1` stays accurate for the small dt values dt_min produces,
            # where `exp(dt) - 1` loses most of its significant digits.
            return keras.ops.log(keras.ops.expm1(dt))

        return initializer

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Create the SSM weights and build every sub-layer ``call`` runs.

        :param input_shape: Shape of input tensor (batch_size, seq_len, d_model).
        :type input_shape: Tuple[Optional[int], ...]
        """
        # A parent block may call this on an instance a forward pass already
        # built, which would re-run add_weight.
        if self.built:
            return

        # S4D real initialization: every row of A is [1, 2, ..., d_state]. NumPy
        # rather than keras.ops, which hits graph context issues inside `build`.
        A_init = np.tile(
            np.arange(1, self.d_state + 1, dtype="float32"), (self.d_inner, 1)
        )

        self.A_log = self.add_weight(
            name="A_log",
            shape=(self.d_inner, self.d_state),
            initializer=keras.initializers.Constant(np.log(A_init)),
            trainable=True,
        )

        # Per-channel scale on the skip connection around the scan.
        self.D = self.add_weight(
            name="D",
            shape=(self.d_inner,),
            initializer="ones",
            trainable=True,
        )

        # Building each sub-layer here keeps a standalone reload from restoring
        # weights into sub-layers that were never built.
        self.in_proj.build(input_shape)

        # The convolution sees the x path only, which is d_inner channels wide.
        conv_input_shape = (input_shape[0], input_shape[1], self.d_inner)
        self.conv1d.build(conv_input_shape)

        self.activation.build(conv_input_shape)

        # x_proj and dt_proj run on tokens folded into the batch axis.
        x_proj_input_shape = (None, self.d_inner)
        self.x_proj.build(x_proj_input_shape)

        dt_proj_input_shape = (None, self.dt_rank)
        self.dt_proj.build(dt_proj_input_shape)

        out_proj_input_shape = (input_shape[0], input_shape[1], self.d_inner)
        self.out_proj.build(out_proj_input_shape)

        # dt_proj's initialization comes from the initializers passed in
        # `__init__`, never from an `.assign()` here (D-084).

        super().build(input_shape)

    def _selective_scan(
        self,
        u: keras.KerasTensor,
        delta: keras.KerasTensor,
        A: keras.KerasTensor,
        B: keras.KerasTensor,
        C: keras.KerasTensor,
        D: keras.KerasTensor,
        z: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """
        Run the selective scan over the sequence, add the skip, and gate.

        The step size and the input and output maps vary per timestep, so the
        recurrence runs in a while loop rather than as a convolution.

        :param u: Input tensor after convolution, shape (batch, d_inner, seq_len).
        :type u: keras.KerasTensor
        :param delta: Step size delta, shape (batch, d_inner, seq_len).
        :type delta: keras.KerasTensor
        :param A: State transition matrix, shape (d_inner, d_state).
        :type A: keras.KerasTensor
        :param B: Input matrix, shape (batch, d_state, seq_len).
        :type B: keras.KerasTensor
        :param C: Output matrix, shape (batch, d_state, seq_len).
        :type C: keras.KerasTensor
        :param D: Skip connection parameter, shape (d_inner,).
        :type D: keras.KerasTensor
        :param z: Gating tensor, shape (batch, d_inner, seq_len).
        :type z: keras.KerasTensor
        :return: Output tensor, shape (batch, d_inner, seq_len).
        :rtype: keras.KerasTensor
        """
        batch_size, d_inner, seq_len = keras.ops.shape(u)

        # DECISION plan-2026-08-19T163559-499b6f0e/D-044: scan runs in variable
        # dtype, not compute dtype; half precision drifts. See decisions.md.
        scan_dtype = self.variable_dtype
        u = keras.ops.cast(u, scan_dtype)
        delta = keras.ops.cast(delta, scan_dtype)
        A = keras.ops.cast(A, scan_dtype)
        B = keras.ops.cast(B, scan_dtype)
        C = keras.ops.cast(C, scan_dtype)
        D = keras.ops.cast(D, scan_dtype)
        # z stays at compute dtype, since the gate applies after the result is
        # cast back down.

        # A_bar = exp(delta * A), one value per channel, state and step.
        deltaA = keras.ops.exp(
            keras.ops.einsum("bdl,dn->bdln", delta, A)
        )

        # B_bar * u = delta * B * u.
        deltaB_u = keras.ops.einsum(
            "bdl,bnl,bdl->bdln", delta, B, u
        )

        h = keras.ops.zeros(
            (batch_size, d_inner, self.d_state),
            dtype=scan_dtype
        )

        # ys is time-major so each step writes one contiguous slice.
        ys = keras.ops.zeros(
            (seq_len, batch_size, d_inner),
            dtype=scan_dtype
        )

        t = keras.ops.convert_to_tensor(0, dtype="int32")

        def condition(t: keras.KerasTensor, h: keras.KerasTensor,
                     ys: keras.KerasTensor) -> keras.KerasTensor:
            """Loop condition: continue while t < seq_len."""
            return keras.ops.less(t, seq_len)

        def body(t: keras.KerasTensor, h: keras.KerasTensor,
                ys: keras.KerasTensor) -> Tuple[keras.KerasTensor, ...]:
            """Advance the state one step and store ``y_t`` in ``ys``."""
            h = deltaA[:, :, t] * h + deltaB_u[:, :, t]

            y_t = keras.ops.einsum("bdn,bn->bd", h, C[:, :, t])

            # A single-slice scatter_update needs indices of shape (1, 1) and
            # updates with a matching leading axis.
            indices = keras.ops.reshape(t, (1, 1))
            updates = keras.ops.expand_dims(y_t, axis=0)
            ys = keras.ops.scatter_update(ys, indices, updates)

            return t + 1, h, ys

        _, _, final_ys = keras.ops.while_loop(
            cond=condition,
            body=body,
            loop_vars=(t, h, ys),
            maximum_iterations=seq_len
        )

        # Put batch back in front of the time axis.
        y = keras.ops.transpose(final_ys, (1, 2, 0))

        y = y + keras.ops.expand_dims(keras.ops.expand_dims(D, 0), -1) * u

        # The gate runs at compute dtype, so the scan result comes down first.
        y = keras.ops.cast(y, self.compute_dtype)
        return y * self.activation(z)

    def call(
        self,
        hidden_states: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Project the input, run the scan, and project back to ``d_model``.

        :param hidden_states: Input tensor, shape (batch, seq_len, d_model).
        :type hidden_states: keras.KerasTensor
        :param training: Whether in training mode. Defaults to None.
        :type training: Optional[bool]
        :return: Output tensor, shape (batch, seq_len, d_model).
        :rtype: keras.KerasTensor
        """
        batch_size, seq_len, _ = keras.ops.shape(hidden_states)

        # The scan works channels-first, so x, z, delta, B and C are transposed
        # to put the sequence axis last.
        xz = self.in_proj(hidden_states, training=training)
        xz = keras.ops.transpose(xz, (0, 2, 1))
        x, z = keras.ops.split(xz, 2, axis=1)

        x = keras.ops.transpose(x, (0, 2, 1))
        x_conv = self.conv1d(x, training=training)
        x_conv = self.activation(x_conv)

        # x_proj is a plain Dense, so the tokens fold into the batch axis.
        x_reshaped = keras.ops.reshape(x_conv, (-1, self.d_inner))
        x_proj_output = self.x_proj(x_reshaped, training=training)

        # `keras.ops.split` takes split indices, not sizes, so three tensors
        # need two split points.
        split_indices = [self.dt_rank, self.dt_rank + self.d_state]
        dt_raw, B_raw, C_raw = keras.ops.split(
            x_proj_output,
            split_indices,
            axis=-1
        )

        dt = self.dt_proj(dt_raw, training=training)
        dt = keras.ops.reshape(dt, (batch_size, seq_len, self.d_inner))
        dt = keras.ops.transpose(dt, (0, 2, 1))

        B = keras.ops.reshape(B_raw, (batch_size, seq_len, self.d_state))
        B = keras.ops.transpose(B, (0, 2, 1))

        C = keras.ops.reshape(C_raw, (batch_size, seq_len, self.d_state))
        C = keras.ops.transpose(C, (0, 2, 1))

        # A stays negative, so the recurrence decays instead of growing.
        A = -keras.ops.exp(keras.ops.cast(self.A_log, "float32"))

        # softplus keeps the step size positive.
        delta = keras.ops.softplus(dt)

        x_conv_transposed = keras.ops.transpose(x_conv, (0, 2, 1))

        y = self._selective_scan(
            u=x_conv_transposed,
            delta=delta,
            A=A,
            B=B,
            C=C,
            D=self.D,
            z=z
        )

        y = keras.ops.transpose(y, (0, 2, 1))
        output = self.out_proj(y, training=training)

        return output

    def compute_output_shape(self, input_shape):
        """Output shape: (batch, seq_len, d_model)."""
        return (*input_shape[:-1], self.d_model)

    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration for serialization.

        :return: Dictionary containing all constructor arguments.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "d_state": self.d_state,
            "d_conv": self.d_conv,
            "expand": self.expand,
            "dt_rank": self.dt_rank,
            "dt_min": self.dt_min,
            "dt_max": self.dt_max,
            "dt_init": self.dt_init,
            "dt_scale": self.dt_scale,
            "dt_init_floor": self.dt_init_floor,
            "conv_bias": self.conv_bias,
            "use_bias": self.use_bias,
            "layer_idx": self.layer_idx,
        })
        return config

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.models.mamba.components")
class MambaResidualBlock(keras.layers.Layer):
    """
    Wrap a MambaLayer in a pre-norm residual block.

    The block adds the incoming residual to the hidden states, normalizes the
    sum, and runs the Mamba layer on the normalized value. It returns the layer
    output and the unnormalized sum as two tensors, so the caller carries the
    residual into the next block instead of the block closing it. Normalizing
    before the sublayer rather than after improves training stability in deep
    networks.

    Architecture:

    .. code-block:: text

          hidden_states [B, L, D]       residual [B, L, D]
                     │                           │
                     └─────────────┬─────────────┘
                                   ▼
                             new_residual
                                   │
                     ┌─────────────┴─────────────┐
                     ▼                           │
              ┌─────────────┐                    │
              │    norm     │                    │
              └─────────────┘                    │
                     │                           │
                     ▼                           │
              ┌─────────────┐                    │
              │    mamba    │                    │
              └─────────────┘                    │
                     │                           │
                     ▼                           ▼
               mamba_output                new_residual
                 [B, L, D]                   [B, L, D]

    ``residual`` arrives as an input and is ``None`` for the first block.

    :param d_model: Dimensionality of the input and output.
    :type d_model: int
    :param norm_epsilon: Epsilon for layer normalization. Defaults to 1e-5.
    :type norm_epsilon: float
    :param mamba_kwargs: Keyword arguments to pass to MambaLayer constructor.
        Should include parameters like d_state, d_conv, expand, etc.
    :type mamba_kwargs: Optional[Dict[str, Any]]
    :param kwargs: Additional keyword arguments for Layer base class.

    Input shape:
        - hidden_states: 3D tensor (batch_size, seq_len, d_model)
        - residual: Optional 3D tensor (batch_size, seq_len, d_model) or None

    Output shape:
        Tuple of:
        - mamba_output: 3D tensor (batch_size, seq_len, d_model)
        - new_residual: 3D tensor (batch_size, seq_len, d_model)

    :ivar norm: Layer normalization applied before the Mamba layer.
    :vartype norm: keras.layers.Layer
    :ivar mamba: The core Mamba SSM layer.
    :vartype mamba: MambaLayer

    Example:
        .. code-block:: python

            block = MambaResidualBlock(
                d_model=768,
                mamba_kwargs={
                    "d_state": 16,
                    "d_conv": 4,
                    "expand": 2,
                    "layer_idx": 0
                }
            )

            x = keras.random.normal((2, 512, 768))
            hidden, residual = block(x, residual=None)
            hidden, residual = block(hidden, residual=residual)

    Note:
        The block never forms ``hidden_states + mamba_output``. It returns the
        two tensors, and the next block adds them.
    """

    def __init__(
        self,
        d_model: int,
        norm_epsilon: float = 1e-5,
        mamba_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.d_model = d_model
        self.norm_epsilon = norm_epsilon
        self.mamba_kwargs = mamba_kwargs or {}

        self.norm = create_normalization_layer(
            'layer_norm',
            epsilon=self.norm_epsilon,
            name="norm"
        )

        self.mamba = MambaLayer(
            d_model=self.d_model,
            **self.mamba_kwargs
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build both sub-layers ``call`` runs: the pre-norm and the Mamba layer.

        Both see the block's own input shape, because ``call`` normalizes the
        residual sum, which has the input shape, and feeds it to the layer.

        :param input_shape: Shape of input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        """
        self.norm.build(input_shape)
        self.mamba.build(input_shape)

        super().build(input_shape)

    def call(
        self,
        hidden_states: keras.KerasTensor,
        residual: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Add the residual, normalize, and run the Mamba layer.

        :param hidden_states: Main input tensor, shape (batch, seq_len, d_model).
        :type hidden_states: keras.KerasTensor
        :param residual: Optional residual from previous block. Defaults to None.
        :type residual: Optional[keras.KerasTensor]
        :param training: Whether in training mode. Defaults to None.
        :type training: Optional[bool]
        :return: Tuple of (mamba_output, new_residual).
        :rtype: Tuple[keras.KerasTensor, keras.KerasTensor]
        """
        # The sum is taken before normalization, so the residual stream stays
        # unnormalized down the stack.
        new_residual = (
            hidden_states + residual if residual is not None else hidden_states
        )

        normalized = self.norm(new_residual, training=training)

        mamba_output = self.mamba(normalized, training=training)

        return mamba_output, new_residual

    def compute_output_shape(self, input_shape):
        """Returns tuple of (hidden_states, residual), both (batch, seq_len, d_model)."""
        output_shape = (*input_shape[:-1], self.d_model)
        return (output_shape, output_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration for serialization.

        :return: Dictionary containing all constructor arguments.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "norm_epsilon": self.norm_epsilon,
            "mamba_kwargs": self.mamba_kwargs,
        })
        return config

# ---------------------------------------------------------------------
