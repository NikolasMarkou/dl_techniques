"""
Mamba-2 selective state space layer and its pre-norm residual block.

This file defines ``Mamba2Layer``, a selective state space layer, and
``Mamba2ResidualBlock``, which wraps that layer in pre-norm residual form and
is the unit a full model stacks. The layer computes its SSM parameters
(``dt``, ``B``, ``C``) in one projection of the input, rather than projecting
them from the convolution output as Mamba v1 does. That same projection also
carves out a gated MLP path that runs beside the SSM path, and the two are
concatenated before the output projection. ``B`` and ``C`` are computed per
group and broadcast to the ``nheads // ngroups`` heads each group serves, so
``nheads`` must be divisible by ``ngroups``. The recurrence runs as a
sequential ``keras.ops.while_loop`` in the layer's variable dtype, so a
half-precision compute dtype does not make it faster. ``Mamba2ResidualBlock``
returns two tensors, the layer output and the running residual, so a stack has
to thread the second value into the next block.
"""

import math
import keras
import numpy as np
from typing import Optional, Any, Dict, Tuple

# ---------------------------------------------------------------------

from dl_techniques.layers.norms.rms_norm import RMSNorm
from dl_techniques.utils.keras_registration import register_dl_technique


# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.models.mamba.components_v2")
class Mamba2Layer(keras.layers.Layer):
    """Apply a Mamba-2 selective state space layer to a sequence.

    One dense projection splits into a gated MLP path and an SSM path. The SSM
    path runs a depthwise causal convolution, then a per-head recurrence over
    the sequence:

        h_t = exp(delta_t * A) h_{t-1} + (delta_t * B_t) x_t
        y_t = <C_t, h_t> + D * x_t

    where ``delta_t = softplus(dt_t + dt_bias)`` and ``A`` is negative. The two
    paths are concatenated and sent through the output projection.

    Architecture:

    .. code-block:: text

            hidden_states [B, L, d_model]
                          │
                          ▼
                 ┌─────────────────┐
                 │     in_proj     │
                 └─────────────────┘
                          │
                          ▼
                   split last axis
              ┌───────────┴───────────┐
              │                       │
           z0, x0                z, xBC, dt
        [B, L, d_mlp]
              │                       │
              ▼                       ▼
        silu(z0) * x0          ┌─────────────┐
              │                │   conv1d    │
              │                └─────────────┘
              │                       │
              │                       ▼
              │                     silu
              │                       │
              │                       ▼
              │                 split x, B, C
              │                       │
              │                       ▼
              │                ┌─────────────┐
              │                │  ssm scan   │
              │                └─────────────┘
              │                       │ [B, L, H, P]
              │                       ▼
              │                    + D * x
              │                       │
              │                       ▼
              │             gate with z + rmsnorm
              │                       │ [B, L, d_ssm]
              └───────────┬───────────┘
                          ▼
             concatenate [B, L, d_inner]
                          │
                          ▼
                 ┌─────────────────┐
                 │    out_proj     │
                 └─────────────────┘
                          │
                          ▼
               output [B, L, d_model]

    The mlp path exists only when ``d_ssm`` is smaller than ``d_inner``.

    Scan internals:

    .. code-block:: text

        x [B, L, H, P]   dt [B, L, H]   B, C [B, L, G, N]
               │               │                │
               │               ▼                ▼
               │         softplus(dt +   repeat b, c to
               │           dt_bias)          h heads
               │               │                │
               ▼               ▼                ▼
        ┌─────────────────────────────────────────────────┐
        │  while_loop over L, in the variable dtype       │
        │  h = exp(delta * A) h + (delta * B) x           │
        │  y = <C, h> over the state axis                 │
        └─────────────────────────────────────────────────┘
                                 │
                                 ▼
                          y [B, L, H, P]

    The scan output is cast back to the compute dtype.

    Gate and norm order:

    .. code-block:: text

             False (default)                    True

            y_ssm      silu(z)            y_ssm      silu(z)
              │           │                 │           │
              └─────┬─────┘                 ▼           │
                    ▼                    rmsnorm        │
                 rmsnorm                    │           │
                    │                       └─────┬─────┘
                    ▼                             ▼
               ssm_output                    ssm_output

    With ``rmsnorm=False`` both branches reduce to ``y_ssm * silu(z)``.

    :param d_model: Dimensionality of input and output.
    :param d_state: Dimensionality of the SSM latent state (N).
    :param d_conv: Kernel size for the causal 1D convolution.
    :param expand: Expansion factor for the internal dimension.
    :param headdim: Dimensionality of each SSM head.
    :param ngroups: Number of groups for the ``B`` and ``C`` projections. Each
        group is broadcast to the ``nheads // ngroups`` consecutive heads it
        serves, so ``nheads`` must be divisible by ``ngroups``.
    :param d_ssm: If not None, applies the SSM on this many dims, with the rest
        using a gated MLP. Defaults to the full inner dimension.
    :param rmsnorm: Whether to apply RMS normalization to the SSM output.
    :param norm_epsilon: Epsilon value for normalization layers.
    :param norm_before_gate: Where the gated RMSNorm sits relative to the
        multiplicative gate. ``False``, the default and what the reference
        Mamba-2 implementation uses, computes ``norm(y * silu(z))``. ``True``
        computes ``norm(y) * silu(z)``. Both branches use the same ``norm``
        weights, so the choice changes numerics but not the ``.keras`` layout.
    :param dt_min: Minimum value for the step size delta.
    :param dt_max: Maximum value for the step size delta.
    :param dt_init_floor: Lower clip on the sampled step size before the
        inverse softplus that produces ``dt_bias``.
    :param bias: Whether to use bias in linear projections.
    :param conv_bias: Whether to use bias in the convolution layer.
    :raises ValueError: If ``d_ssm`` is not divisible by ``headdim``, if
        ``ngroups`` is not positive, or if ``nheads`` is not divisible by
        ``ngroups``.

    Input shape:
        ``(batch, seq_len, d_model)``.

    Output shape:
        ``(batch, seq_len, d_model)``.
    """

    def __init__(
            self,
            d_model: int,
            d_state: int = 128,
            d_conv: int = 4,
            expand: int = 2,
            headdim: int = 64,
            ngroups: int = 1,
            d_ssm: Optional[int] = None,
            rmsnorm: bool = True,
            norm_epsilon: float = 1e-5,
            norm_before_gate: bool = False,
            dt_min: float = 0.001,
            dt_max: float = 0.1,
            dt_init_floor: float = 1e-4,
            bias: bool = False,
            conv_bias: bool = True,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        self.headdim = headdim
        self.ngroups = ngroups
        self.d_ssm = self.d_inner if d_ssm is None else d_ssm
        self.rmsnorm = rmsnorm
        self.norm_epsilon = norm_epsilon
        self.norm_before_gate = norm_before_gate
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init_floor = dt_init_floor
        self.bias = bias
        self.conv_bias = conv_bias

        if self.d_ssm % self.headdim != 0:
            raise ValueError(f"d_ssm ({self.d_ssm}) must be divisible by headdim ({self.headdim})")
        self.nheads = self.d_ssm // self.headdim
        if self.ngroups <= 0:
            raise ValueError(f"ngroups must be positive, got {self.ngroups}")
        # Required by the head to group routing in `_ssm_scan` (D-042): each
        # group serves exactly `nheads // ngroups` consecutive heads.
        if self.nheads % self.ngroups != 0:
            raise ValueError(
                f"nheads ({self.nheads} = d_ssm {self.d_ssm} // headdim "
                f"{self.headdim}) must be divisible by ngroups ({self.ngroups})"
            )

        # in_proj emits z0, x0 (mlp path), z (ssm gate), xBC (conv input) and
        # one dt per head, concatenated in that order.
        d_in_proj = (
                2 * (self.d_inner - self.d_ssm)
                + self.d_ssm
                + (self.d_ssm + 2 * self.ngroups * self.d_state)
                + self.nheads
        )
        self.in_proj = keras.layers.Dense(d_in_proj, use_bias=bias, name="in_proj")

        conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
        self.conv1d = keras.layers.Conv1D(
            filters=conv_dim,
            kernel_size=d_conv,
            # One filter per channel, so no mixing across channels.
            groups=conv_dim,
            padding="causal",
            use_bias=conv_bias,
            name="conv1d",
        )

        self.activation = keras.layers.Activation("silu", name="silu")

        if self.rmsnorm:
            # DECISION plan-2026-08-19T163559-499b6f0e/D-123: use RMSNorm, not
            # LayerNormalization(rms_scaling=True). See decisions.md.
            self.norm = RMSNorm(epsilon=self.norm_epsilon, name="rmsnorm")

        self.out_proj = keras.layers.Dense(self.d_model, use_bias=bias, name="out_proj")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the SSM weights and build every sub-layer ``call`` runs.

        :param input_shape: Shape of the input, ``(batch, seq_len, d_model)``.
        """
        # DECISION plan-2026-08-19T163559-499b6f0e/D-123: return early if built;
        # Mamba2ResidualBlock.build calls this unconditionally. See decisions.md.
        if self.built:
            return

        # np.random, not keras.random: a seeded build must give bit-identical
        # A_log and dt_bias, with no graph-unsafe tensor-to-numpy hop.
        A_init = np.log(np.random.uniform(1, 16, size=self.nheads))
        self.A_log = self.add_weight(
            name="A_log",
            shape=(self.nheads,),
            initializer=keras.initializers.Constant(A_init),
            trainable=True,
        )

        # Per-head scale on the skip connection around the scan.
        self.D = self.add_weight(
            name="D",
            shape=(self.nheads,),
            initializer="ones",
            trainable=True,
        )

        # Inverse softplus, so softplus(dt_bias) starts in [dt_min, dt_max].
        dt_init = np.exp(
            np.random.rand(self.nheads) * (math.log(self.dt_max) - math.log(self.dt_min))
            + math.log(self.dt_min)
        )
        dt_init = np.clip(dt_init, self.dt_init_floor, None)
        inv_dt = dt_init + np.log(-np.expm1(-dt_init))
        self.dt_bias = self.add_weight(
            name="dt_bias",
            shape=(self.nheads,),
            initializer=keras.initializers.Constant(inv_dt),
            trainable=True,
        )

        # DECISION plan-2026-08-14T233721-d4f9beb2/D-042: build every sub-layer
        # call() runs, or a standalone reload finds them unbuilt. See decisions.md.
        conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
        conv_input_shape = (input_shape[0], input_shape[1], conv_dim)

        self.in_proj.build(input_shape)
        self.conv1d.build(conv_input_shape)
        self.activation.build(conv_input_shape)
        if self.rmsnorm:
            self.norm.build((input_shape[0], input_shape[1], self.d_ssm))
        # `call` concatenates the d_mlp-wide MLP path onto the d_ssm-wide SSM
        # path, and d_mlp = d_inner - d_ssm, so out_proj always sees d_inner.
        self.out_proj.build((input_shape[0], input_shape[1], self.d_inner))

        super().build(input_shape)

    def _ssm_scan(
            self,
            x: keras.KerasTensor,
            dt: keras.KerasTensor,
            A: keras.KerasTensor,
            B: keras.KerasTensor,
            C: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Run the selective scan recurrence over the sequence.

        :param x: SSM input of shape ``(batch, seq_len, nheads, headdim)``.
        :param dt: Step size logits of shape ``(batch, seq_len, nheads)``.
        :param A: State decay of shape ``(nheads,)``, negative.
        :param B: Input map of shape ``(batch, seq_len, ngroups, d_state)``.
        :param C: Output map of shape ``(batch, seq_len, ngroups, d_state)``.
        :return: Tensor of shape ``(batch, seq_len, nheads, headdim)`` in the
            compute dtype.
        """
        batch_size, seq_len, nheads, headdim = keras.ops.shape(x)

        # DECISION plan-2026-08-19T163559-499b6f0e/D-044: scan runs in variable
        # dtype, not compute dtype; half precision drifts. See decisions.md.
        scan_dtype = self.variable_dtype
        x = keras.ops.cast(x, scan_dtype)
        dt = keras.ops.cast(dt, scan_dtype)
        A = keras.ops.cast(A, scan_dtype)
        B = keras.ops.cast(B, scan_dtype)
        C = keras.ops.cast(C, scan_dtype)

        # DECISION plan-2026-08-18T140459-7991552f/D-042: broadcast B and C to the
        # heads each group serves, never sum over groups. See decisions.md.
        heads_per_group = self.nheads // self.ngroups
        if self.ngroups != self.nheads:
            B = keras.ops.repeat(B, heads_per_group, axis=2)
            C = keras.ops.repeat(C, heads_per_group, axis=2)

        delta = keras.ops.softplus(
            dt + keras.ops.cast(self.dt_bias, scan_dtype)
        )
        # A_bar = exp(delta * A), one value per batch, step and head.
        deltaA = keras.ops.exp(keras.ops.einsum("blh,h->blh", delta, A))

        h = keras.ops.zeros((batch_size, nheads, headdim, self.d_state), dtype=scan_dtype)
        ys = keras.ops.zeros((seq_len, batch_size, nheads, headdim), dtype=scan_dtype)

        t = keras.ops.convert_to_tensor(0, dtype="int32")

        def cond(t, h, ys):
            return keras.ops.less(t, seq_len)

        def body(t, h, ys):
            # deltaA[:, t] is (B, H) and h is (B, H, P, N), so this scales the
            # whole state of each head.
            h_A_part = keras.ops.einsum("bh,bhpn->bhpn", deltaA[:, t], h)

            # B is already per head, so nothing contracts here: (B, H, N) times
            # (B, H, P) gives the new (B, H, P, N) state contribution.
            B_bar_t = keras.ops.einsum("bh,bhn->bhn", delta[:, t], B[:, t])
            h_B_part = keras.ops.einsum("bhn,bhp->bhpn", B_bar_t, x[:, t])

            h = h_A_part + h_B_part

            # C contracts the state axis only, leaving the head axis intact.
            y_t = keras.ops.einsum('bhpn,bhn->bhp', h, C[:, t])

            updates = keras.ops.expand_dims(y_t, axis=0)
            ys = keras.ops.scatter_update(ys, keras.ops.reshape(t, (1, 1)), updates)

            return t + 1, h, ys

        _, _, final_ys = keras.ops.while_loop(
            cond=cond,
            body=body,
            loop_vars=(t, h, ys),
            maximum_iterations=seq_len
        )
        # ys is time-major inside the loop; put batch back in front.
        return keras.ops.cast(
            keras.ops.transpose(final_ys, (1, 0, 2, 3)), self.compute_dtype
        )

    def call(
            self,
            hidden_states: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Project the input, run the scan, and combine both paths.

        :param hidden_states: Input of shape ``(batch, seq_len, d_model)``.
        :param training: Unused, kept for the Keras call signature.
        :return: Tensor of shape ``(batch, seq_len, d_model)``.
        """
        batch, seqlen, _ = keras.ops.shape(hidden_states)

        zxbcdt = self.in_proj(hidden_states)
        d_mlp = self.d_inner - self.d_ssm
        conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state

        split_indices = [
            d_mlp, 2 * d_mlp, 2 * d_mlp + self.d_ssm, 2 * d_mlp + self.d_ssm + conv_dim
        ]
        z0, x0, z, xBC, dt = keras.ops.split(zxbcdt, split_indices, axis=-1)

        xBC_conv = self.conv1d(xBC)
        xBC_act = self.activation(xBC_conv)

        # x, B and C share one convolution, so they are split after it.
        split_indices_after_conv = [
            self.d_ssm, self.d_ssm + self.ngroups * self.d_state
        ]
        x, B_conv, C_conv = keras.ops.split(xBC_act, split_indices_after_conv, axis=-1)

        # x is laid out per head; B and C are per group.
        x = keras.ops.reshape(x, (batch, seqlen, self.nheads, self.headdim))
        B = keras.ops.reshape(B_conv, (batch, seqlen, self.ngroups, self.d_state))
        C = keras.ops.reshape(C_conv, (batch, seqlen, self.ngroups, self.d_state))

        # A stays negative, so the recurrence decays instead of growing.
        A = -keras.ops.exp(keras.ops.cast(self.A_log, "float32"))

        y_ssm = self._ssm_scan(x, dt, A, B, C)

        y_ssm = y_ssm + keras.ops.einsum("blhp,h->blhp", x, self.D)
        y_ssm = keras.ops.reshape(y_ssm, (batch, seqlen, self.d_ssm))

        if self.rmsnorm:
            if self.norm_before_gate:
                y_norm = self.norm(y_ssm)
                ssm_output = y_norm * self.activation(z)
            else:
                ssm_output = self.norm(y_ssm * self.activation(z))
        else:
            ssm_output = y_ssm * self.activation(z)

        # The MLP path is empty when d_ssm covers the whole inner dimension.
        if d_mlp > 0:
            mlp_output = self.activation(z0) * x0
            combined_output = keras.ops.concatenate([mlp_output, ssm_output], axis=-1)
        else:
            combined_output = ssm_output

        return self.out_proj(combined_output)

    def compute_output_shape(self, input_shape):
        """Output shape: (batch, seq_len, d_model)."""
        return (*input_shape[:-1], self.d_model)

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor arguments needed to rebuild the layer.

        :return: Serializable config dictionary.
        """
        config = super().get_config()
        config.update({
            "d_model": self.d_model, "d_state": self.d_state,
            "d_conv": self.d_conv, "expand": self.expand,
            "headdim": self.headdim, "ngroups": self.ngroups,
            "d_ssm": self.d_ssm, "rmsnorm": self.rmsnorm,
            "norm_epsilon": self.norm_epsilon,
            "norm_before_gate": self.norm_before_gate,
            "dt_min": self.dt_min, "dt_max": self.dt_max,
            "dt_init_floor": self.dt_init_floor, "bias": self.bias,
            "conv_bias": self.conv_bias,
        })
        return config


@register_dl_technique("dl_techniques.models.mamba.components_v2")
class Mamba2ResidualBlock(keras.layers.Layer):
    """Wrap a :class:`Mamba2Layer` in a pre-norm residual block.

    The block adds the incoming residual to the hidden states, normalizes the
    sum, and runs the layer on the normalized value. It returns the layer
    output and the unnormalized sum, so the caller carries the residual to the
    next block instead of the block closing it.

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
              │   mamba2    │                    │
              └─────────────┘                    │
                     │                           │
                     ▼                           ▼
               mamba_output                new_residual
                 [B, L, D]                   [B, L, D]

    ``residual`` arrives as an input and is ``None`` for the first block.

    :param d_model: Dimensionality of input and output.
    :param d_state: Dimensionality of the SSM latent state.
    :param d_conv: Kernel size for the causal 1D convolution.
    :param expand: Expansion factor for the internal dimension.
    :param headdim: Dimensionality of each SSM head.
    :param d_ssm: Number of dims the SSM runs on; the rest use a gated MLP.
    :param norm_epsilon: Epsilon value for the pre-norm and the layer norms.
    :param rmsnorm: If True, the pre-norm is an :class:`RMSNorm`; if False, it
        is a ``LayerNormalization``. Also forwarded to the wrapped layer.
    :param norm_before_gate: Forwarded to the wrapped :class:`Mamba2Layer`; see
        its docstring for the semantics.
    :param ngroups: Forwarded to the wrapped :class:`Mamba2Layer`.
    :param dt_min: Forwarded to the wrapped :class:`Mamba2Layer`.
    :param dt_max: Forwarded to the wrapped :class:`Mamba2Layer`.
    :param dt_init_floor: Forwarded to the wrapped :class:`Mamba2Layer`.
    :param bias: Forwarded to the wrapped :class:`Mamba2Layer`.
    :param conv_bias: Forwarded to the wrapped :class:`Mamba2Layer`.

    Input shape:
        ``(batch, seq_len, d_model)``.

    Output shape:
        Two tensors, each ``(batch, seq_len, d_model)``.

    Note:
        Every default here matches the corresponding :class:`Mamba2Layer`
        default.
    """

    def __init__(
            self,
            d_model: int,
            d_state: int,
            d_conv: int,
            expand: int,
            headdim: int,
            d_ssm: int,
            norm_epsilon: float = 1e-5,
            rmsnorm: bool = True,
            norm_before_gate: bool = False,
            # DECISION plan-2026-08-18T140459-7991552f/D-036: these six are pure
            # pass-throughs; keep their defaults equal to Mamba2Layer's. See decisions.md.
            ngroups: int = 1,
            dt_min: float = 0.001,
            dt_max: float = 0.1,
            dt_init_floor: float = 1e-4,
            bias: bool = False,
            conv_bias: bool = True,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.headdim = headdim
        self.d_ssm = d_ssm
        self.norm_epsilon = norm_epsilon
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate
        self.ngroups = ngroups
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init_floor = dt_init_floor
        self.bias = bias
        self.conv_bias = conv_bias

        if rmsnorm:
            # DECISION plan-2026-08-19T163559-499b6f0e/D-123 (see `Mamba2Layer`)
            self.norm = RMSNorm(epsilon=self.norm_epsilon, name="norm")
        else:
            self.norm = keras.layers.LayerNormalization(
                epsilon=self.norm_epsilon, name="norm"
            )
        self.mamba2 = Mamba2Layer(
            d_model=self.d_model,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
            headdim=self.headdim,
            d_ssm=self.d_ssm,
            rmsnorm=self.rmsnorm,
            norm_epsilon=self.norm_epsilon,
            norm_before_gate=self.norm_before_gate,
            ngroups=self.ngroups,
            dt_min=self.dt_min,
            dt_max=self.dt_max,
            dt_init_floor=self.dt_init_floor,
            bias=self.bias,
            conv_bias=self.conv_bias,
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build both sub-layers ``call`` runs: the pre-norm and the layer.

        Both sub-layers see the block's own input shape, because ``call`` runs
        the pre-norm on the residual sum, which has the input shape, and feeds
        that straight into the wrapped layer.

        :param input_shape: Shape of the input, ``(batch, seq_len, d_model)``.
        """
        self.norm.build(input_shape)
        self.mamba2.build(input_shape)
        super().build(input_shape)

    def call(
            self,
            hidden_states: keras.KerasTensor,
            residual: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None,
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Add the residual, normalize, and run the wrapped layer.

        :param hidden_states: Input of shape ``(batch, seq_len, d_model)``.
        :param residual: Running residual from the previous block, or None.
        :param training: Unused, kept for the Keras call signature.
        :return: Tuple of the layer output and the new residual.
        """
        new_residual = hidden_states + residual if residual is not None else hidden_states
        normalized = self.norm(new_residual)
        mamba_output = self.mamba2(normalized)
        return mamba_output, new_residual

    def compute_output_shape(self, input_shape):
        """Returns tuple of (hidden_states, residual), both (batch, seq_len, d_model)."""
        output_shape = (*input_shape[:-1], self.d_model)
        return (output_shape, output_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor arguments needed to rebuild the block.

        :return: Serializable config dictionary.
        """
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "d_state": self.d_state,
            "d_conv": self.d_conv,
            "expand": self.expand,
            "headdim": self.headdim,
            "d_ssm": self.d_ssm,
            "norm_epsilon": self.norm_epsilon,
            "rmsnorm": self.rmsnorm,
            "norm_before_gate": self.norm_before_gate,
            "ngroups": self.ngroups,
            "dt_min": self.dt_min,
            "dt_max": self.dt_max,
            "dt_init_floor": self.dt_init_floor,
            "bias": self.bias,
            "conv_bias": self.conv_bias,
        })
        return config

# ---------------------------------------------------------------------
