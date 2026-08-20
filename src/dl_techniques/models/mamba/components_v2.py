import math
import keras
import numpy as np
from typing import Optional, Any, Dict, Tuple


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class Mamba2Layer(keras.layers.Layer):
    """
    Core Mamba v2 selective state space model layer.

    This layer implements the Mamba v2 architecture, which refines Mamba v1
    by computing SSM parameters (Δ, B, C) in parallel from the input.
    This differs from v1, where they were computed sequentially after a
    convolution. The layer also integrates an optional gated MLP path and
    RMS normalization for improved performance and stability.

    **Architecture**:

    .. code-block:: text

        Input (x)
           │
           ▼
        Linear Projection → [z0, x0, z, xBC, dt]
           │
           ├───────────────┬──────────────────┐
           ▼               ▼                  ▼
        (z0, x0)      (z, xBC, dt)          Gate(z)
        (MLP Path)      (SSM Path)             │
           │               │                   │
           ▼               ▼                   ▼
        silu(z0)*x0  Conv1D(xBC) → [x, B, C]   │
                         │                     │
                         ▼                     │
                     Selective SSM → y_ssm     │
                                     │         │
                                     └─ Gating ┤
                                         │     │
                                         ▼     │
                                      RMSNorm  │
                                         │     │
                                         ▼     ▼
                            [MLP output, SSM output]
                                         │
                                         ▼
                                     Out Proj

    :param d_model: Dimensionality of input and output.
    :param d_state: Dimensionality of the SSM latent state (N).
    :param d_conv: Kernel size for causal 1D convolution.
    :param expand: Expansion factor for internal dimension.
    :param headdim: Dimensionality of each SSM head.
    :param ngroups: Number of groups for the ``B``/``C`` projections. Each
        group is BROADCAST to the ``nheads // ngroups`` consecutive heads it
        serves (GQA semantics), so ``nheads`` must be divisible by ``ngroups``.
        Until 2026-08-19 the scan summed over the group axis instead, which
        made any ``ngroups > 1`` mathematically identical to ``ngroups = 1``
        with a rescaled ``B``/``C`` (D-042).
    :param d_ssm: If not None, applies SSM on this many dims, with the
        rest using a gated MLP. Defaults to the full inner dimension.
    :param rmsnorm: Whether to apply RMS normalization to the SSM output.
    :param norm_epsilon: Epsilon value for normalization layers.
    :param norm_before_gate: Where the gated RMSNorm sits relative to the
        multiplicative gate. ``False`` (the default, and what the reference
        Mamba-2 implementation does) computes ``norm(y * silu(z))`` -- the gate
        is applied first and the product is normalized, which is the whole
        point of a *gated* norm. ``True`` computes ``norm(y) * silu(z)``.
        This defaulted to ``True`` and described that as "as in the official
        implementation" until 2026-08-15; the reference defaults it to
        ``False``. Both branches use the same ``norm`` weights, so the flip
        changes numerics but not the ``.keras`` layout -- a checkpoint trained
        under the old default must set ``norm_before_gate=True`` explicitly.
    :param dt_min: Minimum value for the step size Δ.
    :param dt_max: Maximum value for the step size Δ.
    :param bias: Whether to use bias in linear projections.
    :param conv_bias: Whether to use bias in the convolution layer.
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

        # Store config
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

        # Validation
        if self.d_ssm % self.headdim != 0:
            raise ValueError(f"d_ssm ({self.d_ssm}) must be divisible by headdim ({self.headdim})")
        self.nheads = self.d_ssm // self.headdim
        if self.ngroups <= 0:
            raise ValueError(f"ngroups must be positive, got {self.ngroups}")
        # Required by the head->group routing in `_ssm_scan` (D-042): each group
        # serves exactly `nheads // ngroups` consecutive heads.
        if self.nheads % self.ngroups != 0:
            raise ValueError(
                f"nheads ({self.nheads} = d_ssm {self.d_ssm} // headdim "
                f"{self.headdim}) must be divisible by ngroups ({self.ngroups})"
            )

        # Input projection for z, x, B, C, dt
        d_in_proj = (
                2 * (self.d_inner - self.d_ssm)  # z0, x0 for MLP
                + self.d_ssm  # z for SSM gating
                + (self.d_ssm + 2 * self.ngroups * self.d_state)  # xBC for conv
                + self.nheads  # dt
        )
        self.in_proj = keras.layers.Dense(d_in_proj, use_bias=bias, name="in_proj")

        # Convolutional layer
        conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
        self.conv1d = keras.layers.Conv1D(
            filters=conv_dim,
            kernel_size=d_conv,
            groups=conv_dim,  # Depthwise
            padding="causal",
            use_bias=conv_bias,
            name="conv1d",
        )

        self.activation = keras.layers.Activation("silu", name="silu")

        # RMS Normalization (or LayerNorm as a fallback)
        if self.rmsnorm:
            self.norm = keras.layers.LayerNormalization(
                epsilon=self.norm_epsilon,
                rms_scaling=True,  # Use RMSNorm
                name="rmsnorm"
            )

        # Output projection
        self.out_proj = keras.layers.Dense(self.d_model, use_bias=bias, name="out_proj")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        # A_log initialization (per head).
        # `np.random`, deliberately and MEASURED: `keras.utils.set_random_seed`
        # calls `np.random.seed(seed)`, so two seeded builds of this layer
        # produce bit-identical `A_log` and `dt_bias` (max diff exactly 0.0).
        # Drawing with `keras.random` instead would mean converting a backend
        # tensor to numpy inside `build()` to feed `Constant`, which is not
        # graph-safe. Pinned by
        # `tests/test_models/test_mamba/test_build_and_reload.py`.
        A_init = np.log(np.random.uniform(1, 16, size=self.nheads))
        self.A_log = self.add_weight(
            name="A_log",
            shape=(self.nheads,),
            initializer=keras.initializers.Constant(A_init),
            trainable=True,
        )

        # D skip connection (per head)
        self.D = self.add_weight(
            name="D",
            shape=(self.nheads,),
            initializer="ones",
            trainable=True,
        )

        # dt_bias initialization (per head)
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

        # DECISION plan-2026-08-14T233721-d4f9beb2/D-042
        # Build EXACTLY the sub-layers `call()` runs. This class overrides
        # `build`, so `build_from_config` takes the `self.build(input_shape)`
        # branch rather than the build-by-run branch; without these lines a
        # standalone `.keras` reload -- or embedding in any parent that also
        # overrides `build` -- restores weights into sub-layers that do not
        # exist. `Mamba2` itself masked this by having no `build` override.
        # `self.norm` is built only when `rmsnorm` is set, because `call` only
        # runs it then: building an unused sub-layer creates weights the lazy
        # path never created and silently changes the `.keras` layout.
        # Sibling that gets this right: `components.py::MambaLayer.build`.
        # See decisions.md D-042 and plans/SYSTEM.md.
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
        """Performs the selective scan recurrence."""
        batch_size, seq_len, nheads, headdim = keras.ops.shape(x)

        # DECISION plan-2026-08-19T163559-499b6f0e/D-044
        # The recurrent scan runs in the VARIABLE dtype (float32 under
        # `mixed_float16`), never in the compute dtype. `call()` already casts
        # `A_log` to float32 on purpose, and `keras.ops.einsum` PROMOTES rather
        # than raising, so `deltaA` came out float32 while `h` was materialised at
        # `compute_dtype` — the exception then landed on `h_A_part + h_B_part`
        # inside `body`, away from its cause. Do NOT "fix" this by casting `A`
        # down to float16: this is a sequential accumulation over `seq_len` steps
        # and half-precision state is exactly where it drifts.
        # See decisions.md D-044.
        scan_dtype = self.variable_dtype
        x = keras.ops.cast(x, scan_dtype)
        dt = keras.ops.cast(dt, scan_dtype)
        A = keras.ops.cast(A, scan_dtype)
        B = keras.ops.cast(B, scan_dtype)
        C = keras.ops.cast(C, scan_dtype)

        # DECISION plan-2026-08-18T140459-7991552f/D-042: BROADCAST B/C from
        # their group to the heads that group serves -- do NOT sum over `g`.
        # This body used to do `einsum("bh,bgn->bhgn")` then
        # `einsum("bhgn,bhp->bhpn")`, and `einsum('bhpn,bgn->bhpg')` then
        # `sum(axis=3)`, i.e. it CONTRACTED the group axis, so every head
        # received `sum_g B_g` and `sum_g C_g`. MEASURED on CPU at
        # `nheads=4, ngroups=4`: the scan was bit-invariant to PERMUTING the
        # group axis (max|delta| 9.5e-07 on |y|max 10.2) and reproduced a
        # `ngroups=1` layer fed the summed `B`/`C` to 1.3e-06 -- i.e. `ngroups`
        # allocated `2*ngroups*d_state` projection channels and then collapsed
        # them additively, exactly `ngroups=1` with a rescaled B/C. GQA (which
        # `mamba_v2.py`'s module docstring invokes) BROADCASTS WITHIN a group;
        # it does not sum across groups.
        # Head->group order matches the reference's
        # `repeat(B, "b l g n -> b l (g h) n")`: heads are contiguous blocks,
        # head `i` reads group `i // (nheads // ngroups)`.
        heads_per_group = self.nheads // self.ngroups
        if self.ngroups != self.nheads:
            B = keras.ops.repeat(B, heads_per_group, axis=2)  # (B, L, H, N)
            C = keras.ops.repeat(C, heads_per_group, axis=2)  # (B, L, H, N)

        # Discretize A: A_bar = exp(Δ * A)
        delta = keras.ops.softplus(
            dt + keras.ops.cast(self.dt_bias, scan_dtype)
        )  # (B, L, H)
        deltaA = keras.ops.exp(keras.ops.einsum("blh,h->blh", delta, A))  # (B, L, H)

        # Recurrent scan
        h = keras.ops.zeros((batch_size, nheads, headdim, self.d_state), dtype=scan_dtype)
        ys = keras.ops.zeros((seq_len, batch_size, nheads, headdim), dtype=scan_dtype)

        t = keras.ops.convert_to_tensor(0, dtype="int32")

        def cond(t, h, ys):
            return keras.ops.less(t, seq_len)

        def body(t, h, ys):
            # State update: h_t = A_bar_t * h_{t-1} + B_bar_t * x_t

            # A part: A_bar_t * h_{t-1}
            # deltaA[:, t] has shape (B, H). h has shape (B, H, P, N).
            h_A_part = keras.ops.einsum("bh,bhpn->bhpn", deltaA[:, t], h)

            # B part: (delta_t * B_t) * x_t, PER HEAD (no contraction).
            # delta[:, t] is (B, H)
            # B[:, t] is (B, H, N)  -- already broadcast from (B, G, N)
            # x[:, t] is (B, H, P)
            B_bar_t = keras.ops.einsum("bh,bhn->bhn", delta[:, t], B[:, t])
            h_B_part = keras.ops.einsum("bhn,bhp->bhpn", B_bar_t, x[:, t])

            h = h_A_part + h_B_part

            # Output: y_t = <C_t, h_t> contracted over the STATE axis only.
            # h is (B, H, P, N). C[:, t] is (B, H, N).
            y_t = keras.ops.einsum('bhpn,bhn->bhp', h, C[:, t])

            # Store output
            updates = keras.ops.expand_dims(y_t, axis=0)
            ys = keras.ops.scatter_update(ys, keras.ops.reshape(t, (1, 1)), updates)

            return t + 1, h, ys

        _, _, final_ys = keras.ops.while_loop(
            cond=cond,
            body=body,
            loop_vars=(t, h, ys),
            maximum_iterations=seq_len
        )
        return keras.ops.cast(
            keras.ops.transpose(final_ys, (1, 0, 2, 3)), self.compute_dtype
        )  # (B, L, H, P)

    def call(
            self,
            hidden_states: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        batch, seqlen, _ = keras.ops.shape(hidden_states)

        # 1. Input Projection
        zxbcdt = self.in_proj(hidden_states)
        d_mlp = self.d_inner - self.d_ssm
        conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state

        # 2. Split into MLP and SSM paths
        split_indices = [
            d_mlp, 2 * d_mlp, 2 * d_mlp + self.d_ssm, 2 * d_mlp + self.d_ssm + conv_dim
        ]
        z0, x0, z, xBC, dt = keras.ops.split(zxbcdt, split_indices, axis=-1)

        # 3. SSM Path
        # Causal convolution
        xBC_conv = self.conv1d(xBC)
        xBC_act = self.activation(xBC_conv)

        # Split into x, B, C after convolution
        split_indices_after_conv = [
            self.d_ssm, self.d_ssm + self.ngroups * self.d_state
        ]
        x, B_conv, C_conv = keras.ops.split(xBC_act, split_indices_after_conv, axis=-1)

        # Reshape for scan
        x = keras.ops.reshape(x, (batch, seqlen, self.nheads, self.headdim))
        B = keras.ops.reshape(B_conv, (batch, seqlen, self.ngroups, self.d_state))
        C = keras.ops.reshape(C_conv, (batch, seqlen, self.ngroups, self.d_state))

        # Get state matrix A
        A = -keras.ops.exp(keras.ops.cast(self.A_log, "float32"))

        # Selective scan
        y_ssm = self._ssm_scan(x, dt, A, B, C)

        # Add skip connection D
        y_ssm = y_ssm + keras.ops.einsum("blhp,h->blhp", x, self.D)
        y_ssm = keras.ops.reshape(y_ssm, (batch, seqlen, self.d_ssm))

        # Optional RMSNorm and Gating
        if self.rmsnorm:
            if self.norm_before_gate:
                y_norm = self.norm(y_ssm)
                ssm_output = y_norm * self.activation(z)
            else:
                ssm_output = self.norm(y_ssm * self.activation(z))
        else:
            ssm_output = y_ssm * self.activation(z)

        # 4. MLP Path
        if d_mlp > 0:
            mlp_output = self.activation(z0) * x0
            # 5. Combine and project
            combined_output = keras.ops.concatenate([mlp_output, ssm_output], axis=-1)
        else:
            combined_output = ssm_output

        return self.out_proj(combined_output)

    def compute_output_shape(self, input_shape):
        """Output shape: (batch, seq_len, d_model)."""
        return (*input_shape[:-1], self.d_model)

    def get_config(self) -> Dict[str, Any]:
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


@keras.saving.register_keras_serializable()
class Mamba2ResidualBlock(keras.layers.Layer):
    """Residual block wrapping a Mamba2Layer with pre-normalization.

    :param norm_before_gate: Forwarded verbatim to the wrapped
        :class:`Mamba2Layer`; see its docstring for the semantics. It is exposed
        here because that docstring names ``norm_before_gate=True`` as the
        remedy for a checkpoint trained under the pre-2026-08-15 default, and
        this block did not accept, forward or serialize it — which made the
        documented escape hatch unreachable from every real model, since
        :class:`~dl_techniques.models.mamba.mamba_v2.Mamba2Model` builds its
        whole stack out of these blocks.
    :param ngroups: Forwarded verbatim to the wrapped :class:`Mamba2Layer`.
    :param dt_min: Forwarded verbatim to the wrapped :class:`Mamba2Layer`.
    :param dt_max: Forwarded verbatim to the wrapped :class:`Mamba2Layer`.
    :param dt_init_floor: Forwarded verbatim to the wrapped
        :class:`Mamba2Layer`.
    :param bias: Forwarded verbatim to the wrapped :class:`Mamba2Layer`.
    :param conv_bias: Forwarded verbatim to the wrapped :class:`Mamba2Layer`.

    .. note::
       Those six joined ``norm_before_gate`` on 2026-08-19 for exactly the
       reason its own ``:param:`` gives. :class:`Mamba2Layer` declares, stores
       and USES all seven, but this block declared only the last one, so from
       any assembled Mamba-2 model ``ngroups`` was pinned at 1 (which is what
       made multi-group state unreachable), ``conv_bias`` could not be turned
       off, and the Δ initialisation range could not be widened. Every default
       here equals the corresponding :class:`Mamba2Layer` default, so the
       default construction path is unchanged. See decisions.md
       plan-2026-08-18T140459-7991552f/D-036.
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
            # DECISION plan-2026-08-18T140459-7991552f/D-036
            # These six are pure pass-throughs and their defaults MUST stay
            # equal to `Mamba2Layer`'s, which is where the semantics live. Do
            # NOT drop them again: until 2026-08-19 this block declared none of
            # them, and since `Mamba2Model` builds its entire stack out of these
            # blocks, six documented `Mamba2Layer` knobs were unreachable from
            # every assembled model -- `ngroups` in particular, which is why the
            # multi-group state path had no caller. Do not give them
            # block-local defaults either: a divergence here silently rebuilds
            # a different `in_proj`/`conv1d` width than the docstring one level
            # down promises. See decisions.md.
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
            self.norm = keras.layers.LayerNormalization(
                epsilon=self.norm_epsilon, rms_scaling=True, name="norm"
            )
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
        """Build both sub-layers `call()` runs: the pre-norm and the Mamba2Layer.

        This class had no `build` at all, which is safe only while every parent
        leaves it on Keras' build-by-run path. `Mamba2Layer` now overrides
        `build` (D-042), so an explicit one here keeps the two consistent and
        makes a standalone reload of this block work. Both sub-layers see the
        block's own input shape: `call` runs `self.norm` on the residual sum
        (same shape as the input) and feeds its output straight to `self.mamba2`.
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
        new_residual = hidden_states + residual if residual is not None else hidden_states
        normalized = self.norm(new_residual)
        mamba_output = self.mamba2(normalized)
        return mamba_output, new_residual

    def compute_output_shape(self, input_shape):
        """Returns tuple of (hidden_states, residual), both (batch, seq_len, d_model)."""
        output_shape = (*input_shape[:-1], self.d_model)
        return (output_shape, output_shape)

    def get_config(self) -> Dict[str, Any]:
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