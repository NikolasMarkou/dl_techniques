"""Selective state space mixer (S6) with input-dependent discretization.

A layer-level port of the Mamba v1 selective scan
(Gu and Dao, 2023) for sequence data of any modality: the discretization
step ``delta`` and the ``B``/``C`` maps are projected from the input at
every timestep, so the recurrence ``h_t = A_bar h_{t-1} + B_bar x_t``,
``y_t = C h_t`` can keep or forget per token. The scan runs sequentially
under ``keras.ops.while_loop`` in the variable dtype.

This is the generic primitive backing :class:`ContextMambaLayer` and any
future vision/language temporal model. The token-sequence foundation model
itself stays in ``models/language/mamba/``.

References:
    - Gu and Dao, 2023. Mamba: Linear-Time Sequence Modeling with Selective
      State Spaces. (https://arxiv.org/abs/2312.00752)
    - Gu et al., 2021. Efficiently Modeling Long Sequences with Structured
      State Spaces. (https://arxiv.org/abs/2111.00396)
"""

import math
import keras
import numpy as np
from typing import Optional, Union, Any, Dict, Tuple

from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.ssm.selective_ssm")
class SelectiveSSMLayer(keras.layers.Layer):
    """Run a causal convolution followed by an input-dependent SSM scan.

    :param d_model: Dimensionality of input and output embeddings. Must be positive.
    :type d_model: int
    :param d_state: Dimensionality of the SSM latent state. Defaults to 16.
    :type d_state: int
    :param d_conv: Kernel size for the causal depthwise 1D convolution. Defaults to 4.
    :type d_conv: int
    :param expand: Expansion factor for the internal dimension
        (``d_inner = expand * d_model``). Defaults to 2.
    :type expand: int
    :param dt_rank: Rank of the step-size projection; ``"auto"`` resolves to
        ``ceil(d_model / 16)``. Defaults to ``"auto"``.
    :type dt_rank: Union[str, int]
    :param dt_min: Lower end of the step-size range at initialization. Defaults to 0.001.
    :type dt_min: float
    :param dt_max: Upper end of the step-size range at initialization. Defaults to 0.1.
    :type dt_max: float
    :param dt_init: Initialization strategy for the delta projection kernel
        (``"random"`` or ``"constant"``). Defaults to ``"random"``.
    :type dt_init: str
    :param dt_scale: Scaling factor for delta initialization. Defaults to 1.0.
    :type dt_scale: float
    :param dt_init_floor: Lower clip on the sampled step size before the inverse
        softplus. Defaults to 1e-4.
    :type dt_init_floor: float
    :param conv_bias: Whether to use bias in the convolution layer. Defaults to True.
    :type conv_bias: bool
    :param use_bias: Whether to use bias in linear projections. Defaults to False.
    :type use_bias: bool
    :param layer_idx: Optional layer index for inference caching. Defaults to None.
    :type layer_idx: Optional[int]

    Input shape:
        3D tensor ``(batch_size, sequence_length, d_model)``.

    Output shape:
        3D tensor ``(batch_size, sequence_length, d_model)``.
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
            raise ValueError(f"d_model must be positive, got {d_model}")
        if d_state <= 0:
            raise ValueError(f"d_state must be positive, got {d_state}")
        if d_conv <= 0:
            raise ValueError(f"d_conv must be positive, got {d_conv}")
        if expand <= 0:
            raise ValueError(f"expand must be positive, got {expand}")
        if isinstance(dt_rank, int) and dt_rank <= 0:
            raise ValueError(f"dt_rank must be positive, got {dt_rank}")
        if dt_min <= 0 or dt_max <= 0 or dt_max <= dt_min:
            raise ValueError(f"need 0 < dt_min ({dt_min}) < dt_max ({dt_max})")
        if dt_init not in ("random", "constant"):
            raise ValueError(f"dt_init must be 'random' or 'constant', got {dt_init!r}")

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        # The raw argument is what get_config serializes ("auto" stays
        # "auto"); the resolved rank drives the layer. Same split as the
        # dt_min/dt_max-style pure-function-of-config resolutions elsewhere.
        self.dt_rank_arg = dt_rank
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
            self.d_inner * 2, use_bias=use_bias, name="in_proj"
        )
        self.conv1d = keras.layers.Conv1D(
            filters=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding="causal",
            use_bias=conv_bias,
            name="conv1d",
        )
        self.activation = keras.layers.Activation("silu", name="silu")
        self.x_proj = keras.layers.Dense(
            self.dt_rank + self.d_state * 2, use_bias=False, name="x_proj"
        )
        self.dt_proj = keras.layers.Dense(
            self.d_inner,
            use_bias=True,
            kernel_initializer=self._dt_kernel_initializer(),
            bias_initializer=self._dt_bias_initializer(),
            name="dt_proj",
        )
        self.out_proj = keras.layers.Dense(
            self.d_model, use_bias=use_bias, name="out_proj"
        )

        self.A_log = None
        self.D = None

    def _dt_kernel_initializer(self):  # type: ignore[no-untyped-def]
        """Build the initializer for ``dt_proj.kernel``.

        :return: A callable ``(shape, dtype) -> tensor``.
        """
        dt_init_std = self.dt_rank ** -0.5 * self.dt_scale
        if self.dt_init == "constant":
            return keras.initializers.Constant(dt_init_std)
        return keras.initializers.RandomUniform(
            minval=-dt_init_std, maxval=dt_init_std
        )

    def _dt_bias_initializer(self):  # type: ignore[no-untyped-def]
        """Build the initializer for ``dt_proj.bias`` (inverse softplus).

        :return: A callable ``(shape, dtype) -> tensor``.
        """
        log_min = math.log(self.dt_min)
        log_max = math.log(self.dt_max)
        floor = self.dt_init_floor
        uniform = keras.initializers.RandomUniform(minval=0.0, maxval=1.0)

        def initializer(shape, dtype=None):  # type: ignore[no-untyped-def]
            u = uniform(shape, dtype=dtype or "float32")
            dt = keras.ops.exp(u * (log_max - log_min) + log_min)
            dt = keras.ops.clip(dt, floor, float("inf"))
            return keras.ops.log(keras.ops.expm1(dt))

        return initializer

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create SSM weights and build every sub-layer ``call`` runs.

        :param input_shape: Shape ``(batch, seq_len, d_model)``.
        :type input_shape: Tuple[Optional[int], ...]
        """
        if self.built:
            return

        A_init = np.tile(
            np.arange(1, self.d_state + 1, dtype="float32"), (self.d_inner, 1)
        )
        self.A_log = self.add_weight(
            name="A_log",
            shape=(self.d_inner, self.d_state),
            initializer=keras.initializers.Constant(np.log(A_init)),
            trainable=True,
        )
        self.D = self.add_weight(
            name="D",
            shape=(self.d_inner,),
            initializer="ones",
            trainable=True,
        )

        self.in_proj.build(input_shape)
        conv_input_shape = (input_shape[0], input_shape[1], self.d_inner)
        self.conv1d.build(conv_input_shape)
        self.activation.build(conv_input_shape)
        self.x_proj.build((None, self.d_inner))
        self.dt_proj.build((None, self.dt_rank))
        self.out_proj.build((input_shape[0], input_shape[1], self.d_inner))

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
        """Run the selective scan, add the skip, and gate.

        :param u: Convolved input ``(batch, d_inner, seq_len)``.
        :param delta: Step sizes ``(batch, d_inner, seq_len)``.
        :param A: Negative state decay ``(d_inner, d_state)``.
        :param B: Input map ``(batch, d_state, seq_len)``.
        :param C: Output map ``(batch, d_state, seq_len)``.
        :param D: Skip parameter ``(d_inner,)``.
        :param z: Gate ``(batch, d_inner, seq_len)`` at compute dtype.
        :return: Output ``(batch, d_inner, seq_len)``.
        """
        batch_size, d_inner, seq_len = keras.ops.shape(u)

        scan_dtype = self.variable_dtype
        u = keras.ops.cast(u, scan_dtype)
        delta = keras.ops.cast(delta, scan_dtype)
        A = keras.ops.cast(A, scan_dtype)
        B = keras.ops.cast(B, scan_dtype)
        C = keras.ops.cast(C, scan_dtype)
        D = keras.ops.cast(D, scan_dtype)

        h = keras.ops.zeros((batch_size, d_inner, self.d_state), dtype=scan_dtype)
        ys = keras.ops.zeros((seq_len, batch_size, d_inner), dtype=scan_dtype)
        t = keras.ops.convert_to_tensor(0, dtype="int32")

        def condition(t: keras.KerasTensor, h: keras.KerasTensor,
                      ys: keras.KerasTensor) -> keras.KerasTensor:
            return keras.ops.less(t, seq_len)

        def body(t: keras.KerasTensor, h: keras.KerasTensor,
                 ys: keras.KerasTensor) -> Tuple[keras.KerasTensor, ...]:
            delta_t = delta[:, :, t]
            u_t = u[:, :, t]
            B_t = B[:, :, t]
            deltaA_t = keras.ops.exp(keras.ops.einsum("bd,dn->bdn", delta_t, A))
            deltaB_u_t = keras.ops.einsum("bd,bn,bd->bdn", delta_t, B_t, u_t)
            h = deltaA_t * h + deltaB_u_t
            y_t = keras.ops.einsum("bdn,bn->bd", h, C[:, :, t])
            indices = keras.ops.reshape(t, (1, 1))
            updates = keras.ops.expand_dims(y_t, axis=0)
            ys = keras.ops.scatter_update(ys, indices, updates)
            return t + 1, h, ys

        _, _, final_ys = keras.ops.while_loop(
            cond=condition,
            body=body,
            loop_vars=(t, h, ys),
            maximum_iterations=seq_len,
        )

        y = keras.ops.transpose(final_ys, (1, 2, 0))
        y = y + keras.ops.expand_dims(keras.ops.expand_dims(D, 0), -1) * u
        y = keras.ops.cast(y, self.compute_dtype)
        return y * self.activation(z)

    def call(
        self,
        hidden_states: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Project the input, run the scan, and project back.

        :param hidden_states: Input ``(batch, seq_len, d_model)``.
        :param training: Training mode flag, forwarded to sub-layers.
        :return: Output ``(batch, seq_len, d_model)``.
        """
        batch_size, seq_len, _ = keras.ops.shape(hidden_states)

        xz = self.in_proj(hidden_states, training=training)
        xz = keras.ops.transpose(xz, (0, 2, 1))
        x, z = keras.ops.split(xz, 2, axis=1)

        x = keras.ops.transpose(x, (0, 2, 1))
        x_conv = self.conv1d(x, training=training)
        x_conv = self.activation(x_conv)

        x_reshaped = keras.ops.reshape(x_conv, (-1, self.d_inner))
        x_proj_output = self.x_proj(x_reshaped, training=training)

        split_indices = [self.dt_rank, self.dt_rank + self.d_state]
        dt_raw, B_raw, C_raw = keras.ops.split(
            x_proj_output, split_indices, axis=-1
        )

        dt = self.dt_proj(dt_raw, training=training)
        dt = keras.ops.reshape(dt, (batch_size, seq_len, self.d_inner))
        dt = keras.ops.transpose(dt, (0, 2, 1))

        B = keras.ops.reshape(B_raw, (batch_size, seq_len, self.d_state))
        B = keras.ops.transpose(B, (0, 2, 1))
        C = keras.ops.reshape(C_raw, (batch_size, seq_len, self.d_state))
        C = keras.ops.transpose(C, (0, 2, 1))

        A = -keras.ops.exp(keras.ops.cast(self.A_log, "float32"))
        delta = keras.ops.softplus(dt)
        x_conv_transposed = keras.ops.transpose(x_conv, (0, 2, 1))

        if keras.backend.backend() == "tensorflow":
            import tensorflow as tf
            scan_fn = tf.recompute_grad(self._selective_scan)
            scan_D = tf.convert_to_tensor(self.D)
        else:
            scan_fn = self._selective_scan
            scan_D = self.D
        y = scan_fn(x_conv_transposed, delta, A, B, C, scan_D, z)

        y = keras.ops.transpose(y, (0, 2, 1))
        return self.out_proj(y, training=training)

    def compute_output_shape(self, input_shape):  # type: ignore[no-untyped-def]
        """Compute output shape from stored config.

        :param input_shape: Input shape ``(batch, seq_len, d_model)``.
        :return: Output shape ``(batch, seq_len, d_model)``.
        """
        return (*input_shape[:-1], self.d_model)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        :return: Dictionary containing all constructor arguments.
        """
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "d_state": self.d_state,
            "d_conv": self.d_conv,
            "expand": self.expand,
            "dt_rank": self.dt_rank_arg,
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
