import math
import keras
import numpy as np
from typing import Optional, Union, Any, Dict, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.norms.factory import create_normalization_layer

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class MambaLayer(keras.layers.Layer):
    """
    Core Mamba selective state space model layer.

    This layer implements the selective SSM computation as the fundamental
    building block of the Mamba architecture. It combines a causal 1D
    convolution with a recurrent state space model where the discretization
    parameters (Δ, B, C) are computed from the input data, enabling the
    model to selectively propagate or forget information.

    **Intent**:
    Provide efficient sequence modeling with linear complexity by using a
    selective state space mechanism that adapts its behavior based on input
    content, unlike traditional SSMs with fixed parameters.

    **Architecture**:

    .. code-block:: text

        Input (x)
           │
           ▼
        Linear Projection → [x_proj, z]  (split into two paths)
           │
           ├─────────────────┐
           ▼                 ▼
        x_proj            z (gate)
           │                 │
           ▼                 │
        Causal Conv1D        │
           │                 │
           ▼                 │
        SiLU                 │
           │                 │
           ▼                 │
        Compute Δ, B, C      │
           │                 │
           ▼                 │
        Selective SSM ───────┤
           │                 │
           ▼                 ▼
        Output ──────── Gating (y * SiLU(z))
           │
           ▼
        Linear Projection
           │
           ▼
        Output

    **Mathematical Foundation**:

    The core SSM equations:
        h_t = A̅ * h_{t-1} + B̅ * x_t    (state update)
        y_t = C * h_t                   (output projection)

    Where:
        A̅ = exp(Δ * A)                  (discretized state matrix)
        B̅ = Δ * B                       (discretized input matrix)
        Δ, B, C = projections(x)        (data-dependent parameters)

    The key innovation is that Δ, B, and C are computed from the input,
    making the state space model "selective" and input-dependent.

    :param d_model: Dimensionality of input and output embeddings.
    :type d_model: int
    :param d_state: Dimensionality of the SSM latent state (N).
        Controls state space capacity. Defaults to 16.
    :type d_state: int
    :param d_conv: Kernel size for causal 1D convolution. Larger values
        increase local context window. Defaults to 4.
    :type d_conv: int
    :param expand: Expansion factor for internal dimension (d_inner = expand * d_model).
        Defaults to 2.
    :type expand: int
    :param dt_rank: Rank for the step size Δ projection. 'auto' sets to
        ceil(d_model/16). Controls expressiveness of temporal discretization.
        Defaults to "auto".
    :type dt_rank: Union[str, int]
    :param dt_min: Minimum clipping value for step size Δ. Defaults to 0.001.
    :type dt_min: float
    :param dt_max: Maximum clipping value for step size Δ. Defaults to 0.1.
    :type dt_max: float
    :param dt_init: Initialization strategy for Δ projection ("random" or "constant").
        Defaults to "random".
    :type dt_init: str
    :param dt_scale: Scaling factor for Δ initialization. Defaults to 1.0.
    :type dt_scale: float
    :param dt_init_floor: Minimum floor value for Δ initialization. Defaults to 1e-4.
    :type dt_init_floor: float
    :param conv_bias: Whether to use bias in convolution layer. Defaults to True.
    :type conv_bias: bool
    :param use_bias: Whether to use bias in linear projections. Defaults to False.
    :type use_bias: bool
    :param layer_idx: Optional layer index for caching in inference. Defaults to None.
    :type layer_idx: Optional[int]
    :param kwargs: Additional keyword arguments for Layer base class.

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

            # Create a Mamba layer
            mamba = MambaLayer(d_model=768, d_state=16, d_conv=4, expand=2)

            # Process a sequence
            x = keras.random.normal((2, 512, 768))
            y = mamba(x)
            print(y.shape)  # (2, 512, 768)

            # With custom parameters
            mamba = MambaLayer(
                d_model=1024,
                d_state=32,
                d_conv=8,
                expand=3,
                dt_rank=64
            )

    Note:
        The selective scan is implemented using `keras.ops.while_loop` because
        the SSM parameters are data-dependent, preventing a convolutional
        implementation. This makes the layer inherently sequential but enables
        the selective mechanism that is key to Mamba's performance.
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

        # Parameter validation
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, but got {d_model}")
        if d_state <= 0:
            raise ValueError(f"d_state must be positive, but got {d_state}")
        if d_conv <= 0:
            raise ValueError(f"d_conv must be positive, but got {d_conv}")
        if expand <= 0:
            raise ValueError(f"expand must be a positive integer, but got {expand}")

        # Store configuration
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

        # CREATE sub-layers in __init__ (following the guide)
        self.in_proj = keras.layers.Dense(
            self.d_inner * 2,
            use_bias=use_bias,
            name="in_proj"
        )

        self.conv1d = keras.layers.Conv1D(
            filters=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,  # Depthwise convolution
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

        # DECISION plan-2026-08-14T233721-d4f9beb2/D-084: the paper's dt_proj
        # initialization is passed as INITIALIZERS, not applied with `.assign()`
        # inside `build()`.
        #
        # WHAT NOT TO DO: do NOT move this back into `build()` as
        #     self.dt_proj.kernel.assign(...)   # dt_init_std
        #     self.dt_proj.bias.assign(inv_dt)  # inverse softplus of a
        #                                       # log-uniform draw in [dt_min, dt_max]
        # Keras 3 runs the symbolic build pass of a sublayer first reached from a
        # PARENT layer's `call()` -- i.e. every real Mamba model, since
        # `MambaLayer` is reached through `MambaResidualBlock` -- inside a
        # `StatelessScope`, which RECORDS the `.assign()` and then DISCARDS it.
        # Measured 2026-08-17, CPU: through a parent's `call()` the bias was
        # ALL ZERO, so `softplus(bias) == 0.693147` uniformly, against the
        # `[dt_min=0.001, dt_max=0.1]` range the paper's scheme exists to cover
        # -- ~7x above dt_max and ~660x above dt_min -- and the kernel was
        # Dense's default glorot (max 0.391 vs the intended 0.9995). The
        # selective-SSM timestep initialization that makes Mamba trainable was
        # simply absent, in every model, with every test green. A direct
        # `.build(...)` gave the correct values, which is why this was invisible.
        # Same defect class as D-021 (RoPE tables) and F-9 (N-BEATS bases).
        # See decisions.md D-084.
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

        # Weight attributes to be created in build()
        self.A_log = None
        self.D = None

    def _dt_kernel_initializer(self):
        """Initializer for ``dt_proj.kernel``: the paper's ``dt_init_std`` scale.

        The whole value is computed INSIDE the returned callable. Do not compute
        a tensor out here and close over it — under the symbolic build pass that
        tensor belongs to a scratch ``FuncGraph`` and raises "cannot be accessed
        from here ... out of scope" on the eager pass (the D-021 trap).

        :return: A callable ``(shape, dtype) -> tensor``.
        """
        dt_init_std = self.dt_rank ** -0.5 * self.dt_scale
        if self.dt_init == "constant":
            return keras.initializers.Constant(dt_init_std)
        # A Keras Initializer, NOT a bare `keras.random.uniform` call: inside the
        # symbolic build pass the global seed generator cannot allocate its state
        # variable and the call dies with "'NoneType' object has no attribute
        # 'assign'". Keras initializers handle their own seeding there.
        return keras.initializers.RandomUniform(
            minval=-dt_init_std, maxval=dt_init_std
        )

    def _dt_bias_initializer(self):
        """Initializer for ``dt_proj.bias``: ``inv_softplus`` of a log-uniform draw.

        Chosen so that ``softplus(bias)`` — the actual SSM timestep — lands
        log-uniformly in ``[dt_min, dt_max]``, which is what the Mamba paper's
        initialization is for. This is a RANDOM draw rather than a closed form,
        which is exactly why it was written as an ``.assign()`` in the first
        place; an initializer callable handles it just as well and, unlike the
        assign, survives the stateless build pass.

        :return: A callable ``(shape, dtype) -> tensor``.
        """
        log_min = math.log(self.dt_min)
        log_max = math.log(self.dt_max)
        floor = self.dt_init_floor
        # The uniform draw comes from a Keras Initializer rather than a bare
        # `keras.random.uniform`: inside the symbolic build pass the global seed
        # generator cannot allocate its state variable and the call dies with
        # "'NoneType' object has no attribute 'assign'". Everything downstream of
        # the draw is a pure `keras.ops` transform and is safe there.
        uniform = keras.initializers.RandomUniform(minval=0.0, maxval=1.0)

        def initializer(shape, dtype=None):
            u = uniform(shape, dtype=dtype or "float32")
            dt = keras.ops.exp(u * (log_max - log_min) + log_min)
            dt = keras.ops.clip(dt, floor, float("inf"))
            # Inverse of softplus: log(exp(x) - 1). `expm1` keeps this accurate
            # for the small dt values `dt_min` produces, where `exp(dt) - 1`
            # loses most of its significant digits.
            return keras.ops.log(keras.ops.expm1(dt))

        return initializer

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Create layer weights and build sub-layers.

        This method initializes:
        1. A_log: Log of state transition matrix using S4D initialization
        2. D: Skip connection parameter
        3. Special initialization for dt_proj weights/bias

        :param input_shape: Shape of input tensor (batch_size, seq_len, d_model).
        :type input_shape: Tuple[Optional[int], ...]
        """
        # Keras traces `call()` more than once on a layer's first invocation, so
        # without this guard `add_weight` runs twice and the second attempt dies
        # with "Variable .../A_log is already initialized." That makes this layer
        # unusable from inside a parent LAYER's `call()` -- the path the D-084
        # anchor below is about. Every sibling in this package already has it.
        if self.built:
            return

        # S4D real initialization for state matrix A.
        # Each row of A is initialized to [1, 2, ..., d_state].
        # We use NumPy for initialization to avoid potential graph context issues
        # with the TensorFlow backend when using keras.ops inside `build`.
        A_init = np.tile(
            np.arange(1, self.d_state + 1, dtype="float32"), (self.d_inner, 1)
        )

        self.A_log = self.add_weight(
            name="A_log",
            shape=(self.d_inner, self.d_state),
            initializer=keras.initializers.Constant(np.log(A_init)),
            trainable=True,
        )

        # D "skip" parameter - allows direct input-output connections
        self.D = self.add_weight(
            name="D",
            shape=(self.d_inner,),
            initializer="ones",
            trainable=True,
        )

        # Build sub-layers explicitly (critical for serialization)
        self.in_proj.build(input_shape)

        # Conv1D expects (batch, length, channels) but we'll transpose
        conv_input_shape = (input_shape[0], input_shape[1], self.d_inner)
        self.conv1d.build(conv_input_shape)

        self.activation.build(conv_input_shape)

        # x_proj input after reshaping
        x_proj_input_shape = (None, self.d_inner)
        self.x_proj.build(x_proj_input_shape)

        # dt_proj input
        dt_proj_input_shape = (None, self.dt_rank)
        self.dt_proj.build(dt_proj_input_shape)

        # out_proj input
        out_proj_input_shape = (input_shape[0], input_shape[1], self.d_inner)
        self.out_proj.build(out_proj_input_shape)

        # dt_proj's special initialization is NOT applied here -- it is carried
        # by the initializers passed at construction. See the D-084 anchor in
        # `__init__`: an `.assign()` at this point is silently discarded whenever
        # this layer is first reached from a parent's `call()`, which is every
        # real model.

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
        Perform the selective scan operation over the sequence.

        This is the core SSM computation implementing the recurrent state updates
        with data-dependent discretization. Uses a while loop because the parameters
        vary per timestep based on input content.

        :param u: Input tensor after convolution, shape (batch, d_inner, seq_len).
        :type u: keras.KerasTensor
        :param delta: Step size Δ, shape (batch, d_inner, seq_len).
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

        # DECISION plan-2026-08-19T163559-499b6f0e/D-044
        # The recurrent scan runs in the VARIABLE dtype (float32 under
        # `mixed_float16`), never in the compute dtype. `call()` already casts
        # `A_log` to float32 on purpose, and `keras.ops.einsum` PROMOTES rather
        # than raising, so `deltaA` came out float32 while `h` was materialised at
        # `compute_dtype` — the exception then landed on `deltaA[:, :, t] * h`
        # inside `body`, two statements away from its cause. Do NOT "fix" this by
        # casting `A` down to float16: this is a sequential accumulation over
        # `seq_len` steps and half-precision state is exactly where it drifts.
        # See decisions.md D-044.
        scan_dtype = self.variable_dtype
        u = keras.ops.cast(u, scan_dtype)
        delta = keras.ops.cast(delta, scan_dtype)
        A = keras.ops.cast(A, scan_dtype)
        B = keras.ops.cast(B, scan_dtype)
        C = keras.ops.cast(C, scan_dtype)
        D = keras.ops.cast(D, scan_dtype)
        # `z` is deliberately NOT lifted: the SiLU gate is a `keras.layers.Activation`,
        # which runs at `compute_dtype`, so the gating is applied AFTER the result
        # comes back down (see the return statement).

        # Discretize continuous parameters A and B
        # A_bar = exp(Δ * A)
        deltaA = keras.ops.exp(
            keras.ops.einsum("bdl,dn->bdln", delta, A)
        )

        # B_bar * u = Δ * B * u
        deltaB_u = keras.ops.einsum(
            "bdl,bnl,bdl->bdln", delta, B, u
        )

        # Initialize hidden state
        h = keras.ops.zeros(
            (batch_size, d_inner, self.d_state),
            dtype=scan_dtype
        )

        # Storage for outputs
        ys = keras.ops.zeros(
            (seq_len, batch_size, d_inner),
            dtype=scan_dtype
        )

        # Time step counter
        t = keras.ops.convert_to_tensor(0, dtype="int32")

        def condition(t: keras.KerasTensor, h: keras.KerasTensor,
                     ys: keras.KerasTensor) -> keras.KerasTensor:
            """Loop condition: continue while t < seq_len."""
            return keras.ops.less(t, seq_len)

        def body(t: keras.KerasTensor, h: keras.KerasTensor,
                ys: keras.KerasTensor) -> Tuple[keras.KerasTensor, ...]:
            """
            Single timestep of SSM computation.

            Updates state: h_t = A_bar * h_{t-1} + B_bar * u_t
            Computes output: y_t = C * h_t
            """
            # State update
            h = deltaA[:, :, t] * h + deltaB_u[:, :, t]

            # Output computation
            y_t = keras.ops.einsum("bdn,bn->bd", h, C[:, :, t])

            # Store output
            # For scatter_update to update a slice, `indices` needs to specify
            # the index of the slice. For a single slice, it should have shape (1, 1).
            # `t` is a scalar, so we reshape it.
            indices = keras.ops.reshape(t, (1, 1))
            # The `updates` tensor needs to have a matching leading dimension with
            # indices, so we add a dimension to `y_t`.
            updates = keras.ops.expand_dims(y_t, axis=0)
            ys = keras.ops.scatter_update(ys, indices, updates)

            return t + 1, h, ys

        # Run the recurrent computation
        _, _, final_ys = keras.ops.while_loop(
            cond=condition,
            body=body,
            loop_vars=(t, h, ys),
            maximum_iterations=seq_len
        )

        # Transpose output to (batch, d_inner, seq_len)
        y = keras.ops.transpose(final_ys, (1, 2, 0))

        # Add skip connection: y = y + D * u
        y = y + keras.ops.expand_dims(keras.ops.expand_dims(D, 0), -1) * u

        # Apply gating: y = y * silu(z), at the layer's compute dtype.
        y = keras.ops.cast(y, self.compute_dtype)
        return y * self.activation(z)

    def call(
        self,
        hidden_states: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass through the Mamba layer.

        :param hidden_states: Input tensor, shape (batch, seq_len, d_model).
        :type hidden_states: keras.KerasTensor
        :param training: Whether in training mode. Defaults to None.
        :type training: Optional[bool]
        :return: Output tensor, shape (batch, seq_len, d_model).
        :rtype: keras.KerasTensor
        """
        batch_size, seq_len, _ = keras.ops.shape(hidden_states)

        # 1. Input projection: split into x and z paths
        xz = self.in_proj(hidden_states, training=training)
        xz = keras.ops.transpose(xz, (0, 2, 1))  # (B, 2*D_inner, L)
        x, z = keras.ops.split(xz, 2, axis=1)    # Each (B, D_inner, L)

        # 2. Causal convolution and activation
        x = keras.ops.transpose(x, (0, 2, 1))    # (B, L, D_inner)
        x_conv = self.conv1d(x, training=training)  # (B, L, D_inner)
        x_conv = self.activation(x_conv)

        # 3. Compute SSM parameters (Δ, B, C)
        # Reshape for projection
        x_reshaped = keras.ops.reshape(x_conv, (-1, self.d_inner))
        x_proj_output = self.x_proj(x_reshaped, training=training)

        # Split into dt_raw, B, C
        # The `keras.ops.split` function expects split *indices*, not sizes.
        # To get 3 tensors, we need 2 split points.
        split_indices = [self.dt_rank, self.dt_rank + self.d_state]
        dt_raw, B_raw, C_raw = keras.ops.split(
            x_proj_output,
            split_indices,
            axis=-1
        )

        # Project dt to get delta
        dt = self.dt_proj(dt_raw, training=training)
        dt = keras.ops.reshape(dt, (batch_size, seq_len, self.d_inner))
        dt = keras.ops.transpose(dt, (0, 2, 1))  # (B, D_inner, L)

        # Reshape B and C
        B = keras.ops.reshape(B_raw, (batch_size, seq_len, self.d_state))
        B = keras.ops.transpose(B, (0, 2, 1))  # (B, N, L)

        C = keras.ops.reshape(C_raw, (batch_size, seq_len, self.d_state))
        C = keras.ops.transpose(C, (0, 2, 1))  # (B, N, L)

        # Get state matrix A (continuous, negative for stability)
        A = -keras.ops.exp(keras.ops.cast(self.A_log, "float32"))

        # Apply softplus to delta for positivity
        delta = keras.ops.softplus(dt)

        # Prepare x_conv for selective scan
        x_conv_transposed = keras.ops.transpose(x_conv, (0, 2, 1))  # (B, D_inner, L)

        # 4. Selective scan (core SSM computation)
        y = self._selective_scan(
            u=x_conv_transposed,
            delta=delta,
            A=A,
            B=B,
            C=C,
            D=self.D,
            z=z
        )

        # 5. Output projection
        y = keras.ops.transpose(y, (0, 2, 1))  # (B, L, D_inner)
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

@keras.saving.register_keras_serializable()
class MambaResidualBlock(keras.layers.Layer):
    """
    Residual block wrapping a MambaLayer with pre-normalization.

    Implements the standard pre-norm residual architecture:
        output = hidden_states + MambaLayer(LayerNorm(hidden_states))

    This is the fundamental building block for stacking multiple Mamba layers
    to create deep sequence models. The pre-normalization pattern (also called
    "pre-activation") has been shown to improve training stability in deep networks.

    **Intent**:
    Provide a reusable residual wrapper that handles normalization and skip
    connections, allowing the core MambaLayer to focus solely on the SSM logic.

    **Architecture**:

    .. code-block:: text

        Input (residual from previous block)
           │
           ├─────────────┐
           │             │
           ▼             │
        LayerNorm        │
           │             │
           ▼             │
        MambaLayer       │
           │             │
           ▼             │
        Add ←────────────┘
           │
           ▼
        Output (new hidden_states + new residual)

    :param d_model: Dimensionality of the input and output.
    :type d_model: int
    :param norm_epsilon: Epsilon for layer normalization. Defaults to 1e-5.
    :type norm_epsilon: float
    :param mamba_kwargs: Keyword arguments to pass to MambaLayer constructor.
        Should include parameters like d_state, d_conv, expand, etc.
    :type mamba_kwargs: Optional[Dict[str, Any]]
    :param kwargs: Additional keyword arguments for Layer base class.

    Input shape:
        Tuple of:
        - hidden_states: 3D tensor (batch_size, seq_len, d_model)
        - residual: Optional 3D tensor (batch_size, seq_len, d_model) or None

    Output shape:
        Tuple of:
        - new_hidden_states: 3D tensor (batch_size, seq_len, d_model)
        - new_residual: 3D tensor (batch_size, seq_len, d_model)

    :ivar norm: Layer normalization applied before the Mamba layer.
    :vartype norm: keras.layers.LayerNormalization
    :ivar mamba: The core Mamba SSM layer.
    :vartype mamba: MambaLayer

    Example:
        .. code-block:: python

            # Create a residual block
            block = MambaResidualBlock(
                d_model=768,
                mamba_kwargs={
                    "d_state": 16,
                    "d_conv": 4,
                    "expand": 2,
                    "layer_idx": 0
                }
            )

            # First block (no residual)
            x = keras.random.normal((2, 512, 768))
            hidden, residual = block(x, residual=None)

            # Subsequent blocks (with residual)
            hidden, residual = block(hidden, residual=residual)

    Note:
        This implementation returns both the new hidden states and the new
        residual separately, allowing efficient residual accumulation without
        repeated additions in deep networks.
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

        # CREATE sub-layers in __init__
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
        Build sub-layers.

        :param input_shape: Shape of input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        """
        # Build sub-layers explicitly for proper serialization
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
        Forward pass through the residual block.

        :param hidden_states: Main input tensor, shape (batch, seq_len, d_model).
        :type hidden_states: keras.KerasTensor
        :param residual: Optional residual from previous block. Defaults to None.
        :type residual: Optional[keras.KerasTensor]
        :param training: Whether in training mode. Defaults to None.
        :type training: Optional[bool]
        :return: Tuple of (new_hidden_states, new_residual).
        :rtype: Tuple[keras.KerasTensor, keras.KerasTensor]
        """
        # Compute new residual (before normalization)
        new_residual = (
            hidden_states + residual if residual is not None else hidden_states
        )

        # Apply pre-normalization
        normalized = self.norm(new_residual, training=training)

        # Apply Mamba layer
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
