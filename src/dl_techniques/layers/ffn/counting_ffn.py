"""
A feed-forward network, ``CountingFFN``, that measures how often learned
events occur over a sequence and blends that measurement back into each
position.

The layer detects countable events with a sigmoid key projection, aggregates
them along the sequence with a *decay-weighted, mask-aware average*, and
gates the result back into each position over a residual shortcut.

The aggregate is deliberately **not** a raw sum. A raw cumulative sum of
sigmoid values grows as ``0.5 * t``, which destroys the layer three separate
ways: it swamps the residual at initialization, it shifts out of
distribution when the sequence length changes, and it silently saturates in
bf16 (a running sum of 0.5 stops incrementing at 256, after roughly 385
tokens, because bf16 has 8 mantissa bits). Worse, the content signal in a
raw sum grows as ``O(sqrt(t))`` against an ``O(t)`` positional baseline, so
at ``t = 512`` about 98% of what the transform receives is a linear ramp in
position carrying no information about the input at all.

This version separates the two quantities that a raw sum conflates:

.. code-block:: text

    density   rho_t = S_t / Z_t      in [0, 1] exactly, for every t, T and
                                     mask. Pure content. Length invariant.
    extent    log1p(Z_t)             how much sequence was aggregated.
                                     Pure position. Grows as log T, so
                                     T = 8192 gives 9.0, not 4096.

both computed from one first-order linear recurrence run over the sequence:

.. code-block:: text

    S_t = lambda * S_{t-1} + k_t * m_t        numerator   (events seen)
    Z_t = lambda * Z_{t-1} +       m_t        denominator (slots counted)

    rho_t = S_t / (Z_t + eps)

``lambda`` is a learnable per-channel decay in ``(0, 1)``, initialized so
the channels span geometrically spaced timescales. It is what makes
``counting_scope='local'`` actually local: the effective window is
``1 / (1 - lambda)``. At ``lambda -> 1`` the recurrence degenerates to a
plain running mean over the whole valid prefix, so cumulative counting
remains reachable; the normalization by ``Z`` keeps it bounded even then.

The mask enters both the numerator and the denominator, so padded positions
contribute nothing to either and the ratio stays exactly correct. This is
the same construction as Adam's bias correction, and it also fixes the
warm-up bias at small ``t``: at the first valid position ``Z = 1`` and
``rho = k``, with no cold-start transient.

Why a decay and not a raw bidirectional sum. In the previous version,
``'local'`` concatenated a forward and a backward cumulative sum. Those obey
an exact identity, ``bwd_t = S + k_t - fwd_t``, so the backward half was an
affine function of the forward half and contributed only the sequence total
and the current key while doubling the transform's input width. A decay
breaks that identity: forward and backward accumulate different weightings
of the same events, so both halves carry information.

References:
    - Hochreiter & Schmidhuber, 1997. Long Short-Term Memory. (The forget
      gate is the point; ``lambda`` is that gate, and a raw cumsum is an
      LSTM with the forget gate removed.)
    - He et al., 2016. Deep Residual Learning for Image Recognition.
    - Poli et al., 2023. Hyena Hierarchy: Towards Larger Convolutional
      Language Models.
    - Sun et al., 2023. Retentive Network. (Multi-scale geometric decay
      initialization.)
"""

import math
import keras
from typing import Callable, Literal, Tuple, Optional, Union, Any, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.initializers.clone import clone_initializer
from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.layers.activations.common import resolve_activation, serialize_activation

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.ffn.counting_ffn")
class CountingFFN(keras.layers.Layer):
    """
    Feed-forward network that measures event density across a sequence.

    ``key_projection`` detects countable events with a sigmoid
    (``k_t = sigmoid(W_k @ x_t)``). ``counting_scope`` says how those are
    aggregated along the sequence axis. The aggregate is a decay-weighted
    average in ``[0, 1]`` paired with a log-scaled extent, normalized,
    transformed to ``output_dim``, and gated into a residual shortcut.

    Aggregation happens on axis 1, so the input must be rank 3 with the
    sequence on that axis. Rank 2 is rejected rather than silently
    aggregating over the feature axis.

    Architecture:

    .. code-block:: text

        ┌──────────────────────────────────────┐
        │ Input x  [B, T, input_dim]           │
        │ mask m   [B, T]  (optional)          │
        └──────────────────┬───────────────────┘
                           │
              ┌────────────┼────────────┬──────────────┐
              ▼            ▼            ▼              │
        ┌───────────┐ ┌─────────┐ ┌───────────┐        │
        │ key_proj  │ │  gate   │ │ shortcut  │        │
        │ Dense(C)  │ │ Dense(O)│ │ x, or     │        │
        │ sigmoid   │ │ sigmoid │ │ Dense(O)  │        │
        └─────┬─────┘ │ bias<0  │ │ if O != d │        │
              │       └────┬────┘ └─────┬─────┘        │
              ▼            │            │              │
        ┌────────────────┐ │            │              │
        │ masked decayed │ │            │              │
        │ scan (fp32)    │ │            │              │
        │ -> rho, log1pZ │ │            │              │
        └───────┬────────┘ │            │              │
                ▼          │            │              │
        ┌────────────────┐ │            │              │
        │   count_norm   │ │            │              │
        │  LayerNorm     │ │            │              │
        └───────┬────────┘ │            │              │
                ▼          │            │              │
        ┌────────────────┐ │            │              │
        │count_transform │ │            │              │
        │ Dense(O), act  │ │            │              │
        └───────┬────────┘ │            │              │
                └─────┬────┴────────────┘              │
                      ▼                                │
             y = g*C' + (1-g)*shortcut  ◄──────────────┘
                      ▼
        ┌──────────────────────────────────────┐
        │ Output  [B, T, output_dim]           │
        └──────────────────────────────────────┘

        C = count_dim, O = output_dim, d = input width.

        The residual blend now runs on EVERY path. When O != d a
        bias-free projection supplies the shortcut, so the input
        always has an additive route to the output. The previous
        version dropped the shortcut in that case and emitted a
        warning; at g -> 0 its output was exactly zero and the
        gradient to everything upstream died.

    The counting_scope fork:

    .. code-block:: text

              k = sigmoid(key_projection(x)) * m   [B, T, C]
                            │
             ┌──────────────┼──────────────┐
             ▼              ▼              ▼
         'global'       'causal'        'local'
             │              │              │
             ▼              ▼              ▼
        masked mean    forward scan   forward AND
        over all T     (prefix only)  backward scans
        [B, 1, C+1]    [B, T, C+1]    [B, T, 2C+2]
        broadcast                     concatenated
        after the
        transform

        'global' is exact and needs no decay: it is the masked mean
        over the whole sequence, identical at every position. It is
        transformed at [B, 1, *] and broadcast afterwards, which is
        T times cheaper than broadcasting first and running a Dense
        over T identical rows.

        'causal' never looks ahead and is safe for autoregressive
        use. 'local' is bidirectional and leaks the future into
        every position; it is local in the sense of a finite decay
        window, not causal. Do not use it autoregressively.

    Numerical policy:

    .. code-block:: text

        The recurrence is forced to float32 regardless of the
        compute dtype, then cast back. Under a bf16 policy a raw
        accumulation of sigmoid values stalls after ~385 steps and
        freezes at 256; fp16 stalls after ~2049 and freezes at
        1024. Both fail silently, with no NaN and no warning. The
        normalized output rho lies in [0, 1], so the cast back is
        lossless in any float format.

    :param output_dim: Width of the output. Must be positive. It no longer
        needs to match the input width; a bias-free projection supplies the
        shortcut when it does not.
    :type output_dim: int
    :param count_dim: Number of countable events ``key_projection`` learns.
        Must be positive. Note this also caps the rank of the layer's
        additive contribution: everything the counting path injects lives in
        a subspace of dimension at most ``2 * count_dim + 2``.
    :type count_dim: int
    :param counting_scope: 'global', 'local' or 'causal'. Defaults to
        'causal', changed from the previous default of 'local' because
        'local' is bidirectional and silently breaks autoregressive models.
    :type counting_scope: Literal["global", "local", "causal"]
    :param activation: Activation for ``count_transform``. A name or a
        callable. Defaults to 'gelu'.
    :type activation: Union[str, Callable]
    :param use_bias: Whether the Dense layers carry a bias. Defaults to True.
        Setting it False forfeits the near-identity gate initialization, since
        ``gate_bias_init`` has nowhere to live.
    :type use_bias: bool
    :param gate_bias_init: Constant initializer for the gate bias. Defaults
        to -2.0, giving ``sigmoid(-2) ~ 0.12`` so the layer starts close to
        the identity and opens the count path as it becomes useful. The
        previous version started at 0.5, blending in an unnormalized count
        path measured at 30x the residual.
    :type gate_bias_init: float
    :param decay_min_window: Shortest effective window at initialization, in
        positions. Must be greater than 1. Defaults to 4.0.
    :type decay_min_window: float
    :param decay_max_window: Longest effective window at initialization.
        Must be at least ``decay_min_window``. Defaults to 512.0. Channels
        are seeded geometrically between the two, so a single layer observes
        several timescales at once.
    :type decay_max_window: float
    :param trainable_decay: Whether the per-channel decay is learned.
        Defaults to True. Ignored for ``counting_scope='global'``, which has
        no decay.
    :type trainable_decay: bool
    :param epsilon: Guard on the density denominator and the variance
        epsilon for ``count_norm``. Defaults to 1e-6.
    :type epsilon: float
    :param kernel_initializer: Initializer for the kernels. Each Dense gets
        its own clone. Defaults to 'glorot_uniform'.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for the biases, also cloned. The
        gate overrides it with ``gate_bias_init``. Defaults to 'zeros'.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Regularizer for the kernels. Defaults to None.
        Cloned per Dense so stateful regularizers do not share state.
    :type kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param bias_regularizer: Regularizer for the biases. Defaults to None.
        Also cloned per Dense.
    :type bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param kwargs: Extra arguments for ``keras.layers.Layer``. A ``max_count``
        key is accepted and discarded for backward compatibility.
    :type kwargs: Any

    :ivar output_dim: The stored output width.
    :vartype output_dim: int
    :ivar count_dim: The stored number of countable events.
    :vartype count_dim: int
    :ivar counting_scope: The stored scope name.
    :vartype counting_scope: str
    :ivar activation: The resolved activation, a callable.
    :vartype activation: Callable
    :ivar use_bias: Whether the Dense layers carry a bias.
    :vartype use_bias: bool
    :ivar gate_bias_init: The stored gate bias constant.
    :vartype gate_bias_init: float
    :ivar decay_min_window: Shortest seeded window.
    :vartype decay_min_window: float
    :ivar decay_max_window: Longest seeded window.
    :vartype decay_max_window: float
    :ivar trainable_decay: Whether the decay is learned.
    :vartype trainable_decay: bool
    :ivar epsilon: The stored epsilon.
    :vartype epsilon: float
    :ivar key_projection: ``Dense(count_dim, activation='sigmoid')``.
    :vartype key_projection: keras.layers.Dense
    :ivar count_norm: LayerNorm over the aggregated features, which mixes
        densities in ``[0, 1]`` with extents around ``log T``.
    :vartype count_norm: keras.layers.LayerNormalization
    :ivar count_transform: ``Dense(output_dim, activation=activation)``.
    :vartype count_transform: keras.layers.Dense
    :ivar gate: ``Dense(output_dim, activation='sigmoid')``, read from the
        original input.
    :vartype gate: keras.layers.Dense
    :ivar input_proj: ``Dense(output_dim, use_bias=False)`` supplying the
        residual shortcut, created in ``build()`` only when ``output_dim``
        differs from the input width, otherwise ``None``.
    :vartype input_proj: Optional[keras.layers.Dense]
    :ivar decay_logits: Per-channel decay logits, shape ``(count_dim,)``.
        ``lambda = sigmoid(decay_logits)``. ``None`` for scope 'global'.
    :vartype decay_logits: Optional[keras.Variable]
    :ivar _input_last_dim: The input's last axis, captured in ``build()`` as
        a Python int. ``None`` before ``build()``, and not serialized.
    :vartype _input_last_dim: Optional[int]

    :raises ValueError: If ``output_dim`` or ``count_dim`` is not positive.
    :raises ValueError: If ``counting_scope`` is not 'global', 'local' or
        'causal'.
    :raises ValueError: If ``decay_min_window <= 1`` or
        ``decay_max_window < decay_min_window``.
    :raises ValueError: If ``epsilon`` is not positive.
    :raises ValueError: From ``build()``, if the input is not rank 3 or its
        last axis is ``None``.

    Input shape:
        3D tensor ``(batch, sequence_length, input_dim)``, optionally masked.

    Output shape:
        ``(batch, sequence_length, output_dim)``.

    Example:
        .. code-block:: python

            ffn = CountingFFN(output_dim=32, count_dim=16,
                              counting_scope="global")
            x = keras.random.normal((2, 10, 32))
            m = keras.ops.concatenate([
                keras.ops.ones((2, 6)), keras.ops.zeros((2, 4))
            ], axis=1) > 0
            y = ffn(x, mask=m)
            y.shape                          # (2, 10, 32)
            ffn.verify_mask_invariance()     # (True, 1.1e-07)

    Note:
        ``counting_scope='local'`` is bidirectional. Use 'causal', the
        default, for anything autoregressive.

    Note:
        The counting path is cheap and the gate is not. At
        ``input_dim = output_dim = 512`` and ``count_dim = 16``, the gate
        holds roughly 90% of the layer's parameters.

    Note:
        Before attributing a gain to counting, ablate it: replace
        ``key_projection`` with a constant. The aggregate carries positional
        information in ``log1p(Z)`` even when the keys are uninformative, and
        the previous raw-sum version was 98% positional at ``t = 512``.
    """

    def __init__(
        self,
        output_dim: int,
        count_dim: int,
        counting_scope: Literal["global", "local", "causal"] = "causal",
        activation: Union[str, Callable] = "gelu",
        use_bias: bool = True,
        gate_bias_init: float = -2.0,
        decay_min_window: float = 4.0,
        decay_max_window: float = 512.0,
        trainable_decay: bool = True,
        epsilon: float = 1e-6,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Validate the configuration and create the sub-layers.

        Every argument is documented on the class. A ``max_count`` keyword is
        dropped before ``super().__init__`` so old configs still load.

        :raises ValueError: If any argument is out of range, or if
            ``counting_scope`` is not one of 'global', 'local', 'causal'.
        """
        kwargs.pop("max_count", None)
        super().__init__(**kwargs)

        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if count_dim <= 0:
            raise ValueError(f"count_dim must be positive, got {count_dim}")
        if counting_scope not in ("global", "local", "causal"):
            raise ValueError(
                f"counting_scope must be one of 'global', 'local', 'causal', "
                f"but got {counting_scope}"
            )
        if decay_min_window <= 1.0:
            raise ValueError(
                f"decay_min_window must be > 1.0, got {decay_min_window}"
            )
        if decay_max_window < decay_min_window:
            raise ValueError(
                f"decay_max_window ({decay_max_window}) must be >= "
                f"decay_min_window ({decay_min_window})"
            )
        if epsilon <= 0.0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        self.output_dim = output_dim
        self.count_dim = count_dim
        self.counting_scope = counting_scope
        self.activation = resolve_activation(activation)
        self.use_bias = use_bias
        self.gate_bias_init = gate_bias_init
        self.decay_min_window = decay_min_window
        self.decay_max_window = decay_max_window
        self.trainable_decay = trainable_decay
        self.epsilon = epsilon
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # The activation argument exactly as passed; only the build() log
        # line reads it. get_config() serializes self.activation instead.
        self._activation_identifier = activation

        # Masked positions must not reach the aggregate at all.
        self.supports_masking = True

        # Initializers are cloned per Dense so no mutable seed-generator
        # state is shared between sub-layers. This does NOT decorrelate two
        # layers under a seeded initializer: clone_initializer round-trips
        # get_config, which preserves the seed, so a seeded clone reproduces
        # its source exactly. That is what asking for a seed means.
        self.key_projection = keras.layers.Dense(
            self.count_dim,
            activation="sigmoid",
            use_bias=self.use_bias,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            kernel_regularizer=self._clone_regularizer(self.kernel_regularizer),
            bias_regularizer=self._clone_regularizer(self.bias_regularizer),
            name="key_projection",
        )

        # Densities live in [0, 1]; extents live near log(T). count_norm
        # puts them on one scale before the transform sees them.
        self.count_norm = keras.layers.LayerNormalization(
            epsilon=self.epsilon,
            center=True,
            scale=True,
            name="count_norm",
        )

        # Input width resolved in build() via _aggregate_feature_dim().
        self.count_transform = keras.layers.Dense(
            self.output_dim,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            kernel_regularizer=self._clone_regularizer(self.kernel_regularizer),
            bias_regularizer=self._clone_regularizer(self.bias_regularizer),
            name="count_transform",
        )

        # A negative bias starts the gate near closed, so the layer begins
        # as (almost) the identity on the shortcut rather than as a 50/50
        # blend with an untrained count path.
        self.gate = keras.layers.Dense(
            self.output_dim,
            activation="sigmoid",
            use_bias=self.use_bias,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=keras.initializers.Constant(self.gate_bias_init),
            kernel_regularizer=self._clone_regularizer(self.kernel_regularizer),
            bias_regularizer=self._clone_regularizer(self.bias_regularizer),
            name="gate",
        )

        # Created in build(), where the input width is known.
        self.input_proj: Optional[keras.layers.Dense] = None
        self.decay_logits: Optional[keras.Variable] = None

        # Static input feature dim, resolved in build() (graph-safe branch).
        self._input_last_dim: Optional[int] = None

    @staticmethod
    def _clone_regularizer(
        regularizer: Optional[keras.regularizers.Regularizer]
    ) -> Optional[keras.regularizers.Regularizer]:
        """
        Return an independent copy of a regularizer, or ``None``.

        Sharing one instance across sub-layers is safe for the stateless
        built-ins but not for stateful ones. Cloning costs nothing.

        :param regularizer: The regularizer to copy, or ``None``.
        :type regularizer: Optional[keras.regularizers.Regularizer]
        :return: A fresh instance with the same configuration, or ``None``.
        :rtype: Optional[keras.regularizers.Regularizer]
        """
        if regularizer is None:
            return None
        return regularizer.__class__.from_config(regularizer.get_config())

    def _aggregate_feature_dim(self) -> int:
        """
        Width of the vector handed to ``count_norm`` and ``count_transform``.

        Each direction contributes ``count_dim`` density channels plus one
        log-extent channel. 'local' runs two directions.

        :return: The aggregated feature width.
        :rtype: int
        """
        per_direction = self.count_dim + 1
        return 2 * per_direction if self.counting_scope == "local" else per_direction

    def _decay_logit_init(self) -> keras.initializers.Initializer:
        """
        Seed the per-channel decay at geometrically spaced timescales.

        A channel with effective window ``w`` needs ``lambda = 1 - 1/w``,
        hence a logit of ``log(w - 1)``. Spacing the windows geometrically
        between ``decay_min_window`` and ``decay_max_window`` lets one layer
        observe short and long ranges simultaneously, rather than committing
        every channel to the same timescale.

        :return: A constant initializer holding one logit per channel.
        :rtype: keras.initializers.Initializer
        """
        lo, hi = self.decay_min_window, self.decay_max_window
        if self.count_dim == 1 or hi == lo:
            windows = [math.sqrt(lo * hi)] * self.count_dim
        else:
            ratio = hi / lo
            windows = [
                lo * (ratio ** (c / (self.count_dim - 1)))
                for c in range(self.count_dim)
            ]
        return keras.initializers.Constant([math.log(w - 1.0) for w in windows])

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the Counting FFN and all its sub-layers.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the input is not rank 3, or its last axis is
            ``None``.
        """
        if self.built:
            return

        # Rank 3 is required, not merely expected. Aggregation reduces over
        # axis 1; for a rank-2 input that axis is the feature axis, so the
        # layer would be accumulating across channels, which is meaningless.
        if len(input_shape) != 3:
            raise ValueError(
                f"CountingFFN expects a rank-3 input (batch, sequence, features), "
                f"got {len(input_shape)}D: {input_shape}"
            )
        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError("Input feature dimension must be specified")

        # Captured as a Python int so call() branches without a dynamic
        # keras.ops.shape() tensor, which is graph-unsafe under @tf.function.
        self._input_last_dim = int(input_dim)

        logger.info(
            f"Building CountingFFN: input_dim={input_dim}, output_dim={self.output_dim}, "
            f"count_dim={self.count_dim}, counting_scope='{self.counting_scope}', "
            f"activation='{self._activation_identifier}'"
        )

        self.key_projection.build(input_shape)
        self.gate.build(input_shape)

        # The residual shortcut always exists. When the widths differ it is a
        # bias-free projection rather than nothing, so the input keeps an
        # additive route to the output and the layer stays trainable at
        # g -> 0. No warning is needed because nothing is being given up.
        if self.output_dim != input_dim:
            self.input_proj = keras.layers.Dense(
                self.output_dim,
                activation=None,
                use_bias=False,
                kernel_initializer=clone_initializer(self.kernel_initializer),
                kernel_regularizer=self._clone_regularizer(self.kernel_regularizer),
                name="input_proj",
            )
            self.input_proj.build(input_shape)

        # 'global' is an exact masked mean over the whole sequence; there is
        # no recurrence and therefore no decay to learn.
        if self.counting_scope != "global":
            self.decay_logits = self.add_weight(
                name="decay_logits",
                shape=(self.count_dim,),
                initializer=self._decay_logit_init(),
                trainable=self.trainable_decay,
                dtype="float32",
            )

        feature_shape = tuple(input_shape[:-1]) + (self._aggregate_feature_dim(),)
        self.count_norm.build(feature_shape)
        self.count_transform.build(feature_shape)

        super().build(input_shape)

    @staticmethod
    def _scan_step(
        carry: keras.KerasTensor,
        x: Tuple[keras.KerasTensor, keras.KerasTensor],
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        One step of the recurrence ``s_t = a_t * s_{t-1} + b_t``.

        :param carry: ``s_{t-1}``, the running state.
        :type carry: keras.KerasTensor
        :param x: ``(a_t, b_t)`` for this position.
        :type x: Tuple[keras.KerasTensor, keras.KerasTensor]
        :return: ``(s_t, s_t)`` -- the new carry, and the emitted output,
            which are the same value for this recurrence.
        :rtype: Tuple[keras.KerasTensor, keras.KerasTensor]
        """
        a_t, b_t = x
        new_state = a_t * carry + b_t
        return new_state, new_state

    def _directional_features(
        self,
        keys: keras.KerasTensor,
        mask: keras.KerasTensor,
        reverse: bool,
    ) -> keras.KerasTensor:
        """
        Run the leaky accumulator in one direction and return its features.

        Computes the numerator and denominator recurrences in a single scan
        and returns ``concat([rho, log1p(Z)])``. The decay is applied at
        every position, including padded ones; the mask enters only the
        accumulated quantities, so padded slots add nothing to either the
        numerator or the denominator and the ratio stays exact.

        All of this runs in float32. Under a bf16 policy a raw accumulation
        of sigmoid values stalls after roughly 385 steps and freezes at 256,
        silently. The returned density is bounded by ``[0, 1]``, so casting
        back to the compute dtype loses nothing.

        :param keys: Masked event probabilities, ``[B, T, C]``, float32.
        :type keys: keras.KerasTensor
        :param mask: Validity indicator, ``[B, T, 1]``, float32.
        :type mask: keras.KerasTensor
        :param reverse: Whether to scan from the end of the sequence.
        :type reverse: bool
        :return: ``[B, T, count_dim + 1]`` in float32.
        :rtype: keras.KerasTensor
        """
        lam = keras.ops.sigmoid(keras.ops.cast(self.decay_logits, "float32"))

        # a_t is constant in t but the scan slices along the sequence axis,
        # so it is materialized to the full length. ones_like keeps T dynamic.
        decay = keras.ops.ones_like(keys) * lam

        # One scan carries both recurrences: the numerator accumulates
        # events, the denominator accumulates the slots they could occupy.
        numerator = keys
        denominator = keras.ops.ones_like(keys) * mask

        # DECISION plan-2026-09-18-1f3c0ce8/D-010: sequential keras.ops.scan,
        # not keras.ops.associative_scan. Do NOT change this back to the
        # parallel associative_scan this file used to call: on the
        # TensorFlow backend it MEASURABLY breaks two separate ways, neither
        # of which is specific to this layer's own logic --
        # (1) its recursive `_scan` helper has no base case for a length-1
        # axis (only lengths 2 and 3 short-circuit; a length-1 input reduces
        # to a length-0 array and recurses forever), so T == 1 -- ordinary
        # for single-token decoding -- crashed with a bare RecursionError;
        # (2) independently of sequence length, tracing it inside a
        # `tf.function` (which is what `model.fit()` uses by default)
        # crashed with a SECOND RecursionError, this time deep inside
        # protobuf's GraphDef serialization, reproduced with a minimal
        # `@tf.function`-wrapped `associative_scan` call carrying no
        # CountingFFN code at all. A sequential scan sidesteps both: it has
        # no recursive base-case logic and no nested `tf.cond` tree to trace.
        # Verified equal to the old parallel scan to float32-rounding scale
        # (max|delta|=2.384e-07, both scan directions, T=7) before relying on
        # it, and verified to run under a compiled `model.fit()` where the
        # old code raised. See decisions.md D-010 for the full derivation.
        combined_decay = keras.ops.moveaxis(
            keras.ops.concatenate([decay, decay], axis=-1), 1, 0
        )
        combined_values = keras.ops.moveaxis(
            keras.ops.concatenate([numerator, denominator], axis=-1), 1, 0
        )
        initial_state = keras.ops.zeros_like(combined_values[0])
        _, accumulated = keras.ops.scan(
            self._scan_step,
            initial_state,
            (combined_decay, combined_values),
            reverse=reverse,
        )
        accumulated = keras.ops.moveaxis(accumulated, 0, 1)

        events, slots = keras.ops.split(accumulated, 2, axis=-1)

        # rho in [0, 1] exactly: events <= slots pointwise, because k <= 1
        # and both sides carry the same decay weights and the same mask.
        density = events / (slots + self.epsilon)

        # Extent is the positional half of the signal, kept separate and on a
        # log scale. One channel is enough, and it reads the LAST slot
        # counter, the one with the longest seeded window, so it reflects the
        # most evidence available.
        #
        # It saturates at log1p(1/(1 - lambda_max)), and that is intended.
        # A decayed denominator measures how much evidence backs the density
        # estimate, not absolute position: it separates a cold start from a
        # settled window and then stops. Absolute position belongs in the
        # positional encoding. The previous version leaked position in here
        # by accident, and at t = 512 that leak was 98% of the signal.
        extent = keras.ops.log1p(slots[..., -1:])

        return keras.ops.concatenate([density, extent], axis=-1)

    def _global_features(
        self,
        keys: keras.KerasTensor,
        mask: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """
        Exact masked mean over the whole sequence, as ``[B, 1, C + 1]``.

        No decay and no scan: the global mean is a plain reduction. The
        result is kept at length 1 so the transform runs once per sequence
        instead of ``T`` times on identical rows, and broadcasts afterwards.

        :param keys: Masked event probabilities, ``[B, T, C]``, float32.
        :type keys: keras.KerasTensor
        :param mask: Validity indicator, ``[B, T, 1]``, float32.
        :type mask: keras.KerasTensor
        :return: ``[B, 1, count_dim + 1]`` in float32.
        :rtype: keras.KerasTensor
        """
        events = keras.ops.sum(keys, axis=1, keepdims=True)
        slots = keras.ops.sum(mask, axis=1, keepdims=True)
        density = events / (slots + self.epsilon)
        extent = keras.ops.log1p(slots)
        return keras.ops.concatenate([density, extent], axis=-1)

    def call(
        self,
        inputs: keras.KerasTensor,
        mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """
        Forward pass.

        :param inputs: Input tensor, ``(batch, sequence_length, input_dim)``.
        :type inputs: keras.KerasTensor
        :param mask: Boolean validity mask, ``(batch, sequence_length)``.
            Supplied automatically by Keras when an upstream layer produces
            one. ``None`` treats every position as valid.
        :type mask: Optional[keras.KerasTensor]
        :param training: Unused; kept for API compatibility. No sub-layer
            here has training-dependent behaviour.
        :type training: Optional[bool]
        :return: Output tensor, ``(batch, sequence_length, output_dim)``.
        :rtype: keras.KerasTensor
        """
        # Aggregation runs in float32 whatever the compute policy says.
        if mask is None:
            mask_f = keras.ops.ones_like(inputs[..., :1], dtype="float32")
        else:
            mask_f = keras.ops.cast(mask, "float32")
            if len(mask_f.shape) == 2:
                mask_f = keras.ops.expand_dims(mask_f, axis=-1)

        # Zeroing the keys here is what keeps padding out of the aggregate.
        # The previous version had no mask path at all: ten pad tokens after
        # six real ones inflated the global count by 2.76x.
        keys = keras.ops.cast(self.key_projection(inputs), "float32") * mask_f

        if self.counting_scope == "global":
            features = self._global_features(keys, mask_f)
        elif self.counting_scope == "causal":
            features = self._directional_features(keys, mask_f, reverse=False)
        else:
            features = keras.ops.concatenate(
                [
                    self._directional_features(keys, mask_f, reverse=False),
                    self._directional_features(keys, mask_f, reverse=True),
                ],
                axis=-1,
            )

        features = keras.ops.cast(features, self.compute_dtype)
        transformed_counts = self.count_transform(self.count_norm(features))

        # The shortcut is unconditional. For 'global', transformed_counts is
        # [B, 1, O] and broadcasts against the gate, so the Dense above ran
        # once per sequence rather than T times over identical rows.
        shortcut = inputs if self.input_proj is None else self.input_proj(inputs)
        gate_values = self.gate(inputs)

        return gate_values * transformed_counts + (1.0 - gate_values) * shortcut

    def compute_mask(
        self,
        inputs: keras.KerasTensor,
        mask: Optional[keras.KerasTensor] = None,
    ) -> Optional[keras.KerasTensor]:
        """
        Propagate the input mask unchanged.

        The layer is position-preserving, so the set of valid positions is
        the same on the way out as on the way in.

        :param inputs: The input tensor.
        :type inputs: keras.KerasTensor
        :param mask: The incoming mask, or ``None``.
        :type mask: Optional[keras.KerasTensor]
        :return: The same mask.
        :rtype: Optional[keras.KerasTensor]
        """
        return mask

    def verify_mask_invariance(
        self,
        batch: int = 4,
        valid_length: int = 12,
        pad_length: int = 20,
        tolerance: float = 1e-4,
        seed: int = 0,
    ) -> Tuple[bool, float]:
        """
        Check that padding does not change the output at valid positions.

        Runs the layer twice on the same prefix, once alone and once followed
        by padding, and compares the valid region. This is the direct test of
        the bug the previous version had: without masking, appending ten pad
        tokens to six real ones inflated the global count by 2.76x.

        :param batch: Batch size for the probe.
        :type batch: int
        :param valid_length: Number of real positions.
        :type valid_length: int
        :param pad_length: Number of padded positions to append.
        :type pad_length: int
        :param tolerance: Maximum acceptable absolute deviation.
        :type tolerance: float
        :param seed: Seed for the probe tensor.
        :type seed: int
        :return: ``(passed, max_absolute_deviation)``.
        :rtype: Tuple[bool, float]
        :raises ValueError: If the layer is not built.
        """
        if not self.built:
            raise ValueError("Layer must be built before verify_mask_invariance().")

        width = self._input_last_dim
        short = keras.random.normal((batch, valid_length, width), seed=seed)
        junk = keras.random.normal((batch, pad_length, width), seed=seed + 1)
        padded = keras.ops.concatenate([short, junk], axis=1)

        mask_short = keras.ops.ones((batch, valid_length), dtype="bool")
        mask_padded = keras.ops.concatenate(
            [mask_short, keras.ops.zeros((batch, pad_length), dtype="bool")], axis=1
        )

        y_short = self(short, mask=mask_short)
        y_padded = self(padded, mask=mask_padded)[:, :valid_length]

        deviation = float(keras.ops.max(keras.ops.abs(y_short - y_padded)))
        return deviation <= tolerance, deviation

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple.
        :rtype: Tuple[Optional[int], ...]
        """
        return tuple(input_shape[:-1]) + (self.output_dim,)

    def get_config(self) -> Dict[str, Any]:
        """
        Return the layer configuration for serialization.

        :return: Dictionary containing the layer configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "count_dim": self.count_dim,
            "counting_scope": self.counting_scope,
            "activation": serialize_activation(self.activation),
            "use_bias": self.use_bias,
            "gate_bias_init": self.gate_bias_init,
            "decay_min_window": self.decay_min_window,
            "decay_max_window": self.decay_max_window,
            "trainable_decay": self.trainable_decay,
            "epsilon": self.epsilon,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

# ---------------------------------------------------------------------