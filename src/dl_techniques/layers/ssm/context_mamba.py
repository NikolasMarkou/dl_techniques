"""Unidirectional temporal Context Mamba for long-term visual tracking.

Implements the Context Mamba module of MambaLCT (Li et al., 2024): search
frame features ``f_i`` from the first frame to the current one are scanned
along time, and target-change cues aggregate into the context tokens
``c_p`` that condition the appearance encoder of the next frame
(``Y_i^T = C H_i^T`` updating ``c_p``).

The scan itself is :class:`SelectiveSSMLayer`; this layer owns the
temporal framing: flatten ``(B, T, L, D)`` to ``(B, T*L, D)``, prepend the
incoming context tokens (so the causal scan carries their cues forward
into every frame) plus one trailing learnable (ones-initialized) bridge token
``T_i`` (paper Eq. 8:
``H_i^T`` aggregates all history and ``Y_i^T = C H_i^T`` updates ``c_p``),
then split the scan back into enhanced frame features ``(B, T, L, D)``
and the history-aggregated context ``(B, Nc, D)``.

References:
    - Li et al., 2024. MambaLCT: Boosting Tracking via Long-term Context
      State Space Model. (https://arxiv.org/abs/2412.13615)
    - Gu and Dao, 2023. Mamba: Linear-Time Sequence Modeling with Selective
      State Spaces. (https://arxiv.org/abs/2312.00752)
"""

import keras
from typing import Optional, Union, Any, Dict, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.norms.factory import create_normalization_layer
from dl_techniques.utils.keras_registration import register_dl_technique
from .selective_ssm import SelectiveSSMLayer

# ---------------------------------------------------------------------


def _as_pair(first: Any, second: Any) -> Tuple[Any, Any]:
    """Split a Keras nested ``input_shape`` into its two members.

    :param first: Either the full nest or the first member.
    :param second: The second member, or ``None`` when nested.
    :return: ``(features_shape, context_shape)`` pair.
    """
    if second is not None:
        return first, second
    if isinstance(first, (list, tuple)) and len(first) == 2 and isinstance(
        first[0], (list, tuple)
    ):
        return first[0], first[1]
    return first, second


@register_dl_technique("dl_techniques.layers.ssm.context_mamba")
class ContextMambaLayer(keras.layers.Layer):
    """Scan flattened frame tokens plus context tokens with a selective SSM.

    :param d_model: Feature dimensionality. Must be positive.
    :type d_model: int
    :param d_state: SSM latent state size. Defaults to 16.
    :type d_state: int
    :param d_conv: Causal convolution kernel size. Defaults to 4.
    :type d_conv: int
    :param expand: SSM internal expansion factor. Defaults to 2.
    :type expand: int
    :param dt_rank: Step-size projection rank; ``"auto"`` resolves to
        ``ceil(d_model / 16)``. Defaults to ``"auto"``.
    :type dt_rank: Union[str, int]
    :param dt_min: Step-size init lower bound. Defaults to 0.001.
    :type dt_min: float
    :param dt_max: Step-size init upper bound. Defaults to 0.1.
    :type dt_max: float
    :param dt_init: Delta kernel init (``"random"`` or ``"constant"``).
        Defaults to ``"random"``.
    :type dt_init: str
    :param dt_scale: Delta init scale. Defaults to 1.0.
    :type dt_scale: float
    :param dt_init_floor: Lower clip before inverse softplus. Defaults to 1e-4.
    :type dt_init_floor: float
    :param conv_bias: Use bias in the SSM convolution. Defaults to True.
    :type conv_bias: bool
    :param use_bias: Use bias in linear projections. Defaults to False.
    :type use_bias: bool
    :param normalization_type: Norm factory key for the pre-scan norm.
        Defaults to ``"layer_norm"``.
    :type normalization_type: str
    :param norm_epsilon: Epsilon for the pre-scan norm. Defaults to 1e-5.
    :type norm_epsilon: float

    .. note::
        The ``Nc`` context rows of the update are a single-summary
        broadcast: all rows copy the one bridge-token output, so for
        ``Nc > 1`` every row is identical by design. Production use is
        ``Nc == 1`` (paper default).

    Input shape:
        Pair ``(features, context)`` with features
        ``(batch, frames, tokens, d_model)`` and context
        ``(batch, context_len, d_model)``.

    Output shape:
        Pair ``(enhanced, updated_context)`` with enhanced
        ``(batch, frames, tokens, d_model)`` and updated context
        ``(batch, context_len, d_model)``.
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
        normalization_type: str = "layer_norm",
        norm_epsilon: float = 1e-5,
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
        if norm_epsilon <= 0:
            raise ValueError(f"norm_epsilon must be positive, got {norm_epsilon}")

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.dt_rank = dt_rank
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init = dt_init
        self.dt_scale = dt_scale
        self.dt_init_floor = dt_init_floor
        self.conv_bias = conv_bias
        self.use_bias = use_bias
        self.normalization_type = normalization_type
        self.norm_epsilon = norm_epsilon

        self.input_proj = keras.layers.Dense(
            self.d_model, use_bias=self.use_bias, name="input_proj"
        )
        self.norm = create_normalization_layer(
            self.normalization_type, epsilon=self.norm_epsilon, name="norm"
        )
        self.ssm = SelectiveSSMLayer(
            d_model=self.d_model,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
            dt_rank=self.dt_rank,
            dt_min=self.dt_min,
            dt_max=self.dt_max,
            dt_init=self.dt_init,
            dt_scale=self.dt_scale,
            dt_init_floor=self.dt_init_floor,
            conv_bias=self.conv_bias,
            use_bias=self.use_bias,
            name="ssm",
        )

        # Created in `build`: the trailing history-bridge token T_i (paper
        # Eq. 8). It must be LEARNABLE, never zeros: a constant token is
        # annihilated by the pre-scan norm (norm(c) = beta = 0) and the
        # zero input then closes the SSM silu(z) gate, pinning the context
        # update at exactly 0.0 for every input.
        self.bridge_token = None

    def build(self, input_shape: Any) -> None:
        """Build the projection, norm, and SSM sub-layers.

        :param input_shape: Nest ``(features_shape, context_shape)``.
        """
        if self.built:
            return
        feat_shape, _ = _as_pair(input_shape, None)
        if feat_shape is None:
            raise ValueError(
                "ContextMambaLayer.build expects a (features, context) nest"
            )
        self.bridge_token = self.add_weight(
            name="bridge_token",
            shape=(1, 1, self.d_model),
            initializer="ones",
            trainable=True,
        )
        seq_shape = (feat_shape[0], None, self.d_model)
        self.input_proj.build(seq_shape)
        self.norm.build(seq_shape)
        self.ssm.build(seq_shape)
        super().build(input_shape)

    def call(
        self,
        inputs: Any,
        training: Optional[bool] = None,
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Enhance frame tokens and roll history into the context tokens.

        :param inputs: Pair ``(features, context)`` with shapes
            ``(B, T, L, D)`` and ``(B, Nc, D)``.
        :param training: Training mode flag, forwarded to sub-layers.
        :return: Pair ``(enhanced (B, T, L, D), updated_context (B, Nc, D))``.
        """
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError(
                "ContextMambaLayer expects a (features, context) pair"
            )
        features, context = inputs[0], inputs[1]
        batch = keras.ops.shape(features)[0]
        frames = keras.ops.shape(features)[1]
        tokens = keras.ops.shape(features)[2]
        ctx_len = keras.ops.shape(context)[1]

        flat = keras.ops.reshape(features, (batch, frames * tokens, self.d_model))
        bridge = keras.ops.cast(
            self.bridge_token, flat.dtype
        ) + keras.ops.zeros((batch, 1, self.d_model), dtype=flat.dtype)
        seq = keras.ops.concatenate([context, flat, bridge], axis=1)
        seq = self.input_proj(seq, training=training)
        seq = self.norm(seq, training=training)
        scanned = self.ssm(seq, training=training)

        total = frames * tokens
        enhanced_flat = scanned[:, ctx_len:ctx_len + total, :]
        bridge_out = scanned[:, ctx_len + total:, :]
        updated_context = bridge_out + keras.ops.zeros_like(context)
        enhanced = keras.ops.reshape(
            enhanced_flat, (batch, frames, tokens, self.d_model)
        )
        return enhanced, updated_context

    def compute_output_shape(self, input_shape: Any) -> Tuple[Any, Any]:
        """Compute output shapes from stored config.

        :param input_shape: Nest ``(features_shape, context_shape)``.
        :return: Pair ``(enhanced_shape, context_shape)``.
        """
        feat_shape, ctx_shape = _as_pair(input_shape, None)
        batch = feat_shape[0] if feat_shape is not None else None
        frames = feat_shape[1] if feat_shape is not None and len(feat_shape) > 1 else None
        tokens = feat_shape[2] if feat_shape is not None and len(feat_shape) > 2 else None
        enhanced = (batch, frames, tokens, self.d_model)
        if ctx_shape is None:
            updated: Any = (batch, None, self.d_model)
        else:
            updated = (ctx_shape[0], ctx_shape[1], self.d_model)
        return enhanced, updated

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
            "dt_rank": self.dt_rank,
            "dt_min": self.dt_min,
            "dt_max": self.dt_max,
            "dt_init": self.dt_init,
            "dt_scale": self.dt_scale,
            "dt_init_floor": self.dt_init_floor,
            "conv_bias": self.conv_bias,
            "use_bias": self.use_bias,
            "normalization_type": self.normalization_type,
            "norm_epsilon": self.norm_epsilon,
        })
        return config

# ---------------------------------------------------------------------
