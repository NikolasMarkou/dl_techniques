"""
TransformerLayer, a configurable transformer block, plus the two helper
functions that build its attention and FFN sub-layer configs.

TransformerLayer wires a self-attention layer and a feed-forward layer around
residual connections, with the normalization position (pre or post) and every
sub-component's type chosen at construction time through factory functions
instead of hard-coded classes. One class this way covers many attention
variants (multi-head, windowed, differential, group-query, and more), many
FFN variants (MLP, SwiGLU, mixture-of-experts, and more), and every
normalization type the norm factory supports.

Dropout is applied once, to the FFN branch only, never after attention.
Attention-internal dropout is a separate ``attention_dropout_rate`` argument
forwarded into the attention sub-layer itself. Setting ``moe_config`` replaces
the FFN with a Mixture-of-Experts layer; ``intermediate_size`` still supplies
the fallback expert hidden size for six FFN types even then.

References:
    - Vaswani et al., 2017. Attention Is All You Need. (https://arxiv.org/abs/1706.03762)
    - Ba et al., 2016. Layer Normalization. (https://arxiv.org/abs/1607.06450)
    - Xiong et al., 2020. On Layer Normalization in the Transformer
      Architecture. (https://arxiv.org/abs/2002.04745)
"""

import keras
import warnings
from keras import layers, initializers, regularizers
from typing import Optional, Union, Any, Dict, Tuple, Literal, Callable

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..moe import MixtureOfExperts, MoEConfig
from ..stochastic_depth import StochasticDepth
from ..layer_scale import LayerScale
from ..ffn import assemble_ffn_config, create_ffn_from_config, FFNType
from ..ffn.factory import FFN_REGISTRY
from ..attention import create_attention_layer, AttentionType
from ..attention.factory import ATTENTION_REGISTRY
from ..norms import create_normalization_layer, NormalizationType
from ...initializers import clone_initializer
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------
# Type definitions for enhanced type safety
# ---------------------------------------------------------------------

NormalizationPositionType = Literal['post', 'pre']


# ---------------------------------------------------------------------
# The transformer family's one FFN parameter-injection policy
# ---------------------------------------------------------------------

#: FFN types whose defining feature is a fixed nonlinearity or gate, so the
#: wrapper's single generic ``activation`` is never forwarded even though the
#: registry accepts an ``activation`` key for some of them.
#:
#: * ``squared_relu`` -- fixed ``relu(x) ** 2``; the registry has no
#:   ``activation`` param at all, so this entry is documentation.
#: * ``reglu`` / ``bilinear`` -- ``GLUFFN`` aliases whose identity is the
#:   fixed relu / linear gate. They do accept ``activation``, so withholding
#:   it must be explicit. (D-005)
_FFN_TYPES_WITH_FIXED_ACTIVATION: Tuple[str, ...] = (
    'squared_relu', 'reglu', 'bilinear',
)


def build_transformer_ffn_config(
        *,
        ffn_type: str,
        name: str,
        hidden_size: int,
        intermediate_size: int,
        activation: Union[str, Callable],
        dropout_rate: float,
        kernel_initializer: Any,
        bias_initializer: Any,
        use_bias: bool,
        output_kernel_initializer: Any = None,
        ffn_args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the FFN factory config for a transformer encoder/decoder block.

    Shared by :meth:`TransformerLayer._get_ffn_config` and
    :meth:`TransformerDecoderLayer._get_ffn_config`, so both blocks produce the
    identical config dict for every registry type.

    Emits the block's generic conveniences (dims derived from
    ``hidden_size``/``intermediate_size``, plus ``activation``,
    ``dropout_rate`` and the initializers), applies the three per-type policy
    adjustments below, then hands the result to
    :func:`~dl_techniques.layers.ffn.factory.assemble_ffn_config`, which
    intersects it with what ``ffn_type`` actually accepts and merges
    ``ffn_args`` on top unfiltered. Returns a config carrying ``type`` and
    ``name``, ready for ``create_ffn_from_config``; raises ``ValueError`` for
    an unregistered ``ffn_type``.

    Per-type policy adjustments (the only branching this function does, since
    the registry intersection above cannot express them):

    1. ``swiglu`` sizes itself (a 2/3 rule from ``ffn_expansion_factor``,
       rounded to ``ffn_multiple_of``) and treats ``hidden_dim`` as optional,
       so the block's ``intermediate_size`` is withheld rather than silently
       overriding that derivation; the two expansion knobs are supplied
       instead.
    2. ``differential`` renames ``activation`` to ``branch_activation``
       (``DifferentialFFN``'s own parameter name, D-016); ``gate_activation``
       is never forwarded, since the sigmoid gate is the layer's fixed
       feature.
    3. ``_FFN_TYPES_WITH_FIXED_ACTIVATION`` withholds ``activation`` (D-005).
    4. ``swiglu`` also withholds ``use_bias`` (D-006, see the branch below).

    # DECISION plan-2026-07-30T140922-8af1028f/D-018: this table stays here,
    # never re-inlined into either caller. Two hand-maintained copies once
    # produced a silent activation drop plus 5 decoder-only coverage gaps. See decisions.md.

    :param ffn_type: An ``FFN_REGISTRY`` key.
    :type ffn_type: str
    :param name: Name for the FFN layer.
    :type name: str
    :param hidden_size: The block's model width; the FFN's output width.
    :type hidden_size: int
    :param intermediate_size: The FFN's inner width.
    :type intermediate_size: int
    :param activation: The block's generic activation.
    :type activation: Union[str, Callable]
    :param dropout_rate: The block's dropout rate.
    :type dropout_rate: float
    :param kernel_initializer: The block's kernel initializer.
    :type kernel_initializer: Any
    :param output_kernel_initializer: Optional initializer for the FFN's
        output/contracting projection alone (the residual-path projection).
        Emitted into the wrapper config only when not ``None``, so it is
        subject to the same registry intersection as every other wrapper
        convenience — only ``'mlp'`` declares it. The caller (``TransformerLayer``)
        is responsible for rejecting the combination earlier if a silent drop
        would be wrong.
    :type output_kernel_initializer: Any
    :param bias_initializer: The block's bias initializer.
    :type bias_initializer: Any
    :param use_bias: The block's bias switch. Forwarded to every registry type
        that declares a ``use_bias`` key except ``swiglu`` (policy 4); types
        that declare none (``kan``, ``tversky``) drop it in the pre-filter.
    :type use_bias: bool
    :param ffn_args: The caller's explicit FFN args; merged last and never
        filtered, so a caller key the type does not accept still reaches
        ``create_ffn_layer``.
    :type ffn_args: Optional[Dict[str, Any]]
    :return: Config dict for ``create_ffn_from_config``.
    :rtype: Dict[str, Any]
    :raises ValueError: If ``ffn_type`` is not a registered FFN type.
    """
    config: Dict[str, Any] = {
        'type': ffn_type,
        'name': name,
        'dropout_rate': dropout_rate,
        'kernel_initializer': kernel_initializer,
        'bias_initializer': bias_initializer,
        'hidden_dim': intermediate_size,
        'output_dim': hidden_size,
        'activation': activation,
        'use_bias': use_bias,
    }

    if output_kernel_initializer is not None:
        config['output_kernel_initializer'] = output_kernel_initializer

    if ffn_type == 'swiglu':
        del config['hidden_dim']
        del config['activation']
        # DECISION plan-2026-08-19T070627-a616f581/D-006: never forward the block's
        # `use_bias` here; swiglu defaults it False and forwarding True adds 3 bias tensors to every consumer's swiglu blocks.
        # A caller wanting biased swiglu passes `ffn_args={'use_bias': True}`. See decisions.md.
        del config['use_bias']
        config['ffn_expansion_factor'] = 4
        config['ffn_multiple_of'] = 256
    elif ffn_type == 'differential':
        config['branch_activation'] = config.pop('activation')
    elif ffn_type in _FFN_TYPES_WITH_FIXED_ACTIVATION:
        del config['activation']

    return assemble_ffn_config(ffn_type, config, ffn_args)


# ---------------------------------------------------------------------

#: Default ``window_size`` for ``attention_type='window'``. Read by both
#: :meth:`TransformerLayer.__init__` and
#: :func:`build_transformer_attention_required_params`, as the fallback for a
#: block with no dedicated ``window_size`` constructor parameter (``TransformerDecoderLayer``).
_DEFAULT_ATTENTION_WINDOW_SIZE: int = 8

#: Default ``lambda_init`` for ``attention_type='differential'``. Same two
#: readers as ``_DEFAULT_ATTENTION_WINDOW_SIZE`` above.
_DEFAULT_ATTENTION_LAMBDA_INIT: float = 0.8

#: MoE expert FFN types whose ``hidden_dim`` falls back to
#: ``TransformerLayer.intermediate_size`` when the caller's
#: ``moe_config.expert_config.ffn_config`` omits it. Read by both the fallback and the warning text below, so they cannot drift apart.
_MOE_EXPERT_TYPES_USING_INTERMEDIATE_SIZE: Tuple[str, ...] = (
    'mlp', 'differential', 'glu', 'geglu', 'residual', 'swin_mlp',
)


def build_transformer_attention_required_params(
        *,
        attention_type: str,
        hidden_size: int,
        num_heads: int,
        window_size: Union[int, Tuple[int, int]] = _DEFAULT_ATTENTION_WINDOW_SIZE,
        num_kv_heads: Optional[int] = None,
        lambda_init: float = _DEFAULT_ATTENTION_LAMBDA_INIT,
) -> Dict[str, Any]:
    """The params the attention factory requires beyond ``dim``/``num_heads``.

    Shared by :meth:`TransformerLayer._get_attention_params` and
    :meth:`TransformerDecoderLayer._self_attention_params`, so both blocks
    agree on every type.

    Returns only the type-specific keys that
    ``dl_techniques.layers.attention.factory`` lists as required and not
    already covered by ``dim``/``num_heads`` — never the generic conveniences
    (``dropout_rate``, ``use_bias``, initializers), which the two blocks
    legitimately differ on. Returns an empty dict for every type with no such
    extra requirement (``multi_head``, ``multi_head_cross``, ``anchor``,
    ``lighthouse``, ``fnet``) and for an unknown type — validating the type is
    the factory's job, not this function's. Never raises.

    The four default values here are the encoder's, verbatim, so a block
    without a dedicated constructor parameter (the decoder has none of
    ``window_size`` / ``n_kv_head`` / ``lambda_init``) gets the same answer
    the encoder would give. A caller's ``attention_args`` still overrides
    everything, since both callers merge it last.

    # DECISION plan-2026-07-31T132403-b3f540cb/D-015: this table stays here,
    # never re-inlined into either caller — a decoder-side copy once made 4 types unconstructable there. See decisions.md.

    :param attention_type: An ``ATTENTION_REGISTRY`` key.
    :type attention_type: str
    :param hidden_size: The block's model width (the factory's ``dim``).
    :type hidden_size: int
    :param num_heads: The block's head count.
    :type num_heads: int
    :param window_size: ``'window'``/``'window_zigzag'``/``'window_band'``/
        ``'beit'`` only, and read differently by each: a scalar spatial edge
        length ``W`` for ``'window'``/``'window_zigzag'`` (windows of
        ``W*W`` tokens); a 1-D half-width in tokens for ``'window_band'``
        (query ``i`` attends key ``j`` iff ``abs(i - j) <= window_size``, no
        grid); the ``(Wh, Ww)`` patch grid for ``'beit'`` (an ``int`` meaning
        the square grid ``(W, W)``, sequence length must be ``Wh*Ww + 1``
        including the cls token).
    :type window_size: Union[int, Tuple[int, int]]
    :param num_kv_heads: ``'group_query'`` only. ``None`` means ``num_heads``
        (i.e. degrade to plain MHA), matching ``TransformerLayer.n_kv_head``.
    :type num_kv_heads: Optional[int]
    :param lambda_init: ``'differential'`` only. Initial lambda.
    :type lambda_init: float
    :return: The type-specific required params; possibly empty.
    :rtype: Dict[str, Any]
    """
    if attention_type in ('window', 'window_zigzag', 'window_band', 'beit'):
        # One table entry for all four; see this function's window_size
        # docstring entry for how each type reads the value differently.
        return {'window_size': window_size}
    if attention_type == 'group_query':
        return {'num_kv_heads': num_kv_heads if num_kv_heads is not None else num_heads}
    if attention_type == 'differential':
        return {'head_dim': hidden_size // num_heads, 'lambda_init': lambda_init}
    if attention_type == 'multi_head_latent':
        # MLA requires kv_latent_dim; neither block has a dedicated ctor
        # param for it, so this documented default is the only source.
        return {'kv_latent_dim': max(1, hidden_size // 4)}
    return {}


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.transformers.transformer")
class TransformerLayer(keras.layers.Layer):
    """
    Generic transformer layer with configurable attention, FFN, and normalization.

    Implements a standard transformer block: multi-head self-attention
    followed by a position-wise feed-forward network, each wrapped in a
    residual connection and normalization. The data flow is determined by
    ``normalization_position`` (pre or post). Every sub-component (attention,
    FFN, normalization) is built through a factory function, so swapping
    architectures is a constructor argument, not a subclass.

    ``Attention(Q, K, V) = softmax((Q K^T) / sqrt(d_k)) V``

    Architecture:

    .. code-block:: text

        ┌───────────────────────────────────────┐
        │  Input (B, seq_len, hidden_size)      │
        └───────────────────┬───────────────────┘
                            ▼
        ┌───────────────────────────────────────┐
        │  [Norm] ─► Attention                  │
        │  ─► [StochasticDepth] ─► [LayerScale] │
        │  ─► + Residual                        │
        └───────────────────┬───────────────────┘
                            ▼
        ┌───────────────────────────────────────┐
        │  [Norm] ─► FFN/MoE ─► Dropout         │
        │  ─► [StochasticDepth] ─► [LayerScale] │
        │  ─► + Residual                        │
        └───────────────────┬───────────────────┘
                            ▼
        ┌───────────────────────────────────────┐
        │  Output (B, seq_len, hidden_size)     │
        └───────────────────────────────────────┘

    :param hidden_size: Hidden dimension of the layer.
    :type hidden_size: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param intermediate_size: FFN intermediate dimension. Not ignored when
        ``moe_config`` is provided: it is then used as the fallback for the
        expert FFN's ``hidden_dim`` whenever
        ``moe_config.expert_config.ffn_config`` omits that key and the expert
        type is one of ``{'mlp', 'differential', 'glu', 'geglu', 'residual',
        'swin_mlp'}``. It is only genuinely unused when ``moe_config`` is set
        and the expert config already carries its own ``hidden_dim`` (or the
        expert type is not one of those six).
    :type intermediate_size: int
    :param attention_type: Attention mechanism type. Default: ``'multi_head'``.

        The ``AttentionType`` annotation is wider than what this layer
        implements. It is the full 33-key ``ATTENTION_REGISTRY`` literal, but
        ``_build_attention`` handles ten of them and its ``else`` raises
        ``ValueError: Unknown attention type``. Measured 2026-08-27, one
        ``TransformerLayer`` per registry key on a ``(2, 16, 32)`` input:

        * usable (10): ``anchor``, ``differential``, ``fnet``,
          ``group_query``, ``lighthouse``, ``multi_head``, ``multi_head_latent``,
          ``window``, ``window_band``, ``window_zigzag``.
        * raise ``Unknown attention type`` (22): every other key. Some are
          inapplicable here — ``cbam``/``channel``/``spatial`` are 4-D
          convolutional attentions and ``multi_head_cross`` is cross-attention
          — but ``linear``, ``performer``, ``gated``, ``ring``, ``rpc``,
          ``hopfield``, ``energy``, ``single_window`` and ``wave_field`` are
          drop-in self-attention mechanisms with no branch here.
        * ``beit`` constructs but raises on sequence length unless
          ``seq_len == Wh*Ww + 1``; that is its own documented constraint, not
          a missing branch.

        ``attention_type='anchor'`` runs in standard self-attention mode
        here. ``AnchorAttention``'s hierarchical bottleneck is selected by
        ``num_anchor_tokens``, a ``call()`` argument this block does not
        forward; ``None`` means standard attention by that layer's own
        contract, and passing it through ``attention_args`` raises since it
        is not a constructor parameter.
    :type attention_type: AttentionType
    :param attention_args: Custom arguments forwarded to the attention factory.
    :type attention_args: Optional[Dict[str, Any]]
    :param normalization_type: Normalization type. Default: ``'layer_norm'``.
    :type normalization_type: NormalizationType
    :param normalization_position: ``'pre'`` or ``'post'`` normalization.
    :type normalization_position: NormalizationPositionType
    :param attention_norm_args: Custom arguments for the attention norm layer.
    :type attention_norm_args: Optional[Dict[str, Any]]
    :param ffn_norm_args: Custom arguments for the FFN norm layer.
    :type ffn_norm_args: Optional[Dict[str, Any]]
    :param ffn_type: FFN architecture type. Default: ``'mlp'``.
    :type ffn_type: FFNType
    :param ffn_args: Custom arguments for the FFN factory. These are the
        caller's explicit keys, merged last, after this layer's own generic
        conveniences have been intersected with what ``ffn_type`` accepts.
        They are never pre-filtered, so they always reach
        ``create_ffn_layer`` — including a misdirected or misspelled one,
        which the factory reports rather than this layer swallowing it.
    :type ffn_args: Optional[Dict[str, Any]]
    :param moe_config: Mixture-of-Experts configuration replacing the FFN.
    :type moe_config: Optional[Union[MoEConfig, Dict[str, Any]]]
    :param dropout_rate: FFN output dropout rate. This is the only dropout
        this layer applies itself, applied to the FFN sub-block only — never
        after attention (see the architecture note above). Default: 0.1.
    :type dropout_rate: float
    :param attention_dropout_rate: Attention-internal (weight) dropout rate,
        not an output dropout: forwarded verbatim as the attention
        sub-layer's own ``dropout_rate`` constructor argument (e.g.
        ``MultiHeadAttention(dropout_rate=...)``), acting on the attention
        probabilities inside that layer. Default: 0.1.
    :type attention_dropout_rate: float
    :param use_stochastic_depth: Enable stochastic depth. Default: False.
    :type use_stochastic_depth: bool
    :param stochastic_depth_rate: Drop-path rate for stochastic depth.
    :type stochastic_depth_rate: float
    :param activation: Activation function for the FFN. Default: ``'gelu'``.
    :type activation: Union[str, Callable]
    :param use_bias: Whether linear layers use bias. Default: True.
    :type use_bias: bool
    :param kernel_initializer: Kernel weight initializer.
    :type kernel_initializer: Union[str, initializers.Initializer]
    :param residual_output_kernel_initializer: Optional initializer applied
        only to the block's two residual-path output projections — the
        attention output projection and the FFN's contracting projection —
        leaving Q/K/V and the FFN expansion on ``kernel_initializer``.
        ``None`` (the default) means every projection keeps
        ``kernel_initializer``. It exists for GPT-2's residual-init rule
        (std scaled by ``1/sqrt(2 * n_layer)``, HF
        ``modeling_gpt2.py::_init_weights``); accepted only when
        ``attention_type`` is ``'multi_head'``/``'multi_head_cross'`` and
        ``ffn_type == 'mlp'`` — any other combination raises from the
        respective factory rather than ignoring it.
    :type residual_output_kernel_initializer: Optional[Union[str, initializers.Initializer]]
    :param bias_initializer: Bias weight initializer.
    :type bias_initializer: Union[str, initializers.Initializer]
    :param kernel_regularizer: Kernel weight regularizer.
    :type kernel_regularizer: Optional[regularizers.Regularizer]
    :param bias_regularizer: Bias weight regularizer.
    :type bias_regularizer: Optional[regularizers.Regularizer]
    :param window_size: Window size for ``attention_type='window'`` (the spatial
        window edge length) and for ``attention_type='beit'`` (the ``(Wh, Ww)``
        patch grid, an ``int`` meaning the square grid ``(W, W)``; the block's
        input sequence must then be exactly ``Wh*Ww + 1`` tokens long, cls
        included). Ignored by every other attention type. Default: 8.
    :type window_size: Union[int, Tuple[int, int]]
    :param n_kv_head: Number of key/value heads for grouped-query attention.
    :type n_kv_head: Optional[int]
    :param lambda_init: Initial lambda for differential attention.
    :type lambda_init: float
    :param kwargs: Additional keyword arguments for the base Layer.
    :type kwargs: Any

    :raises ValueError: If dimension parameters are invalid or sub-layer
        creation fails due to incompatible parameters.
    """

    # DECISION plan_2026-06-12_0bb1729b/D-001: these 3 types' `call` signatures
    # reject an `attention_mask` argument; call() must invoke them without one. See decisions.md.
    # DECISION plan-2026-07-31T132403-b3f540cb/D-016: never add 'window' here;
    # it genuinely honours a causal mask and adding it would make masking silently non-causal. See decisions.md.
    _MASKLESS_ATTENTION_TYPES = frozenset({'fnet', 'anchor', 'lighthouse'})

    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            intermediate_size: int,
            attention_type: AttentionType = 'multi_head',
            attention_args: Optional[Dict[str, Any]] = None,
            normalization_type: NormalizationType = 'layer_norm',
            normalization_position: NormalizationPositionType = 'post',
            attention_norm_args: Optional[Dict[str, Any]] = None,
            ffn_norm_args: Optional[Dict[str, Any]] = None,
            ffn_type: FFNType = 'mlp',
            ffn_args: Optional[Dict[str, Any]] = None,
            moe_config: Optional[Union[MoEConfig, Dict[str, Any]]] = None,
            dropout_rate: float = 0.1,
            attention_dropout_rate: float = 0.1,
            use_stochastic_depth: bool = False,
            stochastic_depth_rate: float = 0.1,
            activation: Union[str, Callable] = 'gelu',
            use_bias: bool = True,
            kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
            residual_output_kernel_initializer: Optional[Union[str, initializers.Initializer]] = None,
            bias_initializer: Union[str, initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            bias_regularizer: Optional[regularizers.Regularizer] = None,
            window_size: Union[int, Tuple[int, int]] = _DEFAULT_ATTENTION_WINDOW_SIZE,
            n_kv_head: Optional[int] = None,
            lambda_init: float = _DEFAULT_ATTENTION_LAMBDA_INIT,
            use_layer_scale: bool = False,
            layer_scale_init_value: float = 1e-5,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
            )
        if intermediate_size <= 0 and moe_config is None:
            raise ValueError(
                f"intermediate_size must be positive when moe_config is None, got {intermediate_size}"
            )
        # call() dispatches on `== 'pre'` with an unguarded else for post-norm,
        # so a typo like 'Pre' would silently run the wrong branch. Same check as `TransformerDecoderLayer.__init__`.
        if normalization_position not in ('pre', 'post'):
            raise ValueError(
                f"normalization_position must be 'pre' or 'post', got {normalization_position}"
            )

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.attention_type = attention_type
        self.attention_args = attention_args or {}
        self.normalization_type = normalization_type
        self.normalization_position = normalization_position
        self.attention_norm_args = attention_norm_args or {}
        self.ffn_norm_args = ffn_norm_args or {}
        self.ffn_type = ffn_type
        self.ffn_args = ffn_args or {}
        self.moe_config = moe_config
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.use_stochastic_depth = use_stochastic_depth
        self.stochastic_depth_rate = stochastic_depth_rate
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.residual_output_kernel_initializer = (
            initializers.get(residual_output_kernel_initializer)
            if residual_output_kernel_initializer is not None else None
        )
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.window_size = window_size
        self.n_kv_head = n_kv_head if n_kv_head is not None else num_heads
        self.lambda_init = lambda_init
        self.use_layer_scale = bool(use_layer_scale)
        self.layer_scale_init_value = float(layer_scale_init_value)

        if self.residual_output_kernel_initializer is not None:
            self._reject_unsupported_residual_output_init()

        # Convert dict to MoEConfig if needed
        if isinstance(self.moe_config, dict):
            self.moe_config = MoEConfig.from_dict(self.moe_config)

        if self.moe_config is not None:
            if self.ffn_type != 'mlp' or self.ffn_args:
                warnings.warn(
                    "moe_config is provided, so the `ffn_type` and `ffn_args` parameters "
                    "of TransformerLayer are ignored. The FFN will be a MixtureOfExperts "
                    "layer configured by `moe_config`. NOTE: `intermediate_size` is NOT "
                    "ignored -- it is still used as the fallback for the expert FFN's "
                    "`hidden_dim` when `moe_config.expert_config.ffn_config` omits it and "
                    "the expert type is one of "
                    f"{set(_MOE_EXPERT_TYPES_USING_INTERMEDIATE_SIZE)}."
                )

            ffn_config = self.moe_config.expert_config.ffn_config

            # Ensure expert output_dim matches transformer's hidden_size
            if 'output_dim' in ffn_config and ffn_config['output_dim'] != self.hidden_size:
                warnings.warn(
                    f"Adjusting moe_config.expert_config.ffn_config['output_dim'] from "
                    f"{ffn_config['output_dim']} to {self.hidden_size} "
                    f"to match TransformerLayer's hidden_size for consistency."
                )
                ffn_config['output_dim'] = self.hidden_size
            elif 'output_dim' not in ffn_config:
                ffn_config['output_dim'] = self.hidden_size

            # This block runs unconditionally, not gated by the warning's own
            # `if` above; the type list lives in one place, `_MOE_EXPERT_TYPES_USING_INTERMEDIATE_SIZE`, which the warning text also reads.
            ffn_type = ffn_config.get('type')
            if ffn_type in _MOE_EXPERT_TYPES_USING_INTERMEDIATE_SIZE:
                if 'hidden_dim' not in ffn_config:
                    ffn_config['hidden_dim'] = self.intermediate_size

        # Per Keras best practices, all sub-layers are created in __init__.
        # They will be built with their weights in the build() method.

        # Normalization layers
        self.attention_norm = self._create_normalization_layer('attention_norm', 'attention')
        self.output_norm = self._create_normalization_layer('output_norm', 'ffn')

        # Attention layer
        self.attention = self._create_attention_layer('attention')

        # Feed-forward network (or MoE)
        self.ffn_layer = self._create_ffn_layer('ffn')

        # Dropout layers
        self.dropout = layers.Dropout(self.dropout_rate, name='dropout')

        # Stochastic depth layers (if enabled)
        self.attention_stochastic_depth = None
        self.ffn_stochastic_depth = None
        if self.use_stochastic_depth:
            self.attention_stochastic_depth = StochasticDepth(
                drop_path_rate=self.stochastic_depth_rate,
                name='attention_stochastic_depth'
            )
            self.ffn_stochastic_depth = StochasticDepth(
                drop_path_rate=self.stochastic_depth_rate,
                name='ffn_stochastic_depth'
            )

        # LayerScale (CaiT, Touvron et al. 2021): a small per-channel scale on
        # each residual branch output, keeping a deep pre-norm stack's activation std from growing unbounded (12-layer ViT without it: 2 -> 23).
        self.attention_layer_scale = None
        self.ffn_layer_scale = None
        if self.use_layer_scale:
            self.attention_layer_scale = LayerScale(
                multiplier_type='CHANNEL',
                initializer=keras.initializers.Constant(self.layer_scale_init_value),
                constraint=None,
                name='attention_layer_scale',
            )
            self.ffn_layer_scale = LayerScale(
                multiplier_type='CHANNEL',
                initializer=keras.initializers.Constant(self.layer_scale_init_value),
                constraint=None,
                name='ffn_layer_scale',
            )

    def _create_normalization_layer(self, name: str, layer_type: str = 'attention') -> keras.layers.Layer:
        """Create a normalization layer using the component factory.

        :param name: Name for the layer.
        :type name: str
        :param layer_type: ``'attention'`` or ``'ffn'`` to select custom args.
        :type layer_type: str
        :return: An unbuilt normalization layer instance.
        :rtype: keras.layers.Layer
        :raises ValueError: If layer creation fails.
        """
        custom_args = self.attention_norm_args if layer_type == 'attention' else self.ffn_norm_args
        try:
            return create_normalization_layer(
                normalization_type=self.normalization_type,
                name=name,
                **custom_args
            )
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Failed to create {self.normalization_type} normalization layer for {layer_type}. "
                f"Check parameter compatibility. Custom args: {list(custom_args.keys())}. "
                f"Original error: {e}"
            )

    def _required_attention_params(self) -> Dict[str, Any]:
        """This block's type-specific required attention params.

        Thin binding of :func:`build_transformer_attention_required_params` to
        this layer's own constructor parameters. It exists so that
        ``TransformerDecoderLayer`` — which has no ``window_size`` /
        ``n_kv_head`` / ``lambda_init`` constructor parameters — reads the same
        table from the same function rather than carrying a second copy (D-015;
        the second copy being absent is exactly what F-07 was).
        """
        return build_transformer_attention_required_params(
            attention_type=self.attention_type,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            window_size=self.window_size,
            num_kv_heads=self.n_kv_head,
            lambda_init=self.lambda_init,
        )

    def _get_attention_params(self, name: str) -> Dict[str, Any]:
        """Consolidate parameters for attention layer creation.

        :param name: Name for the attention layer.
        :type name: str
        :return: Parameter dictionary for the attention factory.
        :rtype: Dict[str, Any]
        """
        if self.attention_type == 'multi_head':
            default_params = {
                'dim': self.hidden_size,
                'num_heads': self.num_heads,
                'dropout_rate': self.attention_dropout_rate,
                'use_bias': self.use_bias,
                'kernel_initializer': self.kernel_initializer,
                'name': name
            }
        elif self.attention_type in ('window', 'window_zigzag', 'window_band'):
            # All three keys wrap the same class, WindowAttention, differing
            # only in partition_mode; window_size's meaning is resolved above.
            #
            # DECISION plan-2026-08-19T070627-a616f581/D-005: forward `use_bias`
            # spelled `qkv_bias`/`proj_bias`, never `use_bias` -- that spelling raises, and before this branch existed ModernBERT loaded 2 stray bias tensors per layer. See decisions.md.
            default_params = {
                'dim': self.hidden_size,
                'num_heads': self.num_heads,
                **self._required_attention_params(),
                'dropout_rate': self.attention_dropout_rate,
                'qkv_bias': self.use_bias,
                'proj_bias': self.use_bias,
                'name': name
            }
        elif self.attention_type == 'beit':
            # Not a copy of the 'window' branch: BeitAttention declares
            # `attn_dropout_rate`/`proj_dropout_rate`, not `dropout_rate`.
            # `proj_dropout_rate` is left at the layer default so it does not double up with self.dropout on the residual branch.
            default_params = {
                'dim': self.hidden_size,
                'num_heads': self.num_heads,
                **self._required_attention_params(),
                'attn_dropout_rate': self.attention_dropout_rate,
                'name': name
            }
        elif self.attention_type == 'group_query':
            default_params = {
                'dim': self.hidden_size,
                'num_heads': self.num_heads,
                **self._required_attention_params(),
                'dropout_rate': self.attention_dropout_rate,
                'use_bias': self.use_bias,
                'name': name
            }
        elif self.attention_type == 'differential':
            default_params = {
                'dim': self.hidden_size,
                'num_heads': self.num_heads,
                **self._required_attention_params(),
                'dropout_rate': self.attention_dropout_rate,
                'name': name
            }
        # DECISION plan_2026-06-12_0bb1729b/D-001: branches below were added
        # additively, keeping the 4 branches above byte-identical. See decisions.md.
        elif self.attention_type == 'multi_head_latent':
            # MLA requires kv_latent_dim; TransformerLayer has no dedicated
            # ctor param for it, so a documented default is used and any
            # user-supplied value in attention_args overrides via the merge below.
            default_params = {
                'dim': self.hidden_size,
                'num_heads': self.num_heads,
                **self._required_attention_params(),
                'name': name
            }
        elif self.attention_type == 'anchor':
            default_params = {
                'dim': self.hidden_size,
                'num_heads': self.num_heads,
                'dropout_rate': self.attention_dropout_rate,
                'use_bias': self.use_bias,
                'name': name
            }
        elif self.attention_type == 'lighthouse':
            default_params = {
                'dim': self.hidden_size,
                'num_heads': self.num_heads,
                'name': name
            }
        elif self.attention_type == 'fnet':
            # FNetFourierTransform is parameter-free (no dim/num_heads); it
            # mixes tokens via a 2D DFT. Maskless — see _MASKLESS_ATTENTION_TYPES.
            default_params = {
                'name': name
            }
        else:
            raise ValueError(f"Unknown attention type: {self.attention_type}")

        # DECISION plan-2026-08-22T035419-a11304c8/D-160: merged here, at
        # attention_args's own precedence, never into each branch -- only 2 of 9 types declare output_kernel_initializer, so an unsupported type raises loudly instead of silently dropping the request. See decisions.md.
        params = {**default_params, **self.attention_args}

        # DECISION plan-2026-08-23T091307-9a110062/D-600: forward kernel_initializer
        # here once, gated on the type's own registry declaration, not per-branch -- 8 of 9 types declare it but only 'multi_head' forwarded it, silently falling back to glorot_uniform. clone_initializer avoids replaying one shared draw across blocks. See decisions.md.
        if 'kernel_initializer' in ATTENTION_REGISTRY[self.attention_type].get(
            'optional_params', {}
        ):
            params.setdefault(
                'kernel_initializer', clone_initializer(self.kernel_initializer)
            )

        if self.residual_output_kernel_initializer is not None:
            params.setdefault(
                'output_kernel_initializer',
                self.residual_output_kernel_initializer,
            )
        return params

    def _create_attention_layer(self, name: str) -> keras.layers.Layer:
        """Create an attention layer using the component factory.

        :param name: Name for the attention layer.
        :type name: str
        :return: An unbuilt attention layer instance.
        :rtype: keras.layers.Layer
        :raises ValueError: If creation fails due to invalid parameters.
        """
        params = self._get_attention_params(name)
        try:
            return create_attention_layer(
                attention_type=self.attention_type,
                **params
            )
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Failed to create {self.attention_type} layer. "
                f"Check for parameter incompatibility. Custom args: {list(self.attention_args.keys())}. "
                f"Original error: {e}"
            )

    def _reject_unsupported_residual_output_init(self) -> None:
        """Raise unless both sub-layer types can honour a residual-only init.

        The FFN side goes through ``assemble_ffn_config``, which intersects
        the wrapper's config with the target type's declared params and
        drops the remainder silently — so an unsupported ``ffn_type`` would
        otherwise build an unscaled residual projection with no error.

        # DECISION plan-2026-08-22T035419-a11304c8/D-160: reject the combination
        # here, naming this class's own parameter, rather than let the FFN factory silently drop it. See decisions.md.

        :raises ValueError: If ``ffn_type`` or ``attention_type`` does not
            declare ``output_kernel_initializer``.
        """
        supported_ffn = sorted(
            t for t, info in FFN_REGISTRY.items()
            if 'output_kernel_initializer' in info['optional_params']
        )
        supported_attention = sorted(
            t for t, info in ATTENTION_REGISTRY.items()
            if 'output_kernel_initializer' in info.get('optional_params', {})
        )
        if self.moe_config is None and self.ffn_type not in supported_ffn:
            raise ValueError(
                f"residual_output_kernel_initializer was supplied but "
                f"ffn_type='{self.ffn_type}' has no separable output "
                f"projection to apply it to. Supported ffn_type values: "
                f"{supported_ffn}. (Passing it anyway would be silently "
                f"dropped by the FFN factory's registry intersection, leaving "
                f"the residual projection unscaled.)"
            )
        if self.attention_type not in supported_attention:
            raise ValueError(
                f"residual_output_kernel_initializer was supplied but "
                f"attention_type='{self.attention_type}' does not expose its "
                f"output projection's initializer. Supported attention_type "
                f"values: {supported_attention}."
            )
        if self.moe_config is not None:
            raise ValueError(
                "residual_output_kernel_initializer is not supported together "
                "with moe_config: the FFN is a MixtureOfExperts layer whose "
                "expert output projections are configured through "
                "moe_config.expert_config.ffn_config, not through this block."
            )

    def _get_ffn_config(self, name: str) -> Dict[str, Any]:
        """Consolidate configuration for FFN layer creation.

        :param name: Name for the FFN layer.
        :type name: str
        :return: Parameter dictionary for the FFN factory.
        :rtype: Dict[str, Any]
        """
        # DECISION plan-2026-08-19T163559-499b6f0e/D-070: the FFN gets a clone
        # of the block's initializer, attention keeps the stored instance -- a shared instance drew bit-identical attention-output and FFN-expand kernels. See decisions.md.
        # DECISION plan-2026-08-22T035419-a11304c8/D-160: output_kernel_initializer
        # rides the wrapper channel, never ffn_args -- injecting it there was measured to make the factory blame the caller for a model-chosen key. See decisions.md.
        return build_transformer_ffn_config(
            ffn_type=self.ffn_type,
            name=name,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            activation=self.activation,
            dropout_rate=self.dropout_rate,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=self.bias_initializer,
            use_bias=self.use_bias,
            output_kernel_initializer=self.residual_output_kernel_initializer,
            ffn_args=self.ffn_args,
        )

    def _create_ffn_layer(self, name: str) -> keras.layers.Layer:
        """Create a feed-forward network or MoE layer.

        :param name: Name for the FFN layer.
        :type name: str
        :return: An unbuilt FFN or MoE layer instance.
        :rtype: keras.layers.Layer
        :raises ValueError: If creation fails due to invalid parameters.
        """
        if self.moe_config is not None:
            return MixtureOfExperts(config=self.moe_config, name=name)

        config = self._get_ffn_config(name)
        try:
            return create_ffn_from_config(config)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Failed to create {self.ffn_type} layer. "
                f"Check for parameter incompatibility. Custom args: {list(self.ffn_args.keys())}. "
                f"Original error: {e}"
            )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build all sub-layers with appropriate shapes.

        :param input_shape: Shape tuple ``(batch, seq_len, hidden_size)``.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If input shape is invalid or incompatible.
        """
        if self.built:
            return

        if len(input_shape) != 3:
            raise ValueError(f"Expected 3D input shape, got {len(input_shape)}D: {input_shape}")
        if input_shape[-1] != self.hidden_size:
            raise ValueError(
                f"Input feature dimension ({input_shape[-1]}) must match hidden_size ({self.hidden_size})"
            )

        # Build all sub-layers in computational order
        self.attention_norm.build(input_shape)
        self.output_norm.build(input_shape)
        self.attention.build(input_shape)
        self.ffn_layer.build(input_shape)
        self.dropout.build(input_shape)
        if self.attention_stochastic_depth is not None:
            self.attention_stochastic_depth.build(input_shape)
        if self.ffn_stochastic_depth is not None:
            self.ffn_stochastic_depth.build(input_shape)
        if self.attention_layer_scale is not None:
            self.attention_layer_scale.build(input_shape)
        if self.ffn_layer_scale is not None:
            self.ffn_layer_scale.build(input_shape)

        # Always call super().build() at the end
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            attention_mask: Optional[keras.KerasTensor] = None,
            layer_idx: int = 0,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass of the transformer layer.

        :param inputs: Input tensor ``(batch, seq_len, hidden_size)``.
        :type inputs: keras.KerasTensor
        :param attention_mask: Optional attention mask.

            Discarded with no warning for three attention types.
            ``_MASKLESS_ATTENTION_TYPES`` is ``{'fnet', 'anchor', 'lighthouse'}``,
            whose sub-layers accept no mask argument at all. Measured
            2026-08-27, ``max|y_masked - y_unmasked|`` with a ``(B, T)``
            keep-mask zeroing the tail half:

            * ``anchor``, ``fnet``, ``lighthouse``: exactly 0.0 — the mask
              does nothing;
            * every other usable type: 0.165 to 1.679.

            Padding a batch and passing a mask produces silently wrong
            results for those three. See each layer's own docstring for the
            measured cost of its missing mask.
        :type attention_mask: Optional[keras.KerasTensor]
        :param layer_idx: Layer index (used by differential attention).
        :type layer_idx: int
        :param training: Training mode flag.
        :type training: Optional[bool]
        :return: Output tensor ``(batch, seq_len, hidden_size)``.
        :rtype: keras.KerasTensor
        """
        residual = inputs

        if self.normalization_position == 'pre':
            # 1. Attention block
            x = self.attention_norm(inputs, training=training)
            if self.attention_type == 'differential':
                x = self.attention(x, attention_mask=attention_mask, layer_idx=layer_idx, training=training)
            elif self.attention_type in self._MASKLESS_ATTENTION_TYPES:
                x = self.attention(x, training=training)
            else:
                x = self.attention(x, attention_mask=attention_mask, training=training)
            if self.attention_stochastic_depth is not None:
                x = self.attention_stochastic_depth(x, training=training)
            if self.attention_layer_scale is not None:
                x = self.attention_layer_scale(x, training=training)
            attention_output = x + residual

            # 2. FFN block
            residual = attention_output
            x = self.output_norm(attention_output, training=training)
            x = self.ffn_layer(x, training=training)
            x = self.dropout(x, training=training)
            if self.ffn_stochastic_depth is not None:
                x = self.ffn_stochastic_depth(x, training=training)
            if self.ffn_layer_scale is not None:
                x = self.ffn_layer_scale(x, training=training)
            layer_output = x + residual
        else:
            # 1. Attention block
            if self.attention_type == 'differential':
                x = self.attention(
                    inputs,
                    attention_mask=attention_mask,
                    layer_idx=layer_idx,
                    training=training)
            elif self.attention_type in self._MASKLESS_ATTENTION_TYPES:
                x = self.attention(inputs, training=training)
            else:
                x = self.attention(
                    inputs,
                    attention_mask=attention_mask,
                    training=training)
            if self.attention_stochastic_depth is not None:
                x = self.attention_stochastic_depth(x, training=training)
            if self.attention_layer_scale is not None:
                x = self.attention_layer_scale(x, training=training)
            attention_output = self.attention_norm(x + residual, training=training)

            # 2. FFN block
            residual = attention_output
            x = self.ffn_layer(attention_output, training=training)
            x = self.dropout(x, training=training)
            if self.ffn_stochastic_depth is not None:
                x = self.ffn_stochastic_depth(x, training=training)
            if self.ffn_layer_scale is not None:
                x = self.ffn_layer_scale(x, training=training)
            layer_output = self.output_norm(x + residual, training=training)

        return layer_output

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape (same as input).

        :param input_shape: Input shape tuple.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary for serialization.

        :return: Dictionary containing all constructor parameters.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'hidden_size': self.hidden_size,
            'num_heads': self.num_heads,
            'intermediate_size': self.intermediate_size,
            'attention_type': self.attention_type,
            'attention_args': self.attention_args,
            'normalization_type': self.normalization_type,
            'normalization_position': self.normalization_position,
            'attention_norm_args': self.attention_norm_args,
            'ffn_norm_args': self.ffn_norm_args,
            'ffn_type': self.ffn_type,
            'ffn_args': self.ffn_args,
            'moe_config': self.moe_config.to_dict() if self.moe_config else None,
            'dropout_rate': self.dropout_rate,
            'attention_dropout_rate': self.attention_dropout_rate,
            'use_stochastic_depth': self.use_stochastic_depth,
            'stochastic_depth_rate': self.stochastic_depth_rate,
            'activation': keras.activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'residual_output_kernel_initializer': (
                initializers.serialize(self.residual_output_kernel_initializer)
                if self.residual_output_kernel_initializer is not None else None
            ),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'window_size': self.window_size,
            'n_kv_head': self.n_kv_head,
            'lambda_init': self.lambda_init,
            'use_layer_scale': self.use_layer_scale,
            'layer_scale_init_value': self.layer_scale_init_value,
        })
        return config

# ---------------------------------------------------------------------
