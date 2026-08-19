"""
Foundational building block of a Transformer network, implementing a highly
configurable and serializable encoder/decoder layer.

This layer encapsulates the two primary sub-components of a standard Transformer
architecture: a multi-head self-attention mechanism and a position-wise
feed-forward network. Each sub-component is enclosed within a residual
connection followed by layer normalization, a crucial design pattern that
enables the stable training of deep sequential models.

**Intent**: To provide a robust, production-ready, and flexible Transformer
layer that serves as a fundamental building block for a wide range of sequence
modeling tasks. It is designed to be highly configurable, allowing for easy
swapping of attention, FFN, and normalization components for architectural
research and experimentation, while strictly adhering to modern Keras 3 best
practices for serialization and composite layer construction.

**Architecture**: The layer processes an input sequence through two main blocks,
with the exact data flow determined by `normalization_position`.

**1. Pre-Normalization (`normalization_position='pre'`)**:
```
Input
  |
  +-- Norm → Attention → [StochasticDepth] → [LayerScale] --+
  |                                                          |
  +--------------------------- Add --------------------------+
                                |
  +-- Norm → FFN/MoE → Dropout → [StochasticDepth] → [LayerScale] --+
  |                                                                 |
  +------------------------------ Add ------------------------------+
                                |
                              Output
```

**2. Post-Normalization (`normalization_position='post'`)**:
```
Input
  |
  +-- Attention → [StochasticDepth] → [LayerScale] → Add → Norm --+
                                                                  |
  +-- FFN/MoE → Dropout → [StochasticDepth] → [LayerScale] → Add → Norm --+
                                                                    |
                                                                  Output
```

Note the asymmetry, which is deliberate and MEASURED, not a drawing shortcut:
``self.dropout`` (the ``dropout_rate`` layer) is applied to the FFN sub-block
ONLY -- it is invoked exactly once per forward pass, at both normalization
positions, and its input is the FFN output. There is NO dropout step after
attention. ``attention_dropout_rate`` is a different thing entirely: it is
forwarded to the attention sub-layer's own attention-weight dropout
constructor argument -- ``dropout_rate`` for most types, ``attn_dropout_rate``
for ``attention_type='beit'``, whose layer spells the two dropouts separately
-- i.e. dropout applied inside the attention layer, not an output dropout
applied here. The bracketed steps are optional and present
only when ``use_stochastic_depth`` / ``use_layer_scale`` are enabled.

**Mathematical Operations**:
1.  **Multi-Head Self-Attention (MHSA)**:
    -   Computes context-aware representations using scaled dot-product attention:
        `Attention(Q, K, V) = softmax( (Q @ K.T) / sqrt(d_k) ) @ V`
    -   Uses multiple "heads" to attend to different representational subspaces
        in parallel, enhancing the model's ability to capture complex relationships.

2.  **Position-wise Feed-Forward Network (FFN)**:
    -   Applies a non-linear transformation independently at each sequence position.
    -   Typically a two-layer MLP: `FFN(x) = activation(x @ W₁ + b₁) @ W₂ + b₂`
    -   This component can be replaced by more advanced structures like SwiGLU or
        a Mixture of Experts (MoE) layer.

**References**:
    - Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS*.
    - Ba, J. L., et al. (2016). Layer Normalization. *arXiv preprint*.
    - Xiong, R., et al. (2020). On Layer Normalization in the Transformer
      Architecture. *ICML*. (Analysis of Pre-LN vs. Post-LN).
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
from ..layer_scale import LearnableMultiplier
from ..ffn import assemble_ffn_config, create_ffn_from_config, FFNType
from ..attention import create_attention_layer, AttentionType
from ..norms import create_normalization_layer, NormalizationType

# ---------------------------------------------------------------------
# Type definitions for enhanced type safety
# ---------------------------------------------------------------------

NormalizationPositionType = Literal['post', 'pre']


# ---------------------------------------------------------------------
# The transformer family's ONE FFN parameter-injection policy
# ---------------------------------------------------------------------

#: FFN types whose defining feature is a FIXED nonlinearity or gate, so the
#: wrapper's single generic ``activation`` must NOT be forwarded even though the
#: registry accepts an ``activation`` key for some of them.
#:
#: * ``squared_relu`` -- fixed ``relu(x) ** 2``; the registry has no
#:   ``activation`` param at all, so this entry is documentation.
#: * ``reglu`` / ``bilinear`` -- ``GLUFFN`` aliases whose whole identity is the
#:   fixed relu / linear gate. They DO accept ``activation``, so the pre-filter
#:   would happily let it through; withholding it must be explicit. (D-005)
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
        ffn_args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the FFN factory config for a transformer encoder/decoder block.

    Interface contract (2 call sites: :meth:`TransformerLayer._get_ffn_config`
    and :meth:`TransformerDecoderLayer._get_ffn_config`; the two producing the
    IDENTICAL dict for every registry type is the reason this exists):

    * Emits the block's generic conveniences (dims derived from
      ``hidden_size``/``intermediate_size``, plus ``activation``,
      ``dropout_rate`` and the initializers), applies the three per-type POLICY
      adjustments below, then hands the result to
      :func:`~dl_techniques.layers.ffn.factory.assemble_ffn_config`, which
      intersects it with what ``ffn_type`` actually accepts and merges
      ``ffn_args`` on top UNFILTERED.
    * Returns a config carrying ``type`` and ``name``, ready for
      ``create_ffn_from_config``.
    * Raises ``ValueError`` for an unregistered ``ffn_type``.

    The three policy adjustments -- the only things the registry intersection
    cannot express, and therefore the only per-type branching left here:

    1. ``swiglu`` sizes ITSELF (2/3 rule from ``ffn_expansion_factor``, rounded
       to ``ffn_multiple_of``) and lists ``hidden_dim`` as OPTIONAL, so passing
       the block's ``intermediate_size`` would silently override that
       derivation. Withheld, and the two expansion knobs are supplied instead.
    2. ``differential`` RENAMES: ``DifferentialFFN`` takes ``branch_activation``,
       not ``activation`` (D-016). ``gate_activation`` is deliberately not
       forwarded -- the sigmoid gate is the layer's defining feature.
    3. ``_FFN_TYPES_WITH_FIXED_ACTIVATION`` withholds ``activation`` (D-005).
    4. ``swiglu`` also withholds ``use_bias``
       (plan-2026-08-19-a616f581/D-006, see the branch below).

    # DECISION plan-2026-07-30T140922-8af1028f/D-018
    Do NOT re-inline this table into either caller. Two independently
    hand-maintained copies of it are exactly what produced the D-016 defect
    (`differential` silently losing its activation on the decoder for the whole
    life of that file) and five further decoder-only coverage gaps (`lowrank`,
    `monarch`, `squared_relu`, `reglu`, `bilinear` raised on the decoder while
    the encoder handled them). Pinned by
    ``TestEncoderDecoderFFNConfigParity`` in
    ``tests/test_layers/test_transformers/test_transformer.py``, which compares
    both dispatchers over every ``FFN_REGISTRY`` key.

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
    :param bias_initializer: The block's bias initializer.
    :type bias_initializer: Any
    :param use_bias: The block's bias switch. Forwarded to every registry type
        that declares a ``use_bias`` key EXCEPT ``swiglu`` (policy 4); types
        that declare none (``kan``, ``tversky``) drop it in the pre-filter.
    :type use_bias: bool
    :param ffn_args: The caller's explicit FFN args; merged LAST and NEVER
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

    if ffn_type == 'swiglu':
        del config['hidden_dim']
        del config['activation']
        # DECISION plan-2026-08-19-a616f581/D-006: `swiglu` is the ONE registry
        # type whose own `use_bias` default is False (measured: every other
        # bias-declaring type defaults True), because a bias-free gated FFN is
        # SwiGLUFFN's defining LLaMA-style design. The block's `use_bias`
        # therefore must NOT be forwarded here. Do NOT "make it uniform" by
        # deleting this line: `TransformerLayer`'s own `use_bias` default is
        # True, so forwarding would ADD `gate_proj/bias`, `up_proj/bias` and
        # `down_proj/bias` (measured) to every swiglu block of every model that
        # never asked for them -- vit_siglip, tiny_recursive_model,
        # qwen3_embeddings, nano_vlm, dino v2/v3 giant and the HRM family all
        # default to `ffn_type='swiglu'` with `use_bias` at its True default,
        # so their `.keras` files would all stop matching. A caller who really
        # wants biased swiglu passes `ffn_args={'use_bias': True}`, which the
        # pre-filter never touches. See decisions.md D-006.
        del config['use_bias']
        config['ffn_expansion_factor'] = 4
        config['ffn_multiple_of'] = 256
    elif ffn_type == 'differential':
        config['branch_activation'] = config.pop('activation')
    elif ffn_type in _FFN_TYPES_WITH_FIXED_ACTIVATION:
        del config['activation']

    return assemble_ffn_config(ffn_type, config, ffn_args)


# ---------------------------------------------------------------------

#: Default ``window_size`` for ``attention_type='window'``. Read by BOTH
#: :meth:`TransformerLayer.__init__` (as its signature default) and
#: :func:`build_transformer_attention_required_params` (as the fallback for a
#: block that has no dedicated ``window_size`` constructor parameter, i.e.
#: ``TransformerDecoderLayer``).
_DEFAULT_ATTENTION_WINDOW_SIZE: int = 8

#: Default ``lambda_init`` for ``attention_type='differential'``. Same two
#: readers as ``_DEFAULT_ATTENTION_WINDOW_SIZE`` above.
_DEFAULT_ATTENTION_LAMBDA_INIT: float = 0.8

#: MoE expert FFN types whose ``hidden_dim`` falls back to
#: ``TransformerLayer.intermediate_size`` when the caller's
#: ``moe_config.expert_config.ffn_config`` omits it. Defined once and read by
#: BOTH the fallback itself and the ``moe_config`` warning text, so the warning
#: cannot drift away from the behaviour again (F-17: it previously claimed
#: ``intermediate_size`` was ignored while this fallback consulted it).
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
    """The params the attention factory REQUIRES beyond ``dim``/``num_heads``.

    Interface contract (2 call sites:
    :meth:`TransformerLayer._get_attention_params` and
    :meth:`TransformerDecoderLayer._self_attention_params`; the two agreeing on
    every type is the reason this exists):

    * Returns ONLY the type-specific keys that
      ``dl_techniques.layers.attention.factory`` lists as REQUIRED and that are
      not already covered by ``dim``/``num_heads`` — never the generic
      conveniences (``dropout_rate``, ``use_bias``, initializers), which the two
      blocks legitimately differ on.
    * Returns an EMPTY dict for every type with no such extra requirement
      (``multi_head``, ``multi_head_cross``, ``anchor``, ``lighthouse``,
      ``fnet``), and for an unknown type — validating the type is the FACTORY's
      job, and this function must not turn a factory error into a different one.
    * Never raises.

    # DECISION plan-2026-07-31T132403-b3f540cb/D-015
    Do NOT re-inline this table into either caller. Two hand-maintained copies of
    the per-type attention parameter table are exactly what F-07 was: the
    decoder's copy listed no type-specific params at all, so ``window``,
    ``group_query``, ``differential`` and ``multi_head_latent`` were
    unconstructable on the decoder (``ValueError: ... Required parameters:
    ['dim', 'window_size', 'num_heads']``) while ``TransformerLayer`` handled all
    four. This is the SAME defect class as D-018 (the FFN table, whose two copies
    produced the ``differential``/``activation`` silent drop plus five
    decoder-only coverage gaps), in the same pair of files, one method over.

    The four default VALUES here are the encoder's, verbatim, so a block without
    a dedicated constructor parameter (the decoder has none of ``window_size`` /
    ``n_kv_head`` / ``lambda_init``) gets the same answer the encoder would give.
    A caller's ``attention_args`` still overrides everything, since both callers
    merge it last.

    :param attention_type: An ``ATTENTION_REGISTRY`` key.
    :type attention_type: str
    :param hidden_size: The block's model width (the factory's ``dim``).
    :type hidden_size: int
    :param num_heads: The block's head count.
    :type num_heads: int
    :param window_size: ``'window'`` and ``'beit'`` only. The spatial window
        edge length for ``'window'``; the ``(Wh, Ww)`` patch grid for
        ``'beit'`` (an ``int`` there meaning the square grid ``(W, W)``).
    :type window_size: Union[int, Tuple[int, int]]
    :param num_kv_heads: ``'group_query'`` only. ``None`` means ``num_heads``
        (i.e. degrade to plain MHA), matching ``TransformerLayer.n_kv_head``.
    :type num_kv_heads: Optional[int]
    :param lambda_init: ``'differential'`` only. Initial lambda.
    :type lambda_init: float
    :return: The type-specific required params; possibly empty.
    :rtype: Dict[str, Any]
    """
    if attention_type in ('window', 'beit'):
        # Both require a 'window_size', and this is the SAME table entry on
        # purpose (D-015) — but the two types read the value differently:
        # 'window' takes a scalar spatial edge length W and attends within
        # W*W-token windows, while 'beit' takes the PATCH GRID and expects a
        # sequence of exactly Wh*Ww + 1 tokens (the +1 being the cls token).
        # A scalar reaching 'beit' is normalized to the square grid (W, W).
        return {'window_size': window_size}
    if attention_type == 'group_query':
        return {'num_kv_heads': num_kv_heads if num_kv_heads is not None else num_heads}
    if attention_type == 'differential':
        return {'head_dim': hidden_size // num_heads, 'lambda_init': lambda_init}
    if attention_type == 'multi_head_latent':
        # MLA requires kv_latent_dim and NEITHER block has a dedicated ctor
        # param for it, so this documented default is the only source.
        return {'kv_latent_dim': max(1, hidden_size // 4)}
    return {}


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class TransformerLayer(keras.layers.Layer):
    """
    Generic transformer layer with configurable attention, FFN, and normalization.

    Implements a standard transformer block consisting of multi-head
    self-attention followed by a position-wise feed-forward network, each
    wrapped in residual connections and normalization. The exact data flow
    is determined by ``normalization_position`` (pre or post). All core
    sub-components (attention, FFN, normalization) are constructed via
    factory functions, enabling easy architectural exploration.

    ``Attention(Q, K, V) = softmax((Q K^T) / sqrt(d_k)) V``

    **Architecture Overview:**

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
    :param intermediate_size: FFN intermediate dimension. NOT ignored when
        ``moe_config`` is provided: it is then used as the fallback for the
        expert FFN's ``hidden_dim`` whenever
        ``moe_config.expert_config.ffn_config`` omits that key and the expert
        type is one of ``{'mlp', 'differential', 'glu', 'geglu', 'residual',
        'swin_mlp'}``. It is only genuinely unused when ``moe_config`` is set
        AND the expert config already carries its own ``hidden_dim`` (or the
        expert type is not one of those six).
    :type intermediate_size: int
    :param attention_type: Attention mechanism type. Default: ``'multi_head'``.
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
        CALLER's explicit keys and are merged LAST, after this layer's own
        generic conveniences have been intersected with what ``ffn_type``
        accepts. They are never pre-filtered, so they always reach
        ``create_ffn_layer`` -- including a misdirected or misspelled one,
        which the factory reports rather than this layer swallowing it.
    :type ffn_args: Optional[Dict[str, Any]]
    :param moe_config: Mixture-of-Experts configuration replacing the FFN.
    :type moe_config: Optional[Union[MoEConfig, Dict[str, Any]]]
    :param dropout_rate: FFN output dropout rate. This is the ONLY dropout this
        layer applies itself, and it is applied to the FFN sub-block only --
        never after attention (see the architecture note above). Default: 0.1.
    :type dropout_rate: float
    :param attention_dropout_rate: Attention-INTERNAL (weight) dropout rate.
        Not an output dropout: this value is forwarded verbatim as the
        attention sub-layer's own ``dropout_rate`` constructor argument (e.g.
        ``MultiHeadAttention(dropout_rate=...)``), so it acts on the attention
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

    # DECISION plan_2026-06-12_0bb1729b/D-001: attention types whose factory
    # layer `call` signature does NOT accept an `attention_mask` argument.
    # `call()` must invoke these WITHOUT passing `attention_mask`, or the
    # sub-layer raises TypeError. Verified empirically (signatures):
    #   - 'fnet'       -> FNetFourierTransform.call(inputs, training)
    #   - 'anchor'     -> AnchorAttention.call(x, num_anchor_tokens=None, training)
    #   - 'lighthouse' -> LighthouseAttention.call(inputs, training)
    # 'multi_head_latent' DOES accept attention_mask and stays on the standard
    # branch. Do NOT add a type here unless its `call` genuinely rejects mask.
    #
    # DECISION plan-2026-07-31T132403-b3f540cb/D-016
    # Do NOT add 'window' here. It was CONSIDERED and REFUTED BY MEASUREMENT
    # (G-07, 2026-07-31): `WindowAttention` (the layer behind the 'window' key)
    # both accepts AND genuinely honours a rank-3 causal keep-mask -- perturbing
    # the last token moved every earlier position by exactly 0.0, against an
    # unmasked control of 1.97e+02 -- whenever `seq_len == window_size ** 2`,
    # and raises a loud ValueError at any other length. Adding it here would
    # convert that into a SILENT non-causal block: the dead-component probe for
    # this decision added 'window' to this very frozenset and measured
    # `TransformerDecoderLayer` leaking the future by 3.713046e+00 at the exact
    # geometry where it is otherwise bit-exactly causal. "Accepts the kwarg" is
    # not the test either way -- 'fnet' accepts it and then dies on a shape
    # mismatch. See decisions.md D-016 and `TestWindowSelfAttentionIsMaskedNotMaskless`.
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

        # --- Input Validation (early) ---
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
        # `call()` dispatches on `== 'pre'` with an unguarded `else` for post-norm,
        # so without this check ANY other spelling ('Pre', 'PRE', 'postt', '')
        # silently ran the POST-norm branch -- a typo became a different
        # architecture with no error. Same check, same message, as the sibling
        # `TransformerDecoderLayer.__init__`.
        if normalization_position not in ('pre', 'post'):
            raise ValueError(
                f"normalization_position must be 'pre' or 'post', got {normalization_position}"
            )

        # --- Configuration Storage ---
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
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.window_size = window_size
        self.n_kv_head = n_kv_head if n_kv_head is not None else num_heads
        self.lambda_init = lambda_init
        self.use_layer_scale = bool(use_layer_scale)
        self.layer_scale_init_value = float(layer_scale_init_value)

        # --- Handle MoE Configuration ---
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

            # If expert_config is for an MLP-like FFN and doesn't have its intermediate size set,
            # use TransformerLayer's intermediate_size as a sensible default.
            # This block is UNCONDITIONAL (it is not gated by the warning's `if`
            # above), which is why the warning must not claim intermediate_size
            # is ignored. The type list lives in exactly one place --
            # `_MOE_EXPERT_TYPES_USING_INTERMEDIATE_SIZE` -- which the warning
            # text above interpolates, so the two cannot drift.
            ffn_type = ffn_config.get('type')
            if ffn_type in _MOE_EXPERT_TYPES_USING_INTERMEDIATE_SIZE:
                if 'hidden_dim' not in ffn_config:
                    ffn_config['hidden_dim'] = self.intermediate_size

        # --- Create Sub-layers (unbuilt) ---
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

        # LayerScale (CaiT, Touvron et al. 2021): a per-channel learnable scale
        # applied to each residual *branch* output before the add, initialized
        # small. It keeps the pre-norm residual stream from blowing up across
        # deep stacks (without it a 12-layer ViT's activation std grew 2 -> 23,
        # which the final LayerNorm then compresses, starving downstream layers).
        self.attention_layer_scale = None
        self.ffn_layer_scale = None
        if self.use_layer_scale:
            self.attention_layer_scale = LearnableMultiplier(
                multiplier_type='CHANNEL',
                initializer=keras.initializers.Constant(self.layer_scale_init_value),
                constraint=None,
                name='attention_layer_scale',
            )
            self.ffn_layer_scale = LearnableMultiplier(
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
        """This block's type-specific REQUIRED attention params.

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
        elif self.attention_type == 'window':
            # DECISION plan-2026-08-19-a616f581/D-005: the block's `use_bias`
            # MUST be forwarded here, and it is spelled `qkv_bias`/`proj_bias`
            # -- NOT `use_bias`. The 'window' registry entry
            # (attention/factory.py, key 'window') declares exactly those two
            # names under `optional_params`, BOTH DEFAULTING TO True. Do NOT
            # "simplify" this to `'use_bias': self.use_bias` to match the
            # 'multi_head'/'group_query'/'anchor' branches: since D-011 the
            # factory RAISES on undeclared keys, so that spelling is a
            # construction failure, and before this branch forwarded anything
            # at all the two `True` defaults silently won -- ModernBERT (all
            # three variants set `use_bias: False`) carried `qkv/bias` and
            # `proj/bias` on every one of its ~68% local layers. This is a
            # weight-SET change: it removes 2 of the 5 tensors in a local
            # layer's attention subtree at `use_bias=False`, so any pre-fix
            # `.keras` built that way will not load. See decisions.md D-005.
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
            # NOT a copy of the 'window' branch: `BeitAttention` declares no
            # `dropout_rate`, it declares `attn_dropout_rate` /
            # `proj_dropout_rate`. HISTORICAL, and the reason for the rename
            # SURVIVES the change: `create_attention_layer` used to FILTER kwargs
            # to the registry's declared names and DROP the rest SILENTLY, so
            # passing 'dropout_rate' here would have looked correct, raised
            # nothing, and left the attention probabilities undropped at 0.0
            # forever. Since 2026-08-17 (plan-2026-08-17T183311-79c63e38/D-011)
            # that factory RAISES instead, so the same mistake would now be a
            # loud construction failure rather than a silent one -- still a
            # mistake, just a findable one. The block's
            # `attention_dropout_rate` is routed to the attention-probability
            # dropout, matching every other branch's intent; `proj_dropout_rate`
            # is deliberately left at the layer default so the block's own
            # `self.dropout` (applied to the residual branch) is not doubled.
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
        # --- DECISION plan_2026-06-12_0bb1729b/D-001: additionally-wired
        # self-attention factory types. Each accepts a standard
        # `(inputs, attention_mask=..., training=...)` self-attention call
        # (except 'fnet', see _MASKLESS_ATTENTION_TYPES). Added additively to
        # keep the 4 branches above byte-identical (regression invariant). ---
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

        # User-provided args override defaults
        return {**default_params, **self.attention_args}

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

    def _get_ffn_config(self, name: str) -> Dict[str, Any]:
        """Consolidate configuration for FFN layer creation.

        :param name: Name for the FFN layer.
        :type name: str
        :return: Parameter dictionary for the FFN factory.
        :rtype: Dict[str, Any]
        """
        return build_transformer_ffn_config(
            ffn_type=self.ffn_type,
            name=name,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            activation=self.activation,
            dropout_rate=self.dropout_rate,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            use_bias=self.use_bias,
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
            # --- Pre-Normalization: Normalize -> SubLayer -> Add ---
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
            # --- Post-Normalization: SubLayer -> Add -> Normalize ---
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
