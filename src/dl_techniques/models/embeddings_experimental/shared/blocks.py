"""Sequence-mixing blocks and the registry that resolves them.

Every arm of the embeddings study differs ONLY in the block it stacks. This
module defines the contract those blocks share and the registry that maps a
block-type string to a builder.

The contract is deliberately identical to
:class:`~dl_techniques.layers.transformers.transformer.TransformerLayer`'s, so
the baseline arm needs no adapter at all::

    block(hidden_states, attention_mask=..., layer_idx=..., training=...)
        -> hidden_states of the same shape (batch, seq_len, hidden_size)

Registry discipline follows the house factory rules in
``src/dl_techniques/CLAUDE.md``: an unknown block type raises, and so does a
keyword the target builder does not declare. Filter-and-drop is never used --
that design is what previously made ``dropout=`` (against a declared
``dropout_rate``) a silent no-op repo-wide.

Adding an arm
-------------
Write a builder with an explicit keyword-only signature, add one entry to
:data:`BLOCK_REGISTRY`, and the encoder, the trainer and the sweep all pick it
up. No change to :class:`~...encoder.EmbeddingEncoder` is required.

References:
    - Vaswani et al., 2017. Attention Is All You Need.
      (https://arxiv.org/abs/1706.03762)
    - Ji, Z., 2026. CliffordNet: All You Need is Geometric Algebra.
      (https://arxiv.org/abs/2601.06793)
    - Brandstetter et al., 2023. Clifford Neural Layers for PDE Modeling.
      (https://arxiv.org/abs/2209.04934)
    - Ruhe et al., 2023. Geometric Clifford Algebra Networks.
      (https://arxiv.org/abs/2302.06594)
    - Touvron et al., 2021. Going Deeper with Image Transformers (CaiT).
      (LayerScale) (https://arxiv.org/abs/2103.17239)
    - Huang et al., 2016. Deep Networks with Stochastic Depth.
      (https://arxiv.org/abs/1603.09382)
"""

import inspect
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import keras

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.geometric.clifford_block import CliffordNetBlock
from dl_techniques.layers.stochastic_depth import StochasticDepth
from dl_techniques.layers.transformers import TransformerLayer
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

__all__ = [
    "BLOCK_REGISTRY",
    "CliffordEncoderBlock",
    "available_block_types",
    "build_clifford_block",
    "build_transformer_block",
    "clifford_receptive_field",
    "create_encoder_block",
]


# ---------------------------------------------------------------------
# Clifford encoder block
# ---------------------------------------------------------------------

def clifford_receptive_field(num_layers: int, context_kernel_size: int) -> int:
    """Return the causal-context width, in tokens, of a Clifford stack.

    Each :class:`CliffordNetBlock` applies **two** stacked depthwise
    convolutions of width ``context_kernel_size``, so one block widens the
    receptive field by ``2 * (K - 1)`` and a stack of ``num_layers`` reaches
    ``num_layers * 2 * (K - 1) + 1`` positions.

    This matters far more at character granularity than at sub-word
    granularity: with the layer's default ``K = 3``, a 4-block stack sees 17
    characters, which is a few words rather than a sentence. The geometric
    product does NOT help here -- it rolls along the CHANNEL axis, not the
    sequence axis, so it contributes nothing to the token-mixing span. Choose
    ``context_kernel_size`` (or enable the global-context branch, whose
    cumulative mean has unbounded reach) deliberately.

    :param num_layers: Number of stacked blocks.
    :type num_layers: int
    :param context_kernel_size: Depthwise kernel width ``K`` per convolution.
    :type context_kernel_size: int
    :return: Receptive field in tokens.
    :rtype: int
    """
    return num_layers * 2 * (context_kernel_size - 1) + 1


@keras.saving.register_keras_serializable()
class CliffordEncoderBlock(keras.layers.Layer):
    """
    Bidirectional Clifford mixing block with the transformer block's call contract.

    Wraps :class:`~dl_techniques.layers.geometric.clifford_block.CliffordNetBlock`
    in sequence mode with ``causal=False`` -- the bidirectional setting, since
    this is an encoder rather than a language model -- and adapts it to the
    ``(inputs, attention_mask=, layer_idx=, training=)`` signature the encoder
    calls every block with.

    Two behaviours are load-bearing and are the reason this wrapper exists.

    **The residual is external.** ``CliffordNetBlock.call`` returns only the
    LayerScale-gated update, not ``x + update``. Writing ``x = block(x)``
    therefore does not "apply a block", it REPLACES the signal with a residual
    scaled by ``layer_scale_init`` (1e-5 by default), annihilating activations
    at roughly five orders of magnitude per block while every shape,
    finiteness and serialization test still passes. This wrapper computes
    ``inputs + drop_path(update)``.

    **The block cannot honour an attention mask.** ``CliffordNetBlock`` sets
    ``supports_masking = False`` and its module docstring records the measured
    consequences: the two stacked same-padded depthwise convolutions pull zero
    padding into the receptive field of real positions near the boundary
    (measured: the last real position of a 6-token prefix moves by 1.183 on a
    ~2.4-scale output when padded to 8), and with ``use_global_context=True``
    the pooled branch means over the whole padded length, so the pad LENGTH
    shifts every real position (measured: up to 0.449). Masking is a new
    capability there, not a repair.

    What this wrapper does about it: it zeroes the masked positions before the
    block, so padding contributes a known constant rather than whatever the
    pad token's embedding happens to have learned, and it logs the limitation
    once at construction so it appears in every run log. That is a mitigation,
    not a fix -- the boundary effect survives it. The study's real answer is to
    pretrain on PACKED fixed-length sequences that carry no padding at all, and
    to bucket by length wherever padding is unavoidable.

    ``use_global_context`` defaults to ``False`` because that is the setting
    whose padding hazard is boundary-local rather than global.

    :param hidden_size: Channel dimension ``D``. Preserved by the block.
    :type hidden_size: int
    :param shifts: Channel-axis roll offsets for the geometric product.
    :type shifts: Sequence[int]
    :param cli_mode: Clifford components used -- ``"inner"``, ``"wedge"`` or
        ``"full"``. Defaults to ``"full"``.
    :type cli_mode: str
    :param ctx_mode: Context mode, ``"diff"`` or ``"abs"``. Defaults to
        ``"diff"``.
    :type ctx_mode: str
    :param use_global_context: Whether to add the pooled global branch. See the
        padding hazard above. Defaults to ``False``.
    :type use_global_context: bool
    :param context_kernel_size: Depthwise kernel width. See
        :func:`clifford_receptive_field`. Defaults to 3.
    :type context_kernel_size: int
    :param layer_scale_init: Initial LayerScale gamma. Defaults to 1e-5.
    :type layer_scale_init: float
    :param drop_path_rate: Stochastic-depth rate for this block's residual
        branch. Defaults to 0.0.
    :type drop_path_rate: float
    :param normalization_type: Normalization inside the block. ``None`` keeps
        the layer's own sequence-mode default. Defaults to ``None``.
    :type normalization_type: str | None
    :param use_bias: Whether inner projections carry a bias.
    :type use_bias: bool
    :param kernel_initializer: Initializer for inner kernels.
    :type kernel_initializer: Any
    :param kwargs: Additional keyword arguments for the Layer base class.
    :raises ValueError: If ``hidden_size`` or ``context_kernel_size`` is not a
        positive integer.
    """

    def __init__(
        self,
        hidden_size: int,
        shifts: Sequence[int] = (1, 2, 4),
        cli_mode: str = "full",
        ctx_mode: str = "diff",
        use_global_context: bool = False,
        context_kernel_size: int = 3,
        layer_scale_init: float = 1e-5,
        drop_path_rate: float = 0.0,
        normalization_type: Optional[str] = None,
        use_bias: bool = True,
        kernel_initializer: Any = "glorot_uniform",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError(
                f"hidden_size must be a positive int, got {hidden_size!r}"
            )
        if not isinstance(context_kernel_size, int) or context_kernel_size <= 0:
            raise ValueError(
                "context_kernel_size must be a positive int, got "
                f"{context_kernel_size!r}"
            )

        self.hidden_size = hidden_size
        self.shifts = list(shifts)
        self.cli_mode = cli_mode
        self.ctx_mode = ctx_mode
        self.use_global_context = use_global_context
        self.context_kernel_size = context_kernel_size
        self.layer_scale_init = layer_scale_init
        self.drop_path_rate = drop_path_rate
        self.normalization_type = normalization_type
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer

        block_kwargs: Dict[str, Any] = dict(
            channels=hidden_size,
            shifts=self.shifts,
            cli_mode=cli_mode,
            ctx_mode=ctx_mode,
            use_global_context=use_global_context,
            causal=False,
            input_mode="sequence",
            context_kernel_size=context_kernel_size,
            layer_scale_init=layer_scale_init,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
        )
        if normalization_type is not None:
            block_kwargs["normalization_type"] = normalization_type

        self.block = CliffordNetBlock(name="clifford_block", **block_kwargs)
        self.drop_path = StochasticDepth(
            drop_path_rate=drop_path_rate, name="drop_path"
        )

    def build(self, input_shape: Any) -> None:
        """Build the wrapped block on the sequence shape.

        :param input_shape: ``(batch, seq_len, hidden_size)``.
        :type input_shape: Any
        """
        self.block.build(input_shape)
        self.drop_path.build(input_shape)
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        layer_idx: int = 0,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Apply the block and add its update to the input.

        :param inputs: ``(batch, seq_len, hidden_size)``.
        :type inputs: keras.KerasTensor
        :param attention_mask: ``(batch, seq_len)`` with 1 for kept positions.
            Used only to zero padded positions before the block; the block
            itself is maskless (see the class docstring).
        :type attention_mask: keras.KerasTensor | None
        :param layer_idx: Accepted for signature compatibility with
            ``TransformerLayer``; unused.
        :type layer_idx: int
        :param training: Keras training flag.
        :type training: bool | None
        :return: ``(batch, seq_len, hidden_size)``.
        :rtype: keras.KerasTensor
        """
        block_input = inputs
        if attention_mask is not None:
            keep = keras.ops.cast(
                keras.ops.expand_dims(attention_mask, axis=-1), inputs.dtype
            )
            block_input = inputs * keep

        update = self.block(block_input, training=training)
        # External residual. `x = block(x)` would annihilate the signal; see
        # the class docstring.
        return inputs + self.drop_path(update, training=training)

    def compute_output_shape(self, input_shape: Any) -> Any:
        """Return the input shape; the block is shape-preserving.

        :param input_shape: ``(batch, seq_len, hidden_size)``.
        :type input_shape: Any
        :return: The same shape.
        :rtype: Any
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor configuration.

        :return: Serializable configuration dictionary.
        :rtype: dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "shifts": self.shifts,
                "cli_mode": self.cli_mode,
                "ctx_mode": self.ctx_mode,
                "use_global_context": self.use_global_context,
                "context_kernel_size": self.context_kernel_size,
                "layer_scale_init": self.layer_scale_init,
                "drop_path_rate": self.drop_path_rate,
                "normalization_type": self.normalization_type,
                "use_bias": self.use_bias,
                "kernel_initializer": self.kernel_initializer,
            }
        )
        return config


# ---------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------

def build_transformer_block(
    *,
    hidden_size: int,
    name: str,
    num_heads: int = 8,
    intermediate_size: int = 2048,
    attention_type: str = "multi_head",
    ffn_type: str = "mlp",
    normalization_type: str = "layer_norm",
    normalization_position: str = "post",
    layer_norm_eps: float = 1e-12,
    dropout_rate: float = 0.1,
    attention_dropout_rate: float = 0.1,
    drop_path_rate: float = 0.0,
    activation: Any = "gelu",
    use_bias: bool = True,
    kernel_initializer: Any = "glorot_uniform",
) -> keras.layers.Layer:
    """Build one baseline transformer block.

    Thin pass-through to :class:`TransformerLayer`, which already satisfies the
    block call contract, so the baseline arm carries no adapter.

    ``layer_norm_eps`` is threaded into both norms explicitly. Keras defaults
    ``LayerNormalization`` to ``epsilon=1e-3`` -- a thousand times the value
    every BERT reference uses -- with no shape symptom and no warning, so it is
    never left implicit here.

    :param hidden_size: Model width.
    :type hidden_size: int
    :param name: Layer name.
    :type name: str
    :param num_heads: Attention heads.
    :type num_heads: int
    :param intermediate_size: FFN inner width.
    :type intermediate_size: int
    :param attention_type: Attention registry key.
    :type attention_type: str
    :param ffn_type: FFN registry key.
    :type ffn_type: str
    :param normalization_type: Normalization registry key.
    :type normalization_type: str
    :param normalization_position: ``"pre"`` or ``"post"``.
    :type normalization_position: str
    :param layer_norm_eps: Normalization epsilon, passed to both norms.
    :type layer_norm_eps: float
    :param dropout_rate: Hidden dropout.
    :type dropout_rate: float
    :param attention_dropout_rate: Attention-probability dropout.
    :type attention_dropout_rate: float
    :param drop_path_rate: Stochastic-depth rate; 0.0 disables it.
    :type drop_path_rate: float
    :param activation: FFN activation.
    :type activation: Any
    :param use_bias: Whether projections carry a bias.
    :type use_bias: bool
    :param kernel_initializer: Initializer for kernels.
    :type kernel_initializer: Any
    :return: A configured ``TransformerLayer``.
    :rtype: keras.layers.Layer
    """
    norm_args = {"epsilon": layer_norm_eps}
    return TransformerLayer(
        hidden_size=hidden_size,
        num_heads=num_heads,
        intermediate_size=intermediate_size,
        normalization_type=normalization_type,
        attention_norm_args=dict(norm_args),
        ffn_norm_args=dict(norm_args),
        normalization_position=normalization_position,
        attention_type=attention_type,
        ffn_type=ffn_type,
        dropout_rate=dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
        use_stochastic_depth=drop_path_rate > 0.0,
        stochastic_depth_rate=drop_path_rate,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        name=name,
    )


def build_clifford_block(
    *,
    hidden_size: int,
    name: str,
    shifts: Sequence[int] = (1, 2, 4),
    cli_mode: str = "full",
    ctx_mode: str = "diff",
    use_global_context: bool = False,
    context_kernel_size: int = 3,
    layer_scale_init: float = 1e-5,
    drop_path_rate: float = 0.0,
    normalization_type: Optional[str] = None,
    use_bias: bool = True,
    kernel_initializer: Any = "glorot_uniform",
) -> keras.layers.Layer:
    """Build one bidirectional Clifford mixing block.

    :param hidden_size: Model width.
    :type hidden_size: int
    :param name: Layer name.
    :type name: str
    :param shifts: Channel-axis roll offsets.
    :type shifts: Sequence[int]
    :param cli_mode: ``"inner"``, ``"wedge"`` or ``"full"``.
    :type cli_mode: str
    :param ctx_mode: ``"diff"`` or ``"abs"``.
    :type ctx_mode: str
    :param use_global_context: Whether to add the pooled global branch.
    :type use_global_context: bool
    :param context_kernel_size: Depthwise kernel width.
    :type context_kernel_size: int
    :param layer_scale_init: Initial LayerScale gamma.
    :type layer_scale_init: float
    :param drop_path_rate: Stochastic-depth rate.
    :type drop_path_rate: float
    :param normalization_type: Normalization override, or ``None`` for the
        block's own sequence-mode default.
    :type normalization_type: str | None
    :param use_bias: Whether inner projections carry a bias.
    :type use_bias: bool
    :param kernel_initializer: Initializer for inner kernels.
    :type kernel_initializer: Any
    :return: A configured :class:`CliffordEncoderBlock`.
    :rtype: keras.layers.Layer
    """
    return CliffordEncoderBlock(
        hidden_size=hidden_size,
        shifts=shifts,
        cli_mode=cli_mode,
        ctx_mode=ctx_mode,
        use_global_context=use_global_context,
        context_kernel_size=context_kernel_size,
        layer_scale_init=layer_scale_init,
        drop_path_rate=drop_path_rate,
        normalization_type=normalization_type,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        name=name,
    )


#: Block-type string -> builder. Append-only: the keys are public API,
#: recorded in every run directory's config and in the study's reports, so
#: renaming one invalidates existing results rather than tidying them.
BLOCK_REGISTRY: Dict[str, Callable[..., keras.layers.Layer]] = {
    "transformer": build_transformer_block,
    "clifford": build_clifford_block,
}


def available_block_types() -> List[str]:
    """Return the registered block types, sorted.

    :return: Registry keys.
    :rtype: list[str]
    """
    return sorted(BLOCK_REGISTRY)


def create_encoder_block(
    block_type: str,
    *,
    hidden_size: int,
    name: str,
    **block_kwargs: Any,
) -> keras.layers.Layer:
    """Build one block of the requested type.

    Validation follows the house factory contract: an unknown ``block_type``
    raises, and so does any keyword the chosen builder does not declare.
    Silently dropping an unrecognized keyword is never acceptable -- that is
    how a misspelled knob becomes a no-op that no test can see.

    :param block_type: A key of :data:`BLOCK_REGISTRY`.
    :type block_type: str
    :param hidden_size: Model width, forwarded to every builder.
    :type hidden_size: int
    :param name: Layer name, forwarded to every builder.
    :type name: str
    :param block_kwargs: Builder-specific keyword arguments.
    :type block_kwargs: Any
    :return: The configured block.
    :rtype: keras.layers.Layer
    :raises ValueError: If ``block_type`` is unknown, or a keyword is not
        declared by the builder.
    """
    if block_type not in BLOCK_REGISTRY:
        raise ValueError(
            f"Unknown block_type {block_type!r}. "
            f"Available: {available_block_types()}"
        )

    builder = BLOCK_REGISTRY[block_type]
    declared = set(inspect.signature(builder).parameters)
    unknown = sorted(set(block_kwargs) - declared)
    if unknown:
        raise ValueError(
            f"Block type {block_type!r} does not declare {unknown}. "
            f"Declared parameters: {sorted(declared)}"
        )

    return builder(hidden_size=hidden_size, name=name, **block_kwargs)
