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

from dl_techniques.layers.convnext_v1_block import ConvNextV1Block
from dl_techniques.layers.convnext_v2_block import ConvNextV2Block
from dl_techniques.layers.geometric.clifford_block import CliffordNetBlock
from dl_techniques.layers.stochastic_depth import StochasticDepth
from dl_techniques.layers.transformers import TransformerLayer
from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

__all__ = [
    "BLOCK_REGISTRY",
    "CliffordEncoderBlock",
    "ConvNextEncoderBlock",
    "available_block_types",
    "build_clifford_block",
    "build_convnext_block",
    "build_convnext_v2_block",
    "build_transformer_block",
    "clifford_receptive_field",
    "conv_receptive_field",
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


@register_dl_technique("dl_techniques.models.shared.blocks")
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
    :param dropout_rate: Dropout applied to the block's update before the
        external residual add -- the same position `ConvNextV1Block` and
        `ConvNextV2Block` apply theirs, so the arms are comparably regularized.
        `CliffordNetBlock` has no dropout parameter of its own and is shared
        with other packages, so this lives in the wrapper rather than in the
        layer. Defaults to 0.0. **This arm trained at 0.0 for Runs 1-4 while
        every other arm carried 0.1**; see RESULTS.md.
    :type dropout_rate: float
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
        dropout_rate: float = 0.0,
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
        self.dropout_rate = dropout_rate
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
        self.dropout = keras.layers.Dropout(dropout_rate, name="dropout")
        self.drop_path = StochasticDepth(
            drop_path_rate=drop_path_rate, name="drop_path"
        )

    def build(self, input_shape: Any) -> None:
        """Build the wrapped block on the sequence shape.

        :param input_shape: ``(batch, seq_len, hidden_size)``.
        :type input_shape: Any
        """
        self.block.build(input_shape)
        self.dropout.build(input_shape)
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
        update = self.dropout(update, training=training)
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
                "dropout_rate": self.dropout_rate,
                "normalization_type": self.normalization_type,
                "use_bias": self.use_bias,
                "kernel_initializer": self.kernel_initializer,
            }
        )
        return config


def conv_receptive_field(num_layers: int, kernel_size: int) -> int:
    """Return the token-mixing span of a stack of ONE-convolution blocks.

    A ConvNeXt block applies a SINGLE depthwise convolution, so a stack reaches
    ``num_layers * (K - 1) + 1`` tokens -- half the span a Clifford stack of the
    same depth and kernel gets, because that block applies two stacked
    convolutions per block. Use :func:`clifford_receptive_field` for the other
    arm; the two are deliberately separate functions rather than one with a
    flag, because silently applying the wrong factor is the kind of error that
    only shows up as a mediocre metric.

    :param num_layers: Number of stacked blocks.
    :type num_layers: int
    :param kernel_size: Depthwise kernel width ``K``.
    :type kernel_size: int
    :return: Receptive field in tokens.
    :rtype: int
    """
    return num_layers * (kernel_size - 1) + 1


@register_dl_technique("dl_techniques.models.shared.blocks")
class ConvNextEncoderBlock(keras.layers.Layer):
    """
    ConvNeXt V1 mixing block over a token sequence, with the block call contract.

    Wraps :class:`~dl_techniques.layers.convnext_v1_block.ConvNextV1Block` --
    depthwise convolution, normalization, pointwise expansion, activation,
    pointwise contraction, LayerScale -- and adapts it from images to sequences.

    **The sequence is lifted to a singleton-height image.** The wrapped block is
    2-D, so ``(B, L, D)`` becomes ``(B, 1, L, D)`` and the depthwise kernel is
    ``(1, K)``: convolution along the sequence axis only, never across the
    (length-1) height axis. The result is squeezed back. This is the same lift
    ``CliffordNetBlock`` performs internally for its own sequence mode.

    **The residual is external**, exactly as for the Clifford arm:
    ``ConvNextV1Block.call`` ends at ``return x`` with no ``+ inputs`` (see its
    step 7), so this wrapper computes ``inputs + drop_path(update)``. Note the
    default LayerScale here is ``gamma_initial_value=1.0``, not the Clifford
    block's ``1e-5``, so the update is full-magnitude from the first step.

    **Padding is not neutral**, for the same reason as the Clifford arm: a
    same-padded depthwise convolution pulls zero padding into the receptive
    field of real positions near the boundary. Masked positions are zeroed
    before the block, which bounds the effect without removing it. Stage 1 of
    the study trains on packed sequences carrying no padding at all.

    :param hidden_size: Channel dimension ``D``. Preserved by the block.
    :type hidden_size: int
    :param kernel_size: Depthwise kernel width along the sequence axis.
    :type kernel_size: int
    :param activation: Activation between the pointwise convolutions.
    :type activation: str
    :param dropout_rate: Dropout inside the block.
    :type dropout_rate: float
    :param drop_path_rate: Stochastic-depth rate for the residual branch.
    :type drop_path_rate: float
    :param gamma_initial_value: Initial LayerScale value.
    :type gamma_initial_value: float
    :param use_gamma: Whether to apply LayerScale at all.
    :type use_gamma: bool
    :param normalization_type: Normalization inside the wrapped block.
    :type normalization_type: str
    :param use_bias: Whether the convolutions carry a bias.
    :type use_bias: bool
    :param version: ``"v1"`` or ``"v2"``. V2 inserts Global Response
        Normalization after the activation and is otherwise the same block.
        One class with a switch rather than two near-identical wrappers,
        because everything around the wrapped block -- the lift, the mask
        zeroing, the external residual -- is identical between them.
    :type version: str
    :param kwargs: Additional keyword arguments for the Layer base class.
    :raises ValueError: If ``hidden_size`` or ``kernel_size`` is not a positive
        integer, or ``version`` is not ``"v1"``/``"v2"``.
    """

    #: Version string -> wrapped block class.
    _BLOCK_CLASSES: Dict[str, Any] = {
        "v1": ConvNextV1Block,
        "v2": ConvNextV2Block,
    }

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int = 7,
        activation: str = "gelu",
        dropout_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        gamma_initial_value: float = 1.0,
        use_gamma: bool = True,
        normalization_type: str = "layernorm",
        use_bias: bool = True,
        depthwise_initializer: Any = None,
        version: str = "v1",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if version not in self._BLOCK_CLASSES:
            raise ValueError(
                f"version must be one of {sorted(self._BLOCK_CLASSES)}, "
                f"got {version!r}"
            )
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError(
                f"hidden_size must be a positive int, got {hidden_size!r}"
            )
        if not isinstance(kernel_size, int) or kernel_size <= 0:
            raise ValueError(
                f"kernel_size must be a positive int, got {kernel_size!r}"
            )

        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.drop_path_rate = drop_path_rate
        self.gamma_initial_value = gamma_initial_value
        self.use_gamma = use_gamma
        self.normalization_type = normalization_type
        self.use_bias = use_bias
        self.depthwise_initializer = depthwise_initializer
        self.version = version

        self.block = self._BLOCK_CLASSES[version](
            # (1, K): convolve along the sequence axis only. The height axis is
            # the singleton introduced by the lift in `call`.
            kernel_size=(1, kernel_size),
            filters=hidden_size,
            activation=activation,
            dropout_rate=dropout_rate,
            use_gamma=use_gamma,
            gamma_initial_value=gamma_initial_value,
            normalization_type=normalization_type,
            use_bias=use_bias,
            depthwise_initializer=depthwise_initializer,
            name=f"convnext_{version}_block",
        )
        self.drop_path = StochasticDepth(
            drop_path_rate=drop_path_rate, name="drop_path"
        )

    def build(self, input_shape: Any) -> None:
        """Build the wrapped block on the lifted, rank-4 shape.

        :param input_shape: ``(batch, seq_len, hidden_size)``.
        :type input_shape: Any
        """
        batch, seq_len, channels = tuple(input_shape)
        self.block.build((batch, 1, seq_len, channels))
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
            Used only to zero padded positions before the block; a same-padded
            convolution is maskless (see the class docstring).
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

        # Lift (B, L, D) -> (B, 1, L, D) for the 2-D block, then squeeze back.
        lifted = keras.ops.expand_dims(block_input, axis=1)
        update = keras.ops.squeeze(self.block(lifted, training=training), axis=1)

        # External residual: both ConvNext block versions return the update
        # only (each ends at `return x`, with no `+ inputs`).
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
                "kernel_size": self.kernel_size,
                "activation": self.activation,
                "dropout_rate": self.dropout_rate,
                "drop_path_rate": self.drop_path_rate,
                "gamma_initial_value": self.gamma_initial_value,
                "use_gamma": self.use_gamma,
                "normalization_type": self.normalization_type,
                "use_bias": self.use_bias,
                "version": self.version,
                "depthwise_initializer": keras.initializers.serialize(
                    keras.initializers.get(self.depthwise_initializer)
                )
                if self.depthwise_initializer is not None
                else None,
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
    dropout_rate: float = 0.0,
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
    :param dropout_rate: Dropout on the update before the external residual.
    :type dropout_rate: float
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
        dropout_rate=dropout_rate,
        normalization_type=normalization_type,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        name=name,
    )


def build_convnext_block(
    *,
    hidden_size: int,
    name: str,
    kernel_size: int = 7,
    activation: str = "gelu",
    dropout_rate: float = 0.0,
    drop_path_rate: float = 0.0,
    gamma_initial_value: float = 1.0,
    use_gamma: bool = True,
    normalization_type: str = "layernorm",
    use_bias: bool = True,
    kernel_initializer: Any = None,
) -> keras.layers.Layer:
    """Build one ConvNeXt V1 sequence-mixing block.

    :param hidden_size: Model width.
    :type hidden_size: int
    :param name: Layer name.
    :type name: str
    :param kernel_size: Depthwise kernel width along the sequence axis.
    :type kernel_size: int
    :param activation: Activation between the pointwise convolutions.
    :type activation: str
    :param dropout_rate: Dropout inside the block.
    :type dropout_rate: float
    :param drop_path_rate: Stochastic-depth rate.
    :type drop_path_rate: float
    :param gamma_initial_value: Initial LayerScale value.
    :type gamma_initial_value: float
    :param use_gamma: Whether to apply LayerScale.
    :type use_gamma: bool
    :param normalization_type: Normalization inside the block.
    :type normalization_type: str
    :param use_bias: Whether the convolutions carry a bias.
    :type use_bias: bool
    :param kernel_initializer: Initializer for the DEPTHWISE convolution. The
        two pointwise convolutions inside ``ConvNextV1Block`` hard-code their
        own ``TruncatedNormal`` and expose no override, so this reaches the
        depthwise kernel only. Declared here so the encoder's shared
        initializer injection has somewhere to land; the registry rejects an
        undeclared keyword rather than dropping it.
    :type kernel_initializer: Any
    :return: A configured :class:`ConvNextEncoderBlock`.
    :rtype: keras.layers.Layer
    """
    return ConvNextEncoderBlock(
        hidden_size=hidden_size,
        kernel_size=kernel_size,
        activation=activation,
        dropout_rate=dropout_rate,
        drop_path_rate=drop_path_rate,
        gamma_initial_value=gamma_initial_value,
        use_gamma=use_gamma,
        normalization_type=normalization_type,
        use_bias=use_bias,
        depthwise_initializer=kernel_initializer,
        version="v1",
        name=name,
    )


def build_convnext_v2_block(
    *,
    hidden_size: int,
    name: str,
    kernel_size: int = 7,
    activation: str = "gelu",
    dropout_rate: float = 0.0,
    drop_path_rate: float = 0.0,
    gamma_initial_value: float = 1.0,
    use_gamma: bool = True,
    normalization_type: str = "layernorm",
    use_bias: bool = True,
    kernel_initializer: Any = None,
) -> keras.layers.Layer:
    """Build one ConvNeXt V2 sequence-mixing block.

    Identical to :func:`build_convnext_block` except for the wrapped block:
    V2 inserts Global Response Normalization after the activation. GRN scores
    each channel by its L2 magnitude **over the sequence axis**, so unlike
    every other block in this registry it performs a GLOBAL reduction along
    the sequence -- see :class:`ConvNextEncoderBlock` for what that means for
    padded batches.

    A separate function rather than a ``version=`` argument on the v1 builder,
    because the registry validates keywords against the builder's signature:
    a shared builder would let ``version`` be passed through
    ``block_config`` from anywhere, and the registry key is meant to be the
    single place the version is decided.

    :param hidden_size: Model width.
    :type hidden_size: int
    :param name: Layer name.
    :type name: str
    :param kernel_size: Depthwise kernel width along the sequence axis.
    :type kernel_size: int
    :param activation: Activation before GRN.
    :type activation: str
    :param dropout_rate: Dropout inside the block.
    :type dropout_rate: float
    :param drop_path_rate: Stochastic-depth rate.
    :type drop_path_rate: float
    :param gamma_initial_value: Initial LayerScale value.
    :type gamma_initial_value: float
    :param use_gamma: Whether to apply LayerScale.
    :type use_gamma: bool
    :param normalization_type: Normalization inside the block.
    :type normalization_type: str
    :param use_bias: Whether the convolutions carry a bias.
    :type use_bias: bool
    :param kernel_initializer: Initializer for the depthwise convolution; see
        :func:`build_convnext_block` for what it does and does not reach.
    :type kernel_initializer: Any
    :return: A configured :class:`ConvNextEncoderBlock` wrapping V2.
    :rtype: keras.layers.Layer
    """
    return ConvNextEncoderBlock(
        hidden_size=hidden_size,
        kernel_size=kernel_size,
        activation=activation,
        dropout_rate=dropout_rate,
        drop_path_rate=drop_path_rate,
        gamma_initial_value=gamma_initial_value,
        use_gamma=use_gamma,
        normalization_type=normalization_type,
        use_bias=use_bias,
        depthwise_initializer=kernel_initializer,
        version="v2",
        name=name,
    )


#: Block-type string -> builder. Append-only: the keys are public API,
#: recorded in every run directory's config and in the study's reports, so
#: renaming one invalidates existing results rather than tidying them.
BLOCK_REGISTRY: Dict[str, Callable[..., keras.layers.Layer]] = {
    "transformer": build_transformer_block,
    "clifford": build_clifford_block,
    "convnext": build_convnext_block,
    "convnext_v2": build_convnext_v2_block,
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
