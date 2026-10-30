"""
Shared skeleton for causal transformer stacks: positional embedding,
then ``depth`` causal ``TransformerLayer`` blocks, then a final
``LayerNormalization``.

Four BLT layers and :class:`PatchCausalTransformer` all run this same
stack and differ only in what sits around it: an entropy projection, a
patch pooler, cross-attention between blocks. This module holds the
stack once, so every caller builds the same blocks, applies the same
causal mask and uses the same final-norm epsilon, while the per-layer
heads stay in their own modules. It provides two builders and a mixin:
:func:`build_causal_transformer_stack` makes the blocks,
:func:`build_causal_stack_norm` makes the final norm, and
:class:`CausalTransformerStackMixin` drives them from a host layer's
``build`` and ``call``.

Use it when writing a new layer that needs an autoregressive
transformer over a ``(batch, seq, dim)`` sequence. The host layer
creates the three parts in ``__init__`` and hands them over; the mixin
adds no constructor arguments, so the host's ``get_config`` is
unchanged::

    class MyLayer(CausalTransformerStackMixin, keras.layers.Layer):
        def __init__(self, ...):
            ...
            self.bind_causal_stack(
                PositionalEmbedding(...),
                build_causal_transformer_stack(dim, heads, depth, rate),
                build_causal_stack_norm(name="final_norm"),
            )

        def build(self, input_shape):
            self.build_causal_stack(input_shape)
            super().build(input_shape)

        def call(self, x, training=None):
            return self.run_causal_stack(x, training=training)

Before use: ``build_causal_stack`` must be called from the host's
``build``, or the sub-layer weights do not reload from a ``.keras``
file. The attention mask is causal only, with no padding term; a padded
batch needs its own mask. Sequence length is bounded by the positional
embedding's table. The final norm uses ``epsilon=1e-3``
(:data:`LAYER_NORM_EPSILON`), not the norm factory's ``1e-6``; changing
it shifts every stack's output. ``intermediate_size`` defaults to
``4 * hidden_size``.

References:
    - Pagnoni et al., 2024. Byte Latent Transformer: Patches Scale Better
      Than Tokens. BLT, the main consumer of this stack.
      (https://arxiv.org/abs/2412.09871)
    - Vaswani et al., 2017. Attention Is All You Need. Transformer blocks
      and causal self-attention. (https://arxiv.org/abs/1706.03762)
    - Ba et al., 2016. Layer Normalization.
      (https://arxiv.org/abs/1607.06450)
"""

import keras
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..embedding.positional_embedding import PositionalEmbedding
from dl_techniques.utils.masking import create_causal_attend_mask
from .transformer import TransformerLayer

# ---------------------------------------------------------------------

#: Epsilon for a stack's FINAL norm, stated explicitly.
#:
#: ``1e-3`` is the value the BLT port inherited from
#: ``keras.layers.LayerNormalization``'s own default, and it is kept rather than
#: moved to the norm factory's ``1e-6``. Two repository rulings forbid the
#: silent swap: ``create_normalization_layer``'s own docstring carries a
#: ``.. warning::`` block ("THIS IS NOT A DROP-IN REPLACEMENT", with a measured
#: 11-of-16 divergence table) and states there is no ``epsilon=None`` sentinel
#: meaning "use the class default"; and decision D-202 rejected exactly this
#: rewrite for ``mobilenet``/``cbam`` (189 sites) on measurement.
#:
#: MEASURED consequence of the alternative, on ``PatchCausalTransformer`` at
#: ``dim=32, depth=2, num_heads=2, max_patches=8``, seed-pinned: moving this one
#: norm ``1e-3 -> 1e-6`` shifts the forward pass by ``max|delta| = 1.7e-03``
#: (relative ``5.0e-04``). So the value is PINNED, not incidental: a norm built
#: without an explicit ``epsilon`` is a defect, and the per-package census in
#: ``tests/test_models/test_the_norm_epsilon_provenance_is_stated.py`` fails if
#: this value ever moves in either direction.
LAYER_NORM_EPSILON = 1e-3


def build_causal_transformer_stack(
        hidden_size: int,
        num_heads: int,
        depth: int,
        dropout_rate: float,
        intermediate_size: Optional[int] = None,
        name_prefix: str = 'transformer',
) -> List[TransformerLayer]:
    """Build the ``depth``-block causal stack every layer here composes.

    Defaults ``intermediate_size`` to ``4 * hidden_size``, the expansion all
    four callers used before this helper existed.

    :param hidden_size: Block width; also the FFN's output width.
    :type hidden_size: int
    :param num_heads: Attention heads per block.
    :type num_heads: int
    :param depth: Number of blocks.
    :type depth: int
    :param dropout_rate: Forwarded to each block's FFN-output dropout.
    :type dropout_rate: float
    :param intermediate_size: FFN hidden width. Defaults to
        ``4 * hidden_size``.
    :type intermediate_size: Optional[int]
    :param name_prefix: Substring of each block's name, so two stacks in one
        model do not collide.
    :type name_prefix: str
    :return: The blocks, unnamed as a group so the caller controls the scope.
    :rtype: List[TransformerLayer]
    """
    if intermediate_size is None:
        intermediate_size = hidden_size * 4

    return [
        TransformerLayer(
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            dropout_rate=dropout_rate,
            name=f'{name_prefix}_{i}',
        )
        for i in range(depth)
    ]


def build_causal_stack_norm(
        name: str,
        epsilon: float = LAYER_NORM_EPSILON,
) -> keras.layers.LayerNormalization:
    """Build a stack's final norm, with ``epsilon`` always stated.

    Constructing ``keras.layers.LayerNormalization`` directly is correct here
    and routing it through the norm factory is not; see
    :data:`LAYER_NORM_EPSILON`. The point of this helper is that no caller can
    forget the argument.

    :param name: Layer name.
    :type name: str
    :param epsilon: Norm epsilon. Defaults to :data:`LAYER_NORM_EPSILON`.
    :type epsilon: float
    :return: The final norm.
    :rtype: keras.layers.LayerNormalization
    """
    return keras.layers.LayerNormalization(epsilon=epsilon, name=name)


class CausalTransformerStackMixin:
    """The ``build``/``call`` skeleton for a causal stack over embeddings.

    Not a ``keras.layers.Layer`` and not registered: it is a mixin that expects
    the composing layer to own three attributes, and adds no constructor
    arguments of its own, so the layer's ``get_config`` is untouched by its
    presence.

    Required attributes on the composing layer:

    ``stack_positional_embedding``
        a :class:`~dl_techniques.layers.embedding.positional_embedding.PositionalEmbedding`
    ``stack_layers``
        the blocks from :func:`build_causal_transformer_stack`
    ``stack_final_norm``
        the norm from :func:`build_causal_stack_norm`

    The ``attention_mask`` is causal over the sequence axis with NO padding
    term, because a padding mask is the caller's to supply and none of these
    layers have one to supply it.
    """

    #: Set by :func:`bind_causal_stack`; the three attributes the skeleton uses.
    stack_positional_embedding: PositionalEmbedding
    stack_layers: List[TransformerLayer]
    stack_final_norm: keras.layers.LayerNormalization

    def bind_causal_stack(self, positional_embedding, layers, final_norm) -> None:
        """Adopt the three sub-layer groups the skeleton drives.

        :param positional_embedding: The learned position table layer.
        :type positional_embedding: PositionalEmbedding
        :param layers: The causal blocks, in order.
        :type layers: List[TransformerLayer]
        :param final_norm: The final norm.
        :type final_norm: keras.layers.LayerNormalization
        """
        self.stack_positional_embedding = positional_embedding
        self.stack_layers = layers
        self.stack_final_norm = final_norm

    def build_causal_stack(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Build every sub-layer explicitly and return the post-stack shape.

        Explicit builds are load-bearing, not stylistic: a lazy first-call build
        leaves the weights unloadable on a ``.keras`` reload.

        :param input_shape: Shape of the tensor entering the stack.
        :type input_shape: Tuple[Optional[int], ...]
        :return: The shape leaving the final norm.
        :rtype: Tuple[Optional[int], ...]
        """
        self.stack_positional_embedding.build(input_shape)
        current_shape = self.stack_positional_embedding.compute_output_shape(input_shape)

        for layer in self.stack_layers:
            layer.build(current_shape)
            current_shape = layer.compute_output_shape(current_shape)

        self.stack_final_norm.build(current_shape)
        return current_shape

    def run_causal_stack(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Run positional embedding, the causal blocks, and the final norm.

        :param inputs: Embedded representations, shape ``(batch, seq, dim)``.
        :type inputs: keras.KerasTensor
        :param training: Whether in training mode.
        :type training: Optional[bool]
        :return: Contextualized representations, same shape as ``inputs``.
        :rtype: keras.KerasTensor
        """
        x = self.stack_positional_embedding(inputs, training=training)

        # Position i must not read position j > i.
        attend_mask = create_causal_attend_mask(x)
        for layer in self.stack_layers:
            x = layer(x, attention_mask=attend_mask, training=training)

        return self.stack_final_norm(x)


__all__ = [
    "LAYER_NORM_EPSILON",
    "CausalTransformerStackMixin",
    "build_causal_stack_norm",
    "build_causal_transformer_stack",
]