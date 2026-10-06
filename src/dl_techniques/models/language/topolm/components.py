"""The decoder block that carries TopoLM's spatial taps.

Why this block exists
---------------------
Every architectural piece a TopoLM block needs already exists in
``dl_techniques.layers.transformers`` -- and none of them can be used here.
:class:`~dl_techniques.layers.transformers.transformer.TransformerLayer` computes
the attention branch and the feed-forward branch, and then adds each straight
into the residual stream; in the pre-norm path the branch tensor is a local
variable overwritten by stochastic depth and layer scale before the residual add,
so it is gone by the time ``call`` returns. The spatial smoothness loss has to be
computed on that tensor, at exactly that point, before normalization and before
the add. There is no flag, no hook and no ``activity_regularizer`` that reaches it.

So the block is written here, and it composes the shared factories
(``create_attention_layer``, ``create_ffn_layer``,
``create_normalization_layer``) rather than reimplementing any of them. That is
the same shape ``hnet``, ``wave_field`` and ``tree_transformer`` use for their own
blocks, and the reuse order is unchanged: the factories first, a bespoke layer
last.

.. code-block:: text

    x
    │
    ├── attention_norm ──► attention ──► attn_tap ──┐   tap: identity forward,
    │                                               │   alpha * SL when training
    └───────────────────────────────────────────────┤
                                                    ▼
                                                  x + attn
                                                    │
                            ┌───────────────────────┘
                            ▼
    x' ──► ffn_norm ──► ffn_dense_1 ──► ffn_dense_2 ──► mlp_tap ──┐
                                                             ▼
                                                       x' + mlp

The taps sit ON the branch outputs, not around them: their forward pass is the
identity, so the residual arithmetic is byte-identical to a plain GPT-2 block and
any difference in the loss curve is attributable to the added term alone.

References:
    - Rathi, Mehrer, AlKhamissi, Binhuraib, Blauch & Schrimpf, 2025. TopoLM.
      ICLR 2025. (https://arxiv.org/abs/2410.11516)
    - Radford et al., 2019. Language Models are Unsupervised Multitask Learners.
    - Vaswani et al., 2017. Attention Is All You Need.
      (https://arxiv.org/abs/1706.03762)
    - Xiong et al., 2020. On Layer Normalization in the Transformer
      Architecture. (https://arxiv.org/abs/2002.04745)
"""

import math
from typing import Any, Dict, Optional, Tuple, Union

import keras

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.attention import create_attention_layer
from dl_techniques.layers.ffn import create_ffn_layer
from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

#: Which branch outputs a block can tap. The paper taps both.
TAP_SITES = ("attention", "mlp")


@register_dl_technique("dl_techniques.models.topolm.components")
class TopoLMBlock(keras.layers.Layer):
    """A pre-norm decoder block whose two branch outputs carry spatial taps.

    Two things distinguish it from a stock block, and both are load-bearing:

    * ``attn_out`` and ``mlp_out`` are held as named locals and passed through
      :class:`~dl_techniques.layers.regularization.spatial_smoothness.SpatialSmoothness`
      before the residual add, which is where the paper computes the loss.
    * ``alpha = 0`` produces a block whose weight set, tap count and layout seeds
      are IDENTICAL to an ``alpha > 0`` block. The non-topographic control is only
      a control if everything except the loss term is held fixed, and "held fixed"
      has to mean the weights too -- otherwise the two arms differ in RNG draws as
      well as in objective and no comparison between them means anything.

    :param hidden_size: Model width, and the tapped axis' length.
    :type hidden_size: int
    :param num_heads: Attention heads. Must divide ``hidden_size``.
    :type num_heads: int
    :param intermediate_size: Feed-forward hidden width.
    :type intermediate_size: int
    :param alpha: Weight on each tap's spatial loss. ``0`` gives the control.
    :type alpha: float
    :param radius: Neighbourhood radius for each tap.
    :type radius: int
    :param num_neighborhoods: Neighbourhoods sampled per tap per step.
    :type num_neighborhoods: int
    :param distance: Distance metric behind each tap's prior.
    :type distance: str
    :param permute: Whether each tap draws its own unit permutation. ``False`` is
        the paper's Fig. 12 ablation.
    :type permute: bool
    :param grid_shape: Explicit ``(height, width)`` for the taps, or ``None``.
    :type grid_shape: Optional[Tuple[int, int]]
    :param tap_sites: Which branch outputs to tap, a subset of
        :data:`TAP_SITES`. ``()`` builds the block with no taps at all, which is
        how a plain GPT-2 block is recovered from the same code.
    :type tap_sites: Tuple[str, ...]
    :param normalization_position: ``'pre'`` (the paper's) or ``'post'``.
    :type normalization_position: str
    :param activation: Feed-forward activation.
    :type activation: Union[str, Any]
    :param dropout_rate: Dropout on the feed-forward branch.
    :type dropout_rate: float
    :param attention_dropout_rate: Dropout on the attention output.
    :type attention_dropout_rate: float
    :param initializer_range: Stddev for the block's ``TruncatedNormal`` inits.
    :type initializer_range: float
    :param layer_norm_eps: Normalization epsilon, passed to BOTH norms. The
        factory default is 1e-6; GPT-2 uses 1e-5, and a stack whose two norms run
        at different epsilons is a silent 1000x spread inside one forward pass.
    :type layer_norm_eps: float
    :param seed: Base seed for this block's taps.
    :type seed: Optional[int]
    :param kwargs: Forwarded to ``keras.layers.Layer``.
    :raises ValueError: If ``hidden_size`` does not divide among ``num_heads``,
        if a name in ``tap_sites`` is not in :data:`TAP_SITES`, or if
        ``normalization_position`` is not ``'pre'`` or ``'post'`` -- naming the
        offending value in each case.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        alpha: float = 2.5,
        radius: int = 5,
        num_neighborhoods: int = 5,
        distance: str = "linf",
        permute: bool = True,
        grid_shape: Optional[Tuple[int, int]] = None,
        tap_sites: Tuple[str, ...] = TAP_SITES,
        normalization_position: str = "pre",
        activation: Union[str, Any] = "gelu",
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-5,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads "
                f"({num_heads})"
            )
        if normalization_position not in ("pre", "post"):
            raise ValueError(
                f"normalization_position must be 'pre' or 'post', got "
                f"{normalization_position!r}"
            )
        tap_sites = tuple(tap_sites)
        unknown = sorted(set(tap_sites) - set(TAP_SITES))
        if unknown:
            raise ValueError(
                f"tap_sites contains unknown site(s) {unknown}; "
                f"accepted: {list(TAP_SITES)}"
            )

        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.intermediate_size = int(intermediate_size)
        self.alpha = float(alpha)
        self.radius = int(radius)
        self.num_neighborhoods = int(num_neighborhoods)
        self.distance = distance
        self.permute = bool(permute)
        self.grid_shape = (
            None if grid_shape is None else tuple(int(v) for v in grid_shape)
        )
        self.tap_sites = tap_sites
        self.normalization_position = normalization_position
        self.activation = activation
        self.dropout_rate = float(dropout_rate)
        self.attention_dropout_rate = float(attention_dropout_rate)
        self.initializer_range = float(initializer_range)
        self.layer_norm_eps = float(layer_norm_eps)
        self.seed = seed

        # One initializer INSTANCE per block is deliberate: Keras hands the same
        # instance to identically-shaped sublayers inside a layer, so sharing it
        # would give bit-identical draws to the two residual projections and the
        # feed-forward pair would start as a rank-deficient map.
        kernel_initializer = keras.initializers.TruncatedNormal(
            stddev=self.initializer_range
        )
        # The two residual-path projections are scaled by depth, matching GPT-2's
        # _init_weights, which divides by sqrt(2 * n_layer) because each block
        # performs two residual additions.
        residual_initializer = keras.initializers.TruncatedNormal(
            stddev=self.initializer_range / math.sqrt(2.0)
        )

        self.attention_norm = create_normalization_layer(
            "layer_norm",
            name="attention_norm",
            epsilon=self.layer_norm_eps,
        )
        self.attention = create_attention_layer(
            "multi_head",
            name="attention",
            dim=self.hidden_size,
            num_heads=self.num_heads,
            dropout_rate=self.attention_dropout_rate,
            kernel_initializer=kernel_initializer,
            output_kernel_initializer=residual_initializer,
            use_bias=False,
        )
        self.ffn_norm = create_normalization_layer(
            "layer_norm",
            name="ffn_norm",
            epsilon=self.layer_norm_eps,
        )
        self.ffn_dense_1 = create_ffn_layer(
            "mlp",
            name="ffn_dense_1",
            hidden_dim=self.hidden_size,
            output_dim=self.intermediate_size,
            activation=self.activation,
            dropout_rate=self.dropout_rate,
            kernel_initializer=kernel_initializer,
            output_kernel_initializer=kernel_initializer,
            use_bias=True,
        )
        self.ffn_dense_2 = create_ffn_layer(
            "mlp",
            name="ffn_dense_2",
            hidden_dim=self.intermediate_size,
            output_dim=self.hidden_size,
            kernel_initializer=kernel_initializer,
            output_kernel_initializer=residual_initializer,
            use_bias=True,
        )
        self.ffn_dropout = keras.layers.Dropout(
            self.dropout_rate, name="ffn_dropout"
        )

        # Taps are created UNCONDITIONALLY and gated in `call`. Conditional
        # creation would shift every auto-generated layer name after it and make
        # the alpha=0 control differ from the alpha>0 arm by more than its loss.
        self.attention_tap = self._make_tap(
            "attention_tap", seed=self._tap_seed(0)
        )
        self.mlp_tap = self._make_tap("mlp_tap", seed=self._tap_seed(1))

        logger.info(
            f"TopoLMBlock: hidden={self.hidden_size}, heads={self.num_heads}, "
            f"ffn={self.intermediate_size}, taps={list(self.tap_sites)}, "
            f"alpha={self.alpha}, radius={self.radius}, "
            f"normalization_position={self.normalization_position!r}"
        )

    def _tap_seed(self, index: int) -> int:
        """The layout seed of one of this block's taps.

        Derived from ``(base seed, depth-independent tap index)`` through the
        layer's own :func:`~dl_techniques.layers.regularization.spatial_smoothness.permutation_seed`,
        so two taps never share a layout and the derivation is reproducible from
        configuration alone.
        """
        from dl_techniques.layers.regularization.spatial_smoothness import (
            permutation_seed,
        )

        return permutation_seed(self.seed, index)

    def _make_tap(self, name: str, seed: int):
        """Construct one tap layer with this block's spatial configuration."""
        from dl_techniques.layers.regularization.spatial_smoothness import (
            SpatialSmoothness,
        )

        return SpatialSmoothness(
            alpha=self.alpha,
            radius=self.radius,
            num_neighborhoods=self.num_neighborhoods,
            distance=self.distance,
            permute=self.permute,
            grid_shape=self.grid_shape,
            seed=seed,
            name=name,
        )

    @property
    def taps(self) -> Tuple[Any, ...]:
        """The tap layers, in site order; empty when ``tap_sites`` is empty."""
        selected = []
        if "attention" in self.tap_sites:
            selected.append(self.attention_tap)
        if "mlp" in self.tap_sites:
            selected.append(self.mlp_tap)
        return tuple(selected)

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build every sub-layer that ``call`` runs, and only those.

        Created-unconditionally / built-conditionally is the asymmetry this block
        depends on: an ``alpha=0`` or ``tap_sites=()`` block keeps a stable object
        graph and stable names, while building nothing it never calls.

        :param input_shape: Block input shape ``(batch, seq_len, hidden_size)``.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the last axis is not the configured
            ``hidden_size`` -- a cross-parameter contract ``call`` relies on, and
            one that a mismatch would otherwise turn into a broadcast error
            deep inside the attention layer.
        """
        if input_shape is not None and input_shape[-1] is not None:
            if int(input_shape[-1]) != self.hidden_size:
                raise ValueError(
                    f"TopoLMBlock was built for hidden_size={self.hidden_size} "
                    f"but received an input with last axis {int(input_shape[-1])}"
                )

        self.attention_norm.build(input_shape)
        self.attention.build(input_shape)

        self.ffn_norm.build(input_shape)
        self.ffn_dense_1.build(input_shape)
        widened = tuple(input_shape[:-1]) + (self.intermediate_size,)
        self.ffn_dense_2.build(widened)
        self.ffn_dropout.build(input_shape)

        if "attention" in self.tap_sites:
            self.attention_tap.build(input_shape)
        if "mlp" in self.tap_sites:
            self.mlp_tap.build(input_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
        taps: Optional[Tuple[Any, ...]] = None,
    ) -> keras.KerasTensor:
        """Run the block, tapping each branch output before its residual add.

        :param inputs: ``(batch, seq_len, hidden_size)``.
        :type inputs: keras.KerasTensor
        :param attention_mask: Rank-3 attend-semantics mask, as produced by
            :func:`~dl_techniques.utils.masking.factory.create_causal_attend_mask`.
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: Training flag, forwarded explicitly to every sub-layer.
            Keras propagates ``training`` through one mutable slot that a sibling
            overwrites, so omitting it is a latent bug even where it is currently
            a no-op.
        :type training: Optional[bool]
        :param taps: Optional pair of identity layers receiving
            ``(attn_out, mlp_out)``, for post-hoc activation extraction. This is a
            measurement hook, not a mechanism: the captures return their inputs
            unchanged, so the residual arithmetic is identical whether or not they
            are supplied. Order is the ``tap_sites`` order.
        :type taps: Optional[Tuple[Any, ...]]
        :return: ``(batch, seq_len, hidden_size)``.
        :rtype: keras.KerasTensor
        :raises ValueError: If ``taps`` has one entry per tapped site but the
            wrong count -- naming what was expected.
        """
        if taps is not None:
            expected = len(self.tap_sites)
            if len(taps) != expected:
                raise ValueError(
                    f"TopoLMBlock was given {len(taps)} capture layer(s) for "
                    f"{expected} tapped site(s) {list(self.tap_sites)}"
                )

        residual = inputs

        if self.normalization_position == "pre":
            normed = self.attention_norm(inputs, training=training)
            attn_out = self.attention(
                normed, attention_mask=attention_mask, training=training
            )
            if "attention" in self.tap_sites:
                attn_out = self.attention_tap(attn_out, training=training)
                if taps is not None and "attention" in self.tap_sites:
                    position = self.tap_sites.index("attention")
                    attn_out = taps[position](attn_out, training=training)
            hidden = attn_out + residual

            normed = self.ffn_norm(hidden, training=training)
            mlp_out = self.ffn_dense_2(
                self.ffn_dense_1(normed, training=training), training=training
            )
            mlp_out = self.ffn_dropout(mlp_out, training=training)
            if "mlp" in self.tap_sites:
                mlp_out = self.mlp_tap(mlp_out, training=training)
                if taps is not None and "mlp" in self.tap_sites:
                    position = self.tap_sites.index("mlp")
                    mlp_out = taps[position](mlp_out, training=training)
            return mlp_out + hidden

        attn_out = self.attention(
            inputs, attention_mask=attention_mask, training=training
        )
        if "attention" in self.tap_sites:
            attn_out = self.attention_tap(attn_out, training=training)
            if taps is not None and "attention" in self.tap_sites:
                position = self.tap_sites.index("attention")
                attn_out = taps[position](attn_out, training=training)
        hidden = self.attention_norm(attn_out + residual, training=training)

        mlp_out = self.ffn_dense_2(
            self.ffn_dense_1(hidden, training=training), training=training
        )
        mlp_out = self.ffn_dropout(mlp_out, training=training)
        if "mlp" in self.tap_sites:
            mlp_out = self.mlp_tap(mlp_out, training=training)
            if taps is not None and "mlp" in self.tap_sites:
                position = self.tap_sites.index("mlp")
                mlp_out = taps[position](mlp_out, training=training)
        return self.ffn_norm(mlp_out + hidden, training=training)

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Return ``input_shape``; the block preserves it."""
        return tuple(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return every constructor argument."""
        config = super().get_config()
        config.update({
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "intermediate_size": self.intermediate_size,
            "alpha": self.alpha,
            "radius": self.radius,
            "num_neighborhoods": self.num_neighborhoods,
            "distance": self.distance,
            "permute": self.permute,
            "grid_shape": self.grid_shape,
            "tap_sites": tuple(self.tap_sites),
            "normalization_position": self.normalization_position,
            "activation": keras.saving.serialize_keras_object(self.activation)
            if callable(self.activation) and not isinstance(self.activation, str)
            else self.activation,
            "dropout_rate": self.dropout_rate,
            "attention_dropout_rate": self.attention_dropout_rate,
            "initializer_range": self.initializer_range,
            "layer_norm_eps": self.layer_norm_eps,
            "seed": self.seed,
        })
        return config